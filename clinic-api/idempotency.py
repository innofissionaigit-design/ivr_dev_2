"""Idempotent writes: a retried POST does what the first one did, once.

A voice agent on a bad line retries: the connection drops after the server has committed, the client never hears the
answer and sends the same request again. Without a guard that is a second booking, a second SMS, a second callback.

The contract (the standard `Idempotency-Key` header, as used by payment APIs):

  * the client sends `Idempotency-Key: <opaque string>` with a write, and REUSES it when it retries the same operation;
  * the first request with a key runs and its JSON answer is stored under (key, scope);
  * a repeat with the same key AND the same body returns the stored answer, and runs nothing;
  * the same key with a DIFFERENT body is a client bug and is refused (422), never silently treated as either;
  * no key: the write runs as before, and the endpoints whose operation has a natural identity (an SMS to the same
    number with the same text, a callback for the same call and window, ...) de-duplicate on that instead.

Race: two concurrent requests with one key may both run before either stores its answer; the unique index makes the
second store fail, and that request then returns the first one's stored answer. It narrows a retry storm to at most one
extra execution of an operation that is itself made safe to repeat by its own guards (holds, confirms, unique slots).
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import json
import typing

from fastapi import HTTPException
from models import IdempotencyRecord
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

KEY_HEADER = "idempotency-key"
MAX_KEY_CHARS = 200

_key: contextvars.ContextVar[str | None] = contextvars.ContextVar("idempotency_key", default=None)


def set_key_from_header(value: str | None) -> None:
    """Called by the request middleware; a blank or oversized key is ignored (the write then runs unguarded)."""
    value = (value or "").strip()
    _key.set(value if 0 < len(value) <= MAX_KEY_CHARS else None)


def current_key() -> str | None:
    return _key.get()


def _fingerprint(kwargs: dict) -> str:
    """A stable hash of the request body(s): every pydantic argument, every path/query value; never the DB session."""
    parts = {}
    for name, value in sorted(kwargs.items()):
        if isinstance(value, Session):
            continue
        parts[name] = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def idempotent(scope: str):
    """Decorator for a synchronous endpoint that takes `db: Session`. The endpoint's signature is unchanged (FastAPI
    still sees the original parameters through functools.wraps).

    WHY THE ANNOTATIONS ARE RESOLVED EAGERLY BELOW -- a real defect, fixed:

    `functools.wraps` copies `__wrapped__`, so `inspect.signature()` reports the
    decorated function's parameters correctly. It does NOT change `__globals__`:
    that still belongs to THIS module, because `wrapper` is defined here. FastAPI
    resolves a parameter's type with `getattr(call, "__globals__", {})`, and
    clinic-api/main.py uses `from __future__ import annotations`, so every
    annotation there is a plain STRING. The result was that FastAPI looked up a
    name like "BookingRequest" in this module's namespace, where it does not
    exist, and raised

        pydantic.errors.PydanticUndefinedAnnotation: name 'BookingRequest' is not defined

    AT IMPORT TIME -- so the whole app could not be imported, and 16 test modules
    could not run at all (including the one that audits every write endpoint's
    schema), and `scripts/gate.sh` could not run either.

    `typing.get_type_hints(fn)` resolves those strings against `fn`'s OWN globals
    -- main.py's -- and assigning the result to `wrapper.__annotations__` leaves
    FastAPI with real classes that need no namespace lookup at all. `__globals__`
    is read-only, so this is the fix available rather than a workaround.

    `include_extras=True` keeps `Annotated[...]` metadata (`Query(...)`,
    `Depends(...)`) intact, which FastAPI needs. If a hint genuinely cannot be
    resolved the original strings are left in place, so this can only ever
    improve on the previous behaviour, never break an endpoint that worked.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key, db = current_key(), kwargs.get("db")
            if not key or db is None:
                return fn(*args, **kwargs)
            digest = _fingerprint(kwargs)
            rec = db.query(IdempotencyRecord).filter_by(key=key, scope=scope).first()
            if rec is not None:
                return _replay(rec, digest)
            result = fn(*args, **kwargs)
            try:
                payload = json.dumps(result, default=str)
            except (TypeError, ValueError):
                return result  # not JSON-shaped (a raw Response): nothing to store
            try:
                db.add(IdempotencyRecord(key=key, scope=scope, request_hash=digest, response_json=payload))
                db.commit()
            except IntegrityError:  # a concurrent twin stored first: answer as it did
                db.rollback()
                rec = db.query(IdempotencyRecord).filter_by(key=key, scope=scope).first()
                if rec is not None:
                    return _replay(rec, digest)
            return result

        # Resolve the endpoint's annotations ONCE, here, against main.py's own
        # globals, and publish them as an explicit `__signature__`.
        #
        # Setting `wrapper.__annotations__` is NOT enough and was tried first:
        # `functools.wraps` sets `__wrapped__`, and `inspect.signature()` follows
        # that to the original function, whose annotations are still the raw
        # strings `from __future__ import annotations` produced. `__signature__`
        # takes precedence over `__wrapped__`, so this is the one place the
        # resolved types actually reach FastAPI.
        try:
            hints = typing.get_type_hints(fn, include_extras=True)
            sig = inspect.signature(fn)
            wrapper.__signature__ = sig.replace(
                parameters=[
                    prm.replace(annotation=hints.get(prm.name, prm.annotation))
                    for prm in sig.parameters.values()
                ]
            )
        except Exception:  # noqa: BLE001  -- an unresolvable hint leaves the signature as it was
            pass
        return wrapper

    return deco


def _replay(rec: IdempotencyRecord, digest: str):
    if rec.request_hash != digest:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key was already used with a different request",
        )
    return json.loads(rec.response_json)
