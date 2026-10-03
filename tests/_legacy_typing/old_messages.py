"""Operator-editable wording (KCD-353, KCD-500, KCD-513): the database can override any phrase.

The clinic API holds a table of messages (clinic-api/agent_messages.py). At the start of each call
the orchestrator refreshes this module from it; everything that speaks a fixed phrase asks here
first and falls back to the text built into the code:

    messages.text("disclosure", "en", default)     -> the database wording if there is one, else `default`

So the operator can change what callers hear -- the disclosure, the "I cannot find that" line, the
thank-you -- by updating a row, with no deploy, and an outage of the API can never leave the agent
without words: an empty or stale cache simply returns the built-in default.

What this does NOT do: it does not validate wording (tools/check_messages.py does, against the
persona rules) and it cannot add a FACT -- prices, times and confirmation numbers are template
substitutions from live answers, not messages. `version()` reports the highest message version seen,
which the call record stores so an audit can tell which wording a caller heard.

Pure Python, no I/O: the refresh itself (an HTTP call) is made by the orchestrator, which hands the
result to `load()`.
"""

from __future__ import annotations

import time

# {(key, lang): (text, version)}
_cache: dict[tuple[str, str], tuple[str, int]] = {}
_loaded_at: float | None = None
_top_version = 0

REFRESH_AFTER_S = 60.0


def load(payload: dict | None, now: float | None = None) -> int:
    """Replace the cache from a GET /api/v1/agent/messages response. Returns how many entries.
    A missing or malformed payload leaves the existing cache untouched (never empties it)."""
    global _loaded_at, _top_version
    try:
        rows = payload["messages"]
    except (TypeError, KeyError):
        return len(_cache)
    if not isinstance(rows, dict):  # a malformed body is ignored, never trusted
        return len(_cache)
    fresh: dict[tuple[str, str], tuple[str, int]] = {}
    for key, by_lang in rows.items():
        if not isinstance(by_lang, dict):
            continue
        for lang, entry in by_lang.items():
            text = (entry or {}).get("text")
            if isinstance(text, str) and text.strip():
                fresh[(key, lang)] = (text.strip(), int((entry or {}).get("version", 1)))
    _cache.clear()
    _cache.update(fresh)
    _top_version = int((payload or {}).get("version", 0) or 0)
    _loaded_at = time.monotonic() if now is None else now
    return len(_cache)


def clear() -> None:
    global _loaded_at, _top_version
    _cache.clear()
    _loaded_at, _top_version = None, 0


def stale(now: float | None = None) -> bool:
    n = time.monotonic() if now is None else now
    return _loaded_at is None or (n - _loaded_at) > REFRESH_AFTER_S


def text(key: str, lang: str, default: str) -> str:
    """The operator's wording for (key, lang) if there is one, else `default`."""
    hit = _cache.get((key, lang))
    return hit[0] if hit else default


def has(key: str, lang: str) -> bool:
    return (key, lang) in _cache


def version() -> int:
    return _top_version


def label() -> str:
    """What the call record stores as the wording version: `db:<n>` when the database supplied any."""
    return f"db:{_top_version}" if _cache else "built-in"
