"""KCD-501: every call leaves a complete record.

"Intent, actions taken, values confirmed, escalation reason and outcome are written
within thirty seconds of the call ending. The write is idempotent and a dropped call
still records what was completed."

How each part is met:

  * WRITTEN AS THE CALL GOES. Every event (a language, an intent, a completed action,
    confirmed values, an escalation) is queued the moment it happens and flushed
    right away, so a call that drops -- socket closed, process killed, network cut --
    already has everything that was completed before it dropped. The end-of-call write
    only adds the outcome.

  * IDEMPOTENT. Each event carries a sequence number that this recorder assigns ONCE
    and keeps until the server acknowledges it (clinic-api applies an event only if
    its `seq` is higher than the last applied). A retry after a lost response is
    acknowledged as a duplicate and changes nothing: no doubled action, no doubled
    intent.

  * WITHIN THIRTY SECONDS. finish() flushes with a deadline (FINISH_DEADLINE_S, 25 s),
    retrying failed events with a short backoff until it either drains or the deadline
    passes; the un-flushed count is returned and logged so a miss is visible, never
    silent.

  * MINIMAL. Values confirmed are redacted server-side (clinic-api redact_confirmed):
    an allow-list of fields, a phone as its last four digits, no name and no address.

Pure asyncio around an injected writer: tests use a fake that fails, duplicates and
drops on demand.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Awaitable, Callable

FINISH_DEADLINE_S = 25.0
RETRY_BACKOFF_S = (0.2, 0.5, 1.0, 2.0)

Writer = Callable[[str, int, str, dict, "str | None"], Awaitable[dict]]


@dataclasses.dataclass
class _Event:
    seq: int
    kind: str
    payload: dict


class CallRecorder:
    def __init__(
        self,
        call_id: str,
        writer: Writer,
        caller_phone: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep=asyncio.sleep,
    ):
        self.call_id, self.caller_phone = call_id, caller_phone
        self._writer, self._clock, self._sleep = writer, clock, sleep
        self._seq = 0
        self._pending: list[_Event] = []
        self._lock = asyncio.Lock()
        self.acknowledged = 0
        self.finished = False
        self.unflushed_at_finish = 0

    # -- events ----------------------------------------------------------
    def add(self, kind: str, payload: dict | None = None) -> int:
        """Queue an event; returns its sequence number. Cheap and synchronous, so it can be
        called from anywhere in a turn without awaiting a network round trip."""
        self._seq += 1
        self._pending.append(_Event(self._seq, kind, dict(payload or {})))
        return self._seq

    def start(self, channel: str = "voice", disclosure_version: str = "") -> None:
        self.add("start", {"channel": channel, "disclosure_version": disclosure_version})

    def language(self, lang: str) -> None:
        self.add("language", {"language": lang})

    def intent(self, name: str) -> None:
        self.add("intent", {"intent": name})

    def action(self, name: str, ref: str | None = None) -> None:
        self.add("action", {"name": name, "ref": ref})

    def confirmed(self, slots: dict) -> None:
        self.add("confirmed", {"slots": dict(slots)})

    def history(self, statement_ids: list[str]) -> None:
        if statement_ids:
            self.add("history", {"statements": list(statement_ids)})

    def patient(self, patient_ref) -> None:
        self.add("patient", {"patient_ref": patient_ref})

    def satisfaction(self, payload: dict) -> None:
        """The call's implicit happiness score (agent/call_score.py): counts, flags and reasons, never any text."""
        self.add("satisfaction", payload)

    def escalation(self, reason: str) -> None:
        self.add("escalation", {"reason": reason})

    # -- delivery --------------------------------------------------------
    @property
    def pending(self) -> int:
        return len(self._pending)

    async def flush(self) -> int:
        """Send what is queued, in order. An event stays queued (with its seq) if the
        write fails, and everything after it waits behind it so order is preserved.
        Returns how many are still pending."""
        async with self._lock:
            while self._pending:
                ev = self._pending[0]
                try:
                    await self._writer(self.call_id, ev.seq, ev.kind, ev.payload, self.caller_phone)
                except Exception:  # noqa: BLE001 - any failure leaves it queued for the retry
                    break
                self._pending.pop(0)
                self.acknowledged += 1
        return len(self._pending)

    async def finish(self, outcome: str) -> int:
        """Close the record. Retries until drained or FINISH_DEADLINE_S; returns the number
        of events that could NOT be written (0 is the goal)."""
        if not self.finished:
            self.add("end", {"outcome": outcome})
            self.finished = True
        deadline = self._clock() + FINISH_DEADLINE_S
        attempt = 0
        left = await self.flush()
        while left and self._clock() < deadline:
            await self._sleep(RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)])
            attempt += 1
            left = await self.flush()
        self.unflushed_at_finish = left
        return left
