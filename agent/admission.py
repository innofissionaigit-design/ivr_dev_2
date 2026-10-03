"""Admission control for AI voice calls.

The rule this enforces (docs/adr/0001, section 5; Blueprint Appendix A/B):
a call the AI cannot serve well goes to a person, immediately. It is never
queued behind the GPU. An unbounded queue turns an overloaded system into a
system where EVERY caller hears dead air; a refused call gets a human.

What can close the door, in priority order:

  bypass       An operator forced human-only mode (the RFP's audited AI
               kill-switch). Two ways to flip it, both without a deploy: the
               flag file (`touch` it) and set_bypass() behind the HTTP
               endpoint in main.py. Either one is enough; the file is
               checked on every admit so it works even if the process's own
               control plane is wedged.
  backend_down A model service the call cannot complete without has failed
               its health probes (see HealthMonitor). "GPU loss lowers
               capacity to zero" -- there is no partial credit for a call
               that will hit a dead TTS on its first reply.
  latency_shed Recent turn latency is over budget. Existing calls keep going;
               NEW calls are refused until p95 recovers. This is the
               "latency-aware" part: the GPU can be saturated long before any
               counter reaches the cap, and turn latency is what a caller
               actually feels.
  capacity     max_calls reached.

Pure logic plus one small async health poller: no model, no audio. Everything
here is unit-tested (tests/test_admission.py) with an injected clock.

What this does NOT do: transfer a live call. Moving a SIP call to the contact
centre happens in the SBC layer, outside this repository. This module makes
the decision and emits it (main.py sends a `handoff_human` frame on the
socket); the telephony bridge acts on that frame.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger("admission")

DEFAULT_MAX_CALLS = 28  # ADR section 5: AI cap 28-30 for the 30-call design point
DEFAULT_LATENCY_WINDOW = 30  # turns considered when judging p95
DEFAULT_MIN_LATENCY_SAMPLES = 10  # too few samples -> do not shed on noise
DEFAULT_BYPASS_FILE = "/workspace/.ai_bypass"


@dataclasses.dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: str  # "ok" on admit; otherwise bypass|backend_down:<n>|latency_shed|capacity
    ticket: int | None = None


def _percentile(values, q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


class AdmissionController:
    """One instance per process, shared by every call (state is process-wide
    by design: it counts THIS process's calls). With two orchestrator
    instances the true global cap is the sum of both caps -- set max_calls
    to half the intended total, or move the counter to shared storage
    (Redis) when the second instance goes live."""

    def __init__(
        self,
        max_calls: int = DEFAULT_MAX_CALLS,
        *,
        latency_shed_p95_s: float | None = None,
        latency_window: int = DEFAULT_LATENCY_WINDOW,
        min_latency_samples: int = DEFAULT_MIN_LATENCY_SAMPLES,
        bypass_file: str | None = DEFAULT_BYPASS_FILE,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        self.max_calls = max_calls
        self.latency_shed_p95_s = latency_shed_p95_s
        self.min_latency_samples = min_latency_samples
        self.bypass_file = bypass_file
        self._clock = clock

        self._lock = threading.Lock()
        self._next_ticket = 1
        self._active: set[int] = set()
        self._latencies: collections.deque[float] = collections.deque(maxlen=latency_window)
        self._bypass = False
        self._bypass_by: str | None = None
        self._down: dict[str, str] = {}
        self._counters: collections.Counter[str] = collections.Counter()
        self._peak = 0

    # ------------------------------------------------------------ decisions
    def _closed_reason(self) -> str | None:
        """Why the door is closed to NEW calls right now, or None if open.
        Caller holds the lock."""
        if self._bypass or (self.bypass_file and os.path.exists(self.bypass_file)):
            return "bypass"
        if self._down:
            return f"backend_down:{sorted(self._down)[0]}"
        if (
            self.latency_shed_p95_s is not None
            and len(self._latencies) >= self.min_latency_samples
            and _percentile(self._latencies, 0.95) > self.latency_shed_p95_s
        ):
            return "latency_shed"
        return None

    def try_admit(self) -> Admission:
        with self._lock:
            reason = self._closed_reason()
            if reason is None and len(self._active) >= self.max_calls:
                reason = "capacity"
            if reason is not None:
                self._counters[f"rejected:{reason.split(':')[0]}"] += 1
                return Admission(False, reason)
            ticket = self._next_ticket
            self._next_ticket += 1
            self._active.add(ticket)
            self._peak = max(self._peak, len(self._active))
            self._counters["admitted"] += 1
            return Admission(True, "ok", ticket)

    def release(self, admission: Admission) -> None:
        """Idempotent: a call's teardown may run twice (disconnect plus the
        poll task's own exit) and must not free a slot it no longer holds."""
        if admission.ticket is None:
            return
        with self._lock:
            self._active.discard(admission.ticket)

    # ------------------------------------------------------------- feedback
    def record_turn_latency(self, seconds: float) -> None:
        """End-to-end time from utterance end to first audio sent, per turn.
        Feeds latency_shed."""
        with self._lock:
            self._latencies.append(float(seconds))

    def set_backend_health(self, name: str, ok: bool, detail: str = "") -> None:
        with self._lock:
            was_down = name in self._down
            if ok:
                self._down.pop(name, None)
            else:
                self._down[name] = detail or "unhealthy"
        if ok and was_down:
            logger.warning("backend %s recovered", name)
        elif not ok and not was_down:
            logger.error("backend %s DOWN (%s): admission closed", name, detail)

    def set_bypass(self, enabled: bool, actor: str = "unknown") -> None:
        with self._lock:
            self._bypass = enabled
            self._bypass_by = actor if enabled else None
        # Audited, as the RFP requires: who flipped it, and to what.
        logger.warning("AI bypass %s by %s", "ENABLED" if enabled else "cleared", actor)

    # ------------------------------------------------------------- read-out
    @property
    def active_calls(self) -> int:
        with self._lock:
            return len(self._active)

    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self._latencies)
            return {
                "open": self._closed_reason() is None and len(self._active) < self.max_calls,
                "closed_reason": self._closed_reason(),
                "active_calls": len(self._active),
                "max_calls": self.max_calls,
                "peak_active": self._peak,
                "bypass": self._bypass or bool(self.bypass_file and os.path.exists(self.bypass_file)),
                "bypass_by": self._bypass_by,
                "backends_down": dict(self._down),
                "turn_latency_p50_s": round(_percentile(lat, 0.50), 3) if lat else None,
                "turn_latency_p95_s": round(_percentile(lat, 0.95), 3) if lat else None,
                "latency_samples": len(lat),
                "counters": dict(self._counters),
            }


# --------------------------------------------------------------- health probe

Probe = Callable[[], Awaitable[bool]]


class HealthMonitor:
    """Polls the model services a call needs and feeds AdmissionController.

    A backend is declared down only after `fail_threshold` CONSECUTIVE
    failed probes: one slow /health under load must not slam the door on
    every caller. It is declared up again on the first success -- recovery
    should be quick, and a flapping backend is already covered by the
    failure threshold on the way down."""

    def __init__(
        self,
        controller: AdmissionController,
        probes: dict[str, Probe],
        interval_s: float = 5.0,
        fail_threshold: int = 3,
    ):
        self.controller = controller
        self.probes = probes
        self.interval_s = interval_s
        self.fail_threshold = fail_threshold
        self._fails: dict[str, int] = {name: 0 for name in probes}

    async def check_once(self) -> None:
        for name, probe in self.probes.items():
            try:
                ok = bool(await probe())
            except Exception as exc:  # noqa: BLE001 - a probe crash is a failed probe
                ok = False
                detail = f"{type(exc).__name__}: {exc}"[:120]
            else:
                detail = "probe returned false" if not ok else ""
            if ok:
                self._fails[name] = 0
                self.controller.set_backend_health(name, True)
            else:
                self._fails[name] += 1
                if self._fails[name] >= self.fail_threshold:
                    self.controller.set_backend_health(name, False, detail)

    async def run_forever(self) -> None:
        import asyncio

        while True:
            await self.check_once()
            await asyncio.sleep(self.interval_s)


def http_probe(client, url: str, timeout_s: float = 3.0) -> Probe:
    """Probe for an httpx.AsyncClient: healthy iff GET url returns 200."""

    async def _probe() -> bool:
        r = await client.get(url, timeout=timeout_s)
        return r.status_code == 200

    return _probe
