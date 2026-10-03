"""KCD-076: "caller-state detection costs almost nothing, so empathy
does not buy latency."

Language ID, code-switch and channel-quality detection all run on every
turn now (or will, as the Empathy epic's detectors land). None of them
may be allowed to make a turn slower than Appendix B's budget for their
slice, so every one of them is called through run_within_budget rather
than directly.

run_within_budget enforces the turn-facing half of the acceptance
criterion for real: `fn` runs in a worker thread (asyncio.to_thread) and
the AWAIT is bounded by `budget_s` via asyncio.wait_for, so the calling
turn is provably never delayed past budget -- it moves on and degrades
to `previous_value` the instant the deadline passes, it does not wait
for the slow call to finish first. What it does NOT do (a real, stated
Python limitation, not swept under the rug): the abandoned background
thread is not forcibly killed -- CPython has no safe preemptive
thread-kill -- so a detector that truly hangs leaves an orphaned thread
running harmlessly to completion and then discarding its result. Every
detector this budget wraps (channel_quality's FFT arithmetic, LID,
code-switch's regex scan) is deterministic, bounded CPU work with no
I/O and no lock it could hang on, so an orphaned thread is a
theoretical concern here, not an operational one -- this module exists
so that stays true by measurement (see the histogram export) instead of
by assumption.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger("detector_budget")

T = TypeVar("T")

# Per-detector histograms of elapsed seconds, for the export KCD-076's
# own acceptance criterion asks for ("measured as a histogram").
_TIMINGS: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=500))
_OVER_BUDGET_COUNTS: dict[str, int] = collections.defaultdict(int)


async def run_within_budget(
    name: str, budget_s: float, fn: Callable[..., T], *args, previous_value: T | None = None, **kwargs
) -> T | None:
    """Runs fn(*args, **kwargs) in a worker thread. Returns its result if
    it completes within budget_s; otherwise returns `previous_value`
    immediately when the deadline passes, without waiting further for
    fn -- see module docstring for exactly what "without waiting" does
    and does not guarantee."""
    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=budget_s)
    except TimeoutError:
        elapsed = time.monotonic() - t0
        _OVER_BUDGET_COUNTS[name] += 1
        logger.warning(
            "detector %s exceeded its %.3fs budget (still running after %.3fs) -- degrading to previous value",
            name,
            budget_s,
            elapsed,
        )
        return previous_value
    _TIMINGS[name].append(time.monotonic() - t0)
    return result


def snapshot() -> dict:
    out = {}
    for name, samples in _TIMINGS.items():
        xs = sorted(samples)
        out[name] = {
            "samples": len(xs),
            "p50_s": round(xs[len(xs) // 2], 4) if xs else None,
            "p95_s": round(xs[int(len(xs) * 0.95)], 4) if xs else None,
            "max_s": round(xs[-1], 4) if xs else None,
            "over_budget_count": _OVER_BUDGET_COUNTS.get(name, 0),
        }
    return out
