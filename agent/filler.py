"""KCD-461: "a natural filler is spoken when a stage exceeds its
threshold... suppressed if the result arrives first. Silence never
exceeds a stated maximum."

Pure racing logic, no TTS/session/vendor dependency (CLAUDE.md's
one-import rule and vendor-boundary rule both apply the same way here
as everywhere else in agent/) -- main.py's _await_with_filler is a thin
wrapper passing the actual "speak the filler" call as `on_timeout`, so
this module is fully unit-testable with a fake awaitable and a fake
callback instead of a real TTS/WebSocket round trip.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def await_with_filler(
    awaitable: Awaitable[T], threshold_s: float, on_timeout: Callable[[], Awaitable[None]]
) -> T:
    """Races `awaitable` against `threshold_s`. If it has not finished by
    then, awaits `on_timeout()` (the filler) exactly once, then keeps
    waiting for `awaitable`'s real result. Never calls `on_timeout` if
    `awaitable` finishes first -- suppressed, per the story's own
    wording, not merely raced and both sides shown."""
    task = asyncio.ensure_future(awaitable)
    done, _pending = await asyncio.wait({task}, timeout=threshold_s)
    if task not in done:
        await on_timeout()
    return await task
