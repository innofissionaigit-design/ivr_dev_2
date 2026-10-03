"""A virtual-time asyncio event loop: sleeping costs no wall-clock time and the
result cannot depend on how busy the machine is.

WHY. tests/test_filler.py raced real asyncio.sleep() calls against a wall-clock
bound (`elapsed < 0.2`). It passed alone and failed when the suite shared the
machine with heavy numpy work: a scheduling stall of a few dozen milliseconds
is enough to turn "0.15 s of sleeping" into "0.21 s of measured time". A test
that asserts on ELAPSED WALL TIME measures the machine, not the code.

HOW. This is a small discrete-event simulator. The loop's clock is a number it
owns. Whenever the loop would BLOCK waiting for the next timer, the selector
wrapper advances that number by exactly the wait and returns at once, so
`await asyncio.sleep(0.15)` completes instantly in real time and takes exactly
0.15 in loop time. Everything else -- task scheduling, ordering, cancellation --
is the stock asyncio machinery, so the code under test runs unmodified.

Consequences worth having:
  * deterministic: two timers 0.02 s apart fire in that order every time, under
    any load;
  * exact: elapsed loop time is 0.15, not "less than 0.2", so a regression that
    sums two waits (0.22) is caught by a tight bound instead of hiding inside a
    generous one;
  * fast: a test that "waits" 30 seconds takes microseconds.

Real I/O still works (the selector is consulted with a zero timeout), but a
test should not mix real sockets with virtual time.

    @virtual_time
    async def test_something(): ...        # runs on a VirtualClockLoop
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import selectors


class _JumpingSelector:
    """Wraps the real selector: a blocking wait becomes a clock jump."""

    def __init__(self, loop: VirtualClockLoop, real: selectors.BaseSelector):
        self._loop = loop
        self._real = real

    def select(self, timeout=None):
        if timeout is None:
            # No timers and nothing ready: a real loop would wait forever for I/O
            # that a virtual-time test never produces. Poll once and let the
            # loop go idle instead of hanging the test run.
            return self._real.select(0)
        if timeout > 0:
            self._loop._virtual_now += timeout
        return self._real.select(0)

    def __getattr__(self, name):
        return getattr(self._real, name)


class VirtualClockLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__()
        self._virtual_now = 0.0
        self._selector = _JumpingSelector(self, self._selector)

    def time(self) -> float:
        return self._virtual_now


def run_virtual(coro):
    """Run `coro` to completion on a fresh VirtualClockLoop; return its result."""
    loop = VirtualClockLoop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def virtual_time(fn):
    """Decorator: run an `async def` test on virtual time. The wrapper is an
    ordinary (sync) test function, so it needs no asyncio plugin."""
    if not inspect.iscoroutinefunction(fn):
        raise TypeError("virtual_time wraps coroutine functions")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return run_virtual(fn(*args, **kwargs))

    # pytest inspects the signature for fixtures; keep the original's
    wrapper.__signature__ = inspect.signature(fn)
    return wrapper
