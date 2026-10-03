"""tests/_virtual_time.py: the virtual-time loop the timing-sensitive tests run on.

python -m pytest tests/test_virtual_time.py -v
"""

import asyncio
import os
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

from _virtual_time import run_virtual, virtual_time


def test_a_long_sleep_costs_no_wall_clock_time():
    async def main():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await asyncio.sleep(30)
        return loop.time() - t0

    real0 = time.monotonic()
    assert run_virtual(main()) == pytest.approx(30.0, abs=1e-6)
    assert time.monotonic() - real0 < 2.0


@virtual_time
async def test_machine_load_cannot_change_measured_loop_time():
    # A blocking stall inside a coroutine is exactly what heavy neighbours do to
    # a real-time test. The loop clock does not move while the CPU is busy.
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    time.sleep(0.3)  # real, blocking
    assert loop.time() - t0 == pytest.approx(0.0, abs=1e-9)


@virtual_time
async def test_timers_fire_in_order_and_at_their_exact_times():
    loop = asyncio.get_running_loop()
    fired = []

    async def at(delay, tag):
        await asyncio.sleep(delay)
        fired.append((tag, round(loop.time(), 6)))

    await asyncio.gather(at(0.10, "b"), at(0.02, "a"), at(0.30, "c"))
    assert fired == [("a", 0.02), ("b", 0.10), ("c", 0.30)]


@virtual_time
async def test_cancellation_and_timeouts_behave_normally():
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.sleep(5), timeout=1.0)
    task = asyncio.ensure_future(asyncio.sleep(10))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
