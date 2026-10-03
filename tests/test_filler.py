"""KCD-461: agent/filler.py's race logic, run on VIRTUAL TIME
(tests/_virtual_time.py): the sleeps below cost no wall-clock time and the
outcome cannot depend on machine load. The last test used to assert
`elapsed < 0.2` on the wall clock and failed whenever the suite shared the
machine with heavy work; the same race between a 0.02 s lookup and a 0.1 s
threshold was exposed to the same stalls. The intent of every test is unchanged
-- only the clock is.

    python -m pytest tests/test_filler.py -v
"""

import asyncio
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _virtual_time import virtual_time

from agent.filler import await_with_filler


async def _fast_lookup() -> str:
    await asyncio.sleep(0.02)
    return "the real answer"


async def _slow_lookup() -> str:
    await asyncio.sleep(0.15)
    return "the real answer, eventually"


@virtual_time
async def test_filler_is_suppressed_when_the_result_arrives_first():
    filler_calls = []

    async def on_timeout():
        filler_calls.append(1)

    result = await await_with_filler(_fast_lookup(), threshold_s=0.1, on_timeout=on_timeout)
    assert result == "the real answer"
    assert filler_calls == []  # never spoken


@virtual_time
async def test_filler_fires_exactly_once_when_the_stage_exceeds_the_threshold():
    filler_calls = []

    async def on_timeout():
        filler_calls.append(1)

    result = await await_with_filler(_slow_lookup(), threshold_s=0.05, on_timeout=on_timeout)
    assert result == "the real answer, eventually"
    assert filler_calls == [1]  # exactly once, not repeated while still waiting


@virtual_time
async def test_the_real_result_is_still_returned_after_the_filler():
    async def on_timeout():
        pass

    result = await await_with_filler(_slow_lookup(), threshold_s=0.05, on_timeout=on_timeout)
    assert result == "the real answer, eventually"


@virtual_time
async def test_an_exception_from_the_awaitable_still_propagates():
    async def failing():
        await asyncio.sleep(0.02)
        raise ValueError("boom")

    async def on_timeout():
        pass

    with pytest.raises(ValueError, match="boom"):
        await await_with_filler(failing(), threshold_s=0.1, on_timeout=on_timeout)


@virtual_time
async def test_the_filler_does_not_delay_the_real_result_past_its_own_completion():
    # The filler callback itself takes real time (as speaking one would);
    # the total elapsed time should still be dominated by the slow
    # lookup, not doubled by waiting for the filler AND then restarting
    # the wait for the lookup from zero.
    async def slow_filler():
        await asyncio.sleep(0.02)

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await await_with_filler(_slow_lookup(), threshold_s=0.05, on_timeout=slow_filler)
    elapsed = loop.time() - t0
    # Exact, not "less than some slack": the filler runs from 0.05 to 0.07 while
    # the lookup keeps running, so the whole thing ends when the lookup does,
    # at 0.15. Summing the waits (0.05 + 0.02 + 0.15) would give 0.22.
    assert elapsed == pytest.approx(0.15, abs=1e-6)
