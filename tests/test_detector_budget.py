"""KCD-076: agent/detector_budget.py. Async, but no network/GPU --
just proving the timeout/degrade behaviour with real (short) sleeps.

    python -m pytest tests/test_detector_budget.py -v
"""

import os
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.detector_budget import run_within_budget, snapshot


def _fast() -> str:
    return "fresh"


def _slow() -> str:
    time.sleep(0.3)
    return "too late"


@pytest.mark.asyncio
async def test_a_fast_detector_returns_its_own_result():
    result = await run_within_budget("unit_test_fast", 0.2, _fast, previous_value="stale")
    assert result == "fresh"


@pytest.mark.asyncio
async def test_a_slow_detector_degrades_to_the_previous_value_without_waiting():
    t0 = time.monotonic()
    result = await run_within_budget("unit_test_slow", 0.05, _slow, previous_value="stale")
    elapsed = time.monotonic() - t0
    assert result == "stale"  # degraded, not the slow function's real answer
    assert elapsed < 0.2  # the TURN was not held up for the full 0.3s sleep


@pytest.mark.asyncio
async def test_a_slow_detector_with_no_previous_value_degrades_to_none():
    result = await run_within_budget("unit_test_slow_none", 0.05, _slow)
    assert result is None


@pytest.mark.asyncio
async def test_snapshot_reports_timings_and_over_budget_counts():
    await run_within_budget("unit_test_snapshot", 0.2, _fast)
    await run_within_budget("unit_test_snapshot", 0.05, _slow, previous_value=None)
    snap = snapshot()
    assert snap["unit_test_snapshot"]["samples"] >= 1
    assert snap["unit_test_snapshot"]["over_budget_count"] >= 1
