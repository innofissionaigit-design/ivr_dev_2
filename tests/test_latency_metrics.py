"""KCD-469: agent/latency_metrics.py. Pure, offline.

python -m pytest tests/test_latency_metrics.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.latency_metrics import PerLanguageLatency


def test_records_are_grouped_by_language():
    tracker = PerLanguageLatency()
    tracker.record("bn", 0.5)
    tracker.record("bn", 0.6)
    tracker.record("hi", 2.0)
    snap = tracker.snapshot()
    assert snap["bn"]["samples"] == 2
    assert snap["hi"]["samples"] == 1


def test_percentiles_are_computed_per_language_independently():
    tracker = PerLanguageLatency()
    for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        tracker.record("bn", v)
    for v in [5.0, 6.0]:
        tracker.record("hi", v)
    snap = tracker.snapshot()
    assert snap["bn"]["p50_s"] < snap["hi"]["p50_s"]
    assert snap["hi"]["p95_s"] >= 5.0


def test_a_language_with_no_samples_is_simply_absent():
    tracker = PerLanguageLatency()
    tracker.record("bn", 1.0)
    assert "en" not in tracker.snapshot()


def test_window_caps_memory_and_keeps_only_recent_samples():
    tracker = PerLanguageLatency(window=5)
    for v in range(10):
        tracker.record("bn", float(v))
    assert tracker.snapshot()["bn"]["samples"] == 5
