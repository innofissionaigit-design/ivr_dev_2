"""The orchestrator's startup warm-up of the intent model must use a deadline that fits a cold load.

Found on the live pod: since KCD-465 made the per-turn extraction deadline 12 s, the startup warm-up (which called
extract_intent with the default) timed out on every restart; the client disconnected, Ollama abandoned the 47-74 s
load, and the first caller after a restart paid the cold start. Both entrypoints logged
"intent model warmup failed ... within its 12.0s deadline".

    python -m pytest tests/test_startup_warmup.py -v
"""

import asyncio
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _pod_stubs import pod_stubs

from agent.llm import DEFAULT_DEADLINE_S

COLD_LOAD_MEASURED_S = 74.0  # agent/llm.py: a cold 7B load on the pod


@pytest.fixture(scope="module")
def m():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


def test_the_warmup_deadline_is_longer_than_a_measured_cold_load_and_than_a_turns(m):
    assert m.INTENT_WARMUP_DEADLINE_S > COLD_LOAD_MEASURED_S
    assert m.INTENT_WARMUP_DEADLINE_S > DEFAULT_DEADLINE_S


def test_the_warmup_passes_that_deadline_to_the_extractor(m, monkeypatch):
    seen = {}

    def fake_extract(text, max_retries=2, lang="bn", deadline_s=DEFAULT_DEADLINE_S):
        seen["deadline_s"] = deadline_s
        return {"intent": "smalltalk"}, {"total_time_s": 0.1}

    monkeypatch.setattr(m, "extract_intent", fake_extract)
    assert asyncio.run(m._warm_intent_model()) is True
    assert seen["deadline_s"] == m.INTENT_WARMUP_DEADLINE_S


def test_a_failed_warmup_is_logged_and_never_stops_startup(m, monkeypatch):
    def failing(*a, **k):
        raise RuntimeError("ollama is not up")

    monkeypatch.setattr(m, "extract_intent", failing)
    assert asyncio.run(m._warm_intent_model()) is False


def test_the_startup_sequence_calls_the_warmup_helper_not_the_default_deadline(m):
    import inspect

    src = inspect.getsource(m._startup)
    assert "_warm_intent_model()" in src
    assert 'extract_intent, "' not in src
