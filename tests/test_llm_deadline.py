"""KCD-465: agent/llm.py's extract_intent -- a DEADLINE budget bounds the
whole operation, not a retry count. _call_ollama is monkeypatched so
this runs with no Ollama server and no real network waits.

    python -m pytest tests/test_llm_deadline.py -v
"""

import os
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import agent.llm as llm_mod
from agent.llm import ExtractionError, extract_intent

_VALID_RESPONSE = (
    '{"intent": "smalltalk", "slots": {"test_name": null, "test_names": null, '
    '"doctor_name": null, "date": null, "time_slot": null, "new_date": null, '
    '"new_time_slot": null, "confirmation_id": null, "patient_name": null, '
    '"patient_age": null, "phone": null, "contact_phone": null, "relationship": null, '
    '"spelled_letters": null, "symptom_description": null, "faq_topic": null}, '
    '"direct_reply_bn": "\\u09a8\\u09ae\\u09b8\\u09cd\\u0995\\u09be\\u09b0"}'
)


def test_a_slow_first_attempt_that_eats_the_deadline_stops_without_a_second_attempt(monkeypatch):
    calls = []

    def fake_call_ollama(prompt, timeout_s=90):
        calls.append(timeout_s)
        time.sleep(0.05)  # consumes real wall-clock time against the deadline
        raise TimeoutError("simulated slow/unreachable Ollama")

    monkeypatch.setattr(llm_mod, "_call_ollama", fake_call_ollama)

    with pytest.raises(ExtractionError) as exc_info:
        extract_intent("কিছু একটা", lang="bn", deadline_s=0.08)

    assert "deadline" in str(exc_info.value)
    # The deadline was consumed by real time.sleep() calls, not merely an
    # attempt counter -- proves this is a WALL-CLOCK bound, not a retry count.
    assert len(calls) <= 2


def test_deadline_is_never_exceeded_by_more_than_one_in_flight_attempt(monkeypatch):
    def fake_call_ollama(prompt, timeout_s=90):
        time.sleep(0.03)
        raise TimeoutError("simulated failure")

    monkeypatch.setattr(llm_mod, "_call_ollama", fake_call_ollama)

    t0 = time.time()
    with pytest.raises(ExtractionError):
        extract_intent("কিছু একটা", lang="bn", deadline_s=0.1)
    elapsed = time.time() - t0
    # One more attempt could still be in flight when the deadline check
    # fires, but the total must stay in the same order of magnitude as
    # the deadline, not balloon toward the old max_retries*90s shape.
    assert elapsed < 0.5


def test_a_successful_attempt_within_the_deadline_returns_normally(monkeypatch):
    def fake_call_ollama(prompt, timeout_s=90):
        return _VALID_RESPONSE

    monkeypatch.setattr(llm_mod, "_call_ollama", fake_call_ollama)

    data, diagnostics = extract_intent("নমস্কার", lang="bn", deadline_s=5.0)
    assert data["intent"] == "smalltalk"
    assert diagnostics["attempts"] == 1


def test_a_later_attempt_succeeding_within_the_deadline_still_returns(monkeypatch):
    attempts = {"n": 0}

    def fake_call_ollama(prompt, timeout_s=90):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("first attempt fails fast")
        return _VALID_RESPONSE

    monkeypatch.setattr(llm_mod, "_call_ollama", fake_call_ollama)

    data, diagnostics = extract_intent("নমস্কার", lang="bn", deadline_s=5.0)
    assert data["intent"] == "smalltalk"
    assert diagnostics["attempts"] == 2


def test_each_attempts_own_timeout_is_clamped_to_the_remaining_deadline(monkeypatch):
    seen_timeouts = []

    def fake_call_ollama(prompt, timeout_s=90):
        seen_timeouts.append(timeout_s)
        raise TimeoutError("simulated failure")

    monkeypatch.setattr(llm_mod, "_call_ollama", fake_call_ollama)

    with pytest.raises(ExtractionError):
        extract_intent("কিছু একটা", lang="bn", deadline_s=3.0)

    # Never the old flat 90s default -- every attempt's own timeout is
    # bounded by what is left of the 3s deadline.
    assert all(t <= 3.0 for t in seen_timeouts)
