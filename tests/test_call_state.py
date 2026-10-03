"""KCD-061: agent/call_state.py -- the single, versioned carrier of
caller signals. Pure, offline.

    python -m pytest tests/test_call_state.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.call_state import (
    CALLER_STATE_NEUTRAL,
    SCHEMA_VERSION,
    VALID_CALLER_STATES,
    apply_confidence,
    apply_language,
    new_call_state,
)


def test_new_call_state_has_neutral_defaults():
    st = new_call_state()
    assert st.schema_version == SCHEMA_VERSION
    assert st.caller_state == CALLER_STATE_NEUTRAL
    assert st.senior is False
    assert st.confirmation_required is False
    assert st.speech_rate == 1.0
    assert st.one_question_at_a_time is True


def test_apply_language_updates_only_language():
    st = new_call_state()
    apply_language(st, "hi")
    assert st.language == "hi"
    assert st.caller_state == CALLER_STATE_NEUTRAL  # nothing else touched


def test_apply_confidence_sets_confirmation_required():
    st = new_call_state()
    apply_confidence(st, True)
    assert st.confirmation_required is True
    apply_confidence(st, False)
    assert st.confirmation_required is False


def test_to_log_dict_carries_no_personal_data_fields():
    st = new_call_state()
    d = st.to_log_dict()
    forbidden = {"name", "phone", "patient_name", "transcript", "text"}
    assert forbidden.isdisjoint(d.keys())
    assert d["language"] == "bn"


def test_caller_state_is_always_one_of_the_valid_categories():
    st = new_call_state()
    assert st.caller_state in VALID_CALLER_STATES
