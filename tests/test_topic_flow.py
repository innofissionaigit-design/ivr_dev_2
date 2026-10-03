"""KCD-104: the pure decisions and wording behind "a topic change or a correction never loses the booking". Offline.

    python -m pytest tests/test_topic_flow.py -v

The orchestrator behaviour (answering the aside, keeping the state, the confirmation step, suspend and resume) is
tested at every stage of the booking flow in tests/test_orchestrator_booking_flow.py.
"""

import os
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import topic_flow as tf
from agent.booking_flow import STATE_IDLE_TIMEOUT_S, mark_confirming, merge_slots, new_state
from agent.enquiry_followup import ENQUIRY_INTENTS
from agent.speech_policy import check_reply, count_questions, derive_policy


def _state(action="book_appointment", stage="collecting", **slots):
    st = new_state(action)
    merge_slots(st, slots)
    st.stage = stage
    return st


# ============================================================================================== classify


def test_no_task_in_progress_means_nothing_to_protect():
    assert tf.classify(None, "test_rate") == tf.NO_TASK
    assert tf.classify(_state(stage="done", doctor_name="Sen"), "test_rate") == tf.NO_TASK


def test_a_stale_task_is_not_protected():
    st = _state(doctor_name="Sen")
    st.last_updated = time.monotonic() - STATE_IDLE_TIMEOUT_S - 1
    assert tf.classify(st, "test_rate") == tf.NO_TASK


@pytest.mark.parametrize("intent", sorted(ENQUIRY_INTENTS) + ["smalltalk", "lookup_booking", "resend_confirmation"])
def test_an_enquiry_or_small_talk_is_a_topic_change(intent):
    assert tf.classify(_state(doctor_name="Sen"), intent) == tf.TOPIC_CHANGE


def test_the_same_action_or_a_bare_answer_continues_the_task():
    st = _state(doctor_name="Sen")
    assert tf.classify(st, "book_appointment") == tf.CONTINUE
    assert tf.classify(st, "unclear") == tf.CONTINUE  # a bare "Ravi Das" the extractor could not label


@pytest.mark.parametrize("other", ["book_test", "reschedule_appointment", "cancel_appointment", "add_test_booking"])
def test_a_different_booking_action_is_a_new_task(other):
    assert tf.classify(_state(doctor_name="Sen"), other) == tf.NEW_TASK


def test_an_unknown_intent_leaves_the_task_alone():
    assert tf.classify(_state(doctor_name="Sen"), "something_new") == tf.CONTINUE


def test_every_enquiry_intent_is_covered_so_no_topic_change_is_missed():
    assert {"test_rate", "doctor_availability", "test_prep", "clinic_faq"} <= set(ENQUIRY_INTENTS)


# ============================================================================================ worth_suspending


def test_only_a_task_with_something_in_it_is_worth_offering_to_resume():
    assert not tf.worth_suspending(None)
    assert not tf.worth_suspending(_state())  # just started: nothing to lose
    assert tf.worth_suspending(_state(doctor_name="Sen"))
    st = _state()
    st.test_names.append("CBC")
    assert tf.worth_suspending(st)
    st = _state()
    st.hold_token = "abc"
    assert tf.worth_suspending(st)
    st = _state(doctor_name="Sen", date="2026-10-01", time_slot="10:00", patient_name="Ravi", phone="9")
    mark_confirming(st)
    assert tf.worth_suspending(st)
    assert not tf.worth_suspending(_state(stage="done", doctor_name="Sen"))


# ========================================================================================= when a resume line is allowed


def test_a_resume_line_is_a_question_so_it_needs_a_policy_that_allows_one_and_a_reply_that_does_not_ask():
    assert tf.resume_allowed(2, "The CBC costs 350 rupees.")
    assert tf.resume_allowed(1, "The CBC costs 350 rupees.")
    assert not tf.resume_allowed(0, "The CBC costs 350 rupees.")  # the emergency policy: no question
    assert not tf.resume_allowed(2, "I found no test by that name. Did you mean CBC?")
    assert not tf.resume_allowed(2, "यह टेस्ट नहीं मिला। क्या आप सीबीसी कहना चाहते हैं?".replace("?", "؟"))


# ============================================================================================== the wording


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_the_resume_lines_are_one_question_each_and_inside_the_policy(lang):
    normal, senior = derive_policy("neutral"), derive_policy("neutral", senior=True)
    for line in (
        tf.resume_line(
            "Which day would you like?" if lang == "en" else "কোন দিন?" if lang == "bn" else "किस दिन?", lang
        ),
        tf.confirm_resume_line(lang),
        tf.offer_resume("cancel_appointment", lang),
    ):
        assert count_questions(line) == 1, line
        assert check_reply(line, normal) == [], line
        assert check_reply(line, senior) == [], line  # one question, inside a senior caller's single question
        assert ":" not in line and "{" not in line and "[" not in line


def test_the_english_wording_is_natural():
    assert (
        tf.resume_line("Which day would you like the appointment for?", "en")
        == "Coming back to your booking. Which day would you like the appointment for?"
    )
    assert tf.confirm_resume_line("en") == "Coming back to your booking. Shall I confirm it?"
    assert (
        tf.offer_resume("book_appointment", "en")
        == "Earlier we were talking about booking an appointment. Shall I go back to it?"
    )
    assert (
        tf.offer_resume("reschedule_appointment", "en")
        == "Earlier we were talking about changing an appointment. Shall I go back to it?"
    )


@pytest.mark.parametrize("action", tf.BOOKING_ACTIONS)
@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_every_action_has_a_resume_offer_in_every_language(action, lang):
    text = tf.offer_resume(action, lang)
    assert text and "{" not in text


def test_an_unknown_language_falls_back_to_english_rather_than_raising():
    assert "Coming back" in tf.confirm_resume_line("ta")
