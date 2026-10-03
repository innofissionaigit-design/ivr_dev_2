"""KCD-103: several fields in one question, except under the senior, distress or emergency policy. Pure and offline.

    python -m pytest tests/test_slot_grouping.py -v

The "median turns to complete a booking falls" claim is shown here on SCRIPTED callers (a fixed set of behaviours,
below), not on real ones; the orchestrator wiring is tested in tests/test_orchestrator_booking_flow.py.
"""

import os
import statistics
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.booking_flow import merge_slots, missing_required, new_state
from agent.slot_grouping import MAX_FIELDS_PER_QUESTION, grouped_prompt, next_fields
from agent.speech_policy import check_reply, count_questions, derive_policy

FULL = {
    "doctor_name": "Sen",
    "date": "2026-10-01",
    "time_slot": "10:00",
    "patient_name": "Ravi Das",
    "phone": "9876543210",
}
FULL_TEST = {"test_names": ["CBC"], "date": "2026-10-01", "patient_name": "Ravi Das", "phone": "9876543210"}


def _policy(name):
    return derive_policy("neutral" if name == "normal" else name)


# ============================================================================================== which fields


def test_the_first_missing_field_decides_the_group_and_only_that_groups_missing_fields_are_asked():
    m = ["doctor_name", "date", "time_slot", "patient_name", "phone"]
    assert next_fields("book_appointment", m, 2) == ["doctor_name", "date", "time_slot"]
    assert next_fields("book_appointment", m[1:], 2) == ["date", "time_slot"]  # the doctor is known
    assert next_fields("book_appointment", m[3:], 2) == ["patient_name", "phone"]
    assert next_fields("book_appointment", ["phone"], 2) == ["phone"]
    assert next_fields("book_test", ["test_names", "date", "patient_name", "phone"], 2) == ["test_names", "date"]


def test_a_group_never_spans_two_groups_or_exceeds_the_maximum():
    got = next_fields("book_appointment", ["doctor_name", "date", "time_slot", "patient_name", "phone"], 2)
    assert "patient_name" not in got and len(got) <= MAX_FIELDS_PER_QUESTION


@pytest.mark.parametrize(
    "action", ["reschedule_appointment", "cancel_appointment", "add_test_booking", "lookup_booking"]
)
def test_actions_with_no_group_ask_one_field(action):
    assert next_fields(action, ["confirmation_id", "new_date"], 2) == ["confirmation_id"]


def test_nothing_missing_asks_nothing():
    assert next_fields("book_appointment", [], 2) == []


# ============================================================ the caller-state table always wins over grouping


@pytest.mark.parametrize("state", ["senior", "distressed", "confused", "angry"])
def test_a_policy_that_caps_questions_at_one_gets_exactly_one_field(state):
    policy = _policy(state)
    assert policy.questions_per_turn <= 1, "this test assumes the state's policy caps questions"
    m = ["doctor_name", "date", "time_slot", "patient_name", "phone"]
    assert next_fields("book_appointment", m, policy.questions_per_turn) == ["doctor_name"]
    assert next_fields("book_test", ["test_names", "date"], policy.questions_per_turn) == ["test_names"]


def test_the_emergency_policy_asks_nothing_grouped():
    assert next_fields("book_appointment", ["doctor_name", "date", "time_slot"], 0) == ["doctor_name"]


def test_a_field_already_asked_is_asked_alone_the_next_time():
    m = ["date", "time_slot", "patient_name", "phone"]
    assert next_fields("book_appointment", m, 2, asked={"date", "time_slot"}) == ["date"]
    assert next_fields("book_appointment", m, 2, asked={"time_slot"}) == ["date"]  # any repeat means: one at a time
    assert next_fields("book_appointment", m[2:], 2, asked={"date", "time_slot"}) == ["patient_name", "phone"]


# ==================================================================================================== wording


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
@pytest.mark.parametrize(
    "fields",
    [
        ["doctor_name", "date", "time_slot"],
        ["date", "time_slot"],
        ["doctor_name", "date"],
        ["doctor_name", "time_slot"],
        ["patient_name", "phone"],
        ["test_names", "date"],
    ],
)
def test_every_group_has_wording_in_every_language_and_is_one_question(fields, lang):
    text = grouped_prompt(fields, lang)
    assert text, (fields, lang)
    assert count_questions(text) == 1, text
    assert check_reply(text, _policy("normal")) == [], text  # inside the normal policy's two questions
    assert ":" not in text and "[" not in text and "{" not in text  # KCD-454: no label-and-colon artefacts


def test_the_english_wording_reads_as_one_natural_clause():
    assert (
        grouped_prompt(["doctor_name", "date", "time_slot"], "en")
        == "For the appointment, which doctor, which day and what time?"
    )
    assert grouped_prompt(["date", "time_slot"], "en") == "For the appointment, which day and what time?"
    assert (
        grouped_prompt(["patient_name", "phone"], "en")
        == "May I have the patient's name and a phone number for the confirmation?"
    )


def test_a_single_field_or_an_unknown_combination_has_no_grouped_wording():
    assert grouped_prompt(["date"], "en") is None
    assert grouped_prompt(["date", "phone"], "en") is None  # not a group: the caller asks the first field alone
    assert grouped_prompt(["date", "time_slot"], "ta") is None


def test_the_grouped_wording_carries_no_sentence_longer_than_the_normal_policy_allows():
    for lang in ("bn", "hi", "en"):
        for fields in (["doctor_name", "date", "time_slot"], ["patient_name", "phone"]):
            assert check_reply(grouped_prompt(fields, lang), _policy("normal")) == []


# ================================================= median turns to complete a booking falls (scripted callers)


def _answer(persona, fields, all_slots, state):
    """What a scripted caller says in reply to a question about `fields`."""
    if persona == "cooperative":
        return {f: all_slots[f] for f in fields if f in all_slots}
    if persona == "partial":  # answers only the first thing asked
        return {fields[0]: all_slots[fields[0]]}
    raise AssertionError(persona)


def _questions_to_complete(action, opening, persona, questions_per_turn, all_slots):
    """How many questions the agent asks before every required field is present."""
    st = new_state(action)
    merge_slots(st, opening)
    asked, n = set(), 0
    while missing_required(st) and n < 30:
        fields = next_fields(action, missing_required(st), questions_per_turn, asked)
        asked.update(fields)
        n += 1
        answer = _answer(persona, fields, all_slots, st)
        merge_slots(st, {k: v for k, v in answer.items() if k != "test_names"})
        if "test_names" in answer:
            st.test_names.extend(answer["test_names"])
    assert not missing_required(st), "the simulation must terminate with a complete booking"
    return n


CALLERS = [  # (opening utterance's slots, how they answer)
    ({}, "cooperative"),
    ({}, "partial"),
    ({"doctor_name": "Sen"}, "cooperative"),
    ({"doctor_name": "Sen"}, "partial"),
    ({"doctor_name": "Sen", "date": "2026-10-01", "time_slot": "10:00"}, "cooperative"),
    (dict(FULL), "cooperative"),
]


def test_median_questions_to_complete_a_booking_falls_against_one_field_per_turn():
    baseline = [_questions_to_complete("book_appointment", o, p, 1, FULL) for o, p in CALLERS]
    grouped = [_questions_to_complete("book_appointment", o, p, 2, FULL) for o, p in CALLERS]
    assert statistics.median(grouped) < statistics.median(baseline), (baseline, grouped)
    assert all(g <= b for g, b in zip(grouped, baseline)), (baseline, grouped)  # never worse for any caller
    assert sum(grouped) < sum(baseline)


def test_the_same_holds_for_a_test_booking():
    callers = [({}, "cooperative"), ({}, "partial"), ({"test_names": ["CBC"]}, "cooperative")]
    baseline = [_questions_to_complete("book_test", o, p, 1, FULL_TEST) for o, p in callers]
    grouped = [_questions_to_complete("book_test", o, p, 2, FULL_TEST) for o, p in callers]
    assert all(g <= b for g, b in zip(grouped, baseline)) and sum(grouped) < sum(baseline), (baseline, grouped)


@pytest.mark.parametrize("state", ["senior", "distressed"])
def test_under_a_one_question_policy_the_turn_count_is_exactly_the_baseline(state):
    q = _policy(state).questions_per_turn
    for opening, persona in CALLERS:
        assert _questions_to_complete("book_appointment", opening, persona, q, FULL) == _questions_to_complete(
            "book_appointment", opening, persona, 1, FULL
        )


def test_a_partial_answer_is_accepted_and_only_what_is_missing_is_asked_next():
    st = new_state("book_appointment")
    merge_slots(st, {"doctor_name": "Sen"})
    asked = set()
    fields = next_fields("book_appointment", missing_required(st), 2, asked)
    assert fields == ["date", "time_slot"]
    asked.update(fields)
    merge_slots(st, {"date": "2026-10-01"})  # the caller gave the day only
    assert next_fields("book_appointment", missing_required(st), 2, asked) == ["time_slot"]
    assert st.slots["doctor_name"] == "Sen" and st.slots["date"] == "2026-10-01"  # nothing captured was lost
