"""agent/booking_flow.py: the cross-turn slot-filling state machine that
KCD-357, 358, 361 (relative-date reuse), 364, 366, 367, 368 all depend on.
Pure and offline -- no network, no database, no GPU.

    python -m pytest tests/test_booking_flow.py -v
"""

import time

import pytest

from agent.booking_flow import (
    classify_yes_no,
    correction_acknowledgement,
    effective_phone,
    is_ready_to_confirm,
    mark_confirming,
    merge_slots,
    merge_spelling,
    missing_required,
    new_state,
    reopen_for_correction,
    try_assemble_spelling,
)


def test_one_sentence_booking_fills_every_slot_at_once():
    # KCD-357: caller gives doctor, date, time, name, phone all in one turn.
    state = new_state("book_appointment")
    changed = merge_slots(
        state,
        {
            "doctor_name": "Sen",
            "date": "2026-10-01",
            "time_slot": "18:15",
            "patient_name": "Ravi",
            "phone": "9800000001",
        },
    )
    assert set(changed) == {"doctor_name", "date", "time_slot", "patient_name", "phone"}
    assert missing_required(state) == []
    assert is_ready_to_confirm(state)


def test_guided_flow_fills_one_slot_per_turn_without_forgetting_earlier_ones():
    # KCD-358: doctor only, then the rest across later turns.
    state = new_state("book_appointment")
    merge_slots(state, {"doctor_name": "Sen"})
    assert missing_required(state) == ["date", "time_slot", "patient_name", "phone"]

    merge_slots(state, {"date": "2026-10-01", "time_slot": "18:15"})
    assert state.slots["doctor_name"] == "Sen"  # not forgotten
    assert missing_required(state) == ["patient_name", "phone"]

    merge_slots(state, {"patient_name": "Ravi", "phone": "9800000001"})
    assert is_ready_to_confirm(state)


def test_changing_mind_mid_booking_updates_only_that_slot():
    # KCD-366: caller picks a different doctor halfway through.
    state = new_state("book_appointment")
    merge_slots(state, {"doctor_name": "Sen", "date": "2026-10-01", "time_slot": "18:15"})
    merge_slots(state, {"patient_name": "Ravi", "phone": "9800000001"})
    assert is_ready_to_confirm(state)

    mark_confirming(state)
    changed = merge_slots(state, {"doctor_name": "Ghosh"})
    assert changed == ["doctor_name"]
    assert state.slots["date"] == "2026-10-01"  # everything else kept
    assert state.slots["patient_name"] == "Ravi"
    assert state.stage == "collecting"  # re-opened, not committed blind


def test_correction_after_confirmation_reopens_collection_and_keeps_the_rest():
    # KCD-367: caller spots a mistake between "shall I confirm" and commit.
    state = new_state("book_appointment")
    merge_slots(
        state,
        {
            "doctor_name": "Sen",
            "date": "2026-10-01",
            "time_slot": "18:15",
            "patient_name": "Ravi",
            "phone": "9800000001",
        },
    )
    mark_confirming(state)
    assert state.stage == "confirming"

    reopen_for_correction(state, "phone", "9800000099")
    assert state.stage == "collecting"
    assert state.slots["phone"] == "9800000099"
    assert state.slots["doctor_name"] == "Sen"  # nothing else lost
    assert is_ready_to_confirm(state)  # ready again immediately


def test_multi_test_booking_accumulates_across_turns_without_duplicates():
    # KCD-365: several tests named across turns.
    state = new_state("book_test")
    merge_slots(state, {"test_names": ["Uric Acid", "Lipid Profile"]})
    merge_slots(state, {"test_names": ["Lipid Profile", "CBC"]})  # a repeat + a new one
    assert state.test_names == ["Uric Acid", "Lipid Profile", "CBC"]


def test_reschedule_and_cancel_and_add_test_have_their_own_required_sets():
    r = new_state("reschedule_appointment")
    merge_slots(r, {"confirmation_id": "KCD-1"})
    assert missing_required(r) == ["new_date", "new_time_slot"]

    c = new_state("cancel_appointment")
    assert missing_required(c) == ["confirmation_id"]

    a = new_state("add_test_booking")
    merge_slots(a, {"confirmation_id": "KCD-1"})
    assert missing_required(a) == ["test_name"]


def test_lookup_accepts_either_phone_or_confirmation_id():
    l1 = new_state("lookup_booking")
    assert missing_required(l1) == ["phone"]
    merge_slots(l1, {"phone": "9800000001"})
    assert missing_required(l1) == []

    l2 = new_state("lookup_booking")
    merge_slots(l2, {"confirmation_id": "KCD-1"})
    assert missing_required(l2) == []


def test_state_becomes_stale_after_the_idle_timeout(monkeypatch):
    state = new_state("book_appointment")
    assert not state.is_stale()
    state.last_updated = time.monotonic() - 999
    assert state.is_stale()


def test_classify_yes_no_across_languages():
    assert classify_yes_no("হ্যাঁ ঠিক আছে", "bn") == "yes"
    assert classify_yes_no("না না", "bn") == "no"
    assert classify_yes_no("हाँ बुक कर दीजिए", "hi") == "yes"
    assert classify_yes_no("नहीं रद्द करो", "hi") == "no"
    assert classify_yes_no("yes that's correct", "en") == "yes"
    assert classify_yes_no("no that's wrong", "en") == "no"
    assert classify_yes_no("what did you say", "en") is None


def test_classify_yes_no_checks_no_before_yes_for_negated_correct():
    # "that's not correct" contains "correct" (a yes-word) but must read as no.
    assert classify_yes_no("that's not correct", "en") == "no"
    assert classify_yes_no("ঠিক না", "bn") == "no"


def test_an_affirmative_cancel_confirmation_is_never_misread_as_no():
    # CodeRabbit-flagged, real bug: "cancel" used to be a NO-word, so a
    # caller confirming "shall I cancel your appointment?" with "yes,
    # cancel it" was told "okay, unchanged" -- the opposite of what they
    # asked. classify_yes_no is only ever asked during a confirming-stage
    # turn, so when the PENDING action is itself a cancellation, "cancel"
    # in the answer is affirmative, never negative.
    assert classify_yes_no("yes, cancel it", "en") == "yes"
    assert classify_yes_no("হ্যাঁ, ক্যানসেল করে দিন", "bn") == "yes"
    assert classify_yes_no("हाँ, कैंसल कर दीजिए", "hi") == "yes"


def test_a_bare_cancel_with_no_other_signal_is_unclear_not_a_guess():
    # Ambiguous on its own (could be "yes, [please] cancel" or a stray
    # utterance) -- asked again rather than guessed, the safe direction.
    assert classify_yes_no("cancel", "en") is None


def test_substring_false_positives_are_no_longer_misclassified():
    # "book it" contains "ok" as a raw substring; "I don't know" contains
    # "no" (via "know"). Both were real false positives under substring
    # matching -- fixed by token-based matching (_has_phrase).
    assert classify_yes_no("book it", "en") is None
    assert classify_yes_no("I don't know", "en") is None


def test_spelling_assembly_ignores_non_letters():
    assert try_assemble_spelling(["r", "a", "3", "v", "", "i"]) == "ravi"
    assert try_assemble_spelling(["", "  "]) is None


def test_spelling_merges_across_turns():
    state = new_state("book_appointment")
    merge_spelling(state, ["r", "a"])
    combined = merge_spelling(state, ["v", "i"])
    assert combined == "ravi"


def test_a_declined_phone_number_stops_blocking_the_booking():
    # KCD-370: refusal still completes the booking, never blocks forever.
    state = new_state("book_appointment")
    merge_slots(state, {"doctor_name": "Sen", "date": "2026-10-01", "time_slot": "18:15", "patient_name": "Ravi"})
    assert missing_required(state) == ["phone"]
    state.phone_declined = True
    assert missing_required(state) == []
    assert is_ready_to_confirm(state)


def test_effective_phone_prefers_contact_phone_then_falls_back_to_sentinel():
    state = new_state("book_appointment")
    assert effective_phone(state) == "not_provided"
    merge_slots(state, {"phone": "9800000001"})
    assert effective_phone(state) == "9800000001"
    merge_slots(state, {"contact_phone": "9800000099"})
    assert effective_phone(state) == "9800000099"


# ------------------------------------------------------------- KCD-453


def test_first_time_slot_fill_is_not_a_correction():
    state = new_state("book_appointment")
    prior = dict(state.slots)
    changed = merge_slots(state, {"doctor_name": "Sen"})
    assert correction_acknowledgement(changed, prior, state.slots, "en") is None


def test_changing_an_already_given_value_is_a_correction():
    state = new_state("book_appointment")
    merge_slots(state, {"doctor_name": "Sen", "date": "2026-10-01"})
    prior = dict(state.slots)
    changed = merge_slots(state, {"doctor_name": "Roy"})
    ack = correction_acknowledgement(changed, prior, state.slots, "en")
    assert ack is not None
    assert "Roy" in ack
    assert "Sen" not in ack  # the agent restates the new value, never defends the old one


def test_correction_acknowledgement_never_speaks_a_label_colon():
    state = new_state("book_appointment")
    merge_slots(state, {"patient_name": "Ravi", "phone": "9800000001"})
    prior = dict(state.slots)
    changed = merge_slots(state, {"phone": "9800000099"})
    for lang in ("bn", "hi", "en"):
        ack = correction_acknowledgement(changed, prior, state.slots, lang)
        assert ":" not in ack and "[" not in ack and "]" not in ack


def test_correction_acknowledgement_is_localised_per_language():
    state = new_state("book_appointment")
    merge_slots(state, {"date": "2026-10-01"})
    prior = dict(state.slots)
    changed = merge_slots(state, {"date": "2026-10-05"})
    assert "2026-10-05" in correction_acknowledgement(changed, prior, state.slots, "bn")
    assert "2026-10-05" in correction_acknowledgement(changed, prior, state.slots, "hi")
    assert "2026-10-05" in correction_acknowledgement(changed, prior, state.slots, "en")


def test_uncorrectable_internal_fields_are_never_acknowledged():
    # _spelling_buffer is not in _CORRECTABLE_FIELDS -- restating it would
    # be noise, and merge_slots itself never puts it in `changed` (it is
    # written directly by merge_spelling, not through merge_slots).
    state = new_state("book_appointment")
    merge_slots(state, {"symptom_description": "chest pain"})
    prior = dict(state.slots)
    changed = merge_slots(state, {"symptom_description": "shortness of breath"})
    assert correction_acknowledgement(changed, prior, state.slots, "en") is None


# ============================================================ KCD-484
# "No write proceeds without an affirmative confirmation on the same call."
# main.py commits only from _handle_booking_confirmation_turn on a
# classify_yes_no == "yes" (untestable off-pod: main.py cannot be
# imported here), so what CAN be pinned locally is the two halves it
# depends on: the state machine never reaches a committed stage on its
# own, and the classifier never reads an uncertain/negated answer as yes.


def test_a_fully_filled_booking_is_ready_to_confirm_but_never_committed():
    state = new_state("book_appointment")
    merge_slots(
        state,
        {
            "doctor_name": "Sen",
            "date": "2026-10-01",
            "time_slot": "18:15",
            "patient_name": "Ravi",
            "phone": "9800000001",
        },
    )
    assert is_ready_to_confirm(state)
    assert state.stage == "collecting", "filling every slot must not, by itself, reach a write stage"
    mark_confirming(state)
    assert state.stage == "confirming"
    assert not is_ready_to_confirm(state), "already awaiting the caller's answer, not re-askable as fresh"


def test_changing_a_value_while_confirming_reopens_collection():
    state = new_state("book_appointment")
    merge_slots(
        state,
        {
            "doctor_name": "Sen",
            "date": "2026-10-01",
            "time_slot": "18:15",
            "patient_name": "Ravi",
            "phone": "9800000001",
        },
    )
    mark_confirming(state)
    merge_slots(state, {"time_slot": "18:30"})
    assert state.stage == "collecting"
    assert state.slots["time_slot"] == "18:30" and state.slots["doctor_name"] == "Sen"


@pytest.mark.parametrize(
    "utterance",
    [
        "not sure",
        "I'm not sure",
        "not really sure",
        "don't confirm",
        "not okay",
        "no wait",
        "hmm",
        "wait a moment",
        "9800000001",
        "what",
        "nope",
    ],
)
def test_an_uncertain_or_negated_answer_is_never_yes(utterance):
    assert classify_yes_no(utterance, "en") != "yes"


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("yes", "yes"),
        ("yeah okay", "yes"),
        ("I am sure", "yes"),
        ("yes I'm sure", "yes"),
        ("okay confirm", "yes"),
        ("no", "no"),
        ("not correct", "no"),
        ("not okay", "no"),
        ("not sure", None),
        ("hmm", None),
    ],
)
def test_english_yes_no_classification_including_negation(utterance, expected):
    assert classify_yes_no(utterance, "en") == expected


def test_bengali_and_hindi_negations_are_never_yes():
    assert classify_yes_no("না", "bn") == "no"  # na
    assert classify_yes_no("হাঁ না", "bn") == "no"  # "yes no" -> refusal wins
    assert classify_yes_no("नहीं", "hi") == "no"  # nahin
    assert classify_yes_no("ठीक नहीं है", "hi") == "no"  # "theek nahin hai"
