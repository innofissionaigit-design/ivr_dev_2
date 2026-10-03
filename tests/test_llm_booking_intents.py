"""agent/llm.py's schema extension for Epic E26's new intents. Only
exercises the offline, deterministic parts (VALID_INTENTS, _validate) --
no Ollama call, so this runs anywhere.

    python -m pytest tests/test_llm_booking_intents.py -v
"""

from agent.llm import VALID_INTENTS, _validate


def test_new_booking_intents_are_recognised():
    for intent in (
        "book_test",
        "reschedule_appointment",
        "cancel_appointment",
        "lookup_booking",
        "add_test_booking",
        "resend_confirmation",
        "department_query",
    ):
        assert intent in VALID_INTENTS


def _full_slots(**overrides) -> dict:
    slots = {
        k: None
        for k in (
            "test_name",
            "test_names",
            "doctor_name",
            "date",
            "time_slot",
            "new_date",
            "new_time_slot",
            "confirmation_id",
            "patient_name",
            "patient_age",
            "phone",
            "contact_phone",
            "relationship",
            "spelled_letters",
            "symptom_description",
            "faq_topic",
        )
    }
    slots.update(overrides)
    return slots


def test_reschedule_intent_validates():
    data = {
        "intent": "reschedule_appointment",
        "slots": _full_slots(confirmation_id="KCD-1", new_date="2026-10-05", new_time_slot="18:15"),
        "direct_reply_bn": None,
    }
    ok, errors = _validate(data)
    assert ok, errors


def test_department_query_validates_with_symptom_description():
    data = {
        "intent": "department_query",
        "slots": _full_slots(symptom_description="chest pain since morning"),
        "direct_reply_bn": None,
    }
    ok, errors = _validate(data)
    assert ok, errors


def test_invalid_intent_is_rejected():
    data = {"intent": "delete_all_bookings", "slots": _full_slots(), "direct_reply_bn": None}
    ok, errors = _validate(data)
    assert not ok
    assert any("invalid intent" in e for e in errors)


def test_model_overstepping_direct_reply_on_a_non_smalltalk_intent_is_stripped():
    data = {
        "intent": "book_test",
        "slots": _full_slots(test_names=["Uric Acid"]),
        "direct_reply_bn": "your test costs 250 taka",
    }
    ok, _ = _validate(data)
    assert ok
    assert data["direct_reply_bn"] is None  # the model never gets to state a fact


# ===================================================== single-item list collapse
# CodeRabbit-flagged, real bug: Ollama's JSON mode only enforces valid
# JSON, not this schema's shape. A documented single-item-list failure
# mode is the model returning a bare string instead of a one-element
# list -- agent/booking_flow.merge_slots then iterates the STRING's
# individual characters as if each were a separate test name.


def test_a_single_test_name_returned_as_a_bare_string_is_wrapped_not_iterated():
    data = {"intent": "book_test", "slots": _full_slots(test_names="CBC"), "direct_reply_bn": None}
    ok, _ = _validate(data)
    assert ok
    assert data["slots"]["test_names"] == ["CBC"]  # never ["C", "B", "C"]


def test_a_normal_test_names_list_is_left_untouched():
    data = {"intent": "book_test", "slots": _full_slots(test_names=["CBC", "Lipid Profile"]), "direct_reply_bn": None}
    ok, _ = _validate(data)
    assert ok
    assert data["slots"]["test_names"] == ["CBC", "Lipid Profile"]


def test_spelled_letters_returned_as_a_bare_string_is_split_into_letters():
    data = {"intent": "book_appointment", "slots": _full_slots(spelled_letters="ravi"), "direct_reply_bn": None}
    ok, _ = _validate(data)
    assert ok
    assert data["slots"]["spelled_letters"] == ["r", "a", "v", "i"]


def test_a_non_string_garbage_value_in_a_list_slot_is_dropped_to_null():
    data = {"intent": "book_test", "slots": _full_slots(test_names=[1, 2, 3]), "direct_reply_bn": None}
    ok, _ = _validate(data)
    assert ok
    assert data["slots"]["test_names"] is None
