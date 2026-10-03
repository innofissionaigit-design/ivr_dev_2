"""Unit tests for agent/enquiry_followup.py (KCD-395/KCD-396).

Pure functions, no DB/LLM/pod needed:

    python -m pytest tests/test_enquiry_followup.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.enquiry_followup import (
    ENQUIRY_INTENTS,
    EnquiryEntity,
    entities_from_turn,
    has_pronoun_reference,
    resolve_followup_slot,
)

# ============================================================= pronoun cues


def test_pronoun_reference_detected_per_language():
    assert has_pronoun_reference("ওটার জন্য কি ফাস্টিং লাগবে?", "bn")
    assert has_pronoun_reference("उसके लिए फास्टिंग चाहिए?", "hi")
    assert has_pronoun_reference("is fasting needed for that one?", "en")


def test_pronoun_reference_not_detected_on_ordinary_new_question():
    assert not has_pronoun_reference("লিপিড প্রোফাইলের রেট কত?", "bn")
    assert not has_pronoun_reference("uric acid test ka rate kya hai", "hi")
    assert not has_pronoun_reference("what is the price of CBC", "en")


# ==================================================== resolve_followup_slot


def test_resolves_single_unambiguous_candidate():
    last = [EnquiryEntity(kind="test_name", value="Complete Blood Count (CBC)")]
    slots, resolved = resolve_followup_slot(
        "test_prep", {"test_name": None}, "is fasting needed for that one?", "en", last
    )
    assert resolved is True
    assert slots["test_name"] == "Complete Blood Count (CBC)"


def test_does_not_resolve_without_a_pronoun_cue():
    last = [EnquiryEntity(kind="test_name", value="Uric Acid")]
    slots, resolved = resolve_followup_slot("test_prep", {"test_name": None}, "what about lipid profile", "en", last)
    assert resolved is False
    assert slots["test_name"] is None


def test_does_not_resolve_when_slot_already_filled():
    last = [EnquiryEntity(kind="test_name", value="Uric Acid")]
    slots, resolved = resolve_followup_slot("test_prep", {"test_name": "ECG"}, "prep for that one?", "en", last)
    assert resolved is False
    assert slots["test_name"] == "ECG"  # untouched, caller's own words win


def test_ambiguous_when_two_entities_of_the_same_kind_are_remembered():
    last = [EnquiryEntity(kind="test_name", value="CBC"), EnquiryEntity(kind="test_name", value="Lipid Profile")]
    slots, resolved = resolve_followup_slot(
        "test_prep", {"test_name": None}, "is fasting needed for that one?", "en", last
    )
    assert resolved is False  # must ask, never guess between the two


def test_wrong_kind_is_not_a_match():
    last = [EnquiryEntity(kind="doctor_name", value="Dr. Sen")]
    slots, resolved = resolve_followup_slot("test_rate", {"test_name": None}, "what's the price of it?", "en", last)
    assert resolved is False


def test_department_query_has_no_primary_slot_to_resolve():
    last = [EnquiryEntity(kind="test_name", value="CBC")]
    slots, resolved = resolve_followup_slot(
        "department_query", {"symptom_description": None}, "and for that?", "en", last
    )
    assert resolved is False


# ======================================================== entities_from_turn


def test_entities_from_a_single_question_turn():
    entities = entities_from_turn(("test_rate", {"test_name": "CBC"}))
    assert entities == [EnquiryEntity(kind="test_name", value="CBC")]


def test_entities_from_a_kcd395_two_question_turn_keeps_both():
    entities = entities_from_turn(
        ("test_rate", {"test_name": "CBC"}),
        ("doctor_availability", {"doctor_name": "Dr. Sen"}),
    )
    assert entities == [
        EnquiryEntity(kind="test_name", value="CBC"),
        EnquiryEntity(kind="doctor_name", value="Dr. Sen"),
    ]


def test_entities_from_turn_with_no_lookup_slot_is_empty():
    assert entities_from_turn(("department_query", {"symptom_description": "chest pain"})) == []


def test_department_query_is_a_valid_secondary_intent_target():
    assert "department_query" in ENQUIRY_INTENTS
    assert "book_appointment" not in ENQUIRY_INTENTS  # stateful actions excluded


# ============================================ KCD-396: three consecutive follow-ups
# "Tested on scripted transcripts with three consecutive follow-ups" --
# the story's own acceptance criterion. Simulates the session-memory loop
# main.py's dispatch performs each turn, without needing a live LLM/pod.


def test_three_consecutive_follow_ups_on_the_same_entity():
    last_entities: list[EnquiryEntity] = []

    # Turn 1: names the test explicitly.
    slots1 = {"test_name": "Lipid Profile"}
    slots1, resolved1 = resolve_followup_slot("test_rate", slots1, "লিপিড প্রোফাইলের রেট কত?", "bn", last_entities)
    assert resolved1 is False  # nothing to resolve -- caller named it themselves
    last_entities = entities_from_turn(("test_rate", slots1))

    # Turn 2: elliptical follow-up, no test named.
    slots2 = {"test_name": None}
    slots2, resolved2 = resolve_followup_slot("test_prep", slots2, "ওটার জন্য কি ফাস্টিং লাগবে?", "bn", last_entities)
    assert resolved2 is True
    assert slots2["test_name"] == "Lipid Profile"
    last_entities = entities_from_turn(("test_prep", slots2))

    # Turn 3: another elliptical follow-up, same remembered entity.
    slots3 = {"test_name": None}
    slots3, resolved3 = resolve_followup_slot("test_rate", slots3, "সেটার স্যাম্পল কী লাগবে?", "bn", last_entities)
    assert resolved3 is True
    assert slots3["test_name"] == "Lipid Profile"


def test_three_consecutive_follow_ups_switches_entity_mid_conversation():
    last_entities: list[EnquiryEntity] = []

    slots1 = {"test_name": "Uric Acid"}
    last_entities = entities_from_turn(("test_rate", slots1))

    slots2 = {"test_name": None}
    slots2, resolved2 = resolve_followup_slot("test_prep", slots2, "prep for that one?", "en", last_entities)
    assert resolved2 is True and slots2["test_name"] == "Uric Acid"
    last_entities = entities_from_turn(("test_prep", slots2))

    # Caller explicitly names a NEW test -- their own words win, no resolution attempted.
    slots3 = {"test_name": "ECG"}
    slots3, resolved3 = resolve_followup_slot("test_rate", slots3, "what about ECG", "en", last_entities)
    assert resolved3 is False
    assert slots3["test_name"] == "ECG"
