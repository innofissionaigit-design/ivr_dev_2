"""agent/doctor_personal_request.py -- asking for a doctor, not for the front desk.

The thing worth testing here is the BOUNDARY, not the recall. This guard sits
next to `agent/human_request.asks_for_a_person()` and they must not blur into
each other: a caller asking for the front desk is handed straight there, while a
caller asking for a doctor cannot be, because there is no clinician on the line.
So the groups below are: it fires on a doctor request in all four families
(including the Devanagari written new for this repository); it does NOT fire on
the two adjacent requests that look almost identical -- asking whether a doctor
is AVAILABLE, and asking to BOOK with one -- since misreading either would stop
a caller doing ordinary clinic business; it does not collide with the other
guards; and the reply declines honestly while still offering the appointment
that is the real route to a doctor.
"""

from __future__ import annotations

import pytest

from agent import persona
from agent.clinical_safety import is_clinical_interpretation
from agent.complaint import is_complaint
from agent.doctor_personal_request import detect_doctor_personal_request, is_doctor_personal_request
from agent.human_request import asks_for_a_person
from agent.phrases import PHRASES, phrase

LANGS = ("bn", "hi", "en")


@pytest.mark.parametrize(
    "text,family",
    [
        ("I want to speak to a doctor", "en"),
        ("connect me to a doctor", "en"),
        ("I want my doctor to call me", "en"),
        ("I want clinical reassurance", "en"),
        ("mujhe doctor se baat karni hai", "latin"),
        ("daktar amake call korun", "latin"),
        ("ডাক্তারের সাথে কথা বলতে চাই", "bn"),
        ("ডাক্তার আমাকে কল করুন", "bn"),
        ("डॉक्टर से बात करनी है", "hi"),
        ("डॉक्टर मुझे कॉल करें", "hi"),
        ("can Dr Sen speak to me", "named"),
        ("please ask Dr Sen to call me", "named"),
    ],
)
def test_a_request_for_a_doctor_personally_is_caught(text, family):
    matched, tag = detect_doctor_personal_request(text)
    assert matched is True
    assert tag == family


@pytest.mark.parametrize("lang", LANGS)
def test_every_language_has_its_own_phrase_list(lang):
    import agent.doctor_personal_request as m

    table = {"bn": m._BN, "hi": m._HI, "en": m._EN}[lang]
    assert len(table) >= 10, f"{lang} list is too thin to be real coverage"


# ------------------------------------------- the two look-alikes it must not eat
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        # availability -- an ordinary enquiry with a real answer behind it
        "is Dr Sen available tomorrow",
        "which days does Dr Sen sit",
        "ডক্টর সেন কি আজ আছেন",
        "डॉक्टर सेन कल हैं क्या",
        # booking -- the thing this guard's own reply offers to do
        "I want to book an appointment with Dr Sen",
        "book me with a doctor tomorrow",
        "ডাক্তার সেনের কাছে অ্যাপয়েন্টমেন্ট চাই",
        # unrelated
        "what is the price of a CBC test",
        "I want to speak to someone",
        "connect me to a person",
    ],
)
def test_availability_booking_and_front_desk_requests_are_left_alone(text):
    """A caller asking when a doctor sits, or to book with one, must keep their
    normal flow -- both have real answers. And the front-desk request belongs to
    agent/human_request.py, which hands off immediately; swallowing it here
    would replace a hand-off with a decline."""
    assert is_doctor_personal_request(text) is False


def test_this_guard_does_not_overlap_the_guards_around_it():
    """Each guard answers a different question, and main.py resolves them by a
    fixed order. The order is only meaningful if the inputs are distinct."""
    for text in (
        "I want to speak to a doctor",
        "ডাক্তার আমাকে কল করুন",
        "डॉक्टर मुझे कॉल करें",
        "can Dr Sen speak to me",
    ):
        assert not asks_for_a_person(text, "en")
        assert not is_complaint(text)
        assert not is_clinical_interpretation(text)


def test_a_doctors_name_is_never_captured():
    """The reply never repeats a name, so the detector has no business
    extracting one -- nothing a mis-extraction could disclose."""
    matched, tag = detect_doctor_personal_request("please ask Dr Sen to call me")
    assert (matched, tag) == (True, "named")
    assert "sen" not in str(tag).lower()


# --------------------------------------------------------------- the reply
@pytest.mark.parametrize("lang", LANGS)
def test_the_reply_exists_in_all_three_languages(lang):
    assert "doctor_personal_request" in PHRASES[lang]
    assert phrase("doctor_personal_request", lang).strip()


def test_phrase_coverage_is_still_identical_across_languages():
    assert sorted(PHRASES["bn"]) == sorted(PHRASES["hi"]) == sorted(PHRASES["en"])


@pytest.mark.parametrize("lang", LANGS)
def test_the_reply_passes_the_persona_and_asks_one_question(lang):
    text = phrase("doctor_personal_request", lang)
    assert persona.violations(text) == [], persona.violations(text)
    assert text.count("?") == 1


def test_the_english_reply_declines_the_connection_but_still_offers_the_appointment():
    """Both halves matter. Declining without an offer is a dead end; offering
    without declining implies a direct connection that does not exist."""
    text = phrase("doctor_personal_request", "en").lower()
    assert "cannot connect" in text
    assert "cannot promise" in text, "the named-doctor callback promise is not ruled out"
    assert "appointment" in text, "the real route to a doctor is not offered"


def test_the_english_reply_never_names_a_doctor_or_claims_a_transfer():
    text = phrase("doctor_personal_request", "en").lower()
    for overclaim in ("putting you through", "transferring you", "hold while i get", "will call you back shortly"):
        assert overclaim not in text
