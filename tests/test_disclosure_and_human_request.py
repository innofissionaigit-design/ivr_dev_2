"""KCD-353: the automated-assistant disclosure (voice greeting, text channel) and the
immediate honouring of a request for a person.

The disclosure WORDING is draft and pending clinical and legal review; these tests
prove the wording is present everywhere it must be, versioned, and consistent -- not
that it is the right wording.

    python -m pytest tests/test_disclosure_and_human_request.py -v
"""

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API = os.path.join(REPO_ROOT, "clinic-api")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import disclosure, persona
from agent.human_request import asks_for_a_person
from agent.phrases import PHRASES, phrase


def _clinic_disclosure():
    spec = importlib.util.spec_from_file_location("clinic_disclosure", os.path.join(CLINIC_API, "disclosure.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ spoken disclosure


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_every_greeting_says_who_the_caller_is_speaking_with(lang):
    # DELIBERATE SPEC CHANGE (owner's instruction, 2026-09-25): the greeting carries the identity sentence only; the
    # automated-assistant disclosure stays available (below, and on the text channel) but is not in the greeting.
    from agent.phrases import GREETING_IDENTITY

    assert GREETING_IDENTITY[lang] in phrase("greeting", lang)


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_the_greeting_still_ends_on_the_question_that_hands_over_the_floor(lang):
    from agent.phrases import GREETING_IDENTITY

    greeting = phrase("greeting", lang)
    assert greeting.rstrip().endswith("?")
    assert greeting.index(GREETING_IDENTITY[lang]) < len(greeting) - 5  # the identity precedes the question


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_the_disclosure_wording_obeys_the_persona(lang):
    assert persona.is_clean(disclosure.disclosure_for(lang), lang)


def test_the_wording_is_versioned_and_says_it_is_unreviewed():
    assert disclosure.DISCLOSURE_VERSION and disclosure.DISCLOSURE_VERSION.endswith("draft")
    assert disclosure.REVIEW_STATUS == "pending_clinical_and_legal_review"


def test_a_greeting_with_no_second_sentence_still_gets_the_disclosure():
    assert disclosure.disclosure_for("en") in disclosure.insert_into_greeting("Hello", "en")


def test_the_reverify_notice_exists_in_all_languages_and_accuses_no_one():
    for lang in ("bn", "hi", "en"):
        text = phrase("reverify_notice", lang)
        assert persona.is_clean(text, lang)
    assert "impersonat" not in PHRASES["en"]["reverify_notice"].lower()


# ------------------------------------------------------------------ text channel


def test_the_text_channel_carries_the_notice_once():
    d = _clinic_disclosure()
    msg = d.with_notice("Your appointment (KC123) is confirmed.")
    assert d.TEXT_NOTICE in msg and msg.count(d.TEXT_NOTICE) == 1
    assert d.with_notice(msg) == msg  # idempotent


def test_the_voice_and_text_disclosures_share_a_version():
    assert _clinic_disclosure().DISCLOSURE_VERSION == disclosure.DISCLOSURE_VERSION


def test_every_queued_sms_says_it_is_automated(tmp_path, monkeypatch):
    fd = os.path.join(str(tmp_path), "t.db")
    monkeypatch.setenv("CLINIC_DB_PATH", fd)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    if CLINIC_API not in sys.path:
        sys.path.insert(0, CLINIC_API)
    for mod in (
        "main",
        "db",
        "models",
        "seed",
        "booking_service",
        "booking_migrate",
        "enquiry_migrate",
        "i18n_content",
        "disclosure",
    ):
        sys.modules.pop(mod, None)
    import db as db_mod
    import models

    models.Base.metadata.create_all(db_mod.engine)
    import booking_service as bs
    import disclosure as d

    session = db_mod.SessionLocal()
    try:
        assert bs.queue_sms(session, "9000000001", "booking_confirmed", "Your booking is confirmed.")["queued"]
        row = session.query(models.SmsOutbox).one()
        assert d.TEXT_NOTICE in row.message and row.message.startswith("Your booking is confirmed.")
    finally:
        session.close()


# ------------------------------------------------------------------ a request for a person


@pytest.mark.parametrize(
    "text,lang",
    [
        ("I want to talk to a person", "en"),
        ("can I speak to someone please", "en"),
        ("connect me to your staff", "en"),
        ("get me a human being", "en"),
        ("I want a real person", "en"),
        ("you are a robot, I want the operator", "en"),
        ("আমি একজন মানুষের সাথে কথা বলতে চাই", "bn"),
        ("কাউকে দিন", "bn"),
        ("স্টাফের সঙ্গে কথা বলব", "bn"),
        ("অপারেটর চাই", "bn"),
        ("मुझे किसी इंसान से बात करनी है", "hi"),
        ("स्टाफ से बात कराइए", "hi"),
        ("किसी को दीजिए", "hi"),
        ("मुझे असली इंसान चाहिए", "hi"),
    ],
)
def test_a_request_for_a_person_is_recognised_in_every_language(text, lang):
    assert asks_for_a_person(text, lang), text


@pytest.mark.parametrize(
    "text,lang",
    [
        ("I want to book an appointment with the doctor", "en"),
        ("what is the price of a CBC test", "en"),
        ("I need a test at the counter", "en"),
        ("I want to know about your staff timings", "en"),
        ("I don't need to talk to a person, just tell me the price", "en"),
        ("আমি ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট চাই", "bn"),
        ("সিবিসি টেস্টের দাম কত", "bn"),
        ("মানুষের সাথে কথা বলতে চাই না", "bn"),
        ("मुझे डॉक्टर के साथ अपॉइंटमेंट चाहिए", "hi"),
        ("सीबीसी टेस्ट की कीमत क्या है", "hi"),
        ("", "en"),
    ],
)
def test_ordinary_requests_and_explicit_refusals_are_not_taken_for_one(text, lang):
    assert not asks_for_a_person(text, lang), text


def test_a_request_in_a_language_other_than_the_calls_is_still_caught():
    assert asks_for_a_person("I want to talk to a person", "bn")
    assert asks_for_a_person("আমি একজন মানুষের সাথে কথা বলতে চাই", "en")
