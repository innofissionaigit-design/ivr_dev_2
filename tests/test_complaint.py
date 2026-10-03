"""agent/complaint.py -- the detector, and the verbatim store behind it.

WHY THESE TESTS EXIST, AND WHAT EACH GROUP IS PROVING:

The ported story has five acceptance criteria, and three of them are the kind
that pass review by inspection and then fail in production:

  * criterion 1 (a complaint is recognised AS a complaint) fails silently. This
    repository's intent set has no "complaint" member at all, so an undetected
    complaint is not an error -- it is classified `smalltalk`, `clinic_faq` or
    `unclear` and answered as a cheerful enquiry. Nothing crashes; the caller is
    simply not heard. So the first group below is about recall, in all four
    phrase families, and the second is about the false positives that recall
    buys -- an enquiry misread as a complaint ends the call at a human.
  * criterion 3 (captured verbatim) is a claim about storage, not about speech,
    and the spoken line promises it out loud ("I have written your complaint
    down exactly as you described it"). A test that only checked the sentence
    was spoken would let the system lie. So the third group stores a complaint
    through the real service function and asserts the text came back byte for
    byte -- and that a retry does not open a second row, since the agent says
    "your complaint", singular.
  * criterion 5 (never defend the clinic, never explain the complaint away) is a
    property of the wording, which is where it can regress later when someone
    edits a phrase. The last group pins it on the actual strings in all three
    languages, through agent/apology.py's own counter rather than by eye.

tests/test_orchestrator_complaint.py covers the thing none of this can: that the
detector is REACHED, and reached before the two other guards that also match.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent import anger, complaint
from agent.apology import count_apologies
from agent.phrases import PHRASES, phrase

CLINIC_API = str(Path(__file__).resolve().parent.parent / "clinic-api")
if CLINIC_API not in sys.path:
    sys.path.insert(0, CLINIC_API)

LANGS = ("bn", "hi", "en")


@pytest.fixture
def db():
    """A real SQLite database with the real schema, created the way the service
    creates it at startup (Base.metadata.create_all) -- no migration is involved
    because `complaints` is a brand-new table, and this fixture is what proves
    that claim rather than leaving it as an assertion in a comment."""
    from models import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


# --------------------------------------------------------------- detection

# Both families the lists deliberately cover: naming the ACT of complaining,
# and an unambiguous statement about the clinic's conduct.
@pytest.mark.parametrize(
    "text,family",
    [
        ("I want to file a complaint", "en"),
        ("and I have a complaint about the billing", "en"),
        ("your staff were rude to me", "en"),
        ("I am very dissatisfied", "en"),
        ("mujhe shikayat darj karni hai", "latin"),
        ("ami obhijog korte chai", "latin"),
        ("staff ne bahut badtameezi ki", "latin"),
        ("আমার একটা অভিযোগ আছে", "bn"),
        ("আমি কমপ্লেইন করতে চাই", "bn"),
        ("স্টাফ খুব খারাপ ব্যবহার করেছে", "bn"),
        ("मुझे शिकायत करनी है", "hi"),
        ("मेरी शिकायत नोट कर लीजिए", "hi"),
        ("स्टाफ ने बहुत बदतमीजी की", "hi"),
    ],
)
def test_a_complaint_is_recognised_in_every_language_and_reports_which_list_matched(text, family):
    matched, tag = complaint.detect_complaint(text)
    assert matched is True
    assert tag == family
    assert complaint.is_complaint(text) is True


def test_a_complaint_added_mid_sentence_is_still_caught():
    """A caller routinely tacks a complaint onto something else. A substring list
    that assumed "I" came first missed every one of these, which is exactly the
    kind of miss criterion 1 fails silently on."""
    assert complaint.is_complaint("I also want to file a complaint about yesterday") is True
    assert complaint.is_complaint("okay, and I want to complain about the waiting time") is True


@pytest.mark.parametrize("lang", LANGS)
def test_every_language_has_its_own_phrase_list(lang):
    """The Hindi list did not exist in the module this was ported from. Without
    it, a Hindi caller's complaint is answered as an enquiry."""
    families = dict(complaint._FAMILIES)
    assert lang in families
    assert len(families[lang]) >= 10, f"{lang} list is too thin to be real coverage"


# ------------------------------------------------------------ no false fire

# Every one of these is an ordinary request. Misreading any of them as a
# complaint acknowledges something that was never said and ends the call at a
# human -- a worse failure than missing a mild complaint.
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "what is the price of a CBC test",
        "is Dr Sen available tomorrow",
        "I have a problem with my knee",
        "the report is bad news, what should I do",
        "আমার রিপোর্ট কোথায়",
        "মাথা খারাপ লাগছে",
        "मुझे रिपोर्ट चाहिए",
        "मेरी तबीयत खराब है",
        "amar report ta kothay",
        "I am not happy with the timing, can I change it",
    ],
)
def test_an_ordinary_request_is_not_a_complaint(text):
    assert complaint.is_complaint(text) is False
    assert complaint.detect_complaint(text) == (False, None)


def test_the_complaint_and_anger_lists_do_not_overlap():
    """They are two stories with two different responses, and main.py resolves
    the overlap by checking complaint first. That ordering is only meaningful if
    the lists describe genuinely different utterances rather than being a
    tie-break between duplicates."""
    for _tag, phrases in complaint._FAMILIES:
        for p in phrases:
            assert not anger.is_anger(p), f"{p!r} is on both the complaint and the anger list"


# ---------------------------------------------------- captured verbatim (AC 3)


def test_the_complaint_is_stored_exactly_as_the_caller_said_it(db):
    """The spoken line promises the words were written down "exactly as you
    described it". This is the test that keeps that promise true."""
    import enquiry_service as eq
    from models import ComplaintRecord

    said = "your staff were rude to me and nobody helped me at the clinic yesterday"
    out = eq.file_complaint(db, "call-1", said, detected_by="en", lang="en", phone="9876543210")
    assert out["success"] is True and out["duplicate"] is False

    row = db.query(ComplaintRecord).filter_by(id=out["id"]).one()
    assert row.text == said, "the complaint was altered on the way to storage"
    assert row.call_id == "call-1" and row.lang == "en" and row.detected_by == "en"
    assert row.status == "open"  # nobody has dealt with it yet, and the row says so
    assert isinstance(row.created_at, datetime.datetime)


def test_a_non_latin_complaint_survives_storage_unchanged(db):
    """Encoding is the obvious way "verbatim" quietly stops being true."""
    import enquiry_service as eq
    from models import ComplaintRecord

    said = "স্টাফ খুব খারাপ ব্যবহার করেছে, আমি অভিযোগ করতে চাই"
    out = eq.file_complaint(db, "call-bn", said, detected_by="bn", lang="bn")
    assert db.query(ComplaintRecord).filter_by(id=out["id"]).one().text == said


def test_the_same_complaint_twice_in_one_call_is_one_complaint(db):
    """The agent says "your complaint", singular. A retried tool call, or a
    caller repeating themselves, must not open a second row."""
    import enquiry_service as eq
    from models import ComplaintRecord

    first = eq.file_complaint(db, "call-2", "I want to file a complaint", lang="en")
    again = eq.file_complaint(db, "call-2", "I want to file a complaint", lang="en")
    assert again["duplicate"] is True and again["id"] == first["id"]
    assert db.query(ComplaintRecord).filter_by(call_id="call-2").count() == 1


def test_a_different_complaint_later_in_the_same_call_is_a_second_complaint(db):
    """The other half of the dedup rule: two different grievances are two
    complaints, and collapsing them would lose one."""
    import enquiry_service as eq
    from models import ComplaintRecord

    eq.file_complaint(db, "call-3", "your staff were rude to me", lang="en")
    eq.file_complaint(db, "call-3", "I want to complain about the billing", lang="en")
    assert db.query(ComplaintRecord).filter_by(call_id="call-3").count() == 2


def test_a_runaway_transcript_is_bounded_but_never_edited(db):
    """The length bound is storage hygiene, not an editorial step: what is kept
    has to be a prefix of what was said, with nothing rewritten."""
    import enquiry_service as eq
    from models import ComplaintRecord

    said = "complaint " + ("x" * (eq.MAX_COMPLAINT_CHARS + 500))
    out = eq.file_complaint(db, "call-4", said, lang="en")
    stored = db.query(ComplaintRecord).filter_by(id=out["id"]).one().text
    assert len(stored) == eq.MAX_COMPLAINT_CHARS
    assert said.startswith(stored)


# ----------------------------------------- the wording itself (AC 2, 4 and 5)


@pytest.mark.parametrize("lang", LANGS)
def test_the_acknowledgement_exists_in_all_three_languages(lang):
    assert "complaint_acknowledged" in PHRASES[lang]
    assert phrase("complaint_acknowledged", lang).strip()


def test_phrase_coverage_is_still_identical_across_languages():
    """The repo's own invariant: a key added to one language must be added to
    all three, or a caller in the missing language hears a lookup failure."""
    assert sorted(PHRASES["bn"]) == sorted(PHRASES["hi"]) == sorted(PHRASES["en"])


@pytest.mark.parametrize("lang", LANGS)
def test_the_acknowledgement_apologises_exactly_once(lang):
    """Counted with agent/apology.py's own counter, which is what the rest of
    the system uses, so this cannot drift from the real rule. It also catches a
    hazard specific to Bengali: the apology marker is matched as a bare
    substring, so an innocent word containing it reads as a second apology."""
    assert count_apologies(phrase("complaint_acknowledged", lang), lang) == 1


@pytest.mark.parametrize("lang", LANGS)
def test_the_acknowledgement_neither_defends_the_clinic_nor_explains_it_away(lang):
    """Acceptance criterion 5, pinned on the strings. It is a fixed phrase, so
    the model can never add any of this -- but a future editor could."""
    text = phrase("complaint_acknowledged", lang).lower()
    forbidden = {
        "en": ("however", "but ", "actually", "policy", "usually", "normally", "should have", "in fact"),
        "bn": ("কিন্তু", "আসলে", "নিয়ম", "সাধারণত"),
        "hi": ("लेकिन", "दरअसल", "नियम", "आमतौर पर"),
    }[lang]
    for word in forbidden:
        assert word not in text, f"{lang} acknowledgement argues or explains: {word!r}"


def test_the_english_acknowledgement_promises_only_what_this_system_can_do():
    """It must not claim a live transfer: there is no SIP transfer in this
    repository (main._handoff_to_human sends a frame and speaks a notice), so a
    promise to "put you through now" would be false."""
    text = phrase("complaint_acknowledged", "en").lower()
    assert "written" in text and "team" in text
    for overclaim in ("transferring you now", "putting you through", "connecting you now", "resolve this today"):
        assert overclaim not in text
