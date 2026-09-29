"""Every way a caller refers to a doctor when booking -- end to end, against
the REAL clinic-api and the REAL seed data.

Only three things are replaced: the classifier (its output is supplied, as the
model produced it on the live pod), the microphone and the speaker. Everything
between them is production code: the safety guards, the REAL fast path built
from the REAL catalogue, the dispatcher, every pending state, the tools
client, clinic-api's matcher (match_band.py) and its SQLite data -- the
agent's HTTP calls are routed in-process to the clinic's ASGI app.

It exists because a run of this matrix on the pod found, beyond the dead ends
it was looking for, answers that were confidently WRONG:
  * "বাচ্চাদের ডাক্তার দেখাতে চাই" (children's doctor)  -> Dr Dey, a
    dermatologist: দে was "contained" in বাচ্চাদের and in দেখাতে.
  * "চোখের ডাক্তার দেখাতে চাই" (eye doctor)            -> Dr Dey again.
  * "হার্ট টেস্টের দাম কত" (heart test price)          -> the LIVER Function
    Test's price: "হার্ট টেস্ট" vs "লিভার টেস্ট" scored 0.73 on "টেস্ট" alone.
  * four doctors (Roy/Ray, Basu/Bose) unbookable by name: each pair shared one
    Bengali alias, so the only question possible was "রায় নাকি রায়?".
  * every department-style request ("কার্ডিওলজির একজন") answered "no doctor
    by that name".
And so that the fixes are held to the standard the user set: the NORMAL flow
-- all 32 doctors, booked by name -- must keep working.
"""
from __future__ import annotations

import asyncio
import datetime
import importlib.util
import os
import pathlib
import sys
import types

import httpx
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLINIC = ROOT / "clinic-api"
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import main_pcm  # noqa: E402
from agent import answer_ledger  # noqa: E402
from agent.fast_path import Catalogue, FastPath  # noqa: E402
from agent.llm import _SLOT_KEYS  # noqa: E402
from agent.state import DialogueState  # noqa: E402
from agent.tools_client import ClinicToolsClient  # noqa: E402

CALL_DAY = datetime.date(2026, 9, 28)

DEAD_ENDS = ("নামে কোনো ডাক্তার খুঁজে পেলাম না", "এই নামে কোনো ডাক্তার খুঁজে পাচ্ছি না",
             "নামে কোনো ডাক্তার আমাদের এখানে নেই", "we don't have a doctor named")
UNREACHABLE = "এই মুহূর্তে দেখতে পারছি না"


class _FrozenDate(datetime.date):
    @classmethod
    def today(cls):
        return cls(CALL_DAY.year, CALL_DAY.month, CALL_DAY.day)


@pytest.fixture(scope="module")
def clinic(tmp_path_factory):
    """The real clinic-api app, seeded by the real seed.py into a throwaway
    SQLite file. Loaded as `clinic_main` so it never shadows the agent's
    `main`; its own db/models/seed/match_band modules are restored after."""
    db_path = tmp_path_factory.mktemp("booking_refs") / "clinic.db"
    old_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    names = ("db", "models", "seed", "match_band")
    saved = {m: sys.modules.pop(m, None) for m in names}
    sys.path.insert(0, str(CLINIC))
    try:
        import db as clinic_db
        import models as clinic_models
        clinic_models.Base.metadata.create_all(clinic_db.engine)
        from seed import seed
        seed()
        spec = importlib.util.spec_from_file_location("clinic_main", CLINIC / "main.py")
        clinic_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clinic_main)
        yield clinic_main
    finally:
        sys.path.remove(str(CLINIC))
        for m in names:
            sys.modules.pop(m, None)
        for m, mod in saved.items():
            if mod is not None:
                sys.modules[m] = mod
        if old_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old_url


@pytest.fixture(scope="module")
def catalogue(clinic):
    async def fetch():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=clinic.app),
                                     base_url="http://clinic") as c:
            return (await c.get("/api/v1/catalogue")).json()
    return asyncio.run(fetch())


class _Result:
    """A transcribed turn. `agreement`/`alt_text` simulate the two decoders:
    agreement 0.0 is what a one-word answer scores when CTC and RNNT spell it
    differently, which sends the turn to the confidence gate's REJECT zone."""
    decoder_used = "rnnt"

    def __init__(self, text, agreement=1.0, alt_text=""):
        self.text = text
        self.decoder_agreement = agreement
        self.alt_text = alt_text
        self.ctc_words = self.rnnt_words = max(1, len(text.split()))


class _ASR:
    text = ""
    agreement = 1.0
    alt_text = ""

    async def transcribe_utterance(self, wav_path):
        return _Result(self.text, self.agreement, self.alt_text)


class _NoCache:
    def get(self, text):
        return None, None

    def put(self, text, data):
        pass


class _AsyncNoOp:
    async def __call__(self, *a, **k):
        return None


def _llm(intent, **slots):
    s = {k: None for k in _SLOT_KEYS}
    s.update(slots)
    return {"intent": intent, "slots": s, "parts": [{"intent": intent, "slots": dict(s)}],
            "direct_reply_bn": None}


@pytest.fixture(params=[main, main_pcm], ids=["main", "main_pcm"])
def call(request, clinic, catalogue, monkeypatch, tmp_path):
    transport = request.param
    spoken: list[str] = []
    classify: dict[str, dict] = {}
    asr = _ASR()

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    tools = ClinicToolsClient("http://clinic")
    tools._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=clinic.app),
                                      base_url="http://clinic")

    monkeypatch.setattr(datetime, "date", _FrozenDate)
    monkeypatch.setattr(transport, "_speak", fake_speak)
    monkeypatch.setattr(transport, "_tools", tools)
    monkeypatch.setattr(transport, "_asr", asr)
    monkeypatch.setattr(transport, "_fast_path", FastPath(Catalogue(catalogue), today=CALL_DAY))
    monkeypatch.setattr(transport, "_intent_cache", _NoCache())
    monkeypatch.setattr(transport, "extract_intent",
                        lambda text: (classify[text], {"total_time_s": 0.0, "attempts": 1}))
    counter = iter(range(100_000))

    def new_session():
        return types.SimpleNamespace(
            call_id="refs", pending=None, dispatch_lock=asyncio.Lock(),
            send_json=_AsyncNoOp(), utt_seq=1, call_state=None, confirm_attempts=0,
            answer_ledger=answer_ledger.AnswerLedger(), state=DialogueState(), deferred=None)

    def say(text, *, llm=None, session=None, agreement=1.0, alt=""):
        """One caller turn. -> (reply, session)."""
        if llm is not None:
            classify[text] = llm
        session = session or new_session()
        asr.text, asr.agreement, asr.alt_text = text, agreement, alt
        before = len(spoken)
        wav = tmp_path / f"u{next(counter)}.wav"
        wav.write_bytes(b"")
        asyncio.run(transport._dispatch_turn(session, str(wav)))
        return " ".join(spoken[before:]), session

    yield types.SimpleNamespace(say=say, catalogue=catalogue, new_session=new_session)
    asyncio.run(tools._client.aclose())


def _dead_end(reply: str) -> bool:
    return any(d in reply for d in DEAD_ENDS) or UNREACHABLE in reply


# =========================================================================
# The normal flow: every seeded doctor, booked by name.
# =========================================================================

def test_every_seeded_doctor_is_bookable_by_bengali_name(call):
    """All 32. Before the fix 28 got through: Roy/Ray and Basu/Bose each
    shared one Bengali alias, and the offer that followed -- "রায় নাকি রায়?"
    -- could not be answered."""
    stuck = []
    for doctor in call.catalogue["doctors"]:
        bn = doctor["aliases_bn"][0]
        said = f"ডাক্তার {bn} এর কাছে অ্যাপয়েন্টমেন্ট চাই"
        reply, session = call.say(said, llm=_llm("book_appointment", doctor_name=f"ডাক্তার {bn}"))
        pending = session.pending or {}
        offered = pending.get("candidates") or []
        if (session.pending or {}).get("awaiting") in ("doctor_name", "entity_choice") and offered:
            # Two doctors really are said the same way (Roy and Ray). Then the
            # offer must be ANSWERABLE: every option said differently.
            spoken = [c.get("name_bn") for c in offered]
            if len(set(spoken)) != len(spoken):
                stuck.append((doctor["name"], reply))
            continue
        if _dead_end(reply) or "ডাঃ" not in reply:
            stuck.append((doctor["name"], reply))
    assert not stuck, stuck


def test_roy_and_ray_are_offered_so_the_caller_can_choose(call):
    reply, session = call.say("ডাক্তার রায়ের কাছে অ্যাপয়েন্টমেন্ট চাই",
                              llm=_llm("book_appointment", doctor_name="রায়"))
    assert "কার্ডিওলজি বিভাগের রায়" in reply and "ডায়াবেটোলজি বিভাগের রায়" in reply, reply

    # ... and answering with the department reaches the right doctor.
    reply, session = call.say("কার্ডিওলজির", session=session)
    assert "ডাঃ" in reply and not _dead_end(reply), reply
    assert (session.pending or {}).get("slots", {}).get("doctor_name") == "Dr. N. Roy", session.pending


def test_basu_and_bose_are_no_longer_the_same_word(call):
    reply, _ = call.say("ডাক্তার বোসের কাছে অ্যাপয়েন্টমেন্ট চাই",
                        llm=_llm("book_appointment", doctor_name="বোস"))
    assert "বলতে চাইছেন" not in reply and "ডাঃ" in reply, reply


@pytest.mark.parametrize("said,doctor", [
    ("ডাক্তার সেনের কাছে অ্যাপয়েন্টমেন্ট চাই", "ডাক্তার সেন"),
    ("ডক্টর মণ্ডল এর কাছে অ্যাপয়েন্টমেন্ট চাই", "ডক্টর মণ্ডল"),
    ("সেনের সাথে দেখা করতে চাই", "সেন"),
    ("Dr Sen er appointment chai", "Dr Sen"),
    ("doctor mondal e appointment", "doctor mondal"),
    ("মন্ডল ডাক্তারবাবুর কাছে যাব", "মন্ডল ডাক্তারবাবু"),
    ("ডাক্তার দে এর কাছে অ্যাপয়েন্টমেন্ট চাই", "ডাক্তার দে"),     # a 2-letter surname
    ("ডাক্তার করকে দেখাতে চাই", "কর"),                              # another
])
def test_a_named_doctor_still_books(call, said, doctor):
    reply, _ = call.say(said, llm=_llm("book_appointment", doctor_name=doctor))
    assert not _dead_end(reply), reply
    assert "ডাঃ" in reply or "Dr." in reply, reply


# =========================================================================
# Confidently wrong -> right.
# =========================================================================

@pytest.mark.parametrize("said,llm,department", [
    ("বাচ্চাদের ডাক্তার দেখাতে চাই",
     _llm("doctor_availability", doctor_name="বাচ্চাদের ডাক্তার"), "পিডিয়াট্রিক্স"),
    ("হার্টের ডাক্তার দেখাতে চাই",
     _llm("doctor_availability", doctor_name="হার্টের ডাক্তার"), "কার্ডিওলজি"),
])
def test_a_kind_of_doctor_never_becomes_the_wrong_doctor(call, said, llm, department):
    reply, session = call.say(said, llm=llm)
    assert "ডাঃ দে " not in reply, f"routed to the dermatologist: {reply}"
    assert department in reply, reply
    assert not _dead_end(reply), reply


def test_an_eye_doctor_is_told_there_is_none_and_given_the_departments(call):
    """There is no eye department. Before: Dr Dey (a dermatologist)."""
    reply, session = call.say("চোখের ডাক্তার দেখাতে চাই",
                              llm=_llm("doctor_availability", doctor_name="চোখের ডাক্তার"))
    assert "ডাঃ দে" not in reply, reply
    assert "এই ধরনের ডাক্তার আমাদের এখানে নেই" in reply, reply
    assert "কার্ডিওলজি" in reply and (session.pending or {}).get("awaiting") == "list_department"


@pytest.mark.parametrize("said,test_name,expected_bn", [
    ("হার্ট টেস্টের দাম কত", "হার্ট টেস্ট", "ইসিজি"),
    ("heart test koto taka", "heart test", "ইসিজি"),
    ("লিভার টেস্টের দাম কত", "লিভার টেস্ট", None),   # the real LFT still resolves
])
def test_a_test_price_is_the_price_of_that_test(call, said, test_name, expected_bn):
    reply, _ = call.say(said, llm=_llm("test_rate", test_name=test_name))
    if expected_bn:
        assert "800" not in reply, f"answered with the LFT price: {reply}"
        assert "355" in reply or "তিনশো" in reply, reply
    else:
        assert "800" in reply or "আটশো" in reply, reply


# =========================================================================
# A department said where a doctor's name goes -> that department's doctors.
# =========================================================================

@pytest.mark.parametrize("said,llm,department_bn", [
    ("কার্ডিওলজি র একজনের সাথে দেখা করার ইচ্ছে রয়েছে",
     _llm("book_appointment", doctor_name="কার্ডিওলজি র একজন"), "কার্ডিওলজি"),
    ("কার্ডিওলজিতে একজনের সাথে দেখা করতে চাই",
     _llm("book_appointment", doctor_name="কার্ডিওলজিতে একজন", department="cardiology"), "কার্ডিওলজি"),
    ("অর্থোতে একজন ডাক্তার দেখাতে চাই",
     _llm("book_appointment", doctor_name="অর্থো", department="orthopedics"), "অর্থোপেডিক্স"),
    ("skin er doctor dekhate chai",
     _llm("doctor_availability", doctor_name="skin er doctor"), "ডার্মাটোলজি"),
    # The model got it right -- department, no doctor -- and booking still
    # asked "which doctor?" because it had no notion of a department.
    ("হার্টের ডাক্তার দেখাতে চাই",
     _llm("book_appointment", department="cardiology"), "কার্ডিওলজি"),
])
def test_a_department_request_lists_that_departments_doctors(call, said, llm, department_bn):
    reply, session = call.say(said, llm=llm)
    assert not _dead_end(reply), reply
    assert department_bn in reply and "ডাঃ" in reply, reply
    assert (session.pending or {}).get("awaiting") == "doctor_choice", session.pending


def test_the_listed_department_continues_into_a_booking(call):
    reply, session = call.say("কার্ডিওলজিতে একজনের সাথে দেখা করতে চাই",
                              llm=_llm("book_appointment", doctor_name="কার্ডিওলজিতে একজন",
                                       department="cardiology"))
    first = session.pending["candidates"][0]
    reply, session = call.say(first["name_bn"], session=session)
    assert not _dead_end(reply), reply
    assert (session.pending or {}).get("slots", {}).get("doctor_name") == first["name"]


def test_a_missing_person_is_still_a_missing_person(call):
    """The fallback must not turn "no such doctor" into a department answer."""
    reply, _ = call.say("ডাক্তার নোবডি এর কাছে অ্যাপয়েন্টমেন্ট চাই",
                        llm=_llm("book_appointment", doctor_name="ডাক্তার নোবডি"))
    assert "বিভাগে" not in reply, reply
    assert UNREACHABLE not in reply, reply


# =========================================================================
# A Banglish caller is answered in Banglish, not English.
#
# detect_language() had no marker for the commonest Banglish words (er, chai,
# koto, taka), so a Latin-script Bengali caller fell through to "english" and
# was answered "Yes, Dr. A. Sen. The next sitting is ...". The templates
# already had a Banglish branch for exactly this caller; it never ran. (Both
# are Latin script -- tts.py sends everything to gTTS as "bn" either way -- so
# this changes the WORDS the caller hears, from English to their own.)
# =========================================================================

@pytest.mark.parametrize("said,llm,banglish,english", [
    ("Dr Sen er appointment chai", _llm("book_appointment", doctor_name="Dr Sen"),
     "Porer din", "The next sitting"),
    ("heart test koto taka", _llm("test_rate", test_name="heart test"),
     "taka", "rupees"),
])
def test_a_banglish_caller_is_not_answered_in_english(call, said, llm, banglish, english):
    reply, _ = call.say(said, llm=llm)
    assert english not in reply, f"answered in English: {reply}"
    assert banglish in reply, reply


# =========================================================================
# "অর্থোপেডিক্সে কোন কোন ডক্টর বসছেন কালকে" -- the department in the
# question IS the answer, not a prompt to list all eight. (Live: all eight
# were listed, and the caller had to say "অর্থোপেডিক্স" three more times.)
# =========================================================================

TOMORROW = "2026-09-29"
ALL_DEPARTMENTS = "আমাদের এখানে"   # how the every-department reply begins

# Every way of asking "which doctors in <department>" -- each run twice: with
# the classifier filling the department slot, and with it missing it.
DEPARTMENT_QUESTIONS = [
    ("অর্থোপেডিক্সে কোন কোন ডক্টর বসছেন কালকে", "orthopaedics", "অর্থোপেডিক্স"),   # live
    ("কাল অর্থোতে কে কে আছেন", "ortho", "অর্থোপেডিক্স"),
    ("হাড়ের ডাক্তার কে কে আছেন কাল", "orthopaedics", "অর্থোপেডিক্স"),
    ("ortho te ke ke doctor boshen kal", "ortho", "অর্থোপেডিক্স"),
    ("অর্থোপেডিক বিভাগে কারা আছেন", "orthopaedics", "অর্থোপেডিক্স"),
    ("কার্ডিওলজিতে কারা বসেন কালকে", "cardiology", "কার্ডিওলজি"),
    ("শিশু বিশেষজ্ঞ কে কে আছেন কাল", "paediatrics", "পিডিয়াট্রিক্স"),
    ("গাইনি বিভাগে কোন কোন ডাক্তার আছেন কাল", "gynaecology", "গাইনি"),
]


@pytest.mark.parametrize("model_found_department", [True, False], ids=["slot", "words"])
@pytest.mark.parametrize("said,slot_value,department_bn", DEPARTMENT_QUESTIONS)
def test_a_department_in_the_question_lists_that_department(call, said, slot_value,
                                                            department_bn, model_found_department):
    slots = {"department": slot_value} if model_found_department else {}
    reply, session = call.say(said, llm=_llm("doctor_schedule", **slots))
    assert not reply.startswith(ALL_DEPARTMENTS), f"listed every department: {reply}"
    # A relative day ("কালকে") is checked before it is used -- the existing,
    # deliberate date rule (_date_gate), the same for every department
    # question however it was recognised: read back when the model's reading
    # agrees ("confirm_date"), asked when only the words gave it
    # ("department_date"). The caller says yes, as a caller would.
    if (session.pending or {}).get("awaiting") in ("confirm_date", "department_date"):
        assert TOMORROW in reply, reply
        assert session.pending["slots"].get("department"), "the department must survive the check"
        reply, session = call.say("হ্যাঁ", session=session)
    assert department_bn in reply, reply
    assert (session.pending or {}).get("awaiting") in ("doctor_choice", "department_date"), session.pending


def test_a_bare_list_question_still_lists_every_department(call):
    reply, session = call.say("কে কে ডাক্তার আছেন", llm=_llm("doctor_availability"))
    assert reply.startswith(ALL_DEPARTMENTS), reply
    assert session.pending["awaiting"] == "list_department"


# =========================================================================
# The answer to "কোন বিভাগের ডাক্তার খুঁজছেন?", however it is said.
# =========================================================================

def _listed(call):
    """A call that has just been read all eight departments and asked which."""
    return call.say("কে কে ডাক্তার আছেন", llm=_llm("doctor_availability"))[1]


@pytest.mark.parametrize("answer,department_bn", [
    ("অর্থোপেডিক্স", "অর্থোপেডিক্স"),
    ("অর্থ পেডিটস", "অর্থোপেডিক্স"),        # live: the ASR's split -> "no such TEST"
    ("অর্থো", "অর্থোপেডিক্স"),
    ("হাড়ের", "অর্থোপেডিক্স"),               # cut short
    ("অর্থোপেডিক্সের ডাক্তার", "অর্থোপেডিক্স"),
    ("ortho", "অর্থোপেডিক্স"),
    ("কার্ডিয়োলজি", "কার্ডিওলজি"),           # spelled as heard
    ("চতুর্থটা", "অর্থোপেডিক্স"),              # its place in the list read out
    ("চার নম্বর", "অর্থোপেডিক্স"),
    ("প্রথমটা", "জেনারেল মেডিসিন"),
    ("শেষেরটা", "ডায়াবেটোলজি"),
])
def test_the_department_answer_is_understood(call, answer, department_bn):
    reply, session = call.say(answer, session=_listed(call))
    assert department_bn in reply and "ডাঃ" in reply, reply
    assert (session.pending or {}).get("awaiting") == "doctor_choice"


def test_a_short_answer_it_cannot_place_is_asked_once_more_with_the_options(call):
    reply, session = call.say("ওই যে", session=_listed(call))
    assert "বিভাগের নামটা ঠিক বুঝতে পারিনি" in reply and "অর্থোপেডিক্স" in reply, reply
    assert session.pending["awaiting"] == "list_department"
    reply, session = call.say("অর্থো", session=session)          # ...and the retry works
    assert "অর্থোপেডিক্স" in reply and "ডাঃ" in reply, reply


def test_a_longer_unrelated_sentence_is_treated_as_a_new_question(call):
    reply, _ = call.say("ডাক্তার সেন কবে বসেন", session=_listed(call),
                        llm=_llm("doctor_schedule", doctor_name="ডাক্তার সেন"))
    assert "ডাঃ সেন" in reply and "বিভাগের নামটা" not in reply, reply


def test_a_doctor_is_chosen_by_place_in_the_list(call):
    reply, session = call.say("কার্ডিওলজির ডাক্তার কে কে আছেন",
                              llm=_llm("doctors_by_department", department="cardiology"))
    second = session.pending["candidates"][1]
    reply, session = call.say("দ্বিতীয় জন", session=session)
    assert (session.pending or {}).get("slots", {}).get("doctor_name") == second["name"], session.pending


def test_the_indefinite_article_is_not_the_first_option(call):
    """"একজন"/"একটা" is "someone"/"a", not "the first one"."""
    reply, _ = call.say("একজন ডাক্তার", session=_listed(call))
    assert "জেনারেল মেডিসিন বিভাগে" not in reply, reply


# =========================================================================
# The confidence gate: a one-word answer the two decoders spelled differently
# scores agreement 0.00 and used to be rejected outright.
# =========================================================================

REJECTED = "ভালো করে শুনতে পাইনি"


def test_a_rejected_one_word_department_answer_is_accepted(call):
    """Live: "অর্থোপেডিক্স" rejected twice at agreement 0.00."""
    reply, _ = call.say("অর্থোপেডিক্স", session=_listed(call), agreement=0.0, alt="অর্থপেডিক্স")
    assert REJECTED not in reply, reply
    assert "অর্থোপেডিক্স" in reply and "ডাঃ" in reply, reply


def test_either_reading_may_carry_the_answer(call):
    reply, _ = call.say("অথ পেডি", session=_listed(call), agreement=0.0, alt="অর্থোপেডিক্স")
    assert REJECTED not in reply and "অর্থোপেডিক্স" in reply, reply


def test_two_readings_naming_different_departments_stay_rejected(call):
    reply, session = call.say("অর্থোপেডিক্স", session=_listed(call), agreement=0.0, alt="কার্ডিওলজি")
    assert REJECTED in reply, reply
    assert session.pending["awaiting"] == "list_department", "the question is still open"


def test_a_reading_that_is_no_option_stays_rejected(call):
    reply, _ = call.say("ইয়ে", session=_listed(call), agreement=0.0, alt="উম")
    assert REJECTED in reply, reply


def test_the_gate_is_unchanged_with_no_closed_question_pending(call):
    reply, _ = call.say("অর্থোপেডিক্স", agreement=0.0, alt="অর্থপেডিক্স",
                        llm=_llm("doctors_by_department", department="orthopaedics"))
    assert REJECTED in reply, reply


def test_a_rejected_yes_to_a_yes_no_question_is_accepted(call):
    session = call.new_session()
    session.pending = {"awaiting": "confirm_date", "slots": {"doctor_name": "Dr. A. Mondal"},
                       "candidates": None, "offered_date": "2026-09-28", "retries": 0,
                       "resume_intent": "doctor_availability", "span_end": "2026-09-28"}
    reply, _ = call.say("হ্যাঁ", session=session, agreement=0.0, alt="হ্যা")
    assert REJECTED not in reply and "মন্ডল" in reply, reply


def test_a_yes_that_would_write_a_booking_needs_both_readings(call):
    """Where "yes" WRITES, one reading is not enough: a mis-heard yes there is
    a booking, not a question."""
    session = call.new_session()
    session.pending = {"awaiting": "confirm_booking", "candidates": None, "retries": 0,
                       "offered_date": None, "turns": 1,
                       "slots": {"doctor_name": "Dr. A. Mondal", "doctor_name_bn": "মন্ডল",
                                 "date": "2026-10-05", "time_slot": "18:00", "date_checked": True,
                                 "patient_name": "রিনা দাস", "phone": "9831234567"}}
    reply, session = call.say("হ্যাঁ", session=session, agreement=0.0, alt="উম")
    assert REJECTED in reply, reply
    assert session.pending["awaiting"] == "confirm_booking", "nothing was booked"


# =========================================================================
# "Which departments do you have?" -- answered with the departments.
# Live: every phrasing below got "কোন বিভাগের ডাক্তার খুঁজছেন?" back -- the
# caller asked for the list and was asked to pick from it, unseen.
# =========================================================================

BARE_DEPARTMENT_QUESTION = "কোন বিভাগের ডাক্তার খুঁজছেন?"


@pytest.mark.parametrize("said,llm", [
    # live transcripts, with what the model actually returned for them
    ("আপনাদের এখানে কোন কোন ডিপার্টমেন্ট রয়েছে", _llm("doctors_by_department")),
    ("আপনাদের ওখানে কোন কোন বিভাগ রয়েছে", _llm("doctors_by_department")),
    ("বিভাগগুলোর নাম বলুন", _llm("doctors_by_department")),
    ("আপনাদের কি কি বিভাগ আছে", _llm("doctors_by_department", department="কি কি")),
    # other ways of asking
    ("কোন কোন ডিপার্টমেন্ট রয়েছে", _llm("doctors_by_department")),
    ("which departments do you have", _llm("unclear")),
    ("ডিপার্টমেন্টের লিস্ট দিন", _llm("doctors_by_department")),
    # a phrasing no cue list knows: the model's doctors_by_department with no
    # department is itself enough -- the list is read, never a bare question
    ("আপনাদের এখানে কী কী পরিষেবা পাওয়া যায়", _llm("doctors_by_department")),
])
def test_which_departments_is_answered_with_the_departments(call, said, llm):
    reply, session = call.say(said, llm=llm)
    assert reply != BARE_DEPARTMENT_QUESTION, reply
    # In the caller's language: the English question gets "We have doctors in".
    assert reply.startswith((ALL_DEPARTMENTS, "We have doctors in")), reply
    assert "অর্থোপেডিক্স" in reply, reply
    assert session.pending["awaiting"] == "list_department"


def test_the_department_list_then_leads_to_a_doctor(call):
    reply, session = call.say("বিভাগগুলোর নাম বলুন", llm=_llm("doctors_by_department"))
    reply, session = call.say("অর্থ পেডিটস", session=session)
    assert "অর্থোপেডিক্স" in reply and "ডাঃ" in reply, reply


# =========================================================================
# "পরশুদিন" means the day after tomorrow -- whatever the model read.
# =========================================================================

def test_an_unambiguous_day_word_is_not_second_guessed_by_the_model(call):
    """Live: the model read "পরশুদিন" as a Saturday twelve days out, and the
    agent offered both ("…৩০ তারিখের কথা বলছেন, নাকি শনিবার ১০ তারিখের?") --
    a question "হ্যাঁ" cannot answer. The parser's reading is read back
    instead, and the model's wrong one is not offered."""
    reply, session = call.say("অর্থোপেটিক্সে পরশুদিন কোন ডক্টর বসছেন",
                              llm=_llm("doctor_schedule", department="orthopaedics",
                                       date_expr="next_saturday"))
    assert "নাকি" not in reply, f"offered the model's wrong day: {reply}"
    assert "2026-09-30" in reply and "হ্যাঁ অথবা না বলুন" in reply, reply
    reply, session = call.say("হ্যাঁ", session=session)
    assert "অর্থোপেডিক্স" in reply and "ডাঃ" in reply, reply
