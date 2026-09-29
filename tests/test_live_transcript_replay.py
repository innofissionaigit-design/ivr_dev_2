"""The live call transcript, replayed turn by turn through the real dispatcher.

Every utterance below is the caller's exact words from the call. Each turn goes
through _dispatch_turn -- ASR result -> confidence zone -> _continue_pending or
_resolve_intent -> dispatch -> templated reply -- on BOTH transports.

The classifier (agent/llm.py's extract_intent) is stubbed to return what it
evidently returned on the call: the doctor-list questions came back as
doctor_availability with no doctor, and the bare "অন্য দিনের জন্যে" as
reschedule_appointment. The fast path and the semantic cache are disabled so
every classification goes through _resolve_intent's correction step exactly as
a live LLM result does. The point: the fixes hold even when the model gets it
wrong the same way again.

The clinic is a spy with plausible answers; the clock is frozen on the day of
the call (Monday 28 September 2026).
"""
from __future__ import annotations

import asyncio
import datetime
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main  # noqa: E402
import main_pcm  # noqa: E402
from agent import answer_ledger  # noqa: E402
from agent.llm import _SLOT_KEYS  # noqa: E402
from agent.reply_templates import (  # noqa: E402
    booking_which_date_prompt, missing_slot_prompt, new_or_move_prompt,
)
from agent.state import DialogueState  # noqa: E402

CALL_DAY = datetime.date(2026, 9, 28)
WHICH_DOCTOR = missing_slot_prompt("doctor_availability", "doctor_name")
TODAY_OR_OTHER_DAY = missing_slot_prompt("book_appointment", "date")
RESCHEDULE_PHONE_CUE = "যে ফোন নম্বরে বুক করেছিলেন"

DEPARTMENTS = [
    {"name": "General Medicine", "department_bn": "জেনারেল মেডিসিন",
     "aliases_bn": ["general medicine", "medicine", "জেনারেল মেডিসিন", "মেডিসিন"], "doctor_count": 4},
    {"name": "Cardiology", "department_bn": "কার্ডিওলজি",
     "aliases_bn": ["cardiology", "cardio", "কার্ডিওলজি", "হার্ট", "হার্টের ডাক্তার"], "doctor_count": 4},
    {"name": "ENT", "department_bn": "ইএনটি", "aliases_bn": ["ent", "ইএনটি"], "doctor_count": 0},
]


class _FrozenDate(datetime.date):
    @classmethod
    def today(cls):
        return cls(CALL_DAY.year, CALL_DAY.month, CALL_DAY.day)


class SpyClinic:
    """Dr A. Sen does not sit on the call day; next free 2026-09-29; sits on
    3 November. Records every call so a test can say what was NOT asked."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def get_departments(self, date=None):
        self.calls.append(("get_departments", date))
        return {"found": True, "date": date, "departments": [dict(d) for d in DEPARTMENTS]}

    async def get_doctors_by_department(self, department, date=None):
        self.calls.append(("get_doctors_by_department", department, date))
        return {"found": True, "department": department, "department_bn": "কার্ডিওলজি",
                "date": date, "doctors": [{"name": "Dr. N. Roy", "doctor_name_bn": "রায়",
                                           "qualifications": "DM"}]}

    async def get_doctor_availability(self, doctor_name, date, *args, **kwargs):
        self.calls.append(("get_doctor_availability", doctor_name, date))
        # Dr Sen does not sit on the call day (that is what opened the
        # "today or another day?" question in the live call). Dr Sinha, the
        # doctor in the second transcript, sits every day.
        sinha = "সিনহা" in doctor_name or "Sinha" in doctor_name
        mondal = "মন্ডল" in doctor_name or "মণ্ডল" in doctor_name or "Mondal" in doctor_name
        sits = sinha or date != CALL_DAY.isoformat()
        name, name_bn = ("Dr. A. Sen", "সেন")
        if sinha:
            name, name_bn = "Dr. K. Sinha", "সিনহা"
        elif mondal:
            name, name_bn = "Dr. A. Mondal", "মন্ডল"
        return {"found": True, "doctor_name": name, "doctor_name_bn": name_bn,
                "date": date, "available": sits,
                "chamber_hours": "10:00-12:00" if sits else None,
                "next_available_date": None if sits else "2026-09-29",
                "slots": ["10:00", "10:15", "10:30"] if sits else []}

    async def get_doctor_schedule(self, doctor_name):
        self.calls.append(("get_doctor_schedule", doctor_name))
        return {"found": True, "doctor_name": "Dr. A. Sen", "doctor_name_bn": "সেন",
                "days": ["Tuesday", "Thursday"]}

    async def find_appointments(self, *args, **kwargs):
        self.calls.append(("find_appointments",))
        return {"found": False, "matches": []}

    def names(self):
        return [c[0] for c in self.calls]


class _Result:
    decoder_used = "ctc"
    decoder_agreement = 1.0

    def __init__(self, text):
        self.text = text
        self.ctc_words = self.rnnt_words = max(1, len(text.split()))


class _ASR:
    def __init__(self):
        self.next_text = ""

    async def transcribe_utterance(self, wav_path):
        return _Result(self.next_text)


class _NoCache:
    def get(self, text):
        return None, None

    def put(self, text, data):
        pass


class _AsyncNoOp:
    async def __call__(self, *a, **k):
        return None


def _slots(**kw):
    s = {k: None for k in _SLOT_KEYS}
    s.update(kw)
    return s


def _llm(intent, **slots):
    s = _slots(**slots)
    return {"intent": intent, "slots": s, "parts": [{"intent": intent, "slots": dict(s)}],
            "direct_reply_bn": None}


@pytest.fixture(params=[main, main_pcm], ids=["main", "main_pcm"])
def call(request, monkeypatch, tmp_path):
    transport = request.param
    spoken: list[str] = []
    clinic = SpyClinic()
    asr = _ASR()
    classify: dict[str, dict] = {}

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    def fake_extract_intent(text):
        return classify[text], {"total_time_s": 0.0, "attempts": 1}

    monkeypatch.setattr(datetime, "date", _FrozenDate)
    monkeypatch.setattr(transport, "_speak", fake_speak)
    monkeypatch.setattr(transport, "_tools", clinic)
    monkeypatch.setattr(transport, "_asr", asr)
    monkeypatch.setattr(transport, "_fast_path", None)
    monkeypatch.setattr(transport, "_intent_cache", _NoCache())
    monkeypatch.setattr(transport, "extract_intent", fake_extract_intent)

    session = types.SimpleNamespace(
        call_id="replay", pending=None, dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
        utt_seq=1, call_state=None, confirm_attempts=0,
        answer_ledger=answer_ledger.AnswerLedger(), state=DialogueState(),
        deferred=None,   # read directly by _drain_deferred, as on a real CallSession
    )
    counter = iter(range(10_000))

    def say(text, *, llm=None):
        """The caller says `text`; returns what the agent said back."""
        if llm is not None:
            classify[text] = llm
        asr.next_text = text
        before = len(spoken)
        wav = tmp_path / f"utt-{next(counter)}.wav"
        wav.write_bytes(b"")
        asyncio.run(transport._dispatch_turn(session, str(wav)))
        return spoken[before:]

    def ask_about_sen_today():
        """Put Dr Sen in the call's memory the way the live call did. A day
        the agent calculated from a day word is read back first ("আজ মানে
        সোমবার ... ঠিক আছে?", E13-S7) -- the caller confirms, as a caller would."""
        call_llm = _llm("doctor_availability", doctor_name="Dr. A. Sen", date_expr="today")
        replies = say("ডাক্তার সেন কি আজ আছেন", llm=call_llm)
        if (session.pending or {}).get("awaiting") == "confirm_date":
            replies += say("হ্যাঁ")
        return replies

    def talk_about_sen():
        """Put Dr Sen in the call's memory (DialogueState). Through
        doctor_schedule, whose answer records the entity -- the confirm_date
        resume path used by ask_about_sen_today() does not record it (a gap
        that predates this change)."""
        say("ডাক্তার সেন কবে বসেন", llm=_llm("doctor_schedule", doctor_name="Dr. A. Sen"))
        assert session.state.active_doctor.primary == "Dr. A. Sen"
        session.pending = None

    return types.SimpleNamespace(say=say, session=session, clinic=clinic, spoken=spoken,
                                 ask_about_sen_today=ask_about_sen_today,
                                 talk_about_sen=talk_about_sen)


# ---------------------------------------------------------------------------
# 1. "Which doctors are there?" no longer loops.

def test_1_a_list_question_gets_the_departments_not_which_doctor(call):
    # "ডাক্তার অবৈলাবল আছে" is singular and ambiguous -- asking once is fair.
    replies = call.say("ডাক্তার অবৈলাবল আছে", llm=_llm("doctor_availability"))
    assert replies == [WHICH_DOCTOR]

    # The live call asked the same question again here. Now: the departments.
    replies = call.say("কে কে ডাক্তার অবৈলাবল আছে সেটা বলুন", llm=_llm("doctor_availability"))
    assert replies == ["আমাদের এখানে জেনারেল মেডিসিন আর কার্ডিওলজি বিভাগে ডাক্তার বসেন। "
                       "কোন বিভাগের ডাক্তার খুঁজছেন?"]
    assert WHICH_DOCTOR not in replies
    assert call.clinic.calls[-1] == ("get_departments", None)
    assert call.session.pending["awaiting"] == "list_department"

    # Picking a department lists its doctors and opens the booking hand-off.
    replies = call.say("কার্ডিওলজি")
    assert call.clinic.calls[-1] == ("get_doctors_by_department", "Cardiology", None)
    assert "রায়" in replies[0]
    assert call.session.pending["awaiting"] == "doctor_choice"


def test_1b_asking_which_doctor_twice_without_a_name_offers_the_list(call):
    call.say("ডাক্তার অবৈলাবল আছে", llm=_llm("doctor_availability"))
    replies = call.say("ডাক্তার পাওয়া যাবে", llm=_llm("doctor_availability"))
    assert replies[0].startswith("আমাদের এখানে"), "never the same question twice in a row"


def test_1c_a_day_in_the_list_question_filters_it(call):
    call.say("আজ কে কে ডাক্তার বসেন", llm=_llm("doctor_availability", date_expr="today"))
    assert call.clinic.calls[-1] == ("get_departments", "2026-09-28")
    assert call.spoken[-1].startswith("আমাদের এখানে আজ ")


def test_1d_a_named_department_still_goes_straight_to_its_doctors(call):
    call.say("কার্ডিওলজিতে কে কে ডাক্তার আছেন",
             llm=_llm("doctors_by_department", department="Cardiology"))
    assert ("get_departments", None) not in call.clinic.calls
    assert call.clinic.calls[-1][:2] == ("get_doctors_by_department", "Cardiology")


# ---------------------------------------------------------------------------
# 2. Dr Sen is not the answer to "which doctors sit here".

def test_2_the_last_doctor_is_not_backfilled_into_a_list_question(call):
    # Dr Sen is discussed first -- this is what put him in the call's memory.
    call.talk_about_sen()
    before = len(call.clinic.calls)

    replies = call.say("আপনাদের কি কি ডাক্তার এখানে বসেন", llm=_llm("doctor_availability"))
    # "ডাঃ সেন", not bare "সেন": the correct reply ends "ডাক্তার বসেন" (sit).
    assert not any("ডাঃ সেন" in r for r in replies), "Dr Sen must not answer a question about all doctors"
    assert replies[0].startswith("আমাদের এখানে")
    assert call.clinic.calls[before:] == [("get_departments", None)]


def test_2_with_the_real_fast_path_on(call, monkeypatch):
    """What actually happened on the live call. The rest of this file turns
    the fast path off to isolate the model's classification; this one turns
    the REAL one on, with Dr Sen's real seeded aliases -- tier 1 committed
    "বসেন" (sits) as Dr Sen before the model or any guard saw the sentence.
    (The run against the live pod is what caught this; the model-only replay
    above passed.)"""
    from agent.fast_path import Catalogue, FastPath
    catalogue = Catalogue({"tests": [], "doctors": [
        {"name": "Dr. A. Sen", "surname": "Sen", "aliases_bn": ["সেন", "sen"]},
        {"name": "Dr. N. Roy", "surname": "Roy", "aliases_bn": ["রায়", "রয়", "roy"]},
    ]})
    for mod in (main, main_pcm):
        monkeypatch.setattr(mod, "_fast_path", FastPath(catalogue, today=CALL_DAY))
    call.talk_about_sen()
    before = len(call.clinic.calls)

    for said in ("আপনাদের কি কি ডাক্তার এখানে বসেন", "কোন ডাক্তার আজ বসেন"):
        call.session.pending = None
        replies = call.say(said, llm=_llm("doctor_availability"))
        assert not any("ডাঃ সেন" in r for r in replies), said
        assert replies[0].startswith("আমাদের এখানে"), said
    assert all(c[0] == "get_departments" for c in call.clinic.calls[before:])


def test_2b_a_real_follow_up_about_him_still_works(call):
    call.talk_about_sen()
    call.say("কাল উনি বসবেন?", llm=_llm("doctor_availability", date_expr="tomorrow"))
    if (call.session.pending or {}).get("awaiting") == "confirm_date":
        call.say("হ্যাঁ")
    assert call.clinic.calls[-1] == ("get_doctor_availability", "Dr. A. Sen", "2026-09-29")


# ---------------------------------------------------------------------------
# 3. "তিন নভেম্বর" / "থার্ড নভেম্বর" are dates.

def _sen_is_off_today(call):
    replies = call.ask_about_sen_today()
    assert call.session.pending["awaiting"] == "date"
    assert call.session.pending["offered_date"] == "2026-09-29"
    return replies


@pytest.mark.parametrize("said", [
    "না অন্য তিন নভেম্বর কি উনি বসবেন",                     # live transcript
    "থার্ড নভেম্বর এর জন্য অযাপ্ফোন্টমেন্ট পাওয়া যাবে",      # live transcript
])
def test_3_a_day_next_to_a_month_is_taken_as_that_date(call, said):
    _sen_is_off_today(call)
    replies = call.say(said)
    assert TODAY_OR_OTHER_DAY not in replies, "the live call re-asked this, then dropped the booking"
    assert ("get_doctor_availability", "Dr. A. Sen", "2026-11-03") in call.clinic.calls
    assert call.session.pending is not None
    assert call.session.pending["slots"]["date"] == "2026-11-03"


# ---------------------------------------------------------------------------
# 4. "অন্য দিনের জন্যে" is not a reschedule.

def test_4_another_day_while_booking_asks_which_day(call):
    _sen_is_off_today(call)
    replies = call.say("অন্য দিনের জন্যে")
    assert replies == [booking_which_date_prompt()]
    assert not any(RESCHEDULE_PHONE_CUE in r for r in replies)
    assert "find_appointments" not in call.clinic.names()
    pending = call.session.pending
    assert pending["awaiting"] == "date" and pending["retries"] == 0, "an answer, not a failed parse"
    assert pending["offered_date"] is None, "the offered day was just declined"

    # ... and the day then given is used.
    call.say("থার্ড নভেম্বর")
    assert call.session.pending["slots"]["date"] == "2026-11-03"


@pytest.mark.parametrize("said", ["আজকের জন্য", "আজকের জন্যই", "আজকে", "আজ"])
def test_3c_answering_with_todays_inflected_form_is_not_re_asked(call, said):
    """The second live transcript. Dr Sinha IS in today, so the agent offers
    "আজকের জন্যই ... নাকি অন্য কোনো দিনের জন্য?" -- and the caller picks the
    first option in the agent's own words. That answer looped on the call."""
    call.say("ডাক্তার সিনহার কাছে অ্যাপয়েন্টমেন্ট চাই",
             llm=_llm("book_appointment", doctor_name="সিনহা"))
    assert call.session.pending["awaiting"] == "date"

    replies = call.say(said)
    assert TODAY_OR_OTHER_DAY not in replies, f"{said!r} was re-asked the question it answered"
    assert call.session.pending is not None
    slots = call.session.pending.get("slots") or {}
    assert slots.get("date") == "2026-09-28", slots


# ---------------------------------------------------------------------------
# 5. Confirming a near-match offer ("did you mean Mondal?" -> "হ্যাঁ").

NEAR_MATCH_UNCLEAR = "দুঃখিত, ঠিক কোনটার কথা বলছেন বুঝতে পারিনি। একটু পরিষ্কার করে নামটা বলবেন?"
# The ASR writes মণ্ডল (ণ্ড); the catalogue alias is মন্ডল (ন্ড). Both are
# ordinary spellings of the same surname, which is why clinic-api offers it as
# a near match instead of resolving it outright.
MONDAL = [{"name": "Dr. A. Mondal", "name_bn": "মন্ডল"}]


def _offered_mondal(call, candidates=MONDAL):
    """The state the live call was in: the agent has just asked "আপনি কি
    মন্ডল বলতে চাইছেন?" and is waiting for the answer."""
    call.session.pending = {
        "awaiting": "entity_choice", "intent": "doctor_schedule", "slots": {},
        "candidates": [dict(c) for c in candidates], "offered_date": None, "retries": 0,
    }


@pytest.mark.parametrize("said", ["হ্যাঁ", "হ্যা", "হুম", "ঠিক আছে", "yes", "haan"])
def test_5_yes_to_a_single_offered_name_is_taken_as_that_name(call, said):
    """The live loop: the offer is a yes/no question, but only a NAME was
    accepted as an answer, so "হ্যাঁ" was met with "I did not understand"."""
    _offered_mondal(call)
    replies = call.say(said)
    assert NEAR_MATCH_UNCLEAR not in replies, f"{said!r} was rejected"
    assert ("get_doctor_schedule", "Dr. A. Mondal") in call.clinic.calls
    assert call.session.pending is None


@pytest.mark.parametrize("said", [
    "ডক্টর মণ্ডল কবে বসছেন",   # live transcript -- the caller repeats the question
    "ডঃ মণ্ডল কবে বসছেন",       # live transcript
    "মণ্ডল",
    "ডাক্তার মন্ডল",
])
def test_5b_repeating_the_name_in_a_whole_sentence_resolves_it(call, said):
    # One short name scored against a whole sentence was 0.31, under the 0.55
    # floor; against the word that IS the name it is 0.80.
    _offered_mondal(call)
    replies = call.say(said)
    assert NEAR_MATCH_UNCLEAR not in replies, f"{said!r} was rejected"
    assert ("get_doctor_schedule", "Dr. A. Mondal") in call.clinic.calls


def test_5c_yes_is_not_taken_when_two_names_were_offered(call):
    """"হ্যাঁ" to "did you mean Mondal or Nandi?" names neither -- guessing
    here would undo the turn that produced the offer."""
    _offered_mondal(call, MONDAL + [{"name": "Dr. S. Nandi", "name_bn": "নন্দী"}])
    replies = call.say("হ্যাঁ")
    assert replies == [NEAR_MATCH_UNCLEAR]
    assert "get_doctor_schedule" not in call.clinic.names()


def test_5d_choosing_between_two_offered_names_still_works(call):
    _offered_mondal(call, MONDAL + [{"name": "Dr. S. Nandi", "name_bn": "নন্দী"}])
    call.say("নন্দী")
    assert ("get_doctor_schedule", "Dr. S. Nandi") in call.clinic.calls


def test_5e_no_asks_for_the_name_again_and_does_not_re_offer(call):
    _offered_mondal(call)
    replies = call.say("না")
    assert replies == [NEAR_MATCH_UNCLEAR]
    assert "get_doctor_schedule" not in call.clinic.names()
    assert call.session.pending is None, "a fresh classification gets the next turn"


# ---------------------------------------------------------------------------
# 6. Answering the offer DURING A BOOKING, and keeping the doctor afterwards.

NO_SUCH_DOCTOR = "দুঃখিত, এই নামে কোনো ডাক্তার খুঁজে পাচ্ছি না। আবার নাম বলবেন?"


def _booking_offered_mondal(call, candidates=MONDAL):
    """_open_booking parks here after clinic-api calls the spoken name
    ambiguous: awaiting "doctor_name" WITH candidates, having just asked
    "আপনি কি মন্ডল বলতে চাইছেন?"."""
    call.session.pending = {
        "awaiting": "doctor_name", "slots": {}, "turns": 1,
        "candidates": [dict(c) for c in candidates], "offered_date": None, "retries": 0,
    }


@pytest.mark.parametrize("said", ["হ্যাঁ", "হুম", "yes", "মণ্ডল", "ডাক্তার মন্ডল"])
def test_6_answering_the_booking_offer_reaches_the_doctor(call, said):
    """The live call: "হ্যাঁ" was sent to the clinic AS A DOCTOR'S NAME,
    because a second handler for this same state shadowed the one that knew
    about `candidates`."""
    _booking_offered_mondal(call)
    replies = call.say(said)
    assert NO_SUCH_DOCTOR not in replies, f"{said!r} was answered 'no such doctor'"
    assert any(c[0] == "get_doctor_availability" and c[1] == "Dr. A. Mondal"
               for c in call.clinic.calls), call.clinic.calls


def test_6b_only_one_handler_owns_the_doctor_name_state(call):
    """Two branches matched `awaiting == "doctor_name"` in _continue_pending;
    the first returned unconditionally, so the second was dead code that
    looked correct while the live one was broken."""
    import inspect
    for mod in (main, main_pcm):
        src = inspect.getsource(mod._continue_pending)
        assert src.count('if awaiting == "doctor_name"') == 1, mod.__name__


def test_6c_an_unmatched_reply_re_offers_rather_than_denying(call):
    _booking_offered_mondal(call)
    replies = call.say("ইয়ে মানে")
    assert NO_SUCH_DOCTOR not in replies
    assert "মন্ডল" in replies[0], replies


def test_6d_the_confirmed_doctor_survives_into_the_booking(call):
    """After the near-match is resolved, doctor_availability_reply() asks
    "shall I book that day?" -- so the doctor has to still be there when the
    caller answers. The state was cleared instead, and the caller who had just
    confirmed WHICH doctor was asked "কোন ডাক্তারের সাথে?" two turns later."""
    call.session.pending = {
        "awaiting": "entity_choice", "intent": "doctor_availability", "slots": {},
        "candidates": [dict(c) for c in MONDAL], "offered_date": "2026-09-29", "retries": 0,
    }
    call.say("হ্যাঁ")
    pending = call.session.pending
    assert pending is not None, "the follow-up question has nothing to answer into"
    # The CANONICAL name from the lookup, not the caller's words -- it is what
    # the booking lookup needs next.
    assert pending["slots"]["doctor_name"] == "Dr. A. Mondal"

    _sen_is_off_today(call)
    for said in ("অন্য দিনের জন্যে", "না অন্য তিন নভেম্বর কি উনি বসবেন"):
        call.say(said)
    assert not any(RESCHEDULE_PHONE_CUE in r for r in call.spoken)
    assert "find_appointments" not in call.clinic.names()
    assert call.session.pending["slots"]["date"] == "2026-11-03"


def test_4c_a_fresh_another_day_classified_as_reschedule_is_asked_about(call):
    replies = call.say("অন্য দিনের জন্যে", llm=_llm("reschedule_appointment"))
    assert replies == [new_or_move_prompt()]
    assert call.session.pending["awaiting"] == "new_or_move"
    assert "find_appointments" not in call.clinic.names()


def test_4d_new_starts_a_booking(call):
    call.say("অন্য দিনের জন্যে", llm=_llm("reschedule_appointment"))
    replies = call.say("নতুন অ্যাপয়েন্টমেন্ট")
    assert not any(RESCHEDULE_PHONE_CUE in r for r in replies)
    assert call.session.pending is not None and call.session.pending.get("flow") != "reschedule"


def test_4e_move_starts_the_reschedule_flow(call):
    call.say("অন্য দিনের জন্যে", llm=_llm("reschedule_appointment"))
    replies = call.say("আগেরটা বদলাতে চাই")
    assert any(RESCHEDULE_PHONE_CUE in r for r in replies)
    assert call.session.pending["flow"] == "reschedule"


def test_4f_a_real_reschedule_request_goes_straight_to_it(call):
    replies = call.say("অ্যাপয়েন্টমেন্টটা অন্য দিনে সরাতে চাই", llm=_llm("reschedule_appointment"))
    assert any(RESCHEDULE_PHONE_CUE in r for r in replies)
    assert call.session.pending["flow"] == "reschedule"
