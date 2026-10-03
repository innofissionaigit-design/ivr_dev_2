"""KCD-103 (several fields in one question), KCD-104 (a correction or a topic change never loses the booking) and
KCD-101 (one wall-clock budget, a holding phrase once, then the apology), through the REAL orchestrator, off-pod.

    python -m pytest tests/test_orchestrator_booking_flow.py -v

Same approach as tests/test_orchestrator_persona.py: the real `_dispatch_turn`, `_speak` and helpers from
main_pcm.py with the pod-only libraries stubbed, and the extractor, ASR and clinic tools replaced by fakes. What is
real is the wiring and the order of the checks. Timing tests use small REAL waits (a fraction of a second) so a
deadline is a wall-clock fact, not a mocked one.
"""

import os
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _pod_stubs import pod_stubs
from _synth_voices import FATHER, utterance
from test_orchestrator_persona import ASRResult, FakeTTS, FakeWS, _wav

from agent.phrases import phrase

ASK_SCHEDULE = "For the appointment, which doctor, which day and what time?"
ASK_DAY_TIME = "For the appointment, which day and what time?"
ASK_DETAILS = "May I have the patient's name and a phone number for the confirmation?"
BACK = "Coming back to your booking."
PRICE = "The CBC costs three hundred and fifty rupees."


class FakeTools:
    """The clinic API as the booking flow uses it. Records what it was asked, answers like the real one."""

    def __init__(self):
        self.holds, self.confirms, self.cancels, self.conflicts = [], [], [], []

    async def hold_slot(self, doctor, date, slot):
        self.holds.append((doctor, date, slot))
        return {"success": True, "hold_token": f"hold-{len(self.holds)}", "doctor_id": 7}

    async def booking_conflict(self, phone, date, slot):
        self.conflicts.append((phone, date, slot))
        return {"conflict": False}

    async def confirm_booking(self, hold_token, doctor_id, date, slot, name, phone, **kw):
        self.confirms.append({"hold": hold_token, "date": date, "slot": slot, "name": name, "phone": phone})
        return {
            "success": True,
            "date": date,
            "time_slot": slot,
            "confirmation_id": "KCD-1001",
            "doctor_name": "Dr. A. Sen",
        }

    async def cancel_appointment(self, confirmation_id, confirm_charge):
        self.cancels.append(confirmation_id)
        return {"success": True, "charge_inr": 0}

    async def send_payment_link(self, confirmation_id):
        # KCD-lay-terms: main.py calls this right after a doctor booking is confirmed, then ends the call.
        self.payment_links = [*getattr(self, "payment_links", []), confirmation_id]
        return {"success": True, "amount_inr": 500, "payment_link": f"https://pay.invalid/{confirmation_id}"}

    async def request_callback(self, phone, call_id, requested_window, reason="", call_summary=""):
        self.callbacks = [*getattr(self, "callbacks", []), {"phone": phone, "reason": reason}]
        return {"success": True, "id": len(self.callbacks)}

    async def get_patient_senior(self, phone):
        return False

    async def set_patient_senior(self, phone, value):
        return None


@pytest.fixture(scope="module")
def m():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


@pytest.fixture
def env(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_admission", None)
    monkeypatch.setattr(m, "_tts_router", FakeTTS())
    monkeypatch.setattr(m, "CONDITION_INPUT", "off")
    tools = FakeTools()
    monkeypatch.setattr(m, "_tools", tools)
    state = {"text": "ok", "lang": "en", "data": {}, "reply": PRICE}

    async def route(session, wav):
        return state["lang"], ASRResult(state["text"])

    async def resolve(session, text, lang):
        return {"secondary_intent": None, "direct_reply_bn": None, **state["data"]}

    async def answer(intent, slots, lang):
        return state["reply"]

    monkeypatch.setattr(m, "_route_and_transcribe", route)
    monkeypatch.setattr(m, "_resolve_intent", resolve)
    monkeypatch.setattr(m, "_answer_enquiry_intent", answer)
    session = m.CallSession(FakeWS())
    session.release_gate()
    session.disclosed_langs.add("en")
    session.call_state = m.new_call_state()
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)

    class Driver:
        pass

    d = Driver()
    d.session, d.state, d.tools = session, state, tools

    async def say(text, intent="unclear", slots=None, **extra):
        """One caller turn: what ASR heard, and what the extractor made of it."""
        state["text"] = text
        state["data"] = {"intent": intent, "slots": dict(slots or {}), **extra}
        session.turn_epoch = session.speak_epoch
        path = _wav(tmp_path, voice, f"u{len(session.ws.texts)}.wav")
        before = len(session.ws.spoken())
        await m._dispatch_turn(session, path)
        return session.ws.spoken()[before:]

    d.say = say
    d.last = lambda: session.ws.spoken()[-1]
    yield d
    session.cleanup()


def book(env, **slots):
    return env.say("I want to book", "book_appointment", slots)


def text_of(said):
    return " ".join(said)


# ========================================================================= KCD-103: several fields in one question


@pytest.mark.asyncio
async def test_a_booking_is_collected_with_grouped_questions(m, env):
    said = await book(env, doctor_name="Sen")
    assert ASK_DAY_TIME in text_of(said)  # the doctor is known: day and time together
    said = await env.say("tomorrow at ten", "book_appointment", {"date": "2026-10-01", "time_slot": "10:00"})
    assert ASK_DETAILS in text_of(said)  # then the name and number together
    said = await env.say("Ravi Das 9876543210", "book_appointment", {"patient_name": "Ravi Das", "phone": "9876543210"})
    assert env.session.booking.stage == "confirming"
    assert env.tools.holds == [("Sen", "2026-10-01", "10:00")]


@pytest.mark.asyncio
async def test_an_opening_with_nothing_asks_for_the_whole_first_group(m, env):
    said = await env.say("I want to book an appointment", "book_appointment", {})
    assert ASK_SCHEDULE in text_of(said)


@pytest.mark.asyncio
async def test_a_partial_answer_keeps_what_was_given_and_asks_the_rest_alone(m, env):
    await book(env, doctor_name="Sen")
    said = await env.say("tomorrow", "book_appointment", {"date": "2026-10-01"})
    st = env.session.booking
    assert st.slots["doctor_name"] == "Sen" and st.slots["date"] == "2026-10-01"
    assert "what time" in text_of(said).lower() and ASK_DAY_TIME not in text_of(said)  # only the time is asked


@pytest.mark.asyncio
async def test_a_senior_caller_is_asked_one_thing_at_a_time_whatever_the_group(m, env):
    m.apply_caller_state(env.session.call_state, senior=True)
    said = await env.say("I want to book an appointment", "book_appointment", {})
    spoken = text_of(said)
    assert ASK_SCHEDULE not in spoken and "which day" not in spoken.lower()
    assert spoken.count("?") == 1


@pytest.mark.asyncio
async def test_a_distressed_caller_is_asked_one_thing_at_a_time(m, env):
    m.apply_caller_state(env.session.call_state, caller_state="distressed")
    said = await env.say("I want to book an appointment", "book_appointment", {"doctor_name": "Sen"})
    assert ASK_DAY_TIME not in text_of(said) and text_of(said).count("?") <= 1


@pytest.mark.asyncio
async def test_a_grouped_question_is_never_more_than_the_normal_policy_allows(m, env):
    from agent.speech_policy import check_reply

    said = await env.say("I want to book", "book_appointment", {})
    assert check_reply(said[-1], env.session.policy) == []


# ======================================================================== KCD-104: a topic change keeps the booking

STAGES = ["after_doctor", "after_date", "after_time", "all_but_phone", "confirming"]


async def reach(env, stage):
    """Drive a booking to `stage`; returns the booking's slots at that point."""
    await book(env, doctor_name="Sen")
    if stage != "after_doctor":
        await env.say("tomorrow", "book_appointment", {"date": "2026-10-01"})
    if stage in ("after_time", "all_but_phone", "confirming"):
        await env.say("at ten", "book_appointment", {"time_slot": "10:00"})
    if stage in ("all_but_phone", "confirming"):
        await env.say("Ravi Das", "book_appointment", {"patient_name": "Ravi Das"})
    if stage == "confirming":
        await env.say("9876543210", "book_appointment", {"phone": "9876543210"})
    return dict(env.session.booking.slots)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", STAGES)
async def test_a_question_at_any_stage_is_answered_and_the_caller_is_brought_back(m, env, stage):
    before = await reach(env, stage)
    st = env.session.booking
    stage_before = st.stage
    said = await env.say("what does the CBC cost", "test_rate", {"test_name": "CBC"})
    spoken = text_of(said)
    assert PRICE in spoken and BACK in spoken  # answered, then back to the booking
    assert env.session.booking is st and dict(st.slots) == before  # nothing captured was lost or changed
    assert st.stage == stage_before  # and the step it was at is still pending
    if stage == "confirming":
        assert "Shall I confirm it?" in spoken


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["after_doctor", "after_date", "after_time", "all_but_phone"])
async def test_after_the_aside_the_booking_carries_on_from_where_it_was(m, env, stage):
    await reach(env, stage)
    await env.say("what does the CBC cost", "test_rate", {"test_name": "CBC"})
    st = env.session.booking
    # finish it with everything still missing, in one go
    await env.say(
        "all of it",
        "book_appointment",
        {"date": "2026-10-01", "time_slot": "10:00", "patient_name": "Ravi Das", "phone": "9876543210"},
    )
    assert st.stage == "confirming" and st.slots["doctor_name"] == "Sen"
    await env.say("yes", "unclear", {})
    assert len(env.tools.confirms) == 1 and env.tools.confirms[0]["slot"] == "10:00"


@pytest.mark.asyncio
async def test_a_question_at_the_confirmation_step_is_answered_and_a_later_yes_still_confirms(m, env):
    await reach(env, "confirming")
    await env.say("do I need to fast for the CBC", "test_prep", {"test_name": "CBC"})
    assert env.session.booking.stage == "confirming" and not env.tools.confirms  # answering it confirmed nothing
    await env.say("yes", "unclear", {})
    assert len(env.tools.confirms) == 1 and env.tools.confirms[0]["name"] == "Ravi Das"
    assert env.session.booking is None


@pytest.mark.asyncio
async def test_small_talk_in_the_middle_of_a_booking_comes_back_to_it(m, env):
    await reach(env, "after_date")
    said = await env.say("thanks", "smalltalk", {}, direct_reply_bn="You are welcome.")
    assert "You are welcome." in text_of(said) and BACK in text_of(said)


@pytest.mark.asyncio
async def test_an_unclear_reply_at_the_confirmation_step_repeats_the_confirmation_without_guessing(m, env):
    await reach(env, "confirming")
    said = await env.say("hmm", "unclear", {})
    assert env.session.booking.stage == "confirming" and not env.tools.confirms
    assert "Ravi Das" in text_of(said)  # the readback, again


@pytest.mark.asyncio
async def test_no_resume_line_is_added_to_a_reply_that_already_asks_a_question(m, env):
    await reach(env, "after_date")
    env.state["reply"] = "I found no test by that name. Did you mean CBC?"
    said = await env.say("what does the CVC cost", "test_rate", {"test_name": "CVC"})
    assert BACK not in text_of(said)


@pytest.mark.asyncio
async def test_without_a_booking_in_progress_nothing_is_added(m, env):
    said = await env.say("what does the CBC cost", "test_rate", {"test_name": "CBC"})
    assert BACK not in text_of(said) and PRICE in text_of(said)


@pytest.mark.asyncio
async def test_a_senior_caller_is_brought_back_with_one_question_only(m, env):
    await reach(env, "after_doctor")
    m.apply_caller_state(env.session.call_state, senior=True)
    said = await env.say("what does the CBC cost", "test_rate", {"test_name": "CBC"})
    assert BACK in text_of(said) and text_of(said).count("?") == 1


# ============================================================================= KCD-104: corrections at every stage


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["after_time", "all_but_phone", "confirming"])
async def test_a_corrected_time_updates_that_slot_only_and_a_new_slot_is_held(m, env, stage):
    before = await reach(env, stage)
    said = await env.say("no, make it eleven", "book_appointment", {"time_slot": "11:00"})
    st = env.session.booking
    assert "11:00" in text_of(said)  # said back: what changed
    assert st.slots["time_slot"] == "11:00"
    assert {k: v for k, v in st.slots.items() if k != "time_slot"} == {
        k: v for k, v in before.items() if k != "time_slot"
    }
    assert env.tools.holds[-1] == ("Sen", "2026-10-01", "11:00")  # the hold moved with the slot


@pytest.mark.asyncio
async def test_a_correction_after_the_final_prompt_is_confirmed_again_and_commits_the_corrected_slot(m, env):
    await reach(env, "confirming")
    await env.say("actually the name is Ravi Dass", "book_appointment", {"patient_name": "Ravi Dass"})
    assert env.session.booking.stage == "confirming"  # read back again before anything is written
    assert not env.tools.confirms
    await env.say("yes", "unclear", {})
    assert env.tools.confirms[0]["name"] == "Ravi Dass" and env.tools.confirms[0]["phone"] == "9876543210"


@pytest.mark.asyncio
async def test_a_corrected_day_is_held_again_and_the_old_hold_is_not_used(m, env):
    await reach(env, "confirming")
    old_hold = env.session.booking.hold_token
    await env.say("no, the second", "book_appointment", {"date": "2026-10-02"})
    await env.say("yes", "unclear", {})
    assert env.tools.holds[-1] == ("Sen", "2026-10-02", "10:00")
    assert env.tools.confirms[0]["hold"] != old_hold


# =========================================================================== KCD-104: a different task is not lost


@pytest.mark.asyncio
async def test_a_different_task_sets_the_booking_aside_and_offers_it_back_when_done(m, env):
    await reach(env, "after_date")
    said = await env.say("cancel my other appointment KCD-9", "cancel_appointment", {"confirmation_id": "KCD-9"})
    assert env.session.suspended is not None and env.session.suspended.slots["doctor_name"] == "Sen"
    assert env.session.booking.action == "cancel_appointment"
    said = await env.say("yes", "unclear", {})
    assert env.tools.cancels == ["KCD-9"]
    assert "Earlier we were talking about booking an appointment. Shall I go back to it?" in text_of(said)
    said = await env.say("yes", "unclear", {})
    st = env.session.booking
    assert st.action == "book_appointment" and st.slots == {"doctor_name": "Sen", "date": "2026-10-01"}
    assert "what time" in text_of(said).lower()  # carries on with the next detail


@pytest.mark.asyncio
async def test_declining_the_offer_drops_the_set_aside_booking(m, env):
    await reach(env, "after_date")
    await env.say("cancel KCD-9", "cancel_appointment", {"confirmation_id": "KCD-9"})
    await env.say("yes", "unclear", {})
    await env.say("no", "unclear", {})
    assert env.session.booking is None and env.session.suspended is None and not env.session.awaiting_resume


@pytest.mark.asyncio
async def test_a_task_started_with_nothing_captured_is_not_offered_back(m, env):
    await env.say("I want to book", "book_appointment", {})
    await env.say("cancel KCD-9", "cancel_appointment", {"confirmation_id": "KCD-9"})
    assert env.session.suspended is None


@pytest.mark.asyncio
async def test_aborting_the_new_task_also_offers_the_old_one_back(m, env):
    await reach(env, "after_doctor")
    await env.say("cancel KCD-9", "cancel_appointment", {"confirmation_id": "KCD-9"})
    said = await env.say("no", "unclear", {})  # "no" aborts a cancellation
    assert "Shall I go back to it?" in text_of(said)


# ===================================================== KCD-101: one budget, a holding phrase once, then the apology


class SlowCache:
    def __init__(self, get_s=0.0):
        self.get_s, self.puts = get_s, []

    def get(self, key):
        time.sleep(self.get_s)
        return None, "miss"

    def put(self, key, data):
        self.puts.append(key)


GOOD = {"intent": "test_rate", "slots": {"test_name": "CBC"}, "secondary_intent": None, "direct_reply_bn": None}


@pytest.fixture
def budget(m, env, monkeypatch):
    """The real `_resolve_intent` with a small budget, a small filler threshold and a fake cache/extractor."""
    monkeypatch.setattr(m, "INTENT_BUDGET_S", 1.0)
    monkeypatch.setattr(m, "CACHE_LOOKUP_MAX_S", 0.2)
    monkeypatch.setattr(m, "_fast_path", None)

    async def no_reload():
        return None

    monkeypatch.setattr(m, "_maybe_reload_fast_path", no_reload)
    real_filler = m._await_with_filler
    monkeypatch.setattr(m, "_await_with_filler", lambda s, aw, lang, threshold_s=0.25: real_filler(s, aw, lang, 0.25))
    monkeypatch.setattr(m, "_intent_cache", SlowCache())
    return env


_REAL_RESOLVE = None


@pytest.fixture(autouse=True)
def _capture_real_resolve(m):
    global _REAL_RESOLVE
    if _REAL_RESOLVE is None:
        _REAL_RESOLVE = m._resolve_intent
    yield


def holding(env):
    return [t for t in env.session.ws.spoken() if t == phrase("please_wait", "en")]


async def resolve_real(m, env, monkeypatch):
    monkeypatch.setattr(m, "_resolve_intent", _REAL_RESOLVE)
    t0 = time.monotonic()
    try:
        return await m._resolve_intent(env.session, "what is the price of a CBC", "en"), time.monotonic() - t0
    except m.ExtractionError as e:
        return e, time.monotonic() - t0


@pytest.mark.asyncio
async def test_a_fast_answer_speaks_no_holding_phrase(m, budget, monkeypatch):
    monkeypatch.setattr(m, "extract_intent", lambda *a, **k: (dict(GOOD), {"total_time_s": 0.01, "attempts": 1}))
    result, took = await resolve_real(m, budget, monkeypatch)
    assert result["intent"] == "test_rate" and took < 0.25 and holding(budget) == []


@pytest.mark.asyncio
async def test_a_slow_answer_speaks_the_holding_phrase_exactly_once_then_answers(m, budget, monkeypatch):
    def slow(text, max_retries=2, lang="bn", deadline_s=12.0):
        time.sleep(0.5)
        return dict(GOOD), {"total_time_s": 0.5, "attempts": 1}

    monkeypatch.setattr(m, "extract_intent", slow)
    result, took = await resolve_real(m, budget, monkeypatch)
    assert result["intent"] == "test_rate" and len(holding(budget)) == 1


@pytest.mark.asyncio
async def test_no_answer_ends_in_the_apology_path_at_the_budget_not_later(m, budget, monkeypatch):
    def hangs(text, max_retries=2, lang="bn", deadline_s=12.0):
        time.sleep(min(deadline_s, 5.0))  # honours its deadline like the real extractor
        raise m.ExtractionError("deadline")

    monkeypatch.setattr(m, "extract_intent", hangs)
    result, took = await resolve_real(m, budget, monkeypatch)
    assert isinstance(result, m.ExtractionError)
    assert 0.8 <= took <= 1.6, took  # the 1.0 s budget, plus scheduling slack
    assert len(holding(budget)) == 1  # said once, not once per attempt


@pytest.mark.asyncio
async def test_a_slow_cache_and_a_slow_model_together_still_end_inside_the_budget(m, budget, monkeypatch):
    monkeypatch.setattr(m, "_intent_cache", SlowCache(get_s=0.9))  # the embedding lookup alone takes 0.9 s
    seen = {}

    def slow(text, max_retries=2, lang="bn", deadline_s=12.0):
        seen["deadline_s"] = deadline_s
        time.sleep(min(deadline_s, 5.0))
        raise m.ExtractionError("deadline")

    monkeypatch.setattr(m, "extract_intent", slow)
    result, took = await resolve_real(m, budget, monkeypatch)
    assert isinstance(result, m.ExtractionError)
    assert took <= 1.0 + 0.6, took  # before this change: 0.9 + the model's whole deadline
    assert seen["deadline_s"] < 1.0  # the model was given what was LEFT, not a fresh budget


@pytest.mark.asyncio
async def test_a_slow_cache_is_a_miss_and_the_model_still_answers(m, budget, monkeypatch):
    monkeypatch.setattr(m, "_intent_cache", SlowCache(get_s=0.6))
    monkeypatch.setattr(m, "extract_intent", lambda *a, **k: (dict(GOOD), {"total_time_s": 0.01, "attempts": 1}))
    result, took = await resolve_real(m, budget, monkeypatch)
    assert result["intent"] == "test_rate" and took < 0.6


@pytest.mark.asyncio
async def test_a_model_that_ignores_its_deadline_is_still_cut_off_at_the_budget(m, budget, monkeypatch):
    def deaf(text, max_retries=2, lang="bn", deadline_s=12.0):
        time.sleep(2.5)  # ignores deadline_s entirely
        return dict(GOOD), {"total_time_s": 2.5, "attempts": 1}

    monkeypatch.setattr(m, "extract_intent", deaf)
    result, took = await resolve_real(m, budget, monkeypatch)
    assert isinstance(result, m.ExtractionError) and took <= 2.3, took


@pytest.mark.asyncio
async def test_when_the_cache_uses_up_the_budget_the_model_is_not_asked_at_all(m, budget, monkeypatch):
    monkeypatch.setattr(m, "INTENT_BUDGET_S", 0.6)
    monkeypatch.setattr(m, "CACHE_LOOKUP_MAX_S", 0.55)
    monkeypatch.setattr(m, "_intent_cache", SlowCache(get_s=0.5))
    called = []
    monkeypatch.setattr(
        m, "extract_intent", lambda *a, **k: called.append(1) or (dict(GOOD), {"total_time_s": 0, "attempts": 1})
    )
    result, took = await resolve_real(m, budget, monkeypatch)
    assert isinstance(result, m.ExtractionError) and not called


@pytest.mark.asyncio
async def test_the_caller_hears_the_apology_when_extraction_fails_and_nothing_else(
    m, env, budget, monkeypatch, tmp_path
):
    """Through the real `_dispatch_turn`: the spoken apology is the llm_failure phrase, and the turn ends."""

    def hangs(text, max_retries=2, lang="bn", deadline_s=12.0):
        time.sleep(min(deadline_s, 5.0))
        raise m.ExtractionError("deadline")

    monkeypatch.setattr(m, "extract_intent", hangs)
    monkeypatch.setattr(m, "_resolve_intent", _REAL_RESOLVE)
    env.state["text"] = "what is the price of a CBC"
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)
    env.session.turn_epoch = env.session.speak_epoch
    t0 = time.monotonic()
    await m._dispatch_turn(env.session, _wav(tmp_path, voice, "late.wav"))
    took = time.monotonic() - t0
    said = env.session.ws.spoken()
    assert said[-1] == phrase("llm_failure", "en") and said.count(phrase("llm_failure", "en")) == 1
    assert took <= 2.2, took


# ========================================= KCD-104: a bare "no" still reopens; "no, ..." is heard as what it says


@pytest.mark.asyncio
async def test_a_bare_no_at_the_confirmation_step_reopens_and_asks_again_without_consulting_the_extractor(
    m, env, monkeypatch
):
    await reach(env, "confirming")
    calls = []

    async def must_not_run(*a, **k):
        calls.append(1)
        return {"intent": "unclear", "slots": {}, "secondary_intent": None, "direct_reply_bn": None}

    monkeypatch.setattr(m, "_resolve_intent", must_not_run)
    said = await env.say("no", "unclear", {})
    assert not calls and env.session.booking.stage == "collecting"
    assert "Which doctor" in text_of(said)
    assert env.session.booking.slots["patient_name"] == "Ravi Das"  # what was captured stays


@pytest.mark.asyncio
async def test_a_no_with_nothing_to_change_after_more_words_still_reopens(m, env):
    await reach(env, "confirming")
    said = await env.say("no that is not right", "unclear", {})
    assert env.session.booking.stage == "collecting" and "Which doctor" in text_of(said)


@pytest.mark.asyncio
async def test_a_no_with_a_correction_in_it_applies_the_correction_and_reads_it_back(m, env):
    await reach(env, "confirming")
    said = await env.say("no, make it eleven", "book_appointment", {"time_slot": "11:00"})
    assert env.session.booking.slots["time_slot"] == "11:00" and env.session.booking.slots["doctor_name"] == "Sen"
    assert "11:00" in text_of(said) and "Which doctor" not in text_of(said)
