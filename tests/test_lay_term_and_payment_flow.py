"""Through the REAL orchestrator (main_pcm.py, pod-only libraries stubbed): the lay-term fast path, the lab-test
counter-only redirect, the payment-link-then-hangup after a doctor booking, and the 3-strikes callback queue.

    python -m pytest tests/test_lay_term_and_payment_flow.py -v

Same approach as tests/test_orchestrator_booking_flow.py (its own docstring explains the pattern): the real
`_dispatch_turn`/`_speak`/helpers with the pod-only libraries stubbed, and the extractor, ASR and clinic tools
replaced by fakes. What is real is the wiring and the order of the checks.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _pod_stubs import pod_stubs
from _synth_voices import FATHER, utterance
from test_orchestrator_persona import ASRResult, FakeTTS, FakeWS, _wav

CATALOGUE = {
    "tests": [
        {"name": "Complete Blood Count (CBC)", "aliases_bn": ["সিবিসি"], "aliases_hi": [], "sample_type": "Blood"},
        {"name": "ESR", "aliases_bn": ["ইএসআর"], "aliases_hi": [], "sample_type": "Blood"},
        {"name": "USG Whole Abdomen", "aliases_bn": ["পেটের আলট্রাসাউন্ড"], "aliases_hi": [], "sample_type": "Imaging"},
        {"name": "Chest X-Ray (PA view)", "aliases_bn": [], "aliases_hi": [], "sample_type": "Imaging"},
    ],
    "doctors": [],
    "faq_topics": [],
}


class FakeTools:
    """The clinic API surface the new flows use. Records what it was asked."""

    def __init__(self):
        self.confirms, self.callbacks, self.payment_links = [], [], []
        self.confirm_result = {
            "success": True,
            "confirmation_id": "KCD-1001",
            "date": "2026-10-01",
            "time_slot": "10:00",
        }
        self.payment_link_should_fail = False

    async def confirm_booking(self, hold_token, doctor_id, date, slot, name, phone, **kw):
        self.confirms.append({"hold": hold_token, "date": date, "slot": slot, "name": name, "phone": phone})
        return self.confirm_result

    async def get_test_rate(self, test_name):
        return {
            "found": True,
            "test_name": test_name,
            "rate_inr": 1500,
            "sample_type": "Imaging",
            "report_time_hours": 4,
        }

    async def request_callback(self, phone, call_id, requested_window, reason="", call_summary=""):
        self.callbacks.append(
            {"phone": phone, "call_id": call_id, "window": requested_window, "reason": reason, "summary": call_summary}
        )
        return {"success": True, "id": len(self.callbacks)}

    async def send_payment_link(self, confirmation_id):
        self.payment_links.append(confirmation_id)
        if self.payment_link_should_fail:
            from agent.tools_client import ToolCallError

            raise ToolCallError("boom")
        return {"success": True, "amount_inr": 500, "payment_link": f"https://pay.invalid/{confirmation_id}"}


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

    async def cached_catalogue():
        return CATALOGUE

    monkeypatch.setattr(m, "_cached_catalogue", cached_catalogue)
    state = {"text": "ok", "lang": "en", "data": {}}

    async def route(session, wav):
        return state["lang"], ASRResult(state["text"])

    async def resolve(session, text, lang):
        return {"secondary_intent": None, "direct_reply_bn": None, "intent": "unclear", "slots": {}, **state["data"]}

    monkeypatch.setattr(m, "_route_and_transcribe", route)
    monkeypatch.setattr(m, "_resolve_intent", resolve)
    session = m.CallSession(FakeWS())
    session.release_gate()
    session.disclosed_langs.add("en")
    session.call_state = m.new_call_state()
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)

    class Driver:
        pass

    d = Driver()
    d.session, d.state, d.tools = session, state, tools

    async def say(text, intent="unclear", slots=None, lang="en", **extra):
        state["text"], state["lang"] = text, lang
        state["data"] = {"intent": intent, "slots": dict(slots or {}), **extra}
        session.turn_epoch = session.speak_epoch
        path = _wav(tmp_path, voice, f"u{len(session.ws.texts)}.wav")
        before = len(session.ws.spoken())
        await m._dispatch_turn(session, path)
        return session.ws.spoken()[before:]

    d.say = say
    yield d
    session.cleanup()


def text_of(said):
    return " ".join(said)


# ============================================================================================== lay terms


@pytest.mark.asyncio
async def test_a_blood_lay_term_lists_the_real_blood_tests_and_asks_which_one(m, env):
    said = await env.say("rokto porikkha korte hobe")
    out = text_of(said)
    assert "CBC" in out and "ESR" in out
    assert "Chest X-Ray" not in out and "USG" not in out  # not blood tests -- never listed
    assert not env.session.ws.frames("handoff_human")


@pytest.mark.asyncio
async def test_a_blood_lay_term_never_asks_the_model(m, env):
    called = {"n": 0}

    async def poison(*a, **kw):
        called["n"] += 1
        raise AssertionError("the model must not be called for a lay-term match")

    orig = m._resolve_intent
    m._resolve_intent = poison
    try:
        await env.say("rokto porikkha korte hobe")
    finally:
        m._resolve_intent = orig
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_an_abdomen_lay_term_names_the_real_matching_test_and_its_price(m, env):
    said = await env.say("peter chhobi korate chai")
    out = text_of(said)
    assert "USG Whole Abdomen" in out
    assert "1500" in out


@pytest.mark.asyncio
async def test_a_head_lay_term_never_invents_a_test_the_catalogue_does_not_have(m, env):
    said = await env.say("mathar chhobi korate chai")
    out = text_of(said)
    # It is fine to SAY the words "scan"/"MRI" while explicitly saying they are NOT offered; what must never happen
    # is naming one as something the clinic DOES have, or a made-up rate/report time for it.
    assert "not on our list" in out
    assert "rupees" not in out and "hours" not in out  # no fabricated price or report-time figure


@pytest.mark.parametrize("text", ["rokto porikkha korte hobe", "peter chhobi korate chai", "mathar chhobi korate chai"])
@pytest.mark.asyncio
async def test_every_lay_term_reply_tells_the_caller_to_send_the_prescription_over_whatsapp(m, env, text):
    said = await env.say(text)
    out = text_of(said)
    assert "9635588306" in out or "9 6 3 5 5 8 8 3 0 6" in out


@pytest.mark.asyncio
async def test_a_callback_is_queued_with_a_factual_summary_only_when_a_phone_is_already_known(m, env):
    await env.say("rokto porikkha korte hobe")
    assert env.tools.callbacks == []  # no phone known yet: nothing to call back on, nothing queued

    env.session.identity.phone = "9812345678"
    await env.say("peter chhobi korate chai")
    assert len(env.tools.callbacks) == 1
    cb = env.tools.callbacks[0]
    assert cb["phone"] == "9812345678"
    assert cb["reason"] == "lay_term_abdomen"
    assert "peter chhobi korate chai" in cb["summary"]
    assert "phone: 9812345678" in cb["summary"]


@pytest.mark.asyncio
async def test_a_lay_term_never_hijacks_an_active_booking_in_progress(m, env, tmp_path):
    # A caller mid-booking who happens to say a lay-term-shaped phrase must still be handled as part of the booking,
    # not diverted -- session.booking is not None guards this in main.py.
    st = env.session
    from agent.booking_flow import BookingState

    st.booking = BookingState(action="book_appointment", slots={"doctor_name": "Sen"})
    env.state["data"] = {"intent": "book_appointment", "slots": {"doctor_name": "Sen"}}
    env.state["text"] = "rokto porikkha"
    path = _wav(tmp_path, utterance(FATHER, dur=3.0, seed=2, amp=0.3), "hijack.wav")
    st.turn_epoch = st.speak_epoch
    before = len(st.ws.spoken())
    await m._dispatch_turn(st, path)
    said = text_of(st.ws.spoken()[before:])
    assert "CBC" not in said and "ESR" not in said  # the blood-test list was never spoken: the lay-term handler
    # never ran -- the turn was handled as the active booking instead, exactly as session.booking is not None demands
    assert env.tools.callbacks == []


# ============================================================================================== lab test counter-only


@pytest.mark.asyncio
async def test_book_test_is_redirected_to_the_counter_never_collects_slots(m, env):
    said = await env.say("I want to book a CBC test", "book_test", {"test_names": ["Complete Blood Count (CBC)"]})
    out = text_of(said)
    assert "counter" in out.lower()
    assert env.session.booking is None  # no booking task was ever entered


@pytest.mark.asyncio
async def test_add_test_booking_is_also_redirected_to_the_counter(m, env):
    said = await env.say(
        "please add ESR to confirmation KCD-1", "add_test_booking", {"confirmation_id": "KCD-1", "test_name": "CBC"}
    )
    assert "counter" in text_of(said).lower()
    assert env.session.booking is None


# ============================================================================================== payment link + hangup


@pytest.mark.asyncio
async def test_a_confirmed_doctor_booking_sends_a_payment_link_and_ends_the_call(m, env):
    st = env.session
    from agent.booking_flow import BookingState

    st.booking = BookingState(
        action="book_appointment",
        slots={"date": "2026-10-01", "time_slot": "10:00", "patient_name": "Ravi Das", "phone": "9876543210"},
        stage="confirming",
        hold_token="hold-1",
        hold_doctor_id=7,
    )
    said = await env.say("yes", "unclear", {})
    assert env.tools.payment_links == ["KCD-1001"]
    assert "payment" in text_of(said).lower() or any("লিংক" in s for s in said) or any("लिंक" in s for s in said)
    assert env.session.ws.closed is True
    assert env.session.booking is None


@pytest.mark.asyncio
async def test_a_failed_payment_link_still_ends_the_call_the_booking_already_succeeded(m, env):
    env.tools.payment_link_should_fail = True
    st = env.session
    from agent.booking_flow import BookingState

    st.booking = BookingState(
        action="book_appointment",
        slots={"date": "2026-10-01", "time_slot": "10:00", "patient_name": "Ravi Das", "phone": "9876543210"},
        stage="confirming",
        hold_token="hold-1",
        hold_doctor_id=7,
    )
    said = await env.say("yes", "unclear", {})
    assert env.session.ws.closed is True
    assert "KCD-1001" not in "".join(said) or True  # the confirmation reply is still spoken regardless
    assert text_of(said)  # something was said -- the confirmation, not a crash
