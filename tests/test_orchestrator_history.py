"""The orchestrator's history flow and call record (KCD-494/495/496/498/500/501), off-pod.

The REAL `_dispatch_turn` and the new history helpers from main_pcm.py, with the pod-only
libraries stubbed and the clinic tools replaced by a fake that RECORDS what it was asked.
What is checked is the wiring: the order of the gate, that nothing personal is spoken
before a verification, that every spoken history sentence is a template with a recorded
provenance id, and that the call's record is written.

    python -m pytest tests/test_orchestrator_history.py -v
"""

import io
import json
import os
import sys
import wave

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _pod_stubs import pod_stubs
from _synth_voices import FATHER, utterance

from agent import history_templates as ht
from agent.tools_client import ToolCallError

SR = 16000
TODAY = None  # the real clock: the fake timeline is dated relative to it


class FakeWS:
    def __init__(self):
        self.texts, self.audio, self.closed = [], [], False

    async def send_text(self, t):
        self.texts.append(t)

    async def send_bytes(self, b):
        self.audio.append(b)

    async def close(self):
        self.closed = True

    def spoken(self):
        out = []
        for t in self.texts:
            try:
                d = json.loads(t)
            except ValueError:
                continue
            if d.get("sender") == "AI":
                out.append(d["text"])
        return out


class FakeTTS:
    async def synthesize(self, lang, text, speed=1.0):
        b = io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(b"\x00\x00" * (SR // 4))
        return b.getvalue()


class ASRResult:
    def __init__(self, text, agreement=0.9):
        self.text, self.decoder_agreement = text, agreement


class FakeCatalogue:
    def match(self, text, kind, lang="bn", floor=0.0):  # follows agent.fast_path.Catalogue.match (KCD-095)
        t = (text or "").lower()
        if "cbc" in t:
            return "CBC", "cbc", 1.0
        return None, None, 0.0


class FakeFastPath:
    catalogue = FakeCatalogue()


class FakeTools:
    """Records every call. `mode` decides what identify says."""

    def __init__(self):
        import datetime

        self.calls, self.events = [], []
        self.mode = "single"
        self.timeline_events = [
            {
                "id": "test_performed:11",
                "kind": "test_performed",
                "fields": {
                    "test_name": "CBC",
                    "performed_on": (datetime.date.today() - datetime.timedelta(days=40)).isoformat(),
                },
            }
        ]
        self.as_of = datetime.datetime.now().isoformat()
        self.fail_timeline = False

    async def identify_patient(self, phone):
        self.calls.append(("identify", phone))
        if self.mode == "new":
            return {"status": "new", "record_exists": False, "count": 0}
        if self.mode == "ambiguous":
            return {"status": "ambiguous", "record_exists": True, "count": 2}
        return {"status": "single", "record_exists": True, "count": 1, "patient_ref": 7}

    async def resolve_patient(self, phone, name, age=None):
        self.calls.append(("resolve", phone, name))
        if (name or "").lower().startswith("ashaa"):
            return {"status": "single", "patient_ref": 9, "basis": "phonetic", "needs_confirmation": True}
        if (name or "").lower().startswith("asha"):
            return {"status": "single", "patient_ref": 9, "basis": "exact"}
        return {"status": "none"}

    async def patient_timeline(self, patient_ref, caller_phone, call_id):
        self.calls.append(("timeline", patient_ref))
        if self.fail_timeline:
            raise ToolCallError("down")
        return {"success": True, "as_of": self.as_of, "events": self.timeline_events}

    async def patient_test_status(self, patient_ref, test_name, caller_phone, call_id):
        self.calls.append(("test_status", patient_ref, test_name))
        return {"success": True, "known": False}

    async def write_call_event(self, call_id, seq, kind, payload=None, caller_phone=None):
        self.events.append((seq, kind, payload))
        return {"success": True}


def _wav(tmp_path, samples, name):
    path = str(tmp_path / name)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


@pytest.fixture(scope="module")
def m():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


@pytest.fixture
def env(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_admission", None)
    monkeypatch.setattr(m, "_tts_router", FakeTTS())
    monkeypatch.setattr(m, "CONDITION_INPUT", "off")
    # These tests are about the fallback for a deployment that verifies some other way: with the security
    # questions OFF an unverified caller is offered a person. The questions themselves (the default) are
    # covered by tests/test_orchestrator_security.py.
    monkeypatch.setattr(m, "SECURITY_QUESTIONS", "off")
    tools = FakeTools()
    monkeypatch.setattr(m, "_tools", tools)
    monkeypatch.setattr(m, "_fast_path", FakeFastPath())
    state = {"text": "when did I last have my CBC", "lang": "en", "slots": {}, "handoffs": []}

    async def route(session, wav):
        return state["lang"], ASRResult(state["text"])

    async def resolve(session, text, lang):
        return {"intent": "unclear", "slots": dict(state["slots"]), "secondary_intent": None, "direct_reply_bn": None}

    async def handoff(session, reason, languages=None):
        state["handoffs"].append(reason)

    monkeypatch.setattr(m, "_route_and_transcribe", route)
    monkeypatch.setattr(m, "_resolve_intent", resolve)
    monkeypatch.setattr(m, "_handoff_to_human", handoff)
    session = m.CallSession(FakeWS())
    session.release_gate()
    session.disclosed_langs.add("en")
    session.call_state = m.new_call_state()
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)

    class Driver:
        pass

    d = Driver()
    d.session, d.state, d.tools = session, state, tools

    async def turn(text=None, **slots):
        if text is not None:
            state["text"] = text
        state["slots"] = slots
        session.turn_epoch = session.speak_epoch
        await m._dispatch_turn(session, _wav(tmp_path, voice, f"u{len(session.ws.texts)}.wav"))

    d.turn = turn
    yield d
    session.cleanup()


def _verified(env):
    """What KCD-203's verifier will do once it exists: a passed check for patient 7."""
    env.session.identity.claim("7", "9876543210")
    env.session.identity.verify("otp", "7", phone="9876543210")


def _no_personal_data(said: list[str]) -> bool:
    return not any(ch.isdigit() for s in said for ch in s)


# ============================================================ KCD-494 / 495: the gate


@pytest.mark.asyncio
async def test_a_history_question_first_asks_for_the_registered_number(m, env):
    await env.turn()
    assert env.session.ws.spoken() == [ht.ASK_PHONE["en"]]
    assert env.session.history_state == "need_phone"
    assert env.tools.calls == []  # nothing looked up yet


@pytest.mark.asyncio
async def test_an_unverified_caller_is_told_nothing_from_the_record_and_offered_a_person(m, env):
    await env.turn()
    await env.turn(text="nine eight seven six five four three two one zero", phone="9876543210")
    said = env.session.ws.spoken()
    assert said[-1] == ht.NEEDS_VERIFICATION["en"]
    assert ("timeline", 7) not in env.tools.calls  # the record was never even fetched
    assert _no_personal_data(said[1:]) and "CBC" not in " ".join(said[1:])
    assert env.session.awaiting_handoff_offer


@pytest.mark.asyncio
async def test_saying_yes_to_the_offer_hands_over_to_staff(m, env):
    await env.turn()
    await env.turn(text="9876543210", phone="9876543210")
    await env.turn(text="yes please")
    assert env.state["handoffs"] == ["history_requested_staff"]


@pytest.mark.asyncio
async def test_saying_no_to_the_offer_returns_to_normal_conversation(m, env):
    await env.turn()
    await env.turn(text="9876543210", phone="9876543210")
    await env.turn(text="no thanks")
    assert env.session.ws.spoken()[-1] == ht.OK_ANYTHING_ELSE["en"]
    assert env.state["handoffs"] == []


@pytest.mark.asyncio
async def test_a_number_with_no_record_says_so_and_nothing_more(m, env):
    env.tools.mode = "new"
    await env.turn()
    await env.turn(text="9876543210", phone="9876543210")
    assert env.session.ws.spoken()[-1] == ht.NO_RECORD["en"]
    assert env.session.history_state is None and env.session.pending_history is None


# ============================================================ verified: answered from fields


@pytest.mark.asyncio
async def test_a_verified_caller_gets_the_last_test_date_from_the_timeline_and_no_result(m, env):
    _verified(env)
    await env.turn(text="when did I last have my CBC")
    said = env.session.ws.spoken()[-1]
    assert "CBC" in said and "last done" in said
    assert ("timeline", 7) in env.tools.calls
    assert not any(w in said.lower() for w in ("normal", "abnormal", "result", "high", "low"))


@pytest.mark.asyncio
async def test_a_verified_caller_with_no_matching_test_hears_that_it_cannot_be_seen(m, env):
    env.tools.timeline_events = []
    _verified(env)
    await env.turn(text="when did I last have my CBC")
    assert env.session.ws.spoken()[-1] == ht.CANNOT_SEE["en"]


@pytest.mark.asyncio
async def test_a_test_that_cannot_be_named_is_asked_about_never_guessed(m, env):
    _verified(env)
    await env.turn(text="what was the last time I had a test done")
    assert env.session.ws.spoken()[-1] == ht.ASK_WHICH_TEST["en"]
    assert env.session.history_state == "need_test"
    assert not any(c[0] == "test_status" for c in env.tools.calls)


@pytest.mark.asyncio
async def test_a_lookup_failure_is_a_failure_not_a_guess(m, env):
    env.tools.fail_timeline = True
    _verified(env)
    await env.turn(text="when did I last have my CBC")
    said = env.session.ws.spoken()[-1]
    assert "CBC" not in said and _no_personal_data([said])


# ============================================================ KCD-494: a shared number


@pytest.mark.asyncio
async def test_a_shared_number_is_settled_by_an_open_question_that_lists_no_names(m, env):
    env.tools.mode = "ambiguous"
    await env.turn()
    await env.turn(text="9876543210", phone="9876543210")
    q = env.session.ws.spoken()[-1]
    assert "Who is the call for" in q
    assert env.session.history_state == "need_name"


@pytest.mark.asyncio
async def test_an_exact_name_settles_it_but_history_still_waits_for_verification(m, env):
    env.tools.mode = "ambiguous"
    await env.turn()
    await env.turn(text="9876543210", phone="9876543210")
    await env.turn(text="Asha Saha", patient_name="Asha Saha")
    assert env.session.identity.patient_ref == "9"
    assert env.session.ws.spoken()[-1] == ht.NEEDS_VERIFICATION["en"]
    assert ("timeline", 9) not in env.tools.calls


@pytest.mark.asyncio
async def test_a_sound_alike_name_is_not_accepted_the_agent_asks_again(m, env):
    env.tools.mode = "ambiguous"
    await env.turn()
    await env.turn(text="9876543210", phone="9876543210")
    await env.turn(text="Ashaa", patient_name="Ashaa")
    assert env.session.identity.patient_ref is None
    assert env.session.history_state == "need_name"


@pytest.mark.asyncio
async def test_two_failed_attempts_go_to_a_person(m, env):
    env.tools.mode = "ambiguous"
    await env.turn()
    await env.turn(text="9876543210", phone="9876543210")
    await env.turn(text="Nobody", patient_name="Nobody")
    await env.turn(text="Nobody again", patient_name="Nobody again")
    assert env.state["handoffs"] == ["history_patient_unclear"]


@pytest.mark.asyncio
async def test_an_unclear_number_twice_goes_to_a_person(m, env):
    await env.turn()
    await env.turn(text="it is one two three", phone="123")
    await env.turn(text="it is one two three", phone="123")
    assert env.state["handoffs"] == ["history_phone_unclear"]


# ============================================================ KCD-500: model text cannot claim history


@pytest.mark.asyncio
async def test_smalltalk_that_claims_the_callers_past_is_replaced(m, env, monkeypatch):
    async def resolve(session, text, lang):
        return {
            "intent": "smalltalk",
            "slots": {},
            "secondary_intent": None,
            "direct_reply_bn": "Good to hear from you again. You had a blood test last month.",
        }

    monkeypatch.setattr(m, "_resolve_intent", resolve)
    await env.turn(text="hello there")
    said = env.session.ws.spoken()
    assert said and "blood test last month" not in said[-1]


# ============================================================ KCD-501: the call record


@pytest.mark.asyncio
async def test_history_statements_are_recorded_with_their_provenance_ids(m, env):
    rec = m.CallRecorder("c1", m._write_call_event)
    env.session.recorder = rec
    _verified(env)
    await env.turn(text="when did I last have my CBC")
    left = await rec.finish("completed")
    assert left == 0
    kinds = [k for _, k, _ in env.tools.events]
    assert "history" in kinds and kinds[-1] == "end"
    ids = [i for _, k, p in env.tools.events if k == "history" for i in (p["statements"])]
    assert "test_performed:11" in ids


@pytest.mark.asyncio
async def test_a_recorder_that_cannot_write_never_slows_or_breaks_a_turn(m, env, monkeypatch):
    async def down(*a, **k):
        raise ToolCallError("clinic down")

    monkeypatch.setattr(m, "_tools", type("T", (FakeTools,), {"write_call_event": down})())
    env.session.recorder = m.CallRecorder("c2", m._write_call_event)
    await env.turn()
    assert env.session.ws.spoken() == [ht.ASK_PHONE["en"]]  # the caller heard the reply regardless
    assert env.session.recorder.pending >= 1
