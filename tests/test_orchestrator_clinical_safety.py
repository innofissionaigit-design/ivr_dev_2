"""The orchestrator's wiring for a clinical-interpretation question, off-pod.

WHY THIS FILE EXISTS SEPARATELY FROM tests/test_clinical_safety.py:

That file proves the detector catches the question and that the reply offers a
clinician. It cannot prove either of the two things this port actually depends
on:

  1. That the guard runs BEFORE the model. "Routed to a human by policy rather
     than by prompt wording" is the acceptance criterion, and the difference
     between policy and prompt wording is entirely a question of WHERE the check
     sits in the turn. A guard placed after intent resolution satisfies every
     unit test in the other file and satisfies none of the criterion.

  2. That the offer it makes can actually be ACCEPTED. The reply ends in a
     question -- "shall I connect you?" -- which means this feature spans two
     turns and a piece of session state. A guard that asks a question nothing
     listens for leaves a frightened caller's "yes" falling through to the
     intent classifier, where it becomes `unclear` and they are asked to repeat
     themselves. That is a worse rebuff than the one the story was written to
     prevent, and it is invisible to single-turn testing.

So this drives the REAL `_dispatch_turn` from main_pcm.py across consecutive
turns, with the pod-only libraries, the model, ASR and the clinic tools faked.

    python -m pytest tests/test_orchestrator_clinical_safety.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _pod_stubs import pod_stubs  # noqa: E402
from _synth_voices import FATHER, utterance  # noqa: E402
from test_orchestrator_persona import FakeTTS, FakeWS, _wav  # noqa: E402

from agent.phrases import phrase  # noqa: E402


class ASRResult:
    def __init__(self, text, agreement=0.9):
        self.text, self.decoder_agreement = text, agreement


class FakeTools:
    def __init__(self):
        self.complaints = []

    async def submit_complaint(self, call_id, text, detected_by="", lang="", phone=None):
        self.complaints.append({"call_id": call_id, "text": text, "lang": lang})
        return {"success": True, "id": len(self.complaints)}

    async def request_callback(self, *a, **k):
        return {"success": True, "id": 1}


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
    state = {
        "text": "what is the price of a CBC test?",
        "lang": "en",
        "intent": "test_rate",
        "reply": "The CBC costs three hundred and fifty rupees.",
        "handoffs": [],
    }

    async def route(session, wav):
        return state["lang"], ASRResult(state["text"])

    async def resolve(session, text, lang):
        return {
            "intent": state["intent"],
            "slots": {"test_name": "CBC"},
            "secondary_intent": None,
            "direct_reply_bn": None,
        }

    async def answer(intent, slots, lang):
        return state["reply"]

    async def handoff(session, reason, languages=None):
        state["handoffs"].append(reason)

    monkeypatch.setattr(m, "_route_and_transcribe", route)
    monkeypatch.setattr(m, "_resolve_intent", resolve)
    monkeypatch.setattr(m, "_answer_enquiry_intent", answer)
    monkeypatch.setattr(m, "_handoff_to_human", handoff)
    session = m.CallSession(FakeWS())
    session.release_gate()
    session.disclosed_langs.update({"en", "bn", "hi"})
    session.call_state = m.new_call_state()
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)

    class Driver:
        pass

    d = Driver()
    d.session, d.state, d.tools, d.voice = session, state, tools, voice

    async def turn(samples=None, **overrides):
        state.update(overrides)
        session.turn_epoch = session.speak_epoch
        path = _wav(tmp_path, voice if samples is None else samples, f"u{len(session.ws.texts)}.wav")
        await m._dispatch_turn(session, path)

    d.turn = turn
    yield d


QUESTIONS = [
    ("en", "is my result dangerous"),
    ("bn", "আমার রিপোর্ট কি অস্বাভাবিক"),
    ("hi", "क्या मेरी रिपोर्ट खतरनाक है"),
]


# ---------------------------------------------------- the guard is reached


@pytest.mark.parametrize("lang,text", QUESTIONS)
async def test_the_caller_is_offered_a_clinician_in_their_own_language(env, lang, text):
    await env.turn(text=text, lang=lang)
    said = " ".join(env.session.ws.spoken())
    assert phrase("clinical_interpretation", lang) in said


@pytest.mark.parametrize("lang,text", QUESTIONS)
async def test_the_question_never_reaches_the_model(env, lang, text):
    """"By policy rather than by prompt wording", tested where the difference
    actually lives. The real answer the faked model would have produced must
    not appear: if it does, the guard ran too late."""
    await env.turn(text=text, lang=lang)
    said = " ".join(env.session.ws.spoken())
    assert "three hundred and fifty" not in said
    assert env.state["handoffs"] == []  # nothing is handed off until the caller says yes


async def test_the_clinical_question_is_counted(env):
    await env.turn(text="am I dying", lang="en")
    assert env.session.signals.clinical_questions == 1


# ------------------------------------------- the offer can be accepted


@pytest.mark.parametrize("lang,text,yes", [("en", "is my result dangerous", "yes please"),
                                           ("bn", "আমার রিপোর্ট কি অস্বাভাবিক", "হ্যাঁ"),
                                           ("hi", "क्या मेरी रिपोर्ट खतरनाक है", "हाँ")])
async def test_a_yes_on_the_next_turn_reaches_a_clinician(env, lang, text, yes):
    """The half no single-turn test can see. The reason code is this story's
    own, not the history flow's -- a clinical question filed as a history
    request would be routed to the wrong desk."""
    await env.turn(text=text, lang=lang)
    assert env.session.awaiting_clinician_offer is True
    await env.turn(text=yes, lang=lang)
    assert env.state["handoffs"] == ["clinical_interpretation"]


async def test_a_no_on_the_next_turn_does_not_hand_off_and_the_call_continues(env):
    await env.turn(text="is my result dangerous", lang="en")
    await env.turn(text="no", lang="en")
    assert env.state["handoffs"] == []
    assert env.session.awaiting_clinician_offer is False
    assert phrase("silence_go_on", "en") in " ".join(env.session.ws.spoken())


async def test_an_answer_that_is_neither_yes_nor_no_lets_the_offer_lapse(env):
    """The courtesy _continue_history_flow() already shows its own offer: a
    caller who ignores the question and asks something else gets that something
    answered, not a repeated question."""
    await env.turn(text="is my result dangerous", lang="en")
    await env.turn(text="what is the price of a CBC test?", lang="en")
    assert env.session.awaiting_clinician_offer is False
    assert env.state["handoffs"] == []
    assert "three hundred and fifty" in " ".join(env.session.ws.spoken())


async def test_the_offer_is_not_still_pending_after_it_is_answered(env):
    """State that is set but never cleared turns the next unrelated "yes" in the
    call into a hand-off."""
    await env.turn(text="is my result dangerous", lang="en")
    await env.turn(text="yes", lang="en")
    assert env.session.awaiting_clinician_offer is False
    env.state["handoffs"].clear()
    await env.turn(text="yes", lang="en")
    assert env.state["handoffs"] == [], "a later bare yes was treated as accepting an offer again"


# -------------------------------------------------------- guard ordering


async def test_a_clinical_question_wrapped_in_a_complaint_takes_the_clinical_path(env):
    """Both guards match this turn. The clinical guard is checked first, which is
    the order the module this was ported from tested: the caller's health
    question is the one that must not go unanswered, and the clinical path also
    reaches a person."""
    await env.turn(text="I want to file a complaint, is my result dangerous", lang="en")
    said = " ".join(env.session.ws.spoken())
    assert phrase("clinical_interpretation", "en") in said
    assert phrase("complaint_acknowledged", "en") not in said


async def test_an_ordinary_question_is_completely_unaffected_by_this_port(env):
    await env.turn(text="what is the price of a CBC test?", lang="en")
    assert env.session.awaiting_clinician_offer is False
    assert env.session.signals.clinical_questions == 0
    said = " ".join(env.session.ws.spoken())
    assert "three hundred and fifty" in said
    assert phrase("clinical_interpretation", "en") not in said
