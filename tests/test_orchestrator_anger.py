"""The orchestrator's per-turn wiring for caller anger (ported story: "Caller is
angry about a previous experience"), off-pod.

WHY THIS FILE EXISTS SEPARATELY FROM tests/test_anger.py:

tests/test_anger.py proves the detector is right and that the "angry" policy row
and acknowledgement it unlocks say what the story demanded. It cannot prove the
one thing that actually mattered about this port: that the detector is REACHED.

That risk is specific and real here. Before the port, `caller_state="angry"`
existed in three definition sites (agent/call_state.py, agent/speech_policy.py,
agent/acknowledgement.py) and was set by nothing at all -- a complete, trilingual
feature that no caller could ever trigger. A port that added a perfect detector
and wired it to nothing would reproduce exactly that bug, and every unit test in
tests/test_anger.py would still pass.

So this file drives the REAL `_dispatch_turn` from main_pcm.py, with only the
pod-only libraries, the language model, ASR and the clinic tools faked -- the
same approach as tests/test_orchestrator_persona.py, whose harness it reuses.
What is under test is the wiring and the ORDER of the checks: that an angry
transcript reaches the caller-state step, that the caller then actually hears the
acknowledgement in their own language, that a person is offered, and that the
guards which already ran before this point (emergency, abuse) still win where
they are supposed to.

    python -m pytest tests/test_orchestrator_anger.py -v
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

from agent.acknowledgement import acknowledgement_for  # noqa: E402


class ASRResult:
    def __init__(self, text, agreement=0.9):
        self.text, self.decoder_agreement = text, agreement


@pytest.fixture(scope="module")
def m():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


@pytest.fixture
def env(m, monkeypatch, tmp_path):
    """A session plus fakes for everything that needs a pod. Mirrors
    tests/test_orchestrator_persona.py's own `env`, so any drift in the real
    dispatch path shows up in both files rather than being hidden here."""
    monkeypatch.setattr(m, "_admission", None)
    monkeypatch.setattr(m, "_tts_router", FakeTTS())
    monkeypatch.setattr(m, "CONDITION_INPUT", "off")
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
            "direct_reply_bn": state.get("smalltalk"),
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
    session.disclosed_langs.update({"en", "bn", "hi"})  # the disclosure is not what these tests are about
    session.call_state = m.new_call_state()
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)

    class Driver:
        pass

    d = Driver()
    d.session, d.state, d.voice = session, state, voice

    async def turn(samples=None, **overrides):
        state.update(overrides)
        session.turn_epoch = session.speak_epoch
        path = _wav(tmp_path, voice if samples is None else samples, f"u{len(session.ws.texts)}.wav")
        await m._dispatch_turn(session, path)

    d.turn = turn
    yield d


# ----------------------------------------------------- the state is reached

# One angry utterance per language this system serves, each in that language's
# own turn. The assertion is deliberately about the SESSION, not about the
# detector: tests/test_anger.py already owns the detector.
@pytest.mark.parametrize(
    "lang,text",
    [
        ("en", "this is absolutely ridiculous, I am fed up"),
        ("bn", "আমি খুব রেগে গেছি, আপনারা কোনো কাজের না"),
        ("hi", "मुझे बहुत गुस्सा आ रहा है, यह बिलकुल बकवास है"),
    ],
)
async def test_an_angry_turn_puts_the_call_into_the_angry_state(env, lang, text):
    await env.turn(text=text, lang=lang)
    assert env.session.call_state.caller_state == "angry"


@pytest.mark.parametrize(
    "lang,text",
    [
        ("en", "this is absolutely ridiculous, I am fed up"),
        ("bn", "আমি খুব রেগে গেছি, আপনারা কোনো কাজের না"),
        ("hi", "मुझे बहुत गुस्सा आ रहा है, यह बिलकुल बकवास है"),
    ],
)
async def test_the_angry_caller_hears_the_acknowledgement_in_their_own_language(env, lang, text):
    """The point of the whole port: a real spoken line, in the caller's language,
    reaching a real caller. `spoken()` is what the client actually receives."""
    await env.turn(text=text, lang=lang)
    said = " ".join(env.session.ws.spoken())
    assert acknowledgement_for("angry", lang) in said, f"the {lang} acknowledgement was not spoken"


async def test_a_person_is_offered_explicitly_rather_than_the_call_just_continuing(env):
    """Acceptance criterion 3. The offer is part of the selected line because the
    angry policy sets offer_human; this proves it survives the trip through
    _dispatch_turn and is not dropped on the way to the caller."""
    await env.turn(text="I am furious about this", lang="en")
    said = " ".join(env.session.ws.spoken())
    bare_ack = acknowledgement_for("angry", "en")
    offer = acknowledgement_for("angry", "en", offer_human=True)
    assert offer != bare_ack, "fixture assumption: offer_human should extend the line"
    assert offer in said


async def test_the_angry_state_is_sticky_and_the_acknowledgement_is_not_repeated(env):
    """Two things at once, because they pull in opposite directions: the state has
    to persist (so the agent does not speed back up on the next calm turn) while
    the apology must NOT be said again (acceptance criterion 4 -- apologise once).
    """
    await env.turn(text="I have had enough of this", lang="en")
    assert env.session.call_state.caller_state == "angry"
    first = env.session.ws.spoken()
    ack = acknowledgement_for("angry", "en")
    assert sum(ack in line for line in first) == 1

    await env.turn(text="what is the price of a CBC test?", lang="en")
    assert env.session.call_state.caller_state == "angry", "the state was silently cleared by a calm turn"
    later = env.session.ws.spoken()[len(first) :]
    assert all(ack not in line for line in later), "the acknowledgement was repeated on a later turn"


async def test_the_anger_is_counted_for_the_call_score(env):
    """Acceptance criterion 7, the half a human picking up the call reads."""
    await env.turn(text="this is so frustrating", lang="en")
    assert env.session.signals.anger_turns == 1
    assert env.session.call_state.to_log_dict()["caller_state"] == "angry"


# -------------------------------------------------- the guards that outrank it


async def test_swearing_is_still_handled_as_abuse_and_not_as_anger(env):
    """main_pcm.py checks abuse BEFORE the caller-state step, and must keep doing
    so: the two have different responses (a calm fixed boundary vs. a slowed turn
    with a person offered) and conflating them would answer the wrong one. The
    abuse turn returns early, so the caller state stays neutral."""
    await env.turn(text="you are a complete bastard", lang="en")
    assert env.session.abuse_count == 1
    assert env.session.call_state.caller_state != "angry"


async def test_an_ordinary_question_is_completely_unaffected_by_this_port(env):
    """The regression guard that matters most: every caller who is not angry must
    get exactly the call they got before the port -- no acknowledgement, no state
    change, and the real answer still spoken."""
    await env.turn(text="what is the price of a CBC test?", lang="en")
    assert env.session.call_state.caller_state == "neutral"
    assert env.session.signals.anger_turns == 0
    said = " ".join(env.session.ws.spoken())
    assert "three hundred and fifty" in said
    assert acknowledgement_for("angry", "en") not in said
