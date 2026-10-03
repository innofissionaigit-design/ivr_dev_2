"""The orchestrator's per-turn wiring for a complaint (ported story: "Caller wants
to make a complaint"), off-pod.

WHY THIS FILE EXISTS SEPARATELY FROM tests/test_complaint.py:

tests/test_complaint.py proves the detector, the verbatim store and the wording.
None of it can prove the two things that decide whether this port works at all:

  1. That the guard is REACHED. The complaint path has to run before intent
     extraction, because this repository's intent set has no "complaint" member
     -- if the guard is placed after the model, or skipped on some turn shape,
     the complaint is answered as a cheerful enquiry and nothing anywhere
     reports a failure.

  2. That it is reached FIRST. "I want to speak to someone about a complaint"
     matches three guards at once: this one, agent/human_request.py's
     asks_for_a_person(), and agent/anger.py's check inside
     _update_caller_state(). Only this one keeps the caller's words. If the
     ordering is ever rearranged, the caller is still routed to a person -- the
     call still "works" -- and the complaint is silently never recorded. That is
     the single most plausible way this feature breaks in future, and no unit
     test can see it.

So this drives the REAL `_dispatch_turn` from main_pcm.py with the pod-only
libraries, the model, ASR and the clinic tools faked, the same approach as
tests/test_orchestrator_persona.py.

    python -m pytest tests/test_orchestrator_complaint.py -v
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
from agent.phrases import phrase  # noqa: E402
from agent.tools_client import ToolCallError  # noqa: E402


class ASRResult:
    def __init__(self, text, agreement=0.9):
        self.text, self.decoder_agreement = text, agreement


class FakeTools:
    """Records what the orchestrator tried to persist. `fail` makes the durable
    write raise the same error the real client raises, so the resilience test
    exercises the real suppression path rather than a mock's convenience."""

    def __init__(self):
        self.complaints, self.callbacks, self.fail = [], [], False

    async def submit_complaint(self, call_id, text, detected_by="", lang="", phone=None):
        if self.fail:
            raise ToolCallError("submit_complaint: clinic-api unreachable")
        self.complaints.append(
            {"call_id": call_id, "text": text, "detected_by": detected_by, "lang": lang, "phone": phone}
        )
        return {"success": True, "id": len(self.complaints)}

    async def request_callback(self, *a, **k):
        self.callbacks.append((a, k))
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


COMPLAINTS = [
    ("en", "I want to file a complaint about yesterday"),
    ("bn", "আমি কমপ্লেইন করতে চাই"),
    ("hi", "मुझे शिकायत करनी है"),
]


# ------------------------------------------------- the guard is reached


@pytest.mark.parametrize("lang,text", COMPLAINTS)
async def test_the_caller_hears_the_complaint_acknowledgement_in_their_own_language(env, lang, text):
    await env.turn(text=text, lang=lang)
    said = " ".join(env.session.ws.spoken())
    assert phrase("complaint_acknowledged", lang) in said


@pytest.mark.parametrize("lang,text", COMPLAINTS)
async def test_the_complaint_is_routed_to_a_person_with_its_own_reason_code(env, lang, text):
    """Acceptance criterion 4. The reason code is "complaint", not the generic
    handoff code, so a human picking the queue up can tell the two apart."""
    await env.turn(text=text, lang=lang)
    assert env.state["handoffs"] == ["complaint"]


@pytest.mark.parametrize("lang,text", COMPLAINTS)
async def test_the_caller_s_own_words_are_sent_verbatim_to_the_durable_store(env, lang, text):
    """Acceptance criterion 3, end to end: not what the service function does
    with the text (tests/test_complaint.py owns that) but what the orchestrator
    actually hands it -- the transcript, unedited, with no summary step."""
    await env.turn(text=text, lang=lang)
    assert len(env.tools.complaints) == 1
    rec = env.tools.complaints[0]
    assert rec["text"] == text, "the orchestrator altered the complaint before storing it"
    assert rec["lang"] == lang
    assert rec["detected_by"] in ("en", "latin", "bn", "hi")
    assert rec["call_id"] == env.session.call_id


async def test_the_complaint_is_counted_for_the_call_score(env):
    await env.turn(text="I want to file a complaint", lang="en")
    assert env.session.signals.complaint_filed is True


# -------------------------------------------- the ordering that cannot regress


async def test_a_complaint_that_also_asks_for_a_person_takes_the_complaint_path(env):
    """The ordering test that matters most. This utterance matches both this
    guard and asks_for_a_person(). If the two are ever swapped, the caller is
    still handed to a person and the call still looks fine -- but the words are
    never recorded. Asserted on the stored complaint, not on the handoff."""
    await env.turn(text="I want to speak to someone about a complaint", lang="en")
    assert len(env.tools.complaints) == 1, "the complaint was not recorded: asks_for_a_person won the race"
    assert env.state["handoffs"] == ["complaint"], "routed as a bare person-request, not as a complaint"


async def test_an_angry_complaint_is_one_acknowledgement_and_one_route(env):
    """"Anger enhances the complaint flow, it does not compete with it." Both
    guards match this turn; the complaint path returns first, so the caller must
    NOT also get the anger acknowledgement, and must not be acknowledged twice."""
    await env.turn(text="this is ridiculous, I want to file a complaint", lang="en")
    said = env.session.ws.spoken()
    assert sum(phrase("complaint_acknowledged", "en") in line for line in said) == 1
    assert all(acknowledgement_for("angry", "en") not in line for line in said), (
        "the caller was acknowledged twice: the anger guard also ran for this turn"
    )
    assert env.state["handoffs"] == ["complaint"]
    assert len(env.tools.complaints) == 1


async def test_swearing_is_still_abuse_and_does_not_file_a_complaint(env):
    """abuse is checked earlier still, and keeps its own calm boundary. Swearing
    is not a complaint and must not open a complaint row."""
    await env.turn(text="you are a complete bastard", lang="en")
    assert env.session.abuse_count == 1
    assert env.tools.complaints == []
    assert env.state["handoffs"] == []


# --------------------------------------------------------------- resilience


async def test_a_storage_failure_still_acknowledges_and_still_routes(env):
    """A clinic-api outage must not cost the caller their acknowledgement or
    their hand-off -- the complaint is the one thing that must not be dropped on
    the floor because a write failed. The tool raises the real ToolCallError
    here, so this exercises the real suppression, not a mock's politeness."""
    env.tools.fail = True
    await env.turn(text="I want to file a complaint", lang="en")
    said = " ".join(env.session.ws.spoken())
    assert phrase("complaint_acknowledged", "en") in said
    assert env.state["handoffs"] == ["complaint"]


async def test_an_ordinary_question_is_completely_unaffected_by_this_port(env):
    """The regression guard: every caller who is not complaining gets exactly
    the call they got before the port."""
    await env.turn(text="what is the price of a CBC test?", lang="en")
    assert env.tools.complaints == []
    assert env.state["handoffs"] == []
    assert env.session.signals.complaint_filed is False
    said = " ".join(env.session.ws.spoken())
    assert "three hundred and fifty" in said
    assert phrase("complaint_acknowledged", "en") not in said
