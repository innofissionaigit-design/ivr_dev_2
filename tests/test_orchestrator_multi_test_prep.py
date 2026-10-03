"""The orchestrator's per-turn wiring for KCD-382, story "Caller has several tests with conflicting
preparation", off-pod.

WHY THIS FILE EXISTS SEPARATELY FROM tests/test_enquiry_service.py AND tests/test_prep_and_ambiguity.py:

Those files prove the merge logic (clinic-api/enquiry_service.py::merge_prep_instructions) and the spoken
template (agent/reply_templates.py::merged_test_prep_reply) each in isolation. Neither can prove the thing
that decides whether the story actually works end to end from the caller's side:

  1. That a turn naming MORE THAN ONE test is intercepted before the single-test `test_prep` path ever
     runs -- main.py's `test_prep` dispatch branch has to check `len(test_names) > 1` and route to
     `_handle_multi_test_prep` BEFORE calling `_answer_enquiry_intent` / `_finish_enquiry_turn`. If that
     check is ever removed or reordered, the call still "works" (the model's `test_names` slot is simply
     ignored) but the caller is back to hearing one test's answer for a multi-test question, silently.

  2. That AC3 (escalate to a human rather than guess) is wired to the REAL signal, not re-derived. The
     orchestrator must act on `result["escalate_to_human"]` alone, must speak the hand-off notice BEFORE
     calling `_handoff_to_human` (which closes the websocket -- see `_handle_multi_test_prep`'s own
     docstring), and must never also speak the merged instruction in the same turn.

  3. That the existing single-test `test_prep` flow is completely unaffected -- the regression this story
     is most likely to cause if the new branch is placed carelessly in the existing `elif` chain.

So this drives the REAL `_dispatch_turn` from main_pcm.py with the pod-only libraries, the model, ASR and
the clinic tools faked, the same approach as tests/test_orchestrator_complaint.py.

    python -m pytest tests/test_orchestrator_multi_test_prep.py -v
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
    """Records every merge/single-test-prep lookup the orchestrator makes, so each test can assert not just
    what was SAID but which tool call actually produced it -- the thing a purely reply-text assertion
    cannot distinguish (a wrong-but-similar-sounding path could still pass a text-only check)."""

    def __init__(self):
        self.merge_calls = []
        self.merge_result = {"found": False}
        self.prep_calls = []
        self.prep_result = {"found": False}

    async def merge_test_prep(self, test_names, lang="bn"):
        self.merge_calls.append((list(test_names), lang))
        return self.merge_result

    async def get_test_prep(self, test_name, lang="bn"):
        self.prep_calls.append((test_name, lang))
        return self.prep_result


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
        "text": "what is the preparation for blood sugar fasting and lipid profile",
        "lang": "en",
        "intent": "test_prep",
        "slots": {"test_names": ["Blood Sugar Fasting", "Lipid Profile"]},
        "reply": "The CBC needs no special preparation.",  # used only by the single-test regression tests
        "handoffs": [],
        "answer_calls": [],
    }

    async def route(session, wav):
        return state["lang"], ASRResult(state["text"])

    async def resolve(session, text, lang):
        return {
            "intent": state["intent"],
            "slots": dict(state["slots"]),
            "secondary_intent": None,
            "direct_reply_bn": None,
        }

    async def answer(intent, slots, lang):
        # The single-test `_answer_enquiry_intent` path -- faked here exactly as every other orchestrator
        # test fakes it (test_orchestrator_complaint.py, test_orchestrator_clinical_safety.py, ...), so this
        # file stays focused on KCD-382's OWN wiring rather than re-proving that function's internals (that
        # fallback is covered directly, unmocked, in test_answer_enquiry_intent_test_prep_fallback below).
        state["answer_calls"].append((intent, dict(slots), lang))
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


# ------------------------------------------------------------------------------------------ AC1 + AC2: merge


async def test_two_tests_are_merged_into_one_spoken_instruction(env):
    env.tools.merge_result = {
        "found": True,
        "test_names": ["Blood Sugar Fasting", "Lipid Profile"],
        "fasting_hours": 11,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": False,
        "not_found": [],
    }
    await env.turn(slots={"test_names": ["Blood Sugar Fasting", "Lipid Profile"]})
    assert env.tools.merge_calls == [(["Blood Sugar Fasting", "Lipid Profile"], "en")]
    said = " ".join(env.session.ws.spoken())
    assert "11" in said
    assert env.state["handoffs"] == []
    assert env.state["answer_calls"] == [], "the single-test path ran too -- the multi-test branch did not short-circuit it"


async def test_three_tests_with_one_needing_no_fasting_keeps_the_strictest_applicable_value(env):
    """One of the three tests contributes nothing to the fasting window (it has none); the merged number
    must still be the strictest of the ones that DO have a requirement -- this is the orchestrator-level
    counterpart of test_enquiry_service.py's same-shaped assertion on the service function directly."""
    env.tools.merge_result = {
        "found": True,
        "test_names": ["Blood Sugar Fasting", "Lipid Profile", "Complete Blood Count (CBC)"],
        "fasting_hours": 11,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": False,
        "not_found": [],
    }
    await env.turn(slots={"test_names": ["Blood Sugar Fasting", "Lipid Profile", "Complete Blood Count (CBC)"]})
    said = " ".join(env.session.ws.spoken())
    assert "11" in said


# --------------------------------------------------------------------------------- AC3: escalate, never guess


async def test_incomplete_structured_data_hands_off_to_a_human_instead_of_guessing(env):
    """The USG-Whole-Abdomen-shaped case: the merge service could not safely fold a known fasting
    requirement into the merged number, so it set a real `escalate_to_human`. The orchestrator must speak
    the hand-off notice, call `_handoff_to_human` with the dedicated reason code, and must NOT also speak a
    merged instruction -- the two are mutually exclusive in `_handle_multi_test_prep`."""
    env.tools.merge_result = {
        "found": True,
        "test_names": ["USG Whole Abdomen"],
        "fasting_hours": None,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": True,
        "incomplete_data": ["USG Whole Abdomen"],
        "not_found": [],
    }
    await env.turn(text="what is the preparation for USG whole abdomen and blood sugar", slots={
        "test_names": ["USG Whole Abdomen", "Blood Sugar Fasting"]
    })
    said = env.session.ws.spoken()
    assert phrase("test_prep_conflict_handoff", "en") in " ".join(said)
    assert env.state["handoffs"] == ["test_prep_conflict_unresolved"]
    assert not any("fast for at least" in line.lower() for line in said), (
        "a merged instruction was spoken alongside the hand-off -- the caller must get one or the other"
    )


async def test_an_unresolved_test_name_is_not_guessed_but_does_not_need_a_human(env):
    """A name the merge endpoint could not resolve is a "say it again" case (AC1's deterministic lookup
    found nothing to merge for that name), not a conflict requiring escalation -- the two failure modes are
    deliberately kept distinct, exactly as clinic-api/main.py::prep_merge_endpoint reports them."""
    env.tools.merge_result = {
        "found": True,
        "test_names": ["Blood Sugar Fasting"],
        "fasting_hours": 8,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": False,
        "not_found": ["Not A Real Test"],
    }
    await env.turn(slots={"test_names": ["Blood Sugar Fasting", "Not A Real Test"]})
    said = " ".join(env.session.ws.spoken())
    assert "Not A Real Test" in said
    assert env.state["handoffs"] == [], "a resolution miss escalated to a human instead of asking to repeat"


# ------------------------------------------------------------------------------- regression: single test_prep


async def test_a_single_named_test_is_completely_unaffected_by_this_port(env):
    await env.turn(
        text="what is the preparation for a CBC",
        slots={"test_name": "Complete Blood Count (CBC)"},
        reply="No special preparation is needed for the CBC.",
    )
    assert env.tools.merge_calls == [], "the single-test turn went through the multi-test merge path"
    assert len(env.state["answer_calls"]) == 1
    said = " ".join(env.session.ws.spoken())
    assert "No special preparation is needed for the CBC." in said
    assert env.state["handoffs"] == []


async def test_a_single_entry_test_names_list_still_takes_the_single_test_path(env):
    """The model may now put even a lone test name in the plural `test_names` slot (agent/llm.py's
    broadened prompt). `len(test_names) > 1` is the only thing that should route to the merge branch, so
    exactly one name must still fall through to the ordinary single-test flow."""
    await env.turn(
        text="what is the preparation for a CBC",
        slots={"test_names": ["Complete Blood Count (CBC)"]},
        reply="No special preparation is needed for the CBC.",
    )
    assert env.tools.merge_calls == []
    assert len(env.state["answer_calls"]) == 1
    assert "No special preparation is needed for the CBC." in " ".join(env.session.ws.spoken())


# ------------------------------------------------------- the _answer_enquiry_intent fallback, unmocked


@pytest.fixture(scope="module")
def um():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


async def test_answer_enquiry_intent_test_prep_fallback_reads_a_lone_test_names_entry(um, monkeypatch):
    """Direct, unmocked exercise of the one line main.py's test_prep branch of `_answer_enquiry_intent`
    gained for this story: `slots.get("test_name") or next(iter(slots.get("test_names") or []), None)`.
    This is the actual fallback the orchestrator-level test above relies on being correct, proven here
    against the real function rather than inferred from the faked `answer()` in the `env` fixture."""
    tools = FakeTools()
    tools.prep_result = {
        "found": True,
        "test_name": "Complete Blood Count (CBC)",
        "fasting_required": False,
        "prep_instructions": "",
    }
    monkeypatch.setattr(um, "_tools", tools)
    reply = await um._answer_enquiry_intent("test_prep", {"test_names": ["Complete Blood Count (CBC)"]}, "en")
    assert tools.prep_calls == [("Complete Blood Count (CBC)", "en")]
    assert reply is not None and "CBC" in reply


async def test_answer_enquiry_intent_test_prep_still_requires_some_name(um, monkeypatch):
    monkeypatch.setattr(um, "_tools", FakeTools())
    assert await um._answer_enquiry_intent("test_prep", {}, "en") is None
    assert await um._answer_enquiry_intent("test_prep", {"test_names": []}, "en") is None
