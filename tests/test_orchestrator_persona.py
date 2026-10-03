"""The orchestrator's per-turn wiring for persona, acknowledgement, apology, disclosure,
human request, near-end and speaker change (KCD-353/513/514/053/054), off-pod.

Same approach as tests/test_orchestrator_duplex.py: the REAL `_dispatch_turn`, `_speak`
and helpers from main_pcm.py, with the pod-only libraries stubbed and the language
model, ASR and clinic tools replaced by fakes. What is real is the wiring and the
order of the checks.

    python -m pytest tests/test_orchestrator_persona.py -v
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
from _synth_voices import DAUGHTER, FATHER, MOTHER, scaled, utterance

from agent.outcome_metrics import apology_events, audio_issue_buckets, golden_buckets
from agent.phrases import phrase

SR = 16000


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
        """The text of every 'AI' message, in order."""
        out = []
        for t in self.texts:
            try:
                d = json.loads(t)
            except ValueError:
                continue
            if d.get("sender") == "AI":
                out.append(d["text"])
        return out

    def frames(self, kind):
        return [d for t in self.texts for d in [json.loads(t)] if d.get("type") == kind]


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


def _wav(tmp_path, samples, name="utt.wav"):
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
    """A session plus fakes for everything that needs a pod; returns a small driver."""
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
    session.disclosed_langs.add("en")  # most tests are not about the disclosure
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
    session.cleanup()


# ================================================================== KCD-513 acknowledgement


@pytest.mark.asyncio
async def test_an_answer_to_the_callers_own_question_has_no_thanks_in_front_of_it(m, env):
    # DELIBERATE SPEC CHANGE (owner, 2026-09-25, third live call): "Thank you for telling me" in front of a price was
    # heard as nonsense. An answer starts with the answer; the bare "Thank you." is spoken only after the caller has
    # answered a question the agent asked (tests/test_filler_thanks_silence.py).
    await env.turn()
    assert env.session.ws.spoken() == [env.state["reply"]]


@pytest.mark.asyncio
async def test_no_reply_to_a_question_of_the_callers_is_thanked_however_many_they_ask(m, env):
    # DELIBERATE SPEC CHANGE (owner, 2026-09-25): was "the thanks is spoken on every substantive reply".
    await env.turn()
    await env.turn()
    assert env.session.ws.spoken() == [env.state["reply"], env.state["reply"]]


def test_the_varied_mode_still_suppresses_repeats():
    # DELIBERATE, MARKED CHANGE: the orchestrator no longer decorates replies with AckTracker, so the varied mode of
    # the class itself is what is checked (it was checked through the orchestrator before).
    from agent.turn_ack import AckTracker

    t = AckTracker(mode="varied")
    reply = "The CBC costs three hundred and fifty rupees."
    first, _ = t.decorate(reply, "en")
    t.next_turn()
    second, _ = t.decorate(reply, "en")
    assert first != reply and second == reply


# ====================================================================== KCD-514 apology


@pytest.mark.asyncio
async def test_a_reply_that_stacks_apologies_is_spoken_with_one(m, env):
    before = apology_events.snapshot().get("stacked_removed", {}).get("reply:en", 0)
    env.state["reply"] = "Sorry, I could not find that doctor. Sorry, please give me the name again."
    await env.turn()
    said = env.session.ws.spoken()[0]
    from agent.apology import count_apologies

    assert count_apologies(said, "en") == 1
    assert "give me the name again" in said
    assert apology_events.snapshot()["stacked_removed"]["reply:en"] == before + 1


# ================================================================== KCD-353 human request


@pytest.mark.asyncio
async def test_a_request_for_a_person_hands_over_at_once_and_consults_no_model(m, env, monkeypatch):
    called = []

    async def must_not_run(*a, **k):
        called.append(1)
        return {"intent": "unclear", "slots": {}}

    monkeypatch.setattr(m, "_resolve_intent", must_not_run)
    await env.turn(text="I want to talk to a person")
    assert env.state["handoffs"] == ["caller_requested"]
    assert called == []  # no intent extraction happened
    assert env.session.ws.spoken() == []  # and nothing else was said first


@pytest.mark.asyncio
async def test_an_ordinary_question_does_not_trigger_a_handoff(m, env):
    await env.turn()
    assert env.state["handoffs"] == []


# ================================================================== KCD-353 disclosure


@pytest.mark.asyncio
async def test_the_disclosure_is_repeated_once_in_a_new_language_and_never_again(m, env):
    from agent.disclosure import disclosure_for

    env.session.disclosed_langs.discard("en")
    await env.turn(lang="en")
    await env.turn(lang="en")
    said = env.session.ws.spoken()
    assert said.count(disclosure_for("en")) == 1
    assert said.index(disclosure_for("en")) == 0  # before the first answer


@pytest.mark.asyncio
async def test_a_bengali_call_hears_it_only_in_the_greeting(m, env):
    from agent.disclosure import disclosure_for

    await env.turn(lang="bn", text="সিবিসি টেস্টের দাম কত", reply="সিবিসির দাম সাড়ে তিনশো টাকা।")
    assert disclosure_for("bn") not in env.session.ws.spoken()  # already in the greeting


# ============================================================= persona guard on small talk


@pytest.mark.asyncio
async def test_a_model_written_smalltalk_reply_that_breaks_the_persona_is_replaced(m, env):
    await env.turn(intent="smalltalk", smalltalk="Don't worry, it is probably nothing serious.", text="how are you")
    assert env.session.ws.spoken() == [phrase("smalltalk_default", "en")]


@pytest.mark.asyncio
async def test_a_clean_smalltalk_reply_is_spoken(m, env):
    await env.turn(intent="smalltalk", smalltalk="I am well, thank you. How can I help you?", text="how are you")
    assert env.session.ws.spoken() == ["I am well, thank you. How can I help you?"]


# ================================================================= KCD-053 near-end attention


@pytest.mark.asyncio
async def test_a_quieter_different_voice_never_opens_a_turn(m, env):
    await env.turn()  # the caller: establishes the profile
    spoken_before = len(env.session.ws.spoken())
    bystander = scaled(utterance(DAUGHTER, dur=3.0, seed=9, amp=0.3), -22.0)
    dropped_before = audio_issue_buckets.snapshot().get("background_utterance", {}).get("dropped:en", 0)
    await env.turn(samples=bystander)
    assert len(env.session.ws.spoken()) == spoken_before  # silence for the bystander
    assert env.session.near_end.rejected == 1
    assert audio_issue_buckets.snapshot()["background_utterance"]["dropped:en"] == dropped_before + 1


@pytest.mark.asyncio
async def test_the_same_voice_quieter_is_still_answered(m, env):
    await env.turn()
    spoken_before = len(env.session.ws.spoken())
    await env.turn(samples=scaled(utterance(FATHER, dur=3.0, seed=4, amp=0.3), -22.0))
    assert len(env.session.ws.spoken()) > spoken_before
    assert env.session.near_end.rejected == 0


# ============================================================ Appendix F buckets


@pytest.mark.asyncio
async def test_every_analysed_turn_is_counted_and_channel_buckets_are_recorded(m, env):
    before = golden_buckets.snapshot().get("turn", {}).get("analysed:en", 0)
    await env.turn()
    snap = golden_buckets.snapshot()
    assert snap["turn"]["analysed:en"] == before + 1


# ============================================================== KCD-054 speaker change


@pytest.mark.asyncio
async def test_a_different_voice_after_verification_revokes_it_and_says_so(m, env):
    env.session.speaker.enroll(utterance(FATHER, dur=3.0, seed=1, amp=0.3), SR)
    env.session.identity.verify("otp", "patient-1")
    await env.turn()  # same voice: nothing changes
    assert env.session.identity.is_verified
    await env.turn(samples=utterance(MOTHER, dur=3.0, seed=7, amp=0.3))  # a different person, as loud
    assert not env.session.identity.is_verified
    assert env.session.identity.revocations == ["speaker_changed"]
    assert phrase("reverify_notice", "en") in env.session.ws.spoken()


@pytest.mark.asyncio
async def test_no_verification_means_nothing_to_revoke_and_no_notice(m, env):
    env.session.speaker.enroll(utterance(FATHER, dur=3.0, seed=1, amp=0.3), SR)  # enrolled, but never verified
    await env.turn(samples=utterance(MOTHER, dur=3.0, seed=7, amp=0.3))
    assert phrase("reverify_notice", "en") not in env.session.ws.spoken()


@pytest.mark.asyncio
async def test_mark_verified_enrolls_the_voice_that_verified(m, env):
    analysis = m._analyze_utterance_from_wav_path(_tmp_wav(env.voice))
    m._mark_verified(env.session, "otp", "patient-1", analysis)
    assert env.session.identity.is_verified and env.session.speaker.enrolled


def _tmp_wav(samples):
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return path
