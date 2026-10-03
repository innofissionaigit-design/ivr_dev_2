"""The orchestrator's turn-taking wiring, off-pod (KCD-049/048/050/051/052/163).

main.py and main_pcm.py could not be imported without a pod (torch, NeMo,
SpeechBrain), so none of this wiring had a test. tests/_pod_stubs.py answers for
those libraries with mocks; nothing here touches them. What runs is the REAL
`_speak`, `_interrupt_playback`, `_handle_control`, `_decide_turn`,
`CallSession.append` and gate, against a fake WebSocket and a fake TTS.

    python -m pytest tests/test_orchestrator_duplex.py -v
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
from _synth_echo import SR, caller, echo_scene

from agent.endpointing import EndpointConfig


def _wav(x, sr=SR):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return b.getvalue()


class FakeWS:
    def __init__(self):
        self.texts, self.audio = [], []

    async def send_text(self, t):
        self.texts.append(t)

    async def send_bytes(self, b):
        self.audio.append(b)

    async def close(self):
        pass

    def frames(self, kind):
        out = []
        for t in self.texts:
            try:
                d = json.loads(t)
            except ValueError:
                continue
            if d.get("type") == kind:
                out.append(d)
        return out


class FakeTTS:
    """Returns a WAV per clause; `on_call(n)` lets a test act while clause n is 'being synthesised'."""

    def __init__(self, wav_for=None, on_call=None):
        self.calls = 0
        self.wav_for = wav_for or (lambda text: _wav(0.1 * np.sin(2 * np.pi * 200 * np.arange(SR // 2) / SR)))
        self.on_call = on_call

    async def synthesize(self, lang, text, speed=1.0):
        self.calls += 1
        if self.on_call:
            self.on_call(self.calls)
        return self.wav_for(text)


@pytest.fixture(scope="module")
def pcm():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


@pytest.fixture(scope="module")
def webm():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main")


@pytest.fixture
def session(pcm, monkeypatch):
    monkeypatch.setattr(pcm, "_admission", None)
    s = pcm.CallSession(FakeWS())
    yield s
    s.cleanup()


# ================================================================= gate wiring (KCD-050)


def test_the_session_gate_behaves_as_the_old_attributes_did(session):
    assert session.agent_speaking is True  # starts closed
    session.release_gate()
    assert session.agent_speaking is False and session.resync_pending is True
    session.hold_gate_for(2.0)
    assert session.agent_speaking is True
    assert session.speak_deadline > session.gate._clock()  # the deadline property tracks the gate
    session.resync_pending = False  # the setter _resync_after_playback uses
    assert session.gate.resync_pending is False


def test_main_and_main_pcm_agree_on_the_gate(webm, pcm):
    for mod in (webm, pcm):
        assert mod.PLAYBACK_GUARD_S == 3.0 and mod.RESYNC_REWIND_S == 0.25


# ============================================================ epochs and discard (KCD-052/163)


@pytest.mark.asyncio
async def test_a_reply_interrupted_mid_synthesis_sends_no_further_clauses(pcm, session):
    def interrupt_during_second_clause(n):
        if n == 2:
            session.speak_epoch += 1  # the caller takes the floor now

    pcm._tts_router = FakeTTS(on_call=interrupt_during_second_clause)
    await pcm._speak(session, "Your appointment is booked. The fee is five hundred rupees. Please arrive early.", "en")
    assert len(session.ws.audio) == 1  # only the clause sent before the interrupt
    assert pcm._tts_router.calls == 2  # and nothing was even synthesised after it


@pytest.mark.asyncio
async def test_a_stale_turn_is_discarded_before_any_work_is_done(pcm, session):
    pcm._tts_router = FakeTTS()
    session.speak_epoch += 1  # interrupted; this turn started earlier
    total = await pcm._speak(session, "This should never be spoken.", "en")
    assert total == 0.0 and session.ws.audio == [] and pcm._tts_router.calls == 0
    session.turn_epoch = session.speak_epoch  # the NEXT turn syncs and speaks normally
    await pcm._speak(session, "This is spoken.", "en")
    assert len(session.ws.audio) == 1


@pytest.mark.asyncio
async def test_the_manual_interrupt_message_stops_playback_and_discards_the_rest(pcm, session):
    pcm._tts_router = FakeTTS()
    await pcm._speak(session, "First.", "en")
    assert session.agent_speaking
    await pcm._handle_control(session, json.dumps({"type": "interrupt"}))
    assert not session.agent_speaking and session.speak_epoch == 1
    assert session.ws.frames("stop_playback") == []  # the client stopped itself; it is not told again
    sent = len(session.ws.audio)
    await pcm._speak(session, "Second, which the caller already interrupted.", "en")
    assert len(session.ws.audio) == sent  # KCD-163: nothing more reaches the caller


@pytest.mark.asyncio
async def test_an_interrupt_with_a_resume_point_moves_the_marker_and_cancels_the_resync(pcm, session):
    session.processed_until_s = 3.0
    await pcm._interrupt_playback(session, "acoustic", resume_from_s=7.5)
    assert session.processed_until_s == 7.5 and session.resync_pending is False
    await pcm._interrupt_playback(session, "acoustic", resume_from_s=2.0)  # never backwards
    assert session.processed_until_s == 7.5


# ======================================================= AEC + barge-in end to end (KCD-051/052)


def _chunks(x, n=2048):
    pcm16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    for i in range(0, pcm16.size, n):
        yield pcm16[i : i + n].tobytes()


async def _run_call(pcm, monkeypatch, with_caller):
    monkeypatch.setattr(pcm, "_admission", None)
    monkeypatch.setattr(pcm, "AEC_BARGE_IN", True)
    far, mic, _ = echo_scene(3, dur_each=5.0)
    onset = int(9.0 * SR)
    if with_caller:
        c = caller(3, dur=1.5, f0=118)
        mic = mic.copy()
        mic[onset : onset + c.size] += c
    session = pcm.CallSession(FakeWS())
    assert session.duplex is not None
    pcm._tts_router = FakeTTS(wav_for=lambda text: _wav(far))  # the agent speaks 15 s of "far end"
    session.turn_epoch = session.speak_epoch
    await pcm._speak(session, "Hello.", "en")
    for chunk in _chunks(mic):
        await session.append(chunk)
    return session, onset


@pytest.mark.asyncio
async def test_a_caller_talking_over_the_agent_stops_playback_and_keeps_their_words(pcm, monkeypatch):
    session, onset = await _run_call(pcm, monkeypatch, with_caller=True)
    try:
        assert len(session.ws.frames("stop_playback")) == 1
        assert session.speak_epoch == 1 and not session.agent_speaking and session.resync_pending is False
        # the utterance start is where the caller began, less the pre-roll -- not the end of the buffer
        assert session.processed_until_s == pytest.approx(onset / SR - pcm.BARGE_IN_PREROLL_S, abs=0.5)
        assert session.processed_until_s < session.audio.duration_s - 1.0
    finally:
        session.cleanup()


@pytest.mark.asyncio
async def test_the_agents_own_echo_alone_never_interrupts_it(pcm, monkeypatch):
    session, _ = await _run_call(pcm, monkeypatch, with_caller=False)
    try:
        assert session.ws.frames("stop_playback") == []
        assert session.speak_epoch == 0 and session.agent_speaking
        assert session.audio.duration_s > 14.0  # the cleaned signal was buffered
    finally:
        session.cleanup()


@pytest.mark.asyncio
async def test_without_aec_the_buffer_holds_the_raw_microphone_and_nothing_can_interrupt(pcm, monkeypatch):
    monkeypatch.setattr(pcm, "_admission", None)
    monkeypatch.setattr(pcm, "AEC_BARGE_IN", False)
    session = pcm.CallSession(FakeWS())
    try:
        assert session.duplex is None
        raw = (0.1 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)).astype(np.float32)
        for chunk in _chunks(raw):
            await session.append(chunk)
        assert session.audio.duration_s == pytest.approx(1.0, abs=0.01)
        assert session.ws.frames("stop_playback") == []
    finally:
        session.cleanup()


# ============================================================ wake scheduling (KCD-049)


def test_wake_delays_follow_the_detectors_own_schedule(pcm):
    pcm.TurnDecision if hasattr(pcm, "TurnDecision") else None
    from agent.endpointing import TurnDecision

    idle = TurnDecision(None, False, "no_speech")
    assert pcm._next_wake_delay(idle) == pcm.POLL_INTERVAL_S
    talking = TurnDecision(None, True, "waiting_silence")
    assert pcm._next_wake_delay(talking) == pcm.ACTIVE_POLL_INTERVAL_S
    soon = TurnDecision(None, True, "waiting_silence", commit_after_s=0.08)
    assert pcm._next_wake_delay(soon) == pytest.approx(0.08 + pcm.WAKE_SLACK_S)
    imminent = TurnDecision(None, True, "waiting_silence", commit_after_s=0.0)
    assert pcm._next_wake_delay(imminent) == pcm.MIN_WAKE_S
    far = TurnDecision(None, True, "waiting_silence", commit_after_s=0.9)
    assert pcm._next_wake_delay(far) == pcm.ACTIVE_POLL_INTERVAL_S  # never sleeps LONGER than the cadence


def test_the_wake_schedule_resets_to_idle_after_it_is_used(session, pcm):
    session.next_wake_s = 0.03
    assert session.take_wake_delay() == 0.03
    assert session.take_wake_delay() == pcm.POLL_INTERVAL_S


def test_the_pcm_variant_trusts_exact_sample_positions_with_a_smaller_tail_guard(webm, pcm):
    assert webm.TURN_TAIL_GUARD_S == 0.3 and pcm.TURN_TAIL_GUARD_S == 0.1


# ====================================================== semantic endpointing (KCD-048)


class FakeDetector:
    def __init__(self, spans, duration_s):
        self._spans, self._duration = spans, duration_s
        self.config = EndpointConfig()

    def spans(self, tail, sr):
        return self._spans, self._duration


class FakeRouter:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.calls = text, error, 0

    async def transcribe(self, lang, path):
        self.calls += 1
        if self.error:
            raise self.error
        return type("R", (), {"text": self.text})()


@pytest.fixture
def endpointing_env(pcm, monkeypatch, tmp_path):
    async def fake_slice(session, start_s, end_s, seq):
        f = tmp_path / f"spec{seq}.wav"
        f.write_bytes(b"x")
        return str(f)

    monkeypatch.setattr(pcm, "_slice_utterance", fake_slice)
    monkeypatch.setattr(pcm, "_turn_detector", FakeDetector([{"start": 0.2, "end": 1.0}], 1.0 + 0.3 + 0.5))
    return pcm


@pytest.mark.asyncio
async def test_a_finished_transcript_ends_the_turn_at_the_shorter_threshold(endpointing_env, session, monkeypatch):
    pcm = endpointing_env
    monkeypatch.setattr(pcm, "SEMANTIC_ENDPOINTING", True)
    pcm._asr_router = FakeRouter("what is the price of a CBC test?")
    session.lang = "en"
    d = await pcm._decide_turn(session, None, 16000)  # 0.5 s of confirmed silence: under the 1.0 s baseline
    assert d.utterance_end_s == 1.0 and pcm._asr_router.calls == 1


@pytest.mark.asyncio
async def test_an_unfinished_transcript_keeps_waiting(endpointing_env, session, monkeypatch):
    pcm = endpointing_env
    monkeypatch.setattr(pcm, "SEMANTIC_ENDPOINTING", True)
    pcm._asr_router = FakeRouter("my number is nine eight and")
    session.lang = "en"
    d = await pcm._decide_turn(session, None, 16000)
    assert d.utterance_end_s is None and d.confirm_s > EndpointConfig().silence_confirm_s


@pytest.mark.asyncio
async def test_the_verdict_is_reused_and_not_recomputed_while_the_speech_end_has_not_moved(
    endpointing_env, session, monkeypatch
):
    pcm = endpointing_env
    monkeypatch.setattr(pcm, "SEMANTIC_ENDPOINTING", True)
    pcm._asr_router = FakeRouter("my number is nine eight and")
    session.lang = "en"
    await pcm._decide_turn(session, None, 16000)
    await pcm._decide_turn(session, None, 16000)
    assert pcm._asr_router.calls == 1


@pytest.mark.asyncio
async def test_speculation_is_bounded_per_turn_and_a_failure_is_only_no_verdict(endpointing_env, session, monkeypatch):
    pcm = endpointing_env
    monkeypatch.setattr(pcm, "SEMANTIC_ENDPOINTING", True)
    pcm._asr_router = FakeRouter(error=RuntimeError("asr down"))
    session.lang = "en"
    for _ in range(6):
        d = await pcm._decide_turn(session, None, 16000)
        assert d.utterance_end_s is None  # the baseline threshold still applies
    assert pcm._asr_router.calls == pcm.MAX_SPECULATIONS_PER_TURN


@pytest.mark.asyncio
async def test_with_the_feature_off_no_transcription_is_ever_requested(endpointing_env, session, monkeypatch):
    pcm = endpointing_env
    monkeypatch.setattr(pcm, "SEMANTIC_ENDPOINTING", False)
    pcm._asr_router = FakeRouter("what is the price of a CBC test?")
    d = await pcm._decide_turn(session, None, 16000)
    assert pcm._asr_router.calls == 0 and d.utterance_end_s is None  # 0.5 s < the 1.0 s baseline
