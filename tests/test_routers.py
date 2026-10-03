"""Unit tests for agent/asr_router.py and agent/tts_router.py -- routing
logic only, with fake engines. No GPU, no models, no pod.

    python -m pytest tests/test_routers.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agent.asr_router import ASRRouter
from agent.asr_router import UnroutableLanguageError as ASRUnroutable
from agent.tts_router import TTSRouter
from agent.tts_router import UnroutableLanguageError as TTSUnroutable


class _FakeASREngine:
    def __init__(self, tag: str):
        self.tag = tag
        self.calls = []

    async def transcribe_utterance(self, wav_path: str):
        self.calls.append(wav_path)
        return f"{self.tag}:{wav_path}"


class _FakeTTSEngine:
    def __init__(self, tag: str):
        self.tag = tag

    async def synthesize(self, text: str, **speech_params):
        return f"{self.tag}:{text}".encode()


def _asr_engines():
    return {"bn": _FakeASREngine("bn"), "hi": _FakeASREngine("hi"), "en": _FakeASREngine("en")}


def _tts_engines():
    return {"bn": _FakeTTSEngine("bn"), "hi": _FakeTTSEngine("hi"), "en": _FakeTTSEngine("en")}


def test_asr_router_rejects_incomplete_registration():
    with pytest.raises(ValueError):
        ASRRouter({"bn": _FakeASREngine("bn")})


def test_asr_router_selects_the_right_engine():
    router = ASRRouter(_asr_engines())
    assert router.engine_for("hi").tag == "hi"


def test_asr_router_unknown_language_is_unroutable():
    router = ASRRouter(_asr_engines())
    with pytest.raises(ASRUnroutable):
        router.engine_for("bengali")  # not the registered key


@pytest.mark.asyncio
async def test_asr_router_transcribe_dispatches_to_one_engine():
    engines = _asr_engines()
    router = ASRRouter(engines)
    result = await router.transcribe("en", "/tmp/clip.wav")
    assert result == "en:/tmp/clip.wav"
    assert engines["bn"].calls == []
    assert engines["en"].calls == ["/tmp/clip.wav"]


@pytest.mark.asyncio
async def test_asr_router_transcribe_dual_runs_both_concurrently():
    engines = _asr_engines()
    router = ASRRouter(engines)
    primary, secondary = await router.transcribe_dual("hi", "bn", "/tmp/clip.wav")
    assert primary == "hi:/tmp/clip.wav"
    assert secondary == "bn:/tmp/clip.wav"


def test_tts_router_rejects_incomplete_registration():
    with pytest.raises(ValueError):
        TTSRouter({"bn": _FakeTTSEngine("bn"), "hi": _FakeTTSEngine("hi")})


@pytest.mark.asyncio
async def test_tts_router_synthesize_dispatches_to_one_engine():
    router = TTSRouter(_tts_engines())
    audio = await router.synthesize("bn", "hello", speech_rate="slow")
    assert audio == b"bn:hello"


def test_tts_router_unknown_language_is_unroutable():
    router = TTSRouter(_tts_engines())
    with pytest.raises(TTSUnroutable):
        router.engine_for("bangla")
