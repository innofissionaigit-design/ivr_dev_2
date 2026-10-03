"""Routes a verbalised reply to its language-specific TTS voice.

Same shape and same reasoning as agent/asr_router.py: a plain registry,
engines injected not imported, fully unit-testable without a GPU. See
that module's docstring for why all three engines stay resident.

Keep this router dumb on purpose. It does not decide caller-state speech
parameters (rate, pause length, first-clause streaming) -- those come
from the Response Policy layer (Blueprint 2.1 [8] / [9], Appendix C) and
are passed through, not decided here.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

from agent.lid import SUPPORTED_LANGUAGES


class TTSEngine(Protocol):
    """Whatever agent/tts.py's TTSClient (or its hi/en siblings) exposes."""

    async def synthesize(self, text: str, **speech_params) -> bytes: ...


@dataclasses.dataclass
class TTSModelSpec:
    language: str
    acoustic_model: str
    vocoder: str
    source: str


PILOT_TTS_MODELS: dict[str, TTSModelSpec] = {
    "bn": TTSModelSpec(
        language="bn",
        acoustic_model="FastPitch",
        vocoder="HiFi-GAN",
        source="ai4bharat/indic-tts (existing checkpoint, already in production)",
    ),
    "hi": TTSModelSpec(
        language="hi",
        acoustic_model="FastPitch",
        vocoder="HiFi-GAN",
        source="ai4bharat/indic-tts, Hindi checkpoint",
    ),
    "en": TTSModelSpec(
        language="en",
        acoustic_model="FastPitch",
        vocoder="HiFi-GAN",
        source="nvidia/tts_en_fastpitch + nvidia/tts_hifigan",
    ),
}


class UnroutableLanguageError(Exception):
    """Same contract as agent.asr_router.UnroutableLanguageError -- treat
    as a hard failure, not a silent fallback to another language's voice."""


class TTSRouter:
    def __init__(self, engines: dict[str, TTSEngine]):
        missing = set(SUPPORTED_LANGUAGES) - set(engines)
        if missing:
            raise ValueError(
                f"TTSRouter constructed without engines for: {sorted(missing)}. "
                f"Register a clearly-failing stub rather than omitting the key."
            )
        self._engines = dict(engines)

    def engine_for(self, language: str) -> TTSEngine:
        try:
            return self._engines[language]
        except KeyError as exc:
            raise UnroutableLanguageError(f"No TTS engine registered for language={language!r}") from exc

    async def synthesize(self, language: str, text: str, **speech_params) -> bytes:
        return await self.engine_for(language).synthesize(text, **speech_params)
