"""Routes one utterance to its language-specific ASR engine.

Pilot architecture (docs/adr/0001-pilot-single-l4-architecture.md): three
ASR checkpoints stay resident on the one production GPU simultaneously --
never unloaded/reloaded on a language switch, because a model-load stall
mid-call is worse than the small fixed VRAM cost of keeping all three warm.
For any one utterance, exactly one (or, on `dual_asr`, two) is actually
run. This module is the "exactly one" part: a plain registry keyed by
language, with the actual engines injected rather than imported, so the
routing decision is unit-testable with fakes and carries no GPU, torch or
NeMo dependency of its own.

Engine construction (loading IndicConformer-bn, IndicConformer-hi, or the
NVIDIA FastConformer English checkpoint) belongs to whoever wires this
router at process startup -- see agent/asr.py for the proven bn loading
shape; hi and en checkpoints are new and unverified off-pod, same caveat
as agent/lid.py.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

from agent.lid import SUPPORTED_LANGUAGES


class ASREngine(Protocol):
    """Whatever agent/asr.py's TurnASR (or its hi/en siblings) exposes.
    The router only ever calls this one method."""

    async def transcribe_utterance(self, wav_path: str): ...


@dataclasses.dataclass
class ASRModelSpec:
    """One row of the exact-model-selection table from the pilot
    architecture. `checkpoint` and `engine_family` are documentation --
    the router does not load anything itself."""

    language: str
    checkpoint: str
    engine_family: str  # "ai4bharat_nemo_fork" | "nvidia_nemo"


PILOT_ASR_MODELS: dict[str, ASRModelSpec] = {
    "bn": ASRModelSpec(
        language="bn",
        checkpoint="ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large",
        engine_family="ai4bharat_nemo_fork",
    ),
    "hi": ASRModelSpec(
        language="hi",
        checkpoint="ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large",
        engine_family="ai4bharat_nemo_fork",
    ),
    "en": ASRModelSpec(
        language="en",
        checkpoint="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
        engine_family="nvidia_nemo",
    ),
}


class HTTPASREngine:
    """ASREngine over HTTP, for a language whose model lives in a
    different process/venv than the orchestrator -- concretely, English
    ASR (english_asr_server.py, mainline NeMo, its own venv; see that
    file's docstring for why it cannot be in-process here). Shares
    /workspace with the orchestrator, so the WAV path is passed, not the
    audio bytes -- both processes can already read the same file.

    Bengali and Hindi do NOT need this: same AI4Bharat NeMo fork, no
    environment conflict, loaded directly as agent.asr.TurnASR instances
    instead.
    """

    def __init__(self, base_url: str, timeout_s: float = 15.0):
        import httpx

        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

    async def transcribe_utterance(self, wav_path: str):
        r = await self._client.post("/transcribe", json={"wav_path": wav_path})
        r.raise_for_status()
        data = r.json()
        # Local import to avoid a hard dependency on agent.asr (and
        # therefore on the AI4Bharat NeMo fork) for a class that exists
        # specifically so English does NOT need that fork installed.
        from agent.asr import ASRResult

        return ASRResult(**{k: v for k, v in data.items() if k in ASRResult.__dataclass_fields__})

    async def aclose(self):
        await self._client.aclose()


class UnroutableLanguageError(Exception):
    """Raised when asked to route a language with no registered engine.
    The caller (the orchestrator) should treat this exactly like an
    ASRLanguageRouter `handoff_human` decision -- do not guess."""


class ASRRouter:
    """Language -> ASR engine. One instance per process (engines are the
    expensive, process-wide singletons); safe to share across calls."""

    def __init__(self, engines: dict[str, ASREngine]):
        missing = set(SUPPORTED_LANGUAGES) - set(engines)
        if missing:
            raise ValueError(
                f"ASRRouter constructed without engines for: {sorted(missing)}. "
                f"All of {SUPPORTED_LANGUAGES} must be registered even if a "
                f"given pod build only warms a subset for now -- register a "
                f"clearly-failing stub rather than omitting the key, so a "
                f"missing engine fails at startup, not mid-call."
            )
        self._engines = dict(engines)

    def engine_for(self, language: str) -> ASREngine:
        try:
            return self._engines[language]
        except KeyError as exc:
            raise UnroutableLanguageError(f"No ASR engine registered for language={language!r}") from exc

    async def transcribe(self, language: str, wav_path: str):
        """Convenience wrapper for the common `commit` routing decision."""
        return await self.engine_for(language).transcribe_utterance(wav_path)

    async def transcribe_many(self, languages: list[str], wav_path: str) -> list[tuple[str, object]]:
        """Run several engines on one clip, concurrently. Returns
        [(language, result)] for the ones that succeeded, in the order given.

        An engine that raises (English ASR is a separate process and can be
        down) is dropped, not fatal: verification exists to pick the best of
        what is available. Only when EVERY engine fails does this raise."""
        import asyncio
        import logging
        import time

        took: dict[str, float] = {}

        async def timed(lang: str):
            t0 = time.perf_counter()
            try:
                return await self.transcribe(lang, wav_path)
            finally:
                took[lang] = (time.perf_counter() - t0) * 1000

        outcomes = await asyncio.gather(*[timed(lang) for lang in languages], return_exceptions=True)
        logging.getLogger("asr_router").info(
            "ASR timings (concurrent): %s", " | ".join(f"{lang} {int(ms)} ms" for lang, ms in took.items())
        )
        good = []
        for lang, out in zip(languages, outcomes):
            if isinstance(out, BaseException):
                logging.getLogger("asr_router").warning("ASR %s failed on verify: %s", lang, out)
            else:
                good.append((lang, out))
        if not good:
            raise next(o for o in outcomes if isinstance(o, BaseException))
        return good

    async def transcribe_dual(self, primary: str, secondary: str, wav_path: str):
        """For the `dual_asr` routing decision: run two engines on the
        same clip and return both results, primary first, for the
        orchestrator to disambiguate (e.g. by decoder confidence or a
        downstream intent-match score). Runs concurrently -- it is
        already paying for two ASR passes; it should not also pay their
        latency serially."""
        import asyncio

        primary_result, secondary_result = await asyncio.gather(
            self.transcribe(primary, wav_path),
            self.transcribe(secondary, wav_path),
        )
        return primary_result, secondary_result
