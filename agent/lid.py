"""Per-utterance language identification, ahead of ASR.

Pilot architecture change (2026-09-15, supersedes the transcript-based LID
in the Architecture & Implementation Plan Section 10): once the agent
routes each utterance to ONE of three language-specific ASR engines
(agent/asr_router.py), the language decision can no longer be made by
reading the ASR transcript -- there is no transcript yet. LID must run on
the AUDIO, before ASR. See docs/adr/0001-pilot-single-l4-architecture.md.

This module is split in two, deliberately:

  - `LanguageIdentifier` is the audio-level model boundary. The pilot
    architecture names SpeechBrain's VoxLingua107 ECAPA-TDNN model
    (bn/hi/en, 16 kHz mono, ~86 MB, CPU-only) as the first implementation
    to try. `SpeechBrainVoxLingua107LID` below has the loading/inference
    shape for it, but IT IS UNVERIFIED -- it has not been run against a
    live model or real audio, because no pod was available when this was
    written. Do not trust it until it has been exercised on the pod; see
    the class docstring.

  - `ASRLanguageRouter` is pure orchestration logic: no model, no audio,
    no I/O. It takes whatever LID confidence you have plus the call's
    running state and decides what to DO about it -- commit to a
    language, run a second ASR for this turn, ask the caller to repeat,
    or give up and hand the call to a human. This half is fully unit
    tested without a GPU, a pod, or even SpeechBrain installed -- see
    tests/test_lid.py, which passes today, on this machine, with nothing
    but the standard library.

Keep language PER UTTERANCE, never latched for the call: a caller may use
Bengali grammar with an English test name, switch to Hindi next turn, then
read a phone number in English.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

SUPPORTED_LANGUAGES = ("bn", "hi", "en")

# Below this, a single LID pass is not trusted to route ASR on its own.
DEFAULT_CONFIDENCE_FLOOR = 0.55

# Consecutive low-confidence/ambiguous turns before the call goes to a
# human rather than continuing to guess. Reasoned, not measured -- see
# the module docstring in agent/fast_path.py for what that label means;
# calibrate this against real Kolkata G.711 calls before trusting it.
DEFAULT_MAX_AMBIGUOUS_STREAK = 3


@dataclasses.dataclass(frozen=True)
class LIDResult:
    """One language-identification pass over one utterance's audio."""

    language: str  # one of SUPPORTED_LANGUAGES, or "unknown"
    confidence: float  # 0.0-1.0
    scores: dict[str, float] = dataclasses.field(default_factory=dict)


class LanguageIdentifier(Protocol):
    """The audio-level boundary. Implementations are model-backed and are
    NOT expected to be unit tested without real audio -- exercise them
    with an integration test on the pod instead."""

    def identify(self, audio_16k_mono: bytes | object) -> LIDResult: ...


class SpeechBrainVoxLingua107LID:
    """SpeechBrain VoxLingua107 ECAPA-TDNN, restricted to bn/hi/en.

    UNVERIFIED. Written from the model card and the pilot architecture
    recommendation, never run: this session had no live pod. Before this
    class is trusted in the call path:

      1. `pip install speechbrain` on the pod (CPU is fine -- the model
         is ~86 MB and the pilot architecture explicitly keeps LID off
         the GPU so the GPU stays free for the six speech models).
      2. Load `speechbrain/lang-id-voxlingua107-ecapa` and confirm it
         actually exposes bn/hi/en in its 107-language label set at the
         installed version -- verify against the live model, not this
         docstring.
      3. Feed it real 8 kHz-origin (resampled to 16 kHz) Kolkata call
         audio, not clean microphone audio, and calibrate
         DEFAULT_CONFIDENCE_FLOOR against THAT distribution.

    Until all three are done, treat this class as a stub with the right
    shape, not a working component.
    """

    MODEL_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._label_idx: dict[str, int] | None = None
        self._model = None  # lazy: keeps import-time cost at zero for
        # every caller (tests, the router, CI) that never needs the model.

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as exc:  # pragma: no cover - exercised on the pod only
            raise RuntimeError(
                "speechbrain is not installed. This is expected off-pod; "
                "install it on the RunPod instance before using this class."
            ) from exc
        import os

        # Without an explicit savedir speechbrain writes ./pretrained_models
        # relative to the process CWD -- i.e. into the repo checkout, or
        # onto the ephemeral overlay depending on where the service starts.
        savedir = os.environ.get("VOICE_AGENT_LID_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "voice-agent-lid"
        )
        self._model = EncoderClassifier.from_hparams(
            source=self.MODEL_SOURCE,
            savedir=savedir,
            run_opts={"device": self.device},
        )

    # A supported-language posterior mass below this means the audio is most
    # likely some OTHER language (or not speech): say "unknown" rather than
    # renormalizing three near-zero numbers into a confident-looking guess.
    MIN_SUPPORTED_MASS = 0.25

    def _require_model(self) -> Any:
        """The loaded model, or a plain error saying it is not (it loads lazily: _ensure_loaded())."""
        if self._model is None:
            raise RuntimeError("the language-identification model is not loaded")
        return self._model

    def _label_index(self) -> dict[str, int]:
        """Model label ("bn: Bengali") -> class index, for the three
        languages we route between. Built once from the loaded model."""
        idx: dict[str, int] = {}
        model = self._require_model()
        for label, i in model.hparams.label_encoder.lab2ind.items():
            code = str(label).split(":")[0].strip().lower()
            if code in SUPPORTED_LANGUAGES:
                idx[code] = int(i)
        return idx

    def identify(self, audio_16k_mono) -> LIDResult:  # pragma: no cover - needs the model
        """`audio_16k_mono`: a float tensor, [T] or [1, T], 16 kHz mono.

        Scoring is restricted to bn/hi/en and renormalized. The model has 107
        classes; taking its raw top-1 would turn every Urdu-sounding Hindi
        utterance into "unknown". Restricting is the right question -- "of
        the languages we can actually serve, which is this?" -- with a
        separate mass check (MIN_SUPPORTED_MASS) for "none of them"."""
        import torch

        self._ensure_loaded()
        wav = audio_16k_mono
        if not torch.is_tensor(wav):
            wav = torch.as_tensor(wav, dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        out_prob, _score, _index, _label = self._require_model().classify_batch(wav)
        log_post = out_prob[0]
        if self._label_idx is None:
            self._label_idx = self._label_index()
        missing = set(SUPPORTED_LANGUAGES) - set(self._label_idx)
        if missing:
            raise RuntimeError(
                f"VoxLingua107 label set at this speechbrain version has no {sorted(missing)}; "
                f"routing cannot work until that is resolved"
            )

        probs = {lang: float(torch.exp(log_post[i])) for lang, i in self._label_idx.items()}
        mass = sum(probs.values())
        if mass < self.MIN_SUPPORTED_MASS:
            return LIDResult(
                language="unknown", confidence=0.0, scores={k: v / mass if mass else 0.0 for k, v in probs.items()}
            )
        scores = {lang: p / mass for lang, p in probs.items()}
        best = max(scores, key=lambda k: scores[k])
        return LIDResult(language=best, confidence=scores[best], scores=scores)

    def load(self) -> None:  # pragma: no cover - needs the model
        """Load the model AND verify its label set has bn/hi/en. Call once at
        startup so a wrong speechbrain version fails at boot, not on the
        first caller's first word."""
        self._ensure_loaded()
        self._label_idx = self._label_index()
        missing = set(SUPPORTED_LANGUAGES) - set(self._label_idx)
        if missing:
            raise RuntimeError(f"VoxLingua107 label set has no {sorted(missing)}")

    def identify_path(self, wav_path: str) -> LIDResult:  # pragma: no cover - needs the model
        """Load a WAV (any rate/channels) as 16 kHz mono and identify it.

        Reads with soundfile, not torchaudio.load: on torch >= 2.9 torchaudio
        delegates file IO to torchcodec, which is one more compiled
        dependency that must match the torch build exactly. LID needs only a
        WAV read; resampling stays in torchaudio.functional (pure torch)."""
        import soundfile as sf
        import torch
        import torchaudio

        data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return self.identify(wav)


@dataclasses.dataclass
class RoutingDecision:
    """What ASRLanguageRouter decided to do with one utterance."""

    action: str  # "commit" | "dual_asr" | "clarify" | "handoff_human"
    language: str | None  # set when action == "commit" or "dual_asr" (primary)
    secondary_language: str | None = None  # set only for "dual_asr"
    reason: str = ""


class ASRLanguageRouter:
    """Turns a raw LID result into a routing decision, per call.

    No model, no audio, no I/O -- fully unit-testable. Owns exactly the
    policy the pilot architecture specifies:

      - Trust a confident LID result outright.
      - Below the confidence floor, prefer the previous turn's language
        (callers rarely switch language mid-thought without a reason).
      - Still ambiguous (no usable prior, or the prior disagrees with a
        low-confidence current read) -> run the top-two ASRs for just
        this one turn rather than all three, to protect the cost/latency
        advantage of routing at all.
      - After DEFAULT_MAX_AMBIGUOUS_STREAK consecutive ambiguous turns in
        the same call, stop guessing and hand off to a human.

    One instance per call (it holds per-call state); do not share across
    concurrent calls.
    """

    def __init__(
        self,
        confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
        max_ambiguous_streak: int = DEFAULT_MAX_AMBIGUOUS_STREAK,
    ):
        self.confidence_floor = confidence_floor
        self.max_ambiguous_streak = max_ambiguous_streak
        self._previous_language: str | None = None
        self._ambiguous_streak = 0

    def route(self, lid: LIDResult) -> RoutingDecision:
        if lid.language in SUPPORTED_LANGUAGES and lid.confidence >= self.confidence_floor:
            self._ambiguous_streak = 0
            self._previous_language = lid.language
            return RoutingDecision(
                action="commit", language=lid.language, reason=f"confidence {lid.confidence:.2f} >= floor"
            )

        if self._previous_language is not None:
            self._ambiguous_streak = 0
            decided = self._previous_language
            return RoutingDecision(
                action="commit",
                language=decided,
                reason=f"low confidence ({lid.confidence:.2f}); used previous-turn prior",
            )

        self._ambiguous_streak += 1
        if self._ambiguous_streak > self.max_ambiguous_streak:
            return RoutingDecision(
                action="handoff_human",
                language=None,
                reason=f"{self._ambiguous_streak} consecutive ambiguous turns with no prior",
            )

        top_two = self._top_two_languages(lid)
        return RoutingDecision(
            action="dual_asr",
            language=top_two[0],
            secondary_language=top_two[1] if len(top_two) > 1 else None,
            reason="no usable prior and low confidence; running top-two ASR for this turn only",
        )

    @property
    def previous_language(self) -> str | None:
        """The language this call has been in (the last committed or answered one), or None on the first turn."""
        return self._previous_language

    def note_response_language(self, language: str) -> None:
        """Call after the response layer picks a language for the reply
        (e.g. from a clarification), so the next turn's prior is correct
        even on a turn where LID itself did not commit."""
        if language in SUPPORTED_LANGUAGES:
            self._previous_language = language

    def _top_two_languages(self, lid: LIDResult) -> list[str]:
        if lid.scores:
            ranked = sorted(
                ((lang, score) for lang, score in lid.scores.items() if lang in SUPPORTED_LANGUAGES),
                key=lambda pair: pair[1],
                reverse=True,
            )
            if ranked:
                return [lang for lang, _ in ranked[:2]]
        return list(SUPPORTED_LANGUAGES[:2])
