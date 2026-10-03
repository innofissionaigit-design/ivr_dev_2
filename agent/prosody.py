"""KCD-162: clause-level prosody with punctuation-aware pauses.

The pure half of tts_server.py -- how a reply is split into chunks, how
long a pause each chunk earns, how ragged padding is trimmed and how the
level is normalised. It lives here, numpy-only, rather than inside
tts_server.py because that file loads three GPU synthesizers and changes
its working directory at import, so nothing in it can be unit tested off a
pod. tts_server.py keeps only the part that genuinely needs the models:
calling synth.tts() once per chunk.

tts_server.py runs in its own venv (coqui-tts and the NeMo fork cannot
share one) but deploy/env.sh puts the repo root on PYTHONPATH, so this
module imports there as `agent.prosody`. It must stay free of anything
beyond numpy and the standard library for exactly that reason.

Every value in ProsodyParams is overridable per request (see
resolve_params) and range-checked, so delivery can be A/B tested against a
real handset without a redeploy -- the story's own requirement.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np

# Terminators that end a SENTENCE. "।" is the Bengali/Hindi danda; "." is
# what English uses exclusively and Hindi text often does too -- the
# earlier splitter omitted it, so an English reply of several sentences
# was rendered as one flat pass with no sentence pause at all.
_SENTENCE_END = "।?!."
_CLAUSE_END = ",;:"

# A "." is a sentence end only when followed by whitespace or the end of
# the text, and not when it closes one of these. "Dr. Sen" must not be
# read as "Dr." / "Sen"; "2.5" never splits because a digit follows.
_ABBREVIATIONS = frozenset({"dr", "mr", "mrs", "ms", "rs", "st", "no", "vs", "prof", "sr", "jr"})


@dataclasses.dataclass(frozen=True)
class ProsodyParams:
    pause_sentence_s: float = 0.28
    pause_clause_s: float = 0.14
    pause_none_s: float = 0.06
    # REASONED, not measured: both were chosen from how synthetic and studio
    # speech behaves, not from handset playback. Treat as starting points until
    # measured on real phone speakers.
    trim_threshold: float = 0.012  # amplitude below which a sample is treated as padding
    target_peak: float = 0.89  # ~-1 dBFS peak target
    max_chunk_chars: int = 90  # a long single clause still gets a breath
    lead_in_s: float = 0.04  # stops the first phoneme clipping on stream start


# (min, max) for each overridable field. A request outside these is
# rejected rather than clamped silently, so a typo in a tuning call is
# visible instead of quietly producing something different from what was asked.
BOUNDS: dict[str, tuple[float, float]] = {
    "pause_sentence_s": (0.0, 2.0),
    "pause_clause_s": (0.0, 2.0),
    "pause_none_s": (0.0, 2.0),
    "trim_threshold": (0.0, 0.2),
    "target_peak": (0.1, 0.99),
    "max_chunk_chars": (20, 400),
    "lead_in_s": (0.0, 0.5),
}


def resolve_params(base: ProsodyParams, overrides: dict | None = None) -> ProsodyParams:
    """`base` with any non-None `overrides` applied. Unknown keys and
    out-of-range values raise ValueError -- see BOUNDS."""
    if not overrides:
        return base
    fields = {f.name for f in dataclasses.fields(ProsodyParams)}
    applied = {}
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in fields:
            raise ValueError(f"unknown prosody parameter {key!r}")
        lo, hi = BOUNDS[key]
        if not (lo <= value <= hi):
            raise ValueError(f"{key}={value} outside [{lo}, {hi}]")
        applied[key] = int(value) if key == "max_chunk_chars" else float(value)
    return dataclasses.replace(base, **cast("dict[str, Any]", applied))


def _is_sentence_boundary(text: str, i: int) -> bool:
    ch = text[i]
    if ch in "।?!\n":
        return True
    if ch != ".":
        return False
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if nxt and not nxt.isspace():
        return False  # "2.5", "a.b", "e.g.x"
    j = i - 1
    while j >= 0 and text[j].isalpha():
        j -= 1
    return text[j + 1 : i].lower() not in _ABBREVIATIONS


def _split_sentences_marked(text: str) -> list[tuple[str, bool]]:
    """[(sentence, ended_at_a_newline)]. A newline is a sentence boundary but
    is stripped from the chunk, so without the flag the chunk would look
    unterminated and earn no pause at all."""
    sentences, start = [], 0
    for i in range(len(text)):
        if _is_sentence_boundary(text, i):
            sentences.append((text[start : i + 1], text[i] == chr(10)))
            start = i + 1
    if start < len(text):
        sentences.append((text[start:], False))
    return [(s.strip(), nl_end) for s, nl_end in sentences if s.strip()]


def _split_sentences(text: str) -> list[str]:
    return [s for s, _ in _split_sentences_marked(text)]


def _wrap_words(clause: str, limit: int) -> list[str]:
    """Split `clause` at word boundaries into pieces of at most `limit`
    characters (a single word longer than the limit is kept whole)."""
    pieces, cur = [], ""
    for word in clause.split():
        if cur and len(cur) + 1 + len(word) > limit:
            pieces.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        pieces.append(cur)
    return pieces


def _split_clauses(sentence: str) -> list[str]:
    clauses, start = [], 0
    for i, ch in enumerate(sentence):
        if ch in _CLAUSE_END:
            clauses.append(sentence[start : i + 1])
            start = i + 1
    if start < len(sentence):
        clauses.append(sentence[start:])
    return [c.strip() for c in clauses if c.strip()]


def pause_kind_for(chunk: str) -> str:
    """The pause a chunk earns from the punctuation that actually ended
    it -- not a label handed down by whichever splitter produced it. A
    chunk cut for length with no terminator gets the shortest pause, not
    a full sentence's worth."""
    end = chunk.rstrip()[-1:] if chunk.strip() else ""
    if end and end in _SENTENCE_END:
        return "sentence"
    if end and end in _CLAUSE_END:
        return "clause"
    return "none"


def split_for_prosody(text: str, max_chunk_chars: int = ProsodyParams.max_chunk_chars) -> list[tuple[str, str]]:
    """-> [(chunk_text, pause_kind)]. Sentences first, then clauses inside
    any sentence longer than `max_chunk_chars`, so a listener gets a
    breath where a person would take one."""
    out: list[tuple[str, str]] = []
    for sentence, at_newline in _split_sentences_marked(text):

        def kind(chunk: str, last_of_sentence: bool, at_newline: bool = at_newline) -> str:
            k = pause_kind_for(chunk)
            return "sentence" if (at_newline and last_of_sentence and k == "none") else k

        if len(sentence) <= max_chunk_chars:
            out.append((sentence, kind(sentence, True)))
            continue
        clauses = _split_clauses(sentence)
        for ci, clause in enumerate(clauses):
            last_clause = ci == len(clauses) - 1
            if len(clause) <= max_chunk_chars:
                out.append((clause, kind(clause, last_clause)))
                continue
            # A long clause with no comma still has to breathe: cut it at word
            # boundaries; intermediate pieces get the shortest pause, the
            # final piece keeps the clause's own.
            pieces = _wrap_words(clause, max_chunk_chars)
            for pi, piece in enumerate(pieces):
                if pi < len(pieces) - 1:
                    out.append((piece, "none"))
                else:
                    out.append((piece, kind(piece, last_clause)))
    if not out and text.strip():
        out = [(text.strip(), pause_kind_for(text))]
    return out


def trim_silence(wav: np.ndarray, threshold: float) -> np.ndarray:
    loud = np.where(np.abs(wav) > threshold)[0]
    if loud.size == 0:
        return wav[:0]
    return wav[loud[0] : loud[-1] + 1]


def normalize_peak(wav: np.ndarray, target: float) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav * (target / peak) if peak > 1e-6 else wav


def assemble(
    chunk_wavs: list[tuple[np.ndarray, str]], sample_rate: int, params: ProsodyParams, pauses: bool = True
) -> np.ndarray:
    """Trim each rendered chunk, pad it by the punctuation that ended it,
    join, add the lead-in and normalise the whole to one peak. `chunk_wavs`
    is [(raw_waveform, pause_kind)] in speaking order."""
    pause_s = {"sentence": params.pause_sentence_s, "clause": params.pause_clause_s, "none": params.pause_none_s}
    pieces: list[np.ndarray] = []
    for raw, kind in chunk_wavs:
        wav = trim_silence(np.asarray(raw, dtype=np.float32), params.trim_threshold)
        if wav.size == 0:
            continue
        pieces.append(wav)
        if pauses:
            pieces.append(np.zeros(int(pause_s[kind] * sample_rate), dtype=np.float32))
    if not pieces:
        return np.zeros(int(0.2 * sample_rate), dtype=np.float32)
    lead_in = np.zeros(int(params.lead_in_s * sample_rate), dtype=np.float32)
    return normalize_peak(np.concatenate([lead_in, *pieces]), params.target_peak)


# ------------------------------------------------------------ speaking rate

RATE_BOUNDS = (0.5, 2.0)
LENGTH_SCALE_BOUNDS = (0.5, 2.5)


def resolve_length_scale(
    default_length_scale: float, rate: float | None = None, length_scale: float | None = None
) -> float:
    """Coqui's FastPitch `length_scale` is a DURATION multiplier: above 1
    is SLOWER. The agent side speaks in RATES (agent/tts.py's
    FIGURE_SPEECH_SPEED = 0.8 is documented as "a fifth slower", and
    agent/speech_policy.py's senior rate is likewise <1 for slower), so the
    server has to convert. It did not: it assigned the incoming `speed`
    straight to length_scale, so a price or phone number sent at 0.8 was
    rendered with length_scale 0.8 -- about 35% FASTER than the 1.08
    default -- the exact opposite of KCD-456's intent. Found while
    implementing KCD-157.

    `length_scale` (explicit) wins for anyone who really wants the raw
    duration multiplier; otherwise `rate` divides the default."""
    if length_scale is not None:
        lo, hi = LENGTH_SCALE_BOUNDS
        if not lo <= length_scale <= hi:
            raise ValueError(f"length_scale={length_scale} outside [{lo}, {hi}]")
        return float(length_scale)
    if rate is None:
        return float(default_length_scale)
    lo, hi = RATE_BOUNDS
    if not lo <= rate <= hi:
        raise ValueError(f"speed={rate} outside [{lo}, {hi}]")
    return float(default_length_scale) / float(rate)
