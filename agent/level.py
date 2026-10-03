"""KCD-057: input level normalisation with a limiter.

A caller speaking quietly, or from arm's length, produces speech at -50 dBFS; one
shouting into the handset clips. Recognisers cope with a range of levels but not
an unlimited one -- quantisation noise dominates a very quiet clip, and a clipped
one has lost its peaks for good. This brings an utterance to a target speech level
before recognition, and limits its peaks so the gain can never clip it.

Design decisions, each with a reason:

  * ONE static gain per utterance, from the level of its ACTIVE speech (the loud
    frames, not the silence around them). A per-sample AGC would pump the noise
    floor up in every pause and reshape syllable dynamics the recogniser has
    learned from.
  * Gain is bounded: at most +30 dB (a clip that quiet is mostly noise; amplifying
    it further only amplifies the noise) and at most -15 dB of cut.
  * Already-clipped input is never amplified: the gain is capped at 0 dB when the
    clipping fraction is high, because amplification cannot restore what was cut
    off and only makes the distortion louder.
  * The limiter is a soft knee (tanh above a threshold), not a hard clip, so it
    adds no sharp corners. Its ceiling is -1 dBFS: the output can never exceed it.

LEVEL BANDS. KCD-057 asks accuracy to be measured across four input level bands
with the worst inside a stated margin of the best. The bands below are REASONED
(bounds of active-speech RMS in dBFS); agent/conditioning.py's bench and
tools/conditioning_eval.py measure per band. Off-pod the measurement is of LEVEL
and of speech-feature preservation (proxies); recognition ACCURACY needs the ASR
and real recordings, which tools/conditioning_eval.py provides on a pod.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from agent.audio_quality import _db, _frames

# (name, lower dBFS, upper dBFS) of ACTIVE-speech RMS. REASONED, not measured.
LEVEL_BANDS = (
    ("very_quiet", -120.0, -45.0),
    ("quiet", -45.0, -35.0),
    ("normal", -35.0, -20.0),
    ("loud", -20.0, 0.0),
)
TARGET_RMS_DBFS = -24.0
MAX_GAIN_DB = 30.0
MAX_CUT_DB = 15.0
CEILING_DBFS = -1.0
LIMITER_THRESHOLD_DBFS = -6.0
CLIPPED_FRACTION = 0.005  # above this the input is treated as already clipped


@dataclasses.dataclass
class LevelReport:
    band_before: str
    speech_rms_before_dbfs: float
    speech_rms_after_dbfs: float
    gain_db: float
    limiter_fraction: float  # share of samples the soft knee touched
    clipped_input: bool


def speech_rms_dbfs(x: np.ndarray, sr: int = 16000) -> float:
    """RMS of the ACTIVE frames (those within 25 dB of the loud end and above the
    quiet third), in dBFS; -120 if there is no speech-level energy."""
    x = np.asarray(x, dtype=np.float32)
    frames = _frames(x - (float(np.mean(x)) if x.size else 0.0), sr)
    if frames.shape[0] < 5:
        return -120.0
    e = _db(np.mean(frames**2, axis=1))
    floor, top = float(np.percentile(e, 10)), float(np.percentile(e, 90))
    active = e >= max(floor + 0.35 * (top - floor), top - 25.0)
    return float(_db(np.mean(frames[active] ** 2))) if active.any() else -120.0


def level_band(rms_dbfs: float) -> str:
    for name, lo, hi in LEVEL_BANDS:
        if lo <= rms_dbfs < hi:
            return name
    return LEVEL_BANDS[-1][0] if rms_dbfs >= 0 else LEVEL_BANDS[0][0]


def soft_limit(
    x: np.ndarray, ceiling_dbfs: float = CEILING_DBFS, threshold_dbfs: float = LIMITER_THRESHOLD_DBFS
) -> tuple[np.ndarray, float]:
    """Soft-knee limiter: identity below the threshold, tanh-compressed above it,
    asymptotically approaching the ceiling and never exceeding it."""
    c, t = 10 ** (ceiling_dbfs / 20.0), 10 ** (threshold_dbfs / 20.0)
    a = np.abs(x)
    over = a > t
    y = x.copy()
    if over.any():
        y[over] = np.sign(x[over]) * (t + (c - t) * np.tanh((a[over] - t) / (c - t)))
    return y.astype(np.float32), float(np.mean(over)) if x.size else 0.0


def normalise(x: np.ndarray, sr: int = 16000, target_dbfs: float = TARGET_RMS_DBFS) -> tuple[np.ndarray, LevelReport]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    before = speech_rms_dbfs(x, sr)
    clipped = bool(x.size and float(np.mean(np.abs(x) >= 0.99)) >= CLIPPED_FRACTION)
    if before <= -100.0:
        return x.copy(), LevelReport(
            level_band(before), before, before, 0.0, 0.0, clipped
        )  # no speech: nothing to scale
    gain_db = float(np.clip(target_dbfs - before, -MAX_CUT_DB, MAX_GAIN_DB))
    if clipped:
        gain_db = min(gain_db, 0.0)
    y, lim = soft_limit(x * (10 ** (gain_db / 20.0)))
    return y, LevelReport(level_band(before), before, speech_rms_dbfs(y, sr), gain_db, lim, clipped)
