"""KCD-053: follow the caller, not the person beside them.

Three things a caller in a busy room needs, all numpy-only and cheap enough to
run inside the per-turn analysis budget (agent/detector_budget.py):

  1. MARK the overlap. `overlap_segments()` returns where in an utterance a
     second, non-harmonic voice is alive under the dominant one -- the same
     two-pitch test agent/audio_quality.assess() uses for its cross-talk verdict,
     but reported as time segments instead of one number.
  2. ATTEND to the dominant talker. `dominant_span()` trims background speech
     that sits at the start or end of the clip (someone answering the caller
     from across the room before or after the caller speaks), keeping the span
     where the near-end voice carries the energy.
  3. NEVER OPEN A TURN on a background utterance. `NearEndProfile` remembers how
     loud, and at what pitch, this call's near-end speaker has been. An
     utterance that is much QUIETER and in a DIFFERENT VOICE is someone else in
     the room, not the caller, and is not a turn. The same voice, merely
     quieter, is still the caller (they moved the handset): that goes to the
     empathetic re-ask path (agent/reask_policy.py), not into the bin.

WHAT THIS CANNOT DO. It cannot separate two people talking at once -- that needs
a trained source-separation model -- and it cannot tell a second person who is
as loud as the caller from the caller. It can only decline to treat a clearly
distant, clearly different voice as a turn, and say when overlap happened so the
conversation can react. The first utterance of a call has no profile to compare
with, so only overlap marking applies to it.

Thresholds are REASONED and validated on SYNTHETIC voices (tests/test_near_end.py);
recalibrate on recorded rooms before trusting them past the pilot. Cross-talk
rate is exported (agent/golden_buckets.py) as the Appendix F `cross_talk` channel
bucket.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from agent.audio_quality import (
    FRAME_S,
    HOP_S,
    VOICED_HARMONICITY,
    _autocorr,
    _db,
    _frames,
    _pitch_lag,
    _refine_lag,
    _second_pitch,
)

# An utterance this much quieter than the call's near-end level, in a different
# voice, is background (REASONED: bystanders are typically 10-25 dB down at a
# handset microphone).
BACKGROUND_DELTA_DB = 12.0
# Two voices whose median pitch differs by more than this ratio are different
# speakers for this purpose (REASONED: within-speaker median f0 wanders a few
# percent between utterances; men and women differ by ~1.6-2x).
DIFFERENT_VOICE_F0_RATIO = 1.15
MIN_OVERLAP_S = 0.08
MERGE_GAP_S = 0.10
PROFILE_ALPHA = 0.3
# When two voices are about equally loud the "dominant" pitch flips from frame to frame, and the
# residual test in audio_quality._second_pitch fires on only a minority of frames. What still shows is
# the pitch track ALTERNATING between two tight, well-separated values. REASONED, conservative: a
# window of ALTERNATION_WINDOW hops in which two f0 clusters (each within CLUSTER_TOL of its centre)
# hold at least CLUSTER_MIN_SHARE of the voiced frames each, are at least ALTERNATION_RATIO apart, and
# leave under CLUSTER_MAX_BETWEEN in between. A single voice gliding from high to low fills the gap
# between the clusters and so does not qualify; ratio 1.4 is above the ~1.15 within-speaker wander.
ALTERNATION_WINDOW = 41
ALTERNATION_RATIO = 1.4
CLUSTER_TOL = 0.08
CLUSTER_MIN_SHARE = 0.25
CLUSTER_MAX_BETWEEN = 0.15


def _voiced_track(samples: np.ndarray, sr: int):
    """(frame start times, energy dB, f0 Hz or 0, second-pitch flag) per 10 ms hop
    over the ACTIVE frames -- the same framing assess() uses."""
    x = np.asarray(samples, dtype=np.float32)
    x = x - float(np.mean(x)) if x.size else x
    frames = _frames(x, sr)
    if frames.shape[0] < 5:
        return None
    energy = _db(np.mean(frames**2, axis=1))
    floor, top = float(np.percentile(energy, 10)), float(np.percentile(energy, 90))
    active = energy >= max(floor + 0.35 * (top - floor), top - 25.0)
    idx = np.where(active)[0]
    f0 = np.zeros(frames.shape[0])
    second = np.zeros(frames.shape[0], dtype=bool)
    if idx.size:
        acf = _autocorr(frames[idx])
        lags, harm = _pitch_lag(acf, sr)
        for j, i in enumerate(idx):
            if harm[j] >= VOICED_HARMONICITY:
                f0[i] = sr / max(float(_refine_lag(acf[j], int(lags[j]))), 1.0)
                second[i] = _second_pitch(frames[i], _refine_lag(acf[j], int(lags[j])), sr)
    return energy, f0, second, active


def _alternation_mask(f0: np.ndarray) -> np.ndarray:
    """Frames inside a window whose voiced pitch track alternates between two well-separated,
    tight values (see ALTERNATION_* above): evidence of two voices when neither dominates."""
    out = np.zeros(f0.shape[0], dtype=bool)
    half = ALTERNATION_WINDOW // 2
    for c in range(0, f0.shape[0], 5):  # every 50 ms; a window is 0.4 s
        lo, hi = max(0, c - half), min(f0.shape[0], c + half + 1)
        v = f0[lo:hi][f0[lo:hi] > 0]
        if v.size < 12:
            continue
        low, high = float(np.percentile(v, 15)), float(np.percentile(v, 85))
        if high / max(low, 1.0) < ALTERNATION_RATIO:
            continue
        near_low = np.abs(v / low - 1.0) <= CLUSTER_TOL
        near_high = np.abs(v / high - 1.0) <= CLUSTER_TOL
        between = 1.0 - float(np.mean(near_low | near_high))
        if (
            near_low.mean() >= CLUSTER_MIN_SHARE
            and near_high.mean() >= CLUSTER_MIN_SHARE
            and between <= CLUSTER_MAX_BETWEEN
        ):
            out[lo:hi] |= f0[lo:hi] > 0
    return out


def overlap_segments(samples: np.ndarray, sr: int = 16000) -> list[tuple[float, float]]:
    """Time segments (seconds) in which a second voice is alive under the dominant
    one. Runs of overlapped frames closer than MERGE_GAP_S are merged; runs
    shorter than MIN_OVERLAP_S are a glitch, not a talker."""
    tr = _voiced_track(samples, sr)
    if tr is None:
        return []
    _, f0, second, _ = tr
    second = second | _alternation_mask(f0)
    segs, start, last = [], None, None
    for i in np.where(second)[0]:
        t = i * HOP_S
        if start is None:
            start, last = t, t
        elif t - last <= MERGE_GAP_S:
            last = t
        else:
            segs.append((start, last))
            start, last = t, t
    if start is not None:
        segs.append((start, last))
    return [(a, b + HOP_S * 4) for a, b in segs if (b - a) + HOP_S * 4 >= MIN_OVERLAP_S]


def dominant_span(
    samples: np.ndarray, sr: int = 16000, drop_db: float = 12.0, smooth_s: float = 0.25, min_len_s: float = 0.3
) -> tuple[float, float] | None:
    """The stretch of the clip in which the loudest voice carries the energy,
    with leading and trailing stretches more than `drop_db` below it removed.
    Energy is smoothed over `smooth_s` first, so the gaps between a talker's own
    syllables never read as "background". None if there is no speech."""
    x = np.asarray(samples, dtype=np.float32)
    frames = _frames(x - (float(np.mean(x)) if x.size else 0.0), sr)
    if frames.shape[0] < 5:
        return None
    energy = np.mean(frames**2, axis=1)
    k = max(1, int(smooth_s / HOP_S))
    smooth_db = _db(np.convolve(energy, np.ones(k) / k, mode="same"))
    peak = float(np.max(smooth_db))
    keep = np.where(smooth_db >= peak - drop_db)[0]
    if keep.size == 0:
        return None
    a, b = keep[0] * HOP_S, keep[-1] * HOP_S + FRAME_S
    b = min(b, x.size / sr)
    return (a, b) if b - a >= min_len_s else None


def level_and_pitch(samples: np.ndarray, sr: int = 16000) -> tuple[float, float]:
    """(active-speech RMS in dBFS, median f0 in Hz or 0.0 if nothing voiced)."""
    tr = _voiced_track(samples, sr)
    if tr is None:
        return -120.0, 0.0
    energy, f0, _, active = tr
    level = float(_db(np.mean(10 ** (energy[active] / 10.0)))) if active.any() else -120.0
    voiced = f0[f0 > 0]
    return level, float(np.median(voiced)) if voiced.size else 0.0


@dataclasses.dataclass
class Verdict:
    background: bool
    reason: str


class NearEndProfile:
    """One per call. Learns the near-end speaker from turns that were ACCEPTED;
    a rejected background utterance never contaminates it."""

    def __init__(self):
        self.level_dbfs: float | None = None
        self.f0_hz: float | None = None
        self.accepted = 0
        self.rejected = 0

    @property
    def established(self) -> bool:
        return self.accepted >= 1 and self.level_dbfs is not None

    def judge(self, level_dbfs: float, f0_hz: float) -> Verdict:
        if not self.established:
            return Verdict(False, "no_profile_yet")
        quieter = self.level_dbfs - level_dbfs
        if quieter < BACKGROUND_DELTA_DB:
            return Verdict(False, "as_loud_as_the_caller")
        if f0_hz <= 0 or not self.f0_hz:
            return Verdict(False, "no_pitch_to_compare")  # unvoiced: too quiet, not a different person
        ratio = max(f0_hz, self.f0_hz) / min(f0_hz, self.f0_hz)
        if ratio <= DIFFERENT_VOICE_F0_RATIO:
            return Verdict(False, "same_voice_but_quieter")  # the caller moved away: re-ask, do not ignore
        return Verdict(True, "quieter_and_a_different_voice")

    def accept(self, level_dbfs: float, f0_hz: float) -> None:
        a = PROFILE_ALPHA
        self.level_dbfs = level_dbfs if self.level_dbfs is None else (1 - a) * self.level_dbfs + a * level_dbfs
        if f0_hz > 0:
            self.f0_hz = (
                f0_hz if self.f0_hz is None else float(np.exp((1 - a) * np.log(self.f0_hz) + a * np.log(f0_hz)))
            )
        self.accepted += 1

    def reject(self) -> None:
        self.rejected += 1

    def consider(self, samples: np.ndarray, sr: int = 16000) -> Verdict:
        """One call: judge the utterance and, if it is the caller, learn from it."""
        level, f0 = level_and_pitch(samples, sr)
        verdict = self.judge(level, f0)
        if verdict.background:
            self.reject()
        else:
            self.accept(level, f0)
        return verdict
