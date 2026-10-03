"""Listen to the caller, not the room (KCD-053, KCD-055).

Once the call has heard its caller for a turn or two, it knows two things about them: HOW LOUD they
speak into this handset and at what PITCH (agent/near_end.py's NearEndProfile). Everything in the
audio that is far below that level is not the caller, and this module turns the two facts into a
gain envelope, frame by frame (10 ms):

  * a BACKGROUND VOICE frame -- voiced, at least BACKGROUND_DELTA_DB below the caller's level, and at
    a pitch more than DIFFERENT_VOICE_F0_RATIO away from theirs -- is someone else in the room;
  * a NOISE frame -- not voiced (no pitch), at least NOISE_DELTA_DB below the caller's level and not
    within 80 ms of their voiced speech -- is the fan, the street, the hiss between phrases. This is the noise-suppression half (KCD-055): a gate
    keyed to the caller's OWN level rather than to a generic noise estimate, which is why it can
    be aggressive about noise without touching the caller's quiet consonants (those sit within a
    few dB of their voiced level, far above the gate);
  * everything else -- the caller, and any frame we cannot classify -- is left EXACTLY as it was.

Frames that qualify are ATTENUATED (ATTENUATION_DB), not deleted, and the gain moves smoothly (a short
ramp) so there are no clicks. Nothing is ever amplified.

GUARDS. It does nothing at all until the profile is established (the first utterance of a call is
passed through untouched). If more than MAX_FRACTION of an utterance would be attenuated, something is
off (the caller moved away, or the profile is stale) and the clip is returned unchanged. And the
frames it may touch are defined by exclusion, so speech at the caller's own level cannot be
attenuated by construction; tests/test_attention.py checks that on synthetic callers, bystanders and
noise, and reports the measured effect. Those numbers are synthetic; how much a recogniser benefits
on real audio is measured by tools/conditioning_eval.py on a pod.

Pure numpy.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from agent.audio_quality import HOP_S, VOICED_HARMONICITY, _autocorr, _db, _frames, _pitch_lag, _refine_lag
from agent.near_end import BACKGROUND_DELTA_DB, DIFFERENT_VOICE_F0_RATIO

NOISE_DELTA_DB = 20.0  # an unvoiced frame this far below the caller's level is noise
PROTECT_HOPS = 8  # ...unless it is within 80 ms of the caller's voiced speech (a consonant)
ATTENUATION_DB = 18.0
RAMP_S = 0.03
MAX_FRACTION = 0.6
MIN_FRAMES = 20


@dataclasses.dataclass
class AttentionReport:
    applied: bool
    reason: str
    frames: int = 0
    background: int = 0
    noise: int = 0

    @property
    def fraction(self) -> float:
        return (self.background + self.noise) / self.frames if self.frames else 0.0


def attend(
    samples: np.ndarray,
    level_dbfs: float | None,
    f0_hz: float | None,
    sr: int = 16000,
) -> tuple[np.ndarray, AttentionReport]:
    """(samples with the room attenuated, what was done). `level_dbfs` and `f0_hz` are the caller's
    profile (active-speech RMS in dBFS, median pitch); with None the clip is returned unchanged."""
    x = np.asarray(samples, dtype=np.float32)
    if level_dbfs is None or level_dbfs <= -100.0:
        return x, AttentionReport(False, "no_profile_yet")
    centred = x - float(np.mean(x)) if x.size else x
    frames = _frames(centred, sr)
    n = frames.shape[0]
    if n < MIN_FRAMES:
        return x, AttentionReport(False, "too_short", n)
    energy = _db(np.mean(frames**2, axis=1))  # per-frame level, dB
    # dBFS of a frame vs the profile's active-speech RMS (same dB scale: mean-square in dB)
    idx = np.where(energy >= level_dbfs - NOISE_DELTA_DB - 25.0)[0]  # frames worth pitch-tracking
    f0 = np.zeros(n)
    voiced = np.zeros(n, dtype=bool)
    if idx.size:
        acf = _autocorr(frames[idx])
        lags, harm = _pitch_lag(acf, sr)
        for j, i in enumerate(idx):
            if harm[j] >= VOICED_HARMONICITY:
                voiced[i] = True
                f0[i] = sr / max(float(_refine_lag(acf[j], int(lags[j]))), 1.0)
    quieter = level_dbfs - energy
    ratio = np.ones(n)
    if f0_hz and f0_hz > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(f0 > 0, np.maximum(f0, f0_hz) / np.minimum(np.maximum(f0, 1.0), f0_hz), 1.0)
    background = voiced & (quieter >= BACKGROUND_DELTA_DB) & (ratio > DIFFERENT_VOICE_F0_RATIO)
    # a weak consonant (an "s", an "f") is unvoiced and can sit well below the vowels around it, so
    # unvoiced frames within PROTECT_HOPS of the caller's own voiced speech are never treated as noise:
    # the gate works in the pauses between phrases, not inside words
    caller_voiced = voiced & ~background
    near_caller = np.convolve(caller_voiced.astype(float), np.ones(2 * PROTECT_HOPS + 1), mode="same") > 0
    noise = (~voiced) & (quieter >= NOISE_DELTA_DB) & ~near_caller
    cut = background | noise
    report = AttentionReport(True, "applied", n, int(background.sum()), int(noise.sum()))
    if not cut.any():
        return x, AttentionReport(False, "nothing_to_attenuate", n)
    if cut.mean() > MAX_FRACTION:
        return x, AttentionReport(False, "too_much_would_go", n, int(background.sum()), int(noise.sum()))
    # gain per hop. Only frames that are being cut change: their gain falls from 1.0 at the edge of
    # the cut to the full attenuation over RAMP_S, so there is no click; a frame that is not cut keeps
    # a gain of exactly 1.0.
    hop = int(round(HOP_S * sr))
    floor = 10 ** (-ATTENUATION_DB / 20.0)
    ramp = max(1, int(round(RAMP_S / HOP_S)))
    dist = np.zeros(n)
    run = 0
    for i in range(n):  # hops since the last frame that is kept
        run = run + 1 if cut[i] else 0
        dist[i] = run
    run = 0
    for i in range(n - 1, -1, -1):  # ... and until the next one
        run = run + 1 if cut[i] else 0
        dist[i] = min(dist[i], run) if cut[i] else 0
    gain_hops = np.where(cut, 1.0 - (1.0 - floor) * np.minimum(1.0, dist / ramp), 1.0)
    gain = np.repeat(gain_hops, hop)
    if gain.size < x.size:
        gain = np.concatenate([gain, np.full(x.size - gain.size, gain_hops[-1])])
    return (x * gain[: x.size]).astype(np.float32), report
