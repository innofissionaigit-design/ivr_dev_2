"""Is this caller audible and intelligible -- and if not, why?

Three jobs, all numpy-only and cheap enough to run inside a per-turn time
budget (agent/detector_budget.py):

1. assess()   what is wrong with this utterance: too quiet, noisy, two
              people talking at once (cross-talk), unvoiced mumbling /
              whisper, clipped, too short. Each with a measured number,
              not just a label.
2. enhance()  filter what CAN be filtered: rumble and DC, stationary and
              slowly-varying background noise (Wiener-style spectral
              gating with a noise profile taken from the utterance's own
              quietest frames), and low level (gain to a target RMS).
3. transcript_problem()  the text-side half of "jumbled": a transcript that
              is empty, one stray fragment for seconds of audio, repeated
              or shattered tokens, or in the wrong script.

WHAT THIS CANNOT DO, STATED PLAINLY
-----------------------------------
* It cannot separate two talkers. Single-channel source separation needs a
  trained model. Cross-talk is DETECTED (a second, non-harmonic pitch
  track alive inside the voiced frames) so the conversation can react --
  ask the caller to move somewhere quieter -- not silently "fixed".
* It cannot judge intelligibility from audio alone. A mumble is a
  perceptual property; the acoustic proxies here (voicing ratio,
  harmonicity, SNR) correlate with it, and the transcript-side checks and
  the ASR's own decoder agreement carry the rest. agent/reask_policy.py
  combines all three.
* Every threshold below is REASONED from how telephone speech behaves and
  validated on synthetic signals (tests/test_audio_quality.py), NOT
  calibrated on real Kolkata call audio -- there is none locally. Same
  "measured vs reasoned" rule as agent/confidence_gate.py: recalibrate
  against real calls before trusting these past the pilot.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import numpy as np

FRAME_S = 0.040
HOP_S = 0.010
F0_MIN_HZ, F0_MAX_HZ = 70.0, 400.0

# ---- thresholds (REASONED, see module docstring) ----
MIN_SPEECH_DURATION_S = 0.6
QUIET_RMS_DBFS = -38.0  # active-speech RMS below this is "low audible"
QUIET_PEAK_DBFS = -22.0
NOISY_SNR_DB = 10.0
VERY_NOISY_SNR_DB = 5.0
CLIPPING_FRACTION = 0.01
VOICED_HARMONICITY = 0.45  # normalised autocorrelation peak for a voiced frame
MUMBLE_VOICED_RATIO = 0.20  # of active frames; speech is normally 0.4-0.7
TWO_PITCH_RESIDUAL_FRACTION = 0.10
TWO_PITCH_RESIDUAL_HARMONICITY = 0.50
# Of voiced frames carrying a second pitch track. MEASURED on synthetic
# signals (108 single voices across f0 90-250 Hz, 1-5% pitch drift, clean
# and with noise at 12 and 4 dB SNR: max 0.012; 36 two-talker mixes across
# f0 pairs and 0.4-1.0 level ratios: min 0.130). The gap is wide, and the
# threshold sits inside it. Two talkers rarely overlap for the whole
# utterance -- syllables interleave -- which is why a fraction of ALL
# voiced frames is small even for genuine cross-talk. Real handset audio
# will be messier: recalibrate before trusting this past the pilot.
CROSSTALK_RATIO = 0.05
CROSSTALK_MIN_FRAMES = 8  # 80 ms of overlapped speech, so one glitch cannot trip it
_LAG_EXCLUSION = 0.07  # a "second" pitch within 7% of k*T is the same voice


@dataclasses.dataclass
class AudioAssessment:
    duration_s: float
    active_ratio: float  # share of frames that carry speech-level energy
    speech_rms_dbfs: float
    peak_dbfs: float
    snr_db: float
    voiced_ratio: float  # of active frames
    two_pitch_ratio: float  # of voiced frames
    clipping_ratio: float
    issues: list[str]
    intelligibility: float  # 0..1, acoustic proxy only

    @property
    def primary_issue(self) -> str | None:
        return self.issues[0] if self.issues else None

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_log_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


# ------------------------------------------------------------------ framing


def _frames(x: np.ndarray, sr: int) -> np.ndarray:
    n, hop = int(FRAME_S * sr), int(HOP_S * sr)
    if x.size < n:
        return np.zeros((0, n), dtype=np.float32)
    count = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


def _db(power: Any) -> Any:
    """Power to decibels. A numpy scalar in gives a scalar out and an array in gives an array out, which a single
    static return type cannot say without a false "or a bool" that every comparison downstream then trips over."""
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def _autocorr(frames: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of every frame (FFT-based, zero-padded so
    the result is linear not circular). r[:, 0] == 1 for non-silent frames."""
    n = frames.shape[1]
    spec = np.fft.rfft(frames, n=2 * n, axis=1)
    r = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :n]
    r0 = np.maximum(r[:, :1], 1e-12)
    return r / r0


def _pitch_lag(r: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """(lag, harmonicity) per frame. The SMALLEST lag within 15% of the
    strongest peak wins, which avoids the classic octave error of locking
    onto a multiple of the true period."""
    lo, hi = int(sr / F0_MAX_HZ), min(int(sr / F0_MIN_HZ), r.shape[1] - 1)
    seg = r[:, lo : hi + 1]
    best = seg.max(axis=1)
    lags = np.zeros(r.shape[0], dtype=int)
    for i in range(r.shape[0]):
        cand = np.where(seg[i] >= 0.85 * best[i])[0]
        # a local maximum, not merely a high shoulder of the zero-lag lobe
        for c in cand:
            left = seg[i, c - 1] if c > 0 else -1
            right = seg[i, c + 1] if c + 1 < seg.shape[1] else -1
            if seg[i, c] >= left and seg[i, c] >= right:
                lags[i] = lo + c
                break
        else:
            lags[i] = lo + int(np.argmax(seg[i]))
    return lags, best


# --------------------------------------------------------------- cross-talk


def _refine_lag(r: np.ndarray, lag: int) -> float:
    """Sub-sample period by parabolic interpolation of the autocorrelation
    peak. An integer lag leaves the dominant voice's HIGH harmonics out of
    phase (at 120 Hz / 16 kHz the period is 133.33 samples; rounding to 133
    misaligns harmonic 25 by ~0.4 rad), so its cancellation is imperfect and
    the leftover masquerades as -- or drowns -- a second talker."""
    if lag <= 0 or lag + 1 >= r.size:
        return float(lag)
    a, b, c = r[lag - 1], r[lag], r[lag + 1]
    denom = a - 2.0 * b + c
    return float(lag) + (0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0)


def _second_pitch(frame: np.ndarray, lag: float, sr: int) -> bool:
    """Is there a second periodic source in this voiced frame?

    Differencing at the dominant period, e[n] = x[n] - x[n+T], cancels the
    dominant voice and every harmonic of its f0. What survives is noise
    (aperiodic) or a SECOND voice (periodic at a different, non-harmonic
    period). So: enough residual energy, and that residual is itself
    strongly periodic at some lag that is NOT the dominant period, a
    multiple of it, or a sub-multiple of it. T is fractional and the
    shifted copy is linearly interpolated, for the reason in _refine_lag."""
    whole = int(np.floor(lag))
    frac = lag - whole
    m = frame.size - whole - 1
    if lag <= 1 or m < 2 * int(sr / F0_MAX_HZ):
        return False
    shifted = (1.0 - frac) * frame[whole : whole + m] + frac * frame[whole + 1 : whole + 1 + m]
    e = frame[:m] - shifted
    ex, ee = float(np.dot(frame[:m], frame[:m])), float(np.dot(e, e))
    if ex < 1e-9 or ee / (2.0 * ex) < TWO_PITCH_RESIDUAL_FRACTION:
        return False
    er = _autocorr(e[None, :])[0]
    lo, hi = int(sr / F0_MAX_HZ), min(int(sr / F0_MIN_HZ), er.size - 1)
    if hi <= lo:
        return False
    lags = np.arange(lo, hi + 1)
    allowed = np.ones(lags.size, dtype=bool)
    for k in (1, 2, 3, 4):
        allowed &= np.abs(lags - k * lag) > _LAG_EXCLUSION * k * lag  # multiples of T
        allowed &= np.abs(lags - lag / k) > _LAG_EXCLUSION * lag / k  # sub-multiples of T
    if not allowed.any():
        return False
    return float(er[lo : hi + 1][allowed].max()) >= TWO_PITCH_RESIDUAL_HARMONICITY


# ------------------------------------------------------------------- assess


def assess(samples: np.ndarray, sr: int = 16000) -> AudioAssessment:
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    duration = x.size / sr if sr else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    clipping = float(np.mean(np.abs(x) >= 0.99)) if x.size else 0.0

    frames = _frames(x - (float(np.mean(x)) if x.size else 0.0), sr)
    if frames.shape[0] < 5:
        return AudioAssessment(duration, 0.0, -120.0, float(_db(peak**2)), 0.0, 0.0, 0.0, clipping, ["too_short"], 0.0)

    energy_db = _db(np.mean(frames**2, axis=1))
    floor, top = float(np.percentile(energy_db, 10)), float(np.percentile(energy_db, 90))
    snr = top - floor
    active = energy_db >= max(floor + 0.35 * (top - floor), top - 25.0)
    active_ratio = float(np.mean(active))
    speech_rms = float(_db(np.mean(frames[active] ** 2))) if active.any() else -120.0

    voiced_ratio = two_pitch = 0.0
    two_pitch_hits = 0
    if active.any():
        act = frames[active]
        acf = _autocorr(act)
        lags, harm = _pitch_lag(acf, sr)
        voiced = harm >= VOICED_HARMONICITY
        voiced_ratio = float(np.mean(voiced))
        if voiced.any():
            hits = sum(_second_pitch(act[i], _refine_lag(acf[i], int(lags[i])), sr) for i in np.where(voiced)[0])
            two_pitch, two_pitch_hits = hits / float(voiced.sum()), hits

    issues: list[str] = []
    if speech_rms < -100.0 or active_ratio * duration < MIN_SPEECH_DURATION_S:
        issues.append("no_speech" if speech_rms < -100.0 else "too_short")
    else:
        if speech_rms < QUIET_RMS_DBFS or _db(peak**2) < QUIET_PEAK_DBFS:
            issues.append("too_quiet")
        if two_pitch >= CROSSTALK_RATIO and two_pitch_hits >= CROSSTALK_MIN_FRAMES:
            issues.append("crosstalk")
        if snr < NOISY_SNR_DB:
            issues.append("noisy")
        if voiced_ratio < MUMBLE_VOICED_RATIO:
            issues.append("unvoiced_mumble")
        if clipping >= CLIPPING_FRACTION:
            issues.append("clipped")

    # A single 0..1 proxy: each factor is a soft penalty, never a veto, so
    # one mediocre number does not by itself condemn an utterance.
    def soft(v, lo, hi):
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    intelligibility = (
        soft(snr, 0.0, 20.0) ** 0.5
        * soft(speech_rms, -50.0, -30.0) ** 0.5
        * (0.4 + 0.6 * soft(voiced_ratio, 0.0, 0.45))
        * (1.0 - 0.6 * min(two_pitch, 1.0))
    )
    return AudioAssessment(
        duration,
        active_ratio,
        speech_rms,
        float(_db(peak**2)),
        snr,
        voiced_ratio,
        two_pitch,
        clipping,
        issues,
        float(np.clip(intelligibility, 0.0, 1.0)),
    )


# ------------------------------------------------------------------ enhance


def enhance(
    samples: np.ndarray,
    sr: int = 16000,
    target_rms_dbfs: float = -24.0,
    max_gain_db: float = 24.0,
    floor_gain: float = 0.12,
) -> np.ndarray:
    """Rumble/DC removal, Wiener-style spectral gating, then gain.

    The noise power spectrum is the mean of the quietest 15% of the
    utterance's own frames -- a profile taken from THIS call's line, not a
    generic one, so it tracks whatever hum or hiss this handset has. The
    gate uses a decision-directed a-priori SNR (smoothed across time) and a
    gain floor of `floor_gain` (-18 dB): a floor is what stops the "musical
    noise" artefacts hard spectral subtraction leaves, which an ASR reads
    as phantom phonemes. Speech that is already clean passes through
    essentially unchanged (gain ~1 where SNR is high)."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if x.size < int(0.2 * sr):
        return x.copy()
    x = x - float(np.mean(x))
    # zero-phase 4th-order Butterworth-magnitude high-pass, ~70 Hz, applied
    # in the frequency domain: removes rumble and handling thump without
    # the phase smear (or a per-sample Python loop) a time-domain IIR costs.
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    spectrum *= 1.0 / np.sqrt(1.0 + (70.0 / np.maximum(freqs, 1e-3)) ** 8)
    x = np.fft.irfft(spectrum, n=x.size).astype(np.float32)

    n_fft, hop = 512, 128
    win = np.hanning(n_fft).astype(np.float32)
    pad = n_fft
    xp = np.concatenate([np.zeros(pad, np.float32), x, np.zeros(pad, np.float32)])
    count = 1 + (xp.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    spec = np.fft.rfft(xp[idx] * win, axis=1)
    power = np.abs(spec) ** 2

    frame_energy = power.sum(axis=1)
    quiet = frame_energy <= np.percentile(frame_energy, 15)
    noise_psd = power[quiet].mean(axis=0) + 1e-12

    gain = np.ones_like(power)
    prev_clean = power[0]
    alpha = 0.92
    for t in range(count):
        post = power[t] / noise_psd
        prior = alpha * (prev_clean / noise_psd) + (1 - alpha) * np.maximum(post - 1.0, 0.0)
        g = np.maximum(prior / (1.0 + prior), floor_gain)
        gain[t] = g
        prev_clean = (g**2) * power[t]
    out_spec = spec * gain
    frames_out = np.fft.irfft(out_spec, n=n_fft, axis=1) * win
    out: np.ndarray = np.zeros(xp.size, dtype=np.float32)
    norm = np.zeros(xp.size, dtype=np.float32)
    for t in range(count):
        s = t * hop
        out[s : s + n_fft] += frames_out[t]
        norm[s : s + n_fft] += win**2
    out = (out / np.maximum(norm, 1e-6))[pad : pad + x.size]

    # gain to a target level, only ever UP to max_gain_db, never into clipping
    frames = _frames(out, sr)
    if frames.shape[0]:
        e = _db(np.mean(frames**2, axis=1))
        act = e >= max(
            float(np.percentile(e, 10)) + 0.35 * (float(np.percentile(e, 90)) - float(np.percentile(e, 10))),
            float(np.percentile(e, 90)) - 25.0,
        )
        cur = float(_db(np.mean(frames[act] ** 2))) if act.any() else -120.0
        if -100.0 < cur < target_rms_dbfs:
            g = 10 ** (min(target_rms_dbfs - cur, max_gain_db) / 20.0)
            peak = float(np.max(np.abs(out))) or 1.0
            out = out * min(g, 0.97 / peak)
    return out.astype(np.float32)


# ------------------------------------------------------- transcript-side

_RE_TOKEN = re.compile(r"\S+")
_SCRIPT = {
    "bn": re.compile(r"[ঀ-৿]"),
    "hi": re.compile(r"[ऀ-ॿ]"),
    "en": re.compile(r"[A-Za-z]"),
}


def transcript_problem(
    text: str,
    lang: str,
    audio_duration_s: float | None = None,
    decoder_agreement: float | None = None,
    slot_answer: bool = False,
) -> str | None:
    """Why this transcript looks jumbled, or None if it looks like speech.

    Order matters: the cheap, unambiguous signals first. A problem here
    does NOT mean the caller did anything wrong -- it means the turn is not
    safe to act on, so the right response is to ask again, kindly.

    `slot_answer` is True while a booking is collecting a slot or awaiting a
    confirmation. There a digit-by-digit phone number ("9 8 7 6 ..."), a name
    spelled letter by letter, a repeated digit run ("5 5 5 5") and a bare "না"
    over a long, slow clip are all LEGITIMATE, and the token-shape checks
    would call each of them jumbled. Only the decoder-disagreement signal --
    which does not depend on token shape -- still applies."""
    tokens = _RE_TOKEN.findall((text or "").strip())
    if slot_answer:
        if tokens and decoder_agreement is not None and 0.0 < decoder_agreement < 0.25:
            return "decoders_disagree"
        return "empty" if not tokens else None
    if not tokens:
        return "empty"
    if audio_duration_s and audio_duration_s >= 2.5 and len(tokens) == 1 and len(tokens[0]) <= 3:
        return "fragment"  # seconds of audio, one stray syllable
    script = _SCRIPT.get(lang)
    letters = [c for c in text if c.isalpha()]
    if script and letters and len(letters) >= 4:
        if sum(1 for c in letters if script.match(c)) / len(letters) < 0.4:
            return "wrong_script"
    if len(tokens) >= 5:
        if sum(1 for t in tokens if len(t) == 1) / len(tokens) >= 0.5:
            return "shattered"  # a run of single characters
        lowered = [t.lower() for t in tokens]
        if len(set(lowered)) / len(lowered) <= 0.4:
            return "repetitive"  # "the the the ... "
    if decoder_agreement is not None and 0.0 < decoder_agreement < 0.25:
        return "decoders_disagree"
    return None
