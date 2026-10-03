"""KCD-084: recognise an older caller from how they sound (and what they say).

Blueprint 4.5: "Detectors live in Call Intelligence, outside the LLM:
age-related speech characteristics ... Acoustic features + lexical cues,
each with a confidence." This is that detector. It writes the `senior`
flag on CallState; agent/speech_policy.py then derives everything else
(slower rate, shorter sentences, one question at a time, echoing each
value) from Appendix C's table.

WHAT IS MEASURED
----------------
Age shows up in speech in ways that are individually weak and jointly
useful. Fundamental frequency is deliberately NOT one of them -- it moves
opposite ways for older men (up) and older women (down), so on its own it
identifies nothing. What is measured instead:

  syllable_rate     slower articulation
  pause_ratio       more, longer pauses inside the utterance
  f0_instability    cycle-to-cycle instability of the pitch track
                    (std of the detrended f0 in semitones)
  tremor_st         RMS pitch variation (semitones) in the 4-12 Hz band,
                    the physiological-tremor range
  harmonicity       lower periodicity / breathier voice

HOW SURE IT CAN BE -- read this before trusting a number
--------------------------------------------------------
There is no labelled age corpus in this repository, so the weights and
thresholds are REASONED from the speech-ageing literature and checked on
synthetic voices whose parameters are controlled (tests/test_senior_voice.py),
not fitted to Kolkata callers. Three things follow, on purpose:

  1. The cost is asymmetric, so the design is asymmetric. A false positive
     gives a younger caller a slower, more-confirming conversation -- mildly
     annoying. A false negative leaves an older caller with today's normal
     service -- no worse than before. So the thresholds lean conservative
     and a decision needs EVIDENCE OVER SEVERAL TURNS, not one clip.
  2. The acoustic estimate is one vote. An explicit request ("please speak
     slowly", "I am an old man") or a stated age of 60+ decides on its own,
     and is far stronger evidence than any of the above. The story says the
     mode "triggers on detection or request" -- both paths exist.
  3. Recalibrate on real, consented, labelled calls before trusting the
     acoustic path. Until then treat the estimate as a hint that shapes
     delivery, never as a fact about a person: nothing in the system states
     or stores an inferred age.

Persistence against the patient (KCD-084's last sentence) is the
clinic-api's job -- see clinic-api/booking_service.set_patient_senior; only
the boolean "delivery mode" is stored, never the score or any audio.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata

import numpy as np

from agent.audio_quality import (
    F0_MAX_HZ,
    F0_MIN_HZ,
    HOP_S,
    VOICED_HARMONICITY,
    _autocorr,
    _db,
    _frames,
    _pitch_lag,
    _refine_lag,
)

MIN_VOICED_SECONDS = 2.0  # below this there is not enough to judge
TREMOR_BAND_HZ = (4.0, 12.0)
MAX_GAP_S = 0.25  # unvoiced gap the pitch track is bridged across
MIN_TREMOR_SPAN_S = 1.0  # continuous stretch needed to resolve a 4-12 Hz modulation
PAUSE_MIN_S = 0.20  # a silence shorter than this is a syllable dip, not a pause
FRAME_RATE_HZ = 1.0 / HOP_S

# (feature, direction, low, high, weight): each feature is mapped to a
# 0..1 "looks older" score by a linear ramp between `low` (0.0) and `high`
# (1.0). REASONED starting points, see module docstring.
_FEATURES = (
    ("syllable_rate", "lower", 2.0, 4.0, 0.30),  # syll/s: <=2.0 slow, >=4.0 brisk
    ("pause_ratio", "higher", 0.15, 0.40, 0.15),
    ("f0_instability", "higher", 0.30, 0.90, 0.20),  # semitones
    ("tremor_st", "higher", 0.05, 0.30, 0.20),  # RMS semitones in the 4-12 Hz band
    ("harmonicity", "lower", 0.60, 0.85, 0.15),
)

# Per-clip score at or above which a clip votes "older". Placed from
# measurement on synthetic voices (tests/_synth_speech.py, 5 seeds x 4 f0
# values): brisk steady speech scores 0.01-0.08, speech with every older-
# voice trait 0.54-0.64, and any SINGLE trait alone no more than 0.30. 0.40
# sits in the gap and above every single-trait score, so one quirk (a slow
# talker, a hoarse voice) never flags anyone on its own.
SENIOR_SCORE = 0.40
DECISION_CLIPS = 2  # clips with enough speech that must agree
EVIDENCE_DECAY = 0.6  # weight of history vs the newest clip


@dataclasses.dataclass
class SeniorEstimate:
    sufficient: bool  # enough voiced speech to judge at all
    score: float  # 0..1, how much this clip sounds older
    features: dict
    contributions: dict

    @property
    def votes_senior(self) -> bool:
        return self.sufficient and self.score >= SENIOR_SCORE


def _ramp(value: float, low: float, high: float, direction: str) -> float:
    t = (value - low) / (high - low)
    t = min(max(t, 0.0), 1.0)
    return 1.0 - t if direction == "lower" else t


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best = (0, 0)
    start = None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def extract_features(samples: np.ndarray, sr: int = 16000) -> dict | None:
    """The five measurements, or None if there is too little voiced speech."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    frames = _frames(x - (float(x.mean()) if x.size else 0.0), sr)
    if frames.shape[0] < 50:
        return None

    energy = _db(np.mean(frames**2, axis=1))
    floor, top = float(np.percentile(energy, 10)), float(np.percentile(energy, 90))
    active = energy >= max(floor + 0.35 * (top - floor), top - 25.0)
    if active.sum() * HOP_S < MIN_VOICED_SECONDS * 0.5:
        return None

    # ---- pitch track over ALL frames (unvoiced -> NaN), keeping time order
    acf = _autocorr(frames)
    lags, harm = _pitch_lag(acf, sr)
    voiced = active & (harm >= VOICED_HARMONICITY)
    if voiced.sum() * HOP_S < MIN_VOICED_SECONDS:
        return None
    f0 = np.full(frames.shape[0], np.nan)
    for i in np.where(voiced)[0]:
        f0[i] = sr / _refine_lag(acf[i], int(lags[i]))
    f0[(f0 < F0_MIN_HZ) | (f0 > F0_MAX_HZ)] = np.nan
    semitones = 12.0 * np.log2(f0)

    # ---- f0 instability + tremor. Voiced runs in syllable-paced speech are
    # only ~100-300 ms, far too short to resolve a 4-12 Hz modulation, so
    # the pitch track is interpolated across unvoiced gaps of up to
    # MAX_GAP_S (a tremor does not stop for a consonant) and analysed over
    # the longest stretch that stays continuous that way. Instability is
    # measured on the voiced frames only, so interpolated values never
    # count as evidence.
    valid = ~np.isnan(semitones)
    idx = np.arange(semitones.size)
    filled = np.interp(idx, idx[valid], semitones[valid])
    gap_ok = np.ones(semitones.size, dtype=bool)
    run_start = None
    for i, v in enumerate(np.append(valid, True)):
        if not v and run_start is None:
            run_start = i
        elif v and run_start is not None:
            if (i - run_start) * HOP_S > MAX_GAP_S:
                gap_ok[run_start:i] = False
            run_start = None
    gap_ok &= (idx >= idx[valid][0]) & (idx <= idx[valid][-1])
    a, b = _longest_true_run(gap_ok)
    instability = tremor = 0.0
    if (b - a) * HOP_S >= MIN_TREMOR_SPAN_S:
        seg = filled[a:b]
        k = 31
        trend = np.convolve(np.pad(seg, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        resid = seg - trend
        instability = float(np.std(resid[valid[a:b]])) if valid[a:b].any() else 0.0
        # RMS, in semitones, of the pitch variation that lies in the tremor
        # band. An AMPLITUDE, not a share of the total: a steady young voice
        # has almost no variation at all, so "what fraction of almost
        # nothing is in the band" is large and meaningless -- the first
        # version of this feature read 0.4-0.8 on the steadiest synthetic
        # voices for exactly that reason.
        spec = np.fft.rfft(resid)
        freqs = np.fft.rfftfreq(resid.size, 1.0 / FRAME_RATE_HZ)
        spec[(freqs < TREMOR_BAND_HZ[0]) | (freqs > TREMOR_BAND_HZ[1])] = 0.0
        tremor = float(np.std(np.fft.irfft(spec, n=resid.size)))

    # ---- rate and pauses from the energy envelope
    env = np.convolve(np.pad(10 ** (energy / 20.0), (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
    span = np.where(active)[0]
    lo, hi = span[0], span[-1] + 1
    seg_env = env[lo:hi]
    peaks = (
        (seg_env[1:-1] > seg_env[:-2]) & (seg_env[1:-1] >= seg_env[2:]) & (seg_env[1:-1] > 0.5 * float(np.max(seg_env)))
    )
    peak_idx = np.where(peaks)[0]
    # peaks closer than 100 ms are one syllable
    kept = [p for j, p in enumerate(peak_idx) if j == 0 or (p - peak_idx[j - 1]) * HOP_S >= 0.10]
    duration_s = max((hi - lo) * HOP_S, 1e-3)
    syllable_rate = len(kept) / duration_s
    # Only silences of PAUSE_MIN_S or more are pauses: the dip between two
    # syllables is silence too, and counting it would rate brisk speech as
    # hesitant.
    pause_frames, run = 0, 0
    for v in np.append(~active[lo:hi], False):
        if v:
            run += 1
        else:
            if run * HOP_S >= PAUSE_MIN_S:
                pause_frames += run
            run = 0
    pause_ratio = pause_frames / float(max(hi - lo, 1))

    return {
        "syllable_rate": float(syllable_rate),
        "pause_ratio": pause_ratio,
        "f0_instability": instability,
        "tremor_st": tremor,
        "harmonicity": float(np.mean(harm[voiced])),
        "voiced_seconds": float(voiced.sum() * HOP_S),
    }


def estimate(samples: np.ndarray, sr: int = 16000) -> SeniorEstimate:
    feats = extract_features(samples, sr)
    if feats is None:
        return SeniorEstimate(False, 0.0, {}, {})
    contributions = {}
    score = 0.0
    for name, direction, low, high, weight in _FEATURES:
        c = weight * _ramp(feats[name], low, high, direction)
        contributions[name] = round(c, 3)
        score += c
    return SeniorEstimate(True, float(score), {k: round(v, 3) for k, v in feats.items()}, contributions)


# ------------------------------------------------------ evidence over a call


class SeniorEvidence:
    """One per call. A single clip never decides: the acoustic path needs
    DECISION_CLIPS separate clips (each with enough speech) that agree, and
    the evidence is smoothed so one odd clip cannot flip the state either
    way. An explicit cue decides at once and is sticky, and once senior the
    state is not silently cleared by later clips -- a caller who needed the
    slower pace a minute ago still does."""

    def __init__(self):
        self.senior = False
        self.reason: str | None = None
        self._ema = 0.0
        self._clips = 0
        self._agree = 0

    def note_explicit(self, reason: str) -> bool:
        changed = not self.senior
        self.senior, self.reason = True, reason
        return changed

    def add(self, est: SeniorEstimate) -> bool:
        """Fold in one clip. Returns True exactly when this clip flipped the
        call into senior mode."""
        if self.senior or not est.sufficient:
            return False
        self._clips += 1
        self._ema = est.score if self._clips == 1 else EVIDENCE_DECAY * self._ema + (1 - EVIDENCE_DECAY) * est.score
        self._agree += 1 if est.votes_senior else 0
        if self._agree >= DECISION_CLIPS and self._ema >= SENIOR_SCORE:
            self.senior, self.reason = True, "acoustic"
            return True
        return False

    @property
    def score(self) -> float:
        return self._ema


# ----------------------------------------------------- explicit / lexical cues

SENIOR_AGE = 60

_CUES = {
    "en": [
        r"\bspeak (more )?slowly\b",
        r"\bslowly please\b",
        r"\bplease (go|speak) slow",
        r"\bi(?:'m| am) (?:an? )?(?:old|elderly|senior)",
        r"\bmy hearing is (?:not|bad|weak)",
        r"\bi(?:'m| am) hard of hearing\b",
    ],
    "hi": [r"धीरे (?:से )?बोल", r"धीमे बोल", r"मैं (?:बूढ़ा|बूढ़ी|बुज़ुर्ग|बुजुर्ग|वरिष्ठ नागरिक)", r"मुझे कम सुनाई", r"मैं कम सुन(?:ता|ती)"],
    "bn": [
        r"ধীরে (?:ধীরে )?বল",
        r"আস্তে (?:আস্তে )?বল",
        r"আমি (?:বৃদ্ধ|বুড়ো|বুড়ি|বয়স্ক)",
        r"আমি প্রবীণ নাগরিক",
        r"আমি কম শুনি",
        r"আমার কম শুনতে",
    ],
    # Every cue that describes the SPEAKER is first-person on purpose: "my
    # father is a senior citizen" is a caller booking for someone else, and
    # slowing that caller down would be the wrong adaptation (the same rule
    # stated_age_is_senior applies to a stated age).
}
_REQUEST_MARKERS: tuple[str, ...] = ("slow", "धीरे", "धीमे", "ধীরে", "আস্তে")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# Bengali and Devanagari have several valid encodings of the same letter
# (the nukta letters -- য়, ज़ -- are in Unicode's composition-exclusion
# list, so NFC decomposes them). ASR output and this file's own patterns
# come from different sources, so BOTH sides are normalised the same way
# before matching or "আমি বয়স্ক" is silently not "আমি বয়স্ক".
_CUES = {lg: [_nfc(p) for p in pats] for lg, pats in _CUES.items()}
_REQUEST_MARKERS = tuple(_nfc(m) for m in _REQUEST_MARKERS)


def explicit_senior_cue(text: str, lang: str = "en") -> str | None:
    """A caller asking for a slower pace, or describing themselves as
    elderly, in any of the three languages (the caller may switch mid-call,
    so all three are always checked, `lang` only orders them)."""
    t = _nfc(text or "").lower()
    for lg in dict.fromkeys((lang, "en", "hi", "bn")):
        for pat in _CUES.get(lg, []):
            if re.search(pat, t):
                return "requested_slower" if any(m in pat for m in _REQUEST_MARKERS) else "self_described"
    return None


def stated_age_is_senior(age: int | None, relationship: str | None = None) -> bool:
    """A patient's own stated age of 60 or more. Only counts when the caller
    IS the patient (relationship self/None): a daughter booking for her
    father is not herself an older caller, and slowing her down would be
    the wrong adaptation."""
    try:
        if age is None or isinstance(age, bool) or int(age) < SENIOR_AGE:
            return False
    except (TypeError, ValueError):
        return False  # a non-numeric "age" is no evidence of anything
    return relationship in (None, "", "self")
