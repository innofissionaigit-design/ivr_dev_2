"""KCD-054: notice when a different person takes the phone after verification.

A patient verified themselves; then the handset is passed to someone else in the
household. Without a speaker signal the agent goes on disclosing to whoever is
now speaking. This module computes, per turn, HOW DIFFERENT the voice is from the
voice that was verified, and a decision the identity layer (agent/identity.py)
acts on: a change re-triggers verification before any further disclosure.

WHAT THE SIGNAL IS. Two cheap, numpy-only cues that survive a telephone channel
and need no trained model: the long-term average spectrum of the voiced frames
(vocal-tract length and timbre) with level removed, and the median pitch. They
are compared across turns of the SAME call, so the channel is constant and
cancels. A distance combines them.

WHAT IT IS NOT. It is not speaker verification and must never be used to
AUTHENTICATE anyone: it cannot say that this voice is the patient, only that this
voice differs from the one that was verified minutes ago. Family members can
sound alike; a sick or tired caller can sound unlike themselves. The two
failure directions are not symmetrical, and the policy reflects that: a MISSED
change means disclosure continues (mitigated by every disclosure being
gated on verification in the first place -- this is a second lock, not the only
one), a FALSE change costs the genuine caller one extra verification question.
So the threshold is set to keep false changes on single-speaker calls rare and
bounded, and the detector abstains ("insufficient") on a turn with too little
voiced speech rather than guess.

Thresholds are REASONED, chosen from a sweep on SYNTHETIC voices (tests/test_speaker_change.py,
tests/_synth_voices.py) whose speakers differ in pitch and vocal-tract length and
whose phrases differ in content. Recalibrate on recorded multi-speaker calls
(KCD-203 and the recording programme) before trusting the numbers on real callers.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from agent.audio_quality import VOICED_HARMONICITY, _autocorr, _frames, _pitch_lag, _refine_lag

N_BANDS = 24
BAND_EDGES_HZ = np.geomspace(150.0, 3800.0, N_BANDS + 1)
MIN_VOICED_S = 0.6  # less voiced speech than this and the turn abstains
FFT = 512

# Combination weights: 1 dB of mean band difference and 0.06 octave of median
# pitch difference count the same. REASONED.
SPEC_DB_SCALE = 1.0
PITCH_OCTAVE_SCALE = 0.06

# Decision thresholds on the combined distance. REASONED, chosen from a sweep on the synthetic
# population (tests/_synth_voices.py: 5 speakers, 20 same-speaker and 120 cross-speaker turns,
# varied content, level and noise). Same-speaker distances: median 2.3, p95 3.8, max 4.7.
# Different-speaker: min 4.6 (two near-identical synthetic men), p5 5.1, median 16. At 5.0 there
# were 0/100 false changes and 6/120 missed changes, all from that one close pair.
CHANGE_HIGH = 5.0  # one turn this far from the verified voice is a change
CHANGE_LOW = 3.8  # two consecutive turns this far are a change

# The same cues agent/near_end.py uses to tell the caller from the room -- pitch and level -- as a
# corroborating signal (KCD-054). A turn that is at least CHANGE_LOW from the verified voice AND ALSO
# sits at a clearly different pitch (PITCH_JUMP_OCTAVES, 0.32 = a 1.25x ratio: a different speaker, not
# a caller's own intonation) is a change at once, without waiting for a second turn; so is one that is
# at least CHANGE_LOW away, at a somewhat different pitch (0.15 octave), AND at a very different level
# (LEVEL_JUMP_DB: the handset went to someone at a different distance). A level shift ALONE is never a
# change -- a caller moves the phone -- and neither is a pitch shift alone: both need the spectral
# distance to agree. REASONED; measured on synthetic voices in tests/test_speaker_change.py.
PITCH_JUMP_OCTAVES = 0.32
PITCH_SOME_OCTAVES = 0.15
LEVEL_JUMP_DB = 8.0


@dataclasses.dataclass
class VoiceEmbedding:
    ltas_db: np.ndarray  # (N_BANDS,), level removed
    log2_f0: float
    voiced_s: float


def voice_embedding(samples: np.ndarray, sr: int = 16000) -> VoiceEmbedding | None:
    """None if the clip has too little voiced speech to describe a voice."""
    x = np.asarray(samples, dtype=np.float32)
    if x.size < int(0.3 * sr):
        return None
    frames = _frames(x - float(np.mean(x)), sr)
    if frames.shape[0] < 20:
        return None
    energy = np.mean(frames**2, axis=1)
    thr = max(float(np.percentile(energy, 90)) * 10 ** (-25 / 10), 1e-10)
    idx = np.where(energy >= thr)[0]
    if idx.size == 0:
        return None
    acf = _autocorr(frames[idx])
    lags, harm = _pitch_lag(acf, sr)
    voiced = harm >= VOICED_HARMONICITY
    if not voiced.any():
        return None
    vi = idx[voiced]
    f0 = np.array([sr / max(_refine_lag(acf[j], int(lags[j])), 1.0) for j in np.where(voiced)[0]])
    voiced_s = vi.size * 0.010
    if voiced_s < MIN_VOICED_S:
        return None
    win = np.hanning(frames.shape[1])
    spec = np.abs(np.fft.rfft(frames[vi] * win, FFT, axis=1)) ** 2
    freqs = np.fft.rfftfreq(FFT, 1.0 / sr)
    mean_spec = spec.mean(axis=0)
    bands = np.array(
        [mean_spec[(freqs >= lo) & (freqs < hi)].sum() + 1e-12 for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])]
    )
    db = 10 * np.log10(bands)
    return VoiceEmbedding(db - db.mean(), float(np.log2(np.median(f0))), voiced_s)


def distance(a: VoiceEmbedding, b: VoiceEmbedding) -> float:
    spec = float(np.mean(np.abs(a.ltas_db - b.ltas_db)))
    pitch = abs(a.log2_f0 - b.log2_f0)
    return spec / SPEC_DB_SCALE + pitch / PITCH_OCTAVE_SCALE


@dataclasses.dataclass
class ChangeResult:
    verdict: str  # "same" | "changed" | "insufficient" | "not_enrolled"
    distance: float | None = None


class SpeakerChangeDetector:
    """One per call. enroll() with the audio of the turn(s) at which identity was
    verified; observe() every later turn."""

    ENROL_TURNS = 3

    def __init__(self):
        self._ref: VoiceEmbedding | None = None
        self._enrol: list[VoiceEmbedding] = []
        self._levels: list[float] = []
        self._streak = 0
        self.turns_observed = 0
        self.changes = 0

    @property
    def enrolled(self) -> bool:
        return self._ref is not None

    def reset(self) -> None:
        """Forget the reference (called when verification is revoked or renewed)."""
        self._ref, self._enrol, self._levels, self._streak = None, [], [], 0

    def enroll(self, samples: np.ndarray, sr: int = 16000) -> bool:
        emb = voice_embedding(samples, sr)
        return False if emb is None else self.enroll_embedding(emb)

    def enroll_embedding(self, emb: VoiceEmbedding, level_dbfs: float | None = None) -> bool:
        """As enroll(), for an embedding already computed (the per-turn analysis
        computes it once and shares it). `level_dbfs` is the turn's active-speech level."""
        if level_dbfs is not None:
            self._levels = (self._levels + [level_dbfs])[-self.ENROL_TURNS :]
        self._enrol = (self._enrol + [emb])[-self.ENROL_TURNS :]
        self._ref = VoiceEmbedding(
            np.mean([e.ltas_db for e in self._enrol], axis=0),
            float(np.mean([e.log2_f0 for e in self._enrol])),
            sum(e.voiced_s for e in self._enrol),
        )
        return True

    def _corroborated(self, emb: VoiceEmbedding, level_dbfs: float | None) -> bool:
        """Pitch, and pitch with level, agree with the spectral distance that this is another voice."""
        octaves = abs(emb.log2_f0 - self._ref.log2_f0)
        if octaves >= PITCH_JUMP_OCTAVES:
            return True
        if level_dbfs is not None and self._levels and octaves >= PITCH_SOME_OCTAVES:
            return abs(level_dbfs - float(np.mean(self._levels))) >= LEVEL_JUMP_DB
        return False

    def observe(self, samples: np.ndarray, sr: int = 16000) -> ChangeResult:
        if self._ref is None:
            return ChangeResult("not_enrolled")
        emb = voice_embedding(samples, sr)
        if emb is None:
            return ChangeResult("insufficient")  # abstain: never guess a change from a cough
        return self.observe_embedding(emb)

    def observe_embedding(self, emb: VoiceEmbedding, level_dbfs: float | None = None) -> ChangeResult:
        if self._ref is None:
            return ChangeResult("not_enrolled")
        self.turns_observed += 1
        d = distance(emb, self._ref)
        if d >= CHANGE_HIGH:
            self._streak = 0
            self.changes += 1
            return ChangeResult("changed", d)
        if d >= CHANGE_LOW and self._corroborated(emb, level_dbfs):
            self._streak = 0
            self.changes += 1
            return ChangeResult("changed", d)
        if d >= CHANGE_LOW:
            self._streak += 1
            if self._streak >= 2:
                self._streak = 0
                self.changes += 1
                return ChangeResult("changed", d)
            return ChangeResult("same", d)
        self._streak = 0
        return ChangeResult("same", d)
