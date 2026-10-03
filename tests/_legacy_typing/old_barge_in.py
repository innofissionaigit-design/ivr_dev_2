"""KCD-052: barge-in -- the caller talks over the agent and is heard.

Runs on the ECHO-CANCELLED microphone signal (agent/echo_cancel.py), so what
it sees while the agent is speaking is the caller plus whatever the canceller
could not remove. It decides one thing: is a person speaking to us right now?

Four tests, all of which must pass for a frame to count as evidence:

  1. LEVEL     -- the cleaned frame is well above the line's noise floor
                  (a running minimum that creeps up, so a noisy street raises
                  the bar instead of tripping it) and above -50 dBFS.
  2. NOT ECHO  -- the cleaned frame is not more than 6 dB below the canceller's
                  own echo estimate. Residual echo of an imperfect filter is
                  bounded by the ERLE achieved; a caller is not.
  3. NOT ECHO-SHAPED -- a frame that is not clearly louder than the echo
                  estimate AND has its spectral shape (cosine similarity of
                  100-3500 Hz magnitude spectra >= 0.70) is residual echo: same
                  pitch, same harmonics. A caller's voice does not share them.
  4. VOICED    -- a clear pitch (harmonicity), which separates speech from a
                  door slam, a click, or broadband street noise.

Evidence ACCUMULATES: +1 per hit (2.0 for a frame clearly LOUDER than the echo estimate,
which cannot be residual echo), -0.4 per miss. Speech is syllables with gaps, so a strict
"N frames in a row" rule missed callers with deep gaps, and noise that hits now and then
never accumulates. `min_speech_s` (0.11 s) of net evidence triggers. A frame that is one
narrow spectral line (a beep, a pure horn) is not a voice and counts for nothing.

MEASURED by tools/echo_eval.py --synthetic 200 on same-voice synthetic scenes (the deployed
condition: the agent speaks in one TTS voice per language), 100 echo-only calls and 100
calls with a caller talking over the agent:
    own-audio triggers   0 in 100 calls         (the story: never over 200 calls)
    callers detected     100 of 100
    detection latency    median 140 ms, p95 170 ms, max 390 ms   (budget: p95 <= 300 ms)
    ERLE, last 3 s       median 27.9 dB, minimum 20.4 dB
With the agent's voice changing every 4-6 s (the worst case; in deployment it changes
only when the caller switches language) barge-in is held off for two seconds after each
change (agent/full_duplex.py) because the canceller has to relearn; see that file.

KNOWN LIMITS, measured and not hidden: a loud horn over traffic rumble looks like a voice
to any acoustic detector -- 4.8 false events per minute of agent speech on the synthetic
street, whose horns come every ~3 s (far more often than a real street); clinic, fan and
market noise, and echo alone, gave 0. Background VOICES cannot be told from a caller by
acoustics at all. A false barge-in stops the agent's reply early; the caller can always ask
again. The network hop to the client is outside this module and is measured on the pod.

Every threshold is REASONED and swept on SYNTHETIC audio; recalibrate on recorded handset
calls before trusting them (KCD-407 measures speakerphone separately). agent/full_duplex.py
additionally refuses to act on an event until the canceller has measurably converged, which
is what keeps the agent from triggering on its own audio.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from agent.audio_quality import VOICED_HARMONICITY, _autocorr, _pitch_lag

FRAME_S = 0.040
HOP_S = 0.010


@dataclasses.dataclass
class BargeInConfig:
    min_speech_s: float = 0.11  # evidence (in hops) needed; see the grid in test_barge_in (REASONED)
    above_floor_db: float = 9.0  # over the tracked noise floor (REASONED)
    above_echo_db: float = (
        -6.0
    )  # cleaned power vs echo estimate; below this => residual echo (swept -13..+3 on synthetic scenes)
    voiced_harmonicity: float = VOICED_HARMONICITY
    miss_penalty: float = 0.4  # evidence lost per non-speech hop (REASONED)
    min_floor_dbfs: float = -70.0  # never treat digital silence as the floor
    absolute_floor_dbfs: float = -50.0  # nothing quieter than this is a caller
    # MEASURED on synthetic scenes (echo-only frames: median similarity 0.89, 25th
    # percentile 0.75; caller frames: median 0.54, 75th percentile 0.72): a frame
    # this similar to the echo estimate and not clearly louder than it is echo.
    echo_like_similarity: float = 0.70
    echo_like_below_db: float = 6.0
    # A frame clearly LOUDER than the echo estimate cannot be residual echo, so it is
    # worth more evidence: this is what lets a real caller be heard sooner without
    # letting residual echo (which sits at or below the estimate) in faster.
    strong_above_echo_db: float = 6.0
    strong_weight: float = 2.0
    # A frame with this share of its energy within +-2 bins of one spectral line is a tone,
    # not speech (REASONED; the synthetic horn is >0.95, synthetic voices measured below 0.7).
    pure_tone_fraction: float = 0.85


@dataclasses.dataclass
class BargeInEvent:
    at_sample: int  # where the evidence began
    detected_at_sample: int  # where it was enough
    latency_ms: float


class BargeInDetector:
    """Feed cleaned audio with the matching echo estimate; get an event or None."""

    def __init__(self, sr: int = 16000, config: BargeInConfig | None = None):
        self.sr = sr
        self.cfg = config or BargeInConfig()
        self._frame, self._hop = int(FRAME_S * sr), int(HOP_S * sr)
        self._buf = np.zeros(0, np.float32)
        self._echo = np.zeros(0, np.float32)
        self._pos = 0  # absolute index of _buf[0]
        self._floor_db = None
        self._streak_start: int | None = None
        self._score = 0.0
        self.frames = self.evidence_frames = 0

    def reset(self) -> None:
        """Call when the agent starts a new utterance: evidence from the
        previous one must not carry over."""
        self._streak_start, self._score = None, 0.0

    def _frame_hit(self, cleaned: np.ndarray, echo: np.ndarray) -> float:
        """Evidence weight of this frame: 0 (not a caller), 1 (a hit), or strong_weight."""
        p = float(np.mean(cleaned**2)) + 1e-12
        db = 10 * np.log10(p)
        # Noise floor: a running minimum that creeps upward (5 dB/s), so a
        # line that gets noisier raises the bar instead of tripping it.
        floor = db if self._floor_db is None else min(db, self._floor_db + 0.05)
        self._floor_db = max(floor, self.cfg.min_floor_dbfs)
        if db < self.cfg.absolute_floor_dbfs or db < self._floor_db + self.cfg.above_floor_db:
            return 0.0
        pe = float(np.mean(echo**2)) + 1e-12
        rel = 10 * np.log10(p / pe) if pe > 1e-8 else float("inf")
        if rel < self.cfg.above_echo_db:
            return 0.0  # quieter than any plausible residual
        # Residual echo has the SPECTRAL SHAPE of the echo estimate (same pitch,
        # same harmonics); a caller's voice does not. So a frame that looks like
        # the echo and is not clearly louder than it is residual.
        if (
            rel < self.cfg.echo_like_below_db
            and self._spectral_similarity(cleaned, echo) >= self.cfg.echo_like_similarity
        ):
            return 0.0
        acf = _autocorr(cleaned[None, :].astype(np.float32))
        _, harm = _pitch_lag(acf, self.sr)
        if float(harm[0]) < self.cfg.voiced_harmonicity:
            return 0.0
        if self._pure_tone(cleaned):
            return 0.0  # a horn or beep is periodic but it is not a voice
        return self.cfg.strong_weight if rel >= self.cfg.strong_above_echo_db else 1.0

    def _pure_tone(self, frame: np.ndarray) -> bool:
        """Almost all of the frame's energy in one narrow spectral peak. Speech puts its
        energy across a harmonic series shaped by formants; a horn, a beep or a whistle
        puts it in one line. Periodic, so the voicing test alone cannot tell them apart."""
        spec = np.abs(np.fft.rfft(frame * np.hanning(frame.size), 1024)) ** 2
        k = int(np.argmax(spec))
        peak = float(spec[max(0, k - 2) : k + 3].sum())
        return peak / (float(spec.sum()) + 1e-12) >= self.cfg.pure_tone_fraction

    def _spectral_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        win = np.hanning(a.size)
        A = np.abs(np.fft.rfft(a * win, 1024))[6:224]  # ~100-3500 Hz at 16 kHz
        B = np.abs(np.fft.rfft(b * win, 1024))[6:224]
        return float(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))

    def process(self, cleaned: np.ndarray, echo_estimate: np.ndarray | None = None) -> BargeInEvent | None:
        cleaned = np.asarray(cleaned, np.float32)
        echo = np.zeros_like(cleaned) if echo_estimate is None else np.asarray(echo_estimate, np.float32)
        self._buf = np.concatenate([self._buf, cleaned])
        self._echo = np.concatenate([self._echo, echo])
        need = self.cfg.min_speech_s / HOP_S
        while self._buf.size >= self._frame:
            weight = self._frame_hit(self._buf[: self._frame], self._echo[: self._frame])
            hit = weight > 0.0
            self.frames += 1
            event = None
            # Evidence accumulates on a hit and drains, more slowly, on a miss:
            # speech is syllables with gaps, so a strict "N in a row" rule misses a
            # real caller whose gaps are deep, while noise that hits now and then
            # never accumulates.
            if hit:
                self.evidence_frames += 1
                self._score += weight
                if self._streak_start is None:
                    self._streak_start = self._pos
            else:
                self._score = max(0.0, self._score - self.cfg.miss_penalty)
                if self._score == 0.0:
                    self._streak_start = None
            if self._score >= need:
                end = self._pos + self._frame
                event = BargeInEvent(
                    self._streak_start if self._streak_start is not None else self._pos,
                    end,
                    (end - (self._streak_start or self._pos)) / self.sr * 1000.0,
                )
            self._buf, self._echo = self._buf[self._hop :], self._echo[self._hop :]
            self._pos += self._hop
            if event is not None:
                self._score, self._streak_start = 0.0, None
                return event
        return None
