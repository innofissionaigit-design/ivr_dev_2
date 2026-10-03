"""KCD-051/052 glue: one object per call that turns "microphone in, agent audio
sent" into "cleaned microphone out, and a barge-in event when the caller talks
over the agent".

It composes agent/echo_cancel.EchoPipeline and agent/barge_in.BargeInDetector
and adds the two policies neither should own:

  * The agent's speech is placed on the microphone timeline WHERE THE CLIENT
    WILL PLAY IT: clips play back to back, so a clip starts at
    max(now, end of the previous clip). `place_reference()` does that
    arithmetic; the residual (network + output buffering) is what the
    canceller's delay estimator measures.
  * Barge-in is only ACTED ON while the agent is speaking and only once the
    canceller has converged. Before convergence the residual echo is as loud as
    a caller and cannot be told apart, so events are ignored rather than
    guessed; the detector is still fed, so its noise floor keeps tracking.

Nothing here touches a socket or a model: it is fed numpy arrays, so the whole
of it runs and is tested off-pod.
"""

from __future__ import annotations

import dataclasses
import io
import wave

import numpy as np

from agent.barge_in import BargeInConfig, BargeInDetector, BargeInEvent
from agent.echo_cancel import EchoPipeline

# ERLE (dB) the canceller must have reached before a barge-in event is trusted.
# MEASURED on synthetic scenes (12 callers over echo, 24 echo-only calls): with no
# trust rule, 44 false events in 21 of 24 echo-only calls (the residual echo of an
# unconverged filter is as loud as a caller); at >= 0 dB, 6 events in 5 calls;
# at >= 4 dB, none, with every caller still detected; at >= 8 dB one caller in
# twelve is missed. 5 dB keeps a margin above the zero-false-event point. Real
# handsets are untested -- tools/echo_eval.py measures this on recorded audio.
MIN_TRUSTED_ERLE_DB = 5.0
# The agent is treated as still audible this long after the last reference
# sample (playback buffer, room reverberation).
TAIL_S = 0.4
# After the agent's VOICE changes (a different TTS voice: the caller switched language) the
# filter has never seen the newly excited frequencies and cancels poorly for a few seconds,
# and the residual is indistinguishable from a caller. MEASURED on synthetic scenes: every
# remaining false trigger in 200 calls fell within 1.3 s of a voice change, and cancellation
# recovers within about 2 s. Barge-in is not acted on until then; the caller still has the
# manual interrupt.
VOICE_CHANGE_INHIBIT_S = 2.0


@dataclasses.dataclass
class DuplexResult:
    cleaned: np.ndarray
    barge_in: BargeInEvent | None  # only ever set when it is trusted and acted on
    suppressed_barge_in: bool  # an event fired but was ignored (agent silent / not converged)
    agent_speaking: bool


class FullDuplexProcessor:
    def __init__(
        self,
        sr: int = 16000,
        min_trusted_erle_db: float = MIN_TRUSTED_ERLE_DB,
        barge_config: BargeInConfig | None = None,
        **echo_kwargs,
    ):
        self.sr = sr
        self.min_trusted_erle_db = min_trusted_erle_db
        self.echo = EchoPipeline(sr=sr, **echo_kwargs)
        self.detector = BargeInDetector(sr=sr, config=barge_config)
        self._mic_pos = 0  # samples of microphone consumed so far
        self._ref_end = 0  # end of the last agent clip on the mic timeline
        self._voice: str | None = None
        self._inhibit_until = 0  # mic sample before which barge-in is not acted on
        self.barge_ins = 0
        self.suppressed = 0

    # -- agent side ---------------------------------------------------
    def place_reference(self, pcm: np.ndarray, voice: str | None = None) -> int:
        """Register a clip the agent is about to send. Returns where on the
        microphone timeline it is expected to start playing. `voice` identifies the
        TTS voice (e.g. the language): when it changes, barge-in is held off for
        VOICE_CHANGE_INHIBIT_S from the moment the new voice starts."""
        start = max(self._mic_pos, self._ref_end)
        if voice is not None:
            if self._voice is not None and voice != self._voice:
                self._inhibit_until = max(self._inhibit_until, start + int(VOICE_CHANGE_INHIBIT_S * self.sr))
            self._voice = voice
        self.echo.feed_reference(np.asarray(pcm, np.float32), start)
        self._ref_end = start + len(pcm)
        return start

    def agent_speaking(self) -> bool:
        return self._mic_pos < self._ref_end + int(TAIL_S * self.sr)

    def cancel_pending(self) -> None:
        """The caller interrupted: whatever was queued after now will not be
        played, so it must not be expected as echo either."""
        self._ref_end = min(self._ref_end, self._mic_pos)
        self.detector.reset()

    # -- caller side --------------------------------------------------
    def process(self, mic: np.ndarray) -> DuplexResult:
        cleaned, echo = self.echo.process_ex(mic)
        self._mic_pos += len(mic)
        speaking = self.agent_speaking()
        event = self.detector.process(cleaned, echo) if cleaned.size else None
        acted = None
        suppressed = False
        if event is not None:
            trusted = (
                speaking
                and self.echo.canceller.erle_db >= self.min_trusted_erle_db
                and self._mic_pos >= self._inhibit_until
            )
            if trusted:
                acted = event
                self.barge_ins += 1
            else:
                suppressed = True
                self.suppressed += 1
        return DuplexResult(cleaned, acted, suppressed, speaking)


def wav_to_pcm16k(wav_bytes: bytes, sr_out: int = 16000) -> np.ndarray:
    """Decode a WAV (the TTS output, typically 22.05 kHz) to float32 mono at the
    microphone's rate, for use as the echo canceller's far-end reference. FFT
    resampling: exact band-limiting, and clips are short. The reference only
    needs the spectral shape of what the caller will hear, not audiophile
    fidelity, and errors here are absorbed by the adaptive filter."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr, ch, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"unsupported sample width {width}")
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr == sr_out or x.size == 0:
        return x
    n_out = int(round(x.size * sr_out / sr))
    spec = np.fft.rfft(x)
    out_bins = n_out // 2 + 1
    resized = np.zeros(out_bins, dtype=np.complex128)
    k = min(out_bins, spec.size)
    resized[:k] = spec[:k]
    return (np.fft.irfft(resized, n=n_out) * (n_out / x.size)).astype(np.float32)
