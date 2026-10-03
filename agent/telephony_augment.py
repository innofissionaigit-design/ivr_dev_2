"""Make studio-clean audio sound like it came down a real rural West Bengal telephone line.

Needs no recording and no pod: every function here is pure numpy (no scipy, no new dependency) over a plain float32
array, so it runs on any audio at all -- a public-domain clip, a TTS-generated sentence, or a synthetic tone -- while
real call recordings do not yet exist. It closes part of the gap between "the ASR sounds fine in the office" and "the
ASR is fed a narrowband, compressed, noisy phone call": table step 20 in the pre-fine-tuning plan.

    band_limit(audio, sr)        keep only what a phone line carries (ITU G.712: 300-3400 Hz)
    resample(audio, sr, 8000)    down- or up-sample (band-limits first to avoid aliasing)
    mu_law_round_trip(audio)     the G.711 8-bit non-linear quantisation every real call already goes through
    add_noise(audio, noise, db)  additive noise at a target signal-to-noise ratio
    clip(audio, threshold)       a level that overdrove the microphone or the line
    apply_gain_db(audio, db)     a quieter or louder caller
    packet_loss(audio, sr, ...)  short zeroed bursts, standing in for dropped RTP packets
    apply_profile(audio, sr, spec)   a named recipe applying several of the above in order

None of this invents accent, dialect or vocabulary -- it only degrades the CHANNEL. It cannot substitute for real
rural West Bengal recordings; it only stops "the model never saw telephone conditions" from being an extra, avoidable
source of error on top of whatever the accent gap turns out to be.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

Audio = NDArray[np.float32]  # mono, -1..1

_MU = 255.0


def _sinc_lowpass(cutoff_hz: float, sample_rate: float, num_taps: int) -> Audio:
    fc = cutoff_hz / sample_rate
    n = np.arange(num_taps) - (num_taps - 1) / 2
    kernel = np.sinc(2 * fc * n) * 2 * fc
    return cast("Audio", (kernel * np.hamming(num_taps)).astype(np.float32))


def band_limit(
    audio: Audio, sample_rate: float, low_hz: float = 300.0, high_hz: float = 3400.0, num_taps: int = 101
) -> Audio:
    """A telephone line carries only 300-3400 Hz (ITU G.712); an FIR bandpass built as the difference of two
    windowed-sinc lowpass kernels (`high_hz` minus `low_hz`), the standard construction -- no scipy needed."""
    num_taps = num_taps | 1  # an odd length gives the kernel a single centre sample
    kernel = _sinc_lowpass(high_hz, sample_rate, num_taps) - _sinc_lowpass(low_hz, sample_rate, num_taps)
    return np.convolve(audio, kernel, mode="same").astype(np.float32)


def resample(audio: Audio, sample_rate: float, target_rate: float) -> tuple[Audio, int]:
    """Linear-interpolation resampling (good enough for augmentation, not for production playback). Callers going to
    8 kHz should `band_limit` to at most `target_rate / 2` first, or a real phone line's own bandpass does it for
    them anyway -- `apply_profile` below does this automatically."""
    if sample_rate == target_rate:
        return audio.astype(np.float32), int(target_rate)
    duration = len(audio) / sample_rate
    old_t = np.linspace(0, duration, len(audio), endpoint=False)
    n_new = max(1, int(round(duration * target_rate)))
    new_t = np.linspace(0, duration, n_new, endpoint=False)
    return np.interp(new_t, old_t, audio).astype(np.float32), int(target_rate)


def mu_law_round_trip(audio: Audio) -> Audio:
    """Encode to G.711 mu-law's 256 non-linear levels and decode back -- the quantisation step every real call
    already goes through, distinct from (and additional to) the bandwidth limit `band_limit` applies."""
    x = np.clip(audio, -1.0, 1.0).astype(np.float64)
    compressed = np.sign(x) * np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    quantized = np.round(compressed * 127.0) / 127.0  # 8-bit mu-law: 127 levels either side of zero
    expanded = np.sign(quantized) * (np.expm1(np.abs(quantized) * np.log1p(_MU)) / _MU)
    return cast("Audio", expanded.astype(np.float32))


def white_noise(num_samples: int, rng: np.random.Generator) -> Audio:
    """Unit-power white noise, for when no real background-noise clip is available yet."""
    return rng.normal(0.0, 1.0, num_samples).astype(np.float32)


def add_noise(audio: Audio, noise: Audio, snr_db: float, rng: np.random.Generator) -> Audio:
    """`noise` at a target signal-to-noise ratio, looped if shorter than `audio` and a random window taken if longer,
    so repeated calls with the same noise clip do not always add the same slice."""
    if len(noise) == 0:
        return audio.astype(np.float32)
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(audio) + 1)) if len(noise) > len(audio) else 0
    segment = noise[start : start + len(audio)]
    signal_power = float(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
    noise_power = float(np.mean(segment.astype(np.float64) ** 2)) + 1e-12
    scaled = segment * np.sqrt((signal_power / (10 ** (snr_db / 10))) / noise_power)
    return cast("Audio", np.clip(audio + scaled, -1.0, 1.0).astype(np.float32))


def clip(audio: Audio, threshold: float = 0.9) -> Audio:
    """A level that overdrove the microphone, the AGC or the line -- everything past `threshold` is flattened."""
    return np.clip(audio, -threshold, threshold).astype(np.float32)


def apply_gain_db(audio: Audio, db: float) -> Audio:
    """A quieter or louder caller (a phone held far from the mouth, a speakerphone, a shout)."""
    return cast("Audio", np.clip(audio * (10.0 ** (db / 20.0)), -1.0, 1.0).astype(np.float32))


def packet_loss(
    audio: Audio, sample_rate: float, loss_rate: float = 0.02, burst_ms: float = 20.0, *, rng: np.random.Generator
) -> Audio:
    """Short bursts zeroed out, standing in for dropped RTP packets on a poor mobile or VoIP link. `loss_rate` is the
    fraction of `burst_ms`-long windows across the clip that are dropped, not a per-sample probability."""
    out = audio.copy()
    burst_len = max(1, int(sample_rate * burst_ms / 1000))
    windows = max(1, len(audio) // burst_len)
    num_bursts = int(round(windows * loss_rate))
    for _ in range(num_bursts):
        start = int(rng.integers(0, max(1, len(audio) - burst_len + 1)))
        out[start : start + burst_len] = 0.0
    return out.astype(np.float32)


class AugmentSpec:
    """One named recipe: the tags a manifest row would carry (agent/asr_manifest.py's TAGS) if this were how it was
    produced, applied in a fixed order: resample -> gain -> noise -> packet loss -> clip -> codec."""

    def __init__(
        self,
        name: str,
        *,
        target_sample_rate: int | None = 8000,
        gain_db_range: tuple[float, float] | None = None,
        snr_db: float | None = None,
        packet_loss_rate: float | None = None,
        clip_threshold: float | None = None,
        codec: bool = True,
    ) -> None:
        self.name = name
        self.target_sample_rate = target_sample_rate
        self.gain_db_range = gain_db_range
        self.snr_db = snr_db
        self.packet_loss_rate = packet_loss_rate
        self.clip_threshold = clip_threshold
        self.codec = codec


# REASONED, not measured: there is no real call audio yet to calibrate these against (see the module docstring).
# Recalibrate once real rural West Bengal recordings exist.
PROFILES: dict[str, AugmentSpec] = {
    "clean_8k": AugmentSpec("clean_8k"),
    "noisy_8k": AugmentSpec("noisy_8k", snr_db=12.0, gain_db_range=(-6.0, 3.0)),
    "speakerphone_8k": AugmentSpec("speakerphone_8k", snr_db=15.0, clip_threshold=0.8, gain_db_range=(-10.0, 0.0)),
    "rural_line_8k": AugmentSpec(
        "rural_line_8k", snr_db=6.0, packet_loss_rate=0.03, gain_db_range=(-12.0, 2.0), clip_threshold=0.85
    ),
}


def apply_profile(
    audio: Audio, sample_rate: int, spec: AugmentSpec, rng: np.random.Generator, noise: Audio | None = None
) -> tuple[Audio, int]:
    """Apply `spec` to `audio`. `noise` is a real background-noise clip if one is available; without one, synthetic
    white noise stands in so this never needs a recording to run."""
    out, sr = audio.astype(np.float32), sample_rate
    if spec.target_sample_rate and spec.target_sample_rate != sr:
        out = band_limit(out, sr, high_hz=min(3400.0, spec.target_sample_rate / 2 - 100.0))
        out, sr = resample(out, sr, spec.target_sample_rate)
    if spec.gain_db_range:
        low, high = spec.gain_db_range
        out = apply_gain_db(out, float(rng.uniform(low, high)))
    if spec.snr_db is not None:
        out = add_noise(out, noise if noise is not None else white_noise(len(out), rng), spec.snr_db, rng)
    if spec.packet_loss_rate:
        out = packet_loss(out, sr, spec.packet_loss_rate, rng=rng)
    if spec.clip_threshold is not None:
        out = clip(out, spec.clip_threshold)
    if spec.codec:
        out = mu_law_round_trip(out)
    return out, sr
