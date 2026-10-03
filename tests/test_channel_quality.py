"""KCD-075: agent/channel_quality.py, against synthetic signals with
known properties (no real labelled call-audio corpus locally -- see the
module's own "reasoned, not measured" caveat on the noise/cross-talk
thresholds). Requires numpy (already a project dependency via librosa).

    python -m pytest tests/test_channel_quality.py -v
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.channel_quality import (
    CHANNEL_CLEAN_16K,
    CHANNEL_CROSSTALK,
    CHANNEL_NARROWBAND_8K,
    CHANNEL_NOISY,
    classify_channel,
)

SR = 16000
DURATION_S = 2.0


def _silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * seconds))


def _tone_burst(freq_hz: float, seconds: float, sr: int = SR, amplitude: float = 1.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _speech_like_clean(sr: int = SR) -> np.ndarray:
    """Alternating tone bursts and real silence gaps, energy spread up
    to 6kHz -- a wideband signal with normal speech-like pauses."""
    parts = []
    for freq in (300, 1200, 2500, 5800, 800, 4200):
        parts.append(_tone_burst(freq, 0.15, sr))
        parts.append(_silence(0.08, sr))
    return np.concatenate(parts)


def _speech_like_narrowband(sr: int = SR) -> np.ndarray:
    """Same shape, but every component is below the telephony cutoff --
    what an 8kHz-bandlimited call looks like even stored in a 16kHz
    container."""
    parts = []
    for freq in (300, 800, 1200, 1800, 2500, 3000):
        parts.append(_tone_burst(freq, 0.15, sr))
        parts.append(_silence(0.08, sr))
    return np.concatenate(parts)


def test_wideband_signal_with_normal_pauses_is_clean():
    samples = _speech_like_clean()
    assert classify_channel(samples, SR) == CHANNEL_CLEAN_16K


def test_bandlimited_signal_is_narrowband():
    samples = _speech_like_narrowband()
    assert classify_channel(samples, SR) == CHANNEL_NARROWBAND_8K


def test_an_actual_8khz_sample_rate_is_always_narrowband():
    # The container itself cannot carry content above 4kHz -- decided
    # before any spectral analysis even runs.
    samples = _tone_burst(1000, DURATION_S, sr=8000)
    assert classify_channel(samples, 8000) == CHANNEL_NARROWBAND_8K


def test_wideband_signal_with_heavy_noise_floor_is_noisy():
    rng = np.random.default_rng(42)
    clean = _speech_like_clean()
    noise = rng.normal(0, 0.35, size=len(clean))
    noisy = clean + noise
    assert classify_channel(noisy, SR) == CHANNEL_NOISY


def test_continuous_high_energy_with_no_silence_is_crosstalk():
    # Wideband content, but never drops near the noise floor -- the
    # heuristic's proxy for "someone is always talking."
    t = np.linspace(0, DURATION_S, int(SR * DURATION_S), endpoint=False)
    samples = np.sin(2 * np.pi * 400 * t) + 0.6 * np.sin(2 * np.pi * 5200 * t) + 0.3 * np.sin(2 * np.pi * 2600 * t)
    assert classify_channel(samples, SR) == CHANNEL_CROSSTALK


def test_empty_clip_defaults_to_clean_rather_than_crashing():
    assert classify_channel(np.array([]), SR) == CHANNEL_CLEAN_16K
