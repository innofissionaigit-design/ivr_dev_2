"""Synthetic noise profiles for tests/test_barge_in.py and tests/test_noise_suppression.py.

Four profiles chosen to differ in the property that matters to a detector or a
suppressor -- spectral shape and stationarity -- not to imitate a place
faithfully. No recorded noise exists locally; these are stand-ins with known
structure. Level is set as RMS in dBFS so a scene can state its noise plainly.
"""

import numpy as np
from _synth_speech import speech

SR = 16000
PROFILES = ("street", "clinic", "market", "fan")


def _shape(white, lo, hi):
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(white.size, 1.0 / SR)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, n=white.size)


def _at_rms(x, rms_dbfs):
    r = np.sqrt(np.mean(x**2)) + 1e-12
    return (x * (10 ** (rms_dbfs / 20.0) / r)).astype(np.float32)


def noise_profile(name, n, rms_dbfs=-40.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    white = rng.standard_normal(n)
    if name == "street":
        # traffic rumble (low-passed, slowly modulated) + horn bursts + hiss
        x = _shape(white, 20, 500) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6.28)))
        x += 0.15 * _shape(rng.standard_normal(n), 500, 6000)
        for _ in range(max(1, int(n / SR / 3))):
            a = int(rng.uniform(0, max(n - SR, 1)))
            L = int(rng.uniform(0.2, 0.5) * SR)
            f = rng.choice([420.0, 510.0, 630.0])
            x[a : a + L] += 0.5 * np.sin(2 * np.pi * f * t[: min(L, n - a)]) * np.hanning(min(L, n - a))
    elif name == "clinic":
        # mains hum and its harmonics + air-handling hiss + occasional clatter
        x = sum(np.sin(2 * np.pi * 50 * k * t + rng.uniform(0, 6.28)) / k for k in (1, 2, 3, 4, 6))
        x = 0.6 * x + 0.5 * _shape(white, 200, 4000)
        for _ in range(max(1, int(n / SR / 2))):
            a = int(rng.uniform(0, max(n - 400, 1)))
            x[a : a + 200] += rng.standard_normal(min(200, n - a)) * 2.0
    elif name == "market":
        # broadband din plus the murmur of several distant voices
        x = 0.6 * _shape(white, 100, 5000)
        for k in range(6):
            x[:] = x + 0.25 * speech(f0=95 + 25 * k, dur=n / SR, seed=seed * 10 + k, amp=1.0)[:n]
    elif name == "fan":
        x = 0.8 * _shape(white, 30, 800) + 0.3 * np.sin(2 * np.pi * 120 * t) + 0.1 * _shape(white, 800, 4000)
    else:
        raise ValueError(name)
    return _at_rms(x, rms_dbfs)
