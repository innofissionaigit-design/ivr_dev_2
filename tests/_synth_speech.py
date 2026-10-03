"""Shared synthetic-speech generator for the audio tests (no real audio
exists locally). Parameters control exactly the properties a detector is
supposed to measure, so ground truth is known by construction."""

import numpy as np

SR = 16000


def speech(
    f0=120.0,
    dur=6.0,
    syll_hz=4.5,
    pause_every=0,
    pause_s=0.0,
    tremor_hz=0.0,
    tremor_st=0.0,
    jitter_st=0.05,
    breath=0.02,
    amp=0.3,
    seed=0,
):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    # pitch track in semitones: slow drift + tremor + smoothed random jitter
    st = 0.6 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 6.28))
    if tremor_hz:
        st += tremor_st * np.sin(2 * np.pi * tremor_hz * t + rng.uniform(0, 6.28))
    if jitter_st:
        n = rng.standard_normal(t.size // 160 + 2)
        st += jitter_st * np.interp(t, np.arange(n.size) * 0.01, n)
    f0_t = f0 * 2.0 ** (st / 12.0)
    phase = 2 * np.pi * np.cumsum(f0_t) / SR
    sig = np.zeros_like(t)
    for k in range(1, 26):
        if k * f0 * 1.3 > 3400:
            break
        sig += np.sin(k * phase + rng.uniform(0, 6.28)) / k
    env = np.clip(np.sin(2 * np.pi * syll_hz * t - np.pi / 2) * 1.4, 0.0, 1.0) ** 0.8
    if pause_every and pause_s:
        period = pause_every / syll_hz + pause_s
        env = env * ((t % period) < pause_every / syll_hz)
    sig = sig * env
    sig = sig / max(float(np.max(np.abs(sig))), 1e-9)
    if breath:
        sig = sig + breath * rng.standard_normal(t.size) * (env > 0.05)
    return (amp * sig / max(float(np.max(np.abs(sig))), 1e-9)).astype(np.float32)


def young(seed=0, f0=120.0):
    return speech(f0=f0, syll_hz=4.8, jitter_st=0.05, breath=0.02, seed=seed)


def older(seed=0, f0=120.0):
    return speech(
        f0=f0,
        syll_hz=2.3,
        pause_every=3,
        pause_s=0.55,
        tremor_hz=6.0,
        tremor_st=0.5,
        jitter_st=0.30,
        breath=0.10,
        seed=seed,
    )
