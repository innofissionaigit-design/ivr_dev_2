"""Synthetic echo scenes for tests/test_echo_cancel.py and tests/test_barge_in.py.

No recorded handset audio exists locally, so the far-end is the synthetic
speech generator, the echo path is a decaying random impulse response after a
bulk delay (the client's network + output buffering), and the near-end (the
caller) is a second synthetic voice. Ground truth is known by construction.
"""

import numpy as np
from _synth_speech import speech

SR = 16000


def rir(delay_ms, len_ms=100, decay_ms=25, gain=0.35, seed=0):
    """A bulk delay followed by a decaying random tail, scaled so the path's
    energy gain is `gain` (about -9 dB at 0.35): echo quieter than the far end,
    as on a real handset."""
    rng = np.random.default_rng(seed)
    n = int(len_ms * SR / 1000)
    t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-t / (decay_ms / 1000))
    h[:8] *= 0.3
    h = h / np.linalg.norm(h) * gain
    return np.concatenate([np.zeros(int(delay_ms * SR / 1000)), h])


def far_end(seed, dur_each=4.0, voices=3, amp=0.4, same_voice=False):
    """Agent speech: several segments back to back with pitch movement and
    aspiration, so the reference is not a steady vowel (which no adaptive
    filter can identify from).

    same_voice=False (the default, the WORST case for a canceller): each segment is a
    different synthetic SPEAKER, as if the agent's voice changed every few seconds.
    same_voice=True (the DEPLOYED case): one speaker, different sentences -- the agent
    speaks in one TTS voice per language, so its voice changes only when the caller
    switches language."""
    parts = []
    for k in range(voices):
        j = 0 if same_voice else k
        parts.append(
            speech(
                f0=100 + 30 * ((seed + j) % 4) + 8 * j,
                dur=dur_each,
                seed=seed * 7 + k,
                amp=amp,
                tremor_hz=1.3 + 0.4 * ((seed + j) % 3),
                tremor_st=3 + (j % 2),
                breath=0.12,
            )
        )
    x = np.concatenate(parts)
    return x + fricatives(x.size, seed)


def fricatives(n, seed, amp=0.12):
    """Broadband noise bursts (80-150 ms, roughly every 0.5 s), high-passed at
    about 2 kHz, standing in for /s/, /sh/, /f/. Real speech carries roughly a
    fifth of its time as noise-like sound; a vowels-only stand-in has too few
    excited frequencies for any adaptive filter to identify an echo path from,
    which understates cancellation on real speech."""
    rng = np.random.default_rng(500 + seed)
    out = np.zeros(n, np.float32)
    t = 0
    while t < n:
        t += int(rng.uniform(0.3, 0.7) * SR)
        L = int(rng.uniform(0.08, 0.15) * SR)
        if t + L >= n:
            break
        burst = rng.standard_normal(L)
        burst = np.diff(burst, prepend=0.0)  # crude high-pass tilt
        out[t : t + L] += (amp * burst / (np.max(np.abs(burst)) + 1e-9) * np.hanning(L)).astype(np.float32)
        t += L
    return out


def speaker(x, drive=0.0):
    """Loudspeaker non-linearity: soft clipping. drive=0 is linear."""
    if drive <= 0:
        return x
    return (np.tanh(drive * x) / drive).astype(np.float32)


def echo_scene(seed, delay_ms=None, noise=0.001, drive=0.0, gain=0.35, dur_each=4.0, voices=3, same_voice=False):
    """-> (far, mic, echo). mic = echo + noise, caller silent."""
    delay_ms = 120 + 50 * (seed % 5) if delay_ms is None else delay_ms
    far = far_end(seed, dur_each, voices, same_voice=same_voice)
    echo = np.convolve(speaker(far, drive), rir(delay_ms, seed=seed, gain=gain))[: far.size].astype(np.float32)
    rng = np.random.default_rng(1000 + seed)
    mic = echo + noise * rng.standard_normal(far.size).astype(np.float32)
    return far, mic.astype(np.float32), echo


def caller(seed, dur=1.5, f0=155.0, amp=0.25):
    return speech(
        f0=f0 + 10 * (seed % 5),
        dur=dur,
        seed=200 + seed,
        amp=amp,
        syll_hz=4.0,
        tremor_hz=1.0,
        tremor_st=2.0,
        breath=0.05,
    )


def voice_segments(far, dur_each, sr=SR):
    """The far end as the clips the server would send: one per voice."""
    n = int(dur_each * sr)
    return [far[i : i + n] for i in range(0, far.size, n)]


def run_pipeline(mic, far, chunk=1600, **kw):
    from agent.echo_cancel import EchoPipeline

    p = EchoPipeline(**kw)
    p.feed_reference(far, 0)
    cleaned, echos = [], []
    for i in range(0, mic.size, chunk):
        c, e = p.process_ex(mic[i : i + chunk])
        cleaned.append(c)
        echos.append(e)
    return p, np.concatenate(cleaned), np.concatenate(echos)
