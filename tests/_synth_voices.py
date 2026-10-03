"""Synthetic SPEAKERS for tests/test_speaker_change.py and tests/test_near_end.py.

tests/_synth_speech.py makes one kind of voice (harmonics of a drifting f0). A
speaker-change detector needs voices that differ the way PEOPLE differ, and
phrases that differ the way SPEECH differs, so that "same speaker, different
words" and "different speaker" are separable only by what a real detector would
have to use:

  * a speaker has a base pitch and a vocal-tract length that scales every
    formant (women and children ~1.15-1.3, men ~1.0);
  * a phrase moves the formants around by phoneme (each ~150 ms segment
    perturbs F1-F3 by up to +-12%) and the pitch by intonation;
  * every utterance gets its own noise and its own slow pitch drift.

Ground truth is known by construction. What this can and cannot show is stated
in tests/test_speaker_change.py: it shows the detector separates speakers who
differ in pitch and tract length, and stays quiet over changes of content. It
does not show how it behaves on two real siblings.
"""

import numpy as np

SR = 16000
BASE_FORMANTS = np.array([500.0, 1500.0, 2500.0])
FORMANT_BW = np.array([90.0, 140.0, 200.0])


class Speaker:
    def __init__(self, f0, vtl, breathiness=0.02, name=""):
        self.f0, self.vtl, self.breathiness, self.name = f0, vtl, breathiness, name

    def __repr__(self):
        return f"Speaker({self.name or ''} f0={self.f0}, vtl={self.vtl})"


def _envelope(freqs, formants):
    """Spectral envelope (linear) at `freqs` for a set of formants: a sum of
    resonance peaks over a -6 dB/octave tilt."""
    env = np.zeros_like(freqs, dtype=np.float64)
    for f, bw in zip(formants, FORMANT_BW):
        env += 1.0 / (1.0 + ((freqs - f) / bw) ** 2)
    return (env + 0.03) / np.maximum(freqs / 500.0, 1.0)


def utterance(speaker, dur=2.0, seed=0, noise_db=-45.0, amp=0.3):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    # intonation + slow drift, per utterance
    contour = 2 ** (
        rng.uniform(-0.25, 0.25) * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t + rng.uniform(0, 6.28)) / 1.0
        + 0.04 * np.sin(2 * np.pi * 4.5 * t)
    )
    f0_t = speaker.f0 * contour
    phase = 2 * np.pi * np.cumsum(f0_t) / SR
    n_harm = int(3800 / (speaker.f0 * 0.75))
    seg = 0.15
    n_seg = int(np.ceil(dur / seg)) + 1
    # phoneme-level formant perturbation, shared by all harmonics in a segment
    formants = np.array([BASE_FORMANTS * speaker.vtl * (1 + rng.uniform(-0.12, 0.12, 3)) for _ in range(n_seg)])
    seg_t = np.arange(n_seg) * seg
    sig = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        fk = k * speaker.f0
        amp_seg = np.array([_envelope(np.array([fk]), formants[i])[0] for i in range(n_seg)])
        sig += np.interp(t, seg_t, amp_seg) * np.sin(k * phase + rng.uniform(0, 6.28))
    # syllable-rate energy envelope with real gaps between syllables
    env = np.clip(np.sin(2 * np.pi * rng.uniform(3.0, 4.5) * t - np.pi / 2) * 1.4, 0.0, 1.0) ** 0.8
    sig = sig * env
    sig = amp * sig / max(float(np.max(np.abs(sig))), 1e-9)
    sig = sig + speaker.breathiness * rng.standard_normal(t.size) * (env > 0.05) * amp
    sig = sig + 10 ** (noise_db / 20.0) * rng.standard_normal(t.size)
    return sig.astype(np.float32)


# a small population
FATHER = Speaker(105, 1.00, name="father")
MOTHER = Speaker(205, 1.16, name="mother")
SON = Speaker(140, 1.04, name="son")
DAUGHTER = Speaker(235, 1.22, name="daughter")
GRANDFATHER = Speaker(120, 0.97, breathiness=0.05, name="grandfather")
POPULATION = (FATHER, MOTHER, SON, DAUGHTER, GRANDFATHER)


def scaled(x, db):
    return (x * 10 ** (db / 20.0)).astype(np.float32)
