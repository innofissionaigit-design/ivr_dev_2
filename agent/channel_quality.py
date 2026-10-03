"""KCD-075: classify each call's audio into the annotation vocabulary --
clean 16 kHz, narrowband 8 kHz, noisy, or cross-talk -- so live traffic
can be compared against the evaluation population's own labelled
distribution (the whole point of a shared vocabulary: a live "noisy"
rate that drifts from the eval set's is a real, actionable signal, and
comparing them needs both sides using the same four labels).

Deterministic signal processing on the decoded PCM, no model call --
this has to run inside KCD-076's per-call latency slice alongside the
other call-intelligence detectors, so it is FFT/energy arithmetic on a
single utterance clip, not a classifier that needs a GPU.

Bandwidth (clean vs narrowband) is judged from the actual FREQUENCY
CONTENT, not the file's declared sample rate: a telephone leg often
arrives as 8 kHz-bandlimited audio stored/upsampled into a 16 kHz
container, and the container's own sample_rate field would call that
"clean" when the content is not.

Noisy vs clean is a coarse energy-percentile SNR estimate. Cross-talk
detection is the weakest of the four by a wide margin -- there is no
real overlapping-speaker corpus locally to calibrate against, so it is
a conservative, explicitly-labelled HEURISTIC (see classify_channel's
docstring), the same "reasoned, not measured" caveat this codebase
already carries on agent/fast_path.py's FAQ_COMMIT_FLOOR and
agent/confidence_gate.py's threshold -- recalibrate against real call
audio before trusting it past the pilot.
"""

from __future__ import annotations

import numpy as np

CHANNEL_CLEAN_16K = "clean_16k"
CHANNEL_NARROWBAND_8K = "narrowband_8k"
CHANNEL_NOISY = "noisy"
CHANNEL_CROSSTALK = "cross_talk"

# A telephone-bandwidth signal has essentially no energy above ~3.4kHz
# (the classic PSTN cutoff); a wideband 16kHz-capable signal does. The
# ratio of energy above vs below this line is the bandwidth test.
_TELEPHONY_CUTOFF_HZ = 3400
_NARROWBAND_HIGH_ENERGY_RATIO = 0.02  # below this fraction above cutoff -> narrowband

# Coarse SNR proxy: energy of the quietest 10th percentile of frames
# (presumed near-silence/noise floor) vs the loudest 10th percentile
# (presumed speech peaks). REASONED, not measured -- no labelled
# clean/noisy call corpus exists locally to calibrate the exact ratio;
# recalibrate against real call audio before trusting this past the
# pilot, per this module's own docstring.
_NOISY_SNR_RATIO_FLOOR = 0.15
_FRAME_MS = 20


def _frame_energies(samples: np.ndarray, sample_rate: int, frame_ms: int = _FRAME_MS) -> np.ndarray:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return np.array([np.mean(samples.astype(np.float64) ** 2)]) if len(samples) else np.array([0.0])
    trimmed = samples[: n_frames * frame_len].astype(np.float64)
    frames = trimmed.reshape(n_frames, frame_len)
    return np.mean(frames**2, axis=1)


def _is_narrowband(samples: np.ndarray, sample_rate: int) -> bool:
    if sample_rate <= 8000:
        return True  # the container itself cannot carry content above 4kHz
    if len(samples) < 32:
        return False
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    total_energy = float(np.sum(spectrum**2))
    if total_energy <= 0:
        return False
    high_energy = float(np.sum(spectrum[freqs > _TELEPHONY_CUTOFF_HZ] ** 2))
    return (high_energy / total_energy) < _NARROWBAND_HIGH_ENERGY_RATIO


def _is_noisy(samples: np.ndarray, sample_rate: int) -> bool:
    energies = _frame_energies(samples, sample_rate)
    if len(energies) < 5:
        return False
    quiet = np.percentile(energies, 10)
    loud = np.percentile(energies, 90)
    if loud <= 0:
        return False
    return (quiet / loud) > _NOISY_SNR_RATIO_FLOOR


_CROSSTALK_MAX_SPECTRAL_FLATNESS = 0.1


def _spectral_flatness(samples: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of the power spectrum -- the
    standard tonality measure. Close to 1.0 for broadband noise (flat
    spectrum), close to 0.0 for tonal/structured energy (a few dominant
    frequencies, the shape voiced speech has). Used to tell "continuous
    BUT structured, voice-shaped energy" (cross-talk's proxy) apart from
    "continuous, featureless energy" (noisy's proxy) -- the two
    occupancy-only heuristics alone could not distinguish (a flat noise
    floor filling silence gaps looks identical to constant talking by
    energy occupancy alone)."""
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64))) ** 2
    spectrum = spectrum[spectrum > 0]
    if len(spectrum) == 0:
        return 1.0
    log_mean = float(np.mean(np.log(spectrum)))
    geometric_mean = np.exp(log_mean)
    arithmetic_mean = float(np.mean(spectrum))
    if arithmetic_mean <= 0:
        return 1.0
    return float(geometric_mean / arithmetic_mean)


def _is_crosstalk(samples: np.ndarray, sample_rate: int) -> bool:
    """HEURISTIC, not a real overlapping-speech detector (see module
    docstring): near-continuous, TONAL/structured energy -- almost no
    frame ever drops close to the noise floor, AND the spectrum is not
    just flat broadband noise -- is a coarse proxy for "someone is
    always talking, possibly two people at once". A single articulate
    speaker with normal pauses does not usually look like this over a
    whole utterance; a merely noisy channel fills the gaps but stays
    spectrally flat, which _spectral_flatness is what separates the
    two."""
    energies = _frame_energies(samples, sample_rate)
    if len(energies) < 10:
        return False
    peak = np.percentile(energies, 95)
    if peak <= 0:
        return False
    quiet_frame_fraction = float(np.mean(energies < 0.05 * peak))
    if quiet_frame_fraction >= 0.05:
        return False  # has real quiet stretches -- not continuous
    return _spectral_flatness(samples) < _CROSSTALK_MAX_SPECTRAL_FLATNESS


def classify_channel(samples: np.ndarray, sample_rate: int) -> str:
    """One of CHANNEL_CLEAN_16K / CHANNEL_NARROWBAND_8K / CHANNEL_NOISY /
    CHANNEL_CROSSTALK for one utterance clip. Checked in this order --
    bandwidth first (a hard property of the audio path itself), then
    cross-talk, then noise. Cross-talk BEFORE noisy deliberately: both
    heuristics key off a flattened energy envelope, but cross-talk's
    check (essentially never near the noise floor) is the more specific
    condition -- a signal that never goes quiet is checked for that
    first, rather than falling into the broader, looser noisy bucket a
    quiet/loud percentile ratio alone would sort it into."""
    if len(samples) == 0:
        return CHANNEL_CLEAN_16K
    if _is_narrowband(samples, sample_rate):
        return CHANNEL_NARROWBAND_8K
    if _is_crosstalk(samples, sample_rate):
        return CHANNEL_CROSSTALK
    if _is_noisy(samples, sample_rate):
        return CHANNEL_NOISY
    return CHANNEL_CLEAN_16K
