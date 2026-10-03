"""agent/telephony_augment.py: does the augmentation actually do what it claims, on synthetic audio.

    python -m pytest tests/test_telephony_augment.py -v

No recording, no pod: a pure sine tone and generated noise stand in for real audio. What is checked is the physical
effect (bandwidth, level, sample count, that a codec round trip changes something), not any claim about accuracy.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent.telephony_augment import (
    PROFILES,
    AugmentSpec,
    add_noise,
    apply_gain_db,
    apply_profile,
    band_limit,
    clip,
    mu_law_round_trip,
    packet_loss,
    resample,
    white_noise,
)

SR = 16000


def _tone(freq: float, seconds: float = 0.5, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2))


def _band_energy_ratio(x: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(x.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    in_band = spectrum[(freqs >= low_hz) & (freqs <= high_hz)].sum()
    return in_band / (spectrum.sum() + 1e-12)


# ------------------------------------------------------------------------------------------------------ band_limit


def test_a_tone_inside_the_telephone_band_survives():
    tone = _tone(1000)
    filtered = band_limit(tone, SR)
    assert _band_energy_ratio(filtered, SR, 300, 3400) > 0.9  # a 101-tap FIR has some roll-off, not a brick wall


def test_a_tone_outside_the_telephone_band_is_suppressed():
    low = _tone(80)  # below 300 Hz
    high = _tone(6000)  # above 3400 Hz
    for tone in (low, high):
        filtered = band_limit(tone, SR)
        assert _power(filtered) < 0.1 * _power(tone)


def test_band_limit_never_changes_the_sample_count():
    tone = _tone(1000)
    assert len(band_limit(tone, SR)) == len(tone)


# -------------------------------------------------------------------------------------------------------- resample


def test_resampling_to_the_same_rate_is_a_no_op():
    tone = _tone(1000)
    out, sr = resample(tone, SR, SR)
    assert sr == SR and np.array_equal(out, tone.astype(np.float32))


def test_downsampling_changes_the_sample_count_and_the_rate_proportionally():
    tone = _tone(1000, seconds=1.0)
    out, sr = resample(tone, SR, 8000)
    assert sr == 8000
    assert abs(len(out) - 8000) <= 1


@given(st.integers(min_value=4000, max_value=48000))
@settings(max_examples=20)
def test_resampling_always_returns_the_requested_rate(target_rate):
    tone = _tone(200, seconds=0.2)
    out, sr = resample(tone, SR, target_rate)
    assert sr == target_rate and len(out) > 0


# --------------------------------------------------------------------------------------------------------- mu-law


def test_mu_law_round_trip_changes_a_smooth_tone_slightly_not_drastically():
    tone = _tone(1000)
    out = mu_law_round_trip(tone)
    assert out.shape == tone.shape
    correlation = np.corrcoef(tone, out)[0, 1]
    assert correlation > 0.95  # recognisably the same signal
    assert not np.array_equal(out, tone)  # but quantised, not identical


def test_mu_law_never_exceeds_full_scale():
    loud = _tone(1000, amplitude=1.5)  # deliberately out of range
    out = mu_law_round_trip(loud)
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_silence_stays_silence():
    silence = np.zeros(SR // 10, dtype=np.float32)
    assert np.allclose(mu_law_round_trip(silence), 0.0, atol=1e-6)


# ------------------------------------------------------------------------------------------------------ add_noise


def test_a_lower_snr_adds_more_noise_power():
    rng = np.random.default_rng(1)
    tone = _tone(1000)
    noise = white_noise(len(tone), rng)
    quiet_noise = add_noise(tone, noise, snr_db=20, rng=np.random.default_rng(1))
    loud_noise = add_noise(tone, noise, snr_db=0, rng=np.random.default_rng(1))
    added_quiet = _power(quiet_noise - tone)
    added_loud = _power(loud_noise - tone)
    assert added_loud > added_quiet


def test_a_noise_clip_shorter_than_the_audio_is_looped_not_truncated():
    rng = np.random.default_rng(2)
    tone = _tone(500, seconds=1.0)
    short_noise = white_noise(100, rng)
    out = add_noise(tone, short_noise, snr_db=5, rng=rng)
    assert len(out) == len(tone)


def test_empty_noise_leaves_the_audio_unchanged():
    tone = _tone(500)
    out = add_noise(tone, np.array([], dtype=np.float32), snr_db=0, rng=np.random.default_rng(3))
    assert np.array_equal(out, tone.astype(np.float32))


# --------------------------------------------------------------------------------------------------- clip / gain


def test_clip_never_exceeds_the_threshold():
    tone = _tone(500, amplitude=1.0)
    out = clip(tone, threshold=0.3)
    assert np.max(np.abs(out)) <= 0.3 + 1e-6


def test_gain_scales_power_by_the_expected_factor_below_the_ceiling():
    tone = _tone(500, amplitude=0.1)
    out = apply_gain_db(tone, db=6.0)
    ratio = _power(out) / _power(tone)
    assert ratio == pytest.approx(10 ** (6.0 / 10), rel=0.05)


def test_gain_never_exceeds_full_scale():
    tone = _tone(500, amplitude=0.9)
    out = apply_gain_db(tone, db=20.0)
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


# ------------------------------------------------------------------------------------------------- packet_loss


def test_packet_loss_zeros_out_whole_bursts_and_keeps_the_length():
    rng = np.random.default_rng(4)
    tone = _tone(500, seconds=1.0, amplitude=1.0)
    out = packet_loss(tone, SR, loss_rate=0.3, burst_ms=20, rng=rng)
    assert len(out) == len(tone)
    # ~50 windows of 20 ms in a 1 s clip, 30% dropped, 320 samples each: far more than the tone's own occasional
    # exact-zero crossing could ever account for, so this only passes if bursts are genuinely being zeroed.
    assert np.sum(out == 0.0) > 1000


def test_zero_loss_rate_changes_nothing():
    rng = np.random.default_rng(5)
    tone = _tone(500)
    out = packet_loss(tone, SR, loss_rate=0.0, rng=rng)
    assert np.array_equal(out, tone)


# ------------------------------------------------------------------------------------------------- apply_profile


@pytest.mark.parametrize("name", list(PROFILES))
def test_every_named_profile_runs_end_to_end_without_error(name):
    tone = _tone(1000, seconds=0.3)
    out, sr = apply_profile(tone, SR, PROFILES[name], rng=np.random.default_rng(7))
    assert sr == (PROFILES[name].target_sample_rate or SR)
    assert len(out) > 0
    assert np.max(np.abs(out)) <= 1.0 + 1e-6
    assert not np.any(np.isnan(out))


def test_the_clean_profile_still_band_limits_and_changes_the_sample_rate():
    tone = _tone(1000, seconds=0.3)
    out, sr = apply_profile(tone, SR, PROFILES["clean_8k"], rng=np.random.default_rng(8))
    assert sr == 8000 and len(out) < len(tone)


def test_profiles_are_reproducible_with_the_same_seeded_generator():
    tone = _tone(1000, seconds=0.2)
    out1, _ = apply_profile(tone, SR, PROFILES["rural_line_8k"], rng=np.random.default_rng(42))
    out2, _ = apply_profile(tone, SR, PROFILES["rural_line_8k"], rng=np.random.default_rng(42))
    assert np.array_equal(out1, out2)


def test_a_custom_spec_with_nothing_enabled_is_close_to_a_no_op():
    spec = AugmentSpec("bare", target_sample_rate=None, codec=False)
    tone = _tone(1000, seconds=0.2)
    out, sr = apply_profile(tone, SR, spec, rng=np.random.default_rng(9))
    assert sr == SR and np.array_equal(out, tone.astype(np.float32))


@given(st.floats(min_value=-20, max_value=20), st.floats(min_value=0, max_value=30))
@settings(max_examples=25)
def test_apply_profile_never_produces_nan_or_out_of_range_samples(gain_db, snr_db):
    spec = AugmentSpec(
        "fuzzed", gain_db_range=(gain_db, gain_db), snr_db=snr_db, packet_loss_rate=0.05, clip_threshold=0.9
    )
    tone = _tone(440, seconds=0.1)
    out, _ = apply_profile(tone, SR, spec, rng=np.random.default_rng(11))
    assert not np.any(np.isnan(out))
    assert np.max(np.abs(out)) <= 1.0 + 1e-6
