"""KCD-055 (noise suppression) and KCD-057 (input level normalisation):
agent/conditioning.py and agent/level.py, on SYNTHETIC speech and noise.

What is measured here is a PROXY for recognition accuracy -- the fraction of the
clean signal's voiced frames that are still voiced, at the same pitch, after noise
and conditioning -- plus SNR against a known clean reference. It is NOT word error
rate and it is not entity accuracy: those need an ASR and real recordings, which
tools/conditioning_eval.py measures on a pod. The four "noise profiles" are
stand-ins that differ in spectral shape and stationarity
(tests/_synth_noise.py), not recordings of a street, a clinic, a market or a fan.

The numbers in the assertions were MEASURED on these synthetic buckets (see the
comments); the bounds sit outside the measurement with room, and are REASONED.

    python -m pytest tests/test_conditioning.py -v
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _synth_noise import PROFILES, noise_profile
from _synth_voices import FATHER, MOTHER, SON, utterance

from agent.conditioning import (
    MAX_F0_ERROR,
    MIN_RETENTION,
    bench,
    condition,
    mix_at_snr,
    preservation,
    speech_frame_accuracy,
)
from agent.level import (
    CEILING_DBFS,
    LEVEL_BANDS,
    MAX_GAIN_DB,
    TARGET_RMS_DBFS,
    level_band,
    normalise,
    soft_limit,
    speech_rms_dbfs,
)

SR = 16000
SNRS = (20, 10, 5, 0)


@pytest.fixture(scope="module")
def clips():
    return [utterance(s, dur=3.0, seed=i + 1, amp=0.3, noise_db=-90) for i, s in enumerate((FATHER, MOTHER, SON))]


@pytest.fixture(scope="module")
def result(clips):
    return bench(clips, lambda p, n, s: noise_profile(p, n, -40.0, seed=s), PROFILES, SNRS)


# ============================================================ KCD-055 noise


def test_at_least_four_noise_profiles_at_graded_snr_are_measured_before_and_after(result):
    profiles = {r.profile for r in result["rows"]}
    snrs = {r.snr_db for r in result["rows"]}
    assert len(profiles) >= 4 and snrs == set(map(float, SNRS))
    assert len(result["rows"]) == len(profiles) * len(snrs)
    assert "PROXY" in result["metric"]  # the report says what it measures


def test_suppression_raises_snr_wherever_the_clip_is_noisy(result):
    for r in result["rows"]:
        if r.snr_db <= 10:
            assert r.snr_gain_db >= 2.0, (r.profile, r.snr_db, r.snr_gain_db)  # measured +3.3 to +9.1 dB


def test_a_clean_clip_is_left_alone(result):
    for r in result["rows"]:
        if r.snr_db >= 20:
            assert r.suppressed_share == 0.0 and abs(r.snr_gain_db) < 0.5


def test_no_bucket_gets_worse_after_suppression(result):
    # the story: suppression "never improves noise at the cost of entity accuracy"
    assert result["buckets_that_got_worse"] == [], result["buckets_that_got_worse"]
    for r in result["rows"]:
        assert r.accuracy_after >= r.accuracy_before - 0.02, (r.profile, r.snr_db)


def test_the_worst_bucket_degradation_stays_under_the_stated_bound(result):
    # STATED BOUND (REASONED): the worst (profile, SNR) bucket loses no more than 0.5 of
    # the proxy accuracy the best bucket keeps; measured 0.45 at 0 dB SNR. Above 5 dB SNR
    # the bound is 0.25 (measured 0.18).
    assert result["worst_bucket_degradation_vs_best"] <= 0.50
    best = max(r.accuracy_after for r in result["rows"])
    for r in result["rows"]:
        if r.snr_db >= 5:
            assert best - r.accuracy_after <= 0.25, (r.profile, r.snr_db)


def test_conditioning_helps_most_where_it_is_needed_most(result):
    gains = {(r.profile, r.snr_db): r.accuracy_change for r in result["rows"]}
    assert max(gains[(p, 0.0)] for p in PROFILES) >= 0.20  # measured up to +0.75 at 0 dB
    assert all(gains[(p, 20.0)] == pytest.approx(0.0, abs=0.02) for p in PROFILES)


# ----------------------------------------------------------- the guard


def test_the_guard_rejects_suppression_that_removes_the_speech():
    x = utterance(FATHER, dur=3.0, seed=1, amp=0.3)
    ruined = x * 0.0 + 0.001 * np.random.default_rng(0).standard_normal(x.size).astype(np.float32)
    retention, f0err = preservation(x, ruined, SR)
    assert retention < MIN_RETENTION  # the harmonic structure is gone

    np.random.default_rng(1)
    noisy = mix_at_snr(x, noise_profile("street", x.size, -40, seed=1), 5)
    import agent.conditioning as c

    real_enhance = c.enhance
    try:
        c.enhance = lambda samples, sr, **kw: ruined  # a suppressor that eats the speech
        out, rep = condition(noisy, SR, normalise_level=False)
    finally:
        c.enhance = real_enhance
    assert rep.suppressed is False and rep.reason == "rejected_it_removed_speech"
    assert np.allclose(out, noisy)  # the UNSUPPRESSED clip is what is used


def test_a_good_suppression_passes_the_guard():
    x = utterance(MOTHER, dur=3.0, seed=2, amp=0.3)
    noisy = mix_at_snr(x, noise_profile("clinic", x.size, -40, seed=2), 8)
    _, rep = condition(noisy, SR)
    assert rep.suppressed and rep.reason == "applied"
    assert rep.voiced_retention >= MIN_RETENTION and rep.f0_error <= MAX_F0_ERROR


def test_never_and_always_modes():
    x = utterance(SON, dur=3.0, seed=3, amp=0.3)
    noisy = mix_at_snr(x, noise_profile("fan", x.size, -40, seed=3), 20)
    assert condition(noisy, SR, suppress="never")[1].suppressed is False
    assert condition(noisy, SR, suppress="auto")[1].reason == "clean_enough"


# ============================================================ KCD-057 level


@pytest.fixture(scope="module")
def band_clips(clips):
    """The same speech scaled into each of the four level bands (active-speech RMS)."""
    base = clips[0]
    cur = speech_rms_dbfs(base, SR)
    targets = {"very_quiet": -55.0, "quiet": -40.0, "normal": -27.0, "loud": -12.0}
    return {name: (base * 10 ** ((t - cur) / 20.0)).astype(np.float32) for name, t in targets.items()}


def test_the_four_bands_are_covered_by_the_scaled_clips(band_clips):
    assert [level_band(speech_rms_dbfs(x, SR)) for x in band_clips.values()] == [n for n, _, _ in LEVEL_BANDS]


def test_every_band_is_brought_to_the_target_level(band_clips):
    # measured: every band lands within 1 dB of -24 dBFS
    after = {n: speech_rms_dbfs(normalise(x, SR)[0], SR) for n, x in band_clips.items()}
    for n, v in after.items():
        assert v == pytest.approx(TARGET_RMS_DBFS, abs=1.5), (n, v)
    assert max(after.values()) - min(after.values()) <= 2.0  # the worst band is within 2 dB of the best


def test_the_limiter_holds_the_ceiling_however_hot_the_input(band_clips):
    hot = band_clips["loud"] * 6.0  # far past full scale
    y, report = normalise(np.clip(hot, -1, 1), SR)
    assert np.max(np.abs(y)) <= 10 ** (CEILING_DBFS / 20.0) + 1e-6
    lim, share = soft_limit(np.array([0.0, 0.3, 0.9, 1.0, 5.0, -5.0], np.float32))
    assert np.max(np.abs(lim)) <= 10 ** (CEILING_DBFS / 20.0) + 1e-6 and share > 0
    assert lim[1] == pytest.approx(0.3)  # below the knee: identity


def test_already_clipped_input_is_never_amplified():
    x = np.clip(utterance(FATHER, dur=3.0, seed=1, amp=0.3) * 12.0, -1, 1)
    y, report = normalise(x, SR)
    assert report.clipped_input and report.gain_db <= 0.0


def test_gain_is_bounded_so_faint_noise_is_not_pumped_up_without_limit():
    x = utterance(FATHER, dur=3.0, seed=1, amp=0.3) * 10 ** (-90 / 20.0)  # -90 dBFS: mostly noise
    _, report = normalise(x, SR)
    assert report.gain_db <= MAX_GAIN_DB


def test_silence_is_not_scaled():
    y, report = normalise(np.zeros(SR, np.float32), SR)
    assert report.gain_db == 0.0 and not np.any(y)


def test_normalisation_does_not_change_what_the_proxy_can_see(band_clips):
    # On this synthetic proxy (voiced frames at the right pitch) level does not matter,
    # and the test says so plainly: it pins that normalisation does not DAMAGE it.
    # Whether level matters to the real recogniser is what tools/conditioning_eval.py measures.
    ref = normalise(band_clips["normal"], SR)[0]
    for n, x in band_clips.items():
        assert speech_frame_accuracy(ref, normalise(x, SR)[0], SR) >= 0.95, n
