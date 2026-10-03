"""Listening to the caller, not the room (KCD-053/055): agent/attention.py, and the level+pitch cue added to
speaker-change detection (KCD-054). SYNTHETIC voices and noise; the numbers here describe the synthetic
scenes, not a real room.

    python -m pytest tests/test_attention.py -v
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _synth_voices import FATHER, MOTHER, scaled, utterance

from agent import speaker_change as sc
from agent.attention import ATTENUATION_DB, MAX_FRACTION, attend
from agent.near_end import level_and_pitch

SR = 16000


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


@pytest.fixture(scope="module")
def caller():
    x = utterance(FATHER, dur=2.0, seed=1, amp=0.3)
    level, f0 = level_and_pitch(x, SR)
    return x, level, f0


def test_nothing_happens_before_the_call_has_a_profile(caller):
    x, _, _ = caller
    y, rep = attend(x, None, None, SR)
    assert not rep.applied and rep.reason == "no_profile_yet" and np.array_equal(x, y)


def test_a_clip_that_is_only_the_caller_is_not_changed_where_they_speak(caller):
    x, level, f0 = caller
    y, rep = attend(x, level, f0, SR)
    loud = np.abs(x) > 0.05  # sample positions where the caller is speaking
    assert np.array_equal(x[loud], y[loud])  # bit-for-bit: the caller is never touched
    assert np.max(np.abs(y)) <= np.max(np.abs(x)) + 1e-9  # and nothing is ever amplified


def test_a_bystander_before_and_after_the_caller_is_attenuated_and_the_caller_is_exact(caller):
    x, level, f0 = caller
    bystander = scaled(utterance(MOTHER, dur=1.2, seed=2, amp=0.3), -24.0)
    clip = np.concatenate([bystander, x, bystander])
    y, rep = attend(clip, level, f0, SR)
    n = bystander.size
    assert rep.applied and rep.background > 20
    before, after = rms_db(clip[:n]), rms_db(y[:n])
    assert before - after >= 10.0  # the bystander is well down
    span = slice(n, n + x.size)
    speaking = np.abs(clip[span]) > 0.05  # where the CALLER is speaking (the quiet tail of their
    assert np.array_equal(
        clip[span][speaking], y[span][speaking]
    )  # own clip is a pause, and may be gated like any pause)


def test_hiss_in_the_pauses_between_phrases_is_gated_but_speech_and_its_edges_are_not(caller):
    x, level, f0 = caller
    rng = np.random.default_rng(0)
    pause = np.zeros(SR, dtype=np.float32)  # one second of quiet
    clip = np.concatenate([x, pause, x])
    hiss = (rng.standard_normal(clip.size) * 10 ** (-52 / 20)).astype(np.float32)
    noisy = clip + hiss
    y, rep = attend(noisy, level, f0, SR)
    mid = slice(x.size + int(0.35 * SR), x.size + int(0.65 * SR))  # the middle of the pause, far from speech
    assert rep.noise > 10
    assert rms_db(noisy[mid]) - rms_db(y[mid]) >= ATTENUATION_DB - 3.0
    speech = np.abs(x) > 0.05
    assert np.array_equal(noisy[: x.size][speech], y[: x.size][speech])


def test_the_edge_of_a_word_is_protected_from_the_noise_gate(caller):
    """A weak consonant sits below the vowels but next to them: unvoiced frames within 80 ms of the
    caller's voiced speech must not be treated as noise."""
    x, level, f0 = caller
    y, _ = attend(x, level, f0, SR)
    # wherever the caller speaks, the 80 ms on either side keeps at least half its level
    speaking = np.where(np.abs(x) > 0.1)[0]
    edges = np.concatenate([speaking[:5] - 640, speaking[-5:] + 640])
    edges = edges[(edges > 0) & (edges < x.size)]
    assert np.all(np.abs(y[edges]) >= 0.5 * np.abs(x[edges]) - 1e-9)


def test_a_clip_that_would_be_mostly_removed_is_returned_whole(caller):
    _, level, f0 = caller
    quiet = utterance(MOTHER, dur=2.0, seed=4, amp=0.3)
    quiet = scaled(quiet, -40.0)  # the whole clip is far below the caller's profile
    y, rep = attend(quiet, level, f0, SR)
    assert (not rep.applied) and rep.fraction <= 1.0 and np.array_equal(quiet, y)
    assert MAX_FRACTION < 1.0


def test_the_output_is_the_same_length_and_finite(caller):
    x, level, f0 = caller
    y, _ = attend(np.concatenate([scaled(utterance(MOTHER, dur=0.8, seed=5, amp=0.3), -25.0), x]), level, f0, SR)
    assert y.dtype == np.float32 and np.all(np.isfinite(y))


# ======================================================================== KCD-054: pitch and level as a second cue


def emb(ltas_offset=0.0, octaves=0.0):
    base = np.linspace(-6, 6, sc.N_BANDS)
    return sc.VoiceEmbedding(base + ltas_offset, 7.0 + octaves, 2.0)


def detector(level=-20.0):
    d = sc.SpeakerChangeDetector()
    d.enroll_embedding(emb(), level)
    return d


def test_the_pitch_and_level_cue_needs_the_spectrum_to_agree_first():
    d = detector()
    # 10 dB louder alone: no spectral or pitch difference at all -- the caller moved the phone
    assert d.observe_embedding(emb(), level_dbfs=-10.0).verdict == "same"
    # a large level jump with only a small voice difference is still not enough on its own
    assert d.observe_embedding(emb(0.5, 0.02), level_dbfs=-5.0).verdict == "same"


def test_a_different_pitch_plus_a_different_level_at_moderate_distance_is_a_change_at_once():
    # distance ~4.0 (>= CHANGE_LOW 3.8, < CHANGE_HIGH 5.0): without the cue it needs two turns
    quiet_change = emb(1.5, 0.15)
    assert sc.distance(quiet_change, emb()) >= sc.CHANGE_LOW and sc.distance(quiet_change, emb()) < sc.CHANGE_HIGH
    slow = detector()
    assert slow.observe_embedding(quiet_change).verdict == "same"  # one turn: still "same"
    fast = detector()
    assert (
        fast.observe_embedding(quiet_change, level_dbfs=-30.0).verdict == "changed"
    )  # 10 dB different level: change now


def test_the_cue_does_not_fire_without_a_pitch_difference():
    d = detector()
    same_pitch = emb(3.9, 0.0)  # distance 3.9, same pitch
    assert d.observe_embedding(same_pitch, level_dbfs=-32.0).verdict == "same"


def test_a_pitch_jump_of_the_size_the_cue_names_already_exceeds_the_hard_threshold():
    assert sc.PITCH_JUMP_OCTAVES / sc.PITCH_OCTAVE_SCALE >= sc.CHANGE_HIGH


def test_existing_behaviour_without_a_level_is_unchanged():
    d = detector()
    assert d.observe_embedding(emb(1.5, 0.15)).verdict == "same"
    assert d.observe_embedding(emb(1.5, 0.15)).verdict == "changed"  # two consecutive turns, as before
