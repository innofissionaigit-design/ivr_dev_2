"""KCD-084: agent/senior_voice.py. Validated on SYNTHETIC speech with
known parameters (tests/_synth_speech.py) -- there is no labelled age
corpus locally, so this proves the detector responds to the traits it is
designed around and is conservative about single traits, NOT that it
predicts age for real callers. See the module docstring.

    python -m pytest tests/test_senior_voice.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _synth_speech import SR, older, speech, young

from agent.senior_voice import (
    SENIOR_SCORE,
    SeniorEstimate,
    SeniorEvidence,
    estimate,
    explicit_senior_cue,
    extract_features,
    stated_age_is_senior,
)


def _scores(gen, f0s=(130, 190, 230), seeds=(0, 1)):
    return [estimate(gen(s, f0), SR) for s in seeds for f0 in f0s]


# ------------------------------------------------------------- acoustics


def test_brisk_steady_speech_does_not_sound_older():
    ests = [e for e in _scores(young) if e.sufficient]
    assert len(ests) >= 4
    assert all(e.score < 0.15 and not e.votes_senior for e in ests)


def test_speech_with_the_older_voice_traits_is_recognised_across_pitch_ranges():
    ests = [e for e in _scores(older) if e.sufficient]
    assert len(ests) >= 4
    assert all(e.votes_senior for e in ests), [round(e.score, 2) for e in ests]


def test_pitch_alone_decides_nothing():
    # An older man's raised pitch and an older woman's lowered pitch are
    # exactly why f0 is not a feature: the same young-sounding delivery at
    # very different pitches must score the same.
    lo = estimate(speech(f0=105, syll_hz=4.8, seed=1), SR)
    hi = estimate(speech(f0=230, syll_hz=4.8, seed=1), SR)
    assert lo.sufficient and hi.sufficient
    assert abs(lo.score - hi.score) < 0.1


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("slow only", dict(syll_hz=2.3)),
        ("tremor only", dict(tremor_hz=6.0, tremor_st=0.5)),
        ("breathy and unsteady only", dict(breath=0.10, jitter_st=0.30)),
    ],
)
def test_a_single_trait_never_flags_a_caller_on_its_own(label, kwargs):
    base = dict(syll_hz=4.8, jitter_st=0.05, breath=0.02)
    e = estimate(speech(seed=3, **{**base, **kwargs}), SR)
    assert e.sufficient, label
    assert not e.votes_senior, (label, e.score)


def test_the_features_move_the_right_way():
    slow = extract_features(speech(syll_hz=2.3, seed=1), SR)
    fast = extract_features(speech(syll_hz=4.8, seed=1), SR)
    assert slow["syllable_rate"] < fast["syllable_rate"] - 1.5

    shaky = extract_features(speech(tremor_hz=6.0, tremor_st=0.5, seed=2), SR)
    steady = extract_features(speech(seed=2), SR)
    assert shaky["tremor_st"] > 3 * steady["tremor_st"]
    assert shaky["f0_instability"] > steady["f0_instability"]

    breathy = extract_features(speech(breath=0.15, seed=4), SR)
    clean = extract_features(speech(breath=0.0, seed=4), SR)
    assert breathy["harmonicity"] < clean["harmonicity"]


def test_a_brisk_voice_has_no_pauses_and_a_hesitant_one_does():
    brisk = extract_features(speech(syll_hz=4.8, seed=1), SR)
    hesitant = extract_features(speech(syll_hz=3.0, pause_every=3, pause_s=0.6, seed=1, dur=8.0), SR)
    assert brisk["pause_ratio"] == 0.0
    assert hesitant["pause_ratio"] > 0.1


def test_too_little_speech_is_not_judged_at_all():
    e = estimate(speech(dur=1.0, seed=1), SR)
    assert not e.sufficient and not e.votes_senior and e.features == {}


def test_silence_is_not_judged():
    import numpy as np

    assert not estimate(np.zeros(SR * 5, dtype=np.float32), SR).sufficient


def test_contributions_explain_the_score():
    e = estimate(older(0, 190), SR)
    assert e.sufficient
    assert sum(e.contributions.values()) == pytest.approx(e.score, abs=0.01)


# --------------------------------------------------- evidence over a call


def _est(score, sufficient=True):
    return SeniorEstimate(sufficient, score, {}, {})


def test_one_clip_never_decides():
    ev = SeniorEvidence()
    assert ev.add(_est(0.9)) is False
    assert not ev.senior


def test_two_agreeing_clips_flip_the_call_and_only_once():
    ev = SeniorEvidence()
    ev.add(_est(0.8))
    assert ev.add(_est(0.8)) is True
    assert ev.senior and ev.reason == "acoustic"
    assert ev.add(_est(0.8)) is False


def test_disagreeing_evidence_does_not_flip_the_call():
    ev = SeniorEvidence()
    for s in (0.8, 0.02, 0.05, 0.03, 0.04):
        ev.add(_est(s))
    assert not ev.senior


def test_clips_without_enough_speech_are_ignored():
    ev = SeniorEvidence()
    for _ in range(5):
        ev.add(_est(0.9, sufficient=False))
    assert not ev.senior


def test_senior_is_sticky_once_detected():
    ev = SeniorEvidence()
    ev.note_explicit("requested_slower")
    for _ in range(4):
        ev.add(_est(0.0))
    assert ev.senior and ev.reason == "requested_slower"


def test_an_explicit_request_decides_immediately_and_reports_the_change():
    ev = SeniorEvidence()
    assert ev.note_explicit("self_described") is True
    assert ev.note_explicit("self_described") is False


_SINGLE_TRAITS = (dict(syll_hz=2.3), dict(tremor_hz=6.0, tremor_st=0.5), dict(breath=0.10, jitter_st=0.30))


def test_the_decision_threshold_is_above_every_single_trait_score():
    # Measured, not restated: vary ONE trait at a time on an otherwise young
    # voice, and the decision threshold must clear the highest score any of
    # them earns -- a single trait alone must never make a caller "senior".
    base = dict(syll_hz=4.8, jitter_st=0.05, breath=0.02)
    scores = [
        estimate(speech(seed=sd, f0=f0, **{**base, **kw}), SR).score
        for kw in _SINGLE_TRAITS
        for sd in (0, 1, 3)
        for f0 in (130, 190, 230)
    ]
    assert SENIOR_SCORE > max(scores), max(scores)


def test_a_third_party_description_is_not_a_self_description():
    assert explicit_senior_cue("my father is a senior citizen", "en") is None
    assert explicit_senior_cue("my mother is hard of hearing", "en") is None
    assert explicit_senior_cue("मेरे पिता वरिष्ठ नागरिक हैं", "hi") is None
    assert explicit_senior_cue("I am a senior citizen", "en") == "self_described"
    assert explicit_senior_cue("मुझे कम सुनाई देता है", "hi") == "self_described"


def test_a_non_numeric_stated_age_is_ignored_not_a_crash():
    assert stated_age_is_senior("seventy") is False
    assert stated_age_is_senior("72") is True


# ------------------------------------------------------ explicit cues


@pytest.mark.parametrize(
    "text,lang,expected",
    [
        ("could you please speak slowly", "en", "requested_slower"),
        ("slowly please, I can't follow", "en", "requested_slower"),
        ("I am an old man", "en", "self_described"),
        ("I'm a senior citizen", "en", "self_described"),
        ("my hearing is not good", "en", "self_described"),
        ("ধীরে বলুন", "bn", "requested_slower"),
        ("আমি বয়স্ক মানুষ", "bn", "self_described"),
        ("धीरे बोलिए", "hi", "requested_slower"),
        ("मैं बुज़ुर्ग हूँ", "hi", "self_described"),
        ("what is the price of a blood test", "en", None),
        ("I want to book for tomorrow", "en", None),
        ("", "en", None),
    ],
)
def test_explicit_cues_in_all_three_languages(text, lang, expected):
    assert explicit_senior_cue(text, lang) == expected


def test_a_cue_in_another_language_than_the_current_one_still_counts():
    # callers switch language mid-call
    assert explicit_senior_cue("please speak slowly", "bn") == "requested_slower"


@pytest.mark.parametrize(
    "age,relationship,expected",
    [
        (72, None, True),
        (60, "self", True),
        (59, "self", False),
        (80, "father", False),
        (80, "mother", False),
        (None, None, False),
        (65, "", True),
    ],
)
def test_only_the_callers_own_stated_age_counts(age, relationship, expected):
    assert stated_age_is_senior(age, relationship) is expected
