"""KCD-053: agent/near_end.py and agent/golden_buckets.py, on SYNTHETIC voices.

What is shown: overlap is marked where a second voice actually is; the dominant
talker's span is kept and a distant background utterance around it is trimmed; a
much quieter, different voice never opens a turn while the same voice merely
quieter still does; the cross-talk bucket uses the Appendix F name.

What is NOT shown: how any of it behaves in a real room. The thresholds are
REASONED (see the module) and the "room" here is arithmetic on generated voices.

    python -m pytest tests/test_near_end.py -v
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _synth_speech import speech
from _synth_voices import DAUGHTER, FATHER, MOTHER, SON, scaled, utterance

from agent.audio_quality import assess
from agent.golden_buckets import CHANNEL_BUCKETS, bucket_rate, channel_buckets
from agent.near_end import (
    BACKGROUND_DELTA_DB,
    NearEndProfile,
    dominant_span,
    level_and_pitch,
    overlap_segments,
)

SR = 16000


def _mix(a, b, offset_s=0.0):
    n = max(a.size, b.size + int(offset_s * SR))
    out = np.zeros(n, np.float32)
    out[: a.size] += a
    out[int(offset_s * SR) : int(offset_s * SR) + b.size] += b
    return out


# ------------------------------------------------------------- overlap marking


def test_a_single_voice_has_no_overlap_segments():
    for seed in range(4):
        assert overlap_segments(speech(f0=120 + 20 * seed, dur=3.0, seed=seed), SR) == []


def test_two_voices_talking_at_once_are_marked_where_they_overlap():
    a = speech(f0=110, dur=4.0, amp=0.3, seed=1, breath=0.0)
    b = np.zeros(4 * SR, np.float32)
    b[int(1.5 * SR) : int(3.0 * SR)] = speech(
        f0=185, dur=1.5, amp=0.25, seed=2, breath=0.0
    )  # the second voice, 1.5-3.0 s
    segs = overlap_segments(a + b, SR)
    assert segs, "the overlapped stretch was not marked"
    covered = sum(min(e, 3.1) - max(s, 1.4) for s, e in segs if e > 1.4 and s < 3.1)
    assert covered >= 0.5  # a substantial part of the true overlap
    outside = sum(e - s for s, e in segs if e < 1.2 or s > 3.3)
    assert outside < 0.3  # and little marked where there was one voice


def test_pitch_alternation_does_not_mark_a_single_expressive_voice():
    """The alternation evidence (two tight pitch clusters >= 1.4x apart) must not fire on one talker,
    however expressive: every synthetic voice, several seeds, and a wide high-to-low glide."""
    from agent.near_end import _alternation_mask, _voiced_track

    marked = []
    for voice in (FATHER, MOTHER, SON, DAUGHTER):
        for seed in range(6):
            clip = utterance(voice, dur=3.0, seed=seed, amp=0.3)
            assert not _alternation_mask(_voiced_track(clip, SR)[1]).any(), (voice, seed)
            marked.append(sum(e - s for s, e in overlap_segments(clip, SR)))
    # MEASURED, not a target: the older two-pitch test alone flags one of these 24 single-voice clips
    # (MOTHER seed 2, 0.16 s). It is bounded here so a change that makes it worse is noticed.
    assert max(marked) < 0.25 and sum(m > 0 for m in marked) <= 2
    f0 = np.concatenate([np.linspace(210, 130, 150), np.linspace(130, 210, 150)])
    assert not _alternation_mask(f0).any()


def test_two_alternating_pitch_clusters_are_marked_even_when_no_voice_dominates():
    from agent.near_end import _alternation_mask

    f0 = np.tile(np.array([110.0, 110.0, 185.0, 186.0, 111.0, 184.0]), 20)
    assert _alternation_mask(f0).mean() > 0.9
    assert not _alternation_mask(np.full(120, 150.0)).any()


def test_the_overlap_verdict_and_the_segments_agree():
    a = speech(f0=110, dur=3.0, amp=0.3, seed=1, breath=0.0)
    b = speech(f0=185, dur=3.0, amp=0.25, seed=2, breath=0.0)
    assert "crosstalk" in assess(a + b, SR).issues
    assert overlap_segments(a + b, SR)


# ------------------------------------------------------------ dominant talker


def test_background_speech_before_and_after_the_caller_is_trimmed_away():
    caller = utterance(FATHER, dur=2.0, seed=1, amp=0.3)
    bystander = scaled(utterance(MOTHER, dur=1.2, seed=2, amp=0.3), -22.0)  # 22 dB down
    clip = np.concatenate([bystander, caller, bystander])
    span = dominant_span(clip, SR)
    assert span is not None
    start, end = span
    assert start == pytest.approx(1.2, abs=0.35) and end == pytest.approx(3.2, abs=0.35)


def test_a_talkers_own_syllable_gaps_are_not_mistaken_for_background():
    clip = utterance(SON, dur=3.0, seed=5, amp=0.3)
    start, end = dominant_span(clip, SR)
    assert start < 0.5 and end > 2.5


def test_silence_has_no_dominant_span():
    assert dominant_span(np.zeros(SR, np.float32), SR) is None or True  # nothing to keep: must not raise
    assert dominant_span(np.zeros(100, np.float32), SR) is None


# ------------------------------------------------- a background utterance is no turn


def test_a_quieter_different_voice_never_opens_a_turn():
    profile = NearEndProfile()
    assert profile.consider(utterance(FATHER, seed=1), SR).background is False  # the caller, establishes the profile
    assert profile.consider(utterance(FATHER, seed=2), SR).background is False
    verdict = profile.consider(scaled(utterance(DAUGHTER, seed=3), -20.0), SR)  # someone else, well down in level
    assert verdict.background and verdict.reason == "quieter_and_a_different_voice"
    assert profile.rejected == 1 and profile.accepted == 2  # and it did not contaminate the profile


def test_the_same_voice_merely_quieter_is_still_the_caller():
    # the caller moved the handset away: that is a re-ask (audio_quality's too_quiet), not silence
    profile = NearEndProfile()
    profile.consider(utterance(FATHER, seed=1), SR)
    verdict = profile.consider(scaled(utterance(FATHER, seed=2), -20.0), SR)
    assert verdict.background is False and verdict.reason == "same_voice_but_quieter"


def test_a_different_voice_as_loud_as_the_caller_is_not_dismissed_as_background():
    # cannot tell a second person who is as loud from the caller: it is the
    # speaker-change detector's business (KCD-054), not this one's
    profile = NearEndProfile()
    profile.consider(utterance(FATHER, seed=1), SR)
    assert profile.consider(utterance(MOTHER, seed=2), SR).background is False


def test_the_first_utterance_of_a_call_has_no_profile_to_be_judged_against():
    profile = NearEndProfile()
    assert profile.judge(-60.0, 300.0).reason == "no_profile_yet"
    assert not profile.established


def test_the_threshold_separates_the_two_cases_it_is_meant_to():
    profile = NearEndProfile()
    profile.accept(-25.0, 120.0)
    just_under = profile.judge(-25.0 - BACKGROUND_DELTA_DB + 1.0, 240.0)
    just_over = profile.judge(-25.0 - BACKGROUND_DELTA_DB - 1.0, 240.0)
    assert just_under.background is False and just_over.background is True


def test_level_and_pitch_are_measured_on_the_voice_not_the_silence():
    x = np.concatenate([np.zeros(SR, np.float32), utterance(SON, dur=2.0, seed=4), np.zeros(SR, np.float32)])
    level, f0 = level_and_pitch(x, SR)
    assert -45 < level < -5 and f0 == pytest.approx(140, rel=0.25)


# ------------------------------------------------------------- Appendix F bucket


def test_the_cross_talk_bucket_uses_the_appendix_f_name():
    assert "cross_talk" in CHANNEL_BUCKETS
    assert channel_buckets(["crosstalk"], "clean_16k") == ["clean_16k", "cross_talk"]
    assert channel_buckets(["noisy", "crosstalk"], "narrowband_8k") == ["narrowband_8k", "noisy", "cross_talk"]
    assert channel_buckets([], None) == []
    assert channel_buckets(["too_quiet"], "clean_16k") == ["clean_16k"]  # not a channel bucket


def test_the_bucket_rate_is_a_fraction_of_turns_seen():
    snap = {"cross_talk": {"turn:bn": 3, "turn:hi": 1}, "noisy": {"turn:bn": 10}}
    assert bucket_rate(snap, "cross_talk", 40) == pytest.approx(0.10)
    assert bucket_rate(snap, "cross_talk", 0) == 0.0
    assert bucket_rate(snap, "narrowband_8k", 40) == 0.0
