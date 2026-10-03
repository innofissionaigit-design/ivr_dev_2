"""KCD-046 / KCD-048 / KCD-049: agent/endpointing.py, on synthetic audio.

KCD-046 asks each false-turn-end guard to carry a test "with synthetic audio
demonstrating the failure it prevents". The spans here come from
endpointing.energy_spans() run over generated audio -- a stand-in for Silero (which
needs torch and a model) that only has to be a consistent yardstick.

Every threshold under test is REASONED (agent/endpointing.THRESHOLD_STATUS); these
tests prove the LOGIC does what it says, not that 0.45 s or 1.0 s is right for
Kolkata callers -- KCD-047's calibration tool exists for that.

    python -m pytest tests/test_endpointing.py -v
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _synth_speech import SR, speech

from agent.endpointing import (
    EndpointConfig,
    classify_completeness,
    decide,
    energy_spans,
)

CFG = EndpointConfig()


def _silence(seconds):
    return np.zeros(int(seconds * SR), np.float32)


def _decide_at(audio, t, cfg=CFG, **kw):
    """The decision a detector would make when `t` seconds of `audio` have arrived."""
    seen = audio[: int(t * SR)]
    return decide(energy_spans(seen, SR), seen.size / SR, cfg, **kw)


# ============================================================ KCD-046 guards


def test_a_cough_or_click_is_not_a_finished_sentence():
    # 0.12 s burst, then a long silence. Without the minimum-duration guard the
    # silence that follows would end a "turn" of nothing and the agent would
    # answer nobody.
    t = np.arange(int(0.12 * SR)) / SR
    burst = (0.4 * np.hanning(t.size) * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # a cough-length burst
    audio = np.concatenate([_silence(0.5), burst, _silence(3.0)])
    d = _decide_at(audio, 3.6)
    assert d.utterance_end_s is None and d.reason == "transient"
    no_guard = EndpointConfig(min_speech_s=0.0)
    assert _decide_at(audio, 3.6, no_guard).utterance_end_s is not None  # the failure the guard prevents


def test_real_speech_does_end_a_turn_once_the_silence_is_long_enough():
    audio = np.concatenate([_silence(0.3), speech(f0=130, dur=1.6, amp=0.3, syll_hz=3.0, breath=0.02), _silence(3.0)])
    d = _decide_at(audio, audio.size / SR)
    assert d.utterance_end_s is not None and d.reason == "silence_confirmed"


def test_the_tail_guard_stops_decode_lag_being_read_as_silence():
    # The caller is still talking, but the freshest 0.3 s has not arrived yet, so
    # the buffer looks quiet at its end. Speech runs to 2.0 s; we are at 3.1 s
    # of audio -- 1.1 s of apparent silence, more than the 1.0 s baseline.
    audio = np.concatenate([_silence(0.2), speech(f0=130, dur=1.8, amp=0.3, syll_hz=3.0, breath=0.02), _silence(1.1)])
    with_guard = _decide_at(audio, 3.1)
    assert with_guard.utterance_end_s is None  # 3.1 - 0.3 - 2.0 = 0.8 < 1.0
    no_guard = EndpointConfig(tail_guard_s=0.0)
    assert _decide_at(audio, 3.1, no_guard).utterance_end_s is not None  # without it: a premature cut


def test_a_runaway_utterance_is_force_cut():
    long_talk = speech(f0=125, dur=24.0, amp=0.3, syll_hz=3.5, breath=0.02, seed=3)
    d = _decide_at(long_talk, 24.0)
    assert d.utterance_end_s is not None and d.reason == "max_utterance"
    assert _decide_at(long_talk, 15.0).utterance_end_s is None  # under the limit, still talking


# ============================================================ KCD-048 semantic


@pytest.mark.parametrize(
    "text,lang,expected",
    [
        ("what is the price of the CBC test?", "en", "complete"),
        ("I want to book an appointment for tomorrow", "en", "complete"),
        ("yes", "en", "complete"),
        ("my number is nine eight and", "en", "incomplete"),
        ("I would like to know about the", "en", "incomplete"),
        ("uh", "en", "incomplete"),
        ("9 8 3 0", "en", "incomplete"),  # half a phone number
        ("9 8 3 0 6 1 4 2 5 5", "en", "complete"),
        ("আমি একটা অ্যাপয়েন্টমেন্ট চাই", "bn", "complete"),
        ("ডাক্তার সেনের সঙ্গে আর", "bn", "incomplete"),
        ("मुझे कल का", "hi", "incomplete"),
        ("मुझे कल आना है", "hi", "complete"),
        ("", "en", "unknown"),
        ("blood sugar", "en", "unknown"),  # a noun phrase: no verdict, baseline applies
    ],
)
def test_completeness_of_a_partial_transcript(text, lang, expected):
    assert classify_completeness(text, lang) == expected


def _talk_then_silence(silence_s):
    return np.concatenate(
        [_silence(0.2), speech(f0=130, dur=1.6, amp=0.3, syll_hz=3.0, breath=0.02), _silence(silence_s)]
    )


def test_a_finished_question_ends_the_turn_sooner_than_the_baseline():
    audio = _talk_then_silence(2.0)

    # find the earliest wake time at which each policy commits
    def first_commit(completeness, semantic=True):
        for t in np.arange(1.9, 4.0, 0.01):
            if _decide_at(audio, t, completeness=completeness, semantic=semantic).utterance_end_s is not None:
                return t
        return None

    baseline = first_commit(None, semantic=False)
    complete = first_commit("complete")
    assert complete is not None and baseline is not None
    recovered = baseline - complete
    assert recovered == pytest.approx(CFG.silence_confirm_s - CFG.complete_confirm_s, abs=0.03)  # ~0.55 s


def test_an_unfinished_utterance_is_given_LONGER_than_the_baseline_never_less():
    audio = _talk_then_silence(2.5)
    # at 1.0 s of confirmed silence the baseline would already have cut...
    t = 0.2 + 1.6 + CFG.tail_guard_s + 1.05
    assert _decide_at(audio, t, semantic=False).utterance_end_s is not None
    # ...but an unfinished utterance is still being waited for
    assert _decide_at(audio, t, completeness="incomplete").utterance_end_s is None
    later = 0.2 + 1.6 + CFG.tail_guard_s + CFG.incomplete_confirm_s + 0.05
    assert _decide_at(audio, later, completeness="incomplete").utterance_end_s is not None


def test_false_cuts_do_not_increase_against_the_baseline():
    # the property the story asks to hold: over pauses of every length inside an
    # utterance, semantic endpointing cuts mid-utterance NO MORE OFTEN than the
    # baseline when the transcript verdict is honest (a pause inside an
    # unfinished sentence is "incomplete"; a real end is "complete").
    rng = np.random.default_rng(0)
    base_cuts = sem_cuts = 0
    for pause in rng.uniform(0.3, 1.4, 60):
        part1 = speech(f0=130, dur=1.0, amp=0.3, syll_hz=3.0, breath=0.02, seed=int(pause * 1000) % 50)
        part2 = speech(f0=130, dur=1.0, amp=0.3, syll_hz=3.0, breath=0.02, seed=7)
        audio = np.concatenate([_silence(0.2), part1, _silence(pause), part2, _silence(2.5)])
        t_pause_seen = 0.2 + 1.0 + pause  # the moment part 2 is about to start
        seen = audio[: int(t_pause_seen * SR) - int(0.05 * SR)]
        spans = energy_spans(seen, SR)
        b = decide(spans, seen.size / SR, CFG, semantic=False)
        s = decide(spans, seen.size / SR, CFG, completeness="incomplete", semantic=True)
        base_cuts += b.utterance_end_s is not None
        sem_cuts += s.utterance_end_s is not None
    assert sem_cuts <= base_cuts
    assert base_cuts > 0  # the scenario does produce baseline cuts to compare against


# ============================================================ KCD-049 scheduling


def _simulate_commit_lag(end_s, wake_policy, chunk_s=0.032):
    """How long after the EARLIEST moment the turn could be committed does the
    loop actually commit it? `wake_policy(decision) -> seconds until next wake`."""
    cfg = EndpointConfig(tail_guard_s=0.1)
    ideal = end_s + cfg.tail_guard_s + cfg.silence_confirm_s
    t = 0.5
    while t < end_s + 5:
        arrived = np.floor(t / chunk_s) * chunk_s  # audio arrives in client-sized blocks
        spans = [{"start": 0.2, "end": end_s}] if arrived > end_s else [{"start": 0.2, "end": min(arrived, end_s)}]
        d = decide(spans, arrived, cfg, semantic=False)
        if d.utterance_end_s is not None:
            return t - ideal
        t += wake_policy(d)
    raise AssertionError("never committed")


def test_a_fixed_half_second_poll_costs_a_quarter_second_on_average():
    rng = np.random.default_rng(1)
    lags = [_simulate_commit_lag(e, lambda d: 0.5) for e in rng.uniform(1.0, 3.0, 400)]
    assert 0.2 < np.mean(lags) < 0.3
    assert np.percentile(lags, 95) > 0.4  # the "250 ms average" the story names


def test_audio_driven_wake_ups_bring_granularity_under_50_ms_at_p95():
    def scheduled(d):
        base = 0.2 if d.had_any_speech else 0.5
        if d.commit_after_s is None:
            return base
        return min(base, max(0.02, d.commit_after_s + 0.005))

    rng = np.random.default_rng(1)
    lags = [_simulate_commit_lag(e, scheduled) for e in rng.uniform(1.0, 3.0, 400)]
    assert np.percentile(lags, 95) < 0.050, np.percentile(lags, 95)
    assert np.mean(lags) < 0.030


def test_a_decision_says_when_it_could_next_change():
    spans = [{"start": 0.2, "end": 1.0}]
    d = decide(spans, 1.6, EndpointConfig(), semantic=False)  # 0.3 s of confirmed silence so far
    assert d.utterance_end_s is None
    assert d.commit_after_s == pytest.approx(EndpointConfig().silence_confirm_s - (1.6 - 0.3 - 1.0), abs=1e-6)


def test_speculation_is_requested_only_once_there_is_enough_silence_to_be_worth_it():
    spans = [{"start": 0.2, "end": 1.0}]
    early = decide(spans, 1.1 + 0.1, EndpointConfig(), semantic=True)  # 0.1 s of silence
    assert early.speculate is False
    later = decide(spans, 1.0 + 0.3 + 0.35, EndpointConfig(), semantic=True)  # 0.35 s
    assert later.speculate is True


def test_a_calibration_file_replaces_the_reasoned_values(tmp_path):
    from agent.endpointing import config_from_json

    f = tmp_path / "cal.json"
    f.write_text('{"status": "MEASURED", "config": {"silence_confirm_s": 0.7, "bogus": 1}}', encoding="utf-8")
    cfg = config_from_json(str(f))
    assert cfg.silence_confirm_s == 0.7 and cfg.tail_guard_s == EndpointConfig().tail_guard_s
