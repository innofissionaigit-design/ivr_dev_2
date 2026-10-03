"""Senior care (KCD-512) and the adaptive pause (KCD-047).

python -m pytest tests/test_senior_care_and_pause.py -v
"""

import math
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import senior_care as sc
from agent.endpointing import EndpointConfig
from agent.pause_profile import (
    BUMP_MAX,
    FALSE_CUT_WINDOW_S,
    MIN_GAPS,
    SILENCE_MAX_S,
    SILENCE_MIN_S,
    PauseProfile,
)
from agent.persona import is_clean

# ======================================================================== senior care


def test_sixty_and_over_is_a_senior_citizen():
    assert sc.is_senior(60) and sc.is_senior(78) and not sc.is_senior(59) and not sc.is_senior(None)


def test_senior_care_starts_only_when_the_registered_age_says_so_and_only_once():
    k = sc.KindnessPlanner()
    assert k.activate(45) is False and not k.active
    assert k.activate(72) is True and k.active
    assert k.activate(72) is False  # already on: not "newly"
    assert k.opening("en") == sc.OPENING["en"] and k.opening("en") is None


def test_an_inactive_planner_adds_nothing():
    k = sc.KindnessPlanner()
    assert k.opening("bn") is None and k.patience("hi") is None and k.warm_ack("en") is None
    assert k.decorate("Your appointment is on Monday morning.", "en") == "Your appointment is on Monday morning."


def test_the_warm_closing_goes_on_every_second_substantive_reply_and_never_on_a_short_one():
    k = sc.KindnessPlanner()
    k.activate(70)
    reply = "Your appointment is on Monday morning."
    outs = [k.decorate(reply, "en") for _ in range(4)]
    assert [o.endswith(sc.CLOSING["en"]) for o in outs] == [False, True, False, True]
    assert k.decorate("Yes.", "en") == "Yes."


def test_a_seniors_reask_carries_patience():
    k = sc.KindnessPlanner()
    k.activate(65)
    assert k.patience("en") == sc.PATIENCE["en"] and k.patience_given == 1


@pytest.mark.parametrize("table", [sc.OPENING, sc.CLOSING, sc.PATIENCE, sc.WARM_ACK])
def test_every_senior_line_exists_in_three_languages_has_no_fact_and_passes_the_persona(table):
    assert set(table) == {"bn", "hi", "en"}
    for lang, text in table.items():
        assert not any(ch.isdigit() for ch in text) and "?" not in text
        assert is_clean(text, lang), (lang, text)


# ======================================================================== adaptive pause


def sample_gap(rng, median, sigma=0.45):
    return float(rng.lognormal(math.log(median), sigma))


def simulate(median_gap, adaptive, utterances=80, seed=3):
    """A caller says utterances of 4 pauses each. A pause longer than the wait CUTS the turn (a false cut);
    a shorter one is observed. Returns (false-cut rate over pauses, the final silence wait)."""
    rng = np.random.default_rng(seed)
    prof, base = PauseProfile(), EndpointConfig()
    cuts = pauses = 0
    for _ in range(utterances):
        confirm = (prof.config(base) if adaptive else base).silence_confirm_s
        observed = []
        for _ in range(4):
            g = sample_gap(rng, median_gap)
            pauses += 1
            if g > confirm:
                cuts += 1
                if adaptive:
                    prof.note_turn_cut(g - confirm, agent_spoke_between=False)
                break
            observed.append(g)
        spans, t = [], 0.0
        for g in observed:
            spans.append({"start": t, "end": t + 0.6})
            t += 0.6 + g
        spans.append({"start": t, "end": t + 0.6})
        if adaptive:
            prof.observe_utterance(spans)
    return cuts / pauses, (prof.config(base) if adaptive else base).silence_confirm_s


def test_without_enough_gaps_the_base_configuration_is_used_unchanged():
    p, base = PauseProfile(), EndpointConfig()
    assert p.config(base) is base
    for _ in range(MIN_GAPS - 1):
        p.gaps.append(0.3)
    assert p.config(base) is base


def test_a_slow_deliberate_speaker_is_cut_off_far_less_than_with_the_fixed_threshold():
    fixed_rate, _ = simulate(median_gap=0.9, adaptive=False)
    adaptive_rate, wait = simulate(median_gap=0.9, adaptive=True)
    assert fixed_rate > 0.30  # the fixed 1.0 s cuts this caller constantly
    assert adaptive_rate < 0.6 * fixed_rate  # learning their rhythm cuts that by more than 40%
    assert wait > 1.2  # ...by waiting longer FOR THIS CALLER


def test_a_brisk_speaker_is_answered_sooner_than_the_fixed_threshold_allows():
    rate, wait = simulate(median_gap=0.15, adaptive=True)
    assert wait <= 0.7 < EndpointConfig().silence_confirm_s and rate < 0.03


def test_the_wait_is_always_inside_its_bounds_even_for_extreme_callers():
    for median in (0.05, 0.15, 1.5, 2.5):
        _, wait = simulate(median, adaptive=True, utterances=40)
        assert SILENCE_MIN_S <= wait <= SILENCE_MAX_S


def test_a_cut_followed_at_once_by_more_speech_and_no_agent_reply_raises_the_wait():
    p, base = PauseProfile(), EndpointConfig()
    p.note_turn_cut(0.3, agent_spoke_between=False)
    assert p.false_cuts == 1 and p.bump > 1.0 and p.config(base).silence_confirm_s > base.silence_confirm_s


def test_speech_that_resumes_after_the_agent_answered_or_much_later_is_not_a_false_cut():
    p = PauseProfile()
    p.note_turn_cut(0.3, agent_spoke_between=True)
    p.note_turn_cut(FALSE_CUT_WINDOW_S + 5, agent_spoke_between=False)
    assert p.false_cuts == 0 and p.bump == 1.0


def test_the_raise_is_capped_and_relaxes_after_a_run_of_clean_turns():
    p = PauseProfile()
    for _ in range(30):
        p.note_turn_cut(0.1, False)
    assert p.bump == pytest.approx(BUMP_MAX)
    for _ in range(60):
        p.note_turn_cut(None, True)
    assert p.bump < BUMP_MAX


def test_only_pauses_inside_one_utterance_are_recorded_and_absurd_ones_are_ignored():
    p = PauseProfile()
    spans = [
        {"start": 0.0, "end": 0.5},
        {"start": 0.55, "end": 1.0},  # 0.05 s: the detector's own hangover
        {"start": 1.6, "end": 2.0},
        {"start": 6.0, "end": 6.5},
    ]  # 0.6 s counts, 4.0 s is not a pause
    assert p.observe_utterance(spans) == 1 and list(p.gaps) == [pytest.approx(0.6)]


def test_the_snapshot_reports_what_was_learned():
    p = PauseProfile()
    for g in (0.2, 0.3, 0.25, 0.9):
        p.gaps.append(g)
    snap = p.snapshot()
    assert snap["adapted"] and snap["gaps"] == 4 and snap["p90_s"] >= snap["median_s"]
