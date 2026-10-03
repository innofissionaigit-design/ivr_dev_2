"""The empathetic re-ask policy (agent/reask_policy.py), Appendix C's
speech policy (agent/speech_policy.py, KCD-149), acknowledgement templates
(agent/acknowledgement.py, KCD-155) and phone/ID grouping
(agent/figures.py, KCD-157).

    python -m pytest tests/test_reask_and_policy.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.acknowledgement import (
    REVIEW_STATUS,
    acknowledgement_for,
    pending_review_count,
    select_acknowledgement,
)
from agent.call_state import apply_caller_state, new_call_state
from agent.figures import id_groups, phone_groups, speak_grouped
from agent.phrases import PHRASES, phrase
from agent.reask_policy import ReaskTracker
from agent.speech_policy import (
    check_reply,
    count_questions,
    derive_policy,
    effective_rate,
    limit_questions,
)

# ================================================================ re-ask


def test_a_usable_turn_proceeds_and_resets_the_count():
    t = ReaskTracker()
    t.decide(asr_empty=True)
    assert t.consecutive == 1
    assert t.decide().action == "proceed"
    assert t.consecutive == 0


def test_acoustic_trouble_alone_never_triggers_a_reask():
    # The recogniser understood the caller despite the noise: nagging them
    # would be its own failure.
    d = ReaskTracker().decide(audio_issues=["noisy", "too_quiet"])
    assert d.action == "proceed"


@pytest.mark.parametrize(
    "issues,expected_key",
    [
        (["too_quiet"], "reask_low_volume"),
        (["crosstalk"], "reask_crosstalk"),
        (["noisy"], "reask_noisy"),
        (["unvoiced_mumble"], "reask_mumbled"),
        (["noisy", "too_quiet"], "reask_low_volume"),  # speaking up fixes the faint voice first
        (["noisy", "crosstalk"], "reask_crosstalk"),
        ([], "reask_generic"),
    ],
)
def test_the_reask_names_the_actual_problem(issues, expected_key):
    d = ReaskTracker().decide(asr_empty=True, audio_issues=issues)
    assert d.action == "reask" and d.phrase_key == expected_key


def test_a_jumbled_transcript_gets_the_gentle_mumble_wording():
    d = ReaskTracker().decide(transcript_issue="shattered")
    assert d.action == "reask" and d.phrase_key == "reask_mumbled"


def test_an_unidentifiable_language_is_asked_about_before_any_handoff():
    # Previously this went straight to a human on the first ambiguous turn.
    d = ReaskTracker().decide(language_ambiguous=True)
    assert d.action == "reask" and d.reason == "unclear"


def test_the_caller_is_asked_twice_then_routed_to_a_human_with_an_apology():
    t = ReaskTracker(max_reasks=2)
    first, second, third = (t.decide(asr_empty=True) for _ in range(3))
    assert [first.action, second.action, third.action] == ["reask", "reask", "handoff"]
    assert third.phrase_key == "reask_final"


def test_success_between_failures_starts_the_count_over():
    t = ReaskTracker(max_reasks=2)
    t.decide(asr_empty=True)
    t.decide(asr_empty=True)
    t.decide()  # understood
    assert t.decide(asr_empty=True).action == "reask"


def test_a_second_failure_marks_the_caller_confused_so_the_policy_slows_down():
    t = ReaskTracker()
    assert t.decide(asr_empty=True).mark_confused is False
    assert t.decide(asr_empty=True).mark_confused is True


def test_every_reask_phrase_exists_in_every_language():
    keys = {"reask_low_volume", "reask_noisy", "reask_crosstalk", "reask_mumbled", "reask_generic", "reask_final"}
    for lang in ("bn", "hi", "en"):
        assert keys <= set(PHRASES[lang]), lang


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_reask_wording_never_blames_the_caller_and_asks_one_question(lang):
    for key in ("reask_low_volume", "reask_noisy", "reask_crosstalk", "reask_mumbled", "reask_generic"):
        text = phrase(key, lang)
        assert count_questions(text) == 1, (lang, key)
        assert ":" not in text and "[" not in text, "spoken text must carry no label punctuation"


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_reask_phrases_satisfy_the_senior_policy(lang):
    senior = derive_policy("neutral", senior=True)
    for key in ("reask_low_volume", "reask_noisy", "reask_crosstalk", "reask_mumbled", "reask_generic", "reask_final"):
        assert check_reply(phrase(key, lang), senior) == [], (lang, key, phrase(key, lang))


# ============================================================ speech policy


def test_appendix_c_rows_are_encoded_exactly():
    normal = derive_policy("neutral")
    assert (normal.speech_rate, normal.sentence_length, normal.questions_per_turn, normal.confirmation) == (
        1.0,
        "default",
        2,
        "implicit",
    )

    senior = derive_policy("neutral", senior=True)
    assert senior.speech_rate < 1.0 and senior.sentence_length == "short"
    assert senior.questions_per_turn == 1 and senior.confirmation == "explicit_high_frequency"
    assert senior.interruption_tolerance == "high" and senior.escalation_threshold == "lowered"

    distress = derive_policy("distressed")
    assert distress.acknowledge_first and distress.interruption_tolerance == "very_high"
    assert distress.escalation_threshold == "low" and distress.questions_per_turn == 1

    anger = derive_policy("angry")
    assert 0.85 < anger.speech_rate < 1.0, "Appendix C: slow-NORMAL, between slow and normal"
    assert anger.offer_human and anger.acknowledge_first

    confusion = derive_policy("confused")
    assert confusion.sentence_length == "very_short" and confusion.confirmation == "explicit_repeat_back"


def test_emergency_stops_the_normal_flow():
    p = derive_policy("emergency")
    assert p.emergency and p.questions_per_turn == 0
    assert check_reply("Any reply at all?", p) == []  # the protocol owns the words


def test_an_unknown_caller_state_falls_back_to_normal_never_raises():
    assert derive_policy("no-such-state") == derive_policy("neutral")


def test_states_combine_to_the_most_conservative_value_of_every_field():
    p = derive_policy("distressed", senior=True)
    assert p.speech_rate == min(derive_policy("distressed").speech_rate, derive_policy("neutral", True).speech_rate)
    assert p.questions_per_turn == 1
    assert p.interruption_tolerance == "very_high"  # distress's, stronger than senior's
    assert p.escalation_threshold == "low"
    assert p.acknowledge_first


def test_senior_mode_echoes_each_value_and_normal_mode_does_not():
    assert derive_policy("neutral", senior=True).confirm_each_slot
    assert not derive_policy("neutral").confirm_each_slot


def test_the_planner_gets_the_policy_as_hard_constraints():
    c = derive_policy("confused").hard_constraints()
    assert c["questions_per_turn"] == 1 and c["max_sentence_words"] == 8 and c["speech_rate"] < 1.0


def test_check_reply_finds_several_questions_and_overlong_sentences():
    senior = derive_policy("neutral", senior=True)
    problems = check_reply("Which doctor? Which day? Which time?", senior)
    assert any("3 questions" in p for p in problems)
    long_sentence = " ".join(["word"] * 20) + "."
    assert any("exceeds" in p for p in check_reply(long_sentence, senior))
    assert check_reply("Which doctor would you like?", senior) == []


def test_limit_questions_drops_only_the_surplus_question_never_a_statement():
    text = "Your price is 500 rupees. Which day? Which time?"
    out = limit_questions(text, 1)
    assert "500 rupees" in out and out.count("?") == 1 and "Which day?" in out
    assert limit_questions("One question only?", 1) == "One question only?"


def test_a_figure_is_spoken_slower_still_for_a_senior_caller():
    senior = derive_policy("neutral", senior=True)
    assert effective_rate(senior, 0.8) < effective_rate(senior)
    assert effective_rate(derive_policy("neutral"), 0.8) == pytest.approx(0.8)
    assert effective_rate(derive_policy("neutral")) == 1.0


def test_apply_caller_state_drives_every_call_state_delivery_field_from_the_table():
    s = new_call_state()
    assert s.speech_rate == 1.0 and s.response_length == "normal"
    apply_caller_state(s, senior=True)
    assert s.senior and s.speech_rate < 1.0 and s.response_length == "short"
    assert s.interruption_tolerance == "patient" and s.escalation_threshold == "lowered"
    apply_caller_state(s, caller_state="confused")
    assert s.caller_state == "confused" and s.senior, "senior is sticky across later states"
    apply_caller_state(s, caller_state="not-a-state")
    assert s.caller_state == "neutral"


# ========================================================= acknowledgement


@pytest.mark.parametrize("state", ["distressed", "angry", "confused"])
@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_every_acknowledgement_state_exists_in_every_language(state, lang):
    text = acknowledgement_for(state, lang)
    assert text and ":" not in text and "[" not in text


def test_an_acknowledgement_is_selected_only_when_the_policy_asks_for_one():
    assert select_acknowledgement(derive_policy("neutral"), "neutral", "en") is None
    assert select_acknowledgement(derive_policy("neutral", senior=True), "neutral", "en") is None
    assert select_acknowledgement(derive_policy("distressed"), "distressed", "en")
    assert select_acknowledgement(derive_policy("emergency"), "emergency", "en") is None


def test_anger_acknowledges_first_and_then_offers_a_human():
    text = select_acknowledgement(derive_policy("angry"), "angry", "en")
    assert text.startswith("I am sorry") and text.rstrip().endswith("?")
    assert count_questions(text) == 1
    assert "connect you" not in select_acknowledgement(derive_policy("distressed"), "distressed", "en")


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_acknowledgements_satisfy_the_senior_policy(lang):
    senior = derive_policy("neutral", senior=True)
    for state in ("distressed", "angry", "confused"):
        assert check_reply(acknowledgement_for(state, lang, offer_human=True), senior) == [], (state, lang)


def test_wording_is_marked_pending_clinical_review_not_silently_approved():
    assert pending_review_count() == len(REVIEW_STATUS) > 0


# ================================================================ figures


@pytest.mark.parametrize(
    "digits,expected",
    [
        ("9876543210", ["98765", "43210"]),  # a mobile number is dictated 5 + 5
        ("12345678", ["1234", "5678"]),
        ("123456789", ["1234", "56789"]),  # remainder folded in, never a lone digit
        ("123", ["123"]),
        ("12345", ["12345"]),  # a lone trailing digit is folded in
    ],
)
def test_phone_grouping(digits, expected):
    got = phone_groups(digits)
    assert "".join(got) == digits
    assert got == expected


def test_a_lone_trailing_digit_is_never_left_on_its_own():
    assert all(len(g) >= 2 for g in phone_groups("123456789"))


def test_confirmation_ids_are_read_a_segment_at_a_time():
    assert id_groups("KCD-20261005-3F9A2B1C") == ["KCD", "2026", "1005", "3F9A", "2B1C"]
    assert id_groups("KCD-4471") == ["KCD", "4471"]


def test_grouped_speech_puts_a_standalone_beat_between_groups():
    assert speak_grouped(["98765", "43210"], lambda g: g) == "98765 , 43210"
