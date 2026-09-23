"""ADDED BY SOURAV -- "Caller goes silent" story (Epic: Conversation --
Difficult, Sensitive and Edge Cases), plus this story's own bugfix
(validation report Bug #2).

story title: Caller goes silent
user story: As a caller who was distracted, I want a gentle prompt rather
    than a disconnection, so that I do not lose the call.
acceptance criteria: Two graduated prompts precede a graceful close, each
    different from the last. The close states what was and was not
    completed. Abandonment is logged with the turn index and the
    preceding prompt.

bugfix covered here: a caller who never said a word before going silent
(session.utt_seq == 0, pending and deferred both None) used to be told
"everything you asked has been taken care of" -- has_unfinished_business()
correctly said False (nothing is pending), but the close text built on
top of that False conflated "nothing outstanding" with "something was
resolved". agent/silence_flow.next_close_state() adds the ONE extra fact
(utt_seq) needed to tell "never engaged" apart from "engaged and
finished cleanly", without touching has_unfinished_business() itself or
introducing any new session field. See that function's own docstring for
the full writeup.

Layout, matching tests/test_correction_flow.py's own convention:
  - TestNextSilenceAction / TestHasUnfinishedBusiness: the pre-existing
    pure functions, regression-pinned so this bugfix cannot accidentally
    change either (the bugfix's own constraint: "keep has_unfinished_
    business() logic unchanged unless absolutely necessary" -- it was not
    necessary, and this file proves it stayed that way).
  - TestNextCloseState: the new pure function this bugfix adds.
  - TestSilenceCloseReply: agent/reply_templates.py's silence_close_reply(),
    now branching on next_close_state()'s three string values instead of
    a bare bool.
"""
from __future__ import annotations

import pytest

from agent import silence_flow
from agent.reply_templates import silence_close_reply


class TestNextSilenceAction:
    """Unchanged by this bugfix -- pinned here so a future change to the
    close-state logic cannot accidentally also change the prompt-stage
    logic, since both live in the same module."""

    def test_stage_normal_goes_to_prompt_1(self):
        assert silence_flow.next_silence_action(silence_flow.STAGE_NORMAL) == \
            silence_flow.ACTION_PROMPT_1

    def test_stage_prompt_1_sent_goes_to_prompt_2(self):
        assert silence_flow.next_silence_action(silence_flow.STAGE_PROMPT_1_SENT) == \
            silence_flow.ACTION_PROMPT_2

    def test_stage_prompt_2_sent_goes_to_abandon(self):
        assert silence_flow.next_silence_action(silence_flow.STAGE_PROMPT_2_SENT) == \
            silence_flow.ACTION_ABANDON

    def test_unrecognised_numeric_stage_fails_closed_to_abandon(self):
        assert silence_flow.next_silence_action(99) == silence_flow.ACTION_ABANDON


class TestHasUnfinishedBusiness:
    """Unchanged by this bugfix -- see this file's own module docstring
    for why next_close_state() calls this rather than reimplementing it."""

    def test_neither_pending_nor_deferred(self):
        assert silence_flow.has_unfinished_business(None, None) is False

    def test_pending_only(self):
        assert silence_flow.has_unfinished_business({"awaiting": "date"}, None) is True

    def test_deferred_only(self):
        assert silence_flow.has_unfinished_business(None, {"parts": [1]}) is True

    def test_both(self):
        assert silence_flow.has_unfinished_business({"awaiting": "x"}, {"parts": [1]}) is True


class TestNextCloseState:
    """The bugfix itself: which of the three close texts an abandonment
    should use, given session.utt_seq/pending/deferred."""

    def test_fresh_silent_call_is_no_engagement(self):
        # utt_seq=0, pending=None, deferred=None -- the exact scenario
        # from the validation report's Bug #2.
        assert silence_flow.next_close_state(0, None, None) == \
            silence_flow.CLOSE_NO_ENGAGEMENT

    def test_caller_engaged_then_nothing_pending_is_completed(self):
        assert silence_flow.next_close_state(1, None, None) == \
            silence_flow.CLOSE_COMPLETED
        assert silence_flow.next_close_state(5, None, None) == \
            silence_flow.CLOSE_COMPLETED

    def test_pending_present_is_unfinished_regardless_of_turns(self):
        assert silence_flow.next_close_state(0, {"awaiting": "date"}, None) == \
            silence_flow.CLOSE_UNFINISHED
        assert silence_flow.next_close_state(3, {"awaiting": "date"}, None) == \
            silence_flow.CLOSE_UNFINISHED

    def test_deferred_present_is_unfinished(self):
        assert silence_flow.next_close_state(0, None, {"parts": [1]}) == \
            silence_flow.CLOSE_UNFINISHED

    def test_both_pending_and_deferred_is_unfinished(self):
        assert silence_flow.next_close_state(2, {"awaiting": "x"}, {"parts": [1]}) == \
            silence_flow.CLOSE_UNFINISHED

    def test_unfinished_business_takes_priority_over_engagement_count(self):
        # Even a hypothetical utt_seq == 0 with something pending (not
        # reachable through this codebase's own state transitions today,
        # since pending/deferred are only ever set from inside a
        # dispatched turn -- but the function's own contract should not
        # depend on that being true forever) still reads as unfinished,
        # never no_engagement: "something is genuinely outstanding" is
        # the more specific, more actionable fact.
        assert silence_flow.next_close_state(0, {"awaiting": "date"}, None) == \
            silence_flow.CLOSE_UNFINISHED


class TestSilenceCloseReply:
    """agent/reply_templates.py's silence_close_reply(), now taking one
    of next_close_state()'s three string values."""

    ALL_LANGUAGES = ("bengali", "english", "hinglish", "banglish")

    @pytest.mark.parametrize("language", ALL_LANGUAGES)
    def test_all_three_states_produce_distinct_text(self, language):
        unfinished = silence_close_reply(silence_flow.CLOSE_UNFINISHED, language=language)
        no_engagement = silence_close_reply(silence_flow.CLOSE_NO_ENGAGEMENT, language=language)
        completed = silence_close_reply(silence_flow.CLOSE_COMPLETED, language=language)
        assert len({unfinished, no_engagement, completed}) == 3, (
            f"expected 3 distinct close texts for language={language!r}, "
            f"got a collision: {unfinished!r} / {no_engagement!r} / {completed!r}"
        )

    def test_no_engagement_does_not_claim_anything_was_completed_or_answered(self):
        # The bug being fixed, pinned as a permanent regression test: none
        # of these phrases (or their non-English equivalents already
        # covered by the distinctness test above) may appear in the
        # no-engagement close.
        text = silence_close_reply(silence_flow.CLOSE_NO_ENGAGEMENT, language="english")
        forbidden_claims = ("taken care of", "answered", "completed", "resolved")
        lowered = text.lower()
        for phrase in forbidden_claims:
            assert phrase not in lowered, (
                f"no-engagement close still contains a completion claim: {phrase!r} in {text!r}"
            )

    def test_no_engagement_bengali_text_matches_expected(self):
        # Pinned exact text so a future edit that reintroduces a
        # completion claim in Bengali (the default language, and the one
        # actually spoken to callers in this deployment) is caught even
        # if the English-phrase check above is bypassed.
        assert silence_close_reply(silence_flow.CLOSE_NO_ENGAGEMENT) == \
            "কোনো সাড়া না পাওয়ায় আমি কলটি এখানে শেষ করছি। ধন্যবাদ।"

    def test_completed_state_keeps_original_wording(self):
        # The bugfix must not change CLOSE_COMPLETED's text -- that was
        # already correct for the case it actually describes (the caller
        # engaged and nothing is outstanding).
        assert silence_close_reply(silence_flow.CLOSE_COMPLETED) == (
            "কোনো সাড়া না পাওয়ায় আমি কলটি এখানে শেষ করছি। আপনি যা জিজ্ঞাসা "
            "করেছিলেন তার উত্তর দেওয়া হয়ে গেছে। ধন্যবাদ, ভালো থাকবেন।"
        )

    def test_unfinished_state_keeps_original_wording(self):
        # Also unchanged by this bugfix.
        assert silence_close_reply(silence_flow.CLOSE_UNFINISHED) == (
            "কোনো সাড়া না পাওয়ায় আমি কলটি এখানে শেষ করছি। আপনি যেটি শুরু "
            "করেছিলেন সেটি এখনও সম্পূর্ণ হয়নি -- চালিয়ে যেতে আবার কল করুন। "
            "ধন্যবাদ।"
        )
