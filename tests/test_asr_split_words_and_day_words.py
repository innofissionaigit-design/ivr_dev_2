"""Two live-call failures found while testing the booking stories by voice.

1. The confidence gate threw away a turn both ASR decoders heard the same.
   The caller picked a slot with "দ্বিতীয়টা"; CTC wrote "দ্বিতীয়টা", RNNT
   "দ্বিতীয় টা". As word sets they share nothing, agreement was 0.00, the zone
   was REJECT and the caller heard "ভালো করে শুনতে পাইনি".

2. A day of the month said as a WORD was not a date. "চব্বিশ তারিখে সকাল পৌনে
   এগারোটায় ..." -- the ASR writes "চব্বিশ", not "২৪" -- so the 24th never
   became a date and the caller was asked to pick a day from 1-31 October.

Both fixes only turn a failure into a result: an input that already worked
gives exactly what it gave before (checked old-vs-new on every string in the
code and tests before this file was written).
"""
from __future__ import annotations

import datetime
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.asr import _word_agreement  # noqa: E402  (torch / nemo stubbed by conftest)
from agent.bn_normalize import number_to_bn_words  # noqa: E402
from agent.confidence import PROCEED, REJECT, zone  # noqa: E402
from agent.slot_parse import parse_date  # noqa: E402

TODAY = datetime.date(2026, 9, 24)


# ------------------------------------------------------------ 1. split words

class TestSameWordsDifferentSpaces:
    def test_the_live_pick_is_full_agreement(self):
        assert _word_agreement("দ্বিতীয়টা", "দ্বিতীয় টা") == 1.0

    def test_the_live_pick_now_proceeds(self):
        turn = types.SimpleNamespace(decoder_used="rnnt",
                                     decoder_agreement=round(_word_agreement("দ্বিতীয়টা", "দ্বিতীয় টা"), 2))
        assert zone(turn) == PROCEED

    @pytest.mark.parametrize("a, b", [
        ("প্রথমটা", "দ্বিতীয়টা"),            # different words heard
        ("প্রথমটা", "প্রথম তা"),              # a different letter heard
        ("হ্যাঁ", "না"),
    ])
    def test_a_real_difference_is_still_disagreement(self, a, b):
        assert _word_agreement(a, b) == 0.0
        turn = types.SimpleNamespace(decoder_used="rnnt", decoder_agreement=_word_agreement(a, b))
        assert zone(turn) == REJECT

    def test_partial_overlap_is_scored_as_before(self):
        assert _word_agreement("ডাক্তার সেন কাল আছেন", "ডাক্তার সেন কাল আসেন") == 3 / 5

    def test_empty_cases_are_unchanged(self):
        assert _word_agreement("", "") == 1.0
        assert _word_agreement("", "হ্যাঁ") == 0.0


# ----------------------------------------------------------- 2. day words

class TestDayOfMonthSaidAsAWord:
    def test_the_live_sentence(self):
        said = "চব্বিশ তারিখে সকাল পৌনে এগারোটায় ডাক্তার সেনের কাছে অ্যাপয়েন্টমেন্ট চাই"
        assert parse_date(said, today=TODAY) == "2026-09-24"

    @pytest.mark.parametrize("day", range(1, 32))
    def test_every_day_word_the_agent_itself_speaks(self, day):
        # January has 31 days, so every day 1-31 exists; a caller repeating
        # the agent's own readback word is understood.
        word = number_to_bn_words(day)
        assert parse_date(f"{word} তারিখ", today=datetime.date(2026, 1, 1)) == f"2026-01-{day:02d}"

    def test_a_past_day_rolls_into_next_month_like_numerals(self):
        assert parse_date("পাঁচ তারিখে", today=TODAY) == parse_date("৫ তারিখে", today=TODAY) == "2026-10-05"

    def test_a_day_the_month_does_not_have_is_asked_again(self):
        assert parse_date("একত্রিশ তারিখ", today=TODAY) is None   # September has 30 days

    def test_the_agents_readback_phrasing_with_its_month(self):
        assert parse_date("সেপ্টেম্বর মাসের আটাশ তারিখ", today=TODAY) == "2026-09-28"
        assert parse_date("অক্টোবর মাসের পাঁচ তারিখ", today=TODAY) == "2026-10-05"

    @pytest.mark.parametrize("said", ["আগস্ট মাসের চব্বিশ তারিখ", "অক্টোবর মাসের আটাশ তারিখ"])
    def test_a_named_month_that_does_not_match_is_never_guessed(self, said):
        assert parse_date(said, today=TODAY) is None

    @pytest.mark.parametrize("said", ["পাঁচ মিনিট", "চব্বিশ ঘণ্টা", "বার বার বলছি", "এক মিনিট দাঁড়ান",
                                      "বারো টাকা", "দশটা নাগাদ"])
    def test_a_number_word_without_tarikh_is_not_a_date(self, said):
        assert parse_date(said, today=TODAY) is None

    def test_numerals_are_unchanged(self):
        assert parse_date("২৪ তারিখে", today=TODAY) == "2026-09-24"
        assert parse_date("কাল", today=TODAY) == "2026-09-25"
