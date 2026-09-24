"""ADDED BY SOURAV -- fixes a real production bug, reported directly from
a live call transcript:

    [User] Is my report ready?
    [AI]   Can you tell me your registered phone number?
    [User] Yes, write nine zero zero zero zero zero zero zero zero one.
    [AI]   Can you tell me your registered phone number?
    [User] I said the number is nine zero zero zero zero zero zero zero zero one.
    [AI]   Can you tell me your registered phone number?
    [User] Nine then eight zeros and one
    [AI]   Sorry, I didn't understand. Could you say it again?
    [User] You can do it by knowing the number and giving the name.
    [AI]   Can you tell me your registered phone number?
    [User] Nine eight zero one

The caller was stuck in an infinite "please give your phone number" loop
because agent/slot_parse.py's parse_phone() only ever understood LITERAL
digit characters (Bengali numerals or plain ASCII digits) -- a caller who
reads their number out as spoken English words ("nine zero zero...")
produced ZERO digits after stripping non-digit characters, no matter how
many times or how clearly they repeated themselves. See parse_phone()'s
own docstring in agent/slot_parse.py for the full root-cause writeup.

parse_phone() had NO dedicated test file at all before this fix -- a real,
pre-existing gap in a function that gates identity verification (RULE
14/15) for real callers' private report data, confirmed by grepping every
file under tests/ for "parse_phone"/"parse_otp" before writing this file:
the only other reference is an unrelated docstring mention in
tests/test_booking_readback.py's own module docstring, and neither
function is exercised directly (only indirectly, through main_pcm.py
dispatch tests) anywhere else. This file covers ONLY the specific
behaviour this fix touched -- a full test suite for the rest of
slot_parse.py's existing behaviour (parse_date, parse_time,
parse_correction_field) is a separate, larger undertaking outside this
fix's scope, same reasoning tests/test_fast_path_schedule_cue.py gave for
the equivalent gap in agent/fast_path.py.
"""
from __future__ import annotations

import pytest

from agent.slot_parse import parse_otp, parse_phone


class TestTheReportedBugIsFixed:
    """Reproduces the exact phrasings from the transcript above."""

    def test_first_phrasing_from_the_transcript(self):
        assert parse_phone("Yes, write nine zero zero zero zero zero zero zero zero one.") == "9000000001"

    def test_second_phrasing_from_the_transcript_identical_number(self):
        assert parse_phone(
            "I said the number is nine zero zero zero zero zero zero zero zero one."
        ) == "9000000001"

    def test_case_insensitive_and_mixed_case_words(self):
        assert parse_phone("Nine Zero ZERO zero zero zero zero zero zero One") == "9000000001"


class TestAmbiguousRepetitionShorthandStillFailsClosed:
    """The caller's own third attempt in the transcript -- "eight zeros"
    meaning the digit 0 repeated eight times. Deliberately NOT supported:
    telling this apart from the caller instead meaning the two separate
    digits "eight" then "zero" is genuinely ambiguous, and getting a phone
    number wrong risks exposing (or refusing) the wrong caller's report.
    See parse_phone()'s own docstring for the full reasoning."""

    def test_eight_zeros_shorthand_is_not_guessed_at(self):
        assert parse_phone("Nine then eight zeros and one") is None

    def test_a_short_fragment_correctly_stays_unresolved(self):
        # Only 4 digit-words -- correctly insufficient regardless of the
        # word-to-digit fix; this was never expected to resolve.
        assert parse_phone("Nine eight zero one") is None


class TestPreExistingBehaviorUnchanged:
    """Regression guards: every way parse_phone() already worked before
    this fix must keep working identically."""

    def test_plain_ascii_digits(self):
        assert parse_phone("my number is 9000000001") == "9000000001"

    def test_bengali_digits(self):
        assert parse_phone("৯০০০০০০০০১") == "9000000001"

    def test_spaced_out_digits(self):
        assert parse_phone("90 00 00 00 01") == "9000000001"

    def test_leading_country_code_is_trimmed_to_last_10(self):
        assert parse_phone("+91 9000000001") == "9000000001"

    def test_too_short_returns_none(self):
        assert parse_phone("12345") is None

    def test_empty_string_returns_none(self):
        assert parse_phone("") is None

    def test_unrelated_text_with_no_digit_words_returns_none(self):
        assert parse_phone("আমি জানি না") is None


class TestNoFalsePositivesFromOrdinaryWords:
    """The word-to-digit substitution must not fire on ordinary words that
    merely contain a digit-word as a substring, or on isolated digit-words
    that don't add up to a real phone number."""

    def test_not_now_maybe_later_does_not_falsely_match(self):
        assert parse_phone("not now, maybe later") is None

    def test_one_moment_please_does_not_resolve_to_a_phone_number(self):
        # "one" is a real digit word, but a single digit is nowhere near
        # the 10 required -- must not be padded or guessed into a number.
        assert parse_phone("one moment please") is None

    def test_a_sentence_with_a_stray_digit_word_and_no_real_number(self):
        assert parse_phone("call me back in one hour") is None


class TestParseOtpIsUnaffectedByTheSharedWordList:
    """Regression guard: this fix moved _DIGIT_WORDS/_DIGIT_WORD_RE above
    parse_phone() (previously defined only just above parse_otp()) and had
    parse_phone() start reusing them -- parse_otp()'s own, already-working
    behaviour must be byte-for-byte unchanged."""

    def test_otp_spoken_as_words_still_works(self):
        assert parse_otp("four eight two nine one three") == "482913"

    def test_otp_as_bengali_digits_still_works(self):
        assert parse_otp("৪৮২৯১৩") == "482913"

    def test_otp_wrong_length_still_rejected(self):
        assert parse_otp("48291333") is None
        assert parse_otp("4829") is None


class TestAsrSpellingsOfEnglishDigitWords:
    """Live reschedule call: the booked number was said as "... সেভেন এইট
    নাইন" and accepted; to find the booking the same caller said it again and
    the ASR wrote "... সেবেন নাইট নাইন ফোন নম্বার". "সেবেন" (seven) and
    "নাইট" (eight, carrying the n of "seven") were not digits, the number came
    up short, and the caller was asked for it three times."""

    def test_the_live_reschedule_answer_is_the_booked_number(self):
        said = "জিরো ওয়ান টু থ্রি ফোর ফাইভ সিক্স সেবেন নাইট নাইন ফোন নম্বার"
        booked = "জিরো ওয়ান টু থ্রি ফোর ফাইভ সিক্স সেভেন এইট নাইন"
        assert parse_phone(said) == parse_phone(booked) == "0123456789"

    @pytest.mark.parametrize("seven", ["সেবেন", "সেভেন", "seven"])
    def test_naait_after_seven_is_eight(self, seven):
        assert parse_phone(f"ওয়ান টু থ্রি ফোর ফাইভ সিক্স {seven} নাইট নাইন জিরো") == "1234567890"

    def test_naait_anywhere_else_is_not_a_digit(self):
        # Opening a number, "নাইট" is as close to "nine" as to "eight":
        # never guessed, so this nine-digit reading is refused.
        assert parse_phone("নাইট সেবেন সেভেন সেবেন সেবেন ফোর সেভেন জিরো ফাইভ টু") is None

    def test_a_number_that_is_really_short_is_still_refused(self):
        # The caller's second try dropped the leading zero: nine digits.
        assert parse_phone("ওয়ান টু থ্রি ফোর ফাইভ সিক্স সেভেন এইট নাইন") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
