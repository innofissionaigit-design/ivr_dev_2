"""Regressions for the third CodeRabbit batch (PR #2). Each test pins the exact
failure the review described, so it cannot come back unnoticed.

    python -m pytest tests/test_review_regressions_batch3.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.audio_quality import transcript_problem
from agent.booking_flow import classify_yes_no
from agent.language_policy import choose_spoken_form, reply_matches_language, script_counts
from agent.llm import _normalize_age
from agent.prosody import split_for_prosody
from agent.speech_policy import count_questions, limit_questions

# --------------------------------------------- confirmation must not be inferred from a maybe


@pytest.mark.parametrize(
    "text",
    [
        "not sure I should confirm",
        "I am not really sure but confirm it",
        "maybe yes",
        "unsure, go ahead",
    ],
)
def test_an_uncertain_answer_is_never_a_yes(text):
    assert classify_yes_no(text, "en") is None


@pytest.mark.parametrize("text", ["yes", "yes I am sure", "yes please confirm", "confirm it", "sure"])
def test_a_plain_affirmative_still_confirms(text):
    assert classify_yes_no(text, "en") == "yes"


# --------------------------------------------- a statement is never dropped with a surplus question


def test_limit_questions_keeps_a_statement_that_sits_between_two_questions():
    out = limit_questions("Which doctor? Your price is 500 rupees. Which day?", 1)
    assert "500 rupees" in out
    assert count_questions(out) == 1


def test_limit_questions_does_not_split_a_decimal():
    out = limit_questions("The fee is 2.5 hundred. Which doctor? Which day?", 1)
    assert "2.5 hundred" in out


# --------------------------------------------- dictated slots are not "jumbled"


@pytest.mark.parametrize(
    "text",
    [
        "9 8 7 6 5 4 3 2 1 0",  # a phone number, digit by digit
        "R A V I D A S",  # a name, letter by letter
        "5 5 5 5 5 5",  # a repeated digit run
    ],
)
def test_a_dictated_slot_answer_is_not_called_jumbled(text):
    # the reviewer's reproduction: outside a booking these ARE flagged ...
    assert transcript_problem(text, "en", 6.0, 0.9) in ("shattered", "repetitive")
    # ... and inside one they are the answer we asked for.
    assert transcript_problem(text, "en", 6.0, 0.9, slot_answer=True) is None


def test_a_short_answer_over_a_long_clip_is_a_fragment_only_outside_a_booking():
    assert transcript_problem("না", "bn", 3.0, 0.9) == "fragment"
    assert transcript_problem("না", "bn", 3.0, 0.9, slot_answer=True) is None


def test_decoder_disagreement_still_counts_inside_a_booking():
    assert transcript_problem("9 8 7 6", "en", 4.0, 0.1, slot_answer=True) == "decoders_disagree"


# --------------------------------------------- language policy


def test_numerals_carry_no_script():
    assert script_counts("১২৩") == {"bengali": 0, "devanagari": 0, "latin": 0}
    assert script_counts("१२३") == {"bengali": 0, "devanagari": 0, "latin": 0}
    assert reply_matches_language("১২৩", "en") is True
    assert reply_matches_language("১২৩ ৪৫", "hi") is True


def test_the_longest_whole_form_the_caller_used_wins():
    forms = ["স্ক্যান", "সিটি স্ক্যান"]
    assert choose_spoken_form("সিটি স্ক্যান", forms, None) == "সিটি স্ক্যান"  # exact
    assert choose_spoken_form("আমার সিটি স্ক্যান লাগবে", forms, None) == "সিটি স্ক্যান"  # longest contained
    assert choose_spoken_form("স্ক্যান", forms, None) == "স্ক্যান"


# --------------------------------------------- prosody


def test_a_newline_boundary_earns_a_sentence_pause():
    assert split_for_prosody("Hello\nWorld") == [("Hello", "sentence"), ("World", "none")]


def test_a_long_clause_without_a_comma_is_cut_at_word_boundaries():
    text = " ".join(["word"] * 40) + "."
    chunks = split_for_prosody(text, 60)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c, _ in chunks)
    assert all(kind == "none" for _, kind in chunks[:-1])
    assert chunks[-1][1] == "sentence"  # the clause keeps its own pause at the end
    assert " ".join(c for c, _ in chunks) == text


# --------------------------------------------- the age slot is a number or nothing


@pytest.mark.parametrize(
    "value,expected",
    [
        (72, 72),
        ("72", 72),
        (" 65 ", 65),
        (72.0, 72),
        ("seventy", None),
        (None, None),
        (True, None),
        (0, None),
        (200, None),
        ("7.5", None),
        (-3, None),
    ],
)
def test_patient_age_is_normalised_at_the_extraction_boundary(value, expected):
    assert _normalize_age(value) == expected
