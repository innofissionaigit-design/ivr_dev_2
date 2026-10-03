"""KCD-445: "Numbers are never rounded, reordered or approximated... A
test asserts byte-level equality between the tool value and the spoken
value for a corpus of amounts, dates and identifiers."

tests/test_speech_norm.py already pins specific known-good outputs and
asserts no raw digit survives verbalization; this file adds the piece
those don't cover -- a general ROUND TRIP proof, across a corpus of
values, that the exact digit SEQUENCE spoken can be reconstructed
digit-for-digit from the source, in order, for every language. This is
the strongest text-level stand-in for "byte-level equality" the pipeline
allows (actual audio bytes are not assertable without a live TTS
service) and is exactly the property the module docstring's measured
incident violated: two different confirmation numbers producing
IDENTICAL audio because the digits were silently dropped, not
mispronounced.

    python -m pytest tests/test_digit_fidelity.py -v
"""

import os
import random
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import bn_normalize
from agent.speech_norm import verbalize_en, verbalize_hi

_BN_DIGIT_WORD = bn_normalize._ONES_TO_99[:10]  # index == digit
_HI_DIGIT_WORD = __import__("agent.speech_norm", fromlist=["_HI_0_99"])._HI_0_99[:10]
_EN_DIGIT_WORD = __import__("agent.speech_norm", fromlist=["_EN_0_19"])._EN_0_19[:10]


def _digits_from_spoken(spoken_text: str, digit_words: list[str]) -> str:
    """Reconstructs the digit sequence a digit-by-digit reading spoke, by
    mapping each recognised digit-word token back to its digit and
    dropping everything else (surrounding sentence words) -- the inverse
    of bn_normalize.digits_one_by_one / speech_norm's hi/en equivalents."""
    reverse = {word: str(i) for i, word in enumerate(digit_words)}
    return "".join(reverse[tok] for tok in spoken_text.split() if tok in reverse)


# CodeRabbit-flagged: seeded, not the module-level random state -- an
# unseeded generator makes PHONE_CORPUS (and therefore each parametrized
# test's own ID) change between runs, so a failure on a generated phone
# number cannot be reproduced by re-running the suite.
_RNG = random.Random(445)


def _random_phone() -> str:
    return "9" + "".join(str(_RNG.randint(0, 9)) for _ in range(9))


PHONE_CORPUS = ["9876543210", "9000000001", "9800000099", "9111111111"] + [_random_phone() for _ in range(20)]


@pytest.mark.parametrize("phone", PHONE_CORPUS)
def test_bengali_phone_number_round_trips_digit_for_digit(phone):
    spoken = bn_normalize.verbalize(f"ফোন নম্বর {phone}")
    assert _digits_from_spoken(spoken, _BN_DIGIT_WORD) == phone


@pytest.mark.parametrize("phone", PHONE_CORPUS)
def test_hindi_phone_number_round_trips_digit_for_digit(phone):
    spoken = verbalize_hi(f"फ़ोन नंबर {phone}")
    assert _digits_from_spoken(spoken, _HI_DIGIT_WORD) == phone


@pytest.mark.parametrize("phone", PHONE_CORPUS)
def test_english_phone_number_round_trips_digit_for_digit(phone):
    spoken = verbalize_en(f"phone number {phone}")
    assert _digits_from_spoken(spoken, _EN_DIGIT_WORD) == phone


# ----------------------------------------------------- price/amount corpus
# Boundary values deliberately included: 0, single/double/triple digit,
# exact hundred/thousand/lakh/crore (no remainder branch), and a value
# exercising every place at once -- the shape of number most likely to
# silently lose a digit in a recursive divmod implementation.
AMOUNT_CORPUS = [0, 1, 9, 50, 99, 100, 250, 999, 1000, 1500, 12345, 100000, 150000, 999999, 10000000, 12345678]


@pytest.mark.parametrize("amount", AMOUNT_CORPUS)
def test_bengali_amount_has_no_digit_left_and_is_not_empty(amount):
    spoken = bn_normalize.verbalize(f"রেট {amount} টাকা।")
    assert not any(c.isdigit() for c in spoken)
    assert spoken.strip()


@pytest.mark.parametrize("amount", AMOUNT_CORPUS)
def test_english_amount_has_no_digit_left_and_is_not_empty(amount):
    spoken = verbalize_en(f"rate {amount} rupees.")
    assert not any(c.isdigit() for c in spoken)
    assert spoken.strip()


@pytest.mark.parametrize("amount", AMOUNT_CORPUS)
def test_hindi_amount_has_no_digit_left_and_is_not_empty(amount):
    spoken = verbalize_hi(f"रेट {amount} रुपये।")
    assert not any(c.isdigit() for c in spoken)
    assert spoken.strip()


def test_different_amounts_never_produce_the_same_spoken_text():
    # The exact historical failure mode (module docstring): two different
    # numbers producing IDENTICAL audio because the digits were dropped.
    # Text-level identity is the necessary (if not sufficient) proxy.
    seen = {}
    for amount in AMOUNT_CORPUS:
        spoken = bn_normalize.verbalize(f"রেট {amount} টাকা।")
        assert spoken not in seen, f"{amount} and {seen.get(spoken)} produced identical spoken text"
        seen[spoken] = amount
