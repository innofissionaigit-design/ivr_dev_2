"""agent/spoken_codes.py: phone numbers, OTPs and the confirmation id, read back digit by digit and letter by letter.

    python -m pytest tests/test_spoken_codes.py -v

Every number here is synthetic (Hypothesis-generated or hand-written); no recording is needed.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent.spoken_codes import parse_confirmation_id, parse_otp, parse_phone

# ------------------------------------------------------------------------------------------------------- phone


@pytest.mark.parametrize(
    "text,expected",
    [
        ("9830112233", "9830112233"),
        ("98301 12233", "9830112233"),
        ("nine eight three zero one one two two three three", "9830112233"),
        ("my number is nine eight three zero one, one two two three three", "9830112233"),
        ("+91 9830112233", "9830112233"),
        ("91 9830112233", "9830112233"),
        ("0 9830112233", "9830112233"),
        ("নয় আট তিন শূন্য এক এক দুই দুই তিন তিন", "9830112233"),
        ("नौ आठ तीन शून्य एक एक दो दो तीन तीन", "9830112233"),
    ],
)
def test_a_ten_digit_mobile_number_is_read_from_digits_or_words_in_any_of_the_three_scripts(text, expected):
    assert parse_phone(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "12345",  # too short
        "12301112233",  # 11 digits, no recognised prefix leaves 11
        "1234567890",  # 10 digits but does not start 6-9: not a mobile range
        "asdf qwer",  # nothing at all
        "",
    ],
)
def test_anything_that_is_not_a_ten_digit_mobile_number_is_refused(text):
    assert parse_phone(text) is None


def test_number_words_are_never_composed_arithmetically():
    # "eighty five" is the two digits 8 and 5, never the number 85 -- composing them would corrupt the phone number
    assert (
        parse_phone("nine eight three zero one eighty five two three three") is None
    )  # "eighty"/"five" are not digit words
    assert parse_phone("nine eight three zero one one two two three three") == "9830112233"


@given(st.text(alphabet="6789", min_size=1, max_size=1), st.text(alphabet="0123456789", min_size=9, max_size=9))
def test_any_ten_digit_mobile_shaped_number_round_trips(first, rest):
    number = first + rest
    assert parse_phone(number) == number
    assert parse_phone("+91 " + number) == number
    assert parse_phone("0" + number) == number


# --------------------------------------------------------------------------------------------------------- otp


@pytest.mark.parametrize(
    "text,length,expected",
    [
        ("482913", None, "482913"),
        ("4 8 2 9 1 3", None, "482913"),
        ("four eight two nine one three", None, "482913"),
        ("4829", None, "4829"),
        ("চার আট দুই নয় এক তিন", None, "482913"),
        ("482913", 6, "482913"),
        ("482913", 4, None),
        ("4829", 6, None),
    ],
)
def test_an_otp_is_a_run_of_digits_of_the_expected_or_a_common_length(text, length, expected):
    assert parse_otp(text, length) == expected


def test_an_otp_of_an_uncommon_length_needs_the_length_to_be_given():
    assert parse_otp("1 2 3 4 5") is None  # 5 is not 4 or 6
    assert parse_otp("1 2 3 4 5", length=5) == "12345"


@given(st.text(alphabet="0123456789", min_size=4, max_size=4) | st.text(alphabet="0123456789", min_size=6, max_size=6))
def test_any_digit_run_of_a_common_otp_length_round_trips(code):
    assert parse_otp(code) == code
    assert parse_otp(" ".join(code)) == code


# ---------------------------------------------------------------------------------------- confirmation id


@pytest.mark.parametrize(
    "text",
    [
        "KCD-20260928-1F3A9C0B",
        "kcd 2 0 2 6 0 9 2 8 1 f 3 a 9 c 0 b",
        "k c d two zero two six zero nine two eight one f three a nine c zero b",
        "the code is k c d 20260928 1f3a9c0b thank you",
        "kay cee dee 20260928 1f3a9c0b",
        "কে সি ডি ২০২৬০৯২৮ ১ এফ ৩ এ ৯ সি ০ বি",
    ],
)
def test_a_confirmation_id_is_read_from_letters_and_digits_in_any_of_the_three_scripts(text):
    assert parse_confirmation_id(text) == "KCD-20260928-1F3A9C0B"


@pytest.mark.parametrize(
    "text",
    [
        "KCD-2026092-1F3A9C0B",  # 7-digit date
        "KCD-20260928-1F3A9C0",  # 7-char reference
        "KPC-20260928-1F3A9C0B",  # wrong prefix
        "KCD-20260928-1G3A9C0B",  # G is not hex
        "just a random sentence",
        "",
    ],
)
def test_anything_that_does_not_fit_the_shape_is_refused(text):
    assert parse_confirmation_id(text) is None


_HEX = "0123456789ABCDEF"


@given(st.text(alphabet="0123456789", min_size=8, max_size=8), st.text(alphabet=_HEX, min_size=8, max_size=8))
def test_any_kcd_shaped_code_round_trips_letter_by_letter_and_digit_by_digit(date, ref):
    code = f"KCD-{date}-{ref}"
    assert parse_confirmation_id(code) == code
    assert parse_confirmation_id(" ".join(f"KCD{date}{ref}")) == code
