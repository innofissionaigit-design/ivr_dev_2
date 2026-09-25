"""The answer to "রোগীর নাম আর ফোন নম্বর?" split into (name, phone).

Regression for a live call: the caller said the phone number as digit WORDS,
and the whole sentence -- label, phone phrase and digits -- was read back as
the patient's name:

    ডাঃ সেন, 2026-09-24, সময় 10:00, রোগীর নাম রাজস্বী চক্রবর্তী রোগীর নাম আর
    ফোন নম্বার হল নাইট সেবেন সেভেন সেবেন সেবেন ফোর সেভেন জিরো ফাইভ টু, ...

main.py and main_pcm.py carry the same function, so every case runs on both.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main  # noqa: E402
import main_pcm  # noqa: E402
from agent.slot_parse import parse_phone, split_phone_span  # noqa: E402

MODULES = pytest.mark.parametrize("mod", [main, main_pcm], ids=["main", "main_pcm"])

LIVE_UTTERANCE = ("রাজস্বী চক্রবর্তী রোগীর নাম আর ফোন নম্বার হল "
                  "নাইট সেবেন সেভেন সেবেন সেবেন ফোর সেভেন জিরো ফাইভ টু")


@MODULES
def test_the_live_call_sentence_yields_only_the_name(mod):
    name, phone = mod._split_name_and_phone(LIVE_UTTERANCE)
    assert name == "রাজস্বী চক্রবর্তী"
    # "নাইট" OPENS this number, where it could be eight or nine, so it is not
    # read as a digit (it is only after "seven"): one digit short, no phone,
    # and the caller is asked for it again.
    assert phone is None


@MODULES
@pytest.mark.parametrize("said, expected", [
    # Same sentence with every digit word recognisable.
    ("রাজস্বী চক্রবর্তী রোগীর নাম আর ফোন নম্বার হল "
     "নাইন সেবেন সেভেন সেবেন সেবেন ফোর সেভেন জিরো ফাইভ টু",
     ("রাজস্বী চক্রবর্তী", "9777747052")),
    # Label first, number as words after a cue.
    ("রোগীর নাম হল রিয়া দাস, ফোন নম্বর নাইন ওয়ান টু থ্রি ফোর ফাইভ সিক্স সেভেন এইট নাইন",
     ("রিয়া দাস", "9123456789")),
    # Number first, name after it.
    ("ফোন নম্বর 9123456789, নাম রিয়া দাস", ("রিয়া দাস", "9123456789")),
    # Digit words with no cue word at all.
    ("রিয়া দাস নাইন ওয়ান টু থ্রি ফোর ফাইভ সিক্স সেভেন এইট নাইন",
     ("রিয়া দাস", "9123456789")),
])
def test_name_and_phone_in_one_answer(mod, said, expected):
    assert mod._split_name_and_phone(said) == expected


@MODULES
def test_numerals_still_split_as_before(mod):
    assert mod._split_name_and_phone("রিয়া দাস 9123456789") == ("রিয়া দাস", "9123456789")


@MODULES
def test_a_bare_name_is_the_whole_answer(mod):
    assert mod._split_name_and_phone("রিয়া দাস") == ("রিয়া দাস", None)


@MODULES
def test_an_unparseable_number_never_lands_in_the_name(mod):
    # Nine numerals: not a phone. It used to become part of the name.
    assert mod._split_name_and_phone("রিয়া দাস, ফোন 912345678") == ("রিয়া দাস", None)


@MODULES
def test_number_word_containing_naam_is_not_a_name_marker(mod):
    # "নাম্বার" (number) contains "নাম" (name). Treating it as a name marker
    # would end the phone part early and leave its digits in the name.
    assert mod._split_name_and_phone("রিয়া দাস, ফোন নাম্বার নাইট ওয়ান টু") == ("রিয়া দাস", None)


@MODULES
def test_one_filler_prefix_rule_still_holds(mod):
    # _clean_patient_name strips at most one leading filler; the new label
    # handling must not reintroduce the double strip that ate first names.
    assert mod._split_name_and_phone("নাম আমি সেন") == ("আমি সেন", None)


def test_split_phone_span_leaves_a_plain_answer_alone():
    assert split_phone_span("রিয়া দাস") == ("রিয়া দাস", "", None)


def test_asr_spelling_of_seven_is_a_digit():
    assert parse_phone("ওয়ান টু থ্রি ফোর ফাইভ সিক্স সেবেন এইট নাইন জিরো") == "1234567890"
