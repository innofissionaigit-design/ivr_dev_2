"""speech_norm: numbers, times, dates and IDs per language. Pure functions.

The point of pinning these is the failure mode they prevent, not the words:
a Latin digit reaching an Indic FastPitch tokenizer is silently DROPPED
(agent/bn_normalize.py has the measurement), so a wrong table entry here is
a wrong price spoken to a caller, with no error anywhere.
"""

import re

import pytest

from agent.speech_norm import (
    number_to_en_words,
    number_to_hi_words,
    time_to_en_words,
    time_to_hi_words,
    unspeakable_spans,
    verbalize,
)

DIGIT = re.compile(r"\d")


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "zero"),
        (7, "seven"),
        (19, "nineteen"),
        (20, "twenty"),
        (45, "forty five"),
        (100, "one hundred"),
        (250, "two hundred fifty"),
        (1500, "one thousand five hundred"),
        (100000, "one lakh"),
        (150000, "one lakh fifty thousand"),
        (12345678, "one crore twenty three lakh forty five thousand six hundred seventy eight"),
    ],
)
def test_english_numbers(n, expected):
    assert number_to_en_words(n) == expected


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "शून्य"),
        (1, "एक"),
        (6, "छह"),
        (11, "ग्यारह"),
        (15, "पंद्रह"),
        (19, "उन्नीस"),
        (20, "बीस"),
        (21, "इक्कीस"),
        (29, "उनतीस"),
        (31, "इकतीस"),
        (39, "उनतालीस"),
        (40, "चालीस"),
        (49, "उनचास"),
        (50, "पचास"),
        (59, "उनसठ"),
        (69, "उनहत्तर"),
        (79, "उनासी"),
        (89, "नवासी"),
        (99, "निन्यानवे"),
        (250, "दो सौ पचास"),
        (400, "चार सौ"),
        (1500, "एक हज़ार पाँच सौ"),
        (100000, "एक लाख"),
        (2200, "दो हज़ार दो सौ"),
    ],
)
def test_hindi_numbers(n, expected):
    assert number_to_hi_words(n) == expected


def test_hindi_table_has_no_gaps_or_duplicates():
    words = [number_to_hi_words(i) for i in range(100)]
    assert all(w and not DIGIT.search(w) for w in words)
    assert len(set(words)) == 100


@pytest.mark.parametrize(
    "hh,mm,expected",
    [
        (18, 0, "six PM"),
        (10, 0, "ten AM"),
        (17, 30, "five thirty PM"),
        (9, 5, "nine oh five AM"),
    ],
)
def test_english_times(hh, mm, expected):
    assert time_to_en_words(hh, mm) == expected


@pytest.mark.parametrize(
    "hh,mm,expected",
    [
        (18, 0, "शाम छह बजे"),
        (10, 0, "सुबह दस बजे"),
        (17, 30, "शाम साढ़े पाँच बजे"),
        (13, 30, "दोपहर डेढ़ बजे"),
        (14, 30, "दोपहर ढाई बजे"),
        (9, 15, "सुबह सवा नौ बजे"),
        (8, 45, "सुबह पौने नौ बजे"),
    ],
)
def test_hindi_times(hh, mm, expected):
    assert time_to_hi_words(hh, mm) == expected


def test_english_price_sentence_has_no_digits_left():
    out = verbalize("The price of the Uric Acid test is 250 rupees. Report within 12 hours.", "en")
    assert "two hundred fifty" in out and "twelve" in out
    assert not DIGIT.search(out)


def test_hindi_price_sentence_has_no_digits_left():
    out = verbalize("यूरिक एसिड टेस्ट की कीमत 250 रुपये है। रिपोर्ट 12 घंटे में मिल जाएगी।", "hi")
    assert "दो सौ पचास" in out and "बारह" in out
    assert not DIGIT.search(out)


def test_hindi_devanagari_digits_take_the_same_path():
    assert verbalize("कीमत ४०० रुपये", "hi") == verbalize("कीमत 400 रुपये", "hi")


def test_dates_times_ranges_ids_and_phones_per_language():
    en = verbalize("On 2026-09-16, 18:00-20:00, confirmation KCD-4471, phone 9876543210.", "en")
    assert "September sixteenth" in en and "six PM to eight PM" in en
    assert "kay see dee" in en and "four four seven one" in en
    # KCD-157: a phone number is dictated in groups (5 + 5), with a beat
    # between them, so a caller can write it down -- no longer one
    # unbroken ten-digit run.
    assert "nine eight seven six five , four three two one zero" in en
    assert not DIGIT.search(en)

    hi = verbalize("2026-09-16 को 18:00-20:00, नंबर KCD-4471, फ़ोन 9876543210।", "hi")
    assert "सितंबर की सोलह तारीख" in hi and "शाम छह बजे से रात आठ बजे तक" in hi
    assert not DIGIT.search(hi)


def test_hindi_maps_specimen_words_it_can_say():
    assert "खून" in verbalize("सैंपल: Blood।", "hi")


def test_bengali_dispatch_is_unchanged():
    from agent import bn_normalize

    s = "রেট 250 টাকা।"
    assert verbalize(s, "bn") == bn_normalize.verbalize(s)
    assert verbalize(s) == bn_normalize.verbalize(s)


def test_unspeakable_spans_are_per_language():
    assert unspeakable_spans("यूरिक Acid", "hi") == ["Acid"]
    assert unspeakable_spans("hello यूरिक", "en") == ["यूरिक"]
    assert unspeakable_spans("plain english only", "en") == []
    assert unspeakable_spans("केवल हिंदी", "hi") == []
