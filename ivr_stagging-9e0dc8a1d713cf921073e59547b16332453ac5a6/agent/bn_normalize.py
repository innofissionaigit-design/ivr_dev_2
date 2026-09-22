"""Verbalize a reply string into something the Bengali TTS can actually SAY.

This is not a cosmetic prettifier -- it fixes a proven, silent data-loss
bug. AI4Bharat's Bengali FastPitch tokenizer drops Latin digits entirely.
Measured on the live pod (22050Hz mono 16-bit, so bytes/2/22050 = seconds):

    "রেট টাকা।"            -> 102476 bytes   (no number at all)
    "রেট 250 টাকা।"        -> 100940 bytes   <- identical...
    "রেট 987654 টাকা।"     -> 100940 bytes   <- ...to a 6-digit number
    "রেট দুইশো পঞ্চাশ টাকা।" -> 143948 bytes   (spelled out: actually spoken)

Two different numbers producing the same audio, shorter than the sentence
with the number removed, is conclusive: every price, report time, chamber
hour and confirmation number this agent has ever "spoken" was silence.
Callers heard "রেট ___ টাকা" and reported it as the agent skipping words.

So every number reaching TTS gets spelled into Bengali words HERE, in
code, before synthesis. Deliberately not asked of the LLM: the model is
never allowed to restate a figure (see llm.py's module docstring), and
that rule does not get to be quietly relaxed just because the figure
needs reformatting.

LANGUAGE SUPPORT:
- Primary: Bengali (বাংলা) - for TTS synthesis
- Secondary: Hinglish support - for mixed-language contexts
- The system handles Bengali, English, and Hinglish (Hindi-English mix)
while maintaining digit-faithful number verbalization.
"""
from __future__ import annotations

import math
import re

# ---------------------------------------------------------------- numbers

_ONES_TO_99 = [
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়",
    "দশ", "এগারো", "বারো", "তেরো", "চোদ্দো", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ",
    "কুড়ি", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আটাশ", "ঊনত্রিশ",
    "ত্রিশ", "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ", "ঊনচল্লিশ",
    "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ", "ঊনপঞ্চাশ",
    "পঞ্চাশ", "একান্ন", "বাহান্ন", "তিপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "ঊনষাট",
    "ষাট", "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি", "সাতষট্টি", "আটষট্টি", "ঊনসত্তর",
    "সত্তর", "একাত্তর", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "ঊনআশি",
    "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "অষ্টাশি", "ঊননব্বই",
    "নব্বই", "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
]

_HUNDREDS = [
    "", "একশো", "দুইশো", "তিনশো", "চারশো", "পাঁচশো", "ছয়শো", "সাতশো", "আটশো", "নয়শো",
]

# Bengali digit glyphs -> ASCII, so ২৫০ and 250 take the same path.
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# ---------------------------------------------------------------- Hinglish support

# English number words for Hinglish context (digit-faithful conversion)
_ENGLISH_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven", "twenty-eight", "twenty-nine",
    "thirty", "thirty-one", "thirty-two", "thirty-three", "thirty-four", "thirty-five", "thirty-six", "thirty-seven", "thirty-eight", "thirty-nine",
    "forty", "forty-one", "forty-two", "forty-three", "forty-four", "forty-five", "forty-six", "forty-seven", "forty-eight", "forty-nine",
    "fifty", "fifty-one", "fifty-two", "fifty-three", "fifty-four", "fifty-five", "fifty-six", "fifty-seven", "fifty-eight", "fifty-nine",
    "sixty", "sixty-one", "sixty-two", "sixty-three", "sixty-four", "sixty-five", "sixty-six", "sixty-seven", "sixty-eight", "sixty-nine",
    "seventy", "seventy-one", "seventy-two", "seventy-three", "seventy-four", "seventy-five", "seventy-six", "seventy-seven", "seventy-eight", "seventy-nine",
    "eighty", "eighty-one", "eighty-two", "eighty-three", "eighty-four", "eighty-five", "eighty-six", "eighty-seven", "eighty-eight", "eighty-nine",
    "ninety", "ninety-one", "ninety-two", "ninety-three", "ninety-four", "ninety-five", "ninety-six", "ninety-seven", "ninety-eight", "ninety-nine",
]

_ENGLISH_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
]

_ENGLISH_HUNDREDS = [
    "", "one hundred", "two hundred", "three hundred", "four hundred", "five hundred",
    "six hundred", "seven hundred", "eight hundred", "nine hundred",
]


def number_to_bn_words(n: int) -> str:
    """Indian numbering system (হাজার / লাখ / কোটি), not the short scale."""
    # UPDATED BY SOURAV -- defense-in-depth for the combined report_status/
    # report_send story's regression testing: clinic-api's rate_inr column
    # is a SQLAlchemy Float, so a caller can end up handing this function a
    # whole-number float (e.g. 250.0) instead of an int -- divmod(250.0,
    # 100) returns a float `head`, and `_HUNDREDS[head]` then raises
    # TypeError: list indices must be integers or slices, not float.
    # verbalize()'s own regex path already coerces every matched digit
    # substring to int before calling here, so this was previously
    # unreachable from normal spoken output; it IS reachable via a direct
    # call (see tests/test_live_test_price_lookup.py's digit-fidelity
    # test, which does exactly that against a real DB-sourced rate). Only
    # a WHOLE-number float is coerced -- a genuinely fractional value
    # (e.g. 199.55) still falls through to the `n < 0` / int-indexing logic
    # below and still raises, exactly as before, because silently
    # truncating a real fraction would violate this codebase's digit-
    # fidelity discipline (see tests/test_number_fidelity*.py).
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if n < 0:
        return "মাইনাস " + number_to_bn_words(-n)
    if n < 100:
        return _ONES_TO_99[n]
    if n < 1000:
        head, rest = divmod(n, 100)
        out = _HUNDREDS[head]
        return out if rest == 0 else f"{out} {number_to_bn_words(rest)}"
    for divisor, word in ((10_000_000, "কোটি"), (100_000, "লাখ"), (1_000, "হাজার")):
        if n >= divisor:
            head, rest = divmod(n, divisor)
            out = f"{number_to_bn_words(head)} {word}"
            return out if rest == 0 else f"{out} {number_to_bn_words(rest)}"
    return str(n)  # unreachable


def number_to_english_words(n: int) -> str:
    """Convert number to English words for Hinglish support (digit-faithful)."""
    # UPDATED BY SOURAV -- same whole-number-float tolerance as
    # number_to_bn_words() above, for the same reason (a raw rate_inr
    # float reaching this function directly); see that function's comment
    # for the full explanation. A genuine fraction is still not coerced.
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if n < 0:
        return "minus " + number_to_english_words(-n)
    if n < 100:
        return _ENGLISH_ONES[n]
    if n < 1000:
        head, rest = divmod(n, 100)
        out = _ENGLISH_HUNDREDS[head]
        return out if rest == 0 else f"{out} and {number_to_english_words(rest)}"
    # Use Indian numbering system for consistency with Bengali
    for divisor, word in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        if n >= divisor:
            head, rest = divmod(n, divisor)
            out = f"{number_to_english_words(head)} {word}"
            return out if rest == 0 else f"{out} {number_to_english_words(rest)}"
    return str(n)  # unreachable


def number_to_hinglish_words(n: int) -> str:
    """Convert number to Hinglish (Hindi-English mix) words.
    
    Hinglish commonly uses Hindi number words for small numbers (1-10)
    and English for larger numbers, or mixes them naturally.
    This provides a digit-faithful conversion that sounds natural in
    Hinglish contexts while preserving exact values.
    """
    # Common Hindi numbers used in Hinglish
    _HINDI_SMALL = {
        0: "zero", 1: "ek", 2: "do", 3: "teen", 4: "chaar",
        5: "paanch", 6: "chhe", 7: "saat", 8: "aath", 9: "nau",
        10: "das", 20: "bees", 30: "tees", 40: "chaalis", 50: "pachaas",
        60: "saath", 70: "sattar", 80: "assi", 90: "nabbe", 100: "ek sau"
    }
    
    # UPDATED BY SOURAV -- same whole-number-float tolerance as
    # number_to_bn_words() above (see its comment for the full rationale).
    # Without this, the recursive divmod() calls further down would hand
    # a float `head`/`rest` to number_to_english_words(), reintroducing
    # the same TypeError one level removed.
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if n < 0:
        return "minus " + number_to_hinglish_words(-n)

    # Use Hindi words for small numbers (common in Hinglish)
    if n in _HINDI_SMALL:
        return _HINDI_SMALL[n]
    
    # For larger numbers, use English with Indian system
    if n < 1000:
        return number_to_english_words(n)
    
    # Use Indian numbering system with English words
    for divisor, word in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        if n >= divisor:
            head, rest = divmod(n, divisor)
            out = f"{number_to_hinglish_words(head)} {word}"
            return out if rest == 0 else f"{out} {number_to_hinglish_words(rest)}"
    
    return str(n)


# Latin letters read aloud in Bengali. Confirmation IDs ("KCD-4471") are
# the only place these reach TTS, and without this the letters vanish the
# same way the digits did -- the caller hears four digits and no prefix.
_LETTER_BN = {
    "a": "এ", "b": "বি", "c": "সি", "d": "ডি", "e": "ই", "f": "এফ", "g": "জি",
    "h": "এইচ", "i": "আই", "j": "জে", "k": "কে", "l": "এল", "m": "এম",
    "n": "এন", "o": "ও", "p": "পি", "q": "কিউ", "r": "আর", "s": "এস",
    "t": "টি", "u": "ইউ", "v": "ভি", "w": "ডব্লিউ", "x": "এক্স", "y": "ওয়াই", "z": "জেড",
}


# story title: Figures are spoken at a pace a caller can write down
# user story: As a patient noting a price or a reference, I want it grouped
#   and slower, so that I do not have to ask twice.
# acceptance criteria: Prices, phone numbers and reference identifiers are
#   spoken with grouping and a reduced rate through the per-request speed
#   parameter. A listening test confirms callers transcribe correctly on
#   first hearing.
#
# GROUPING POLICY -- PROVISIONAL, PENDING THE LISTENING STUDY.
#
# None of the numbers below is an acceptance criterion. The story asks for
# "grouping"; how many digits go in a group is a design decision, and these
# are the ones a Kolkata caller is most likely to expect rather than ones
# anybody has measured. They are gathered here, in one table with a version
# on it, precisely so the study can move them without touching any logic --
# and so a result can say WHICH policy it validated. See
# docs/listening-test.md.
#
# THE SEPARATOR IS A COMMA, and that is a deliberate REUSE of an existing
# meaning rather than a free win. tts_server._split_for_prosody already
# treats "," as a clause boundary and splices PAUSE_S["clause"] of real
# silence there, so a comma inside a number produces an audible group break
# with no server change at all. But the server cannot tell a grouping comma
# from a grammatical one, and it gives both the same pause. Whether a
# clause-length gap is the right gap between digit groups is exactly the
# kind of question only a handset can answer.
#
# One consequence worth knowing before listening: a sentence that had no
# comma becomes clause-split as a whole once a grouped number is in it, so
# the words AROUND the number gain a boundary too. That is the intended
# effect -- a small breath before a number is what a person does -- but it
# is a prosody change to the carrier sentence, not only to the figure.
GROUPING_POLICY_VERSION = "2026-09-12.a"

GROUP_SEPARATOR = ","

# Digit-count -> group sizes, for lengths with a convention worth honouring.
# Ten digits is an Indian mobile number and 5+5 is how one is written on a
# form and read back at a counter.
_GROUPS_BY_LENGTH = {10: (5, 5)}

# Everything else is cut into near-equal groups no longer than this.
MAX_GROUP_DIGITS = 4


def _group_sizes(n: int) -> tuple[int, ...]:
    """-> how many characters go in each spoken group."""
    if n <= 0:
        return ()
    if n in _GROUPS_BY_LENGTH:
        return _GROUPS_BY_LENGTH[n]
    if n <= MAX_GROUP_DIGITS:
        return (n,)
    # Near-equal rather than "fours with a stray remainder": a 9-digit run
    # reads better as 3+3+3 than as 4+4+1, and a group of one is not a group.
    count = -(-n // MAX_GROUP_DIGITS)
    base, extra = divmod(n, count)
    return tuple(base + (1 if i < extra else 0) for i in range(count))


def _in_groups(chars: str) -> list[str]:
    out, index = [], 0
    for size in _group_sizes(len(chars)):
        out.append(chars[index:index + size])
        index += size
    return out


def _read_chars(chars: str) -> str:
    spoken = []
    for ch in chars:
        if ch.isdigit():
            spoken.append(_ONES_TO_99[int(ch)])
        elif ch.isalpha() and ch.lower() in _LETTER_BN:
            spoken.append(_LETTER_BN[ch.lower()])
    return " ".join(spoken)


def spell_out(s: str) -> str:
    """Character-by-character, the way an ID is read over a phone -- now in
    the groups the ID itself already has.

    CONTRACT CHANGE, stated rather than slipped in: this used to return a
    flat run of words and now returns a GROUPED one. The old behaviour threw
    away the hyphens in "KCD-20260914-4A2F" before speech, so a caller heard
    fifteen tokens in one breath with no boundary where their pen would
    pause -- the structure was in the string and was discarded on the way to
    the synthesiser.

    Group boundaries come from the identifier's own separators first, and
    any run longer than MAX_GROUP_DIGITS is cut again so no single group is
    too long to hold. Nothing is reordered and no character is dropped: the
    digit-fidelity guarantee in tests/test_number_fidelity.py is unchanged,
    and this function is still the one place that decides how an ID sounds.
    """
    groups = []
    for part in re.split(r"[^0-9A-Za-z]+", s):
        if not part:
            continue
        groups.extend(_read_chars(chunk) for chunk in _in_groups(part))
    return f"{GROUP_SEPARATOR} ".join(g for g in groups if g)


def digits_one_by_one(s: str) -> str:
    """Digit-by-digit, ungrouped, the way a person reads a short run aloud --
    "নয় আট সাত" not "নয়শো সাতাশি".

    Deliberately NOT grouped, and left exactly as it was. Its other caller is
    the fractional half of a decimal, where a group separator would be wrong
    -- "দুইশো পঞ্চাশ দশমিক পাঁচ, শূন্য" is not a thing anyone says. Grouping
    is opt-in through grouped_digits() below, so the decimal path stays
    byte-identical to what it produced before this story.
    """
    return " ".join(_ONES_TO_99[int(c)] if c.isdigit() else c for c in s if not c.isspace())


def grouped_digits(s: str) -> str:
    """Digit-by-digit, in groups, for a number the caller is writing down."""
    digits = "".join(c for c in s if not c.isspace())
    return f"{GROUP_SEPARATOR} ".join(_read_chars(chunk) for chunk in _in_groups(digits))


# ------------------------------------------------------------------ time

_HOUR_WORD = {
    1: "একটা", 2: "দুটো", 3: "তিনটে", 4: "চারটে", 5: "পাঁচটা", 6: "ছটা",
    7: "সাতটা", 8: "আটটা", 9: "নটা", 10: "দশটা", 11: "এগারোটা", 12: "বারোটা",
}


def _day_part(hour24: int) -> str:
    if 4 <= hour24 < 12:
        return "সকাল"
    if 12 <= hour24 < 16:
        return "দুপুর"
    if 16 <= hour24 < 18:
        return "বিকেল"
    if 18 <= hour24 < 20:
        return "সন্ধ্যা"
    return "রাত"


def time_to_bn_words(hh: int, mm: int) -> str:
    """Bengali speakers say সাড়ে/সোয়া/পৌনে for :30/:15/:45 -- reading
    "দশটা ত্রিশ মিনিট" instead is understandable but immediately marks the
    voice as a machine."""
    part = _day_part(hh)
    h12 = hh % 12 or 12
    if mm == 0:
        return f"{part} {_HOUR_WORD[h12]}"
    if mm == 30:
        return f"{part} সাড়ে {_HOUR_WORD[h12]}"
    if mm == 15:
        return f"{part} সোয়া {_HOUR_WORD[h12]}"
    if mm == 45:
        nxt = (h12 % 12) + 1
        return f"{_day_part((hh + 1) % 24)} পৌনে {_HOUR_WORD[nxt]}"
    return f"{part} {_HOUR_WORD[h12]} বেজে {number_to_bn_words(mm)} মিনিট"


def time_to_english_words(hh: int, mm: int) -> str:
    """Convert time to English words for Hinglish support."""
    h12 = hh % 12 or 12
    period = "AM" if hh < 12 else "PM"
    
    if mm == 0:
        return f"{_ENGLISH_ONES[h12]} o'clock {period}"
    if mm == 30:
        return f"half past {_ENGLISH_ONES[h12]} {period}"
    if mm == 15:
        return f"quarter past {_ENGLISH_ONES[h12]} {period}"
    if mm == 45:
        nxt = (h12 % 12) + 1
        return f"quarter to {_ENGLISH_ONES[nxt]} {period}"
    return f"{_ENGLISH_ONES[h12]} {number_to_english_words(mm)} {period}"


def time_to_hinglish_words(hh: int, mm: int) -> str:
    """Convert time to Hinglish (Hindi-English mix) words.
    
    Hinglish speakers often use Hindi time words like "baje" for hours
    and English for minutes, or mix them naturally.
    """
    h12 = hh % 12 or 12
    # Hindi number words for hours (common in Hinglish)
    _HINDI_HOURS = {
        1: "ek", 2: "do", 3: "teen", 4: "chaar", 5: "paanch",
        6: "chhe", 7: "saat", 8: "aath", 9: "nau", 10: "das", 11: "gyaarah", 12: "baarah"
    }
    
    period = "morning" if 4 <= hh < 12 else ("afternoon" if 12 <= hh < 16 else ("evening" if 16 <= hh < 20 else "night"))
    
    if mm == 0:
        return f"{_HINDI_HOURS[h12]} baje {period}"
    if mm == 30:
        return f"{_HINDI_HOURS[h12]} baje {number_to_hinglish_words(mm)} {period}"
    if mm == 15:
        return f"{_HINDI_HOURS[h12]} baje {number_to_hinglish_words(mm)} {period}"
    if mm == 45:
        nxt = (h12 % 12) + 1
        return f"{_HINDI_HOURS[nxt]} baje {number_to_hinglish_words(60 - mm)} {period}"
    return f"{_HINDI_HOURS[h12]} baje {number_to_hinglish_words(mm)} {period}"


# ------------------------------------------------------------------ date

_MONTHS_BN = [
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
]


def date_to_bn_words(y: int, m: int, d: int) -> str:
    if not 1 <= m <= 12:
        return f"{number_to_bn_words(d)} তারিখ"
    return f"{_MONTHS_BN[m - 1]} মাসের {number_to_bn_words(d)} তারিখ"


_MONTHS_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def date_to_english_words(y: int, m: int, d: int) -> str:
    """Convert date to English words for Hinglish support."""
    if not 1 <= m <= 12:
        return f"{number_to_english_words(d)} date"
    return f"{_MONTHS_EN[m - 1]} {number_to_english_words(d)}, {number_to_english_words(y)}"


_MONTHS_HI = [
    "Janvari", "Febrari", "March", "April", "May", "June",
    "Julai", "August", "September", "October", "November", "December",
]


def date_to_hinglish_words(y: int, m: int, d: int) -> str:
    """Convert date to Hinglish (Hindi-English mix) words.

    Hinglish speakers often use English month names with Hindi grammar
    or mixed date formats.
    """
    if not 1 <= m <= 12:
        return f"{number_to_hinglish_words(d)} tarikh"
    return f"{_MONTHS_HI[m - 1]} {number_to_hinglish_words(d)}, {number_to_hinglish_words(y)}"


# --------------------------------------------------------------- weekday
#
# ADDED BY SOURAV -- "Caller asks when a doctor sits" story. clinic-api's
# DoctorSchedule.weekday is an int 0=Monday..6=Sunday (see that model's own
# docstring); this is the SPEAKING direction (index -> a word a caller
# hears), the mirror image of agent/slot_parse.py's `_WEEKDAYS_BN` dict
# (word -> index, used to PARSE a caller's spoken weekday back when they're
# choosing a booking date). The two are deliberately separate: different
# module, different direction, different caller -- slot_parse.py's dict
# only needs Bengali (bookings are parsed from what the caller literally
# said), while this needs all four reply languages so
# reply_templates.py::doctor_schedule_reply() can speak a schedule back in
# whichever language it was asked to answer in. Kept in this module rather
# than reply_templates.py because it is a pure "index -> spoken word"
# lookup, the same shape as _MONTHS_BN/_MONTHS_EN/_MONTHS_HI just above.
_WEEKDAYS_BN = [
    "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার", "রবিবার",
]
_WEEKDAYS_EN = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]
_WEEKDAYS_HI = [
    "Somwar", "Mangalwar", "Budhwar", "Guruwar", "Shukrawar", "Shanivar", "Raviwar",
]
_WEEKDAYS_BANGLISH = [
    "Sombar", "Mongolbar", "Budhbar", "Brihospotibar", "Shukrobar", "Shonibar", "Robibar",
]


def weekday_to_words(weekday: int, language: str = "bengali") -> str:
    """0=Monday..6=Sunday -> the spoken weekday name in `language`.

    Raises IndexError/ValueError on an out-of-range int rather than
    silently wrapping or returning a placeholder -- a weekday value only
    ever comes from clinic-api's DoctorSchedule column (always 0-6 by
    that model's own CHECK-equivalent usage) or a value this codebase
    computed itself with `datetime.date.weekday()` (also always 0-6), so
    an out-of-range value here means a real bug upstream that should
    surface loudly, not one this function should paper over.
    """
    if language == "english":
        return _WEEKDAYS_EN[weekday]
    elif language == "hinglish":
        return _WEEKDAYS_HI[weekday]
    elif language == "banglish":
        return _WEEKDAYS_BANGLISH[weekday]
    else:  # bengali (default)
        return _WEEKDAYS_BN[weekday]


# Buckets chosen so every value clinic-api/seed.py actually seeds today
# (1, 4, 6, 12, 24, 48, 72) lands in a distinct bucket -- see
# reply_templates.py's test_rate_reply() for the one call site.
_DURATION_BUCKETS = (
    (6, {"english": "within a few hours", "bengali": "কয়েক ঘণ্টার মধ্যে",
         "hinglish": "kuch hi ghanton mein", "banglish": "kayek ghontar modhye"}),
    (12, {"english": "within half a day", "bengali": "অর্ধেক দিনের মধ্যে",
          "hinglish": "aadhe din mein", "banglish": "ordhek diner modhye"}),
    (24, {"english": "within a day", "bengali": "একদিনের মধ্যে",
          "hinglish": "ek din mein", "banglish": "ek diner modhye"}),
    (48, {"english": "within two days", "bengali": "দুই দিনের মধ্যে",
          "hinglish": "do din mein", "banglish": "dui diner modhye"}),
    (72, {"english": "within three days", "bengali": "তিন দিনের মধ্যে",
          "hinglish": "teen din mein", "banglish": "tin diner modhye"}),
)


def hours_to_duration_phrase(hours: int, language: str = "bengali") -> str:
    """"The agent says how long results take" (Epic: Conversation --
    Information and Enquiry). AC: "Reporting time comes from the catalogue
    and is expressed as a natural duration rather than a number of hours
    read as a figure."

    Turns a raw clinic-api report_time_hours integer into a natural
    spoken phrase instead of a bare figure -- "within a day" rather than
    "24 hours". This is deliberately NOT the same discipline as rate_inr,
    dates and confirmation IDs (see tests/test_number_fidelity*.py and the
    "Numbers are never rounded, reordered or approximated" story): those
    are amounts, dates and identifiers, which that story protects
    byte-exact. A duration in hours is exactly the opposite case this
    story targets -- a person never says "your report will be ready in
    twenty-four hours", they say "by tomorrow" / "within a day". The
    catalogue value still drives every phrase below; nothing is
    approximated or guessed, only re-expressed the way a person would say
    it.

    Deliberately scoped: clinic-api/models.py's LabTest has exactly one
    report_time_hours integer per test -- no per-branch column, no
    per-weekday column, no variance of any kind exists in the schema
    today. This function never fabricates "it varies by day" or "it
    varies by branch" wording; the AC's "where turnaround varies ... that
    is stated" clause is a no-op against today's data, on purpose. If the
    catalogue ever gains real per-branch/per-weekday data, that is new
    catalogue data flowing into new slots -- a separate change, not
    something to invent here.

    For any hours value beyond today's catalogue (>72), falls back to an
    explicit day count ("within {N} days"). The bare integer this
    produces is not a violation of the "no raw figure" intent -- it is
    read through bn_normalize.verbalize() exactly like every other number
    in this codebase (the module you are in), which is what turns it into
    a spoken word before synthesis; a person genuinely does say "within
    five days" for a long turnaround, unlike reading out an hour count.
    """
    langs = {"english", "bengali", "hinglish", "banglish"}
    if language not in langs:
        language = "bengali"

    for ceiling, phrases in _DURATION_BUCKETS:
        if hours <= ceiling:
            return phrases[language]

    days = math.ceil(hours / 24)
    if language == "english":
        return f"within {days} days"
    elif language == "hinglish":
        return f"{days} din mein"
    elif language == "banglish":
        return f"{days} diner modhye"
    else:  # bengali
        return f"{days} দিনের মধ্যে"


# ------------------------------------------------------------- the pass

_RE_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b(\s*তারিখে?)?")
_RE_TIME_RANGE = re.compile(r"\b(\d{1,2}):(\d{2})\s*[-–—to]{1,2}\s*(\d{1,2}):(\d{2})\b")
_RE_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_RE_PHONE = re.compile(r"\b(\d{10,})\b")
# E4-S3: the optional trailing group. clinic-api issues IDs shaped
# KCD-20260915-4F0C; the pattern used to stop at "KCD-20260915", so the
# suffix's digits were read as a bare number and its Latin letters dropped by
# the tokenizer -- the caller heard "... পাঁচ-চারFশূন্যC" on every booking
# confirmation. A reference that cannot be read back cannot be quoted back.
_RE_CONF_ID = re.compile(r"\b([A-Z]{2,}-?\d{3,}(?:-[0-9A-Z]{2,})?)\b")
_RE_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_RE_INT = re.compile(r"\d+")

# Latin fragments that survive in clinic data (test names like "Uric Acid",
# "CBC", sample types like "Blood"). The Bengali tokenizer drops these the
# same way it drops digits, so anything still in Latin script after
# verbalization is a word the caller will never hear. The lookup service
# owns the Bengali aliases (clinic-api seeds aliases_bn); this table is the
# last-resort spoken form for the handful of fields the API returns in
# English regardless of how the caller phrased the question.
#
# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
# THIS TABLE IS THE "REWRITE" HALF OF E12-S2.
# Every entry is hand-chosen for a CLOSED, enumerated set -- there are five
# sample types in the catalogue and eight departments, and they are known in
# advance. That is what separates this from transliterating arbitrary text at
# runtime, which this codebase does not do: an automatic transliteration turns
# a detectable hole into a confident mispronunciation, which is worse. When a
# span is not in this table the reply is BLOCKED, not guessed at -- see
# agent/speakability.py.
#
# The Bengali-script renderings of borrowed clinical words ("ইমেজিং",
# "কার্ডিয়াক") are how a Kolkata caller actually says them, rather than a
# translation into literary Bengali they would not use -- the same principle
# E2-S4E states for the caller's own borrowings. They are NOT yet signed off
# by a native listener; that sign-off is E2-S4F's criterion and is outstanding.
_LATIN_SPOKEN_BN = {
    "blood": "রক্ত",
    "urine": "মূত্র",
    "stool": "মল",
    "serum": "সিরাম",
    "saliva": "লালা",
    "swab": "সোয়াব",
    "plasma": "প্লাজমা",
}


def _sub_int(match: re.Match) -> str:
    return number_to_bn_words(int(match.group(0)))


def verbalize(text: str, language: str = "bengali") -> str:
    """Rewrite `text` so every token in it is actually pronounceable by the
    Bengali FastPitch model. Order matters: the most specific patterns
    (dates, time ranges, long digit runs) must run before the bare-integer
    sweep, or "2026-08-25" gets read as three unrelated numbers.
    
    Args:
        text: The text to verbalize
        language: Target language - "bengali" (default), "english", or "hinglish"
    
    Returns:
        Verbalized text with numbers converted to words in the target language
    """
    if not text:
        return text

    # Select appropriate functions based on language
    if language == "english":
        number_func = number_to_english_words
        date_func = date_to_english_words
        time_func = time_to_english_words
        currency_word = " rupees "
        percent_word = " percent "
    elif language == "hinglish":
        number_func = number_to_hinglish_words
        date_func = date_to_hinglish_words
        time_func = time_to_hinglish_words
        currency_word = " rupaye "
        percent_word = " percent "
    else:  # bengali (default)
        number_func = number_to_bn_words
        date_func = date_to_bn_words
        time_func = time_to_bn_words
        currency_word = " টাকা "
        percent_word = " শতাংশ "

    text = text.translate(_BN_DIGITS)
    text = text.replace("₹", currency_word).replace("%", percent_word)

    # Date conversion - language-specific
    if language == "bengali":
        # group(4) is a "তারিখ"/"তারিখে" the template already supplied. Absorb it:
        # date_to_bn_words ends in "তারিখ", so leaving it produces "... তারিখ তারিখে".
        text = _RE_DATE.sub(
            lambda m: date_func(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            + ("ে" if (m.group(4) or "").strip().endswith("ে") else ""), text,
        )
    else:
        text = _RE_DATE.sub(
            lambda m: date_func(int(m.group(1)), int(m.group(2)), int(m.group(3))), text,
        )

    # Time range conversion
    if language == "bengali":
        text = _RE_TIME_RANGE.sub(
            lambda m: (f"{time_func(int(m.group(1)), int(m.group(2)))} থেকে "
                       f"{time_func(int(m.group(3)), int(m.group(4)))} পর্যন্ত"), text,
        )
    elif language == "hinglish":
        text = _RE_TIME_RANGE.sub(
            lambda m: (f"{time_func(int(m.group(1)), int(m.group(2)))} se "
                       f"{time_func(int(m.group(3)), int(m.group(4)))} tak"), text,
        )
    else:  # english
        text = _RE_TIME_RANGE.sub(
            lambda m: (f"{time_func(int(m.group(1)), int(m.group(2)))} to "
                       f"{time_func(int(m.group(3)), int(m.group(4)))}"), text,
        )

    # Time conversion
    text = _RE_TIME.sub(lambda m: time_func(int(m.group(1)), int(m.group(2))), text)
    
    # Confirmation ID and phone conversion
    text = _RE_CONF_ID.sub(lambda m: spell_out(m.group(1)), text)
    text = _RE_PHONE.sub(lambda m: digits_one_by_one(m.group(1)), text)
    text = _RE_DECIMAL.sub(
        lambda m: f"{number_to_bn_words(int(m.group(1)))} দশমিক {digits_one_by_one(m.group(2))}", text,
    )
    text = _RE_INT.sub(_sub_int, text)

    # Whole-word, case-insensitive: only rewrites a Latin word we have a
    # spoken Bengali form for. Anything else Latin is left alone and
    # reported by `unspeakable_spans()` rather than silently mangled.
    for latin, bn in _LATIN_SPOKEN_BN.items():
        text = re.sub(rf"\b{latin}\b", bn, text, flags=re.IGNORECASE)

    return re.sub(r"\s{2,}", " ", text).strip()


# UPDATED BY SOURAV -- real production bug, reported directly by the
# caller: "why voice is giving response only in bengali, not in english
# or hinglish or hindi, when the user asks in hindi aur hinglish or
# english." Root cause traced to TWO separate things:
#
# 1. Every reply-generating function in reply_templates.py and
#    report_flow.py already accepts a `language` argument (confirmed:
#    grepped every `def ..._reply(` / `def interpret_...` signature in
#    both files -- all of them already default to language="bengali" and
#    already have full 4-language bodies). detect_language() existed the
#    whole time but was NEVER CALLED anywhere in this repo -- only
#    imported into reply_templates.py, itself a pre-existing, separately-
#    flagged "imported but unused" pyflakes warning. main.py's dispatch
#    simply never called it and never passed `language=` to anything, so
#    every reply defaulted to Bengali on every real call regardless of
#    what the caller said. Fixed in main.py -- see that file's dispatch
#    functions for where this is now actually called and threaded
#    through.
# 2. Wiring this function in for real exposed a genuine bug in its OWN
#    logic (fixed below): text with SOME Bengali-script characters mixed
#    with mostly Latin script was labelled "hinglish" (Hindi+English).
#    That is the wrong label -- Bengali script mixed with Latin is a
#    BENGALI+ENGLISH code-switch, i.e. Banglish, not Hindi+English.
#    Confirmed live: detect_language("ami office e achi, phone \u09A7\u09B0\u09A4\u09C7
#    \u09AA\u09BE\u09B0\u09BF\u09A8\u09BF") returned "hinglish" despite containing not one Hindi word.
#    Meanwhile genuine Hinglish (transliterated Hindi + English, e.g.
#    "mera number kya hai") contains ZERO Bengali-script characters, so
#    under the old logic it always fell through to "english" -- the
#    function could never actually detect real Hinglish at all, despite
#    its own docstring and return type claiming to.
_HINGLISH_MARKERS = (
    "hai", "hain", "kya", "chahiye", "matlab", "nahi", "nahin", "aap",
    "mera", "mujhe", "kaise", "kab", "kitna", "kitne", "bhi", "abhi",
    "accha", "achha", "theek hai", "haan", "han", "sahi hai",
    "shukriya", "dhanyawad",
)
# Bengali-transliterated (Latin script) markers -- distinguishes a
# BANGLISH caller (Bengali+English code-switch, spoken/typed in Latin
# script with no actual Bengali unicode characters at all) from a
# HINGLISH one (Hindi+English). Reuses this project's own existing
# Banglish vocabulary precedent (agent/slot_parse.py's _AFFIRMATIVE/
# _NEGATIVE sets already list "thik ache"/"thik achhe"/"sob thik ache" as
# Banglish, distinct from Hinglish's "haan"/"theek hai") rather than
# inventing a new word list from scratch.
_BANGLISH_MARKERS = (
    "ache", "achi", "achhe", "hocche", "hoyeche", "korbo", "korchi",
    "korte", "lagbe", "hobe", "bolo", "bolchi", "bhalo", "kemon",
    "eta", "ota", "amar", "tumi", "apni", "kotha",
)
_HINGLISH_MARKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _HINGLISH_MARKERS) + r")\b", re.IGNORECASE
)
_BANGLISH_MARKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _BANGLISH_MARKERS) + r")\b", re.IGNORECASE
)


def detect_language(text: str) -> str:
    """Detect which of this project's 4 supported reply languages
    (see reply_templates.py's own LANGUAGE SUPPORT note -- English,
    Hinglish, Banglish, Bengali) a caller's utterance was most likely
    spoken in, so main.py's dispatch can pass the right `language=`
    argument to whichever reply function it calls next. See the
    UPDATED BY SOURAV comment just above for the real bug this fixes and
    the bug found while fixing it.

    Two-stage approach:
      1. Bengali-script ratio decides the clear-cut cases: mostly Bengali
         script -> "bengali"; SOME Bengali script mixed with mostly Latin
         -> "banglish" (a Bengali+English code-switch).
      2. For text that is entirely or almost entirely Latin script
         (where script-ratio alone cannot tell a transliterated Bengali
         speaker, a transliterated Hindi speaker, and a genuine English
         speaker apart), a small recognizable marker-word vocabulary
         decides between "hinglish" and "banglish" -- reusing this
         project's own existing Hinglish/Banglish word-list precedent
         (agent/slot_parse.py's _AFFIRMATIVE/_NEGATIVE) rather than
         inventing new vocabulary. No markers of either kind -> "english".

    This is a best-effort heuristic, same trust model as fast_path.py's
    own local matching -- it will not be perfect on every utterance
    (a bare "CBC" or a doctor's surname alone gives no language signal at
    all and falls through to "english"), but it is a large, concrete
    improvement over "always Bengali" or "always the wrong label for a
    mixed-script utterance", and it reaches every one of the 4 languages
    reply_templates.py already knows how to speak.
    """
    if not text:
        return "bengali"

    bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    total_chars = len(text.replace(" ", "").replace("\n", ""))

    if total_chars == 0:
        return "bengali"

    bengali_ratio = bengali_chars / total_chars

    if bengali_ratio > 0.7:
        return "bengali"
    if bengali_ratio > 0.2:
        # UPDATED BY SOURAV -- was "hinglish" (wrong label, see the
        # comment above this function). Bengali script mixed with Latin
        # script is a Bengali+English code-switch: Banglish.
        return "banglish"

    # Almost entirely Latin script -- script-ratio alone cannot tell
    # transliterated Bengali, transliterated Hindi, and genuine English
    # apart. Fall back to recognizable marker words.
    banglish_hits = len(_BANGLISH_MARKER_RE.findall(text))
    hinglish_hits = len(_HINGLISH_MARKER_RE.findall(text))

    if banglish_hits > hinglish_hits and banglish_hits > 0:
        return "banglish"
    if hinglish_hits > 0:
        return "hinglish"
    return "english"


_RE_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z .'-]*")


# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
def unspeakable_spans(text: str) -> list[str]:
    """Latin-script runs left after verbalize() -- these WILL be dropped
    silently by the tokenizer, exactly like the digits were.

    SINGLE LETTERS COUNT. This used to filter out anything one character
    long, which was a blind spot rather than noise reduction: the tokenizer
    drops a lone "T" exactly as completely as it drops "TSH", and the
    catalogue really does contain names shaped that way -- "Thyroid Profile
    (T3 T4 TSH)" verbalizes to "(Tতিন Tচার TSH)", of which the old filter
    reported only "TSH" and let both stray T's through unrecorded. Anything
    built on top of this function inherits its blind spots, so a gate that
    blocks unspeakable replies could not be trusted while this one existed.

    KNOWN LIMIT: this models Latin-versus-Bengali only, because the matrix
    language is currently hardcoded Bengali (agent/asr.py). It does not see
    Devanagari, symbols, or anything else outside the checkpoint's vocabulary
    -- those are caught, if at all, by tts_server.py's empty-chunk log.
    """
    return [s for s in (m.group(0).strip() for m in _RE_LATIN_RUN.finditer(text)) if s]
