"""Lightweight local parsers for slot values that fill an appointment
booking a field at a time, driven by main.py's CallSession.pending state
machine.

WHY LOCAL PARSING, NOT ANOTHER LLM CALL
----------------------------------------
Once a caller is inside a booking flow, main.py already knows exactly
which single field it just asked for. Re-running the LLM's full
intent+multi-slot extraction on a bare reply like "আজ" or "সাড়ে দশটা" is
both slower (a 7B-model round trip per field) and unreliable: llm.py's
SYSTEM_PROMPT_TEMPLATE classifies intent from cue words like "বুক" /
"অ্যাপয়েন্টমেন্ট", none of which appear in a bare "আজ" -- a caller who is
already three questions into booking is not going to repeat "আমি
অ্যাপয়েন্টমেন্ট করতে চাই" every turn just so the classifier has something
to key off. This is also the root cause of the original bug report: the
LLM sees each utterance in isolation with no memory of the conversation,
so a follow-up like "10 টায়" on its own has no doctor_name/date attached
to it and the LLM has nothing to extract them from.

So each of these functions answers one narrow question -- "does this
utterance look like a date/time/phone number, and if so, which one" --
using the same trust model as fast_path.py: return None whenever not
confident, and let main.py re-prompt (or give up and fall back to a
fresh LLM classification) rather than guess.
"""
from __future__ import annotations

import datetime
import re
import unicodedata

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_WEEKDAYS_BN = {
    "সোমবার": 0, "সোম": 0,
    "মঙ্গলবার": 1, "মঙ্গল": 1,
    "বুধবার": 2, "বুধ": 2,
    "বৃহস্পতিবার": 3, "বৃহস্পতি": 3, "বিহস্পতি": 3,
    "শুক্রবার": 4, "শুক্র": 4,
    "শনিবার": 5, "শনি": 5,
    "রবিবার": 6, "রবি": 6,
}

_RELATIVE_DAYS = {
    "আজ": 0, "আজকে": 0, "আজকেই": 0,
    "কাল": 1, "আগামীকাল": 1, "কালকে": 1,
    "পরশু": 2, "পরশুদিন": 2,
}

# E13-S7 follow-up: the same three days said in Latin script -- English, and
# the Banglish/Hinglish spellings a code-switching caller (or the ASR) uses.
# Matched as whole words with \b, which IS reliable for Latin script (the
# Bengali-aware boundary below exists because \b is not). "kal" here means
# tomorrow: this parser only ever reads a day for something being booked or
# asked about ahead, never a day gone by.
_RELATIVE_DAYS_LATIN = {
    "today": 0, "aaj": 0, "aj": 0, "aajke": 0, "ajke": 0,
    "tomorrow": 1, "kal": 1, "kaal": 1, "kalke": 1, "agamikal": 1,
    "day after tomorrow": 2, "porshu": 2, "porshudin": 2, "parso": 2, "parson": 2,
}

# "Caller says tomorrow, day after, or next Monday": weekday names in Latin
# script (English, Hinglish, Banglish) are read by code too, so a Latin-script
# caller's day can be checked against the model's reading. Like the Bengali
# names they mean the COMING one; a qualifier ("next", "agle") is not read
# here on purpose -- where the model reads it as "a week later", the two
# readings differ and the caller is asked (docs/stories/relative-dates-plan.md).
_WEEKDAYS_LATIN = {
    "monday": 0, "somvaar": 0, "somvar": 0, "sombar": 0,
    "tuesday": 1, "mangalvaar": 1, "mangalvar": 1, "mongolbar": 1,
    "wednesday": 2, "budhvaar": 2, "budhvar": 2, "budhbar": 2,
    "thursday": 3, "guruvaar": 3, "guruvar": 3, "brihaspativaar": 3, "brihospotibar": 3,
    "friday": 4, "shukravaar": 4, "shukravar": 4, "shukrobar": 4,
    "saturday": 5, "shanivaar": 5, "shanivar": 5, "shonibar": 5,
    "sunday": 6, "ravivaar": 6, "ravivar": 6, "itvaar": 6, "robibar": 6,
}

# What a day word MEANS, as a key the reply templates can say in any
# language ("tomorrow" -> "আগামীকাল" / "Tomorrow" / "Kal"), for the recheck.
_RELATIVE_KEY = {0: "today", 1: "tomorrow", 2: "day_after_tomorrow"}
_WEEKDAY_KEY = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# Kept deliberately small and exact-match only (see is_affirmative /
# is_negative below) -- these gate whole-utterance decisions like "abandon
# the booking flow" and, since the booking-readback story, "confirm the
# write", so a false hit on a substring inside an unrelated reply (e.g. a
# patient name that happens to contain "না") would be a much worse failure
# than occasionally not recognising a yes/no.
#
# This system's own docstrings (reply_templates.py's LANGUAGE SUPPORT
# note) commit to three reply languages -- Bengali, English, Hinglish --
# and Kolkata callers code-switch Bengali with English just as often as
# Hindi with English ("Banglish"), so the affirmative/negative vocabulary
# a caller might actually SAY is not Bengali-only regardless of which
# language the AGENT chose to speak the prompt in. Previously this set
# only recognised Bengali words, so an English "yes"/"no" or a
# transliterated "haan"/"nahi" fell through to the unparseable-reply retry
# path instead of being understood immediately.
_AFFIRMATIVE = {
    # Bengali
    "হ্যাঁ", "হ্যা", "হুম", "হুঁ", "ঠিক", "ঠিক আছে", "ওই দিন", "ওইদিন",
    "সেদিন", "সেদিনই", "সেই দিন", "চলবে", "ওকে", "হবে",
    # English
    "yes", "yeah", "yep", "yup", "correct", "right", "that's right",
    "ok", "okay", "sure", "confirmed", "confirm",
    # Hinglish / Banglish (Latin-script transliteration -- shared
    # vocabulary between the two, since both are "regional language +
    # English" code-switches)
    "haan", "han", "haa", "thik ache", "thik achhe", "theek hai",
    "sahi hai", "sob thik ache", "sob thik",
}
_NEGATIVE = {
    # Bengali
    "না", "নাহ", "না না", "লাগবে না", "থাক", "দরকার নেই", "ইচ্ছা নেই",
    "না থাক", "লাগবে নাহ",
    # English
    "no", "nope", "not correct", "wrong", "incorrect", "not right",
    # Hinglish / Banglish
    "nahi", "nahin", "na", "galat", "thik na", "thik nei",
}


def _strip(text: str) -> str:
    # .lower() is a no-op on Bengali script (it has no case), so this is
    # safe for the existing Bengali entries and required for the English /
    # Hinglish / Banglish ones added above ("Yes", "YES" and "yes" must
    # all match).
    return text.strip().strip("।!?., ").lower()


def is_affirmative(text: str) -> bool:
    return _strip(text) in _AFFIRMATIVE


def is_negative(text: str) -> bool:
    return _strip(text) in _NEGATIVE


# story title: Every critical value is read back before it is used
# user story: As a patient giving a phone number, I want it read back, so that
#   a misheard digit does not send my report to a stranger.
# acceptance criteria: Phone numbers, dates, times and names are confirmed
#   aloud before any write, and a rejection opens a correction path rather than
#   repeating the prompt. Readback is mandatory regardless of confidence for
#   values that affect a write.
#
# Ported from dev_sourav. Used only after the pre-write readback is REJECTED:
# the caller is asked which single value is wrong instead of the five-field
# flow restarting, and this maps whatever they name back to one booking field.
#
# ORDER MATTERS, and it is not _BOOKING_FIELDS order or alphabetical. "নাম"
# (name) is a substring of how a caller says "the doctor's name"
# ("ডাক্তারের নাম") at least as often as they mean the patient's, so
# doctor_name and phone are matched FIRST on their own unambiguous words. A
# bare "নাম" then falls through to patient_name, which is the only reading
# left once the others are excluded.
#
# English/Hinglish variants are matched alongside the Bengali words (merged
# from dev_sourav) since callers code-switch mid-sentence.
_CORRECTION_FIELD_WORDS = {
    "doctor_name": ("ডাক্তার", "ডক্তার", "doctor"),
    "date": ("তারিখ", "দিন", "date"),
    "time_slot": ("সময়", "টাইম", "time"),
    "phone": ("ফোন", "নম্বর", "নাম্বার", "phone", "number"),
    "patient_name": ("নাম", "name"),
}
_CORRECTION_FIELD_ORDER = ("doctor_name", "date", "time_slot", "phone", "patient_name")


def parse_correction_field(text: str) -> str | None:
    """-> one booking field name, or None if the reply does not confidently
    name one.

    Same trust model as the rest of this module: return None rather than
    guess, and let main.py re-ask. Guessing here would silently re-collect the
    wrong field and then read the SAME wrong value back, which is worse than
    asking twice.
    """
    t = _strip(text).lower()
    if not t:
        return None
    for field in _CORRECTION_FIELD_ORDER:
        if any(word in t for word in _CORRECTION_FIELD_WORDS[field]):
            return field
    return None


# Any character in the Bengali Unicode block -- letters, vowel signs, the
# nukta, and the ০-৯ digits. Used as a word boundary that actually works for
# this script; see the comment inside parse_date().
_BN_CHAR = r"[ঀ-৿]"


def _bn_bounded(word: str, text: str) -> bool:
    """Is `word` present in `text` as a whole word, Bengali-aware?"""
    return re.search(rf"(?<!{_BN_CHAR}){re.escape(word)}(?!{_BN_CHAR})", text) is not None


def _match_day_word(t: str) -> tuple[str, int] | None:
    """-> ("relative", days from today) or ("weekday", 0-6), for the FIRST
    kind found: a relative day word (Bengali, then Latin script) before a
    weekday name, longest word first -- the order parse_date always used."""
    for word in sorted(_RELATIVE_DAYS, key=len, reverse=True):
        if _bn_bounded(word, t):
            return "relative", _RELATIVE_DAYS[word]
    lowered = t.lower()
    for word in sorted(_RELATIVE_DAYS_LATIN, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return "relative", _RELATIVE_DAYS_LATIN[word]
    for word in sorted(_WEEKDAYS_BN, key=len, reverse=True):
        if _bn_bounded(word, t):
            return "weekday", _WEEKDAYS_BN[word]
    for word in sorted(_WEEKDAYS_LATIN, key=len, reverse=True):
        if re.search(rf"\b{word}\b", lowered):
            return "weekday", _WEEKDAYS_LATIN[word]
    return None


def spoken_day_word(text: str) -> str | None:
    """-> what the caller's day word MEANS ("today" | "tomorrow" |
    "day_after_tomorrow" | "monday" .. "sunday"), or None when the sentence
    names no such word (an explicit "১৫ তারিখ", or no day at all).

    E13-S7 follow-up. A date the agent CALCULATED from one of these words is
    read back to the caller with the word and the weekday ("আগামীকাল মানে
    রবিবার, ...") before it is used -- see main.py's _apply_booking_date."""
    matched = _match_day_word(_nfc(text or "").translate(_BN_DIGITS))
    if matched is None:
        return None
    kind, n = matched
    return _RELATIVE_KEY[n] if kind == "relative" else _WEEKDAY_KEY[n]


def parse_date(text: str, today: datetime.date | None = None,
                offered_date: str | None = None) -> str | None:
    """-> ISO date string, or None if not confident.

    `offered_date` is the ISO date main.py already spoke out loud (e.g.
    doctor_availability_reply's "next available: 2026-09-08, do you want
    that day or another one?") -- a bare affirmative reply ("হ্যাঁ", "ওই
    দিন") confirms THAT date, not literally "today"."""
    today = today or datetime.date.today()
    t = text.translate(_BN_DIGITS).strip()

    if offered_date and is_affirmative(t):
        return offered_date

    # story title: The model never originates a fact
    # user story: As a clinical lead, I want every price, date and identifier
    #   to come from a verified system response, so that a wrong answer is a
    #   data bug rather than a model bug.
    # acceptance criteria: Every factual sentence is a template substitution
    #   from a validated tool response and the model is never shown a figure
    #   it could restate. An automated assertion on every commit proves no
    #   model-composed span reaches synthesis on a factual intent.
    #
    # These two loops were `if word in t` -- a bare substring test, which is
    # a real, reproducible bug and not a theoretical one:
    #
    #     parse_date("সকাল দশটায়")   -> TOMORROW      ("সকাল" contains "কাল")
    #     parse_date("বিকাল পাঁচটায়") -> TOMORROW      ("বিকাল" contains "কাল")
    #
    # A caller answering "কোন দিন চান?" with "সকালে" was silently given
    # tomorrow's date. This function is the highest authority in
    # agent/date_calc.resolve()'s order of precedence -- it outranks the
    # model's own interpretation -- so a parser that invents a date is the
    # same defect as a model that invents one, only harder to notice.
    #
    # _bn_bounded() is the fix, and it is the same fix parse_time() already
    # documents for টা/টার/টায়: \b cannot be used here because Bengali vowel
    # signs and the nukta are combining marks that Python's \w does not count
    # as word characters, so \b matches in the middle of a word. Asserting
    # "no Bengali character adjacent" instead does what \b was meant to do.
    #
    # Longest key first so a shorter key can never consume part of a longer
    # one -- "কাল" must not fire inside "আগামীকাল". The matching itself now
    # lives in _match_day_word(), shared with spoken_day_word() below, so the
    # word that is CONFIRMED with the caller is always the word the date was
    # computed from.
    matched = _match_day_word(t)
    if matched is not None:
        kind, n = matched
        if kind == "relative":
            return (today + datetime.timedelta(days=n)).isoformat()
        days_ahead = (n - today.weekday()) % 7
        days_ahead = days_ahead or 7  # naming today's weekday means NEXT week's
        return (today + datetime.timedelta(days=days_ahead)).isoformat()

    # Explicit ISO date (e.g. carried over from an earlier LLM extraction).
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return None

    # "১৫ তারিখ" / "15 তারিখে" -- day-of-month in the current month,
    # rolling into next month if that day has already passed.
    #
    # (?<!\w) / (?!\w) here, NOT \b: Bengali vowel signs and the nukta
    # (e.g. the ে in তারিখে) are Unicode combining marks, which Python's
    # \w does NOT count as word characters. That makes \b fail to match
    # at the boundary right after them -- so "তারিখে" sitting at the very
    # end of an utterance (the normal case) silently never matched. The
    # lookaround forms only check "not a word character adjacent", which
    # is true at end-of-string and before whitespace/punctuation either
    # way, so they don't have this blind spot. See parse_time() below for
    # the same fix applied to টা/টার/টায়, where it mattered even more.
    m = re.search(r"(?<!\w)(\d{1,2})\s*(?:তারিখ|তারিখে|ই)(?!\w)", t)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 31:
            try:
                candidate = today.replace(day=day)
            except ValueError:
                return None  # e.g. "31 তারিখ" in a 30-day month -- ask again
            if candidate < today:
                next_month = today.month % 12 + 1
                next_year = today.year + (1 if today.month == 12 else 0)
                try:
                    candidate = candidate.replace(year=next_year, month=next_month)
                except ValueError:
                    return None
            return candidate.isoformat()

    return None


_HOUR_WORD_TO_NUM = {
    "একটা": 1, "দুটো": 2, "দুইটা": 2, "তিনটে": 3, "তিনটা": 3, "চারটে": 4, "চারটা": 4,
    "পাঁচটা": 5, "ছটা": 6, "ছয়টা": 6, "সাতটা": 7, "আটটা": 8, "নটা": 9, "নয়টা": 9,
    "দশটা": 10, "এগারোটা": 11, "বারোটা": 12,
}

# Bengali day-part words -> the 24h hours they cover, used only to decide
# whether a bare 1-12 number means AM or PM.
_DAYPART_WORDS = ("সকাল", "দুপুর", "বিকেল", "সন্ধ্যা", "রাত")


def _to_24h(hour12: int, daypart: str | None) -> int:
    hour12 = hour12 % 12
    if daypart in ("দুপুর", "বিকেল", "সন্ধ্যা", "রাত"):
        return hour12 + 12 if hour12 != 0 else 12
    if daypart == "সকাল":
        return hour12 if hour12 != 0 else 12
    # No day-part cue spoken: this clinic's chamber hours run evenings
    # (see clinic-api/seed.py's DoctorSchedule rows, typically 17:00-20:00),
    # so treat a bare 1-7 as PM and 8-12 as AM -- the common case for "the
    # doctor's hours are 6 to 8" said without "সন্ধ্যা" in front of it.
    if 1 <= hour12 <= 7:
        return hour12 + 12
    return hour12 if hour12 != 0 else 12


def _extract_hour12_after(t: str, prefix: str) -> int | None:
    """Look immediately after `prefix` (সাড়ে/সোয়া/পৌনে) for the hour it
    modifies, in EITHER digit form ("সাড়ে ৭টা") or word form ("সাড়ে
    সাতটা") -- real callers and ASR output mix both freely, and the
    original version here only ever checked the digit form, so "সাড়ে
    দশটা" (half past ten, said as a word) silently lost its "half past"
    and was parsed as a bare 10:00."""
    idx = t.find(prefix)
    if idx == -1:
        return None
    rest = t[idx + len(prefix):].lstrip()
    # The digit and টা/টার/টায় are written with NO space between them
    # ("সাড়ে ৭টা"), so a plain (?!\w) right after the digit would wrongly
    # reject it -- ট is a word character. Consume that suffix as PART of
    # the match instead of asserting against it.
    m = re.match(r"(\d{1,2})(?:টা|টার|টায়)?(?!\w)", rest)
    if m:
        return int(m.group(1))
    for word, hour12 in _HOUR_WORD_TO_NUM.items():
        if rest.startswith(word):
            return hour12
    return None


def parse_time(text: str) -> str | None:
    """-> "HH:MM" in 24h, or None if not confident."""
    t = text.translate(_BN_DIGITS).strip()

    m = re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", t)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"

    daypart = next((w for w in _DAYPART_WORDS if w in t), None)

    hour12 = _extract_hour12_after(t, "সাড়ে")
    if hour12 is not None:
        return f"{_to_24h(hour12, daypart):02d}:30"
    hour12 = _extract_hour12_after(t, "সোয়া")
    if hour12 is not None:
        return f"{_to_24h(hour12, daypart):02d}:15"
    hour12 = _extract_hour12_after(t, "পৌনে")
    if hour12 is not None:
        h = _to_24h(hour12, daypart)
        return f"{(h - 1) % 24:02d}:45"

    # (?<!\w) / (?!\w), NOT \b -- see parse_date()'s তারিখ regex above for
    # why. This one matters even more: টা is the single most common way a
    # caller states an o'clock hour ("৭টা", "সন্ধ্যা ৭টা"), and it almost
    # always sits at the very end of the utterance -- exactly where \b
    # after a combining vowel sign (the া in টা) silently failed to match.
    # Confirmed by hand: "সন্ধ্যা ৭টা" and even a bare "৭টা" both returned
    # None under the old \b version.
    m = re.search(r"(?<!\w)(\d{1,2})\s*(?:টা|টার|টায়)(?!\w)", t)
    if m:
        return f"{_to_24h(int(m.group(1)), daypart):02d}:00"

    for word, hour12 in _HOUR_WORD_TO_NUM.items():
        if word in t:
            return f"{_to_24h(hour12, daypart):02d}:00"

    # Transliterated/English callers: "10 am", "10am", "10 pm".
    m = re.search(r"\b(\d{1,2})\s*([ap])\.?m\.?\b", t, re.IGNORECASE)
    if m:
        h = int(m.group(1)) % 12
        if m.group(2).lower() == "p":
            h += 12
        return f"{h:02d}:00"

    return None


# ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story.
# main_pcm.py's new "otp_code" pending state (see _continue_pending) uses
# this instead of running the caller's reply through agent/llm.py, for
# the SAME reason every other field in this file is parsed locally (see
# this module's docstring) -- PLUS a security reason unique to this one
# field: an OTP must never be sent to the LLM or the semantic cache at
# all. The LLM call is a real network hop to Ollama with its own request
# log, and agent/semantic_cache.py persists normalized utterance text as
# its cache key -- routing a 6-digit secret through either would be
# exactly the "secret-shaped literal in a log line" DoD gate 6 warns
# about, even though the OTP itself is a deliberately hardcoded prototype
# value (see clinic-api/seed.py and main.py's FRESH_OTP_CODE). Handling
# it here, alongside parse_phone, keeps it out of both.
#
# UPDATED BY SOURAV -- moved above parse_phone() (was originally defined
# only just above parse_otp(), further down this file) because parse_phone()
# now reuses this SAME word list for a real production bug fix -- see
# parse_phone()'s own docstring immediately below for the full writeup.
# Reused as-is, not duplicated or extended: still English digit words
# only (see parse_phone()'s docstring for why Bengali/Hindi spoken digit
# words are a separate, flagged, not-yet-closed gap, symmetric with the
# same pre-existing limitation this already had for OTP entry).
_DIGIT_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_DIGIT_WORD_RE = re.compile(r"\b(" + "|".join(_DIGIT_WORDS) + r")\b", re.IGNORECASE)


# E4-S3. NFC so "য়" arrives in one encoding whether it came from the ASR or
# from a literal in this file -- it has two, and they do not compare equal.
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# Single digits as a Bengali caller reads a number aloud, including the
# English words said in Bengali script. Matched as WHOLE TOKENS only (see
# _tokens), which is why these are not folded into _DIGIT_WORD_RE: \b does
# not behave on Bengali vowel signs.
_BN_DIGIT_WORDS = {_nfc(k): v for k, v in {
    "শূন্য": "0", "জিরো": "0",
    "এক": "1", "ওয়ান": "1",
    "দুই": "2", "দু": "2", "টু": "2",
    "তিন": "3", "থ্রি": "3",
    "চার": "4", "ফোর": "4",
    "পাঁচ": "5", "ফাইভ": "5",
    "ছয়": "6", "ছ": "6", "সিক্স": "6",
    "সাত": "7", "সেভেন": "7",
    "আট": "8", "এইট": "8",
    "নয়": "9", "নাইন": "9",
}.items()}

# How a Latin letter in a confirmation reference is spoken. The reverse of
# bn_normalize._LETTER_BN, kept local so this module stays a leaf; a
# round-trip test against bn_normalize.spell_out() catches any divergence.
_LETTER_WORDS = {_nfc(k): v for k, v in {
    "এ": "A", "বি": "B", "সি": "C", "ডি": "D", "ই": "E", "এফ": "F", "জি": "G",
    "এইচ": "H", "আই": "I", "জে": "J", "কে": "K", "এল": "L", "এম": "M",
    "এন": "N", "ও": "O", "পি": "P", "কিউ": "Q", "আর": "R", "এস": "S",
    "টি": "T", "ইউ": "U", "ভি": "V", "ডব্লিউ": "W", "এক্স": "X", "ওয়াই": "Y", "জেড": "Z",
}.items()}

_RE_TOKEN_SPLIT = re.compile(r"[\s,।\-–—.:/]+")


def _tokens(text: str) -> list[str]:
    return [t for t in _RE_TOKEN_SPLIT.split(_nfc(text).translate(_BN_DIGITS)) if t]


def parse_phone(text: str) -> str | None:
    """-> a 10-digit phone number, or None if the utterance doesn't
    contain enough digits to be one.

    UPDATED BY SOURAV -- fixes a real production bug, reported directly
    from a live call transcript. A caller was asked "Is my report ready?"
    (report_status), the agent asked for their registered phone number
    (RULE 14/15 -- identity is resolved by phone, never by name), and the
    caller answered by SPEAKING THE DIGITS AS WORDS:

        "Yes, write nine zero zero zero zero zero zero zero zero one."

    This function used to only understand literal digit characters
    (Bengali numerals, via _BN_DIGITS, or plain ASCII digits) -- spoken
    English number words were never converted to digits at all, so
    stripping every non-digit character from "nine zero zero zero zero
    zero zero zero zero one" left an EMPTY string every single time,
    which is always < 10 digits, which always returned None. The caller
    could repeat themselves as many times and as clearly as they liked
    (confirmed in the transcript -- they tried three different phrasings)
    and would be stuck in an infinite "can you tell me your registered
    phone number?" loop forever, because nothing about repeating the same
    kind of answer could ever succeed.

    Fixed by resolving spoken English digit words into digits FIRST,
    using the exact same _DIGIT_WORDS/_DIGIT_WORD_RE word list parse_otp()
    below already uses for the identical reason -- reused, not duplicated.
    "nine zero zero zero zero zero zero zero zero one" now correctly
    resolves to 9000000001.

    Deliberately still returns None (fails closed) for a spoken REPETITION
    shorthand like "eight zeros" (meaning the digit 0 repeated eight
    times) -- seen in the same transcript ("Nine then eight zeros and
    one") as the caller's own retry after the first phrasing wasn't
    understood. This is NOT fixed here: telling "eight zeros" (0 repeated
    8 times) apart from the caller instead meaning the two separate
    digits "eight" then "zero" is genuinely ambiguous, and a phone number
    gates a real caller's private report data -- guessing wrong here would
    silently produce the WRONG number and risk exposing (or refusing)
    the wrong person's report, which is a worse failure than asking the
    caller to repeat themselves once more. Same fail-closed posture this
    whole module already commits to everywhere else (see the module's own
    docstring: "return None whenever not confident, and let main.py
    re-prompt... rather than guess"). Flagged in this story's test report,
    not silently absorbed.
    """
    # E4-S3: Bengali single-digit words too ("নয় আট সাত ..."), resolved
    # TOKEN BY TOKEN so a digit word is never pulled out of the middle of an
    # unrelated word. Compound numbers ("নিরানব্বই" = 99) are deliberately
    # not recognised: a number read in pairs yields too few digits and the
    # caller is asked again -- the same fail-closed rule as "eight zeros".
    #
    # NUMERALS WIN. "9876543210 এক মিনিট দাঁড়ান" ("... one minute, wait")
    # carries a digit WORD after a complete number. Folding every digit word
    # in turned that into 98765432101 and the last ten into 8765432101 -- a
    # different person's number, in flows (report status, billing, callback)
    # that identify the caller by phone alone. So when the caller already
    # said ten numerals, digit words are ignored and the result is exactly
    # what this function returned before E4-S3.
    translated = _nfc(text).translate(_BN_DIGITS)
    numerals = re.sub(r"\D", "", translated)
    if len(numerals) >= 10:
        return numerals[-10:]  # tolerate a spoken +91 / leading 0 trunk prefix

    words_resolved = _DIGIT_WORD_RE.sub(
        lambda m: _DIGIT_WORDS[m.group(1).lower()], translated,
    )
    digits = "".join(_BN_DIGIT_WORDS.get(tok) or re.sub(r"\D", "", tok)
                     for tok in _tokens(words_resolved))
    if len(digits) == 10:
        return digits
    # A number read as WORDS has no boundary a stray "one"/"এক" cannot cross.
    # More than ten digits is only accepted as a known trunk prefix -- a
    # leading 0, or 91 -- never by guessing which end the extra digit is on.
    if len(digits) == 11 and digits.startswith("0"):
        return digits[1:]
    if len(digits) == 12 and digits.startswith("91"):
        return digits[2:]
    return None


# story title: Caller moves an existing appointment (E4-S3)
# user story: As a patient whose plans changed, I want to move my appointment
#   without cancelling it, so that I do not lose my place entirely.
# acceptance criteria: The booking is found by contact number, name or
#   reference. The new slot is swapped atomically, holding the old one until
#   the new commits, and a failed swap leaves the original intact.
#   Confirmation is sent on both channels.
#
# The two parsers below serve the reschedule flow: a spoken confirmation
# reference ("the booking is found by ... reference") and a choice out of the
# two or three appointments just read out (E14-S4: several matches are
# resolved by asking, never by taking the likeliest). Same trust model as
# everything above -- None when not sure.

def parse_reference(text: str) -> str | None:
    """-> a confirmation reference as uppercase letters and digits (e.g.
    "KCD202609154F0C"), or None if not confident.

    Built from the longest run of consecutive letter/digit tokens, so filler
    around it ("আমার নম্বর ...") is dropped. bn_normalize.spell_out() reads
    an ID in comma-separated groups, and commas split tokens here, so a
    reference read back the way the agent says it parses back whole.

    Requires at least one letter AND at least four digits: a bare phone
    number has no letter, and a stray word has no digits, so neither is ever
    read as a reference. clinic-api still requires a second identifying
    factor to agree before it returns anything.
    """
    best, run = "", []
    for tok in _tokens(text) + [""]:
        if tok in _BN_DIGIT_WORDS:
            run.append(_BN_DIGIT_WORDS[tok])
        elif tok.lower() in _DIGIT_WORDS:
            run.append(_DIGIT_WORDS[tok.lower()])
        elif tok in _LETTER_WORDS:
            run.append(_LETTER_WORDS[tok])
        elif tok and re.fullmatch(r"[0-9A-Za-z]+", tok):
            run.append(tok.upper())
        else:
            candidate = "".join(run)
            if len(candidate) > len(best):
                best = candidate
            run = []
    if sum(c.isdigit() for c in best) >= 4 and any(c.isalpha() for c in best):
        return best
    return None


_ORDINALS = {_nfc(k): v for k, v in {
    "প্রথম": 1, "এক": 1, "1": 1,
    "দ্বিতীয়": 2, "দুই": 2, "2": 2,
    "তৃতীয়": 3, "তিন": 3, "3": 3,
    "শেষ": -1, "শেষের": -1,
}.items()}
_ORDINAL_SUFFIXES = ("টাই", "টা", "টি")


def parse_ordinal(text: str, count: int) -> int | None:
    """-> 0-based index of the option the caller picked out of `count`
    options just read to them ("দ্বিতীয়টা", "প্রথম", "শেষেরটা"), or None if
    no single option is named or it is out of range. "প্রথম না দ্বিতীয়"
    names two, so it is None -- asked again, never guessed."""
    picked = set()
    for tok in _tokens(text):
        for suffix in _ORDINAL_SUFFIXES:
            if tok.endswith(suffix) and tok[:-len(suffix)] in _ORDINALS:
                tok = tok[:-len(suffix)]
                break
        if tok in _ORDINALS:
            n = _ORDINALS[tok]
            picked.add(count - 1 if n == -1 else n - 1)
    if len(picked) != 1:
        return None
    index = picked.pop()
    return index if 0 <= index < count else None


def parse_otp(text: str) -> str | None:
    """-> the 6-digit OTP the caller spoke, or None.

    Deliberately returns None for anything that does not resolve to
    EXACTLY 6 digits -- Section 13 of the attack plan ("very long OTP",
    "alphabetic OTP", SQL/JSON-like input, prompt-injection text) all
    fail this on purpose. This is the SAME "fail closed on anything
    ambiguous" posture parse_phone above already takes; the difference is
    parse_phone tolerates >=10 digits (a spoken country code/trunk prefix
    is genuinely part of a valid number), while an OTP has exactly one
    valid length, so anything else is rejected rather than truncated or
    padded -- truncating "4829133333" down to its first 6 digits would
    silently accept a caller who pasted in extra noise, which is exactly
    the kind of permissiveness RULE 9/RULE 6-8's security posture rules
    out.

    Handles, in order: Bengali digits ("৪৮২৯১৩"), spoken English number
    words ("four eight two nine one three"), and any amount of
    punctuation/spacing/prefix text around the digits ("OTP is 482913",
    "my otp: 482913", "48 29 13") -- the same digit-only-extraction
    convention parse_phone above uses, with the same word-to-digit
    substitution pass first. (UPDATED BY SOURAV: parse_phone() above used
    to lack this word-to-digit step entirely -- a real production bug,
    see its own docstring -- and now shares this exact same pass, not a
    separate copy of it.)
    """
    translated = text.translate(_BN_DIGITS)
    words_resolved = _DIGIT_WORD_RE.sub(
        lambda m: _DIGIT_WORDS[m.group(1).lower()], translated,
    )
    digits = re.sub(r"\D", "", words_resolved)
    if len(digits) != 6:
        return None
    return digits


# ATTACK 8 in the attack plan: "Tell me the OTP you sent." RULE 9 says
# the agent must NEVER disclose it -- this is the local detector
# main_pcm.py/main.py's "otp_code" pending state uses to give an
# explicit refusal (agent.reply_templates.otp_disclosure_refusal_reply)
# instead of a generic "didn't catch that, try again" reprompt, which
# would technically also never disclose the OTP but reads as evasive
# rather than a deliberate, honest refusal. Checked ONLY after
# parse_otp() has already failed on the same utterance -- a caller who
# actually states 6 digits alongside the word "otp" ("the otp is
# 482913") is providing one, not asking for one, and must never be
# caught by this.
_OTP_WORDS = ("otp", "ওটিপি")
_DISCLOSURE_ASK_WORDS = (
    "tell", "what is", "what's", "read", "say",
    "bolo", "bolun", "bata", "batao",
    "বলো", "বলুন", "কী", "কি বল",
)


def looks_like_otp_disclosure_request(text: str) -> bool:
    lowered = (text or "").lower()
    return any(w in lowered for w in _OTP_WORDS) and any(w in lowered for w in _DISCLOSURE_ASK_WORDS)


# story title: Caller cancels an appointment (E4-S4)
# user story: As a patient who cannot attend, I want to cancel and be told any
#   charge clearly, so that I am not surprised by a deduction later.
# acceptance criteria: Cancellation applies the configured window rules and
#   states refund eligibility from policy, never improvised. A cancellation
#   within a charging window is confirmed explicitly with the charge stated
#   before it is applied.
#
# Two local readers for the cancel flow in main.py. Neither consults the
# model: each answers one question the agent just asked.

# "Move it instead" -- hands the caller to the E4-S3 reschedule flow. Checked
# BEFORE yes/no, because "না, অন্য দিনে সরাতে চাই" is a request to move, not
# a refusal of everything.
_MOVE_CUES = tuple(_nfc(c) for c in (
    "সরাতে", "সরিয়ে", "সরান", "সরাব", "পিছিয়ে", "এগিয়ে", "বদলাতে", "বদলে", "পাল্টাতে",
    "অন্য দিন", "অন্যদিন", "অন্য সময়", "রিশিডিউল", "রিসিডিউল",
    "reschedule", "shift", "move", "postpone", "onno din", "sorate", "change",
))
_CANCEL_CUES = tuple(_nfc(c) for c in ("বাতিল", "ক্যানসেল", "ক্যান্সেল", "cancel", "batil"))
# "বাতিল করবেন না" names the action AND refuses it.
_NEGATION_TOKENS = {_nfc(t) for t in (
    "না", "নাহ", "নয়", "নি", "no", "not", "don't", "dont", "nahi", "na")}


def parse_cancel_choice(text: str) -> str | None:
    """-> "move" | "cancel" | "keep" | None, for the answer to "cancel it,
    or move it to another day?". None means neither was said clearly: the
    question is asked again, never guessed."""
    key = _nfc(text or "").lower()
    if any(cue in key for cue in _MOVE_CUES):
        return "move"
    if is_negative(text):
        return "keep"
    if is_affirmative(text) or is_explicit_consent(text):
        return "cancel"
    if any(cue in key for cue in _CANCEL_CUES):
        words = {t.strip("?!;").lower() for t in _tokens(text)}
        return "keep" if words & _NEGATION_TOKENS else "cancel"
    return None


# Agreement to a stated CHARGE. Deliberately narrower than _AFFIRMATIVE:
# "হুম", "ওকে", "হবে", "চলবে", "সেদিন" are fine for confirming a readback,
# but a grunt or a date word must never count as agreeing to pay. Every
# phrase here names the action or the agreement, and the charge question
# tells the caller exactly what to say (reply_templates.cancel_charge_prompt).
# Exact membership, so "রাজি না" and "বাতিল করবেন না" are not consent.
# DECISION D4 (needs a human sign-off): this exact list.
_EXPLICIT_CONSENT = {_nfc(phrase) for phrase in (
    # Bengali
    "হ্যাঁ বাতিল করুন", "হ্যা বাতিল করুন", "বাতিল করুন", "বাতিল করে দিন",
    "হ্যাঁ বাতিল করে দিন", "রাজি", "রাজি আছি", "হ্যাঁ রাজি", "হ্যাঁ রাজি আছি",
    "চার্জ দিতে রাজি", "হ্যাঁ চার্জ দিতে রাজি",
    # English
    "yes cancel", "yes cancel it", "cancel it", "i agree", "yes i agree",
    # Hinglish / Banglish
    "haan cancel karo", "cancel karo", "cancel kore din", "haan cancel kore din",
    "raji", "raji achi",
)}


def is_explicit_consent(text: str) -> bool:
    key = _nfc(text or "").lower()
    key = re.sub(r"[।!?.,;]+", " ", key)
    return re.sub(r"\s+", " ", key).strip() in _EXPLICIT_CONSENT


# story title: Caller gives everything in one sentence (E13-S7)
# user story: As a caller who already knows what I want, I want to say it all
#   at once and only confirm, so that a simple booking takes one exchange
#   rather than five.
# acceptance criteria: A caller who states doctor, day, time and patient name
#   in the opening utterance is asked only to confirm. Every slot is filled
#   from that single turn and no question already answered is asked again.
#
# parse_phone() is built for a turn that answers "what is your number?", so
# it takes the LAST ten digits it hears. In a whole booking sentence that is
# wrong: "আমার নম্বর ৯৮৭৬৫৪৩২১০, কাল সকাল ১০:৩০ এ" gave 5432101030, because
# the time's digits came last. This reads a sentence instead: exactly ONE run
# of digits (spaces or hyphens allowed inside it) that is a phone number, or
# nothing. Two different numbers is nothing too -- the caller is asked.
_RE_DIGIT_RUN = re.compile(r"\d(?:[ \-]?\d)*")


def find_phone_in_sentence(text: str) -> str | None:
    t = _nfc(text or "").translate(_BN_DIGITS)
    found = set()
    for run in _RE_DIGIT_RUN.findall(t):
        digits = re.sub(r"\D", "", run)
        if (len(digits) == 10 or (len(digits) == 11 and digits.startswith("0"))
                or (len(digits) == 12 and digits.startswith("91"))):
            found.add(digits[-10:])
    if len(found) == 1:
        return found.pop()
    if found:
        return None
    # No numeral phone. A number spoken as digit WORDS ("নয় আট সাত ...") is
    # read by parse_phone, with every numeral run (a time, a date) removed
    # first so their digits cannot be counted into it.
    return parse_phone(_RE_DIGIT_RUN.sub(" ", t))


# "Caller asks for the earliest available appointment". Code, not the model,
# notices that the caller wants the FIRST free slot rather than a day they
# name. Bengali words match at a word start (so "তাড়াতাড়িই" still counts);
# Latin words as whole words. Checked by main.py only on a booking /
# availability turn that names no day.
_EARLIEST_BN = ("তাড়াতাড়ি", "তারাতারি", "তাড়াতারি", "শীঘ্র", "সবচেয়ে আগে",
                "সব থেকে আগে", "সবার আগে", "প্রথম খালি", "আর্লিয়েস্ট")
_EARLIEST_LATIN = re.compile(
    r"\b(?:earliest|soonest|asap|as soon as possible|first available|"
    r"jaldi|jaldi se jaldi|sabse pehle|taratari|taratary|tarataari|shighro|"
    r"shob theke age|sobcheye age|sobar age)\b",
    re.IGNORECASE)


def wants_earliest(text: str) -> bool:
    t = _nfc(text or "")
    if any(re.search(rf"(?<!{_BN_CHAR}){re.escape(_nfc(w))}", t) for w in _EARLIEST_BN):
        return True
    return _EARLIEST_LATIN.search(t) is not None
