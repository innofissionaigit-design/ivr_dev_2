"""Phone numbers, OTPs and the clinic's booking reference, read back digit by digit and letter by letter.

Distinct from agent/security_input.py (the four identity questions) -- this is booking/contact data, not identity
verification, and the parsing rule is different in a way that matters: a date-of-birth or an age composes number WORDS
arithmetically ("eighty five" -> 85), but a phone number, an OTP and a confirmation code do not -- each word is one
digit, said in sequence ("nine eight five" is the three digits "985", never the number 1080). Composing them the DOB
way would silently turn a correctly heard phone number into a wrong one.

    parse_phone(text)            -> "9830112233" | None   10 digits, optionally with a +91/91/0 prefix stripped
    parse_otp(text, length=6)    -> "482913" | None        a run of exactly `length` digits
    parse_confirmation_id(text)  -> "KCD-20260928-1F..." | None   letters spelled singly, digits spoken or written

It parses; it never guesses. The clinic API matches a confirmation id EXACTLY (no fuzzy lookup), so a near-miss here
must return None, not a best guess that fails silently at the server instead of here where the agent can ask again.
"""

from __future__ import annotations

import re
import unicodedata

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_HI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# One spoken digit, 0-9, in three scripts -- NOT the fuller number-word table security_input.py uses for dates and
# ages, which also composes "eighty five" into 85. Here every word is exactly one digit.
_DIGIT_WORD = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9",
    "শূন্য": "0", "এক": "1", "দুই": "2", "তিন": "3", "চার": "4", "পাঁচ": "5", "ছয়": "6", "সাত": "7", "আট": "8", "নয়": "9",
    "शून्य": "0", "ज़ीरो": "0", "जीरो": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पाँच": "5", "पांच": "5",
    "छह": "6", "छः": "6", "सात": "7", "आठ": "8", "नौ": "9",
}  # fmt: skip

# The letters used in a confirmation id ("K", "C", "D", and the hex digits A-F), spelled singly, in three scripts --
# a single bare Latin letter ("a", "k") is handled by the generic fallback below; this covers the fuller spoken form
# of a letter's name ("kay", "cee") and the nearest Bengali/Hindi syllable an ASR gives for it, so "কে সি ডি" (as
# heard) still normalises to "KCD". Deliberately NOT a single bare Latin letter or a common short word (no "a"/"e"):
# this table is only ever applied where the agent has specifically asked for the code read back (see the module
# docstring), but a collision with an everyday word is still avoided wherever the fuller spoken form allows it.
_LETTER_WORD = {
    "এ": "A", "ए": "A",
    "bee": "B", "বি": "B", "बी": "B",
    "cee": "C", "sea": "C", "সি": "C", "सी": "C",
    "dee": "D", "ডি": "D", "डी": "D",
    "ef": "F", "এফ": "F", "एफ": "F",
    "kay": "K", "কে": "K", "के": "K",
}  # fmt: skip

_CONFIRMATION_SHAPE = re.compile(r"KCD(\d{8})([0-9A-F]{8})")
_HEX_CHARS = frozenset("0123456789abcdef")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").translate(_BN_DIGITS).translate(_HI_DIGITS).lower()


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s,\-]+", _norm(text)) if t]


def _digit_run(text: str) -> str:
    """Every digit in `text`, in the order said: a numeral token kept whole, a single-digit word converted. Anything
    else (a filler word, "my number is", a bigger number word like "twenty") ends that run without being part of it --
    it is skipped, never merged into the digits around it."""
    out = []
    for tok in _tokens(text):
        tok = tok.lstrip("+")
        if tok.isdigit():
            out.append(tok)
        elif tok in _DIGIT_WORD:
            out.append(_DIGIT_WORD[tok])
    return "".join(out)


def parse_phone(text: str) -> str | None:
    """The 10-digit Indian mobile number in `text`, or None. A leading +91 / 91 / 0 is dropped when what is left is
    exactly 10 digits; the result must start 6-9 (India's mobile ranges since number portability), so an 11-digit
    landline number or a mis-heard run of digits does not pass as a mobile number."""
    digits = _digit_run(text)
    for prefix in ("0091", "091", "91", "0"):
        if digits.startswith(prefix) and len(digits) - len(prefix) == 10:
            digits = digits[len(prefix) :]
            break
    return digits if len(digits) == 10 and digits[0] in "6789" else None


_OTP_LENGTHS = (4, 6)  # the two lengths clinics commonly use; pass `length` when the system's own OTP length is known


def parse_otp(text: str, length: int | None = None) -> str | None:
    """A one-time code read back digit by digit. With `length` given, exactly that many digits are required; otherwise
    a run of 4 or 6 digits (both in use across OTP systems) is accepted. Never composed arithmetically -- see the
    module docstring."""
    digits = _digit_run(text)
    if length is not None:
        return digits if len(digits) == length else None
    return digits if len(digits) in _OTP_LENGTHS else None


def parse_confirmation_id(text: str) -> str | None:
    """The clinic's booking reference, KCD-YYYYMMDD-XXXXXXXX (clinic-api/booking_service.py:_confirmation_id), from a
    caller reading it back -- letters spelled singly, digits spoken or written, or the code already written out as-is.
    Returned in the exact stored form (hyphens, upper case) so the server's exact-match lookup can find it, or None if
    what was said does not fit the shape: a near-miss must fail here, not reach the server as a silent guess."""
    already_written = re.sub(r"[\s\-]", "", text or "").translate(_BN_DIGITS).translate(_HI_DIGITS).upper()
    m = _CONFIRMATION_SHAPE.search(already_written)
    if m:
        return f"KCD-{m.group(1)}-{m.group(2)}"
    chars = []
    for tok in _tokens(text):
        if tok.isdigit():
            chars.append(tok)
        elif tok in _DIGIT_WORD:
            chars.append(_DIGIT_WORD[tok])
        elif tok in _LETTER_WORD:
            chars.append(_LETTER_WORD[tok])
        elif tok and set(tok) <= _HEX_CHARS:
            chars.append(tok.upper())  # ASR glued several hex characters into one token, e.g. "1f3a9c0b"
        elif len(tok) == 1 and tok.isalpha():
            chars.append(tok.upper())
    m = _CONFIRMATION_SHAPE.search("".join(chars))
    return f"KCD-{m.group(1)}-{m.group(2)}" if m else None
