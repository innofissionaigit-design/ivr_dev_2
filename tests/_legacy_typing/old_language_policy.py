"""KCD-087 / KCD-150: answer in the language asked, preserving borrowings.

Blueprint 4.4: the reply language comes from the identification result for
the CURRENT utterance, not a call-level setting. main.py already assigns
session.lang from the per-utterance router each turn; what was missing was
(a) one place that states the rule, (b) a way to catch a reply in the
wrong language as a DEFECT rather than a surprise on a live call ("a
mismatch between question and answer language is a defect caught by the
multilingual suite"), and (c) the register half of KCD-150: a term the
caller said in the everyday (often English-derived) form is spoken back in
that form, not swapped for a formal literary synonym.

No model is involved. Script detection is a character-class count.
"""

from __future__ import annotations

import re
import unicodedata

_BENGALI = re.compile(r"[ঀ-৿]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

SCRIPT_FOR_LANGUAGE = {"bn": "bengali", "hi": "devanagari", "en": "latin"}

# A reply in language L may carry this fraction of words written ONLY in
# another script -- a borrowed English word spoken through the pronunciation
# path, a Latin unit. Above it the reply is in the wrong language.
# REASONED, not measured: chosen so one borrowed word in a short sentence
# ("Your report is ready" style, 1 of 3 words) is tolerated and two of three is
# not. Not calibrated against real replies.
_FOREIGN_TOLERANCE = 0.34


def resolve_reply_language(identified: str | None, explicit_request: str | None = None, fallback: str = "bn") -> str:
    """The language to answer in. An explicit request ("speak Hindi",
    agent/language_switch.py) outranks identification for the turn it is
    made in; otherwise it is whatever THIS utterance was identified as;
    only an unidentifiable utterance falls back to the last committed
    language. Never a call-level constant."""
    for candidate in (explicit_request, identified, fallback):
        if candidate in SCRIPT_FOR_LANGUAGE:
            return candidate
    return "bn"


def script_counts(text: str) -> dict[str, int]:
    text = unicodedata.normalize("NFC", text or "")
    # LETTERS only: the Bengali and Devanagari blocks also hold their own
    # digits, and a numeral carries no language.
    return {
        "bengali": sum(1 for c in _BENGALI.findall(text) if c.isalpha()),
        "devanagari": sum(1 for c in _DEVANAGARI.findall(text) if c.isalpha()),
        "latin": len(_LATIN.findall(text)),
    }


def dominant_script(text: str) -> str | None:
    counts = script_counts(text)
    best = max(counts, key=counts.get)
    return best if counts[best] else None


def reply_matches_language(reply: str, lang: str) -> bool:
    """Is `reply` written in `lang`? Judged per WORD, not per letter: a
    borrowed "Report" inside a Bengali sentence is one foreign word among
    several, and a long Latin word must not outweigh the Bengali around it
    just because it has more letters. Words with no letters (numbers,
    punctuation) carry no script and are ignored; a reply with no lettered
    words at all is given the benefit of the doubt."""
    expected = SCRIPT_FOR_LANGUAGE.get(lang)
    if expected is None:
        return True
    words = [w for w in unicodedata.normalize("NFC", reply or "").split() if sum(script_counts(w).values())]
    if not words:
        return True
    foreign = sum(1 for w in words if script_counts(w)[expected] == 0)
    return foreign / len(words) <= _FOREIGN_TOLERANCE


def language_mismatch(question_lang: str, reply: str) -> bool:
    """True when the caller asked in `question_lang` and `reply` is not in
    it -- the defect the multilingual suite exists to catch."""
    return not reply_matches_language(reply, question_lang)


def choose_spoken_form(caller_term: str | None, known_forms: list[str], default: str | None) -> str | None:
    """KCD-150 (the code half): among the forms an entity is known by
    (aliases_bn/aliases_hi), speak back the one the caller USED. A caller
    who said the everyday borrowed word gets it back; one who said the
    formal word gets that. Falls back to `default` (the first, canonical
    form) when the caller's word is not one of them -- never invents a
    form. The native-reviewer confirmation over fifty sampled calls per
    language is the part this codebase cannot do."""
    if caller_term:
        heard = unicodedata.normalize("NFC", caller_term).strip().lower()
        norm = [(form, unicodedata.normalize("NFC", form).strip().lower()) for form in known_forms]
        norm = [(form, f) for form, f in norm if f]
        # An exact match wins outright; otherwise the LONGEST form the
        # caller's words contain, so "সিটি স্ক্যান" is not answered with the
        # shorter "স্ক্যান" and the caller's qualifier is not lost.
        for form, f in norm:
            if f == heard:
                return form
        contained = [(form, f) for form, f in norm if f in heard]
        if contained:
            return max(contained, key=lambda ff: len(ff[1]))[0]
        for form, f in norm:
            if heard in f:
                return form
    return default
