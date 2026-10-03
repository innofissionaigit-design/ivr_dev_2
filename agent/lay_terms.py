"""Rural lay terms for what a test is, matched deterministically to a CATEGORY -- never to a specific test name.

A caller who has never heard "CBC" or "USG" describes what the doctor wants checked, not the test's clinical name:
"rokto porikkha" (blood test), "peter chhobi" (a picture of the stomach), "mathar chhobi" (a picture of the head).
This module recognises the CATEGORY the phrase describes; agent/reply_templates.py's lay_term_reply() (or main.py,
for the blood list) turns that into an answer built from the REAL catalogue -- this file never states a test exists.

    BLOOD    -- every test the catalogue marks sample_type == "Blood"; ask which one (main.py lists them)
    ABDOMEN  -- resolves to the catalogue's Imaging test(s) whose name mentions the abdomen (today: USG Whole
                Abdomen); state it directly, there is nothing to ask
    HEAD     -- the catalogue has NO head/brain imaging test today (checked against the seeded data: only Chest
                X-Ray and USG Whole Abdomen exist under "Imaging"). Naming "brain scan" or "brain MRI" here would be
                exactly the fabricated fact CLAUDE.md's truth boundary exists to prevent -- a clinic that does not
                offer a service must never be told it does. So this category resolves to NOTHING; the caller is told
                to send the prescription so a colleague can check what the doctor actually wants (see main.py).

Phrases are matched in Bengali script (what a real caller's ASR transcript looks like), Devanagari, and the
phonetic-romanised Latin spelling the person who wrote this table typed it in -- kept because rural callers on a
poor line, or a WhatsApp/SMS channel, sometimes come through in Latin script too. REASONED, not measured: this list
has not been checked against a real call corpus; grow it from real transcripts once they exist (docs/transcription-
rulebook.md), and confirm with the clinic which imaging tests actually exist before this category list grows past
BLOOD/ABDOMEN (docs/REMINDERS-if-client-asks.md).
"""

from __future__ import annotations

import re
import unicodedata

BLOOD = "blood"
ABDOMEN = "abdomen"
HEAD = "head"

# Phrase -> category. Substring match on the normalised text (see normalise()); a phrase is chosen to be distinctive
# enough that it does not fire on an unrelated sentence ("chhobi" alone is too short -- "peter chhobi"/"mathar
# chhobi" are not).
_PHRASES: dict[str, str] = {
    # ---- blood ----
    "রক্ত পরীক্ষা": BLOOD,
    "রক্তের পরীক্ষা": BLOOD,
    "রক্ত টেস্ট": BLOOD,
    "রক্ত দেওয়া": BLOOD,
    "রক্তের টেস্ট": BLOOD,
    "खून की जांच": BLOOD,
    "खून टेस्ट": BLOOD,
    "रक्त जांच": BLOOD,
    "rokto porikkha": BLOOD,
    "rokto porikha": BLOOD,
    "rokter test": BLOOD,
    "rokto test": BLOOD,
    "khoon ki jaanch": BLOOD,
    "khun ki jaanch": BLOOD,
    # ---- abdomen (USG whole abdomen) ----
    "পেটের ছবি": ABDOMEN,
    "পেটের পরীক্ষা": ABDOMEN,
    "পেটের আলট্রাসাউন্ড": ABDOMEN,
    "পেট স্ক্যান": ABDOMEN,
    "पेट की तस्वीर": ABDOMEN,
    "पेट की जांच": ABDOMEN,
    "peter chhobi": ABDOMEN,
    "peter porikkha": ABDOMEN,
    "peter porikha": ABDOMEN,
    "pete ki jaanch": ABDOMEN,
    "pet ki jaanch": ABDOMEN,
    "pet ki tasvir": ABDOMEN,
    # ---- head ----
    "মাথার ছবি": HEAD,
    "মাথার পরীক্ষা": HEAD,
    "মাথা স্ক্যান": HEAD,
    "सिर की तस्वीर": HEAD,
    "सिर की जांच": HEAD,
    "mathar chhobi": HEAD,
    "mathar porikkha": HEAD,
    "mathar porikha": HEAD,
    "matha scan": HEAD,
    "sar ki jaanch": HEAD,
}

# Words a caller wraps a phrase in that carry no category of their own -- stripped before matching so "আমার একটা
# পেটের ছবি করাতে হবে" still contains the bare phrase "পেটের ছবি".
_STRIP = frozenset({"amar", "amake", "ekta", "korte", "korate", "hobe", "hocche", "chai", "please", "mera", "mujhe"})

_BENGALI_RANGE = re.compile(r"[ঀ-৿]")
_DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")


def normalise(text: str) -> str:
    """NFC, lower case, punctuation to spaces -- the same comparison form agent/gazetteer.py uses, kept separate
    (not imported) because this file matches whole PHRASES, not names, and never needs gazetteer's fuzzy tiers: a
    lay term is either recognisably one of the phrases above or it is not, never a sound-alike guess."""
    t = unicodedata.normalize("NFC", text or "").lower()
    kept = "".join(
        ch if (ch.isalnum() or unicodedata.category(ch) in ("Mn", "Mc") or ch.isspace()) else " " for ch in t
    )
    words = [w for w in kept.split() if w not in _STRIP]
    return " ".join(words)


def match_lay_term(text: str) -> str | None:
    """BLOOD / ABDOMEN / HEAD if `text` contains one of the recognised phrases, else None. The longest matching
    phrase wins when more than one appears (a transcript naming two categories in one sentence is rare enough that
    picking the more specific -- usually longer -- phrase is the safer default; a caller who meant both will be
    asked again on the next turn regardless, since only one category is ever acted on per turn)."""
    norm = normalise(text)
    if not norm:
        return None
    hits = [(len(phrase), cat) for phrase, cat in _PHRASES.items() if normalise(phrase) in norm]
    return max(hits)[1] if hits else None
