"""Phonetic folding for name matching across Bengali, Devanagari and Latin
script (KCD-436).

Deliberately duplicated from agent/phonetic_match.py rather than
imported: clinic-api simulates a separate Spring Boot service this repo
does not own (see clinic-api/main.py's module docstring), so it never
imports from agent/ the way agent/tools_client.py never imports clinic-
api's Python directly -- the two only ever talk over HTTP. Keep the two
copies in sync if the folding rules change on one side.

The existing character-level fuzzy match (difflib.SequenceMatcher, in
clinic-api/main.py's _find_doctor) catches typo-distance errors but not
genuine mispronunciation: "Bhattacharya" heard and transcribed as
"Bhattacharyya" is a large character edit distance despite being the same
name said/spelled imperfectly. A pure phonetic fold, on its own, is
exactly the "Doctor Nobody" bug (CLAUDE.md section 1): folding throws away
enough information that unrelated short names can collide. So this module
is used as an ADDITIONAL gate, never a replacement -- a candidate must
clear BOTH the phonetic fold and the existing character-similarity floor
before it counts as a match. See clinic-api/main.py's _find_doctor.

Deliberately simple: consonant-class folding (classic Soundex idea) for
Latin script, and a matching consonant-class fold for the Bengali and
Devanagari consonant inventories, so a name folded in one script can be
compared against a name folded in another once both are transliterated
to the same class alphabet.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------- Latin

# Classic Soundex-style consonant classes: sounds a mishearing commonly
# confuses land in the same class (b/p/f/v are all made with the lips;
# c/g/j/k/q/s/x/z share a sibilant/velar confusion zone in ASR output).
_LATIN_CLASS = {
    **dict.fromkeys("bpfv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    "l": "4",
    **dict.fromkeys("mn", "5"),
    "r": "6",
}


def _collapse(classes: list[str]) -> str:
    """Adjacent-duplicate collapse, classic Soundex style ("Bhattacharya"'s
    doubled T collapses to one code) -- but unlike Soundex, EVERY
    consonant is coded, none kept as a literal letter. Keeping the first
    letter literal is what breaks cross-script comparison: "s" (Latin)
    and "স" (Bengali) are the same sound but different characters, so a
    fold that treats them differently can never let one match the other.
    """
    out = []
    prev = None
    for cls in classes:
        if cls != prev:
            out.append(cls)
        prev = cls
    return "".join(out)[:6]


def _fold_latin(name: str) -> str:
    letters = [c for c in name.lower() if c.isalpha()]
    classes = [_LATIN_CLASS[c] for c in letters if c in _LATIN_CLASS]
    return _collapse(classes)


# ------------------------------------------------------------- Bengali

# Bengali consonants grouped by the confusions actually seen in this
# project's ASR output (see agent/asr.py's decoder-agreement work): the
# sibilant trio শ/ষ/স, the two retroflex-vs-dental pairs, and the
# aspirated/unaspirated pairs that a non-native or hurried speaker often
# collapses. Matras (vowel signs) and the virama are stripped entirely --
# folding is consonant-skeleton only, same principle as Latin Soundex.
_BENGALI_CLASS = {
    "ব": "1",
    "ভ": "1",
    "প": "1",
    "ফ": "1",
    "ক": "2",
    "খ": "2",
    "গ": "2",
    "ঘ": "2",
    "চ": "2",
    "ছ": "2",
    "জ": "2",
    "ঝ": "2",
    "শ": "2",
    "ষ": "2",
    "স": "2",
    "ড": "3",
    "ঢ": "3",
    "দ": "3",
    "ধ": "3",
    "ট": "3",
    "ঠ": "3",
    "ত": "3",
    "থ": "3",
    "ল": "4",
    "ম": "5",
    "ন": "5",
    "ণ": "5",
    "র": "6",
}

_BENGALI_VOWEL_SIGNS = re.compile(r"[া-ৌৗঁ-ঃ্]")  # matras, chandrabindu, anusvara, visarga, virama


# CodeRabbit-flagged, real bug (same as agent/phonetic_match.py -- kept
# in sync, see this file's own module docstring): the flap consonants
# are in Unicode's NFC composition EXCLUSION list, so NFC always leaves
# them as base consonant + nukta, never the precomposed codepoint the
# old _BENGALI_CLASS entry was keyed on. See agent/phonetic_match.py's
# _BENGALI_FLAPS comment for the full empirical confirmation.
_BENGALI_FLAPS = {"ড়": "র", "ঢ়": "র"}  # (ড+nukta)->র, (ঢ+nukta)->র


def _fold_bengali(name: str) -> str:
    text = unicodedata.normalize("NFC", name)
    for decomposed, base in _BENGALI_FLAPS.items():
        text = text.replace(decomposed, base)
    stripped = _BENGALI_VOWEL_SIGNS.sub("", text)
    classes = [_BENGALI_CLASS[c] for c in stripped if c in _BENGALI_CLASS]
    return _collapse(classes)


# ----------------------------------------------------------- Devanagari

_DEVANAGARI_CLASS = {
    "ब": "1",
    "भ": "1",
    "प": "1",
    "फ": "1",
    "व": "1",
    "क": "2",
    "ख": "2",
    "ग": "2",
    "घ": "2",
    "च": "2",
    "छ": "2",
    "ज": "2",
    "झ": "2",
    "श": "2",
    "ष": "2",
    "स": "2",
    "ड": "3",
    "ढ": "3",
    "द": "3",
    "ध": "3",
    "ट": "3",
    "ठ": "3",
    "त": "3",
    "थ": "3",
    "ल": "4",
    "म": "5",
    "न": "5",
    "ण": "5",
    "र": "6",
}

_DEVANAGARI_VOWEL_SIGNS = re.compile(r"[ा-ौ॑-ॗऀ-ः़्]")  # matras, accents, nasals, virama, nukta


# Same NFC composition-exclusion issue as _BENGALI_FLAPS above, same fix.
_DEVANAGARI_FLAPS = {"ड़": "र", "ढ़": "र"}  # (ड+nukta)->र, (ढ+nukta)->र


def _fold_devanagari(name: str) -> str:
    text = unicodedata.normalize("NFC", name)
    for decomposed, base in _DEVANAGARI_FLAPS.items():
        text = text.replace(decomposed, base)
    stripped = _DEVANAGARI_VOWEL_SIGNS.sub("", text)
    classes = [_DEVANAGARI_CLASS[c] for c in stripped if c in _DEVANAGARI_CLASS]
    return _collapse(classes)


_DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")
_BENGALI_RANGE = re.compile(r"[ঀ-৿]")


def phonetic_key(name: str) -> str:
    """Script-detecting dispatcher: folds whichever script the name is
    mostly written in. Mixed-script input folds each run separately and
    concatenates, so a name is still comparable even if ASR output
    switched scripts mid-word (rare, but seen on short garbled utterances).
    """
    if _DEVANAGARI_RANGE.search(name):
        return _fold_devanagari(name)
    if _BENGALI_RANGE.search(name):
        return _fold_bengali(name)
    return _fold_latin(name)


def phonetic_match(a: str, b: str) -> bool:
    """True if two names -- possibly in different scripts -- fold to the
    same consonant-class skeleton. Never used alone for a real match (see
    module docstring); always combined with the existing character-level
    similarity floor."""
    ka, kb = phonetic_key(a), phonetic_key(b)
    return bool(ka) and ka == kb
