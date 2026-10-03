"""KCD-159: mixed-script synthesis with an explicit pronunciation path.

The Indic FastPitch tokenizers DROP any character outside their own
script (agent/bn_normalize.py documents the measurement): a Latin word
inside a Bengali or Hindi reply is not mispronounced, it is silently
absent, and the caller hears a sentence with a hole in it. Until now the
only remedy was a seven-entry table, and everything else was blocked by
agent/tts.py (KCD-455) -- correct, but it means "Your report is pending"
type sentences can never be spoken at all.

This module is the explicit path: an English word (or acronym) inside a
Bengali/Hindi reply is rendered through a curated LEXICON of how a
speaker of that language actually says it -- "সুগার", not the Latin
letters. It is deliberately NOT a general transliterator. A guessed
pronunciation of a medical term is worse than a blocked reply: a
mispronounced drug or test name is a wrong fact in the caller's ear, the
same class of failure CLAUDE.md's truth boundary exists to prevent. So:

  1. a word in the lexicon is spoken the way the lexicon says;
  2. an ALL-CAPS acronym (CBC, MRI, PSA) not in the lexicon is spelled
     letter by letter -- how a person reads an unknown acronym, and safe;
  3. anything else stays in Latin script, is reported by
     unspeakable_spans(), and blocks the reply (KCD-455) exactly as before.

SIGN-OFF IS NOT DONE, AND THIS MODULE SAYS SO
---------------------------------------------
The story requires the two hundred most frequent English clinical terms
to be signed off by a native listener per language. That listening review
cannot be performed by this codebase. What is here: LEXICON_TARGET = 200,
the entries below (every one marked unreviewed), coverage() to report the
gap honestly, and review_sheet() to export exactly what a native listener
needs to tick off. The spellings below are a good-faith first pass by a
non-native speaker: treat every one as a draft.
"""

from __future__ import annotations

import json
import os
import re

LEXICON_TARGET = 200

# term -> (Bengali, Hindi). Terms are matched whole-word, case-insensitively;
# a multi-word term ("ct scan") wins over its parts.
_ENTRIES: dict[str, tuple[str, str]] = {
    # specimens (the original seven-entry table; these are TRANSLATIONS, kept
    # exactly as they were -- the lab's own spoken form, not a transliteration)
    "blood": ("রক্ত", "खून"),
    "urine": ("মূত্র", "पेशाब"),
    "stool": ("মল", "मल"),
    "serum": ("সিরাম", "सीरम"),
    "saliva": ("লালা", "लार"),
    "swab": ("সোয়াব", "स्वैब"),
    "plasma": ("প্লাজমা", "प्लाज़्मा"),
    # everyday clinic words
    "report": ("রিপোর্ট", "रिपोर्ट"),
    "test": ("টেস্ট", "टेस्ट"),
    "sample": ("স্যাম্পল", "सैंपल"),
    "fasting": ("ফাস্টিং", "फास्टिंग"),
    "appointment": ("অ্যাপয়েন্টমেন্ট", "अपॉइंटमेंट"),
    "doctor": ("ডাক্তার", "डॉक्टर"),
    "profile": ("প্রোফাইল", "प्रोफाइल"),
    "package": ("প্যাকেজ", "पैकेज"),
    "counter": ("কাউন্টার", "काउंटर"),
    "online": ("অনলাইন", "ऑनलाइन"),
    "whatsapp": ("হোয়াটসঅ্যাপ", "व्हाट्सऐप"),
    "email": ("ইমেল", "ईमेल"),
    "sms": ("এসএমএস", "एसएमएस"),
    "otp": ("ওটিপি", "ओटीपी"),
    "upi": ("ইউপিআই", "यूपीआई"),
    "card": ("কার্ড", "कार्ड"),
    "cashless": ("ক্যাশলেস", "कैशलेस"),
    "insurance": ("ইনস্যুরেন্স", "इंश्योरेंस"),
    "parking": ("পার্কিং", "पार्किंग"),
    # tests and analytes
    "sugar": ("সুগার", "शुगर"),
    "cholesterol": ("কোলেস্টেরল", "कोलेस्ट्रॉल"),
    "thyroid": ("থাইরয়েড", "थायरॉइड"),
    "vitamin": ("ভিটামিন", "विटामिन"),
    "hemoglobin": ("হিমোগ্লোবিন", "हीमोग्लोबिन"),
    "haemoglobin": ("হিমোগ্লোবিন", "हीमोग्लोबिन"),
    "platelet": ("প্লেটলেট", "प्लेटलेट"),
    "creatinine": ("ক্রিয়েটিনিন", "क्रिएटिनिन"),
    "urea": ("ইউরিয়া", "यूरिया"),
    "lipid": ("লিপিড", "लिपिड"),
    "liver": ("লিভার", "लिवर"),
    "kidney": ("কিডনি", "किडनी"),
    "function": ("ফাংশন", "फंक्शन"),
    "uric": ("ইউরিক", "यूरिक"),
    "acid": ("অ্যাসিড", "एसिड"),
    "calcium": ("ক্যালসিয়াম", "कैल्शियम"),
    "electrolytes": ("ইলেকট্রোলাইট", "इलेक्ट्रोलाइट"),
    "hormone": ("হরমোন", "हार्मोन"),
    "insulin": ("ইনসুলিন", "इंसुलिन"),
    "protein": ("প্রোটিন", "प्रोटीन"),
    "glucose": ("গ্লুকোজ", "ग्लूकोज़"),
    "hba1c": ("এইচবিএ ওয়ান সি", "एचबीए वन सी"),
    "culture": ("কালচার", "कल्चर"),
    "biopsy": ("বায়োপসি", "बायोप्सी"),
    # imaging and procedures
    "x-ray": ("এক্স-রে", "एक्स-रे"),
    "xray": ("এক্স-রে", "एक्स-रे"),
    "ultrasound": ("আল্ট্রাসাউন্ড", "अल्ट्रासाउंड"),
    "ct scan": ("সিটি স্ক্যান", "सीटी स्कैन"),
    "mri": ("এমআরআই", "एमआरआई"),
    "ecg": ("ইসিজি", "ईसीजी"),
    "echo": ("ইকো", "इको"),
    "scan": ("স্ক্যান", "स्कैन"),
    "endoscopy": ("এন্ডোস্কোপি", "एंडोस्कोपी"),
    # conditions frequently named in lab enquiries
    "dengue": ("ডেঙ্গু", "डेंगू"),
    "malaria": ("ম্যালেরিয়া", "मलेरिया"),
    "covid": ("কোভিড", "कोविड"),
    "typhoid": ("টাইফয়েড", "टाइफाइड"),
    "diabetes": ("ডায়াবেটিস", "डायबिटीज़"),
    "anaemia": ("অ্যানিমিয়া", "एनीमिया"),
    # departments
    "pathology": ("প্যাথলজি", "पैथोलॉजी"),
    "radiology": ("রেডিওলজি", "रेडियोलॉजी"),
    "cardiology": ("কার্ডিওলজি", "कार्डियोलॉजी"),
    "dermatology": ("ডার্মাটোলজি", "डर्मेटोलॉजी"),
    "paediatrics": ("পেডিয়াট্রিক্স", "पीडियाट्रिक्स"),
    "ortho": ("অর্থো", "ऑर्थो"),
    "ent": ("ইএনটি", "ईएनटी"),
    # A live call (2026-09-28): the blood-test-list reply blocked outright --
    # these mixed-case lab abbreviations ("Rh", "HBsAg", "IgG", "IgM") defeat
    # the ALL-CAPS acronym spell-out below (KCD-455's other safety net), so
    # without an explicit entry here they had no spoken form at all, not even
    # a spelled-out one.
    "grouping": ("গ্রুপিং", "ग्रुपिंग"),
    "typing": ("টাইপিং", "टाइपिंग"),
    "rh": ("আর এইচ", "आर एच"),
    "widal": ("উইডাল", "विडाल"),
    "antigen": ("অ্যান্টিজেন", "एंटीजन"),
    "ns1": ("এন এস ওয়ান", "एन एस वन"),
    "hbsag": ("এইচ বি এস এ জি", "एच बी एस ए जी"),
    "igg": ("আই জি জি", "आई जी जी"),
    "igm": ("আই জি এম", "आई जी एम"),
}

REVIEWED: set[tuple[str, str]] = set()  # (term, lang) pairs a native listener has approved

# The seven specimen words are TRANSLATIONS the lab already used before this
# module existed (agent/bn_normalize.py's original table), not first-pass
# transliterations, so they are spoken without waiting for sign-off.
ESTABLISHED = frozenset({"blood", "urine", "stool", "serum", "saliva", "swab", "plasma"})

# A draft (unreviewed) form is NOT spoken by default: a wrongly rendered
# clinical term is a wrong fact in the caller's ear. Setting this to 1 lets
# the drafts through so a native listener can HEAR them on a pod -- which is
# how they get reviewed at all. It is a listening aid, never a release setting.
ALLOW_DRAFT_ENV = "PRONUNCIATION_ALLOW_DRAFT"


def _allow_draft() -> bool:
    return os.environ.get(ALLOW_DRAFT_ENV, "0") == "1"


def load_reviewed(path: str) -> int:
    """Approve the (term, lang) pairs listed in a JSON file
    [["ct scan", "bn"], ...] -- what a native reviewer hands back from
    review_sheet(). Returns how many pairs are now approved."""
    with open(path, encoding="utf-8") as f:
        for term, lang in json.load(f):
            REVIEWED.add((str(term).lower(), lang))
    return len(REVIEWED)


def speakable(term: str, lang: str) -> bool:
    key = term.lower()
    return key in ESTABLISHED or (key, lang) in REVIEWED or _allow_draft()


_INDEX = {"bn": 0, "hi": 1}


def _compile():
    terms = sorted(_ENTRIES, key=len, reverse=True)  # longest first: "ct scan" before "scan"
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(t) for t in terms) + r")(?![A-Za-z0-9])", re.IGNORECASE)


_RE_TERMS = _compile()
_RE_ACRONYM = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,6})(?![A-Za-z0-9])")


def lookup(term: str, lang: str) -> str | None:
    entry = _ENTRIES.get(term.lower())
    return entry[_INDEX[lang]] if entry and lang in _INDEX else None


def apply(text: str, lang: str, letters: dict[str, str] | None = None) -> str:
    """Rewrite the Latin words in `text` that have an explicit spoken form
    in `lang` (bn or hi). `letters` is that language's letter-name table
    (agent.bn_normalize._LETTER_BN / agent.speech_norm._HI_LETTER) used to
    spell an unlisted acronym; passed in rather than imported so this module
    stays importable from both without a cycle. Other Latin words are left
    alone for unspeakable_spans() to report."""
    if lang not in _INDEX or not text:
        return text
    text = _RE_TERMS.sub(
        lambda m: (lookup(m.group(1), lang) if speakable(m.group(1), lang) else None) or m.group(1), text
    )
    if letters:
        text = _RE_ACRONYM.sub(lambda m: " ".join(letters[c.lower()] for c in m.group(1) if c.lower() in letters), text)
    return text


def coverage() -> dict:
    """How far this is from the story's acceptance criterion -- reported,
    never rounded up."""
    reviewed = sum(1 for term in _ENTRIES for lang in _INDEX if (term, lang) in REVIEWED)
    return {
        "target_terms": LEXICON_TARGET,
        "terms": len(_ENTRIES),
        "shortfall": max(LEXICON_TARGET - len(_ENTRIES), 0),
        "entries_needing_native_review": len(_ENTRIES) * len(_INDEX) - reviewed,
        "signed_off": reviewed,
    }


def review_sheet() -> list[dict]:
    """One row per (term, language) for a native listener to approve or
    correct -- the exact hand-off the story's sign-off requires."""
    return [
        {
            "term": term,
            "lang": lang,
            "spoken_form": forms[idx],
            "status": "approved" if (term, lang) in REVIEWED else "pending_native_review",
        }
        for term, forms in sorted(_ENTRIES.items())
        for lang, idx in _INDEX.items()
    ]
