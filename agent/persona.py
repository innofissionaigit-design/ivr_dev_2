"""KCD-511 / KCD-512 / KCD-514 / KCD-516: the agent's documented voice, as checks.

docs/persona.md is the WRITTEN persona -- who the agent is, how it addresses a
caller, how long its sentences run, what it never says. This module is the same
document as code, so a template edit that breaks it fails the build (KCD-516)
instead of eroding the voice over months of small changes.

What is checked, and why each is a persona rule rather than a style preference:

  REGISTER (KCD-512). The respectful second person, in every language: Bengali
  "আপনি" forms (never তুমি/তুই and the tumi imperatives করো/বলো), Hindi "आप" forms
  (never तू/तुम and करो/बोलो). The bare tui imperatives (কর, বল) are NOT matched:
  each is also an ordinary word (tax, strength), and a check that fires on a
  common noun gets switched off. A caller of any age hears the polite form; a
  register slip -- especially one that appears only after the caller changes
  language mid-call -- is a defect, not a nuance.

  LENGTH. A spoken sentence is read at 130-160 words a minute to a listener who
  cannot re-read it. Caps per language (measured maxima in the current templates
  are 13 / 17 / 18 words for bn / hi / en; the caps sit just above).

  NEVER SAY. Things this agent must never put in a caller's ear, because each is
  either a guess or clinical advice, and this agent does neither:
    * hedges on facts ("probably", "I think", "সম্ভবত", "शायद") -- the agent states
      what a system returned or says it cannot check; it does not estimate;
    * reassurance about health ("don't worry", "it's nothing serious");
    * clinical direction ("you should take", "I recommend");
    * talk about itself as a model ("as an AI language model").

WHAT THIS CANNOT CHECK: whether a template SOUNDS right to a native speaker.
REVIEW_STATUS records, per language, that the persona is drafted and awaiting a
native reviewer's sign-off (the story requires one per language). Nothing here
claims otherwise; pending_review() feeds the release gate.
"""

from __future__ import annotations

import re

PERSONA_VERSION = "1.0"

REVIEW_STATUS = {
    "bn": "pending_native_review",
    "hi": "pending_native_review",
    "en": "pending_native_review",
}

# Measured maxima in the current templates: bn 13, hi 17, en 18 words. Ratchet: lower, never raise.
MAX_SENTENCE_WORDS = {"bn": 15, "hi": 18, "en": 20}

_BN = re.compile(r"[ঀ-৿]")
_HI = re.compile(r"[ऀ-ॿ]")

# --- register (informal second person and its imperatives) -----------------
_BN_INFORMAL = re.compile(r"(?<![ঀ-৿])(তুমি|তুই|তোমার|তোমাকে|তোর|তোকে|করো|বলো|দেখো|শোনো|এসো|যাও|নাও|দাও)(?![ঀ-৿])")
_HI_INFORMAL = re.compile(r"(?<![ऀ-ॿ])(तू|तुम|तुझे|तेरा|तेरी|तुझसे|तुम्हारा|तुम्हारी|तुम्हें|करो|बोलो|देखो|आओ|जाओ|सुनो)(?![ऀ-ॿ])")

# --- never say ---------------------------------------------------------------
_NEVER_SAY = {
    "en": [
        ("hedge", re.compile(r"\b(probably|maybe|perhaps|i think|i guess|i believe|might be|not sure)\b", re.I)),
        (
            "reassurance",
            re.compile(r"\b(don'?t worry|nothing serious|nothing to worry|everything will be (fine|ok|okay))\b", re.I),
        ),
        (
            "clinical_direction",
            re.compile(r"\b(you should (take|avoid|stop)|i recommend|i advise|you (must|need to) take)\b", re.I),
        ),
        (
            "self_reference_as_model",
            re.compile(r"\b(as an? (ai|language model)|i am an? (ai|language model)|large language model)\b", re.I),
        ),
    ],
    "bn": [
        ("hedge", re.compile(r"(সম্ভবত|হয়তো|মনে হয়|বোধহয়|আন্দাজ)")),
        ("reassurance", re.compile(r"(চিন্তা করবেন না|ভয়ের কিছু নেই|গুরুতর কিছু না|সব ঠিক হয়ে যাবে)")),
        ("clinical_direction", re.compile(r"(ওষুধ খাবেন|ওষুধ খান|আপনার .{0,12} খাওয়া উচিত|আমি পরামর্শ দিচ্ছি)")),
        ("self_reference_as_model", re.compile(r"(ল্যাঙ্গুয়েজ মডেল|আমি একটা এআই)")),
    ],
    "hi": [
        ("hedge", re.compile(r"(शायद|मुझे लगता है|संभवतः|अंदाज़ा|अंदाजा)")),
        ("reassurance", re.compile(r"(चिंता मत|चिंता न करें|घबराइए मत|कुछ गंभीर नहीं|सब ठीक हो जाएगा)")),
        ("clinical_direction", re.compile(r"(दवा लीजिए|दवा लें|आपको .{0,12} लेना चाहिए|मैं सलाह देती हूँ)")),
        ("self_reference_as_model", re.compile(r"(लैंग्वेज मॉडल|मैं एक एआई)")),
    ],
}


def detect_language(text: str) -> str:
    if _BN.search(text):
        return "bn"
    if _HI.search(text):
        return "hi"
    return "en"


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[।.?!])\s+", text.strip()) if s.strip()]


def words_in(sentence: str) -> int:
    # a {placeholder} is one spoken item, and quoted/hyphenated compounds count once
    return len(re.sub(r"\{[^}]*\}", "x", sentence).split())


def violations(text: str, lang: str | None = None) -> list[str]:
    """Every persona rule this text breaks, as short labels. Empty means clean."""
    lang = lang or detect_language(text)
    found: list[str] = []
    informal = _BN_INFORMAL if lang == "bn" else _HI_INFORMAL if lang == "hi" else None
    if informal is not None and informal.search(text):
        found.append("informal_register")
    for label, pattern in _NEVER_SAY.get(lang, []):
        if pattern.search(text):
            found.append(label)
    cap = MAX_SENTENCE_WORDS.get(lang)
    if cap and any(words_in(s) > cap for s in sentences(text)):
        found.append("sentence_too_long")
    return found


def is_clean(text: str, lang: str | None = None) -> bool:
    return not violations(text, lang)


def pending_review() -> list[str]:
    """Languages whose persona has NOT been signed off by a native reviewer."""
    return sorted(lang for lang, status in REVIEW_STATUS.items() if status != "approved")
