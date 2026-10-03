"""A caller asking what their result MEANS: never answered here, always a clinician.

Ported from `merged_code/agent/clinical_safety.py` (story: "Caller asks whether
their result is dangerous"). Its acceptance criteria were:

  * clinical interpretation is routed to a human BY POLICY, not by prompt wording;
  * the reply says plainly that a doctor will explain the result and OFFERS to
    connect one, rather than simply refusing;
  * an adversarial set of a hundred prompts per language produces zero breaches.

WHAT THE GAP ACTUALLY IS IN THIS REPOSITORY -- IT IS NOT THE ONE THE ORIGINAL
MODULE WAS WRITTEN AGAINST:

The module this was ported from argued that the danger is the model inventing
clinical advice through the `smalltalk` path, the one place model-composed text
reaches a caller. That argument does not hold here, and it is worth saying so
plainly rather than importing a rationale that no longer applies: this
repository already passes that text through `_speakable()`, `persona_clean()`
and `mentions_personal_history()` before speaking it, and agent/persona.py
already blocks reassurance ("don't worry", "nothing serious", and their Bengali
and Hindi equivalents) in all three languages. The model cannot tell a caller
their result is fine.

The real gap, measured against this code: a caller asking "is my result
dangerous?" or "am I dying?" names no test, no doctor and no date, so there is
nothing to look up. agent/emergency.py does not fire (correctly -- none of these
is an emergency). So the turn resolves to `smalltalk` or `unclear`, and the
frightened caller is answered with a generic greeting or asked to repeat
themselves. That is precisely the "procedural limit that feels like a rebuff"
the story exists to prevent, and no amount of persona filtering turns a greeting
into an offer of a clinician.

WHY IT IS STILL A PRE-CLASSIFIER GUARD:

"By policy rather than by prompt wording" is the acceptance criterion itself. A
prompt instruction is advice a 7B model can be argued out of, and this module's
own phrase lists exist because an adversarial suite proved that: a third of the
entries below were added to close roleplay frames ("pretend you are a doctor"),
hypotheticals, and forced binaries ("just say yes or no") that got past earlier
versions. Running before agent/fast_path.py, the semantic cache and Ollama makes
a misclassification structurally impossible rather than merely unlikely.

THIS MODULE MAKES NO MEDICAL JUDGEMENT. It does the opposite: it recognises the
SHAPE of a question asking for one -- a danger or mortality word, a
normal-versus-abnormal question about a report, a named lab parameter sitting
next to a raw number, or an attempt to route around a refusal -- and sends every
one of them to the same fixed, clinician-offering reply.

REASONED, not measured, for this repository: the Devanagari lists are new here
(the original covered Bengali, English and Latin-script transliteration only),
and the adversarial suite the original closed its gaps against was never run in
Hindi. A native speaker should review them, and the suite should be re-run per
language on the pod before this is called verified for Hindi.
"""

from __future__ import annotations

import re
import unicodedata

# Word boundaries that actually work for Indic scripts. Python's \b treats a
# vowel sign or the nukta as a non-word character, so a short syllable would
# otherwise match inside an unrelated longer word. Same idiom as
# agent/persona.py's `_BN_INFORMAL` / `_HI_INFORMAL`.
_BN_CHAR = r"[ঀ-৿]"
_HI_CHAR = r"[ऀ-ॿ]"


def _bounded(word: str, text: str, char_class: str) -> bool:
    """Is `word` in `text` as a whole word, for a script whose marks combine?"""
    return re.search(rf"(?<!{char_class}){re.escape(word)}(?!{char_class})", text) is not None


# ======================================================================= English
# Checked as substrings of the lower-cased text: every entry is already a
# distinctive multi-character English phrase, so no boundary check is needed.
_EN_DANGER_PANIC = (
    "dangerous", "am i dying", "going to die", "will i die", "could i die",
    "might i die", "life threatening", "life-threatening", "is it fatal",
    "is this fatal", "do i have cancer", "is it cancer", "is this cancer",
    "should i be worried", "should i worry", "am i going to be okay",
    "am i going to be alright", "am i okay", "am i alright",
    "worst case scenario", "is my life at risk", "in danger",
    "scared", "terrified", "am i in trouble", "is something wrong with me",
    # Closed by the original's adversarial suite: natural paraphrases carrying
    # the same meaning without containing any phrase above as a substring.
    "kill me", "should i panic", "wrong with me", "in trouble",
    "real trouble", "could this be cancer", "could it be cancer",
    "worst that can happen",
)
_EN_ABNORMAL_NORMAL = (
    "is this normal", "is this abnormal", "is my result normal",
    "is my results normal", "is my report normal", "is my report abnormal",
    "is my result abnormal", "what does this mean", "what does it mean",
    "what does my result mean", "what does my report mean",
    "is this a bad result", "is this a good result", "is that bad",
    "is that good", "out of range", "is this out of range",
    "too high", "too low", "is this serious", "is it serious",
    "how serious is this", "is this concerning", "is this a bad sign",
    "normal or not", "to worry about",
)
_EN_ROLEPLAY = (
    "pretend you are a doctor", "pretend to be a doctor", "act as a doctor",
    "as a doctor tell me", "imagine you are a doctor", "imagine you're a doctor",
    "hypothetically", "in theory", "just theoretically",
    "if you were a doctor", "role play", "roleplay", "just between us",
    "off the record", "as a friend, not an agent", "speaking as a human",
    "forget the rules",
)
_EN_FORCED_BINARY = (
    "just say yes or no", "yes or no", "one word answer", "one-word answer",
    "simple yes or no", "just tell me yes or no", "just answer yes or no",
    "give me a yes or no",
)

# ========================================== Hindi and Bengali in Latin script
_LATIN_DANGER_PANIC = (
    "khatarnak", "bipodjonok", "main mar jaunga", "mai mar jaunga",
    "kya main mar jaunga", "ami ki mara jabo", "ami ki more jabo",
    "mara jabo naki", "more jabo naki", "amar ki bipod",
    "ami ki bipode achi", "jibon songkoto", "jibon er jonno bipodjonok",
    "gurutor", "khub gurutor", "cancer naki", "amar ki cancer",
    "bhoy lagche", "bhoi lagche", "dar lagche", "main dar gaya hoon",
    "jaanleva", "cancer hai", "dar lag raha hai", "ghabrana", "gadbad hai",
)
_LATIN_ABNORMAL_NORMAL = (
    "eta ki normal", "ata ki normal", "eta ki abnormal", "ata ki abnormal",
    "report ta kharap naki", "result ta kharap naki",
    "ata ki bhalo naki kharap", "eta ki bhalo naki kharap",
    "ei value ta thik ache naki", "yeh normal hai kya", "yeh abnormal hai kya",
    "yeh khatarnak hai kya", "report kharab hai kya", "matlab kya hai iska",
    "eta mane ki", "ata mane ki", "iska matlab kya hai",
    "normal hai ya", "ya kharab", "bahut zyada hai", "bahut kam hai",
    "kitna serious hai", "chinta karni chahiye",
)
_LATIN_ROLEPLAY = (
    "doctor hoke bolo", "doctor hokar bolun", "farz karo aap doctor ho",
    "dhorun apni doctor", "ekta hypothetically bolun",
    "ekhon doctor er moto bolun", "aap doctor ki tarah bolo",
    "man lijiye aap doctor hain", "rules bhool jao", "dost ki tarah",
)
_LATIN_FORCED_BINARY = (
    "sirf haan ya na bolo", "sirf haan ya naa boliye", "ek shabd mein batao",
    "shudhu hae ba na bolun", "ek kothay bolun", "just haan ya na",
    "haan ya na bolo", "ek shabd",
)

_ALL_LATIN = (
    _EN_DANGER_PANIC + _EN_ABNORMAL_NORMAL + _EN_ROLEPLAY + _EN_FORCED_BINARY
    + _LATIN_DANGER_PANIC + _LATIN_ABNORMAL_NORMAL + _LATIN_ROLEPLAY + _LATIN_FORCED_BINARY
)

# ======================================================================= Bengali
# Word-bounded, never bare substrings: several entries are short enough that a
# substring check would match inside an unrelated word.
_BN_DANGER_PANIC = (
    "বিপজ্জনক", "মারা যাব", "মারা যাচ্ছি", "মরে যাব", "মরে যাচ্ছি",
    "গুরুতর", "ভয়ঙ্কর", "জীবন সংশয়", "জীবনের ঝুঁকি", "বিপদে আছি",
    "ক্যান্সার কিনা", "আমার কি ক্যান্সার", "ভয় লাগছে", "ভয়ে আছি",
    "চিন্তিত", "আতঙ্কিত",
    # An inflected form is a DIFFERENT word to the boundary check above, by
    # design (the left boundary has to stay strict). The forms callers actually
    # use are listed rather than loosening the shared rule.
    "জীবন সংশয়ের", "সমস্যা আছে",
)
_BN_ABNORMAL_NORMAL = (
    "স্বাভাবিক কিনা", "অস্বাভাবিক", "রিপোর্ট খারাপ", "রেজাল্ট খারাপ",
    "ভালো নাকি খারাপ", "মানে কী", "এর মানে কী", "সমস্যা আছে কিনা",
    "কতটা গুরুতর", "অনেক বেশি", "অনেক কম",
)
_BN_ROLEPLAY = (
    "ধরুন আপনি ডাক্তার", "ডাক্তার হিসেবে বলুন", "যদি ধরি",
    "একজন ডাক্তারের মতো বলুন", "নিয়ম ভুলে", "বন্ধুর মতো", "অফ দ্য রেকর্ড",
)
_BN_FORCED_BINARY = (
    "শুধু হ্যাঁ বা না বলুন", "এক কথায় বলুন", "শুধু হ্যাঁ নাকি না",
    "হ্যাঁ বা না বলুন",
)
_ALL_BN = _BN_DANGER_PANIC + _BN_ABNORMAL_NORMAL + _BN_ROLEPLAY + _BN_FORCED_BINARY

# ========================================================= Hindi (Devanagari)
# NEW in this repository: the module this was ported from had no Devanagari
# lists, so a Hindi caller reached none of this. Same four categories, same
# word-bounded treatment as Bengali -- "गंभीर" and "घबरा" are short enough to
# need it.
_HI_DANGER_PANIC = (
    "खतरनाक", "ख़तरनाक", "जानलेवा", "क्या मैं मर जाऊँगा", "क्या मैं मर जाऊंगी",
    "मैं मर जाऊँगा", "मर जाऊँगी", "जान को खतरा", "जान को ख़तरा",
    "जान का खतरा", "गंभीर", "बहुत गंभीर", "कैंसर है क्या", "मुझे कैंसर है",
    "क्या मुझे कैंसर", "डर लग रहा है", "डर लगता है", "घबरा", "घबराहट",
    "मैं खतरे में", "कुछ गड़बड़ है", "गड़बड़ है क्या", "मुझे कुछ हो जाएगा",
)
_HI_ABNORMAL_NORMAL = (
    "सामान्य है क्या", "असामान्य", "क्या यह सामान्य है", "रिपोर्ट खराब",
    "रिपोर्ट ख़राब", "रिजल्ट खराब", "इसका मतलब क्या", "इसका मतलब क्या है",
    "मतलब क्या है", "कितना गंभीर", "कितना सीरियस", "बहुत ज्यादा है",
    "बहुत ज़्यादा है", "बहुत कम है", "चिंता करनी चाहिए",
    "अच्छा है या खराब", "ठीक है या नहीं", "रेंज से बाहर",
)
_HI_ROLEPLAY = (
    "मान लीजिए आप डॉक्टर", "मान लो तुम डॉक्टर", "डॉक्टर की तरह बताइए",
    "डॉक्टर बनकर बताइए", "अगर आप डॉक्टर होते", "काल्पनिक रूप से",
    "नियम भूल", "दोस्त की तरह", "हमारे बीच की बात",
)
_HI_FORCED_BINARY = (
    "सिर्फ हाँ या ना", "सिर्फ़ हाँ या ना", "हाँ या ना बताइए",
    "एक शब्द में बताइए", "एक शब्द में",
)
_ALL_HI = _HI_DANGER_PANIC + _HI_ABNORMAL_NORMAL + _HI_ROLEPLAY + _HI_FORCED_BINARY

# ------------------------------------------------- gap-tolerant bypass framings
# The substring lists above cannot absorb an inserted clause ("pretend FOR A
# SECOND you're a doctor"), so these close each whole family at once.
_RE_EN_PRETEND_DOCTOR = re.compile(r"\bpretend\b[^.?!]{0,40}\bdoctor\b", re.IGNORECASE)
_RE_EN_WHAT_DOES_MEAN = re.compile(r"\bwhat does\b[^.?!]{0,30}\bmean\b", re.IGNORECASE)
_RE_LATIN_FARZ_DOCTOR = re.compile(r"farz karo\b[^.?!]{0,30}\bdoctor\b", re.IGNORECASE)
_RE_BN_DHORUN_DOCTOR = re.compile(r"ধরুন আপনি[^.?!।]{0,30}ডাক্তার")
_RE_HI_MAAN_DOCTOR = re.compile(r"मान (?:लीजिए|लो|लीजिये)[^.?!।]{0,30}डॉक्टर")

# ------------------------------------- a lab parameter next to a raw number
# A caller reading a figure off their own report and asking this system to
# interpret it, in either order, within one clause. The parameter NAME is what
# makes this unambiguous: no other intent in this system ever needs a caller to
# name a clinical measurement.
_CLINICAL_TERMS = (
    "hemoglobin", "haemoglobin", "hb", "sugar", "glucose", "creatinine",
    "bilirubin", "wbc", "rbc", "platelet", "platelets", "cholesterol",
    "hba1c", "tsh", "sgpt", "sgot", "urea", "esr", "crp", "vitamin d",
    "vitamin b12",
)
_TERMS_RX = "|".join(re.escape(t) for t in _CLINICAL_TERMS)
_RE_TERM_THEN_NUMBER = re.compile(rf"\b(?:{_TERMS_RX})\b[^.?!]{{0,25}}\d", re.IGNORECASE)
_RE_NUMBER_THEN_TERM = re.compile(rf"\d[^.?!]{{0,25}}\b(?:{_TERMS_RX})\b", re.IGNORECASE)

# A LIVE FALSE POSITIVE this rule produced, and the two defences against it.
#
# "please add ESR to confirmation KCD-1" fired the rule: "esr" is a clinical term
# and "KCD-1" supplied a digit inside the 25-character window. The caller was
# asking to add a test to an existing booking and would have been answered with
# "a doctor will explain your result" -- a working flow (add-test booking) broken
# by a guard whose whole job is to sit quietly in front of it. Caught by
# tests/test_lay_term_and_payment_flow.py, which is why that suite exists.
#
# The PHRASE LISTS above are untouched by both defences. They are what the
# adversarial suite hardened and nothing here may weaken them. Only the
# term-NEAR-number heuristic is narrowed, because only it infers an
# interpretation request from the mere presence of a figure.
#
# 1. A digit inside an alphanumeric IDENTIFIER is not a measurement. Booking
#    references here look like "KCD-1" (agent/bn_normalize.py's own `_RE_CONF_ID`
#    matches the same shape), so identifiers are removed before the check runs.
_RE_IDENTIFIER = re.compile(r"\b[A-Za-z]{2,}[-/]?\d+\b")
# 2. A caller reading a lab value off their report is not simultaneously asking
#    to add, book, cancel or move an appointment. An explicit booking cue means
#    the figure in the sentence belongs to a reference, a date or a time.
_BOOKING_CONTEXT = (
    "confirmation",
    "booking",
    "appointment",
    "reference",
    "add ",
    "cancel",
    "reschedule",
    "resend",
    "book ",
    "কনফার্মেশন",
    "বুকিং",
    "অ্যাপয়েন্টমেন্ট",
    "রেফারেন্স",
    "বাতিল",
    "যোগ কর",
    "कन्फ़र्मेशन",
    "कन्फर्मेशन",
    "बुकिंग",
    "अपॉइंटमेंट",
    "रेफरेंस",
    "रद्द",
    "जोड़ ",
)


def _reading_a_lab_value(lowered: str, raw: str) -> bool:
    """The term-near-number heuristic, with identifiers and booking turns excluded."""
    if any(cue in lowered for cue in _BOOKING_CONTEXT) or any(cue in raw for cue in _BOOKING_CONTEXT):
        return False
    stripped = _RE_IDENTIFIER.sub(" ", lowered)
    return bool(_RE_TERM_THEN_NUMBER.search(stripped) or _RE_NUMBER_THEN_TERM.search(stripped))


_BN_CLINICAL_TERMS = ("হিমোগ্লোবিন", "সুগার", "ক্রিয়েটিনিন", "কোলেস্টেরল", "বিলিরুবিন")
_HI_CLINICAL_TERMS = ("हीमोग्लोबिन", "शुगर", "क्रिएटिनिन", "कोलेस्ट्रॉल", "बिलीरुबिन", "शर्करा")
# Both Indic and Latin digits turn up in real transcripts here.
_RE_ANY_DIGIT = re.compile(r"[0-9০-৯०-९]")


def _indic_value_mention(text: str, terms: tuple[str, ...], char_class: str) -> bool:
    """An Indic-script clinical term with a number within ~25 characters.

    Checked separately from the Latin regexes because an Indic term needs the
    word-boundary-safe check, not a bare substring.
    """
    for term in terms:
        if not _bounded(term, text, char_class):
            continue
        idx = text.find(term)
        window = text[max(0, idx - 25) : idx + len(term) + 25]
        if _RE_ANY_DIGIT.search(window):
            return True
    return False


def is_clinical_interpretation(text: str) -> bool:
    """True when `text` asks this system to render a clinical judgement --
    whether a result is dangerous, normal, or means something -- however it is
    dressed up: a direct question, a panic, a roleplay frame, a hypothetical, or
    a forced binary.

    Never a determination of what the ANSWER is; only that the question is this
    kind of question, which is all the orchestrator needs in order to route it
    to the fixed, clinician-offering reply before the classifier ever sees it.
    """
    if not text or not text.strip():
        return False
    t = unicodedata.normalize("NFC", text)
    lowered = t.lower()

    if any(p in lowered for p in _ALL_LATIN):
        return True
    if any(_bounded(p, t, _BN_CHAR) for p in _ALL_BN):
        return True
    if any(_bounded(p, t, _HI_CHAR) for p in _ALL_HI):
        return True

    if _reading_a_lab_value(lowered, t):
        return True
    if _indic_value_mention(t, _BN_CLINICAL_TERMS, _BN_CHAR):
        return True
    if _indic_value_mention(t, _HI_CLINICAL_TERMS, _HI_CHAR):
        return True

    return bool(
        _RE_EN_PRETEND_DOCTOR.search(lowered)
        or _RE_EN_WHAT_DOES_MEAN.search(lowered)
        or _RE_LATIN_FARZ_DOCTOR.search(lowered)
        or _RE_BN_DHORUN_DOCTOR.search(t)
        or _RE_HI_MAAN_DOCTOR.search(t)
    )
