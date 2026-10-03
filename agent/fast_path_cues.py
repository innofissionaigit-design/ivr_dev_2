"""Per-language cue tables for the deterministic fast path (KCD-095, KCD-094).

The fast path decides "is this a price question, a doctor-availability question, a test-preparation question, a
clinic FAQ, a greeting" from CUE words before it looks for the entity (which test, which doctor). Those cues used to
be Bengali literals inline in agent/fast_path.py, so a Hindi or English caller was never served by it: speed was a
Bengali-only privilege. They are now DATA, one `CueTable` per language behind one interface:

    register(CueTable(lang="ta", ...))      # adding a language is data, not code

`FastPath.resolve(text, lang)` picks the table for `lang`. A language with no table -- including "unknown", the
language-identification result for an utterance that is none of the supported languages -- is not guessed at: the
fast path abstains and the turn goes to the model (Blueprint 4.4), so a Bengali cue table can never fire on a
Hindi sentence.

REASONED, NOT MEASURED. The Bengali table is the pre-existing one, unchanged. The Hindi and English tables were
written from the same intent categories by a non-native reviewer; they are deliberately conservative (a cue that is
ambiguous in that language -- Hindi "kal" means both tomorrow and yesterday -- is treated as "I am not sure" and the
turn abstains). The serve-rate gap between languages is REPORTED (`FastPath.snapshot()["language_gap"]`) and a gap
wider than LANGUAGE_GAP_MARGIN is a defect to fix in this data, not to accept. Recalibrate against real call
transcripts, and have a native speaker of each language review the cues, before trusting either.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

_BN_RATE = (
    "রেট",
    "দাম",
    "খরচ",
    "চার্জ",
    "মূল্য",
    "কত টাকা",
    "কত পড়বে",
    "কত লাগবে",
    "কত নেবে",
    "প্রাইস",
    "টাকা লাগে",
)
_BN_AVAIL = (
    "কবে",
    "কখন",
    "বসবেন",
    "বসেন",
    "চেম্বার",
    "আছেন",
    "থাকবেন",
    "পাওয়া যাবে",
    "ভিজিট",
    "সময়সূচি",
    "শিডিউল",
)
_BN_BOOK = ("বুক", "বুকিং", "অ্যাপয়েন্টমেন্ট", "অ্যাপয়েনমেন্ট", "সিরিয়াল", "নাম লেখা", "স্লট")
# What must the caller DO before/for a test -- distinct from _BN_RATE
# ("কত টাকা") and _BN_AVAIL (a DOCTOR's schedule), so a test-prep
# question never gets misrouted as a price or a doctor question even
# though all three can mention a test/doctor name in the same sentence
# shape.
_BN_PREP = (
    "প্রস্তুতি",
    "উপবাস",
    "উপোস",
    "খালি পেটে",
    "ফাস্টিং",
    "আগে কী করতে হবে",
    "আগে কি করতে হবে",
    "কী মানতে হবে",
    "কি মানতে হবে",
    "খাওয়া যাবে কিনা",
    "আগে খাওয়া যাবে",
)
_BN_GREETING = ("নমস্কার", "নমষ্কার", "হ্যালো", "হ্যালো?", "শুভ সকাল", "আসসালামু")
_BN_THANKS = ("ধন্যবাদ", "থ্যাঙ্ক", "থ্যাংক")

# Relative day words the fast path is willing to resolve itself. Anything
# else with a date in it (weekday names, "১৫ তারিখে", explicit dates) goes
# to the LLM, which already has date-resolution rules and today's date.
_BN_RELATIVE_DAYS = {"আজ": 0, "আজকে": 0, "কাল": 1, "আগামীকাল": 1, "কালকে": 1, "পরশু": 2}

# Words that make an utterance more than a simple lookup: a comparison, a
# list request, a negation, a follow-up. Cheap insurance -- if any appear,
# abstain rather than answer half the question.
_BN_COMPLEXITY = (
    "সব",
    "সবগুলো",
    "তালিকা",
    "কোন কোন",
    "আর",
    "এবং",
    "না",
    "নাকি",
    "বদলে",
    "চেয়ে",
    "ছাড়া",
    "কিন্তু",
    "অন্য",
)

# Little words that carry no test or doctor name ("do I have to ...", "is it needed"). A follow-up question with a cue
# and nothing but these (and pointing words) names nothing -- it is about the topic we were just on (KCD-396). A word
# NOT in this list is treated as a possible name, and the fast path abstains: the failure direction is the model.
_BN_FUNCTION_WORDS = (
    "কত",
    "কতটা",
    "করতে",
    "করা",
    "করব",
    "হবে",
    "হয়",
    "হলে",
    "লাগবে",
    "লাগে",
    "লাগছে",
    "কি",
    "কী",
    "কিনা",
    "কি না",
    "আমার",
    "আমাকে",
    "আমি",
    "বলুন",
    "বলবেন",
    "বলেন",
    "তো",
    "এখন",
    "জানতে",
    "চাই",
    "চাইছি",
    "আছে",
    "দরকার",
    "প্রয়োজন",
    "যাবে",
    "পারব",
    "পারি",
    "একটু",
    "জন্য",
    "এর",
    "হ্যাঁ",
    "আচ্ছা",
    "তাহলে",
    "তবে",
    "থাকতে",
    "খেয়ে",
    "আসতে",
)

_BN_DATE_WORDS = ("সোম", "মঙ্গল", "বুধ", "বৃহস্পতি", "শুক্র", "শনি", "রবি", "তারিখ")

# Hindi. Whitespace-delimited and, for these cues, not inflected, so whole-word matching is right (a substring
# hit would fire "sab" inside "sabse"). "kal" (tomorrow AND yesterday) and "parson" (day after tomorrow AND day
# before yesterday) are deliberately date words, not relative days: they cannot be resolved without the model.
_HI_RATE = (
    "रेट",
    "दाम",
    "कीमत",
    "प्राइस",
    "चार्ज",
    "फीस",
    "खर्चा",
    "कितने रुपये",
    "कितने पैसे",
    "कितने का",
    "कितना खर्च",
    "कितना पड़ेगा",
)
_HI_AVAIL = (
    "कब",
    "कितने बजे",
    "बैठते",
    "बैठती",
    "बैठेंगे",
    "चैंबर",
    "उपलब्ध",
    "मिलेंगे",
    "मिलते",
    "विज़िट",
    "विजिट",
    "शेड्यूल",
)
_HI_BOOK = (
    "बुक",
    "बुकिंग",
    "अपॉइंटमेंट",
    "अपॉइन्टमेंट",
    "अपोइंटमेंट",
    "सीरियल",
    "स्लॉट",
    "नाम लिखवा",
    "नंबर लगा",
)
_HI_PREP = ("तैयारी", "उपवास", "खाली पेट", "फास्टिंग", "पहले क्या करना", "पहले क्या करें")
_HI_GREETING = ("नमस्ते", "नमस्कार", "हैलो", "हेलो", "सुप्रभात")
_HI_THANKS = ("धन्यवाद", "शुक्रिया", "थैंक यू", "थैंक्स")
_HI_COMPLEXITY = (
    "सब",
    "सभी",
    "सारे",
    "सूची",
    "लिस्ट",
    "और",
    "तथा",
    "नहीं",
    "या",
    "बजाय",
    "लेकिन",
    "मगर",
    "दूसरा",
    "दूसरे",
    "अलग",
    "बनाम",
)
_HI_FUNCTION_WORDS = (
    "कितना",
    "कितने",
    "कितनी",
    "करना",
    "करने",
    "करनी",
    "होगा",
    "होगी",
    "होता",
    "पड़ेगा",
    "पड़ता",
    "पड़ेगी",
    "क्या",
    "मुझे",
    "मेरा",
    "मेरे",
    "मैं",
    "है",
    "हैं",
    "के",
    "का",
    "की",
    "लिए",
    "भी",
    "तो",
    "बताइए",
    "बताओ",
    "चाहिए",
    "जरूरी",
    "ज़रूरी",
    "आना",
    "आना",
    "थोड़ा",
    "अच्छा",
    "तब",
)
_HI_DATE_WORDS = (
    "सोमवार",
    "मंगलवार",
    "बुधवार",
    "गुरुवार",
    "शुक्रवार",
    "शनिवार",
    "रविवार",
    "तारीख",
    "कल",
    "परसों",
    "सुबह",
    "शाम",
    "दोपहर",
    "रात",
    "हफ्ते",
    "सप्ताह",
    "अगले",
    "पिछले",
)

# English. Whole-word matching only ("rate" must not fire inside "separate"). Date words are many on purpose: an
# English "tonight" or "next Monday" that the fast path did not parse would answer for the wrong day.
_EN_RATE = (
    "price",
    "prices",
    "pricing",
    "cost",
    "costs",
    "rate",
    "rates",
    "charge",
    "charges",
    "fee",
    "fees",
    "how much",
)
_EN_AVAIL = (
    "when",
    "what time",
    "available",
    "availability",
    "timing",
    "timings",
    "sit",
    "sits",
    "sitting",
    "chamber",
    "schedule",
    "visit",
)
_EN_BOOK = (
    "book",
    "booking",
    "appointment",
    "appointments",
    "reserve",
    "reservation",
    "slot",
    "slots",
    "arrange",
)
_EN_PREP = (
    "preparation",
    "preparations",
    "prepare",
    "fasting",
    "fast",
    "empty stomach",
    "before the test",
    "eat before",
    "drink before",
    "what should i do before",
    "instructions",
)
_EN_GREETING = (
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "namaste",
)
_EN_THANKS = ("thank you", "thanks", "thank you very much")
_EN_COMPLEXITY = (
    "all",
    "list",
    "every",
    "and",
    "or",
    "not",
    "no",
    "instead",
    "other",
    "another",
    "but",
    "compare",
    "both",
    "also",
    "than",
    "versus",
    "vs",
    "cheaper",
    "cheapest",
    "best",
    "different",
    "else",
)
_EN_FUNCTION_WORDS = (
    "do",
    "does",
    "did",
    "i",
    "me",
    "my",
    "need",
    "needs",
    "have",
    "has",
    "to",
    "should",
    "must",
    "a",
    "an",
    "is",
    "it",
    "its",
    "for",
    "what",
    "about",
    "how",
    "much",
    "is",
    "there",
    "any",
    "please",
    "tell",
    "so",
    "then",
    "the",
    "of",
    "be",
    "required",
    "necessary",
    "needed",
    "before",
    "that",
    "this",
    "one",
    "same",
    "test",
    "can",
    "will",
    "would",
    "you",
    "ok",
    "okay",
)
_EN_DATE_WORDS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "yesterday",
    "tonight",
    "morning",
    "afternoon",
    "evening",
    "night",
    "week",
    "weekend",
    "next",
    "last",
    "later",
    "date",
)


@dataclass(frozen=True)
class CueTable:
    lang: str
    rate: tuple[str, ...]
    avail: tuple[str, ...]
    book: tuple[str, ...]
    prep: tuple[str, ...]
    greeting: tuple[str, ...]
    thanks: tuple[str, ...]
    complexity: tuple[str, ...]
    relative_days: dict = field(default_factory=dict)  # word -> days from today; checked in this order
    date_words: tuple[str, ...] = ()  # a date-ish word the fast path will not parse
    function_words: tuple[str, ...] = ()  # words that name nothing (see _BN_FUNCTION_WORDS)
    greeting_reply: str = ""  # spoken for a bare greeting; empty = do not serve one
    thanks_reply: str = ""
    # "substring": a cue may sit inside an inflected word (Bengali "রেট" inside "রেটটা"); "word": whole words only.
    match: str = "word"
    # A spoken form shorter than this must match a window EXACTLY: at three letters, one character of difference
    # is a ratio of 0.67-0.86, which is how "days" would be read as the surname "Das". 0 = the old behaviour.
    exact_below_chars: int = 5
    # True: a date-ish word or a digit anywhere makes the date "not sure" even when a relative day ("tomorrow") is
    # also present. False is the original Bengali order (a relative day wins), kept so Bengali is unchanged.
    strict_dates: bool = True
    # True: an availability cue ("when", "what time") with no doctor named falls through to the FAQ topics, so "what
    # time do you open" is answered as opening hours. False is the original Bengali behaviour (give up at once).
    faq_after_failed_avail: bool = True

    def contains(self, text: str, cue: str) -> bool:
        if self.match == "substring":
            return cue in text
        return f" {cue} " in f" {text} "

    def any(self, text: str, cues) -> bool:
        return any(self.contains(text, c) for c in cues)


def normalise_cue(text: str) -> str:
    """The same normalisation the transcript gets (NFC, nukta dropped, lower case), so a cue written in either
    Unicode form still matches."""
    t = unicodedata.normalize("NFC", text).replace("़", "").lower()
    return " ".join(t.replace("?", " ").replace("!", " ").replace(",", " ").split())


def _norm_all(cues) -> tuple[str, ...]:
    return tuple(dict.fromkeys(c for c in (normalise_cue(x) for x in cues) if c))


def _table(**kw) -> CueTable:
    for key in (
        "rate",
        "avail",
        "book",
        "prep",
        "greeting",
        "thanks",
        "complexity",
        "date_words",
        "function_words",
    ):
        if key in kw:
            kw[key] = _norm_all(kw[key])
    if "relative_days" in kw:
        kw["relative_days"] = {normalise_cue(k): v for k, v in kw["relative_days"].items()}
    return CueTable(**kw)


_TABLES: dict[str, CueTable] = {}


def register(table: CueTable) -> None:
    """Add or replace a language. Nothing else needs to change: FastPath and Catalogue read this registry."""
    _TABLES[table.lang] = table


def table_for(lang: str | None) -> CueTable | None:
    return _TABLES.get(lang or "")


def languages() -> tuple[str, ...]:
    return tuple(_TABLES)


register(
    _table(
        lang="bn",
        rate=_BN_RATE,
        avail=_BN_AVAIL,
        book=_BN_BOOK,
        prep=_BN_PREP,
        greeting=_BN_GREETING,
        thanks=_BN_THANKS,
        complexity=_BN_COMPLEXITY,
        relative_days=_BN_RELATIVE_DAYS,
        date_words=_BN_DATE_WORDS,
        function_words=_BN_FUNCTION_WORDS,
        greeting_reply="নমস্কার, কী সাহায্য করতে পারি?",
        thanks_reply="ধন্যবাদ। আর কিছু জানতে চান?",
        match="substring",
        exact_below_chars=0,
        strict_dates=False,
        faq_after_failed_avail=False,  # exactly the behaviour this had before it was data
    )
)
register(
    _table(
        lang="hi",
        rate=_HI_RATE,
        avail=_HI_AVAIL,
        book=_HI_BOOK,
        prep=_HI_PREP,
        greeting=_HI_GREETING,
        thanks=_HI_THANKS,
        complexity=_HI_COMPLEXITY,
        relative_days={"आज": 0},
        date_words=_HI_DATE_WORDS,
        function_words=_HI_FUNCTION_WORDS,
        greeting_reply="नमस्ते, मैं आपकी क्या मदद कर सकती हूँ?",
        thanks_reply="धन्यवाद। क्या आप कुछ और जानना चाहेंगे?",
    )
)
register(
    _table(
        lang="en",
        rate=_EN_RATE,
        avail=_EN_AVAIL,
        book=_EN_BOOK,
        prep=_EN_PREP,
        greeting=_EN_GREETING,
        thanks=_EN_THANKS,
        complexity=_EN_COMPLEXITY,
        relative_days={"day after tomorrow": 2, "tomorrow": 1, "today": 0},
        date_words=_EN_DATE_WORDS,
        function_words=_EN_FUNCTION_WORDS,
        greeting_reply="Hello, how can I help you?",
        thanks_reply="You are welcome. Is there anything else you would like to know?",
    )
)
