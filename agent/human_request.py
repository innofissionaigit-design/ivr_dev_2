"""KCD-353: "a request for a person is honoured immediately."

A caller who asks for a person must reach one, without the agent deciding they do
not really mean it, asking them to repeat it, or routing the request through a
model. So the detection is deterministic and runs BEFORE any language model call
and before intent extraction: a fixed set of phrases in each language, checked
against the transcript, with a negation guard ("I don't need a person").

It is deliberately generous. The two errors are not symmetric: a caller who says
"can I get someone to help" and is offered a person they did not quite want loses
a few seconds; a caller who asked for a person and got another round of the bot
has been refused. When in doubt this returns True. "Nurse", "doctor" and
"receptionist" as bare words do NOT count -- "book me with the doctor" is not a
request for a person -- only a request to SPEAK to someone does.

Phrase lists are provisional and were written by a non-native speaker
(agent/persona.py REVIEW_STATUS): expect a native reviewer to add more.
"""

from __future__ import annotations

import re
import unicodedata

# ---- English -------------------------------------------------------------
_EN_PERSON = r"(a |an |the |some |your |any )?(real |live |actual )?(person|human|man|woman|someone|somebody|agent|operator|representative|receptionist|staff|staff member|executive|manager|counter)"
# "I want a person" needs a narrower list than "let me talk to <X>": "I need a test at the
# counter" and "I want to know about your staff" are not requests to speak to anyone.
_EN_PERSON_ONLY = (
    r"(a |an |the |some )?(real |live |actual )?(person|human|someone|somebody|operator|representative|receptionist)"
)
_EN = [
    re.compile(
        rf"\b(talk|speak|connect|transfer|put me|get me|give me|pass me)\b.{{0,25}}\b(to|with|through to)\b\s*{_EN_PERSON}\b",
        re.I,
    ),
    re.compile(
        rf"\b(i (want|need|would like|wish)|can i (get|have)|let me (talk|speak)).{{0,25}}{_EN_PERSON_ONLY}\b", re.I
    ),
    re.compile(
        r"\b(human being|real person|live agent|customer (care|service|support)|call ?center|call centre)\b", re.I
    ),
    re.compile(r"\bnot (a )?(robot|bot|machine)\b", re.I),
]
# ---- Bengali -------------------------------------------------------------
_BN = [
    re.compile(
        r"(মানুষের|লোকের|কারও|কারো|কাউকে|কোনো একজনের|স্টাফের|কর্মীর|অপারেটরের|রিসেপশনের|কাউন্টারের|ম্যানেজারের)\s*(সাথে|সঙ্গে|সাথে|কাছে)?\s*(কথা|কথা বলতে|কথা বলব|কথা বলতে চাই|কথা বলি)"
    ),
    re.compile(r"(কাউকে|কোনো মানুষকে|স্টাফকে|অপারেটরকে|রিসেপশনকে)\s*(দিন|ধরিয়ে দিন|লাইনে দিন|দেবেন|ধরিয়ে দেবেন)"),
    re.compile(r"(আসল|সত্যিকারের|জ্যান্ত)\s*(মানুষ|লোক)"),
    re.compile(r"(অপারেটর|রিসেপশনিস্ট|কাস্টমার কেয়ার)"),
    re.compile(r"(যন্ত্র|মেশিন|রোবট)\s*(না|নয়|চাই না)"),
]
# ---- Hindi ---------------------------------------------------------------
_HI = [
    re.compile(r"(किसी|एक)?\s*(इंसान|इन्सान|आदमी|व्यक्ति|स्टाफ|ऑपरेटर|रिसेप्शन|मैनेजर|काउंटर)\s*(से|को)\s*(बात|बोल|मिला|जोड़)"),
    re.compile(r"(असली|सचमुच के)\s*(इंसान|इन्सान|आदमी|व्यक्ति)"),
    re.compile(r"(किसी को)\s*(दीजिए|दीजिये|जोड़िए|लाइन पर)"),
    re.compile(r"(ऑपरेटर|रिसेप्शनिस्ट|कस्टमर केयर)"),
    re.compile(r"(मशीन|रोबोट|बॉट)\s*(नहीं|से नहीं)"),
]

_NEGATED_EN = re.compile(
    r"\b(don'?t|do not|no need|not need|never|without)\b.{0,20}(talk|speak|person|human|agent|staff|operator)", re.I
)
_NEGATED_BN = re.compile(r"(লাগবে না|চাই না|দরকার নেই|দরকার নাই)")
_NEGATED_HI = re.compile(r"(नहीं चाहिए|नहीं चाहिये|ज़रूरत नहीं|जरूरत नहीं)")

_PATTERNS = {"en": _EN, "bn": _BN, "hi": _HI}
_NEGATED = {"en": _NEGATED_EN, "bn": _NEGATED_BN, "hi": _NEGATED_HI}


def _nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t or "")


def asks_for_a_person(text: str, lang: str = "en") -> bool:
    """True if the caller asked to speak to a person. All three languages are
    always checked (a caller may switch mid-call); `lang` only orders them."""
    t = _nfc(text).strip()
    if not t:
        return False
    for lg in dict.fromkeys((lang, "en", "bn", "hi")):
        if any(p.search(t) for p in _PATTERNS.get(lg, [])):
            # a bare keyword hit is dropped if the caller said they do NOT want that
            # ("customer care বলে কিছু লাগবে না"); but a NEGATED_EN/BN/HI match only
            # cancels a request that is not also a clear request form
            if _NEGATED.get(lg) and _NEGATED[lg].search(t):
                continue
            return True
    return False
