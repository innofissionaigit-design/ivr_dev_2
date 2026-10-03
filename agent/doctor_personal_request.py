"""A caller asking to speak to a doctor personally: say plainly what is possible.

Ported from `merged_code/agent/doctor_personal_request.py` (story: "Caller wants
to speak to a doctor personally"). Its acceptance criteria were: never claim a
direct connection this system cannot make, never promise that a NAMED doctor
will ring back, never repeat or disclose a doctor's personal contact details,
and offer what the system genuinely can do instead.

WHY THIS IS NOT COVERED BY agent/human_request.py:

That module's noun list is `person|human|someone|somebody|operator|
representative|receptionist|staff|manager|counter` -- deliberately the FRONT
DESK. It does not contain "doctor" or "clinician", so "connect me to a doctor"
matches nothing today and falls through to the model, which has no fact to
answer it with and no instruction about what may be promised. The two requests
also want different answers: a caller asking for the front desk can be handed
straight to the front desk, which is why `asks_for_a_person()` hands off
immediately. A caller asking for a doctor cannot be -- there is no clinician on
this line and no way to reach one from it -- so the honest reply has to decline
the direct connection AND offer the appointment that is the real route to a
doctor.

WHY A DOCTOR'S NAME IS NEVER CAPTURED:

A doctor's name is open vocabulary -- this system has no fixed list to match
against without the live catalogue -- so the named case is a regex over a
"Dr <name>"-shaped token near a personal-contact trigger. The name itself is
deliberately **not** captured: the reply never repeats it, so there is nothing
to extract correctly in the first place, and nothing a mis-extraction could
disclose. That is also why this module returns only a boolean and a family tag.

REASONED, not measured: the Devanagari list and the Devanagari named-doctor
regex are new in this repository; the original covered Bengali, English and
Latin-script transliteration only. A native speaker should review them.
"""

from __future__ import annotations

import re
import unicodedata

_BN_CHAR = r"[ঀ-৿]"


def _bounded(word: str, text: str, char_class: str = _BN_CHAR) -> bool:
    return re.search(rf"(?<!{char_class}){re.escape(word)}(?!{char_class})", text) is not None


# ---- English: built combinatorially, both word orders ---------------------
_EN_CONTACT_VERBS = (
    "speak to", "speak with", "talk to", "talk with",
    "speak directly to", "speak directly with",
    "talk directly to", "talk directly with",
    "connect me to", "connect me with",
)
_EN_DOCTOR_NOUNS = (
    "a doctor", "the doctor", "my doctor",
    "a clinician", "the clinician", "my clinician",
)
_EN_TRAILING_CONTACT = (
    "to call me", "to ring me", "to talk to me",
    "to speak to me", "to speak with me", "to reassure me",
)
_EN = (
    tuple(f"{v} {n}" for v in _EN_CONTACT_VERBS for n in _EN_DOCTOR_NOUNS)
    + tuple(f"{n} {t}" for n in _EN_DOCTOR_NOUNS for t in _EN_TRAILING_CONTACT)
    + ("i want clinical reassurance", "i need clinical reassurance")
)

# ---- Hindi and Bengali in Latin script -----------------------------------
_LATIN = (
    "doctor se baat karna chahta hoon", "doctor se baat karni hai",
    "doctor se baat karna chahti hoon", "mujhe doctor se baat karni hai",
    "doctor se seedhe baat karna chahta hoon", "doctor se seedhe baat karni hai",
    "doctor se connect karo", "doctor se connect kar dijiye",
    "clinician se baat karna chahta hoon", "clinician se baat karni hai",
    "doctor mujhe call kare", "doctor mujhe phone kare",
    "doctor se call karwa dijiye", "doctor ko bolo mujhe call kare",
    "doctor ko kahiye mujhe call kare", "mera doctor mujhe call kare",
    "mujhe apne doctor se baat karni hai",
    "daktarer sathe kotha bolte chai", "daktarer sathe sojasuji kotha bolte chai",
    "daktar amake call korun", "daktar amake phone korun",
    "amar daktar amake call korun", "daktar ke bolben amake call korte",
    "daktarer sathe directly kotha bolte chai",
)

# ---- Bengali -------------------------------------------------------------
_BN = (
    "ডাক্তারের সাথে কথা বলতে চাই", "ডাক্তারের সাথে সরাসরি কথা বলতে চাই",
    "একজন ডাক্তারের সাথে কথা বলতে চাই", "ডাক্তারের সাথে যোগাযোগ করতে চাই",
    "ডাক্তার আমাকে কল করুন", "ডাক্তার আমাকে ফোন করুন",
    "আমার ডাক্তার আমাকে কল করুন", "আমার ডাক্তার আমাকে ফোন করুন",
    "একজন ডাক্তার আমাকে কল করুন", "ডাক্তারকে বলুন আমাকে কল করতে",
    "ডাক্তারকে বলুন আমাকে ফোন করতে",
)

# ---- Hindi (Devanagari) -- new in this repository -------------------------
_HI = (
    "डॉक्टर से बात करनी है", "मुझे डॉक्टर से बात करनी है",
    "डॉक्टर से बात करना चाहता हूँ", "डॉक्टर से बात करना चाहती हूँ",
    "डॉक्टर से सीधे बात करनी है", "डॉक्टर से जोड़ दीजिए",
    "डॉक्टर से मिलाइए", "किसी डॉक्टर से बात करनी है",
    "डॉक्टर मुझे फ़ोन करें", "डॉक्टर मुझे फोन करें",
    "डॉक्टर मुझे कॉल करें", "मेरे डॉक्टर मुझे कॉल करें",
    "डॉक्टर से कहिए मुझे कॉल करें", "डॉक्टर को बोलिए मुझे फ़ोन करें",
)

# ---- a NAMED doctor plus a personal-contact trigger, either order --------
_DR = r"(?:dr\.?|doctor)\s+[a-z]+"
_TRIGGER = r"(?:call me|ring me|speak to me|speak with me|talk to me|talk with me)"
_RE_EN_NAMED = re.compile(
    rf"(?:{_DR}.{{0,40}}{_TRIGGER})"
    rf"|(?:{_TRIGGER}.{{0,40}}{_DR})"
    rf"|(?:(?:speak to|speak with|talk to|talk with|connect me to|connect me with).{{0,10}}{_DR})"
    rf"|(?:(?:ask|tell).{{0,15}}{_DR}.{{0,20}}(?:to call me|to ring me|to speak to me))"
    rf"|(?:can.{{0,5}}{_DR}.{{0,20}}(?:speak to me|speak with me|talk to me|call me))",
    re.IGNORECASE,
)
_RE_LATIN_NAMED = re.compile(
    r"(?:dr\.?\s+[a-z]+.{0,30}(?:call kare|phone kare|call karo|baat kare|amake call korun|amake phone korun))"
    r"|(?:(?:bolo|bolen|bolben|kahiye).{0,20}dr\.?\s+[a-z]+.{0,20}(?:call kare|phone kare))",
    re.IGNORECASE,
)
_RE_BN_NAMED = re.compile(
    r"(?:(?:ডা\.?|ডক্টর)\s*\S+.{0,20}(?:কল করুন|ফোন করুন|কল করবেন|ফোন করবেন))"
)
# New in this repository, same shape as the Bengali one above.
_RE_HI_NAMED = re.compile(
    r"(?:(?:डॉ\.?|डॉक्टर)\s*\S+.{0,20}(?:कॉल करें|फ़ोन करें|फोन करें|कॉल कीजिए))"
)


def detect_doctor_personal_request(text: str) -> tuple[bool, str | None]:
    """True, plus which phrase family matched, when `text` asks to speak to a
    doctor personally or to have one ring back.

    The tag is for logging only, never the reply language. A doctor's name is
    never captured -- see the module docstring.
    """
    if not text or not text.strip():
        return False, None
    t = unicodedata.normalize("NFC", text)
    lowered = t.lower()

    if any(p in lowered for p in _EN):
        return True, "en"
    if any(p in lowered for p in _LATIN):
        return True, "latin"
    if any(_bounded(p, t) for p in _BN):
        return True, "bn"
    if any(_bounded(p, t, r"[ऀ-ॿ]") for p in _HI):
        return True, "hi"
    if _RE_EN_NAMED.search(lowered) or _RE_LATIN_NAMED.search(lowered):
        return True, "named"
    if _RE_BN_NAMED.search(t) or _RE_HI_NAMED.search(t):
        return True, "named"
    return False, None


def is_doctor_personal_request(text: str) -> bool:
    matched, _ = detect_doctor_personal_request(text)
    return matched
