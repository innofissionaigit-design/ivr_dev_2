"""A possible medical emergency, detected deterministically, before any model.

An external review of this repository (the "no guessing" release gate) pointed out that the speech
policy has an `emergency` row but nothing could ever put the call into that state: the caller-state
set did not contain it and no detector existed. This is the detector. It is deliberately NOT a
classifier and NOT clinical: it recognises a fixed set of phrases for the situations a clinic's
phone line should never keep booking through -- trouble breathing, chest pain, loss of
consciousness, heavy bleeding, fits, poisoning or overdose, a serious accident, self-harm, or an
explicit call for an ambulance or "emergency" -- in Bengali, Hindi and English, and it reports only
"a possible emergency was said". The agent never says what the condition is or what to do about it
beyond the fixed notice in agent/phrases.py (`emergency_notice`).

RECALL OVER PRECISION. A false alarm costs the caller one short notice and a person joining the
call; a miss costs far more. So this does not try to understand negation ("no chest pain") and it is
run on the transcript of EVERY turn in EVERY language regardless of which language was routed, and
regardless of the ASR confidence -- a garbled transcript that still contains "ambulance" is treated
as one.

WHAT THIS CANNOT DO, and it is a real limit: it runs on a finished transcript, one turn at a time,
after the caller has stopped speaking. A caller who is mid-sentence, or whose audio never yields
these words in the transcript, is not caught. A true safety path needs a streaming recogniser on
the audio itself (an external review's recommendation; not built here). The phrase lists were
written by a non-native speaker; the project owner has approved the lists and the notice wording
(`REVIEW_STATUS` below), and a clinician or native reviewer may still ask for changes.
"""

from __future__ import annotations

import re
import unicodedata

from agent.phrase_proximity import any_phrase_matches

# Approved by the project owner in chat on 2026-09-24: the phrase samples, the notice (wording A) and the
# greeting pointer in docs/emergency-for-approval.md. The document's sign-off table still has no named
# clinician or native reviewer; if the clinic requires one, that is a separate sign-off.
REVIEW_STATUS = "approved_by_project_owner"

# The number the notice tells the caller to dial. 112 is India's national emergency number; it is a
# setting, not a constant, because the clinic decides what its notice says.
DEFAULT_EMERGENCY_NUMBER = "112"

_EN = re.compile(
    r"\b(chest pain|pain in (my|the) chest|(can'?t|cannot|can not|unable to|not able to|difficulty|trouble) breath\w*|"
    r"short(ness)? of breath|stopped breathing|not breathing|struggling to breathe|"
    r"unconscious|fainted|passed out|collapsed|not responding|unresponsive|"
    r"heavy(ly)? bleeding|bleeding (a lot|heavily|badly|profusely|(won'?t|will not|does not|doesn'?t) stop)|"
    r"seizure|convulsion|having a fit|fits|"
    r"poison\w*|overdos\w*|swallowed (something|pills|poison|acid)|"
    r"heart attack|stroke|snake ?bite|"
    r"serious accident|met with an accident|road accident|accident happened|badly (hurt|injured)|"
    r"suicid\w*|kill (myself|himself|herself)|want to die|end my life|"
    r"call (an )?ambulance|need (an )?ambulance|ambulance|"
    r"(this|it) is an emergency|it'?s an emergency|medical emergency|emergency case)\b",
    re.I,
)

_BN = re.compile(
    r"(বুকে ব্যথা|বুকে যন্ত্রণা|বুকে চাপ|শ্বাস(কষ্ট| নিতে (পারছ|পারছি|কষ্ট))|নিঃশ্বাস (নিতে|বন্ধ)|দম (বন্ধ|আটকে)|"
    r"অজ্ঞান|জ্ঞান (নেই|হারিয়ে)|সাড়া (দিচ্ছে না|দিচ্ছেন না)|"
    r"রক্তক্ষরণ|প্রচুর রক্ত|রক্ত (পড়ছে|বন্ধ হচ্ছে না|থামছে না)|"
    r"খিঁচুনি|খিচুনি|ফিট (হয়েছে|হচ্ছে)|"
    r"বিষ (খেয়ে|খেয়েছ)|বিষক্রিয়া|ওভারডোজ|"
    r"হার্ট অ্যাটাক|হার্টঅ্যাটাক|স্ট্রোক|সাপে কামড়|"
    r"দুর্ঘটনা|অ্যাক্সিডেন্ট|অ্যাকসিডেন্ট|"
    r"আত্মহত্যা|মরে যেতে চাই|"
    r"অ্যাম্বুলেন্স|এম্বুলেন্স|জরুরি (অবস্থা|অবস্থা|ভিত্তিতে)|ইমার্জেন্সি|এমারজেন্সি)"
)

_HI = re.compile(
    r"(सीने में (दर्द|जलन|भारीपन)|छाती में दर्द|सांस (नहीं|लेने में (तकलीफ|दिक्कत|परेशानी))|साँस (नहीं|लेने में)|दम घुट|"
    r"बेहोश|होश (नहीं|खो)|जवाब नहीं दे रहे|"
    r"खून (बह रहा|बहुत बह|नहीं रुक)|ज़्यादा खून|ज्यादा खून|"
    r"दौरा|मिर्गी|झटके|"
    r"ज़हर|जहर|ओवरडोज़|ओवरडोज|"
    r"दिल का दौरा|हार्ट अटैक|स्ट्रोक|साँप ने काट|सांप ने काट|"
    r"दुर्घटना|एक्सीडेंट|"
    r"आत्महत्या|मर जाना चाहता|मर जाना चाहती|"
    r"एम्बुलेंस|एंबुलेंस|आपातकाल|इमरजेंसी)"
)

_PATTERNS = (_EN, _BN, _HI)

# A live call (2026-09-28): "বুকে খুব ব্যথা করছে" ("my chest REALLY hurts") matched NONE of the patterns above,
# because "বুকে ব্যথা" there is a fixed two-word ADJACENT phrase and "খুব" (an intensifier) sits between the two
# words. An ordinary way to say "my chest hurts a lot" must not silently fail to be an emergency. Rather than
# rewrite (and re-review) every approved phrase above into a gap-tolerant regex by hand, this is a SECOND,
# ADDITIVE check over the plain multi-word phrases most exposed to exactly this (an intensifier or filler word
# inserted between two content words): it can only make detect_emergency() MORE likely to fire on a real case
# (RECALL OVER PRECISION, this module's own rule), never less -- the patterns above are unchanged and still run.
_GAP_TOLERANT_PHRASES = (
    "chest pain",
    "not responding",
    "passed out",
    "heart attack",
    "serious accident",
    "road accident",
    "medical emergency",
    "বুকে ব্যথা",
    "বুকে যন্ত্রণা",
    "বুকে চাপ",
    "প্রচুর রক্ত",
    "হার্ট অ্যাটাক",
    "সাপে কামড়",
    "মরে যেতে চাই",
    "छाती में दर्द",
    "ज़्यादा खून",
    "ज्यादा खून",
    "दिल का दौरा",
    "हार्ट अटैक",
    "साँप ने काट",
    "सांप ने काट",
    "मर जाना चाहता",
    "मर जाना चाहती",
)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def detect_emergency(text: str) -> bool:
    """True if the transcript contains an emergency phrase in ANY of the three languages."""
    t = _nfc(text)
    if not t:
        return False
    return any(p.search(t) for p in _PATTERNS) or any_phrase_matches(_GAP_TOLERANT_PHRASES, t)
