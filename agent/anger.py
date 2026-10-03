"""Plain anger or frustration from the caller: hear it, slow down, offer a person.

Ported from `merged_code/agent/anger_flow.py` (story: "Caller is angry about a
previous experience"). Its acceptance criteria were:

  1. Anger lowers the escalation threshold.   2. Delivery slows.
  3. A human is offered explicitly.           4. The agent apologises once.
  5. The agent does not argue.                6. It does not defend the hospital.
  7. The frustration is captured in the context packet.

WHY THIS MODULE IS ONLY A DETECTOR, AND NOT A FLOW:

Every one of those seven criteria is ALREADY implemented in this repository, in
all three languages, and was dead code before this module existed. The
`"angry"` row of agent/speech_policy.py already sets `escalation_threshold="low"`
(AC 1), `speech_rate="slow-normal"` (AC 2), `offer_human=True` with
`confirmation="explicit_offer_human"` (AC 3) and `acknowledge_first=True`; the
`"angry"` entry of agent/acknowledgement.py already carries one apology, in bn,
hi and en, which agent/apology.py's single-apology rule then holds to exactly one
(AC 4); the acknowledgement is fixed text, so the model never gets a turn in
which it could argue or defend (AC 5, AC 6); and agent/call_state.py already
records `caller_state` in `to_log_dict()` (AC 7).

What was missing was the one thing no module did: `"angry"` appeared in
agent/call_state.py's VALID_CALLER_STATES, in agent/speech_policy.py's `_ROWS`
and in agent/acknowledgement.py's `_ACK`, and NOWHERE ELSE in the repository --
nothing ever set it. So porting the original 220-line flow would have duplicated
six working mechanisms to reach the same behaviour. This module supplies the
detector alone, and `main.py`/`main_pcm.py`'s existing caller-state step does the
rest. The original module's phrase lists are what is actually ported here.

WHY DETECTION IS DETERMINISTIC AND RUNS BEFORE THE MODEL:

Same reasoning as agent/abuse.py and agent/human_request.py: a policy the model
is asked to apply is a policy the model can be talked out of. Anger is caught by
a fixed list of unambiguous, multi-word statements, checked in every language at
once -- a caller speaking Bengali may well say "this is ridiculous" in English,
exactly as agent/abuse.py already assumes for swearing. A bare "bad", "annoyed"
or "problem" is deliberately NOT on any list: the point is to catch anger, not
to police tone, and a false positive costs a slower, gentler turn with a human
offered, which is never a harmful outcome.

REASONED, not measured: a native speaker should review all four lists, and the
Hindi list in particular has no counterpart in the module this was ported from
(the original covered Bengali, English and Latin-script transliteration only --
this repository serves Hindi callers too, so Devanagari was written for it here).
"""

from __future__ import annotations

import unicodedata

# ---- English -------------------------------------------------------------
# Substrings of the lower-cased utterance. Every entry is an explicit,
# multi-word statement of anger; none is a single word that has a calm use.
_EN = (
    "i am very angry",
    "i'm very angry",
    "i am so angry",
    "i'm so angry",
    "i am extremely angry",
    "i'm extremely angry",
    "i am really angry",
    "i'm really angry",
    "i am angry",
    "i'm angry",
    "i am furious",
    "i'm furious",
    "i am absolutely furious",
    "i am fed up",
    "i'm fed up",
    "i am so fed up",
    "i'm so fed up",
    "you people are useless",
    "you guys are useless",
    "your staff are useless",
    "this hospital is useless",
    "this is extremely frustrating",
    "this is so frustrating",
    "this is very frustrating",
    "this is really frustrating",
    "i am extremely frustrated",
    "i'm extremely frustrated",
    "i am so frustrated",
    "i'm so frustrated",
    "i am very frustrated",
    "i'm very frustrated",
    "i have had enough",
    "i've had enough",
    "this is ridiculous",
    "this is absolutely ridiculous",
    "this is completely ridiculous",
    "i am sick of this",
    "i'm sick of this",
    "i am sick and tired of this",
    "i'm sick and tired of this",
    "i am losing my patience",
    "i'm losing my patience",
    "this is infuriating",
    "i am outraged",
    "i'm outraged",
)

# ---- Hindi and Bengali written in Latin script ---------------------------
# The recogniser and callers both produce these as often as the native script.
_LATIN = (
    "mujhe bahut gussa aa raha hai",
    "mujhe gussa aa raha hai",
    "mujhe bahut gussa hai",
    "mujhe bohot gussa aa raha hai",
    "main bahut gussa hoon",
    "main bahut naraz hoon",
    "main bahut pareshan ho gaya hoon",
    "main bahut pareshan ho gayi hoon",
    "main bahut frustrated hoon",
    "yeh bahut frustrating hai",
    "mujhe bahut irritation ho raha hai",
    "bas bahut ho gaya",
    "bahut ho gaya ab",
    "bahut ho chuka hai ab",
    "yeh bilkul bakwas hai",
    "aap log bekar hain",
    "aap log nikamme hain",
    "mera sabr khatam ho gaya hai",
    "ami khub raag korchi",
    "amar khub raag hocche",
    "amar khub raag uthche",
    "ami onek frustrated",
    "ami khub frustrated",
    "eta khub frustrating",
    "eta ekdom frustrating",
    "ami ar shoy korte parchi na",
    "ami r shoy korte parchi na",
    "onek hoyeche ekhon",
    "onek shoyjo korechi ar na",
    "tomra kono kajer na",
    "apnara kono kajer na",
)

# ---- Bengali -------------------------------------------------------------
# Matched as substrings of the NFC-normalised text, which is safe here because
# every entry is multi-word: the embedded space is the word boundary that
# Python's \b cannot provide for a script whose vowel signs are combining
# marks (agent/abuse.py's `_PHRASES` relies on the same property).
_BN = (
    "আমি খুব রাগান্বিত",
    "আমার খুব রাগ হচ্ছে",
    "আমি প্রচণ্ড রেগে আছি",
    "আমি খুব রেগে গেছি",
    "আমার প্রচণ্ড রাগ হচ্ছে",
    "এটা খুবই বিরক্তিকর",
    "এটা একদম বিরক্তিকর",
    "আমি আর সহ্য করতে পারছি না",
    "অনেক হয়েছে এখন",
    "অনেক সহ্য করেছি আর না",
    "আমি খুব বিরক্ত",
    "আমি অত্যন্ত বিরক্ত",
    "আপনারা কোনো কাজের না",
    "তোমরা কোনো কাজের না",
)

# ---- Hindi ---------------------------------------------------------------
# NEW in this repository: the module this was ported from had no Devanagari
# list at all. Same multi-word discipline as the Bengali list above.
_HI = (
    "मुझे बहुत गुस्सा आ रहा है",
    "मुझे गुस्सा आ रहा है",
    "मुझे बहुत गुस्सा है",
    "मैं बहुत गुस्सा हूँ",
    "मैं बहुत गुस्से में हूँ",
    "मैं बहुत नाराज़ हूँ",
    "मैं बहुत नाराज हूँ",
    "मैं बहुत परेशान हो गया हूँ",
    "मैं बहुत परेशान हो गई हूँ",
    "यह बहुत परेशान करने वाला है",
    "बस बहुत हो गया",
    "बहुत हो गया अब",
    "बहुत हो चुका है अब",
    "यह बिलकुल बकवास है",
    "यह बिल्कुल बकवास है",
    "आप लोग बेकार हैं",
    "आप लोग निकम्मे हैं",
    "मेरा सब्र खत्म हो गया है",
    "मेरा सब्र ख़त्म हो गया है",
    "मैं और सहन नहीं कर सकता",
    "मैं और सहन नहीं कर सकती",
)

_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("en", _EN),
    ("latin", _LATIN),
    ("bn", _BN),
    ("hi", _HI),
)


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


def detect_anger(text: str) -> tuple[bool, str | None]:
    """True, plus which phrase family matched, when `text` is a caller saying
    they are angry or fed up.

    The returned tag ("en" | "latin" | "bn" | "hi") says which list matched and
    is for logging only -- it is NOT the language to reply in. The reply
    language is the sticky one the orchestrator already resolved for the turn,
    exactly as agent/abuse.py's call site treats its own match.
    """
    t = _normalise(text)
    if not t.strip():
        return False, None
    for tag, phrases in _FAMILIES:
        if any(p in t for p in phrases):
            return True, tag
    return False, None


def is_anger(text: str) -> bool:
    """Bare boolean form, matching agent/abuse.py::is_abusive()'s shape at the
    one call site that uses it."""
    matched, _ = detect_anger(text)
    return matched
