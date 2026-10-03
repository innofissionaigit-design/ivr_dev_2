"""Abuse and slang from the caller: stay calm, do not mirror it, do not be misread as a request, and do not drift into
another language.

A live call (2026-09-25): the caller swore at the agent. The model heard "how are you?" ("আপনি কেমন আছেন?"), the next
turn was re-asked in HINDI, and nothing set a limit. What a calm, capable assistant does instead:

  1st time   a short, unruffled boundary and an offer to carry on: no lecture, no apology, no imitation;
  2nd time   the same boundary, plainer;
  3rd time   a courteous close of the call ("call again when you like").

The wording is fixed text (agent/phrases.py: abuse_first / abuse_second / abuse_final) in the caller's own language --
the model composes nothing here, so it can neither echo the abuse nor answer it. Detection is by a SMALL list of
unambiguous vulgarities, matched as whole words in any of the three languages and in the common Latin spellings
(callers, and the recogniser, write "madarchod" as often as "মাদারচোদ"). A word that also has an innocent use is not on
the list ("বালক", "বালি", "साला" are not matched); mild insults are left out on purpose -- the point is to catch
swearing, not to police tone. REASONED, not measured: a native speaker should review the list. A transcript that is only
a misrecognition of an ordinary word costs one polite sentence, never a hang-up (the close needs three in one call).
"""

from __future__ import annotations

import re
import unicodedata

# Whole words (after normalisation). Inflected Bengali forms are listed rather than matched by prefix, so
# "বালক" (boy) and "বালি" (sand) are never caught by "বাল".
_WORDS = frozenset(
    [
        "বাল",
        "বালের",
        "বালটা",
        "বালে",
        "চুদি",
        "চুদ",
        "চুদবো",
        "চুদে",
        "চোদা",
        "চোদন",
        "চোদ",
        "খানকি",
        "খানকির",
        "খানকিরপো",
        "মাগি",
        "মাগী",
        "মাগির",
        "মাগীর",
        "হারামি",
        "হারামজাদা",
        "হারামজাদি",
        "হারামজাদী",
        "শুয়োর",
        "শুওর",
        "শুয়োরের",
        "কুত্তা",
        "কুত্তার",
        "কুত্তি",
        "মাদারচোদ",
        "মাদারচুদ",
        "বাঞ্চোদ",
        "বোকাচোদা",
        "বোকাচুদা",
        "গান্ডু",
        "গাঁড়",
        "গাঁড়ের",
        "গাড়",
        "লেওড়া",
        "भोसड़ी",
        "भोसडी",
        "भोसड़ीके",
        "भोसडीके",
        "मादरचोद",
        "मादरचोद",
        "बहनचोद",
        "बहनचोद",
        "चूतिया",
        "चूत",
        "लौड़ा",
        "लौडा",
        "लंड",
        "गांडू",
        "गांड",
        "हरामी",
        "हरामज़ादा",
        "हरामजादा",
        "रंडी",
        "कुत्ते",
        "कुतिया",
        "fuck",
        "fucking",
        "fucker",
        "fuckers",
        "motherfucker",
        "bastard",
        "bitch",
        "bitches",
        "asshole",
        "dickhead",
        "cunt",
        "slut",
        "whore",
        "madarchod",
        "madarchodh",
        "bhenchod",
        "behenchod",
        "bhosdi",
        "bhosdike",
        "bhosadike",
        "chutiya",
        "chutiye",
        "gandu",
        "randi",
        "harami",
        "haramzada",
        "haramzade",
    ]
)
# multi-word abuse ("কুত্তার বাচ্চা"): a phrase is matched as a substring of the normalised text
_PHRASES = (
    "কুত্তার বাচ্চা",
    "কুকুরের বাচ্চা",
    "শুয়োরের বাচ্চা",
    "कुत्ते का बच्चा",
    "कुत्ते की औलाद",
    "son of a bitch",
)


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").replace("़", "").lower()


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s।,.!?;:\"'()\[\]-]+", _normalise(text)) if t]


def is_abusive(text: str) -> bool:
    t = _normalise(text)
    if not t.strip():
        return False
    if any(p in t for p in _PHRASES):
        return True
    return any(tok in _WORDS for tok in _tokens(t))


def response_key(count: int) -> str:
    """Which fixed line answers the `count`-th abusive turn of this call (1-based): the boundary, a plainer boundary,
    then the courteous close."""
    return "abuse_first" if count <= 1 else "abuse_second" if count == 2 else "abuse_final"


def closes_the_call(count: int) -> bool:
    return count >= 3
