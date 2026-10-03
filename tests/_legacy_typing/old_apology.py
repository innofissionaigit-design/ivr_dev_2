"""KCD-514: apologies that are calibrated and never stacked.

A caller who hears "Sorry, I could not hear you. Sorry, sorry, please repeat." is
being made to feel that a small problem is a crisis. Two rules, both enforced:

  1. AT MOST ONE apology per spoken reply. Templates are written to that rule
     and tests/test_persona_and_apology.py fails the build if one is not; but a
     reply is sometimes ASSEMBLED at run time from two templates (a secondary
     answer appended to a primary one, an acknowledgement in front of a failure),
     so enforce_single_apology() runs on every reply just before it is spoken and
     removes the later ones. It only ever removes an apology; it never edits a fact.

  2. The WORDING says what actually went wrong, because an apology for the wrong
     thing is worse than none. Four causes, kept distinct:

        NOT_HEARD           I could not hear you well enough          (audio)
        CANNOT_CHECK        I cannot look that up right now           (a system is down)
        INSUFFICIENT_INFO   I do not have verified information on it  (nothing to say)
        NOT_EXIST           what you named is not one of ours         (a real "not found")

     "Cannot check" and "does not exist" must never be interchangeable: telling a
     caller a doctor does not exist because a service was down is a false
     statement, and the reverse sends them away from a doctor who is there. The
     mapping from the existing phrase keys to a cause is in CAUSE_OF_PHRASE_KEY.

Nothing here contains a fact, a number or a name. The wording is provisional and
pending native review (agent/persona.py REVIEW_STATUS).
"""

from __future__ import annotations

import re

NOT_HEARD = "not_heard"
CANNOT_CHECK = "cannot_check"
INSUFFICIENT_INFO = "insufficient_info"
NOT_EXIST = "not_exist"
CAUSES = (NOT_HEARD, CANNOT_CHECK, INSUFFICIENT_INFO, NOT_EXIST)

# cause -> lang -> the apology sentence (exactly one apology marker each)
_APOLOGY = {
    NOT_HEARD: {
        "bn": "দুঃখিত। আমি ঠিক শুনতে পাইনি।",
        "hi": "माफ़ कीजिए। मैं ठीक से सुन नहीं पाई।",
        "en": "Sorry. I could not hear that.",
    },
    CANNOT_CHECK: {
        "bn": "দুঃখিত। এখন আমি এটা দেখতে পারছি না।",
        "hi": "माफ़ कीजिए। अभी मैं यह देख नहीं पा रही।",
        "en": "Sorry. I cannot check that now.",
    },
    INSUFFICIENT_INFO: {
        "bn": "দুঃখিত। এ বিষয়ে আমার কাছে তথ্য নেই।",
        "hi": "माफ़ कीजिए। इस बारे में मेरे पास जानकारी नहीं है।",
        "en": "Sorry. I do not have that information.",
    },
    NOT_EXIST: {
        "bn": "দুঃখিত। এটা আমাদের এখানে নেই।",
        "hi": "माफ़ कीजिए। यह हमारे यहाँ नहीं है।",
        "en": "Sorry. We do not have that.",
    },
}

# Which cause each existing phrase key speaks to (agent/phrases.py).
CAUSE_OF_PHRASE_KEY = {
    "asr_empty": NOT_HEARD,
    "unclear": NOT_HEARD,
    "tool_failure": CANNOT_CHECK,
    "llm_failure": CANNOT_CHECK,
}

_MARKERS = {
    "bn": re.compile(r"(দুঃখিত|দুঃখিত|ক্ষমা করবেন|ক্ষমা চাইছি|সরি)"),
    "hi": re.compile(r"(माफ़ कीजिए|माफ कीजिए|माफ़ करें|माफ करें|क्षमा|खेद|सॉरी)"),
    "en": re.compile(r"\b(sorry|apologi[sz]e[sd]?|apologies|pardon me|my apologies)\b", re.I),
}
# an apology clause at the START of a sentence: the marker plus its comma
_LEADING = {
    "bn": re.compile(r"^\s*(দুঃখিত|ক্ষমা করবেন|সরি)\s*[,،।.]?\s*"),
    "hi": re.compile(r"^\s*(माफ़ कीजिए|माफ कीजिए|माफ़ करें|माफ करें|क्षमा कीजिए|खेद है)\s*[,،।.]?\s*"),
    "en": re.compile(r"^\s*(sorry|my apologies|apologies|i apologi[sz]e)\s*[,.]?\s*", re.I),
}


def apology_for(cause: str, lang: str) -> str:
    return _APOLOGY[cause].get(lang) or _APOLOGY[cause]["bn"]


def count_apologies(text: str, lang: str) -> int:
    """Apology markers in `text`. One per marker occurrence: "sorry, sorry" is two."""
    return len(_MARKERS.get(lang, _MARKERS["en"]).findall(text or ""))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[।.?!])\s+", text.strip()) if s]


def ends_on_bare_apology(text: str, lang: str) -> bool:
    """A reply whose LAST sentence is nothing but an apology leaves the caller with
    no next step. (KCD-516: no reply ends on a bare apology.)"""
    parts = _sentences(text or "")
    if not parts:
        return False
    last = parts[-1]
    if count_apologies(last, lang) == 0:
        return False
    stripped = _MARKERS.get(lang, _MARKERS["en"]).sub("", last)
    return len(re.sub(r"[\s,।.!?،-]+", "", stripped)) <= 4


def enforce_single_apology(text: str, lang: str) -> tuple[str, int]:
    """(text with at most one apology, how many were removed). The FIRST apology
    is kept; a later sentence that is only an apology is dropped, and a later
    sentence that merely OPENS with one has the apology clause cut off and keeps
    its content. Never touches a sentence with no apology in it."""
    if count_apologies(text, lang) <= 1:
        return text, 0
    out, seen, removed = [], False, 0
    marker, leading = _MARKERS.get(lang, _MARKERS["en"]), _LEADING.get(lang, _LEADING["en"])
    for s in _sentences(text):
        n = len(marker.findall(s))
        if n == 0:
            out.append(s)
            continue
        if not seen:
            seen = True
            if n > 1:  # "sorry, sorry" inside the first sentence
                first = marker.search(s)
                head, tail = s[: first.end()], s[first.end() :]
                tail, extra = marker.subn("", tail)
                removed += extra
                s = (head + tail).strip()
            out.append(s)
            continue
        # a later sentence: remove the apology clause; drop the sentence if nothing is left
        cut = leading.sub("", s, count=1)
        removed += 1
        if cut != s:
            rest = marker.sub("", cut)
            if re.sub(r"[\s,।.!?،-]+", "", rest):
                out.append(rest[0].upper() + rest[1:] if lang == "en" and rest else rest)
        else:
            rest = marker.sub("", s)
            if re.sub(r"[\s,।.!?،-]+", "", rest):
                out.append(rest.strip())
    return " ".join(out), removed
