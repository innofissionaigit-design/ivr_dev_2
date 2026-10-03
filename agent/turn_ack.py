"""KCD-513: acknowledge before answering.

A caller who asks a question and gets the answer with no preamble hears a
machine reading a table. A brief acknowledgement first -- "Sure." "ঠিক আছে।"
"ठीक है।" -- shows the request was heard, and it costs nothing if it is one
spoken word. Two constraints keep it honest:

  * It CANNOT INTRODUCE A FACT. Every entry is a fixed string with no digit, no
    placeholder, no name, at most three words (validate_table() and the tests
    enforce this on every entry). The acknowledgement is chosen by this module
    from a table; a model never composes it. It says "heard", never "I will" or
    "it is" -- an acknowledgement that promised something would be a claim.

  * It is SUPPRESSED on repeat turns, so it does not become a verbal tic. It is
    not spoken on consecutive replies, not in front of an apology or another
    acknowledgement (the distress acknowledgement of agent/acknowledgement.py
    already opened that turn), not before a one-word reply, and its variant
    rotates so the same word is not heard turn after turn.

Wording is provisional and pending native review (agent/persona.py).
"""

from __future__ import annotations

import re

from agent.apology import count_apologies

# lang -> variants, rotated
_ACK = {
    "bn": ["ঠিক আছে।", "বুঝেছি।", "আচ্ছা।"],
    "hi": ["ठीक है।", "समझ गई।", "जी।"],
    "en": ["Sure.", "Understood.", "Alright."],
}
MAX_ACK_WORDS = 3
MIN_TURNS_BETWEEN = 2  # an acknowledgement at most every other reply
MIN_REPLY_WORDS = 3  # a one- or two-word reply needs no preamble

_LEADING_ACK = {
    lang: re.compile("^\\s*(" + "|".join(re.escape(a.rstrip("।.")) for a in variants) + ")\\b", re.I)
    for lang, variants in _ACK.items()
}


def validate_table() -> list[str]:
    """Problems with the acknowledgement table; empty means every entry is safe."""
    problems = []
    for lang, variants in _ACK.items():
        if not variants:
            problems.append(f"{lang}: no variants")
        for a in variants:
            if re.search(r"\d|[{}\[\]:]", a):
                problems.append(f"{lang}: {a!r} carries a digit, placeholder or punctuation artefact")
            if len(a.split()) > MAX_ACK_WORDS:
                problems.append(f"{lang}: {a!r} is longer than {MAX_ACK_WORDS} words")
            if "?" in a or "؟" in a:
                problems.append(f"{lang}: {a!r} is a question")
            if count_apologies(a, lang):
                problems.append(f"{lang}: {a!r} is an apology")
    return problems


# KCD-513, as the operator now asks (2026-09-25): a bare "Thank you" -- and ONLY when the caller has just answered a
# question the agent asked (a name, a number, a date). An answer to the caller's own question (a price, a preparation
# rule) opens with the answer, not with thanks: "thank you for telling me" after "how much is the test" was heard as
# nonsense. The wording still comes from agent/messages.py so it can be changed from the database.
THANKS = {
    "bn": "ধন্যবাদ।",
    "hi": "धन्यवाद।",
    "en": "Thank you.",
}
MAX_THANKS_WORDS = 8


def thanks_for(lang: str) -> str:
    from agent import messages

    return messages.text("thanks_ack", lang, THANKS.get(lang) or THANKS["bn"])


def validate_thanks() -> list[str]:
    problems = []
    for lang, t in THANKS.items():
        if re.search(r"\d|[{}\[\]:]", t) or "?" in t or count_apologies(t, lang):
            problems.append(f"{lang}: {t!r} carries a digit, placeholder, question or apology")
        if len(t.split()) > MAX_THANKS_WORDS:
            problems.append(f"{lang}: {t!r} is longer than {MAX_THANKS_WORDS} words")
    return problems


class AckTracker:
    """One per call. mode "always": every reply passed to decorate() opens with the thanks; mode "varied": short
    rotating acknowledgements, suppressed on repeat turns. main.py no longer decorates enquiry answers with it
    (see THANKS above); the thanks is spoken on its own, after a caller's answer to a question."""

    def __init__(self, mode: str = "varied"):
        self.mode = mode
        self.turn = 0
        self._last_ack_turn = -MIN_TURNS_BETWEEN
        self._variant = 0
        self.spoken = 0
        self.suppressed = 0

    def next_turn(self) -> None:
        self.turn += 1

    def decorate(
        self, reply: str, lang: str, substantive: bool = True, already_acknowledged: bool = False
    ) -> tuple[str, bool]:
        """(reply, whether an acknowledgement was prefixed)."""
        if self.mode == "always":
            skip = (
                not substantive
                or already_acknowledged
                or len((reply or "").split()) < MIN_REPLY_WORDS
                or count_apologies((reply or "")[:60], lang) > 0
            )
            if skip:
                self.suppressed += 1
                return reply, False
            self.spoken += 1
            return f"{thanks_for(lang)} {reply}", True
        table = _ACK.get(lang) or _ACK["bn"]
        skip = (
            not substantive
            or already_acknowledged
            or len((reply or "").split()) < MIN_REPLY_WORDS
            or self.turn - self._last_ack_turn < MIN_TURNS_BETWEEN
            or count_apologies((reply or "")[:60], lang) > 0  # an apology opens this reply
            or bool(_LEADING_ACK.get(lang, _LEADING_ACK["bn"]).match(reply or ""))
        )
        if skip:
            self.suppressed += 1
            return reply, False
        ack = table[self._variant % len(table)]
        self._variant += 1
        self._last_ack_turn = self.turn
        self.spoken += 1
        return f"{ack} {reply}", True
