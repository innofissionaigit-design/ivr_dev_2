"""Kinder, slower service for a senior citizen (KCD-512).

The age comes from the REGISTERED date of birth, computed on the server after the caller passed the
security questions (clinic-api/registry.py returns `age_years` and `is_senior`, 60 and over, which is
India's definition of a senior citizen). It is NOT inferred from the voice: agent/senior_voice.py's
acoustic cues stay a weak hint that never overrides this. Once the registry says senior, the call
switches to senior care for its remaining turns:

  * the SPEECH POLICY row (agent/speech_policy.py): slower rate, one question at a time, longer
    pauses -- set through `apply_caller_state(senior=True)`;
  * a WARM OPENING, once, right after the check: thank them, tell them there is no hurry;
  * a WARM CLOSING on some answers, so the caller is invited to ask more without being pressed;
  * more PATIENCE when a turn fails: a re-ask carries "there is no hurry" too;
  * the same respectful "you" (आप / আপনি) as everyone -- kindness is warmth and patience, never
    baby talk or a change of register (agent/persona.py still applies).

`KindnessPlanner` decides WHICH of these a given reply gets, deterministically, so it is warm without
being smothering: the opening once, the closing on every second substantive reply, the patience line
on a re-ask, and nothing on a one-word reply. It adds no fact and no advice, and every line is a
fixed string with no digit or name. Wording is provisional and pending native review.
"""

from __future__ import annotations

import dataclasses

SENIOR_AGE = 60

OPENING = {
    "bn": "ধন্যবাদ। আপনি আরাম করে বলুন। কোনও তাড়া নেই।",
    "hi": "धन्यवाद। आप आराम से बोलिए। कोई जल्दी नहीं है।",
    "en": "Thank you. Please take your time. There is no hurry.",
}
CLOSING = {
    "bn": "আর কিছু জানার থাকলে বলবেন। আমি আছি।",
    "hi": "और कुछ जानना हो तो बताइए। मैं यहीं हूँ।",
    "en": "Tell me if you need anything else. I am here.",
}
PATIENCE = {
    "bn": "কোনও তাড়া নেই। ধীরে ধীরে বলুন।",
    "hi": "कोई जल्दी नहीं है। धीरे-धीरे बोलिए।",
    "en": "There is no hurry. Please go slowly.",
}
# a warmer acknowledgement used in place of the plain one in senior mode
WARM_ACK = {
    "bn": "জি, বলার জন্য ধন্যবাদ।",
    "hi": "जी, बताने के लिए धन्यवाद।",
    "en": "Yes, thank you for telling me.",
}
MIN_WORDS_FOR_CLOSING = 4
CLOSING_EVERY = 2  # a closing on every second substantive reply


def is_senior(age_years: int | None) -> bool:
    return age_years is not None and age_years >= SENIOR_AGE


def _t(table: dict, lang: str) -> str:
    return table.get(lang) or table["bn"]


@dataclasses.dataclass
class KindnessPlanner:
    """One per call. Inactive until the registry says the caller is a senior."""

    active: bool = False
    opened: bool = False
    substantive_replies: int = 0
    patience_given: int = 0

    def activate(self, age_years: int | None) -> bool:
        """Turn senior care on from a verified age. Returns True if it was newly turned on."""
        was = self.active
        self.active = self.active or is_senior(age_years)
        return self.active and not was

    def opening(self, lang: str) -> str | None:
        """Said once, straight after the security check passes."""
        if not self.active or self.opened:
            return None
        self.opened = True
        return _t(OPENING, lang)

    def patience(self, lang: str) -> str | None:
        """Added to a re-ask when a senior's turn could not be understood."""
        if not self.active:
            return None
        self.patience_given += 1
        return _t(PATIENCE, lang)

    def warm_ack(self, lang: str) -> str | None:
        return _t(WARM_ACK, lang) if self.active else None

    def decorate(self, reply: str, lang: str, substantive: bool = True) -> str:
        """The reply, with the warm closing added when it is due."""
        if not self.active or not substantive or len((reply or "").split()) < MIN_WORDS_FOR_CLOSING:
            return reply
        self.substantive_replies += 1
        if self.substantive_replies % CLOSING_EVERY == 0:
            return f"{reply} {_t(CLOSING, lang)}"
        return reply
