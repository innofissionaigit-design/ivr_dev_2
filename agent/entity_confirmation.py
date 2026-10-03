"""Read the entity back when the recogniser could not vouch for it.

The rule from the "no guessing" review: an uncertain speech recognition result must never be
turned into a fact. `agent/confidence_gate.py` sorts a turn into VERIFIED, LOW or UNAVAILABLE:

  LOW          both decoders ran and disagreed        -> the factual lookup does not run (KCD-442)
  UNAVAILABLE  only one decoder produced the text, so there was nothing to compare it with
               -> the lookup does not run YET: the agent asks "do you mean <what I heard>?" and
                  runs it only after a yes. This module holds the question and the choice of WHAT
                  to read back.

The entity is whichever slot decides which record the answer comes from: the test, the doctor, a
confirmation number or a phone number. A turn whose intent has no such slot (an FAQ topic, a
department) is not read back -- see `entity_to_confirm` -- and that is a stated residual: those
answers are clinic-wide static facts and a wrong topic is bounded by the fast-path and model floors,
but it is not the same guarantee.

Wording is provisional and pending native review (agent/persona.py).
"""

from __future__ import annotations

import dataclasses

# Which slot decides the record, per read-only intent, most specific first.
ENTITY_SLOTS = {
    "test_rate": ("test_name",),
    "test_prep": ("test_name",),
    "doctor_availability": ("doctor_name",),
    "lookup_booking": ("confirmation_id", "phone"),
    "resend_confirmation": ("confirmation_id", "phone"),
}
_NUMERIC = frozenset({"confirmation_id", "phone"})

CONFIRM_NAME = {
    "bn": "আপনি কি {value}-এর কথা বলছেন?",
    "hi": "क्या आप {value} के बारे में पूछ रहे हैं?",
    "en": "Do you mean {value}?",
}
CONFIRM_NUMBER = {
    "bn": "আপনি কি {value} নম্বরটা বললেন?",
    "hi": "क्या आपने {value} नंबर बताया?",
    "en": "Did you say the number {value}?",
}
REASK = {
    "bn": "ঠিক আছে। আবার একবার বলবেন?",
    "hi": "ठीक है। क्या आप फिर से बताएँगे?",
    "en": "All right. Could you say it again?",
}


@dataclasses.dataclass
class PendingEntity:
    """A lookup held back until the caller confirms what was heard."""

    intent: str
    slot: str
    value: str
    data: dict  # the whole intent-extraction result, replayed on a yes


def entity_to_confirm(intent: str, slots: dict) -> tuple[str, str] | None:
    for slot in ENTITY_SLOTS.get(intent, ()):
        value = slots.get(slot)
        if value not in (None, ""):
            return slot, str(value)
    return None


def confirm_question(slot: str, value: str, lang: str) -> str:
    table = CONFIRM_NUMBER if slot in _NUMERIC else CONFIRM_NAME
    return (table.get(lang) or table["bn"]).format(value=value)


def reask(lang: str) -> str:
    return REASK.get(lang) or REASK["bn"]
