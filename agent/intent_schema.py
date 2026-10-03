"""The shape of what the intent model may say, as a declared schema (pydantic) instead of hand-written dict checks.

The model is an untrusted, probabilistic dependency (CLAUDE.md rule 1): whatever it returns is parsed here, once, into
typed objects, and only those objects are used downstream. What the schema does:

  * `intent` must be one of a fixed set, and `slots` must be an object -- otherwise the answer is rejected outright
    (`parse_extraction` returns no model) and the caller retries within its deadline.
  * every slot is typed and NORMALISED, never trusted: a bare string where a list was meant is wrapped, a spelled name is
    split into letters, an age must be a whole number in a human range, a phone given as a number is made a string, an
    invented FAQ topic becomes null. A wrong-typed value is dropped to null, never guessed at.
  * only a smalltalk intent may carry a model-written reply (`direct_reply_bn`); on any other intent it is stripped, because
    the model never gets to state a fact.
  * a second question is kept only when it is one of the stateless enquiry intents and carries slots.
  * a slot the model omitted is not fatal (it means "not said") but is reported, as before.

`agent/llm.py` builds the prompt and calls the model; this module decides what an answer is allowed to be.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from agent.enquiry_followup import ENQUIRY_INTENTS

Intent = Literal[
    "test_rate",
    "doctor_availability",
    "book_appointment",
    "test_prep",
    "clinic_faq",
    "smalltalk",
    "unclear",
    "book_test",
    "reschedule_appointment",
    "cancel_appointment",
    "lookup_booking",
    "add_test_booking",
    "resend_confirmation",
    "department_query",
]
VALID_INTENTS: frozenset[str] = frozenset(get_args(Intent))

# The FAQ topic keys FastPath.FAQCatalogue matches locally against /api/v1/catalogue's faq_topics.
FAQ_TOPICS: tuple[str, ...] = (
    "hours",
    "location",
    "payment_methods",
    "insurance",
    "parking",
    "report_collection",
    "contact_number",
    "home_collection",
)

SLOT_KEYS: tuple[str, ...] = (
    "test_name",
    "test_names",
    "doctor_name",
    "date",
    "time_slot",
    "new_date",
    "new_time_slot",
    "confirmation_id",
    "patient_name",
    "patient_age",
    "phone",
    "contact_phone",
    "relationship",
    "spelled_letters",
    "symptom_description",
    "faq_topic",
)
_TEXT_SLOTS = tuple(k for k in SLOT_KEYS if k not in ("test_names", "patient_age", "spelled_letters", "faq_topic"))
_MAX_AGE = 130


def normalize_age(value: Any) -> int | None:
    """ "number or null" in the schema; a model may return "72", "seventy" or 7.2e1. Only a whole number in a human range
    survives -- a wrong-typed age must never reach code that compares it."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            return None
    if isinstance(value, int) and 0 < value < _MAX_AGE:
        return value
    return None


class Slots(BaseModel):
    """The slots the model may fill. Every value is optional (null = not said); every value is normalised."""

    model_config = ConfigDict(extra="ignore")

    test_name: str | None = None
    test_names: list[str] | None = None
    doctor_name: str | None = None
    date: str | None = None
    time_slot: str | None = None
    new_date: str | None = None
    new_time_slot: str | None = None
    confirmation_id: str | None = None
    patient_name: str | None = None
    patient_age: int | None = None
    phone: str | None = None
    contact_phone: str | None = None
    relationship: str | None = None
    spelled_letters: list[str] | None = None
    symptom_description: str | None = None
    faq_topic: str | None = None

    @field_validator(*_TEXT_SLOTS, mode="before")
    @classmethod
    def _text(cls, v: Any) -> str | None:
        if isinstance(v, bool) or v is None:
            return None
        if isinstance(v, (int, float)):
            return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
        return v if isinstance(v, str) else None  # an object or a list where text was meant: dropped

    @field_validator("test_names", mode="before")
    @classmethod
    def _test_names(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            return [v]  # one test was meant: never iterated as characters
        return list(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else None

    @field_validator("spelled_letters", mode="before")
    @classmethod
    def _letters(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            return list(v)  # the slot IS the letters: a collapsed string's characters are them
        return list(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else None

    @field_validator("patient_age", mode="before")
    @classmethod
    def _age(cls, v: Any) -> int | None:
        return normalize_age(v)

    @field_validator("faq_topic", mode="before")
    @classmethod
    def _faq(cls, v: Any) -> str | None:
        return v if isinstance(v, str) and v in FAQ_TOPICS else None  # an invented topic key is nulled, not passed on


class IntentExtraction(BaseModel):
    """What one model answer is allowed to be."""

    model_config = ConfigDict(extra="ignore")

    intent: Intent
    slots: Slots
    direct_reply_bn: str | None = None
    secondary_intent: str | None = None
    secondary_slots: Slots | None = None

    @model_validator(mode="after")
    def _guards(self) -> IntentExtraction:
        # the model may write a reply for smalltalk ONLY: anywhere else it would be stating something
        if self.intent != "smalltalk":
            self.direct_reply_bn = None
        # a second question is one of the stateless enquiry intents and has slots; otherwise it is "no second question"
        if self.secondary_intent not in ENQUIRY_INTENTS or self.secondary_slots is None:
            self.secondary_intent, self.secondary_slots = None, None
        return self

    def to_dict(self) -> dict[str, Any]:
        """The plain dict every caller of agent.llm.extract_intent already reads (intent, slots, direct_reply_bn,
        secondary_intent, secondary_slots), so the schema can be introduced without touching them."""
        return {
            "intent": self.intent,
            "slots": self.slots.model_dump(),
            "direct_reply_bn": self.direct_reply_bn,
            "secondary_intent": self.secondary_intent,
            "secondary_slots": self.secondary_slots.model_dump() if self.secondary_slots else None,
        }


def parse_extraction(data: Any, slots_optional: bool = False) -> tuple[IntentExtraction | None, list[str]]:
    """(the parsed answer, problems). The answer is None when it cannot be used at all: not an object, an intent outside the
    fixed set, or `slots` that is not an object. Slots the model left out are reported ("slots.x: missing") but do not
    reject the answer -- a missing slot means "not said"."""
    if not isinstance(data, dict):
        return None, [f"expected a JSON object, got {type(data).__name__}"]
    errors: list[str] = []
    intent = data.get("intent")
    if not isinstance(intent, str) or intent not in VALID_INTENTS:  # (a list is unhashable: never `in` a set)
        errors.append(f"invalid intent: {intent!r}")
    slots = data.get("slots")
    if slots is None and slots_optional:
        slots, data = {}, {**data, "slots": {}}  # the compact answer leaves out an empty "slots": nothing was said
    if not isinstance(slots, dict):
        errors.append("slots: expected object")
    else:
        if not slots_optional:  # the compact answer omits empty slots on purpose
            errors += [f"slots.{k}: missing" for k in SLOT_KEYS if k not in slots]
    if any("invalid intent" in e or "slots: expected" in e for e in errors):
        return None, errors
    secondary_slots = data.get("secondary_slots")
    payload = {**data, "secondary_slots": secondary_slots if isinstance(secondary_slots, dict) else None}
    if not isinstance(payload.get("direct_reply_bn"), str):
        payload["direct_reply_bn"] = None
    if not isinstance(payload.get("secondary_intent"), str):
        payload["secondary_intent"] = None
    return IntentExtraction.model_validate(payload), errors
