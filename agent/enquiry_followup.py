"""KCD-396: pronoun/elliptical follow-up resolution for the stateless
enquiry intents (test_rate, test_prep, doctor_availability, clinic_faq).

Deterministic, no LLM call -- same reasoning as agent/booking_flow.py's
classify_yes_no and agent/language_switch.py's detect_language_switch_request:
zero added latency, and CLAUDE.md's truth boundary keeps "which entity is
the caller referring to" out of the model's hands wherever a cheap
substring check can decide it just as well. The model still extracts
slots normally; this module only fires when the model left the intent's
primary slot empty AND the caller's own words contain a pronoun/ellipsis
cue, so a caller who simply forgot to name a test still gets the
ordinary missing_slot_prompt ask, not a wrong guess.

Markers favour multi-character phrases over single short words on
purpose -- a one-syllable marker ("তার", "उसे") is exactly the kind of
short substring that misfired in classify_yes_no's own "cancel" bug
(agent/booking_flow.py), so this module accepts a narrower recall for a
much lower false-positive rate.
"""

from __future__ import annotations

from dataclasses import dataclass

# Which slot each stateless enquiry intent resolves its "subject" from.
# department_query has no such slot (symptom_description is free text,
# not a lookup key) -- it is not coreference-resolvable and is left out.
PRIMARY_SLOT = {
    "test_rate": "test_name",
    "test_prep": "test_name",
    "doctor_availability": "doctor_name",
    "clinic_faq": "faq_topic",
}

# The stateless, single tool-call intents KCD-395's second-question
# handling is allowed to answer automatically in the same turn. Booking
# actions are deliberately excluded -- they are stateful, multi-turn, and
# "two questions in one breath" only ever means two FACTUAL questions in
# this story's own acceptance criteria, never a second action.
ENQUIRY_INTENTS = frozenset(PRIMARY_SLOT) | {"department_query"}

_PRONOUN_MARKERS: dict[str, tuple[str, ...]] = {
    "bn": ("ওটা", "ওটার", "সেটা", "সেটার", "এটার", "তার জন্য", "সেটার জন্য", "ওটার জন্য", "এর জন্য", "একই টেস্ট", "একই জিনিস"),
    "hi": ("उसका", "उसकी", "उसके", "उसके लिए", "इसका", "इसकी", "इसके लिए", "वही टेस्ट", "वही चीज़", "उसी के लिए"),
    "en": ("it", "that one", "the same test", "the same one", "for that", "about that", "its ", " its"),
}


# How many turns back the topic still counts as "what we were just talking about". REASONED, not measured: a
# follow-up to the last answer is almost always the very next turn or the one after; past this, a bare question with no
# test named is asked about, not assumed.
FOLLOWUP_MAX_GAP = 3

# Words that can only POINT at a test or doctor, never name one: "this", "that", "the same", the classifier suffixes
# Bengali attaches to them, and the ASR's spellings of them (the recogniser wrote "same" as "সমে"). A slot value made of
# nothing but these (plus the words for "test" / "doctor") is a reference, not a name -- it must never be sent to the
# clinic lookup as if it were one ("'সমে টেস্ট' নামে কোনো টেস্ট নেই", heard on the live pod).
_POINTER_WORDS = frozenset(
    {
        # Bengali
        "এই",
        "ওই",
        "ঐ",
        "সেই",
        "এটা",
        "ওটা",
        "সেটা",
        "এটি",
        "ওটি",
        "সেটি",
        "এটার",
        "ওটার",
        "সেটার",
        "এইটা",
        "এইটার",
        "ওইটা",
        "একই",
        "সেম",
        "সমে",
        "সেমে",
        "সেইম",
        "টা",
        "টি",
        "টাই",
        "টাও",
        "ই",
        "তো",
        # Hindi
        "यह",
        "वह",
        "यही",
        "वही",
        "इस",
        "उस",
        "इसी",
        "उसी",
        "ये",
        "वो",
        "सेम",
        "समे",
        "ही",
        # English
        "same",
        "this",
        "that",
        "the",
        "one",
        "it",
        "ones",
        "these",
        "those",
        "such",
    }
)
_NOUN_WORDS = frozenset(
    {
        "টেস্ট",
        "টেস্টের",
        "টেস্টটা",
        "টেস্টটি",
        "টেস্টগুলো",
        "ডাক্তার",
        "ডাক্তারের",
        "ডাঃ",
        "ডক্টর",
        "टेस्ट",
        "जांच",
        "जाँच",
        "डॉक्टर",
        "डॉ",
        "test",
        "tests",
        "doctor",
        "dr",
        "doc",
    }
)


def recent_unique(last_entities: list, kind: str, turns_since_last: int | None) -> str | None:
    """The one entity of `kind` that the last few turns were about, or None (nothing recent, or two candidates --
    an ambiguous reference asks, it never guesses). resolve_followup_slot applies the same two rules."""
    if turns_since_last is None or turns_since_last > FOLLOWUP_MAX_GAP:
        return None
    matches = [e for e in last_entities if e.kind == kind]
    return matches[0].value if len(matches) == 1 else None


def is_reference_only(value: str | None, lang: str = "bn") -> bool:
    """True when `value` is made only of pointing words and the word for "test"/"doctor" -- i.e. the caller pointed
    at something and named nothing. An empty value is not a reference (nothing was said)."""
    tokens = [t for t in (value or "").lower().replace("।", " ").replace("?", " ").replace(",", " ").split() if t]
    return bool(tokens) and all(t in _POINTER_WORDS or t in _NOUN_WORDS for t in tokens)


def has_pronoun_reference(text: str, lang: str) -> bool:
    t = text.strip().lower()
    markers = _PRONOUN_MARKERS.get(lang, _PRONOUN_MARKERS["en"])
    return any(m in t for m in markers)


@dataclass(frozen=True)
class EnquiryEntity:
    kind: str  # one of PRIMARY_SLOT's values: "test_name" | "doctor_name" | "faq_topic"
    value: str


def resolve_followup_slot(
    intent: str,
    slots: dict,
    text: str,
    lang: str,
    last_entities: list[EnquiryEntity],
    turns_since_last: int | None = None,
) -> tuple[dict, bool]:
    """If `intent`'s primary slot is empty, the caller's turn reads as a
    pronoun/elliptical follow-up, and exactly ONE remembered entity of
    the right kind exists, fill the slot from it. Returns (slots,
    resolved) -- `slots` is only mutated (a shallow copy) when resolved
    is True, so a caller can tell "nothing to do" apart from "resolved".

    Two or more candidates of the right kind (e.g. the previous turn
    itself asked about two different tests via KCD-395) is treated the
    same as zero: ambiguous reference must ask, never guess -- the
    story's own acceptance criterion."""
    needed_kind = PRIMARY_SLOT.get(intent)
    if needed_kind is None:
        return slots, False
    value = slots.get(needed_kind)
    pointed = bool(value) and is_reference_only(value, lang)  # "the same test": a reference, not a name
    if value and not pointed:
        return slots, False  # the caller named something: use it
    if not last_entities:
        return slots, False
    # When is an empty (or pointing-only) slot filled from what we were just talking about?
    #   * the caller pointed ("the same test", "ওটার জন্য"), at any distance; or
    #   * they asked a test or doctor question with nothing named and the topic was one of the last few turns
    #     ("do I need to fast?" straight after a price for one test). Only for tests and doctors: a FAQ question with
    #     no topic is not a follow-up, it is a question we did not understand. The reply always NAMES what it answers
    #     about, so a wrong assumption is heard and can be corrected.
    recent = (
        needed_kind in ("test_name", "doctor_name")
        and turns_since_last is not None
        and turns_since_last <= FOLLOWUP_MAX_GAP
    )
    if not (pointed or has_pronoun_reference(text, lang) or recent):
        return slots, False

    matches = [e for e in last_entities if e.kind == needed_kind]
    if len(matches) != 1:
        return slots, False

    resolved = dict(slots)
    resolved[needed_kind] = matches[0].value
    return resolved, True


def entities_from_turn(*intent_slot_pairs: tuple[str, dict]) -> list[EnquiryEntity]:
    """Builds the "what was this turn about" memory for the NEXT turn's
    coreference resolution, from one or two (intent, slots) pairs (a
    plain single-question turn, or a KCD-395 primary+secondary pair).
    Call with the SAME slots actually used to answer -- i.e. after any
    resolve_followup_slot substitution has already been applied -- so a
    resolved pronoun becomes a real, named entity for the turn after."""
    out: list[EnquiryEntity] = []
    for intent, slots in intent_slot_pairs:
        kind = PRIMARY_SLOT.get(intent)
        if kind and slots.get(kind):
            entity = EnquiryEntity(kind=kind, value=slots[kind])
            if entity not in out:
                out.append(entity)
    return out
