"""Ask for several booking details in one question, except when the caller's state says one at a time (KCD-103).

THE PROBLEM. `missing_slot_prompt` asked for exactly one field per turn, always. A caller who says "I want to book
Dr. Sen" was then asked the day, then the time, then the name, then the phone number: five turns and four "who
are you calling for" pauses to book one appointment. Callers who know all of it were interrogated.

THE RULE. Ask the next GROUP of missing fields in one question (the day and the time together; the name and the
number together), and accept any subset back -- the merge in agent/booking_flow.py takes whatever the extractor
found and the next turn asks for what is still missing. Two things always win over grouping:

  * The caller-state table (agent/speech_policy.py). A senior caller, a distressed caller and an emergency
    are held to `questions_per_turn` = 1 or 0 ("one question, listen, confirm, next"), and this module returns
    exactly one field for them, whatever the group would have been. Grouping is a convenience for a caller who
    can take it; it is never a reason to load one who cannot.
  * A field already asked once. If the caller did not answer a grouped question fully, the leftover fields are
    asked one at a time: repeating the same long question at someone who has just shown it did not land is the
    fastest way to lose them.

A grouped prompt is ONE question sentence (one question mark), so it satisfies `check_reply` for the normal policy,
which allows two per turn. Wording per language is in the tables below; the fragments are joined into a natural
clause, never a "label: value" pair (KCD-454). Adding a group or a language is data.

REASONED, NOT MEASURED: which fields belong together is a judgement about what a caller knows at once (a day and a
time; their own name and number), not a measurement; "median turns to complete a booking falls" is shown on
scripted callers in tests/test_slot_grouping.py, not on real ones.
"""

from __future__ import annotations

from typing import TypedDict

# Fields asked together, in the order they are asked. A group is only ever asked from the front: the first missing
# field decides which group is in play, and only that group's still-missing fields are asked.
GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "book_appointment": (
        ("doctor_name", "date", "time_slot"),
        ("patient_name", "phone"),
    ),
    "book_test": (("test_names", "date"), ("patient_name", "phone")),
}

# No question ever carries more fields than this, whatever the tables say.
MAX_FIELDS_PER_QUESTION = 3

# One frame per (action-group, language): `frame` has `{}` where the joined fragments go, `f` maps a field to the words that
# ask for it. (Functional TypedDict syntax because "and" is not a valid attribute name.)
_Frame = TypedDict("_Frame", {"frame": str, "and": str, "comma": str, "f": dict[str, str]})

_FRAMES: dict[tuple[str, str], _Frame] = {
    ("schedule", "en"): {
        "frame": "For the appointment, {}?",
        "and": " and ",
        "comma": ", ",
        "f": {
            "doctor_name": "which doctor",
            "date": "which day",
            "time_slot": "what time",
        },
    },
    ("schedule", "hi"): {
        "frame": "अपॉइंटमेंट के लिए {} बताइए?",
        "and": " और ",
        "comma": ", ",
        "f": {
            "doctor_name": "कौन से डॉक्टर",
            "date": "कौन सा दिन",
            "time_slot": "कौन सा समय",
        },
    },
    ("schedule", "bn"): {
        "frame": "অ্যাপয়েন্টমেন্টের জন্য {} বলবেন?",
        "and": " আর ",
        "comma": ", ",
        "f": {"doctor_name": "কোন ডাক্তার", "date": "কোন দিন", "time_slot": "কোন সময়"},
    },
    ("tests", "en"): {
        "frame": "For the test booking, {}?",
        "and": " and ",
        "comma": ", ",
        "f": {"test_names": "which tests", "date": "which day"},
    },
    ("tests", "hi"): {
        "frame": "टेस्ट बुकिंग के लिए {} बताइए?",
        "and": " और ",
        "comma": ", ",
        "f": {"test_names": "कौन से टेस्ट", "date": "कौन सा दिन"},
    },
    ("tests", "bn"): {
        "frame": "টেস্ট বুকিংয়ের জন্য {} বলবেন?",
        "and": " আর ",
        "comma": ", ",
        "f": {"test_names": "কোন টেস্টগুলো", "date": "কোন দিন"},
    },
    ("details", "en"): {
        "frame": "May I have {}?",
        "and": " and ",
        "comma": ", ",
        "f": {
            "patient_name": "the patient's name",
            "phone": "a phone number for the confirmation",
        },
    },
    ("details", "hi"): {
        "frame": "{} बताइए?",
        "and": " और ",
        "comma": ", ",
        "f": {"patient_name": "मरीज़ का नाम", "phone": "कन्फ़र्मेशन के लिए एक फ़ोन नंबर"},
    },
    ("details", "bn"): {
        "frame": "{} বলবেন?",
        "and": " আর ",
        "comma": ", ",
        "f": {"patient_name": "রোগীর নাম", "phone": "কনফার্মেশনের জন্য একটা ফোন নম্বর"},
    },
}


def _kind(fields: tuple[str, ...]) -> str | None:
    if all(f in ("doctor_name", "date", "time_slot") for f in fields):
        return "schedule"
    if all(f in ("test_names", "date") for f in fields):
        return "tests"
    if all(f in ("patient_name", "phone") for f in fields):
        return "details"
    return None


def next_fields(
    action: str,
    missing: list[str],
    questions_per_turn: int,
    asked: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """The fields the next question asks for: one, or a group. `missing` is what is still needed, in the order
    booking_flow.missing_required lists it. Always at least one field when anything is missing."""
    if not missing:
        return []
    first = missing[0]
    if questions_per_turn <= 1:
        return [first]  # the caller-state table always wins
    for group in GROUPS.get(action, ()):
        if first in group:
            wanted = [f for f in group if f in missing][:MAX_FIELDS_PER_QUESTION]
            if len(wanted) > 1 and not any(f in asked for f in wanted):
                return wanted
            return [first]
    return [first]


def grouped_prompt(fields: list[str], lang: str) -> str | None:
    """One natural question covering `fields`, or None if there is no wording for that combination (the caller then
    asks the first field alone, as before)."""
    if len(fields) < 2:
        return None
    kind = _kind(tuple(fields))
    table = _FRAMES.get((kind, lang)) if kind else None
    if table is None or any(f not in table["f"] for f in fields):
        return None
    parts = [table["f"][f] for f in fields]
    joined = parts[0] if len(parts) == 1 else table["comma"].join(parts[:-1]) + table["and"] + parts[-1]
    return table["frame"].format(joined)
