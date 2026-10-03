"""Cross-turn booking state (Epic E26).

The gap this file exists to close: before this, main.py's dispatch threw
the ENTIRE turn's slots away and re-prompted from scratch whenever one
required slot was missing (see main.py's book_appointment branch history)
-- so a caller who gave doctor, date and time in one sentence but not a
phone number would be asked for the doctor's name again on the very next
turn. That is KCD-357/358/366's "no cross-turn slot memory" repo evidence.

BookingState is attached to a CallSession (main.py) and MERGES each
turn's newly-extracted slots into what earlier turns already gave,
instead of replacing them. It never talks to clinic-api itself and never
touches the network or a database -- that stays in agent/tools_client.py
and main.py, keeping this module pure and unit-testable, per CLAUDE.md's
one-import rule (nothing here imports main.py).

Deliberately NOT using the LLM to decide "was that a yes" -- a caller
confirming a booking is exactly the kind of binary, high-stakes decision
this project's whole architecture exists to keep deterministic (see
CLAUDE.md's truth boundary) and fast (zero extra round trip to Qwen).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------- actions

REQUIRED_SLOTS: dict[str, tuple[str, ...]] = {
    "book_appointment": ("doctor_name", "date", "time_slot", "patient_name", "phone"),
    "book_test": ("test_names", "date", "patient_name", "phone"),
    "reschedule_appointment": ("confirmation_id", "new_date", "new_time_slot"),
    "cancel_appointment": ("confirmation_id",),
    "add_test_booking": ("confirmation_id", "test_name"),
    "lookup_booking": (),  # phone/confirmation_id/name -- any one is enough, checked specially
}

# How long an in-progress booking survives a caller going silent or
# changing subject mid-flow before it is considered abandoned in-call
# (KCD-366 "changes their mind" is a slot CHANGE, not silence -- this is
# the separate case of just never coming back to it this call).
STATE_IDLE_TIMEOUT_S = 180.0


@dataclass
class BookingState:
    action: str
    stage: str = "collecting"  # collecting -> confirming -> awaiting_charge_confirm -> done
    slots: dict = field(default_factory=dict)
    test_names: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.monotonic)
    # Set once a hold has been won (agent/tools_client.hold_booking), so a
    # correction after confirming (KCD-367) can re-enter collection without
    # losing the slot it already holds.
    hold_token: str | None = None
    hold_doctor_id: int | None = None
    pending_charge_inr: int | None = None  # KCD-372's "state the charge before applying it"
    # How many turns in a row missing_required() has asked for each field,
    # without getting it -- drives KCD-368 (offer spelling after one failed
    # name capture) and KCD-370 (proceed without a phone number after a
    # caller declines to give one, rather than blocking the booking
    # forever). Never used to give up on a REQUIRED fact like the doctor
    # or date -- only on the two fields that have a graceful degraded path.
    retry_counts: dict = field(default_factory=dict)
    phone_declined: bool = False
    # KCD-103: fields the agent has already asked for, so a question that did not land is not repeated as the same
    # long group (agent/slot_grouping.py asks a repeated field alone).
    asked_fields: set = field(default_factory=set)

    def touch(self) -> None:
        self.last_updated = time.monotonic()

    def is_stale(self) -> bool:
        return (time.monotonic() - self.last_updated) > STATE_IDLE_TIMEOUT_S

    def note_retry(self, field_name: str) -> int:
        self.retry_counts[field_name] = self.retry_counts.get(field_name, 0) + 1
        return self.retry_counts[field_name]


def new_state(action: str) -> BookingState:
    return BookingState(action=action)


_SLOT_FIELDS = (
    "doctor_name",
    "date",
    "time_slot",
    "patient_name",
    "phone",
    "contact_phone",
    "confirmation_id",
    "new_date",
    "new_time_slot",
    "test_name",
    "relationship",
    "patient_age",
    "symptom_description",
)


def merge_slots(state: BookingState, new_slots: dict) -> list[str]:
    """Folds this turn's non-null slots into the running state. Returns
    the list of field names that actually changed, so main.py can decide
    whether a change after "confirming" needs a fresh readback (KCD-366).

    A later non-null value ALWAYS overwrites an earlier one -- that is
    precisely what lets a caller correct themselves (KCD-366, KCD-367)
    just by saying the new value, with no special "no I meant" phrasing
    required."""
    changed = []
    for key in _SLOT_FIELDS:
        val = new_slots.get(key)
        if val not in (None, "") and state.slots.get(key) != val:
            state.slots[key] = val
            changed.append(key)
    incoming_tests = new_slots.get("test_names")
    if incoming_tests:
        for t in incoming_tests:
            if t not in state.test_names:
                state.test_names.append(t)
                changed.append("test_names")
    state.touch()
    if changed and state.stage in ("confirming", "awaiting_charge_confirm"):
        # A correction after the readback re-opens collection rather than
        # silently committing the old value or throwing away everything
        # else already captured -- KCD-367.
        state.stage = "collecting"
        state.pending_charge_inr = None
    return changed


def missing_required(state: BookingState) -> list[str]:
    required = REQUIRED_SLOTS.get(state.action, ())
    missing = []
    for field_name in required:
        if field_name == "phone" and state.phone_declined:
            # KCD-370: a caller who refuses to give any number still gets
            # the booking, with a stated consequence (no written
            # confirmation) instead of being blocked forever.
            continue
        if field_name == "test_names":
            if not state.test_names:
                missing.append(field_name)
            continue
        if not state.slots.get(field_name):
            missing.append(field_name)
    if state.action == "lookup_booking" and not (state.slots.get("phone") or state.slots.get("confirmation_id")):
        missing.append("phone")
    return missing


def is_ready_to_confirm(state: BookingState) -> bool:
    return not missing_required(state) and state.stage == "collecting"


def effective_phone(state: BookingState) -> str:
    """The number the confirmation actually goes to: a caller-stated
    contact_phone wins (KCD-370's "a different number for the
    confirmation"), otherwise the phone slot, otherwise the sentinel that
    marks a declined booking -- never guessed, never left as an empty
    string that could be silently confused with a real one downstream."""
    return state.slots.get("contact_phone") or state.slots.get("phone") or "not_provided"


def mark_confirming(state: BookingState) -> None:
    state.stage = "confirming"
    state.touch()


def mark_awaiting_charge_confirm(state: BookingState, charge_inr: int) -> None:
    state.stage = "awaiting_charge_confirm"
    state.pending_charge_inr = charge_inr
    state.touch()


def mark_done(state: BookingState) -> None:
    state.stage = "done"
    state.touch()


def reopen_for_correction(state: BookingState, field_name: str, value: str) -> None:
    """KCD-367: a correction between confirmation and commit. Re-enters
    collection for just that field -- everything else captured stays."""
    if value:
        state.slots[field_name] = value
    state.stage = "collecting"
    state.pending_charge_inr = None
    state.touch()


# --------------------------------------------------- deterministic yes/no

_YES_WORDS = {
    "bn": ("হ্যাঁ", "হ্যা", "ঠিক আছে", "ওকে", "কনফার্ম", "হবে", "বুক করুন"),
    "hi": ("हाँ", "हां", "ठीक है", "ओके", "कन्फर्म", "बुक कर दीजिए"),
    "en": ("yes", "yeah", "yep", "confirm", "ok", "okay", "correct", "right", "sure"),
}
# "cancel"/"ক্যানসেল"/"বাতিল"/"कैंसल"/"रद्द" are deliberately NOT here.
# classify_yes_no is only ever asked as an answer to "shall I <action>
# this?" (main.py only calls it during booking.stage in
# ("confirming", "awaiting_charge_confirm")) -- when the pending action
# IS itself a cancellation, an affirmative answer is exactly "yes,
# cancel it", which contains the word "cancel". Treating that word as a
# NO-signal misread a caller's own "yes" as "no" and left their
# cancellation un-actioned -- CodeRabbit-flagged, confirmed by tracing
# this exact call path. A caller who says a bare "cancel" with no other
# word now gets None (asked again) rather than a guessed answer, which
# is the safe direction to be wrong in.
_NO_WORDS = {
    "bn": ("না", "নয়", "ঠিক না"),
    "hi": ("नहीं", "नही", "गलत"),
    "en": ("no", "nope", "wrong", "not correct"),
}

_RE_TOKEN_SPLIT = re.compile(r"[\s,.!?।|]+")


def _tokenize(t: str) -> list[str]:
    return [tok for tok in _RE_TOKEN_SPLIT.split(t) if tok]


def _phrase_starts(tokens: list[str], phrase: str) -> list[int]:
    """Start index of every occurrence of `phrase` in `tokens`. A
    single-word cue matches only a WHOLE token (never a substring -- "ok"
    must not fire on "book", "no" must not fire on "know"). A multi-word
    phrase ("not correct", "ঠিক না") matches a contiguous run of tokens,
    not a raw substring of the sentence."""
    phrase_tokens = phrase.split()
    n = len(phrase_tokens)
    return [i for i in range(len(tokens) - n + 1) if tokens[i : i + n] == phrase_tokens]


def _has_phrase(tokens: list[str], phrase: str) -> bool:
    return bool(_phrase_starts(tokens, phrase))


# KCD-484: "No write proceeds without an affirmative confirmation." A
# yes-word directly after a negator is not an affirmative -- "not sure"
# used to classify as "yes" (via "sure") and would have committed a
# booking write on an explicitly uncertain answer. English only: bn/hi
# negators ("না", "नहीं") are standalone tokens already caught by the
# no-words check that runs first, so they never reach this path.
_NEGATORS_EN = frozenset(
    {
        "not",
        "don't",
        "dont",
        "doesn't",
        "doesnt",
        "isn't",
        "isnt",
        "aren't",
        "arent",
        "wasn't",
        "wasnt",
        "can't",
        "cant",
        "cannot",
        "won't",
        "wont",
        "never",
        "hardly",
    }
)
# A negated hedge is uncertainty, not refusal: re-ask instead of treating
# it as a "no" that reopens slot collection.
_HEDGE_WORDS_EN = frozenset({"sure"})
_NEGATION_LOOKBACK = 2  # "not really sure", "don't think so" style gaps
_UNCERTAIN_WORDS_EN = frozenset({"unsure", "uncertain", "maybe", "perhaps", "dunno"})


def _expresses_uncertainty(tokens: list[str]) -> bool:
    """A hedge word, or a negator followed within a few tokens by one
    ("not sure", "not really sure", "don't feel certain"). Deliberately
    generous: the cost of a false positive is one re-ask, the cost of a false
    negative is a booking written on a maybe."""
    if any(t in _UNCERTAIN_WORDS_EN for t in tokens):
        return True
    hedges = _HEDGE_WORDS_EN | {"certain", "convinced", "confident"}
    for i, tok in enumerate(tokens):
        if tok in _NEGATORS_EN and any(t in hedges for t in tokens[i + 1 : i + 4]):
            return True
    return False


def classify_yes_no(transcript: str, lang: str) -> str | None:
    """Deterministic, no LLM call -- confirming or cancelling a booking is
    exactly the kind of binary, high-stakes decision this project keeps
    out of the model's hands (CLAUDE.md's truth boundary) and off the
    Qwen round trip (this session's zero-latency requirement). Returns
    "yes", "no", or None if neither matched -- callers treat None as
    "unclear", the same as any other unrecognised turn.

    Token-based, not raw substring matching (see _has_phrase) -- a
    substring check let "book it" match "ok" and "I don't know" match
    "no" (via "know"), both CodeRabbit-flagged and confirmed by tracing
    this function against those exact utterances."""
    t = transcript.strip().lower()
    tokens = _tokenize(t)
    yes_words = _YES_WORDS.get(lang, _YES_WORDS["en"])
    no_words = _NO_WORDS.get(lang, _NO_WORDS["en"])
    # "no" is checked first: "not correct" / "ঠিক না" contain "correct"/
    # "ঠিক", which are also yes-leaning words on their own.
    if any(_has_phrase(tokens, w) for w in no_words):
        return "no"

    # Uncertainty ANYWHERE in the answer withholds a "yes": "not sure I should
    # confirm" has its negated hedge more than _NEGATION_LOOKBACK tokens before
    # the yes cue, so the per-cue window below would never see it and a later
    # "confirm" would commit a write on an explicitly uncertain answer.
    if lang == "en" and _expresses_uncertainty(tokens):
        return None

    negated_other = False
    for w in yes_words:
        for start in _phrase_starts(tokens, w):
            window = tokens[max(0, start - _NEGATION_LOOKBACK) : start]
            if lang in _YES_WORDS and lang != "en":
                negated = False
            else:
                negated = any(tok in _NEGATORS_EN for tok in window)
            if not negated:
                return "yes"
            if w not in _HEDGE_WORDS_EN:
                negated_other = True
    return "no" if negated_other else None


# --------------------------------------------------------- spelling capture

_LATIN_LETTER = "abcdefghijklmnopqrstuvwxyz"


def try_assemble_spelling(letters_spoken: list[str]) -> str | None:
    """KCD-368: a caller spells a name letter by letter. `letters_spoken`
    is whatever agent/llm.py's slot extractor pulled out as individual
    letter tokens for THIS turn (it may take more than one turn for a long
    name -- callers keep speaking and each turn's letters are assembled
    onto whatever spelling_buffer main.py is already holding on the
    BookingState, via merge_spelling below). Returns None if nothing
    letter-like was found, so main.py knows to fall back to asking again
    rather than silently accepting an empty name."""
    letters = [c.strip().lower() for c in letters_spoken if c and c.strip()]
    letters = [c for c in letters if len(c) == 1 and c in _LATIN_LETTER]
    if not letters:
        return None
    return "".join(letters)


def merge_spelling(state: BookingState, letters_spoken: list[str]) -> str:
    piece = try_assemble_spelling(letters_spoken) or ""
    current = state.slots.get("_spelling_buffer", "")
    combined = current + piece
    state.slots["_spelling_buffer"] = combined
    state.touch()
    return combined


# ------------------------------------------------------------- corrections

# Which fields KCD-453 acknowledges by name when corrected -- the
# free-text/internal ones (_spelling_buffer, symptom_description) are
# left out because restating them verbatim would be noise, not
# confirmation; a caller correcting one of those hears it through the
# ordinary readback/reply instead.
_CORRECTABLE_FIELDS = (
    "doctor_name",
    "date",
    "time_slot",
    "patient_name",
    "phone",
    "contact_phone",
    "confirmation_id",
    "new_date",
    "new_time_slot",
    "relationship",
)

# A natural CLAUSE per field, never a "label: value" pair (KCD-454 --
# spoken punctuation artefacts are never acceptable, including ones this
# module itself might otherwise introduce).
_FIELD_CLAUSE = {
    "bn": {
        "doctor_name": "ডাক্তার {v}",
        "date": "{v} তারিখে",
        "time_slot": "{v} সময়ে",
        "patient_name": "নাম {v}",
        "phone": "ফোন নম্বর {v}",
        "contact_phone": "ফোন নম্বর {v}",
        "confirmation_id": "কনফার্মেশন নম্বর {v}",
        "new_date": "নতুন তারিখ {v}",
        "new_time_slot": "নতুন সময় {v}",
        "relationship": "সম্পর্ক {v}",
    },
    "hi": {
        "doctor_name": "डॉक्टर {v}",
        "date": "{v} तारीख को",
        "time_slot": "{v} बजे",
        "patient_name": "नाम {v}",
        "phone": "फ़ोन नंबर {v}",
        "contact_phone": "फ़ोन नंबर {v}",
        "confirmation_id": "कन्फ़र्मेशन नंबर {v}",
        "new_date": "नई तारीख {v}",
        "new_time_slot": "नया समय {v}",
        "relationship": "रिश्ता {v}",
    },
    "en": {
        "doctor_name": "doctor {v}",
        "date": "{v}",
        "time_slot": "{v}",
        "patient_name": "the name {v}",
        "phone": "the phone number {v}",
        "contact_phone": "the phone number {v}",
        "confirmation_id": "confirmation number {v}",
        "new_date": "{v}",
        "new_time_slot": "{v}",
        "relationship": "relationship {v}",
    },
}


def correction_acknowledgement(changed: list[str], prior_slots: dict, new_slots: dict, lang: str) -> str | None:
    """KCD-453: a caller correcting a value already captured gets an
    explicit acknowledgment and restatement -- never a silent overwrite,
    and the agent never re-asserts the value it is replacing. Returns
    None when `changed` has nothing that counts as a CORRECTION: a field
    filled in for the first time this call is ordinary slot-filling, not
    a correction, and gets its usual flow/prompt instead, not this.

    Only the field's OWN new value is restated -- this is deliberately
    not a full booking_confirmation_readback, which already exists and
    still runs before anything is written; this is the narrower "I heard
    you, here is what changed" turn a caller expects immediately after
    correcting themselves, which the readback alone does not give them
    on an intermediate slot-filling turn."""
    corrected = [f for f in changed if f in _CORRECTABLE_FIELDS and prior_slots.get(f) not in (None, "")]
    if not corrected:
        return None
    clauses = _FIELD_CLAUSE.get(lang, _FIELD_CLAUSE["en"])
    parts = [clauses[f].format(v=new_slots.get(f)) for f in corrected]
    if lang == "hi":
        return f"ठीक है, अब {', '.join(parts)} -- सही कर दिया।"
    if lang == "en":
        return f"Got it, changed to {', '.join(parts)}."
    return f"ঠিক আছে, বদলে {', '.join(parts)} করে দিলাম।"
