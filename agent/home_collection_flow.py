"""Cross-turn state for "Caller asks whether a test can be collected at home" (KCD-387 full flow).

ADDED BY SOURAV: this is a NEW file, not a repurposing of agent/booking_flow.py's BookingState.
BookingState's "collecting -> confirming -> done" shape fits an appointment or a lab-test booking,
where every slot is captured BEFORE a single confirm/commit step. This story is structurally
different: it has several real backend calls interleaved with the conversation (an eligibility
check, a slot hold, a priced quote, a booking write) and -- unlike any existing booking flow -- TWO
separate yes/no gates before anything is written (section 19's "would you like to proceed?" and
section 23's "is this address right?"), plus a final full-summary confirm. Collapsing all of that
into BookingState's one "confirming" stage would make it impossible to tell, from the state alone,
which of those three questions is in flight -- exactly the ambiguity the story's own confirmation
rules (19/23/30) are written to prevent. Hence a dedicated, parallel state machine, following
BookingState's OWN conventions (a plain dataclass, touch()/is_stale(), deterministic pure helper
functions, no network/DB access here -- that stays in agent/tools_client.py and main.py).

Nothing in this module calls the clinic API or speaks to the caller: it only tracks what is known
so far and what is still needed, so it stays unit-testable with no server running, the same
discipline booking_flow.py's own docstring describes.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# ------------------------------------------------------------------------------------- stages
# Each stage names exactly one open question. A turn either answers the stage's own question (and
# the state moves on) or asks something else entirely (handled as an ordinary fresh turn by
# main.py, same as BookingState's "collecting" stage never blocks an unrelated question).
#
# main.py only ever sets/reads `stage` for the gates below that need to tell WHICH yes/no (or
# slot-choice) question is pending, for its early, pre-LLM interception
# (_handle_home_collection_confirmation_turn): CHOOSE_SLOT, AWAITING_PROCEED,
# OFFER_STORED_ADDRESS, CONFIRM_ADDRESS, FINAL_CONFIRM. The earlier, open-ended collection phases
# (COLLECT_TESTS/COLLECT_POSTAL_CODE/COLLECT_DATE/COLLECT_ADDRESS, and the NOT_ELIGIBLE/DONE
# outcomes) are driven by which fields are already filled in, not by this field -- the same
# "ask about missing_required(), not a stage" shape BookingState itself uses for its own
# "collecting" stage -- so they are listed here for completeness/documentation even though no code
# assigns them.
STAGE_COLLECT_TESTS = "collect_tests"
STAGE_COLLECT_POSTAL_CODE = "collect_postal_code"
STAGE_NOT_ELIGIBLE = "not_eligible"  # terminal: nothing eligible/serviceable -- nothing left to book
STAGE_COLLECT_DATE = "collect_date"
STAGE_CHOOSE_SLOT = "choose_slot"
STAGE_AWAITING_PROCEED = "awaiting_proceed"  # quote + payment policy spoken; waiting on yes/no
STAGE_COLLECT_PATIENT = "collect_patient"  # patient name / phone for the booking record
STAGE_OFFER_STORED_ADDRESS = "offer_stored_address"  # "use the address on file?" yes/no
STAGE_COLLECT_ADDRESS = "collect_address"
STAGE_CONFIRM_ADDRESS = "confirm_address"  # readback; waiting on yes/no
STAGE_FINAL_CONFIRM = "final_confirm"  # full summary; waiting on yes/no -- the only stage that books
STAGE_DONE = "done"

# Longer than BookingState.STATE_IDLE_TIMEOUT_S (180s): this flow has more real round trips (an
# eligibility check, a hold, a quote, a booking write) interleaved with the conversation, so a
# caller who is mid-flow but momentarily quiet (reading an address off a letter, say) needs more
# room before the flow is treated as abandoned.
IDLE_TIMEOUT_S = 240.0

# A caller may be read back up to this many home-collection time windows before being asked to
# choose -- kept in one place so main.py and any reply template agree on it.
MAX_SLOTS_READ_ALOUD = 4


@dataclass
class HomeCollectionState:
    stage: str = STAGE_COLLECT_TESTS

    # ---- test identification (sections 6/7/8) ----
    test_names: list[str] = field(default_factory=list)
    from_booking: bool = False  # tests came from an existing booking lookup (CASE 2), not caller-named
    confirmation_id: str | None = None  # the booking the tests were resolved from, if any
    registration_asked: bool = False  # KCD-382b-style "are you already booked?" question, asked at most once

    # ---- eligibility / serviceability (sections 8/9/10) ----
    postal_code: str | None = None
    eligibility: list[dict] = field(default_factory=list)  # one multi_test_eligibility() row per requested test
    eligible_tests: list[str] = field(default_factory=list)
    not_found_tests: list[str] = field(default_factory=list)

    # ---- date / slot (sections 11/12/13/14) ----
    date: str | None = None
    slots_offered: list[dict] = field(default_factory=list)
    chosen_slot: dict | None = None  # {"slot_id", "date", "start_time", "end_time", ...}
    hold_token: str | None = None

    # ---- pricing / payment (sections 15/16/17/18) ----
    quote: dict | None = None
    payment_policy_text: str | None = None

    # ---- patient / contact (section 20) ----
    patient_name: str | None = None
    phone: str | None = None
    caller_phone: str | None = None
    patient_id: int | None = None

    # ---- address (sections 21/22/23/24) ----
    address_line: str = ""
    locality: str = ""
    city: str = ""
    state: str = ""
    landmark: str = ""
    using_stored_address: bool = False
    stored_address_offered: bool = False
    stored_address_text: str | None = None  # the address on file, held here only between the offer and its answer

    # ---- booking result (sections 25/26/27) ----
    booking_result: dict | None = None

    retry_counts: dict = field(default_factory=dict)
    last_updated: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_updated = time.monotonic()

    def is_stale(self) -> bool:
        return (time.monotonic() - self.last_updated) > IDLE_TIMEOUT_S

    def note_retry(self, field_name: str) -> int:
        self.retry_counts[field_name] = self.retry_counts.get(field_name, 0) + 1
        return self.retry_counts[field_name]


def new_state() -> HomeCollectionState:
    return HomeCollectionState()


# ------------------------------------------------------------------------------- slot merging


def merge_test_names(state: HomeCollectionState, names: list[str] | None) -> bool:
    """Folds newly-named tests into the running list -- never replaces it, so a caller who names
    tests across more than one turn ("Uric Acid" then, a turn later, "and CBC too") ends up with
    both, the same accumulate-don't-overwrite rule BookingState.merge_slots uses for test_names."""
    changed = False
    for n in names or []:
        n = (n or "").strip()
        if n and n not in state.test_names:
            state.test_names.append(n)
            changed = True
    return changed


def merge_postal_code(state: HomeCollectionState, postal_code: str | None) -> bool:
    """A later postal code always overwrites an earlier one -- the caller correcting themselves
    ("sorry, not 700091, it's 700019") is exactly BookingState.merge_slots' own "a later value
    always wins" rule, applied to this one field. `postal_code` is already a clean 6-digit string
    or None by the time it reaches here (agent/intent_schema.py's Slots validator normalises it) --
    this function never re-validates or re-extracts digits itself, so there is exactly one place in
    the codebase that decides what counts as a postal code."""
    if postal_code and postal_code != state.postal_code:
        state.postal_code = postal_code
        return True
    return False


def missing_address_fields(state: HomeCollectionState) -> list[str]:
    """What is still needed before the address can be read back and confirmed. `landmark` is
    deliberately never required -- section 21 calls it out as "where needed", i.e. optional."""
    missing = []
    if not state.address_line.strip():
        missing.append("address_line")
    if not state.postal_code:
        missing.append("postal_code")
    return missing


_DIGIT_RUN_RE = re.compile(r"\d+")


def address_postal_code_conflict(state: HomeCollectionState) -> str | None:
    """Section 22: "validate the final address against the final pincode -- mismatch -> ask, don't
    guess." A caller's spoken address often repeats the pincode inside it ("...700091"); if it does
    and that 6-digit run DIFFERS from the postal code already given separately, that is a real
    conflict worth asking about rather than silently trusting either one. Returns the conflicting
    run, or None when the address mentions no 6-digit run of its own (nothing to compare) or it
    agrees with `state.postal_code`."""
    if not state.postal_code:
        return None
    runs = [r for r in _DIGIT_RUN_RE.findall(state.address_line) if len(r) == 6]
    conflicting = [r for r in runs if r != state.postal_code]
    return conflicting[0] if conflicting else None


def address_display(state: HomeCollectionState) -> str:
    """One spoken clause for the readback/confirmation step -- never a label:value dump (the same
    "a natural clause per field" rule agent/booking_flow.correction_acknowledgement's own
    _FIELD_CLAUSE table follows), and never split into house/road/locality/city sub-fields the
    caller never heard asked back to them as separate facts."""
    parts = [state.address_line.strip()]
    if state.landmark.strip():
        parts.append(state.landmark.strip())
    if state.postal_code:
        parts.append(state.postal_code)
    return ", ".join(p for p in parts if p)


# ------------------------------------------------------------------------------- slot selection

_ORDINAL_WORDS: dict[str, tuple[tuple[str, ...], ...]] = {
    # index 0 = "first", index 1 = "second", index 2 = "third", index 3 = "fourth" -- only as many
    # as MAX_SLOTS_READ_ALOUD ever need naming.
    "bn": (("প্রথম",), ("দ্বিতীয়",), ("তৃতীয়",), ("চতুর্থ",)),
    "hi": (("पहला", "पहली", "pehla", "pahla"), ("दूसरा", "दूसरी", "dusra", "doosra"), ("तीसरा", "teesra"), ("चौथा", "chautha")),
    # ADDED BY SOURAV: "one"/"two"/"three"/"four" are deliberately NOT listed as cardinal fallbacks
    # here -- "the second one" contains the word "one" and would otherwise match index 0 ("first")
    # before ever reaching "second" at index 1. "first"/"second"/... and their digit-suffix forms
    # are unambiguous and kept; a bare cardinal number is left to the start_time match above, or to
    # a re-ask, rather than risk picking the wrong slot.
    "en": (("first", "1st"), ("second", "2nd"), ("third", "3rd"), ("fourth", "4th")),
}


def resolve_slot_choice(slots_offered: list[dict], utterance: str, time_slot_text: str | None, lang: str) -> dict | None:
    """Which of `slots_offered` the caller meant, from whatever free text was heard -- either the
    whole turn's transcript or just the extracted "time_slot" slot, whichever was given. Returns
    None (never a guess) when nothing in the utterance distinguishes one offered slot from another.

    Deliberately simple, on purpose (the story's own "no unnecessarily complex parser" rule,
    applied here too): a slot is picked by (a) its own start_time appearing in what was said, or
    (b) an ordinal word ("the first one", "প্রথমটা"). Nothing fuzzier than that -- an unmatched
    utterance is asked again, which is the safe direction to be wrong in."""
    if not slots_offered:
        return None
    if len(slots_offered) == 1:
        return slots_offered[0]
    text = f"{utterance or ''} {time_slot_text or ''}".lower()
    for slot in slots_offered:
        start = str(slot.get("start_time") or "")
        if start and start.lower() in text:
            return slot
    for i, words in enumerate(_ORDINAL_WORDS.get(lang, _ORDINAL_WORDS["en"])):
        if i < len(slots_offered) and any(w in text for w in words):
            return slots_offered[i]
    return None
