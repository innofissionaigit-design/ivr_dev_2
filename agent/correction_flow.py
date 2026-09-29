"""ADDED BY SOURAV -- "The agent accepts a correction and restates" story
(Epic: Answer Quality and Grounding).

story title: The agent accepts a correction and restates
user story: As a caller correcting the agent, I want the correction taken
    and confirmed, so that I am not arguing with a machine.
acceptance criteria: A correction updates the named value, is acknowledged
    explicitly and the corrected value is restated. The agent never defends
    a previous answer. Corrections are tested at every point in every flow.

WHAT WAS ALREADY THERE (KCD-448, "Every critical value is read back before
it is used"): a two-step correction path for the booking and callback
flows, but ONLY reachable by rejecting the final readback ("no" ->
"which one should I fix?" -> name a field -> re-collect it -> read the
whole thing back again). Three real gaps against THIS story's AC:

1. No acknowledgment. Once the caller named the field, the agent went
   straight back to the SAME question it asked the first time that field
   was ever collected ("Would you like today or another day?") -- nothing
   ever said "okay, updating the date" before asking. The AC's "acknowledged
   explicitly and the corrected value is restated" is a promise that
   two-step dance never kept; it only ever restated the correction
   incidentally, buried inside the full five-value readback at the end.

2. Only reachable from the final readback. A caller who spontaneously
   corrects an EARLIER field while being asked about a LATER one (asked
   for the time slot, says "actually, make the date Friday instead")
   had no path at all -- the date parser was never even tried against
   that turn's text, so "Friday instead" either failed to parse as a time
   and triggered a repeat of the time question, or silently derailed the
   booking. "at every point in every flow" is the literal gap this closes.

3. A dead end for "doctor". The correction menu ("which one should I fix
   -- the doctor, date, time, name, or phone number?") names the doctor as
   a correctable field, and parse_correction_field() (agent/slot_parse.py)
   dutifully recognises "doctor" in the caller's answer -- but
   main.py/main_pcm.py's _continue_pending() then set
   pending["awaiting"] = "doctor_name" and re-asked "which doctor", yet no
   branch of the generic date/time_slot/patient_name/phone tail that
   follows ever matches `awaiting == "doctor_name"` (matching a doctor
   needs an async catalogue lookup, unlike the other four fields' pure
   synchronous parsers). Whatever the caller said next fell straight
   through to "value is None", so naming "doctor" as the wrong field could
   never actually be fixed through this path -- three retries later the
   whole booking (four still-correct values included) was silently thrown
   away for a fresh LLM classification. See main.py/main_pcm.py's own
   `awaiting == "doctor_name"` block (added alongside this module) for the
   fix -- an actual async lookup, matching how a doctor's name is
   validated everywhere else in this codebase.

WHY THIS IS A SEPARATE, PURE MODULE, NOT WRITTEN INLINE IN
main.py/main_pcm.py's tails a second time each: same drift reason as
agent/report_flow.py's own module docstring gives at length -- main.py and
main_pcm.py each keep a complete, independently-maintained copy of
_continue_pending (WAV vs raw PCM transports over otherwise-identical turn
logic), and every decision that can be pure -- "is this utterance a
correction, and if so, to which field, and what is the new value" -- is
kept here once so it cannot drift between the two copies. The two
transport files are left responsible only for what genuinely differs
between them: awaiting `_tools`, calling their own `_speak()`, and
mutating `session.pending`.

WHY A CORRECTION CUE IS REQUIRED, NOT JUST "does this utterance parse as
some OTHER already-known field's type": running every field's parser
against every turn unconditionally, with no cue, risks silently
reinterpreting an ordinary, correctly-targeted answer as a correction to
something else entirely -- the same "return None rather than guess" trust
model agent/slot_parse.py's own parse_correction_field() docstring states
outright. A caller who is currently being asked for a phone number and
happens to mention a date in passing (rare, but not impossible in natural
speech) is answering the phone question, not correcting the date, unless
they actually SIGNAL a correction ("actually...", "no wait...", "I meant
..."). Matching only when both a cue AND a confident value are present is
what keeps this from ever overriding a plain, correctly-targeted answer.

WHAT THIS MODULE DELIBERATELY DOES NOT COVER, AND WHY -- TWO DIFFERENT
REASONS:

1. The DOCTOR. Correcting it spontaneously (mid-collection, without first
   naming "doctor" via the two-step menu) is not attempted here --
   validating a doctor name needs an async catalogue call, which a pure,
   synchronous detector cannot make. A caller who wants to correct the
   doctor still says so via the existing "which one should I fix?" menu
   (now reachable at any point, and now actually completable -- see gap 3
   above).

2. PATIENT NAME and the CALLBACK TIME WINDOW. These are free text with no
   grammar of their own -- unlike a date, a time or a phone number, there
   is no shape a parser can reject a non-value on. An early version of
   this module DID include them here, and two of test_booking_readback.py
   / test_request_callback.py's own pre-existing tests caught the real
   bug that caused: a caller merely NAMING "patient_name" as the field to
   fix ("রোগীর নাম", "the patient's name") was itself accepted by
   _clean_patient_name() as though it were the new name, and naming
   "callback_time_window" ("সময়টা ঠিক না", "the time isn't right") was
   accepted by a bare "non-empty string" check as though it were the new
   time window -- both fields "never guess" only by having NO structure
   to check against, so this module cannot tell "the caller is naming the
   field" from "the caller is also giving the value" for either one, and
   does not try. Both fields keep the deliberate, pre-existing two-step
   path (name the field, THEN a separate turn supplies the new value) --
   see main.py/main_pcm.py's own "confirm_correction"/"confirm_callback_
   correction" states for where they still get their acknowledgment, just
   one turn later than date/time_slot/phone do.

This module only auto-detects spontaneous corrections to the three fields
with genuine, structurally-validating parsers: date, time_slot, and phone
(both flows' phone field).

report_status/report_send (agent/report_flow.py) is also out of scope,
deliberately: unlike booking/callback, those flows never hold a
multi-field record waiting to be corrected. Their one caller-supplied
value (a phone number) is used immediately for a single identity lookup,
with nothing persisted afterward to "restate" -- a caller who misspoke it
is already free to simply ask again with the right number, which is the
correct-and-only analogue of "correcting a value" for a single-shot
lookup with no record. Inventing a multi-turn correction dance for a
flow that never asks for a second field would be scope invention, not a
gap this AC names.
"""
from __future__ import annotations

from typing import Callable

# --------------------------------------------------------------------- #
# Correction cues, by script/language family. Deliberately literal
# substring phrases (same discipline as agent/human_fast_path.py's own
# phrase tuples) rather than a broad regex -- a missed cue just means the
# caller is asked again (safe, if slightly repetitive); a FALSE cue could
# reinterpret an ordinary answer as a correction to something else, which
# is the failure mode this module exists to avoid.
# --------------------------------------------------------------------- #
_EN_CUES = (
    "actually", "no wait", "no, wait", "wait, no", "not that", "i meant",
    "sorry i meant", "sorry, i meant", "change it to", "change that to",
    "instead", "make it", "wait no", "sorry no", "no no", "correction,",
    "correction ", "i mean", "scratch that", "sorry, actually",
)
_LATIN_TRANSLIT_CUES = (
    "actually", "nahi ruko", "nahi, ruko", "ruko nahi", "matlab",
    "sorry mera matlab", "mera matlab", "badal do", "ulta", "nahi galti",
    "galti se bola", "wait nahi", "na thik na", "asol e", "vul bolechi",
    "vul hoyeche", "sorry actually",
)
_BN_CUES = (
    "আসলে", "না দাঁড়ান", "মানে", "সরি ভুল বললাম", "বদলে দিন", "না ভুল হয়েছে",
    "ভুল বলেছি", "থামুন না",
)


def has_correction_cue(text: str) -> bool:
    """True when `text` contains an explicit signal that the caller is
    correcting something already said, rather than answering the
    question currently on the table. See this module's own docstring for
    why a cue is required rather than inferring a correction purely from
    "this text happens to parse as some other field"."""
    if not text or not text.strip():
        return False
    lowered = text.lower()
    if any(cue in lowered for cue in _EN_CUES):
        return True
    if any(cue in lowered for cue in _LATIN_TRANSLIT_CUES):
        return True
    # Bengali script has no case to lower -- checked against the raw text.
    if any(cue in text for cue in _BN_CUES):
        return True
    return False


# --------------------------------------------------------------------- #
# Booking. Only the fields with a pure, synchronous, STRUCTURALLY-
# VALIDATING parser -- "doctor_name" is excluded because it needs an
# async catalogue lookup (see module docstring point 1); "patient_name"
# is excluded because it has no structure a parser could ever reject a
# non-value on (see module docstring point 2 for the real bug that
# caught this). Both still go through the existing two-step "which one
# should I fix?" menu -- main.py/main_pcm.py's own `awaiting ==
# "doctor_name"` branch is what actually completes a doctor correction.
# --------------------------------------------------------------------- #
_BOOKING_SPONTANEOUS_FIELDS = ("date", "time_slot", "phone")


def detect_booking_correction(
    text: str,
    known_slots: dict,
    awaiting: str | None,
    parsers: dict[str, Callable[[str], object | None]],
) -> tuple[str, object] | None:
    """-> (field, new_value) when `text` both carries a correction cue and
    confidently reparses as a booking field OTHER than `awaiting` that is
    already present in `known_slots` -- or None.

    `awaiting` is excluded so the CURRENTLY-asked field is left to its own
    normal parsing at the call site (a plain answer to the field on the
    table is never reinterpreted as a correction to itself); pass None
    when nothing is specifically "the current field" (e.g. at the final
    readback, where every field is already known and any of them may be
    the one being corrected).

    `parsers` maps each of _BOOKING_SPONTANEOUS_FIELDS to a callable
    text -> value | None -- passed in rather than imported here so this
    module stays a pure function of its inputs and never has to duplicate
    (or drift from) agent/slot_parse.py's own parsing rules or main.py/
    main_pcm.py's own offered-date context for parse_date().

    Never guesses: returns None the moment either the cue is missing or
    no OTHER known field's parser confidently accepts the text, same
    trust model as every parser in agent/slot_parse.py.
    """
    if not has_correction_cue(text):
        return None
    for field in _BOOKING_SPONTANEOUS_FIELDS:
        if field == awaiting:
            continue
        if field not in known_slots:
            continue
        parser = parsers.get(field)
        if parser is None:
            continue
        value = parser(text)
        if value is not None:
            return field, value
    return None


# --------------------------------------------------------------------- #
# Callback (request_callback). Only "phone" -- "callback_time_window" is
# excluded for the same reason "patient_name" is above (module docstring
# point 2: free text with no structure to validate a spontaneous
# correction against). `known_slots` uses the SAME key the flow itself
# does for the phone field ("phone", not "callback_phone" -- see
# agent/slot_parse.py's parse_callback_correction_field() docstring for
# the same "awaiting" name vs "slots" key distinction).
# --------------------------------------------------------------------- #
_CALLBACK_SPONTANEOUS_FIELDS = ("phone",)
_CALLBACK_AWAITING_FOR_FIELD = {"phone": "callback_phone"}


def detect_callback_correction(
    text: str,
    known_slots: dict,
    awaiting: str | None,
    parsers: dict[str, Callable[[str], object | None]],
) -> tuple[str, object] | None:
    """Callback's own analogue of detect_booking_correction() above --
    same cue-gated, never-guess trust model, scoped to the two fields a
    callback request actually collects."""
    if not has_correction_cue(text):
        return None
    for field in _CALLBACK_SPONTANEOUS_FIELDS:
        if _CALLBACK_AWAITING_FOR_FIELD[field] == awaiting:
            continue
        if field not in known_slots:
            continue
        parser = parsers.get(field)
        if parser is None:
            continue
        value = parser(text)
        if value is not None:
            return field, value
    return None
