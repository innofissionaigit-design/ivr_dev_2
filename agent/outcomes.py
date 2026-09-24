"""The insufficient-verified-information outcome.

Story: "The agent says it cannot confirm rather than guessing" (Epic:
Answer Quality and Grounding; near-duplicate acceptance criteria also sit
under the Truth Validator epic's "The insufficient-verified-information
outcome as a first-class path" -- same owner, not yet picked up, flagged
rather than silently merged).

This is a THIRD outcome, distinct from the two that already exist in this
codebase:

  * "not-found"              -- the tool answered cleanly and the answer
                                 is "no" (agent/reply_templates.py's
                                 found=False / success=False branches).
  * "infrastructure apology" -- the tool call itself failed
                                 (agent/tools_client.py's ToolCallError,
                                 caught in four places in main_pcm.py,
                                 all four speaking the same hardcoded
                                 "can't check this right now" line).
  * "insufficient-verified-information" (this file) -- the tool call
    SUCCEEDED and returned a value, but the value cannot be trusted
    enough to speak. The agent says so plainly instead of guessing.

SCOPE -- deliberately narrow, per the three constraints agreed before this
was built:

  1. REAL TRIGGER, and only this one: main_pcm.py's _finish_booking()
     calls missing_booking_write_fields() after book_appointment() reports
     success, and checks only that confirmation_id, date and time_slot
     came back non-empty before any of them is read aloud. That is the
     bare minimum any caller of a write API should do, and it is
     deliberately NOT:
       - shape/type validation of the response       ("Schema drift
         rejected at the boundary" -- a different, unpicked story; that
         story's own acceptance criteria route a malformed response to
         the INFRASTRUCTURE APOLOGY, not to this outcome)
       - a plausibility/business-rule check on the values ("Business rule
         validation before verbalisation" -- different, unpicked story)
       - a freshness/staleness check                 ("Freshness bounds
         per field with withholding" -- different, unpicked story)
       - a re-verification of who is allowed to hear the value
         ("Authorisation re-verified at the validator" -- different,
         unpicked story)
       - a re-query of the database to confirm the row really persisted
         ("Write confirmation validated before it is spoken" -- different,
         unpicked story)
       - a fix for the doctor-name fuzzy-matcher silently picking its best
         guess with no confirmation ("Near matches are offered rather than
         guessed or refused" -- different, unpicked story; that one is
         about OFFERING candidates, a different UX from this outcome)
     Building any of those is the rest of the "Truth Validator" epic, not
     this story.

  2. NO FAKE RATE. The acceptance criterion says "its rate is reported per
     intent". A rate needs a denominator -- attempts per intent -- and
     nothing in this codebase counts attempts per intent today (there is
     no metrics system here at all; see below). This file tracks the
     COUNT per intent only, via insufficient_verified_information_counts().
     Dividing by a future attempts-counter, wherever that ends up living,
     is a one-line change against this function -- it is not something
     this file fabricates a denominator for now.

  3. ESCALATION PATH is the JSONL ledger this file appends to
     (ESCALATION_LOG_PATH). This codebase has no metrics or audit
     infrastructure of any kind today -- every existing signal is a plain
     `logging.getLogger("main")` call with no aggregation. A JSONL ledger
     is the smallest thing that is still durable, greppable by a human,
     and diffable, without inventing a dependency this repo doesn't have.
     Wiring it to a real paging/alerting system (PagerDuty, a Slack
     webhook, whatever the clinic actually uses) is a deployment
     integration decision that this file cannot honestly make up against
     a backend that has none today -- record_insufficient_verified_
     information() is exactly the function such an integration would call
     into.
"""
from __future__ import annotations

import datetime
import json
import os
import threading

# One JSON object per line: {"timestamp", "intent", "field", "reason",
# "call_id"}. Overridable via environment variable, the same convention
# agent/tools_client.py's base URL and clinic-api/db.py's DATABASE_URL
# already use, so tests and deployments can point this elsewhere without
# editing this file.
ESCALATION_LOG_PATH = os.environ.get("ESCALATION_LOG_PATH", "logs/escalations.jsonl")

_lock = threading.Lock()
_counts: dict[str, int] = {}

# The bare-minimum check this story wires up. Presence only -- see the
# module docstring for exactly why shape, plausibility, freshness,
# authorization and DB re-confirmation are all deliberately NOT here.
REQUIRED_BOOKING_WRITE_FIELDS = ("confirmation_id", "date", "time_slot")


def missing_booking_write_fields(result: dict) -> list[str]:
    """Which of confirmation_id/date/time_slot came back missing or empty
    on a booking response that otherwise reported success=True.

    Called from main_pcm.py's _finish_booking() -- the ONE real trigger
    this story wires up. Returns an empty list when the write looks
    complete (the overwhelmingly common case against a healthy backend);
    a non-empty list means _finish_booking() must not read any of these
    values aloud with false confidence.
    """
    return [f for f in REQUIRED_BOOKING_WRITE_FIELDS if not result.get(f)]


# ADDED BY SOURAV -- "Caller asks to be called back" story. Same "the
# agent says it cannot confirm rather than guessing" discipline as the
# booking write just above, extended to this story's own write
# (POST /api/v1/callbacks): a callback_id is this endpoint's equivalent of
# a confirmation_id, and main.py's _finish_callback() must not read one
# aloud unless it actually came back non-empty on a success=True response.
REQUIRED_CALLBACK_WRITE_FIELDS = ("callback_id",)


def missing_callback_write_fields(result: dict) -> list[str]:
    """Which of REQUIRED_CALLBACK_WRITE_FIELDS came back missing or empty
    on a callback-request response that otherwise reported success=True.
    Mirrors missing_booking_write_fields() above exactly, for the one new
    write this story adds; both functions share the same
    record_insufficient_verified_information() escalation path and the
    same insufficient_verified_information_reply() spoken template --
    neither of those needed any change for this story."""
    return [f for f in REQUIRED_CALLBACK_WRITE_FIELDS if not result.get(f)]


def record_insufficient_verified_information(intent: str, field: str, reason: str,
                                              call_id: str | None = None) -> None:
    """Called at the ONE trigger point this story wires up. Does two
    things:

      1. Increments an in-process count for `intent` -- see
         insufficient_verified_information_counts(). Deliberately a count,
         not a rate; see the module docstring, constraint 2.
      2. Appends one JSON line to ESCALATION_LOG_PATH -- the escalation
         path the acceptance criteria requires. A human, or a real
         alerting integration once one exists, can tail or grep this
         file.

    A single process-wide lock is enough for this workload (one call
    center's worth of concurrent calls on one process, the same scale
    every other piece of in-process state in this codebase -- session
    dicts, the semantic cache -- already assumes); this is not meant to
    survive a multi-process deployment without becoming a real metrics
    store, which is exactly the upgrade path constraint 2 and 3 above are
    written to not block.
    """
    with _lock:
        _counts[intent] = _counts.get(intent, 0) + 1
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "intent": intent,
            "field": field,
            "reason": reason,
            "call_id": call_id,
        }
        log_dir = os.path.dirname(ESCALATION_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(ESCALATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def insufficient_verified_information_counts(intent: str | None = None) -> dict[str, int] | int:
    """Per-intent occurrence COUNTS, not a rate -- deliberately (see the
    module docstring, constraint 2). Pass `intent` for just that intent's
    count (0 if it has never been recorded); omit it for a snapshot dict
    of every intent's count so far.
    """
    with _lock:
        if intent is not None:
            return _counts.get(intent, 0)
        return dict(_counts)


# =============================================================================
# ADDED BY SOURAV -- "Caller asks how to prepare for a test" story's bundled
# human_fallback config (lab_tests_with_fallback_config sample file's
# voice_agent_config.human_fallback block). A FOURTH outcome, distinct from
# the three the module docstring above already lists: the caller's query
# could not be resolved at all (agent/llm.py's "unclear" intent).
#
# The config's own action is "transfer_to_human_agent", but there is no
# telephony transfer capability anywhere in this codebase (no SIP/PSTN
# library, no call-control API of any kind) -- an actual call transfer is
# not something this file can honestly build. What IS real and buildable,
# following this file's own constraint 3 above ("ESCALATION PATH is the
# JSONL ledger this file appends to... wiring it to a real paging/alerting
# system is a deployment integration decision this file cannot honestly
# make up against a backend that has none today"): the same ledger gets a
# new record, so a human follow-up process has something durable and
# greppable to act on. agent/reply_templates.human_fallback_reply() is the
# spoken half of this same story; this is only the logging half.
#
# Kept as a SEPARATE counter (_handoff_counts) from _counts/
# insufficient_verified_information_counts() above -- these are two
# different outcomes, and merging their counts under one dict would make
# both numbers meaningless.
# =============================================================================

_handoff_counts: dict[str, int] = {}


def record_human_handoff(intent: str, call_id: str | None = None,
                          reason: str = "query_unresolved_or_low_confidence") -> None:
    """Records one "-> hand off to a human" event: bumps the per-intent
    count and appends a JSON line to the same ESCALATION_LOG_PATH ledger
    record_insufficient_verified_information() above writes to
    (distinguished by "event": "human_handoff", so the two outcomes remain
    greppable apart in one shared, durable file).

    `reason` defaults to this function's original, and still overwhelmingly
    common, story: "unclear"/"out_of_scope"/"clinical_interpretation" all
    hand off because the query itself was confusing, ambiguous, or outside
    what this system can answer. ADDED BY SOURAV -- "Caller asks for a
    person immediately" story (Epic: Conversation -- Difficult, Sensitive
    and Edge Cases): that default reason is actively WRONG for this new
    story's own handoff -- the caller was perfectly understood and simply
    asked for a human, so the ledger must say that plainly rather than
    implying confusion that never happened. See
    record_immediate_human_handoff() below for the caller that passes an
    explicit reason.
    """
    with _lock:
        _handoff_counts[intent] = _handoff_counts.get(intent, 0) + 1
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": "human_handoff",
            "intent": intent,
            "reason": reason,
            "call_id": call_id,
        }
        log_dir = os.path.dirname(ESCALATION_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(ESCALATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def human_handoff_counts(intent: str | None = None) -> dict[str, int] | int:
    """Per-intent human-handoff occurrence COUNTS -- mirrors
    insufficient_verified_information_counts() above exactly, but reads
    the separate _handoff_counts dict, never _counts."""
    with _lock:
        if intent is not None:
            return _handoff_counts.get(intent, 0)
        return dict(_handoff_counts)


# =============================================================================
# ADDED BY SOURAV -- "Caller asks for a person immediately" story (Epic:
# Conversation -- Difficult, Sensitive and Edge Cases). AC: "A request for a
# human in any supported language escalates on the same turn with no
# retention attempt and no question about why... the rate is reported as a
# quality signal rather than something to minimise."
#
# WHY THIS NEEDED A NEW DENOMINATOR THIS MODULE'S OWN DOCSTRING SAID DIDN'T
# EXIST: constraint 2 up top ("NO FAKE RATE... nothing in this codebase
# counts attempts per intent today") was written for the insufficient-
# verified-information outcome, where no denominator was needed yet and
# fabricating one would have been dishonest. This story's AC explicitly asks
# for a rate, so the missing piece -- a genuine attempts counter -- is built
# here, exactly along the upgrade path that docstring already pointed to
# ("dividing by a future attempts-counter... is a one-line change against
# this function"). _total_turns is that counter: main.py/main_pcm.py call
# record_turn_attempt() once per dispatched caller turn (see
# _dispatch_turn_inner's own first line in both files), before any intent is
# even resolved, so it is a true count of turns attempted, not of only the
# turns that happened to reach this intent.
#
# WHY THE RATE IS COMPUTED FROM _handoff_counts RATHER THAN A SEPARATE
# COUNTER: a second counter incremented alongside record_human_handoff()
# would eventually drift from it (two numbers meant to always agree, kept in
# two places, is exactly the "two files that look like duplicates aren't
# automatically in sync" trap this codebase's own report_flow.py module
# docstring warns about for a different pair of files). human_handoff_counts
# ("human_direct_request") is already the authoritative count of this
# story's escalations; the rate is a read-only division on top of it.
_total_turns = 0

HUMAN_DIRECT_REQUEST_INTENT = "human_direct_request"
_HUMAN_DIRECT_REQUEST_REASON = "caller_requested_human_directly"


def record_turn_attempt() -> None:
    """Denominator for immediate_human_escalation_rate() below. Called once
    per caller turn dispatched, regardless of what intent it resolves to --
    a turn that never reaches human_direct_request still belongs in the
    denominator, or the rate would only ever measure "escalations per
    escalation-adjacent turn", not "escalations per turn" as the AC asks."""
    global _total_turns
    with _lock:
        _total_turns += 1


def total_turns() -> int:
    """The current denominator. Exposed mainly for tests and for whatever
    eventually reads immediate_human_escalation_rate() -- see this
    section's own module comment on why no dashboard/metrics endpoint
    exists in this codebase to read it FROM yet."""
    with _lock:
        return _total_turns


def record_immediate_human_handoff(call_id: str | None = None) -> None:
    """The zero-negotiation escalation event itself: records a handoff
    under HUMAN_DIRECT_REQUEST_INTENT with a reason that honestly reflects
    what happened (the caller was understood perfectly and asked for a
    human -- see record_human_handoff()'s own docstring for why its default
    reason string would be wrong here). Thin wrapper so main.py/main_pcm.py
    call one function instead of repeating the intent name and reason
    string at each dispatch site."""
    record_human_handoff(HUMAN_DIRECT_REQUEST_INTENT, call_id=call_id,
                          reason=_HUMAN_DIRECT_REQUEST_REASON)


HUMAN_COMPLAINT_INTENT = "complaint"
_HUMAN_COMPLAINT_REASON = "caller_filed_complaint"


def record_complaint_filed(call_id: str | None = None) -> None:
    """ADDED BY SOURAV -- "Caller wants to make a complaint" story. Records
    one "-> complaint filed, routed to a person" event under the same
    shared ESCALATION_LOG_PATH ledger record_human_handoff() writes to,
    with its own honest reason string (the caller was understood
    perfectly and is reporting a problem -- not confusion, not a direct
    request for a human) -- same pattern
    record_immediate_human_handoff() just above already established for
    its own story. Deliberately logs ONLY intent/reason/call_id, never the
    complaint text itself: the instruction for this story is explicit that
    logs/escalations.jsonl must keep avoiding caller free text (see this
    module's own docstring on why that ledger has that discipline at all)
    -- the verbatim text goes to clinic-api's ComplaintRecord table
    instead (see agent/tools_client.py's submit_complaint()), which is a
    different, narrower-access durable store built for exactly that."""
    record_human_handoff(HUMAN_COMPLAINT_INTENT, call_id=call_id,
                          reason=_HUMAN_COMPLAINT_REASON)


def immediate_human_escalation_rate() -> dict:
    """The quality signal itself: how often a caller turn ended in an
    immediate, zero-negotiation human handoff, out of every turn attempted.
    Returned as a dict (count/denominator/rate) rather than a bare float so
    a log line or a future dashboard can show the numbers the rate was
    built from, not just the ratio -- the AC calls this "a quality signal
    rather than something to minimise", and a rate with no visible
    denominator invites exactly that minimising ("just push the number
    down") instead of the intended reading ("this many people needed a
    human, unfiltered").
    """
    handoffs = human_handoff_counts(HUMAN_DIRECT_REQUEST_INTENT)
    turns = total_turns()
    rate = (handoffs / turns) if turns > 0 else 0.0
    return {
        "immediate_human_handoffs": handoffs,
        "total_turns": turns,
        "immediate_human_escalation_rate": round(rate, 4),
    }


# =============================================================================
# ADDED BY SOURAV -- "Caller goes silent" story (Epic: Conversation --
# Difficult, Sensitive and Edge Cases). AC: "Abandonment is logged with the
# turn index and the preceding prompt."
#
# A FIFTH event in the same ESCALATION_LOG_PATH ledger, distinguished from
# "human_handoff" (and from the un-tagged insufficient-verified-information
# records above, which predate the "event" field entirely) by its own
# "event": "call_abandoned" value -- same discipline record_human_handoff()
# already established: one shared, durable, greppable JSONL file, one
# process-wide lock, one dict of counts kept SEPARATE per event type so no
# two different things are ever added together.
#
# Called from main.py's/main_pcm.py's _turn_poll_loop, and from there only
# once per abandoned call: the ACTION_ABANDON branch this call sits in ends
# with `return`, so _turn_poll_loop never ticks again for that call
# afterward, which is what keeps this a one-shot log rather than something
# that needs its own re-entrancy guard.
# =============================================================================

_abandonment_count = 0


def record_call_abandoned(turn_index: int, preceding_prompt: str | None,
                           call_id: str | None = None) -> None:
    """Records one "caller went silent through both graduated prompts and
    the call was closed" event: bumps the process-wide abandonment count
    and appends a JSON line to ESCALATION_LOG_PATH.

    `turn_index` is session.utt_seq at the moment of abandonment -- the
    count of utterances this call had already dispatched, the same
    counter _turn_poll_loop itself increments for every confirmed
    utterance (see that function's own `session.utt_seq += 1`), not a
    value invented for this story.

    `preceding_prompt` is the exact text of whichever graduated prompt
    (agent/reply_templates.silence_prompt_one()/silence_prompt_two())
    was spoken immediately before the close -- per the acceptance
    criterion, this is always the SECOND prompt, since abandonment is
    only ever reached after both have gone unanswered.
    """
    global _abandonment_count
    with _lock:
        _abandonment_count += 1
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": "call_abandoned",
            "call_id": call_id,
            "turn_index": turn_index,
            "preceding_prompt": preceding_prompt,
        }
        log_dir = os.path.dirname(ESCALATION_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(ESCALATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_abandoned_count() -> int:
    """The current count of logged call_abandoned events -- mirrors
    total_turns() above exactly, but reads _abandonment_count, never
    _total_turns or either of the other two counters in this file."""
    with _lock:
        return _abandonment_count


# =============================================================================
# ADDED BY SOURAV -- Story 5 ("Caller asks about another person's
# report" / privacy gateway enforcement). AC: "The unauthorised attempt
# is audited."
#
# A SIXTH event in the same ESCALATION_LOG_PATH ledger, distinguished
# from every other event here by its own "event": "report_access_denied"
# value -- same discipline record_call_abandoned() above already
# established: one shared, durable, greppable JSONL file, one
# process-wide lock, its own counter dict (_denied_counts) kept SEPARATE
# from every other counter here so no two different things are ever
# added together.
#
# Called from main.py's/main_pcm.py's report-flow dispatch, exactly
# once, at the single point agent/report_flow.py's interpret_*()
# functions signal a denial via the "__report_access_denied__" sentinel
# (see that module and main.py's _apply_report_outcome()). Deliberately
# logs ONLY identifiers and a fixed reason code -- per this story's own
# explicit instruction, and matching record_complaint_filed()'s already-
# established discipline in this exact file, report STATUS, report
# VALUES, and any caller free text are never written to this ledger.
# `patient_id` is the one patient-identifying field logged, and even
# that is an internal database id, never a name, phone number, or any
# spoken value.
# =============================================================================

REPORT_ACCESS_DENIED_EVENT = "report_access_denied"

_denied_counts: dict[str, int] = {}


def record_report_access_denied(intent: str, reason: str, call_id: str | None = None,
                                 turn_index: int | None = None,
                                 patient_id: int | None = None) -> None:
    """Records one "-> report access denied by the authorization gate"
    event. Bumps a per-intent count (report_access_denied_counts()
    below) and appends one JSON line to the shared ESCALATION_LOG_PATH
    ledger.

    `intent` is whichever of "report_status"/"report_send"/
    "report_delivery" the denial happened under (see main.py's call
    sites -- the delivery/OTP continuation states do not track which
    original intent started the flow, so they log the generic
    "report_delivery"). `reason` is a fixed, non-sensitive code such as
    "VERIFIED_IDENTITY_MISMATCH" -- never a sentence built from caller
    speech.

    CALLER RESPONSIBILITY, not enforced here by design: main.py/
    main_pcm.py must call this AFTER already deciding to speak the
    fixed decline (agent/reply_templates.report_access_denied_reply())
    and clear session.pending, and must wrap this call in its own
    try/except -- see main.py's _apply_report_outcome(). This function
    intentionally does not swallow its own exceptions: a caller that
    wraps it can log a warning and continue; a caller that does NOT
    wrap it would surface the failure loudly rather than silently, and
    either way the decision to deny access was already made in pure,
    I/O-free code (agent/report_flow.py) before this function is ever
    reached -- a failure HERE can at worst mean a denial goes
    unrecorded, never that it gets turned into a grant. This mirrors
    every other record_*() function in this file, none of which are
    ever allowed to influence the outcome they are only recording.
    """
    with _lock:
        _denied_counts[intent] = _denied_counts.get(intent, 0) + 1
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": REPORT_ACCESS_DENIED_EVENT,
            "intent": intent,
            "reason": reason,
            "patient_id": patient_id,
            "call_id": call_id,
            "turn_index": turn_index,
        }
        log_dir = os.path.dirname(ESCALATION_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(ESCALATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def report_access_denied_counts(intent: str | None = None) -> dict[str, int] | int:
    """Per-intent report-access-denial occurrence COUNTS -- mirrors
    human_handoff_counts() above exactly, but reads the separate
    _denied_counts dict, never _counts or _handoff_counts."""
    with _lock:
        if intent is not None:
            return _denied_counts.get(intent, 0)
        return dict(_denied_counts)


# =============================================================================
# ADDED BY SOURAV -- "Caller states something the agent cannot verify"
# story. AC: "A caller assertion never becomes a system fact... where the
# system cannot verify the claim, the agent clearly says it cannot
# confirm it."
#
# A SEVENTH event in the same ESCALATION_LOG_PATH ledger, distinguished
# from every other event here by its own "event": "unverifiable_claim"
# value -- same discipline record_report_access_denied() above already
# established: one shared, durable, greppable JSONL file, one process-
# wide lock, its own counter dict (_unverifiable_claim_counts) kept
# SEPARATE from every other counter here so no two different things are
# ever added together.
#
# Called from main.py's/main_pcm.py's _finish_unverifiable_claim() --
# the ONE place agent/unverifiable_claim.py's detect_unverifiable_claim()
# guard hands off to once it fires, whether that happened on a fresh turn
# (_resolve_intent()) or mid-flow (_continue_pending()). Deliberately
# logs ONLY a fixed category reason code (one of
# agent/unverifiable_claim.py's CATEGORY_* constants), never the caller's
# claim text itself -- per this story's own explicit instruction ("do not
# store unnecessary caller free text"), matching record_complaint_filed()'s
# and record_report_access_denied()'s own already-established discipline
# in this exact file.
# =============================================================================

UNVERIFIABLE_CLAIM_EVENT = "unverifiable_claim"

_unverifiable_claim_counts: dict[str, int] = {}


def record_unverifiable_claim(intent: str, reason: str, call_id: str | None = None,
                               turn_index: int | None = None) -> None:
    """Records one "-> caller asserted something this system cannot
    verify" event. Bumps a per-`reason` count
    (unverifiable_claim_counts() below) and appends one JSON line to the
    shared ESCALATION_LOG_PATH ledger.

    `intent` is always "unverifiable_claim" today (the one guard-fired
    intent this event is ever recorded for) -- accepted as a parameter
    rather than hardcoded so this function's shape matches every other
    record_*() function in this file (`intent` keyed) and stays open to a
    second caller without a signature change, exactly as
    record_human_handoff()'s own `intent` parameter already is. `reason`
    is a fixed, non-sensitive category code from
    agent/unverifiable_claim.py's CLAIM_PATTERNS (e.g.
    "APPOINTMENT_EXISTING_CLAIM") -- never a sentence built from caller
    speech.

    CALLER RESPONSIBILITY, not enforced here by design: main.py/
    main_pcm.py must call this from inside their own try/except (see
    main.py's _finish_unverifiable_claim()) -- same discipline
    record_report_access_denied()'s own docstring already establishes,
    and for the identical reason: the fixed "I can't confirm that" reply
    must be spoken, and the guard's decision must stand, regardless of
    whether this audit write succeeds. A failure HERE can at worst mean
    an unverifiable-claim event goes unrecorded, never that the claim
    becomes confirmed or the reply changes.
    """
    with _lock:
        _unverifiable_claim_counts[reason] = _unverifiable_claim_counts.get(reason, 0) + 1
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": UNVERIFIABLE_CLAIM_EVENT,
            "intent": intent,
            "reason": reason,
            "call_id": call_id,
            "turn_index": turn_index,
        }
        log_dir = os.path.dirname(ESCALATION_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(ESCALATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def unverifiable_claim_counts(reason: str | None = None) -> dict[str, int] | int:
    """Per-category unverifiable-claim occurrence COUNTS -- mirrors
    report_access_denied_counts() above exactly, but reads the separate
    _unverifiable_claim_counts dict, keyed by reason code rather than
    intent (every occurrence today shares the same "unverifiable_claim"
    intent, so counting by intent alone would collapse all four claim
    categories into one indistinguishable number)."""
    with _lock:
        if reason is not None:
            return _unverifiable_claim_counts.get(reason, 0)
        return dict(_unverifiable_claim_counts)


def _reset_for_testing() -> None:
    """Test-only: clear the in-process counters between test cases. Never
    called from production code -- tests import and call this explicitly
    in a fixture, the same pattern as clearing any other module-level
    cache in a test suite."""
    global _total_turns, _abandonment_count
    with _lock:
        _counts.clear()
        _handoff_counts.clear()
        _denied_counts.clear()
        _unverifiable_claim_counts.clear()
        _total_turns = 0
        _abandonment_count = 0
