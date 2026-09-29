"""Shared decision logic for the "Lab Report Status & Secure Delivery"
combined story's multi-turn flow.

ADDED BY SOURAV -- new file for this combined story (previously two
separate stories: "is my report ready" and "send my report"), per the
VOICE CARE AGENT FINAL TESTING / STORY / RULES / EDGE-CASE ATTACK PLAN.

WHY THIS EXISTS AS ITS OWN MODULE, NOT WRITTEN TWICE INLINE IN
main.py / main_pcm.py:
Those two files each implement their own COMPLETE copy of _dispatch_turn
and _continue_pending (see main_pcm.py's own docstring on
_continue_pending for why two copies exist at all: main.py speaks WAV,
main_pcm.py speaks raw PCM, over otherwise-identical turn logic). That
duplication already drifted once for real: the "test_sample" intent
(Story 5, "Caller asks what sample is needed") was wired into
main_pcm.py's dispatch but was never mirrored into main.py's, so a
caller on the WAV transport (ports 8080/8100 per deploy/start_all.sh)
asking only about sample type hit no matching branch at all, even
though agent/llm.py -- shared by both transports -- classifies that
intent correctly either way. This story adds THREE new intents
(report_status, report_send, plus OTP verification as its own pending
state) and FIVE new pending states across both files; hand-duplicating
that much branching logic a second time is exactly how the test_sample
gap happens again, just at several times the size. So the actual
DECISIONS -- what to say, what state to enter next -- live here, in one
place, as plain functions with no dependency on CallSession, WebSocket,
or asyncio. main.py and main_pcm.py each call these functions and are
responsible only for the transport-specific parts: awaiting the tools
client, calling their own _speak(), and reading/writing
session.pending. A change to the business rules therefore only has to
happen once, here, to apply identically to both transports -- which is
also exactly what plan RULE 18 requires ("language/channel must not
change authorization").

Every function below is a pure function: (some result dict, maybe some
already-known slots) in, (reply text or None, new pending dict or None)
out. None of them perform I/O, call the tools client, or touch a
CallSession -- that keeps them trivially unit-testable without any of
main.py/main_pcm.py's WebSocket/asyncio machinery, and it is what makes
"the backend decision must remain the same" (plan Section 2) something
this test suite can actually assert directly, instead of only ever
observing it indirectly through two different dispatch files.
"""
from __future__ import annotations

import difflib

from agent.reply_templates import (
    patient_not_found_reply,
    report_not_found_reply,
    report_ambiguous_reply,
    report_status_reply,
    delivery_blocked_reply,
    otp_requested_reply,
    otp_verify_reply,
    report_access_denied_reply,
)
from agent.report_access_control import resolve_verified_patient_id, authorize_report_access

# ADDED BY SOURAV -- Story 5 ("Caller asks about another person's
# report"). The sentinel main.py/main_pcm.py recognize, the exact same
# way they already recognize "__request_delivery_now__" below: "pure
# function says do exactly this next," here meaning "stop, this must
# not be disclosed." See main.py's _apply_report_outcome() for the one
# place this sentinel is ever handled -- audit the denial (best-effort;
# see agent/outcomes.record_report_access_denied()'s own docstring on
# why an audit failure must never change the decision) and speak
# report_access_denied_reply(), never report_status_reply() or any
# other reply that would confirm or deny what was actually found.
REPORT_ACCESS_DENIED_AWAITING = "__report_access_denied__"


def _denied(reason: str, patient_id: int | None) -> dict:
    return {"awaiting": REPORT_ACCESS_DENIED_AWAITING, "reason": reason, "patient_id": patient_id}

# Pending "awaiting" values this module introduces, mirroring the
# existing booking flow's convention (see main_pcm.py's
# _continue_pending docstring for the full pending-dict shape). Not an
# enum -- this codebase's existing pending states are also plain
# strings -- kept here as named constants purely so a typo in one of
# these strings is a NameError in code that imports them, not a silent
# "awaiting state that never matches anything" bug at runtime.
# NOT "phone" -- the pre-existing booking flow already uses that exact
# string as an "awaiting" value (see main.py's _continue_pending: a
# caller correcting the phone field of an in-progress BOOKING re-enters
# awaiting="phone" through the shared date/time_slot/patient_name/phone
# tail). Reusing the same string here was caught by test_booking_readback.py
# failing after this story's states were wired in -- a caller correcting a
# BOOKING's phone number was being routed into the REPORT flow's phone
# handler instead, because _continue_pending's new report-flow check ran
# first and matched on the bare string. Named distinctly so the two flows'
# pending states can never collide again, no matter which is checked first.
AWAITING_REPORT_PHONE = "report_phone"      # RULE 15: identity by phone first
AWAITING_WHICH_REPORT = "which_report"      # RULE 13: multiple reports -> ask
AWAITING_CONFIRM_DELIVERY = "confirm_delivery"  # the "would you like it sent?" offer
AWAITING_OTP_CODE = "otp_code"              # RULE 4-9: OTP verification


def interpret_report_status_result(result: dict, flow: str, language: str = "bengali") -> tuple[str, dict | None]:
    """Called right after tools_client.get_report_status() returns, for
    BOTH "report_status" and "report_send" (flow tells the two apart --
    same lookup, different thing to do once a READY+enabled report is
    found).

    Returns (reply_text, new_pending). new_pending is None whenever the
    flow ends here (nothing left to ask); the caller (main.py /
    main_pcm.py) is responsible for setting session.pending to whatever
    is returned, or None.
    """
    if not result.get("patient_found"):
        # RULE 1's identity equivalent: never guess who the caller is.
        return patient_not_found_reply(language), None

    # ADDED BY SOURAV -- Story 5's AC1/AC4 gateway: verification
    # (resolve_verified_patient_id -- phone-to-patient resolution,
    # clinic-api's job, never the model's) and authorization
    # (authorize_report_access -- a SEPARATE, explicit check) both run
    # here, in code, BEFORE anything about the report is spoken.
    # `verified_patient_id` either comes from the CALLER (main.py
    # carries it forward on `result["verified_patient_id"]` when this
    # is a "which_report" disambiguation re-entry -- see that pending
    # dict below and main.py's own comment at that call site) or, on a
    # flow's first turn, IS this same lookup's own patient_id -- see
    # agent/report_access_control.py's docstring for exactly what this
    # can and cannot catch under today's schema, and why it is still a
    # real, separate, tested gate rather than a no-op.
    resolved_patient_id = result.get("patient_id")
    verified_patient_id = result.get("verified_patient_id", resolved_patient_id)
    if not authorize_report_access(verified_patient_id, resolved_patient_id):
        return None, _denied("VERIFIED_IDENTITY_MISMATCH", resolved_patient_id)

    if not result.get("found"):
        reason = result.get("reason")
        if reason == "AMBIGUOUS":
            # RULE 13.
            return report_ambiguous_reply(result, language), {
                "awaiting": AWAITING_WHICH_REPORT, "flow": flow,
                "candidates": result.get("candidates") or [], "retries": 0,
                "verified_patient_id": verified_patient_id,
            }
        return report_not_found_reply(language), None  # RULE 1.

    status = result["status"]
    delivery_enabled = bool(result.get("delivery_enabled"))

    if status != "READY":
        if flow == "report_send":
            # Caller directly asked for delivery of a report that isn't
            # eligible -- RULE 2/RULE 3: explain, do not start OTP.
            return delivery_blocked_reply(status, language), None
        # flow == "report_status": just answer the status question.
        return report_status_reply(result, language), None

    if not delivery_enabled:
        if flow == "report_send":
            return delivery_blocked_reply("DELIVERY_DISABLED", language), None  # RULE 16.
        return report_status_reply(result, language), None

    # READY and delivery-enabled from here on.
    if flow == "report_send":
        # Caller already asked for delivery -- skip the offer question
        # and go straight to requesting it. The transport file still has
        # to actually AWAIT tools_client.request_report_delivery() and
        # then call interpret_delivery_request_result() below with its
        # response before it can compose a reply or set otp_code pending
        # -- that I/O step cannot live in this pure function. This
        # sentinel pending tells the transport file exactly one thing to
        # do next, so there is nothing left to branch on there either.
        return None, {"awaiting": "__request_delivery_now__", "report_number": result["report_number"],
                       "verified_patient_id": verified_patient_id}

    # flow == "report_status": offer, don't send yet (RULE 4 -- OTP/
    # delivery only ever starts after an explicit yes, tracked by
    # AWAITING_CONFIRM_DELIVERY below).
    #
    # ADDED BY SOURAV -- KCD-383: "test_name" is carried on the pending
    # dict (never previously needed here, since only report_status_reply()
    # above read it) so that a caller who answers this offer with a
    # callback preference instead of yes/no can be routed into the
    # request_callback flow with an honest, already-known reason attached
    # (see main.py's _pivot_report_offer_to_callback(), which reads this
    # exact key -- never re-asks the caller which report they meant).
    return report_status_reply(result, language), {
        "awaiting": AWAITING_CONFIRM_DELIVERY,
        "report_number": result["report_number"],
        "test_name": result.get("test_name"),
        "retries": 0,
        "verified_patient_id": verified_patient_id,
    }


def interpret_delivery_request_result(result: dict, report_number: str, language: str = "bengali",
                                       verified_patient_id: int | None = None) -> tuple[str, dict | None]:
    """Called after tools_client.request_report_delivery() returns --
    either because the caller said yes to the AWAITING_CONFIRM_DELIVERY
    offer, or because interpret_report_status_result() above returned
    the "__request_delivery_now__" sentinel for a "report_send" flow.

    `report_number` is passed in explicitly (not read off `result`) --
    clinic-api's delivery/request response never echoes it back (see
    clinic-api/main.py's request_report_delivery(): it only ever returns
    success/reason/masked_phone), so the caller here is the only place
    that still knows which report this request was for; it has to carry
    that value forward into the next pending state itself.

    `verified_patient_id` -- ADDED BY SOURAV, Story 5 -- is the id
    interpret_report_status_result() established (or re-confirmed) on
    this same flow's earlier turn(s), carried forward on the pending
    dict's own "verified_patient_id" key (see main.py's
    _apply_report_outcome()). Optional and defaulting to None purely so
    every pre-existing caller/test of this function keeps working
    unchanged; when it IS given, the SAME authorize_report_access()
    gate interpret_report_status_result() already ran is re-asserted
    here, right before an OTP is even requested -- see
    agent/report_access_control.py's docstring for why this is a real,
    if today mostly-invariant, check rather than a no-op."""
    if result.get("success"):
        resolved_patient_id = result.get("patient_id")
        if verified_patient_id is not None and not authorize_report_access(verified_patient_id, resolved_patient_id):
            return None, _denied("VERIFIED_IDENTITY_MISMATCH", resolved_patient_id)
        pending = {
            "awaiting": AWAITING_OTP_CODE, "report_number": report_number, "retries": 0,
            "verified_patient_id": verified_patient_id if verified_patient_id is not None else resolved_patient_id,
        }
        return otp_requested_reply(result, language), pending

    reason = result.get("reason")
    if reason == "PATIENT_NOT_FOUND":
        return patient_not_found_reply(language), None
    if reason == "NOT_FOUND":
        return report_not_found_reply(language), None
    # NOT_READY / PROCESSING / CANCELLED / DELIVERY_DISABLED -- the
    # report's state changed between the status check and this request
    # (RULE 3/16 re-checked, same as clinic-api's own defense in depth).
    return delivery_blocked_reply(reason, language), None


def interpret_otp_verify_result(result: dict, report_number: str, language: str = "bengali",
                                 verified_patient_id: int | None = None) -> tuple[str, dict | None]:
    """Called after tools_client.verify_report_otp() returns. On
    OTP_INVALID, stays in the SAME otp_code pending state so the caller
    can try again -- bounded by the transport file's own
    pending["retries"] counter (same convention as every other awaiting
    state in main_pcm.py's _continue_pending), which is a SEPARATE,
    session-local backstop from clinic-api's own server-side
    attempt_count/max_attempts (RULE 8) -- the server's OTP_MAX_ATTEMPTS
    is authoritative and always ends the flow here regardless of the
    local retry counter, so a caller can never out-wait a local retry
    cap to get more real attempts than the server allows.

    `verified_patient_id` -- ADDED BY SOURAV, Story 5 -- same meaning
    and default as interpret_delivery_request_result()'s own parameter
    above: carried forward from this same flow's earlier turns, and
    re-asserted against clinic-api's own patient_id on a successful
    verification, one last time, immediately before the flow ends by
    reporting delivery as sent. See agent/report_access_control.py.
    """
    reply = otp_verify_reply(result, language)
    if result.get("reason") == "OTP_INVALID":
        return reply, {"awaiting": AWAITING_OTP_CODE, "report_number": report_number, "retries": 0,
                        "verified_patient_id": verified_patient_id}
    if result.get("success") and verified_patient_id is not None:
        resolved_patient_id = result.get("patient_id")
        if not authorize_report_access(verified_patient_id, resolved_patient_id):
            return None, _denied("VERIFIED_IDENTITY_MISMATCH", resolved_patient_id)
    # Every other reason (success, expired, used, maxed, not_ready,
    # disabled, delivery failed, not found, patient not found) ends the
    # flow -- none of them are something retrying the SAME OTP fixes.
    return reply, None


# RULE 13's disambiguation match. Mirrors main_pcm.py's own
# _match_candidate_doctor (same scoring shape: difflib ratio, boosted to
# 0.85 on a substring hit, committed only above a floor) -- kept here
# instead of duplicated into main.py/main_pcm.py a second time, for the
# same drift reason as everything else in this module.
_REPORT_MATCH_FLOOR = 0.55


def match_candidate_report(text: str, candidates: list[dict]) -> str | None:
    """-> the `report_number` of the candidate the caller just named out
    of a list session.pending offered a moment ago (RULE 13's "ask,
    using safe identifying information"), or None if the utterance does
    not confidently match exactly one."""
    if not candidates:
        return None
    norm_text = (text or "").strip().lower()
    if not norm_text:
        return None
    best_number, best_score = None, 0.0
    for c in candidates:
        test_name = (c.get("test_name") or "").lower()
        if not test_name:
            continue
        score = difflib.SequenceMatcher(None, test_name, norm_text).ratio()
        if test_name in norm_text or norm_text in test_name:
            score = max(score, 0.85)
        if score > best_score:
            best_number, best_score = c.get("report_number"), score
    return best_number if best_score >= _REPORT_MATCH_FLOOR else None
