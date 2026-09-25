"""ADDED BY SOURAV -- Story 5: "Caller asks about another person's
report" (As a patient, I want my results disclosed only to me or an
authorised person, so that a phone line is not the weak point in my
privacy).

WHY THIS IS ITS OWN MODULE: the investigation done before this story was
implemented (see the delivered inspection report,
"Another_person_report_privacy_investigation_report.txt") found that
this codebase had exactly ONE mechanism standing in for both "who is
this" and "are they allowed to hear this": a caller-spoken phone number
matched against Patient.phone, which happens to behave like an
authorization check ONLY because that column carries a `unique=True`
constraint (see clinic-api/models.py) -- not because any code anywhere
expresses authorization as its own, separate, enforced decision.

This module gives that decision a real, separate, testable, code-only
seam, per this story's own explicit AC:
  "1. Disclosure requires verification of the caller AND an
      authorisation check for that patient."
  "2. Both checks must be enforced in the gateway/code, NOT by the
      model."

Both functions below are pure -- no I/O, no CallSession, no clinic-api
call, no model involvement of any kind -- matching this codebase's own
established discipline for shared decision logic (see
agent/report_flow.py's module docstring on why business rules live in
one place shared by main.py/main_pcm.py rather than duplicated).
Neither function is ever given anything the model extracted from
caller speech (e.g. agent/llm.py's "patient_name" slot) -- both only
ever see identifiers clinic-api already resolved deterministically.

WHAT THIS DOES NOT DO, ON PURPOSE: this module does not invent a
shared-phone-number chooser or an authorized-person/proxy/guardian
model. The investigation found no existing, documented product policy
for either (no such concept exists anywhere in clinic-api/models.py,
and no plan/story document in this repo defines one) -- inventing one
here would be exactly the "invented policy" the story explicitly
forbade. authorize_report_access() below is deliberately written as a
narrow, honest, currently-mostly-trivial identity-consistency check
(see its own docstring for exactly what it can and cannot catch today)
rather than a fabricated stand-in for either feature. It is, however,
the single seam either feature would extend once a real policy for
them exists -- callers pass it two patient ids today; a future
proxy/shared-phone feature would still call it with two patient ids
(the verified caller's, and the one on whose behalf they claim to be
authorized), it would just resolve `verified_patient_id` differently
upstream.
"""
from __future__ import annotations


def resolve_verified_patient_id(result: dict) -> int | None:
    """The entire "verification" step this story requires, made
    explicit: clinic-api has already, deterministically and only in
    code (see clinic-api/main.py's report_status() /
    request_report_delivery() / verify_report_otp(), and
    clinic-api/models.py's Patient.phone unique=True constraint), taken
    the phone number the caller spoke and resolved it to AT MOST ONE
    Patient row's id. This function just reads that id back out of
    whichever of clinic-api's three report-related response shapes is
    passed in, or returns None when no identity was established at all.

    Works uniformly across all three response shapes this story's
    flows use, because each one has its own success indicator:
      - report_status():             {"patient_found": bool, ...}
      - request_report_delivery():   {"success": bool, ...}
      - verify_report_otp():         {"success": bool, ...}
    Never given raw caller speech, an LLM slot, or anything the model
    produced -- only an already-resolved clinic-api response dict.
    """
    if result.get("patient_found") is False:
        return None
    if result.get("success") is False:
        return None
    return result.get("patient_id")


def authorize_report_access(verified_patient_id: int | None, target_patient_id: int | None) -> bool:
    """The SEPARATE, second gate this story's AC1 requires: proving a
    phone resolves to *a* patient (verification, above) is not the same
    claim as "the caller is authorized to hear THIS patient's report"
    (authorization). This function is where that second claim is
    actually checked, in code, so it exists as a real decision a test
    can call directly -- not an assumption buried inside a dispatch
    branch.

    Returns True only when both ids are known and identical.

    HONEST LIMITATION, stated once here rather than left implicit:
    under this codebase's CURRENT schema, Patient.phone is unique, so
    within a single, un-tampered turn `target_patient_id` is always
    sourced from the exact same phone resolution that produced
    `verified_patient_id` -- there is no real caller utterance today
    that can make these two values diverge. This function is still
    real and still worth having, for two reasons:

      1. Every call site in agent/report_flow.py threads
         `verified_patient_id` forward, turn to turn, from the FIRST
         phone resolution in a report_status/report_send flow (see
         that module's AWAITING_WHICH_REPORT / AWAITING_CONFIRM_DELIVERY
         / AWAITING_OTP_CODE pending dicts, each of which now carries
         "verified_patient_id"). Every later turn's result is checked
         against THAT original value, not merely against itself -- so
         this genuinely guards against the patient identity silently
         changing mid-flow (a pending-state bug, a future code change
         that reuses these pending states for a different purpose,
         etc.), even though no code path can trigger that today.
      2. It is the one seam a future shared-phone-number chooser or
         authorized-person/proxy feature (neither built here -- see
         this module's own docstring) would have to route through:
         whoever builds either feature only has to make
         `verified_patient_id` resolve correctly for their new case;
         this function's contract does not change.

    Never given an LLM-extracted value for either argument.
    """
    if verified_patient_id is None or target_patient_id is None:
        return False
    return verified_patient_id == target_patient_id
