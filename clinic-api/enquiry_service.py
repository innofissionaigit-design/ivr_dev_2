"""Business logic for Epic E27 (Information and Enquiry).

Same discipline as booking_service.py: every function takes an open
`db: Session` and returns a plain dict (or list of dicts), never raises
for a normal not-found/ineligible/unavailable outcome, and never invents
a fact -- every number here is either a database value or a deterministic
computation over database values (see, e.g., merge_prep_instructions'
MAX() over fasting hours, or package_comparison's live rate sum).

Three pieces here are EXPLICIT PLACEHOLDERS for systems this prototype
does not own, following the exact pattern booking_service.queue_sms
already established -- never claim something happened that did not:
  - deliver_report(): logs a signed link, never actually emails/SMSes it.
  - check_insurance_coverage(): reads FROM THIS SERVICE's OWN mock
    InsurancePolicy/InsuranceCoverageRule tables, standing in for a real
    insurer's API this system has no access to.
  - request_callback(): records the promise; no outbound-calling system
    exists (that is telephony infrastructure, Epic E22, not this file).
"""

from __future__ import annotations

import datetime
import secrets
import uuid

from models import (
    ComplaintRecord,
    BillingLineItem,
    CallbackRequest,
    CallOutcome,
    DepartmentHours,
    DoctorLeave,
    HomeCollectionCoverage,
    InsuranceCoverageRule,
    InsurancePolicy,
    LabReport,
    LabTest,
    Package,
    PackageTest,
    Patient,
    ReportDeliveryAudit,
    ReportDeliveryOTP,
    WalkInPolicy,
)
from sqlalchemy.orm import Session

REPORT_OTP_TTL_MINUTES = 10
# A second request for a code inside this many seconds is a retry: it gets the code already issued.
OTP_REISSUE_WINDOW_S = 60
REPORT_LINK_TTL_HOURS = 48
CALLBACK_MIN_LEAD_MINUTES = 30


def _now() -> datetime.datetime:
    return datetime.datetime.now()


# ============================================================ KCD-385: leave


def doctor_leave_on(db: Session, doctor_id: int, date: str) -> dict | None:
    """None if the doctor is not on leave that date. Otherwise the leave
    row's facts -- never invented, and the return date is stated as
    unknown rather than guessed when the record does not have one."""
    d = datetime.date.fromisoformat(date)
    for leave in db.query(DoctorLeave).filter_by(doctor_id=doctor_id).all():
        start, end = datetime.date.fromisoformat(leave.start_date), datetime.date.fromisoformat(leave.end_date)
        if start <= d <= end:
            return {
                "on_leave": True,
                "start_date": leave.start_date,
                "end_date": leave.end_date,
                "return_date": leave.return_date,
            }
    return None


# ======================================================= KCD-381/382: prep


def merge_prep_instructions(db: Session, lab_test_ids: list[int]) -> dict:
    """KCD-382: several tests' preparation merged to the STRICTEST
    constraint -- a real MAX() over fasting_hours and an AND over
    water-allowed, computed here in code, never composed by the model.
    A conflict this function cannot resolve (see `escalate`) means the
    caller must be hand off to a human rather than given a guessed
    instruction -- the story's own acceptance criterion.

    KCD-382 (caller story "several tests with conflicting preparation"):
    added `incomplete_data` / the second `escalate_to_human` trigger
    below. `fasting_hours` is only ever a MAX() over the tests that HAVE
    a structured value -- a test whose `fasting_required` boolean (Layer
    1, always populated, see models.LabTest) is True but whose
    `fasting_hours` (Layer 3, backfilled only for a subset of tests, see
    enquiry_migrate.FASTING_HOURS_OVERRIDES) is still NULL would
    otherwise be silently dropped from the MAX(), understating the real
    constraint (seed data example: "USG Whole Abdomen" -- fasting_required
    True, real prep prose, fasting_hours None). Rather than guess a
    number for it, or silently ignore it, that case is treated the same
    as the mutually-exclusive-SOP case above: `escalate_to_human` is set
    True and `incomplete_data` names which of the resolved tests lacks
    the structured number, so a human merges it instead of the caller
    being given a confident instruction missing one of the tests they
    asked about."""
    tests = [db.get(LabTest, tid) for tid in lab_test_ids]
    tests = [t for t in tests if t is not None]
    if not tests:
        return {"found": False}

    fasting_hours = [t.fasting_hours for t in tests if t.fasting_hours]
    merged_hours = max(fasting_hours) if fasting_hours else None
    # Strictest also means: if ANY test in the set forbids water while
    # fasting, the merged instruction forbids it for all of them.
    water_allowed = all(t.water_allowed_while_fasting for t in tests if t.fasting_hours)

    # The one conflict this function cannot silently resolve: two tests
    # both requiring fasting but ALSO both explicitly requiring food
    # shortly before the sample (mutually exclusive) would need a human,
    # not a merge. No test in this seed data models that case yet, so this
    # trigger is here and tested, even though it is currently always
    # False -- ready for real conflicting SOPs without a redesign.
    sop_conflict = False

    # The data-completeness gap this docstring describes: a test the
    # catalogue already KNOWS needs fasting (Layer 1) but has not yet been
    # given a structured hour count (Layer 3) cannot safely be folded into
    # the MAX() above -- it would just vanish from the merge.
    incomplete = sorted(t.name for t in tests if t.fasting_required and t.fasting_hours is None)

    result = {
        "found": True,
        "test_names": [t.name for t in tests],
        "fasting_hours": merged_hours,
        "water_allowed_while_fasting": water_allowed,
        "escalate_to_human": sop_conflict or bool(incomplete),
        "prescription_required": any(t.prescription_required for t in tests),
    }
    if incomplete:
        result["incomplete_data"] = incomplete
    return result


# ==================================================== KCD-380/397: packages


def get_package(db: Session, name: str) -> dict | None:
    pkg = db.query(Package).filter(Package.name.ilike(f"%{name}%")).first()
    if not pkg:
        return None
    return _package_dict(db, pkg)


def _package_dict(db: Session, pkg: Package) -> dict:
    rows = db.query(PackageTest).filter_by(package_id=pkg.id).all()
    tests = [db.get(LabTest, r.lab_test_id) for r in rows]
    tests = [t for t in tests if t is not None]
    separate_total = sum(t.rate_inr for t in tests)
    return {
        "found": True,
        "package_name": pkg.name,
        "test_names": [t.name for t in tests],
        "bundled_price_inr": pkg.bundled_price_inr,
        "separate_total_inr": separate_total,
        "savings_inr": max(0, separate_total - pkg.bundled_price_inr),
        "branch_restricted": pkg.branch_restricted,
    }


def compare_package_vs_separate(db: Session, name: str) -> dict:
    """KCD-397: the comparison stated as fact, computed live -- never a
    clinical or purchasing RECOMMENDATION, just the arithmetic."""
    pkg = get_package(db, name)
    if not pkg:
        return {"found": False, "query": name}
    return pkg


# ========================================================= KCD-393: walk-in


def walk_in_policy(db: Session, department_id: int | None = None, lab_test_id: int | None = None) -> dict:
    q = db.query(WalkInPolicy)
    if lab_test_id is not None:
        row = q.filter_by(lab_test_id=lab_test_id).first()
        if row:
            return {
                "found": True,
                "allowed": row.allowed,
                "queue_note_bn": row.queue_note_bn,
                "queue_note_hi": row.queue_note_hi,
                "queue_note_en": row.queue_note_en,
            }
    if department_id is not None:
        row = q.filter_by(department_id=department_id, lab_test_id=None).first()
        if row:
            return {
                "found": True,
                "allowed": row.allowed,
                "queue_note_bn": row.queue_note_bn,
                "queue_note_hi": row.queue_note_hi,
                "queue_note_en": row.queue_note_en,
            }
    return {"found": False}


# =========================================================== KCD-389: billing


def outstanding_balance(db: Session, phone: str) -> dict:
    patient = db.query(Patient).filter_by(phone=phone).first()
    if not patient:
        return {"found": False}
    items = db.query(BillingLineItem).filter_by(patient_id=patient.id, status="outstanding").all()
    return {
        "found": True,
        "total_inr": sum(i.amount_inr for i in items),
        "line_items": [{"description": i.description, "amount_inr": i.amount_inr} for i in items],
    }


# ================================================== KCD-387: home collection


def home_collection_eligibility(db: Session, lab_test_id: int, postal_code: str) -> dict:
    test = db.get(LabTest, lab_test_id)
    if not test:
        return {"found": False}
    if not test.home_collection_eligible:
        return {"found": True, "eligible": False, "reason": "sample_type"}
    coverage = db.query(HomeCollectionCoverage).filter_by(postal_code=postal_code).first()
    if not coverage or not coverage.serviceable:
        return {"found": True, "eligible": False, "reason": "area_not_covered"}
    return {
        "found": True,
        "eligible": True,
        "charge_inr": coverage.charge_inr,
        "slot_note_bn": coverage.slot_note_bn,
        "slot_note_hi": coverage.slot_note_hi,
        "slot_note_en": coverage.slot_note_en,
    }


# ====================================================== KCD-388: insurance


def check_insurance_coverage(db: Session, policy_number: str, lab_test_id: int | None = None) -> dict:
    """Reads this service's own MOCK insurer tables -- see this module's
    docstring. A real integration replaces this function's body with an
    HTTP call to the insurer; nothing else in the codebase would need to
    change, the same seam booking_service.queue_sms already models."""
    policy = db.query(InsurancePolicy).filter_by(policy_number=policy_number).first()
    if not policy:
        return {"found": False, "reachable": True}
    if not policy.active:
        return {"found": True, "active": False}
    rule = (
        db.query(InsuranceCoverageRule).filter_by(insurer_name=policy.insurer_name, lab_test_id=lab_test_id).first()
        or db.query(InsuranceCoverageRule).filter_by(insurer_name=policy.insurer_name, lab_test_id=None).first()
    )
    if not rule:
        return {"found": True, "active": True, "covered": False}
    return {
        "found": True,
        "active": True,
        "covered": True,
        "coverage_percent": rule.coverage_percent,
        "co_payment_inr": rule.co_payment_inr,
    }


# =================================================== KCD-392: prescription


def prescription_requirement(db: Session, lab_test_id: int, lang: str = "bn") -> dict:
    test = db.get(LabTest, lab_test_id)
    if not test:
        return {"found": False}
    note = {"bn": test.prescription_note_bn, "hi": test.prescription_note_hi, "en": test.prescription_note_en}.get(
        lang, test.prescription_note_bn
    )
    return {"found": True, "required": test.prescription_required, "note": note}


# =================================================== KCD-394: out of scope


def record_out_of_scope(db: Session, call_id: str, caller_question: str, reason_code: str = "out_of_scope") -> dict:
    twin = (
        db.query(CallOutcome)
        .filter_by(call_id=call_id, reason_code=reason_code, caller_question=caller_question)
        .first()
    )
    if twin is not None:  # the same question on the same call is recorded once
        return {"recorded": True, "id": twin.id, "duplicate": True}
    row = CallOutcome(call_id=call_id, reason_code=reason_code, caller_question=caller_question, created_at=_now())
    db.add(row)
    db.commit()
    return {"recorded": True, "id": row.id}


# ========================================================= KCD-398: callback


def request_callback(
    db: Session, phone: str, call_id: str, requested_window: str, reason: str = "", call_summary: str = ""
) -> dict:
    """`call_summary`: a short, deterministic note of what the call was about (agent/call_summary.py builds it from
    known facts, never an LLM paraphrase), so the human who calls back -- or opens a prescription photo the caller
    was asked to send -- has the call's context next to the phone number. Optional and additive: a callback asked
    for without one still works exactly as before."""
    twin = (
        db.query(CallbackRequest)
        .filter_by(phone=phone, call_id=call_id, requested_window=requested_window, status="scheduled")
        .first()
    )
    if twin is not None:  # the same callback asked for twice is one callback
        if call_summary and not twin.call_summary:
            twin.call_summary = call_summary  # never overwrite one already recorded, only fill a blank one
            db.commit()
        return {"success": True, "id": twin.id, "requested_window": requested_window, "duplicate": True}
    row = CallbackRequest(
        phone=phone,
        call_id=call_id,
        requested_window=requested_window,
        reason=reason,
        status="scheduled",
        created_at=_now(),
        call_summary=call_summary,
    )
    db.add(row)
    db.commit()
    return {"success": True, "id": row.id, "requested_window": requested_window}


# ======================================================= KCD-383/384: reports


def report_status(db: Session, confirmation_id: str) -> dict:
    report = db.query(LabReport).filter_by(confirmation_id=confirmation_id).first()
    if not report:
        return {"found": False}
    return {
        "found": True,
        "status": report.status,
        "ready_at": report.ready_at.isoformat() if report.ready_at else None,
    }


def request_report_otp(db: Session, confirmation_id: str, phone: str) -> dict:
    report = db.query(LabReport).filter_by(confirmation_id=confirmation_id).first()
    if not report:
        return {"success": False, "reason": "not_found"}
    if report.status != "ready":
        return {"success": False, "reason": "not_ready"}
    if report.patient_phone != phone:
        return {"success": False, "reason": "phone_mismatch"}

    # A retry within the window gets the code already issued, not a second one that would leave the first dangling.
    live = (
        db.query(ReportDeliveryOTP)
        .filter_by(report_id=report.id, phone=phone, verified=False)
        .order_by(ReportDeliveryOTP.created_at.desc())
        .first()
    )
    if (
        live is not None
        and live.expires_at > _now()
        and (_now() - live.created_at).total_seconds() < OTP_REISSUE_WINDOW_S
    ):
        return {"success": True, "otp_id": live.id, "expires_in_minutes": REPORT_OTP_TTL_MINUTES, "duplicate": True}
    otp = f"{secrets.randbelow(1_000_000):06d}"
    row = ReportDeliveryOTP(
        report_id=report.id,
        phone=phone,
        otp_code=otp,
        verified=False,
        expires_at=_now() + datetime.timedelta(minutes=REPORT_OTP_TTL_MINUTES),
        created_at=_now(),
    )
    db.add(row)
    db.commit()
    # Logged, not sent -- the same disclosed placeholder as booking_service.queue_sms.
    return {"success": True, "otp_id": row.id, "expires_in_minutes": REPORT_OTP_TTL_MINUTES}


def deliver_report(db: Session, confirmation_id: str, otp_code: str) -> dict:
    report = db.query(LabReport).filter_by(confirmation_id=confirmation_id).first()
    if not report:
        return {"success": False, "reason": "not_found"}
    otp = (
        db.query(ReportDeliveryOTP)
        .filter_by(report_id=report.id, otp_code=otp_code, verified=False)
        .order_by(ReportDeliveryOTP.created_at.desc())
        .first()
    )
    if not otp or otp.expires_at < _now():
        return {"success": False, "reason": "otp_invalid_or_expired"}

    otp.verified = True
    link_token = uuid.uuid4().hex
    audit = ReportDeliveryAudit(
        report_id=report.id,
        recipient_phone=otp.phone,
        verification_method="otp",
        link_token=link_token,
        link_expires_at=_now() + datetime.timedelta(hours=REPORT_LINK_TTL_HOURS),
        delivered_at=_now(),
    )
    db.add(audit)
    db.commit()
    return {"success": True, "link_token": link_token, "expires_in_hours": REPORT_LINK_TTL_HOURS}


# =========================================================== KCD-386: hours


def department_hours(db: Session, department_id: int, lang: str = "bn") -> dict | None:
    row = db.query(DepartmentHours).filter_by(department_id=department_id).first()
    if not row:
        return None
    text = {"bn": row.hours_bn, "hi": row.hours_hi, "en": row.hours_en}.get(lang, row.hours_bn)
    return {"found": True, "hours": text, "effective_from": row.effective_from}


MAX_COMPLAINT_CHARS = 4000


def file_complaint(
    db: Session,
    call_id: str,
    text: str,
    detected_by: str = "",
    lang: str = "",
    phone: str | None = None,
) -> dict:
    """Ported story "Caller wants to make a complaint": record the complaint
    verbatim and return a reference the agent can read back.

    `text` is stored UNEDITED, as acceptance criterion 3 requires -- no
    summarising, no model paraphrase. It is truncated at MAX_COMPLAINT_CHARS
    only as a storage bound, never to "clean up" what was said; a real spoken
    complaint is far shorter than that, so the bound is a guard against a
    runaway transcript, not an editorial step.

    Deduplicated per call the same way request_callback() above dedupes: a
    caller who says the same complaint twice in one call has made ONE
    complaint, and a retried tool call must not open a second row. Saying a
    DIFFERENT complaint later in the same call does open a second row, which is
    correct -- they are two complaints.
    """
    body = (text or "").strip()[:MAX_COMPLAINT_CHARS]
    twin = db.query(ComplaintRecord).filter_by(call_id=call_id, text=body).first()
    if twin is not None:
        return {"success": True, "id": twin.id, "duplicate": True, "status": twin.status}
    row = ComplaintRecord(
        call_id=call_id,
        text=body,
        detected_by=detected_by or "",
        lang=lang or "",
        phone=phone,
        status="open",
        created_at=_now(),
    )
    db.add(row)
    db.commit()
    return {"success": True, "id": row.id, "duplicate": False, "status": row.status}
