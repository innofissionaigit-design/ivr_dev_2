"""Booking-lifecycle business logic for Epic E26 (Booking, Rescheduling
and Cancellation).

Kept separate from clinic-api/main.py's route handlers so the
atomicity-critical parts -- hold_slot, confirm_booking,
reschedule_appointment -- can be unit tested directly, including under
real concurrency, without a FastAPI TestClient in the loop.

Every function takes an open `db: Session` and returns a plain dict --
same contract as clinic-api/main.py's existing endpoints -- and never
raises for a normal not-found/taken/policy-blocked outcome. Only
IntegrityError from a genuine slot race is caught here, in hold_slot;
anything else propagates, matching agent/tools_client.py's ToolCallError
convention on the caller side.

No fact in here is ever composed by an LLM: prices, charges and slot
availability come from this module's own deterministic policy constants
and the database, the same discipline agent/reply_templates.py already
applies to test prices and doctor schedules.
"""

from __future__ import annotations

import datetime
import uuid

from disclosure import with_notice as _with_disclosure_notice
from models import (
    Appointment,
    CancellationPolicy,
    Department,
    DepartmentRoute,
    Doctor,
    DoctorSchedule,
    DraftBooking,
    LabTest,
    Patient,
    PatientProxy,
    SlotLock,
    SmsOutbox,
    TestBooking,
)
from phrase_proximity import phrase_matches
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

HOLD_TTL_SECONDS = 90
DRAFT_BOOKING_TTL_MINUTES = 30
SMS_RESEND_MIN_INTERVAL_S = 60
# A repeat of the same message to the same number for the same booking within this many seconds is one message.
SMS_DEDUP_WINDOW_S = 60


def _now() -> datetime.datetime:
    return datetime.datetime.now()


def _confirmation_id(date: str) -> str:
    # 8 hex chars (32 bits, ~4.3 billion per date) rather than 4 (16 bits,
    # 65536): Epic E26 added /api/v1/bookings/lookup?confirmation_id=...,
    # an unauthenticated endpoint (this service has no auth layer at all
    # yet -- Epic E14) that returns patient name, doctor and time for a
    # match. 4 hex chars made that a realistically enumerable guess
    # against one date's bookings; 8 does not, without removing the
    # underlying "add real authentication" gap this alone cannot close.
    return f"KCD-{date.replace('-', '')}-{uuid.uuid4().hex[:8].upper()}"


def _generate_slots(start: str, end: str, step_min: int = 15) -> list[str]:
    t = datetime.datetime.strptime(start, "%H:%M")
    end_t = datetime.datetime.strptime(end, "%H:%M")
    out = []
    while t < end_t:
        out.append(t.strftime("%H:%M"))
        t += datetime.timedelta(minutes=step_min)
    return out


def _slot_minutes(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _alias(aliases: str | None) -> str | None:
    return aliases.split("|")[0] if aliases else None


# ============================================================= holds


def _release_expired_hold(db: Session, doctor_id: int, date: str, time_slot: str) -> None:
    # CodeRabbit-flagged, real race: sqlite3's legacy transaction mode
    # (db.py does not override it) does not BEGIN on a plain SELECT, so
    # two concurrent callers can both read the same expired row before
    # either writes. A read-then-delete-by-primary-key would let caller B
    # delete caller A's REPLACEMENT hold (inserted between B's read and
    # B's delete) instead of the expired one B actually read -- both
    # hold_slot calls report success, and A's later confirm_booking then
    # fails with hold_expired on a hold that should still be live. The
    # fix is a single DELETE whose WHERE clause re-checks status/expiry
    # at delete time, so a hold someone else already replaced is simply
    # not matched and stays in place; the duplicate insert that follows
    # in hold_slot then correctly reports slot_taken.
    db.query(SlotLock).filter(
        SlotLock.doctor_id == doctor_id,
        SlotLock.date == date,
        SlotLock.time_slot == time_slot,
        SlotLock.status == "held",
        SlotLock.hold_expires_at < _now(),
    ).delete()
    db.flush()


def hold_slot(db: Session, doctor_id: int, date: str, time_slot: str) -> dict:
    """Atomically claim a slot. SlotLock's own primary key IS
    (doctor_id, date, time_slot); SQLite serializes writers and enforces
    that key, so of any number of simultaneous callers, exactly one INSERT
    here can ever succeed -- see models.SlotLock's docstring. Everyone
    else gets success=False, reason="slot_taken", not a crash."""
    _release_expired_hold(db, doctor_id, date, time_slot)
    token = uuid.uuid4().hex
    expires = _now() + datetime.timedelta(seconds=HOLD_TTL_SECONDS)
    db.add(
        SlotLock(
            doctor_id=doctor_id,
            date=date,
            time_slot=time_slot,
            status="held",
            hold_token=token,
            hold_expires_at=expires,
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return {"success": False, "reason": "slot_taken"}
    return {"success": True, "hold_token": token, "expires_at": expires.isoformat()}


def _taken_slots(db: Session, doctor_id: int, date: str) -> set[str]:
    now = _now()
    return {
        r.time_slot
        for r in db.query(SlotLock).filter_by(doctor_id=doctor_id, date=date).all()
        if r.status == "confirmed" or (r.status == "held" and r.hold_expires_at and r.hold_expires_at >= now)
    }


def available_slots(db: Session, doctor_id: int, date: str) -> list[str]:
    target = datetime.date.fromisoformat(date)
    sched = db.query(DoctorSchedule).filter_by(doctor_id=doctor_id, weekday=target.weekday()).first()
    if not sched:
        return []
    all_slots = _generate_slots(sched.start_time, sched.end_time)
    taken = _taken_slots(db, doctor_id, date)
    free = [s for s in all_slots if s not in taken]
    now = _now()
    if target == now.date():
        # CodeRabbit-flagged, real bug: a same-day slot earlier than the
        # current time was offered and bookable -- hold_booking's own
        # date check is date-level only ("target < today"), never
        # time-of-day, so "today at 10am" stayed offerable at 3pm.
        now_minutes = now.hour * 60 + now.minute
        free = [s for s in free if _slot_minutes(s) > now_minutes]
    return free


def earliest_available(db: Session, doctor_id: int, from_date: datetime.date, horizon_days: int = 14) -> dict | None:
    for i in range(horizon_days):
        d = (from_date + datetime.timedelta(days=i)).isoformat()
        free = available_slots(db, doctor_id, d)
        if free:
            return {"date": d, "time_slot": free[0], "alternatives": free[1:3]}
    return None


def nearest_alternatives(db: Session, doctor_id: int, date: str, time_slot: str) -> list[dict]:
    """KCD-361's exact shape: the nearest other time the same day, plus
    the same time on the nearest other day -- never a slot that is not
    actually free at the moment of asking."""
    out = []
    same_day = [s for s in available_slots(db, doctor_id, date) if s != time_slot]
    if same_day:
        nearest = min(same_day, key=lambda s: abs(_slot_minutes(s) - _slot_minutes(time_slot)))
        out.append({"date": date, "time_slot": nearest})
    start = datetime.date.fromisoformat(date)
    for i in range(1, 8):
        d = (start + datetime.timedelta(days=i)).isoformat()
        free = available_slots(db, doctor_id, d)
        if time_slot in free:
            out.append({"date": d, "time_slot": time_slot})
            break
        if free:
            out.append({"date": d, "time_slot": free[0]})
            break
    return out


# ========================================================== patients

# Mirrors agent/booking_flow.py's effective_phone() sentinel by VALUE,
# not by import -- clinic-api is a separate service this repo does not
# own (see this module's own docstring / CLAUDE.md's service-boundary
# discipline), the same reason agent/phonetic_match.py and
# clinic-api/phonetic_match.py are deliberately duplicated rather than
# shared.
NOT_PROVIDED_PHONE = "not_provided"


def find_or_create_patient(db: Session, name: str, phone: str, age: int | None = None) -> Patient:
    """CodeRabbit-flagged, real bug: the "not_provided" sentinel (a
    caller who declined to give any phone number) used to be looked up
    like a real phone value -- two DIFFERENT patients who share a name
    and BOTH decline a number would silently become the SAME Patient
    row, cross-contaminating whatever booking history/proxy
    authorisation is attached to it. A declined phone always creates a
    fresh Patient instead of searching for one."""
    if phone == NOT_PROVIDED_PHONE:
        p = Patient(name=name, phone=phone, age=age, created_at=_now())
        db.add(p)
        db.flush()
        return p
    p = db.query(Patient).filter_by(name=name, phone=phone).first()
    if p:
        if age and not p.age:
            p.age = age
        return p
    p = Patient(name=name, phone=phone, age=age, created_at=_now())
    db.add(p)
    db.flush()
    return p


def set_patient_senior(db: Session, phone: str, senior: bool, caller_phone: str | None = None) -> int:
    """KCD-084: remember (or clear) that callers using this phone number are
    served in senior mode. Applies to every Patient row with that phone --
    the number is what a returning caller can be recognised by. Returns how
    many rows changed. A declined number is never a key (it would flag
    every stranger who withheld one).

    Each row is changed only if `caller_phone` (defaulting to `phone`, as in
    lookup_bookings) is authorised for that patient by authorize_disclosure:
    self, or a verified proxy. Until real CallerID reaches this API the
    default means the number the caller SAID, so this is exactly as strong
    as the booking lookups -- it closes the door for a bridge that supplies a
    real caller_phone, and it is stated here so it is not mistaken for more."""
    if not phone or phone == NOT_PROVIDED_PHONE:
        return 0
    asking_as = caller_phone or phone
    rows = [p for p in db.query(Patient).filter(Patient.phone == phone).all() if authorize_disclosure(db, p, asking_as)]
    changed = 0
    for p in rows:
        if bool(p.senior_mode) != bool(senior):
            p.senior_mode = bool(senior)
            changed += 1
    if changed:
        db.commit()
    return changed


def get_patient_senior(db: Session, phone: str, caller_phone: str | None = None) -> bool:
    if not phone or phone == NOT_PROVIDED_PHONE:
        return False
    asking_as = caller_phone or phone
    return any(
        authorize_disclosure(db, p, asking_as)
        for p in db.query(Patient).filter(Patient.phone == phone, Patient.senior_mode.is_(True)).all()
    )


def record_proxy(db: Session, patient: Patient, caller_phone: str, relationship_label: str, verified_by: str) -> None:
    existing = db.query(PatientProxy).filter_by(patient_id=patient.id, caller_phone=caller_phone).first()
    if existing:
        return
    db.add(
        PatientProxy(
            patient_id=patient.id,
            caller_phone=caller_phone,
            relationship_label=relationship_label,
            verified_by=verified_by,
            created_at=_now(),
        )
    )


def authorize_disclosure(db: Session, patient: Patient, caller_phone: str, confirmed_age: int | None = None) -> bool:
    """Whether an EXISTING record may be read out to this caller (KCD-373,
    KCD-378) or modified by them. Booking a brand-new appointment never
    needs this check -- only looking up or changing one that was not made
    from this same phone."""
    if caller_phone == patient.phone:
        return True
    proxy = db.query(PatientProxy).filter_by(patient_id=patient.id, caller_phone=caller_phone).first()
    if not proxy:
        return False
    if proxy.verified_by == "dob_confirmed":
        return True
    if proxy.verified_by == "relationship_stated" and confirmed_age is not None and patient.age == confirmed_age:
        proxy.verified_by = "dob_confirmed"
        db.commit()
        return True
    return False


# ======================================================== appointments


def _verify_appointment_persisted(db: Session, appointment_id: int, expected: dict) -> bool:
    """KCD-486: "the write is verified in the system of record with the
    expected values before it is verbalised" -- re-reads the row AFTER
    commit and checks it, rather than trusting that db.commit() raising
    nothing means the caller's specific values actually landed. Against
    this prototype's single-process SQLite this will realistically never
    fail (a successful commit() IS durable here) -- but that is exactly
    why this check belongs at this layer rather than being skipped:
    swapping the real hospital/lab system in behind this same function
    (this module's own docstring's stated migration path) is precisely
    the case where "the commit call didn't raise" and "the system of
    record now reflects it" can come apart, e.g. an async-replicated
    write or a queue-backed integration. Uses a FRESH query rather than
    the in-memory `appt` object, which would just echo back the same
    Python values that were assigned before commit and verify nothing.
    """
    db.expire_all()
    row = db.get(Appointment, appointment_id)
    if row is None:
        return False
    return all(getattr(row, field) == value for field, value in expected.items())


def confirm_booking(
    db: Session,
    hold_token: str,
    doctor_id: int,
    date: str,
    time_slot: str,
    patient_name: str,
    phone: str,
    caller_phone: str,
    patient_age: int | None = None,
    relationship_label: str = "self",
) -> dict:
    lock = db.get(SlotLock, (doctor_id, date, time_slot))
    # IDEMPOTENT on the hold token: a retried request (a dropped connection, a repeated "yes") for a
    # slot this SAME hold already confirmed returns the booking that exists -- the same confirmation
    # number, no second appointment and no false "hold expired" for a booking that did succeed.
    if lock and lock.status == "confirmed" and lock.hold_token == hold_token and lock.appointment_id:
        appt = db.get(Appointment, lock.appointment_id)
        if appt is not None and appt.status == "confirmed" and appt.phone == phone:
            doctor = db.get(Doctor, doctor_id)
            return {
                "success": True,
                "confirmation_id": appt.confirmation_id,
                "doctor_name": doctor.name,
                "doctor_name_bn": _alias(doctor.aliases_bn),
                "doctor_name_hi": _alias(doctor.aliases_hi),
                "date": date,
                "time_slot": time_slot,
                "replayed": True,
            }
    if not lock or lock.status != "held" or lock.hold_token != hold_token:
        return {"success": False, "reason": "hold_expired"}
    if lock.hold_expires_at and lock.hold_expires_at < _now():
        db.delete(lock)
        db.commit()
        return {"success": False, "reason": "hold_expired"}

    doctor = db.get(Doctor, doctor_id)
    patient = find_or_create_patient(db, patient_name, phone, patient_age)
    verified_by = "self" if caller_phone == phone else "relationship_stated"
    record_proxy(db, patient, caller_phone, relationship_label, verified_by)

    confirmation_id = _confirmation_id(date)
    appt = Appointment(
        confirmation_id=confirmation_id,
        doctor_id=doctor_id,
        date=date,
        time_slot=time_slot,
        patient_name=patient_name,
        phone=phone,
        created_at=_now(),
        patient_id=patient.id,
        caller_phone=caller_phone,
        status="confirmed",
    )
    db.add(appt)
    db.flush()
    lock.status = "confirmed"
    lock.hold_expires_at = None
    lock.appointment_id = appt.id
    db.commit()

    # KCD-486: a confirmation number is spoken only once the write is
    # verified -- an unverified write produces a distinct outcome (never
    # a false "success", and never the ordinary tool_failure apology
    # either) so main.py can put the caller on hold and escalate instead
    # of reading out a reference that may not be real.
    if not _verify_appointment_persisted(
        db,
        appt.id,
        {
            "confirmation_id": confirmation_id,
            "doctor_id": doctor_id,
            "date": date,
            "time_slot": time_slot,
            "patient_name": patient_name,
            "phone": phone,
            "status": "confirmed",
        },
    ):
        return {"success": False, "reason": "write_unverified", "confirmation_id": confirmation_id}

    queue_sms(
        db,
        phone,
        "booking_confirmed",
        f"Your appointment with {doctor.name} on {date} at {time_slot} is confirmed. Ref: {confirmation_id}.",
        confirmation_id,
    )

    return {
        "success": True,
        "confirmation_id": confirmation_id,
        "doctor_name": doctor.name,
        "doctor_name_bn": _alias(doctor.aliases_bn),
        "doctor_name_hi": _alias(doctor.aliases_hi),
        "date": date,
        "time_slot": time_slot,
    }


def find_conflict(db: Session, patient_phone: str, date: str, time_slot: str) -> dict | None:
    """KCD-378: the SAME patient already has a confirmed appointment at
    this date and time. Checked against the hospital system (this table),
    never a local cache, so it is current at the moment of the new
    booking."""
    existing = (
        db.query(Appointment).filter_by(phone=patient_phone, date=date, time_slot=time_slot, status="confirmed").first()
    )
    if not existing:
        return None
    doctor = db.get(Doctor, existing.doctor_id)
    return {
        "confirmation_id": existing.confirmation_id,
        "doctor_name": doctor.name if doctor else None,
        "date": existing.date,
        "time_slot": existing.time_slot,
    }


def active_cancellation_policy(db: Session, as_of: datetime.date | None = None) -> CancellationPolicy | None:
    """KCD-488: the policy row IN FORCE at `as_of` (default: today) --
    versioned configuration with effective dates, never a flat constant a
    developer would have to redeploy to change. Whichever row has the
    LATEST `effective_from` that is not after `as_of` wins, so adding a
    new version with a future effective_from schedules a change without
    touching any row already applied to a past cancellation."""
    as_of_iso = (as_of or _now().date()).isoformat()
    return (
        db.query(CancellationPolicy)
        .filter(CancellationPolicy.effective_from <= as_of_iso)
        .order_by(CancellationPolicy.effective_from.desc(), CancellationPolicy.version.desc())
        .first()
    )


def cancellation_charge(db: Session, appt: Appointment) -> tuple[int, CancellationPolicy | None]:
    """-> (charge_inr, policy_used). policy_used is None only if the
    cancellation_policies table has no row at all -- seed_default_
    cancellation_policy() backfills one on every boot, so this is an
    infrastructure gap to fix, not a normal outcome; treated as "no
    charge" rather than raising, so a missing policy row fails open on
    price (never silently overcharges) but is loud in the return shape
    for whoever calls this to notice."""
    # The rule in force when the caller CANCELS (today), not on the day of the
    # appointment: a rule change scheduled for a date before the appointment
    # must not be applied to a cancellation made before it takes effect, and
    # cancellation_policy_version must record the rule that actually applied.
    policy = active_cancellation_policy(db)
    if not policy:
        return 0, None
    doctor = db.get(Doctor, appt.doctor_id)
    fee = doctor.consultation_fee_inr or 0
    # refund_eligible=False marks a NON-REFUNDABLE policy: the full fee is
    # charged however much notice was given, so free_window_hours and
    # charge_percent (which only describe the refundable case) are
    # deliberately ignored -- checked FIRST, otherwise a long-notice
    # cancellation would fall through the free window and this flag
    # would never affect anything.
    if not policy.refund_eligible:
        return fee, policy
    appt_dt = datetime.datetime.combine(
        datetime.date.fromisoformat(appt.date),
        datetime.datetime.strptime(appt.time_slot, "%H:%M").time(),
    )
    if appt_dt - _now() >= datetime.timedelta(hours=policy.free_window_hours):
        return 0, policy
    return round(fee * policy.charge_percent / 100), policy


def cancel_appointment(db: Session, confirmation_id: str, confirm_charge: bool = False) -> dict:
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if not appt:
        return {"success": False, "reason": "not_found"}
    charge, policy = cancellation_charge(db, appt)
    if charge > 0 and not confirm_charge:
        # Stated BEFORE it is applied (KCD-372) -- the caller must say yes
        # a second time, with the amount already in their ear, before
        # anything is written.
        return {"success": False, "reason": "charge_confirmation_required", "charge_inr": charge}

    lock = db.get(SlotLock, (appt.doctor_id, appt.date, appt.time_slot))
    if lock:
        db.delete(lock)
    appt.status = "cancelled"
    appt.cancelled_at = _now()
    appt.cancellation_charge_inr = charge
    # KCD-488: which rule produced this charge, recorded on the row it
    # applied to -- so a later audit sees the actual rule in force at
    # cancellation time, not just today's policy (which may since differ).
    appt.cancellation_policy_version = policy.version if policy else None
    db.commit()

    queue_sms(
        db,
        appt.phone,
        "booking_cancelled",
        f"Your appointment ({confirmation_id}) has been cancelled."
        + (f" A charge of Rs.{charge} applies." if charge else ""),
        confirmation_id,
    )
    return {"success": True, "confirmation_id": confirmation_id, "charge_inr": charge}


def reschedule_appointment(db: Session, confirmation_id: str, new_date: str, new_time_slot: str) -> dict:
    """Holds the NEW slot first; only once that succeeds is the old one
    touched at all. A failed hold returns immediately with the original
    appointment completely untouched -- KCD-371's "a failed swap leaves
    the original intact"."""
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if not appt:
        return {"success": False, "reason": "not_found"}

    hold = hold_slot(db, appt.doctor_id, new_date, new_time_slot)
    if not hold["success"]:
        return {
            "success": False,
            "reason": "slot_taken",
            "alternative_slots": available_slots(db, appt.doctor_id, new_date)[:3],
        }

    doctor = db.get(Doctor, appt.doctor_id)
    new_confirmation_id = _confirmation_id(new_date)
    new_appt = Appointment(
        confirmation_id=new_confirmation_id,
        doctor_id=appt.doctor_id,
        date=new_date,
        time_slot=new_time_slot,
        patient_name=appt.patient_name,
        phone=appt.phone,
        created_at=_now(),
        patient_id=appt.patient_id,
        caller_phone=appt.caller_phone,
        status="confirmed",
        rescheduled_from_id=appt.id,
    )
    db.add(new_appt)
    db.flush()

    new_lock = db.get(SlotLock, (appt.doctor_id, new_date, new_time_slot))
    new_lock.status, new_lock.hold_expires_at, new_lock.appointment_id = "confirmed", None, new_appt.id

    old_lock = db.get(SlotLock, (appt.doctor_id, appt.date, appt.time_slot))
    if old_lock:
        db.delete(old_lock)
    appt.status = "rescheduled"
    db.commit()

    # KCD-486, same discipline as confirm_booking: verify before speaking
    # the new reference number.
    if not _verify_appointment_persisted(
        db,
        new_appt.id,
        {
            "confirmation_id": new_confirmation_id,
            "doctor_id": appt.doctor_id,
            "date": new_date,
            "time_slot": new_time_slot,
            "status": "confirmed",
        },
    ):
        return {"success": False, "reason": "write_unverified", "confirmation_id": new_confirmation_id}

    queue_sms(
        db,
        appt.phone,
        "booking_rescheduled",
        f"Your appointment with {doctor.name} has been moved to {new_date} at {new_time_slot}. "
        f"New ref: {new_confirmation_id}.",
        new_confirmation_id,
    )

    return {
        "success": True,
        "confirmation_id": new_confirmation_id,
        "doctor_name": doctor.name,
        "date": new_date,
        "time_slot": new_time_slot,
    }


def lookup_bookings(
    db: Session,
    phone: str | None = None,
    confirmation_id: str | None = None,
    name: str | None = None,
    upcoming_only: bool = True,
    caller_phone: str | None = None,
) -> list[dict]:
    """SECURITY (CWE-639/IDOR, CodeRabbit-flagged): every row this used to
    return on a bare phone/name search, unauthenticated -- anyone who
    spoke a phone number, guessed or overheard, could hear a stranger's
    doctor/date/time/name. authorize_disclosure() already existed for
    exactly this (its own docstring says so) but was never called
    anywhere. Fixed here, the one place every phone/confirmation_id/name
    lookup in the codebase funnels through.

    A caller who supplies the exact confirmation_id already holds the
    bearer token for that one row (unchanged from before -- 8 random hex
    chars is the existing access-control model for that path, per
    _confirmation_id's own comment) and skips the extra check. A caller
    searching by PHONE or NAME ALONE must additionally be authorized for
    EACH row: `caller_phone` (defaulting to `phone`, since every existing
    call site already treats the phone it searches by as the asking
    caller's own number) is checked against the row's patient via
    authorize_disclosure -- the same self/proxy/dob-confirmed tiers
    booking time already builds and records, now actually load-bearing.
    A name-only search with no phone at all now discloses nothing: a
    name alone was never a meaningful access-control check to begin
    with."""
    q = db.query(Appointment).filter(Appointment.status == "confirmed")
    if confirmation_id:
        q = q.filter(Appointment.confirmation_id == confirmation_id)
    if phone:
        q = q.filter((Appointment.phone == phone) | (Appointment.caller_phone == phone))
    if name:
        q = q.filter(Appointment.patient_name.ilike(f"%{name}%"))
    if upcoming_only:
        q = q.filter(Appointment.date >= _now().date().isoformat())
    rows = q.order_by(Appointment.date, Appointment.time_slot).all()

    require_auth = not confirmation_id
    asking_as = caller_phone or phone

    out = []
    for a in rows:
        if require_auth:
            patient = db.get(Patient, a.patient_id) if a.patient_id else None
            if not patient or not asking_as or not authorize_disclosure(db, patient, asking_as):
                continue
        doctor = db.get(Doctor, a.doctor_id)
        out.append(
            {
                "confirmation_id": a.confirmation_id,
                "doctor_name": doctor.name if doctor else None,
                "date": a.date,
                "time_slot": a.time_slot,
                "patient_name": a.patient_name,
            }
        )
    return out


# ============================================================ tests


def book_tests(
    db: Session,
    test_ids: list[int],
    date: str,
    patient_name: str,
    phone: str,
    caller_phone: str,
    patient_age: int | None = None,
    relationship_label: str = "self",
) -> dict:
    patient = find_or_create_patient(db, patient_name, phone, patient_age)
    verified_by = "self" if caller_phone == phone else "relationship_stated"
    record_proxy(db, patient, caller_phone, relationship_label, verified_by)

    group_id = uuid.uuid4().hex[:10]
    confirmation_id = _confirmation_id(date)
    booked, total, combined_prep = [], 0, []
    for tid in test_ids:
        t = db.get(LabTest, tid)
        if not t:
            continue
        db.add(
            TestBooking(
                booking_group_id=group_id,
                confirmation_id=confirmation_id,
                lab_test_id=t.id,
                patient_id=patient.id,
                patient_name=patient_name,
                date=date,
                phone=phone,
                caller_phone=caller_phone,
                status="confirmed",
                created_at=_now(),
            )
        )
        booked.append(t.name)
        total += t.rate_inr
        if t.prep_instructions_bn:
            combined_prep.append(t.prep_instructions_bn)
    if not booked:
        return {"success": False, "reason": "no_valid_tests"}
    db.commit()

    queue_sms(
        db,
        phone,
        "tests_confirmed",
        f"Your tests ({', '.join(booked)}) on {date} are confirmed. Total Rs.{total}. Ref: {confirmation_id}.",
        confirmation_id,
    )

    return {
        "success": True,
        "confirmation_id": confirmation_id,
        "test_names": booked,
        "total_rate_inr": total,
        "date": date,
        "combined_prep": " ".join(combined_prep),
    }


def add_test_to_booking(db: Session, confirmation_id: str, test_name: str) -> dict:
    existing = db.query(TestBooking).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if not existing:
        return {"success": False, "reason": "not_found"}
    test = db.query(LabTest).filter(func.lower(LabTest.name) == test_name.lower()).first()
    if not test:
        return {"success": False, "reason": "test_not_found"}
    already = {
        tb.lab_test_id
        for tb in db.query(TestBooking).filter_by(booking_group_id=existing.booking_group_id, status="confirmed").all()
    }
    if test.id in already:
        return {"success": False, "reason": "already_booked"}
    db.add(
        TestBooking(
            booking_group_id=existing.booking_group_id,
            confirmation_id=confirmation_id,
            lab_test_id=test.id,
            patient_id=existing.patient_id,
            patient_name=existing.patient_name,
            date=existing.date,
            phone=existing.phone,
            caller_phone=existing.caller_phone,
            status="confirmed",
            created_at=_now(),
        )
    )
    db.commit()
    return {"success": True, "confirmation_id": confirmation_id, "test_name": test.name, "date": existing.date}


# ========================================================= notifications


def send_payment_link(db: Session, confirmation_id: str) -> dict:
    """Queue a payment-link message for a CONFIRMED DOCTOR appointment. Only a doctor appointment is payable by
    phone (CLAUDE.md's truth boundary already keeps money out of the model's hands; this keeps it out of a lab test's
    hands too -- a lab test's confirmation_id belongs to TestBooking, not Appointment, so it is never found here and
    never gets a link. A lab test is paid and booked at the counter, never over the call.

    The link is an EXPLICIT PLACEHOLDER on an RFC 2606 reserved, never-resolvable domain -- the same "log exactly
    what WOULD be sent, never claim it was" discipline as queue_sms itself, which this calls. No payment gateway is
    integrated; wiring one means building the real link in a new clinic-api/payments.py and calling it from here."""
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if not appt:
        return {"success": False, "reason": "not_found"}
    if appt.phone == NOT_PROVIDED_PHONE:
        return {"success": False, "reason": "no_phone_on_file"}
    doctor = db.get(Doctor, appt.doctor_id)
    fee = (doctor.consultation_fee_inr if doctor else None) or 0
    link = f"https://pay.kolkata-care-voice-agent.invalid/{confirmation_id}"
    message = f"Pay Rs {fee} to confirm your appointment ({confirmation_id}): {link}"
    result = queue_sms(db, appt.phone, "payment_link", message, confirmation_id)
    out = {"success": result["queued"], "amount_inr": fee, "payment_link": link, "confirmation_id": confirmation_id}
    if not result["queued"]:
        out["reason"] = result.get("reason", "unknown")
    return out


def queue_sms(
    db: Session, to_phone: str, template_key: str, message: str, related_confirmation_id: str | None = None
) -> dict:
    """The open integration placeholder. Logs exactly what WOULD be sent,
    in the shape a real provider needs, and returns status="queued" --
    never "sent". Wiring a real SMS/WhatsApp provider means implementing
    send() in a new clinic-api/notifications.py and calling it from here;
    nothing else in this module needs to change, and nothing anywhere else
    claims a message reached a caller's phone until that exists.

    CodeRabbit-flagged: a caller who declined to give any phone number
    (NOT_PROVIDED_PHONE) has nowhere to send anything -- queuing a row
    addressed to the literal string "not_provided" is not merely useless,
    it is a fake record of an outbound message that was never actually
    addressable. Skipped, with a distinct, honest outcome."""
    if to_phone == NOT_PROVIDED_PHONE:
        return {"queued": False, "reason": "no_phone_on_file"}
    # KCD-353: a text channel states it is automated, every time (idempotent).
    message = _with_disclosure_notice(message)
    # The same notice to the same number for the same booking inside the window is a retry, not a second message.
    twin = (
        db.query(SmsOutbox)
        .filter_by(
            to_phone=to_phone,
            template_key=template_key,
            message=message,
            related_confirmation_id=related_confirmation_id,
        )
        .order_by(SmsOutbox.created_at.desc())
        .first()
    )
    if twin is not None and (_now() - twin.created_at).total_seconds() < SMS_DEDUP_WINDOW_S:
        return {"queued": True, "id": twin.id, "duplicate": True}
    row = SmsOutbox(
        to_phone=to_phone,
        template_key=template_key,
        message=message,
        status="queued",
        related_confirmation_id=related_confirmation_id,
        created_at=_now(),
    )
    db.add(row)
    db.commit()
    return {"queued": True, "id": row.id}


def resend_confirmation(db: Session, confirmation_id: str) -> dict:
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    to_phone = message = None
    if appt:
        to_phone = appt.phone
        message = f"Your appointment ({confirmation_id}) is confirmed for {appt.date} at {appt.time_slot}."
    else:
        tb = db.query(TestBooking).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
        if tb:
            to_phone = tb.phone
            message = f"Your test booking ({confirmation_id}) is confirmed for {tb.date}."
    if not to_phone:
        return {"success": False, "reason": "not_found"}
    if to_phone == NOT_PROVIDED_PHONE:
        # CodeRabbit-flagged: to_phone[-4:] on the literal sentinel string
        # produced the nonsense "ided" (the last 4 characters of
        # "not_provided") as if it were a real phone's last 4 digits --
        # confusing at best, and could read as a real (wrong) number.
        return {"success": False, "reason": "no_phone_on_file"}

    last = (
        db.query(SmsOutbox)
        .filter_by(related_confirmation_id=confirmation_id, template_key="resend")
        .order_by(SmsOutbox.created_at.desc())
        .first()
    )
    if last and (_now() - last.created_at).total_seconds() < SMS_RESEND_MIN_INTERVAL_S:
        return {"success": False, "reason": "rate_limited"}

    queue_sms(db, to_phone, "resend", message, confirmation_id)
    return {"success": True, "sent_to_last4": to_phone[-4:]}


# ========================================================== department routing


MAX_SUGGESTED_DOCTORS = 3


def _department_doctor_names(db: Session, department_id: int, limit: int = MAX_SUGGESTED_DOCTORS) -> list[str]:
    """The real, currently-seeded doctors in a department, short-name form -- never a name the model made up
    (CLAUDE.md's truth boundary): this is a live query, the same table booking_service's own doctor search reads."""
    return [
        d.name for d in db.query(Doctor).filter_by(department_id=department_id).order_by(Doctor.id).limit(limit).all()
    ]


def route_department(db: Session, query_text: str, lang: str) -> dict:
    """Symptom text -> department, plus (KCD-lay-terms) the real doctors in it, so the agent can name someone
    instead of only the department. `phrase_matches` tolerates a word or two inserted inside a keyword phrase --
    "বুকে ব্যথা" (chest pain) must still match "বুকে খুব ব্যথা" (chest REALLY hurts), an ordinary way to say it."""
    q = query_text.lower()
    matches: list[int] = []
    for r in db.query(DepartmentRoute).all():
        kws = {"bn": r.keywords_bn, "hi": r.keywords_hi, "en": r.keywords_en}.get(lang, r.keywords_en)
        if any(kw and phrase_matches(kw.lower(), q) for kw in kws.split("|")):
            matches.append(r.department_id)
    matches = list(dict.fromkeys(matches))
    if not matches:
        return {"matched": False}
    if len(matches) > 1:
        depts = [db.get(Department, m) for m in matches[:2]]
        return {
            "matched": False,
            "ambiguous": True,
            "candidates": [d.name for d in depts],
            "doctors_by_department": {d.name: _department_doctor_names(db, d.id) for d in depts},
        }
    dept = db.get(Department, matches[0])
    return {
        "matched": True,
        "department_name": dept.name,
        "department_id": dept.id,
        "doctors": _department_doctor_names(db, dept.id),
    }


# ============================================================= drafts


def save_draft(db: Session, caller_phone: str, call_id: str, slots_json: str) -> None:
    existing = db.query(DraftBooking).filter_by(caller_phone=caller_phone).first()
    expires = _now() + datetime.timedelta(minutes=DRAFT_BOOKING_TTL_MINUTES)
    if existing:
        existing.call_id, existing.slots_json = call_id, slots_json
        existing.updated_at, existing.expires_at = _now(), expires
    else:
        db.add(
            DraftBooking(
                caller_phone=caller_phone, call_id=call_id, slots_json=slots_json, updated_at=_now(), expires_at=expires
            )
        )
    db.commit()


def find_draft(db: Session, caller_phone: str) -> dict | None:
    row = db.query(DraftBooking).filter_by(caller_phone=caller_phone).first()
    if not row or row.expires_at < _now():
        return None
    return {"call_id": row.call_id, "slots_json": row.slots_json, "updated_at": row.updated_at.isoformat()}


def clear_draft(db: Session, caller_phone: str) -> None:
    db.query(DraftBooking).filter_by(caller_phone=caller_phone).delete()
    db.commit()
