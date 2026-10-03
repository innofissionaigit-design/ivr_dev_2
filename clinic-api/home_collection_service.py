# ADDED BY SOURAV: the full home-collection VISIT flow (multi-test eligibility aggregation, real
# slot availability/hold/confirm, a deterministic quote, the payment policy, and the actual
# booking/cancellation transaction) lives here, as its own service module -- parallel to
# booking_service.py (doctor appointments) and enquiry_service.py (read-only enquiries) -- rather
# than inside either of those, because a home-collection VISIT is neither: it is a real write
# transaction (like booking_service.py) over a resource with no doctor_id (unlike anything in
# booking_service.py), and it must not duplicate enquiry_service.home_collection_eligibility()'s
# own eligibility rules, only call them.
"""Home-collection visit scheduling and booking (KCD-387, full flow).

Every function here returns a plain dict and never raises for a normal not-found/unavailable/
full outcome -- same discipline as booking_service.py and enquiry_service.py. The one thing this
module is NOT allowed to do is invent a fact: every price, slot, charge and assignment is either
a live DB read or a deterministic computation over DB values (a sum, a "first row with capacity
left", "latest policy version in effect") -- never guessed, never left to the caller-facing layer
to compute.
"""

from __future__ import annotations

import datetime
import uuid

from enquiry_service import home_collection_eligibility
from models import (
    HomeCollectionBooking,
    HomeCollectionBookingTest,
    HomeCollectionCollector,
    HomeCollectionPaymentPolicy,
    HomeCollectionSlot,
    HomeCollectionSlotHold,
    LabTest,
    Patient,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# ADDED BY SOURAV: longer than booking_service.HOLD_TTL_SECONDS (90s) on purpose -- picking a
# home-collection slot is one step in a much longer multi-turn conversation (address capture,
# readback, confirmation) than picking a doctor's time, so the hold needs to outlive that.
HOLD_TTL_SECONDS = 300


def _now() -> datetime.datetime:
    return datetime.datetime.now()


def _booking_reference() -> str:
    # Same shape family as Appointment/TestBooking confirmation ids elsewhere in this codebase
    # (a short, speakable, unique token) -- "HC-" prefix makes a home-collection reference visually
    # distinct from a doctor/test confirmation id in logs and audits.
    return f"HC-{uuid.uuid4().hex[:10].upper()}"


# ================================================= multi-test eligibility (per acceptance criterion 1)


def multi_test_eligibility(db: Session, lab_test_ids: list[int], postal_code: str) -> list[dict]:
    """ADDED BY SOURAV: evaluate EACH requested test independently and return every result --
    never merge or silently pick one (the story's own "do not silently choose one test" rule,
    and unlike merge_prep_instructions, eligibility is not a mergeable quantity). Calls the
    existing, unmodified home_collection_eligibility() once per test; the business rule itself is
    never duplicated here."""
    results = []
    for lab_test_id in lab_test_ids:
        result = home_collection_eligibility(db, lab_test_id, postal_code)
        test = db.get(LabTest, lab_test_id)
        results.append({**result, "lab_test_id": lab_test_id, "test_name": test.name if test else None})
    return results


# ========================================================================= slot availability


def available_slots(db: Session, postal_code: str, date: str) -> list[dict]:
    """ADDED BY SOURAV: real, DB-backed slots with capacity actually remaining -- never a
    calculated/estimated window. Mirrors booking_service.available_slots()'s shape (a list the
    caller can be read) but counts remaining capacity against HomeCollectionSlotHold rows instead
    of booking_service's own SlotLock table, since this is a different resource (Section 7/12 of
    the architecture investigation ruled out reusing SlotLock directly)."""
    now = _now()
    slots = db.query(HomeCollectionSlot).filter_by(postal_code=postal_code, date=date).all()
    out = []
    for slot in slots:
        occupied = (
            db.query(HomeCollectionSlotHold)
            .filter(HomeCollectionSlotHold.slot_id == slot.id)
            .filter(
                (HomeCollectionSlotHold.status == "confirmed")
                | ((HomeCollectionSlotHold.status == "held") & (HomeCollectionSlotHold.hold_expires_at >= now))
            )
            .count()
        )
        remaining = slot.capacity - occupied
        if remaining > 0:
            out.append(
                {
                    "slot_id": slot.id,
                    "date": slot.date,
                    "start_time": slot.start_time,
                    "end_time": slot.end_time,
                    "remaining_capacity": remaining,
                }
            )
    out.sort(key=lambda s: s["start_time"])
    return out


# =============================================================================== hold / release


def _release_expired_home_collection_holds(db: Session, slot_id: int) -> None:
    # Same re-check-at-delete-time reasoning as booking_service._release_expired_hold's own
    # docstring: the WHERE clause re-checks status/expiry at delete time, so a hold someone else
    # already replaced after this read is simply not matched and stays in place.
    db.query(HomeCollectionSlotHold).filter(
        HomeCollectionSlotHold.slot_id == slot_id,
        HomeCollectionSlotHold.status == "held",
        HomeCollectionSlotHold.hold_expires_at < _now(),
    ).delete()
    db.flush()


def hold_home_collection_slot(
    db: Session, slot_id: int, caller_phone: str | None = None, call_id: str | None = None
) -> dict:
    """ADDED BY SOURAV: atomically claim ONE unit of a slot's capacity. HomeCollectionSlotHold's
    primary key is (slot_id, unit_index); of any number of simultaneous callers trying the SAME
    unit_index, exactly one INSERT can ever succeed (see that model's own docstring) -- so this
    tries unit_index 0..capacity-1 in order and only reports "slot_full" once every unit is
    genuinely occupied by a live hold or a confirmed booking."""
    slot = db.get(HomeCollectionSlot, slot_id)
    if not slot:
        return {"success": False, "reason": "slot_not_found"}
    _release_expired_home_collection_holds(db, slot_id)
    token = uuid.uuid4().hex
    expires = _now() + datetime.timedelta(seconds=HOLD_TTL_SECONDS)
    for unit_index in range(slot.capacity):
        db.add(
            HomeCollectionSlotHold(
                slot_id=slot_id,
                unit_index=unit_index,
                status="held",
                hold_token=token,
                caller_phone=caller_phone,
                call_id=call_id,
                hold_expires_at=expires,
                created_at=_now(),
            )
        )
        try:
            db.commit()
            return {
                "success": True,
                "hold_token": token,
                "slot_id": slot_id,
                "date": slot.date,
                "start_time": slot.start_time,
                "end_time": slot.end_time,
                "expires_at": expires.isoformat(),
            }
        except IntegrityError:
            db.rollback()
            continue
    return {"success": False, "reason": "slot_full"}


def release_home_collection_hold(db: Session, hold_token: str) -> dict:
    """ADDED BY SOURAV: the caller changed their mind before confirming -- free the unit
    immediately rather than waiting for its TTL, per the story's own "a caller abandoning the
    flow must not permanently consume capacity" requirement (though the TTL alone already
    guarantees that eventually)."""
    row = db.query(HomeCollectionSlotHold).filter_by(hold_token=hold_token, status="held").first()
    if not row:
        return {"success": False, "reason": "hold_not_found"}
    db.delete(row)
    db.commit()
    return {"success": True}


# =================================================================================== quote


def quote_home_collection(db: Session, lab_test_ids: list[int], postal_code: str) -> dict:
    """ADDED BY SOURAV: the deterministic price breakdown (test prices + the home-collection
    charge = total), computed here in code so the reply template only ever formats already-
    calculated numbers (same "never let a template calculate money" rule as
    agent/compare_flow.py's own docstring). Only ELIGIBLE tests are priced into the total --
    an ineligible test is reported but contributes nothing to test_charges_inr, since it cannot
    be part of a home-collection visit at all."""
    per_test = multi_test_eligibility(db, lab_test_ids, postal_code)
    # ADDED BY SOURAV: Section 17 needs a caller-facing Test1/Test2/.../Total breakdown, not just a
    # summed figure -- each ELIGIBLE row gets its own real rate_inr (never computed by a reply
    # template: agent/reply_templates.py only ever formats this already-priced list) so a caller
    # with several tests hears what each one costs, not only the combined number.
    for r in per_test:
        if r.get("eligible"):
            test = db.get(LabTest, r["lab_test_id"])
            r["rate_inr"] = test.rate_inr if test else None
    eligible = [r for r in per_test if r.get("eligible")]
    if not eligible:
        return {"found": True, "eligible_tests": [], "per_test": per_test, "quote_available": False}
    test_charges_inr = 0
    for r in eligible:
        test_charges_inr += r.get("rate_inr") or 0
    # Every eligible row carries the SAME charge_inr (one coverage row per postal code) -- take
    # the first rather than summing it per test, which would silently multiply a per-VISIT charge
    # by however many tests happen to be in it.
    home_collection_charge_inr = eligible[0].get("charge_inr", 0) or 0
    total_inr = test_charges_inr + home_collection_charge_inr
    return {
        "found": True,
        "per_test": per_test,
        "eligible_tests": [r["test_name"] for r in eligible],
        "quote_available": True,
        "test_charges_inr": test_charges_inr,
        "home_collection_charge_inr": home_collection_charge_inr,
        "total_inr": total_inr,
    }


# ============================================================================= payment policy


def active_payment_policy(db: Session, as_of: datetime.date | None = None) -> HomeCollectionPaymentPolicy | None:
    """ADDED BY SOURAV: same versioned "latest effective_from wins" pattern as
    booking_service.active_cancellation_policy() (KCD-488), reused unchanged for this new
    policy table rather than inventing a different lookup rule."""
    as_of_iso = (as_of or _now().date()).isoformat()
    return (
        db.query(HomeCollectionPaymentPolicy)
        .filter(HomeCollectionPaymentPolicy.effective_from <= as_of_iso)
        .order_by(HomeCollectionPaymentPolicy.effective_from.desc(), HomeCollectionPaymentPolicy.version.desc())
        .first()
    )


def payment_policy_reply_dict(db: Session, lang: str = "bn") -> dict:
    policy = active_payment_policy(db)
    if not policy:
        return {"found": False}
    text = {"bn": policy.description_bn, "hi": policy.description_hi, "en": policy.description_en}.get(
        lang, policy.description_bn
    )
    return {"found": True, "policy": policy.policy, "description": text}


# ========================================================================= collector assignment


def assign_collector(db: Session, postal_code: str) -> HomeCollectionCollector | None:
    """ADDED BY SOURAV: a REAL, deterministic assignment -- the active collector whose
    `postal_codes` covers this area, least-currently-loaded (fewest open assignments) first.
    Returns None, never an invented name, when no active collector covers this postal code; the
    caller-facing layer must then say assignment will be confirmed by staff later (see
    main.py's home-collection reply templates), exactly as Section 26 of the brief requires."""
    candidates = (
        db.query(HomeCollectionCollector)
        .filter(HomeCollectionCollector.active.is_(True))
        .filter(HomeCollectionCollector.postal_codes.like(f"%{postal_code}%"))
        .all()
    )
    # Exact token match against the "|"-joined list, not a bare substring hit (a postal code can
    # be a substring of another) -- same discipline LabTest.aliases_bn callers already use.
    candidates = [c for c in candidates if postal_code in (c.postal_codes or "").split("|")]
    if not candidates:
        return None

    def _open_load(c: HomeCollectionCollector) -> int:
        return (
            db.query(HomeCollectionBooking)
            .filter(
                HomeCollectionBooking.collector_id == c.id,
                HomeCollectionBooking.status == "confirmed",
                HomeCollectionBooking.dispatch_state.in_(("ASSIGNED", "IN_PROGRESS")),
            )
            .count()
        )

    return min(candidates, key=lambda c: (_open_load(c), c.id))


# ========================================================================= booking creation


def create_home_collection_booking(
    db: Session,
    *,
    hold_token: str,
    lab_test_ids: list[int],
    postal_code: str,
    patient_name: str,
    phone: str,
    caller_phone: str | None,
    address_line: str,
    locality: str = "",
    city: str = "",
    state: str = "",
    landmark: str = "",
    patient_id: int | None = None,
) -> dict:
    """ADDED BY SOURAV: the actual, persistent home-collection booking transaction. A retried
    "please confirm" request is handled at the HTTP layer by the clinic-api's existing
    @idempotent(...) decorator (clinic-api/idempotency.py, the same Idempotency-Key-header
    mechanism already used by every other write endpoint in this file) -- this function is not
    re-made idempotent on its own, by design, to avoid a second, competing idempotency scheme.
    Re-prices the eligible tests and the home-collection charge from live data at booking time
    (never trusts a stale quote the caller may have been given minutes earlier) and freezes those
    numbers onto the booking row."""
    hold = db.query(HomeCollectionSlotHold).filter_by(hold_token=hold_token, status="held").first()
    if not hold or (hold.hold_expires_at and hold.hold_expires_at < _now()):
        return {"success": False, "reason": "hold_expired"}

    quote = quote_home_collection(db, lab_test_ids, postal_code)
    if not quote.get("quote_available"):
        return {"success": False, "reason": "no_eligible_test"}

    eligible_ids = [r["lab_test_id"] for r in quote["per_test"] if r.get("eligible")]
    slot = db.get(HomeCollectionSlot, hold.slot_id)
    policy = active_payment_policy(db)
    patient = db.get(Patient, patient_id) if patient_id else None

    booking = HomeCollectionBooking(
        booking_reference=_booking_reference(),
        patient_id=patient.id if patient else None,
        patient_name=patient_name,
        phone=phone,
        caller_phone=caller_phone,
        date=slot.date,
        slot_id=slot.id,
        start_time=slot.start_time,
        end_time=slot.end_time,
        address_line=address_line,
        locality=locality,
        city=city,
        state=state,
        pincode=postal_code,
        landmark=landmark,
        test_charges_inr=quote["test_charges_inr"],
        home_collection_charge_inr=quote["home_collection_charge_inr"],
        total_inr=quote["total_inr"],
        payment_policy=policy.policy if policy else "",
        payment_status="pending",
        status="confirmed",
        dispatch_state="UNASSIGNED",
        created_at=_now(),
    )
    db.add(booking)
    db.flush()  # booking.id is needed below, before the final commit

    for lab_test_id in eligible_ids:
        test = db.get(LabTest, lab_test_id)
        db.add(HomeCollectionBookingTest(booking_id=booking.id, lab_test_id=lab_test_id, rate_inr=test.rate_inr))

    hold.status = "confirmed"
    hold.hold_expires_at = None
    hold.booking_id = booking.id

    collector = assign_collector(db, postal_code)
    if collector is not None:
        booking.collector_id = collector.id
        booking.assignment_status = "assigned"
        booking.dispatch_state = "ASSIGNED"

    try:
        db.commit()
    except IntegrityError:
        # Only reachable on an astronomically unlikely booking_reference collision (a fresh
        # uuid4 each call) or the hold having been consumed by a concurrent request between the
        # read above and this commit -- either way, never silently succeed with half-written data.
        db.rollback()
        return {"success": False, "reason": "booking_failed"}

    return _booking_dict(db, booking)


def _booking_dict(db: Session, booking: HomeCollectionBooking) -> dict:
    collector = db.get(HomeCollectionCollector, booking.collector_id) if booking.collector_id else None
    return {
        "success": True,
        "booking_reference": booking.booking_reference,
        "date": booking.date,
        "start_time": booking.start_time,
        "end_time": booking.end_time,
        "address_line": booking.address_line,
        "pincode": booking.pincode,
        "test_charges_inr": booking.test_charges_inr,
        "home_collection_charge_inr": booking.home_collection_charge_inr,
        "total_inr": booking.total_inr,
        "payment_policy": booking.payment_policy,
        "assignment_status": booking.assignment_status,
        "collector_name": collector.name if collector else None,
        "dispatch_state": booking.dispatch_state,
        "status": booking.status,
    }


def get_home_collection_booking(db: Session, booking_reference: str) -> dict:
    booking = db.query(HomeCollectionBooking).filter_by(booking_reference=booking_reference).first()
    if not booking:
        return {"found": False}
    return {"found": True, **_booking_dict(db, booking)}


def cancel_home_collection_booking(db: Session, booking_reference: str) -> dict:
    """ADDED BY SOURAV: mirrors booking_service.cancel_appointment()'s own shape -- delete the
    occupying hold row (freeing real capacity back to available_slots()) and mark the booking
    cancelled, never just one or the other."""
    booking = db.query(HomeCollectionBooking).filter_by(booking_reference=booking_reference, status="confirmed").first()
    if not booking:
        return {"success": False, "reason": "not_found"}
    hold = db.query(HomeCollectionSlotHold).filter_by(booking_id=booking.id, status="confirmed").first()
    if hold:
        db.delete(hold)
    booking.status = "cancelled"
    booking.dispatch_state = "CANCELLED"
    booking.cancelled_at = _now()
    db.commit()
    return {"success": True, "booking_reference": booking_reference}
