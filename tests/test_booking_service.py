"""Unit tests for clinic-api/booking_service.py's core lifecycle:
confirm, cancel, reschedule, multi-test booking, proxy authorisation,
conflict detection, and the SMS placeholder -- covers KCD-371, 372, 364,
365, 374, 377, 378 at the data-layer level. Conversation-turn behaviour
(what the caller hears) is covered separately in tests/test_booking_flow.py
and the i18n reply-template tests.

    python -m pytest tests/test_booking_service.py -v
"""

import datetime
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def clinic_modules():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)
    for mod in (
        "main",
        "db",
        "models",
        "seed",
        "booking_service",
        "booking_migrate",
        "enquiry_migrate",
        "i18n_content",
    ):
        sys.modules.pop(mod, None)

    import seed as seed_mod

    seed_mod.seed()
    import booking_migrate

    booking_migrate.migrate_booking_schema()
    booking_migrate.finish_booking_schema_setup()
    import booking_service as bs
    import db as db_mod
    import models as m

    yield bs, db_mod, m

    try:
        os.remove(db_path)
    except OSError:
        pass


def _next_weekday(target_weekday: int) -> str:
    # CodeRabbit-flagged: start tomorrow, not today -- if today IS
    # target_weekday but this doctor's chamber hours for today have
    # already passed, available_slots() correctly returns no slots (the
    # same-day past-time filter), and a caller indexing free[0] fails on
    # an empty list for a reason that has nothing to do with what the
    # test is actually checking.
    d = datetime.date.today() + datetime.timedelta(days=1)
    while d.weekday() != target_weekday:
        d += datetime.timedelta(days=1)
    return d.isoformat()


def _doctor(db, m, weekday: int = 0):
    """A doctor who actually sits on `weekday`, so booking calls don't
    depend on which SHIFT_TEMPLATES rotation seed.py happened to assign."""
    row = db.query(m.DoctorSchedule).filter_by(weekday=weekday).first()
    return db.get(m.Doctor, row.doctor_id)


def test_confirm_books_and_creates_a_patient_and_a_self_proxy(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        assert hold["success"]

        result = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "Ravi Das", "9800000001", "9800000001")
        assert result["success"] and result["confirmation_id"].startswith("KCD-")

        patient = db.query(m.Patient).filter_by(name="Ravi Das", phone="9800000001").one()
        proxy = db.query(m.PatientProxy).filter_by(patient_id=patient.id).one()
        assert proxy.verified_by == "self"

        sms = db.query(m.SmsOutbox).filter_by(related_confirmation_id=result["confirmation_id"]).one()
        assert sms.status == "queued"  # never claims "sent"
    finally:
        db.close()


def test_confirm_with_a_different_caller_phone_records_a_proxy(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        result = bs.confirm_booking(
            db,
            hold["hold_token"],
            doc.id,
            date,
            slot,
            "Anita Sen",
            "9800000002",
            caller_phone="9800000099",
            patient_age=62,
            relationship_label="son",
        )
        assert result["success"]
        patient = db.query(m.Patient).filter_by(name="Anita Sen", phone="9800000002").one()
        proxy = db.query(m.PatientProxy).filter_by(patient_id=patient.id, caller_phone="9800000099").one()
        assert proxy.relationship_label == "son"
        assert proxy.verified_by == "relationship_stated"

        # KCD-364's stricter disclosure rule: relationship alone is not
        # enough to read the record back; the age must also match.
        assert bs.authorize_disclosure(db, patient, "9800000099") is False
        assert bs.authorize_disclosure(db, patient, "9800000099", confirmed_age=62) is True
    finally:
        db.close()


def test_expired_hold_cannot_be_confirmed(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        lock = db.get(m.SlotLock, (doc.id, date, slot))
        lock.hold_expires_at = datetime.datetime.now() - datetime.timedelta(seconds=1)
        db.commit()

        result = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "1", "1")
        assert result == {"success": False, "reason": "hold_expired"}
        # and the slot is free again for someone else
        assert slot in bs.available_slots(db, doc.id, date)
    finally:
        db.close()


def test_cancel_outside_the_charging_window_is_free(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        # Force a date well outside the 24h window regardless of "today".
        far = datetime.date.today() + datetime.timedelta(days=10)
        while far.weekday() != (doc.schedule[0].weekday if doc.schedule else 0):
            far += datetime.timedelta(days=1)
        date = far.isoformat()
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "111", "111")

        result = bs.cancel_appointment(db, booked["confirmation_id"])
        assert result == {"success": True, "confirmation_id": booked["confirmation_id"], "charge_inr": 0}
        # the slot must be bookable again
        assert slot in bs.available_slots(db, doc.id, date)
    finally:
        db.close()


def test_a_cancelled_slot_can_actually_be_rebooked_by_a_second_caller(clinic_modules):
    # CodeRabbit-flagged, real bug: available_slots() correctly showed the
    # slot free again (the test above), but the OLD blanket
    # UniqueConstraint("doctor_id", "date", "time_slot") on Appointment
    # still had the CANCELLED row occupying that key, so the second
    # caller's confirm_booking raised an uncaught IntegrityError on
    # INSERT -- "free" and "actually rebookable" were not the same thing.
    # Fixed via models.Appointment's partial unique index (status='confirmed'
    # only) plus booking_migrate.rebuild_appointments_partial_unique_index()
    # for an existing database. This test proves the FULL cycle, not just
    # that SlotLock looks clear.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(0)
        slot = bs.available_slots(db, doc.id, date)[0]

        hold1 = bs.hold_slot(db, doc.id, date, slot)
        first = bs.confirm_booking(db, hold1["hold_token"], doc.id, date, slot, "First Patient", "111", "111")
        assert first["success"]

        cancelled = bs.cancel_appointment(db, first["confirmation_id"])
        assert cancelled["success"]

        hold2 = bs.hold_slot(db, doc.id, date, slot)
        assert hold2["success"]
        second = bs.confirm_booking(db, hold2["hold_token"], doc.id, date, slot, "Second Patient", "222", "222")
        assert second["success"], second  # used to raise IntegrityError here
        assert second["confirmation_id"] != first["confirmation_id"]
    finally:
        db.close()


def test_cancel_inside_the_charging_window_requires_explicit_confirmation(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        # hold_slot/confirm_booking do not themselves validate against
        # DoctorSchedule (that happens one layer up, when a slot is
        # offered) -- so a slot just over an hour from now, well inside
        # the 24h charging window, is deterministic regardless of which
        # weekday this doctor sits or what time "now" happens to be.
        soon = datetime.datetime.now() + datetime.timedelta(hours=1)
        date, slot = soon.date().isoformat(), soon.strftime("%H:%M")
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "111", "111")

        first = bs.cancel_appointment(db, booked["confirmation_id"])
        assert first["success"] is False and first["reason"] == "charge_confirmation_required"
        assert first["charge_inr"] == round(doc.consultation_fee_inr * 0.5)

        second = bs.cancel_appointment(db, booked["confirmation_id"], confirm_charge=True)
        assert second["success"] is True and second["charge_inr"] == first["charge_inr"]
    finally:
        db.close()


def test_available_slots_excludes_already_passed_times_today(clinic_modules):
    # CodeRabbit-flagged, real bug: available_slots() only checked
    # SlotLock, never the current time-of-day, so "today at 10am" stayed
    # offerable (and directly bookable) at 3pm.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        today = datetime.date.today()
        # Overwrite the schedule to span the whole day, so this test's
        # expectations do not depend on the seeded doctor's actual
        # chamber hours relative to whenever it happens to run.
        sched = db.query(m.DoctorSchedule).filter_by(doctor_id=doc.id, weekday=today.weekday()).first()
        if not sched:
            sched = m.DoctorSchedule(doctor_id=doc.id, weekday=today.weekday(), start_time="00:00", end_time="23:45")
            db.add(sched)
        else:
            sched.start_time, sched.end_time = "00:00", "23:45"
        db.commit()

        now = datetime.datetime.now()
        now_minutes = now.hour * 60 + now.minute

        # Slots are on 15-minute boundaries (SLOT_STEP_MIN) starting at
        # 00:00 -- round to one to guarantee it is actually IN the
        # generated list, not just near a real clock time.
        def _slot_at(offset_minutes: int) -> str:
            total = max(0, min(23 * 60 + 45, (now_minutes // 15) * 15 + offset_minutes))
            return f"{total // 60:02d}:{total % 60:02d}"

        past_slot = _slot_at(-60) if now_minutes >= 75 else None
        future_slot = _slot_at(60) if now_minutes <= 22 * 60 + 45 else None

        free = bs.available_slots(db, doc.id, today.isoformat())
        if past_slot:
            assert past_slot not in free
        if future_slot:
            assert future_slot in free
    finally:
        db.close()


def test_reschedule_moves_the_appointment_and_frees_the_old_slot(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slots = bs.available_slots(db, doc.id, date)
        old_slot, new_slot = slots[0], slots[1]
        hold = bs.hold_slot(db, doc.id, date, old_slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, old_slot, "X", "111", "111")

        result = bs.reschedule_appointment(db, booked["confirmation_id"], date, new_slot)
        assert result["success"] and result["time_slot"] == new_slot
        assert old_slot in bs.available_slots(db, doc.id, date)
        assert new_slot not in bs.available_slots(db, doc.id, date)

        old_appt = db.query(m.Appointment).filter_by(confirmation_id=booked["confirmation_id"]).one()
        assert old_appt.status == "rescheduled"
    finally:
        db.close()


def test_reschedule_onto_a_taken_slot_leaves_the_original_intact(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slots = bs.available_slots(db, doc.id, date)
        mine, contested = slots[0], slots[1]
        hold = bs.hold_slot(db, doc.id, date, mine)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, mine, "X", "111", "111")
        # someone else takes the slot we're about to try to move onto
        other_hold = bs.hold_slot(db, doc.id, date, contested)
        bs.confirm_booking(db, other_hold["hold_token"], doc.id, date, contested, "Y", "222", "222")

        result = bs.reschedule_appointment(db, booked["confirmation_id"], date, contested)
        assert result["success"] is False and result["reason"] == "slot_taken"

        still = db.query(m.Appointment).filter_by(confirmation_id=booked["confirmation_id"]).one()
        assert still.status == "confirmed" and still.time_slot == mine
        assert mine not in bs.available_slots(db, doc.id, date)  # original booking still holds it
    finally:
        db.close()


# ============================================================ KCD-486
# "A confirmation number is spoken only after the write is verified" --
# confirm_booking/reschedule_appointment now re-read the row after commit
# rather than trusting that db.commit() not raising means the caller's
# specific values landed. See booking_service._verify_appointment_persisted's
# own docstring for why this matters more once a real system of record is
# behind this function than it does against this prototype's SQLite today.


def test_write_is_verified_before_a_confirmation_number_is_returned(clinic_modules):
    # The ordinary path: verification passes, success looks exactly as it
    # always did -- this is a regression guard that adding the check did
    # not change the happy path's outcome shape.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        result = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "111", "111")
        assert result["success"] is True
        assert result["confirmation_id"].startswith("KCD-")
    finally:
        db.close()


def test_an_unverifiable_booking_write_holds_instead_of_confirming(clinic_modules):
    # Simulates the system-of-record verification failing (the case that
    # can't actually happen against this prototype's own SQLite -- see
    # _verify_appointment_persisted's docstring) by monkeypatching it
    # directly, to prove the CALLER (main.py) gets a distinct outcome it
    # can hold-and-escalate on, never a confirmation number for a write
    # that wasn't actually confirmed as landed.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)

        original = bs._verify_appointment_persisted
        bs._verify_appointment_persisted = lambda *a, **kw: False
        try:
            result = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "111", "111")
        finally:
            bs._verify_appointment_persisted = original

        assert result["success"] is False
        assert result["reason"] == "write_unverified"
        assert result["confirmation_id"].startswith("KCD-")  # logged, never spoken as confirmed
    finally:
        db.close()


def test_an_unverifiable_reschedule_write_holds_instead_of_confirming(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slots = bs.available_slots(db, doc.id, date)
        old_slot, new_slot = slots[0], slots[1]
        hold = bs.hold_slot(db, doc.id, date, old_slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, old_slot, "X", "111", "111")

        original = bs._verify_appointment_persisted
        bs._verify_appointment_persisted = lambda *a, **kw: False
        try:
            result = bs.reschedule_appointment(db, booked["confirmation_id"], date, new_slot)
        finally:
            bs._verify_appointment_persisted = original

        assert result["success"] is False
        assert result["reason"] == "write_unverified"
    finally:
        db.close()


# ============================================================ KCD-487
# "A reschedule swaps atomically or leaves the original intact... The
# failure path is exercised by fault injection in the integration suite."
# test_reschedule_onto_a_taken_slot_leaves_the_original_intact (above)
# already covers the ORDINARY business-logic failure (slot taken); this
# covers a genuine unexpected crash mid-transaction, which is what "fault
# injection" actually means -- proving the atomicity guarantee holds even
# when something goes wrong that the code did not anticipate, not just
# when it correctly detects an expected condition.


def test_a_fault_during_reschedule_leaves_the_original_appointment_untouched(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        doctor_id = doc.id  # captured now: `doc` is detached once this session closes below
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slots = bs.available_slots(db, doc.id, date)
        old_slot, new_slot = slots[0], slots[1]
        hold = bs.hold_slot(db, doc.id, date, old_slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, old_slot, "X", "111", "111")
        confirmation_id = booked["confirmation_id"]

        # Let the FIRST db.commit() (hold_slot's own, claiming the new
        # slot) succeed normally, then raise on the SECOND (reschedule_
        # appointment's own final commit) -- simulating a crash after the
        # new slot is provisionally held but before the swap is durable.
        real_commit = db.commit
        calls = {"n": 0}

        def flaky_commit():
            calls["n"] += 1
            if calls["n"] == 1:
                return real_commit()
            raise RuntimeError("simulated database failure mid-reschedule")

        db.commit = flaky_commit
        try:
            with pytest.raises(RuntimeError, match="simulated database failure"):
                bs.reschedule_appointment(db, confirmation_id, date, new_slot)
        finally:
            db.commit = real_commit
            db.rollback()
    finally:
        db.close()

    # Verify from a FRESH session/connection, not the one the fault was
    # injected into -- proves the failure never reached the database at
    # all, not just that this session's own view looks right.
    with db_mod.SessionLocal() as verify:
        appt = verify.query(m.Appointment).filter_by(confirmation_id=confirmation_id).one()
        assert appt.status == "confirmed"
        assert appt.time_slot == old_slot, "the original appointment must be untouched by the failed swap"

        leaked = verify.query(m.Appointment).filter_by(doctor_id=doctor_id, date=date, time_slot=new_slot).all()
        assert leaked == [], "no new appointment row may exist for a swap that never committed"

        old_lock = verify.get(m.SlotLock, (doctor_id, date, old_slot))
        assert old_lock is not None and old_lock.status == "confirmed", (
            "the original slot lock must still be held, never deleted by the failed swap"
        )


def test_earliest_available_finds_the_soonest_free_slot(clinic_modules):
    # KCD-360: caller asks for the earliest available appointment.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        result = bs.earliest_available(db, doc.id, datetime.date.today())
        assert result is not None
        assert result["date"] and result["time_slot"]
        # what it found must actually be free right now
        assert result["time_slot"] in bs.available_slots(db, doc.id, result["date"])

        # once that exact slot is taken, the search must skip past it
        hold = bs.hold_slot(db, doc.id, result["date"], result["time_slot"])
        bs.confirm_booking(
            db, hold["hold_token"], doc.id, result["date"], result["time_slot"], "Earliest Taker", "666", "666"
        )
        second = bs.earliest_available(db, doc.id, datetime.date.today())
        assert second is not None
        assert not (second["date"] == result["date"] and second["time_slot"] == result["time_slot"])
    finally:
        db.close()


def test_earliest_available_returns_none_when_nothing_is_free_in_the_horizon(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        # A doctor who never sits (no DoctorSchedule rows at all, simulated
        # by asking about a horizon of zero days) has nothing to offer --
        # must say so plainly, never guess a date.
        doc = _doctor(db, m)
        result = bs.earliest_available(db, doc.id, datetime.date.today(), horizon_days=0)
        assert result is None
    finally:
        db.close()


def test_lookup_bookings_finds_by_phone_and_by_confirmation_id(clinic_modules):
    # KCD-373: caller asks what they have booked.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "Lookup Patient", "777", "777")

        by_phone = bs.lookup_bookings(db, phone="777")
        assert len(by_phone) == 1 and by_phone[0]["confirmation_id"] == booked["confirmation_id"]

        by_id = bs.lookup_bookings(db, confirmation_id=booked["confirmation_id"])
        assert len(by_id) == 1 and by_id[0]["date"] == date

        assert bs.lookup_bookings(db, phone="no-such-number") == []

        # cancelled bookings must not be offered as "what you have booked"
        bs.cancel_appointment(db, booked["confirmation_id"], confirm_charge=True)
        assert bs.lookup_bookings(db, phone="777") == []
    finally:
        db.close()


def test_conflict_detection_finds_the_same_patient_double_booked(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "Same Patient", "333", "333")

        conflict = bs.find_conflict(db, "333", date, slot)
        assert conflict is not None and conflict["date"] == date
        assert bs.find_conflict(db, "not-a-real-number", date, slot) is None
    finally:
        db.close()


def test_multi_test_booking_and_add_test(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        tests = db.query(m.LabTest).limit(2).all()
        date = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        result = bs.book_tests(db, [t.id for t in tests], date, "Multi Test Patient", "444", "444")
        assert result["success"] and len(result["test_names"]) == 2
        assert result["total_rate_inr"] == sum(t.rate_inr for t in tests)

        third = db.query(m.LabTest).offset(2).first()
        added = bs.add_test_to_booking(db, result["confirmation_id"], third.name)
        assert added["success"]

        group_rows = (
            db.query(m.TestBooking)
            .filter_by(
                booking_group_id=db.query(m.TestBooking)
                .filter_by(confirmation_id=result["confirmation_id"])
                .first()
                .booking_group_id
            )
            .all()
        )
        assert len(group_rows) == 3

        dup = bs.add_test_to_booking(db, result["confirmation_id"], third.name)
        assert dup == {"success": False, "reason": "already_booked"}
    finally:
        db.close()


def test_resend_confirmation_is_rate_limited(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "5550001", "5550001")

        first = bs.resend_confirmation(db, booked["confirmation_id"])
        assert first["success"] and first["sent_to_last4"] == "0001"
        second = bs.resend_confirmation(db, booked["confirmation_id"])
        assert second == {"success": False, "reason": "rate_limited"}
    finally:
        db.close()


# ============================== CodeRabbit-flagged: "not_provided" sentinel


def test_two_different_patients_who_both_decline_a_phone_are_never_merged(clinic_modules):
    # Real bug: find_or_create_patient looked up by (name, phone), and
    # "not_provided" is not a real phone -- two DIFFERENT "Ravi Das"es who
    # both declined a number used to become the SAME Patient row.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        p1 = bs.find_or_create_patient(db, "Ravi Das", bs.NOT_PROVIDED_PHONE)
        p2 = bs.find_or_create_patient(db, "Ravi Das", bs.NOT_PROVIDED_PHONE)
        assert p1.id != p2.id
    finally:
        db.close()


def test_a_declined_phone_never_gets_a_queued_sms(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        result = bs.queue_sms(db, bs.NOT_PROVIDED_PHONE, "booking_confirmed", "hi")
        assert result == {"queued": False, "reason": "no_phone_on_file"}
        assert db.query(m.SmsOutbox).count() == 0
    finally:
        db.close()


def test_resend_confirmation_on_a_declined_phone_gives_an_honest_reason_not_a_nonsense_last4(clinic_modules):
    # Real bug: "not_provided"[-4:] == "ided", spoken as if it were a
    # real phone number's last four digits.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(
            db, hold["hold_token"], doc.id, date, slot, "X", bs.NOT_PROVIDED_PHONE, bs.NOT_PROVIDED_PHONE
        )
        result = bs.resend_confirmation(db, booked["confirmation_id"])
        assert result == {"success": False, "reason": "no_phone_on_file"}
    finally:
        db.close()


def test_department_routing_matches_and_flags_ambiguity(clinic_modules):
    bs, db_mod, _m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        result = bs.route_department(db, "amar buke khub betha hocche", "en")
        assert result == {"matched": False}  # no English keyword hit -- transliteration is out of scope here

        result = bs.route_department(db, "I have chest pain", "en")
        assert result["matched"] and result["department_name"] == "Cardiology"

        result = bs.route_department(db, "বুকে ব্যথা", "bn")
        assert result["matched"] and result["department_name"] == "Cardiology"
    finally:
        db.close()


def test_draft_booking_round_trip_and_expiry(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        bs.save_draft(db, "9990001", "call-1", '{"doctor_name": "Sen"}')
        found = bs.find_draft(db, "9990001")
        assert found and found["slots_json"] == '{"doctor_name": "Sen"}'

        row = db.query(m.DraftBooking).filter_by(caller_phone="9990001").one()
        row.expires_at = datetime.datetime.now() - datetime.timedelta(seconds=1)
        db.commit()
        assert bs.find_draft(db, "9990001") is None

        bs.save_draft(db, "9990002", "call-2", "{}")
        bs.clear_draft(db, "9990002")
        assert bs.find_draft(db, "9990002") is None
    finally:
        db.close()


def test_a_hand_pain_symptom_is_ambiguous_between_general_medicine_and_orthopaedics(clinic_modules):
    """KCD-lay-terms: general limb/body pain genuinely could be either -- route_department() must offer both, not
    silently pick one (booking_migrate.DEPARTMENT_ROUTE_KEYWORDS deliberately lists "hand pain" under both)."""
    bs, db_mod, _m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        result = bs.route_department(db, "I have bad hand pain", "en")
        assert result["matched"] is False and result["ambiguous"] is True
        assert set(result["candidates"]) == {"General Medicine", "Orthopaedics"}

        result_bn = bs.route_department(db, "আমার হাতে ব্যথা করছে", "bn")
        assert result_bn["ambiguous"] is True
        assert set(result_bn["candidates"]) == {"General Medicine", "Orthopaedics"}

        # An unambiguous General Medicine symptom is unaffected by the new shared keyword.
        result_fever = bs.route_department(db, "fever", "en")
        assert result_fever["matched"] is True and result_fever["department_name"] == "General Medicine"
    finally:
        db.close()


def test_send_payment_link_only_ever_finds_a_confirmed_doctor_appointment(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "Ravi Das", "9800000002", "9800000002")

        result = bs.send_payment_link(db, booked["confirmation_id"])
        assert result["success"] is True
        assert result["amount_inr"] == doc.consultation_fee_inr
        assert result["payment_link"].endswith(booked["confirmation_id"])
        assert result["payment_link"].startswith("https://") and ".invalid/" in result["payment_link"]

        sms = (
            db.query(m.SmsOutbox)
            .filter_by(related_confirmation_id=booked["confirmation_id"], template_key="payment_link")
            .one()
        )
        assert sms.status == "queued"  # never claims "sent" -- same discipline as every other queue_sms call

        # A lab test booking's confirmation_id lives in a different table entirely: never finds an appointment.
        cbc = db.query(m.LabTest).filter_by(name="Complete Blood Count (CBC)").one()
        tb_result = bs.book_tests(db, [cbc.id], date, "Ravi Das", "9800000003", "9800000003")
        assert bs.send_payment_link(db, tb_result["confirmation_id"]) == {"success": False, "reason": "not_found"}

        assert bs.send_payment_link(db, "KCD-NOPE") == {"success": False, "reason": "not_found"}
    finally:
        db.close()


def test_send_payment_link_with_no_phone_on_file_is_refused_not_queued(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(
            db, hold["hold_token"], doc.id, date, slot, "Ravi Das", bs.NOT_PROVIDED_PHONE, bs.NOT_PROVIDED_PHONE
        )
        result = bs.send_payment_link(db, booked["confirmation_id"])
        assert result == {"success": False, "reason": "no_phone_on_file"}
    finally:
        db.close()


def test_department_routing_names_real_doctors_from_the_matched_departments(clinic_modules):
    """KCD-lay-terms: a caller asking about chest pain hears actual doctor names, not just a department -- and a
    word inserted between the two words of a keyword ("বুকে খুব ব্যথা" for "বুকে ব্যথা") still matches."""
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        cardiology = db.query(m.Department).filter_by(name="Cardiology").one()
        expected = {d.name for d in db.query(m.Doctor).filter_by(department_id=cardiology.id).all()}

        result = bs.route_department(db, "I have chest pain", "en")
        assert result["matched"] and result["department_name"] == "Cardiology"
        assert result["doctors"] and set(result["doctors"]) <= expected
        assert len(result["doctors"]) <= bs.MAX_SUGGESTED_DOCTORS

        # the live-call regression case: an intensifier word inserted between the two keyword words
        gap = bs.route_department(db, "আমার বুকে খুব ব্যথা করছে", "bn")
        assert gap["matched"] and gap["department_name"] == "Cardiology" and gap["doctors"]

        ambiguous = bs.route_department(db, "I have bad hand pain", "en")
        assert ambiguous["ambiguous"] is True
        assert set(ambiguous["doctors_by_department"]) == set(ambiguous["candidates"])
        assert all(ambiguous["doctors_by_department"][c] for c in ambiguous["candidates"])
    finally:
        db.close()
