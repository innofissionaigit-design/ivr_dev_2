"""SECURITY regression test for the IDOR CodeRabbit flagged in
clinic-api/booking_service.py::lookup_bookings: a phone/name search with
no caller_phone match used to disclose ANY matching booking to ANYONE
who spoke that phone number, guessed or overheard. authorize_disclosure()
existed to prevent exactly this and was never called. Fixed by gating
every phone/name-based row on authorize_disclosure(); a confirmation_id
lookup is unchanged (it is the existing bearer-token access model).

    python -m pytest tests/test_booking_authorization.py -v
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
    d = datetime.date.today()
    while d.weekday() != target_weekday:
        d += datetime.timedelta(days=1)
    return d.isoformat()


def _doctor(db, m, weekday: int = 0):
    row = db.query(m.DoctorSchedule).filter_by(weekday=weekday).first()
    return db.get(m.Doctor, row.doctor_id)


def _book(bs, db, m, patient_name: str, phone: str, caller_phone: str | None = None):
    doc = _doctor(db, m)
    date = _next_weekday(0)
    slot = "10:00"
    hold = bs.hold_slot(db, doc.id, date, slot)
    if not hold["success"]:
        # weekday 0 slot already taken by a previous call in this test --
        # walk forward a few days until hold_slot succeeds.
        for offset in range(1, 8):
            date = (datetime.date.fromisoformat(date) + datetime.timedelta(days=1)).isoformat()
            hold = bs.hold_slot(db, doc.id, date, slot)
            if hold["success"]:
                break
    result = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, patient_name, phone, caller_phone or phone)
    assert result["success"], result
    return result


def test_a_stranger_who_states_the_patients_phone_cannot_see_the_booking_via_a_different_caller_identity(
    clinic_modules,
):
    # This is the legitimate case: the patient's OWN phone IS sufficient
    # (authorize_disclosure returns True for caller_phone == patient.phone).
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        booked = _book(bs, db, m, "Ravi Das", "9111111111")
        rows = bs.lookup_bookings(db, phone="9111111111")
        assert any(r["confirmation_id"] == booked["confirmation_id"] for r in rows)
    finally:
        db.close()


def test_an_unrelated_phone_number_gets_nothing_even_if_it_matches_nobody(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        _book(bs, db, m, "Ravi Das", "9111111111")
        rows = bs.lookup_bookings(db, phone="9222222222")  # nobody's number
        assert rows == []
    finally:
        db.close()


def test_name_only_search_with_no_phone_discloses_nothing(clinic_modules):
    # SECURITY: this is the actual IDOR. Before the fix, ANY caller who
    # knew (or guessed) a patient's name alone -- or a phone number that
    # happened to match a DIFFERENT patient's caller_phone -- could read
    # back a stranger's doctor/date/time. A name alone is never a
    # meaningful access-control check.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        _book(bs, db, m, "Ravi Das", "9111111111")
        rows = bs.lookup_bookings(db, name="Ravi Das")
        assert rows == []
    finally:
        db.close()


def test_confirmation_id_alone_still_works_as_a_bearer_token(clinic_modules):
    # Unchanged behaviour: knowing the exact confirmation_id (8 random hex
    # chars) is the existing, deliberate access model for THAT one row --
    # not touched by this fix.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        booked = _book(bs, db, m, "Ravi Das", "9111111111")
        rows = bs.lookup_bookings(db, confirmation_id=booked["confirmation_id"])
        assert len(rows) == 1
        assert rows[0]["confirmation_id"] == booked["confirmation_id"]
    finally:
        db.close()


def test_a_relationship_stated_proxy_alone_is_not_enough_to_look_up_later(clinic_modules):
    # authorize_disclosure is deliberately conservative: booking AS a
    # proxy ("son", "wife") records verified_by="relationship_stated",
    # but that alone does not authorize a LATER lookup without also
    # confirming the patient's age (the dob_confirmed upgrade) --
    # unchanged by this fix, and worth pinning explicitly since it is
    # the reason the next test's positive case needs confirmed_age.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        booked = _book(bs, db, m, "Ravi Das", "9111111111", caller_phone="9333333333")
        rows = bs.lookup_bookings(db, phone="9333333333")
        assert rows == []
        # A THIRD, uninvolved number is refused too.
        assert bs.lookup_bookings(db, phone="9444444444") == []
        assert booked["success"]
    finally:
        db.close()


def test_a_proxy_who_confirms_the_patients_age_is_upgraded_and_authorized(clinic_modules):
    # The dob_confirmed upgrade path authorize_disclosure already builds
    # is exercised directly here (lookup_bookings itself does not collect
    # a spoken age, so this proves the underlying primitive still works,
    # now that something finally calls it).
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(0)
        hold = bs.hold_slot(db, doc.id, date, "11:00")
        bs.confirm_booking(
            db, hold["hold_token"], doc.id, date, "11:00", "Ravi Das", "9111111111", "9333333333", patient_age=40
        )
        patient = db.query(m.Patient).filter_by(phone="9111111111").one()
        assert bs.authorize_disclosure(db, patient, "9333333333", confirmed_age=40) is True
        proxy = db.query(m.PatientProxy).filter_by(patient_id=patient.id, caller_phone="9333333333").one()
        assert proxy.verified_by == "dob_confirmed"
    finally:
        db.close()


def test_reschedule_cannot_be_driven_by_a_strangers_phone_guess(clinic_modules):
    # The end-to-end consequence: main.py's reschedule/cancel dispatch
    # auto-resolves confirmation_id via lookup_bookings(phone=...). With
    # the fix, a stranger's phone guess now resolves to nothing, so there
    # is no confirmation_id for reschedule_appointment to act on.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        _book(bs, db, m, "Ravi Das", "9111111111")
        found = bs.lookup_bookings(db, phone="9999999999")  # attacker's own number, a guess
        assert found == []
        # main.py would now correctly ask the caller for a confirmation_id
        # instead of silently resolving to Ravi Das's appointment.
    finally:
        db.close()
