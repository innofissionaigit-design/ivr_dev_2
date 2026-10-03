"""KCD-084 persistence: the senior-mode flag is read and written only for a
caller authorised for that patient (authorize_disclosure), and the schema
default for the new BOOLEAN columns is the same on a fresh and a migrated
database.

    python -m pytest tests/test_senior_mode_authorization.py -v
"""

import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def clinic(monkeypatch):
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("CLINIC_DB_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
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


def _patient(db, m, phone="9000000001", name="Asha"):
    p = m.Patient(name=name, phone=phone, age=70, created_at=__import__("datetime").datetime(2030, 1, 1))
    db.add(p)
    db.commit()
    return p


def test_the_patients_own_number_can_set_and_read_the_flag(clinic):
    bs, db_mod, m = clinic
    db = db_mod.SessionLocal()
    try:
        _patient(db, m)
        assert bs.set_patient_senior(db, "9000000001", True, caller_phone="9000000001") == 1
        assert bs.get_patient_senior(db, "9000000001", caller_phone="9000000001") is True
    finally:
        db.close()


def test_a_different_number_cannot_change_or_read_someone_elses_flag(clinic):
    bs, db_mod, m = clinic
    db = db_mod.SessionLocal()
    try:
        _patient(db, m)
        assert bs.set_patient_senior(db, "9000000001", True, caller_phone="9111111111") == 0
        assert bs.get_patient_senior(db, "9000000001", caller_phone="9111111111") is False
        # ... and the flag really was not set by the refused write
        assert bs.get_patient_senior(db, "9000000001", caller_phone="9000000001") is False
    finally:
        db.close()


def test_a_flag_that_is_set_is_not_readable_by_a_stranger(clinic):
    bs, db_mod, m = clinic
    db = db_mod.SessionLocal()
    try:
        _patient(db, m)
        bs.set_patient_senior(db, "9000000001", True, caller_phone="9000000001")
        assert bs.get_patient_senior(db, "9000000001", caller_phone="9111111111") is False
    finally:
        db.close()


def test_a_verified_proxy_may_set_the_flag(clinic):
    bs, db_mod, m = clinic
    db = db_mod.SessionLocal()
    try:
        p = _patient(db, m)
        db.add(
            m.PatientProxy(
                patient_id=p.id,
                caller_phone="9222222222",
                relationship_label="son",
                verified_by="dob_confirmed",
                created_at=__import__("datetime").datetime(2030, 1, 1),
            )
        )
        db.commit()
        assert bs.set_patient_senior(db, "9000000001", True, caller_phone="9222222222") == 1
    finally:
        db.close()


def test_boolean_columns_have_a_database_default_on_a_fresh_schema(clinic):
    """create_all() must give the same default the ALTER migration gives, or a
    raw insert that omits the column works on one database and fails on the other."""
    bs, db_mod, m = clinic
    from sqlalchemy import text

    db = db_mod.SessionLocal()
    try:
        db.execute(
            text("INSERT INTO patients (name, phone, created_at) VALUES ('Raw', '9333333333', '2030-01-01 00:00:00')")
        )
        db.execute(
            text(
                "INSERT INTO cancellation_policies (version, effective_from, free_window_hours, charge_percent) "
                "VALUES (9, '2031-01-01', 1, 1)"
            )
        )
        db.commit()
        row = db.query(m.Patient).filter_by(phone="9333333333").one()
        assert not row.senior_mode
        assert db.query(m.CancellationPolicy).filter_by(version=9).one().refund_eligible
    finally:
        db.close()
