"""A database created BEFORE Epic E26's booking-lifecycle columns must
still boot and keep its existing rows.

Same shape of bug as tests/test_i18n_migration.py: SQLAlchemy selects
every mapped column, so a first ORM query against an old `appointments`
or `doctors` table would crash with "no such column" the moment
booking_service.py (which reads Appointment.status, Doctor.
consultation_fee_inr, etc.) touches it. Built from raw SQL matching the
CURRENT live schema (Appointment/Doctor as they exist before this
session's changes -- see clinic-api/models.py's git history), so this
test cannot pass by accident.

    python -m pytest tests/test_booking_migration.py -v
"""

import importlib
import os
import sqlite3
import sys

import pytest

CLINIC_API = os.path.join(os.path.dirname(__file__), "..", "clinic-api")


@pytest.fixture()
def old_db(tmp_path, monkeypatch):
    path = tmp_path / "old_booking.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.syspath_prepend(CLINIC_API)
    for mod in ("db", "models", "seed", "booking_service", "booking_migrate", "i18n_content"):
        sys.modules.pop(mod, None)

    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE);
        CREATE TABLE doctors (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, qualifications VARCHAR NOT NULL,
            aliases_bn VARCHAR NOT NULL DEFAULT '', aliases_hi VARCHAR NOT NULL DEFAULT '',
            department_id INTEGER NOT NULL);
        CREATE TABLE doctor_schedule (id INTEGER PRIMARY KEY, doctor_id INTEGER NOT NULL,
            weekday INTEGER NOT NULL, start_time VARCHAR NOT NULL, end_time VARCHAR NOT NULL);
        CREATE TABLE lab_tests (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE,
            aliases_bn VARCHAR NOT NULL DEFAULT '', aliases_hi VARCHAR NOT NULL DEFAULT '',
            rate_inr INTEGER NOT NULL, sample_type VARCHAR NOT NULL, report_time_hours INTEGER NOT NULL,
            fasting_required BOOLEAN NOT NULL DEFAULT 0, prep_instructions_bn VARCHAR NOT NULL DEFAULT '',
            prep_instructions_hi VARCHAR NOT NULL DEFAULT '', prep_instructions_en VARCHAR NOT NULL DEFAULT '');
        CREATE TABLE faqs (id INTEGER PRIMARY KEY, topic VARCHAR NOT NULL UNIQUE,
            keywords_bn VARCHAR NOT NULL DEFAULT '', answer_bn VARCHAR NOT NULL,
            answer_hi VARCHAR NOT NULL DEFAULT '', answer_en VARCHAR NOT NULL DEFAULT '');
        CREATE TABLE appointments (id INTEGER PRIMARY KEY, confirmation_id VARCHAR NOT NULL UNIQUE,
            doctor_id INTEGER NOT NULL, date VARCHAR NOT NULL, time_slot VARCHAR NOT NULL,
            patient_name VARCHAR NOT NULL, phone VARCHAR NOT NULL, created_at DATETIME NOT NULL,
            UNIQUE(doctor_id, date, time_slot));
        INSERT INTO departments VALUES (1, 'General Medicine');
        INSERT INTO doctors VALUES (1, 'Dr. A. Sen', 'MBBS', 'সেন', 'सेन', 1);
        INSERT INTO doctor_schedule VALUES (1, 1, 0, '10:00', '12:00');
        INSERT INTO lab_tests VALUES (1, 'Uric Acid', 'ইউরিক অ্যাসিড', 'यूरिक एसिड', 250, 'Blood', 12, 0, '', '', '');
        INSERT INTO appointments VALUES
            (1, 'KCD-OLD-0001', 1, '2026-01-05', '10:15', 'Existing Patient', '9000000000', '2026-01-01 09:00:00');
    """)
    con.commit()
    con.close()
    yield path
    for mod in ("db", "models", "seed", "booking_service", "booking_migrate", "i18n_content"):
        sys.modules.pop(mod, None)


def test_old_appointments_table_gains_columns_without_losing_the_existing_row(old_db):
    booking_migrate = importlib.import_module("booking_migrate")
    models = importlib.import_module("models")
    db_mod = importlib.import_module("db")

    # Matches clinic-api/main.py's real startup order: create_all() adds
    # any brand-new table (SlotLock, Patient, DepartmentRoute, ...) --
    # a no-op on tables that already exist -- BEFORE the column migration,
    # which both ALTERs the old tables and queries DepartmentRoute. Then
    # finish_booking_schema_setup() only AFTER, once departments/doctors
    # are known to have rows (this fixture's raw SQL already inserted one
    # of each, but a fresh empty database would not have at this point --
    # see finish_booking_schema_setup()'s docstring for the bug that caught).
    models.Base.metadata.create_all(db_mod.engine)
    result = booking_migrate.migrate_booking_schema()
    assert "appointments.status" in result["columns_added"]
    assert "appointments.patient_id" in result["columns_added"]
    assert "doctors.consultation_fee_inr" in result["columns_added"]

    db = db_mod.SessionLocal()
    try:
        appt = db.query(models.Appointment).filter_by(confirmation_id="KCD-OLD-0001").one()
        assert appt.patient_name == "Existing Patient"  # nothing lost
        assert appt.status == "confirmed"  # ALTER TABLE default applied
        assert appt.patient_id is None  # not retroactively linked -- honest, not guessed

        doc = db.get(models.Doctor, 1)
        assert doc.consultation_fee_inr == 500  # raw ALTER default before backfill

        # And the new tables this old DB never had are now queryable too.
        assert db.query(models.SlotLock).count() == 0
        assert db.query(models.Patient).count() == 0
    finally:
        db.close()

    finish = booking_migrate.finish_booking_schema_setup()
    assert finish["department_routes_added"] == 1  # the fixture's one department


def test_old_blanket_unique_constraint_is_rebuilt_into_a_partial_index(old_db):
    """CodeRabbit-flagged, real bug: this fixture's old appointments table
    (line ~52 above) has the ORIGINAL blanket UNIQUE(doctor_id, date,
    time_slot) baked in via raw SQL, with an existing CONFIRMED row
    already occupying (doctor_id=1, date='2026-01-05', time_slot='10:15').
    Proves the full end-to-end consequence on an actually-migrated
    database: cancel that row, then rebook the SAME slot -- which used to
    raise an uncaught IntegrityError on the INSERT, because the cancelled
    row still occupied the old constraint's key."""
    booking_migrate = importlib.import_module("booking_migrate")
    models = importlib.import_module("models")
    db_mod = importlib.import_module("db")
    booking_service = importlib.import_module("booking_service")

    models.Base.metadata.create_all(db_mod.engine)
    result = booking_migrate.migrate_booking_schema()
    assert result["appointments_table_rebuilt"] is True

    db = db_mod.SessionLocal()
    try:
        # The pre-existing row survived the table rebuild.
        appt = db.query(models.Appointment).filter_by(confirmation_id="KCD-OLD-0001").one()
        assert appt.patient_name == "Existing Patient"
        assert appt.status == "confirmed"

        # confirm_charge=True unconditionally: the fixture's fixed date
        # (2026-01-05) is in the past relative to whenever this test
        # actually runs, which cancellation_charge() reads as "inside
        # the charging window" and would otherwise refuse the first call.
        cancelled = booking_service.cancel_appointment(db, "KCD-OLD-0001", confirm_charge=True)
        assert cancelled["success"], cancelled

        hold = booking_service.hold_slot(db, 1, "2026-01-05", "10:15")
        assert hold["success"]
        rebooked = booking_service.confirm_booking(
            db, hold["hold_token"], 1, "2026-01-05", "10:15", "New Patient", "9111111111", "9111111111"
        )
        assert rebooked["success"], rebooked  # used to raise IntegrityError here
    finally:
        db.close()

    # Idempotent: a second call on the now-rebuilt table is a no-op.
    assert booking_migrate.migrate_booking_schema()["appointments_table_rebuilt"] is False


def test_migration_is_idempotent_and_never_overwrites_a_hand_edited_fee(old_db):
    booking_migrate = importlib.import_module("booking_migrate")
    models = importlib.import_module("models")
    db_mod = importlib.import_module("db")

    models.Base.metadata.create_all(db_mod.engine)
    booking_migrate.migrate_booking_schema()
    first = booking_migrate.finish_booking_schema_setup()
    # The fixture's one doctor is General Medicine, whose fee (500) equals
    # the raw ALTER default -- so the backfill correctly has nothing to do.
    assert first["doctor_fees_filled"] == 0

    db = db_mod.SessionLocal()
    try:
        doc = db.get(models.Doctor, 1)
        doc.consultation_fee_inr = 999  # a clinician's manual edit
        db.commit()
    finally:
        db.close()

    second_columns = booking_migrate.migrate_booking_schema()
    second = booking_migrate.finish_booking_schema_setup()
    assert second_columns["columns_added"] == []  # already added
    assert second["doctor_fees_filled"] == 0  # 999 != 500, so the backfill left it alone

    db = db_mod.SessionLocal()
    try:
        assert db.get(models.Doctor, 1).consultation_fee_inr == 999
    finally:
        db.close()
