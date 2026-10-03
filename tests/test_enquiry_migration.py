"""Epic E27's lab_tests column migration, exercised the way the earlier
Epic E26 migration bug was actually caught: booting the REAL app via
TestClient, not calling migration functions by hand in a pre-corrected
order. See tests/test_i18n_migration.py::
test_the_real_app_startup_boots_an_old_database_end_to_end's docstring
for why that distinction matters -- it is the only thing that caught the
last ordering bug.

    python -m pytest tests/test_enquiry_migration.py -v
"""

import importlib
import os
import sqlite3
import sys

import pytest

CLINIC_API = os.path.join(os.path.dirname(__file__), "..", "clinic-api")


def test_fresh_database_boots_and_seeds_enquiry_facts(tmp_path, monkeypatch):
    path = tmp_path / "fresh.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.syspath_prepend(CLINIC_API)
    for mod in (
        "main",
        "db",
        "models",
        "seed",
        "booking_service",
        "booking_migrate",
        "enquiry_migrate",
        "i18n_content",
        "phonetic_match",
    ):
        sys.modules.pop(mod, None)

    from fastapi.testclient import TestClient

    clinic_main = importlib.import_module("main")
    with TestClient(clinic_main.app) as c:
        health = c.get("/api/health").json()
        assert health == {"status": "ok", "departments": 8, "doctors": 32, "lab_tests": 34}

        prep = c.get("/api/v1/tests/prep", params={"name": "Lipid Profile", "lang": "en"}).json()
        assert prep["found"] and "hours" in prep["prep_instructions"].lower()
    sys.modules.pop("main", None)


@pytest.fixture()
def old_db(tmp_path, monkeypatch):
    """An `appointments`-and-booking-era database (post Epic E26, pre
    Epic E27) -- lab_tests has every E26-era column but none of E27's."""
    path = tmp_path / "pre_e27.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.syspath_prepend(CLINIC_API)
    for mod in (
        "db",
        "models",
        "seed",
        "booking_service",
        "booking_migrate",
        "enquiry_migrate",
        "i18n_content",
        "main",
    ):
        sys.modules.pop(mod, None)

    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE);
        CREATE TABLE doctors (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, qualifications VARCHAR NOT NULL,
            aliases_bn VARCHAR NOT NULL DEFAULT '', aliases_hi VARCHAR NOT NULL DEFAULT '',
            department_id INTEGER NOT NULL, consultation_fee_inr INTEGER NOT NULL DEFAULT 500);
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
            status VARCHAR NOT NULL DEFAULT 'confirmed', UNIQUE(doctor_id, date, time_slot));
        INSERT INTO departments VALUES (1, 'General Medicine');
        INSERT INTO doctors VALUES (1, 'Dr. A. Sen', 'MBBS', 'সেন', 'सेन', 1, 500);
        INSERT INTO doctor_schedule VALUES (1, 1, 0, '10:00', '12:00');
        INSERT INTO lab_tests VALUES
            (1, 'Lipid Profile', 'লিপিড প্রোফাইল', 'लिपिड प्रोफाइल', 650, 'Blood', 24, 1, 'বাংলা', '', '');
    """)
    con.commit()
    con.close()
    yield path
    for mod in (
        "db",
        "models",
        "seed",
        "booking_service",
        "booking_migrate",
        "enquiry_migrate",
        "i18n_content",
        "main",
    ):
        sys.modules.pop(mod, None)


def test_old_database_gains_enquiry_columns_without_losing_the_existing_row(old_db):
    from fastapi.testclient import TestClient

    clinic_main = importlib.import_module("main")
    with TestClient(clinic_main.app) as c:
        health = c.get("/api/health").json()
        assert health["lab_tests"] == 1  # the pre-existing row, not reseeded

        prep = c.get("/api/v1/tests/prep", params={"name": "Lipid Profile", "lang": "en"}).json()
        assert prep["found"]
        assert "hours" in prep["prep_instructions"].lower()  # the OLD bengali-sourced prep, not lost

    models = importlib.import_module("models")
    db_mod = importlib.import_module("db")
    db = db_mod.SessionLocal()
    try:
        t = db.query(models.LabTest).filter_by(name="Lipid Profile").one()
        # Backfilled from enquiry_migrate.FASTING_HOURS_OVERRIDES.
        assert t.fasting_hours == 11
        assert t.home_collection_eligible is True
    finally:
        db.close()
    sys.modules.pop("main", None)
