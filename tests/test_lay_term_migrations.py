"""A database created BEFORE the lay-term/prescription-callback work must still boot and keep its existing rows.

Same shape of bug as tests/test_booking_migration.py and tests/test_i18n_migration.py: SQLAlchemy selects every
mapped column, so the first ORM query against an old `callback_requests` table (no `call_summary` column) or an old
`department_routes` row (missing the new ambiguous-symptom keywords) needs the right migration to have run first.

    python -m pytest tests/test_lay_term_migrations.py -v
"""

import importlib
import os
import sqlite3
import sys

import pytest

CLINIC_API = os.path.join(os.path.dirname(__file__), "..", "clinic-api")


@pytest.fixture()
def old_db(tmp_path, monkeypatch):
    path = tmp_path / "old_lay_terms.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.syspath_prepend(CLINIC_API)
    for mod in ("db", "models", "seed", "booking_service", "booking_migrate", "enquiry_migrate", "i18n_content"):
        sys.modules.pop(mod, None)

    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE);
        CREATE TABLE doctors (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, qualifications VARCHAR NOT NULL,
            aliases_bn VARCHAR NOT NULL DEFAULT '', aliases_hi VARCHAR NOT NULL DEFAULT '',
            department_id INTEGER NOT NULL);
        CREATE TABLE department_routes (id INTEGER PRIMARY KEY, department_id INTEGER NOT NULL,
            keywords_bn VARCHAR NOT NULL DEFAULT '', keywords_hi VARCHAR NOT NULL DEFAULT '',
            keywords_en VARCHAR NOT NULL DEFAULT '');
        CREATE TABLE callback_requests (id INTEGER PRIMARY KEY, phone VARCHAR NOT NULL, call_id VARCHAR NOT NULL,
            requested_window VARCHAR NOT NULL, reason VARCHAR NOT NULL DEFAULT '',
            status VARCHAR NOT NULL DEFAULT 'scheduled', created_at DATETIME NOT NULL);
        INSERT INTO departments VALUES (1, 'General Medicine'), (2, 'Orthopaedics');
        -- an OLD route row, seeded before "hand pain" existed, and a hand-edited extra keyword a clinician added
        INSERT INTO department_routes VALUES (1, 1, 'জ্বর|clinician added this', '', 'fever');
        INSERT INTO callback_requests VALUES
            (1, '9000000000', 'old-call-1', 'this evening', 'billing question', 'scheduled', '2026-01-01 09:00:00');
    """)
    con.commit()
    con.close()
    yield path
    for mod in ("db", "models", "seed", "booking_service", "booking_migrate", "enquiry_migrate", "i18n_content"):
        sys.modules.pop(mod, None)


def test_old_callback_requests_table_gains_call_summary_without_losing_the_existing_row(old_db):
    enquiry_migrate = importlib.import_module("enquiry_migrate")
    models = importlib.import_module("models")
    db_mod = importlib.import_module("db")

    models.Base.metadata.create_all(db_mod.engine)  # a no-op on the table that already exists (raw SQL, above)
    added = enquiry_migrate.add_callback_request_columns()
    assert added == ["callback_requests.call_summary"]

    db = db_mod.SessionLocal()
    try:
        row = db.query(models.CallbackRequest).filter_by(call_id="old-call-1").one()
        assert row.phone == "9000000000"  # nothing lost
        assert row.call_summary == ""  # the ALTER TABLE default
    finally:
        db.close()

    # idempotent: running it again adds nothing more
    assert enquiry_migrate.add_callback_request_columns() == []


def test_old_department_route_row_gains_new_keywords_without_losing_a_clinicians_own_edit(old_db):
    booking_migrate = importlib.import_module("booking_migrate")
    models = importlib.import_module("models")
    db_mod = importlib.import_module("db")

    models.Base.metadata.create_all(db_mod.engine)
    updated = booking_migrate.backfill_department_route_keywords()
    assert updated >= 1

    db = db_mod.SessionLocal()
    try:
        row = db.query(models.DepartmentRoute).filter_by(department_id=1).one()
        kws = row.keywords_bn.split("|")
        assert "জ্বর" in kws  # the original seed keyword survives
        assert "clinician added this" in kws  # a hand-edited addition is never removed
        assert "হাতে ব্যথা" in kws  # the new keyword was appended
        # Orthopaedics (department_id=2) had NO route row at all yet -- backfill only touches EXISTING rows;
        # seed_department_routes() (run first in finish_booking_schema_setup, not called by this narrow test) is
        # what creates a department's first row.
        assert db.query(models.DepartmentRoute).filter_by(department_id=2).first() is None
    finally:
        db.close()

    # idempotent and additive: running it again does not duplicate the keyword
    booking_migrate.backfill_department_route_keywords()
    db = db_mod.SessionLocal()
    try:
        row = db.query(models.DepartmentRoute).filter_by(department_id=1).one()
        assert row.keywords_bn.split("|").count("হাতে ব্যথা") == 1
    finally:
        db.close()
