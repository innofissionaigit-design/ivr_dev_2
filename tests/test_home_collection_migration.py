"""KCD-387 full flow, section 39 (mandatory): a fresh database boots with the new home-collection
tables, and an EXISTING database (pre-dating this story, with real departments/doctors/lab_tests/
coverage data already in it) upgrades in place -- the new tables are added by
Base.metadata.create_all's own ordinary "create what's missing" behaviour (these are brand-new
tables, never a column ALTER on an existing one, so there is no hand-written migration function to
test here, unlike enquiry_migrate.add_enquiry_columns) -- and every pre-existing row survives
untouched. Boots the REAL app via TestClient, the same "the real app startup, not migration
functions called by hand" discipline tests/test_enquiry_migration.py's own docstring explains.

    python -m pytest tests/test_home_collection_migration.py -v
"""

import importlib
import os
import sys

import pytest

CLINIC_API = os.path.join(os.path.dirname(__file__), "..", "clinic-api")

NEW_TABLES = {
    "home_collection_slots",
    "home_collection_slot_holds",
    "home_collection_collectors",
    "home_collection_payment_policies",
    "home_collection_bookings",
    "home_collection_booking_tests",
}

_RESET_MODULES = (
    "main",
    "db",
    "models",
    "seed",
    "booking_service",
    "booking_migrate",
    "enquiry_migrate",
    "enquiry_service",
    "home_collection_service",
    "home_collection_migrate",
    "i18n_content",
    "phonetic_match",
    "patient_context",
    "patient_seed",
)


def test_fresh_database_boots_with_every_home_collection_table_and_seeds_operational_data(tmp_path, monkeypatch):
    path = tmp_path / "fresh.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.syspath_prepend(CLINIC_API)
    for mod in _RESET_MODULES:
        sys.modules.pop(mod, None)

    from fastapi.testclient import TestClient

    clinic_main = importlib.import_module("main")
    with TestClient(clinic_main.app) as c:
        health = c.get("/api/health").json()
        assert health["status"] == "ok"

        import db as db_mod
        import models as m

        db = db_mod.SessionLocal()
        try:
            # Every new table exists and is queryable (create_all ran, nothing raised).
            for table_name, table in m.Base.metadata.tables.items():
                if table_name in NEW_TABLES:
                    assert db.execute(table.select().limit(1)) is not None
            # home_collection_migrate seeded a real payment policy row -- never left for the agent
            # to fabricate one (section 18's own rule).
            assert db.query(m.HomeCollectionPaymentPolicy).count() >= 1
            # And at least one slot for whatever postal codes seed.py's own catalogue marks
            # serviceable (seed.py seeds HomeCollectionCoverage rows as part of the base catalogue).
            serviceable = db.query(m.HomeCollectionCoverage).filter_by(serviceable=True).count()
            if serviceable:
                assert db.query(m.HomeCollectionSlot).count() > 0
        finally:
            db.close()
    sys.modules.pop("main", None)


@pytest.fixture()
def pre_kcd387_db(tmp_path, monkeypatch):
    """A database from BEFORE this story: every table home_collection_eligibility() itself already
    depended on (departments/doctors/lab_tests/home_collection_coverage/patients), real rows in it,
    but none of the 6 tables this story adds."""
    path = tmp_path / "pre_kcd387.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.syspath_prepend(CLINIC_API)
    for mod in _RESET_MODULES:
        sys.modules.pop(mod, None)

    import db as db_mod
    import models as m

    old_tables = [t for name, t in m.Base.metadata.tables.items() if name not in NEW_TABLES]
    m.Base.metadata.create_all(db_mod.engine, tables=old_tables)

    db = db_mod.SessionLocal()
    try:
        dept = m.Department(name="Pathology")
        db.add(dept)
        db.flush()
        db.add(
            m.Doctor(
                name="Dr. Pre Existing",
                qualifications="MBBS",
                department_id=dept.id,
                consultation_fee_inr=500,
            )
        )
        test = m.LabTest(
            name="Uric Acid",
            rate_inr=250,
            sample_type="Blood",
            report_time_hours=24,
            home_collection_eligible=True,
        )
        db.add(test)
        db.add(m.HomeCollectionCoverage(postal_code="700091", serviceable=True, charge_inr=100))
        db.commit()
        pre_existing_test_id = test.id
    finally:
        db.close()

    yield pre_existing_test_id


def test_an_existing_pre_kcd387_database_upgrades_without_losing_any_existing_row(pre_kcd387_db, monkeypatch):
    pre_existing_test_id = pre_kcd387_db
    for mod in ("main",):
        sys.modules.pop(mod, None)

    from fastapi.testclient import TestClient

    clinic_main = importlib.import_module("main")
    with TestClient(clinic_main.app) as c:
        health = c.get("/api/health").json()
        assert health["status"] == "ok"

        import db as db_mod
        import models as m

        db = db_mod.SessionLocal()
        try:
            # The pre-existing row is untouched -- same id, same facts, not duplicated or altered.
            still_there = db.get(m.LabTest, pre_existing_test_id)
            assert still_there is not None
            assert still_there.name == "Uric Acid" and still_there.rate_inr == 250
            assert db.query(m.LabTest).filter_by(name="Uric Acid").count() == 1  # never duplicated
            assert db.query(m.Doctor).filter_by(name="Dr. Pre Existing").count() == 1

            # The new tables now exist and were populated FROM the pre-existing coverage row --
            # never a second, hard-coded postal-code list.
            assert db.query(m.HomeCollectionSlot).filter_by(postal_code="700091").count() > 0
            assert db.query(m.HomeCollectionPaymentPolicy).count() >= 1
        finally:
            db.close()

        # And the new endpoints actually work against the upgraded database, end to end.
        resp = c.get(
            "/api/v1/home-collection/eligibility-multi",
            params={"test_names": ["Uric Acid"], "postal_code": "700091"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["found"] is True
        assert body["results"][0]["eligible"] is True
    sys.modules.pop("main", None)
