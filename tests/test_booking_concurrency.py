"""KCD-376 (P0): two callers want the last slot at the same moment.

Exercises booking_service.hold_slot() directly, under REAL thread
concurrency, against a real (throwaway) SQLite file -- not the FastAPI
TestClient, which does not guarantee genuinely concurrent dispatch. Each
thread opens its OWN database session, the same way two simultaneous
requests in production each get their own session from the connection
pool, so this proves the actual guarantee: SlotLock's composite primary
key plus SQLite's serialized writers mean exactly one INSERT for the same
(doctor_id, date, time_slot) can ever succeed.

    python -m pytest tests/test_booking_concurrency.py -v
"""

import concurrent.futures
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


def _first_doctor_id(db_mod, m) -> int:
    db = db_mod.SessionLocal()
    try:
        return db.query(m.Doctor).first().id
    finally:
        db.close()


def test_thirty_simultaneous_holds_produce_exactly_one_success(clinic_modules):
    bs, db_mod, m = clinic_modules
    doctor_id = _first_doctor_id(db_mod, m)
    # A weekday this doctor actually sits -- seed.py's SHIFT_TEMPLATES only
    # cover Mon-Sat, so pick the next Monday to be schedule-agnostic.
    import datetime

    # CodeRabbit-flagged: start tomorrow, not today -- see
    # test_booking_service.py's _next_weekday for why (same-day chamber
    # hours already past would make available_slots() correctly empty).
    d = datetime.date.today() + datetime.timedelta(days=1)
    while d.weekday() != 0:
        d += datetime.timedelta(days=1)
    date = d.isoformat()

    with db_mod.SessionLocal() as probe:
        free = bs.available_slots(probe, doctor_id, date)
    assert free, "fixture assumption: doctor has at least one free slot that day"
    time_slot = free[0]

    def attempt(_):
        db = db_mod.SessionLocal()
        try:
            return bs.hold_slot(db, doctor_id, date, time_slot)
        finally:
            db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as pool:
        results = list(pool.map(attempt, range(30)))

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    assert len(successes) == 1, f"expected exactly 1 success, got {len(successes)}: {results}"
    assert len(failures) == 29
    assert all(f["reason"] == "slot_taken" for f in failures)

    with db_mod.SessionLocal() as probe:
        rows = probe.query(m.SlotLock).filter_by(doctor_id=doctor_id, date=date, time_slot=time_slot).all()
    assert len(rows) == 1, "no partial writes: exactly one SlotLock row must exist for the slot"


def test_the_two_losers_get_useful_alternatives(clinic_modules):
    bs, db_mod, m = clinic_modules
    doctor_id = _first_doctor_id(db_mod, m)
    import datetime

    # CodeRabbit-flagged: start tomorrow, not today -- see
    # test_booking_service.py's _next_weekday for why (same-day chamber
    # hours already past would make available_slots() correctly empty).
    d = datetime.date.today() + datetime.timedelta(days=1)
    while d.weekday() != 0:
        d += datetime.timedelta(days=1)
    date = d.isoformat()

    db = db_mod.SessionLocal()
    try:
        free = bs.available_slots(db, doctor_id, date)
        taken_slot = free[0]
        assert bs.hold_slot(db, doctor_id, date, taken_slot)["success"]
        alts = bs.nearest_alternatives(db, doctor_id, date, taken_slot)
        assert alts, "a taken slot must produce at least one real alternative"
        for alt in alts:
            still_free = bs.available_slots(db, doctor_id, alt["date"])
            assert alt["time_slot"] in still_free, f"alternative {alt} is not actually free"
    finally:
        db.close()
