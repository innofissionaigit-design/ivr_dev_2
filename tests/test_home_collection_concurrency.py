"""KCD-387 full flow, section 38 (mandatory): "at least one REAL regression test proving confirmed
home-collection capacity <= configured capacity under concurrent booking attempts... must not use
an in-memory fake lock as the actual production mechanism."

Exercises home_collection_service.hold_home_collection_slot() directly, under REAL thread
concurrency, against a real (throwaway) SQLite file -- the exact same approach
tests/test_booking_concurrency.py already uses for booking_service.hold_slot()/SlotLock, applied to
HomeCollectionSlotHold's own composite primary key (slot_id, unit_index) instead. Each thread opens
its OWN database session, the same way two simultaneous requests in production each get their own
session from the connection pool, so this proves the actual DB-enforced guarantee: of any number of
threads racing to insert the SAME (slot_id, unit_index) row, SQLite's serialized writers plus the
composite primary key mean exactly one INSERT for that pair can ever succeed -- generalised here to
a slot of capacity N by giving each unit of capacity its own composite-key row.

    python -m pytest tests/test_home_collection_concurrency.py -v
"""

import concurrent.futures
import datetime
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")

CAPACITY = 3
N_CONCURRENT_CALLERS = 20


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
        "enquiry_service",
        "home_collection_service",
        "home_collection_migrate",
        "booking_migrate",
        "enquiry_migrate",
        "i18n_content",
    ):
        sys.modules.pop(mod, None)

    import seed as seed_mod

    seed_mod.seed()
    import db as db_mod
    import home_collection_service as hcs
    import models as m

    # A slot with real capacity > 1, independent of whatever home_collection_migrate seeds, so this
    # test's capacity figure is explicit and never silently changed by that seeding's own defaults.
    db = db_mod.SessionLocal()
    try:
        slot = m.HomeCollectionSlot(
            postal_code="700091",
            date=(datetime.date.today() + datetime.timedelta(days=3)).isoformat(),
            start_time="08:00",
            end_time="10:00",
            capacity=CAPACITY,
        )
        db.add(slot)
        db.commit()
        slot_id = slot.id
    finally:
        db.close()

    yield hcs, db_mod, m, slot_id

    try:
        os.remove(db_path)
    except OSError:
        pass


def test_concurrent_holds_never_exceed_the_configured_capacity(clinic_modules):
    hcs, db_mod, m, slot_id = clinic_modules

    def attempt(_):
        db = db_mod.SessionLocal()
        try:
            return hcs.hold_home_collection_slot(db, slot_id)
        finally:
            db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CONCURRENT_CALLERS) as pool:
        results = list(pool.map(attempt, range(N_CONCURRENT_CALLERS)))

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    assert len(successes) == CAPACITY, f"expected exactly {CAPACITY} successes, got {len(successes)}: {results}"
    assert len(failures) == N_CONCURRENT_CALLERS - CAPACITY
    assert all(f["reason"] == "slot_full" for f in failures)

    # The real, durable proof: never more live (held/confirmed) rows for this slot than its
    # configured capacity -- not merely that the function RETURNED the right count, in case a bug
    # let an extra row slip in without being reflected in the return value.
    with db_mod.SessionLocal() as probe:
        rows = (
            probe.query(m.HomeCollectionSlotHold)
            .filter_by(slot_id=slot_id, status="held")
            .all()
        )
    assert len(rows) == CAPACITY
    assert len({r.unit_index for r in rows}) == CAPACITY, "no two holds were ever allowed to share a unit_index"


def test_concurrent_holds_plus_confirmed_bookings_never_exceed_capacity(clinic_modules):
    # The same guarantee must hold across a MIX of already-confirmed bookings and newly-contested
    # holds -- available capacity is capacity-minus-(held-or-confirmed), never just capacity-minus-held.
    hcs, db_mod, m, slot_id = clinic_modules
    db = db_mod.SessionLocal()
    try:
        # Pre-occupy one unit as a CONFIRMED booking (not merely a pending hold), the state a real
        # booking leaves behind once create_home_collection_booking() has run.
        db.add(
            m.HomeCollectionSlotHold(
                slot_id=slot_id,
                unit_index=0,
                status="confirmed",
                hold_token="pre-confirmed",
                hold_expires_at=None,
                created_at=datetime.datetime.now(),
            )
        )
        db.commit()
    finally:
        db.close()

    def attempt(_):
        db = db_mod.SessionLocal()
        try:
            return hcs.hold_home_collection_slot(db, slot_id)
        finally:
            db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CONCURRENT_CALLERS) as pool:
        results = list(pool.map(attempt, range(N_CONCURRENT_CALLERS)))

    successes = [r for r in results if r["success"]]
    assert len(successes) == CAPACITY - 1, f"one unit was already confirmed; expected {CAPACITY - 1} new holds"

    with db_mod.SessionLocal() as probe:
        live = (
            probe.query(m.HomeCollectionSlotHold)
            .filter(m.HomeCollectionSlotHold.slot_id == slot_id)
            .filter(m.HomeCollectionSlotHold.status.in_(("held", "confirmed")))
            .all()
        )
    assert len(live) == CAPACITY, "held + confirmed rows must never exceed the slot's configured capacity"


def test_a_released_hold_frees_its_unit_for_a_later_caller(clinic_modules):
    hcs, db_mod, m, slot_id = clinic_modules
    db = db_mod.SessionLocal()
    try:
        holds = [hcs.hold_home_collection_slot(db, slot_id) for _ in range(CAPACITY)]
        assert all(h["success"] for h in holds)
        full = hcs.hold_home_collection_slot(db, slot_id)
        assert full == {"success": False, "reason": "slot_full"}

        released = hcs.release_home_collection_hold(db, holds[0]["hold_token"])
        assert released["success"] is True

        after_release = hcs.hold_home_collection_slot(db, slot_id)
        assert after_release["success"] is True, "a released hold's unit must become available again"
    finally:
        db.close()
