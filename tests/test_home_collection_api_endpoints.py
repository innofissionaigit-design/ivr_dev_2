"""Integration check for every new /api/v1/home-collection/* endpoint (KCD-387 full flow) --
eligibility-multi, slots, hold, release-hold, quote, payment-policy, book, booking status, cancel.

Runs clinic-api directly against a throwaway SQLite file via FastAPI's TestClient -- no live pod
needed, the same discipline tests/test_clinic_api_new_endpoints.py already uses.

    python -m pytest tests/test_home_collection_api_endpoints.py -v
"""

import datetime
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def clinic_client():
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
        "enquiry_service",
        "home_collection_service",
        "home_collection_migrate",
    ):
        sys.modules.pop(mod, None)

    from fastapi.testclient import TestClient

    import main as clinic_main  # noqa: PLC0415

    with TestClient(clinic_main.app) as client:
        yield client

    try:
        os.remove(db_path)
    except OSError:
        pass


def _serviceable_postal_code(client) -> str:
    """seed.py's own catalogue decides which postal codes are serviceable; this reads whichever one
    it actually seeded rather than hard-coding a pincode this test might silently stop covering."""
    import db as db_mod
    import models as m

    db = db_mod.SessionLocal()
    try:
        row = db.query(m.HomeCollectionCoverage).filter_by(serviceable=True).first()
        assert row is not None, "fixture assumption: seed.py marks at least one postal code serviceable"
        return row.postal_code
    finally:
        db.close()


def _eligible_test_name(client, postal_code: str) -> str:
    import db as db_mod
    import models as m

    db = db_mod.SessionLocal()
    try:
        for t in db.query(m.LabTest).filter_by(home_collection_eligible=True).all():
            r = client.get(
                "/api/v1/home-collection/eligibility", params={"test_name": t.name, "postal_code": postal_code}
            ).json()
            if r.get("eligible"):
                return t.name
        pytest.skip("fixture assumption: no eligible+serviceable test found in this seed")
    finally:
        db.close()


def test_eligibility_multi_reports_every_test_independently(clinic_client):
    pc = _serviceable_postal_code(clinic_client)
    eligible_name = _eligible_test_name(clinic_client, pc)
    r = clinic_client.get(
        "/api/v1/home-collection/eligibility-multi",
        params={"test_names": [eligible_name, "Not A Real Test Xyz"], "postal_code": pc},
    ).json()
    assert r["found"] is True
    assert len(r["results"]) == 1 and r["results"][0]["eligible"] is True
    assert r["not_found"] == ["Not A Real Test Xyz"]


def test_slots_only_lists_windows_with_real_remaining_capacity(clinic_client):
    pc = _serviceable_postal_code(clinic_client)
    date = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
    r = clinic_client.get("/api/v1/home-collection/slots", params={"postal_code": pc, "date": date}).json()
    assert r["found"] is True
    for slot in r["slots"]:
        assert slot["remaining_capacity"] > 0


def test_hold_then_release_frees_the_same_capacity_back(clinic_client):
    pc = _serviceable_postal_code(clinic_client)
    date = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
    slots = clinic_client.get("/api/v1/home-collection/slots", params={"postal_code": pc, "date": date}).json()["slots"]
    assert slots, "fixture assumption: at least one slot is seeded for this postal code"
    slot_id = slots[0]["slot_id"]
    before = slots[0]["remaining_capacity"]

    hold = clinic_client.post("/api/v1/home-collection/hold", json={"slot_id": slot_id}).json()
    assert hold["success"] is True and hold["hold_token"]

    after_hold = clinic_client.get(
        "/api/v1/home-collection/slots", params={"postal_code": pc, "date": date}
    ).json()["slots"]
    assert next(s for s in after_hold if s["slot_id"] == slot_id)["remaining_capacity"] == before - 1

    released = clinic_client.post("/api/v1/home-collection/release-hold", json={"hold_token": hold["hold_token"]}).json()
    assert released["success"] is True

    after_release = clinic_client.get(
        "/api/v1/home-collection/slots", params={"postal_code": pc, "date": date}
    ).json()["slots"]
    assert next(s for s in after_release if s["slot_id"] == slot_id)["remaining_capacity"] == before


def test_quote_itemizes_each_eligible_test_and_sums_a_real_total(clinic_client):
    pc = _serviceable_postal_code(clinic_client)
    eligible_name = _eligible_test_name(clinic_client, pc)
    rate = clinic_client.get("/api/v1/tests/search", params={"name": eligible_name}).json()["rate_inr"]
    quote = clinic_client.get(
        "/api/v1/home-collection/quote", params={"test_names": [eligible_name], "postal_code": pc}
    ).json()
    assert quote["quote_available"] is True
    assert quote["test_charges_inr"] == rate
    assert quote["total_inr"] == quote["test_charges_inr"] + quote["home_collection_charge_inr"]
    row = next(r for r in quote["per_test"] if r["test_name"] == eligible_name)
    assert row["rate_inr"] == rate  # itemized, never only the summed total


def test_quote_with_no_eligible_test_is_honest_about_it_never_a_fake_number(clinic_client):
    quote = clinic_client.get(
        "/api/v1/home-collection/quote", params={"test_names": ["Not A Real Test Xyz"], "postal_code": "000000"}
    ).json()
    assert quote["quote_available"] is False
    assert quote.get("total_inr") is None


def test_payment_policy_is_a_real_versioned_db_row(clinic_client):
    r = clinic_client.get("/api/v1/home-collection/payment-policy", params={"lang": "en"}).json()
    assert r["found"] is True
    assert r["policy"] and r["description"]


def test_booking_a_hold_persists_a_real_booking_and_consumes_the_hold(clinic_client):
    pc = _serviceable_postal_code(clinic_client)
    eligible_name = _eligible_test_name(clinic_client, pc)
    date = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
    slot_id = clinic_client.get(
        "/api/v1/home-collection/slots", params={"postal_code": pc, "date": date}
    ).json()["slots"][0]["slot_id"]
    hold = clinic_client.post("/api/v1/home-collection/hold", json={"slot_id": slot_id}).json()

    booked = clinic_client.post(
        "/api/v1/home-collection/book",
        json={
            "hold_token": hold["hold_token"],
            "test_names": [eligible_name],
            "postal_code": pc,
            "patient_name": "Test Patient",
            "phone": "9876543210",
            "address_line": "12 Test Road",
        },
    ).json()
    assert booked["success"] is True
    assert booked["booking_reference"].startswith("HC-")
    assert booked["total_inr"] == booked["test_charges_inr"] + booked["home_collection_charge_inr"]

    status = clinic_client.get(f"/api/v1/home-collection/booking/{booked['booking_reference']}").json()
    assert status["found"] is True and status["status"] == "confirmed"

    # The hold is consumed: trying to book again with the same (now-confirmed) token fails honestly.
    again = clinic_client.post(
        "/api/v1/home-collection/book",
        json={
            "hold_token": hold["hold_token"],
            "test_names": [eligible_name],
            "postal_code": pc,
            "patient_name": "Test Patient",
            "phone": "9876543210",
            "address_line": "12 Test Road",
        },
        headers={"Idempotency-Key": "a-different-key"},
    ).json()
    assert again["success"] is False and again["reason"] == "hold_expired"


def test_booking_with_an_already_expired_hold_token_is_refused_not_faked(clinic_client):
    # A real, eligible test name is used here so this isolates the hold-token check itself --
    # the endpoint resolves test_names to real lab_test_ids BEFORE touching the hold (see
    # main.py's home_collection_book_endpoint), so an unresolvable test name would legitimately
    # short-circuit with "test_not_found" first; that is a separate, correctly-ordered check
    # (test_booking_with_an_unresolvable_test_name_is_refused_as_test_not_found below), not this one.
    pc = _serviceable_postal_code(clinic_client)
    eligible_name = _eligible_test_name(clinic_client, pc)
    booked = clinic_client.post(
        "/api/v1/home-collection/book",
        json={
            "hold_token": "never-issued-token",
            "test_names": [eligible_name],
            "postal_code": pc,
            "patient_name": "Test Patient",
            "phone": "9876543210",
            "address_line": "12 Test Road",
        },
    ).json()
    assert booked["success"] is False and booked["reason"] == "hold_expired"


def test_booking_with_an_unresolvable_test_name_is_refused_as_test_not_found(clinic_client):
    booked = clinic_client.post(
        "/api/v1/home-collection/book",
        json={
            "hold_token": "never-issued-token",
            "test_names": ["Not A Real Test Xyz"],
            "postal_code": "000000",
            "patient_name": "Test Patient",
            "phone": "9876543210",
            "address_line": "12 Test Road",
        },
    ).json()
    assert booked["success"] is False and booked["reason"] == "test_not_found"
    assert booked["not_found"] == ["Not A Real Test Xyz"]


def test_cancel_releases_the_confirmed_capacity_back_to_available(clinic_client):
    pc = _serviceable_postal_code(clinic_client)
    eligible_name = _eligible_test_name(clinic_client, pc)
    date = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
    slots = clinic_client.get("/api/v1/home-collection/slots", params={"postal_code": pc, "date": date}).json()["slots"]
    slot_id, before = slots[0]["slot_id"], slots[0]["remaining_capacity"]
    hold = clinic_client.post("/api/v1/home-collection/hold", json={"slot_id": slot_id}).json()
    booked = clinic_client.post(
        "/api/v1/home-collection/book",
        json={
            "hold_token": hold["hold_token"],
            "test_names": [eligible_name],
            "postal_code": pc,
            "patient_name": "Test Patient",
            "phone": "9876543210",
            "address_line": "12 Test Road",
        },
    ).json()

    after_book = clinic_client.get(
        "/api/v1/home-collection/slots", params={"postal_code": pc, "date": date}
    ).json()["slots"]
    assert next(s for s in after_book if s["slot_id"] == slot_id)["remaining_capacity"] == before - 1

    cancelled = clinic_client.post(
        "/api/v1/home-collection/cancel", json={"booking_reference": booked["booking_reference"]}
    ).json()
    assert cancelled["success"] is True

    after_cancel = clinic_client.get(
        "/api/v1/home-collection/slots", params={"postal_code": pc, "date": date}
    ).json()["slots"]
    assert next(s for s in after_cancel if s["slot_id"] == slot_id)["remaining_capacity"] == before


# ================================================================================================
# ADDED BY SOURAV: KCD-387 bugfix regression -- a caller asking for the exact test "TSH" was being
# silently resolved by _resolve_test_ids's old _find_test() call to "Thyroid Profile (T3 T4 TSH)"
# instead, because "tsh" is a substring of that longer name too. These pin the fixed behavior down
# permanently so a future change cannot quietly reintroduce the substring-wins-over-exact bug.
# ================================================================================================


def _test_row_by_exact_name(name: str):
    import db as db_mod
    import models as m

    db = db_mod.SessionLocal()
    try:
        return db.query(m.LabTest).filter(m.LabTest.name == name).first()
    finally:
        db.close()


def test_exact_tsh_resolves_to_the_tsh_test_not_the_thyroid_profile(clinic_client):
    # ADDED BY SOURAV: KCD-387 bugfix regression -- the exact reproduction from the bug report.
    tsh = _test_row_by_exact_name("TSH")
    thyroid = _test_row_by_exact_name("Thyroid Profile (T3 T4 TSH)")
    if tsh is None or thyroid is None:
        pytest.skip("fixture assumption: seed.py no longer seeds both 'TSH' and 'Thyroid Profile (T3 T4 TSH)'")
    pc = _serviceable_postal_code(clinic_client)
    r = clinic_client.get(
        "/api/v1/home-collection/eligibility-multi", params={"test_names": ["TSH"], "postal_code": pc}
    ).json()
    assert r["not_found"] == []
    assert len(r["results"]) == 1
    assert r["results"][0]["test_name"] == "TSH"
    assert r["results"][0]["lab_test_id"] == tsh.id
    assert r["results"][0]["lab_test_id"] != thyroid.id


def test_exact_thyroid_profile_still_resolves_correctly(clinic_client):
    # ADDED BY SOURAV: KCD-387 bugfix regression -- the fix must not break the longer test's own
    # exact resolution.
    thyroid = _test_row_by_exact_name("Thyroid Profile (T3 T4 TSH)")
    if thyroid is None:
        pytest.skip("fixture assumption: seed.py no longer seeds 'Thyroid Profile (T3 T4 TSH)'")
    pc = _serviceable_postal_code(clinic_client)
    r = clinic_client.get(
        "/api/v1/home-collection/eligibility-multi",
        params={"test_names": ["Thyroid Profile (T3 T4 TSH)"], "postal_code": pc},
    ).json()
    assert r["not_found"] == []
    assert r["results"][0]["lab_test_id"] == thyroid.id


def test_an_ordinary_unambiguous_substring_name_still_resolves_normally(clinic_client):
    # ADDED BY SOURAV: KCD-387 bugfix regression -- the fix must not turn every substring match
    # into "not found"; a name that unambiguously matches exactly one test (the ordinary case
    # _resolve_test_ids has always handled) must still resolve exactly as before.
    cbc = _test_row_by_exact_name("Complete Blood Count (CBC)")
    if cbc is None:
        pytest.skip("fixture assumption: seed.py no longer seeds 'Complete Blood Count (CBC)'")
    pc = _serviceable_postal_code(clinic_client)
    r = clinic_client.get(
        "/api/v1/home-collection/eligibility-multi", params={"test_names": ["CBC"], "postal_code": pc}
    ).json()
    assert r["not_found"] == []
    assert r["results"][0]["lab_test_id"] == cbc.id


def test_an_unknown_test_name_still_resolves_to_not_found(clinic_client):
    # ADDED BY SOURAV: KCD-387 bugfix regression -- an unresolvable name must still be reported as
    # not_found, never fabricated, after the exact-match-first change.
    pc = _serviceable_postal_code(clinic_client)
    r = clinic_client.get(
        "/api/v1/home-collection/eligibility-multi",
        params={"test_names": ["Totally Fake Test Name"], "postal_code": pc},
    ).json()
    assert r["not_found"] == ["Totally Fake Test Name"]
    assert r["results"] == []


def test_tests_search_endpoint_behavior_is_unchanged_by_the_home_collection_fix(clinic_client):
    # ADDED BY SOURAV: KCD-387 bugfix regression -- /api/v1/tests/search (and _find_test_candidates/
    # _find_test themselves) are untouched by this fix; a bare "TSH" search still correctly reports
    # the pre-existing ambiguity (KCD-446) rather than silently resolving either way, and a search
    # for the longer name still finds it.
    tsh = _test_row_by_exact_name("TSH")
    thyroid = _test_row_by_exact_name("Thyroid Profile (T3 T4 TSH)")
    if tsh is None or thyroid is None:
        pytest.skip("fixture assumption: seed.py no longer seeds both 'TSH' and 'Thyroid Profile (T3 T4 TSH)'")
    r = clinic_client.get("/api/v1/tests/search", params={"name": "TSH"}).json()
    assert r.get("ambiguous") is True
    assert set(r.get("did_you_mean") or []) == {"TSH", "Thyroid Profile (T3 T4 TSH)"}
    r2 = clinic_client.get("/api/v1/tests/search", params={"name": "Thyroid Profile"}).json()
    assert r2.get("found") is True
    assert r2["test_name"] == "Thyroid Profile (T3 T4 TSH)"
