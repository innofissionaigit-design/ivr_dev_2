"""Integration check for the test_prep and clinic_faq endpoints added
alongside agent/fast_path.py's test_prep/clinic_faq intents.

Runs clinic-api directly against a throwaway SQLite file via FastAPI's
TestClient -- no live pod needed, unlike tests/test_smoke.py. This is
exactly the kind of check CLAUDE.md asks for: verified against a live
system (even if a local, temporary one), not asserted from reading the
code.

    python -m pytest tests/test_clinic_api_new_endpoints.py -v
"""

import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def clinic_client():
    """A fresh clinic-api app + freshly seeded SQLite file per test."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)

    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)

    # Fresh import each time so module-level `engine`/`SessionLocal` in
    # db.py pick up THIS test's CLINIC_DB_PATH rather than a previous
    # test's cached module.
    for mod in ("main", "db", "models", "seed", "booking_service", "booking_migrate", "enquiry_migrate"):
        sys.modules.pop(mod, None)

    from fastapi.testclient import TestClient

    import main as clinic_main  # noqa: PLC0415

    with TestClient(clinic_main.app) as client:
        yield client

    try:
        os.remove(db_path)
    except OSError:
        pass


def test_catalogue_includes_faq_topics(clinic_client):
    resp = clinic_client.get("/api/v1/catalogue")
    assert resp.status_code == 200
    payload = resp.json()
    assert "faq_topics" in payload
    topics = {row["topic"] for row in payload["faq_topics"]}
    assert "hours" in topics
    assert "location" in topics
    hours_row = next(r for r in payload["faq_topics"] if r["topic"] == "hours")
    assert any("সময়" in k for k in hours_row["keywords_bn"])


def test_faq_answer_found(clinic_client):
    resp = clinic_client.get("/api/v1/faq", params={"topic": "hours"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["topic"] == "hours"
    assert "খোলা" in body["answer"] or "সময়" in body["answer"]


def test_faq_answer_not_found(clinic_client):
    resp = clinic_client.get("/api/v1/faq", params={"topic": "does_not_exist"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False


def test_prep_found_for_fasting_test(clinic_client):
    resp = clinic_client.get("/api/v1/tests/prep", params={"name": "লিপিড প্রোফাইল"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["fasting_required"] is True
    assert "উপবাস" in body["prep_instructions"]


def test_prep_found_for_non_fasting_test(clinic_client):
    resp = clinic_client.get("/api/v1/tests/prep", params={"name": "সিবিসি"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["fasting_required"] is False
    assert body["prep_instructions"]  # the generic default, not empty


def test_prep_not_found_returns_suggestions_shape(clinic_client):
    resp = clinic_client.get("/api/v1/tests/prep", params={"name": "নাথিং টেস্ট"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert "did_you_mean" in body


def test_existing_test_search_endpoint_still_works(clinic_client):
    """Regression guard: the _find_test refactor in clinic-api/main.py must
    not change search_test's existing, already-relied-upon behaviour."""
    resp = clinic_client.get("/api/v1/tests/search", params={"name": "সিবিসি"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["rate_inr"] == 400


# ============================================================== KCD-446


def test_search_offers_near_matches_for_a_genuinely_ambiguous_name(clinic_client):
    # "সুগার" (sugar) alone matches BOTH "Blood Sugar Fasting" (alias
    # "সুগার ফাস্টিং") and "Blood Sugar PP" (alias "পিপি সুগার") equally
    # well -- silently picking one would risk quoting the wrong test's
    # price for something that genuinely exists under two names.
    resp = clinic_client.get("/api/v1/tests/search", params={"name": "সুগার"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["ambiguous"] is True
    assert "Blood Sugar Fasting" in body["did_you_mean"]
    assert "Blood Sugar PP" in body["did_you_mean"]
    assert len(body["did_you_mean"]) <= 3


def test_prep_also_offers_near_matches_for_an_ambiguous_name(clinic_client):
    resp = clinic_client.get("/api/v1/tests/prep", params={"name": "সুগার"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["ambiguous"] is True
    assert len(body["did_you_mean"]) >= 2


def test_a_genuinely_unambiguous_name_is_not_flagged_ambiguous(clinic_client):
    resp = clinic_client.get("/api/v1/tests/search", params={"name": "সিবিসি"})
    body = resp.json()
    assert body["found"] is True
    assert "ambiguous" not in body


# ======================================================= doctor-side KCD-446
# "AI can ask correct questions back" -- doctor lookup had the same silent-
# pick-one gap test lookup was fixed for: _find_doctor happily returns
# whichever seeded doctor scores a hair higher when two surnames both clear
# the fuzzy floor, which is exactly the "Doctor Nobody" class of bug
# CLAUDE.md warns about, just with a REAL doctor's schedule read out
# instead of a fabricated one. _find_doctor_candidates + the ambiguous/
# did_you_mean branch below close that gap the same way search_test's did.


def test_doctor_availability_offers_a_choice_for_a_genuinely_ambiguous_surname(clinic_client):
    # "Dr. N. Roy" and "Dr. P. Ray" are both seeded (Cardiology). A short,
    # garbled fragment like "ry" scores identically (0.8) against both
    # surnames -- silently picking one would risk reading out a different
    # cardiologist's real chamber hours.
    resp = clinic_client.get("/api/v1/doctors/availability", params={"name": "ry"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["ambiguous"] is True
    # DELIBERATE SPEC CHANGE (2026-09-25, duplicate-doctor handling): the choice is named by WHOLE name, because two
    # doctors can share a short name ("Dr. A. Sen") and a "which one?" that repeats it cannot be answered.
    assert "Dr. Nirmal Roy" in body["did_you_mean"]
    assert "Dr. Partha Ray" in body["did_you_mean"]
    assert len(body["did_you_mean"]) <= 3


def test_doctor_availability_still_resolves_an_unambiguous_fuzzy_name(clinic_client):
    # A clear favourite (exact surname) must not be caught by the new
    # ambiguity check -- regression guard for the existing single-match path.
    resp = clinic_client.get("/api/v1/doctors/availability", params={"name": "Sen"})
    body = resp.json()
    assert body["found"] is True
    assert "ambiguous" not in body


# ============================================================== KCD-449


def test_repeating_the_same_question_three_times_gives_the_identical_answer(clinic_client):
    """ "The same question gets the same answer within one call" -- three
    repeats with the backend unchanged (KCD-444, Done, already keeps
    every hit live -- this is the consistency PROOF that live-fetching
    actually implies, not a new fetch behaviour)."""
    answers = [clinic_client.get("/api/v1/tests/search", params={"name": "সিবিসি"}).json() for _ in range(3)]
    assert answers[0] == answers[1] == answers[2]


def test_a_real_data_change_is_reflected_on_the_very_next_call(clinic_client):
    """The other half of KCD-449: consistency holds ONLY while the data is
    unchanged -- a genuine change must show up immediately, not be
    masked by whatever made the repeats above consistent."""
    before = clinic_client.get("/api/v1/tests/search", params={"name": "সিবিসি"}).json()

    import db as db_mod  # noqa: PLC0415 - clinic-api module, imported by the fixture's sys.path setup
    import models as m  # noqa: PLC0415

    db = db_mod.SessionLocal()
    try:
        test_row = db.query(m.LabTest).filter_by(name="Complete Blood Count (CBC)").one()
        test_row.rate_inr = before["rate_inr"] + 50
        db.commit()
    finally:
        db.close()

    after = clinic_client.get("/api/v1/tests/search", params={"name": "সিবিসি"}).json()
    assert after["rate_inr"] == before["rate_inr"] + 50
