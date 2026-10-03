"""HTTP-level integration tests for Epic E27's endpoints in
clinic-api/main.py -- proves the routes are wired to enquiry_service.py
correctly, on top of test_enquiry_service.py's direct unit tests.

    python -m pytest tests/test_enquiry_endpoints.py -v
"""

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
        "enquiry_service",
        "enquiry_migrate",
        "i18n_content",
        "phonetic_match",
    ):
        sys.modules.pop(mod, None)

    from fastapi.testclient import TestClient

    import main as clinic_main

    with TestClient(clinic_main.app) as c:
        yield c
    try:
        os.remove(db_path)
    except OSError:
        pass


def test_doctor_leave_endpoint(clinic_client):
    cat = clinic_client.get("/api/v1/catalogue").json()
    doctor_name = cat["doctors"][0]["surname"]
    result = clinic_client.get(f"/api/v1/doctors/{doctor_name}/leave", params={"date": "2020-01-01"}).json()
    assert result["found"] and result["on_leave"] is False  # far in the past, before the seeded leave window


def test_prep_merge_endpoint(clinic_client):
    result = clinic_client.post(
        "/api/v1/tests/prep/merge", json={"test_names": ["Blood Sugar Fasting", "Lipid Profile"]}
    ).json()
    assert result["found"] and result["fasting_hours"] == 11
    assert result["not_found"] == []


def test_prep_merge_endpoint_surfaces_the_escalation_signal(clinic_client):
    """KCD-382 AC3, at the HTTP boundary: enquiry_service.merge_prep_instructions()'s `escalate_to_human` /
    `incomplete_data` keys (test_enquiry_service.py proves the logic directly) must reach the voice layer
    through this endpoint unchanged -- the route forwards the service dict verbatim, so nothing here should
    strip or recompute them."""
    result = clinic_client.post(
        "/api/v1/tests/prep/merge", json={"test_names": ["USG Whole Abdomen"]}
    ).json()
    assert result["found"] is True
    assert result["escalate_to_human"] is True
    assert result["incomplete_data"] == ["USG Whole Abdomen"]
    assert result["not_found"] == []


def test_prep_merge_endpoint_does_not_guess_an_unresolved_test_name(clinic_client):
    """An unrecognised test name is reported, never silently dropped or substituted."""
    result = clinic_client.post(
        "/api/v1/tests/prep/merge", json={"test_names": ["Blood Sugar Fasting", "Not A Real Test"]}
    ).json()
    assert result["not_found"] == ["Not A Real Test"]
    assert result["found"] is True  # the one resolvable test is still merged...
    assert result["fasting_hours"] == 8  # ...but the caller-facing reply template treats non-empty
    # not_found as "ask again" regardless (see agent/reply_templates.py::merged_test_prep_reply), so the
    # voice layer never speaks a partial merge as if it were complete.


def test_prep_merge_endpoint_when_nothing_resolves(clinic_client):
    result = clinic_client.post("/api/v1/tests/prep/merge", json={"test_names": ["Not A Real Test"]}).json()
    assert result["found"] is False
    assert result["not_found"] == ["Not A Real Test"]


def test_package_endpoint(clinic_client):
    result = clinic_client.get("/api/v1/packages/Full Body Checkup Basic").json()
    assert result["found"] and result["savings_inr"] > 0


def test_walk_in_endpoint(clinic_client):
    result = clinic_client.get("/api/v1/walk-in", params={"department": "General Medicine"}).json()
    assert result["found"] and result["allowed"] is True


def test_home_collection_endpoint(clinic_client):
    eligible = clinic_client.get(
        "/api/v1/home-collection/eligibility", params={"test_name": "Uric Acid", "postal_code": "700091"}
    ).json()
    assert eligible["found"] and eligible["eligible"] is True

    not_eligible = clinic_client.get(
        "/api/v1/home-collection/eligibility", params={"test_name": "ECG", "postal_code": "700091"}
    ).json()
    assert not_eligible["eligible"] is False and not_eligible["reason"] == "sample_type"


def test_insurance_coverage_endpoint(clinic_client):
    result = clinic_client.get("/api/v1/insurance/coverage", params={"policy_number": "DEMO-POLICY-001"}).json()
    assert result["found"] and result["covered"]


def test_prescription_requirement_endpoint(clinic_client):
    required = clinic_client.get(
        "/api/v1/tests/HIV Test (ELISA)/prescription-requirement", params={"lang": "en"}
    ).json()
    assert required["found"] and required["required"] is True

    not_required = clinic_client.get("/api/v1/tests/Complete Blood Count (CBC)/prescription-requirement").json()
    assert not_required["required"] is False


def test_out_of_scope_and_callback_endpoints(clinic_client):
    oos = clinic_client.post(
        "/api/v1/calls/out-of-scope", json={"call_id": "c1", "caller_question": "Do you sell medicine?"}
    ).json()
    assert oos["recorded"]

    cb = clinic_client.post(
        "/api/v1/callbacks", json={"phone": "9111111111", "call_id": "c1", "requested_window": "tomorrow morning"}
    ).json()
    assert cb["success"]


def test_report_status_otp_and_delivery_endpoints(clinic_client):
    # No report exists yet for this confirmation_id.
    missing = clinic_client.get("/api/v1/reports/status", params={"confirmation_id": "KCD-NOPE"}).json()
    assert missing["found"] is False


def test_department_hours_endpoint(clinic_client):
    result = clinic_client.get("/api/v1/departments/Cardiology/hours", params={"lang": "en"}).json()
    assert result["found"] and "10am" in result["hours"]

    no_override = clinic_client.get("/api/v1/departments/Dermatology/hours").json()
    assert no_override["found"] is False


def test_the_callback_endpoint_accepts_and_stores_a_call_summary(clinic_client):
    cb = clinic_client.post(
        "/api/v1/callbacks",
        json={
            "phone": "9222222222",
            "call_id": "c9",
            "requested_window": "asap",
            "reason": "lay_term_abdomen",
            "call_summary": 'reason: lay_term_abdomen | language: bn | caller said: "পেটের ছবি"',
        },
    ).json()
    assert cb["success"] is True


def test_the_payment_link_endpoint_refuses_a_confirmation_id_that_is_not_a_confirmed_appointment(clinic_client):
    result = clinic_client.post("/api/v1/payments/send-link", json={"confirmation_id": "KCD-NOPE"}).json()
    assert result == {"success": False, "reason": "not_found"}
