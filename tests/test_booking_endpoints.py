"""HTTP-level integration tests for Epic E26's endpoints in
clinic-api/main.py -- proves the routes are actually wired to
booking_service.py correctly (request parsing, doctor/test name
resolution, response shape), on top of test_booking_service.py's direct
unit tests of the underlying logic.

    python -m pytest tests/test_booking_endpoints.py -v
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
        "i18n_content",
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


def _next_weekday_with_schedule(client, doctor_name: str, min_days_ahead: int = 1) -> str:
    # Defaults to tomorrow onward, not today: booking_service.available_slots
    # (and now clinic-api/main.py's _validate_doctor_slot) correctly
    # exclude a same-day slot whose time has already passed, so a test
    # that does not care about same-day/past-time behaviour specifically
    # should not pick a "today" date whose first chamber slot may already
    # be behind the current wall-clock time.
    for i in range(min_days_ahead, min_days_ahead + 14):
        d = datetime.date.today() + datetime.timedelta(days=i)
        r = client.get("/api/v1/doctors/availability", params={"name": doctor_name, "date": d.isoformat()})
        if r.json().get("available"):
            return d.isoformat()
    pytest.fail("no available day found in range")


def test_hold_then_confirm_full_flow(clinic_client):
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    doctor_name = cat["doctors"][0]["name"]
    # >24h out, so cancellation below is unambiguously inside the free window.
    date = _next_weekday_with_schedule(c, doctor_name, min_days_ahead=2)

    hold = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": "__probe__"}
    ).json()
    assert hold["success"] is False and hold["reason"] == "invalid_slot"
    valid_slot = hold["valid_slots"][0]

    hold = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": valid_slot}
    ).json()
    assert hold["success"], hold

    confirm = c.post(
        "/api/v1/bookings/confirm",
        json={
            "hold_token": hold["hold_token"],
            "doctor_id": hold["doctor_id"],
            "date": date,
            "time_slot": valid_slot,
            "patient_name": "Test Patient",
            "phone": "9123456780",
            "caller_phone": "9123456780",
        },
    ).json()
    assert confirm["success"] and confirm["confirmation_id"].startswith("KCD-")

    lookup = c.get("/api/v1/bookings/lookup", params={"phone": "9123456780"}).json()
    assert lookup["found"] and lookup["bookings"][0]["confirmation_id"] == confirm["confirmation_id"]

    cancel = c.post("/api/v1/bookings/cancel", json={"confirmation_id": confirm["confirmation_id"]}).json()
    assert cancel["success"]


def test_hold_rejects_a_past_date(clinic_client):
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    doctor_name = cat["doctors"][0]["name"]
    past = (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
    result = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": past, "time_slot": "10:00"}
    ).json()
    assert result == {"success": False, "reason": "date_in_past"}


def test_multi_test_booking_endpoint(clinic_client):
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    names = [t["name"] for t in cat["tests"][:2]]
    date = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()

    result = c.post(
        "/api/v1/bookings/tests",
        json={
            "test_names": names,
            "date": date,
            "patient_name": "Multi",
            "phone": "9111111111",
            "caller_phone": "9111111111",
        },
    ).json()
    assert result["success"] and len(result["test_names"]) == 2 and result["not_found"] == []

    added = c.post(
        "/api/v1/bookings/add-test",
        json={
            "confirmation_id": result["confirmation_id"],
            "test_name": cat["tests"][2]["name"],
        },
    ).json()
    assert added["success"]


def test_earliest_available_endpoint(clinic_client):
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    doctor_name = cat["doctors"][0]["name"]
    result = c.get("/api/v1/doctors/earliest", params={"name": doctor_name}).json()
    assert result["found"] and result["available"]
    assert result["date"] and result["time_slot"]

    missing = c.get("/api/v1/doctors/earliest", params={"name": "Doctor Nobody"}).json()
    assert missing == {"found": False, "query": "Doctor Nobody"}


def test_department_route_endpoint(clinic_client):
    c = clinic_client
    result = c.get("/api/v1/departments/route", params={"query": "chest pain", "lang": "en"}).json()
    assert result["matched"] and result["department_name"] == "Cardiology"


def test_sms_placeholder_never_claims_sent(clinic_client):
    c = clinic_client
    result = c.post(
        "/api/v1/notifications/sms", json={"to": "9000000000", "message": "test message", "template_key": "generic"}
    ).json()
    assert result["queued"] is True
    assert "sent" not in str(result).lower() or result.get("status") != "sent"


def test_conflict_endpoint_detects_double_booking(clinic_client):
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    doctor_name = cat["doctors"][0]["name"]
    date = _next_weekday_with_schedule(c, doctor_name)
    hold = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": "__probe__"}
    ).json()
    slot = hold["valid_slots"][0]
    hold = c.post("/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": slot}).json()
    c.post(
        "/api/v1/bookings/confirm",
        json={
            "hold_token": hold["hold_token"],
            "doctor_id": hold["doctor_id"],
            "date": date,
            "time_slot": slot,
            "patient_name": "Conflict Patient",
            "phone": "9222222222",
            "caller_phone": "9222222222",
        },
    )
    conflict = c.get(
        "/api/v1/bookings/conflict", params={"phone": "9222222222", "date": date, "time_slot": slot}
    ).json()
    assert conflict["conflict"] is True


# ==================== legacy /api/v1/appointments -- CodeRabbit-flagged ====


def test_legacy_endpoint_books_successfully(clinic_client):
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    doctor_name = cat["doctors"][0]["name"]
    date = _next_weekday_with_schedule(c, doctor_name)
    # A real valid slot, via a throwaway hold probe (same trick the other tests use).
    hold = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": "__probe__"}
    ).json()
    slot = hold["valid_slots"][0]

    result = c.post(
        "/api/v1/appointments",
        json={
            "doctor_name": doctor_name,
            "date": date,
            "time_slot": slot,
            "patient_name": "Legacy Patient",
            "phone": "9333333001",
        },
    ).json()
    assert result["success"], result
    assert result["confirmation_id"].startswith("KCD-")


def test_legacy_endpoint_now_sees_a_hold_from_the_new_flow(clinic_client):
    # CodeRabbit-flagged, real bug: the legacy endpoint used to check only
    # its own unguarded Appointment query, invisible to models.SlotLock --
    # a slot HELD (not yet confirmed) via the new /api/v1/bookings/hold
    # flow could still be double-booked through this one. Now routed
    # through the same hold_slot()/confirm_booking() primitives, so it
    # correctly sees the hold and refuses.
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    doctor_name = cat["doctors"][0]["name"]
    date = _next_weekday_with_schedule(c, doctor_name)
    hold = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": "__probe__"}
    ).json()
    slot = hold["valid_slots"][0]

    new_flow_hold = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": slot}
    ).json()
    assert new_flow_hold["success"]

    legacy_result = c.post(
        "/api/v1/appointments",
        json={
            "doctor_name": doctor_name,
            "date": date,
            "time_slot": slot,
            "patient_name": "Double Booker",
            "phone": "9333333002",
        },
    ).json()
    assert legacy_result["success"] is False
    assert legacy_result["reason"] == "slot_taken"


def test_new_flow_now_sees_a_legacy_booking(clinic_client):
    # The other half: a slot booked through the legacy endpoint must show
    # up as unavailable to the new flow too (available_slots() reads
    # SlotLock, which the legacy endpoint now populates).
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    doctor_name = cat["doctors"][0]["name"]
    date = _next_weekday_with_schedule(c, doctor_name)
    hold = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": "__probe__"}
    ).json()
    slot = hold["valid_slots"][0]

    legacy_result = c.post(
        "/api/v1/appointments",
        json={
            "doctor_name": doctor_name,
            "date": date,
            "time_slot": slot,
            "patient_name": "Legacy Patient",
            "phone": "9333333003",
        },
    ).json()
    assert legacy_result["success"]

    new_hold = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": slot}
    ).json()
    assert new_hold["success"] is False
    assert new_hold["reason"] == "slot_taken"


# ============ reschedule/book-tests now validate like hold does (CodeRabbit) ==


def test_reschedule_rejects_a_past_date(clinic_client):
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    doctor_name = cat["doctors"][0]["name"]
    date = _next_weekday_with_schedule(c, doctor_name, min_days_ahead=2)
    hold = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": "__probe__"}
    ).json()
    slot = hold["valid_slots"][0]
    hold = c.post("/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": slot}).json()
    confirm = c.post(
        "/api/v1/bookings/confirm",
        json={
            "hold_token": hold["hold_token"],
            "doctor_id": hold["doctor_id"],
            "date": date,
            "time_slot": slot,
            "patient_name": "Test Patient",
            "phone": "9123456781",
            "caller_phone": "9123456781",
        },
    ).json()
    assert confirm["success"]

    past = (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
    result = c.post(
        "/api/v1/bookings/reschedule",
        json={
            "confirmation_id": confirm["confirmation_id"],
            "new_date": past,
            "new_time_slot": "10:00",
        },
    ).json()
    # Used to skip validation entirely and attempt a hold on a past date.
    assert result == {"success": False, "reason": "date_in_past"}


def test_reschedule_rejects_a_slot_outside_chamber_hours(clinic_client):
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    doctor_name = cat["doctors"][0]["name"]
    date = _next_weekday_with_schedule(c, doctor_name, min_days_ahead=2)
    hold = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": "__probe__"}
    ).json()
    slot = hold["valid_slots"][0]
    hold = c.post("/api/v1/bookings/hold", json={"doctor_name": doctor_name, "date": date, "time_slot": slot}).json()
    confirm = c.post(
        "/api/v1/bookings/confirm",
        json={
            "hold_token": hold["hold_token"],
            "doctor_id": hold["doctor_id"],
            "date": date,
            "time_slot": slot,
            "patient_name": "Test Patient",
            "phone": "9123456782",
            "caller_phone": "9123456782",
        },
    ).json()
    assert confirm["success"]

    result = c.post(
        "/api/v1/bookings/reschedule",
        json={
            "confirmation_id": confirm["confirmation_id"],
            "new_date": date,
            "new_time_slot": "03:45",
        },
    ).json()
    assert result["success"] is False
    assert result["reason"] == "invalid_slot"


def test_book_tests_rejects_a_past_date(clinic_client):
    c = clinic_client
    cat = c.get("/api/v1/catalogue").json()
    test_name = cat["tests"][0]["name"]
    past = (datetime.date.today() - datetime.timedelta(days=3)).isoformat()
    result = c.post(
        "/api/v1/bookings/tests",
        json={
            "test_names": [test_name],
            "date": past,
            "patient_name": "X",
            "phone": "9123456783",
            "caller_phone": "9123456783",
        },
    ).json()
    # Used to skip validation and attempt to book tests for a past date.
    assert result == {"success": False, "reason": "date_in_past"}
