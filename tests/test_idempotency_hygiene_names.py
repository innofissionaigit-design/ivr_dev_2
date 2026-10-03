"""Code hygiene pass (2026-09-25): idempotent writes, request models, tuples/lists, and doctors' whole names.

python -m pytest tests/test_idempotency_hygiene_names.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests"), os.path.join(REPO_ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import hygiene_scan
from _clinic_app import clinic_app


@pytest.fixture(scope="module")
def api():
    with clinic_app(sample_patients=False) as (app, client):
        yield app, client


# ============================================================================================== idempotent writes


def test_a_retried_sms_with_the_same_key_is_one_message(api):
    _app, c = api
    body = {
        "to": "9830011111",
        "template_key": "confirm",
        "message": "Your booking is confirmed.",
        "related_confirmation_id": "KCD-T1",
    }
    a = c.post("/api/v1/notifications/sms", json=body, headers={"Idempotency-Key": "k-sms-1"}).json()
    b = c.post("/api/v1/notifications/sms", json=body, headers={"Idempotency-Key": "k-sms-1"}).json()
    assert a == b and a["queued"] is True


def test_the_same_key_with_a_different_body_is_refused(api):
    _app, c = api
    c.post(
        "/api/v1/callbacks",
        json={"phone": "9830022222", "call_id": "c1", "requested_window": "morning"},
        headers={"Idempotency-Key": "k-cb-1"},
    )
    r = c.post(
        "/api/v1/callbacks",
        json={"phone": "9830022222", "call_id": "c1", "requested_window": "evening"},
        headers={"Idempotency-Key": "k-cb-1"},
    )
    assert r.status_code == 422


def test_the_same_key_on_two_endpoints_is_two_keys(api):
    _app, c = api
    a = c.post(
        "/api/v1/callbacks",
        json={"phone": "9830033333", "call_id": "c2", "requested_window": "morning"},
        headers={"Idempotency-Key": "shared"},
    )
    b = c.post(
        "/api/v1/calls/out-of-scope",
        json={"call_id": "c2", "caller_question": "q", "reason_code": "out_of_scope"},
        headers={"Idempotency-Key": "shared"},
    )
    assert a.status_code == 200 and b.status_code == 200


def test_without_a_key_the_same_sms_within_the_window_is_still_one_message(api):
    _app, c = api
    body = {
        "to": "9830044444",
        "template_key": "confirm",
        "message": "Hello there.",
        "related_confirmation_id": "KCD-T2",
    }
    a = c.post("/api/v1/notifications/sms", json=body).json()
    b = c.post("/api/v1/notifications/sms", json=body).json()
    assert a["id"] == b["id"] and b.get("duplicate") is True


def test_without_a_key_a_callback_asked_for_twice_is_one_callback(api):
    _app, c = api
    body = {"phone": "9830055555", "call_id": "c3", "requested_window": "morning", "reason": "r"}
    a, b = c.post("/api/v1/callbacks", json=body).json(), c.post("/api/v1/callbacks", json=body).json()
    assert a["id"] == b["id"] and b.get("duplicate") is True


def test_without_a_key_the_same_out_of_scope_question_is_recorded_once(api):
    _app, c = api
    body = {"call_id": "c4", "caller_question": "can you diagnose me", "reason_code": "out_of_scope"}
    a, b = (
        c.post("/api/v1/calls/out-of-scope", json=body).json(),
        c.post("/api/v1/calls/out-of-scope", json=body).json(),
    )
    assert a["id"] == b["id"]


def test_a_different_message_is_a_different_sms(api):
    _app, c = api
    base = {"to": "9830066666", "template_key": "confirm", "related_confirmation_id": "KCD-T3"}
    a = c.post("/api/v1/notifications/sms", json={**base, "message": "First."}).json()
    b = c.post("/api/v1/notifications/sms", json={**base, "message": "Second."}).json()
    assert a["id"] != b["id"]


# ================================================================================================== request models


def test_resend_takes_a_request_model_and_still_accepts_the_old_query_form(api):
    _app, c = api
    not_found = {"success": False, "reason": "not_found"}
    assert c.post("/api/v1/bookings/resend", json={"confirmation_id": "KCD-NOPE"}).json() == not_found
    assert c.post("/api/v1/bookings/resend", params={"confirmation_id": "KCD-NOPE"}).json() == not_found
    assert c.post("/api/v1/bookings/resend").status_code == 422


def test_every_write_endpoint_with_a_body_declares_a_schema(api):
    """Every POST/PUT body is a declared schema (a pydantic model), never an untyped free-form object."""
    app, _c = api
    for path, ops in app.openapi()["paths"].items():
        for method, op in ops.items():
            if method not in ("post", "put", "patch") or op.get("requestBody") is None:
                continue
            schema = op["requestBody"]["content"]["application/json"]["schema"]
            assert "$ref" in schema or "anyOf" in schema, (path, method, schema)


def test_a_model_never_defaults_a_field_to_a_shared_mutable():
    src = "from pydantic import BaseModel\nclass M(BaseModel):\n    x: dict = {}\n"
    assert hygiene_scan.scan_source(src)[0][2] == "pydantic-mutable"
    assert not [f for f in hygiene_scan.scan_tree(os.path.join(REPO_ROOT, "clinic-api")) if f[2] == "pydantic-mutable"]


# ================================================================================================ tuples and lists


@pytest.mark.parametrize(
    "source,rule",
    [
        ("def f(x=[]):\n    return x\n", "mutable-default"),
        ("def f(*, x={}):\n    return x\n", "mutable-default"),
        ("from dataclasses import dataclass\n@dataclass\nclass A:\n    xs: list = []\n", "dataclass-mutable"),
        ('ok = "b" in ("abc")\n', "str-in-str"),
        ('if word in ("ok"):\n    pass\n', "str-in-str"),
        ("s.startswith(['a', 'b'])\n", "startswith-list"),
        ("isinstance(x, [int, str])\n", "isinstance-list"),
        ('NAMES = ("only")\n', "tuple-of-one-str"),
    ],
)
def test_the_scanner_catches_each_trap(source, rule):
    assert rule in [f[2] for f in hygiene_scan.scan_source(source)]


@pytest.mark.parametrize(
    "source",
    [
        'ok = "b" in ("abc",)\n',
        'NAMES = ("only",)\n',
        "def f(x=None, y=()):\n    return x, y\n",
        "s.startswith(('a', 'b'))\n",
        "isinstance(x, (int, str))\n",
        'ok = word in ("a", "b")\n',
        "from dataclasses import dataclass, field\n@dataclass\nclass A:\n    xs: list = field(default_factory=list)\n",
    ],
)
def test_the_scanner_leaves_the_correct_forms_alone(source):
    assert hygiene_scan.scan_source(source) == []


def test_the_whole_repository_has_no_tuple_list_or_default_traps():
    found = hygiene_scan.scan_tree(REPO_ROOT)
    assert found == [], found


# ================================================================================================ doctors' whole names


def test_every_doctor_has_a_whole_name_that_extends_the_short_one(api):
    _app, c = api
    doctors = c.get("/api/v1/catalogue").json()["doctors"]
    assert len(doctors) == 32
    for d in doctors:
        first_initial, surname = d["name"].split()[1][0], d["name"].split()[-1]
        parts = d["full_name"].split()
        assert parts[0] == "Dr." and len(parts) == 3 and parts[1][0] == first_initial and parts[-1] == surname, d


def test_the_backfill_fills_only_empty_names_and_is_idempotent(api):
    _app, c = api
    import booking_migrate
    from db import SessionLocal
    from models import Doctor

    db = SessionLocal()
    try:
        db.query(Doctor).filter_by(name="Dr. A. Sen").one().full_name = ""
        db.query(Doctor).filter_by(name="Dr. P. Ghosh").one().full_name = "Dr. Prabir Kumar Ghosh"  # a clinician's edit
        db.commit()
    finally:
        db.close()
    assert booking_migrate.backfill_doctor_full_names() == 1
    assert booking_migrate.backfill_doctor_full_names() == 0
    by = {x["name"]: x["full_name"] for x in c.get("/api/v1/catalogue").json()["doctors"]}
    assert by["Dr. A. Sen"] == "Dr. Arindam Sen" and by["Dr. P. Ghosh"] == "Dr. Prabir Kumar Ghosh"
