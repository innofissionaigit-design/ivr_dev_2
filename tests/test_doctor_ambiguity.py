"""Two doctors, one name: "Dr. Ashok Sen" and "Dr. Abhishek Sen" are both "Dr. A. Sen", and a caller asks for "Dr. A. Sen".
The agent must never pick one: every duplicate or confusion is a question, asked by WHOLE name in the caller's script.

    python -m pytest tests/test_doctor_ambiguity.py -v
"""

import datetime
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app

from agent.fast_path import Catalogue, FastPath
from agent.reply_templates import booking_reply, doctor_availability_reply

TODAY = datetime.date(2026, 9, 25)


@pytest.fixture(scope="module")
def api():
    """The seeded clinic, with its one Dr. A. Sen turned into Ashok Sen and an Abhishek Sen added beside him."""
    with clinic_app(sample_patients=False) as (app, client):
        from db import SessionLocal
        from models import Doctor

        db = SessionLocal()
        try:
            sen = db.query(Doctor).filter_by(name="Dr. A. Sen").one()
            sen.full_name, sen.full_name_bn, sen.full_name_hi = "Dr. Ashok Sen", "অশোক সেন", "अशोक सेन"
            db.add(
                Doctor(
                    name="Dr. A. Sen",
                    full_name="Dr. Abhishek Sen",
                    full_name_bn="অভিষেক সেন",
                    full_name_hi="अभिषेक सेन",
                    qualifications="MBBS",
                    aliases_bn=sen.aliases_bn,
                    aliases_hi=sen.aliases_hi,
                    department_id=sen.department_id,
                )
            )
            db.commit()
        finally:
            db.close()
        yield app, client


def _avail(client, name, **params):
    return client.get("/api/v1/doctors/availability", params={"name": name, **params}).json()


# ================================================================================ the server asks, by whole name


@pytest.mark.parametrize("said", ["Dr. A. Sen", "Sen", "Dr Sen", "সেন", "ডাক্তার সেন", "सेन"])
def test_a_name_two_doctors_share_is_a_question_never_an_answer(api, said):
    _app, c = api
    body = _avail(c, said)
    assert body["found"] is False and body["ambiguous"] is True
    # in Bengali or Devanagari script "সেন" is also inside Sengupta's surname alias, which is a genuine candidate too
    assert {"Dr. Ashok Sen", "Dr. Abhishek Sen"} <= set(body["did_you_mean"])
    assert {"অশোক সেন", "অভিষেক সেন"} <= set(body["did_you_mean_bn"])  # a Bengali voice can say these
    assert {"अशोक सेन", "अभिषेक सेन"} <= set(body["did_you_mean_hi"])
    if said in ("Dr. A. Sen", "Sen", "Dr Sen"):
        assert set(body["did_you_mean"]) == {"Dr. Ashok Sen", "Dr. Abhishek Sen"}


@pytest.mark.parametrize(
    "said,doctor_full",
    [
        ("Ashok Sen", "Dr. Ashok Sen"),
        ("Dr. Abhishek Sen", "Dr. Abhishek Sen"),
        ("Abhishek", "Dr. Abhishek Sen"),
        ("অশোক সেন", "Dr. Ashok Sen"),
        ("অভিষেক সেন", "Dr. Abhishek Sen"),
        ("अशोक सेन", "Dr. Ashok Sen"),
    ],
)
def test_the_whole_name_reaches_exactly_one_doctor(api, said, doctor_full):
    _app, c = api
    body = _avail(c, said)
    assert body["found"] is True, body
    assert body["doctor_name"] == doctor_full  # the reply names him so it can be looked up again


def test_a_doctor_with_a_name_of_his_own_is_unaffected(api):
    _app, c = api
    body = _avail(c, "Mukherjee")
    assert body["found"] is True and body["doctor_name"] == "Dr. S. Mukherjee"


def test_booking_a_shared_name_is_a_question_too(api):
    _app, c = api
    body = c.post(
        "/api/v1/bookings/hold", json={"doctor_name": "Dr. A. Sen", "date": "2026-10-05", "time_slot": "10:00"}
    ).json()
    assert body["success"] is False and body["reason"] == "doctor_ambiguous"
    assert set(body["did_you_mean"]) == {"Dr. Ashok Sen", "Dr. Abhishek Sen"}
    legacy = c.post(
        "/api/v1/appointments",
        json={
            "doctor_name": "Sen",
            "date": "2026-10-05",
            "time_slot": "10:00",
            "patient_name": "X",
            "phone": "9830000000",
        },
    ).json()
    assert legacy["success"] is False and legacy["reason"] == "doctor_ambiguous"


def test_earliest_and_leave_ask_as_well(api):
    _app, c = api
    assert c.get("/api/v1/doctors/earliest", params={"name": "Dr. A. Sen"}).json()["ambiguous"] is True
    assert c.get("/api/v1/doctors/Sen/leave", params={"date": "2026-10-05"}).json()["ambiguous"] is True


def test_two_doctors_who_still_read_the_same_are_told_apart_by_department(api):
    _app, c = api
    from db import SessionLocal
    from models import Department, Doctor

    db = SessionLocal()
    try:
        sen = db.query(Doctor).filter_by(full_name="Dr. Abhishek Sen").one()
        other_dept = db.query(Department).filter(Department.id != sen.department_id).first()
        twin = Doctor(
            name="Dr. A. Sen",
            full_name="Dr. Ashok Sen",
            full_name_bn="অশোক সেন",
            full_name_hi="अशोक सेन",
            qualifications="MBBS",
            department_id=other_dept.id,
        )
        db.add(twin)
        db.commit()
        twin_id = twin.id
    finally:
        db.close()
    try:
        body = _avail(c, "Ashok Sen")
        assert body["ambiguous"] is True and all("(" in label for label in body["did_you_mean"]), body
    finally:
        db = SessionLocal()
        db.query(Doctor).filter_by(id=twin_id).delete()
        db.commit()
        db.close()


# ================================================================================================ the agent's words


def test_the_agent_asks_which_one_in_every_language():
    result = {
        "found": False,
        "ambiguous": True,
        "did_you_mean": ["Dr. Ashok Sen", "Dr. Abhishek Sen"],
        "did_you_mean_bn": ["অশোক সেন", "অভিষেক সেন"],
        "did_you_mean_hi": ["अशोक सेन", "अभिषेक सेन"],
    }
    bn = doctor_availability_reply({"doctor_name": "Dr. A. Sen"}, result, "bn")
    assert "অশোক সেন নাকি অভিষেক সেন" in bn and bn.endswith("?")
    hi = doctor_availability_reply({"doctor_name": "Dr. A. Sen"}, result, "hi")
    assert "अशोक सेन" in hi and "अभिषेक सेन" in hi and hi.endswith("?")
    en = doctor_availability_reply({"doctor_name": "Dr. A. Sen"}, result, "en")
    assert "Dr. Ashok Sen or Dr. Abhishek Sen" in en and en.endswith("?")


def test_a_booking_that_hits_a_shared_name_asks_which_one_in_every_language():
    result = {
        "success": False,
        "reason": "doctor_ambiguous",
        "did_you_mean": ["Dr. Ashok Sen", "Dr. Abhishek Sen"],
        "did_you_mean_bn": ["অশোক সেন", "অভিষেক সেন"],
        "did_you_mean_hi": ["अशोक सेन", "अभिषेक सेन"],
    }
    assert "অশোক সেন নাকি অভিষেক সেন" in booking_reply({}, result, "bn")
    assert "अशोक सेन या अभिषेक सेन" in booking_reply({}, result, "hi")
    assert "Dr. Ashok Sen or Dr. Abhishek Sen" in booking_reply({}, result, "en")
    for lang in ("bn", "hi", "en"):
        assert "not found" not in booking_reply({}, result, lang).lower()  # not the "no such doctor" apology


def test_the_orchestrator_clears_the_doctor_after_an_ambiguous_hold():
    src = open(os.path.join(REPO_ROOT, "main.py"), encoding="utf-8").read()
    assert '"doctor_ambiguous": "doctor_name"' in src


# =================================================================================================== the fast path


def test_the_fast_path_never_answers_on_a_surname_two_doctors_share(api):
    _app, c = api
    fp = FastPath(Catalogue(c.get("/api/v1/catalogue").json()), today=TODAY)
    for text, lang in [
        ("ডাক্তার সেন কবে বসবেন", "bn"),
        ("sen kab baithte hain", "hi"),
        ("डॉक्टर सेन कब बैठते हैं", "hi"),
        ("when does dr sen sit", "en"),
    ]:
        assert fp.resolve(text, lang) is None, (text, lang)
    hit = fp.resolve("ডাক্তার মুখার্জী কবে বসবেন", "bn")  # a surname of one doctor is still served
    assert hit is not None and hit.intent == "doctor_availability"


def test_the_seeded_whole_names_exist_in_all_three_scripts():
    sys.path.insert(0, os.path.join(REPO_ROOT, "clinic-api"))
    from seed import doctor_full_names

    names = doctor_full_names()
    assert len(names) == 32
    for short, (latin, bn, hi) in names.items():
        assert latin and bn and hi, short
        assert any("ঀ" <= ch <= "৿" for ch in bn) and any("ऀ" <= ch <= "ॿ" for ch in hi), short


# ==================================================================== the live call: "ডক্টর পার্থ রায় কবে বসছে"


def test_a_doctor_named_in_full_is_one_doctor_even_when_the_surname_is_shared(api):
    """Live call 2026-09-25: "ডক্টর পার্থ রায় কবে বসছে" was answered "রায় নাকি রায়?" -- Dr. N. Roy and Dr. P. Ray share the
    Bengali spelling of their surname, and the whole name was never consulted. It is now."""
    _app, c = api
    for said in ("ডক্টর পার্থ রায়", "পার্থ রায়", "Partha Ray", "Dr. Partha Ray", "पार्थ रे"):
        body = _avail(c, said)
        assert body["found"] is True and body["doctor_name"] == "Dr. P. Ray", (said, body)
    body = _avail(c, "রায়")  # the bare surname IS a question
    assert body["ambiguous"] is True and set(body["did_you_mean_bn"]) >= {"পার্থ রায়", "নির্মল রায়"}, body


def test_the_fast_path_answers_a_fully_named_doctor_and_asks_about_a_shared_surname(api):
    _app, c = api
    fp = FastPath(Catalogue(c.get("/api/v1/catalogue").json()), today=TODAY)
    hit = fp.resolve("ডক্টর পার্থ রায় কবে বসছে এই সপ্তাহে বসছে", "bn")
    assert hit is not None and hit.intent == "doctor_availability" and hit.slots["doctor_name"] == "পার্থ রায়"
    assert fp.resolve("ডক্টর রায় কবে বসছে", "bn") is None
