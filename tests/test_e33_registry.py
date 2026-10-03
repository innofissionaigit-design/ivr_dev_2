"""The patient registry, security questions, call-outcome history, one-day cache, operator messages and
audit log (KCD-493..501, KCD-353, KCD-512) -- against a REAL clinic API seeded with the sample patients.

    python -m pytest tests/test_e33_registry.py -v
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

ASHA, RAKESH, TANIA = "KCP-100001", "KCP-100002", "KCP-100003"
DEBASHIS, SUNITA, RAVI = "KCP-100004", "KCP-100005", "KCP-100006"
SHARED_PHONE = "9830012345"


@pytest.fixture(scope="module")
def api():
    with clinic_app() as (app, client):
        yield client


_calls = iter(range(10_000))


def new_call() -> str:
    return f"call-{next(_calls)}"


def ref_of(c, uid: str) -> int:
    r = c.post("/api/v1/patients/find", json={"call_id": new_call(), "patient_id": uid}).json()
    assert r["status"] == "single", (uid, r)
    return r["patient_ref"]


def verify(c, call, ref, **answers):
    return c.post("/api/v1/patients/verify", json={"call_id": call, "patient_ref": ref, **answers}).json()


def timeline(c, call, ref):
    return c.get(f"/api/v1/patients/{ref}/timeline", params={"caller_phone": "0", "call_id": call}).json()


# ================================================================ sample data is really there


def test_the_sample_cast_is_seeded_with_a_shared_number(api):
    assert api.get("/api/v1/patients/identify", params={"phone": SHARED_PHONE}).json()["status"] == "ambiguous"
    assert api.get("/api/v1/patients/identify", params={"phone": "9830023456"}).json()["status"] == "single"
    assert api.get("/api/v1/patients/identify", params={"phone": "9000000000"}).json()["status"] == "new"
    for uid in (ASHA, RAKESH, TANIA, DEBASHIS, SUNITA, RAVI):
        ref_of(api, uid)


def test_sample_data_is_never_reseeded_over_existing_rows(api):
    with clinic_app(sample_patients=False) as (_, empty):
        assert empty.post("/api/v1/patients/find", json={"call_id": "x", "patient_id": ASHA}).json()["status"] == "none"


# ================================================================ the rule: two facts, one strong


def test_date_of_birth_and_name_verify(api):
    ref, call = ref_of(api, DEBASHIS), new_call()
    r = verify(api, call, ref, dob="1985-06-30", name="my name is Debashis Mondal")
    assert r["verified"] is True and r["age_years"] >= 40 and r["is_senior"] is False


def test_patient_id_and_address_verify(api):
    ref, call = ref_of(api, RAVI), new_call()
    assert verify(api, call, ref, patient_id="kcp 100006", address="Howrah Maidan 711101")["verified"] is True


def test_one_fact_is_never_enough_and_is_not_even_evaluated(api):
    ref, call = ref_of(api, DEBASHIS), new_call()
    r = verify(api, call, ref, dob="1985-06-30")
    assert r["verified"] is False and r.get("need_more") is True and r["attempts_left"] == 3


# NOTE: each patient tolerates five failed evaluations per 24 h across ALL calls (the brute-force rule is
# itself tested below), so the tests that fail on purpose spread over different patients.
def test_two_weak_facts_are_not_enough_without_a_strong_one(api):
    ref, call = ref_of(api, RAKESH), new_call()
    r = verify(api, call, ref, name="Rakesh Saha", address="Ballygunge Gariahat 700019")
    assert r["verified"] is False and r["attempts_left"] == 2


def test_a_first_name_alone_is_not_a_full_name(api):
    ref, call = ref_of(api, TANIA), new_call()
    assert verify(api, call, ref, dob="2010-11-21", name="Tania")["verified"] is False


def test_a_registered_name_in_bengali_script_verifies(api):
    ref, call = ref_of(api, DEBASHIS), new_call()
    assert verify(api, call, ref, dob="1985-06-30", name="আমার নাম দেবাশীষ মণ্ডল")["verified"] is True


def test_a_wrong_date_of_birth_fails_and_the_reply_never_says_which_answer_was_wrong(api):
    ref, call = ref_of(api, RAKESH), new_call()
    r = verify(api, call, ref, dob="1975-08-01", name="Rakesh Saha")
    assert r == {"verified": False, "attempts_left": 2, "locked": False}


def test_three_failed_evaluations_lock_the_check_for_the_call_even_if_the_next_answers_are_right(api):
    ref, call = ref_of(api, RAKESH), new_call()
    for _ in range(3):
        r = verify(api, call, ref, dob="1990-01-01", name="Rakesh Saha")
    assert r["locked"] is True and r["attempts_left"] == 0
    assert verify(api, call, ref, dob="1975-08-02", name="Rakesh Saha")["verified"] is False


def test_failures_spread_over_many_calls_lock_the_patient(api):
    ref = ref_of(api, RAVI)
    for _ in range(3):  # three calls x two failed evaluations = six failures
        call = new_call()
        for _ in range(2):
            verify(api, call, ref, dob="1990-01-01", name="Ravi Kumar")
    r = verify(api, new_call(), ref, dob="1990-12-05", name="Ravi Kumar")  # the RIGHT answers, a fresh call
    assert r["verified"] is False and r["locked"] is True


def test_address_needs_two_distinct_words_or_the_pincode_and_one(api):
    ref = ref_of(api, SUNITA)
    assert verify(api, new_call(), ref, dob="1962-01-09", address="Burrabazar")["verified"] is False  # one word
    assert verify(api, new_call(), ref, dob="1962-01-09", address="Burrabazar 700007")["verified"] is True


# ================================================================ history is refused until verified


def test_a_timeline_read_is_refused_until_this_call_has_verified_this_patient(api):
    ref, call = ref_of(api, DEBASHIS), new_call()
    assert timeline(api, call, ref) == {"success": False, "reason": "not_verified"}
    verify(api, call, ref, dob="1985-06-30", name="Debashis Mondal")
    assert timeline(api, call, ref)["success"] is True
    assert timeline(api, new_call(), ref)["reason"] == "not_verified"  # another call has not passed
    other = ref_of(api, RAVI)
    assert timeline(api, call, other)["reason"] == "not_verified"  # nor is this call verified for someone else


def test_test_status_is_behind_the_same_gate_and_states_due_only_from_an_approved_interval(api):
    ref, call = ref_of(api, SUNITA), new_call()
    denied = api.get(
        f"/api/v1/patients/{ref}/test-status", params={"test_name": "HbA1c", "caller_phone": "0", "call_id": call}
    ).json()
    assert denied["reason"] == "not_verified"
    verify(api, call, ref, dob="1962-01-09", name="Sunita Devi")
    s = api.get(
        f"/api/v1/patients/{ref}/test-status", params={"test_name": "HbA1c", "caller_phone": "0", "call_id": call}
    ).json()
    assert s["success"] and s["known"] and s["due"] is True and s["approved_by"] == "SAMPLE DATA"
    kft = api.get(
        f"/api/v1/patients/{ref}/test-status",
        params={"test_name": "Kidney Function Test (KFT)", "caller_phone": "0", "call_id": call},
    ).json()
    assert kft["known"] is True and kft["due"] is None  # no approved interval: no "due"


def test_the_timeline_carries_medicines_tests_and_appointments_with_names_in_the_callers_script(api):
    ref, call = ref_of(api, DEBASHIS), new_call()
    verify(api, call, ref, dob="1985-06-30", name="Debashis Mondal")
    events = timeline(api, call, ref)["events"]
    kinds = {e["kind"] for e in events}
    assert {"medicine_prescribed", "test_performed", "appointment"} <= kinds
    med = next(e for e in events if e["kind"] == "medicine_prescribed")
    assert (
        med["fields"]["medicine_name"] == "Levothyroxine"
        and med["fields"]["medicine_name_bn"]
        and med["fields"]["medicine_name_hi"]
    )
    assert all(k not in med["fields"] for k in ("dose", "result", "value", "advice"))
    test = next(e for e in events if e["kind"] == "test_performed" and "CBC" in e["fields"]["test_name"])
    assert test["fields"]["test_name_bn"] and test["fields"]["test_name_hi"]


# ================================================================ find without a phone record (KCD-497)


def test_a_patient_id_finds_the_record_and_returns_only_an_opaque_reference(api):
    r = api.post("/api/v1/patients/find", json={"call_id": new_call(), "patient_id": "kcp-100005"}).json()
    assert r["status"] == "single" and set(r) == {"status", "patient_ref"}


def test_a_date_of_birth_with_a_name_finds_the_record(api):
    r = api.post(
        "/api/v1/patients/find", json={"call_id": new_call(), "dob": "1962-01-09", "name": "Sunita Devi"}
    ).json()
    assert r["status"] == "single"


def test_a_wrong_pair_finds_nobody_and_a_lone_name_is_insufficient(api):
    assert api.post(
        "/api/v1/patients/find", json={"call_id": new_call(), "dob": "1962-01-10", "name": "Sunita Devi"}
    ).json() == {"status": "none"}
    assert api.post("/api/v1/patients/find", json={"call_id": new_call(), "name": "Sunita Devi"}).json() == {
        "status": "insufficient"
    }


# ================================================================ senior citizen from the registered date of birth (KCD-512)


def test_the_registry_says_who_is_a_senior_citizen(api):
    asha = verify(api, new_call(), ref_of(api, ASHA), dob="1948-03-14", name="Asha Saha")
    sunita = verify(api, new_call(), ref_of(api, SUNITA), dob="1962-01-09", name="Sunita Devi")
    tania = verify(api, new_call(), ref_of(api, TANIA), dob="2010-11-21", name="Tania Saha")
    assert asha["is_senior"] and asha["age_years"] >= 77
    assert sunita["is_senior"]
    assert tania["is_senior"] is False and tania["age_years"] < 20


# ================================================================ call outcome -> history + one-day cache (KCD-496/501)


def _events(c, call, items, phone="9830023456"):
    for seq, (kind, payload) in enumerate(items, start=1):
        assert c.post(
            f"/api/v1/calls/{call}/events", json={"seq": seq, "kind": kind, "payload": payload, "caller_phone": phone}
        ).json()["applied"]


def test_closing_a_call_writes_history_rows_a_cache_entry_and_an_audit_row(api):
    ref, call = ref_of(api, DEBASHIS), new_call()
    _events(
        api,
        call,
        [
            ("start", {"channel": "voice"}),
            ("patient", {"patient_ref": ref}),
            ("intent", {"intent": "book_appointment"}),
            ("action", {"name": "booking_confirmed", "ref": "KCD-20990101-AAAA1111"}),
            ("end", {"outcome": "completed"}),
        ],
    )
    ctx = api.get("/api/v1/continuity", params={"caller_phone": "9830023456", "patient_ref": ref}).json()
    assert (
        ctx["cached"]
        and ctx["cached"][0]["outcome"] == "completed"
        and ctx["cached"][0]["actions"][0]["name"] == "booking_confirmed"
    )
    log = api.get("/api/v1/audit", params={"call_id": call}).json()["entries"]
    assert {"call_closed", "action:booking_confirmed"} <= {e["action"] for e in log}
    sess = sys.modules["db"].SessionLocal()
    try:
        rows = sess.query(sys.modules["models"].PatientHistory).filter_by(call_id=call).all()
        assert {r.kind for r in rows} == {"booking_confirmed", "call_outcome"}
    finally:
        sess.close()


def test_a_retried_end_event_does_not_double_the_history(api):
    ref, call = ref_of(api, RAVI), new_call()
    _events(
        api,
        call,
        [
            ("patient", {"patient_ref": ref}),
            ("action", {"name": "tests_booked", "ref": "K1"}),
            ("end", {"outcome": "completed"}),
        ],
    )
    api.post(f"/api/v1/calls/{call}/events", json={"seq": 3, "kind": "end", "payload": {"outcome": "completed"}})
    sess = sys.modules["db"].SessionLocal()
    try:
        assert sess.query(sys.modules["models"].PatientHistory).filter_by(call_id=call).count() == 2
        assert sess.query(sys.modules["models"].PatientCallCache).filter_by(call_id=call).count() == 1
    finally:
        sess.close()


def test_the_cache_is_kept_for_one_day_and_an_expired_entry_is_never_used(api):
    ref, call = ref_of(api, RAVI), new_call()
    _events(
        api,
        call,
        [("patient", {"patient_ref": ref}), ("intent", {"intent": "book_test"}), ("end", {"outcome": "abandoned"})],
        phone="9830045678",
    )
    fresh = api.get("/api/v1/continuity", params={"caller_phone": "9830045678", "patient_ref": ref}).json()["cached"]
    assert any(c["outcome"] == "abandoned" and c["unfinished"] for c in fresh)
    sess = sys.modules["db"].SessionLocal()
    try:
        m = sys.modules["models"]
        row = sess.query(m.PatientCallCache).filter_by(call_id=call).one()
        assert (row.expires_at - row.created_at) == datetime.timedelta(hours=24)
        row.expires_at = datetime.datetime.now() - datetime.timedelta(minutes=1)
        sess.commit()
    finally:
        sess.close()
    after = api.get("/api/v1/continuity", params={"caller_phone": "9830045678", "patient_ref": ref}).json()["cached"]
    assert all(c["at"] != fresh[0]["at"] or c["outcome"] != "abandoned" for c in after)


# ================================================================ audit trail (KCD-501)


def test_every_identification_verification_and_history_read_is_audited(api):
    ref, call = ref_of(api, SUNITA), new_call()
    api.get("/api/v1/patients/identify", params={"phone": "9830034567"})
    verify(api, call, ref, dob="1962-01-01", name="Sunita Devi")  # fails
    verify(api, call, ref, dob="1962-01-09", name="Sunita Devi")  # passes
    timeline(api, call, ref)
    entries = api.get("/api/v1/audit", params={"call_id": call}).json()["entries"]
    seen = {(e["action"], e["outcome"]) for e in entries}
    assert ("verify", "failed") in seen and ("verify", "ok") in seen and ("history:timeline_read", "ok") in seen
    assert all("1962" not in str(e["detail"]) for e in entries)  # the answers themselves are never logged
    assert any(
        e["action"] == "identify" for e in api.get("/api/v1/audit", params={"action": "identify"}).json()["entries"]
    )


def test_a_refused_read_is_audited_as_denied(api):
    ref, call = ref_of(api, RAVI), new_call()
    timeline(api, call, ref)
    entries = api.get("/api/v1/audit", params={"call_id": call}).json()["entries"]
    assert any(e["action"].startswith("history:timeline_denied") and e["outcome"] == "denied" for e in entries)


# ================================================================ operator-editable messages (KCD-353/500/513)


def test_the_default_messages_are_served_and_equal_the_agents_built_in_text(api):
    from agent.disclosure import DISCLOSURE
    from agent.history_templates import CANNOT_FIND
    from agent.turn_ack import THANKS

    got = api.get("/api/v1/agent/messages").json()["messages"]
    for lang in ("bn", "hi", "en"):
        assert got["disclosure"][lang]["text"] == DISCLOSURE[lang]
        assert got["cannot_find"][lang]["text"] == CANNOT_FIND[lang]
        assert got["thanks_ack"][lang]["text"] == THANKS[lang]


def test_changing_a_message_takes_effect_without_a_deploy_and_bumps_the_version(api):
    before = api.get("/api/v1/agent/messages", params={"lang": "en"}).json()["messages"]["thanks_ack"]["en"]
    r = api.put(
        "/api/v1/agent/messages",
        json={"key": "thanks_ack", "lang": "en", "text": "Thanks for letting me know.", "updated_by": "clinic-lead"},
    ).json()
    assert r == {"success": True, "key": "thanks_ack", "lang": "en", "version": before["version"] + 1}
    after = api.get("/api/v1/agent/messages", params={"lang": "en"}).json()["messages"]["thanks_ack"]["en"]
    assert after["text"] == "Thanks for letting me know."
    same = api.put(
        "/api/v1/agent/messages", json={"key": "thanks_ack", "lang": "en", "text": "Thanks for letting me know."}
    ).json()
    assert same["version"] == r["version"]  # no change, no new version
    assert any(
        e["action"] == "message_changed" and e["actor"] == "operator"
        for e in api.get("/api/v1/audit", params={"action": "message_changed"}).json()["entries"]
    )


def test_an_invalid_message_is_refused_and_an_inactive_one_is_not_served(api):
    assert (
        api.put("/api/v1/agent/messages", json={"key": "x", "lang": "fr", "text": "bonjour"}).json()["success"] is False
    )
    assert api.put("/api/v1/agent/messages", json={"key": "x", "lang": "en", "text": "  "}).json()["success"] is False
    assert (
        api.put("/api/v1/agent/messages", json={"key": "x", "lang": "en", "text": "y" * 601}).json()["success"] is False
    )
    api.put("/api/v1/agent/messages", json={"key": "temp_key", "lang": "en", "text": "Temporary."})
    api.put("/api/v1/agent/messages", json={"key": "temp_key", "lang": "en", "text": "Temporary.", "active": False})
    assert "temp_key" not in api.get("/api/v1/agent/messages").json()["messages"]


# ================================================================ the service token covers the new routes


def test_the_new_routes_are_behind_the_service_token(api, monkeypatch):
    monkeypatch.setenv("CLINIC_API_TOKEN", "tok-for-test")
    assert api.post("/api/v1/patients/verify", json={"call_id": "c", "patient_ref": 1}).status_code == 401
    assert api.post("/api/v1/patients/find", json={"call_id": "c", "patient_id": ASHA}).status_code == 401
    assert api.get("/api/v1/agent/messages").status_code == 401
    assert api.put("/api/v1/agent/messages", json={"key": "k", "lang": "en", "text": "t"}).status_code == 401
    assert api.get("/api/v1/audit").status_code == 401
    ok = api.get("/api/v1/agent/messages", headers={"Authorization": "Bearer tok-for-test"})
    assert ok.status_code == 200
