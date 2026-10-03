"""Epic E33, service layer (clinic-api/patient_context.py): call records (KCD-501), identifying
a caller without assuming (KCD-494), the timeline and its audit (KCD-493/495), test history
without advice (KCD-498), preferences (KCD-499), continuity (KCD-496) and finding a booking
without a reference number (KCD-497).

Runs against a real, freshly seeded SQLite database; the clock is pinned.

    python -m pytest tests/test_patient_context_service.py -v
"""

import datetime
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")

PINNED_NOW = datetime.datetime(2030, 1, 7, 10, 0)  # a Monday
TODAY = PINNED_NOW.date()


@pytest.fixture()
def svc(monkeypatch):
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("CLINIC_DB_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
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
        "patient_context",
        "phonetic_match",
        "disclosure",
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
    import patient_context as pc

    monkeypatch.setattr(bs, "_now", lambda: PINNED_NOW)
    monkeypatch.setattr(pc, "_now", lambda: PINNED_NOW)
    session = db_mod.SessionLocal()

    class S:
        pass

    s = S()
    s.bs, s.m, s.pc, s.db = bs, m, pc, session
    yield s
    session.close()
    try:
        os.remove(db_path)
    except OSError:
        pass


def _patient(s, name, phone, age=None):
    p = s.bs.find_or_create_patient(s.db, name, phone, age)
    s.db.commit()
    return p


def _book_appointment(s, patient, days_ahead=10, phone=None):
    row = s.db.query(s.m.DoctorSchedule).filter_by(weekday=0).first()
    doc = s.db.get(s.m.Doctor, row.doctor_id)
    d = TODAY + datetime.timedelta(days=days_ahead)
    while d.weekday() != row.weekday:
        d += datetime.timedelta(days=1)
    date = d.isoformat()
    slot = s.bs.available_slots(s.db, doc.id, date)[0]
    hold = s.bs.hold_slot(s.db, doc.id, date, slot)
    booked = s.bs.confirm_booking(
        s.db,
        hold["hold_token"],
        doc.id,
        date,
        slot,
        patient.name,
        patient.phone,
        phone or patient.phone,
        patient_age=patient.age,
    )
    return booked["confirmation_id"], date, slot, doc


def _book_test(s, patient, test_name="CBC", days_ahead=3):
    test = s.db.query(s.m.LabTest).filter(s.m.LabTest.name.ilike(f"%{test_name}%")).first()
    date = (TODAY + datetime.timedelta(days=days_ahead)).isoformat()
    res = s.bs.book_tests(s.db, [test.id], date, patient.name, patient.phone, patient.phone, patient.age)
    return res["confirmation_id"], date, test


# ================================================================== KCD-501 call records


def test_a_call_record_is_built_as_events_arrive(svc):
    pc = svc.pc
    assert pc.record_call_event(
        svc.db, "c1", 1, "start", {"channel": "voice", "disclosure_version": "1.0-draft"}, "9000000001"
    )["applied"]
    pc.record_call_event(svc.db, "c1", 2, "language", {"language": "bn"})
    pc.record_call_event(svc.db, "c1", 3, "intent", {"intent": "book_appointment"})
    pc.record_call_event(
        svc.db,
        "c1",
        4,
        "confirmed",
        {"slots": {"doctor_name": "Dr Sen", "date": "2030-01-14", "phone": "9000000001", "patient_name": "Asha Das"}},
    )
    pc.record_call_event(svc.db, "c1", 5, "action", {"name": "booking_confirmed", "ref": "KC1"})
    pc.record_call_event(svc.db, "c1", 6, "end", {"outcome": "completed"})
    rec = pc.get_call_record(svc.db, "c1")
    assert rec["languages"] == ["bn"] and rec["intents"] == ["book_appointment"]
    assert rec["actions"][0]["name"] == "booking_confirmed" and rec["outcome"] == "completed" and rec["ended_at"]
    assert rec["disclosure_version"] == "1.0-draft"
    assert rec["confirmed"] == {"doctor_name": "Dr Sen", "date": "2030-01-14", "phone_last4": "0001"}
    assert "Asha" not in str(rec) and "9000000001" not in str(rec)  # no name, no full number


def test_a_retried_event_changes_nothing(svc):
    pc = svc.pc
    pc.record_call_event(svc.db, "c2", 1, "start")
    first = pc.record_call_event(svc.db, "c2", 2, "action", {"name": "booking_confirmed", "ref": "KC9"})
    again = pc.record_call_event(svc.db, "c2", 2, "action", {"name": "booking_confirmed", "ref": "KC9"})
    older = pc.record_call_event(svc.db, "c2", 1, "action", {"name": "booking_cancelled", "ref": "KCX"})
    assert first["applied"] and not again["applied"] and again["duplicate"] and older["duplicate"]
    assert len(pc.get_call_record(svc.db, "c2")["actions"]) == 1
    assert svc.db.query(svc.m.CallRecord).filter_by(call_id="c2").count() == 1


def test_a_call_that_drops_still_records_what_was_completed(svc):
    pc = svc.pc
    pc.record_call_event(svc.db, "c3", 1, "start")
    pc.record_call_event(svc.db, "c3", 2, "action", {"name": "booking_confirmed", "ref": "KC7"})
    rec = pc.get_call_record(svc.db, "c3")  # no "end" ever arrives
    assert rec["outcome"] == "in_progress" and rec["actions"][0]["ref"] == "KC7" and rec["ended_at"] is None


def test_an_unknown_event_kind_is_refused_not_stored(svc):
    r = svc.pc.record_call_event(svc.db, "c4", 1, "delete_everything")
    assert r["applied"] is False and "unknown" in r["error"]


def test_the_history_statement_ids_and_escalation_are_recorded(svc):
    pc = svc.pc
    pc.record_call_event(svc.db, "c5", 1, "history", {"statements": ["appointment:1", "cannot_see"]})
    pc.record_call_event(svc.db, "c5", 2, "history", {"statements": ["appointment:1"]})  # not doubled
    pc.record_call_event(svc.db, "c5", 3, "escalation", {"reason": "caller_requested"})
    rec = pc.get_call_record(svc.db, "c5")
    assert (
        rec["history_statements"] == ["appointment:1", "cannot_see"] and rec["escalation_reason"] == "caller_requested"
    )


# ============================================================ KCD-494 identify, never assume


def test_a_new_number_is_a_new_caller(svc):
    assert svc.pc.identify(svc.db, "9111111111") == {"status": "new", "record_exists": False, "count": 0}
    assert svc.pc.identify(svc.db, svc.bs.NOT_PROVIDED_PHONE)["status"] == "new"


def test_one_record_is_a_single_and_the_answer_carries_no_name(svc):
    p = _patient(svc, "Asha Das", "9000000001", 70)
    r = svc.pc.identify(svc.db, "9000000001")
    assert r["status"] == "single" and r["record_exists"] and r["patient_ref"] == p.id
    assert "Asha" not in str(r) and "70" not in str(r)


def test_a_shared_number_is_ambiguous_and_the_count_is_all_that_is_said(svc):
    _patient(svc, "Asha Das", "9000000002", 70)
    _patient(svc, "Ravi Das", "9000000002", 45)
    r = svc.pc.identify(svc.db, "9000000002")
    assert r == {"status": "ambiguous", "record_exists": True, "count": 2}


def test_a_name_settles_a_shared_number_and_an_age_narrows_but_never_picks(svc):
    _patient(svc, "Asha Das", "9000000003", 70)
    _patient(svc, "Ravi Das", "9000000003", 45)
    asha = svc.db.query(svc.m.Patient).filter_by(name="Asha Das").one()
    assert svc.pc.resolve_named(svc.db, "9000000003", "Asha") == {
        "status": "single",
        "patient_ref": asha.id,
        "basis": "exact",
    }
    assert svc.pc.resolve_named(svc.db, "9000000003", "Das")["status"] == "ambiguous"  # fits both
    assert svc.pc.resolve_named(svc.db, "9000000003", "Das", age=45)["status"] == "single"  # age narrows
    assert (
        svc.pc.resolve_named(svc.db, "9000000003", "Zoya")["status"] == "none"
    )  # sounds like "Asha" once vowels fold away: NOT a match


def test_two_patients_with_the_same_name_and_age_stay_ambiguous(svc):
    _patient(svc, "Ravi Das", "9000000004", 45)
    p2 = svc.m.Patient(name="Ravi Das", phone="9000000004", age=45, created_at=PINNED_NOW)
    svc.db.add(p2)
    svc.db.commit()
    assert svc.pc.resolve_named(svc.db, "9000000004", "Ravi Das", age=45)["status"] == "ambiguous"


# ===================================================================== KCD-493/495 timeline


def test_the_timeline_joins_bookings_tests_and_contacts_with_source_and_as_of(svc):
    p = _patient(svc, "Asha Das", "9000000010", 70)
    conf, date, slot, doc = _book_appointment(svc, p)
    _book_test(svc, p)
    svc.pc.record_call_event(svc.db, "cX", 1, "start", {}, "9000000010")
    svc.pc.record_call_event(svc.db, "cX", 2, "patient", {"patient_ref": p.id})
    t = svc.pc.timeline(svc.db, p.id, "9000000010", "cur")
    assert t["success"] and t["as_of"]
    kinds = {e["kind"] for e in t["events"]}
    assert {"appointment", "test_booking", "contact"} <= kinds
    for e in t["events"]:
        assert e["source"] and e["as_of"] and e["id"]  # provenance on EVERY event
    appt = next(e for e in t["events"] if e["kind"] == "appointment")
    assert appt["fields"]["confirmation_id"] == conf and appt["fields"]["doctor_name"] == doc.name


def test_the_timeline_names_the_sources_it_could_not_read(svc):
    p = _patient(svc, "Asha Das", "9000000011", 70)
    t = svc.pc.timeline(svc.db, p.id, "9000000011", "cur")
    assert t["sources"] == {"local": "ok", "his": "not_connected", "lis": "not_connected"}


def test_every_read_is_audited_and_attributed_to_the_call(svc):
    p = _patient(svc, "Asha Das", "9000000012", 70)
    _book_appointment(svc, p)
    svc.pc.timeline(svc.db, p.id, "9000000012", "call-77")
    rows = svc.db.query(svc.m.HistoryAudit).all()
    assert (
        len(rows) == 1
        and rows[0].call_id == "call-77"
        and rows[0].kind == "timeline_read"
        and rows[0].patient_id == p.id
    )
    assert "appointment" in rows[0].fields_json


def test_a_stranger_gets_nothing_and_the_attempt_is_audited(svc):
    p = _patient(svc, "Asha Das", "9000000013", 70)
    _book_appointment(svc, p)
    r = svc.pc.timeline(svc.db, p.id, "9999999999", "call-bad")
    assert r == {"success": False, "reason": "not_authorized"}
    rows = svc.db.query(svc.m.HistoryAudit).all()
    assert [(x.kind, x.call_id) for x in rows] == [("timeline_denied", "call-bad")]


def test_a_verified_proxy_can_read_and_an_unverified_one_cannot(svc):
    p = _patient(svc, "Asha Das", "9000000014", 70)
    svc.bs.record_proxy(svc.db, p, "9222222222", "son", "relationship_stated")
    svc.db.commit()
    assert svc.pc.timeline(svc.db, p.id, "9222222222", "c")["success"] is False  # stated is not enough to READ
    svc.bs.record_proxy(svc.db, p, "9333333333", "daughter", "dob_confirmed")
    svc.db.commit()
    assert svc.pc.timeline(svc.db, p.id, "9333333333", "c")["success"] is True


def test_no_event_may_carry_a_clinical_result(svc, monkeypatch):
    p = _patient(svc, "Asha Das", "9000000015", 70)
    bad = [
        {
            "id": "test_performed:1",
            "kind": "test_performed",
            "at": "2030-01-01",
            "source": "lis",
            "as_of": "x",
            "fields": {"test_name": "CBC", "Result": "Hb 9.1"},
        }
    ]
    monkeypatch.setattr(svc.pc, "_local_events", lambda db, patient, now: bad)
    with pytest.raises(ValueError, match="result"):
        svc.pc.timeline(svc.db, p.id, "9000000015", "c")


def test_the_tables_have_no_column_that_could_hold_a_result(svc):
    for table in (svc.m.TestPerformance, svc.m.LabReport, svc.m.RetestInterval):
        cols = {c.name.lower() for c in table.__table__.columns}
        assert not (cols & svc.pc.FORBIDDEN_KEYS), (table.__tablename__, cols & svc.pc.FORBIDDEN_KEYS)


def test_a_missing_patient_is_not_found(svc):
    assert svc.pc.timeline(svc.db, 99999, "9000000016", "c") == {"success": False, "reason": "not_found"}


def test_staleness_is_reported(svc):
    old = (PINNED_NOW - datetime.timedelta(hours=svc.pc.STALE_AFTER_HOURS + 1)).isoformat()
    assert svc.pc.is_stale(old) and svc.pc.is_stale(None) and not svc.pc.is_stale(PINNED_NOW.isoformat())


# ============================================================ KCD-498 test history, no advice


def _perform(svc, patient, test_name, on):
    test = svc.db.query(svc.m.LabTest).filter(svc.m.LabTest.name.ilike(f"%{test_name}%")).first()
    svc.db.add(
        svc.m.TestPerformance(
            patient_id=patient.id, lab_test_id=test.id, performed_on=on, source="lis", created_at=PINNED_NOW
        )
    )
    svc.db.commit()
    return test


def test_an_unknown_test_history_says_unknown(svc):
    p = _patient(svc, "Asha Das", "9000000020", 70)
    test = svc.db.query(svc.m.LabTest).first()
    assert svc.pc.test_status(svc.db, p.id, test.id, TODAY) == {
        "known": False,
        "last_performed_on": None,
        "due": None,
        "interval_days": None,
    }


def test_a_last_date_without_an_approved_interval_says_nothing_about_need(svc):
    p = _patient(svc, "Asha Das", "9000000021", 70)
    test = _perform(svc, p, "CBC", "2029-06-01")
    s = svc.pc.test_status(svc.db, p.id, test.id, TODAY)
    assert s["known"] and s["last_performed_on"] == "2029-06-01" and s["due"] is None and s["interval_days"] is None


def test_due_is_computed_only_from_a_clinician_approved_interval(svc):
    p = _patient(svc, "Asha Das", "9000000022", 70)
    test = _perform(svc, p, "CBC", "2029-06-01")
    svc.db.add(
        svc.m.RetestInterval(lab_test_id=test.id, interval_days=90, approved_by="Dr Sen", approved_at="2029-01-01")
    )
    svc.db.commit()
    s = svc.pc.test_status(svc.db, p.id, test.id, TODAY)
    assert s["due"] is True and s["approved_by"] == "Dr Sen" and s["due_on"] == "2029-08-30"
    recent = svc.pc.test_status(svc.db, p.id, test.id, datetime.date(2029, 7, 1))
    assert recent["due"] is False


def test_the_latest_performance_wins(svc):
    p = _patient(svc, "Asha Das", "9000000023", 70)
    test = _perform(svc, p, "CBC", "2028-01-01")
    _perform(svc, p, "CBC", "2029-09-09")
    assert svc.pc.test_status(svc.db, p.id, test.id, TODAY)["last_performed_on"] == "2029-09-09"


# ================================================================== KCD-499 preferences


def test_preferences_are_stored_against_the_patient_and_read_back(svc):
    p = _patient(svc, "Asha Das", "9000000030", 70)
    r = svc.pc.set_preferences(
        svc.db, p.id, "9000000030", delivery_channel="sms", language_bias="bn", accessibility_mode="slower"
    )
    assert r == {"success": True, "stored": ["accessibility_mode", "delivery_channel", "language_bias"]}
    got = svc.pc.get_preferences(svc.db, p.id, "9000000030")
    assert got["preferences"] == {"delivery_channel": "sms", "accessibility_mode": "slower", "language_bias": "bn"}


def test_a_preference_follows_the_patient_not_the_handset(svc):
    p = _patient(svc, "Asha Das", "9000000031", 70)
    svc.bs.record_proxy(svc.db, p, "9444444444", "son", "dob_confirmed")
    svc.db.commit()
    svc.pc.set_preferences(svc.db, p.id, "9000000031", branch="Salt Lake")
    assert svc.pc.get_preferences(svc.db, p.id, "9444444444")["preferences"]["branch"] == "Salt Lake"
    assert svc.pc.get_preferences(svc.db, p.id, "9555555555")["success"] is False


def test_invalid_or_unknown_preferences_are_refused(svc):
    p = _patient(svc, "Asha Das", "9000000032", 70)
    assert (
        svc.pc.set_preferences(svc.db, p.id, "9000000032", delivery_channel="carrier_pigeon")["reason"]
        == "invalid_value:delivery_channel"
    )
    assert (
        svc.pc.set_preferences(svc.db, p.id, "9000000032", language_bias="fr")["reason"]
        == "invalid_value:language_bias"
    )
    assert svc.pc.set_preferences(svc.db, p.id, "9000000032", blood_group="O+")["reason"] == "unknown_field:blood_group"
    assert (
        svc.pc.set_preferences(svc.db, p.id, "9000000032", collection_address="x" * 500)["reason"]
        == "too_long:collection_address"
    )
    assert svc.pc.set_preferences(svc.db, p.id, "9666666666", branch="x")["reason"] == "not_authorized"


# ================================================================== KCD-496 continuity


def test_the_last_three_interactions_across_voice_and_messaging(svc):
    pc = svc.pc
    for i, (cid, outcome) in enumerate(
        [("a", "completed"), ("b", "abandoned"), ("c", "completed"), ("d", "handed_off")]
    ):
        pc.record_call_event(svc.db, cid, 1, "start", {}, "9000000040")
        pc.record_call_event(svc.db, cid, 2, "intent", {"intent": f"intent_{cid}"})
        pc.record_call_event(svc.db, cid, 3, "end", {"outcome": outcome})
        row = svc.db.query(svc.m.CallRecord).filter_by(call_id=cid).one()
        row.started_at = PINNED_NOW - datetime.timedelta(days=10 - i)
        svc.db.commit()
    svc.db.add(
        svc.m.SmsOutbox(
            to_phone="9000000040",
            template_key="booking_confirmed",
            message="m",
            status="queued",
            created_at=PINNED_NOW - datetime.timedelta(days=1),
        )
    )
    svc.db.commit()
    got = pc.recent_interactions(svc.db, "9000000040", limit=3)
    assert len(got) == 3
    assert got[0]["channel"] == "sms" and got[0]["intent"] == "booking_confirmed"  # the most recent, across channels
    assert [g["intent"] for g in got[1:]] == ["intent_d", "intent_c"]
    assert all({"at", "channel", "intent", "outcome"} <= set(g) for g in got)


def test_the_current_call_is_not_its_own_history(svc):
    pc = svc.pc
    pc.record_call_event(svc.db, "now", 1, "start", {}, "9000000041")
    assert pc.recent_interactions(svc.db, "9000000041", exclude_call_id="now") == []


def test_an_unfinished_booking_is_reported_and_expires(svc):
    svc.bs.save_draft(svc.db, "9000000042", "prev-call", '{"doctor_name": "Dr Sen"}')
    c = svc.pc.continuity(svc.db, "9000000042")
    assert c["unfinished"] and c["draft"]["call_id"] == "prev-call"
    svc.bs.clear_draft(svc.db, "9000000042")
    assert svc.pc.continuity(svc.db, "9000000042")["unfinished"] is False


# ====================================================== KCD-497 a booking without a reference number


def test_a_booking_is_found_by_name_and_approximate_date(svc):
    p = _patient(svc, "Asha Das", "9000000050", 70)
    conf, date, slot, doc = _book_appointment(svc, p, days_ahead=10)
    near = (datetime.date.fromisoformat(date) + datetime.timedelta(days=2)).isoformat()
    r = svc.pc.search_bookings(svc.db, "9000000050", name="Asha", approx_date=near)
    assert r["status"] == "single" and r["matches"][0]["confirmation_id"] == conf


def test_a_test_booking_is_found_by_test_name(svc):
    p = _patient(svc, "Asha Das", "9000000051", 70)
    conf, date, test = _book_test(svc, p, "CBC")
    r = svc.pc.search_bookings(svc.db, "9000000051", phone="9000000051", test_name="CBC")
    assert (
        r["status"] == "single"
        and r["matches"][0]["confirmation_id"] == conf
        and r["matches"][0]["test_name"] == test.name
    )


def test_several_matches_are_never_collapsed_to_the_likeliest(svc):
    p = _patient(svc, "Asha Das", "9000000052", 70)
    c1, d1, _, _ = _book_appointment(svc, p, days_ahead=10)
    c2, d2, _, _ = _book_appointment(svc, p, days_ahead=17)
    r = svc.pc.search_bookings(svc.db, "9000000052", phone="9000000052")
    assert r["status"] == "several" and {m["confirmation_id"] for m in r["matches"]} == {c1, c2}
    assert "date" in r["distinguishing"]  # what the agent can ask about


def test_a_stranger_finds_nothing_and_learns_nothing(svc):
    p = _patient(svc, "Asha Das", "9000000053", 70)
    _book_appointment(svc, p)
    r = svc.pc.search_bookings(svc.db, "9888888888", phone="9000000053")
    assert r["status"] == "none" and r["matches"] == []


def test_the_date_window_is_a_window_not_a_guess(svc):
    p = _patient(svc, "Asha Das", "9000000054", 70)
    _, date, _, _ = _book_appointment(svc, p, days_ahead=10)
    far = (datetime.date.fromisoformat(date) + datetime.timedelta(days=30)).isoformat()
    assert svc.pc.search_bookings(svc.db, "9000000054", phone="9000000054", approx_date=far)["status"] == "none"


def test_no_criteria_is_insufficient_and_branch_is_reported_as_ignored(svc):
    assert svc.pc.search_bookings(svc.db, "9000000055")["status"] == "insufficient"
    r = svc.pc.search_bookings(svc.db, "9000000055", phone="9000000055", branch="Salt Lake")
    assert r["ignored_filters"] == ["branch"]


def test_a_multi_test_booking_is_one_booking(svc):
    p = _patient(svc, "Asha Das", "9000000056", 70)
    tests = svc.db.query(svc.m.LabTest).limit(2).all()
    date = (TODAY + datetime.timedelta(days=3)).isoformat()
    res = svc.bs.book_tests(svc.db, [t.id for t in tests], date, p.name, p.phone, p.phone, p.age)
    r = svc.pc.search_bookings(svc.db, "9000000056", phone="9000000056")
    assert r["status"] == "single" and r["matches"][0]["confirmation_id"] == res["confirmation_id"]


def test_phonetic_matching_is_a_suggestion_that_must_be_confirmed(svc):
    _patient(svc, "Chatterjee Ravi", "9000000060", 60)
    assert svc.pc.resolve_named(svc.db, "9000000060", "Chatterjee")["basis"] == "exact"
    sound = svc.pc.resolve_named(svc.db, "9000000060", "Chaterji")  # mis-spelt / mis-heard
    assert sound["status"] == "single" and sound["basis"] == "phonetic"  # found, but flagged


def test_short_names_whose_phonetic_keys_collapse_never_match_by_sound(svc):
    _patient(svc, "Asha Das", "9000000061", 70)
    assert svc.pc.resolve_named(svc.db, "9000000061", "Saha")["status"] == "none"
    assert svc.pc.resolve_named(svc.db, "9000000061", "Zoya")["status"] == "none"


def test_a_search_that_matched_only_by_sound_asks_for_confirmation(svc):
    p = _patient(svc, "Chatterjee Ravi", "9000000062", 60)
    _book_appointment(svc, p)
    r = svc.pc.search_bookings(svc.db, "9000000062", phone="9000000062", name="Chaterji")
    assert r["status"] == "single" and r["needs_confirmation"] is True
    exact = svc.pc.search_bookings(svc.db, "9000000062", phone="9000000062", name="Chatterjee")
    assert exact["needs_confirmation"] is False
