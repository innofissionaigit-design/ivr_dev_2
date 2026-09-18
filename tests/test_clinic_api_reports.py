"""ADDED BY SOURAV -- tests for the 4 new clinic-api endpoints backing the
"Lab Report Status & Secure Delivery" combined story, against the REAL
clinic-api FastAPI app, a REAL SQLite database, seeded with the REAL
clinic-api/seed.py Patient A-J data (not hand-typed dicts). Mirrors
tests/test_live_test_price_lookup.py's established "real backend, real
seed data, real HTTP round-trip" pattern -- the `real_clinic_api` fixture
below is duplicated from that file rather than imported, per this repo's
one-fixture-per-file convention.

Endpoints under test: GET /api/v1/reports/status, POST /api/v1/reports/
delivery/request, POST /api/v1/reports/otp/verify, GET /api/v1/reports/
link/{token} -- see clinic-api/main.py's own docstrings on each for the
RULE references. This file is the plan's Section 15 database requirement
plus the RULE/ATTACK matrix, run against the actual seeded rows rather
than smoke-tested ad hoc.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CLINIC_API_DIR = os.path.join(TESTS_DIR, "..", "clinic-api")


@pytest.fixture(scope="module")
def real_clinic_api(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("clinic_reports_live") / "clinic.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("db", "models", "seed", "main"):
        sys.modules.pop(mod, None)
    import db as clinic_db
    import models as clinic_models
    clinic_models.Base.metadata.create_all(clinic_db.engine)
    from seed import seed
    seed()

    import main as clinic_main
    from fastapi.testclient import TestClient
    client = TestClient(clinic_main.app)

    # ADDED BY SOURAV -- KCD-384: exposed so tests can monkeypatch
    # clinic_main.send_report_link_via_provider (the name main.py's own
    # `from report_delivery_config import ...` bound into ITS module
    # globals -- verify_report_otp() resolves that name against `main`'s
    # own namespace, so patching it there, not on report_delivery_config
    # itself, is what actually intercepts the call).
    yield types.SimpleNamespace(client=client, db=clinic_db, models=clinic_models, main=clinic_main)

    sys.path.remove(CLINIC_API_DIR)


def status(client, phone, test_name=None):
    params = {"phone": phone}
    if test_name:
        params["test_name"] = test_name
    return client.get("/api/v1/reports/status", params=params).json()


def request_delivery(client, phone, report_number):
    return client.post(
        "/api/v1/reports/delivery/request",
        json={"phone": phone, "report_number": report_number},
    ).json()


def verify_otp(client, phone, report_number, otp_code, simulate_delivery_failure=False):
    return client.post(
        "/api/v1/reports/otp/verify",
        json={"phone": phone, "report_number": report_number, "otp_code": otp_code,
              "simulate_delivery_failure": simulate_delivery_failure},
    ).json()


def link(client, token):
    return client.get(f"/api/v1/reports/link/{token}").json()


def seeded_otp_code(real_clinic_api, report_number, used=None):
    """ADDED BY SOURAV -- "otp will not be hardcoded". Every ReportOTP
    row's code is a real random value now (models.generate_otp_code()),
    never a fixed literal -- any test that needs "the seeded row's actual
    code" reads it back from the database here instead of assuming a
    value. `used` narrows to that row's `used` flag when a report has
    more than one row (e.g. Patient E's two rows on RPT-10008); omit it
    to just take the most recently created row overall."""
    with real_clinic_api.db.SessionLocal() as db:
        report = db.query(real_clinic_api.models.LabReport).filter_by(
            report_number=report_number).first()
        query = db.query(real_clinic_api.models.ReportOTP).filter_by(report_id=report.id)
        if used is not None:
            query = query.filter_by(used=used)
        otp_row = query.order_by(
            real_clinic_api.models.ReportOTP.created_at.desc(),
            real_clinic_api.models.ReportOTP.id.desc(),
        ).first()
        return otp_row.otp_code


def create_fresh_ready_report(real_clinic_api, phone, name, report_number,
                               test_name="Complete Blood Count (CBC)"):
    """ADDED BY SOURAV -- KCD-384: the delivery-trigger tests below need a
    READY, delivery_enabled report belonging to a patient nobody else in
    this file has already used (every seeded 9000000001-9000000011 phone
    and RPT-1000X/1001X report number is already consumed by another test
    in this class or TestOtpVerify/TestReportLink above -- reusing one
    would risk colliding with a concurrent OTP row or a delivery already
    recorded against it). Mirrors seed.py's own Patient/LabReport
    construction exactly, just against a brand-new phone/report_number
    pair instead of the seeded PATIENTS/REPORT_DATA tables."""
    with real_clinic_api.db.SessionLocal() as db:
        import datetime as _dt
        patient = real_clinic_api.models.Patient(name=name, phone=phone)
        db.add(patient)
        db.flush()
        test_row = db.query(real_clinic_api.models.LabTest).filter_by(
            name=test_name).first()
        now = _dt.datetime.now()
        report = real_clinic_api.models.LabReport(
            report_number=report_number,
            patient_id=patient.id,
            lab_test_id=test_row.id,
            collected_at=now - _dt.timedelta(hours=8),
            expected_ready_at=now - _dt.timedelta(hours=1),
            ready_at=now - _dt.timedelta(hours=1),
            status="READY",
            delivery_enabled=True,
        )
        db.add(report)
        db.commit()


# --------------------------------------------------------------------- #
# GET /api/v1/reports/status
# --------------------------------------------------------------------- #

class TestReportStatus:
    def test_unknown_phone_is_patient_not_found(self, real_clinic_api):
        r = status(real_clinic_api.client, "9000099999")
        assert r == {"patient_found": False}

    def test_known_patient_with_zero_reports_is_honest_not_found(self, real_clinic_api):
        # Patient G -- Priyanka Sengupta, exists, no reports at all.
        r = status(real_clinic_api.client, "9000000008")
        assert r["patient_found"] is True
        assert r["found"] is False
        assert r["reason"] == "NOT_FOUND"

    def test_not_ready_report(self, real_clinic_api):
        # Patient B -- Riya Das, Vitamin D, NOT_READY.
        r = status(real_clinic_api.client, "9000000002")
        assert r["found"] is True
        assert r["status"] == "NOT_READY"
        assert r["delivery_enabled"] is False

    def test_processing_report(self, real_clinic_api):
        # Patient C -- Rahul Ghosh, TSH, PROCESSING.
        r = status(real_clinic_api.client, "9000000003", test_name="TSH")
        assert r["found"] is True
        assert r["status"] == "PROCESSING"

    def test_cancelled_report(self, real_clinic_api):
        # Patient D -- Debashish Roy, Uric Acid, CANCELLED.
        r = status(real_clinic_api.client, "9000000007")
        assert r["found"] is True
        assert r["status"] == "CANCELLED"

    def test_ready_and_delivery_enabled(self, real_clinic_api):
        # Patient A -- Arjun Sen, CBC, READY, delivery_enabled.
        r = status(real_clinic_api.client, "9000000001")
        assert r["found"] is True
        assert r["status"] == "READY"
        assert r["delivery_enabled"] is True
        assert r["report_number"] == "RPT-10001"

    def test_ready_but_delivery_disabled(self, real_clinic_api):
        # Patient I -- Indrani Ghosh, KFT, READY, delivery_enabled=False.
        r = status(real_clinic_api.client, "9000000011")
        assert r["found"] is True
        assert r["status"] == "READY"
        assert r["delivery_enabled"] is False

    def test_multiple_reports_without_test_name_is_ambiguous(self, real_clinic_api):
        # Patient F -- Mita Roy, 3 reports (RULE 13).
        r = status(real_clinic_api.client, "9000000004")
        assert r["found"] is False
        assert r["reason"] == "AMBIGUOUS"
        assert {c["report_number"] for c in r["candidates"]} == {
            "RPT-10004", "RPT-10005", "RPT-10010",
        }

    def test_test_name_filter_narrows_to_a_single_match(self, real_clinic_api):
        # Same Patient F, but naming the test disambiguates without
        # needing the which_report round trip.
        r = status(real_clinic_api.client, "9000000004", test_name="TSH")
        assert r["found"] is True
        assert r["report_number"] == "RPT-10010"

    def test_test_name_filter_matching_nothing_this_patient_has_is_not_found(self, real_clinic_api):
        r = status(real_clinic_api.client, "9000000004", test_name="MRI Brain")
        assert r["found"] is False
        assert r["reason"] == "NOT_FOUND"

    def test_same_name_patients_are_never_merged(self, real_clinic_api):
        # RULE 14 -- both "Rahul Das" rows must resolve to only THEIR OWN
        # report, keyed by phone, never each other's.
        r1 = status(real_clinic_api.client, "9000000009")
        r2 = status(real_clinic_api.client, "9000000010")
        assert r1["report_number"] == "RPT-10011"
        assert r2["report_number"] == "RPT-10012"
        assert r1["report_number"] != r2["report_number"]


# --------------------------------------------------------------------- #
# POST /api/v1/reports/delivery/request
# --------------------------------------------------------------------- #

class TestDeliveryRequest:
    def test_unknown_phone_is_patient_not_found(self, real_clinic_api):
        r = request_delivery(real_clinic_api.client, "9000099999", "RPT-10001")
        assert r == {"success": False, "reason": "PATIENT_NOT_FOUND"}

    def test_nonexistent_report_number_is_not_found(self, real_clinic_api):
        r = request_delivery(real_clinic_api.client, "9000000001", "RPT-NOPE")
        assert r == {"success": False, "reason": "NOT_FOUND"}

    def test_cross_patient_report_number_is_not_found_not_a_distinct_reason(self, real_clinic_api):
        # RULE 15 / ATTACK: a real report_number belonging to a DIFFERENT
        # patient must return the exact same NOT_FOUND as a nonexistent
        # one -- never a distinguishable "wrong patient" signal.
        r = request_delivery(real_clinic_api.client, "9000000002", "RPT-10001")  # Riya, Arjun's report
        assert r == {"success": False, "reason": "NOT_FOUND"}

    @pytest.mark.parametrize("phone,report,expected_reason", [
        ("9000000002", "RPT-10002", "NOT_READY"),
        ("9000000003", "RPT-10003", "PROCESSING"),
        ("9000000007", "RPT-10009", "CANCELLED"),
        ("9000000011", "RPT-10013", "DELIVERY_DISABLED"),
    ])
    def test_ineligible_reports_are_blocked_with_the_true_reason(
        self, real_clinic_api, phone, report, expected_reason
    ):
        r = request_delivery(real_clinic_api.client, phone, report)
        assert r == {"success": False, "reason": expected_reason}

    def test_ready_and_enabled_succeeds_and_masks_the_phone(self, real_clinic_api):
        # Patient H1 -- fresh report with no pre-seeded OTP row, so this
        # exercises the FRESH_OTP_CODE minting path.
        r = request_delivery(real_clinic_api.client, "9000000009", "RPT-10011")
        assert r["success"] is True
        assert r["reason"] == "OTP_REQUIRED"
        assert r["masked_phone"] == "0009"
        assert "9000000009" not in str(r)

    def test_existing_valid_unused_otp_is_reused_not_replaced(self, real_clinic_api):
        # Patient A already carries a VALID, unused, unexpired OTP
        # (a real random code -- see models.generate_otp_code(), never a
        # fixed literal since "otp will not be hardcoded") -- requesting
        # delivery again must reuse it (RULE 8's other half: don't rotate
        # a perfectly good code away). Checked by reading the row back
        # from the DB rather than by verifying it (which would consume
        # it -- TestOtpVerify below still needs this exact seeded row
        # untouched).
        with real_clinic_api.db.SessionLocal() as db:
            report = db.query(real_clinic_api.models.LabReport).filter_by(
                report_number="RPT-10001").first()
            before = (
                db.query(real_clinic_api.models.ReportOTP)
                .filter_by(report_id=report.id).order_by(
                    real_clinic_api.models.ReportOTP.created_at.desc()).first()
            )
            before_id, before_code = before.id, before.otp_code

        req = request_delivery(real_clinic_api.client, "9000000001", "RPT-10001")
        assert req["success"] is True

        with real_clinic_api.db.SessionLocal() as db:
            report = db.query(real_clinic_api.models.LabReport).filter_by(
                report_number="RPT-10001").first()
            after = (
                db.query(real_clinic_api.models.ReportOTP)
                .filter_by(report_id=report.id).order_by(
                    real_clinic_api.models.ReportOTP.created_at.desc()).first()
            )
            assert after.id == before_id  # same row, not a freshly minted one
            assert after.otp_code == before_code  # code itself untouched, whatever it is


# --------------------------------------------------------------------- #
# POST /api/v1/reports/otp/verify
# --------------------------------------------------------------------- #

class TestOtpVerify:
    def test_correct_seeded_otp_delivers_with_a_real_signed_link(self, real_clinic_api):
        # Patient A's seeded VALID row's code is a real random value now
        # (see models.generate_otp_code()), never a fixed literal -- read
        # it back from the DB rather than assuming what it is.
        code = seeded_otp_code(real_clinic_api, "RPT-10001", used=False)

        r = verify_otp(real_clinic_api.client, "9000000001", "RPT-10001", code)
        assert r["success"] is True
        assert r["reason"] == "DELIVERY_SENT"
        assert r["masked_phone"] == "0001"
        assert r["signed_link_expires_minutes"] == 15

    def test_wrong_otp_is_invalid_and_never_reveals_the_right_one(self, real_clinic_api):
        # Fresh delivery request for Patient H2 so this test doesn't
        # collide with the max-attempts test below re-using the same row.
        request_delivery(real_clinic_api.client, "9000000010", "RPT-10012")
        real_code = seeded_otp_code(real_clinic_api, "RPT-10012", used=False)

        r = verify_otp(real_clinic_api.client, "9000000010", "RPT-10012", "000000")
        assert r == {"success": False, "reason": "OTP_INVALID"}
        # RULE 9: the real code -- whatever this run happened to generate
        # it as -- must never be echoed back, not just some old constant.
        assert real_code not in str(r)

    def test_no_otp_ever_requested_for_this_report(self, real_clinic_api):
        # Patient I's report is delivery-disabled so a request would be
        # blocked anyway -- verify called directly, with no prior
        # request, on an eligible-but-never-requested report instead:
        # Patient F's TSH report (RPT-10010) has no pre-seeded OTP and no
        # delivery request has been made against it in this test module.
        # The code below is deliberately arbitrary -- OTP_NOT_REQUESTED
        # fires because no row exists at all for this report, before any
        # code comparison happens, so its value is irrelevant here.
        r = verify_otp(real_clinic_api.client, "9000000004", "RPT-10010", "000000")
        assert r == {"success": False, "reason": "OTP_NOT_REQUESTED"}

    def test_expired_otp(self, real_clinic_api):
        # NOTE ON DESIGN, found while writing this test: verify_report_otp
        # only ever checks the SINGLE most-recently-created OTP row for a
        # (report, patient) -- by design, matching "a fresh request
        # supersedes the old one" (RULE 8). That means Patient E's
        # originally-seeded 615204 (EXPIRED) row can never actually be
        # reached by verify once 903217 (MAXED, created after it) exists
        # on the same report -- only ever the most recent row is "the
        # active OTP". So EXPIRED is exercised here the way it can
        # actually happen in reality: request delivery for an eligible
        # report to mint a real, fresh row, then let time pass past its
        # expiry (simulated by moving expires_at into the past directly,
        # rather than sleeping the test) and try to verify it.
        request_delivery(real_clinic_api.client, "9000000004", "RPT-10004")  # Mita Roy, CBC
        import datetime as _dt
        with real_clinic_api.db.SessionLocal() as db:
            report = db.query(real_clinic_api.models.LabReport).filter_by(
                report_number="RPT-10004").first()
            otp_row = (
                db.query(real_clinic_api.models.ReportOTP)
                .filter_by(report_id=report.id, used=False)
                .order_by(real_clinic_api.models.ReportOTP.created_at.desc(),
                          real_clinic_api.models.ReportOTP.id.desc())
                .first()
            )
            code = otp_row.otp_code
            otp_row.expires_at = _dt.datetime.now() - _dt.timedelta(minutes=1)
            db.commit()

        r = verify_otp(real_clinic_api.client, "9000000004", "RPT-10004", code)
        assert r == {"success": False, "reason": "OTP_EXPIRED"}

    def test_max_attempts_already_reached(self, real_clinic_api):
        # Patient E's SECOND row is the most recently created, so it is
        # the one verify_report_otp actually checks (most recent per
        # report+patient) -- already at attempt_count==max_attempts. The
        # MAX_ATTEMPTS check fires before any code comparison (see
        # verify_report_otp()'s own fixed-order docstring), so the actual
        # code value passed here doesn't matter -- an arbitrary one is
        # used deliberately, to prove that.
        r = verify_otp(real_clinic_api.client, "9000000006", "RPT-10008", "000000")
        assert r == {"success": False, "reason": "OTP_MAX_ATTEMPTS"}

    def test_already_used_otp_is_rejected_even_with_the_exact_right_code(self, real_clinic_api):
        # Patient J's row, already used (RULE 7: single-use, no
        # exceptions even for the correct value) -- this test's whole
        # point is proving that, so it genuinely needs the real code, not
        # an arbitrary one, even though verify_report_otp's `used` check
        # currently fires before comparing it either way.
        code = seeded_otp_code(real_clinic_api, "RPT-10006", used=True)
        r = verify_otp(real_clinic_api.client, "9000000005", "RPT-10006", code)
        assert r == {"success": False, "reason": "OTP_ALREADY_USED"}

    def test_cross_report_otp_reuse_is_rejected(self, real_clinic_api):
        # ATTACK: Arjun's own real OTP code (by this point in the class
        # already consumed by test_correct_seeded_otp_delivers_with_a_
        # real_signed_link above -- irrelevant here, since identity
        # resolution fails before any OTP validity check even runs) --
        # try it against a DIFFERENT report he does not even have (his
        # identity resolves, but this report_number belongs to nobody
        # named in his rows).
        code = seeded_otp_code(real_clinic_api, "RPT-10001")
        r = verify_otp(real_clinic_api.client, "9000000001", "RPT-10011", code)
        assert r == {"success": False, "reason": "NOT_FOUND"}  # not Arjun's report

    def test_cross_patient_otp_reuse_is_rejected(self, real_clinic_api):
        # ATTACK: Arjun's own real report/OTP code, but calling as a
        # different patient (Riya) -- her identity resolves but the
        # report is not hers, so this must be NOT_FOUND, never leak into
        # Arjun's OTP verification at all.
        code = seeded_otp_code(real_clinic_api, "RPT-10001")
        r = verify_otp(real_clinic_api.client, "9000000002", "RPT-10001", code)
        assert r == {"success": False, "reason": "NOT_FOUND"}

    def test_ineligible_report_is_re_checked_even_at_verify_time(self, real_clinic_api):
        # Defense in depth: even if somehow asked to verify against a
        # CANCELLED report, the blocking check re-runs here too, not only
        # at delivery/request time.
        r = verify_otp(real_clinic_api.client, "9000000007", "RPT-10009", "000000")
        assert r == {"success": False, "reason": "CANCELLED"}

    def test_simulated_delivery_failure_after_correct_otp(self, real_clinic_api):
        # RULE 17: OTP success must not be conflated with delivery
        # success. Patient H1's report already has a used OTP from an
        # earlier test in this class chain, so request a fresh one first.
        request_delivery(real_clinic_api.client, "9000000009", "RPT-10011")
        # The row is now either the original reused OTP or a freshly
        # minted one -- either way it is unused, so read the live code
        # back out of the DB rather than assuming which constant applies
        # (keeps this test correct regardless of reuse-vs-mint).
        with real_clinic_api.db.SessionLocal() as db:
            report = db.query(real_clinic_api.models.LabReport).filter_by(
                report_number="RPT-10011").first()
            otp_row = (
                db.query(real_clinic_api.models.ReportOTP)
                .filter_by(report_id=report.id, used=False)
                .order_by(real_clinic_api.models.ReportOTP.created_at.desc())
                .first()
            )
            code = otp_row.otp_code
        r = verify_otp(real_clinic_api.client, "9000000009", "RPT-10011", code,
                        simulate_delivery_failure=True)
        assert r == {"success": False, "reason": "DELIVERY_FAILED"}

    def test_max_attempts_lockout_then_fresh_delivery_request_recovers(self, real_clinic_api):
        # RULE 8's full cycle: lock out with 3 wrong guesses, confirm
        # OTP_MAX_ATTEMPTS, request delivery again, confirm the NEW row
        # actually works -- a caller is never permanently stuck.
        phone, report_number = "9000000004", "RPT-10010"  # Mita Roy's TSH, READY+enabled
        request_delivery(real_clinic_api.client, phone, report_number)
        real_code = seeded_otp_code(real_clinic_api, report_number, used=False)

        for _ in range(3):
            r = verify_otp(real_clinic_api.client, phone, report_number, "000000")
        assert r == {"success": False, "reason": "OTP_MAX_ATTEMPTS"}

        # Locked out -- even the real code (read back from the DB, never
        # a fixed literal now) no longer works against the maxed row.
        r = verify_otp(real_clinic_api.client, phone, report_number, real_code)
        assert r == {"success": False, "reason": "OTP_MAX_ATTEMPTS"}

        # Requesting delivery again must mint a genuinely NEW, usable row
        # -- read ITS real code back too, rather than assuming any value.
        req = request_delivery(real_clinic_api.client, phone, report_number)
        assert req["success"] is True
        new_code = seeded_otp_code(real_clinic_api, report_number, used=False)
        assert new_code != real_code  # genuinely a different, fresh row
        r = verify_otp(real_clinic_api.client, phone, report_number, new_code)
        assert r["success"] is True
        assert r["reason"] == "DELIVERY_SENT"


# --------------------------------------------------------------------- #
# GET /api/v1/reports/link/{token}
# --------------------------------------------------------------------- #

class TestReportLink:
    def test_valid_unexpired_link(self, real_clinic_api):
        r = link(real_clinic_api.client, "SIGNED-ARJUN-10001")
        assert r["valid"] is True
        assert r["report_number"] == "RPT-10001"

    def test_expired_link(self, real_clinic_api):
        # Patient J's previously-generated, now-expired link.
        r = link(real_clinic_api.client, "SIGNED-AMIT-10006")
        assert r == {"valid": False, "reason": "LINK_EXPIRED"}

    def test_random_unknown_token(self, real_clinic_api):
        r = link(real_clinic_api.client, "totally-random-token-xyz")
        assert r == {"valid": False, "reason": "LINK_INVALID"}

    def test_modified_token_one_char_flipped(self, real_clinic_api):
        r = link(real_clinic_api.client, "SIGNED-ARJUN-1000X")
        assert r == {"valid": False, "reason": "LINK_INVALID"}

    def test_a_freshly_generated_link_from_a_live_otp_verify_validates(self, real_clinic_api):
        # End-to-end: request -> verify -> the token that comes back out
        # of the DB actually validates.
        request_delivery(real_clinic_api.client, "9000000010", "RPT-10012")
        with real_clinic_api.db.SessionLocal() as db:
            report = db.query(real_clinic_api.models.LabReport).filter_by(
                report_number="RPT-10012").first()
            otp_row = (
                db.query(real_clinic_api.models.ReportOTP)
                .filter_by(report_id=report.id, used=False)
                .order_by(real_clinic_api.models.ReportOTP.created_at.desc())
                .first()
            )
            code = otp_row.otp_code if otp_row else "135790"
        verify_result = verify_otp(real_clinic_api.client, "9000000010", "RPT-10012", code)
        assert verify_result["success"] is True, verify_result

        with real_clinic_api.db.SessionLocal() as db:
            report = db.query(real_clinic_api.models.LabReport).filter_by(
                report_number="RPT-10012").first()
            delivery = (
                db.query(real_clinic_api.models.ReportDelivery)
                .filter_by(report_id=report.id, delivery_status="SENT")
                .order_by(real_clinic_api.models.ReportDelivery.created_at.desc())
                .first()
            )
            token = delivery.signed_link_token
        r = link(real_clinic_api.client, token)
        assert r == {"valid": True, "report_number": "RPT-10012"}


# --------------------------------------------------------------------- #
# KCD-384 -- signed-link delivery dispatch trigger
#
# ADDED BY SOURAV. verify_report_otp() in clinic-api/main.py now calls
# report_delivery_config.send_report_link_via_provider() unconditionally
# on every successful OTP verification (see that function's own
# surrounding comments). These tests cover the three outcomes that call
# site's try/except is actually designed for: a configured provider that
# succeeds, no provider configured at all (the honest default in this
# repo, since REPORT_DELIVERY_WEBHOOK_URL is unset in this test run), and
# a configured provider that raises -- in every case the OTP-verify HTTP
# response and the ReportDelivery row's delivery_status must come out
# identically, because RULE 17 (OTP success != delivery-transport
# success) means only audit_note is allowed to vary.
#
# Patches clinic_main.send_report_link_via_provider, not
# report_delivery_config.send_report_link_via_provider -- see the
# real_clinic_api fixture's own comment for why (verify_report_otp()
# resolves the name against main's module globals, bound there once at
# import time by `from report_delivery_config import ...`).
# --------------------------------------------------------------------- #

class TestKcd384ReportLinkDeliveryTrigger:
    def test_configured_provider_that_succeeds_is_called_with_the_real_link_and_recorded(
        self, real_clinic_api, monkeypatch,
    ):
        phone, report_number = "9000000091", "RPT-90001"
        create_fresh_ready_report(real_clinic_api, phone, "Test Patient KCD384-A", report_number)

        calls = []

        def fake_send(dest_phone, report_link_url):
            calls.append((dest_phone, report_link_url))
            return True

        monkeypatch.setattr(real_clinic_api.main, "send_report_link_via_provider", fake_send)

        request_delivery(real_clinic_api.client, phone, report_number)
        code = seeded_otp_code(real_clinic_api, report_number, used=False)
        r = verify_otp(real_clinic_api.client, phone, report_number, code)
        assert r["success"] is True
        assert r["reason"] == "DELIVERY_SENT"

        with real_clinic_api.db.SessionLocal() as db:
            report = db.query(real_clinic_api.models.LabReport).filter_by(
                report_number=report_number).first()
            delivery = (
                db.query(real_clinic_api.models.ReportDelivery)
                .filter_by(report_id=report.id, delivery_status="SENT")
                .order_by(real_clinic_api.models.ReportDelivery.created_at.desc())
                .first()
            )
            token = delivery.signed_link_token
            audit_note = delivery.audit_note

        # Called exactly once, with this patient's real phone and the
        # actual signed-link URL for the token this verify just minted --
        # not a placeholder, not some other report's token.
        assert len(calls) == 1
        assert calls[0] == (phone, real_clinic_api.main.build_report_link_url(token))
        assert "dispatched via configured delivery provider" in audit_note

    def test_no_provider_configured_still_succeeds_and_says_so_in_the_audit_trail(
        self, real_clinic_api,
    ):
        # No monkeypatch here -- exercises the REAL
        # report_delivery_config.send_report_link_via_provider() with
        # REPORT_DELIVERY_WEBHOOK_URL unset (this test run's actual
        # environment), i.e. the honest default a fresh clone of this
        # repo has out of the box.
        assert not os.environ.get("REPORT_DELIVERY_WEBHOOK_URL")

        phone, report_number = "9000000092", "RPT-90002"
        create_fresh_ready_report(real_clinic_api, phone, "Test Patient KCD384-B", report_number)

        request_delivery(real_clinic_api.client, phone, report_number)
        code = seeded_otp_code(real_clinic_api, report_number, used=False)
        r = verify_otp(real_clinic_api.client, phone, report_number, code)
        # RULE 17: no delivery provider configured must never surface as
        # a caller-facing failure -- the caller still hears success.
        assert r["success"] is True
        assert r["reason"] == "DELIVERY_SENT"

        with real_clinic_api.db.SessionLocal() as db:
            report = db.query(real_clinic_api.models.LabReport).filter_by(
                report_number=report_number).first()
            delivery = (
                db.query(real_clinic_api.models.ReportDelivery)
                .filter_by(report_id=report.id, delivery_status="SENT")
                .order_by(real_clinic_api.models.ReportDelivery.created_at.desc())
                .first()
            )
            assert delivery.signed_link_token  # still generated and stored
            assert "No delivery provider configured" in delivery.audit_note

    def test_provider_raising_never_blocks_the_success_response_or_the_delivery_status(
        self, real_clinic_api, monkeypatch,
    ):
        phone, report_number = "9000000093", "RPT-90003"
        create_fresh_ready_report(real_clinic_api, phone, "Test Patient KCD384-C", report_number)

        def fake_send_that_raises(dest_phone, report_link_url):
            raise RuntimeError("provider is down")

        monkeypatch.setattr(
            real_clinic_api.main, "send_report_link_via_provider", fake_send_that_raises,
        )

        request_delivery(real_clinic_api.client, phone, report_number)
        code = seeded_otp_code(real_clinic_api, report_number, used=False)
        r = verify_otp(real_clinic_api.client, phone, report_number, code)
        assert r["success"] is True
        assert r["reason"] == "DELIVERY_SENT"

        with real_clinic_api.db.SessionLocal() as db:
            report = db.query(real_clinic_api.models.LabReport).filter_by(
                report_number=report_number).first()
            delivery = (
                db.query(real_clinic_api.models.ReportDelivery)
                .filter_by(report_id=report.id)
                .order_by(real_clinic_api.models.ReportDelivery.created_at.desc())
                .first()
            )
            # The exception must not have rolled back or altered the
            # already-committed successful delivery row -- still SENT,
            # still has its token, only the audit_note reflects the
            # provider failure.
            assert delivery.delivery_status == "SENT"
            assert delivery.signed_link_token
            assert "No delivery provider configured (or the send attempt failed)" in delivery.audit_note


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
