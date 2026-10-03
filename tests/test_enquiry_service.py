"""Unit tests for clinic-api/enquiry_service.py (Epic E27) against a real
seeded database, same fixture pattern as test_booking_service.py.

    python -m pytest tests/test_enquiry_service.py -v
"""

import datetime
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


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
        "booking_service",
        "booking_migrate",
        "enquiry_service",
        "enquiry_migrate",
        "i18n_content",
        "phonetic_match",
    ):
        sys.modules.pop(mod, None)

    import seed as seed_mod

    seed_mod.seed()
    import booking_migrate

    booking_migrate.migrate_booking_schema()
    booking_migrate.finish_booking_schema_setup()
    import enquiry_migrate

    enquiry_migrate.add_enquiry_columns()
    enquiry_migrate.backfill_enquiry_facts()
    enquiry_migrate.seed_enquiry_demo_data()
    import db as db_mod
    import enquiry_service as es
    import models as m

    yield es, db_mod, m

    try:
        os.remove(db_path)
    except OSError:
        pass


def test_doctor_leave_reports_the_return_date(clinic_modules):
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = db.query(m.Doctor).first()
        leave = db.query(m.DoctorLeave).filter_by(doctor_id=doc.id).one()
        result = es.doctor_leave_on(db, doc.id, leave.start_date)
        assert result["on_leave"] is True and result["return_date"] == leave.return_date
        # A day well outside the leave window is not reported as leave.
        far_future = (datetime.date.today() + datetime.timedelta(days=365)).isoformat()
        assert es.doctor_leave_on(db, doc.id, far_future) is None
    finally:
        db.close()


def test_merge_prep_instructions_takes_the_strictest_fasting_window(clinic_modules):
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        sugar = db.query(m.LabTest).filter_by(name="Blood Sugar Fasting").one()  # 8h
        lipid = db.query(m.LabTest).filter_by(name="Lipid Profile").one()  # 11h
        result = es.merge_prep_instructions(db, [sugar.id, lipid.id])
        assert result["found"] and result["fasting_hours"] == 11  # the strictest of the two
        assert result["escalate_to_human"] is False

        # A test with no fasting requirement at all does not lower the merge.
        cbc = db.query(m.LabTest).filter_by(name="Complete Blood Count (CBC)").one()
        result2 = es.merge_prep_instructions(db, [sugar.id, lipid.id, cbc.id])
        assert result2["fasting_hours"] == 11
    finally:
        db.close()


def test_merge_prep_instructions_unknown_test_ids_are_skipped_not_guessed(clinic_modules):
    es, db_mod, _m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        assert es.merge_prep_instructions(db, [999999]) == {"found": False}
    finally:
        db.close()


def test_merge_prep_instructions_escalates_a_test_with_no_structured_fasting_hours(clinic_modules):
    """KCD-382, story "Caller has several tests with conflicting preparation", AC3 + the partial-data
    safety requirement: "USG Whole Abdomen" has `fasting_required=True` and real prep prose (seed.py's
    PREP_OVERRIDES) but no structured `fasting_hours` (it is absent from enquiry_migrate's
    FASTING_HOURS_OVERRIDES). The merge must never silently drop it from the MAX() and answer as if no
    fasting were needed -- it must escalate instead, naming the test that could not be safely folded in."""
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        usg = db.query(m.LabTest).filter_by(name="USG Whole Abdomen").one()
        assert usg.fasting_required is True and usg.fasting_hours is None  # the data gap this test guards

        result = es.merge_prep_instructions(db, [usg.id])
        assert result["escalate_to_human"] is True
        assert result["incomplete_data"] == ["USG Whole Abdomen"]

        # Paired with a test that DOES have a structured number: the known number must not be thrown away,
        # but the merge as a whole still escalates rather than quietly answering with only part of the facts.
        sugar = db.query(m.LabTest).filter_by(name="Blood Sugar Fasting").one()  # 8h
        combined = es.merge_prep_instructions(db, [usg.id, sugar.id])
        assert combined["escalate_to_human"] is True
        assert combined["incomplete_data"] == ["USG Whole Abdomen"]
        assert combined["fasting_hours"] == 8  # the one structured value present is preserved, not discarded

        # A fully-structured pair stays exactly as before -- the new check is additive, never a regression.
        lipid = db.query(m.LabTest).filter_by(name="Lipid Profile").one()  # 11h
        clean = es.merge_prep_instructions(db, [sugar.id, lipid.id])
        assert clean["escalate_to_human"] is False
        assert "incomplete_data" not in clean
    finally:
        db.close()


def test_package_comparison_computes_real_savings(clinic_modules):
    es, db_mod, _m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        result = es.compare_package_vs_separate(db, "Full Body Checkup Basic")
        assert result["found"]
        assert result["separate_total_inr"] > result["bundled_price_inr"]
        assert result["savings_inr"] == result["separate_total_inr"] - result["bundled_price_inr"]
        assert len(result["test_names"]) == 5

        assert es.compare_package_vs_separate(db, "Nonexistent Package") == {
            "found": False,
            "query": "Nonexistent Package",
        }
    finally:
        db.close()


def test_walk_in_policy_by_department_and_by_test(clinic_modules):
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        gen_med = db.query(m.Department).filter_by(name="General Medicine").one()
        result = es.walk_in_policy(db, department_id=gen_med.id)
        assert result["found"] and result["allowed"] is True

        uric = db.query(m.LabTest).filter_by(name="Uric Acid").one()
        result2 = es.walk_in_policy(db, lab_test_id=uric.id)
        assert result2["found"]

        derm = db.query(m.Department).filter_by(name="Dermatology").one()
        assert es.walk_in_policy(db, department_id=derm.id) == {"found": False}
    finally:
        db.close()


def test_home_collection_eligibility_by_postal_code_and_sample_type(clinic_modules):
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        uric = db.query(m.LabTest).filter_by(name="Uric Acid").one()
        covered = es.home_collection_eligibility(db, uric.id, "700091")
        assert covered["eligible"] is True and covered["charge_inr"] == 100

        not_covered = es.home_collection_eligibility(db, uric.id, "700001")
        assert not_covered["eligible"] is False and not_covered["reason"] == "area_not_covered"

        ecg = db.query(m.LabTest).filter_by(name="ECG").one()  # ineligible sample type
        ineligible = es.home_collection_eligibility(db, ecg.id, "700091")
        assert ineligible["eligible"] is False and ineligible["reason"] == "sample_type"
    finally:
        db.close()


def test_insurance_coverage_active_policy_and_unknown_policy(clinic_modules):
    es, db_mod, _m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        covered = es.check_insurance_coverage(db, "DEMO-POLICY-001")
        assert covered["found"] and covered["active"] and covered["covered"]
        assert covered["coverage_percent"] == 80

        unknown = es.check_insurance_coverage(db, "NOT-A-REAL-POLICY")
        assert unknown["found"] is False
    finally:
        db.close()


def test_prescription_requirement_reads_the_policy_table_not_the_name(clinic_modules):
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        hiv = db.query(m.LabTest).filter_by(name="HIV Test (ELISA)").one()
        result = es.prescription_requirement(db, hiv.id, "en")
        assert result["required"] is True and result["note"]

        cbc = db.query(m.LabTest).filter_by(name="Complete Blood Count (CBC)").one()
        result2 = es.prescription_requirement(db, cbc.id)
        assert result2["required"] is False
    finally:
        db.close()


def test_out_of_scope_question_is_recorded(clinic_modules):
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        result = es.record_out_of_scope(db, "call-123", "Do you treat pets?")
        assert result["recorded"]
        row = db.query(m.CallOutcome).filter_by(id=result["id"]).one()
        assert row.caller_question == "Do you treat pets?"
    finally:
        db.close()


def test_callback_request_is_recorded_never_claims_a_call_was_placed(clinic_modules):
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        result = es.request_callback(db, "9999999999", "call-1", "this evening 6-8pm", "billing question")
        assert result["success"]
        row = db.query(m.CallbackRequest).filter_by(id=result["id"]).one()
        assert row.status == "scheduled"  # never "completed" or "called"
    finally:
        db.close()


def test_report_status_and_otp_delivery_flow(clinic_modules):
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        report = m.LabReport(
            confirmation_id="KCD-TEST-0001",
            patient_phone="9888888888",
            status="pending",
            created_at=datetime.datetime.now(),
        )
        db.add(report)
        db.commit()

        assert es.report_status(db, "KCD-TEST-0001")["status"] == "pending"
        # Not ready yet -- OTP request must be refused, never a guessed status.
        assert es.request_report_otp(db, "KCD-TEST-0001", "9888888888") == {"success": False, "reason": "not_ready"}

        report.status = "ready"
        report.ready_at = datetime.datetime.now()
        db.commit()

        wrong_phone = es.request_report_otp(db, "KCD-TEST-0001", "0000000000")
        assert wrong_phone == {"success": False, "reason": "phone_mismatch"}

        otp_result = es.request_report_otp(db, "KCD-TEST-0001", "9888888888")
        assert otp_result["success"]
        real_otp = db.query(m.ReportDeliveryOTP).filter_by(report_id=report.id).one().otp_code

        bad = es.deliver_report(db, "KCD-TEST-0001", "000000")
        assert bad == {"success": False, "reason": "otp_invalid_or_expired"}

        good = es.deliver_report(db, "KCD-TEST-0001", real_otp)
        assert good["success"] and good["link_token"]

        audit = db.query(m.ReportDeliveryAudit).filter_by(report_id=report.id).one()
        assert audit.recipient_phone == "9888888888" and audit.verification_method == "otp"
    finally:
        db.close()


def test_department_hours_override(clinic_modules):
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        cardio = db.query(m.Department).filter_by(name="Cardiology").one()
        result = es.department_hours(db, cardio.id, "en")
        assert result and "10am" in result["hours"]

        derm = db.query(m.Department).filter_by(name="Dermatology").one()
        assert es.department_hours(db, derm.id) is None  # no override -- falls back to clinic-wide FAQ
    finally:
        db.close()


def test_a_call_summary_is_stored_with_the_callback_and_never_overwritten_by_a_duplicate(clinic_modules):
    es, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        result = es.request_callback(
            db,
            "9888888888",
            "call-2",
            "asap",
            "lay_term_blood",
            call_summary="reason: lay_term_blood | phone: 9888888888",
        )
        row = db.query(m.CallbackRequest).filter_by(id=result["id"]).one()
        assert row.call_summary == "reason: lay_term_blood | phone: 9888888888"

        # the SAME callback asked for again (duplicate) does not overwrite an existing summary with a different one
        again = es.request_callback(db, "9888888888", "call-2", "asap", "lay_term_blood", call_summary="different text")
        assert again["duplicate"] is True
        db.refresh(row)
        assert row.call_summary == "reason: lay_term_blood | phone: 9888888888"

        # a callback asked for with no summary at all still works exactly as before (optional, additive)
        bare = es.request_callback(db, "9777777777", "call-3", "asap")
        bare_row = db.query(m.CallbackRequest).filter_by(id=bare["id"]).one()
        assert bare_row.call_summary == ""
    finally:
        db.close()
