"""In-place migration for Epic E27's new LabTest columns.

Same discipline as booking_migrate.py: every brand-new TABLE (LabReport,
DoctorLeave, Package, WalkInPolicy, BillingLineItem, HomeCollectionCoverage,
InsurancePolicy, CallOutcome, CallbackRequest, DepartmentHours, ...) needs
no migration code at all -- Base.metadata.create_all(engine), already
called at startup, creates a missing table and never touches an existing
one. Only LabTest gained new COLUMNS on an existing table, which SQLite
cannot add without an explicit ALTER TABLE, run before any ORM query
against LabTest -- same "SQLAlchemy selects every mapped column" reason
booking_migrate.py's own columns exist.
"""

from __future__ import annotations

from db import SessionLocal, engine
from sqlalchemy import inspect, text

LAB_TEST_ENQUIRY_COLUMNS: dict[str, str] = {
    "fasting_hours": "INTEGER",
    # CodeRabbit-flagged: TRUE/FALSE, not 1/0 -- PostgreSQL rejects an
    # integer literal as a BOOLEAN column default (SQLite accepts both,
    # so this only ever broke the DATABASE_URL=postgres path db.py's own
    # docstring promises is a one-line change, not the pilot's SQLite one).
    "water_allowed_while_fasting": "BOOLEAN NOT NULL DEFAULT TRUE",
    "home_collection_eligible": "BOOLEAN NOT NULL DEFAULT TRUE",
    "prescription_required": "BOOLEAN NOT NULL DEFAULT FALSE",
    "prescription_note_bn": "VARCHAR NOT NULL DEFAULT ''",
    "prescription_note_hi": "VARCHAR NOT NULL DEFAULT ''",
    "prescription_note_en": "VARCHAR NOT NULL DEFAULT ''",
}


def add_enquiry_columns() -> list[str]:
    added = []
    insp = inspect(engine)
    have = {c["name"] for c in insp.get_columns("lab_tests")}
    with engine.begin() as conn:
        for col, decl in LAB_TEST_ENQUIRY_COLUMNS.items():
            if col not in have:
                conn.execute(text(f"ALTER TABLE lab_tests ADD COLUMN {col} {decl}"))
                added.append(f"lab_tests.{col}")
    return added


# The lay-term / prescription-by-WhatsApp flow's call summary (models.CallbackRequest.call_summary): a NEW column on
# a table `create_all()` may already have created without it on a database that was up before this column existed --
# same "SQLAlchemy selects every mapped column" reason add_enquiry_columns() exists.
CALLBACK_REQUEST_COLUMNS: dict[str, str] = {"call_summary": "TEXT NOT NULL DEFAULT ''"}


def add_callback_request_columns() -> list[str]:
    added = []
    insp = inspect(engine)
    if "callback_requests" not in insp.get_table_names():
        return added  # create_all() has not made the table yet on a brand-new database; nothing to ALTER
    have = {c["name"] for c in insp.get_columns("callback_requests")}
    with engine.begin() as conn:
        for col, decl in CALLBACK_REQUEST_COLUMNS.items():
            if col not in have:
                conn.execute(text(f"ALTER TABLE callback_requests ADD COLUMN {col} {decl}"))
                added.append(f"callback_requests.{col}")
    return added


# Per-test structured prep facts, keyed by the exact LAB_TESTS name from
# seed.py. Only tests with a genuine requirement get an override; every
# other test keeps the ALTER TABLE defaults (no fasting, home-collection
# eligible, no prescription needed) -- same "only list the exceptions"
# convention seed.py's own PREP_OVERRIDES already uses.
FASTING_HOURS_OVERRIDES: dict[str, tuple[int, bool]] = {  # name -> (hours, water_allowed)
    "Blood Sugar Fasting": (8, True),
    "Lipid Profile": (11, True),
    "Kidney Function Test (KFT)": (7, True),
    "Liver Function Test (LFT)": (8, True),
}

# Samples that physically cannot survive a courier trip, or need
# same-visit equipment -- ineligible for home collection regardless of
# the general default.
HOME_COLLECTION_INELIGIBLE = {
    "ECG",
    "2D Echocardiography",
    "TMT (Treadmill Test)",
    "USG Whole Abdomen",
    "USG Pregnancy Profile",
    "Chest X-Ray (PA view)",
    "Pap Smear",
}

# Tests a walk-in caller cannot simply request without a doctor's note.
PRESCRIPTION_REQUIRED = {
    "HIV Test (ELISA)",
    "HBsAg",
    "HCV",
}
_PRESCRIPTION_NOTE = (
    "এই টেস্টের জন্য ডাক্তারের প্রেসক্রিপশন লাগবে। প্রেসক্রিপশনের ছবি হোয়াটসঅ্যাপে পাঠাতে পারেন।",
    "इस टेस्ट के लिए डॉक्टर का पर्चा चाहिए। पर्चे की फोटो व्हाट्सऐप पर भेज सकते हैं।",
    "A doctor's prescription is required for this test. You can send a photo of it over WhatsApp.",
)


def seed_enquiry_demo_data(db=None) -> dict:
    """Representative rows for the NEW E27 tables that need departments/
    doctors/tests to already exist -- same "fictional but clinically
    plausible" convention as seed.py's own data, and same idempotence
    rule: only inserts what is not already there, keyed on a natural
    unique field per table, so a clinician's own added rows are never
    touched or duplicated on a later boot."""
    from models import (
        Department,
        DepartmentHours,
        Doctor,
        DoctorLeave,
        HomeCollectionCoverage,
        InsuranceCoverageRule,
        InsurancePolicy,
        LabTest,
        Package,
        PackageTest,
        WalkInPolicy,
    )

    own = db is None
    db = db or SessionLocal()
    added = 0
    try:
        dept_by_name = {d.name: d for d in db.query(Department).all()}
        test_by_name = {t.name: t for t in db.query(LabTest).all()}
        doctor = db.query(Doctor).first()

        if dept_by_name and not db.query(Package).filter_by(name="Full Body Checkup Basic").first():
            pkg = Package(
                name="Full Body Checkup Basic",
                name_bn="ফুল বডি চেকআপ বেসিক",
                name_hi="फुल बॉडी चेकअप बेसिक",
                bundled_price_inr=1800,
            )
            db.add(pkg)
            db.flush()
            for test_name in (
                "Complete Blood Count (CBC)",
                "Blood Sugar Fasting",
                "Lipid Profile",
                "Liver Function Test (LFT)",
                "Kidney Function Test (KFT)",
            ):
                t = test_by_name.get(test_name)
                if t:
                    db.add(PackageTest(package_id=pkg.id, lab_test_id=t.id))
            added += 1

        if (
            "General Medicine" in dept_by_name
            and not db.query(WalkInPolicy)
            .filter_by(department_id=dept_by_name["General Medicine"].id, lab_test_id=None)
            .first()
        ):
            db.add(
                WalkInPolicy(
                    department_id=dept_by_name["General Medicine"].id,
                    lab_test_id=None,
                    allowed=True,
                    queue_note_bn="সাধারণত ২০-৩০ মিনিট অপেক্ষা করতে হতে পারে।",
                    queue_note_hi="आमतौर पर 20-30 मिनट इंतज़ार करना पड़ सकता है।",
                    queue_note_en="A wait of around 20-30 minutes is typical.",
                )
            )
            added += 1
        uric = test_by_name.get("Uric Acid")
        if uric and not db.query(WalkInPolicy).filter_by(lab_test_id=uric.id).first():
            db.add(
                WalkInPolicy(
                    department_id=None,
                    lab_test_id=uric.id,
                    allowed=True,
                    queue_note_bn="",
                    queue_note_hi="",
                    queue_note_en="",
                )
            )
            added += 1

        if doctor and not db.query(HomeCollectionCoverage).filter_by(postal_code="700091").first():
            db.add(
                HomeCollectionCoverage(
                    postal_code="700091",
                    serviceable=True,
                    charge_inr=100,
                    slot_note_bn="সকাল ৭টা থেকে ১০টার মধ্যে",
                    slot_note_hi="सुबह 7 से 10 बजे के बीच",
                    slot_note_en="Between 7 and 10 in the morning",
                )
            )
            db.add(
                HomeCollectionCoverage(
                    postal_code="700001",
                    serviceable=False,
                    charge_inr=0,
                    slot_note_bn="",
                    slot_note_hi="",
                    slot_note_en="",
                )
            )
            added += 1

        if not db.query(InsurancePolicy).filter_by(policy_number="DEMO-POLICY-001").first():
            db.add(
                InsurancePolicy(
                    policy_number="DEMO-POLICY-001", insurer_name="Star Assure", patient_phone="9000000001", active=True
                )
            )
            db.add(
                InsuranceCoverageRule(
                    insurer_name="Star Assure", lab_test_id=None, coverage_percent=80, co_payment_inr=100
                )
            )
            added += 1

        if not db.query(DepartmentHours).first() and "Cardiology" in dept_by_name:
            db.add(
                DepartmentHours(
                    department_id=dept_by_name["Cardiology"].id,
                    hours_bn="কার্ডিওলজি বিভাগ সকাল ১০টা থেকে দুপুর ১২টা এবং সন্ধ্যা ৬টা থেকে ৮টা পর্যন্ত খোলা।",
                    hours_hi="कार्डियोलॉजी विभाग सुबह 10 से 12 बजे और शाम 6 से 8 बजे तक खुला रहता है।",
                    hours_en="The Cardiology department is open 10am-12pm and 6pm-8pm.",
                )
            )
            added += 1

        if doctor and not db.query(DoctorLeave).filter_by(doctor_id=doctor.id).first():
            future = _now_date_plus(10)
            db.add(DoctorLeave(doctor_id=doctor.id, start_date=future, end_date=future, return_date=_now_date_plus(11)))
            added += 1

        db.commit()
    finally:
        if own:
            db.close()
    return {"rows_added": added}


def _now_date_plus(days: int) -> str:
    import datetime

    return (datetime.date.today() + datetime.timedelta(days=days)).isoformat()


def backfill_enquiry_facts(db=None) -> dict:
    """Idempotent, never overwrites a value a clinician has since edited
    by hand -- same rule seed.backfill_i18n() already applies."""
    from models import LabTest

    added = add_enquiry_columns()
    own = db is None
    db = db or SessionLocal()
    filled = 0
    try:
        for t in db.query(LabTest).all():
            if t.fasting_hours is None and t.name in FASTING_HOURS_OVERRIDES:
                hours, water_ok = FASTING_HOURS_OVERRIDES[t.name]
                t.fasting_hours, t.water_allowed_while_fasting = hours, water_ok
                filled += 1
            if t.name in HOME_COLLECTION_INELIGIBLE and t.home_collection_eligible:
                t.home_collection_eligible = False
                filled += 1
            if t.name in PRESCRIPTION_REQUIRED and not t.prescription_required:
                t.prescription_required = True
                t.prescription_note_bn, t.prescription_note_hi, t.prescription_note_en = _PRESCRIPTION_NOTE
                filled += 1
        db.commit()
    finally:
        if own:
            db.close()
    return {"columns_added": added, "values_filled": filled}
