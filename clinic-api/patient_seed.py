"""SAMPLE patient data, so the registry, security questions and history can be run end to end.

Everything here is invented. The names, dates of birth, addresses, ids, tests and medicines belong to
no real person, and the one retest interval carries `approved_by = "SAMPLE DATA"` on purpose: it is not
a clinician's decision and must never be read as one. In production the same tables are filled by the
hospital connector (KCD-131) and this module is never run.

Seeded only when `patient_registry` is EMPTY, and never overwrites anything: a database that already
holds registry rows is left exactly as it is.

The cast, chosen so each behaviour has a case:
    Asha, Rakesh and Tania Saha   three patients on ONE phone number (a shared family number)
    Asha Saha                     78 -- a senior citizen; tests and medicines on record
    Debashis Mondal               has an upcoming appointment
    Sunita Devi                   64 -- a senior; Hindi-script aliases
    Ravi Kumar                    a patient with one test and one medicine, and stored preferences
"""

from __future__ import annotations

import datetime

import booking_service as bs
from models import (
    Doctor,
    DoctorSchedule,
    LabTest,
    Medicine,
    Patient,
    PatientMedicine,
    PatientPreference,
    PatientRegistry,
    RetestInterval,
    TestPerformance,
)
from registry import age_years
from sqlalchemy.orm import Session

PATIENTS = [
    dict(
        uid="KCP-100001",
        name="Asha Saha",
        phone="9830012345",
        dob="1948-03-14",
        gender="F",
        address="24/B Gariahat Road, Ballygunge, Kolkata 700019",
        locality="Ballygunge",
        pincode="700019",
        name_aliases="আশা সাহা|आशा साहा",
        address_aliases="গড়িয়াহাট রোড বালিগঞ্জ|गड़ियाहाट रोड बालीगंज",
    ),
    dict(
        uid="KCP-100002",
        name="Rakesh Saha",
        phone="9830012345",
        dob="1975-08-02",
        gender="M",
        address="24/B Gariahat Road, Ballygunge, Kolkata 700019",
        locality="Ballygunge",
        pincode="700019",
        name_aliases="রাকেশ সাহা|राकेश साहा",
        address_aliases="গড়িয়াহাট রোড বালিগঞ্জ|गड़ियाहाट रोड बालीगंज",
    ),
    dict(
        uid="KCP-100003",
        name="Tania Saha",
        phone="9830012345",
        dob="2010-11-21",
        gender="F",
        address="24/B Gariahat Road, Ballygunge, Kolkata 700019",
        locality="Ballygunge",
        pincode="700019",
        name_aliases="তানিয়া সাহা|तानिया साहा",
        address_aliases="গড়িয়াহাট রোড বালিগঞ্জ|गड़ियाहाट रोड बालीगंज",
    ),
    dict(
        uid="KCP-100004",
        name="Debashis Mondal",
        phone="9830023456",
        dob="1985-06-30",
        gender="M",
        address="12 Lake Town, Block A, Kolkata 700089",
        locality="Lake Town",
        pincode="700089",
        name_aliases="দেবাশীষ মণ্ডল|देबाशीष मंडल",
        address_aliases="লেক টাউন ব্লক এ|लेक टाउन ब्लॉक ए",
    ),
    dict(
        uid="KCP-100005",
        name="Sunita Devi",
        phone="9830034567",
        dob="1962-01-09",
        gender="F",
        address="5 Burrabazar, Kolkata 700007",
        locality="Burrabazar",
        pincode="700007",
        name_aliases="সুনীতা দেবী|सुनीता देवी",
        address_aliases="বড়বাজার|बड़ाबाज़ार",
    ),
    dict(
        uid="KCP-100006",
        name="Ravi Kumar",
        phone="9830045678",
        dob="1990-12-05",
        gender="M",
        address="Howrah Maidan, Howrah 711101",
        locality="Howrah",
        pincode="711101",
        name_aliases="রবি কুমার|रवि कुमार",
        address_aliases="হাওড়া ময়দান|हावड़ा मैदान",
    ),
]

MEDICINES = [
    ("Metformin", "Metformin", "মেটফর্মিন", "मेटफॉर्मिन"),
    ("Amlodipine", "Amlodipine", "অ্যামলোডিপিন", "एम्लोडिपिन"),
    ("Atorvastatin", "Atorvastatin", "অ্যাটোরভাস্টাটিন", "एटोरवास्टेटिन"),
    ("Levothyroxine", "Levothyroxine", "লিভোথাইরক্সিন", "लेवोथायरोक्सिन"),
    ("Paracetamol", "Paracetamol", "প্যারাসিটামল", "पैरासिटामोल"),
    ("Pantoprazole", "Pantoprazole", "প্যান্টোপ্রাজল", "पैंटोप्राज़ोल"),
]

# (patient uid, test name, days ago)
PERFORMED = [
    ("KCP-100001", "Complete Blood Count (CBC)", 40),
    ("KCP-100001", "HbA1c", 120),
    ("KCP-100001", "Blood Sugar Fasting", 20),
    ("KCP-100002", "Lipid Profile", 200),
    ("KCP-100004", "Thyroid Profile (T3 T4 TSH)", 65),
    ("KCP-100004", "Complete Blood Count (CBC)", 10),
    ("KCP-100005", "HbA1c", 95),
    ("KCP-100005", "Kidney Function Test (KFT)", 150),
    ("KCP-100006", "Vitamin D (25-OH)", 30),
]
# (patient uid, medicine, days ago)
PRESCRIBED = [
    ("KCP-100001", "Metformin", 70),
    ("KCP-100001", "Amlodipine", 70),
    ("KCP-100004", "Levothyroxine", 65),
    ("KCP-100005", "Atorvastatin", 95),
    ("KCP-100005", "Metformin", 95),
    ("KCP-100006", "Paracetamol", 30),
]


def _iso(days_ago: int) -> str:
    return (bs._now().date() - datetime.timedelta(days=days_ago)).isoformat()


def seed_patient_context(db: Session) -> dict:
    """Insert the sample rows if the registry is empty. Returns counts (all zero if it was not)."""
    if db.query(PatientRegistry).count() > 0:
        return {"seeded": False}
    now = bs._now()
    by_uid: dict[str, Patient] = {}
    for spec in PATIENTS:
        p = db.query(Patient).filter_by(name=spec["name"], phone=spec["phone"]).first()
        if p is None:
            p = Patient(name=spec["name"], phone=spec["phone"], age=age_years(spec["dob"]), created_at=now)
            db.add(p)
            db.flush()
        by_uid[spec["uid"]] = p
        db.add(
            PatientRegistry(
                patient_id=p.id,
                patient_uid=spec["uid"],
                dob=spec["dob"],
                gender=spec["gender"],
                address_text=spec["address"],
                locality=spec["locality"],
                pincode=spec["pincode"],
                name_aliases=spec["name_aliases"],
                address_aliases=spec["address_aliases"],
                registered_on=_iso(900),
                status="active",
                source="local",
            )
        )
    meds = {}
    for name, generic, bn, hi in MEDICINES:
        m = db.query(Medicine).filter_by(name=name).first() or Medicine(
            name=name, generic_name=generic, aliases_bn=bn, aliases_hi=hi
        )
        db.add(m)
        db.flush()
        meds[name] = m
    tests = {t.name: t for t in db.query(LabTest).all()}
    n_tests = n_meds = 0
    for uid, test, days in PERFORMED:
        if test in tests:
            db.add(
                TestPerformance(
                    patient_id=by_uid[uid].id,
                    lab_test_id=tests[test].id,
                    performed_on=_iso(days),
                    source="sample",
                    created_at=now,
                )
            )
            n_tests += 1
    for uid, med, days in PRESCRIBED:
        db.add(
            PatientMedicine(
                patient_id=by_uid[uid].id,
                medicine_id=meds[med].id,
                prescribed_on=_iso(days),
                prescribed_by="Dr. A. Sen (sample)",
                source="sample",
                created_at=now,
            )
        )
        n_meds += 1
    # The one retest interval: SAMPLE ONLY, and labelled so. Without an interval no "due" is ever said.
    hba1c = tests.get("HbA1c")
    if hba1c is not None and not db.query(RetestInterval).filter_by(lab_test_id=hba1c.id).first():
        db.add(RetestInterval(lab_test_id=hba1c.id, interval_days=90, approved_by="SAMPLE DATA", approved_at=_iso(0)))
    # KCD-499: stored against the PATIENT, offered for confirmation, never applied silently
    db.add(
        PatientPreference(patient_id=by_uid["KCP-100006"].id, branch="Howrah", delivery_channel="sms", updated_at=now)
    )
    n_appts = _sample_appointment(db, by_uid["KCP-100004"])
    db.commit()
    return {
        "seeded": True,
        "patients": len(PATIENTS),
        "tests_performed": n_tests,
        "medicines": n_meds,
        "appointments": n_appts,
    }


def _sample_appointment(db: Session, patient: Patient) -> int:
    doctor = db.query(Doctor).filter(Doctor.name.like("%Sen")).first()
    if doctor is None:
        return 0
    for offset in range(3, 20):
        d = bs._now().date() + datetime.timedelta(days=offset)
        if db.query(DoctorSchedule).filter_by(doctor_id=doctor.id, weekday=d.weekday()).first():
            slots = bs.available_slots(db, doctor.id, d.isoformat())
            if slots:
                # The LAST slot of the day, never the first: every booking test (and a real caller told "the
                # first free slot") takes the first one, so a sample appointment there made those tests pass
                # or fail depending on the weekday they ran on.
                slot = slots[-1]
                # through the SAME hold-then-confirm primitives a call uses, so the slot is really taken
                held = bs.hold_slot(db, doctor.id, d.isoformat(), slot)
                if held.get("success"):
                    done = bs.confirm_booking(
                        db,
                        held["hold_token"],
                        doctor.id,
                        d.isoformat(),
                        slot,
                        patient.name,
                        patient.phone,
                        patient.phone,
                        patient.age,
                    )
                    return 1 if done.get("success") else 0
    return 0
