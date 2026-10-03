"""Populates dummy clinic data: 8 departments, 32 doctors (4 each), a
weekly chamber schedule per doctor, and a ~32-row lab test price list.

All fictional -- names, quals, prices, and schedule patterns are
representative of a real Kolkata polyclinic but invented for this
prototype, not sourced from any real clinic's data.

Run:  python3 seed.py     (idempotent -- safe to re-run, wipes and reloads)
"""

from __future__ import annotations

from db import SessionLocal, engine
from i18n_content import (
    DEFAULT_PREP_EN,
    DEFAULT_PREP_HI,
    FAQ_I18N,
    FAQ_KEYWORDS_I18N,
    PREP_I18N,
    SURNAME_HI,
    TEST_ALIASES_HI,
)
from models import FAQ, Base, Department, Doctor, DoctorSchedule, LabTest

# ---------------------------------------------------------------------------
# 8 departments, 4 doctors each = 32 doctors.
# Schedule pattern rotates across 4 shift templates so the 32 doctors don't
# all sit at the same time -- a caller asking "is anyone free this evening"
# gets a realistic mixed answer.
# ---------------------------------------------------------------------------
# Bengali-script spelling(s) for each surname, so a caller saying "ডক্টর
# সেন" actually matches "Dr. A. Sen". Keyed on the surname as it appears
# as the last word of the English name below.
SURNAME_BN = {
    "Mukherjee": ["মুখার্জী", "মুখোপাধ্যায়"],
    "Sen": ["সেন"],
    "Ghosh": ["ঘোষ"],
    "Chowdhury": ["চৌধুরী"],
    "Bhattacharya": ["ভট্টাচার্য"],
    "Roy": ["রায়"],
    "Banerjee": ["ব্যানার্জী", "বন্দ্যোপাধ্যায়"],
    "Dutta": ["দত্ত"],
    "Chatterjee": ["চ্যাটার্জী", "চট্টোপাধ্যায়"],
    "Basu": ["বসু"],
    "Mitra": ["মিত্র"],
    "Sengupta": ["সেনগুপ্ত"],
    "Das": ["দাস"],
    "Bose": ["বসু"],
    "Kar": ["কর"],
    "Nandi": ["নন্দী"],
    "Pal": ["পাল"],
    "Halder": ["হালদার"],
    "Guha": ["গুহ"],
    "Chanda": ["চন্দ", "চাঁদা"],
    "Saha": ["সাহা"],
    "Dey": ["দে"],
    "Adhikari": ["অধিকারী"],
    "Bagchi": ["বাগচী"],
    "Biswas": ["বিশ্বাস"],
    "Majumder": ["মজুমদার"],
    "Mondal": ["মন্ডল"],
    "Ganguly": ["গাঙ্গুলী", "গঙ্গোপাধ্যায়"],
    "Sinha": ["সিনহা"],
    "Ray": ["রায়"],
    "Sarkar": ["সরকার"],
    "Chakraborty": ["চক্রবর্তী"],
}

SHIFT_TEMPLATES = [
    {"days": [0, 2, 4], "start": "10:00", "end": "12:00"},  # Mon/Wed/Fri morning
    {"days": [1, 3, 5], "start": "10:00", "end": "12:00"},  # Tue/Thu/Sat morning
    {"days": [0, 2, 4], "start": "18:00", "end": "20:00"},  # Mon/Wed/Fri evening
    {"days": [1, 3, 5], "start": "17:30", "end": "19:30"},  # Tue/Thu/Sat evening
]

# The whole name of each seeded doctor, keyed by the short name the rest of the system uses. FICTIONAL placeholders
# (the seed only ever held initials): the given name starts with the initial the short name carries. The clinic replaces
# them with real names by editing doctors.full_name; the backfill never overwrites a value that is already there.
DOCTOR_FULL_NAMES = {
    "Dr. S. Mukherjee": "Dr. Sourav Mukherjee",
    "Dr. A. Sen": "Dr. Arindam Sen",
    "Dr. P. Ghosh": "Dr. Prabir Ghosh",
    "Dr. R. Chowdhury": "Dr. Rajib Chowdhury",
    "Dr. K. Bhattacharya": "Dr. Kaushik Bhattacharya",
    "Dr. N. Roy": "Dr. Nirmal Roy",
    "Dr. S. Banerjee": "Dr. Subrata Banerjee",
    "Dr. M. Dutta": "Dr. Manas Dutta",
    "Dr. S. Chatterjee": "Dr. Sutapa Chatterjee",
    "Dr. A. Basu": "Dr. Aparna Basu",
    "Dr. R. Mitra": "Dr. Rina Mitra",
    "Dr. P. Sengupta": "Dr. Paromita Sengupta",
    "Dr. D. Das": "Dr. Debasish Das",
    "Dr. T. Bose": "Dr. Tapan Bose",
    "Dr. A. Kar": "Dr. Amit Kar",
    "Dr. S. Nandi": "Dr. Subhankar Nandi",
    "Dr. R. Pal": "Dr. Ranjan Pal",
    "Dr. K. Halder": "Dr. Kalyan Halder",
    "Dr. S. Guha": "Dr. Sanjay Guha",
    "Dr. B. Chanda": "Dr. Biswajit Chanda",
    "Dr. M. Saha": "Dr. Mitali Saha",
    "Dr. A. Dey": "Dr. Ashok Dey",
    "Dr. P. Adhikari": "Dr. Pritha Adhikari",
    "Dr. S. Bagchi": "Dr. Samir Bagchi",
    "Dr. N. Biswas": "Dr. Nandini Biswas",
    "Dr. R. Majumder": "Dr. Rupa Majumder",
    "Dr. A. Mondal": "Dr. Anirban Mondal",
    "Dr. S. Ganguly": "Dr. Shreya Ganguly",
    "Dr. K. Sinha": "Dr. Kunal Sinha",
    "Dr. P. Ray": "Dr. Partha Ray",
    "Dr. A. Sarkar": "Dr. Abhijit Sarkar",
    "Dr. S. Chakraborty": "Dr. Snehasish Chakraborty",
}

# How the given names above are written and said in Bengali and Hindi (placeholders like the names themselves).
_GIVEN_BN_HI = {
    "Sourav": ("সৌরভ", "सौरव"),
    "Arindam": ("অরিন্দম", "अरिंदम"),
    "Prabir": ("প্রবীর", "प्रबीर"),
    "Rajib": ("রাজীব", "राजीव"),
    "Kaushik": ("কৌশিক", "कौशिक"),
    "Nirmal": ("নির্মল", "निर्मल"),
    "Subrata": ("সুব্রত", "सुब्रत"),
    "Manas": ("মানস", "मानस"),
    "Sutapa": ("সুতপা", "सुतपा"),
    "Aparna": ("অপর্ণা", "अपर्णा"),
    "Rina": ("রিনা", "रीना"),
    "Paromita": ("পারমিতা", "पारमिता"),
    "Debasish": ("দেবাশিস", "देबाशीष"),
    "Tapan": ("তপন", "तपन"),
    "Amit": ("অমিত", "अमित"),
    "Subhankar": ("শুভঙ্কর", "शुभंकर"),
    "Ranjan": ("রঞ্জন", "रंजन"),
    "Kalyan": ("কল্যাণ", "कल्याण"),
    "Sanjay": ("সঞ্জয়", "संजय"),
    "Biswajit": ("বিশ্বজিৎ", "बिस्वजीत"),
    "Mitali": ("মিতালী", "मिताली"),
    "Ashok": ("অশোক", "अशोक"),
    "Pritha": ("পৃথা", "पृथा"),
    "Samir": ("সমীর", "समीर"),
    "Nandini": ("নন্দিনী", "नंदिनी"),
    "Rupa": ("রূপা", "रूपा"),
    "Anirban": ("অনির্বাণ", "अनिर्बान"),
    "Shreya": ("শ্রেয়া", "श्रेया"),
    "Kunal": ("কুণাল", "कुणाल"),
    "Partha": ("পার্থ", "पार्थ"),
    "Abhijit": ("অভিজিৎ", "अभिजीत"),
    "Snehasish": ("স্নেহাশিস", "स्नेहाशीष"),
}


def doctor_full_names() -> dict[str, tuple[str, str, str]]:
    """short name -> (whole name, the same in Bengali script, the same in Devanagari)."""
    out = {}
    for short, full in DOCTOR_FULL_NAMES.items():
        given, surname = full.split()[1], full.split()[-1]
        bn_given, hi_given = _GIVEN_BN_HI[given]
        bn_surname = (SURNAME_BN.get(surname) or [""])[0]
        hi_surname = (SURNAME_HI.get(surname) or [""])[0]
        out[short] = (
            full,
            f"{bn_given} {bn_surname}".strip() if bn_surname else "",
            f"{hi_given} {hi_surname}".strip() if hi_surname else "",
        )
    return out


DEPARTMENTS = {
    "General Medicine": [
        ("Dr. S. Mukherjee", "MBBS, MD (Gen. Med.)"),
        ("Dr. A. Sen", "MBBS, MD (Gen. Med.)"),
        ("Dr. P. Ghosh", "MBBS, DNB (Gen. Med.)"),
        ("Dr. R. Chowdhury", "MBBS, MD"),
    ],
    "Cardiology": [
        ("Dr. K. Bhattacharya", "MBBS, MD, DM (Cardiology)"),
        ("Dr. N. Roy", "MBBS, DM (Cardiology)"),
        ("Dr. S. Banerjee", "MBBS, MD, DM (Cardiology)"),
        ("Dr. M. Dutta", "MBBS, DM (Cardiology)"),
    ],
    "Gynaecology & Obstetrics": [
        ("Dr. S. Chatterjee", "MBBS, MS (Obs & Gynae)"),
        ("Dr. A. Basu", "MBBS, DGO"),
        ("Dr. R. Mitra", "MBBS, MS (Obs & Gynae)"),
        ("Dr. P. Sengupta", "MBBS, DNB (Obs & Gynae)"),
    ],
    "Orthopaedics": [
        ("Dr. D. Das", "MBBS, MS (Ortho)"),
        ("Dr. T. Bose", "MBBS, D.Ortho"),
        ("Dr. A. Kar", "MBBS, MS (Ortho)"),
        ("Dr. S. Nandi", "MBBS, DNB (Ortho)"),
    ],
    "ENT": [
        ("Dr. R. Pal", "MBBS, MS (ENT)"),
        ("Dr. K. Halder", "MBBS, DLO"),
        ("Dr. S. Guha", "MBBS, MS (ENT)"),
        ("Dr. B. Chanda", "MBBS, DLO"),
    ],
    "Dermatology": [
        ("Dr. M. Saha", "MBBS, MD (Dermatology)"),
        ("Dr. A. Dey", "MBBS, DVD"),
        ("Dr. P. Adhikari", "MBBS, MD (Dermatology)"),
        ("Dr. S. Bagchi", "MBBS, DDV"),
    ],
    "Paediatrics": [
        ("Dr. N. Biswas", "MBBS, MD (Paediatrics)"),
        ("Dr. R. Majumder", "MBBS, DCH"),
        ("Dr. A. Mondal", "MBBS, MD (Paediatrics)"),
        ("Dr. S. Ganguly", "MBBS, DCH"),
    ],
    "Diabetology & Endocrinology": [
        ("Dr. K. Sinha", "MBBS, MD, DM (Endocrinology)"),
        ("Dr. P. Ray", "MBBS, MD (Diabetology)"),
        ("Dr. A. Sarkar", "MBBS, DM (Endocrinology)"),
        ("Dr. S. Chakraborty", "MBBS, MD (Diabetology)"),
    ],
}

# (name, [Bengali aliases], rate_inr, sample_type, report_time_hours)
# Aliases are what a real caller says, not a transliteration of the
# English column -- e.g. nobody says "কমপ্লিট ব্লাড কাউন্ট", they say "সিবিসি".
LAB_TESTS = [
    ("Complete Blood Count (CBC)", ["সিবিসি", "সি বি সি"], 400, "Blood", 6),
    ("ESR", ["ইএসআর", "ই এস আর"], 150, "Blood", 6),
    ("Blood Sugar Fasting", ["ব্লাড সুগার ফাস্টিং", "সুগার ফাস্টিং", "খালি পেটে সুগার"], 120, "Blood", 4),
    ("Blood Sugar PP", ["সুগার পিপি", "পিপি সুগার", "খাওয়ার পরে সুগার"], 120, "Blood", 4),
    ("HbA1c", ["এইচবিএ১সি", "হিমোগ্লোবিন এ১সি"], 650, "Blood", 24),
    ("Lipid Profile", ["লিপিড প্রোফাইল", "কোলেস্টেরল টেস্ট"], 650, "Blood", 24),
    ("Liver Function Test (LFT)", ["লিভার ফাংশন টেস্ট", "এলএফটি", "লিভার টেস্ট"], 800, "Blood", 24),
    ("Kidney Function Test (KFT)", ["কিডনি ফাংশন টেস্ট", "কেএফটি", "কিডনি টেস্ট"], 750, "Blood", 24),
    ("Thyroid Profile (T3 T4 TSH)", ["থাইরয়েড প্রোফাইল", "থাইরয়েড টেস্ট"], 700, "Blood", 24),
    ("TSH", ["টিএসএইচ"], 350, "Blood", 24),
    ("Urine Routine Examination", ["ইউরিন টেস্ট", "প্রস্রাব পরীক্ষা", "ইউরিন রুটিন"], 200, "Urine", 6),
    ("Widal Test", ["ওয়াইডাল টেস্ট", "উইডাল টেস্ট", "টাইফয়েড টেস্ট"], 250, "Blood", 12),
    ("Dengue NS1 Antigen", ["ডেঙ্গু এনএস১", "ডেঙ্গু টেস্ট"], 900, "Blood", 6),
    ("Dengue IgG/IgM", ["ডেঙ্গু আইজিজি", "ডেঙ্গু আইজিএম"], 900, "Blood", 6),
    ("Malaria Antigen", ["ম্যালেরিয়া টেস্ট", "ম্যালেরিয়া এন্টিজেন"], 400, "Blood", 4),
    ("CRP (C-Reactive Protein)", ["সিআরপি"], 500, "Blood", 12),
    ("Vitamin D (25-OH)", ["ভিটামিন ডি"], 1800, "Blood", 72),
    ("Vitamin B12", ["ভিটামিন বি১২", "বি১২"], 1200, "Blood", 48),
    ("Serum Creatinine", ["ক্রিয়াটিনিন", "সিরাম ক্রিয়াটিনিন"], 250, "Blood", 12),
    ("Serum Electrolytes", ["ইলেক্ট্রোলাইটস", "ইলেকট্রোলাইট টেস্ট"], 450, "Blood", 12),
    ("Blood Grouping & Rh Typing", ["ব্লাড গ্রুপ", "রক্তের গ্রুপ"], 200, "Blood", 4),
    ("HIV Test (ELISA)", ["এইচআইভি টেস্ট", "এইডস টেস্ট"], 500, "Blood", 24),
    ("HBsAg", ["এইচবিএসএজি", "হেপাটাইটিস বি"], 400, "Blood", 24),
    ("HCV", ["এইচসিভি", "হেপাটাইটিস সি"], 600, "Blood", 24),
    ("ECG", ["ইসিজি", "ইলেক্ট্রোকার্ডিওগ্রাম"], 300, "Cardiac", 1),
    ("Chest X-Ray (PA view)", ["বুকের এক্স-রে", "চেস্ট এক্সরে"], 400, "Imaging", 4),
    ("USG Whole Abdomen", ["পেটের আলট্রাসাউন্ড", "হোল অ্যাবডোমেন ইউএসজি", "পেটের ইউএসজি"], 1500, "Imaging", 4),
    ("USG Pregnancy Profile", ["প্রেগন্যান্সি আলট্রাসাউন্ড", "প্রেগনেন্সি ইউএসজি"], 1600, "Imaging", 4),
    ("2D Echocardiography", ["ইকো টেস্ট", "একোকার্ডিওগ্রাফি", "ইকোকার্ডিওগ্রাম"], 2000, "Cardiac", 4),
    ("TMT (Treadmill Test)", ["টিএমটি", "ট্রেডমিল টেস্ট"], 2200, "Cardiac", 4),
    ("Pap Smear", ["প্যাপ স্মিয়ার"], 900, "Sample (Cervical)", 72),
    ("PSA (Prostate Specific Antigen)", ["পিএসএ"], 900, "Blood", 48),
    ("Uric Acid", ["ইউরিক অ্যাসিড", "ইউরিক এসিড", "ইউরিক এসিদ"], 250, "Blood", 12),
    ("Calcium (Serum)", ["ক্যালসিয়াম", "সিরাম ক্যালসিয়াম"], 250, "Blood", 12),
]

# Per-test preparation, keyed by the exact LAB_TESTS name. Only tests with
# a genuine requirement are listed; every other test gets the generic
# no-special-preparation default in _prep_for() below. Fictional but
# clinically plausible, matching the rest of this file's seed data.
PREP_OVERRIDES: dict[str, tuple[bool, str]] = {
    "Blood Sugar Fasting": (
        True,
        "এই টেস্টের আগে অন্তত আট ঘণ্টা কিছু খাবেন না, শুধু জল খেতে পারেন। সকালে খালি পেটে এসে টেস্ট করানো ভালো।",
    ),
    "Lipid Profile": (
        True,
        "এই টেস্টের আগে দশ থেকে বারো ঘণ্টা উপবাস থাকতে হবে, জল ছাড়া আর কিছু খাবেন না। আগের রাতে হালকা খাবার খাওয়া ভালো।",
    ),
    "HbA1c": (False, "এই টেস্টের জন্য উপবাস থাকার দরকার নেই, স্বাভাবিক খাওয়াদাওয়ার পরেও করানো যায়।"),
    "Kidney Function Test (KFT)": (True, "এই টেস্টের আগে ছয় থেকে আট ঘণ্টা উপবাস থাকার পরামর্শ দেওয়া হয়।"),
    "Liver Function Test (LFT)": (True, "এই টেস্টের আগে আট ঘণ্টা উপবাস থাকার পরামর্শ দেওয়া হয়।"),
    "USG Whole Abdomen": (True, "এই টেস্টের আগে ছয় ঘণ্টা কিছু খাবেন না এবং প্রস্রাব চেপে রাখতে হবে, মূত্রথলি ভর্তি থাকা দরকার।"),
    "USG Pregnancy Profile": (True, "এই টেস্টের আগে জল বেশি খেয়ে মূত্রথলি ভর্তি রাখতে হবে, টেস্টের ঠিক আগে প্রস্রাব করবেন না।"),
    "TMT (Treadmill Test)": (False, "হালকা, আরামদায়ক পোশাক ও জুতো পরে আসবেন। টেস্টের দুই ঘণ্টা আগে ভারী খাবার না খাওয়াই ভালো।"),
}

_DEFAULT_PREP_BN = "এই টেস্টের জন্য বিশেষ কোনো প্রস্তুতির প্রয়োজন নেই, স্বাভাবিকভাবে এসে করাতে পারেন।"


def _prep_for(test_name: str) -> tuple[bool, str]:
    return PREP_OVERRIDES.get(test_name, (False, _DEFAULT_PREP_BN))


# (topic, [Bengali keyword/phrase cues], answer)
# Fictional clinic details, representative of a real Kolkata polyclinic --
# same convention as DEPARTMENTS/LAB_TESTS above, not sourced from a real
# clinic. This is the Tier-3 "approved content" table (models.FAQ).
FAQ_ENTRIES = [
    (
        "hours",
        ["সময়", "কখন খোলে", "কখন বন্ধ", "খোলা থাকে", "ভিজিটিং আওয়ার্স", "ক্লিনিকের সময়"],
        "আমাদের ক্লিনিক প্রতিদিন সকাল আটটা থেকে রাত আটটা পর্যন্ত খোলা থাকে, রবিবার সকাল আটটা থেকে দুপুর দুটো পর্যন্ত।",
    ),
    (
        "location",
        ["কোথায়", "ঠিকানা", "লোকেশন", "কোন জায়গায়", "কীভাবে আসব"],
        "আমাদের ক্লিনিক কলকাতার রাজারহাট নিউ টাউনে, সিটি সেন্টার টু-এর কাছে।",
    ),
    (
        "payment_methods",
        ["পেমেন্ট", "টাকা কীভাবে দেব", "কার্ড চলে", "ইউপিআই", "নগদ"],
        "নগদ, সব ধরনের কার্ড এবং ইউপিআই -- সব মাধ্যমেই পেমেন্ট নেওয়া হয়।",
    ),
    (
        "insurance",
        ["ইনসিওরেন্স", "ইন্সুরেন্স", "বিমা", "ক্যাশলেস", "মেডিক্লেম"],
        "প্রধান বিমা সংস্থাগুলোর ক্যাশলেস সুবিধা আছে। আপনার কার্ডের নাম বললে কাউন্টার থেকে নিশ্চিত করে দেওয়া হবে।",
    ),
    ("parking", ["পার্কিং", "গাড়ি রাখার জায়গা", "গাড়ি কোথায় রাখব"], "ক্লিনিকের নিজস্ব পার্কিং আছে, কোনো চার্জ লাগে না।"),
    (
        "report_collection",
        ["রিপোর্ট কীভাবে পাব", "রিপোর্ট নিতে", "রিপোর্ট কোথা থেকে"],
        "রিপোর্ট সরাসরি কাউন্টার থেকে সংগ্রহ করতে পারেন, অথবা হোয়াটসঅ্যাপ ও ইমেলেও পাঠানো হয়।",
    ),
    (
        "contact_number",
        ["ফোন নম্বর", "যোগাযোগ", "নম্বরটা কী"],
        "আমাদের হেল্পডেস্ক নম্বরে ফোন করে সরাসরি কথা বলতে পারেন, এই কলটার পরেও নম্বরটা এসএমএস করে দেওয়া হবে।",
    ),
    (
        "home_collection",
        ["বাড়িতে এসে", "হোম কালেকশন", "বাড়ি থেকে স্যাম্পল"],
        "বেশিরভাগ ব্লাড টেস্টের জন্য বাড়িতে এসে স্যাম্পল নেওয়ার সুবিধা আছে। বুক করার সময় বলে দেবেন।",
    ),
]


I18N_COLUMNS = {
    "doctors": ["aliases_hi"],
    "lab_tests": ["aliases_hi", "prep_instructions_hi", "prep_instructions_en"],
    "faqs": ["answer_hi", "answer_en", "keywords_hi", "keywords_en"],
}


def add_i18n_columns() -> list[str]:
    """ALTER TABLE for any Hindi/English column an older database lacks.

    MUST run before the first ORM query on these tables: SQLAlchemy selects
    every mapped column, so on an old database even `count()` fails with
    "no such column" -- which is exactly how clinic-api crashed at boot on
    the pod's existing clinic.db."""
    from sqlalchemy import inspect, text

    added = []
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in I18N_COLUMNS.items():
            have = {c["name"] for c in insp.get_columns(table)}
            for col in cols:
                if col not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} VARCHAR NOT NULL DEFAULT ''"))
                    added.append(f"{table}.{col}")
    return added


def backfill_i18n(db=None) -> dict:
    """Bring an EXISTING database up to date with the Hindi/English columns
    without wiping it. seed() is destructive (drop_all), so the boot-time
    path cannot use it on a database that already holds bookings.

    Adds any missing column (add_i18n_columns), then fills only rows whose
    value is still empty -- so a value someone edited by hand is never
    overwritten. Safe to run on every boot."""
    added = add_i18n_columns()

    own = db is None
    db = db or SessionLocal()
    filled = 0
    try:
        for d in db.query(Doctor).all():
            if not d.aliases_hi:
                d.aliases_hi = "|".join(SURNAME_HI.get(d.name.split()[-1], []))
                filled += 1
        for t in db.query(LabTest).all():
            prep_hi, prep_en = PREP_I18N.get(t.name, (DEFAULT_PREP_HI, DEFAULT_PREP_EN))
            for attr, val in (
                ("aliases_hi", "|".join(TEST_ALIASES_HI.get(t.name, []))),
                ("prep_instructions_hi", prep_hi),
                ("prep_instructions_en", prep_en),
            ):
                if not getattr(t, attr):
                    setattr(t, attr, val)
                    filled += 1
        for f in db.query(FAQ).all():
            ans_hi, ans_en = FAQ_I18N.get(f.topic, ("", ""))
            kw_hi, kw_en = FAQ_KEYWORDS_I18N.get(f.topic, ([], []))
            for attr, val in (
                ("answer_hi", ans_hi),
                ("answer_en", ans_en),
                ("keywords_hi", "|".join(kw_hi)),
                ("keywords_en", "|".join(kw_en)),
            ):
                if not getattr(f, attr):
                    setattr(f, attr, val)
                    filled += 1
        db.commit()
    finally:
        if own:
            db.close()
    return {"columns_added": added, "values_filled": filled}


def seed():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        doctor_index = 0
        for dept_name, doctors in DEPARTMENTS.items():
            dept = Department(name=dept_name)
            db.add(dept)
            db.flush()  # get dept.id

            for doc_name, quals in doctors:
                surname = doc_name.split()[-1]
                aliases = "|".join(SURNAME_BN.get(surname, []))
                latin, full_bn, full_hi = doctor_full_names().get(doc_name, ("", "", ""))
                doc = Doctor(
                    name=doc_name,
                    full_name=latin,
                    full_name_bn=full_bn,
                    full_name_hi=full_hi,
                    qualifications=quals,
                    aliases_bn=aliases,
                    aliases_hi="|".join(SURNAME_HI.get(surname, [])),
                    department_id=dept.id,
                )
                db.add(doc)
                db.flush()

                template = SHIFT_TEMPLATES[doctor_index % len(SHIFT_TEMPLATES)]
                for weekday in template["days"]:
                    db.add(
                        DoctorSchedule(
                            doctor_id=doc.id,
                            weekday=weekday,
                            start_time=template["start"],
                            end_time=template["end"],
                        )
                    )
                doctor_index += 1

        for name, aliases_bn, rate, sample, hours in LAB_TESTS:
            fasting, prep_bn = _prep_for(name)
            prep_hi, prep_en = PREP_I18N.get(name, (DEFAULT_PREP_HI, DEFAULT_PREP_EN))
            db.add(
                LabTest(
                    name=name,
                    aliases_bn="|".join(aliases_bn),
                    rate_inr=rate,
                    aliases_hi="|".join(TEST_ALIASES_HI.get(name, [])),
                    sample_type=sample,
                    report_time_hours=hours,
                    fasting_required=fasting,
                    prep_instructions_bn=prep_bn,
                    prep_instructions_hi=prep_hi,
                    prep_instructions_en=prep_en,
                )
            )

        for topic, keywords_bn, answer_bn in FAQ_ENTRIES:
            ans_hi, ans_en = FAQ_I18N.get(topic, ("", ""))
            kw_hi, kw_en = FAQ_KEYWORDS_I18N.get(topic, ([], []))
            db.add(
                FAQ(
                    topic=topic,
                    keywords_bn="|".join(keywords_bn),
                    answer_bn=answer_bn,
                    answer_hi=ans_hi,
                    answer_en=ans_en,
                    keywords_hi="|".join(kw_hi),
                    keywords_en="|".join(kw_en),
                )
            )

        db.commit()
        print(
            f"Seeded {len(DEPARTMENTS)} departments, {doctor_index} doctors, "
            f"{len(LAB_TESTS)} lab tests, {len(FAQ_ENTRIES)} FAQ topics."
        )
    finally:
        db.close()


if __name__ == "__main__":
    seed()
