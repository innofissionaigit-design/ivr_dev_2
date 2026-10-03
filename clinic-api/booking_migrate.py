"""In-place migration for Epic E26's booking-lifecycle columns.

Same discipline as seed.py's add_i18n_columns(): ALTER TABLE for any
column an existing `appointments` row is missing, run BEFORE the first
ORM query on that table (SQLAlchemy selects every mapped column, so an
old database fails "no such column" on even a plain count()). Brand NEW
tables (SlotLock, Patient, PatientProxy, TestBooking, SmsOutbox,
DraftBooking, DepartmentRoute) need no migration at all --
Base.metadata.create_all(engine), already called at startup, creates a
missing table but never touches an existing one, so it is safe to call
every boot alongside this.
"""

from __future__ import annotations

from db import SessionLocal, engine
from sqlalchemy import inspect, text

APPOINTMENT_BOOKING_COLUMNS: dict[str, str] = {
    "patient_id": "INTEGER",
    "caller_phone": "VARCHAR",
    "status": "VARCHAR NOT NULL DEFAULT 'confirmed'",
    # TIMESTAMP, not DATETIME (CodeRabbit-flagged): SQLite treats both
    # identically (neither matches its INT/CHAR/TEXT/CLOB/BLOB/REAL/FLOA/
    # DOUB affinity rules, so both fall through to the same NUMERIC
    # affinity -- behaviourally a no-op change here), but DATETIME is not
    # a real PostgreSQL type, and db.py's own docstring states pointing
    # DATABASE_URL at a real Postgres is meant to be a one-line change.
    "cancelled_at": "TIMESTAMP",
    "cancellation_charge_inr": "INTEGER",
    "rescheduled_from_id": "INTEGER",
    "booking_group_id": "VARCHAR",
    "cancellation_policy_version": "INTEGER",
}

PATIENT_COLUMNS: dict[str, str] = {
    # KCD-084. TRUE/FALSE, not 1/0: PostgreSQL rejects an integer default on
    # a BOOLEAN column (same reason enquiry_migrate uses them).
    "senior_mode": "BOOLEAN NOT NULL DEFAULT FALSE",
}

DOCTOR_BOOKING_COLUMNS: dict[str, str] = {
    # Needed to STATE a cancellation charge (KCD-372) instead of inventing
    # one -- 50% of this, within the charging window, per
    # booking_service.CANCELLATION_CHARGE_FRACTION. Fictional, same
    # convention as every other price in seed.py.
    "consultation_fee_inr": "INTEGER NOT NULL DEFAULT 500",
    "full_name": "VARCHAR NOT NULL DEFAULT ''",
    "full_name_bn": "VARCHAR NOT NULL DEFAULT ''",
    "full_name_hi": "VARCHAR NOT NULL DEFAULT ''",
}


def add_appointment_booking_columns() -> list[str]:
    added = []
    insp = inspect(engine)
    have = {c["name"] for c in insp.get_columns("appointments")}
    with engine.begin() as conn:
        for col, decl in APPOINTMENT_BOOKING_COLUMNS.items():
            if col not in have:
                conn.execute(text(f"ALTER TABLE appointments ADD COLUMN {col} {decl}"))
                added.append(f"appointments.{col}")
    return added


def _has_blanket_unique_slot_index(conn) -> bool:
    """True if `appointments` still has a UNIQUE index over exactly
    (doctor_id, date, time_slot) that is NOT partial -- i.e. the old
    blanket constraint, regardless of what it happened to be named (a
    hand-built old-schema fixture may use a bare UNIQUE(...) column
    clause instead of a named CONSTRAINT; a real database created by an
    earlier version of models.py always named it "uq_doctor_slot", but
    this check does not rely on that name, only on the SHAPE, so it
    catches either). Uses PRAGMA index_list/index_info rather than
    string-matching sqlite_master's CREATE TABLE text, which is exact
    but fragile to phrasing differences that carry the same meaning."""
    rows = conn.execute(text("PRAGMA index_list('appointments')")).fetchall()
    for row in rows:
        # (seq, name, unique, origin, partial) -- column order is stable
        # across SQLite versions for this pragma, but accessed by name
        # via ._mapping to be explicit rather than trust positional index.
        m = row._mapping
        if not m["unique"] or m["partial"]:
            continue
        cols = [r._mapping["name"] for r in conn.execute(text(f"PRAGMA index_info('{m['name']}')")).fetchall()]
        if cols == ["doctor_id", "date", "time_slot"]:
            return True
    return False


def rebuild_appointments_partial_unique_index() -> bool:
    """Moves an EXISTING SQLite database off the old blanket
    UniqueConstraint("doctor_id", "date", "time_slot") the appointments
    table originally shipped with, onto the partial unique index (scoped
    to status='confirmed') models.Appointment now defines instead --
    CodeRabbit-flagged, real bug: the blanket constraint also blocked a
    CANCELLED slot's row from ever being rebooked, since a cancelled row
    still occupies that key; a second caller who legitimately holds and
    confirms the now-free slot hit an uncaught IntegrityError on the
    INSERT. See models.Appointment's own docstring.

    Idempotent (see _has_blanket_unique_slot_index) -- a no-op on a
    database that never had the blanket form, including every freshly
    create_all()'d one, since the model no longer defines it.

    MUST run after add_appointment_booking_columns(): the copy below
    references `status`, which does not exist yet on a database that
    predates Epic E26 entirely.

    SQLite cannot ALTER a table to drop or narrow a UNIQUE constraint,
    only rebuild it -- the standard recipe: rename the old table aside,
    let SQLAlchemy CREATE the new one directly from models.Appointment
    (so this migration can never drift from the model it targets), copy
    every row across by explicit column name, then drop the renamed
    original. All inside one transaction."""
    from models import Appointment

    insp = inspect(engine)
    if "appointments" not in insp.get_table_names():
        return False

    # CodeRabbit-flagged: everything below this point is SQLite-specific
    # (PRAGMA index_list/index_info, the rename-rebuild-drop recipe --
    # PostgreSQL supports neither statement and DOES support dropping/
    # replacing a constraint directly). An EXISTING PostgreSQL database
    # still ships the legacy blanket uq_doctor_slot CONSTRAINT (created by
    # an earlier version of models.py); create_all() never touches an
    # existing table, so switching DATABASE_URL to Postgres without this
    # branch would silently leave that constraint in place forever,
    # continuing to let a cancelled appointment block a real rebooking --
    # models.Appointment's own postgresql_where handles this only for a
    # BRAND NEW database, never an existing one.
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE appointments DROP CONSTRAINT IF EXISTS uq_doctor_slot"))
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_doctor_slot_confirmed "
                    "ON appointments (doctor_id, date, time_slot) "
                    "WHERE status = 'confirmed'"
                )
            )
        return True

    # This whole function runs on ONE connection in AUTOCOMMIT mode, not
    # engine.begin()'s usual transaction wrapper -- SQLite refuses to
    # change `PRAGMA foreign_keys` at all while a transaction is open
    # (silently a no-op, confirmed empirically), and db.py's own
    # on-connect handler sets foreign_keys=ON for every new connection,
    # so the toggle has to happen, and stick, on the specific connection
    # that then does the rename. The risky middle section (rename,
    # create, copy, drop) is wrapped in an explicit SQL transaction
    # issued as raw statements instead, so it is still all-or-nothing.
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        if not _has_blanket_unique_slot_index(conn):
            return False

        # Without foreign_keys=OFF, SQLite (3.25+) "helpfully" rewrites
        # every OTHER table's foreign key that points at `appointments`
        # (slot_locks, test_bookings, ...) to follow the RENAME below, so
        # they end up pointing at appointments_pre_partial_index instead
        # of the new `appointments` table created two statements later --
        # then DROP TABLE at the end leaves those foreign keys dangling.
        # legacy_alter_table restores the pre-3.25 rename behaviour on
        # top of that (belt and braces; foreign_keys=OFF alone is
        # sufficient in modern SQLite, but costs nothing to also set).
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        conn.execute(text("PRAGMA legacy_alter_table=ON"))
        try:
            conn.execute(text("BEGIN IMMEDIATE"))
            conn.execute(text("ALTER TABLE appointments RENAME TO appointments_pre_partial_index"))
            Appointment.__table__.create(conn)

            columns = ", ".join(Appointment.__table__.columns.keys())
            conn.execute(
                text(f"INSERT INTO appointments ({columns}) SELECT {columns} FROM appointments_pre_partial_index")
            )
            conn.execute(text("DROP TABLE appointments_pre_partial_index"))
            conn.execute(text("COMMIT"))
        except Exception:
            conn.execute(text("ROLLBACK"))
            raise
        finally:
            conn.execute(text("PRAGMA legacy_alter_table=OFF"))
            conn.execute(text("PRAGMA foreign_keys=ON"))
    return True


def add_patient_columns() -> list[str]:
    added = []
    insp = inspect(engine)
    if "patients" not in insp.get_table_names():
        return added
    have = {c["name"] for c in insp.get_columns("patients")}
    with engine.begin() as conn:
        for col, decl in PATIENT_COLUMNS.items():
            if col not in have:
                conn.execute(text(f"ALTER TABLE patients ADD COLUMN {col} {decl}"))
                added.append(f"patients.{col}")
    return added


CALL_RECORD_COLUMNS: dict[str, str] = {
    "satisfaction_score": "INTEGER",
    "satisfaction_band": "VARCHAR NOT NULL DEFAULT ''",
    "satisfaction_json": "TEXT NOT NULL DEFAULT '{}'",
}


def add_call_record_columns() -> list[str]:
    """The implicit-happiness columns on `call_records`. Idempotent; a database without the table yet has nothing to add
    (create_all() makes it with the columns)."""
    added = []
    insp = inspect(engine)
    if "call_records" not in insp.get_table_names():
        return added
    have = {c["name"] for c in insp.get_columns("call_records")}
    with engine.begin() as conn:
        for col, decl in CALL_RECORD_COLUMNS.items():
            if col not in have:
                conn.execute(text(f"ALTER TABLE call_records ADD COLUMN {col} {decl}"))
                added.append(f"call_records.{col}")
    return added


def add_doctor_booking_columns() -> list[str]:
    added = []
    insp = inspect(engine)
    have = {c["name"] for c in insp.get_columns("doctors")}
    with engine.begin() as conn:
        for col, decl in DOCTOR_BOOKING_COLUMNS.items():
            if col not in have:
                conn.execute(text(f"ALTER TABLE doctors ADD COLUMN {col} {decl}"))
                added.append(f"doctors.{col}")
    return added


# Department -> fictional consultation fee, clinically plausible variation
# by specialty, same spirit as seed.py's LAB_TESTS pricing.
DEPARTMENT_FEE_INR: dict[str, int] = {
    "General Medicine": 500,
    "Cardiology": 900,
    "Gynaecology & Obstetrics": 700,
    "Orthopaedics": 700,
    "ENT": 500,
    "Dermatology": 600,
    "Paediatrics": 500,
    "Diabetology & Endocrinology": 800,
}


def backfill_doctor_fees() -> int:
    """Fills consultation_fee_inr only where it is still at the raw ALTER
    TABLE default (500) *and* the department has a more specific fee --
    never overwrites a fee a clinician has since edited by hand."""
    from models import Department, Doctor

    db = SessionLocal()
    filled = 0
    try:
        depts = {d.id: d.name for d in db.query(Department).all()}
        for doc in db.query(Doctor).filter(Doctor.consultation_fee_inr == 500).all():
            fee = DEPARTMENT_FEE_INR.get(depts.get(doc.department_id))
            if fee and fee != 500:
                doc.consultation_fee_inr = fee
                filled += 1
        db.commit()
    finally:
        db.close()
    return filled


# (department name, [Bengali keyword cues], [Hindi keyword cues], [English keyword cues])
# Clinician-approved, fictional, same convention as seed.py's FAQ_ENTRIES --
# administrative routing only, never a diagnosis (see
# reply_templates.department_route_reply's framing).
DEPARTMENT_ROUTE_KEYWORDS: list[tuple[str, list[str], list[str], list[str]]] = [
    (
        "Cardiology",
        ["বুকে ব্যথা", "বুক ধড়ফড়", "হার্টের সমস্যা", "হৃদযন্ত্র"],
        ["सीने में दर्द", "दिल की धड़कन", "दिल की बीमारी"],
        ["chest pain", "heart palpitation", "heart problem"],
    ),
    (
        "Orthopaedics",
        # "হাতে ব্যথা"/"hand pain" is deliberately in BOTH this row and General Medicine's below: a caller saying
        # only "my hand/body hurts" (no fever, no clearly bone/joint framing) genuinely could mean either a GP visit
        # or an orthopaedic one -- route_department() already returns both as candidates when a query matches more
        # than one department's keywords, rather than guessing; this is what makes that happen here.
        ["হাড়ে ব্যথা", "গাঁটে ব্যথা", "কোমরে ব্যথা", "পিঠে ব্যথা", "হাতে ব্যথা", "পায়ে ব্যথা"],
        ["हड्डी में दर्द", "जोड़ों में दर्द", "कमर दर्द", "पीठ दर्द", "हाथ में दर्द", "पैर में दर्द"],
        ["bone pain", "joint pain", "back pain", "knee pain", "hand pain", "leg pain"],
    ),
    (
        "ENT",
        ["কানে ব্যথা", "গলা ব্যথা", "নাক বন্ধ", "শুনতে অসুবিধা"],
        ["कान में दर्द", "गले में दर्द", "नाक बंद"],
        ["ear pain", "throat pain", "sore throat", "blocked nose"],
    ),
    (
        "Dermatology",
        ["চামড়ায় সমস্যা", "ত্বকের সমস্যা", "চুলকানি", "র‍্যাশ"],
        ["त्वचा की समस्या", "खुजली", "चकत्ते"],
        ["skin problem", "itching", "rash"],
    ),
    (
        "Paediatrics",
        ["বাচ্চার জ্বর", "শিশুর সমস্যা", "বাচ্চার সর্দি"],
        ["बच्चे को बुखार", "बच्चे की समस्या"],
        ["child fever", "baby problem", "infant"],
    ),
    (
        "Gynaecology & Obstetrics",
        ["প্রেগন্যান্সি", "গর্ভাবস্থা", "মহিলাদের সমস্যা"],
        ["गर्भावस्था", "महिलाओं की समस्या"],
        ["pregnancy", "women's problem", "gynaecology"],
    ),
    (
        "Diabetology & Endocrinology",
        ["সুগারের সমস্যা", "ডায়াবেটিস", "থাইরয়েড সমস্যা"],
        ["शुगर की समस्या", "डायबिटीज़", "थायराइड"],
        ["diabetes", "sugar problem", "thyroid problem"],
    ),
    (
        "General Medicine",
        # See the Orthopaedics row above for why "হাতে ব্যথা"/"hand pain" and general body ache are listed here too.
        ["জ্বর", "সর্দি কাশি", "দুর্বলতা", "পেট খারাপ", "হাতে ব্যথা", "গায়ে ব্যথা", "শরীরে ব্যথা"],
        ["बुखार", "सर्दी खांसी", "कमज़ोरी", "पेट खराब", "हाथ में दर्द", "शरीर में दर्द"],
        ["fever", "cold and cough", "weakness", "stomach upset", "hand pain", "body ache"],
    ),
]


def seed_department_routes() -> int:
    """Idempotent: only inserts a department that has no route row yet, so
    a clinician's later edit to an existing row is never overwritten by a
    re-run -- same rule seed.backfill_i18n already applies to translated
    content."""
    from models import Department, DepartmentRoute

    db = SessionLocal()
    added = 0
    try:
        existing = {r.department_id for r in db.query(DepartmentRoute).all()}
        by_name = {d.name: d for d in db.query(Department).all()}
        for dept_name, kw_bn, kw_hi, kw_en in DEPARTMENT_ROUTE_KEYWORDS:
            dept = by_name.get(dept_name)
            if not dept or dept.id in existing:
                continue
            db.add(
                DepartmentRoute(
                    department_id=dept.id,
                    keywords_bn="|".join(kw_bn),
                    keywords_hi="|".join(kw_hi),
                    keywords_en="|".join(kw_en),
                )
            )
            added += 1
        db.commit()
    finally:
        db.close()
    return added


def backfill_department_route_keywords() -> int:
    """Adds any keyword from DEPARTMENT_ROUTE_KEYWORDS an EXISTING route row does not already have -- additive only,
    appended after what is already there, never removing or reordering it, so a clinician's own edit (including a
    deliberate removal) is never put back. Same discipline as backfill_doctor_fees(); a department with no route row
    yet is left to seed_department_routes(), which must run first."""
    from models import Department, DepartmentRoute

    db = SessionLocal()
    updated = 0
    try:
        by_name = {d.name: d for d in db.query(Department).all()}
        rows = {r.department_id: r for r in db.query(DepartmentRoute).all()}
        for dept_name, kw_bn, kw_hi, kw_en in DEPARTMENT_ROUTE_KEYWORDS:
            dept = by_name.get(dept_name)
            row = rows.get(dept.id) if dept else None
            if row is None:
                continue
            changed = False
            for attr, new_kws in (("keywords_bn", kw_bn), ("keywords_hi", kw_hi), ("keywords_en", kw_en)):
                have = [w for w in getattr(row, attr).split("|") if w]
                missing = [w for w in new_kws if w not in have]
                if missing:
                    setattr(row, attr, "|".join(have + missing))
                    changed = True
            updated += changed
        db.commit()
    finally:
        db.close()
    return updated


def migrate_booking_schema() -> dict:
    """The column-ALTER half only. Run at startup BEFORE any ORM query --
    see main.py's startup handler -- because SQLAlchemy selects every
    mapped column, so a query against Doctor/Appointment needs these to
    already exist. Safe to call every boot.

    Deliberately does NOT call seed_department_routes()/
    backfill_doctor_fees() here: both need `departments`/`doctors` rows to
    already exist, which is not yet true on a brand-new database at this
    point in startup -- seed() has not run yet. Call
    finish_booking_schema_setup() below once seeding is done instead. A
    version of this function that called them together shipped once and
    silently seeded zero department routes on every fresh database,
    caught by tests/test_booking_endpoints.py::test_department_route_endpoint
    failing against a throwaway DB."""
    columns_added = (
        add_appointment_booking_columns()
        + add_doctor_booking_columns()
        + add_patient_columns()
        + add_call_record_columns()
    )
    # Must run AFTER add_appointment_booking_columns(): see this
    # function's own docstring for why.
    rebuilt = rebuild_appointments_partial_unique_index()
    return {"columns_added": columns_added, "appointments_table_rebuilt": rebuilt}


def seed_default_cancellation_policy() -> bool:
    """KCD-488: without at least one CancellationPolicy row, a versioned
    lookup at cancellation time would have nothing to find -- this backfills
    version 1 with exactly the values booking_service's old flat constants
    used to hardcode (24h free window, 50% charge), so behaviour is
    unchanged the moment this ships. Idempotent -- only inserts when the
    table is empty, so a clinician's later addition of version 2+ (or an
    edit to version 1) is never touched by a re-run, same rule
    seed_department_routes() already follows."""
    from models import CancellationPolicy

    db = SessionLocal()
    try:
        if db.query(CancellationPolicy).count() > 0:
            return False
        db.add(
            CancellationPolicy(
                version=1,
                effective_from="2020-01-01",
                free_window_hours=24,
                charge_percent=50,
                refund_eligible=True,
            )
        )
        db.commit()
        return True
    finally:
        db.close()


def backfill_doctor_full_names() -> int:
    """Fills `doctors.full_name`, `full_name_bn` and `full_name_hi` where still empty, from seed.DOCTOR_FULL_NAMES (keyed
    by the short name). Idempotent, and never overwrites a name a clinician has since entered. Returns how many values
    were filled."""
    from models import Doctor
    from seed import doctor_full_names

    db = SessionLocal()
    try:
        n = 0
        names = doctor_full_names()
        for d in db.query(Doctor).all():
            latin, bn, hi = names.get(d.name, ("", "", ""))
            for column, value in (("full_name", latin), ("full_name_bn", bn), ("full_name_hi", hi)):
                if value and not getattr(d, column):
                    setattr(d, column, value)
                    n += 1
        db.commit()
        return n
    finally:
        db.close()


def finish_booking_schema_setup() -> dict:
    """The half that needs `departments`/`doctors` to already have rows --
    call this AFTER seed() or backfill_i18n() has run, not before."""
    routes_added = seed_department_routes()
    routes_backfilled = backfill_department_route_keywords()
    fees_filled = backfill_doctor_fees()
    policy_seeded = seed_default_cancellation_policy()
    return {
        "department_routes_added": routes_added,
        "department_route_keywords_backfilled": routes_backfilled,
        "doctor_fees_filled": fees_filled,
        "cancellation_policy_seeded": policy_seeded,
        "doctor_full_names_filled": backfill_doctor_full_names(),
    }
