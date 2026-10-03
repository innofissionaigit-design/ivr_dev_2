# ADDED BY SOURAV: keep the home-collection operational data (slots/collectors/payment policy)
# seeded from HERE, separate from seed.py and enquiry_migrate.py, so it is unambiguous that this
# is development/demo data populating the DATABASE -- the only thing agent/tools_client.py ever
# talks to is clinic-api's HTTP endpoints, which read this same DB, never this file or seed.py
# directly. All three tables below are BRAND NEW (Base.metadata.create_all() already creates
# them at startup, same as every other new table in this schema -- see enquiry_migrate.py's own
# module docstring), so this file only INSERTs rows, it never ALTERs a table.
"""Home-collection operational data: slots, collectors, and the payment policy.

Guarded the same way enquiry_migrate.add_enquiry_facts() is guarded: every insert checks for an
existing row first, so calling this on every boot is always safe and never duplicates data.

Deliberately derives its serviceable postal codes from HomeCollectionCoverage (the real
coverage table KCD-387 already established) rather than hard-coding "700091" a second time here
-- if the coverage table's serviceable postal codes ever change, this seeding logic does not need
to change with it.
"""

from __future__ import annotations

import datetime

from models import HomeCollectionCollector, HomeCollectionCoverage, HomeCollectionPaymentPolicy, HomeCollectionSlot
from sqlalchemy.orm import Session

# REASONED, not measured: a plausible small development roster/slot grid for the one
# serviceable area this prototype's coverage table already has. Swapping in real operational
# capacity numbers later changes only these seed rows, never home_collection_service.py's logic.
DEFAULT_SLOT_WINDOWS: tuple[tuple[str, str, int], ...] = (
    ("08:00", "10:00", 2),
    ("10:00", "12:00", 2),
    ("16:00", "18:00", 1),
)
SLOT_HORIZON_DAYS = 7
DEFAULT_COLLECTOR_NAMES: tuple[str, ...] = ("Ananya Das", "Partha Sarkar")


def seed_home_collection_operational_data(db: Session, today: datetime.date | None = None) -> dict:
    """Idempotent. Seeds (a) one payment policy row, (b) a roster of collectors covering every
    currently-serviceable postal code, and (c) a rolling window of slot rows for each such
    postal code, for the next SLOT_HORIZON_DAYS days. Returns a small summary dict for the
    startup log, the same shape enquiry_migrate's own seed functions already return."""
    today = today or datetime.date.today()
    added = {"payment_policies": 0, "collectors": 0, "slots": 0}

    if not db.query(HomeCollectionPaymentPolicy).first():
        db.add(
            HomeCollectionPaymentPolicy(
                version=1,
                effective_from=today.isoformat(),
                policy="pay_on_collection",
                description_bn="বাড়িতে নমুনা নেওয়ার সময় সরাসরি টাকা দিতে হবে। ফোনে কোনো পেমেন্ট নেওয়া হয় না।",
                description_hi="सैंपल लेने के समय ही सीधे भुगतान करना होगा। फ़ोन पर कोई भुगतान नहीं लिया जाता।",
                description_en="Payment is collected in person at the time of sample collection. "
                "No payment is taken over this call.",
            )
        )
        added["payment_policies"] += 1

    serviceable_codes = [
        row.postal_code for row in db.query(HomeCollectionCoverage).filter_by(serviceable=True).all()
    ]

    for code in serviceable_codes:
        if not db.query(HomeCollectionCollector).filter_by(postal_codes=code).first():
            for name in DEFAULT_COLLECTOR_NAMES:
                db.add(HomeCollectionCollector(name=name, postal_codes=code, active=True))
                added["collectors"] += 1

        for i in range(SLOT_HORIZON_DAYS):
            date_iso = (today + datetime.timedelta(days=i)).isoformat()
            for start, end, capacity in DEFAULT_SLOT_WINDOWS:
                exists = (
                    db.query(HomeCollectionSlot)
                    .filter_by(postal_code=code, date=date_iso, start_time=start, end_time=end)
                    .first()
                )
                if not exists:
                    db.add(
                        HomeCollectionSlot(
                            postal_code=code, date=date_iso, start_time=start, end_time=end, capacity=capacity
                        )
                    )
                    added["slots"] += 1

    db.commit()
    return added
