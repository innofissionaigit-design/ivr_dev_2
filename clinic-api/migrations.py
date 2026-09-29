"""Versioned schema migrations for an EXISTING clinic database.

E4-S4 "Caller cancels an appointment". See docs/stories/E4-S4-cancel-plan.md.

create_all() creates missing tables but never adds a column to, or changes a
constraint on, a table that already exists. The live /workspace/clinic.db
predates `appointments.status`, so without this it would keep the old
full-table uq_doctor_slot constraint and a cancelled appointment would hold
its slot for ever.

Deliberately small: each step runs once, in its own transaction, and is
recorded in `schema_migrations`. A fresh database (seed.py's drop_all +
create_all) already has the current schema, and upgrade() only records the
steps as applied. Alembic (E0-S4) is the long-term home for this; each step
below translates one-to-one into an Alembic revision.
"""
from __future__ import annotations

import datetime
import logging

from sqlalchemy import inspect, text
from sqlalchemy.schema import CreateIndex, CreateTable

log = logging.getLogger("clinic-api.migrations")


def _appointments_has_status(conn) -> bool:
    return "status" in {c["name"] for c in inspect(conn).get_columns("appointments")}


def _0001_appointment_status_sqlite(conn) -> None:
    """SQLite cannot drop an inline UNIQUE constraint, so the table is
    rebuilt (sqlite.org/lang_altertable.html, "other kinds of table schema
    changes"). Foreign keys are switched off for the rebuild by the caller:
    appointment_changes / notification_outbox rows point at appointments.id,
    and the ids are copied unchanged."""
    from models import Appointment

    table = Appointment.__table__
    ddl = str(CreateTable(table).compile(dialect=conn.dialect)).strip()
    head = "CREATE TABLE appointments "
    if not ddl.startswith(head):
        raise RuntimeError(f"unexpected DDL shape: {ddl[:40]!r}")
    conn.exec_driver_sql(ddl.replace(head, "CREATE TABLE appointments_new ", 1))
    shared = [c.name for c in table.columns if c.name not in ("status", "cancelled_at")]
    cols = ", ".join(shared)
    conn.exec_driver_sql(
        f"INSERT INTO appointments_new ({cols}, status, cancelled_at) "
        f"SELECT {cols}, 'active', NULL FROM appointments")
    conn.exec_driver_sql("DROP TABLE appointments")
    conn.exec_driver_sql("ALTER TABLE appointments_new RENAME TO appointments")
    for index in table.indexes:  # uq_doctor_slot_active, on the renamed table
        conn.execute(CreateIndex(index))
    broken = conn.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if broken:
        raise RuntimeError(f"foreign key check failed after rebuild: {broken[:3]}")


def _0001_appointment_status_generic(conn) -> None:
    conn.execute(text("ALTER TABLE appointments ADD COLUMN status VARCHAR NOT NULL DEFAULT 'active'"))
    conn.execute(text("ALTER TABLE appointments ADD COLUMN cancelled_at TIMESTAMP NULL"))
    conn.execute(text("ALTER TABLE appointments DROP CONSTRAINT uq_doctor_slot"))
    conn.execute(text(
        "CREATE UNIQUE INDEX uq_doctor_slot_active ON appointments (doctor_id, date, time_slot) "
        "WHERE status = 'active'"))


MIGRATIONS = ("0001_appointment_status",)


def upgrade(engine) -> list[str]:
    """Apply every pending step. Returns the versions applied by THIS call
    (empty when already current). Safe to run on every startup."""
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version VARCHAR PRIMARY KEY, applied_at TIMESTAMP NOT NULL)"))
        done = {row[0] for row in conn.execute(text("SELECT version FROM schema_migrations"))}

    applied = []
    for version in MIGRATIONS:
        if version in done:
            continue
        sqlite = engine.dialect.name == "sqlite"
        with engine.connect() as conn:
            if sqlite:
                # A no-op inside a transaction, so set it and end the
                # transaction SQLAlchemy opened for it before starting ours.
                conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
                conn.commit()
            try:
                with conn.begin():
                    # Left behind if an earlier attempt died mid-rebuild.
                    conn.exec_driver_sql("DROP TABLE IF EXISTS appointments_new")
                    needed = (inspect(conn).has_table("appointments")
                              and not _appointments_has_status(conn))
                    if needed:
                        (_0001_appointment_status_sqlite if sqlite
                         else _0001_appointment_status_generic)(conn)
                    conn.execute(text("INSERT INTO schema_migrations (version, applied_at) "
                                      "VALUES (:v, :t)"),
                                 {"v": version, "t": datetime.datetime.now()})
            finally:
                if sqlite:
                    conn.exec_driver_sql("PRAGMA foreign_keys=ON")
                    conn.commit()
        log.info("migration %s %s", version,
                 "applied" if needed else "recorded (schema already current)")
        applied.append(version)
    return applied
