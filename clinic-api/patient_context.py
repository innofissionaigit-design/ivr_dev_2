"""Epic E33: patient context and history -- the read model behind "the agent already
knows what I booked here", and the audit trail behind "and only tells me".

WHAT THIS IS NOT. It is not a health record and does not compete with one. The
hospital's systems of record (HIS for visits, LIS for tests performed and reports)
are reached through connectors that do not exist yet (KCD-131, external); until they
do, `timeline()` says so in its `sources` block instead of pretending the local
mirror is the whole picture. Nothing here stores or returns a result value, a
reading or an interpretation -- there is no column for one -- and `_assert_no_results`
refuses to emit an event that carries such a key, so a future connector cannot leak
one through this layer by accident.

WHAT IT ENFORCES.
  * every read of a patient's history is authorised (booking_service.authorize_disclosure:
    self, or a verified proxy) and AUDITED, attributed to the call that made it -- and
    every refused attempt is audited too (KCD-495);
  * before verification the only thing this service will say about a number is that a
    record EXISTS: identify() returns a status and a count, never a name (KCD-494/495);
  * ambiguity is never resolved by taking the most likely candidate: a shared number,
    or a name that fits two patients, comes back as "ambiguous" for the caller to
    settle (KCD-494/497);
  * every field of every event carries its source and the time it was read
    (KCD-493): the caller of this service can always say where a fact came from and
    how old it is, or say that it cannot see it (KCD-500).

The authorisation here is phone-based, exactly as strong as the booking lookups it
mirrors: until real caller ID reaches this API, "the number the caller said" is what
is checked. The voice agent adds its own gate in front (agent/identity.py): nothing
personal is asked for until a verification has passed.
"""

from __future__ import annotations

import datetime
import json
import unicodedata

import registry as reg
from booking_service import NOT_PROVIDED_PHONE, _now, authorize_disclosure, find_draft
from models import (
    Appointment,
    CallRecord,
    Doctor,
    HistoryAudit,
    LabReport,
    LabTest,
    Patient,
    PatientMedicine,
    PatientPreference,
    RetestInterval,
    SmsOutbox,
    TestBooking,
    TestPerformance,
)
from phonetic_match import phonetic_key, phonetic_match
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# Keys that would make an event a clinical fact. None may pass through this layer.
FORBIDDEN_KEYS = frozenset(
    {
        "result",
        "results",
        "value",
        "values",
        "reading",
        "readings",
        "interpretation",
        "finding",
        "findings",
        "diagnosis",
        "report_text",
        "outcome_value",
        "impression",
    }
)
# A history field older than this is STALE: spoken history must say it cannot confirm
# rather than state it. REASONED, not measured; the clinic decides the real figure.
STALE_AFTER_HOURS = 24
EXTERNAL_SOURCES = ("his", "lis")  # not connected until KCD-131

CONFIRMED_ALLOWLIST = {
    "doctor_name",
    "date",
    "time_slot",
    "test_name",
    "test_names",
    "confirmation_id",
    "new_date",
    "new_time_slot",
    "relationship",
    "department",
}
DELIVERY_CHANNELS = {"sms", "whatsapp", "voice"}
ACCESSIBILITY_MODES = {"slower", "shorter", "none"}
LANGUAGES = {"bn", "hi", "en"}


def _a1(csv: str | None) -> str | None:
    """The first alias in a '|'-separated list (the caller's own script), or None."""
    return next((a for a in (csv or "").split("|") if a.strip()), None)


def _iso(x) -> str | None:
    return x.isoformat() if isinstance(x, (datetime.date, datetime.datetime)) else x


# =============================================================== KCD-501: call records


def redact_confirmed(slots: dict) -> dict:
    """What of a booking's confirmed values goes into a call record: the allow-listed
    fields only, and a phone number as its last four digits. A name and an address
    are personal data this record does not need in order to be auditable."""
    out = {k: v for k, v in (slots or {}).items() if k in CONFIRMED_ALLOWLIST and v not in (None, "")}
    for key in ("phone", "contact_phone"):
        v = (slots or {}).get(key)
        if v and v != NOT_PROVIDED_PHONE and len(str(v)) >= 4:
            out[f"{key}_last4"] = str(v)[-4:]
    return out


def _load(row: CallRecord) -> dict:
    return {
        "call_id": row.call_id,
        "channel": row.channel,
        "patient_ref": row.patient_id,
        "started_at": _iso(row.started_at),
        "ended_at": _iso(row.ended_at),
        "languages": [x for x in row.languages.split(",") if x],
        "intents": json.loads(row.intents_json),
        "actions": json.loads(row.actions_json),
        "confirmed": json.loads(row.confirmed_json),
        "history_statements": json.loads(row.history_statements_json),
        "escalation_reason": row.escalation_reason,
        "outcome": row.outcome,
        "disclosure_version": row.disclosure_version,
        "last_seq": row.last_seq,
    }


def get_call_record(db: Session, call_id: str) -> dict | None:
    row = db.query(CallRecord).filter_by(call_id=call_id).first()
    return _load(row) if row else None


def _new_record(db: Session, call_id: str, caller_phone: str | None) -> CallRecord:
    row = CallRecord(call_id=call_id, caller_phone=caller_phone, started_at=_now(), updated_at=_now())
    db.add(row)
    try:
        db.commit()
    except IntegrityError:  # a concurrent writer created it first: use theirs
        db.rollback()
        row = db.query(CallRecord).filter_by(call_id=call_id).one()
    return row


_BANDS = ("happy", "neutral", "unhappy", "very_unhappy", "unscored")


def clean_satisfaction(payload: dict) -> dict:
    """What is kept of a satisfaction event: the score, the band, the reasons and the counts -- and nothing else. Anything
    that is not a number, a flag or a short label is dropped, so free text can never be stored through this door."""
    score = payload.get("score")
    score = (
        int(score) if isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 100 else None
    )
    band = payload.get("band") if payload.get("band") in _BANDS else ("unscored" if score is None else "")
    reasons = [
        [str(r[0])[:40], int(r[1])]
        for r in (payload.get("reasons") or [])[:20]
        if isinstance(r, (list, tuple)) and len(r) == 2 and isinstance(r[1], (int, float))
    ]
    signals = {
        str(k)[:40]: v
        for k, v in (payload.get("signals") or {}).items()
        if isinstance(v, (bool, int, float)) or v is None
    }
    return {
        "score": score,
        "band": band,
        "reasons": reasons,
        "reason": str(payload.get("reason") or "")[:40],
        "version": str(payload.get("version") or "")[:40],
        "basis": "behaviour",
        "signals": signals,
    }


def record_call_event(
    db: Session, call_id: str, seq: int, kind: str, payload: dict | None = None, caller_phone: str | None = None
) -> dict:
    """Apply one event to the call's record. IDEMPOTENT: an event whose `seq` has
    already been applied is acknowledged and ignored, so a retried request (a dropped
    response, a duplicate delivery) cannot double an action or an intent. Events are
    applied as they happen, so a call that drops still records what was completed."""
    payload = payload or {}
    row = db.query(CallRecord).filter_by(call_id=call_id).first() or _new_record(db, call_id, caller_phone)
    if seq <= row.last_seq:
        return {"applied": False, "duplicate": True, "last_seq": row.last_seq}
    if caller_phone and not row.caller_phone:
        row.caller_phone = caller_phone
    if kind == "start":
        row.channel = payload.get("channel", row.channel)
        row.disclosure_version = payload.get("disclosure_version", row.disclosure_version)
    elif kind == "language":
        langs = [x for x in row.languages.split(",") if x]
        if payload.get("language") and payload["language"] not in langs:
            langs.append(payload["language"])
        row.languages = ",".join(langs)
    elif kind == "intent":
        intents = json.loads(row.intents_json)
        if payload.get("intent") and (not intents or intents[-1] != payload["intent"]):
            intents.append(payload["intent"])
        row.intents_json = json.dumps(intents)
    elif kind == "action":
        actions = json.loads(row.actions_json)
        actions.append({"name": payload.get("name"), "ref": payload.get("ref"), "at": _iso(_now())})
        row.actions_json = json.dumps(actions)
        reg.audit(
            db,
            "action:" + str(payload.get("name")),
            "ok",
            call_id=call_id,
            patient_id=row.patient_id,
            ref=payload.get("ref"),
        )
    elif kind == "confirmed":
        merged = json.loads(row.confirmed_json)
        merged.update(redact_confirmed(payload.get("slots") or {}))
        row.confirmed_json = json.dumps(merged)
    elif kind == "history":
        stmts = json.loads(row.history_statements_json)
        for s in payload.get("statements", []):
            if s not in stmts:
                stmts.append(s)
        row.history_statements_json = json.dumps(stmts)
    elif kind == "patient":
        row.patient_id = payload.get("patient_ref")
    elif kind == "escalation":
        row.escalation_reason = payload.get("reason")
        reg.audit(db, "escalation", "ok", call_id=call_id, patient_id=row.patient_id, reason=payload.get("reason"))
    elif kind == "satisfaction":
        cleaned = clean_satisfaction(payload)
        row.satisfaction_score, row.satisfaction_band = cleaned["score"], cleaned["band"]
        row.satisfaction_json = json.dumps(cleaned, ensure_ascii=False)
    elif kind == "end":
        row.ended_at = _now()
        row.outcome = payload.get("outcome", "completed")
    else:
        return {"applied": False, "duplicate": False, "error": f"unknown event kind {kind!r}"}
    row.last_seq = seq
    row.updated_at = _now()
    db.commit()
    if kind == "end":
        # KCD-501: what the call did becomes durable history, its summary is cached for a day
        # (KCD-496) and the close is audited. Idempotent, so a retried `end` changes nothing.
        reg.finalize_call(db, row)
    return {"applied": True, "duplicate": False, "last_seq": row.last_seq}


# ===================================================== KCD-494: identify without assuming


def identify(db: Session, phone: str) -> dict:
    """What can be said about a contact number BEFORE anyone has verified anything:
    that a record exists, and whether it is one person or several. No name, no age."""
    if not phone or phone == NOT_PROVIDED_PHONE:
        return {"status": "new", "record_exists": False, "count": 0}
    rows = db.query(Patient).filter(Patient.phone == phone).all()
    if not rows:
        reg.audit(db, "identify", "ok", result="new", phone_last4=phone[-4:])
        return {"status": "new", "record_exists": False, "count": 0}
    if len(rows) == 1:
        reg.audit(db, "identify", "ok", patient_id=rows[0].id, result="single", phone_last4=phone[-4:])
        return {"status": "single", "record_exists": True, "count": 1, "patient_ref": rows[0].id}
    reg.audit(db, "identify", "ok", result="ambiguous", count=len(rows), phone_last4=phone[-4:])
    return {"status": "ambiguous", "record_exists": True, "count": len(rows)}


def _name_tokens(name: str) -> list[str]:
    return [unicodedata.normalize("NFC", t).lower() for t in (name or "").replace(".", " ").split() if len(t) > 1]


# A phonetic key shorter than this is too collapsed to mean anything: the folding drops
# vowels, so "Zoya", "Asha" and "Saha" all reduce to the single class "2". Measured on
# this codebase's own matcher; such names match only EXACTLY.
MIN_PHONETIC_KEY_LEN = 3


def _name_fit(spoken: str, stored: str) -> str | None:
    """'exact' when every spoken token IS a stored token; 'phonetic' when every one
    merely SOUNDS like one (and its key is long enough to carry information); None
    otherwise. Phonetic is a SUGGESTION -- callers must confirm it, never act on it."""
    spoken_t, stored_t = _name_tokens(spoken), _name_tokens(stored)
    if not spoken_t:
        return None
    if all(t in stored_t for t in spoken_t):
        return "exact"
    if all(
        len(phonetic_key(t)) >= MIN_PHONETIC_KEY_LEN and any(phonetic_match(t, u) for u in stored_t) for t in spoken_t
    ):
        return "phonetic"
    return None


def resolve_named(db: Session, phone: str, spoken_name: str, age: int | None = None) -> dict:
    """A shared number: the caller says WHO the call is for (an open question; the
    candidates are never listed, since listing them would disclose who is on the
    number). One fit is a match; several are ambiguous; none is none. When several fit
    and an age was given, the age narrows them -- it never picks among equals.

    EXACT name matches are used if there are any; only when there are none do
    phonetic ones count, and then the answer says `basis: "phonetic"` and the caller
    must confirm it with the patient before relying on it."""
    scored = [(p, _name_fit(spoken_name, p.name)) for p in db.query(Patient).filter(Patient.phone == phone).all()]
    exact = [p for p, basis in scored if basis == "exact"]
    rows, basis = (exact, "exact") if exact else ([p for p, bs_ in scored if bs_ == "phonetic"], "phonetic")
    if age is not None and len(rows) > 1:
        narrowed = [p for p in rows if p.age == age]
        rows = narrowed or rows
    if not rows:
        return {"status": "none"}
    if len(rows) == 1:
        return {"status": "single", "patient_ref": rows[0].id, "basis": basis}
    return {"status": "ambiguous", "count": len(rows), "basis": basis}


# ==================================================== KCD-493/495/500: the timeline


def _assert_no_results(events: list[dict]) -> None:
    for e in events:
        bad = FORBIDDEN_KEYS & {k.lower() for k in e.get("fields", {})}
        if bad:
            raise ValueError(f"event {e.get('id')} carries clinical result keys {sorted(bad)}: refused")


def _local_events(db: Session, patient: Patient, now: datetime.datetime) -> list[dict]:
    as_of = _iso(now)
    ev: list[dict] = []
    for a in db.query(Appointment).filter(Appointment.patient_id == patient.id).all():
        doctor = db.get(Doctor, a.doctor_id)
        ev.append(
            {
                "id": f"appointment:{a.id}",
                "kind": "appointment",
                "at": a.date,
                "source": "local",
                "as_of": as_of,
                "fields": {
                    "confirmation_id": a.confirmation_id,
                    "doctor_name": doctor.name if doctor else None,
                    "doctor_name_bn": _a1(doctor.aliases_bn) if doctor else None,
                    "doctor_name_hi": _a1(doctor.aliases_hi) if doctor else None,
                    "date": a.date,
                    "time_slot": a.time_slot,
                    "status": a.status,
                },
            }
        )
    tb = db.query(TestBooking).filter(TestBooking.patient_id == patient.id).all()
    for t in tb:
        ev.append(
            {
                "id": f"test_booking:{t.id}",
                "kind": "test_booking",
                "at": t.date,
                "source": "local",
                "as_of": as_of,
                "fields": {
                    "confirmation_id": t.confirmation_id,
                    "test_name": t.lab_test.name if t.lab_test else None,
                    "date": t.date,
                    "status": t.status,
                },
            }
        )
    for p in db.query(TestPerformance).filter(TestPerformance.patient_id == patient.id).all():
        test = db.get(LabTest, p.lab_test_id)
        ev.append(
            {
                "id": f"test_performed:{p.id}",
                "kind": "test_performed",
                "at": p.performed_on,
                "source": p.source,
                "as_of": as_of,
                "fields": {
                    "test_name": test.name if test else None,
                    "test_name_bn": _a1(test.aliases_bn) if test else None,
                    "test_name_hi": _a1(test.aliases_hi) if test else None,
                    "performed_on": p.performed_on,
                },
            }
        )
    for m in db.query(PatientMedicine).filter(PatientMedicine.patient_id == patient.id).all():
        ev.append(
            {
                "id": f"medicine_prescribed:{m.id}",
                "kind": "medicine_prescribed",
                "at": m.prescribed_on,
                "source": m.source,
                "as_of": as_of,
                "fields": {
                    "medicine_name": m.medicine.name if m.medicine else None,
                    "medicine_name_bn": _a1(m.medicine.aliases_bn) if m.medicine else None,
                    "medicine_name_hi": _a1(m.medicine.aliases_hi) if m.medicine else None,
                    "prescribed_on": m.prescribed_on,
                },
            }
        )
    for cid in {t.confirmation_id for t in tb}:
        for r in db.query(LabReport).filter(LabReport.confirmation_id == cid, LabReport.status == "ready").all():
            ev.append(
                {
                    "id": f"report_ready:{r.id}",
                    "kind": "report_ready",
                    "at": _iso(r.ready_at),
                    "source": "local",
                    "as_of": as_of,
                    "fields": {"confirmation_id": cid, "ready_at": _iso(r.ready_at)},
                }
            )
    for c in (
        db.query(CallRecord)
        .filter((CallRecord.patient_id == patient.id) | (CallRecord.caller_phone == patient.phone))
        .all()
    ):
        ev.append(
            {
                "id": f"contact:{c.id}",
                "kind": "contact",
                "at": _iso(c.started_at),
                "source": "local",
                "as_of": as_of,
                "fields": {"channel": c.channel, "outcome": c.outcome, "intents": json.loads(c.intents_json)[-3:]},
            }
        )
    return ev


def _audit(db: Session, call_id: str, patient_id: int | None, kind: str, fields: list[str]) -> None:
    db.add(
        HistoryAudit(
            call_id=call_id or "unknown", patient_id=patient_id, kind=kind, fields_json=json.dumps(fields), at=_now()
        )
    )
    db.commit()
    reg.audit(
        db,
        "history:" + kind,
        "denied" if kind.endswith("denied") or "unverified" in kind else "ok",
        call_id=call_id or "unknown",
        patient_id=patient_id,
        fields=fields,
    )


def timeline(db: Session, patient_id: int, caller_phone: str, call_id: str, verified: bool | None = None) -> dict:
    """The patient's timeline, assembled on request from the local records (never kept
    as a second copy), with every event carrying its source and as-of time and the
    sources that could NOT be read named as such.

    `verified` is the server's own answer to "has this call passed the security questions for
    this patient" (registry.is_verified). The HTTP endpoint always passes it; None means the caller
    is an in-process user of the older phone-based rule, which is kept for them unchanged."""
    patient = db.get(Patient, patient_id)
    if patient is None:
        return {"success": False, "reason": "not_found"}
    if verified is False:
        _audit(db, call_id, patient_id, "timeline_denied_unverified", [])
        return {"success": False, "reason": "not_verified"}
    if verified is None and (not caller_phone or not authorize_disclosure(db, patient, caller_phone)):
        _audit(db, call_id, patient_id, "timeline_denied", [])
        return {"success": False, "reason": "not_authorized"}
    now = _now()
    events = sorted(_local_events(db, patient, now), key=lambda e: (e["at"] or "", e["id"]), reverse=True)
    _assert_no_results(events)
    _audit(db, call_id, patient_id, "timeline_read", sorted({e["kind"] for e in events}))
    return {
        "success": True,
        "patient_ref": patient_id,
        "as_of": _iso(now),
        "events": events,
        "sources": {"local": "ok", **{s: "not_connected" for s in EXTERNAL_SOURCES}},
    }


def is_stale(as_of_iso: str | None, now: datetime.datetime | None = None) -> bool:
    if not as_of_iso:
        return True
    return ((now or _now()) - datetime.datetime.fromisoformat(as_of_iso)) > datetime.timedelta(hours=STALE_AFTER_HOURS)


# ======================================================= KCD-498: test history, no advice


def test_status(db: Session, patient_id: int, lab_test_id: int, today: datetime.date | None = None) -> dict:
    """WHEN a test was last performed and whether it is DUE -- and nothing about what
    it showed. "Due" exists only where a clinician approved an interval; with none,
    `due` is None and the agent may state the date but says nothing about need."""
    today = today or _now().date()
    last = (
        db.query(TestPerformance)
        .filter_by(patient_id=patient_id, lab_test_id=lab_test_id)
        .order_by(TestPerformance.performed_on.desc())
        .first()
    )
    interval = db.query(RetestInterval).filter_by(lab_test_id=lab_test_id).first()
    if last is None:
        return {"known": False, "last_performed_on": None, "due": None, "interval_days": None}
    if interval is None:
        return {"known": True, "last_performed_on": last.performed_on, "due": None, "interval_days": None}
    due_on = datetime.date.fromisoformat(last.performed_on) + datetime.timedelta(days=interval.interval_days)
    return {
        "known": True,
        "last_performed_on": last.performed_on,
        "due": today >= due_on,
        "interval_days": interval.interval_days,
        "due_on": due_on.isoformat(),
        "approved_by": interval.approved_by,
    }


# ============================================================ KCD-499: preferences


def get_preferences(db: Session, patient_id: int, caller_phone: str) -> dict:
    patient = db.get(Patient, patient_id)
    if patient is None or not caller_phone or not authorize_disclosure(db, patient, caller_phone):
        return {"success": False, "reason": "not_authorized"}
    row = db.query(PatientPreference).filter_by(patient_id=patient_id).first()
    if row is None:
        return {"success": True, "preferences": {}}
    return {
        "success": True,
        "preferences": {
            k: getattr(row, k)
            for k in ("branch", "collection_address", "delivery_channel", "accessibility_mode", "language_bias")
            if getattr(row, k) is not None
        },
        "updated_at": _iso(row.updated_at),
    }


def set_preferences(db: Session, patient_id: int, caller_phone: str, **fields) -> dict:
    patient = db.get(Patient, patient_id)
    if patient is None or not caller_phone or not authorize_disclosure(db, patient, caller_phone):
        return {"success": False, "reason": "not_authorized"}
    allowed = {
        "branch": None,
        "collection_address": None,
        "delivery_channel": DELIVERY_CHANNELS,
        "accessibility_mode": ACCESSIBILITY_MODES,
        "language_bias": LANGUAGES,
    }
    clean = {}
    for k, v in fields.items():
        if k not in allowed:
            return {"success": False, "reason": f"unknown_field:{k}"}
        if v in (None, ""):
            continue
        if allowed[k] is not None and v not in allowed[k]:
            return {"success": False, "reason": f"invalid_value:{k}"}
        if isinstance(v, str) and len(v) > 200:
            return {"success": False, "reason": f"too_long:{k}"}
        clean[k] = v
    row = db.query(PatientPreference).filter_by(patient_id=patient_id).first()
    if row is None:
        row = PatientPreference(patient_id=patient_id, updated_at=_now())
        db.add(row)
    for k, v in clean.items():
        setattr(row, k, v)
    row.updated_at = _now()
    db.commit()
    return {"success": True, "stored": sorted(clean)}


# ==================================================== KCD-496: continuity across calls


def recent_interactions(
    db: Session, caller_phone: str, limit: int = 3, exclude_call_id: str | None = None
) -> list[dict]:
    """The last `limit` interactions on this number across voice and messaging, with the
    intent and outcome of each -- the ONLY thing said about them before verification is
    that they exist; the content is for the agent to offer, never to assume."""
    items: list[dict] = []
    q = db.query(CallRecord).filter(CallRecord.caller_phone == caller_phone)
    if exclude_call_id:
        q = q.filter(CallRecord.call_id != exclude_call_id)
    for c in q.order_by(CallRecord.started_at.desc()).limit(limit).all():
        intents = json.loads(c.intents_json)
        items.append(
            {
                "at": _iso(c.started_at),
                "channel": c.channel,
                "intent": intents[-1] if intents else None,
                "outcome": c.outcome,
            }
        )
    for m in (
        db.query(SmsOutbox)
        .filter(SmsOutbox.to_phone == caller_phone)
        .order_by(SmsOutbox.created_at.desc())
        .limit(limit)
        .all()
    ):
        items.append({"at": _iso(m.created_at), "channel": "sms", "intent": m.template_key, "outcome": m.status})
    items.sort(key=lambda i: i["at"] or "", reverse=True)
    return items[:limit]


def continuity(
    db: Session, caller_phone: str, exclude_call_id: str | None = None, patient_id: int | None = None
) -> dict:
    """Is there something unfinished, and what were the last interactions? The unfinished
    booking's DETAILS are returned separately (`draft`) and are for the agent to speak
    only after verification; before it the agent may say only that something was left
    unfinished."""
    draft = find_draft(db, caller_phone)
    interactions = recent_interactions(db, caller_phone, 3, exclude_call_id)
    # KCD-496: the last calls kept for ONE DAY -- what the next call is greeted with. Older calls are
    # in patient_history and are not used as context.
    cached = reg.cached_context(db, patient_id, caller_phone, exclude_call_id)
    unfinished = draft is not None or any(c.get("unfinished") for c in cached[:1])
    return {"unfinished": unfinished, "draft": draft, "interactions": interactions, "cached": cached}


# ====================================================== KCD-497: find a booking without a number


def _within(date_iso: str, approx: str, window: int) -> bool:
    try:
        return abs((datetime.date.fromisoformat(date_iso) - datetime.date.fromisoformat(approx)).days) <= window
    except ValueError:
        return False


def search_bookings(
    db: Session,
    caller_phone: str,
    *,
    phone: str | None = None,
    name: str | None = None,
    approx_date: str | None = None,
    date_window_days: int = 3,
    test_name: str | None = None,
    branch: str | None = None,
) -> dict:
    """Find bookings by whatever the caller remembers: contact number, name (phonetic, any
    script), approximate date, test name. Any workable combination narrows; one match is a
    match; SEVERAL come back as `several` with what distinguishes them, for the agent to ask
    -- never the most likely one taken. Rows the caller is not authorised for are dropped
    silently (their existence is not disclosed). `branch` is accepted and reported as
    ignored: this schema has no branch to filter on (that data arrives with KCD-131)."""
    if not any([phone, name, approx_date, test_name]):
        return {"status": "insufficient", "matches": [], "ignored_filters": ["branch"] if branch else []}
    q = db.query(Appointment).filter(Appointment.status == "confirmed")
    if phone:
        q = q.filter((Appointment.phone == phone) | (Appointment.caller_phone == phone))
    rows = q.all()
    cands = []
    for a in rows:
        if test_name:
            continue  # an appointment with a doctor is not a test booking
        basis = _name_fit(name, a.patient_name) if name else None
        if name and basis is None:
            continue
        if approx_date and not _within(a.date, approx_date, date_window_days):
            continue
        patient = db.get(Patient, a.patient_id) if a.patient_id else None
        if not patient or not caller_phone or not authorize_disclosure(db, patient, caller_phone):
            continue
        doctor = db.get(Doctor, a.doctor_id)
        cands.append(
            {
                "kind": "appointment",
                "confirmation_id": a.confirmation_id,
                "date": a.date,
                "time_slot": a.time_slot,
                "doctor_name": doctor.name if doctor else None,
                "test_name": None,
                "name_basis": basis,
            }
        )
    tq = db.query(TestBooking).filter(TestBooking.status == "confirmed")
    if phone:
        tq = tq.filter((TestBooking.phone == phone) | (TestBooking.caller_phone == phone))
    for t in tq.all():
        names = [t.lab_test.name] + [
            x for x in (t.lab_test.aliases_bn or "").split("|") + (t.lab_test.aliases_hi or "").split("|") if x
        ]
        if test_name and not any(phonetic_match(test_name, n) or test_name.lower() in n.lower() for n in names):
            continue
        basis = _name_fit(name, t.patient_name) if name else None
        if name and basis is None:
            continue
        if approx_date and not _within(t.date, approx_date, date_window_days):
            continue
        patient = db.get(Patient, t.patient_id) if t.patient_id else None
        if not patient or not caller_phone or not authorize_disclosure(db, patient, caller_phone):
            continue
        cands.append(
            {
                "kind": "test_booking",
                "confirmation_id": t.confirmation_id,
                "date": t.date,
                "time_slot": None,
                "doctor_name": None,
                "test_name": t.lab_test.name,
                "name_basis": basis,
            }
        )
    # one row per confirmation id (a multi-test booking is one booking)
    seen, matches = set(), []
    for c in cands:
        if c["confirmation_id"] not in seen:
            seen.add(c["confirmation_id"])
            matches.append(c)
    ignored = ["branch"] if branch else []
    if not matches:
        return {"status": "none", "matches": [], "ignored_filters": ignored}
    if len(matches) == 1:
        # a single match found only by SOUND is a suggestion: the caller confirms it
        return {
            "status": "single",
            "matches": matches,
            "ignored_filters": ignored,
            "needs_confirmation": matches[0].get("name_basis") == "phonetic",
        }
    distinguishing = [f for f in ("date", "doctor_name", "test_name", "time_slot") if len({m[f] for m in matches}) > 1]
    return {"status": "several", "matches": matches, "distinguishing": distinguishing, "ignored_filters": ignored}


# ============================================== KCD-382b: registered/booked-patient preparation


def booking_test_names(db: Session, confirmation_id: str) -> list[str]:
    """Every real, confirmed lab-test name in ONE test-booking group (KCD-382b).

    search_bookings() above deliberately collapses a multi-test booking to ONE representative
    candidate per confirmation_id -- "a multi-test booking is one booking", so disambiguation
    never double-counts it as several. That is correct for FINDING which booking the caller
    means, but once the caller's single intended confirmation_id is already resolved (by
    search_bookings, already authorised against this caller), KCD-382's preparation path needs
    the FULL set of tests inside it, not just the one representative name -- so this reads every
    TestBooking row for that id directly.

    No separate authorization check here: a confirmation_id is already the bearer-token
    identifier this codebase's other booking lookups treat as sufficient on its own (see
    lookup_bookings()'s own docstring in booking_service.py), and the only caller of this
    function passes an id that was already returned by an authorised search_bookings() call in
    the same turn -- never one typed in from outside that flow.

    Confirmed rows only: a cancelled test within an otherwise-confirmed booking group must never
    reappear in a preparation answer. Order is insertion order (the order the tests were booked
    in), not alphabetical -- deterministic, but not a claim about clinical priority."""
    rows = (
        db.query(TestBooking)
        .filter(TestBooking.confirmation_id == confirmation_id, TestBooking.status == "confirmed")
        .order_by(TestBooking.id)
        .all()
    )
    return [r.lab_test.name for r in rows if r.lab_test]
