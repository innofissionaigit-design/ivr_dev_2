"""The patient registry and the SECURITY QUESTIONS (KCD-495, KCD-497, KCD-512), the call-outcome
history and one-day cache (KCD-496, KCD-501), and the audit log.

WHAT COUNTS AS VERIFIED. A caller is verified for a patient when at least TWO different facts they
gave match the registry AND at least one of the two is a strong one:

    patient id      the id on the patient's card                       strong
    date of birth   compared as a date                                 strong
    full name       every part of the registered name was said         weak (also used to find the record)
    address         two distinctive words, or the pincode + one word   weak

The agent asks for them in that order of usefulness and stops as soon as the rule is met. The
answers are checked HERE, on the server that holds the registry, so nothing in the voice agent can
decide a caller is verified: it can only submit answers and read the yes or no. The reply never says
WHICH answer was wrong (that would turn the questions into an oracle), a wrong pair counts as an
attempt, three attempts in a call lock the check for that call, and more than six failures for one
patient in 24 hours lock it across calls (a brute force spread over many calls is still a brute
force). Every attempt -- pass, fail, lock -- is written to the audit log.

A passed check is stored (`VerificationSession`) and is what authorises a history read: the timeline
endpoint refuses a call that has no live verification for that patient.

Nothing clinical is here. The registry holds identity facts; the medicine and test tables hold THAT
something was prescribed or done and when, never what it showed or what to do about it.
"""

from __future__ import annotations

import datetime
import json
import re
import unicodedata

from booking_service import NOT_PROVIDED_PHONE, _now
from models import (
    AuditLog,
    CallRecord,
    Patient,
    PatientCallCache,
    PatientHistory,
    PatientRegistry,
    VerificationSession,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

MAX_ATTEMPTS_PER_CALL = 3
MAX_FAILED_PER_PATIENT_24H = 6
VERIFICATION_TTL_MIN = 15
CACHE_TTL_HOURS = 24
SENIOR_AGE = 60  # India's definition of a senior citizen
_WRITE_INTENTS = frozenset(
    {"book_appointment", "book_test", "reschedule_appointment", "cancel_appointment", "add_test_booking"}
)
STRONG_FACTORS = frozenset({"patient_id", "dob"})
FACTORS = ("patient_id", "dob", "name", "address")
MIN_ADDRESS_TOKEN_LEN = 3
_ADDRESS_STOPWORDS = frozenset(
    {
        "road",
        "street",
        "lane",
        "near",
        "opposite",
        "behind",
        "flat",
        "floor",
        "block",
        "house",
        "apartment",
        "kolkata",
        "calcutta",
        "west",
        "bengal",
        "india",
        "the",
        "and",
        "para",
        "nagar",
        "pin",
        "pincode",
        "code",
        "রোড",
        "লেন",
        "কলকাতা",
        "পশ্চিমবঙ্গ",
        "সড়ক",
        "ফ্ল্যাট",
        "বাড়ি",
        "নম্বর",
        "रोड",
        "गली",
        "कोलकाता",
        "पश्चिम",
        "बंगाल",
        "मकान",
        "नंबर",
        "फ्लैट",
    }
)


# ------------------------------------------------------------------------------------- audit


def audit(
    db: Session,
    action: str,
    outcome: str = "ok",
    *,
    call_id: str | None = None,
    patient_id: int | None = None,
    actor: str = "agent",
    **detail,
) -> None:
    db.add(
        AuditLog(
            at=_now(),
            call_id=call_id,
            patient_id=patient_id,
            actor=actor,
            action=action,
            outcome=outcome,
            detail_json=json.dumps(detail, default=str),
        )
    )
    db.commit()


# ------------------------------------------------------------------------------- normalising


def _nfc(s: str | None) -> str:
    return unicodedata.normalize("NFC", s or "").lower()


def _tokens(s: str | None) -> list[str]:
    # Python's \w does not match the vowel signs and virama of Devanagari and Bengali (they are combining
    # marks), so a plain \w split would cut every Indic word into pieces; the two Indic blocks are kept whole.
    return [t for t in re.split(r"[^\wऀ-ॿঀ-৿]+", _nfc(s), flags=re.UNICODE) if t]


def _uid(s: str | None) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", s or "").upper()


def _uid_matches(spoken: str | None, stored: str) -> bool:
    """A spoken patient id matches the registered one when the letters and digits are the same, or --
    for a caller who reads only the digits off the card -- when the spoken digits are the whole numeric
    part (five or more of them, so a short number cannot match by accident)."""
    a, b = _uid(spoken), _uid(stored)
    if not a:
        return False
    if a == b:
        return True
    return a.isdigit() and len(a) >= 5 and b.endswith(a) and b[: len(b) - len(a)].isalpha()


def age_years(dob_iso: str, today: datetime.date | None = None) -> int | None:
    try:
        d = datetime.date.fromisoformat(dob_iso)
    except (TypeError, ValueError):
        return None
    t = today or _now().date()
    return t.year - d.year - ((t.month, t.day) < (d.month, d.day))


def _aliases(s: str | None) -> list[str]:
    return [a for a in (s or "").split("|") if a.strip()]


# ------------------------------------------------------------------------ the four factors


def _name_matches(spoken: str | None, patient: Patient, reg: PatientRegistry) -> bool:
    """Every part of a registered name (or of one of its aliases) was said. The reverse direction
    on purpose: "my name is Asha Saha" contains "asha" and "saha", and a filler word cannot fail it;
    "Asha" alone does not contain "saha", so a first name alone is not a full name."""
    said = set(_tokens(spoken))
    if not said:
        return False
    for stored in [patient.name] + _aliases(reg.name_aliases):
        parts = [t for t in _tokens(stored) if len(t) > 1]
        if parts and all(p in said for p in parts):
            return True
    return False


def _address_tokens(text: str | None) -> set[str]:
    return {
        t
        for t in _tokens(text)
        if len(t) >= MIN_ADDRESS_TOKEN_LEN and t not in _ADDRESS_STOPWORDS and not (t.isdigit() and len(t) != 6)
    }


def _address_matches(spoken: str | None, reg: PatientRegistry) -> bool:
    said = _address_tokens(spoken)
    if not said:
        return False
    stored: set[str] = set()
    for text in [reg.address_text, reg.locality] + _aliases(reg.address_aliases):
        stored |= _address_tokens(text)
    words_shared = {t for t in said & stored if not t.isdigit()}
    pin_ok = bool(reg.pincode) and reg.pincode in said
    return len(words_shared) >= 2 or (pin_ok and len(words_shared) >= 1)


def factor_matches(db: Session, patient: Patient, reg: PatientRegistry, answers: dict) -> set[str]:
    """Which of the caller's answers match. Internal: never returned to the caller."""
    hit = set()
    if answers.get("patient_id") and _uid_matches(answers["patient_id"], reg.patient_uid):
        hit.add("patient_id")
    if answers.get("dob") and str(answers["dob"]).strip() == reg.dob:
        hit.add("dob")
    if answers.get("name") and _name_matches(answers["name"], patient, reg):
        hit.add("name")
    if answers.get("address") and _address_matches(answers["address"], reg):
        hit.add("address")
    return hit


def _provided(answers: dict) -> dict:
    return {k: v for k, v in (answers or {}).items() if k in FACTORS and v not in (None, "")}


# ------------------------------------------------------------------------------ verification


def _session(db: Session, call_id: str, patient_id: int) -> VerificationSession:
    row = db.query(VerificationSession).filter_by(call_id=call_id, patient_id=patient_id).first()
    if row is None:
        row = VerificationSession(call_id=call_id, patient_id=patient_id)
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            row = db.query(VerificationSession).filter_by(call_id=call_id, patient_id=patient_id).one()
    return row


def _recent_failures(db: Session, patient_id: int) -> int:
    since = _now() - datetime.timedelta(hours=24)
    return (
        db.query(AuditLog)
        .filter(
            AuditLog.action == "verify",
            AuditLog.outcome.in_(("failed", "locked")),
            AuditLog.patient_id == patient_id,
            AuditLog.at >= since,
        )
        .count()
    )


def is_verified(db: Session, call_id: str, patient_id: int) -> bool:
    row = db.query(VerificationSession).filter_by(call_id=call_id, patient_id=patient_id).first()
    return bool(row and row.verified_at and not row.locked and row.expires_at and row.expires_at > _now())


def verify(db: Session, call_id: str, patient_id: int, answers: dict, caller_phone: str | None = None) -> dict:
    """Check the caller's answers against the registry. See the module docstring for the rule."""
    patient = db.get(Patient, patient_id)
    reg = db.query(PatientRegistry).filter_by(patient_id=patient_id).first() if patient else None
    if patient is None or reg is None or reg.status != "active":
        audit(
            db,
            "verify",
            "failed",
            call_id=call_id,
            patient_id=patient_id if patient else None,
            reason="no_registry_record",
        )
        return {"verified": False, "reason": "cannot_verify", "attempts_left": 0, "locked": True}
    sess = _session(db, call_id, patient_id)
    if sess.locked or _recent_failures(db, patient_id) >= MAX_FAILED_PER_PATIENT_24H:
        if not sess.locked:
            sess.locked = True
            db.commit()
        audit(db, "verify", "locked", call_id=call_id, patient_id=patient_id, reason="attempt_limit")
        return {"verified": False, "reason": "locked", "attempts_left": 0, "locked": True}
    if is_verified(db, call_id, patient_id):
        return {
            "verified": True,
            "attempts_left": MAX_ATTEMPTS_PER_CALL - sess.attempts,
            "locked": False,
            **_age_block(reg),
        }
    given = _provided(answers)
    if len(given) < 2:
        # one fact is never enough, and evaluating it would only leak whether it was right
        return {
            "verified": False,
            "need_more": True,
            "attempts_left": MAX_ATTEMPTS_PER_CALL - sess.attempts,
            "locked": False,
        }
    matched = factor_matches(db, patient, reg, given)
    sess.attempts += 1
    if len(matched) >= 2 and matched & STRONG_FACTORS:
        sess.verified_at = _now()
        sess.expires_at = sess.verified_at + datetime.timedelta(minutes=VERIFICATION_TTL_MIN)
        sess.factors_json = json.dumps(sorted(matched))
        db.commit()
        audit(
            db, "verify", "ok", call_id=call_id, patient_id=patient_id, factors=sorted(matched), attempt=sess.attempts
        )
        return {
            "verified": True,
            "attempts_left": MAX_ATTEMPTS_PER_CALL - sess.attempts,
            "locked": False,
            **_age_block(reg),
        }
    if sess.attempts >= MAX_ATTEMPTS_PER_CALL:
        sess.locked = True
    db.commit()
    audit(
        db,
        "verify",
        "locked" if sess.locked else "failed",
        call_id=call_id,
        patient_id=patient_id,
        factors_given=sorted(given),
        attempt=sess.attempts,
    )
    return {"verified": False, "attempts_left": max(0, MAX_ATTEMPTS_PER_CALL - sess.attempts), "locked": sess.locked}


def _age_block(reg: PatientRegistry) -> dict:
    """Computed on the server from the registered date of birth, after verification (KCD-512)."""
    a = age_years(reg.dob)
    return {"age_years": a, "is_senior": bool(a is not None and a >= SENIOR_AGE)}


# ---------------------------------------------------------------------------- find (KCD-497)


def find_by_details(db: Session, call_id: str, answers: dict) -> dict:
    """Find a patient from what the caller remembers when the phone number found nobody: a patient
    id, or a date of birth with a name. Returns only an opaque reference -- never a name, a date or
    a list -- and a caller who has to be told "ambiguous" is asked for one more fact, not shown who
    the candidates are. Finding is not verifying: `verify` still has to pass."""
    given = _provided(answers)
    ref = None
    candidates: list[Patient] = []
    if given.get("patient_id"):
        reg = next(
            (r for r in db.query(PatientRegistry).all() if _uid_matches(given["patient_id"], r.patient_uid)), None
        )
        if reg is not None and reg.status == "active":
            candidates = [db.get(Patient, reg.patient_id)]
    elif given.get("dob") and given.get("name"):
        for reg in db.query(PatientRegistry).filter_by(dob=str(given["dob"]).strip(), status="active").all():
            p = db.get(Patient, reg.patient_id)
            if p is not None and _name_matches(given["name"], p, reg):
                candidates.append(p)
    else:
        audit(db, "find", "failed", call_id=call_id, reason="insufficient_details", given=sorted(given))
        return {"status": "insufficient"}
    if not candidates:
        audit(db, "find", "failed", call_id=call_id, given=sorted(given))
        return {"status": "none"}
    if len(candidates) > 1:
        audit(db, "find", "ok", call_id=call_id, result="ambiguous", given=sorted(given))
        return {"status": "ambiguous"}
    ref = candidates[0].id
    audit(db, "find", "ok", call_id=call_id, patient_id=ref, result="single", given=sorted(given))
    return {"status": "single", "patient_ref": ref}


# ---------------------------------------------------------------- call outcome -> history + cache


def finalize_call(db: Session, row: CallRecord) -> None:
    """The call has closed: turn what it did into durable history (KCD-501), cache its summary for a
    day (KCD-496) and audit the close. Idempotent (unique keys), so a retried `end` changes nothing."""
    actions = json.loads(row.actions_json)
    intents = json.loads(row.intents_json)
    if row.patient_id:
        for a in actions:
            _history_once(
                db, row.patient_id, row.call_id, a.get("name") or "action", a.get("ref") or "", {"at": a.get("at")}
            )
        _history_once(
            db,
            row.patient_id,
            row.call_id,
            "call_outcome",
            "",
            {"outcome": row.outcome, "intents": intents[-5:], "escalation": row.escalation_reason},
        )
    summary = {
        "intents": intents[-3:],
        "outcome": row.outcome,
        "escalation": row.escalation_reason,
        "actions": [{"name": a.get("name"), "ref": a.get("ref")} for a in actions[-3:]],
        "unfinished": row.outcome in ("abandoned", "failed")
        or (bool(intents) and not actions and intents[-1] in _WRITE_INTENTS),
    }
    now = _now()
    if not db.query(PatientCallCache).filter_by(call_id=row.call_id).first():
        db.add(
            PatientCallCache(
                patient_id=row.patient_id,
                caller_phone=row.caller_phone,
                call_id=row.call_id,
                summary_json=json.dumps(summary),
                created_at=now,
                expires_at=now + datetime.timedelta(hours=CACHE_TTL_HOURS),
            )
        )
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
    db.query(PatientCallCache).filter(PatientCallCache.expires_at < now).delete()
    db.commit()
    audit(
        db,
        "call_closed",
        "ok",
        call_id=row.call_id,
        patient_id=row.patient_id,
        call_outcome=row.outcome,
        actions=len(actions),
        escalation=row.escalation_reason,
    )


def _history_once(db: Session, patient_id: int, call_id: str, kind: str, ref: str, detail: dict) -> None:
    if db.query(PatientHistory).filter_by(call_id=call_id, kind=kind, ref=ref).first():
        return
    db.add(
        PatientHistory(
            patient_id=patient_id,
            call_id=call_id,
            kind=kind,
            ref=ref,
            detail_json=json.dumps(detail, default=str),
            at=_now(),
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()


def cached_context(
    db: Session, patient_id: int | None, caller_phone: str | None, exclude_call_id: str | None = None
) -> list[dict]:
    """The last calls (at most three) of this patient or number that are still inside their day.
    Expired rows are never returned."""
    now = _now()
    q = db.query(PatientCallCache).filter(PatientCallCache.expires_at > now)
    conds = []
    if patient_id is not None:
        conds.append(PatientCallCache.patient_id == patient_id)
    if caller_phone and caller_phone != NOT_PROVIDED_PHONE:
        conds.append(PatientCallCache.caller_phone == caller_phone)
    if not conds:
        return []
    from sqlalchemy import or_

    q = q.filter(or_(*conds))
    if exclude_call_id:
        q = q.filter(PatientCallCache.call_id != exclude_call_id)
    out = []
    for c in q.order_by(PatientCallCache.created_at.desc()).limit(3).all():
        s = json.loads(c.summary_json)
        out.append({"at": c.created_at.isoformat(), "expires_at": c.expires_at.isoformat(), **s})
    return out
