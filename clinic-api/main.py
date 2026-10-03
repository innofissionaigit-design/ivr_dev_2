"""Clinic data service -- implements the exact 3-endpoint contract
agent/tools_client.py in the voice agent already expects. Backed by
PostgreSQL, seeded with dummy departments/doctors/schedules/tests via
seed.py.

Resolving a name is deliberately simple and exact (written-form containment, see _find_test and
_find_doctor). "Who might the caller have meant" -- the suggestions read back to a caller who mispronounced
a test or a doctor -- comes from a phonetic-fold gazetteer with an index (gazetteer.py, KCD-096), built once
per catalogue and looked up, not scanned per request. A suggestion is never a resolution.
"""

from __future__ import annotations

import datetime
import difflib
import json
import logging
import os
import secrets
import time
import unicodedata

import agent_messages as am
import booking_service as bs
import enquiry_service as eq
import gazetteer as gz
# ADDED BY SOURAV: the full home-collection visit flow (slots/hold/confirm/booking/assignment) --
# see home_collection_service.py's own module docstring for why this is a separate service module
# from booking_service.py/enquiry_service.py rather than folded into either.
import home_collection_service as hcs
import patient_context as pc
import registry as reg
from db import SessionLocal, get_db
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from idempotency import idempotent
from idempotency import set_key_from_header as set_idempotency_key
from models import FAQ, Appointment, AuditLog, Department, Doctor, DoctorSchedule, LabTest
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

app = FastAPI(title="Kolkata Care Diagnostics -- Clinic Data API (dummy)")


@app.middleware("http")
async def require_service_token(request: Request, call_next):
    """Service-to-service authentication for everything under /api/v1/ (an external review: patient
    data was served to anyone who could reach the port).

    Enforced when CLINIC_API_TOKEN is set, or when CLINIC_API_REQUIRE_TOKEN=1 (then a missing token
    refuses every request -- fail closed). With neither, the API runs open as before and logs that
    at startup; that is only acceptable on a closed dev network. This is a shared secret between
    the orchestrator and this API: it authenticates the CALLER OF THE API, not the phone caller,
    and is not a substitute for verifying a patient's identity (agent/identity.py). Read per
    request so it can be rotated without a code change; compared in constant time."""
    token = os.environ.get("CLINIC_API_TOKEN", "")
    set_idempotency_key(request.headers.get("idempotency-key"))  # for the write endpoints (idempotency.py)
    if request.url.path.startswith("/api/v1/") and (token or os.environ.get("CLINIC_API_REQUIRE_TOKEN") == "1"):
        supplied = request.headers.get("authorization", "")
        if not token or not secrets.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


SLOT_STEP_MIN = 15


@app.on_event("startup")
def _warn_if_open():
    if not os.environ.get("CLINIC_API_TOKEN") and os.environ.get("CLINIC_API_REQUIRE_TOKEN") != "1":
        logging.getLogger("clinic-api").warning(
            "CLINIC_API_TOKEN is not set: /api/v1/* is served WITHOUT authentication. "
            "Acceptable only on a closed development network."
        )


@app.on_event("startup")
def _ensure_seeded():
    """Create the schema, and seed it only if it is EMPTY.

    The catalogue is derived data -- 8 departments, 32 doctors, 34 tests,
    all defined in seed.py -- so regenerating it costs nothing and removes
    the manual reseed step that every pod restart used to require.

    Guarded on emptiness because seed() itself is destructive (drop_all
    then create_all). Running it unconditionally at startup would wipe
    every appointment booked since the last boot, turning a convenience
    into data loss.
    """
    from db import engine
    from models import Base, LabTest

    Base.metadata.create_all(engine)
    from seed import add_i18n_columns

    add_i18n_columns()  # before ANY ORM query: an old database lacks the new columns

    # Epic E26 (booking/reschedule/cancel): new columns on `appointments`
    # and `doctors`, and the department-routing seed table. MUST also run
    # before any ORM query that touches those tables -- same reason as
    # add_i18n_columns above, and it caught a REAL bug here: backfill_i18n()
    # below queries Doctor, which SQLAlchemy selects every mapped column
    # of, so it crashed on a database migrated only as far as
    # add_i18n_columns(). create_all() already created every brand-new
    # table (SlotLock, Patient, PatientProxy, TestBooking, SmsOutbox,
    # DraftBooking, DepartmentRoute) a moment ago; this only ALTERs what
    # already existed.
    from booking_migrate import migrate_booking_schema

    logging.getLogger("clinic-api").info("booking schema migration: %s", migrate_booking_schema())

    # Epic E27 (information and enquiry): new columns on `lab_tests`.
    # Column-ALTER only, same "before any ORM query" reasoning, same
    # class of bug if this ran after the LabTest.count() query below
    # instead of before it.
    from enquiry_migrate import add_callback_request_columns, add_enquiry_columns

    logging.getLogger("clinic-api").info("enquiry schema migration: %s", add_enquiry_columns())
    logging.getLogger("clinic-api").info("callback request schema migration: %s", add_callback_request_columns())

    db = SessionLocal()
    try:
        if db.query(LabTest).count() == 0:
            logging.getLogger("clinic-api").info("empty database -- seeding catalogue")
            from seed import seed

            seed()
        else:
            logging.getLogger("clinic-api").info("catalogue already present, not reseeding")
            from seed import backfill_i18n

            logging.getLogger("clinic-api").info("i18n backfill: %s", backfill_i18n(db))

        # Only now do departments/doctors/tests definitely have rows --
        # either seed() just created them, or they already existed.
        # Calling either of these any earlier seeds zero rows on a
        # brand-new database, the exact bug finish_booking_schema_setup's
        # own docstring documents.
        from booking_migrate import finish_booking_schema_setup

        logging.getLogger("clinic-api").info("booking schema setup (routes/fees): %s", finish_booking_schema_setup())
        from patient_seed import seed_patient_context

        logging.getLogger("clinic-api").info("agent messages: %s inserted", am.seed_defaults(db))
        if os.environ.get("CLINIC_SEED_SAMPLE_PATIENTS", "1") == "1":
            logging.getLogger("clinic-api").info("sample patient data: %s", seed_patient_context(db))
        from enquiry_migrate import backfill_enquiry_facts, seed_enquiry_demo_data

        logging.getLogger("clinic-api").info("enquiry facts backfill: %s", backfill_enquiry_facts(db))
        logging.getLogger("clinic-api").info("enquiry demo data: %s", seed_enquiry_demo_data(db))

        # ADDED BY SOURAV: home-collection slots/collectors/payment-policy are operational data,
        # not catalogue data -- seeded here, every boot, same idempotent-insert discipline as the
        # enquiry demo data just above, and derived from HomeCollectionCoverage's own serviceable
        # rows rather than a second hard-coded postal-code list.
        from home_collection_migrate import seed_home_collection_operational_data

        logging.getLogger("clinic-api").info(
            "home collection operational data: %s", seed_home_collection_operational_data(db)
        )
    finally:
        db.close()


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    return {
        "status": "ok",
        "departments": db.query(Department).count(),
        "doctors": db.query(Doctor).count(),
        "lab_tests": db.query(LabTest).count(),
    }


# =============================================================================
# Tool 1: GET /api/v1/tests/search?name=...
# =============================================================================
def _first_alias_bn(aliases_bn: str) -> str | None:
    """The spoken form. The reply the caller HEARS is synthesized by a
    Bengali-only tokenizer that silently drops Latin script, so returning
    only `t.name` ("Uric Acid") means the caller is read a price with the
    test name missing from the sentence. Every row is seeded with at least
    one Bengali alias for exactly this reason -- see seed.py."""
    for alias in (aliases_bn or "").split("|"):
        if alias.strip():
            return alias.strip()
    return None


def _test_reply_dict(t: LabTest) -> dict:
    return {
        "found": True,
        "test_name": t.name,
        "test_name_bn": _first_alias_bn(t.aliases_bn),
        "test_name_hi": _first_alias_bn(t.aliases_hi),
        "rate_inr": t.rate_inr,
        "sample_type": t.sample_type,
        "report_time_hours": t.report_time_hours,
    }


@app.get("/api/v1/catalogue")
def catalogue(db: Session = Depends(get_db)):
    """Every test and doctor with their Bengali aliases, in one call.

    Exists for the voice agent's deterministic fast path: matching a
    caller's words against a 74-row catalogue is a local string operation,
    but only if the caller HAS the catalogue. Fetching it once at startup
    turns "which test did they say" from a 7B-model inference into a
    microsecond comparison -- see agent/fast_path.py.
    """
    return {
        "tests": [
            {
                "name": t.name,
                "aliases_bn": [a for a in (t.aliases_bn or "").split("|") if a],
                "aliases_hi": [a for a in (t.aliases_hi or "").split("|") if a],
                # The lay-term fast path (agent/lay_terms.py) reads this to answer "রক্ত পরীক্ষা" ("blood test") with
                # the REAL list of blood tests, off the same cached catalogue fetch every other lookup already uses --
                # never a second endpoint, never a hardcoded list that could drift from what the lab actually runs.
                "sample_type": t.sample_type,
            }
            for t in db.query(LabTest).all()
        ],
        "doctors": [
            {
                "name": d.name,
                "full_name": d.full_name or d.name,
                "full_name_bn": d.full_name_bn,
                "full_name_hi": d.full_name_hi,
                "surname": d.name.split()[-1],
                "aliases_bn": [a for a in (d.aliases_bn or "").split("|") if a],
                "aliases_hi": [a for a in (d.aliases_hi or "").split("|") if a],
            }
            for d in db.query(Doctor).all()
        ],
        # FAQ topics + their keyword sets, for FastPath.FAQCatalogue --
        # same "fetch the small table once, match locally" shape as tests
        # and doctors above. NOT the answer text itself: the answer is
        # still fetched live via /api/v1/faq on every turn, the same
        # discipline test_rate and doctor_availability already apply, so a
        # cached routing decision can never serve a stale FAQ answer.
        "faq_topics": [
            {
                "topic": f.topic,
                "keywords_bn": [k for k in (f.keywords_bn or "").split("|") if k],
                "keywords_hi": [k for k in (f.keywords_hi or "").split("|") if k],
                "keywords_en": [k for k in (f.keywords_en or "").split("|") if k],
            }
            for f in db.query(FAQ).all()
        ],
    }


# Words that carry no test identity when a caller wraps a test name in a
# sentence: "lipid profile TEST" must match "Lipid Profile". Bengali/Hindi
# spellings of "test" included.
_GENERIC_WORDS = {"test", "tests", "the", "a", "of", "for", "price", "rate", "টেস্ট", "টেস্টের", "टेस्ट", "जांच"}


def _norm_name(s: str) -> str:
    """Comparison form for a caller-said test name.

    Lowercases, drops generic words, and removes the Devanagari nukta
    (U+093C): the Hindi ASR emits both "प्रोफाइल" and "प्रोफ़ाइल" for the same
    spoken word, and NFC keeps the nukta as a separate mark, so removing it
    makes the two spellings equal. Never fuzzy -- a wrong test's real price is
    the failure this file exists to prevent, so a near-miss is offered as a
    suggestion to the caller, not silently accepted."""
    s = unicodedata.normalize("NFC", s).lower().replace("\u093c", "").replace("-", " ")
    toks = [t.strip("?.,;:!'\"()") for t in s.split()]
    return " ".join(t for t in toks if t and t not in _GENERIC_WORDS)


def _find_test(db: Session, name: str) -> LabTest | None:
    """The exact / alias / containment cascade search_test() and the prep endpoint both need --
    factored out so "which test did they mean" has exactly one implementation.

    RESOLVES ONLY ON A WRITTEN-FORM MATCH. A sound-alike is never a resolution: "CBC" and "CRP"
    are one phoneme apart, and a price or a preparation instruction for the wrong one is a
    confident wrong answer that the database will happily supply. Phonetic matches come back
    only as suggestions (`_phonetic_test_matches`, `_test_suggestions`) for the caller to confirm."""
    exact = db.query(LabTest).filter(func.lower(LabTest.name).contains(name.lower())).first()
    if exact:
        return exact

    all_tests = db.query(LabTest).all()
    for t in all_tests:
        aliases = [a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            return t

    # Normalised, two-way containment against the English name and every
    # alias: the caller's phrase may contain the test name ("lipid profile
    # test") or be a part of it, in any script.
    q = _norm_name(name)
    if q:
        for t in all_tests:
            forms = [t.name] + [a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a]
            for form in forms:
                n = _norm_name(form)
                if n and (q in n or n in q):
                    return t

    return None


# ------------------------------------------------------------ the gazetteer (KCD-096)

# The gazetteer is built from every row once and reused; this is how long a built one is trusted before the
# catalogue is read again. A change in the NUMBER of rows rebuilds it at once (see _catalogue_stamp); an edit to
# a single alias shows up within this many seconds. REASONED: catalogue edits are rare and a suggestion that lags
# an alias edit by half a minute costs nothing, whereas rebuilding on every request would be the scan again.
GAZETTEER_TTL_S = 30.0
_gazetteers: dict[tuple, tuple[float, tuple, gz.Gazetteer]] = {}


def invalidate_gazetteers() -> None:
    """Drop every built gazetteer. Call after writing catalogue rows or aliases so the next lookup sees them."""
    _gazetteers.clear()


def _catalogue_stamp(db: Session, model) -> tuple:
    return (db.query(func.count(model.id)).scalar(), db.query(func.max(model.id)).scalar())


def _test_forms(t: LabTest) -> list[str]:
    """Every way a test is written or said: its name, the name without its bracket, the bracketed code
    ("Complete Blood Count (CBC)" is called "CBC"), and each Bengali and Hindi alias."""
    forms = [t.name]
    if "(" in t.name and ")" in t.name:
        forms += [t.name.split("(")[0].strip(), t.name.split("(")[1].split(")")[0].strip()]
    forms += [a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a.strip()]
    return [f for f in forms if f]


def _doctor_forms(d: Doctor) -> list[str]:
    """A doctor is called by surname or by an alias; the initials in the formal name ("Dr. S. Mukherjee")
    are not something a caller says and only add noise to a similarity comparison."""
    forms = [d.name.split()[-1]] + [a for a in (d.aliases_bn + "|" + (d.aliases_hi or "")).split("|") if a.strip()]
    # The whole name in every script, so "Ashok Sen" / "অশোক সেন" reaches one doctor and not every Sen.
    forms += [f for f in (_strip_title(d.full_name), d.full_name_bn, d.full_name_hi) if f]
    return forms


def _strip_title(name: str) -> str:
    return " ".join(w for w in (name or "").split() if w.strip(".,").lower() not in _TITLE_WORDS)


def _gazetteer(db: Session, kind: str) -> gz.Gazetteer:
    model, forms_of, drop = (
        (LabTest, _test_forms, _GENERIC_WORDS) if kind == "test" else (Doctor, _doctor_forms, _TITLE_WORDS)
    )
    key = (kind, id(db.get_bind()))
    stamp = _catalogue_stamp(db, model)
    now = time.monotonic()
    cached = _gazetteers.get(key)
    if cached and cached[1] == stamp and now - cached[0] < GAZETTEER_TTL_S:
        return cached[2]
    entries = [(row.name, form) for row in db.query(model).all() for form in forms_of(row)]
    built = gz.Gazetteer(entries, drop_words=drop)
    _gazetteers[key] = (now, stamp, built)
    return built


def _phonetic_test_matches(db: Session, name: str) -> list[LabTest]:
    """Tests that sound like `name`, from the gazetteer's sound tiers (KCD-434/096): a romanised Bengali or
    Hindi spelling, or a loanword written in another script. Suggestions only -- what the caller MIGHT have
    meant; the agent asks, it does not act on them."""
    names = [s.canonical for s in _gazetteer(db, "test").suggest(name, limit=3) if s.basis.startswith("sound")]
    by_name = {t.name: t for t in db.query(LabTest).filter(LabTest.name.in_(names)).all()} if names else {}
    return [by_name[n] for n in names if n in by_name]


def _find_test_candidates(db: Session, name: str) -> list[LabTest]:
    """KCD-446: every test matching `name` at the FIRST cascade tier that
    produces any match at all (exact substring, then alias, then
    normalised containment) -- the same first three tiers _find_test
    uses, but collecting every match at the winning tier instead of
    silently returning the first one.

    Deliberately stops BEFORE _find_test's fourth, phonetic-fold tier
    (KCD-434): that tier is a loose, best-effort fallback for
    mishearings and romanised spellings, and its short folded keys are
    short precisely because they are generic -- gating IT on "more than
    one match" would flag routine romanised lookups (e.g. "sibisi" for
    "সিবিসি") as ambiguous and break KCD-434's own resolution, which
    this function's callers fall through to _find_test for unchanged
    when tiers 1-3 find nothing here."""
    all_tests = db.query(LabTest).all()

    exact = [t for t in all_tests if name.lower() in t.name.lower()]
    if exact:
        return exact

    alias_matches = []
    for t in all_tests:
        aliases = [a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            alias_matches.append(t)
    if alias_matches:
        return alias_matches

    q = _norm_name(name)
    norm_matches = []
    if q:
        for t in all_tests:
            forms = [t.name] + [a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a]
            if any(_norm_name(f) and (q in _norm_name(f) or _norm_name(f) in q) for f in forms):
                norm_matches.append(t)
    if norm_matches:
        return norm_matches

    return []


def _first_alias(csv: str | None) -> str | None:
    return next((a for a in (csv or "").split("|") if a.strip()), None)


def _scripted(db: Session, body: dict, model) -> dict:
    """Add the same suggestions in Bengali and Hindi script (`did_you_mean_bn`, `did_you_mean_hi`).
    A Bengali or Hindi voice cannot say a Latin name, so the reply for that language uses these; the
    English canonical names stay in `did_you_mean`. A single suggestion that is not an exact match is
    marked `needs_confirmation` so the agent asks and follows through on a yes."""
    names = body.get("did_you_mean") or []
    if not names:
        return body
    rows = {r.name: r for r in db.query(model).filter(model.name.in_(names)).all()}
    body["did_you_mean_bn"] = [_first_alias(rows[n].aliases_bn) or n for n in names if n in rows]
    body["did_you_mean_hi"] = [_first_alias(rows[n].aliases_hi) or n for n in names if n in rows]
    if len(names) == 1 and not body.get("ambiguous"):
        body["needs_confirmation"] = True
    return body


def _test_suggestions(db: Session, name: str) -> list[str]:
    return [sg.canonical for sg in _gazetteer(db, "test").suggest(name, limit=3)]


@app.get("/api/v1/tests/search")
def search_test(name: str = Query(...), db: Session = Depends(get_db)):
    # English substring match, then Bengali-script match -- covers callers
    # who say the test name in English/transliterated form, then the
    # actual common case. A caller saying "ইউরিক এসিড" was matched against
    # nothing before the Bengali path existed: the DB only stored the
    # English name "Uric Acid", and Bengali script shares zero characters
    # with Latin script, so substring AND fuzzy matching against the
    # English column alone can NEVER succeed on Bengali input, regardless
    # of how close the pronunciation is.
    # KCD-446: two or three catalogue rows matching equally well is a
    # genuinely different outcome from "nothing matched" -- silently
    # picking one (what _find_test does, correctly, for booking) would
    # risk quoting the price of a DIFFERENT test than the one meant.
    candidates = _find_test_candidates(db, name)
    if len(candidates) > 1:
        return _scripted(
            db,
            {"found": False, "query": name, "ambiguous": True, "did_you_mean": [t.name for t in candidates[:3]]},
            LabTest,
        )

    found = candidates[0] if candidates else _find_test(db, name)
    if found:
        return _test_reply_dict(found)

    # Fuzzy fallback -- try both the English name and every Bengali alias,
    # so suggestions are useful regardless of which script the caller used.
    return _scripted(db, {"found": False, "query": name, "did_you_mean": _test_suggestions(db, name)}, LabTest)


# =============================================================================
# Tool 4: GET /api/v1/tests/prep?name=...
# =============================================================================
def _by_lang(lang: str, bn: str, hi: str, en: str) -> str:
    """Requested-language text, falling back to Bengali (the system of
    record) if a translation is missing rather than returning an empty
    string the caller would hear as silence."""
    return {"hi": hi, "en": en}.get(lang) or bn


def _test_prep_reply_dict(t: LabTest, lang: str = "bn") -> dict:
    return {
        "found": True,
        "test_name": t.name,
        "test_name_bn": _first_alias_bn(t.aliases_bn),
        "test_name_hi": _first_alias_bn(t.aliases_hi),
        "fasting_required": t.fasting_required,
        "lang": lang,
        "prep_instructions": _by_lang(lang, t.prep_instructions_bn, t.prep_instructions_hi, t.prep_instructions_en),
    }


@app.get("/api/v1/tests/prep")
def test_prep(name: str = Query(...), lang: str = Query("bn"), db: Session = Depends(get_db)):
    """Same entity resolution as /tests/search, different fact -- kept as a
    separate endpoint rather than folding prep fields into every search
    response, because prep instructions are Tier-3 "approved content"
    (Blueprint 2.2) conceptually distinct from the Tier-1 price/turnaround
    facts search_test returns, and the two may end up backed by different
    systems of record later."""
    candidates = _find_test_candidates(db, name)  # KCD-446: see search_test's own comment
    if len(candidates) > 1:
        return _scripted(
            db,
            {"found": False, "query": name, "ambiguous": True, "did_you_mean": [t.name for t in candidates[:3]]},
            LabTest,
        )

    found = candidates[0] if candidates else _find_test(db, name)
    if found:
        return _test_prep_reply_dict(found, lang)
    return _scripted(db, {"found": False, "query": name, "did_you_mean": _test_suggestions(db, name)}, LabTest)


# =============================================================================
# Tool 2: GET /api/v1/doctors/availability?name=...&date=YYYY-MM-DD (optional)
# =============================================================================
# FUZZY_SURNAME_FLOOR -- LOW confidence, reasoned not measured (no real call
# audio to calibrate against yet, unlike voicerx/gate.py's SIMILARITY_FLOOR).
#
# This exists because of a bug caught in local testing: matching the raw
# query against the full formatted name ("Dr. A. Sen") let a query for
# "Doctor Nobody" fuzzy-match "Dr. N. Roy" at ratio 0.522 -- HIGHER than the
# ratio for a real garbled name against its own doctor ("sen" vs "Dr. A. Sen"
# scores only 0.462, because SequenceMatcher penalizes the length mismatch
# against the "Dr. X." prefix on both sides, so short queries and wrong
# queries land in the same range). That is this system's own small version
# of the "Naloxone" bug: confidently answering with the wrong doctor's real
# schedule instead of saying "not found".
#
# Fix: match against the SURNAME only, which cleanly separates the two
# cases in testing -- genuine garbles (e.g. "mukharji" vs "Mukherjee")
# scored 0.70-0.80; unrelated queries (e.g. "doctor nobody" vs "Roy")
# scored <=0.44. 0.60 sits in the gap. Recalibrate once real call audio
# exists, the same way gate.py's floors were tightened from real samples.
FUZZY_SURNAME_FLOOR = 0.60

# KCD-436: a mishearing severe enough to drop the character ratio below
# FUZZY_SURNAME_FLOOR can still be the right doctor if the CONSONANT
# SKELETON matches (agent/phonetic_match.py) -- e.g. an aspirated/
# unaspirated swap or a cross-script transliteration variant. Used only
# as a SECOND, independent gate on top of phonetic_match(), not instead
# of the character floor: this floor sits just above the 0.44 the
# documented "doctor nobody" vs "Roy" false positive scored, so that
# regression stays blocked even with phonetic matching turned on (its own
# phonetic keys don't match, either -- belt and suspenders).
PHONETIC_ASSISTED_FLOOR = 0.45


# Honorifics stripped from what the caller said before it is compared with a doctor's name.
_TITLE_WORDS = frozenset({"dr", "doctor", "doc", "ডাক্তার", "ডক্টর", "ডাঃ", "डॉक्टर", "डॉ", "डाक्टर"})

# A name fragment shorter than this is not matched as a substring of a doctor's name (REASONED).
MIN_SUBSTRING_CHARS = 3


def _doctor_exact_matches(db: Session, name: str) -> list[Doctor]:
    """Doctors whose WRITTEN name or alias contains what the caller said ("Sen", "Dr Sen", a Bengali
    or Hindi alias). Several matches are returned as several: choosing among them is the caller's job."""
    needle_tokens = [w for w in (x.strip(".,") for x in (name or "").lower().split()) if w and w not in _TITLE_WORDS]
    needle = " ".join(needle_tokens)
    if not needle:
        return []
    doctors = db.query(Doctor).all()

    def tokens(d):
        # the short name's words and the whole name's words ("dr", "a", "sen" and "ashok"): both are written forms
        return [t.strip(".,").lower() for t in (d.name + " " + (d.full_name or "")).split()]

    # A whole name token that IS what was said ("Sen" is Dr. Sen's surname) outranks a mere substring
    # of another name ("Sen" inside "Sengupta"): it is the written form, not a coincidence of letters.
    whole = [d for d in doctors if all(w in tokens(d) for w in needle_tokens)]
    if whole:
        return whole
    # The whole name in Bengali or Devanagari ("অশোক সেন") is more specific than the surname alias every Sen shares:
    # when it was said in full, that doctor and no other.
    said_in_full = [d for d in doctors if any(fn and fn in name for fn in (d.full_name_bn, d.full_name_hi))]
    if said_in_full:
        return said_in_full
    out = []
    for d in doctors:
        aliases = [
            a
            for a in (
                d.aliases_bn + "|" + (d.aliases_hi or "") + "|" + (d.full_name_bn or "") + "|" + (d.full_name_hi or "")
            ).split("|")
            if a
        ]
        # a one- or two-letter fragment ("ry") sits inside far too many names to identify any of them
        if (len(needle) >= MIN_SUBSTRING_CHARS and needle in d.name.lower()) or any(
            name in a or a in name for a in aliases
        ):
            out.append(d)
    return out


def _doctor_choices(db: Session, name: str, doctors: list[Doctor], near: bool = False) -> dict:
    """The "which one?" answer: every doctor in the running, each named so the caller can tell them apart -- by the
    WHOLE name (and in Bengali / Devanagari script for a voice that cannot say a Latin one), never by the short name two
    doctors can share ("Dr. A. Sen" for Ashok Sen and Abhishek Sen). If two still read the same, their department is
    added. Ordered by id so the question is the same every time it is asked."""
    ordered = sorted(doctors, key=lambda d: d.id)[:4]
    depts = {d.id: (db.get(Department, d.department_id).name if d.department_id else "") for d in ordered}

    def label(d: Doctor, script: str) -> str:
        if script == "bn":
            return d.full_name_bn or _first_alias_bn(d.aliases_bn) or d.full_name or d.name
        if script == "hi":
            return d.full_name_hi or _first_alias_bn(d.aliases_hi) or d.full_name or d.name
        return d.full_name or d.name

    body = {"found": False, "query": name, "ambiguous": True}
    for script, key in (("en", "did_you_mean"), ("bn", "did_you_mean_bn"), ("hi", "did_you_mean_hi")):
        labels = [label(d, script) for d in ordered]
        if len(set(labels)) < len(labels):  # still alike: say which department
            labels = [f"{lab} ({depts[d.id]})" if depts[d.id] else lab for lab, d in zip(labels, ordered)]
        body[key] = labels
    if near:
        body["needs_confirmation"] = True
    return body


def _doctor_ref(db: Session, d: Doctor) -> str:
    """What a reply calls this doctor: the short name, unless another doctor has the same short name ("Dr. A. Sen" twice),
    in which case the whole name -- a name that can be looked up again must point at one doctor."""
    twins = db.query(Doctor).filter(Doctor.name == d.name, Doctor.id != d.id).count()
    return (d.full_name or d.name) if twins else d.name


def _ambiguous_doctor(db: Session, name: str) -> dict | None:
    """The "which one?" body when the name fits more than one doctor (two Sens, "Dr. A. Sen" for two A. Sens), else None."""
    matches = _doctor_exact_matches(db, name)
    return _doctor_choices(db, name, matches) if len(matches) > 1 else None


def _find_doctor(db: Session, name: str) -> Doctor | None:
    """The ONE doctor the caller unambiguously named, or None.

    Resolves only on a unique written-form match. A fuzzy or phonetic near-match is NEVER a
    resolution (OpenAI review; CLAUDE.md "Doctor Nobody"): it answers "who might they have meant",
    not "who is it", and reading a different doctor's real schedule is worse than asking again.
    Those come back through `_doctor_suggestions` and the caller confirms. Two doctors matching
    the same words (e.g. "Sen" for Dr. Sen and Dr. Sengupta) is also None here -- the availability
    endpoint reports it as ambiguous instead of picking whichever the database returned first."""
    matches = _doctor_exact_matches(db, name)
    return matches[0] if len(matches) == 1 else None


# Same ratio floor already used to ACCEPT a single fuzzy match above
# (FUZZY_SURNAME_FLOOR) -- reused here as the bar for COLLECTING a name
# into the ambiguity set, so "which doctors are even in the running" and
# "would _find_doctor have accepted this one alone" stay the same
# question asked twice, never two different bars that could disagree.
def _find_doctor_candidates(db: Session, name: str) -> list[Doctor]:
    """Every doctor whose surname or alias clears FUZZY_SURNAME_FLOOR against `name`, not just the single best
    one -- so two similarly-spelled doctors surface as a genuine "which one" choice (KCD-446's doctor-side
    counterpart) instead of one being picked because it scored a hair higher. Exact/alias matches never reach
    this function -- _doctor_exact_matches already returned on those.

    The floor and the "within a hair of the top" rule are unchanged; what changed (KCD-096) is which doctors
    they are applied to: the gazetteer's shortlist, not every row. The floor is a character similarity of at
    least 0.60, and every form that clears it shares character pairs with the query, so it is on the shortlist."""
    shortlisted = {c for c, _f, _r in _gazetteer(db, "doctor").neighbours(name)}
    if not shortlisted:
        return []
    scored = []
    for d in db.query(Doctor).filter(Doctor.name.in_(shortlisted)).all():
        candidates = _doctor_forms(d)
        best_ratio = max(
            (difflib.SequenceMatcher(None, name.lower(), c.lower()).ratio() for c in candidates),
            default=0.0,
        )
        if best_ratio >= FUZZY_SURNAME_FLOOR:
            scored.append((best_ratio, d))
    if not scored:
        return []
    top = max(r for r, _ in scored)
    # Only doctors within a hair of the top score are genuinely "in the running" -- a query that clearly
    # favours one doctor over another is not an ambiguous case, it is a confident match with a distant runner-up.
    return [d for r, d in scored if top - r <= 0.05]


def _doctor_suggestions(db: Session, name: str) -> list[str]:
    """Who the caller MIGHT have meant: sound-alikes (including one- and two-consonant surnames such as Sen,
    Das, Roy) and near-spellings, best first (KCD-436/096). Never acted on -- see _find_doctor."""
    return [sg.canonical for sg in _gazetteer(db, "doctor").suggest(name, limit=3)]


def _schedule_for_weekday(db: Session, doctor_id: int, weekday: int) -> DoctorSchedule | None:
    return db.query(DoctorSchedule).filter_by(doctor_id=doctor_id, weekday=weekday).first()


def _next_available_date(db: Session, doctor_id: int, from_date: datetime.date, horizon_days: int = 14) -> str | None:
    for offset in range(horizon_days):
        d = from_date + datetime.timedelta(days=offset)
        if _schedule_for_weekday(db, doctor_id, d.weekday()):
            return d.isoformat()
    return None


@app.get("/api/v1/doctors/availability")
def doctor_availability(name: str = Query(...), date: str | None = Query(None), db: Session = Depends(get_db)):
    # Doctor-side counterpart of KCD-446: two similarly-spelled doctors
    # tying on the fuzzy floor is a genuinely different outcome from
    # "nothing matched" -- silently picking one (what _find_doctor does,
    # correctly, when there is no tie) would risk reading out a DIFFERENT
    # doctor's real schedule, the CLAUDE.md "Doctor Nobody" class of bug.
    exact = _doctor_exact_matches(db, name)
    if len(exact) > 1:
        return _doctor_choices(db, name, exact)
    doctor = exact[0] if exact else None
    if not doctor:
        near = _find_doctor_candidates(db, name)
        if len(near) > 1:
            return _doctor_choices(db, name, near, near=True)
        return _scripted(
            db,
            {"found": False, "query": name, "did_you_mean": _doctor_suggestions(db, name), "needs_confirmation": True},
            Doctor,
        )

    today = datetime.date.today()

    if date:
        try:
            target = datetime.date.fromisoformat(date)
        except ValueError:
            return {"found": False, "query": name}
        sched = _schedule_for_weekday(db, doctor.id, target.weekday())
        if sched:
            return {
                "found": True,
                "doctor_name": _doctor_ref(db, doctor),
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
                "doctor_name_hi": _first_alias_bn(doctor.aliases_hi),
                "date": target.isoformat(),
                "available": True,
                "chamber_hours": f"{sched.start_time}-{sched.end_time}",
                "next_available_date": None,
            }
        next_date = _next_available_date(db, doctor.id, target + datetime.timedelta(days=1))
        return {
            "found": True,
            "doctor_name": _doctor_ref(db, doctor),
            "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
            "doctor_name_hi": _first_alias_bn(doctor.aliases_hi),
            "date": target.isoformat(),
            "available": False,
            "chamber_hours": None,
            "next_available_date": next_date,
        }

    # No date given -> "when is this doctor next available"
    next_date = _next_available_date(db, doctor.id, today)
    if not next_date:
        return {
            "found": True,
            "doctor_name": _doctor_ref(db, doctor),
            "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
            "doctor_name_hi": _first_alias_bn(doctor.aliases_hi),
            "date": None,
            "available": False,
            "chamber_hours": None,
            "next_available_date": None,
        }
    sched = _schedule_for_weekday(db, doctor.id, datetime.date.fromisoformat(next_date).weekday())
    return {
        "found": True,
        "doctor_name": _doctor_ref(db, doctor),
        "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
        "doctor_name_hi": _first_alias_bn(doctor.aliases_hi),
        "date": next_date,
        "available": True,
        "chamber_hours": f"{sched.start_time}-{sched.end_time}",
        "next_available_date": None,
    }


# =============================================================================
# Tool 3: POST /api/v1/appointments
# =============================================================================
class BookingRequest(BaseModel):
    doctor_name: str
    date: str
    time_slot: str
    patient_name: str
    phone: str


def _generate_slots(start: str, end: str, step_min: int = SLOT_STEP_MIN) -> list[str]:
    t = datetime.datetime.strptime(start, "%H:%M")
    end_t = datetime.datetime.strptime(end, "%H:%M")
    slots = []
    while t < end_t:
        slots.append(t.strftime("%H:%M"))
        t += datetime.timedelta(minutes=step_min)
    return slots


@app.post("/api/v1/appointments")
@idempotent("appointments")
def book_appointment(req: BookingRequest, db: Session = Depends(get_db)):
    """This is the ORIGINAL, pre-Epic-E26 booking endpoint, kept for
    backward compatibility. CodeRabbit-flagged, real bug: it used to
    create an Appointment row directly, with its own "taken" check
    against nothing but a raw Appointment query (which did not even
    filter by status, so a CANCELLED row here blocked the slot forever)
    and total invisibility to models.SlotLock -- the actual atomicity
    guard hold_slot()/confirm_booking() give the newer /api/v1/bookings/*
    flow (KCD-376). A slot booked through this endpoint was silently NOT
    reflected in available_slots() (which only reads SlotLock), so the
    voice agent could offer and double-book it, and conversely a slot
    HELD by the voice agent was invisible here too. Fixed by routing
    through the SAME hold_slot()/confirm_booking() primitives instead of
    maintaining a second, unguarded booking path -- one atomicity
    guarantee, not two that can silently disagree."""
    amb = _ambiguous_doctor(db, req.doctor_name)
    if amb is not None:
        return {
            "success": False,
            "reason": "doctor_ambiguous",
            **{k: v for k, v in amb.items() if k.startswith("did_you_mean")},
        }
    doctor = _find_doctor(db, req.doctor_name)
    if not doctor:
        return {
            "success": False,
            "reason": "doctor_not_found",
            "did_you_mean": _doctor_suggestions(db, req.doctor_name),
        }

    try:
        target = datetime.date.fromisoformat(req.date)
    except ValueError:
        return {"success": False, "reason": "missing_field"}

    sched = _schedule_for_weekday(db, doctor.id, target.weekday())
    if not sched:
        # Doctor doesn't sit that day at all -- not in the caller-facing
        # reason enum reply_templates.booking_reply() specifically handles,
        # so it falls to that function's generic "couldn't book" message,
        # which remains true and safe rather than a false "slot taken".
        return {"success": False, "reason": "doctor_not_available_that_day"}

    valid_slots = _generate_slots(sched.start_time, sched.end_time)
    if req.time_slot not in valid_slots:
        return {"success": False, "reason": "slot_taken", "alternative_slots": valid_slots[:3]}

    hold = bs.hold_slot(db, doctor.id, req.date, req.time_slot)
    if not hold["success"]:
        return {
            "success": False,
            "reason": "slot_taken",
            "alternative_slots": bs.available_slots(db, doctor.id, req.date)[:3],
        }

    # 8 hex chars, not 4 -- see booking_service._confirmation_id's comment:
    # this ID is now also accepted by /api/v1/bookings/lookup, unauthenticated.
    result = bs.confirm_booking(
        db, hold["hold_token"], doctor.id, req.date, req.time_slot, req.patient_name, req.phone, req.phone
    )
    if not result["success"]:
        return {
            "success": False,
            "reason": "slot_taken",
            "alternative_slots": bs.available_slots(db, doctor.id, req.date)[:3],
        }

    return {
        "success": True,
        "confirmation_id": result["confirmation_id"],
        "doctor_name": _doctor_ref(db, doctor),
        "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
        "doctor_name_hi": _first_alias_bn(doctor.aliases_hi),
        "date": req.date,
        "time_slot": req.time_slot,
    }


# =============================================================================
# Tool 5: GET /api/v1/faq?topic=...
# =============================================================================
@app.get("/api/v1/faq")
def faq_answer(topic: str = Query(...), lang: str = Query("bn"), db: Session = Depends(get_db)):
    """Looked up by TOPIC KEY, not free text -- the caller-facing entity
    resolution (which topic did they mean) already happened in
    FastPath.FAQCatalogue against the keyword sets from /api/v1/catalogue.
    This endpoint's only job is "give me the current answer for this
    topic", fetched live on every turn for the same reason test_rate and
    doctor_availability never trust a cached VALUE -- only a cached
    ROUTING decision."""
    row = db.query(FAQ).filter_by(topic=topic).first()
    if not row:
        return {"found": False, "topic": topic}
    return {
        "found": True,
        "topic": row.topic,
        "lang": lang,
        "answer": _by_lang(lang, row.answer_bn, row.answer_hi, row.answer_en),
    }


# =============================================================================
# Epic E26: booking, rescheduling and cancellation
#
# Two-phase everywhere a slot is claimed (hold, then confirm) so the actual
# concurrency guard -- booking_service.hold_slot()'s atomic INSERT into
# SlotLock -- runs BEFORE the caller has to speak a patient name and phone
# number, not after. See booking_service.py's module docstring and
# models.SlotLock's for the full reasoning (KCD-376).
# =============================================================================
class HoldRequest(BaseModel):
    doctor_name: str
    date: str
    time_slot: str


def _validate_date_not_past(date_str: str) -> dict | None:
    """Shared by every booking endpoint that takes a caller-supplied
    date. Returns an error dict in hold_booking's own shape, or None
    when the date is at least parseable and not in the past."""
    try:
        target = datetime.date.fromisoformat(date_str)
    except ValueError:
        return {"success": False, "reason": "invalid_date"}
    if target < datetime.date.today():
        # KCD-363: never book a past date, and never silently roll it
        # forward -- the caller is told plainly (main.py's reply template)
        # and offered the same weekday next week.
        return {"success": False, "reason": "date_in_past"}
    return None


def _validate_doctor_slot(db: Session, doctor: Doctor, date_str: str, time_slot: str) -> dict | None:
    """CodeRabbit-flagged: this used to live only inside hold_booking,
    which reschedule_booking never called -- it went straight to
    bs.reschedule_appointment(), which calls hold_slot(), a bare
    ATOMICITY primitive that deliberately does not validate against the
    doctor's schedule at all (see its own docstring). A malformed date
    (a bad LLM extraction) or a genuinely past date could reach a
    "successful" reschedule to a broken state, confirmed to the caller,
    then crash something unrelated later (e.g. cancellation_charge()
    parsing that date/time). Extracted so every endpoint that resolves a
    date+slot against a specific doctor applies the SAME checks."""
    err = _validate_date_not_past(date_str)
    if err:
        return err
    target = datetime.date.fromisoformat(date_str)
    sched = _schedule_for_weekday(db, doctor.id, target.weekday())
    if not sched:
        return {"success": False, "reason": "doctor_not_available_that_day"}
    valid_slots = _generate_slots(sched.start_time, sched.end_time)
    if time_slot not in valid_slots:
        return {"success": False, "reason": "invalid_slot", "valid_slots": valid_slots}
    if target == datetime.date.today():
        # CodeRabbit-flagged, real bug: _validate_date_not_past is
        # DATE-level only ("target < today"), so a same-day slot earlier
        # than right now was still requestable and holdable directly
        # (available_slots() no longer OFFERS it -- see that function's
        # own fix -- but nothing stopped a caller from asking for it by
        # name anyway).
        now = datetime.datetime.now()
        try:
            slot_h, slot_m = (int(x) for x in time_slot.split(":"))
        except ValueError:
            return {"success": False, "reason": "invalid_slot", "valid_slots": valid_slots}
        if slot_h * 60 + slot_m <= now.hour * 60 + now.minute:
            return {
                "success": False,
                "reason": "invalid_slot",
                "valid_slots": [
                    s
                    for s in valid_slots
                    if int(s.split(":")[0]) * 60 + int(s.split(":")[1]) > now.hour * 60 + now.minute
                ],
            }
    return None


@app.post("/api/v1/bookings/hold")
@idempotent("bookings.hold")
def hold_booking(req: HoldRequest, db: Session = Depends(get_db)):
    amb = _ambiguous_doctor(db, req.doctor_name)
    if amb is not None:
        return {
            "success": False,
            "reason": "doctor_ambiguous",
            **{k: v for k, v in amb.items() if k.startswith("did_you_mean")},
        }
    doctor = _find_doctor(db, req.doctor_name)
    if not doctor:
        return {
            "success": False,
            "reason": "doctor_not_found",
            "did_you_mean": _doctor_suggestions(db, req.doctor_name),
        }
    err = _validate_doctor_slot(db, doctor, req.date, req.time_slot)
    if err:
        return err

    result = bs.hold_slot(db, doctor.id, req.date, req.time_slot)
    if not result["success"]:
        result["alternative_slots"] = [
            a["time_slot"] for a in bs.nearest_alternatives(db, doctor.id, req.date, req.time_slot)
        ]
    else:
        result["doctor_id"] = doctor.id
        result["doctor_name"] = _doctor_ref(db, doctor)
    return result


class ConfirmRequest(BaseModel):
    hold_token: str
    doctor_id: int
    date: str
    time_slot: str
    patient_name: str
    phone: str
    caller_phone: str
    patient_age: int | None = None
    relationship: str = "self"


@app.post("/api/v1/bookings/confirm")
@idempotent("bookings.confirm")
def confirm_booking_endpoint(req: ConfirmRequest, db: Session = Depends(get_db)):
    return bs.confirm_booking(
        db,
        req.hold_token,
        req.doctor_id,
        req.date,
        req.time_slot,
        req.patient_name,
        req.phone,
        req.caller_phone,
        req.patient_age,
        req.relationship,
    )


class RescheduleRequest(BaseModel):
    confirmation_id: str
    new_date: str
    new_time_slot: str


@app.post("/api/v1/bookings/reschedule")
@idempotent("bookings.reschedule")
def reschedule_booking(req: RescheduleRequest, db: Session = Depends(get_db)):
    appt = db.query(Appointment).filter_by(confirmation_id=req.confirmation_id, status="confirmed").first()
    if not appt:
        return {"success": False, "reason": "not_found"}
    doctor = db.get(Doctor, appt.doctor_id)
    err = _validate_doctor_slot(db, doctor, req.new_date, req.new_time_slot)
    if err:
        return err
    return bs.reschedule_appointment(db, req.confirmation_id, req.new_date, req.new_time_slot)


class CancelRequest(BaseModel):
    confirmation_id: str
    confirm_charge: bool = False


@app.post("/api/v1/bookings/cancel")
@idempotent("bookings.cancel")
def cancel_booking(req: CancelRequest, db: Session = Depends(get_db)):
    return bs.cancel_appointment(db, req.confirmation_id, req.confirm_charge)


@app.get("/api/v1/bookings/lookup")
def lookup_booking(
    phone: str | None = Query(None),
    confirmation_id: str | None = Query(None),
    name: str | None = Query(None),
    db: Session = Depends(get_db),
):
    if not (phone or confirmation_id):
        return {"found": False, "bookings": []}
    rows = bs.lookup_bookings(db, phone=phone, confirmation_id=confirmation_id, name=name)
    return {"found": bool(rows), "bookings": rows}


class SeniorModeRequest(BaseModel):
    phone: str
    senior: bool = True
    # The number the call actually came from, when the bridge knows it.
    # Defaults to `phone` (as lookup_booking does) until real CallerID exists.
    caller_phone: str | None = None


@app.post("/api/v1/patients/senior")
@idempotent("patients.senior")
def set_patient_senior(req: SeniorModeRequest, db: Session = Depends(get_db)):
    """KCD-084: persist the delivery mode against the patient. Boolean only."""
    return {"updated": bs.set_patient_senior(db, req.phone, req.senior, req.caller_phone)}


@app.get("/api/v1/patients/senior")
def get_patient_senior(phone: str = Query(...), caller_phone: str | None = Query(None), db: Session = Depends(get_db)):
    return {"senior": bs.get_patient_senior(db, phone, caller_phone)}


@app.get("/api/v1/bookings/conflict")
def booking_conflict(
    phone: str = Query(...), date: str = Query(...), time_slot: str = Query(...), db: Session = Depends(get_db)
):
    conflict = bs.find_conflict(db, phone, date, time_slot)
    return {"conflict": conflict is not None, "existing": conflict}


class TestsBookingRequest(BaseModel):
    test_names: list[str]
    date: str
    patient_name: str
    phone: str
    caller_phone: str
    patient_age: int | None = None
    relationship: str = "self"


@app.post("/api/v1/bookings/tests")
@idempotent("bookings.tests")
def book_tests_endpoint(req: TestsBookingRequest, db: Session = Depends(get_db)):
    err = _validate_date_not_past(req.date)
    if err:
        return err
    ids, not_found = [], []
    for name in req.test_names:
        t = _find_test(db, name)
        (ids if t else not_found).append(t.id if t else name)
    if not ids:
        return {"success": False, "reason": "no_valid_tests", "not_found": not_found}
    result = bs.book_tests(
        db, ids, req.date, req.patient_name, req.phone, req.caller_phone, req.patient_age, req.relationship
    )
    result["not_found"] = not_found
    return result


class AddTestRequest(BaseModel):
    confirmation_id: str
    test_name: str


@app.post("/api/v1/bookings/add-test")
@idempotent("bookings.add_test")
def add_test_endpoint(req: AddTestRequest, db: Session = Depends(get_db)):
    return bs.add_test_to_booking(db, req.confirmation_id, req.test_name)


@app.get("/api/v1/doctors/earliest")
def doctor_earliest(name: str = Query(...), db: Session = Depends(get_db)):
    amb = _ambiguous_doctor(db, name)
    if amb is not None:
        return amb
    doctor = _find_doctor(db, name)
    if not doctor:
        return {"found": False, "query": name}
    result = bs.earliest_available(db, doctor.id, datetime.date.today())
    if not result:
        return {"found": True, "available": False, "doctor_name": _doctor_ref(db, doctor)}
    return {
        "found": True,
        "available": True,
        "doctor_name": _doctor_ref(db, doctor),
        "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
        **result,
    }


@app.get("/api/v1/departments/route")
def department_route(query: str = Query(...), lang: str = Query("bn"), db: Session = Depends(get_db)):
    return bs.route_department(db, query, lang)


class SmsRequest(BaseModel):
    to: str
    message: str
    template_key: str = "generic"
    related_confirmation_id: str | None = None


@app.post("/api/v1/notifications/sms")
@idempotent("notifications.sms")
def send_sms(req: SmsRequest, db: Session = Depends(get_db)):
    """The open placeholder the caller-confirmation flow already writes
    to via booking_service.queue_sms(). Exposed as its own endpoint too,
    so an external system (or a future clinic-api/notifications.py) has
    one clear place to either call in, or be wired up as the thing THIS
    function calls out to. `status` is always "queued": nothing in this
    codebase claims a message reached a phone until a real provider is
    plugged in here."""
    return bs.queue_sms(db, req.to, req.template_key, req.message, req.related_confirmation_id)


class ResendRequest(BaseModel):
    confirmation_id: str


@app.post("/api/v1/bookings/resend")
@idempotent("bookings.resend")
def resend_booking_confirmation(
    req: ResendRequest | None = None, confirmation_id: str | None = Query(None), db: Session = Depends(get_db)
):
    """The confirmation id comes in the body (ResendRequest); the old `?confirmation_id=` form is still accepted."""
    cid = req.confirmation_id if req is not None else confirmation_id
    if not cid:
        raise HTTPException(status_code=422, detail="confirmation_id is required")
    return bs.resend_confirmation(db, cid)


class DraftRequest(BaseModel):
    caller_phone: str
    call_id: str
    slots_json: str


@app.post("/api/v1/bookings/draft")
@idempotent("bookings.draft")
def save_draft_endpoint(req: DraftRequest, db: Session = Depends(get_db)):
    bs.save_draft(db, req.caller_phone, req.call_id, req.slots_json)
    return {"saved": True}


@app.get("/api/v1/bookings/draft")
def get_draft_endpoint(phone: str = Query(...), db: Session = Depends(get_db)):
    draft = bs.find_draft(db, phone)
    return {"found": draft is not None, "draft": draft}


# =============================================================================
# Epic E27: information and enquiry
# =============================================================================


@app.get("/api/v1/doctors/{doctor_name}/leave")
def doctor_leave_endpoint(doctor_name: str, date: str = Query(...), db: Session = Depends(get_db)):
    amb = _ambiguous_doctor(db, doctor_name)
    if amb is not None:
        return amb
    doctor = _find_doctor(db, doctor_name)
    if not doctor:
        return {"found": False, "query": doctor_name}
    leave = eq.doctor_leave_on(db, doctor.id, date)
    return {"found": True, "doctor_name": _doctor_ref(db, doctor), "on_leave": leave is not None, "leave": leave}


class PrepMergeRequest(BaseModel):
    test_names: list[str]


@app.post("/api/v1/tests/prep/merge")
def prep_merge_endpoint(req: PrepMergeRequest, db: Session = Depends(get_db)):
    ids, not_found = [], []
    for name in req.test_names:
        t = _find_test(db, name)
        (ids if t else not_found).append(t.id if t else name)
    result = eq.merge_prep_instructions(db, ids)
    result["not_found"] = not_found
    return result


@app.get("/api/v1/packages/{name}")
def package_endpoint(name: str, db: Session = Depends(get_db)):
    return eq.compare_package_vs_separate(db, name)


@app.get("/api/v1/walk-in")
def walk_in_endpoint(
    department: str | None = Query(None), test_name: str | None = Query(None), db: Session = Depends(get_db)
):
    dept_id = None
    if department:
        dept = db.query(Department).filter(Department.name.ilike(f"%{department}%")).first()
        dept_id = dept.id if dept else None
    test_id = None
    if test_name:
        t = _find_test(db, test_name)
        test_id = t.id if t else None
    return eq.walk_in_policy(db, department_id=dept_id, lab_test_id=test_id)


@app.get("/api/v1/billing/outstanding")
def billing_outstanding_endpoint(phone: str = Query(...), db: Session = Depends(get_db)):
    return eq.outstanding_balance(db, phone)


@app.get("/api/v1/home-collection/eligibility")
def home_collection_endpoint(test_name: str = Query(...), postal_code: str = Query(...), db: Session = Depends(get_db)):
    t = _find_test(db, test_name)
    if not t:
        return {"found": False, "query": test_name}
    return eq.home_collection_eligibility(db, t.id, postal_code)


# =============================================================================
# ADDED BY SOURAV: home-collection full flow -- multi-test eligibility, real slot
# availability/hold/confirm, a deterministic quote, the payment policy, and the actual
# booking/cancellation transaction. See home_collection_service.py for the business logic;
# every endpoint here only resolves names/does request-shape validation and calls it.
# =============================================================================


def _resolve_test_ids(db: Session, test_names: list[str]) -> tuple[list[int], list[str]]:
    """ADDED BY SOURAV: the same exact/near-exact test-name resolution every other endpoint in
    this file already uses (_find_test) -- never a second, competing matcher for home collection.
    Returns (resolved lab_test ids, names that did not resolve to any test)."""
    ids, not_found = [], []
    for name in test_names:
        t = _find_test(db, name)
        if t:
            ids.append(t.id)
        else:
            not_found.append(name)
    return ids, not_found


@app.get("/api/v1/home-collection/eligibility-multi")
def home_collection_eligibility_multi_endpoint(
    test_names: list[str] = Query(...), postal_code: str = Query(...), db: Session = Depends(get_db)
):
    # ADDED BY SOURAV: CASE 3 of the story (several booked/requested tests) needs every test's
    # eligibility reported individually -- never merged, never one silently chosen.
    ids, not_found = _resolve_test_ids(db, test_names)
    return {"found": True, "results": hcs.multi_test_eligibility(db, ids, postal_code), "not_found": not_found}


@app.get("/api/v1/home-collection/slots")
def home_collection_slots_endpoint(
    postal_code: str = Query(...), date: str = Query(...), db: Session = Depends(get_db)
):
    # ADDED BY SOURAV: real, DB-backed slot availability -- never a calculated/estimated window.
    return {"found": True, "slots": hcs.available_slots(db, postal_code, date)}


class HomeCollectionHoldRequest(BaseModel):
    slot_id: int
    caller_phone: str | None = None
    call_id: str | None = None


@app.post("/api/v1/home-collection/hold")
@idempotent("home_collection.hold")
def home_collection_hold_endpoint(req: HomeCollectionHoldRequest, db: Session = Depends(get_db)):
    # ADDED BY SOURAV: the same hold-then-confirm two-phase pattern booking_service.py already
    # uses for doctor slots (KCD-376), so two simultaneous callers can never both confirm the
    # same capacity unit.
    return hcs.hold_home_collection_slot(db, req.slot_id, req.caller_phone, req.call_id)


class HomeCollectionReleaseHoldRequest(BaseModel):
    hold_token: str


@app.post("/api/v1/home-collection/release-hold")
@idempotent("home_collection.release_hold")
def home_collection_release_hold_endpoint(req: HomeCollectionReleaseHoldRequest, db: Session = Depends(get_db)):
    # ADDED BY SOURAV: a caller who changes their mind must not permanently consume capacity.
    return hcs.release_home_collection_hold(db, req.hold_token)


@app.get("/api/v1/home-collection/quote")
def home_collection_quote_endpoint(
    test_names: list[str] = Query(...), postal_code: str = Query(...), db: Session = Depends(get_db)
):
    # ADDED BY SOURAV: the deterministic test-price + home-collection-charge + total breakdown,
    # computed here, never by the agent or a reply template.
    ids, not_found = _resolve_test_ids(db, test_names)
    quote = hcs.quote_home_collection(db, ids, postal_code)
    quote["not_found"] = not_found
    return quote


@app.get("/api/v1/home-collection/payment-policy")
def home_collection_payment_policy_endpoint(lang: str = Query("bn"), db: Session = Depends(get_db)):
    # ADDED BY SOURAV: the real, versioned payment policy (never a string literal in the agent).
    return hcs.payment_policy_reply_dict(db, lang)


class HomeCollectionBookRequest(BaseModel):
    hold_token: str
    test_names: list[str]
    postal_code: str
    patient_name: str
    phone: str
    caller_phone: str | None = None
    address_line: str
    locality: str = ""
    city: str = ""
    state: str = ""
    landmark: str = ""
    patient_id: int | None = None


@app.post("/api/v1/home-collection/book")
@idempotent("home_collection.book")
def home_collection_book_endpoint(req: HomeCollectionBookRequest, db: Session = Depends(get_db)):
    # ADDED BY SOURAV: the actual, persistent home-collection booking transaction. A retried
    # confirm (same Idempotency-Key header) is handled by @idempotent above -- see
    # home_collection_service.create_home_collection_booking's own docstring for why this
    # function is not independently idempotent as well.
    ids, not_found = _resolve_test_ids(db, req.test_names)
    if not_found:
        return {"success": False, "reason": "test_not_found", "not_found": not_found}
    return hcs.create_home_collection_booking(
        db,
        hold_token=req.hold_token,
        lab_test_ids=ids,
        postal_code=req.postal_code,
        patient_name=req.patient_name,
        phone=req.phone,
        caller_phone=req.caller_phone,
        address_line=req.address_line,
        locality=req.locality,
        city=req.city,
        state=req.state,
        landmark=req.landmark,
        patient_id=req.patient_id,
    )


@app.get("/api/v1/home-collection/booking/{booking_reference}")
def home_collection_booking_status_endpoint(booking_reference: str, db: Session = Depends(get_db)):
    return hcs.get_home_collection_booking(db, booking_reference)


class HomeCollectionCancelRequest(BaseModel):
    booking_reference: str


@app.post("/api/v1/home-collection/cancel")
@idempotent("home_collection.cancel")
def home_collection_cancel_endpoint(req: HomeCollectionCancelRequest, db: Session = Depends(get_db)):
    return hcs.cancel_home_collection_booking(db, req.booking_reference)


@app.get("/api/v1/insurance/coverage")
def insurance_coverage_endpoint(
    policy_number: str = Query(...), test_name: str | None = Query(None), db: Session = Depends(get_db)
):
    test_id = None
    if test_name:
        t = _find_test(db, test_name)
        test_id = t.id if t else None
    return eq.check_insurance_coverage(db, policy_number, test_id)


@app.get("/api/v1/tests/{test_name}/prescription-requirement")
def prescription_requirement_endpoint(test_name: str, lang: str = Query("bn"), db: Session = Depends(get_db)):
    t = _find_test(db, test_name)
    if not t:
        return {"found": False, "query": test_name}
    return eq.prescription_requirement(db, t.id, lang)


class OutOfScopeRequest(BaseModel):
    call_id: str
    caller_question: str
    reason_code: str = "out_of_scope"


@app.post("/api/v1/calls/out-of-scope")
@idempotent("calls.out_of_scope")
def out_of_scope_endpoint(req: OutOfScopeRequest, db: Session = Depends(get_db)):
    return eq.record_out_of_scope(db, req.call_id, req.caller_question, req.reason_code)


class CallbackRequestBody(BaseModel):
    phone: str
    call_id: str
    requested_window: str
    reason: str = ""
    call_summary: str = ""


@app.post("/api/v1/callbacks")
@idempotent("callbacks")
def callback_endpoint(req: CallbackRequestBody, db: Session = Depends(get_db)):
    return eq.request_callback(db, req.phone, req.call_id, req.requested_window, req.reason, req.call_summary)


class PaymentLinkRequest(BaseModel):
    confirmation_id: str


@app.post("/api/v1/payments/send-link")
@idempotent("payments.send_link")
def send_payment_link_endpoint(req: PaymentLinkRequest, db: Session = Depends(get_db)):
    """Only a CONFIRMED DOCTOR appointment's confirmation_id ever gets a link back -- a lab test is paid and booked
    at the counter, never over the call (bs.send_payment_link's own docstring)."""
    return bs.send_payment_link(db, req.confirmation_id)


@app.get("/api/v1/reports/status")
def report_status_endpoint(confirmation_id: str = Query(...), db: Session = Depends(get_db)):
    return eq.report_status(db, confirmation_id)


class ReportOTPRequest(BaseModel):
    confirmation_id: str
    phone: str


@app.post("/api/v1/reports/request-otp")
@idempotent("reports.request_otp")
def report_request_otp_endpoint(req: ReportOTPRequest, db: Session = Depends(get_db)):
    return eq.request_report_otp(db, req.confirmation_id, req.phone)


class ReportDeliverRequest(BaseModel):
    confirmation_id: str
    otp_code: str


@app.post("/api/v1/reports/deliver")
@idempotent("reports.deliver")
def report_deliver_endpoint(req: ReportDeliverRequest, db: Session = Depends(get_db)):
    return eq.deliver_report(db, req.confirmation_id, req.otp_code)


def _doctor_listing(d) -> dict:
    """One doctor, in the shape every other doctor payload here already uses.

    Ported story "Caller asks which doctors are available" (agent/doctor_list.py).
    The same keys as /api/v1/catalogue's `doctors` entries, plus the department,
    so agent/reply_templates.py can pick a name the caller's own voice can
    actually SAY: a Bengali or Hindi synthesiser silently drops Latin script, so
    the reply for those languages has to use the seeded alias, never `name`.

    Nothing is composed here. Every string is a column.
    """
    return {
        "name": d.name,
        "full_name": d.full_name or d.name,
        "full_name_bn": d.full_name_bn,
        "full_name_hi": d.full_name_hi,
        "surname": d.name.split()[-1],
        "alias_bn": _first_alias(d.aliases_bn),
        "alias_hi": _first_alias(d.aliases_hi),
        "qualifications": d.qualifications,
        "department": d.department.name if d.department else None,
    }


@app.get("/api/v1/departments")
def list_departments_endpoint(db: Session = Depends(get_db)):
    """Every department, with how many doctors sit in each.

    Ported story "Caller asks which doctors are available". This repository had
    no plain department list -- only /api/v1/departments/route, which maps a
    SYMPTOM to one department and cannot answer "what departments do you have".
    """
    rows = db.query(Department).order_by(Department.name).all()
    return {
        "found": bool(rows),
        "departments": [{"name": r.name, "doctor_count": len(r.doctors)} for r in rows],
    }


@app.get("/api/v1/doctors")
def list_all_doctors_endpoint(db: Session = Depends(get_db)):
    """Every doctor the clinic has, with their department.

    The agent decides how many of these to actually SAY (agent/reply_templates.py
    caps the spoken list and offers the rest by department) -- this endpoint
    returns the facts and makes no presentation decision.
    """
    rows = db.query(Doctor).join(Department).order_by(Department.name, Doctor.name).all()
    return {"found": bool(rows), "count": len(rows), "doctors": [_doctor_listing(d) for d in rows]}


@app.get("/api/v1/doctors/by-department")
def doctors_by_department_endpoint(department: str = Query(...), db: Session = Depends(get_db)):
    """The doctors in one named department.

    Resolution is the SAME discipline as _find_doctor/_find_test: a written-form
    match resolves, anything less comes back as a suggestion the agent must ask
    about, never as a silent best guess. An exact (case-insensitive) name wins;
    otherwise a substring match resolves only when exactly ONE department
    contains the query, and several matches are `ambiguous` with the names
    offered -- which is how "Doctor Nobody" was stopped from resolving to a real
    doctor's real schedule, applied here to departments.
    """
    query = (department or "").strip()
    if not query:
        return {"found": False, "query": department}

    rows = db.query(Department).order_by(Department.name).all()
    low = query.lower()
    exact = [r for r in rows if r.name.lower() == low]
    # Matched in BOTH directions on purpose, so the agent never has to carve a
    # department name out of a sentence before asking. "which doctors are in
    # cardiology" contains "Cardiology", and "cardio" is contained BY it; both
    # resolve here, against the real department table, instead of being guessed
    # at by a text extractor in agent/ that has no list to check itself against.
    # The no-guessing discipline is unchanged: several matches are still
    # `ambiguous` rather than silently resolving to the first row.
    matches = exact or [r for r in rows if low in r.name.lower() or r.name.lower() in low]

    if not matches:
        near = [r.name for r in rows if r.name.lower()[:3] == query.lower()[:3]][:3]
        body = {"found": False, "query": query}
        if near:
            body["did_you_mean"] = near
            body["needs_confirmation"] = len(near) == 1
        return body
    if len(matches) > 1:
        return {"found": False, "query": query, "ambiguous": True, "did_you_mean": [r.name for r in matches[:3]]}

    dept = matches[0]
    doctors = db.query(Doctor).filter_by(department_id=dept.id).order_by(Doctor.name).all()
    return {
        "found": True,
        "department": dept.name,
        "count": len(doctors),
        "doctors": [_doctor_listing(d) for d in doctors],
    }


@app.get("/api/v1/departments/{department_name}/hours")
def department_hours_endpoint(department_name: str, lang: str = Query("bn"), db: Session = Depends(get_db)):
    dept = db.query(Department).filter(Department.name.ilike(f"%{department_name}%")).first()
    if not dept:
        return {"found": False, "query": department_name}
    hours = eq.department_hours(db, dept.id, lang)
    return hours or {"found": False}


# =============================================================================
# Epic E33: patient context and history (clinic-api/patient_context.py)
# =============================================================================


@app.get("/api/v1/calls/satisfaction/summary")
def satisfaction_summary(
    since: str | None = Query(None, description="ISO date; calls that started on or after it"),
    language: str | None = Query(None, description="bn | hi | en: the call's first language"),
    db: Session = Depends(get_db),
):
    """How the calls went, from the implicit happiness score (agent/call_score.py): scored calls, their mean and
    median, the share in each band, the same by language, and what pulled scores down most. A PROXY built from behaviour,
    not from anything callers said; no call, phone number or text is returned."""
    from models import CallRecord

    q = db.query(CallRecord)
    if since:
        try:
            q = q.filter(CallRecord.started_at >= datetime.datetime.fromisoformat(since))
        except ValueError:
            raise HTTPException(status_code=422, detail="since must be an ISO date") from None
    rows = q.all()
    if language:
        rows = [r for r in rows if (r.languages or "").split(",")[0] == language]

    def block(subset):
        scored = sorted(r.satisfaction_score for r in subset if r.satisfaction_score is not None)
        bands = {
            b: sum(1 for r in subset if r.satisfaction_score is not None and r.satisfaction_band == b)
            for b in ("happy", "neutral", "unhappy", "very_unhappy")
        }
        return {
            "calls": len(subset),
            "scored": len(scored),
            "mean": round(sum(scored) / len(scored), 1) if scored else None,
            "median": scored[len(scored) // 2] if scored else None,
            "bands": bands,
            "share_unhappy": round((bands["unhappy"] + bands["very_unhappy"]) / len(scored), 3) if scored else None,
        }

    by_language: dict[str, list] = {}
    for r in rows:
        by_language.setdefault((r.languages or "").split(",")[0] or "unknown", []).append(r)
    costs: dict[str, list[int]] = {}
    unscored: dict[str, int] = {}
    for r in rows:
        try:
            data = json.loads(r.satisfaction_json or "{}")
        except ValueError:
            data = {}
        if r.satisfaction_score is None:
            unscored[data.get("reason") or "no_score"] = unscored.get(data.get("reason") or "no_score", 0) + 1
            continue
        for name, pts in data.get("reasons", []):
            if pts < 0:
                costs.setdefault(name, []).append(pts)
    top = sorted(
        ({"signal": n, "calls": len(v), "points_lost": -sum(v)} for n, v in costs.items()),
        key=lambda d: (-d["points_lost"], d["signal"]),
    )[:8]
    return {
        "basis": "behaviour",
        "version": "implicit-v1",
        "overall": block(rows),
        "by_language": {lang: block(rs) for lang, rs in sorted(by_language.items())},
        "top_causes": top,
        "unscored_reasons": unscored,
    }


class CallEventRequest(BaseModel):
    seq: int
    kind: str
    payload: dict = Field(default_factory=dict)
    caller_phone: str | None = None


class ComplaintBody(BaseModel):
    """Ported story "Caller wants to make a complaint" (agent/complaint.py)."""

    call_id: str
    text: str
    detected_by: str = ""
    lang: str = ""
    phone: str | None = None


@app.post("/api/v1/complaints")
def file_complaint_endpoint(req: ComplaintBody, db: Session = Depends(get_db)):
    """Record a complaint verbatim (clinic-api/enquiry_service.file_complaint).

    DELIBERATELY NOT @idempotent, unlike its neighbours. Two reasons, and the
    second is the load-bearing one:

      * the service layer already dedupes by (call_id, text), the same way
        eq.request_callback() dedupes its own twin, so a retried tool call
        cannot open a second row for the same words;
      * @idempotent wraps the endpoint in a function defined in
        clinic-api/idempotency.py, so FastAPI resolves this module's type
        annotations against THAT module's globals. Under
        `from __future__ import annotations` every annotation here is a string,
        so a decorated endpoint whose body model is defined in this file fails
        at import with `NameError: name '<Model>' is not defined`. That is a
        live defect on this file's existing @idempotent endpoints
        (book_appointment at /api/v1/appointments is the first one) and is why
        14 test modules cannot import the app today. It is not worked around
        here -- it is simply not re-created.
    """
    return eq.file_complaint(db, req.call_id, req.text, req.detected_by, req.lang, req.phone)


@app.post("/api/v1/calls/{call_id}/events")
def call_event(call_id: str, req: CallEventRequest, db: Session = Depends(get_db)):
    """KCD-501: idempotent by (call_id, seq)."""
    return pc.record_call_event(db, call_id, req.seq, req.kind, req.payload, req.caller_phone)


@app.get("/api/v1/calls/{call_id}")
def call_record(call_id: str, db: Session = Depends(get_db)):
    rec = pc.get_call_record(db, call_id)
    return {"found": rec is not None, "record": rec}


@app.get("/api/v1/patients/identify")
def patient_identify(phone: str = Query(...), db: Session = Depends(get_db)):
    """KCD-494: existence and count only -- never a name."""
    return pc.identify(db, phone)


class ResolveRequest(BaseModel):
    phone: str
    name: str
    age: int | None = None


@app.post("/api/v1/patients/resolve")
def patient_resolve(req: ResolveRequest, db: Session = Depends(get_db)):
    return pc.resolve_named(db, req.phone, req.name, req.age)


@app.get("/api/v1/patients/{patient_ref}/timeline")
def patient_timeline(
    patient_ref: int, caller_phone: str = Query(...), call_id: str = Query("unknown"), db: Session = Depends(get_db)
):
    """KCD-493/495: authorised, audited, and free of result values. Refused unless THIS CALL has
    passed the security questions for THIS patient (POST /api/v1/patients/verify): the server decides,
    not the caller."""
    return pc.timeline(db, patient_ref, caller_phone, call_id, verified=reg.is_verified(db, call_id, patient_ref))


@app.get("/api/v1/patients/{patient_ref}/preferences")
def patient_preferences(patient_ref: int, caller_phone: str = Query(...), db: Session = Depends(get_db)):
    return pc.get_preferences(db, patient_ref, caller_phone)


class PreferencesRequest(BaseModel):
    caller_phone: str
    branch: str | None = None
    collection_address: str | None = None
    delivery_channel: str | None = None
    accessibility_mode: str | None = None
    language_bias: str | None = None


@app.post("/api/v1/patients/{patient_ref}/preferences")
@idempotent("patients.preferences")
def set_patient_preferences(patient_ref: int, req: PreferencesRequest, db: Session = Depends(get_db)):
    fields = req.model_dump(exclude={"caller_phone"})
    return pc.set_preferences(db, patient_ref, req.caller_phone, **fields)


@app.get("/api/v1/patients/{patient_ref}/test-status")
def patient_test_status(
    patient_ref: int,
    test_name: str = Query(...),
    caller_phone: str = Query(...),
    call_id: str = Query("unknown"),
    db: Session = Depends(get_db),
):
    """KCD-498: when a test was last performed and whether it is due -- authorised and audited
    like the timeline, and carrying no result."""
    patient = db.get(pc.Patient, patient_ref)
    if patient is None or not reg.is_verified(db, call_id, patient_ref):
        pc._audit(db, call_id, patient_ref, "test_status_denied", [])
        return {"success": False, "reason": "not_verified"}
    test = db.query(LabTest).filter(LabTest.name.ilike(test_name)).first()
    if test is None:
        return {"success": False, "reason": "unknown_test"}
    pc._audit(db, call_id, patient_ref, "test_status_read", [test.name])
    return {"success": True, "test_name": test.name, **pc.test_status(db, patient_ref, test.id)}


@app.get("/api/v1/continuity")
def continuity_endpoint(
    caller_phone: str = Query(...),
    call_id: str | None = Query(None),
    patient_ref: int | None = Query(None),
    db: Session = Depends(get_db),
):
    """KCD-496: what was left unfinished, the last three interactions on this number, and the last
    calls cached for ONE DAY. The draft's details are for the agent to speak only after verification."""
    return pc.continuity(db, caller_phone, call_id, patient_id=patient_ref)


# ------------------------------------------------ security questions, find, messages, audit


class VerifyRequest(BaseModel):
    call_id: str
    patient_ref: int
    patient_id: str | None = None  # the id on the patient's card
    dob: str | None = None  # ISO date, parsed by the agent
    name: str | None = None
    address: str | None = None
    caller_phone: str | None = None


@app.post("/api/v1/patients/verify")
def patient_verify(req: VerifyRequest, db: Session = Depends(get_db)):
    """KCD-495: check the caller's answers against the registry. Two matching facts, one of them
    strong (patient id or date of birth). The reply never says which answer was wrong."""
    return reg.verify(
        db,
        req.call_id,
        req.patient_ref,
        {"patient_id": req.patient_id, "dob": req.dob, "name": req.name, "address": req.address},
        req.caller_phone,
    )


class FindRequest(BaseModel):
    call_id: str
    patient_id: str | None = None
    dob: str | None = None
    name: str | None = None
    address: str | None = None


@app.post("/api/v1/patients/find")
def patient_find(req: FindRequest, db: Session = Depends(get_db)):
    """KCD-497: find a patient by a patient id, or a date of birth with a name, when the phone number
    found nobody. Returns an opaque reference only; finding is not verifying."""
    return reg.find_by_details(
        db, req.call_id, {"patient_id": req.patient_id, "dob": req.dob, "name": req.name, "address": req.address}
    )


@app.get("/api/v1/agent/messages")
def agent_messages(lang: str | None = Query(None), db: Session = Depends(get_db)):
    """KCD-353/500/513: the operator-editable wording, read by the agent at the start of a call."""
    return am.get_messages(db, lang)


class MessageRequest(BaseModel):
    key: str
    lang: str
    text: str
    updated_by: str = "operator"
    active: bool = True


@app.put("/api/v1/agent/messages")
def agent_message_set(req: MessageRequest, db: Session = Depends(get_db)):
    """Change what callers hear, without a deploy. Behind the service token like every /api/v1 route;
    check the wording first with tools/check_messages.py."""
    return am.set_message(db, req.key, req.lang, req.text, req.updated_by, req.active)


@app.get("/api/v1/audit")
def audit_log_read(
    call_id: str | None = Query(None),
    patient_ref: int | None = Query(None),
    action: str | None = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    """The audit trail, newest first, filtered. Read-only."""
    q = db.query(AuditLog)
    if call_id:
        q = q.filter(AuditLog.call_id == call_id)
    if patient_ref is not None:
        q = q.filter(AuditLog.patient_id == patient_ref)
    if action:
        q = q.filter(AuditLog.action == action)
    rows = q.order_by(AuditLog.at.desc(), AuditLog.id.desc()).limit(limit).all()
    return {
        "entries": [
            {
                "at": r.at.isoformat(),
                "call_id": r.call_id,
                "patient_ref": r.patient_id,
                "actor": r.actor,
                "action": r.action,
                "outcome": r.outcome,
                "detail": json.loads(r.detail_json),
            }
            for r in rows
        ]
    }


class BookingSearchRequest(BaseModel):
    caller_phone: str
    phone: str | None = None
    name: str | None = None
    approx_date: str | None = None
    date_window_days: int = 3
    test_name: str | None = None
    branch: str | None = None


@app.post("/api/v1/bookings/search")
def bookings_search(req: BookingSearchRequest, db: Session = Depends(get_db)):
    """KCD-497: several matches are returned as several, never collapsed to the likeliest."""
    return pc.search_bookings(
        db,
        req.caller_phone,
        phone=req.phone,
        name=req.name,
        approx_date=req.approx_date,
        date_window_days=req.date_window_days,
        test_name=req.test_name,
        branch=req.branch,
    )


@app.get("/api/v1/bookings/{confirmation_id}/tests")
def booking_tests_endpoint(confirmation_id: str, db: Session = Depends(get_db)):
    """KCD-382b: the full set of confirmed lab tests in one test-booking group, for the
    registered/booked-patient preparation path -- see pc.booking_test_names's own docstring for
    why this is a separate read from /api/v1/bookings/search (which intentionally returns only
    one representative test per confirmation_id)."""
    return {"confirmation_id": confirmation_id, "test_names": pc.booking_test_names(db, confirmation_id)}
