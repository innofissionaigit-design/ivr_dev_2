"""Clinic data service -- implements the exact 3-endpoint contract
agent/tools_client.py in the voice agent already expects. Backed by
PostgreSQL, seeded with dummy departments/doctors/schedules/tests via
seed.py.

Entity matching lives in match_band.py, which scores every spelling the
catalogue holds and returns one of three verdicts -- commit, ambiguous,
none. It is still simple (difflib plus a containment tier) rather than the
phonetic-fold gazetteer that production callers slurring "লিপিড প্রোফাইল"
through a phone mic deserve. What changed is that it no longer resolves an
ambiguity by picking a row: several plausible rows come back as several, and
the agent asks. The single-test lookups below (search, preparation, sample,
duration, walk-in eligibility, prescription requirements) share a simpler
exact/Bengali-alias-then-fuzzy ladder (_find_lab_test() /
_lab_test_fuzzy_suggestions()) -- see search_test()'s own docstring for why
that endpoint alone goes through match_band instead.

======================================================================
UPDATED BY SOURAV -- "Lab Report Status & Secure Delivery" combined
story (previously two separate stories: "is my report ready" and "send
my report"), per the VOICE CARE AGENT FINAL TESTING / STORY / RULES /
EDGE-CASE ATTACK PLAN doc.

Added below (search "SOURAV" for every changed/new block):
  - GET  /api/v1/reports/status           (Rule 1-3, 13, 14; Section 4)
  - POST /api/v1/reports/delivery/request (Rule 3, 4, 15, 16)
  - POST /api/v1/reports/otp/verify       (Rule 4-9, 17; the whole OTP
                                            attack surface in Section 6)
  - GET  /api/v1/reports/link/{token}     (Rule 11, 12; Section 8 link
                                            attacks)

FAIL SAFE, NOT FAIL OPEN (plan Section 23) is the guiding principle for
every one of these: every endpoint below returns a NAMED reason instead
of a generic error/boolean whenever it refuses to act, precisely so the
voice agent (main.py / main_pcm.py) never has to guess why something
didn't happen, and never defaults to acting just because a check was
inconclusive.
======================================================================
"""
from __future__ import annotations

import logging

import datetime
import os
import difflib
import json
import re
import secrets
import unicodedata
import uuid

from fastapi import FastAPI, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import match_band
from db import get_db, SessionLocal
from models import (
    Department, Doctor, DoctorSchedule, LabTest, Appointment,
    # SOURAV: needed for the report-status/delivery/OTP endpoints below.
    Patient, LabReport, ReportOTP, ReportDelivery,
    # ADDED BY SOURAV -- "Caller asks about a health package" and "Caller
    # asks opening hours, address or directions" stories, below.
    ClinicInfo, HealthPackage, HealthPackageTest,
    # ADDED BY SOURAV -- "otp will not be hardcoded" -- the single shared
    # random-OTP generator every ReportOTP row now goes through (see
    # models.py's own comment on generate_otp_code() for why this lives
    # there and not duplicated here and in seed.py).
    generate_otp_code,
    # ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables. Walk-in
    # Eligibility / Prescription Requirements / Insurance Coverage Policy /
    # Outstanding Balance stories, below.
    InsuranceProvider, InsurancePolicy, PatientBilling,
    # ADDED BY SOURAV -- "Caller asks to be called back" story, below.
    CallbackRequest,
    # E4-S3 "Caller moves an existing appointment" -- the move's history
    # (which is also its idempotency record) and its queued confirmation.
    AppointmentChange, NotificationOutbox,
    # E4-S4 "Caller cancels an appointment" -- the cancellation and the
    # charge it carried (also its idempotency record).
    AppointmentCancellation,
)
# E4-S4: the admin's cancellation rules (cancellation_rules.txt), parsed and
# applied by one pure module -- the only place a charge is ever computed.
import charging_for_cancellation as charging
# ADDED BY SOURAV -- "otp will not be hardcoded": how the freshly
# generated code actually reaches the patient is a separate, pluggable
# concern -- see this module's own docstring on the file below.
from otp_messaging_config import send_otp_via_provider

app = FastAPI(title="Kolkata Care Diagnostics -- Clinic Data API (dummy)")

SLOT_STEP_MIN = 15

# "Caller asks for the earliest available appointment": how many days ahead
# the earliest-slot search looks, and how soon a slot TODAY may start (a slot
# 10 minutes from now cannot be reached). Clinic settings, from the
# environment; defaults 14 days (the same window _next_available_date uses)
# and 30 minutes.
EARLIEST_SLOT_HORIZON_DAYS = int(os.environ.get("EARLIEST_SLOT_HORIZON_DAYS", "14"))
EARLIEST_SLOT_LEAD_MINUTES = int(os.environ.get("EARLIEST_SLOT_LEAD_MINUTES", "30"))


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
    # E4-S4: converts an existing appointments table (adds status, and
    # replaces the full-table uq_doctor_slot with an index over active rows)
    # BEFORE create_all, which can create new tables but never alter one.
    import migrations
    migrations.upgrade(engine)
    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        if db.query(LabTest).count() == 0:
            logging.getLogger("clinic-api").info("empty database -- seeding catalogue")
            from seed import seed
            seed()
        else:
            logging.getLogger("clinic-api").info("catalogue already present, not reseeding")
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
        "found": True, "test_name": t.name, "test_name_bn": _first_alias_bn(t.aliases_bn),
        "rate_inr": t.rate_inr,
        "sample_type": t.sample_type, "report_time_hours": t.report_time_hours,
    }


# story title: Near matches are offered rather than guessed or refused
# user story: As a caller naming something loosely, I want the close matches
#   offered, so that I am not told my test does not exist when it does.
# acceptance criteria: When several catalogue rows fall within the match band
#   the agent offers up to three by name and asks which. Candidates are
#   generated across every supported language and romanised spelling. The
#   did-you-mean path covers the ambiguous case and not only total failure.
#
# EVERY FORM THE CATALOGUE HOLDS, in every script, scored through one
# function. The English canonical name and the Bengali aliases go into the
# same pool because a caller may say either and the ASR may land on either --
# the same argument that put aliases_bn into the old fuzzy fallback, applied
# now to the whole lookup rather than only to its last resort.
def _forms(row, *, extra=()) -> list[str]:
    aliases = [a.strip() for a in (row.aliases_bn or "").split("|") if a.strip()]
    return [row.name, *extra, *aliases]


def _candidate_dicts(rows) -> list[dict]:
    """-> [{"name", "name_bn"}, ...], or [] if ANY row cannot be said aloud.

    All or nothing, deliberately. Dropping the one candidate that has no
    Bengali alias would turn "which of these two did you mean" back into
    "did you mean this one" -- a guess wearing a question mark, which is the
    exact failure this story exists to remove. An empty list tells the agent
    to ask the caller to name it again instead, which is honest.

    Every seeded row has an alias today (seed.py guarantees it), so this is a
    trap being closed rather than a bug being fixed.
    """
    out = []
    for row in rows:
        spoken = _first_alias_bn(row.aliases_bn)
        if not spoken:
            return []
        out.append({"name": row.name, "name_bn": spoken})
    return out


def _ambiguous_reply(query: str, rows) -> dict:
    """The third response shape, alongside found and not-found.

    `ambiguous` is its own flag rather than an overloaded found=false,
    because agent/tool_outcome.py reads a falsy `found` as NOT_FOUND and an
    ambiguity counted as a thing-not-existing would poison not_found_rate --
    the one metric the preceding story built to tell an empty catalogue from
    a working one.
    """
    return {"found": False, "ambiguous": True, "query": query,
            "candidates": _candidate_dicts(rows)}


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
            {"name": t.name,
             "aliases_bn": [a for a in (t.aliases_bn or "").split("|") if a]}
            for t in db.query(LabTest).all()
        ],
        "doctors": [
            {"name": d.name,
             "surname": d.name.split()[-1],
             "aliases_bn": [a for a in (d.aliases_bn or "").split("|") if a]}
            for d in db.query(Doctor).all()
        ],
    }


def _find_lab_test(db: Session, name: str) -> LabTest | None:
    """Factored out so /api/v1/tests/preparation and the other single-test
    lookups below (sample, duration, walk-in eligibility, prescription
    requirements, ...) can reuse the exact same exact/Bengali-alias matching
    ladder rather than duplicating it. Returns None when neither stage finds
    anything; each caller decides its own not-found shape.

    NOTE: search_test() below deliberately does NOT call this -- see its own
    docstring. It goes through the match_band ranker instead so a near-miss is
    OFFERED rather than silently resolved to the first substring hit, which
    is exactly the bug this function's simpler ladder would reintroduce for
    the one endpoint whose whole story is about not doing that. Every other
    lookup here still wants the plain exact-then-fuzzy shape, so both ladders
    coexist on purpose."""
    # English substring match -- covers callers who say the test name in
    # English/transliterated form.
    exact = db.query(LabTest).filter(func.lower(LabTest.name).contains(name.lower())).first()
    if exact:
        return exact

    # Bengali-script match -- covers the actual common case. A caller
    # saying "ইউরিক এসিড" was matched against nothing before this existed:
    # the DB only stored the English name "Uric Acid", and Bengali script
    # shares zero characters with Latin script, so substring AND fuzzy
    # matching against the English column alone can NEVER succeed on
    # Bengali input, regardless of how close the pronunciation is.
    for t in db.query(LabTest).all():
        aliases = [a for a in t.aliases_bn.split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            return t

    return None


def _lab_test_fuzzy_suggestions(db: Session, name: str) -> list[str]:
    """Fuzzy-fallback logic paired with _find_lab_test() above, shared by
    /api/v1/tests/preparation and the other single-test lookups."""
    all_tests = db.query(LabTest).all()
    candidates = []
    for t in all_tests:
        candidates.append(t.name)
        candidates.extend(a for a in t.aliases_bn.split("|") if a)
    suggestions = difflib.get_close_matches(name, candidates, n=3, cutoff=0.5)
    # Map suggested aliases back to their canonical English name for display.
    alias_to_name = {a: t.name for t in all_tests for a in t.aliases_bn.split("|") if a}
    return list(dict.fromkeys(alias_to_name.get(s, s) for s in suggestions))


@app.get("/api/v1/tests/search")
def search_test(name: str = Query(...), db: Session = Depends(get_db)):
    # story title: Near matches are offered rather than guessed or refused
    # user story: As a caller naming something loosely, I want the close matches
    #   offered, so that I am not told my test does not exist when it does.
    # acceptance criteria: When several catalogue rows fall within the match band
    #   the agent offers up to three by name and asks which. Candidates are
    #   generated across every supported language and romanised spelling. The
    #   did-you-mean path covers the ambiguous case and not only total failure.
    #
    # WHAT THIS REPLACED, because the shape of the bug is not the shape people
    # expect. There were three passes: an English substring query ending in
    # .first(), a Bengali alias loop ending in `return` on its first hit, and
    # only then a fuzzy fallback. The first two did no scoring AT ALL -- so
    # "Blood Sugar", which matches both "Blood Sugar Fasting" and "Blood Sugar
    # PP", was resolved by row order, silently, and so were "সুগার",
    # "ভিটামিন" and "Vitamin". A runner-up margin alone would not have touched
    # any of them; there was no runner-up to compare against, only a list and
    # an index.
    #
    # Now every form of every row is scored once, and the same question --
    # is second place close? -- is asked on every path. A substring hit is a
    # high score rather than an early return.
    all_tests = db.query(LabTest).all()
    verdict, candidates = match_band.decide(
        match_band.rank(name, [(t, _forms(t)) for t in all_tests]))

    if verdict == match_band.COMMIT:
        return _test_reply_dict(candidates[0].key)

    if verdict == match_band.AMBIGUOUS:
        return _ambiguous_reply(name, [c.key for c in candidates])

    # Nothing cleared match_band.BAND_FLOOR, which is the same 0.50 the old
    # difflib.get_close_matches(cutoff=0.5) used -- so this is exactly the
    # case that used to produce an EMPTY suggestion list, and the keys are
    # still emitted, still empty, so a client reading them sees no change.
    #
    # The non-empty case they used to carry is now the ambiguous branch
    # above. That is the story's third clause: did-you-mean stops being a
    # total-failure consolation and becomes the same mechanism that handles
    # two rows tying at 0.90.
    return {"found": False, "query": name, "did_you_mean": [], "did_you_mean_bn": []}


# =============================================================================
# Tool 12: GET /api/v1/tests/preparation?name=...
#
# ADDED BY SOURAV -- "Caller asks how to prepare for a test" story.
# Reuses /api/v1/tests/search's own exact/Bengali-alias/fuzzy matching
# ladder unchanged (see _find_lab_test()/_lab_test_fuzzy_suggestions()
# above) so a test that can already be priced or looked up can also be
# asked about by name here -- this endpoint only differs in what it
# returns once the row is found.
# =============================================================================

def _test_preparation_reply_dict(t: LabTest) -> dict:
    """`advisory_available=False` is a real, honest, DIFFERENT outcome
    from `found=False` (test_preparation()'s own not-found branch below):
    the test itself exists and can be looked up fine, but nobody has
    ever supplied real preparation content for it (see models.py's own
    comment on why LabTest's advisory columns are nullable with no
    default). The voice agent must tell these two apart -- suggesting
    "did you mean" alternatives for a test that WAS found correctly
    would be nonsensical, and guessing "no special preparation" for one
    with no advisory row would be fabricating a medical instruction."""
    if t.fasting_required is None:
        return {
            "found": True, "test_name": t.name, "test_name_bn": _first_alias_bn(t.aliases_bn),
            "advisory_available": False,
        }
    return {
        "found": True, "test_name": t.name, "test_name_bn": _first_alias_bn(t.aliases_bn),
        "advisory_available": True,
        "fasting_required": t.fasting_required,
        "fasting_hours": t.fasting_hours,
        "water_allowance": t.water_allowance,
        "medication_hold": t.medication_hold,
        "timing_rule": t.timing_rule,
        "advisory_script_en": t.advisory_script_en,
        "advisory_script_hinglish": t.advisory_script_hinglish,
        "advisory_script_banglish": t.advisory_script_banglish,
        "advisory_script_bn": t.advisory_script_bn,
    }


@app.get("/api/v1/tests/preparation")
def test_preparation(name: str = Query(...), db: Session = Depends(get_db)):
    t = _find_lab_test(db, name)
    if t:
        return _test_preparation_reply_dict(t)
    suggestions = _lab_test_fuzzy_suggestions(db, name)
    return {"found": False, "query": name, "did_you_mean": suggestions}


# =============================================================================
# Tool 13: GET /api/v1/tests/walkin-policy?name=...
#
# ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables. Walk-in
# Eligibility story. Reuses /api/v1/tests/search's own exact/Bengali-alias/
# fuzzy matching ladder unchanged (see _find_lab_test()/
# _lab_test_fuzzy_suggestions() above), same as /api/v1/tests/preparation
# just above -- only what gets returned once the row is found differs.
# =============================================================================

def _walkin_policy_reply_dict(t: LabTest) -> dict:
    """`policy_available=False` is a real, honest, DIFFERENT outcome from
    `found=False` below -- same "found vs. has-real-content" split as
    test_preparation's own advisory_available (see that function's own
    docstring for the full reasoning): the test exists and can be looked
    up fine, but nobody has confirmed its walk-in policy yet. Never
    guess "walk-ins welcome" for a test with no reviewed answer."""
    if t.walkin_eligible is None:
        return {
            "found": True, "test_name": t.name, "test_name_bn": _first_alias_bn(t.aliases_bn),
            "policy_available": False,
        }
    return {
        "found": True, "test_name": t.name, "test_name_bn": _first_alias_bn(t.aliases_bn),
        "policy_available": True,
        "walkin_eligible": t.walkin_eligible,
        "walkin_hours": t.walkin_hours,
    }


@app.get("/api/v1/tests/walkin-policy")
def test_walkin_policy(name: str = Query(...), db: Session = Depends(get_db)):
    t = _find_lab_test(db, name)
    if t:
        return _walkin_policy_reply_dict(t)
    suggestions = _lab_test_fuzzy_suggestions(db, name)
    return {"found": False, "query": name, "did_you_mean": suggestions}


# =============================================================================
# Tool 14: GET /api/v1/tests/prescription-policy?name=...
#
# ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables. Prescription
# Requirements story. Same matching ladder and found/policy_available
# split as walkin-policy just above.
# =============================================================================

def _prescription_policy_reply_dict(t: LabTest) -> dict:
    if t.prescription_required is None:
        return {
            "found": True, "test_name": t.name, "test_name_bn": _first_alias_bn(t.aliases_bn),
            "policy_available": False,
        }
    # prescription_channels is stored "|"-joined (same convention as
    # aliases_bn) -- split it into a real list for the caller-facing
    # layer here, same as aliases_bn is split wherever it's read.
    channels = [c for c in (t.prescription_channels or "").split("|") if c]
    return {
        "found": True, "test_name": t.name, "test_name_bn": _first_alias_bn(t.aliases_bn),
        "policy_available": True,
        "prescription_required": t.prescription_required,
        "prescription_channels": channels,
    }


@app.get("/api/v1/tests/prescription-policy")
def test_prescription_policy(name: str = Query(...), db: Session = Depends(get_db)):
    t = _find_lab_test(db, name)
    if t:
        return _prescription_policy_reply_dict(t)
    suggestions = _lab_test_fuzzy_suggestions(db, name)
    return {"found": False, "query": name, "did_you_mean": suggestions}


# =============================================================================
# Tool 15: GET /api/v1/insurance/coverage?test_name=...&provider_name=...
#
# ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables. Insurance
# Coverage Policy story. Two names to resolve, not one: the test (same
# ladder as every other LabTest lookup above) AND the insurer (its own
# exact/alias ladder below, same shape as _find_lab_test's, over
# InsuranceProvider.aliases instead of LabTest.aliases_bn).
# =============================================================================

def _find_insurance_provider(db: Session, name: str) -> InsuranceProvider | None:
    """Same two-stage ladder as _find_lab_test() above: substring match on
    the canonical name first, then a "|"-joined alias match -- covers a
    caller naming their insurer in English, Hinglish/Banglish, or Bengali
    script, the same way LabTest.aliases_bn covers a test name."""
    exact = db.query(InsuranceProvider).filter(
        func.lower(InsuranceProvider.name).contains(name.lower())
    ).first()
    if exact:
        return exact
    for p in db.query(InsuranceProvider).all():
        aliases = [a for a in p.aliases.split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            return p
    return None


@app.get("/api/v1/insurance/coverage")
def insurance_coverage(test_name: str = Query(...), provider_name: str = Query(...),
                        db: Session = Depends(get_db)):
    t = _find_lab_test(db, test_name)
    if not t:
        suggestions = _lab_test_fuzzy_suggestions(db, test_name)
        return {"test_found": False, "query": test_name, "did_you_mean": suggestions}

    provider = _find_insurance_provider(db, provider_name)
    if not provider:
        # Honest "we don't recognise that insurer", never silently
        # matched to the wrong one and never assumed "not covered".
        return {
            "test_found": True, "test_name": t.name, "provider_found": False,
            "query_provider": provider_name,
        }

    policy = db.query(InsurancePolicy).filter_by(test_id=t.id, provider_id=provider.id).first()
    if policy is None or policy.coverage_status is None:
        # No reviewed (test, provider) row -- honest "we don't know yet",
        # never a guessed COVERED/NOT_COVERED (see models.py's own
        # comment on InsurancePolicy.coverage_status).
        return {
            "test_found": True, "test_name": t.name,
            "provider_found": True, "provider_name": provider.name,
            "policy_available": False,
        }
    return {
        "test_found": True, "test_name": t.name,
        "provider_found": True, "provider_name": provider.name,
        "policy_available": True,
        "coverage_status": policy.coverage_status,
        "pre_auth_required": policy.pre_auth_required,
    }


# =============================================================================
# Tool 2: GET /api/v1/doctors/availability?name=...&date=YYYY-MM-DD (optional)
# Tool 2b: GET /api/v1/doctors/by-department?department=...
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


# story title: Near matches are offered rather than guessed or refused
# user story: As a caller naming something loosely, I want the close matches
#   offered, so that I am not told my test does not exist when it does.
# acceptance criteria: When several catalogue rows fall within the match band
#   the agent offers up to three by name and asks which. Candidates are
#   generated across every supported language and romanised spelling. The
#   did-you-mean path covers the ambiguous case and not only total failure.
#
# The surname is passed as an extra form because it is what callers actually
# say -- "Sen", not "Dr. A. Sen" -- and because it is the form that makes the
# tiering matter: "সেন" EQUALS Dr Sen's alias (1.0) and is CONTAINED IN Dr
# Sengupta's (0.90), so it clears the margin and commits, where a flat
# containment score for both would have asked the caller to choose between a
# doctor they named exactly and one they did not.
#
# FUZZY_SURNAME_FLOOR's 0.60 is no longer a commit threshold -- match_band
# commits at 0.72 and OFFERS between 0.50 and 0.72. That band used to be a
# silent commit. The Doctor Nobody incident that set 0.60 in the first place
# (a wrong doctor matched at 0.522) lands in it: the caller is now asked
# "did you mean Dr Roy?" and can say no, instead of being answered about a
# doctor they never named.
def _resolve_doctor(db: Session, name: str) -> tuple[str, Doctor | None, list[Doctor]]:
    """-> (verdict, the one doctor if committing, the rows to offer)."""
    all_doctors = db.query(Doctor).all()
    rows = [(d, _forms(d, extra=[d.name.split()[-1]])) for d in all_doctors]
    verdict, candidates = match_band.decide(match_band.rank(name, rows))
    if verdict == match_band.COMMIT:
        return verdict, candidates[0].key, []
    return verdict, None, [c.key for c in candidates]


# story title: Near matches are offered rather than guessed or refused
# user story: As a caller naming something loosely, I want the close matches
#   offered, so that I am not told my test does not exist when it does.
# acceptance criteria: When several catalogue rows fall within the match band
#   the agent offers up to three by name and asks which. Candidates are
#   generated across every supported language and romanised spelling. The
#   did-you-mean path covers the ambiguous case and not only total failure.
#
# Departments are the one entity type whose aliases ALREADY span scripts --
# "কার্ডিওলজি", "হার্ট", "heart", "cardio". They go into the pool exactly as
# they are, which is why the criterion's romanised-spelling clause is met here
# and not for tests or doctors: it is a data question, and the seed rows for
# those two carry Bengali script plus the English name only. Noted as the
# known gap rather than closed, because seed() is destructive and reseeding is
# an operational decision, not a deploy.
def _resolve_department(db: Session, department_name: str) -> tuple[str, Department | None, list[Department]]:
    """-> (verdict, the one department if committing, the rows to offer)."""
    all_departments = db.query(Department).all()
    verdict, candidates = match_band.decide(
        match_band.rank(department_name, [(d, _forms(d)) for d in all_departments]))
    if verdict == match_band.COMMIT:
        return verdict, candidates[0].key, []
    return verdict, None, [c.key for c in candidates]


def _schedule_for_weekday(db: Session, doctor_id: int, weekday: int) -> DoctorSchedule | None:
    return db.query(DoctorSchedule).filter_by(doctor_id=doctor_id, weekday=weekday).first()


def _next_available_date(db: Session, doctor_id: int, from_date: datetime.date,
                          horizon_days: int = 14) -> str | None:
    for offset in range(horizon_days):
        d = from_date + datetime.timedelta(days=offset)
        if _schedule_for_weekday(db, doctor_id, d.weekday()):
            return d.isoformat()
    return None


@app.get("/api/v1/doctors/availability")
def doctor_availability(name: str = Query(...), date: str | None = Query(None),
                         time: str | None = Query(None, pattern=r"^\d{2}:\d{2}$"),
                         db: Session = Depends(get_db)):
    verdict, doctor, offered = _resolve_doctor(db, name)
    if verdict == match_band.AMBIGUOUS:
        return _ambiguous_reply(name, offered)
    if not doctor:
        return {"found": False, "query": name}

    today = datetime.date.today()

    if date:
        try:
            target = datetime.date.fromisoformat(date)
        except ValueError:
            return {"found": False, "query": name}
        sched = _schedule_for_weekday(db, doctor.id, target.weekday())
        if sched:
            reply = {
                "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": target.isoformat(),
                "available": True, "chamber_hours": f"{sched.start_time}-{sched.end_time}",
                "next_available_date": None,
            }
            # "Requested slot is already taken": with a time, also say
            # whether that slot is free right now, and if not, what is.
            if time:
                now = datetime.datetime.now()
                lead = now + datetime.timedelta(minutes=EARLIEST_SLOT_LEAD_MINUTES)
                too_soon = target < now.date() or (
                    target == now.date() and (lead.date() != target or time < lead.strftime("%H:%M")))
                reply["slot_free"] = not too_soon and time in _free_slots(
                    db, doctor.id, target.isoformat(), sched, limit=None)
                if not reply["slot_free"]:
                    reply.update(_slot_alternatives(db, doctor.id, target.isoformat(), time))
            return reply
        next_date = _next_available_date(db, doctor.id, target + datetime.timedelta(days=1))
        return {
            "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": target.isoformat(),
            "available": False, "chamber_hours": None, "next_available_date": next_date,
        }

    # No date given -> "when is this doctor next available"
    next_date = _next_available_date(db, doctor.id, today)
    if not next_date:
        return {
            "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": None,
            "available": False, "chamber_hours": None, "next_available_date": None,
        }
    sched = _schedule_for_weekday(db, doctor.id, datetime.date.fromisoformat(next_date).weekday())
    return {
        "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": next_date,
        "available": True, "chamber_hours": f"{sched.start_time}-{sched.end_time}",
        "next_available_date": None,
    }


@app.get("/api/v1/doctors/schedule")
def doctor_schedule(name: str = Query(...), db: Session = Depends(get_db)):
    """ADDED BY SOURAV -- "Caller asks when a doctor sits" story.

    Deliberately DATE-FREE, unlike doctor_availability() just above. That
    endpoint always resolves to one particular day (today, an explicit
    date, or the computed "next available" day) because it answers "is
    the doctor in THEN". This endpoint answers a different, more general
    question a caller actually asks in practice -- "which days does Dr X
    usually sit?" -- with no date involved at all: it returns the
    doctor's FULL recurring weekly schedule, every DoctorSchedule row
    they have, in weekday order (0=Monday .. 6=Sunday, matching
    DoctorSchedule's own docstring). The caller-facing wording is built
    from this list in agent/reply_templates.py::doctor_schedule_reply().

    Response shapes:
      found=false: {"found": false, "query": "..."}
      found=true, doctor has 1+ scheduled weekdays:
        {"found": true, "doctor_name": "...", "doctor_name_bn": "...",
         "schedule": [{"weekday": 0, "start_time": "10:00", "end_time": "12:00"}, ...]}
      found=true, doctor exists but has ZERO DoctorSchedule rows (a real,
      honest edge case -- e.g. a doctor on indefinite leave with no
      chamber days configured at all): "schedule": [] -- the caller-facing
      reply function must say so plainly rather than fabricating a day.
    """
    doctor = _find_doctor(db, name)
    if not doctor:
        return {"found": False, "query": name}

    rows = (
        db.query(DoctorSchedule)
        .filter_by(doctor_id=doctor.id)
        .order_by(DoctorSchedule.weekday.asc())
        .all()
    )
    return {
        "found": True,
        "doctor_name": doctor.name,
        "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
        "schedule": [
            {"weekday": r.weekday, "start_time": r.start_time, "end_time": r.end_time}
            for r in rows
        ],
    }


@app.get("/api/v1/doctors/by-department")
def doctors_by_department(department: str = Query(...), date: str | None = Query(None),
                           db: Session = Depends(get_db)):
    """Get all doctors in a department, supports short forms like 'ortho' for
    Orthopaedics.

    `date` is OPTIONAL, same convention as /doctors/availability above.
    Omit it for a plain "who's in this department" listing (every doctor,
    unfiltered -- unchanged from before this parameter existed). Pass it
    for "which ortho doctor is available TODAY/that day": the list is
    filtered down to doctors who actually have a DoctorSchedule row for
    that date's weekday, and each gets its chamber_hours attached, mirroring
    what doctor_availability() already reports for a single named doctor.
    """
    verdict, dept, offered = _resolve_department(db, department)
    if verdict == match_band.AMBIGUOUS:
        return _ambiguous_reply(department, offered)
    if not dept:
        return {"found": False, "query": department}

    doctors = db.query(Doctor).filter_by(department_id=dept.id).all()

    target = None
    if date:
        try:
            target = datetime.date.fromisoformat(date)
        except ValueError:
            target = None  # malformed date -- fall back to the unfiltered listing

    out = []
    for d in doctors:
        entry = {
            "name": d.name,
            "doctor_name_bn": _first_alias_bn(d.aliases_bn),
            "qualifications": d.qualifications,
        }
        if target is not None:
            sched = _schedule_for_weekday(db, d.id, target.weekday())
            if not sched:
                continue  # doesn't sit that day -- excluded, not just flagged
            entry["chamber_hours"] = f"{sched.start_time}-{sched.end_time}"
        out.append(entry)

    return {
        "found": True,
        "department": dept.name,
        # STORY [Answer Quality and Grounding]
        # As a patient, I want to hear the whole sentence, so that I am
        # not left guessing what the agent tried to say.
        # The SPOKEN department name. Without it the agent's listing reply
        # reads "<English> বিভাগে ... আছেন" and the Bengali tokenizer drops the
        # Latin word, so the caller loses the SUBJECT of the sentence -- on
        # every one of the eight seeded departments, not an edge case. Same
        # helper, same reason, as test_name_bn and doctor_name_bn above; every
        # department is seeded with a Bengali alias first (see seed.py's
        # DEPARTMENT_ALIASES), so this needs no new data.
        "department_bn": _first_alias_bn(dept.aliases_bn),
        "date": target.isoformat() if target else None,
        "doctors": out,
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
def book_appointment(req: BookingRequest, db: Session = Depends(get_db)):
    # story title: Near matches are offered rather than guessed or refused
    # user story: As a caller naming something loosely, I want the close matches
    #   offered, so that I am not told my test does not exist when it does.
    # acceptance criteria: When several catalogue rows fall within the match band
    #   the agent offers up to three by name and asks which. Candidates are
    #   generated across every supported language and romanised spelling. The
    #   did-you-mean path covers the ambiguous case and not only total failure.
    #
    # A WRITE NEVER PROCEEDS UNDER AMBIGUITY. Booking the higher-scoring of
    # two plausible doctors is the worst version of this bug: the caller
    # leaves believing they have an appointment, and they do -- with someone
    # else. The refusal carries the candidates so the agent can ask, rather
    # than reporting "no such doctor" for a doctor who exists twice over.
    verdict, doctor, offered = _resolve_doctor(db, req.doctor_name)
    if verdict == match_band.AMBIGUOUS:
        return {"success": False, "reason": "doctor_ambiguous",
                "candidates": _candidate_dicts(offered)}
    if not doctor:
        return {"success": False, "reason": "doctor_not_found"}

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
        # Not one of his 15-minute slots. Used to return valid_slots[:3]
        # with no check for bookings -- a booked slot could be offered.
        return {"success": False, "reason": "slot_taken",
                **_slot_alternatives(db, doctor.id, req.date, req.time_slot)}

    taken = {
        a.time_slot for a in db.query(Appointment).filter_by(
            # E4-S4: a cancelled appointment no longer holds its slot.
            doctor_id=doctor.id, date=req.date, status="active",
        ).all()
    }
    if req.time_slot in taken:
        return {"success": False, "reason": "slot_taken",
                **_slot_alternatives(db, doctor.id, req.date, req.time_slot)}

    confirmation_id = f"KCD-{req.date.replace('-', '')}-{uuid.uuid4().hex[:4].upper()}"
    appt = Appointment(
        confirmation_id=confirmation_id, doctor_id=doctor.id, date=req.date,
        time_slot=req.time_slot, patient_name=req.patient_name, phone=req.phone,
        created_at=datetime.datetime.now(),
    )
    db.add(appt)
    # E4-S3: the "is it taken?" check above and this insert are two
    # statements, so two callers can both pass the check. uq_doctor_slot then
    # rejects the second insert -- which used to escape as an unhandled
    # IntegrityError (HTTP 500), so the losing caller heard "cannot check
    # right now" about a slot that was simply taken. Same root cause the
    # reschedule swap below relies on, handled the same way.
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return {"success": False, "reason": "slot_taken",
                **_slot_alternatives(db, doctor.id, req.date, req.time_slot)}

    return {
        "success": True, "confirmation_id": confirmation_id,
        "doctor_name": doctor.name,
        "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": req.date, "time_slot": req.time_slot,
    }


# =============================================================================
# E4-S3 -- Caller moves an existing appointment
#
#   GET   /api/v1/appointments/lookup?phone=&name=&reference=
#   GET   /api/v1/appointments/{reference}/availability?date=
#   PATCH /api/v1/appointments/{reference}
#
# story title: Caller moves an existing appointment
# user story: As a patient whose plans changed, I want to move my appointment
#   without cancelling it, so that I do not lose my place entirely.
# acceptance criteria: The booking is found by contact number, name or
#   reference. The new slot is swapped atomically, holding the old one until
#   the new commits, and a failed swap leaves the original intact.
#   Confirmation is sent on both channels.
#
# Plan and decisions D1-D6: docs/stories/E4-S3-reschedule-plan.md.
# =============================================================================

def _free_slots(db: Session, doctor_id: int, date_iso: str, sched: DoctorSchedule,
                limit: int | None = 3) -> list[str]:
    """The doctor's bookable slots on that date that nobody holds, earliest
    first. Read fresh from the table, so it reflects every committed write.
    limit=None -> all of them."""
    taken = {a.time_slot for a in db.query(Appointment).filter_by(
        doctor_id=doctor_id, date=date_iso, status="active").all()}  # E4-S4
    return [s for s in _generate_slots(sched.start_time, sched.end_time)
            if s not in taken][:limit]


def _hhmm_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _slot_alternatives(db: Session, doctor_id: int, date_iso: str, time_slot: str) -> dict:
    """"Requested slot is already taken" (docs/stories/slot-taken-alternatives-plan.md).

    THE one definition of what may be offered instead of a taken (or
    invalid) time, used by the booking write's refusals AND by the check
    before the readback, so both say the same thing:
      alternative_slots -- the 2 free slots that day nearest in minutes to
                           the time asked (a tie goes to the earlier),
                           in time order;
      other_day_slot    -- the same time on the nearest other day (either
                           side, never in the past, within the horizon) on
                           which the doctor sits and that time is free. If
                           the time asked is not a slot, the nearest slot
                           time that day stands in for it.
    Read from the ACTIVE appointments at call time. Today: nothing already
    passed or inside the lead time. (A future slot hold -- E4-S1 -- belongs
    here too: a held slot is not free.)
    """
    now = datetime.datetime.now()
    today = now.date()
    lead = now + datetime.timedelta(minutes=EARLIEST_SLOT_LEAD_MINUTES)
    cut = lead.strftime("%H:%M") if lead.date() == today else "24:00"

    def usable(day: datetime.date, t: str) -> bool:
        return day > today or (day == today and t >= cut)

    target = datetime.date.fromisoformat(date_iso)
    asked = _hhmm_minutes(time_slot)
    same, ref = [], time_slot
    sched = _schedule_for_weekday(db, doctor_id, target.weekday())
    if sched:
        valid = _generate_slots(sched.start_time, sched.end_time)
        free = [t for t in _free_slots(db, doctor_id, date_iso, sched, limit=None)
                if t != time_slot and usable(target, t)]
        nearest = sorted(free, key=lambda t: (abs(_hhmm_minutes(t) - asked), _hhmm_minutes(t)))[:2]
        same = sorted(nearest)
        if time_slot not in valid and valid:
            ref = min(valid, key=lambda t: (abs(_hhmm_minutes(t) - asked), _hhmm_minutes(t)))

    other = None
    for dist in range(1, EARLIEST_SLOT_HORIZON_DAYS + 1):
        for day in (target - datetime.timedelta(days=dist), target + datetime.timedelta(days=dist)):
            if day < today or not usable(day, ref):
                continue
            day_sched = _schedule_for_weekday(db, doctor_id, day.weekday())
            if day_sched and ref in _free_slots(db, doctor_id, day.isoformat(), day_sched, limit=None):
                other = {"date": day.isoformat(), "time_slot": ref}
                break
        if other:
            break
    return {"alternative_slots": same, "other_day_slot": other}


# =============================================================================
# "Caller asks for the earliest available appointment"
#
#   GET /api/v1/doctors/earliest-slots?name=...&limit=3
#
# Read only. The doctor's first free slots, earliest first, walking forward
# from today up to EARLIEST_SLOT_HORIZON_DAYS. Built from the SAME functions
# the booking write uses to accept a slot (_schedule_for_weekday,
# _generate_slots, _free_slots over ACTIVE appointments), so an offered slot
# is one the write would accept at the moment it was read. Nothing is held:
# the write re-checks, and refuses a slot taken in between.
# Plan: docs/stories/earliest-appointment-plan.md.
# =============================================================================
@app.get("/api/v1/doctors/earliest-slots")
def doctor_earliest_slots(name: str = Query(...), limit: int = Query(3, ge=1, le=10),
                          db: Session = Depends(get_db)):
    verdict, doctor, offered = _resolve_doctor(db, name)
    if verdict == match_band.AMBIGUOUS:
        return _ambiguous_reply(name, offered)
    if not doctor:
        return {"found": False, "query": name}

    now = datetime.datetime.now()
    today = now.date()
    earliest_today = (now + datetime.timedelta(minutes=EARLIEST_SLOT_LEAD_MINUTES)).strftime("%H:%M")
    slots = []
    for offset in range(EARLIEST_SLOT_HORIZON_DAYS):
        day = today + datetime.timedelta(days=offset)
        sched = _schedule_for_weekday(db, doctor.id, day.weekday())
        if not sched:
            continue
        # A lead time past midnight means nothing is left today.
        if offset == 0 and (now + datetime.timedelta(minutes=EARLIEST_SLOT_LEAD_MINUTES)).date() != today:
            continue
        for t in _free_slots(db, doctor.id, day.isoformat(), sched, limit=None):
            if offset == 0 and t < earliest_today:
                continue
            slots.append({"date": day.isoformat(), "time_slot": t,
                          "chamber_hours": f"{sched.start_time}-{sched.end_time}"})
            if len(slots) >= limit:
                break
        if len(slots) >= limit:
            break

    return {
        "found": True, "doctor_name": doctor.name,
        "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
        "horizon_days": EARLIEST_SLOT_HORIZON_DAYS, "slots": slots,
        "as_of": now.isoformat(timespec="seconds"),
    }


# [REASONED, not measured] How close a spoken patient name must be to the
# stored one. Never decisive on its own: D1 below means a phone number or a
# reference must ALSO agree, so a false accept needs two independent errors.
# Calibrate on real ASR spellings of patient names before trusting it.
NAME_MATCH_FLOOR = 0.75

# DECISION D1 -- any TWO of {phone, name, reference} must point at the same
# appointment before anything about it is returned. One factor alone is
# refused, identically whether or not something would have matched, so the
# endpoint cannot be used to probe. A stand-in for OTP (Blueprint 4.10,
# E7-S2), which does not exist for bookings yet. One constant, not a design.
MIN_IDENTITY_FACTORS = 2


def _phone_key(s: str | None) -> str | None:
    """Last 10 digits, or None. "+91 98765 43210" == "9876543210"."""
    digits = re.sub(r"\D", "", s or "")
    return digits[-10:] if len(digits) >= 10 else None


def _name_key(s: str | None) -> str:
    return " ".join(unicodedata.normalize("NFC", s or "").split()).strip().lower()


def _reference_key(s: str | None) -> str:
    return re.sub(r"[^0-9A-Z]", "", (s or "").upper())


def _name_matches(spoken: str | None, stored: str | None) -> bool:
    a, b = _name_key(spoken), _name_key(stored)
    if not a or not b:
        return False
    if a == b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= NAME_MATCH_FLOOR


def _reference_matches(spoken: str | None, stored: str | None) -> bool:
    """Tolerates stray letters around a spoken ID (stored contained in
    spoken) and a dropped "KCD" prefix or a suffix-only answer (stored ends
    with spoken). At least 4 characters, so "C" cannot match everything."""
    a, b = _reference_key(spoken), _reference_key(stored)
    if len(a) < 4 or not b:
        return False
    return b in a or b.endswith(a)


def _appointment_summary(appt: Appointment) -> dict:
    """EVERYTHING a caller may be told about an appointment. Phone and
    patient name are never returned by any endpoint in this section."""
    return {
        "reference": appt.confirmation_id,
        "doctor_name": appt.doctor.name,
        "doctor_name_bn": _first_alias_bn(appt.doctor.aliases_bn),
        "date": appt.date,
        "time_slot": appt.time_slot,
    }


@app.get("/api/v1/appointments/lookup")
def lookup_appointments(phone: str | None = Query(None), name: str | None = Query(None),
                        reference: str | None = Query(None),
                        db: Session = Depends(get_db)):
    """AC1. Future appointments that at least MIN_IDENTITY_FACTORS of the
    given factors agree on. Several matches are all returned (earliest
    first); the agent asks which, and never takes the likeliest (E14-S4)."""
    phone_k, name_k, ref_k = _phone_key(phone), _name_key(name), _reference_key(reference)
    given = sum(1 for k in (phone_k, name_k, ref_k) if k)
    if given < MIN_IDENTITY_FACTORS:
        return {"found": False, "reason": "insufficient_identification", "matches": []}

    today = datetime.date.today().isoformat()
    rows = (db.query(Appointment)
            .filter(Appointment.date >= today,
                    Appointment.status == "active")  # E4-S4: never offer a cancelled one
            .order_by(Appointment.date.asc(), Appointment.time_slot.asc())
            .all())
    matches = []
    for appt in rows:
        factors = 0
        if phone_k and _phone_key(appt.phone) == phone_k:
            factors += 1
        if name_k and _name_matches(name, appt.patient_name):
            factors += 1
        if ref_k and _reference_matches(reference, appt.confirmation_id):
            factors += 1
        if factors >= MIN_IDENTITY_FACTORS:
            matches.append(_appointment_summary(appt))
    return {"found": bool(matches), "matches": matches}


@app.get("/api/v1/appointments/{reference}/availability")
def appointment_availability(reference: str, date: str = Query(...),
                             db: Session = Depends(get_db)):
    """Does THIS appointment's doctor sit on `date`?

    Scoped to the appointment, never to a doctor NAME: re-matching by name
    goes through match_band, which can land on a different doctor or come
    back ambiguous (the seed has both "Dr. Sen" and "Dr. A. Sen"), and a
    reschedule must only ever talk about the doctor it already has. Returns
    no doctor and no patient field -- a reference alone reveals nothing but
    chamber hours.
    """
    appt = db.query(Appointment).filter_by(confirmation_id=reference,
                                           status="active").first()  # E4-S4
    if appt is None:
        return {"found": False, "query": reference}
    try:
        target = datetime.date.fromisoformat(date)
    except ValueError:
        return {"found": False, "query": reference}
    sched = _schedule_for_weekday(db, appt.doctor_id, target.weekday())
    if sched:
        return {"found": True, "date": target.isoformat(), "available": True,
                "chamber_hours": f"{sched.start_time}-{sched.end_time}",
                "next_available_date": None}
    return {"found": True, "date": target.isoformat(), "available": False,
            "chamber_hours": None,
            "next_available_date": _next_available_date(
                db, appt.doctor_id, target + datetime.timedelta(days=1))}


class RescheduleRequest(BaseModel):
    new_date: str
    new_time_slot: str
    # The slot that was READ BACK to the caller. The move only happens if
    # the appointment is still in it (see the conditional UPDATE below).
    expected_date: str
    expected_time_slot: str
    # Blueprint 4.9: call_id + action id, so a retry can never move twice.
    idempotency_key: str
    call_id: str | None = None


def _stored_result(db: Session, key: str) -> dict | None:
    row = db.query(AppointmentChange).filter_by(idempotency_key=key).first()
    return json.loads(row.result_json) if row else None


_OUTBOX_FIELDS = ("reference", "doctor_name", "doctor_name_bn",
                  "old_date", "old_time_slot", "new_date", "new_time_slot")


def _record_change(db: Session, appt: Appointment, req: RescheduleRequest,
                   result: dict) -> None:
    """History + outbox rows, INSIDE the caller's transaction. Never commits."""
    now = datetime.datetime.now()
    db.add(AppointmentChange(
        appointment_id=appt.id,
        old_date=result["old_date"], old_time_slot=result["old_time_slot"],
        new_date=result["new_date"], new_time_slot=result["new_time_slot"],
        call_id=req.call_id, idempotency_key=req.idempotency_key,
        result_json=json.dumps(result, ensure_ascii=False), created_at=now,
    ))
    db.add(NotificationOutbox(
        appointment_id=appt.id, kind="reschedule_confirmation", channel="sms",
        template_id="reschedule_confirmation_v1",
        payload_json=json.dumps({k: result[k] for k in _OUTBOX_FIELDS}, ensure_ascii=False),
        status="pending", created_at=now,
    ))
    db.flush()


@app.patch("/api/v1/appointments/{reference}")
def reschedule_appointment(reference: str, req: RescheduleRequest,
                           db: Session = Depends(get_db)):
    """AC2 -- the atomic swap.

    Every refusal below returns BEFORE any write, so the original is
    untouched. The write itself is one transaction: a conditional UPDATE of
    the appointment row (it matches only if the row is still in the slot
    that was read back), the history row and the outbox row. The row keeps
    its old slot until COMMIT; uq_doctor_slot makes the new slot exclusive;
    any failure rolls all three back together. The doctor is ALWAYS
    appt.doctor_id -- never re-matched by name (D4).
    """
    stored = _stored_result(db, req.idempotency_key)
    if stored is not None:
        return stored

    # E4-S4: a cancelled appointment cannot be moved -- it reads as not found.
    appt = db.query(Appointment).filter_by(confirmation_id=reference, status="active").first()
    if appt is None:
        return {"success": False, "reason": "not_found"}

    today = datetime.date.today()
    if appt.date < today.isoformat():
        return {"success": False, "reason": "past"}
    if (appt.date, appt.time_slot) != (req.expected_date, req.expected_time_slot):
        return {"success": False, "reason": "conflict",
                "current_date": appt.date, "current_time_slot": appt.time_slot}
    try:
        target = datetime.date.fromisoformat(req.new_date)
    except ValueError:
        return {"success": False, "reason": "invalid_date"}
    if target < today:  # D3: date-level only, no minimum notice
        return {"success": False, "reason": "past"}
    if (req.new_date, req.new_time_slot) == (appt.date, appt.time_slot):
        return {"success": False, "reason": "same_slot"}

    sched = _schedule_for_weekday(db, appt.doctor_id, target.weekday())
    if not sched:
        return {"success": False, "reason": "doctor_not_available_that_day",
                "next_available_date": _next_available_date(
                    db, appt.doctor_id, target + datetime.timedelta(days=1))}
    if req.new_time_slot not in _generate_slots(sched.start_time, sched.end_time):
        return {"success": False, "reason": "slot_taken",
                "alternative_slots": _free_slots(db, appt.doctor_id, req.new_date, sched)}

    # Built BEFORE the update, while the ORM object still holds the old slot.
    result = {
        "success": True, **_appointment_summary(appt),
        "old_date": appt.date, "old_time_slot": appt.time_slot,
        "new_date": req.new_date, "new_time_slot": req.new_time_slot,
        "date": req.new_date, "time_slot": req.new_time_slot,
    }
    appt_id, doctor_id = appt.id, appt.doctor_id

    try:
        moved = db.execute(
            update(Appointment)
            .where(Appointment.id == appt_id,
                   Appointment.status == "active",  # E4-S4: not cancelled meanwhile
                   Appointment.date == req.expected_date,
                   Appointment.time_slot == req.expected_time_slot)
            .values(date=req.new_date, time_slot=req.new_time_slot)
            .execution_options(synchronize_session=False)
        ).rowcount
        if moved != 1:
            # Changed between our read and our write. Say so; never overwrite.
            db.rollback()
            stored = _stored_result(db, req.idempotency_key)
            if stored is not None:
                return stored
            current = db.get(Appointment, appt_id)
            return {"success": False, "reason": "conflict",
                    "current_date": current.date if current else None,
                    "current_time_slot": current.time_slot if current else None}
        _record_change(db, appt, req, result)
        db.commit()
    except IntegrityError:
        # uq_doctor_slot: somebody holds the new slot. Or a concurrent retry
        # of THIS request won the idempotency_key -- then replay its answer.
        db.rollback()
        stored = _stored_result(db, req.idempotency_key)
        if stored is not None:
            return stored
        return {"success": False, "reason": "slot_taken",
                "alternative_slots": _free_slots(db, doctor_id, req.new_date, sched)}
    except Exception:
        db.rollback()
        raise
    return result


# =============================================================================
# E4-S4 -- Caller cancels an appointment
#
#   GET  /api/v1/appointments/{reference}/cancellation-quote
#   POST /api/v1/appointments/{reference}/cancel
#
# story title: Caller cancels an appointment
# user story: As a patient who cannot attend, I want to cancel and be told any
#   charge clearly, so that I am not surprised by a deduction later.
# acceptance criteria: Cancellation applies the configured window rules and
#   states refund eligibility from policy, never improvised. A cancellation
#   within a charging window is confirmed explicitly with the charge stated
#   before it is applied.
#
# The rules are the admin's text file (cancellation_rules.txt), applied by
# charging_for_cancellation.py. The quote the caller HEARS and the charge that
# is RECORDED both come from _quote_now() below -- and the cancel endpoint
# computes it AGAIN at commit, refusing if it differs from what the caller
# was told. Plan and decisions: docs/stories/E4-S4-cancel-plan.md.
# =============================================================================

def _now() -> datetime.datetime:
    """The only clock the cancellation rules see (tests replace it)."""
    return datetime.datetime.now(charging.CLINIC_TZ)


def _cancellation_policy(now: datetime.datetime):
    """The admin's rules, read fresh on EVERY request -- a saved edit applies
    to the next caller with no restart. None whenever they cannot be used:
    missing, not understood, unapproved, or not yet in force. None means
    nothing is cancelled by phone; a charge is never guessed."""
    path = os.environ.get("CANCELLATION_RULES_PATH", charging.DEFAULT_RULES_PATH)
    try:
        policy = charging.load_policy(path)
    except charging.PolicyError as e:
        logging.getLogger("clinic-api").error("cancellation rules unusable: %s", e)
        return None
    if not charging.is_in_force(policy, now):
        logging.getLogger("clinic-api").warning(
            "cancellation rules %s start on %s -- not in force yet",
            policy.version, policy.effective_from)
        return None
    return policy


def _quote_now(appt: Appointment, policy, now: datetime.datetime) -> dict:
    start = charging.appointment_start(appt.date, appt.time_slot)
    return charging.quote_cancellation(policy, start, now).to_dict()


# Every key a quote can carry, so both shapes of a found=true response have
# the same keys (null where they do not apply) and the agent's contract can
# require all of them.
_EMPTY_QUOTE = {"cancellable": False, "reason": None, "window_id": None, "hours_before": None,
                "charge_inr": 0, "refund_eligibility": None, "refund_percent": None,
                "policy_version": None}


@app.get("/api/v1/appointments/{reference}/cancellation-quote")
def cancellation_quote(reference: str, db: Session = Depends(get_db)):
    """What cancelling THIS appointment NOW would cost, under the rules.

    Writes nothing. Returns no doctor, date, phone or name: the agent already
    holds the appointment it identified through the two-factor lookup."""
    appt = db.query(Appointment).filter_by(confirmation_id=reference).first()
    if appt is None:
        return {"found": False, "reason": "not_found", "query": reference}
    if appt.status != "active":
        return {"found": False, "reason": "already_cancelled", "query": reference}
    now = _now()
    policy = _cancellation_policy(now)
    if policy is None:
        return {"found": True, "reference": reference,
                **_EMPTY_QUOTE, "reason": "policy_unavailable"}
    return {"found": True, "reference": reference, **_quote_now(appt, policy, now)}


class CancelRequest(BaseModel):
    # The slot that was READ BACK to the caller.
    expected_date: str
    expected_time_slot: str
    # The quote that was SPOKEN to the caller. The cancel only happens if
    # the rules, applied now, still give exactly this.
    expected_window_id: str
    expected_charge_inr: int
    policy_version: str
    # True only after the caller explicitly agreed to a stated charge.
    charge_confirmed: bool = False
    # Blueprint 4.9: call_id + action id, so a retry can never cancel -- or
    # charge -- twice.
    idempotency_key: str
    call_id: str | None = None


def _stored_cancellation(db: Session, key: str) -> dict | None:
    row = db.query(AppointmentCancellation).filter_by(idempotency_key=key).first()
    return json.loads(row.result_json) if row else None


_CANCEL_OUTBOX_FIELDS = ("reference", "doctor_name", "doctor_name_bn", "date", "time_slot",
                         "charge_inr", "charge_status", "refund_eligibility",
                         "refund_percent", "policy_version")


def _record_cancellation(db: Session, appointment_id: int, req: CancelRequest,
                         quote: dict, result: dict) -> None:
    """Cancellation + outbox rows, INSIDE the caller's transaction. Never
    commits. The outbox payload carries no phone and no patient name."""
    now = datetime.datetime.now()
    db.add(AppointmentCancellation(
        appointment_id=appointment_id, policy_version=quote["policy_version"],
        window_id=quote["window_id"], hours_before=quote["hours_before"],
        charge_inr=quote["charge_inr"], charge_status=result["charge_status"],
        refund_eligibility=quote["refund_eligibility"], refund_percent=quote["refund_percent"],
        call_id=req.call_id, idempotency_key=req.idempotency_key,
        result_json=json.dumps(result, ensure_ascii=False), created_at=now,
    ))
    db.add(NotificationOutbox(
        appointment_id=appointment_id, kind="cancellation_confirmation", channel="sms",
        template_id="cancellation_confirmation_v1",
        payload_json=json.dumps({k: result[k] for k in _CANCEL_OUTBOX_FIELDS}, ensure_ascii=False),
        status="pending", created_at=now,
    ))
    db.flush()


@app.post("/api/v1/appointments/{reference}/cancel")
def cancel_appointment(reference: str, req: CancelRequest, db: Session = Depends(get_db)):
    """AC1 + AC2, enforced here and not only in the dialogue.

    Every refusal returns BEFORE any write. The write is one transaction: a
    conditional UPDATE (matches only while the row is still active and still
    in the slot that was read back), the cancellation row with its charge,
    and the outbox row. Any failure rolls all three back together.
    """
    stored = _stored_cancellation(db, req.idempotency_key)
    if stored is not None:
        return stored

    appt = db.query(Appointment).filter_by(confirmation_id=reference).first()
    if appt is None:
        return {"success": False, "reason": "not_found"}
    if appt.status != "active":
        return {"success": False, "reason": "already_cancelled"}
    if (appt.date, appt.time_slot) != (req.expected_date, req.expected_time_slot):
        return {"success": False, "reason": "conflict",
                "current_date": appt.date, "current_time_slot": appt.time_slot}

    now = _now()
    policy = _cancellation_policy(now)
    if policy is None:
        return {"success": False, "reason": "policy_unavailable"}
    quote = _quote_now(appt, policy, now)
    if not quote["cancellable"]:
        return {"success": False, "reason": quote["reason"], "quote": quote}
    # The caller agreed to what they HEARD. If a window boundary passed, or
    # the admin changed the rules, while they were answering, that is no
    # longer the charge -- never apply a charge the caller did not hear.
    if (quote["window_id"], quote["charge_inr"], quote["policy_version"]) != (
            req.expected_window_id, req.expected_charge_inr, req.policy_version):
        return {"success": False, "reason": "quote_changed", "quote": quote}
    if quote["charge_inr"] > 0 and not req.charge_confirmed:
        return {"success": False, "reason": "charge_not_confirmed", "quote": quote}

    # Built BEFORE the update, while the ORM object still holds the slot.
    result = {
        "success": True, **_appointment_summary(appt),
        "charge_inr": quote["charge_inr"],
        "charge_status": "pending_collection" if quote["charge_inr"] > 0 else "none",
        "refund_eligibility": quote["refund_eligibility"],
        "refund_percent": quote["refund_percent"],
        "window_id": quote["window_id"], "policy_version": quote["policy_version"],
    }
    appt_id = appt.id

    try:
        cancelled = db.execute(
            update(Appointment)
            .where(Appointment.id == appt_id,
                   Appointment.status == "active",
                   Appointment.date == req.expected_date,
                   Appointment.time_slot == req.expected_time_slot)
            # Naive clinic wall time, like every other DateTime column here.
            .values(status="cancelled", cancelled_at=now.replace(tzinfo=None))
            .execution_options(synchronize_session=False)
        ).rowcount
        if cancelled != 1:
            # Changed between our read and our write. Say so; never overwrite.
            db.rollback()
            stored = _stored_cancellation(db, req.idempotency_key)
            if stored is not None:
                return stored
            current = db.get(Appointment, appt_id)
            if current is None or current.status != "active":
                return {"success": False, "reason": "already_cancelled"}
            return {"success": False, "reason": "conflict",
                    "current_date": current.date, "current_time_slot": current.time_slot}
        _record_cancellation(db, appt_id, req, quote, result)
        db.commit()
    except IntegrityError:
        # A concurrent retry of THIS request won the idempotency key.
        db.rollback()
        stored = _stored_cancellation(db, req.idempotency_key)
        if stored is not None:
            return stored
        raise
    except Exception:
        db.rollback()
        raise
    return result

# =============================================================================
# Tool 5: GET /api/v1/reports/status?phone=...&test_name=... (optional)
# Tool 6: POST /api/v1/reports/delivery/request
# Tool 7: POST /api/v1/reports/otp/verify
# Tool 8: GET /api/v1/reports/link/{token}
#
# ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story.
# See this file's module docstring for the rule references. Every helper
# and endpoint below is new; nothing above this line was touched except
# the import block.
# =============================================================================

OTP_VALIDITY_MINUTES = 10
SIGNED_LINK_VALIDITY_MINUTES = 15
# UPDATED BY SOURAV -- "otp will not be hardcoded" (the user's own
# explicit instruction, superseding the earlier prototype decision this
# comment used to document: "the otp [is] hardcoded, for now user will
# tell the otp and the matching will be done"). A report with no
# pre-seeded ReportOTP row (or whose only rows are used/expired/maxed)
# now gets a genuinely random code from models.generate_otp_code() every
# time delivery is requested live, exactly like every other ReportOTP
# row (seeded or live) already does -- see that function's own docstring.
# Seeded patients that already carry their own ReportOTP row (Arjun/
# Sohini/Amit -- see seed.py SECTION 9) still take priority via the
# `reusable` check just below; this path only fires for reports with no
# usable row yet (e.g. Mita's and the two Rahul Das reports).
#
# Whoever actually needs the code now (a real caller, or a tester with no
# provider connected) gets it via send_otp_via_provider() -- best-effort,
# see clinic-api/otp_messaging_config.py -- or, with no provider
# connected, by reading the freshly-created ReportOTP row directly from
# the database, the same way this file's own test suite already does.


def _mask_phone_last4(phone: str) -> str:
    """Rule 10: never say the full registered number back to the caller.
    Used only in this file's own response payloads (masked_phone) --
    reply_templates.py has its own copy for anything it composes
    directly from a phone slot the caller spoke, since that module must
    not import from clinic-api (they are separate deployables)."""
    digits = "".join(c for c in phone if c.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits


def _find_patient_by_phone(db: Session, phone: str) -> Patient | None:
    return db.query(Patient).filter_by(phone=phone).first()


def _report_summary(report: LabReport, test_name: str) -> dict:
    return {
        "report_number": report.report_number,
        "test_name": test_name,
        "status": report.status,
        "delivery_enabled": report.delivery_enabled,
        "expected_ready_at": report.expected_ready_at.isoformat() if report.expected_ready_at else None,
        "ready_at": report.ready_at.isoformat() if report.ready_at else None,
    }


@app.get("/api/v1/reports/status")
def report_status(phone: str = Query(...), test_name: str | None = Query(None),
                   db: Session = Depends(get_db)):
    """RULE 1 (never invent a report), RULE 13 (multiple reports require
    clarification), RULE 14 (same-name patients must not be merged).

    Identity is resolved by PHONE, never by name -- this is what makes
    RULE 14 hold structurally rather than by convention: Patient.phone is
    a unique column (see models.py), so two "Rahul Das" rows can never
    collide here regardless of how the caller pronounces the name. A
    caller who states only a name and no phone is a slot the voice agent
    must ask for BEFORE calling this endpoint at all (see main_pcm.py's
    new "phone" pending state) -- this endpoint has no name-based lookup
    path to accidentally fall back to.
    """
    patient = _find_patient_by_phone(db, phone)
    if not patient:
        return {"patient_found": False}

    reports = db.query(LabReport).filter_by(patient_id=patient.id).all()
    if not reports:
        # RULE 1: an honest NOT_FOUND, never a fabricated report.
        return {"patient_found": True, "found": False, "reason": "NOT_FOUND"}

    lab_test_ids = {r.lab_test_id for r in reports}
    tests_by_id = {t.id: t for t in db.query(LabTest).filter(LabTest.id.in_(lab_test_ids)).all()}

    if test_name:
        matches = [r for r in reports if test_name.lower() in tests_by_id[r.lab_test_id].name.lower()]
    else:
        matches = reports

    if not matches:
        # A test_name was given but nothing this patient has matches it --
        # honest NOT_FOUND rather than silently falling back to "all
        # reports" (which would let a misheard test name return an
        # unrelated report).
        return {"patient_found": True, "found": False, "reason": "NOT_FOUND"}

    if len(matches) > 1:
        # RULE 13: multiple candidates -> ask, never guess.
        return {
            "patient_found": True, "found": False, "reason": "AMBIGUOUS",
            "candidates": [_report_summary(r, tests_by_id[r.lab_test_id].name) for r in matches],
        }

    report = matches[0]
    return {"patient_found": True, "found": True, **_report_summary(report, tests_by_id[report.lab_test_id].name)}


def _blocking_reason_for(report: LabReport) -> str | None:
    """RULE 3 / RULE 16: only a READY, delivery-enabled report may enter
    the delivery flow. Returns the specific reason delivery is blocked,
    or None if it is allowed to proceed -- used identically by both the
    delivery-request and otp-verify endpoints below, so a report that
    changes state between those two calls (e.g. cancelled in between) is
    re-checked, not trusted from the first call."""
    if report.status != "READY":
        return report.status  # NOT_READY / PROCESSING / CANCELLED
    if not report.delivery_enabled:
        return "DELIVERY_DISABLED"
    return None


def _resolve_patient_and_report(db: Session, phone: str, report_number: str):
    """Shared identity+report resolution for delivery/request and
    otp/verify. Returns (patient, report, error_reason). error_reason is
    None on success.

    RULE 15: the report must belong to the PHONE-resolved patient, not
    merely exist. A real report_number for a DIFFERENT patient (ATTACK
    15) and a nonexistent one both return the same "NOT_FOUND" -- never a
    distinct "wrong patient" reason, which would let an attacker probe
    which report numbers are real by watching the error change.
    """
    patient = _find_patient_by_phone(db, phone)
    if not patient:
        return None, None, "PATIENT_NOT_FOUND"

    report = db.query(LabReport).filter_by(report_number=report_number).first()
    if not report or report.patient_id != patient.id:
        return patient, None, "NOT_FOUND"

    return patient, report, None


class DeliveryRequest(BaseModel):
    phone: str
    report_number: str


@app.post("/api/v1/reports/delivery/request")
def request_report_delivery(req: DeliveryRequest, db: Session = Depends(get_db)):
    """RULE 4 (OTP required before delivery), RULE 6/7/8 (an exhausted,
    expired or already-used OTP is never silently reused -- a fresh
    request always gets a fresh, USABLE OTP row instead)."""
    patient, report, error = _resolve_patient_and_report(db, req.phone, req.report_number)
    if error:
        return {"success": False, "reason": error}

    blocked = _blocking_reason_for(report)
    if blocked:
        return {"success": False, "reason": blocked}

    now = datetime.datetime.now()

    # Reuse the most recent OTP row for this (report, patient) ONLY if it
    # is still usable. RULE 8's "require a new verification flow" is what
    # this is: a maxed-out, expired or already-used row is treated as
    # dead, and a brand-new row is minted instead of ever handing the
    # same exhausted OTP back out.
    active = (
        db.query(ReportOTP)
        .filter_by(report_id=report.id, patient_id=patient.id)
        # UPDATED BY SOURAV -- tie-break on id, not just created_at.
        # Two OTP rows for the same (report, patient) can share an
        # identical created_at timestamp (Python's datetime.now() and
        # SQLite's storage resolution both make this a real possibility,
        # not just a seed-data artifact -- clinic-api/seed.py's own
        # Patient E rows originally did exactly this before being fixed
        # to use distinct timestamps). Without a secondary sort key,
        # "most recent" was undefined on a tie -- caught by
        # tests/test_clinic_api_reports.py::TestOtpVerify::
        # test_max_attempts_already_reached returning OTP_EXPIRED instead
        # of OTP_MAX_ATTEMPTS. `id` is autoincrement, so it is a reliable
        # insertion-order tie-breaker regardless of what the clock reads.
        .order_by(ReportOTP.created_at.desc(), ReportOTP.id.desc())
        .first()
    )
    reusable = (
        active is not None
        and not active.used
        and active.expires_at > now
        and active.attempt_count < active.max_attempts
    )

    if not reusable:
        active = ReportOTP(
            report_id=report.id,
            patient_id=patient.id,
            phone=patient.phone,
            otp_code=generate_otp_code(),
            created_at=now,
            expires_at=now + datetime.timedelta(minutes=OTP_VALIDITY_MINUTES),
            used=False,
            attempt_count=0,
        )
        db.add(active)
        db.commit()

    # ADDED BY SOURAV -- "otp will not be hardcoded". Every delivery
    # request -- whether it just minted a fresh row above or is reusing
    # an existing valid one -- attempts to actually push the current
    # active code out through whatever provider a deploying company has
    # configured (see otp_messaging_config.py, the one file they need to
    # touch). Best-effort and never blocking: `active`'s row is already
    # committed to the database by this point regardless of whether this
    # send succeeds, matching this whole file's "fail safe, not fail
    # open" posture (module docstring) -- a provider hiccup must never be
    # the reason a patient can't proceed. send_otp_via_provider() already
    # promises never to raise on its own, but this endpoint's own success
    # response is still wrapped in a try/except here too -- defense in
    # depth, the same posture every other tie-break/edge case in this
    # file already takes, rather than resting entirely on another file's
    # contract holding forever.
    try:
        send_otp_via_provider(patient.phone, active.otp_code)
    except Exception as e:
        logging.getLogger("clinic-api").error(
            "send_otp_via_provider() raised unexpectedly for report %s: %s",
            report.report_number, e,
        )

    return {
        "success": True, "reason": "OTP_REQUIRED",
        "masked_phone": _mask_phone_last4(patient.phone),
    }


class OtpVerifyRequest(BaseModel):
    phone: str
    report_number: str
    otp_code: str
    # SOURAV: test-only hook for CASE 2 ("Delivery failure tests" /
    # provider failure) in the attack plan's Section 9. The voice agent
    # itself never sets this -- there is no real SMS/e-mail provider in
    # this prototype to fail on its own, so this is how the automated
    # test suite deterministically exercises "OTP was right, but the
    # provider failed" (RULE 17) without needing real delivery infra.
    simulate_delivery_failure: bool = False


@app.post("/api/v1/reports/otp/verify")
def verify_report_otp(req: OtpVerifyRequest, db: Session = Depends(get_db)):
    """The entire OTP attack surface (plan Section 6) lives in this one
    function, in a fixed order, on purpose -- RULE 9 says never reveal
    the correct OTP, so the code is compared LAST, after every other
    reason to refuse has already been ruled out; an attacker never
    learns anything about the correct value from which check fired."""
    patient, report, error = _resolve_patient_and_report(db, req.phone, req.report_number)
    if error:
        return {"success": False, "reason": error}

    blocked = _blocking_reason_for(report)
    if blocked:
        return {"success": False, "reason": blocked}

    now = datetime.datetime.now()

    active = (
        db.query(ReportOTP)
        .filter_by(report_id=report.id, patient_id=patient.id)
        # UPDATED BY SOURAV -- tie-break on id, not just created_at.
        # Two OTP rows for the same (report, patient) can share an
        # identical created_at timestamp (Python's datetime.now() and
        # SQLite's storage resolution both make this a real possibility,
        # not just a seed-data artifact -- clinic-api/seed.py's own
        # Patient E rows originally did exactly this before being fixed
        # to use distinct timestamps). Without a secondary sort key,
        # "most recent" was undefined on a tie -- caught by
        # tests/test_clinic_api_reports.py::TestOtpVerify::
        # test_max_attempts_already_reached returning OTP_EXPIRED instead
        # of OTP_MAX_ATTEMPTS. `id` is autoincrement, so it is a reliable
        # insertion-order tie-breaker regardless of what the clock reads.
        .order_by(ReportOTP.created_at.desc(), ReportOTP.id.desc())
        .first()
    )
    if active is None:
        return {"success": False, "reason": "OTP_NOT_REQUESTED"}

    if active.used:
        # RULE 7: an OTP is single-use, full stop -- even if the caller
        # types the exact right value again.
        return {"success": False, "reason": "OTP_ALREADY_USED"}

    if active.expires_at <= now:
        # RULE 6.
        return {"success": False, "reason": "OTP_EXPIRED"}

    if active.attempt_count >= active.max_attempts:
        # RULE 8 -- checked BEFORE the code comparison, so a caller who
        # is already locked out never gets another "wrong"/"right"
        # signal about a code that no longer matters.
        return {"success": False, "reason": "OTP_MAX_ATTEMPTS"}

    if active.otp_code != req.otp_code:
        # RULE 9: the response says only that it was wrong, never what
        # the right one is.
        active.attempt_count += 1
        db.commit()
        if active.attempt_count >= active.max_attempts:
            return {"success": False, "reason": "OTP_MAX_ATTEMPTS"}
        return {"success": False, "reason": "OTP_INVALID"}

    # Correct code, and every gate above passed -- RULE 5 held throughout
    # because `active` was scoped to (report.id, patient.id) from the
    # very first query above: an OTP that is valid for a DIFFERENT report
    # or patient is simply never the row being compared against here, so
    # ATTACK 5/6 (cross-report / cross-patient OTP reuse) fail not
    # because of a special case, but because the right row was never
    # found in the first place.
    active.used = True
    active.verified_at = now
    db.commit()

    if req.simulate_delivery_failure:
        # RULE 17: OTP success must NOT be conflated with delivery
        # success -- the two are recorded and reported separately.
        delivery = ReportDelivery(
            report_id=report.id, patient_id=patient.id,
            recipient=f"registered contact for {patient.phone}",
            delivery_channel="EMAIL", verification_status="VERIFIED",
            delivery_status="FAILED", created_at=now, verified_at=now,
            failed_at=now, failure_reason="DELIVERY_PROVIDER_FAILURE",
            audit_note="OTP verified successfully; delivery provider failed on send.",
        )
        db.add(delivery)
        db.commit()
        return {"success": False, "reason": "DELIVERY_FAILED"}

    token = secrets.token_urlsafe(16)
    expires_at = now + datetime.timedelta(minutes=SIGNED_LINK_VALIDITY_MINUTES)
    delivery = ReportDelivery(
        report_id=report.id, patient_id=patient.id,
        recipient=f"registered contact for {patient.phone}",
        delivery_channel="EMAIL", verification_status="VERIFIED",
        delivery_status="SENT", signed_link_token=token,
        signed_link_expires_at=expires_at, created_at=now,
        verified_at=now, sent_at=now,
        audit_note="OTP verified; report delivered successfully.",
    )
    db.add(delivery)
    db.commit()

    return {
        "success": True, "reason": "DELIVERY_SENT",
        "masked_phone": _mask_phone_last4(patient.phone),
        "signed_link_expires_minutes": SIGNED_LINK_VALIDITY_MINUTES,
    }


@app.get("/api/v1/reports/link/{token}")
def validate_report_link(token: str, db: Session = Depends(get_db)):
    """RULE 11 (links must expire), RULE 12 (no direct access without
    authorization). This is an HTTP-level endpoint, not something the
    voice agent itself calls -- the caller never speaks a token back
    over the phone. It exists so the plan's Section 8 link/token attacks
    (23-28: expired, reused, modified, empty, random, cross-report) have
    something real to attack, matching a real "click the link we sent
    you" step in the actual delivery channel."""
    if not token:
        return {"valid": False, "reason": "LINK_INVALID"}

    delivery = db.query(ReportDelivery).filter_by(signed_link_token=token).first()
    if not delivery:
        # Covers a genuinely random token, an empty one, AND a modified
        # one (flipping one character of a real token overwhelmingly
        # lands on a value nothing in signed_link_token equals) --
        # ATTACK 25/26/27 all resolve to this same branch, which is the
        # point: a corrupted token carries no partial credit.
        return {"valid": False, "reason": "LINK_INVALID"}

    if not delivery.signed_link_expires_at or delivery.signed_link_expires_at <= datetime.datetime.now():
        return {"valid": False, "reason": "LINK_EXPIRED"}

    linked_report = db.query(LabReport).filter_by(id=delivery.report_id).first()
    return {
        "valid": True,
        "report_number": linked_report.report_number if linked_report else None,
    }


# =============================================================================
# Tool 16: GET /api/v1/patient/billing?phone=...
#
# ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables. Outstanding
# Balance / Billing story. Identity resolved by PHONE (RULE 14/15), same
# as /api/v1/reports/status above, reusing the same _find_patient_by_phone
# helper -- deliberately NOT gated behind OTP verification the way report
# delivery is (see models.py's PatientBilling docstring for that scoping
# decision).
# =============================================================================

@app.get("/api/v1/patient/billing")
def patient_billing(phone: str = Query(...), db: Session = Depends(get_db)):
    patient = _find_patient_by_phone(db, phone)
    if not patient:
        return {"patient_found": False}

    billing = db.query(PatientBilling).filter_by(patient_id=patient.id).first()
    if billing is None or billing.outstanding_amount is None:
        # No billing record for this patient at all -- honest NOT_FOUND,
        # never a guessed/defaulted zero balance (see models.py's
        # PatientBilling docstring: no row and a real 0.0 are different,
        # both real, outcomes).
        return {"patient_found": True, "found": False, "reason": "NOT_FOUND"}

    return {
        "patient_found": True, "found": True,
        "outstanding_amount": billing.outstanding_amount,
        "due_date": billing.due_date.isoformat() if billing.due_date else None,
    }


# =============================================================================
# Tool 9: GET /api/v1/clinic/info
#
# ADDED BY SOURAV -- "Caller asks opening hours, address or directions"
# story. models.py's own ClinicInfo docstring lists the exact caller
# phrasings this backs ("When do you open?" / "Clinic kab khulta hai?" /
# "ঠিকানাটা কী?") -- the DB already stored everything needed (see that
# model's own comment: "The actual voice response language should ideally
# be handled by the agent/template layer... The DB stores the factual
# information"); this endpoint is simply the missing HTTP surface over it.
# A singleton table (seed.py inserts exactly one row) -- no query params.
# =============================================================================

# Ordered Monday-first to match DoctorSchedule's own weekday convention
# (0=Monday .. 6=Sunday) used throughout this file, so any future caller
# of this endpoint can zip the two together without a re-mapping step.
_CLINIC_WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)


@app.get("/api/v1/clinic/info")
def clinic_info(db: Session = Depends(get_db)):
    """found=false is a real, honest edge case (an unseeded/emptied
    ClinicInfo table), not something the voice agent should ever silently
    paper over with a guessed address -- same fail-safe posture as every
    other endpoint in this file. Normally there is exactly one row."""
    info = db.query(ClinicInfo).first()
    if not info:
        return {"found": False}

    hours = {}
    for day in _CLINIC_WEEKDAYS:
        closed = bool(getattr(info, f"{day}_closed", False))
        hours[day] = {
            "closed": closed,
            "open": None if closed else getattr(info, f"{day}_open"),
            "close": None if closed else getattr(info, f"{day}_close"),
        }

    return {
        "found": True,
        "clinic_name": info.clinic_name,
        "phone": info.phone,
        "address": info.address,
        "directions": info.directions,
        "hours": hours,
    }


# =============================================================================
# Tool 10: GET /api/v1/health-packages
# Tool 11: GET /api/v1/health-packages/search?name=...
#
# ADDED BY SOURAV -- "Caller asks about a health package" story. Mirrors
# /api/v1/tests/search's own English -> Bengali-alias -> fuzzy matching
# ladder exactly (see that endpoint's comments for why each stage exists),
# now that seed.py actually seeds HealthPackage.aliases -- see this file's
# own "ADDED BY SOURAV" comment in seed.py's HEALTH_PACKAGES list for that
# half of this story.
# =============================================================================

def _package_tests(db: Session, pkg: HealthPackage) -> list[LabTest]:
    return (
        db.query(LabTest)
        .join(HealthPackageTest, HealthPackageTest.lab_test_id == LabTest.id)
        .filter(HealthPackageTest.package_id == pkg.id)
        .all()
    )


def _package_reply_dict(db: Session, pkg: HealthPackage) -> dict:
    tests = _package_tests(db, pkg)
    return {
        "found": True,
        "package_name": pkg.name,
        # _first_alias_bn() is named for its original (LabTest/Doctor)
        # callers, but its behaviour -- "first non-empty '|'-separated
        # entry" -- has nothing Bengali-specific about it; HealthPackage's
        # own alias list (seed.py) deliberately mixes English/Hinglish/
        # Banglish/Bengali-script entries the same way, so it is reused
        # here as-is rather than duplicated under a new name.
        "package_name_bn": _first_alias_bn(pkg.aliases),
        "description": pkg.description,
        "price_inr": pkg.price_inr,
        "tests": [t.name for t in tests],
        "tests_bn": [_first_alias_bn(t.aliases_bn) for t in tests],
    }


def _find_health_package(db: Session, name: str) -> HealthPackage | None:
    active = db.query(HealthPackage).filter_by(active=True)

    # English substring match.
    exact = active.filter(func.lower(HealthPackage.name).contains(name.lower())).first()
    if exact:
        return exact

    all_packages = active.all()

    # Alias match (English/Hinglish/Banglish/Bengali-script -- see
    # seed.py's HEALTH_PACKAGES aliases for exactly what this catches).
    for p in all_packages:
        aliases = [a for a in (p.aliases or "").split("|") if a]
        if any(name.lower() in alias.lower() or alias.lower() in name.lower() for alias in aliases):
            return p

    # Fuzzy fallback -- same shape as _find_department()'s own fallback.
    best_pkg, best_ratio = None, 0.0
    for p in all_packages:
        candidates = [p.name.lower()] + [a.lower() for a in (p.aliases or "").split("|") if a]
        for c in candidates:
            ratio = difflib.SequenceMatcher(None, name.lower(), c).ratio()
            if ratio > best_ratio:
                best_pkg, best_ratio = p, ratio

    return best_pkg if best_ratio >= 0.6 else None


@app.get("/api/v1/health-packages")
def list_health_packages(db: Session = Depends(get_db)):
    """A plain "what packages do you have?" listing -- every ACTIVE
    package, unfiltered. Deliberately excludes inactive packages (same
    posture as active-only lookups elsewhere): a caller should never be
    offered, or able to ask follow-up questions about, a package the
    clinic has withdrawn."""
    packages = db.query(HealthPackage).filter_by(active=True).all()
    return {
        "packages": [_package_reply_dict(db, p) for p in packages],
    }


@app.get("/api/v1/health-packages/search")
def search_health_package(name: str = Query(...), db: Session = Depends(get_db)):
    pkg = _find_health_package(db, name)
    if pkg:
        return _package_reply_dict(db, pkg)

    # Fuzzy "did you mean" suggestions -- same pattern as search_test()'s
    # own fallback: try both the English name and every alias, then map
    # suggested aliases back to their canonical English name for display.
    active_packages = db.query(HealthPackage).filter_by(active=True).all()
    candidates = []
    for p in active_packages:
        candidates.append(p.name)
        candidates.extend(a for a in (p.aliases or "").split("|") if a)
    suggestions = difflib.get_close_matches(name, candidates, n=3, cutoff=0.5)
    alias_to_name = {
        a: p.name for p in active_packages for a in (p.aliases or "").split("|") if a
    }
    suggestions = list(dict.fromkeys(alias_to_name.get(s, s) for s in suggestions))
    return {"found": False, "query": name, "did_you_mean": suggestions}


# =============================================================================
# Tool 12: POST /api/v1/callbacks
#
# ADDED BY SOURAV -- "Caller asks to be called back" story. See
# models.py's own CallbackRequest docstring for why this is its own table,
# and agent/callback_flow.py's module docstring for why "operating hours"
# is checked entirely on the VOICE AGENT side (main.py), before this
# endpoint is ever called -- this endpoint's only job is to persist a
# request that has already been decided to be acceptable; it does not
# re-check availability itself, the same division of labour
# book_appointment() above already has with the agent's own slot
# collection (the agent decides WHEN to call this; this endpoint decides
# only HOW to store what it's given).
# =============================================================================
class CallbackRequestIn(BaseModel):
    phone: str
    time_window: str
    reason: str | None = None


@app.post("/api/v1/callbacks")
def request_callback(req: CallbackRequestIn, db: Session = Depends(get_db)):
    """Always succeeds for a syntactically valid body (FastAPI/Pydantic
    already reject a request missing `phone`/`time_window` with a 422
    before this function body ever runs) -- unlike book_appointment()
    above, there is no slot-taken/doctor-not-found domain check to fail:
    a callback request has no capacity limit and needs nothing else to
    already exist in the database. `success` is still returned, matching
    every other write in this file's response shape, so main.py's existing
    "check success, then check the fields that must be non-empty before
    speaking them" pattern (see agent/outcomes.py's missing_booking_write_
    fields(), mirrored for this endpoint as missing_callback_write_fields())
    needs no special-casing for this endpoint."""
    callback_id = f"CB-{datetime.date.today().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
    row = CallbackRequest(
        callback_id=callback_id,
        phone=req.phone,
        time_window=req.time_window,
        reason=req.reason,
        status="pending",
        created_at=datetime.datetime.now(),
    )
    db.add(row)
    db.commit()

    return {
        "success": True,
        "callback_id": callback_id,
        "phone": req.phone,
        "time_window": req.time_window,
        "reason": req.reason,
        "status": "pending",
    }
