"""Unit tests for the four fixes behind one live call transcript.

    1. [User] ডাক্তার অবৈলাবল আছে            [AI] কোন ডাক্তারের কথা জিজ্ঞেস করছেন?
       [User] কে কে ডাক্তার অবৈলাবল আছে সেটা বলুন   [AI] কোন ডাক্তারের কথা জিজ্ঞেস করছেন?
    2. [User] আপনাদের কি কি ডাক্তার এখানে বসেন  [AI] ডাঃ সেন ওই দিন বসবেন না ...
    3. [User] না অন্য তিন নভেম্বর কি উনি বসবেন   [AI] আজকের জন্য চান, নাকি অন্য কোনো দিনের জন্য ...
       [User] থার্ড নভেম্বর এর জন্য অযাপ্ফোন্টমেন্ট পাওয়া যাবে  [AI] (the same question)
    4. [User] অন্য দিনের জন্যে                   [AI] ... যে ফোন নম্বরে বুক করেছিলেন সেটা বলবেন?

Causes, one per section below:
    1. no intent for "which doctors are there"      -> agent/doctor_list.py
    2. the last doctor backfilled into a list question -> agent/state.py
    3. no date rule for a day next to a month name   -> agent/slot_parse.py
    4. "another day" read as a reschedule            -> agent/slot_parse.py

Everything here is pure or runs the REAL clinic-api in-process against the
REAL seed data; tests/test_live_transcript_replay.py replays the call itself
through the dispatcher.
"""
from __future__ import annotations

import asyncio
import datetime
import importlib.util
import os
import pathlib
import re
import sys
import types

import httpx
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLINIC_API_DIR = str(ROOT / "clinic-api")

# The AGENT's main, bound now: real_clinic_api below swaps clinic-api's own
# `main` into sys.modules for the length of this module, so an `import main`
# inside a test body would get the wrong one.
import main as agent_main  # noqa: E402
from agent import speakability  # noqa: E402
from agent.doctor_list import (  # noqa: E402
    LIST_DOCTORS, apply_list_doctors_override, is_doctor_list_request, match_department,
)
from agent.llm import VALID_INTENTS, _SLOT_KEYS, _validate  # noqa: E402
from agent.reply_templates import (  # noqa: E402
    anything_else_reply, booking_which_date_prompt, doctor_list_reply, new_or_move_prompt,
)
from agent.slot_parse import (  # noqa: E402
    has_reschedule_cue, parse_date, parse_new_or_move, wants_unspecified_other_day,
)
from agent.state import DialogueState, resolve_follow_up  # noqa: E402

TODAY = datetime.date(2026, 9, 28)   # the day of the live call (a Monday)


def _slots(**kw):
    s = {k: None for k in _SLOT_KEYS}
    s.update(kw)
    return s


# =========================================================================
# 1. "Which doctors are there?"
# =========================================================================

LIST_QUESTIONS = [
    "কে কে ডাক্তার অবৈলাবল আছে সেটা বলুন",          # live transcript
    "আপনাদের কি কি ডাক্তার এখানে বসেন",             # live transcript
    "কোন কোন ডাক্তার আছেন",
    "সব ডাক্তারের নাম বলুন",
    "ডাক্তারদের লিস্ট দিন",
    "ডাক্তার কারা আছেন",
    "কতজন ডাক্তার বসেন",
    "কোন ডাক্তার আজ বসেন",
    "ডাক্তাররা কবে বসেন",
    "which doctors are available",
    "who are the doctors here",
    "all doctors",
    "ke ke doctor ache",
    "kaun kaun se doctor hai",
]
NOT_LIST_QUESTIONS = [
    "ডাক্তার সেন কবে বসেন",
    "ডাক্তার সেন কি আজ আছেন",
    "সব ঠিক আছে, ডাক্তার সেন কবে বসেন",   # "সব" + "ডাক্তার", but not next to each other
    "কি কি টেস্ট করাতে হবে",               # a list -- of tests
    "CBC এর দাম কত",
    "অন্য দিনের জন্যে",
    "is Dr Sen available today",
    "ওনার ফি কত",
    "",
]


class TestListRequestDetector:
    @pytest.mark.parametrize("said", LIST_QUESTIONS)
    def test_list_questions_are_recognised(self, said):
        assert is_doctor_list_request(said)

    @pytest.mark.parametrize("said", NOT_LIST_QUESTIONS)
    def test_everything_else_is_left_alone(self, said):
        assert not is_doctor_list_request(said)

    def test_a_singular_nameless_question_is_not_forced_into_a_list(self):
        # Genuinely ambiguous ("is THE doctor in?"), so asking which doctor
        # once is fair; the loop-breaker in main.py handles a second time.
        assert not is_doctor_list_request("ডাক্তার অবৈলাবল আছে")


class TestListOverride:
    """apply_list_doctors_override() only corrects classifications that
    could not have been answered anyway."""

    @pytest.mark.parametrize("intent,slot", [
        ("doctor_availability", "doctor_name"), ("doctor_schedule", "doctor_name"),
        ("doctors_by_department", "department"),
    ])
    def test_a_doctor_intent_missing_its_entity_becomes_list_doctors(self, intent, slot):
        data = {"intent": intent, "slots": _slots(), "parts": [{"intent": intent, "slots": _slots()}]}
        out = apply_list_doctors_override("কে কে ডাক্তার আছেন", data)
        assert out["intent"] == LIST_DOCTORS
        assert out["parts"][0]["intent"] == LIST_DOCTORS
        assert out["slots"]["doctor_name"] is None

    @pytest.mark.parametrize("intent", ["unclear", "smalltalk", "out_of_scope"])
    def test_a_no_lookup_intent_becomes_list_doctors_and_loses_its_free_text(self, intent):
        data = {"intent": intent, "slots": _slots(), "direct_reply_bn": "মডেলের নিজের কথা"}
        out = apply_list_doctors_override("আপনাদের কি কি ডাক্তার এখানে বসেন", data)
        assert out["intent"] == LIST_DOCTORS
        assert out["direct_reply_bn"] is None, "a factual answer is never the model's own text"

    def test_a_named_department_is_never_overridden(self):
        data = {"intent": "doctors_by_department", "slots": _slots(department="Cardiology")}
        assert apply_list_doctors_override("কার্ডিওলজিতে কে কে ডাক্তার আছেন", data) is data

    def test_a_named_doctor_is_never_overridden(self):
        data = {"intent": "doctor_availability", "slots": _slots(doctor_name="Dr. A. Sen")}
        assert apply_list_doctors_override("কোন ডাক্তার সেন আছেন", data) is data

    def test_an_unrelated_intent_is_never_overridden(self):
        data = {"intent": "test_rate", "slots": _slots(test_name="CBC")}
        assert apply_list_doctors_override("কে কে ডাক্তার CBC করেন", data) is data

    def test_the_input_is_never_mutated(self):
        # The semantic cache may hold this dict.
        slots = _slots(doctor_name="stale")
        data = {"intent": "unclear", "slots": slots}
        apply_list_doctors_override("কে কে ডাক্তার আছেন", data)
        assert data["intent"] == "unclear" and slots["doctor_name"] == "stale"

    def test_only_the_correctable_part_of_a_multi_part_turn_changes(self):
        rate = {"intent": "test_rate", "slots": _slots(test_name="CBC")}
        avail = {"intent": "doctor_availability", "slots": _slots()}
        data = {"intent": "test_rate", "slots": rate["slots"], "parts": [rate, avail]}
        out = apply_list_doctors_override("CBC এর দাম কত আর কে কে ডাক্তার আছেন", data)
        assert out["intent"] == "test_rate"
        assert [p["intent"] for p in out["parts"]] == ["test_rate", LIST_DOCTORS]

    def test_the_llm_may_emit_list_doctors_and_it_validates(self):
        assert "list_doctors" in VALID_INTENTS
        ok, errors = _validate({"intent": "list_doctors", "slots": _slots()})
        assert ok and not errors


def _seeded_departments():
    sys.path.insert(0, CLINIC_API_DIR)
    try:
        import seed as clinic_seed
    finally:
        sys.path.remove(CLINIC_API_DIR)
    out = []
    for name, aliases in clinic_seed.DEPARTMENT_ALIASES.items():
        bn = next(a for a in aliases if any("ঀ" <= c <= "৿" for c in a))
        out.append({"name": name, "department_bn": bn, "aliases_bn": aliases, "doctor_count": 4})
    return out


@pytest.fixture(scope="module")
def departments():
    return _seeded_departments()


class TestDepartmentMatcher:
    """The answer to "কোন বিভাগের ডাক্তার খুঁজছেন?", against the REAL seeded
    aliases -- the matcher keeps no vocabulary of its own."""

    @pytest.mark.parametrize("said,expected", [
        ("কার্ডিওলজি", "Cardiology"),
        ("কার্ডিওলজিতে", "Cardiology"),               # case ending
        ("হার্টের ডাক্তার", "Cardiology"),
        ("মেডিসিন", "General Medicine"),
        ("জেনারেল মেডিসিন", "General Medicine"),
        ("অর্থোর ডাক্তার", "Orthopaedics"),
        ("ইএনটি", "ENT"),
        ("ENT", "ENT"),
        ("বাচ্চাদের ডাক্তার", "Paediatrics"),
        ("সুগারের ডাক্তার", "Diabetology & Endocrinology"),
        ("গাইনি", "Gynaecology & Obstetrics"),
        ("cardio", "Cardiology"),
    ])
    def test_a_department_in_any_of_its_forms(self, departments, said, expected):
        assert match_department(said, departments)["name"] == expected

    @pytest.mark.parametrize("said", ["appointment chai",     # "ent" inside a word
                                      "ডাক্তার সেন", "না থাক", "CBC এর দাম কত"])
    def test_anything_else_is_none(self, departments, said):
        assert match_department(said, departments) is None


class TestDoctorListReply:
    R = {"found": True, "date": None, "departments": [
        {"name": "Cardiology", "department_bn": "কার্ডিওলজি", "doctor_count": 4},
        {"name": "Empty", "department_bn": "খালি", "doctor_count": 0},
        {"name": "Orthopaedics", "department_bn": "অর্থোপেডিক্স", "doctor_count": 3},
        {"name": "Medicine", "department_bn": "মেডিসিন", "doctor_count": 5},
    ]}

    def test_lists_only_departments_with_a_doctor_and_asks_which(self):
        reply = doctor_list_reply(self.R)
        assert reply == ("আমাদের এখানে কার্ডিওলজি, অর্থোপেডিক্স আর মেডিসিন বিভাগে ডাক্তার বসেন। "
                         "কোন বিভাগের ডাক্তার খুঁজছেন?")
        assert "খালি" not in reply

    def test_a_dated_list_names_the_day_word(self):
        reply = doctor_list_reply(dict(self.R, date="2026-09-28"), day_key="today")
        assert reply.startswith("আমাদের এখানে আজ কার্ডিওলজি")

    def test_nobody_in_that_day_invites_another_day(self):
        reply = doctor_list_reply({"found": True, "date": "2026-09-27", "departments": []},
                                  day_key="today")
        assert reply == "দুঃখিত, আজ কোনো ডাক্তার বসবেন না। অন্য কোনো দিনের কথা জিজ্ঞেস করতে পারেন।"

    @pytest.mark.parametrize("reply", [
        doctor_list_reply(R), doctor_list_reply(dict(R, date="2026-11-03")),
        doctor_list_reply({"found": True, "date": None, "departments": []}),
        booking_which_date_prompt(), new_or_move_prompt(), anything_else_reply(),
    ])
    def test_every_bengali_reply_is_speakable(self, reply):
        assert speakability.check(reply).state == speakability.SPEAKABLE, reply

    def test_no_department_is_spoken_in_latin_script(self):
        # The Bengali tokenizer drops Latin script; a list of dropped names is
        # a sentence with holes in it.
        reply = doctor_list_reply(self.R)
        assert not any("a" <= c.lower() <= "z" for c in reply)


# ---------------------------------------------------------------------------
# The endpoint, against the REAL app and the REAL seed data.

@pytest.fixture(scope="module")
def real_clinic_api(tmp_path_factory):
    """Same self-contained fixture pattern as tests/test_doctor_schedule_
    dispatch.py: a throwaway SQLite file seeded by the real seed.py, the real
    FastAPI app in-process."""
    db_path = tmp_path_factory.mktemp("clinic_departments") / "clinic.db"
    old_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    saved = {m: sys.modules.pop(m, None) for m in ("db", "models", "seed", "main")}
    sys.path.insert(0, CLINIC_API_DIR)
    try:
        import db as clinic_db
        import models as clinic_models
        clinic_models.Base.metadata.create_all(clinic_db.engine)
        from seed import seed
        seed()
        import main as clinic_main
        from fastapi.testclient import TestClient
        yield types.SimpleNamespace(client=TestClient(clinic_main.app), db=clinic_db,
                                    models=clinic_models)
    finally:
        sys.path.remove(CLINIC_API_DIR)
        for m in ("db", "models", "seed", "main"):
            sys.modules.pop(m, None)
        for m, mod in saved.items():
            if mod is not None:
                sys.modules[m] = mod
        if old_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old_url


class TestDepartmentsEndpoint:
    def test_every_seeded_department_with_its_real_doctor_count(self, real_clinic_api):
        body = real_clinic_api.client.get("/api/v1/departments").json()
        assert body["found"] is True and body["date"] is None
        with real_clinic_api.db.SessionLocal() as db:
            m = real_clinic_api.models
            expected = {d.name: db.query(m.Doctor).filter_by(department_id=d.id).count()
                        for d in db.query(m.Department).all()}
        assert {d["name"]: d["doctor_count"] for d in body["departments"]} == expected
        assert sum(expected.values()) > 0

    def test_every_department_is_named_in_bengali_script(self, real_clinic_api):
        # seed.py lists the English aliases FIRST, so the first alias is Latin;
        # the endpoint must pick the first Bengali one or the list goes silent.
        body = real_clinic_api.client.get("/api/v1/departments").json()
        for d in body["departments"]:
            assert any("ঀ" <= c <= "৿" for c in d["department_bn"]), d
            assert d["aliases_bn"], d

    def test_a_date_counts_only_the_doctors_who_sit_that_weekday(self, real_clinic_api):
        date = "2026-09-28"
        body = real_clinic_api.client.get("/api/v1/departments", params={"date": date}).json()
        assert body["date"] == date
        with real_clinic_api.db.SessionLocal() as db:
            m = real_clinic_api.models
            weekday = datetime.date.fromisoformat(date).weekday()
            for d in body["departments"]:
                dept = db.query(m.Department).filter_by(name=d["name"]).one()
                sitting = sum(
                    1 for doc in db.query(m.Doctor).filter_by(department_id=dept.id)
                    if db.query(m.DoctorSchedule).filter_by(doctor_id=doc.id, weekday=weekday).first())
                assert d["doctor_count"] == sitting, d["name"]

    def test_the_department_listing_also_names_it_in_bengali(self, real_clinic_api):
        # The step after "which department?": /doctors/by-department had the
        # same Latin-first alias bug, so "cardiology বিভাগে ..." lost its subject.
        for name in ("Cardiology", "General Medicine", "ENT"):
            body = real_clinic_api.client.get("/api/v1/doctors/by-department",
                                              params={"department": name}).json()
            assert any("ঀ" <= c <= "৿" for c in body["department_bn"]), body

    @pytest.mark.parametrize("query", [
        "ডক্টর মণ্ডল",      # live transcript -- the honorific was the whole problem
        "ডঃ মণ্ডল", "ডাঃ মণ্ডল", "ডাক্তার মণ্ডল", "মণ্ডল",
        "Dr Mondal", "doctor mondal", "Dr. Mondal",
    ])
    def test_a_doctor_resolves_with_or_without_a_title(self, real_clinic_api, query):
        # "ডাক্তার মন্ডল" committed while "ডক্টর মণ্ডল" -- the same request --
        # landed in the offer band, so the caller was asked "did you mean
        # Mondal?" about the doctor they had just named.
        r = real_clinic_api.client.get("/api/v1/doctors/schedule",
                                       params={"name": query}).json()
        assert r.get("found") is True, r
        assert r["doctor_name"] == "Dr. A. Mondal", r

    def test_a_name_that_does_not_exist_is_still_offered_not_refused(self, real_clinic_api):
        # Stripping the title unconditionally would turn "Doctor Nobody" into
        # "Nobody", matching nothing, and lose the near-match band entirely.
        r = real_clinic_api.client.get("/api/v1/doctors/schedule",
                                       params={"name": "Doctor Nobody"}).json()
        assert r["found"] is False and r.get("ambiguous") is True, r

    def test_a_bare_title_is_not_treated_as_a_name(self, real_clinic_api):
        for query in ("ডাক্তার", "doctor", "ডাঃ"):
            r = real_clinic_api.client.get("/api/v1/doctors/schedule",
                                           params={"name": query}).json()
            assert r.get("found") is not True, (query, r)

    def test_a_malformed_date_falls_back_to_the_roster(self, real_clinic_api):
        body = real_clinic_api.client.get("/api/v1/departments", params={"date": "tomorrow"}).json()
        assert body["date"] is None


class TestToolsClient:
    """get_departments() through the real contract and cache code, with
    the transport faked at the httpx layer."""

    def _client(self, handler):
        from agent.tools_client import ClinicToolsClient
        tc = ClinicToolsClient("http://clinic.test")
        tc._client = httpx.AsyncClient(base_url="http://clinic.test",
                                       transport=httpx.MockTransport(handler))
        return tc

    def test_returns_the_payload_and_caches_only_the_undated_roster(self):
        calls = []

        def handler(request):
            calls.append(dict(request.url.params))
            return httpx.Response(200, json={"found": True, "date": request.url.params.get("date"),
                                             "departments": []})

        tc = self._client(handler)
        asyncio.run(tc.get_departments())
        asyncio.run(tc.get_departments())
        asyncio.run(tc.get_departments("2026-09-28"))
        asyncio.run(tc.get_departments("2026-09-28"))
        assert calls == [{}, {"date": "2026-09-28"}, {"date": "2026-09-28"}], \
            "the roster is reference data; who sits on a date is live"

    def test_a_contract_violation_is_a_tool_call_error(self):
        from agent.tools_client import ToolCallError
        tc = self._client(lambda r: httpx.Response(200, json={"found": True}))
        with pytest.raises(ToolCallError):
            asyncio.run(tc.get_departments())


# =========================================================================
# 2. The last doctor is not backfilled into a question about other doctors
# =========================================================================

class TestStaleSlotCarryover:
    @pytest.fixture
    def state(self):
        s = DialogueState()
        s.mark("doctor", "Dr. A. Sen")
        s.mark("test", "CBC")
        return s

    @pytest.mark.parametrize("intent,said", [
        ("doctor_availability", "আপনাদের কি কি ডাক্তার এখানে বসেন"),   # live transcript
        ("doctor_availability", "কে কে ডাক্তার অবৈলাবল আছে"),
        ("doctor_schedule", "অন্য ডাক্তার কবে বসেন"),
        ("doctor_schedule", "আর কোন ডাক্তার আছেন"),
        ("doctor_availability", "another doctor tomorrow"),
    ])
    def test_a_caller_pointing_away_from_the_doctor_gets_no_backfill(self, state, intent, said):
        slots, ambiguous = resolve_follow_up(state, intent, _slots(), text=said)
        assert slots["doctor_name"] is None and ambiguous is None

    @pytest.mark.parametrize("intent,said", [
        ("doctor_schedule", "কবে বসেন?"),                        # pronoun dropped
        ("doctor_availability", "ওনার কাল সময় আছে?"),
        # "অন্য" modifies the DAY here; "উনি" is Dr Sen (live transcript).
        ("doctor_availability", "না অন্য তিন নভেম্বর কি উনি বসবেন"),
    ])
    def test_a_real_follow_up_still_backfills(self, state, intent, said):
        slots, _ = resolve_follow_up(state, intent, _slots(), text=said)
        assert slots["doctor_name"] == "Dr. A. Sen"

    def test_a_follow_up_may_still_change_intent(self, state):
        # Deliberately NOT "intent changed -> stale": that is the feature.
        slots, _ = resolve_follow_up(state, "test_duration", _slots(), text="ওটার রিপোর্ট কবে পাব")
        assert slots["test_name"] == "CBC"

    @pytest.mark.parametrize("said", ["অন্য টেস্টের দাম কত", "কি কি টেস্ট আছে", "other test"])
    def test_the_same_rule_covers_tests(self, state, said):
        slots, _ = resolve_follow_up(state, "test_rate", _slots(), text=said)
        assert slots["test_name"] is None

    def test_callers_that_pass_no_text_behave_exactly_as_before(self, state):
        slots, _ = resolve_follow_up(state, "doctor_availability", _slots())
        assert slots["doctor_name"] == "Dr. A. Sen"


class TestFastPathVerbsAreNotSurnames:
    """What actually fired on the live call: the fast path (tier 1, before the
    model) read "বসেন" (sits) as Dr Sen at 0.86. These are the thirteen
    collisions measured against the seeded catalogue; the doctors' alias
    lists here are the seeded ones."""

    CATALOGUE = {"tests": [], "doctors": [
        {"name": "Dr. A. Sen", "surname": "Sen", "aliases_bn": ["সেন", "sen"]},
        {"name": "Dr. N. Roy", "surname": "Roy", "aliases_bn": ["রায়", "রয়", "roy"]},
        {"name": "Dr. A. Basu", "surname": "Basu", "aliases_bn": ["বসু", "basu"]},
        {"name": "Dr. A. Kar", "surname": "Kar", "aliases_bn": ["কর", "kar"]},
        {"name": "Dr. A. Dey", "surname": "Dey", "aliases_bn": ["দে", "dey"]},
        {"name": "Dr. K. Bhattacharya", "surname": "Bhattacharya",
         "aliases_bn": ["ভট্টাচার্য", "bhattacharya"]},
    ]}

    @pytest.fixture(scope="class")
    def catalogue(self):
        from agent.fast_path import Catalogue
        return Catalogue(self.CATALOGUE)

    @pytest.mark.parametrize("said", [
        "আপনাদের কি কি ডাক্তার এখানে বসেন",      # live transcript
        "কে কে ডাক্তার বসেন", "কোন ডাক্তার বসবেন", "উনি কবে বসছেন", "আগে কখন বসতেন",
        "ডাক্তার কখন আসেন", "কাল আসবেন", "একটু বসুন", "বুক করে দিন", "আমি কী করব",
        "এটা করা যাবে", "আমি করি", "রিপোর্টটা দেন", "report send karo",
    ])
    def test_a_verb_is_never_a_doctor(self, catalogue, said):
        from agent.fast_path import COMMIT_FLOOR
        name, form, score, _ = catalogue.match(said, "doctor")
        assert score < COMMIT_FLOOR, f"{said!r} committed to {name} via {form!r} ({score:.2f})"

    @pytest.mark.parametrize("said,expected", [
        ("ডাক্তার সেন কি আজ আছেন", "Dr. A. Sen"),
        ("সেনের চেম্বার কবে", "Dr. A. Sen"),
        ("সেনবাবু কবে বসেন", "Dr. A. Sen"),
        ("Dr Sen available today", "Dr. A. Sen"),
        ("ডাক্তার বসুর সময়", "Dr. A. Basu"),
        ("ডাক্তার করকে দেখাব", "Dr. A. Kar"),
        ("ডাক্তার দে কবে বসেন", "Dr. A. Dey"),
        ("ডাক্তার রায় আজ আছেন", "Dr. N. Roy"),
        ("ভট্টাচার্য্য কবে বসেন", "Dr. K. Bhattacharya"),   # long forms stay fuzzy
    ])
    def test_a_named_doctor_is_still_found(self, catalogue, said, expected):
        from agent.fast_path import COMMIT_FLOOR
        name, _, score, _ = catalogue.match(said, "doctor")
        assert name == expected and score >= COMMIT_FLOOR, (name, score)


# =========================================================================
# 3. A day said next to a month name
# =========================================================================

class TestMonthNamedDates:
    @pytest.mark.parametrize("said", [
        # every form in the request, plus the two live utterances
        "তিন নভেম্বর", "৩ নভেম্বর", "3 নভেম্বর", "থার্ড নভেম্বর", "November 3",
        "নভেম্বরের তিন", "নভেম্বর মাসের তিন তারিখ", "নভেম্বর এর তিন তারিখ",
        "তেসরা নভেম্বর", "৩রা নভেম্বর", "3rd November", "3rd of November", "third november",
        "মঙ্গলবার তিন নভেম্বর",          # 3 Nov 2026 IS a Tuesday: they agree
        "না অন্য তিন নভেম্বর কি উনি বসবেন",
        "থার্ড নভেম্বর এর জন্য অযাপ্ফোন্টমেন্ট পাওয়া যাবে",
    ])
    def test_the_third_of_november(self, said):
        assert parse_date(said, today=TODAY) == "2026-11-03"

    @pytest.mark.parametrize("said,expected", [
        ("২১শে নভেম্বর", "2026-11-21"), ("একুশে নভেম্বর", "2026-11-21"),
        ("টুয়েন্টি থার্ড নভেম্বর", "2026-11-23"), ("টুয়েন্টি টু নভেম্বর", "2026-11-22"),
        ("পয়লা ডিসেম্বর", "2026-12-01"), ("দোসরা অক্টোবর", "2026-10-02"),
        ("চৌঠা অক্টোবর", "2026-10-04"), ("পাঁচই অক্টোবর", "2026-10-05"),
        ("৫ই অক্টোবর", "2026-10-05"), ("১লা নভেম্বর", "2026-11-01"),
        ("December 25", "2026-12-25"), ("25th December", "2026-12-25"),
        ("অক্টোবর মাসের আটাশ তারিখ", "2026-10-28"),
        ("নবেম্বরের দশ তারিখ", "2026-11-10"),                     # ASR spelling
        ("তিন জানুয়ারি", "2027-01-03"),                          # the coming January
        ("৩ নভেম্বর ২০২৬", "2026-11-03"),
        ("২৮ সেপ্টেম্বর", "2026-09-28"),                          # today
    ])
    def test_other_days_and_forms(self, said, expected):
        assert parse_date(said, today=TODAY) == expected

    @pytest.mark.parametrize("said", [
        "সোমবার তিন নভেম্বর",          # contradicts itself -- asked, not guessed
        "কাল তিন নভেম্বর",              # "tomorrow" is 29 September
        "৩১ নভেম্বর",                   # no such date
        "৩ নভেম্বর ২০২৫",               # explicitly in the past
        "২০ সেপ্টেম্বর",                # last week; next year's is 357 days off
        "তিন নভেম্বর আর পাঁচ নভেম্বর",  # two different dates
        "শিফট টু নভেম্বর",              # "টু" is also English "to"
        "নভেম্বরের তিনটায়",             # a time, not a day
    ])
    def test_anything_unclear_is_asked_again(self, said):
        assert parse_date(said, today=TODAY) is None

    @pytest.mark.parametrize("said,expected", [
        ("সকাল দশটায়", None), ("১৫ তারিখ", "2026-10-15"), ("চব্বিশ তারিখে", "2026-10-24"),
        ("কাল", "2026-09-29"), ("সোমবার", "2026-10-05"), ("তিন তারিখ", "2026-10-03"),
        ("2026-11-03", "2026-11-03"), ("মেয়ের তিন তারিখ", "2026-10-03"),
        ("পাঁচ মিনিট", None), ("বারো টাকা", None),
    ])
    def test_nothing_that_parsed_before_changes(self, said, expected):
        assert parse_date(said, today=TODAY) == expected

    def test_date_calc_takes_the_parsed_date_over_the_model(self):
        # parse_date is the highest authority in date_calc.resolve, so every
        # date path (availability, booking, department) gets this for free.
        from agent import date_calc
        span = date_calc.resolve("থার্ড নভেম্বর এর জন্য", date_expr="unmapped", today=TODAY)
        assert span.start == "2026-11-03"


def _load_match_band():
    spec = importlib.util.spec_from_file_location("match_band_under_test",
                                                  pathlib.Path(CLINIC_API_DIR) / "match_band.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_literal(name):
    src = (pathlib.Path(CLINIC_API_DIR) / "seed.py").read_text(encoding="utf-8")
    m = re.search(name + r"\s*=\s*[\[{].*?\n[\]}]", src, re.S)
    ns: dict = {}
    exec(m.group(0), ns)  # noqa: S102 - our own source
    return ns[name]


class TestMatchBandReadsBengaliWords:
    """clinic-api's matcher (match_band.score), against the REAL seed data.

    Containment used to be `q in f or f in q`, guarded by the QUERY's length,
    so a two-letter alias was "inside" any sentence. Bengali inflects with a
    suffix, never a prefix: a form counts only where it starts a word and is
    followed by a case ending at most."""

    @pytest.fixture(scope="class")
    def mb(self):
        return _load_match_band()

    @pytest.fixture(scope="class")
    def rows(self):
        sb, d = _seed_literal("SURNAME_BN"), _seed_literal("DEPARTMENTS")
        tests = [(t[0], [t[0], *t[1]]) for t in _seed_literal("LAB_TESTS")]
        docs = [(n, [n, n.split()[-1], *sb.get(n.split()[-1], [])])
                for docs_ in d.values() for n, _ in docs_]
        return types.SimpleNamespace(tests=tests, docs=docs)

    @staticmethod
    def _v(mb, q, rows):
        verdict, cands = mb.decide(mb.rank(q, rows))
        return verdict, [c.key for c in cands]

    @pytest.mark.parametrize("query", [
        "বাচ্চাদের ডাক্তার",   # দে inside বাচ্চাদের -> was Dr Dey, a dermatologist
        "চোখের ডাক্তার", "ডাক্তার দেখাতে", "দেখাতে",   # দে starts দেখাতে
        "করতে", "বুক করে দিন", "করা",                   # কর (Dr Kar) in "to do"
        "রিপোর্টটা দেন",                                 # দে + ন
        "কোন ডাক্তার বসেন",                              # সেন inside বসেন
        "দের", "বাচ্চা দের ডাক্তার",                    # the plural suffix, ASR-split
    ])
    def test_a_verb_or_suffix_never_commits_a_doctor(self, mb, rows, query):
        verdict, keys = self._v(mb, query, rows.docs)
        assert verdict != mb.COMMIT, f"{query!r} committed {keys}"

    @pytest.mark.parametrize("query,doctor", [
        ("ডাক্তার দে", "Dr. A. Dey"), ("দে", "Dr. A. Dey"), ("ডাক্তার কর", "Dr. A. Kar"),
        ("করকে", "Dr. A. Kar"), ("সেন", "Dr. A. Sen"), ("সেনের", "Dr. A. Sen"),
        ("সেনবাবু", "Dr. A. Sen"), ("ডাক্তার সেন", "Dr. A. Sen"), ("Dr Sen", "Dr. A. Sen"),
        ("সেনগুপ্ত", "Dr. P. Sengupta"), ("মণ্ডল", "Dr. A. Mondal"),
        ("বোস", "Dr. T. Bose"), ("বসু", "Dr. A. Basu"),
    ])
    def test_a_real_name_still_commits(self, mb, rows, query, doctor):
        assert self._v(mb, query, rows.docs) == (mb.COMMIT, [doctor])

    def test_a_one_letter_asr_slip_is_still_offered(self, mb, rows):
        assert self._v(mb, "সেণ", rows.docs) == (mb.AMBIGUOUS, ["Dr. A. Sen"])
        assert self._v(mb, "ডাক্তার সেণ", rows.docs) == (mb.AMBIGUOUS, ["Dr. A. Sen"])

    @pytest.mark.parametrize("query,expected", [
        ("হার্ট টেস্ট", "ECG"),          # was the LIVER Function Test, on "টেস্ট" alone
        ("হার্টের টেস্ট", "ECG"), ("heart test", "ECG"), ("ইসিজি", "ECG"),
        ("লিভার টেস্ট", "Liver Function Test (LFT)"),
        ("pap smear", "Pap Smear"),
    ])
    def test_a_test_resolves_to_itself(self, mb, rows, query, expected):
        assert self._v(mb, query, rows.tests) == (mb.COMMIT, [expected])

    @pytest.mark.parametrize("query", ["সুগার", "ভিটামিন", "Blood Sugar"])
    def test_a_shared_word_is_still_offered_not_guessed(self, mb, rows, query):
        assert self._v(mb, query, rows.tests)[0] == mb.AMBIGUOUS

    @pytest.mark.parametrize("short,long,expected", [
        ("ent", "appointment", False), ("ear", "smear", False), ("ear", "heart", False),
        ("সেন", "বসেন", False), ("দে", "বাচ্চাদের", False), ("কর", "করতে", False),
        ("সেন", "সেনের কাছে", True), ("কার্ডিওলজি", "কার্ডিওলজিতে একজন", True),
        ("সুগার", "সুগার ফাস্টিং", True), ("দে", "ডাক্তার দে", True),
    ])
    def test_containment_is_a_word_not_a_substring(self, mb, short, long, expected):
        assert mb._contained(short, long) is expected


class TestSpecialtyRequest:
    @pytest.mark.parametrize("said", ["চোখের ডাক্তার", "দাঁতের ডাক্তার দেখাতে চাই",
                                      "বাচ্চাদের ডাক্তার", "skin er doctor", "eye specialist"])
    def test_a_kind_of_doctor(self, said):
        from agent.doctor_list import looks_like_specialty_request
        assert looks_like_specialty_request(said)

    @pytest.mark.parametrize("said", ["ডাক্তার নোবডি", "ডাক্তার সেন", "ডক্টর মণ্ডল", "Dr Sen"])
    def test_a_person(self, said):
        from agent.doctor_list import looks_like_specialty_request
        assert not looks_like_specialty_request(said)


class TestBanglishIsNotEnglish:
    @pytest.mark.parametrize("said,expected", [
        ("skin er doctor dekhate chai", "banglish"), ("Dr Sen er appointment chai", "banglish"),
        ("heart test koto taka", "banglish"), ("pap smear koto", "banglish"),
        # unchanged
        ("I want to see a doctor", "english"), ("is Dr Sen available today", "english"),
        ("CBC ka rate kitna hai", "hinglish"), ("kaun kaun se doctor hai", "hinglish"),
        ("mujhe appointment chahiye", "hinglish"), ("CBC", "english"),
    ])
    def test_detect_language(self, said, expected):
        from agent.bn_normalize import detect_language
        assert detect_language(said) == expected


class TestYesNoQuestionsSaySo:
    """Every prompt whose answer is read by is_affirmative()/is_negative()
    alone tells the caller to answer yes or no -- in the FIRST ask, not only
    after they got it wrong once (the reschedule/cancel retries already did)."""

    S = {"doctor_name": "Dr. A. Sen", "doctor_name_bn": "সেন", "date": "2026-10-05",
         "time_slot": "10:00", "patient_name": "রিনা দাস", "phone": "9831234567",
         "date_said": "date", "date_weekday": 0}
    A = {"reference": "KCD-1", "doctor_name": "Dr. A. Sen", "doctor_name_bn": "সেন",
         "date": "2026-10-05", "time_slot": "10:00"}

    def _prompts(self):
        import agent.reply_templates as rt
        return {
            "confirm_booking (full readback)": rt.booking_confirmation_prompt(self.S),
            "confirm_booking (short readback)": rt.booking_confirm_prompt(self.S),
            "confirm_booking_date": rt.booking_date_check_prompt(self.S),
            "confirm_date (range)": rt.date_range_confirm_prompt("2026-09-14", "2026-09-20"),
            "confirm_transcript": rt.heard_confirm_prompt("ডাক্তার সেন"),
            "entity_choice (one name)": rt.near_match_prompt([{"name": "Dr. A. Mondal",
                                                               "name_bn": "মন্ডল"}]),
            "confirm_callback": rt.callback_confirmation_prompt(
                {"callback_time_window": "বিকেল", "phone": "9831234567"}),
            "clinical_interpretation_choice": rt.clinical_interpretation_reply(),
            "resched_confirm": rt.reschedule_confirm_prompt(self.A, "2026-10-06", "11:00"),
            "cancel_confirm": rt.cancel_confirm_prompt(self.A, {"charge_inr": 0, "refund_inr": 0}),
        }

    def test_every_plain_yes_no_question_ends_with_the_hint(self):
        missing = {k: v for k, v in self._prompts().items() if not v.endswith("হ্যাঁ অথবা না বলুন।")}
        assert not missing, missing

    @pytest.mark.parametrize("name", ["out_of_scope_reply", "unverifiable_claim_reply"])
    def test_an_either_or_says_which_answer_picks_which(self, name):
        import agent.reply_templates as rt
        reply = getattr(rt, name)()
        assert "চাইলে হ্যাঁ বলুন" in reply and "চাইলে না বলুন" in reply, reply

    def test_the_angry_offer_says_it_too_both_times(self):
        import agent.reply_templates as rt
        for apologised in (False, True):
            assert "চাইলে হ্যাঁ বলুন" in rt.anger_reply(apologised)

    def test_the_report_offer_maps_yes_and_callback(self):
        import agent.reply_templates as rt
        reply = rt.report_status_reply({"patient_found": True, "found": True, "status": "READY",
                                        "delivery_enabled": True, "test_name": "CBC"})
        assert "হ্যাঁ বলুন" in reply and "কল ব্যাক বলুন" in reply, reply

    @pytest.mark.parametrize("language,hint", [("english", "Please say yes or no."),
                                               ("hinglish", "Haan ya na boliye."),
                                               ("banglish", "Hyan ba na bolun.")])
    def test_in_the_callers_language(self, language, hint):
        import agent.reply_templates as rt
        assert rt.booking_confirmation_prompt(self.S, language=language).endswith(hint)

    def test_a_choice_between_names_gets_no_yes_no_hint(self):
        import agent.reply_templates as rt
        reply = rt.near_match_prompt([{"name": "a", "name_bn": "রায়"}, {"name": "b", "name_bn": "বোস"}])
        assert "হ্যাঁ" not in reply, reply

    def test_never_said_twice(self):
        import agent.reply_templates as rt
        assert rt.cancel_prompt("confirm").count("হ্যাঁ") == 1
        once = rt.with_yes_no_hint("ঠিক আছে?")
        assert rt.with_yes_no_hint(once) == once

    def test_every_hint_is_speakable(self):
        for name, reply in self._prompts().items():
            assert speakability.check(reply).state == speakability.SPEAKABLE, name


class TestLanguageOfAShortAnswer:
    """A word or two of Latin script says nothing about the caller's language;
    it keeps the call's language instead of flipping to English."""

    def _session(self, last):
        return types.SimpleNamespace(last_language=last)

    def _lang(self, session, text):
        return agent_main._turn_language(session, text)

    @pytest.mark.parametrize("said", ["ortho", "CBC", "Dr Sen", "Dr. Gupta", "ok"])
    def test_a_loanword_name_or_interjection_keeps_bengali(self, said):
        assert self._lang(self._session("bengali"), said) == "bengali"

    @pytest.mark.parametrize("said", ["tomorrow morning", "I want to see a doctor tomorrow",
                                      "next week"])
    def test_a_real_switch_to_english_is_still_followed(self, said):
        s = self._session("bengali")
        assert self._lang(s, said) == "english"
        assert s.last_language == "english"

    def test_a_marked_banglish_answer_is_banglish(self):
        assert self._lang(self._session("bengali"), "ortho er doctor chai") == "banglish"

    def test_the_first_turn_is_detected_as_before(self):
        assert self._lang(self._session(None), "CBC") == "english"


class TestDayOfMonthOutranksAWeekday:
    """"সোমবার বারো তারিখ" -- Monday the 12th, said as one phrase.

    The weekday block returned first and the "তারিখ" half was never read, so
    the agent answered with 5 October for a caller who had pinned the 12th.
    12 Oct 2026 IS a Monday, so the two halves agreed all along."""

    @pytest.mark.parametrize("said,expected", [
        ("সোমবার বারো তারিখ", "2026-10-12"),    # live transcript
        ("সোমবার ১২ তারিখ", "2026-10-12"),
        ("সোমবার বারো তারিখে বসবেন", "2026-10-12"),
        ("বারো তারিখ", "2026-10-12"),
        ("পাঁচ তারিখ", "2026-10-05"),
        ("সোমবার পাঁচ তারিখ", "2026-10-05"),
    ])
    def test_the_day_of_the_month_wins(self, said, expected):
        assert parse_date(said, today=TODAY) == expected

    @pytest.mark.parametrize("said", [
        "মঙ্গলবার বারো তারিখ",   # 12 Oct is a Monday -- the two contradict
        "বুধবার পাঁচ তারিখ",
        "কাল বারো তারিখ",         # tomorrow is 29 September
        "আজ বারো তারিখ",
    ])
    def test_a_contradiction_is_asked_again_never_guessed(self, said):
        assert parse_date(said, today=TODAY) is None


class TestInflectedDayWords:
    """A day word carrying a Bengali case ending is still that day word.

    Second live loop, and the agent walked into this one itself: it asked
    "আজকের জন্যই অ্যাপয়েন্টমেন্ট করবেন, নাকি অন্য কোনো দিনের জন্য?", the
    caller answered "আজকের জন্য" in the agent's own words, and the date came
    back empty -- so the identical question was asked again."""

    @pytest.mark.parametrize("said,expected,day_word", [
        ("আজকের জন্য", "2026-09-28", "today"),            # live transcript
        ("আজকের জন্যই", "2026-09-28", "today"),
        ("আজকেই", "2026-09-28", "today"),
        ("কালকের জন্য", "2026-09-29", "tomorrow"),
        ("আগামীকালের জন্য", "2026-09-29", "tomorrow"),
        ("কালই", "2026-09-29", "tomorrow"),
        ("পরশুর জন্য", "2026-09-30", "day_after_tomorrow"),
        ("মঙ্গলবারে", "2026-09-29", "tuesday"),
        ("বৃহস্পতিবারের জন্য", "2026-10-01", "thursday"),
        ("শনিবারও", "2026-10-03", "saturday"),
        ("রবিবারের জন্য", "2026-10-04", "sunday"),
        ("সোমবারের জন্য", "2026-10-05", "monday"),
    ])
    def test_an_inflected_day_word_is_that_day(self, said, expected, day_word):
        from agent.slot_parse import spoken_day_word
        assert parse_date(said, today=TODAY) == expected
        # The readback must name the same word the date was computed from.
        assert spoken_day_word(said) == day_word

    @pytest.mark.parametrize("said", [
        # Why the LEFT boundary stays strict: these contain "কাল" and used to
        # parse as tomorrow -- the bug _bn_bounded() was written for.
        "সকালে", "বিকালে", "সকাল দশটায়", "বিকাল পাঁচটায়",
        "আজকাল",        # "nowadays" -- "কাল" is not an allowed ending
        "কালো", "কালীঘাট", "পাঁচ মিনিট",
    ])
    def test_a_word_that_merely_contains_a_day_word_is_not_a_date(self, said):
        assert parse_date(said, today=TODAY) is None


# =========================================================================
# 4. "Another day" is not a reschedule
# =========================================================================

class TestAnotherDayIsNotAReschedule:
    @pytest.mark.parametrize("said", ["অন্য দিনের জন্যে", "অন্য দিন", "অন্যদিনে চাই",
                                      "another day", "onno din"])
    def test_another_day_without_a_day(self, said):
        assert wants_unspecified_other_day(said)

    @pytest.mark.parametrize("said", ["অন্য দিন, তিন নভেম্বর", "কাল", "হ্যাঁ"])
    def test_a_day_that_is_named_is_not_unspecified(self, said):
        assert not wants_unspecified_other_day(said)

    @pytest.mark.parametrize("said", [
        # the LLM prompt's own reschedule examples must all still qualify
        "অ্যাপয়েন্টমেন্টটা অন্য দিনে সরাতে চাই", "বুকিংটা পিছিয়ে দিন",
        "appointment ta shift korte chai", "can I change my appointment date",
        "আমি আগে বুক করেছিলাম, অন্য দিনে চাই",
    ])
    def test_a_real_reschedule_has_a_cue(self, said):
        assert has_reschedule_cue(said)

    @pytest.mark.parametrize("said", ["অন্য দিনের জন্যে", "অন্য দিন চাই",
                                      "বুক করা যাবে?"])   # a NEW booking question
    def test_another_day_alone_is_not_a_reschedule(self, said):
        assert not has_reschedule_cue(said)

    @pytest.mark.parametrize("said,expected", [
        ("নতুন", "new"), ("নতুন অ্যাপয়েন্টমেন্ট", "new"), ("new one", "new"), ("প্রথমটা", "new"),
        ("আগেরটা বদলাতে চাই", "move"), ("দ্বিতীয়টা", "move"), ("the old one", "move"),
        ("হ্যাঁ", None), ("না", None),
    ])
    def test_the_answer_to_new_or_move(self, said, expected):
        assert parse_new_or_move(said) == expected
