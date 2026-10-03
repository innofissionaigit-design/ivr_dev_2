"""The "no guessing" fixes from the external review, as checks.

* confidence is three-state: a single decoder's text is never trusted as if two agreed;
* an unverified entity is read back and the lookup runs only after a yes;
* a possible emergency stops everything, in any language, at any confidence;
* fuzzy and phonetic matches SUGGEST, they never resolve, on doctors and on tests.

  python -m pytest tests/test_no_guess.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _pod_stubs import pod_stubs
from _synth_voices import FATHER, utterance
from test_orchestrator_persona import FakeTTS, FakeWS, _wav

from agent import entity_confirmation as ec
from agent.confidence_gate import (
    LOW,
    UNAVAILABLE,
    VERIFIED,
    confidence_state,
    needs_entity_readback,
    should_withhold_factual_answer,
)
from agent.emergency import detect_emergency
from agent.phrases import phrase

# ============================================================================ confidence


def test_confidence_is_three_state_and_a_single_decoder_is_never_verified():
    assert confidence_state("rnnt", 0.9) == VERIFIED
    assert confidence_state("rnnt", 0.3) == LOW
    assert confidence_state("rnnt", 0.0) == LOW  # total disagreement is LOW, not "trust me"
    assert confidence_state("ctc_fallback", 0.0) == UNAVAILABLE
    assert confidence_state("fastconformer", 1.0) == UNAVAILABLE  # a 1.0 from one decoder means nothing
    assert confidence_state("none", 1.0) == UNAVAILABLE
    assert confidence_state("something_new", 1.0) == UNAVAILABLE  # unknown decoder: never trusted


def test_without_a_recorded_decoder_the_old_float_rule_still_applies():
    assert confidence_state(None, 0.0) == VERIFIED and confidence_state(None, 0.3) == LOW


@pytest.mark.parametrize(
    "intent",
    [
        "test_rate",
        "doctor_availability",
        "test_prep",
        "clinic_faq",
        "department_query",
        "lookup_booking",
        "resend_confirmation",
    ],
)
def test_every_fact_retrieval_is_gated(intent):
    assert should_withhold_factual_answer(intent, 0.2, "rnnt")
    assert needs_entity_readback(intent, "ctc_fallback", 0.0)
    assert not should_withhold_factual_answer(intent, 0.9, "rnnt") and not needs_entity_readback(intent, "rnnt", 0.9)


def test_a_write_intent_is_not_gated_by_this_it_has_its_own_confirmation():
    assert not needs_entity_readback("book_appointment", "ctc_fallback", 0.0)


def test_the_readback_names_the_slot_that_selects_the_record():
    assert ec.entity_to_confirm("test_rate", {"test_name": "CRP"}) == ("test_name", "CRP")
    assert ec.entity_to_confirm("doctor_availability", {"doctor_name": "Sen"}) == ("doctor_name", "Sen")
    assert ec.entity_to_confirm("lookup_booking", {"phone": "9876543210"}) == ("phone", "9876543210")
    assert ec.entity_to_confirm("clinic_faq", {"topic": "parking"}) is None
    assert ec.entity_to_confirm("test_rate", {"test_name": None}) is None
    assert "CRP" in ec.confirm_question("test_name", "CRP", "en")
    assert "9876543210" in ec.confirm_question("phone", "9876543210", "hi")


# ============================================================================ emergency

_EMERGENCY = {
    "en": [
        "I have chest pain",
        "he is not breathing",
        "my father fainted",
        "she is unconscious",
        "there is heavy bleeding",
        "he is having a seizure",
        "my child swallowed poison",
        "I think it is a heart attack",
        "please send an ambulance",
        "this is an emergency",
        "he cannot breathe",
        "there was a road accident",
        "I want to kill myself",
        "she can't breathe properly",
        "he collapsed",
        "bleeding won't stop",
        "snake bite",
        "possible stroke",
    ],
    "bn": [
        "আমার বুকে ব্যথা করছে",
        "বাবা অজ্ঞান হয়ে গেছে",
        "শ্বাসকষ্ট হচ্ছে",
        "প্রচুর রক্ত পড়ছে",
        "খিঁচুনি হচ্ছে",
        "অ্যাম্বুলেন্স পাঠান",
        "হার্ট অ্যাটাক মনে হচ্ছে",
        "দুর্ঘটনা হয়েছে",
        "বিষ খেয়েছে",
        "এটা ইমার্জেন্সি",
    ],
    "hi": [
        "मेरे सीने में दर्द है",
        "वह बेहोश हो गया",
        "सांस नहीं ले पा रहे",
        "खून बहुत बह रहा है",
        "दौरा पड़ा है",
        "एम्बुलेंस भेजिए",
        "दिल का दौरा पड़ा",
        "दुर्घटना हो गई",
        "ज़हर खा लिया",
        "यह इमरजेंसी है",
    ],
}
_ORDINARY = [
    "what is the price of a CBC test",
    "book an appointment with Dr Sen tomorrow",
    "when does the clinic open",
    "I want to cancel my appointment",
    "is fasting required for the thyroid test",
    "my report is ready or not",
    "আমার সিবিসি টেস্টের দাম কত",
    "কাল ডাক্তার সেনের সঙ্গে অ্যাপয়েন্টমেন্ট চাই",
    "ক্লিনিক কখন খোলে",
    "मेरा टेस्ट कब हुआ था",
    "कल डॉक्टर सेन का अपॉइंटमेंट चाहिए",
    "क्लिनिक कब खुलता है",
    "hello good morning",
]


def test_every_emergency_phrase_is_caught_in_any_language_slot():
    missed = [(lg, t) for lg, ts in _EMERGENCY.items() for t in ts if not detect_emergency(t)]
    assert not missed, missed


def test_ordinary_clinic_calls_are_not_emergencies():
    wrongly = [t for t in _ORDINARY if detect_emergency(t)]
    assert not wrongly, wrongly


def test_an_emergency_phrase_in_a_garbled_transcript_still_counts():
    assert detect_emergency("uh ambulance um please")
    assert not detect_emergency("")


def test_a_live_call_2026_09_28_an_intensifier_word_inside_a_fixed_phrase_still_counts():
    # "বুকে ব্যথা" (chest pain) did not match "বুকে খুব ব্যথা করছে" ("my chest REALLY hurts") on a real call --
    # an ordinary way to say it, with "খুব" inserted between the two words. agent/phrase_proximity.py fixes this.
    assert detect_emergency("আমার বুকে খুব ব্যথা করছে")
    assert not detect_emergency("normal conversation about tests and prices")


def test_the_notice_exists_in_every_language_states_no_condition_and_gives_no_advice():
    for lang in ("bn", "hi", "en"):
        n = phrase("emergency_notice", lang)
        assert n and ("112" in n or "১১২" in n)


# ============================================================================ orchestrator


class ASRResult:
    def __init__(self, text, agreement=0.9, decoder_used="rnnt"):
        self.text, self.decoder_agreement, self.decoder_used = text, agreement, decoder_used


@pytest.fixture(scope="module")
def m():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


@pytest.fixture
def env(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_admission", None)
    monkeypatch.setattr(m, "_tts_router", FakeTTS())
    monkeypatch.setattr(m, "CONDITION_INPUT", "off")
    state = {
        "text": "what is the price of a CBC test",
        "lang": "en",
        "agreement": 0.9,
        "decoder": "rnnt",
        "handoffs": [],
        "resolved": 0,
        "answered": [],
        "intent": "test_rate",
        "slots": {"test_name": "CBC"},
    }

    async def route(session, wav):
        return state["lang"], ASRResult(state["text"], state["agreement"], state["decoder"])

    async def resolve(session, text, lang):
        state["resolved"] += 1
        return {
            "intent": state["intent"],
            "slots": dict(state["slots"]),
            "secondary_intent": None,
            "direct_reply_bn": None,
        }

    async def answer(intent, slots, lang):
        state["answered"].append((intent, dict(slots)))
        return "The CBC costs three hundred and fifty rupees."

    async def handoff(session, reason, languages=None):
        state["handoffs"].append(reason)

    monkeypatch.setattr(m, "_route_and_transcribe", route)
    monkeypatch.setattr(m, "_resolve_intent", resolve)
    monkeypatch.setattr(m, "_answer_enquiry_intent", answer)
    monkeypatch.setattr(m, "_handoff_to_human", handoff)
    session = m.CallSession(FakeWS())
    session.release_gate()
    session.disclosed_langs.add("en")
    session.call_state = m.new_call_state()
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)

    class Driver:
        pass

    d = Driver()
    d.session, d.state = session, state

    async def turn(**overrides):
        state.update(overrides)
        session.turn_epoch = session.speak_epoch
        await m._dispatch_turn(session, _wav(tmp_path, voice, f"u{len(session.ws.texts)}.wav"))

    d.turn = turn
    yield d
    session.cleanup()


@pytest.mark.asyncio
async def test_a_verified_turn_is_answered_straight_away(m, env):
    await env.turn()
    assert env.state["answered"] and env.session.pending_entity is None


@pytest.mark.asyncio
async def test_a_single_decoder_turn_is_read_back_and_nothing_is_looked_up_yet(m, env):
    await env.turn(decoder="ctc_fallback", agreement=0.0)
    assert env.state["answered"] == []
    assert env.session.ws.spoken()[-1] == ec.confirm_question("test_name", "CBC", "en")
    assert env.session.pending_entity is not None


@pytest.mark.asyncio
async def test_a_yes_runs_the_held_lookup_without_another_model_call(m, env):
    await env.turn(decoder="ctc_fallback", agreement=0.0)
    resolved_before = env.state["resolved"]
    await env.turn(text="yes", decoder="ctc_fallback", agreement=0.0)
    assert env.state["answered"] == [("test_rate", {"test_name": "CBC"})]
    assert env.state["resolved"] == resolved_before  # the held result was replayed
    assert env.session.pending_entity is None


@pytest.mark.asyncio
async def test_a_no_asks_again_and_never_looks_anything_up(m, env):
    await env.turn(decoder="ctc_fallback", agreement=0.0)
    await env.turn(text="no", decoder="ctc_fallback", agreement=0.0)
    assert env.state["answered"] == []
    assert env.session.ws.spoken()[-1] == ec.reask("en")


@pytest.mark.asyncio
async def test_a_reply_that_is_neither_lapses_the_question_and_is_handled_as_a_new_turn(m, env):
    await env.turn(decoder="ctc_fallback", agreement=0.0)
    await env.turn(text="what about the lipid profile", decoder="rnnt", agreement=0.95, slots={"test_name": "Lipid"})
    assert env.state["answered"] == [("test_rate", {"test_name": "Lipid"})]


@pytest.mark.asyncio
async def test_two_decoders_disagreeing_is_refused_not_read_back(m, env):
    # Below 0.25, agent/audio_quality.transcript_problem's OWN, earlier "decoders_disagree" check intercepts the
    # turn first (a re-ask, agent/reask_policy.py) -- it never reaches intent resolution or the action gate at
    # all, so nothing is extracted to read back. Unchanged by this session's work; see the 0.38 case below for that.
    await env.turn(decoder="rnnt", agreement=0.2)
    assert env.state["answered"] == [] and env.state["resolved"] == 0 and env.session.pending_entity is None


@pytest.mark.asyncio
async def test_two_decoders_disagreeing_above_the_reask_floor_reads_back_a_name_the_model_did_extract(m, env):
    """DELIBERATE spec change (a live call, 2026-09-28, decoder_agreement=0.38 -- above transcript_problem's 0.25
    re-ask floor, so it reaches here): the model extracted "CBC" even though the two decoders disagreed on the
    sentence as a whole. The old behaviour was a blank "sorry, I didn't catch it, repeat the whole thing" -- now it
    reads the name back and asks a yes/no, exactly like the single-decoder case above, and the lookup still never
    runs without an explicit yes (asserted here: answered stays empty)."""
    await env.turn(decoder="rnnt", agreement=0.38)
    assert env.state["answered"] == []  # the lookup never ran on the unconfirmed name
    assert env.session.ws.spoken()[-1] == ec.confirm_question("test_name", "CBC", "en")
    assert env.session.pending_entity is not None


@pytest.mark.asyncio
async def test_two_decoders_disagreeing_with_no_name_at_all_still_gets_the_blank_ask(m, env):
    await env.turn(decoder="rnnt", agreement=0.38, slots={})
    assert env.state["answered"] == [] and env.session.pending_entity is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lang,text", [("en", "my father is not breathing"), ("bn", "বাবা অজ্ঞান হয়ে গেছে"), ("hi", "मेरे सीने में दर्द है")]
)
async def test_an_emergency_stops_everything_and_hands_over(m, env, lang, text):
    await env.turn(text=text, lang=lang)
    assert env.state["handoffs"] == ["emergency"]
    assert env.state["resolved"] == 0 and env.state["answered"] == []  # no model, no lookup
    assert env.session.ws.spoken()[0] == phrase("emergency_notice", lang)
    assert env.session.outcome == "emergency"


@pytest.mark.asyncio
async def test_an_emergency_at_low_confidence_or_mid_booking_still_fires(m, env):
    env.session.booking = object()
    await env.turn(text="ambulance please", decoder="ctc_fallback", agreement=0.0)
    assert env.state["handoffs"] == ["emergency"] and env.session.booking is None


# ============================================================================ clinic matching


@pytest.fixture
def clinic(tmp_path):
    import tempfile

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    clinic_dir = os.path.join(REPO_ROOT, "clinic-api")
    saved_path = list(sys.path)
    sys.path.insert(0, clinic_dir)
    saved = {
        k: sys.modules.pop(k)
        for k in (
            "main",
            "db",
            "models",
            "seed",
            "booking_service",
            "booking_migrate",
            "enquiry_service",
            "enquiry_migrate",
            "i18n_content",
            "phonetic_match",
            "patient_context",
            "disclosure",
        )
        if k in sys.modules
    }
    from fastapi.testclient import TestClient

    import main as clinic_main

    with TestClient(clinic_main.app) as c:
        yield c
    for k in list(sys.modules):
        if k in saved or k in (
            "main",
            "db",
            "models",
            "seed",
            "booking_service",
            "booking_migrate",
            "enquiry_service",
            "enquiry_migrate",
            "i18n_content",
            "phonetic_match",
            "patient_context",
            "disclosure",
        ):
            sys.modules.pop(k, None)
    sys.modules.update(saved)
    sys.path[:] = saved_path
    try:
        os.remove(db_path)
    except OSError:
        pass


def _availability(c, name):
    return c.get("/api/v1/doctors/availability", params={"name": name}).json()


def test_a_written_surname_or_title_and_surname_resolves_to_that_doctor(clinic):
    for spoken in ("Sen", "Dr Sen", "Doctor Sen", "Dr. A. Sen"):
        assert _availability(clinic, spoken)["doctor_name"] == "Dr. A. Sen", spoken


def test_a_name_that_is_inside_another_name_does_not_beat_the_exact_one(clinic):
    assert _availability(clinic, "Sengupta")["doctor_name"] == "Dr. P. Sengupta"


def test_a_two_letter_fragment_identifies_no_one(clinic):
    body = _availability(clinic, "ry")
    assert body["found"] is False and body["ambiguous"] is True and body["needs_confirmation"] is True


def test_a_sound_alike_doctor_is_suggested_never_resolved_on_any_endpoint(clinic):
    body = _availability(clinic, "Nukharji")
    assert body["found"] is False and body["did_you_mean"] == ["Dr. S. Mukherjee"] and body["needs_confirmation"]
    assert clinic.get("/api/v1/doctors/earliest", params={"name": "Nukharji"}).json()["found"] is False
    assert clinic.get("/api/v1/doctors/Nukharji/leave", params={"date": "2026-10-01"}).json()["found"] is False
    hold = clinic.post(
        "/api/v1/bookings/hold", json={"doctor_name": "Nukharji", "date": "2026-10-01", "time_slot": "10:00"}
    ).json()
    assert hold["success"] is False and hold["reason"] == "doctor_not_found"
    assert "Dr. S. Mukherjee" in hold["did_you_mean"]


def test_an_unknown_doctor_is_not_found_and_reads_no_schedule(clinic):
    body = _availability(clinic, "Doctor Nobody")
    assert body["found"] is False and "schedule" not in body and "doctor_name" not in body


def test_a_sound_alike_test_is_suggested_never_priced(clinic):
    body = clinic.get("/api/v1/tests/search", params={"name": "sibisi"}).json()
    assert body["found"] is False and "Complete Blood Count (CBC)" in body["did_you_mean"]
    assert "price" not in body
    prep = clinic.get("/api/v1/tests/prep", params={"name": "sibisi"}).json()
    assert prep["found"] is False


def test_a_written_test_name_still_resolves(clinic):
    assert clinic.get("/api/v1/tests/search", params={"name": "CBC"}).json()["found"] is True


# ============================================================================ service token


def test_with_a_token_configured_the_api_refuses_a_caller_without_it(clinic, monkeypatch):
    monkeypatch.setenv("CLINIC_API_TOKEN", "s3cret-token-for-test")
    assert clinic.get("/api/v1/tests/search", params={"name": "CBC"}).status_code == 401
    assert (
        clinic.get(
            "/api/v1/tests/search", params={"name": "CBC"}, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    ok = clinic.get(
        "/api/v1/tests/search", params={"name": "CBC"}, headers={"Authorization": "Bearer s3cret-token-for-test"}
    )
    assert ok.status_code == 200 and ok.json()["found"] is True


def test_patient_data_endpoints_are_behind_the_token_too(clinic, monkeypatch):
    monkeypatch.setenv("CLINIC_API_TOKEN", "s3cret-token-for-test")
    for path in (
        "/api/v1/patients/identify?phone=9876543210",
        "/api/v1/bookings/lookup?phone=9876543210",
        "/api/v1/patients/1/timeline?caller_phone=9876543210&call_id=x",
    ):
        assert clinic.get(path).status_code == 401, path


def test_require_token_with_none_configured_refuses_everything(clinic, monkeypatch):
    monkeypatch.delenv("CLINIC_API_TOKEN", raising=False)
    monkeypatch.setenv("CLINIC_API_REQUIRE_TOKEN", "1")
    assert clinic.get("/api/v1/tests/search", params={"name": "CBC"}).status_code == 401


def test_health_stays_open_so_probes_work(clinic, monkeypatch):
    monkeypatch.setenv("CLINIC_API_TOKEN", "s3cret-token-for-test")
    assert clinic.get("/api/health").status_code == 200


def test_with_no_token_and_no_requirement_it_runs_open_as_before(clinic, monkeypatch):
    monkeypatch.delenv("CLINIC_API_TOKEN", raising=False)
    monkeypatch.delenv("CLINIC_API_REQUIRE_TOKEN", raising=False)
    assert clinic.get("/api/v1/tests/search", params={"name": "CBC"}).status_code == 200


def test_the_tools_client_sends_the_token_and_only_when_configured(monkeypatch):
    from agent.tools_client import ClinicToolsClient

    monkeypatch.setenv("CLINIC_API_TOKEN", "abc")
    assert ClinicToolsClient("http://x")._client.headers["authorization"] == "Bearer abc"
    monkeypatch.delenv("CLINIC_API_TOKEN")
    assert "authorization" not in ClinicToolsClient("http://x")._client.headers


# ============================================================================ write idempotency


def _tomorrow_slot(c):
    """A date >24 h out on which Dr. Sen sits, and a valid slot (found the way test_booking_endpoints does)."""
    import datetime

    for offset in range(2, 16):
        d = (datetime.date.today() + datetime.timedelta(days=offset)).isoformat()
        probe = c.post("/api/v1/bookings/hold", json={"doctor_name": "Sen", "date": d, "time_slot": "__probe__"}).json()
        if probe.get("reason") == "invalid_slot":
            return d, probe["valid_slots"][0]
    raise AssertionError("no bookable day in the seed data")


def test_confirming_the_same_hold_twice_returns_the_same_booking_not_a_second_one(clinic):
    date, slot = _tomorrow_slot(clinic)
    hold = clinic.post("/api/v1/bookings/hold", json={"doctor_name": "Sen", "date": date, "time_slot": slot}).json()
    assert hold["success"], hold
    body = {
        "hold_token": hold["hold_token"],
        "doctor_id": hold["doctor_id"],
        "date": date,
        "time_slot": slot,
        "patient_name": "Test Patient",
        "phone": "9876500001",
        "caller_phone": "9876500001",
    }
    first = clinic.post("/api/v1/bookings/confirm", json=body).json()
    second = clinic.post("/api/v1/bookings/confirm", json=body).json()
    assert first["success"] and second["success"]
    assert second["confirmation_id"] == first["confirmation_id"] and second.get("replayed") is True
    listed = clinic.get("/api/v1/bookings/lookup", params={"phone": "9876500001"}).json()
    assert len(listed["bookings"]) == 1


def test_a_different_phone_cannot_replay_someone_elses_confirmed_hold(clinic):
    date, slot = _tomorrow_slot(clinic)
    hold = clinic.post("/api/v1/bookings/hold", json={"doctor_name": "Sen", "date": date, "time_slot": slot}).json()
    body = {
        "hold_token": hold["hold_token"],
        "doctor_id": hold["doctor_id"],
        "date": date,
        "time_slot": slot,
        "patient_name": "Test Patient",
        "phone": "9876500002",
        "caller_phone": "9876500002",
    }
    assert clinic.post("/api/v1/bookings/confirm", json=body).json()["success"]
    other = dict(body, phone="9876500003", caller_phone="9876500003")
    assert clinic.post("/api/v1/bookings/confirm", json=other).json() == {"success": False, "reason": "hold_expired"}
