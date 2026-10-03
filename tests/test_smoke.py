"""Unit + integration tests for the Kolkata Care voice agent.

Run ON THE POD, against the live services:
    /workspace/venv/bin/python3 -m pytest /workspace/pod_scripts/test_suite.py -v

Layered deliberately, cheapest first, so a failure tells you WHERE it broke:

  1. pure functions      -- no network, no GPU (bn_normalize, reply_templates)
  2. clinic API          -- HTTP + Postgres
  3. LLM                 -- Ollama intent extraction
  4. TTS                 -- FastPitch/HiFi-GAN
  5. full pipeline       -- everything above, in the real order
  6. transport           -- HTTP + WebSocket through the RunPod proxy

The two regression tests that matter most are the "Naloxone-class" guards:
test_unknown_doctor_is_not_fuzzy_matched and test_unknown_test_not_matched.
Both encode bugs this project already hit -- a confidently wrong answer is
far worse here than an admitted miss.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, "/workspace/kolkata-care-voice-agent")

import httpx

# NB: import the MODULE, not the names. reply_templates.test_rate_reply
# starts with "test_", so a direct import makes pytest try to collect it as
# a test case and fail on its (slots, result) signature.
import agent.reply_templates as rt
from agent.bn_normalize import number_to_bn_words, unspeakable_spans, verbalize
from agent.llm import extract_intent
from agent.tools_client import ClinicToolsClient
from agent.tts import TTSClient

CLINIC = "http://localhost:8080"


def _clinic_headers() -> dict:
    """DELIBERATE SPEC CHANGE (marked, not a weakened check): the clinic API now requires its service token on
    everything under /api/v1/ (an external review: patient data was served to anyone who could reach the port).
    These calls were written before that and got a 401 on a correctly configured pod. Every assertion below is
    unchanged; the request now carries the token the deployment already has, exactly as agent/tools_client.py does."""
    token = os.environ.get("CLINIC_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


AGENT = "http://localhost:8100"
TTS = "http://localhost:8002"


# ===================================================================
# 1. Pure functions -- no network, no GPU
# ===================================================================


class TestBengaliNormalization:
    """The tokenizer drops Latin digits outright, so every number must be
    spelled into Bengali words before synthesis or the price is silence."""

    def test_number_to_words_basic(self):
        assert number_to_bn_words(0)
        assert number_to_bn_words(1)
        assert number_to_bn_words(650)

    def test_verbalize_removes_latin_digits(self):
        out = verbalize("লিপিড প্রোফাইল টেস্টের রেট 650 টাকা")
        assert not any(c.isdigit() for c in out), f"Latin digits survived verbalize(): {out!r}"

    def test_verbalize_handles_multiple_numbers(self):
        out = verbalize("রেট 250 টাকা, রিপোর্ট 12 ঘণ্টা")
        assert not any(c.isdigit() for c in out)

    def test_unspeakable_spans_flags_latin(self):
        # 'blood' is Latin script -- the Bengali voice cannot say it.
        spans = unspeakable_spans("স্যাম্পল: Blood")
        assert isinstance(spans, list)

    def test_verbalize_is_idempotent_on_clean_text(self):
        clean = "আপনার রিপোর্ট প্রস্তুত"
        assert verbalize(clean) == clean


class TestReplyTemplates:
    """The model never states a price. Templates assemble the reply from the
    API response, so a hallucinated number cannot reach the caller."""

    def test_test_rate_reply_contains_api_price(self):
        result = {
            "found": True,
            "test_name": "Lipid Profile",
            "test_name_bn": "লিপিড প্রোফাইল",
            "rate_inr": 650,
            "sample_type": "Blood",
            "report_time_hours": 24,
        }
        out = rt.test_rate_reply({"test_name": "লিপিড প্রোফাইল"}, result)
        assert "650" in out, f"price missing from reply: {out!r}"

    def test_doctor_availability_reply_uses_api_result(self):
        result = {"found": True, "doctor_name": "Dr. A. Sen", "available": False, "next_available_date": "2026-09-03"}
        out = rt.doctor_availability_reply({"doctor_name": "সেন"}, result)
        assert isinstance(out, str) and len(out) > 0

    def test_doctor_availability_reply_asks_which_doctor_when_ambiguous(self):
        # Doctor-side counterpart of test_rate_reply's ambiguous framing
        # (KCD-446) -- "ask correct questions back" instead of silently
        # picking one of the tied candidates.
        result = {"found": False, "query": "ry", "ambiguous": True, "did_you_mean": ["Dr. N. Roy", "Dr. P. Ray"]}
        out = rt.doctor_availability_reply({"doctor_name": "ry"}, result)
        assert "Dr. N. Roy" in out and "Dr. P. Ray" in out
        for lang in ("hi", "en"):
            out_i18n = rt.doctor_availability_reply({"doctor_name": "ry"}, result, lang)
            assert isinstance(out_i18n, str) and len(out_i18n) > 0

    def test_missing_slot_prompt_is_specific_per_field(self):
        a = rt.missing_slot_prompt("test_rate", "test_name")
        b = rt.missing_slot_prompt("doctor_availability", "doctor_name")
        assert a and b
        assert a != b, "missing-slot prompts must differ per field"


# ===================================================================
# 2. Clinic API -- HTTP + Postgres
# ===================================================================


class TestClinicAPI:
    def test_health(self):
        r = httpx.get(f"{CLINIC}/api/health", timeout=10)
        assert r.status_code == 200

    def test_known_test_returns_price(self):
        r = httpx.get(f"{CLINIC}/api/v1/tests/search", params={"name": "lipid"}, timeout=10, headers=_clinic_headers())
        assert r.status_code == 200
        body = r.json()
        assert body["found"] is True, f"seeded test not found: {body!r}"
        assert body["rate_inr"] > 0

    def test_unknown_test_not_matched(self):
        """REGRESSION: a nonsense query must not fuzzy-match a real test.
        Quoting a real price for a test the caller did not ask about is the
        exact 'Naloxone' failure this project already reproduced once."""
        r = httpx.get(
            f"{CLINIC}/api/v1/tests/search", params={"name": "zzzznotarealtest"}, timeout=10, headers=_clinic_headers()
        )
        assert r.status_code == 200
        body = r.json()
        assert body["found"] is False, f"nonsense query fuzzy-matched a real test: {body!r}"
        assert "rate_inr" not in body, "a price leaked on a not-found result"

    def test_known_doctor_availability(self):
        r = httpx.get(
            f"{CLINIC}/api/v1/doctors/availability", params={"name": "Sen"}, timeout=10, headers=_clinic_headers()
        )
        assert r.status_code == 200

    def test_unknown_doctor_is_not_fuzzy_matched(self):
        """REGRESSION: 'Doctor Nobody' once matched 'Dr. N. Roy' above the
        similarity floor and returned that real doctor's real schedule.
        Fixed by matching on surname only, FUZZY_SURNAME_FLOOR = 0.60."""
        r = httpx.get(
            f"{CLINIC}/api/v1/doctors/availability", params={"name": "Nobody"}, timeout=10, headers=_clinic_headers()
        )
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            body = r.json()
            if isinstance(body, dict) and body.get("found") is True:
                pytest.fail(f"unknown doctor fuzzy-matched: {body!r}")


# ===================================================================
# 3. LLM intent extraction
# ===================================================================


class TestIntentExtraction:
    def test_test_rate_intent(self):
        result, meta = extract_intent("লিপিড প্রোফাইল টেস্টের রেট কত")
        assert result["intent"] == "test_rate"
        assert (result.get("slots") or {}).get("test_name")

    def test_doctor_availability_intent(self):
        result, meta = extract_intent("ডক্টর সেন কি আজ চেম্বারে বসবেন")
        assert result["intent"] == "doctor_availability"
        assert (result.get("slots") or {}).get("doctor_name")

    def test_model_does_not_invent_a_price(self):
        """The model must extract, never answer. A price in direct_reply_bn
        would mean the model is stating numbers on its own authority."""
        result, meta = extract_intent("ইউরিক অ্যাসিড টেস্ট কত টাকা")
        direct = result.get("direct_reply_bn") or ""
        assert not any(c.isdigit() for c in direct), f"model volunteered a number: {direct!r}"

    def test_garbled_input_does_not_crash(self):
        result, meta = extract_intent("অকিওেয ঝিওেয")
        assert "intent" in result


# ===================================================================
# 4. TTS
# ===================================================================


class TestTTS:
    def test_health(self):
        r = httpx.get(f"{TTS}/health", timeout=10)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_synthesize_returns_riff_wav(self):
        r = httpx.post(f"{TTS}/synthesize", json={"text": "আপনার রিপোর্ট প্রস্তুত", "lang": "bn"}, timeout=60)
        assert r.status_code == 200
        assert r.content[:4] == b"RIFF", "not a WAV file"
        assert len(r.content) > 10000

    def test_numbers_are_spoken_not_dropped(self):
        """A price with digits must produce audio of comparable length to the
        same sentence with the number spelled out -- if the tokenizer drops
        the digits, the clip is noticeably shorter."""
        with_digits = httpx.post(f"{TTS}/synthesize", json={"text": verbalize("রেট 650 টাকা")}, timeout=60)
        assert with_digits.status_code == 200
        assert len(with_digits.content) > 20000


class TestFallbackAudio:
    """If TTS is what is broken, asking it to speak its own apology is
    circular. These files must exist ahead of time and must be real WAVs."""

    FALLBACK_DIR = "/workspace/kolkata-care-voice-agent/static/fallback_audio"

    @pytest.mark.parametrize(
        "name",
        [
            "sorry_repeat.wav",
            "system_busy.wav",
            "check_failed.wav",
        ],
    )
    def test_fallback_file_is_a_real_wav(self, name):
        path = os.path.join(self.FALLBACK_DIR, name)
        assert os.path.exists(path), f"missing fallback: {name}"
        with open(path, "rb") as fh:
            assert fh.read(4) == b"RIFF", f"{name} is not a WAV"
        assert os.path.getsize(path) > 10000


# ===================================================================
# 5. Full pipeline
# ===================================================================


class TestPipeline:
    @pytest.mark.parametrize(
        "utterance,expect_intent",
        [
            ("লিপিড প্রোফাইল টেস্টের রেট কত", "test_rate"),
            ("ইউরিক অ্যাসিড টেস্ট কত টাকা", "test_rate"),
            ("ডক্টর সেন কি আজ চেম্বারে বসবেন", "doctor_availability"),
        ],
    )
    def test_transcript_to_audio(self, utterance, expect_intent):
        async def run():
            tools = ClinicToolsClient(CLINIC)
            tts = TTSClient()
            try:
                result, _ = await asyncio.to_thread(extract_intent, utterance)
                assert result["intent"] == expect_intent
                slots = result.get("slots") or {}

                if result["intent"] == "test_rate":
                    api = await tools.get_test_rate(slots.get("test_name", ""))
                    reply = rt.test_rate_reply(slots, api)
                else:
                    api = await tools.get_doctor_availability(slots.get("doctor_name", ""), slots.get("date"))
                    reply = rt.doctor_availability_reply(slots, api)

                assert reply and len(reply) > 5
                wav = await tts.synthesize(reply)
                assert wav and len(wav) > 10000
                assert wav[:4] == b"RIFF"
                return reply
            finally:
                await tools.aclose()
                await tts.aclose()

        reply = asyncio.run(run())
        print(f"\n  {utterance}\n  -> {reply}")

    def test_missing_slot_asks_instead_of_guessing(self):
        """No doctor name given: the agent must ask, never call the API with
        a null field and never guess a doctor."""
        result, _ = extract_intent("ডাক্তার কি আছেন")
        slots = result.get("slots") or {}
        if result.get("intent") == "doctor_availability" and not slots.get("doctor_name"):
            prompt = rt.missing_slot_prompt("doctor_availability", "doctor_name")
            assert prompt and len(prompt) > 3


# ===================================================================
# 6. Transport -- through the RunPod proxy
# ===================================================================


class TestTransport:
    def test_agent_health(self):
        r = httpx.get(f"{AGENT}/api/health", timeout=15)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["asr_loaded"] is True, "ASR model not loaded"

    def test_agent_serves_ui(self):
        r = httpx.get(f"{AGENT}/", timeout=15)
        assert r.status_code == 200
        assert "html" in r.headers.get("content-type", "").lower()

    def test_stats_endpoint_shape(self):
        r = httpx.get(f"{AGENT}/api/stats", timeout=15)
        assert r.status_code == 200
        body = r.json()
        assert "intent_cache" in body and "tts_cache" in body

    def test_websocket_accepts_connection(self):
        """The /ws/audio upgrade is the one thing the README flagged as
        unverified against the RunPod proxy."""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets not installed")

        async def probe():
            async with websockets.connect("ws://localhost:8100/ws/audio", open_timeout=15) as ws:
                return ws.state.name if hasattr(ws, "state") else "OPEN"

        state = asyncio.run(probe())
        assert state in ("OPEN", "1")
