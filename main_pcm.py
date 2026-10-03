"""Kolkata Care Diagnostics -- voice agent on the RAW PCM transport.

GENERATED FILE -- do not edit directly. Produced by
tools/make_pcm_variant.py from main.py; edit main.py and re-run that.

This variant changes only the TRANSPORT (how audio arrives and how the
turn detector reads it):
  * No ffmpeg, no WebM, no temp audio files, no subprocess per poll.
  * Appending is O(chunk) and reading the tail is O(tail), where the WebM
    path was O(call length) EVERY poll -- O(T^2) per call.
  * The sample index IS the timeline, exactly, so processed_until_s
    cannot drift away from real time.
See agent/pcm_buffer.py for the argument and the arithmetic.

----------------------------------------------------------------------


Turn loop, once a caller's utterance is judged complete (agent/vad_stream.py):

  utterance WAV -> ASR (agent/asr.py, IndicConformer)
                -> intent+slots (agent/llm.py, Ollama JSON-mode,
                   fronted by agent/semantic_cache.py)
                -> Spring Boot lookup (agent/tools_client.py) -- ALWAYS
                   live, never cached; see semantic_cache.py's docstring
                -> reply text, TEMPLATED from the API response, never
                   restated by the model (agent/reply_templates.py)
                -> TTS (agent/tts.py) -> WAV bytes back over the socket

Every stage has a named failure path (see _dispatch_turn) so a caller never
gets dead air: ASR-empty, LLM-failure, tool-failure and TTS-failure each
speak a distinct, pre-recorded apology rather than the process hanging or
the socket just going quiet. See README.md "Error handling" for the full
table and the reasoning behind each choice.

HALF-DUPLEX GATE
----------------
The mic is open for the entire call, and the agent's replies play out of
the caller's speaker. With no gate, the agent hears itself: its own
greeting lands in the same buffer the turn detector is watching, so VAD
fires a "the caller finished talking" on the agent's own voice, ASR
transcribes the agent, and processed_until_s advances past audio the
caller never produced. That is a self-sustaining loop, and it is what
made real calls cut the caller off in the first second and then run a
turn behind for the rest of the call.

Browser echoCancellation does not save this. It is built to cancel a
remote WebRTC peer's rendered stream; here the audio is synthesized
locally and played through Web Audio, which the canceller never sees as a
far-end reference.

So the pipeline is explicitly half-duplex, gated from BOTH ends:
  * client mutes the mic track while agent audio is playing (static/
    index.html) -- the track stays live and keeps emitting, so the WebM
    timeline never breaks, it just carries silence;
  * server refuses to run turn detection while `agent_speaking`, then
    resynchronizes processed_until_s past the muted region once playback
    is confirmed finished.

The cost is no barge-in: a caller cannot interrupt the agent mid-sentence.
That is a real limitation, chosen deliberately over the alternative, which
was a system that interrupted ITSELF. Supporting barge-in properly needs
an acoustic echo canceller with the played audio as a reference signal
(WebRTC APM or speex AEC), which is a much larger change.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime
import io
import json
import logging
import os
import secrets
import tempfile
import time
import uuid
import wave

import numpy as np
import torchaudio
from agent.pcm_buffer import PcmCallBuffer, SAMPLE_RATE
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import abuse, action_gate, anger, call_end, slot_grouping, topic_flow
from agent import entity_confirmation as entity_text
from agent import history_templates as history_text
from agent import messages as agent_messages
from agent import patient_context as patient_policy
from agent import security_check as sec_text
from agent.acknowledgement import select_acknowledgement, slot_echo
from agent.admission import AdmissionController, HealthMonitor, http_probe
from agent.apology import enforce_single_apology
from agent.asr import TurnASR
from agent.asr_router import ASRRouter, HTTPASREngine, UnroutableLanguageError
from agent.attention import attend
from agent.audio_quality import assess as assess_audio
from agent.audio_quality import transcript_problem
from agent.booking_flow import (
    BookingState,
    classify_yes_no,
    correction_acknowledgement,
    effective_phone,
    is_ready_to_confirm,
    mark_awaiting_charge_confirm,
    mark_confirming,
    merge_slots,
    merge_spelling,
    missing_required,
    new_state,
)
from agent.call_record import FINISH_DEADLINE_S as CALL_RECORD_DEADLINE_S
from agent.call_record import CallRecorder
from agent.call_score import CallSignals, score_call
from agent.call_state import (
    CallState,
    apply_caller_state,
    apply_channel_quality,
    apply_confidence,
    apply_language,
    new_call_state,
)
from agent.call_summary import build_call_summary
from agent.channel_quality import CHANNEL_CLEAN_16K, classify_channel
from agent.clause_split import split_into_clauses
from agent.code_switch import mixture_bucket
from agent.conditioning import condition as condition_audio
from agent.confidence_gate import VERIFIED, confidence_state, is_low_confidence
from agent.detector_budget import run_within_budget
from agent.detector_budget import snapshot as detector_budget_snapshot
from agent.disclosure import disclosure_for
from agent.disclosure import version_label as disclosure_version_label
from agent.emergency import detect_emergency
from agent.endpointing import classify_completeness
from agent.endpointing import decide as decide_turn_end
from agent.enquiry_followup import FOLLOWUP_MAX_GAP, entities_from_turn, recent_unique, resolve_followup_slot
from agent.fast_path import COMMIT_FLOOR, Catalogue, FastPath
from agent.filler import await_with_filler
from agent.full_duplex import FullDuplexProcessor, wav_to_pcm16k
from agent.golden_buckets import channel_buckets
from agent.history_intent import detect_history_question
from agent.clinical_safety import is_clinical_interpretation
from agent.complaint import detect_complaint, is_complaint
from agent.compare_flow import KIND_NOT_FOUND, KIND_TEST, asks_to_compare, build_comparison
from agent.doctor_list import asks_for_the_doctor_list
from agent.doctor_personal_request import is_doctor_personal_request
from agent.callback_config import CALLBACKS_ENABLED
from agent.callback_request import asks_for_a_callback, build_callback_reason
from agent.spoken_codes import parse_phone as _parse_spoken_phone
from agent.unverifiable_claim import CATEGORY_REPORT_RESULT, detect_unverifiable_claim
from agent.human_request import asks_for_a_person
from agent.identity import IdentityState
from agent.lang_select import engines_needed, languages_to_verify, pick_candidate
from agent.lang_select import speakable as _speakable
from agent.language_policy import language_mismatch
from agent.language_switch import detect_language_switch_request
from agent.latency_metrics import turn_latency_by_language
from agent.lay_terms import ABDOMEN, BLOOD, match_lay_term
from agent.lid import (
    SUPPORTED_LANGUAGES,
    ASRLanguageRouter,
    LIDResult,
    SpeechBrainVoxLingua107LID,
)
from agent.llm import ExtractionError, extract_intent
from agent.near_end import NearEndProfile, level_and_pitch
from agent.outcome_metrics import (
    apology_events,
    audio_issue_buckets,
    barge_in_interrupts,
    channel_quality_buckets,
    code_switch_buckets,
    golden_buckets,
    insufficient_information,
    language_mismatches,
    reask_outcomes,
    senior_detections,
)
from agent.pause_profile import PauseProfile
from agent.persona import is_clean as persona_clean
from agent.phrases import HANDOFF_ALL_LANGUAGES, phrase, prewarm_lines
from agent.playback_gate import PlaybackGate
from agent.reask_policy import ReaskTracker
from agent.reply_templates import (
    add_test_reply,
    blood_test_list_reply,
    booking_confirmation_readback,
    booking_reply,
    cancel_reply,
    clinic_faq_reply,
    compare_options_reply,
    doctor_list_reply,
    doctors_by_department_reply,
    conflict_reply,
    department_route_reply,
    doctor_availability_reply,
    insufficient_information_reply,
    lookup_reply,
    merged_test_prep_reply,
    missing_slot_prompt,
    multi_test_reply,
    multiple_bookings_reply,
    multiple_test_bookings_reply,
    reschedule_reply,
    resend_reply,
    spelling_prompt,
    spelling_readback,
    test_prep_reply,
    test_rate_reply,
)
from agent.security_check import SecurityCheck
from agent.semantic_cache import SemanticCache
from agent.semantic_cache import embed as _embed_probe
from agent.senior_care import KindnessPlanner
from agent.senior_voice import SeniorEvidence, explicit_senior_cue, stated_age_is_senior
from agent.senior_voice import estimate as estimate_senior
from agent.speaker_change import SpeakerChangeDetector, voice_embedding
from agent.speech_norm import contains_critical_figure
from agent.speech_policy import derive_policy, effective_rate, limit_questions
from agent.tools_client import ClinicToolsClient, ToolCallError
from agent.tts import FIGURE_SPEECH_SPEED, TTSClient, UnspeakableTextError
from agent.tts_router import TTSRouter
from agent.turn_ack import AckTracker, thanks_for
from agent.vad_stream import TurnDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

POLL_INTERVAL_S = 0.5
IDLE_TIMEOUT_S = 90.0
UTTERANCE_PAD_S = 0.15  # small trailing pad so ASR doesn't clip the last phoneme

# A live call was observed closing itself ~26s after the last exchange --
# far short of IDLE_TIMEOUT_S, which only fires at 90s. That gap points to
# an intermediate proxy (RunPod's or an nginx in front of it) closing
# WebSocket connections that go quiet for a while, independent of this
# app's own idle logic. A small periodic heartbeat keeps real traffic
# flowing on the socket so no proxy in between decides it's abandoned.
HEARTBEAT_INTERVAL_S = 15.0

# Backstop for the half-duplex gate. Normally the client reports playback
# finished and the gate lifts immediately; this only fires when that
# message never arrives (JS error, stale cached page, a client that
# predates the control channel). Generous on purpose -- lifting the gate
# early puts the agent back to hearing itself, which is the bug.
PLAYBACK_GUARD_S = 3.0

# On resync, rewind slightly before the buffer's decoded end. The region
# being skipped is muted silence, so rewinding into it costs nothing,
# while NOT rewinding risks clipping the caller's first syllable if the
# WebM decode is running a beat behind real time.
RESYNC_REWIND_S = 0.25

# KCD-049: poll granularity. The fixed 0.5 s poll costs half its period on
# average (0.25 s) on EVERY turn. Instead: an ACTIVE cadence while the caller
# has spoken this turn, and -- the part that removes the tax -- a wake-up at
# exactly the moment the detector says its answer could change
# (TurnDecision.commit_after_s). REASONED, not measured under load on the pod:
# each active poll re-runs Silero over the unprocessed tail, so a shorter
# cadence costs CPU. Set ACTIVE_POLL_INTERVAL_S=0.5 to get the old behaviour.
ACTIVE_POLL_INTERVAL_S = float(os.environ.get("ACTIVE_POLL_INTERVAL_S", "0.2"))
MIN_WAKE_S = 0.02
WAKE_SLACK_S = 0.005

# KCD-048: semantic endpointing (agent/endpointing.py). OFF by default: it
# spends one extra speculative ASR call per candidate turn end, and shortens the
# silence needed after a "finished" transcript, which changes how the agent
# feels to talk to. That needs an A/B on the pod before it is anyone's default.
SEMANTIC_ENDPOINTING = os.environ.get("SEMANTIC_ENDPOINTING", "0") == "1"
SPECULATIVE_ASR_TIMEOUT_S = 1.5
MAX_SPECULATIONS_PER_TURN = 3
SPECULATION_MATCH_TOLERANCE_S = 0.25

# Trailing audio never trusted as "confirmed silence". 0.3 s covers the decode
# lag of a growing WebM; the PCM variant reads exact sample positions and
# overrides this default (tools/make_pcm_variant.py).
TURN_TAIL_GUARD_S = float(os.environ.get("TURN_TAIL_GUARD_S", "0.1"))

# KCD-051/052: acoustic echo cancellation + barge-in on the PCM transport. OFF by
# default: the canceller and detector are validated on SYNTHETIC echo only and
# need the pod's real handset audio (tools/echo_eval.py) before anyone relies on
# them. Off, the half-duplex gate behaves exactly as before.
AEC_BARGE_IN = os.environ.get("AEC_BARGE_IN", "0") == "1"
# How far before the detected onset the caller's utterance is taken to start
# (the detector needs ~0.1 s of evidence, so speech began before it fired).
BARGE_IN_PREROLL_S = 0.35

# KCD-513: "always" opens every substantive reply with "Thank you for telling me" (the operator's
# wording, editable from the database); "varied" is the short rotating acknowledgement.
ACK_MODE = os.environ.get("ACK_MODE", "always")
# KCD-495: history is spoken only after the security questions pass. "off" restores the older behaviour
# of offering a person instead of asking (for a deployment that verifies some other way).
SECURITY_QUESTIONS = os.environ.get("SECURITY_QUESTIONS", "on")
# KCD-047: learn each caller's own pause length as the call goes and wait accordingly.
ADAPTIVE_PAUSE = os.environ.get("ADAPTIVE_PAUSE", "on")
# KCD-053/055: attenuate what is far below the caller's own level and not their voice.
NEAR_END_ATTENTION = os.environ.get("NEAR_END_ATTENTION", "on")

# A test the caller named must match the catalogue at least this well before it is used to look up
# their history; below it the agent ASKS which test (KCD-500: never guess). REASONED.
HISTORY_TEST_MATCH_MIN = 0.85

# KCD-055/057: condition the audio the recogniser hears -- noise suppression (only when
# the clip is noisy, and guarded so it cannot eat the speech) and level normalisation
# with a limiter (agent/conditioning.py). "off" (the default), "level" or "full".
# OFF by default: the proxies it is validated on are synthetic, and whether it helps
# the real recognisers is exactly what tools/conditioning_eval.py measures on a pod.
# The RETRY path (a poor first result on a noisy or faint line) always uses it.
CONDITION_INPUT = os.environ.get("CONDITION_INPUT", "off")

# KCD-461: "a natural filler is spoken when a stage exceeds its
# threshold... silence never exceeds a stated maximum." REASONED, not
# measured against real telephony (no pod to time it against) --
# comfortably above a fast_path/cache hit's near-zero latency and a
# warm LLM call's own "~9s across retries" evidence agent/llm.py's own
# KCD-465 fix cites, short enough that a genuinely slow turn does not
# leave the caller wondering whether the line is still open.
# OWNER'S INSTRUCTION, 2026-09-25: a holding phrase only when the answer is taking MORE THAN 0.7 s, counted from the
# moment the caller stopped speaking (so recognition time counts); an answer that comes sooner has no filler at all.
FILLER_THRESHOLD_S = float(os.environ.get("FILLER_THRESHOLD_S", "0.7"))  # was 2.5 s of silence, then 0.9 s
# ...but the semantic cache is given this long to answer before the model is waited on at all, so a hit is never
# announced by a filler that the answer was about to make pointless.
FILLER_MIN_WAIT_S = float(os.environ.get("FILLER_MIN_WAIT_S", "0.25"))

# OWNER'S INSTRUCTION, 2026-09-25: when the caller says nothing for this long after the agent finished speaking, ask
# once whether there is anything else (the call ends if not); if there is still nothing after SILENCE_CLOSE_S, end it.
SILENCE_PROMPT_S = float(os.environ.get("SILENCE_PROMPT_S", "5"))
SILENCE_CLOSE_S = float(os.environ.get("SILENCE_CLOSE_S", "5"))

# KCD-076: agent/channel_quality.py's FFT-based classification is pure
# CPU arithmetic over one short clip -- REASONED, not measured against
# Appendix B's real per-call budget breakdown (no live pod to time it
# against right now); generous enough that it should never fire in
# practice on hardware this pipeline already assumes, tight enough that
# a genuinely pathological clip cannot silently eat into the turn.
CHANNEL_QUALITY_BUDGET_S = 0.08

# agent/audio_quality.assess + agent/senior_voice.estimate: FFT/autocorrelation
# over one utterance, run through detector_budget so a pathological clip
# cannot delay the turn. REASONED, not measured on the pod: generous for a
# 3-6 s clip on the hardware this pipeline already assumes.
AUDIO_ANALYSIS_BUDGET_S = 0.5

CLINIC_API_BASE = os.environ.get("CLINIC_API_BASE", "http://localhost:8080")

# Languages the live path routes between. "bn" alone reproduces the original
# Bengali-only behaviour (no LID, no Hindi ASR loaded) -- the rollback lever.
# Bengali is always active: it is the clinic's primary language and the
# greeting language.
ACTIVE_LANGUAGES = tuple(
    dict.fromkeys(
        ["bn"]
        + [
            x
            for x in os.environ.get("VOICE_AGENT_LANGUAGES", "bn,hi,en").replace(" ", "").split(",")
            if x in SUPPORTED_LANGUAGES
        ]
    )
)
ENGLISH_ASR_URL = os.environ.get("ENGLISH_ASR_URL", "http://localhost:8003")
TTS_HEALTH_URL = os.environ.get("TTS_HEALTH_URL", "http://localhost:8002/health")
OLLAMA_HEALTH_URL = os.environ.get("OLLAMA_HEALTH_URL", "http://localhost:11434/api/tags")

# Admission control (agent/admission.py). Cap is per PROCESS: with the WebM
# and PCM services both running, each has its own counter.
ADMISSION_MAX_CALLS = int(os.environ.get("ADMISSION_MAX_CALLS", "28"))
ADMISSION_SHED_P95_S = float(os.environ["ADMISSION_SHED_P95_S"]) if os.environ.get("ADMISSION_SHED_P95_S") else None
ADMISSION_BYPASS_FILE = os.environ.get("ADMISSION_BYPASS_FILE", "/workspace/.ai_bypass")
ADMISSION_ADMIN_TOKEN = os.environ.get("ADMISSION_ADMIN_TOKEN", "")

app = FastAPI()

# KCD-467: "time from process start to ready is measured and bounded."
# Captured at import time -- as close to "process start" as this module
# can observe -- so _startup()'s own duration can be reported once every
# model has answered ready, rather than assumed. See /api/health.
_PROCESS_STARTED_AT = time.monotonic()
_READY_AT: float | None = None  # set once, at the end of _startup()

# ---- process-wide singletons: loaded once, shared by every call ----
_asr: TurnASR | None = None
_turn_detector: TurnDetector | None = None
_tools: ClinicToolsClient | None = None
_tts: TTSClient | None = None
_intent_cache: SemanticCache | None = None
_fast_path: FastPath | None = None
_answer_warm_task = None
_asr_router: ASRRouter | None = None
_lid: SpeechBrainVoxLingua107LID | None = None
_tts_router: TTSRouter | None = None
_admission: AdmissionController | None = None
_health: HealthMonitor | None = None
_health_task: asyncio.Task | None = None
_languages_active: tuple[str, ...] = ("bn",)


class _UnavailableASR:
    """Registered for a language this build does not serve. Fails loudly on
    use (ASRRouter's contract) instead of silently sending Hindi audio to
    the Bengali model."""

    def __init__(self, language: str):
        self.language = language

    async def transcribe_utterance(self, wav_path: str):
        raise UnroutableLanguageError(f"language {self.language!r} is not active in this build")


_fast_path_retry_at = 0.0


async def _warm_speech_models() -> None:
    """Push one real utterance per language through its ASR, and through LID.

    The first real inference on a CUDA model compiles kernels and allocates
    workspaces; measured on the pod, the first turn of the first call took
    ~23 s (three ASRs plus LID, all cold) against ~2 s once warm. A tone or
    silence does NOT do it -- the decode path only runs on speech -- so the
    warm-up speech is the agent's own greeting, synthesized by the TTS."""
    t0 = time.monotonic()
    for lang in _languages_active:
        path = os.path.join(tempfile.gettempdir(), f"kcd_warm_{lang}_{uuid.uuid4().hex[:6]}.wav")
        try:
            wav = await _tts_router.synthesize(lang, phrase("greeting", lang))
            with open(path, "wb") as f:
                f.write(wav)
            await _asr_router.transcribe(lang, path)
            if _lid is not None:
                await asyncio.to_thread(_lid.identify_path, path)
            if lang == "bn":
                # The turn detector and the audio writer are on every call's
                # first-turn path too, and both are lazy: Silero's first
                # inference and torchaudio's first save (torchcodec, loaded
                # from the network volume) are where the first caller's wait
                # went if they are not exercised here.
                import soundfile as sf

                data, sr = await asyncio.to_thread(sf.read, path, dtype="float32", always_2d=True)
                mono = torchaudio.functional.resample(__import__("torch").from_numpy(data.mean(axis=1)), sr, 16000)
                await asyncio.to_thread(_turn_detector.poll, mono, 16000)
                warm_out = path + ".save.wav"
                await asyncio.to_thread(torchaudio.save, warm_out, mono.unsqueeze(0), 16000)
                with contextlib.suppress(OSError):
                    os.remove(warm_out)
        except Exception as e:  # noqa: BLE001 - warmup is advisory
            logger.warning("speech warmup failed for %s: %s", lang, e)
        finally:
            with contextlib.suppress(OSError):
                os.remove(path)
    logger.info("speech models warm (%.1fs)", time.monotonic() - t0)


async def _fetch_catalogue() -> dict:
    """GET /api/v1/catalogue with the service token; raises on any failure or on a body that is not a catalogue."""
    import httpx as _hx

    token = os.environ.get("CLINIC_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with _hx.AsyncClient(timeout=10) as c:
        resp = await c.get(f"{CLINIC_API_BASE}/api/v1/catalogue", headers=headers)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or "tests" not in payload:
        raise ValueError("clinic-api /api/v1/catalogue did not return a catalogue")
    return payload


# The lay-term fast path (agent/lay_terms.py, below) is a deterministic, rare gate -- not every turn -- but still
# reads the catalogue's sample_type, which the FastPath.Catalogue index does not keep. Cached separately, refetched
# at most every CATALOGUE_CACHE_TTL_S, rather than hitting the network on every matching turn.
CATALOGUE_CACHE_TTL_S = 300.0
_catalogue_cache: dict = {"payload": None, "at": 0.0}


async def _cached_catalogue() -> dict:
    now = time.monotonic()
    if _catalogue_cache["payload"] is None or now - _catalogue_cache["at"] > CATALOGUE_CACHE_TTL_S:
        _catalogue_cache["payload"] = await _fetch_catalogue()
        _catalogue_cache["at"] = now
    return _catalogue_cache["payload"]


# ---- the answers every caller asks for, rendered ahead of time -------------------------------------------------
ANSWER_WARM_INTERVAL_S = float(os.environ.get("ANSWER_WARM_INTERVAL_S", "600"))
_answers_warm: dict = {"rounds": 0, "pinned_clips": 0, "last_round_s": None, "last_error": None}


async def _warm_answers_once() -> int:
    """Render, and pin in the TTS cache, the spoken audio of every test price, every test-preparation answer and every
    clinic FAQ answer, in every active language -- exactly the clauses a live turn would synthesise, so the next caller
    who asks gets them from the cache with no synthesis. The TEXT is still built from live clinic data every time
    (the truth boundary is unchanged: nothing here stores a price as a fact); only the rendered AUDIO is kept, keyed
    by the exact text, so a changed price is simply a different clip. Each round re-reads the catalogue, renders only
    clips it does not already hold, and lets clips for answers that no longer exist become evictable."""
    if _tools is None or _tts is None:
        return 0
    t0 = time.monotonic()
    cat = await _fetch_catalogue()
    keep: set[str] = set()
    rate = derive_policy("neutral")
    for lang in _languages_active:
        replies: list[str] = []
        for row in cat.get("tests", []):
            name = row["name"]
            try:
                replies.append(test_rate_reply({"test_name": name}, await _tools.get_test_rate(name), lang))
                replies.append(test_prep_reply({"test_name": name}, await _tools.get_test_prep(name, lang), lang))
            except ToolCallError as e:
                logger.warning("answer warm-up: lookup for %r failed (%s)", name, e)
        for f in cat.get("faq_topics", []):
            try:
                replies.append(
                    clinic_faq_reply({"faq_topic": f["topic"]}, await _tools.get_faq(f["topic"], lang), lang)
                )
            except ToolCallError as e:
                logger.warning("answer warm-up: FAQ %r failed (%s)", f["topic"], e)
        for reply in replies:
            for clause in split_into_clauses(reply) or [reply]:  # as _finish_enquiry_turn does
                speed = effective_rate(rate, FIGURE_SPEECH_SPEED if contains_critical_figure(clause) else None)
                try:
                    await _tts.synthesize(clause, lang, speed=speed, pin=keep)
                except Exception as e:  # noqa: BLE001 - an unspeakable or failed clip is skipped, never fatal
                    logger.debug("answer warm-up: skipped %r (%s)", clause[:30], e)
                await asyncio.sleep(0)  # never hog the loop while calls are live
    n = _tts.retain_pinned(keep)
    _answers_warm.update(
        rounds=_answers_warm["rounds"] + 1,
        pinned_clips=n,
        last_round_s=round(time.monotonic() - t0, 1),
        last_error=None,
    )
    logger.info("answer warm-up: %d clips pinned in %.1fs", n, time.monotonic() - t0)
    return n


async def _answer_warm_loop() -> None:
    while True:
        try:
            await _warm_answers_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - advisory: a failure only means the next caller pays a synthesis
            _answers_warm["last_error"] = str(e)
            logger.warning("answer warm-up failed: %s", e)
        await asyncio.sleep(ANSWER_WARM_INTERVAL_S)


async def _load_fast_path() -> FastPath | None:
    """Fetch the catalogue and build the fast path, or None if clinic-api is
    not reachable. Optional by design: without it every turn goes to the LLM,
    which is the behaviour that existed before this path did."""
    try:
        payload = await _fetch_catalogue()
        fp = FastPath(Catalogue(payload))
        logger.info("fast path ready over %d catalogue rows", len(fp.catalogue))
        return fp
    except Exception as e:  # noqa: BLE001 - degrade to LLM-only, never fail startup
        logger.warning("catalogue unavailable, fast path disabled: %s", e)
        return None


async def _maybe_reload_fast_path() -> None:
    """If the fast path failed to load at boot (clinic-api was not up yet --
    it starts AFTER this service in start_all.sh), retry at most every 30 s.
    Without this a boot-order race silently costs the fast path until the
    next restart: every turn pays an LLM call and nothing says why."""
    global _fast_path, _fast_path_retry_at
    if _fast_path is not None or time.monotonic() < _fast_path_retry_at:
        return
    _fast_path_retry_at = time.monotonic() + 30.0
    _fast_path = await _load_fast_path()


# How long the startup warm-up may take to get the intent model into VRAM. NOT the per-turn deadline
# (agent.llm.DEFAULT_DEADLINE_S, 12 s, KCD-465): a cold 7B load measured 47-74 s on this pod, so a warm-up under
# the turn deadline times out, the client disconnects, and Ollama abandons the load -- the model never becomes
# resident and the FIRST CALLER pays the cold start against a 12 s deadline (seen on the live pod: "intent model
# warmup failed ... 12.0s deadline" in both entrypoints' logs). REASONED headroom above the measured 74 s.
INTENT_WARMUP_DEADLINE_S = float(os.environ.get("INTENT_WARMUP_DEADLINE_S", "300"))


async def _warm_intent_model() -> bool:
    """Load the intent model now, with a deadline that fits a cold load. Advisory: a failure is logged and
    startup continues (calls then pay the cold start, which is the behaviour without a warm-up)."""
    try:
        _, diag = await asyncio.to_thread(extract_intent, "নমস্কার", 2, "bn", INTENT_WARMUP_DEADLINE_S)
        logger.info("intent model warm (%.1fs)", diag["total_time_s"])
        return True
    except Exception as e:  # noqa: BLE001 - warmup is advisory
        logger.warning("intent model warmup failed: %s", e)
        return False


@app.on_event("startup")
async def _startup():
    global _asr, _turn_detector, _tools, _tts, _intent_cache, _fast_path
    global _asr_router, _lid, _tts_router, _admission, _health, _health_task, _languages_active
    import httpx as _httpx

    active = list(ACTIVE_LANGUAGES)
    engines: dict = {}
    logger.info("loading IndicConformer bn...")
    _asr = await asyncio.to_thread(TurnASR)
    engines["bn"] = _asr
    if "hi" in active:
        logger.info("loading IndicConformer hi...")
        engines["hi"] = await asyncio.to_thread(TurnASR, None, "hi")
    if "en" in active:
        engines["en"] = HTTPASREngine(ENGLISH_ASR_URL)
    for lang in SUPPORTED_LANGUAGES:
        engines.setdefault(lang, _UnavailableASR(lang))
    _asr_router = ASRRouter(engines)

    _lid = None
    if len(active) > 1:
        logger.info("loading language ID (SpeechBrain VoxLingua107, CPU)...")
        try:
            lid = SpeechBrainVoxLingua107LID(device="cpu")
            await asyncio.to_thread(lid.load)
            _lid = lid
        except Exception as e:  # noqa: BLE001 - degrade LOUDLY to Bengali-only, never guess a language
            logger.error("language ID unavailable (%s) -- serving Bengali ONLY until fixed", e)
            active = ["bn"]
    _languages_active = tuple(active)
    logger.info("active languages: %s", ",".join(_languages_active))

    logger.info("loading Silero VAD...")
    _turn_detector = await asyncio.to_thread(TurnDetector, tail_guard_s=TURN_TAIL_GUARD_S)
    _tools = ClinicToolsClient(CLINIC_API_BASE)
    _tts = TTSClient()
    _tts_router = TTSRouter({lang: _tts.for_language(lang) for lang in SUPPORTED_LANGUAGES})
    _intent_cache = SemanticCache()

    _admission = AdmissionController(
        max_calls=ADMISSION_MAX_CALLS,
        latency_shed_p95_s=ADMISSION_SHED_P95_S,
        bypass_file=ADMISSION_BYPASS_FILE or None,
    )
    probe_client = _httpx.AsyncClient()
    probes = {
        "tts": http_probe(probe_client, TTS_HEALTH_URL),
        "clinic-api": http_probe(probe_client, f"{CLINIC_API_BASE}/api/health"),
        "ollama": http_probe(probe_client, OLLAMA_HEALTH_URL),
    }
    if "en" in _languages_active:
        probes["english-asr"] = http_probe(probe_client, f"{ENGLISH_ASR_URL}/health")
    _health = HealthMonitor(_admission, probes)
    _health.probe_client = probe_client
    _health_task = asyncio.create_task(_health.run_forever())

    # Pull bge-m3 into VRAM before the first caller needs it. Cold-loading
    # it inside a live turn measured past the client's patience AND past
    # the embed timeout, which silently degraded the cache to exact-match
    # only for the opening minutes of the process -- healthy-looking logs,
    # zero semantic hits. OLLAMA_KEEP_ALIVE=-1 keeps it resident after.
    try:
        await asyncio.to_thread(_embed_probe, "warmup")
        logger.info("embedding model warm")
    except Exception as e:  # noqa: BLE001 - cache is optional, the call is not
        logger.warning("embedding warmup failed, cache starts L1-only: %s", e)

    # Load Qwen into VRAM now. Cold, it measured 74 s for its first intent --
    # that is the first caller's wait, and it would also poison the latency
    # window admission control sheds load on.
    await _warm_intent_model()

    # Load the 74-row catalogue once so the fast path can identify a test
    # or doctor locally. Optional: if the clinic API is not up yet, every
    # turn simply goes to the LLM, which is the behaviour that existed
    # before this path did.
    _fast_path = await _load_fast_path()

    logger.info("prewarming TTS...")
    lines = prewarm_lines()
    await _tts.prewarm({lang: lines[lang] + [thanks_for(lang)] for lang in _languages_active})
    await _warm_speech_models()

    global _READY_AT, _answer_warm_task
    _READY_AT = time.monotonic()
    _answer_warm_task = asyncio.create_task(_answer_warm_loop())  # background: startup does not wait for it
    logger.info("startup complete -- ready for calls (%.1fs from process start)", _READY_AT - _PROCESS_STARTED_AT)


@app.on_event("shutdown")
async def _shutdown():
    if _answer_warm_task is not None:
        _answer_warm_task.cancel()
    if _health_task:
        _health_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _health_task
    if _health is not None:
        await _health.probe_client.aclose()
    if _asr_router is not None:
        with contextlib.suppress(Exception):
            await _asr_router.engine_for("en").aclose()
    if _tools:
        await _tools.aclose()
    if _tts:
        await _tts.aclose()


@app.get("/api/health")
async def health():
    # KCD-467: "instances receive traffic only once every model answers
    # a probe" -- ASGI/uvicorn already refuses connections until
    # _startup() (which awaits every model load) returns, so `ready`
    # here is never False for a request that could reach this handler
    # at all; it exists so an orchestrator polling this URL through a
    # proxy that itself started early sees an honest, explicit signal
    # rather than inferring readiness from a 200 alone. `startup_duration_s`
    # is the "measured and bounded" half of the acceptance criterion --
    # the actual bound (a deploy pipeline gating traffic on this field
    # rather than on the process merely existing) is an infra concern
    # outside this repository.
    return {
        "status": "ok",
        "ready": _READY_AT is not None,
        "startup_duration_s": round(_READY_AT - _PROCESS_STARTED_AT, 1) if _READY_AT else None,
        "asr_loaded": _asr is not None,
        "clinic_api_base": CLINIC_API_BASE,
        "languages_active": list(_languages_active),
        "lid_loaded": _lid is not None,
        "admission_open": bool(_admission and _admission.snapshot()["open"]),
    }


@app.get("/api/admission")
async def admission_status():
    """Read-only: safe to leave unauthenticated behind the private network."""
    return _admission.snapshot() if _admission else {"open": False, "closed_reason": "starting"}


class _BypassRequest(BaseModel):
    enabled: bool


@app.post("/api/admission/bypass")
async def admission_bypass(req: _BypassRequest, x_admin_token: str = Header(default="")):
    """The audited AI kill-switch: enabled=true sends every NEW call to the
    human contact centre without a deploy. Disabled entirely (503) unless
    ADMISSION_ADMIN_TOKEN is set -- an unauthenticated switch that can take
    the whole service offline is worse than none."""
    if not ADMISSION_ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="bypass endpoint disabled: no admin token configured")
    if not secrets.compare_digest(x_admin_token, ADMISSION_ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="bad admin token")
    _admission.set_bypass(req.enabled, actor="http")
    return _admission.snapshot()


@app.get("/api/stats")
async def stats():
    """Cache effectiveness, for tuning the similarity threshold against
    real traffic rather than against my assumptions about it."""
    return {
        "fast_path": _fast_path.snapshot() if _fast_path else None,
        "answer_warmup": _answers_warm,
        "intent_cache": _intent_cache.snapshot() if _intent_cache else None,
        "tts_cache": _tts.snapshot() if _tts else None,
        "admission": _admission.snapshot() if _admission else None,
        "insufficient_information": insufficient_information.snapshot(),
        "code_switch_buckets": code_switch_buckets.snapshot(),
        "channel_quality_buckets": channel_quality_buckets.snapshot(),
        "detector_budget": detector_budget_snapshot(),
        "turn_latency_by_language": turn_latency_by_language.snapshot(),
        "barge_in_interrupts": barge_in_interrupts.snapshot(),
        "reask_outcomes": reask_outcomes.snapshot(),
        "audio_issue_buckets": audio_issue_buckets.snapshot(),
        "golden_buckets": golden_buckets.snapshot(),
        "apology_events": apology_events.snapshot(),
        "language_mismatches": language_mismatches.snapshot(),
        "senior_detections": senior_detections.snapshot(),
    }


def _wav_duration_s(wav_bytes: bytes) -> float:
    try:
        with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001 - a fallback clip may not be canonical WAV
        return 5.0


def _classify_channel_from_wav_path(path: str) -> str:
    """KCD-075's sync (CPU-bound) half -- run through
    agent.detector_budget.run_within_budget, never called directly, so a
    pathological clip cannot delay the turn past its budget slice."""
    with contextlib.closing(wave.open(path, "rb")) as w:
        raw = w.readframes(w.getnframes())
        sample_rate = w.getframerate()
        sampwidth = w.getsampwidth()
    if sampwidth != 2:
        # This pipeline only ever produces 16-bit PCM clips (torchaudio.save's
        # default); anything else is unexpected input this classifier was not
        # built to read faithfully, so default to the least alarming label
        # rather than mis-parse it into a bogus one.
        return CHANNEL_CLEAN_16K
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return classify_channel(samples, sample_rate)


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with contextlib.closing(wave.open(path, "rb")) as w:
        raw, sr, width, channels = w.readframes(w.getnframes()), w.getframerate(), w.getsampwidth(), w.getnchannels()
    if width != 2:
        raise ValueError(f"unsupported sample width {width}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return (x.reshape(-1, channels).mean(axis=1) if channels > 1 else x), sr


def _analyze_utterance_from_wav_path(path: str) -> dict:
    """agent/audio_quality.assess (why this clip is hard to hear) plus
    agent/senior_voice.estimate (does the speaker sound older), on one
    read of the clip. Sync and CPU-bound: only ever called through
    detector_budget.run_within_budget."""
    samples, sr = _read_wav_mono(path)
    level_dbfs, f0_hz = level_and_pitch(samples, sr)
    return {
        "audio": assess_audio(samples, sr),
        "senior": estimate_senior(samples, sr),
        "near_end": (level_dbfs, f0_hz),
        "voice": voice_embedding(samples, sr),
    }


def _enhance_wav_to_path(path: str) -> str | None:
    """A noise-filtered, level-corrected copy of the clip for a retry, or
    None if it cannot be produced."""
    try:
        samples, sr = _read_wav_mono(path)
        out, _report = condition_audio(samples, sr, suppress="always")
        enhanced = path + ".enh.wav"
        with contextlib.closing(wave.open(enhanced, "wb")) as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        return enhanced
    except Exception as e:  # noqa: BLE001 - enhancement is an optional retry, never a failure
        logger.warning("enhance failed: %s", e)
        return None


def _attend_wav_to_path(path: str, level_dbfs: float, f0_hz: float) -> str | None:
    """A copy of the clip with the room attenuated (agent/attention.py), or None if nothing was
    changed or it could not be produced. Never raises."""
    try:
        samples, sr = _read_wav_mono(path)
        out, report = attend(samples, level_dbfs, f0_hz, sr)
        if not report.applied:
            return None
        cleaned = path + ".att.wav"
        with contextlib.closing(wave.open(cleaned, "wb")) as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        golden_buckets.record("attention", "applied", "any")
        return cleaned
    except Exception as e:  # noqa: BLE001 - attention is an optional cleanup, never a failure
        logger.warning("attention failed: %s", e)
        return None


def _condition_wav_to_path(path: str, mode: str) -> str | None:
    """A conditioned COPY of the clip for the recogniser (KCD-055/057), or None. The
    original is left alone: audio analysis, senior and near-end judgements are made on
    what the caller actually sounded like, not on what was done to help the ASR."""
    try:
        samples, sr = _read_wav_mono(path)
        out, _report = condition_audio(samples, sr, suppress="never" if mode == "level" else "auto")
        conditioned = path + ".cond.wav"
        with contextlib.closing(wave.open(conditioned, "wb")) as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        return conditioned
    except Exception as e:  # noqa: BLE001 - conditioning is an enhancement, never a failure
        logger.warning("input conditioning failed: %s", e)
        return None


class CallSession:
    """One PCM buffer for the ENTIRE call, and a marker for how much of it
    has been consumed.

    The continuous-buffer design is inherited from the WebM version, where
    it was forced: MediaRecorder puts the container header only in the
    first chunk, so resetting the buffer mid-call produced audio that
    could never be decoded again. That constraint is GONE here -- raw PCM
    has no header and any byte range is independently valid.

    It is kept anyway, because the second reason for it was always the
    better one: `processed_until_s` gives every turn an absolute,
    monotonic position on one call-long timeline. Silero's segment
    boundaries move as more audio arrives, so a turn detector run against
    a buffer that keeps restarting drops any utterance straddling the
    seam. Each poll therefore looks only at the UNPROCESSED TAIL.

    What PCM changes is the cost and the accuracy of that lookup: the tail
    is a slice rather than a full re-decode, and sample index maps to
    wall-clock exactly, so the timeline cannot drift.
    """

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.call_id = uuid.uuid4().hex[:8]
        self.tmpdir = tempfile.mkdtemp(prefix=f"kcd_call_{self.call_id}_")
        self.last_activity = time.time()
        self.dispatch_lock = asyncio.Lock()
        self.processed_until_s = 0.0
        self.utt_seq = 0
        self.last_heartbeat = time.time()
        self.audio = PcmCallBuffer()
        self.declared_rate: int | None = None
        if AEC_BARGE_IN:
            # KCD-051/052: echo cancellation + acoustic barge-in. Raw PCM only.
            self.duplex = FullDuplexProcessor(sr=SAMPLE_RATE)

        # Starts True: the greeting goes out before the caller has said
        # anything, so the gate must already be closed when the first poll
        # tick runs, not opened a moment later by _speak().
        self.gate = PlaybackGate(PLAYBACK_GUARD_S)

        # KCD-049/048: when the poll loop next wakes, and the speculative
        # transcript state for the candidate turn end currently in view.
        self.next_wake_s = POLL_INTERVAL_S
        self.spec_end_abs: float | None = None
        self.spec_verdict: str | None = None
        self.spec_count = 0

        # KCD-052/163: an interrupt bumps speak_epoch; the turn in flight keeps
        # the epoch it started under, so everything it has yet to say is
        # discarded rather than played over the caller.
        self.speak_epoch = 0
        self.turn_epoch = 0
        # KCD-051: set by the PCM variant when AEC_BARGE_IN is on.
        # (the PCM variant assigns it earlier in __init__ when AEC_BARGE_IN is on)
        self.duplex: FullDuplexProcessor | None = getattr(self, "duplex", None)

        # `lang` is the call's REPLY language -- sticky by design (KCD-lay-terms, a live call 2026-09-28): a caller
        # who started in Bengali and later says one English or Hindi word/name is not asking to be answered in a
        # different language, and re-flipping mid-call, re-playing the disclosure notice each time, was
        # disorienting on a real call. Recognition (which ASR engine transcribes THIS turn) still follows
        # per-utterance LID every turn, via ASRLanguageRouter/_route_and_transcribe -- only the REPLY language is
        # locked. It changes only for: the first turn (nothing is locked yet), an explicit request ("speak in
        # Hindi" -- agent/language_switch.py), or a detected different voice after identity verification
        # (agent/speaker_change.py revokes verification on a voice change; see _check_speaker_change) -- see
        # _resolve_reply_language below, which is the one place this decision is made.
        self.lang = "bn"
        self.lang_locked = False
        self.lang_router = ASRLanguageRouter()
        self.turn_started_at: float | None = None
        self.admission = None

        # Epic E26: the in-progress booking/reschedule/cancel/add-test flow
        # for this call, if any -- see agent/booking_flow.py. One at a time;
        # a caller starting a second booking mid-flow replaces it (their
        # most recent words win, same principle as merge_slots' overwrite
        # rule). No real telephony CallerID exists yet (see HANDOVER.md /
        # CLAUDE.md's "STILL NOT BUILT" notes), so there is no independent
        # "number this call came from" distinct from whatever contact phone
        # the caller states -- booking_service's proxy-authorisation model
        # is real, but is only as strong as a phone-based system without
        # SIP CallerID can be until that lands.
        self.booking: BookingState | None = None

        # KCD-395/KCD-396: what the last answered enquiry turn(s) were
        # about (agent/enquiry_followup.EnquiryEntity), so a bare
        # elliptical follow-up ("and the sample for that?") can resolve
        # without asking the caller to repeat the test/doctor name.
        # Booking turns neither read nor write this -- it is scoped to
        # the stateless enquiry intents only.
        self.last_enquiry_entities: list = []

        # KCD-061: the single, versioned carrier of caller signals
        # (Blueprint 4.5) -- language and confirmation_required are kept
        # current every turn (see _dispatch_turn); every other field
        # stays at its neutral default until a real caller-state
        # detector exists to drive it (see agent/call_state.py's own
        # docstring).
        self.call_state: CallState = new_call_state()

        # Call intelligence (KCD-084/149/155): the empathetic re-ask tracker,
        # the older-caller evidence accumulator, and bookkeeping so an
        # acknowledgement is spoken once per state entry, not every turn.
        self.reask = ReaskTracker()
        self.senior_evidence = SeniorEvidence()
        self.acknowledged_state: str | None = None
        self.senior_lookup_done = False

        # KCD-513/514: acknowledgements are suppressed on repeat turns; KCD-155's distress
        # acknowledgement, if spoken this turn, already opened it.
        self.acks = AckTracker(mode=ACK_MODE)
        self.turn_acknowledged = False
        # The agent's last utterance asked something (it contained a question mark), and whether THIS turn is the
        # caller's answer to it: only an answer to our question is thanked ("Thank you."), never a question of theirs.
        self.awaiting_answer = False
        self.answering = False
        # Silence handling (SILENCE_PROMPT_S): when the quiet started, how many times we have asked, and whether the
        # caller's next words answer "anything else?".
        self.silence_since: float | None = None
        self.silence_prompts = 0
        self.awaiting_close_answer = False
        self.abuse_count = 0  # abusive turns so far this call (agent/abuse.py)
        # The implicit happiness score (agent/call_score.py): counters updated as the call goes, scored once at hang-up.
        self.signals = CallSignals()
        self.reply_noted = False  # this turn's time-to-first-audio has been counted
        self.speaking_filler = False  # the "achha bolchi" filler is being spoken: not the answer
        self.lang_history: list[str] = []  # the language of each accepted turn, to see a flip that is flipped back
        # KCD-353: the disclosure was spoken in the greeting (Bengali); once more, in the
        # caller's own language, the first time it differs.
        self.disclosed_langs: set[str] = {"bn"}
        # KCD-053: who this call's near-end speaker is (level, pitch), to tell them from a
        # bystander. KCD-054/495: what we actually know about who is on the call, and the
        # voice reference that lets a change of speaker revoke a verification.
        self.near_end = NearEndProfile()
        self.identity = IdentityState()
        self.speaker = SpeakerChangeDetector()

        # KCD-501: the call's record, written as the call goes and closed at hang-up.
        self.recorder: CallRecorder | None = None
        self.outcome = "completed"
        self.turn_count = 0
        # KCD-494/495/498: a question about the caller's own history is a small state machine --
        # find the record (an OPEN question when the number is shared), then, only if a
        # verification has passed, answer from retrieved fields.
        self.history_state: str | None = None  # None | need_phone | need_name | need_test
        self.pending_history = None  # the HistoryQuestion being answered
        self.history_attempts = 0
        self.awaiting_handoff_offer = False
        # Ported story "Caller asks whether their result is dangerous" (agent/clinical_safety.py): the
        # reply offers a clinician and the caller answers on the NEXT turn, the same shape as
        # awaiting_handoff_offer just above. Its own flag rather than reusing that one, because that one
        # is answered by _continue_history_flow() and hands off with the reason "history_requested_staff"
        # -- a clinical question is not a history request, and a shared flag would file it as one.
        self.awaiting_clinician_offer = False
        # Which story made that offer, so the hand-off is filed under the right reason. Defaults to the
        # clinical one because that is the story the flag was introduced for; the unverifiable-claim guard
        # sets its own. Kept beside the flag rather than becoming a third awaiting_* field.
        self.clinician_offer_reason = "clinical_interpretation"
        # Ported story "Caller asks to be called back": set when a callback was asked for and no number is
        # known yet, so the NEXT turn is read for digits (agent/spoken_codes.parse_phone).
        self.awaiting_callback_phone = False
        # A factual lookup held back until the caller confirms what the recogniser (with nothing to
        # check it against) heard: agent/entity_confirmation.py.
        self.pending_entity: entity_text.PendingEntity | None = None
        # KCD-495/497: the security questions in progress, and the senior-care planner (KCD-512).
        self.security: SecurityCheck | None = None
        self.kindness = KindnessPlanner()
        # KCD-496: an unfinished booking from an earlier call, offered after verification.
        self.pending_draft: dict | None = None
        self.awaiting_draft_answer = False
        # KCD-104: a booking set aside while the caller did a different task, and whether we are waiting for their
        # yes or no to "shall I go back to it".
        self.suspended = None
        self.awaiting_resume = False
        self.pending_enquiry = None  # (intent, turn) when we just asked "which test / which doctor"
        self.last_enquiry_turn = 0  # turn_count when the last test/doctor answer was given (agent/enquiry_followup.py)
        self.no_at_confirm = False  # the caller said "no, ..." at the confirmation step and it was not a bare no
        # KCD-382b: the registered/booked-patient preparation flow. Asked at most once per call
        # (test_prep_registration_asked); the two "what are we waiting for next" flags are answered
        # deterministically, the same shape awaiting_callback_phone above already uses, never by the model.
        self.test_prep_registration_asked = False
        self.awaiting_test_prep_registration_answer = False
        self.awaiting_test_prep_booking_phone = False
        # The test name(s) last resolved for PREPARATION in this call -- named directly and merged (KCD-382),
        # or retrieved from a registered/booking lookup (KCD-382b). NOT general conversation memory: only ever
        # a resolved test-name set, read back by a same-call follow-up that names no test, within the same
        # recency window agent/enquiry_followup.py already uses for a single test name (FOLLOWUP_MAX_GAP).
        self.last_resolved_prep_tests: tuple[str, ...] | None = None
        self.last_resolved_prep_turn = 0
        # KCD-499: stored preferences, offered after the answer and applied only if the caller says yes.
        self.pending_pref = None
        self.awaiting_pref_answer = False
        self.confirmed_preferences: dict = {}
        # KCD-047: this caller's own pause rhythm, and the evidence for "that cut was too early".
        self.pause = PauseProfile()
        self.last_commit_end_abs: float | None = None
        self.last_confirm_s = 0.0
        self.agent_spoke_since_commit = False
        self.commit_checked = True

    @property
    def policy(self):
        """Appendix C's delivery parameters for the caller's CURRENT state
        (agent/speech_policy.py) -- derived, never stored, so it can never
        disagree with call_state."""
        return derive_policy(self.call_state.caller_state, self.call_state.senior)

    # The gate itself lives in agent/playback_gate.py (KCD-050) so its rules can be
    # tested without a pod; these keep the names the rest of this file uses.
    @property
    def agent_speaking(self) -> bool:
        return self.gate.speaking

    @property
    def speak_deadline(self) -> float:
        return self.gate.deadline

    @property
    def resync_pending(self) -> bool:
        return self.gate.resync_pending

    @resync_pending.setter
    def resync_pending(self, value: bool) -> None:
        self.gate.resync_pending = value

    def hold_gate_for(self, audio_duration_s: float):
        """Called before each reply goes out. Extends rather than replaces
        the deadline: replies queue on the client, so a second clip starts
        playing only after the first finishes."""
        self.gate.hold(audio_duration_s)

    def release_gate(self):
        """Playback is over. Don't touch processed_until_s here -- the poll
        loop owns the decoded buffer and does the resync on its next tick."""
        self.gate.release()

    def register_agent_audio(self, wav_bytes: bytes, voice: str | None = None) -> None:
        """KCD-051: hand the clip about to be sent to the echo canceller as its
        far-end reference. A no-op unless the PCM variant enabled AEC."""
        if self.duplex is None or not wav_bytes:
            return
        try:
            self.duplex.place_reference(wav_to_pcm16k(wav_bytes), voice=voice)
        except Exception as e:  # noqa: BLE001 - AEC is an enhancement, never a reason to drop a reply
            logger.warning("[%s] could not register agent audio for echo cancellation: %s", self.call_id, e)

    def take_wake_delay(self) -> float:
        """Seconds to sleep before the next poll; the schedule resets to the
        idle cadence unless the last decision said otherwise."""
        delay, self.next_wake_s = self.next_wake_s, POLL_INTERVAL_S
        return delay

    def reset_speculation(self) -> None:
        self.spec_end_abs, self.spec_verdict, self.spec_count = None, None, 0

    async def append(self, chunk: bytes):
        self.last_activity = time.time()
        if self.duplex is None:
            self.audio.append(chunk)
            return
        # The buffer the turn detector reads holds the ECHO-CANCELLED signal,
        # so the agent's own voice is not what it hears (agent/full_duplex.py).
        x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        result = self.duplex.process(x)
        if result.cleaned.size:
            self.audio.append((np.clip(result.cleaned, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        if result.barge_in is not None:
            await _handle_barge_in(self, result.barge_in)

    async def send_json(self, sender: str, text: str):
        await self.ws.send_text(json.dumps({"sender": sender, "text": text}, ensure_ascii=False))

    async def send_audio(self, wav_bytes: bytes):
        if wav_bytes:
            await self.ws.send_bytes(wav_bytes)

    def cleanup(self):
        import shutil

        with contextlib.suppress(OSError):
            shutil.rmtree(self.tmpdir, ignore_errors=True)


async def _synthesize_one_clause(session: CallSession, clause: str, lang: str, fallback_reason: str | None) -> bytes:
    # KCD-149/157: the caller's policy rate, and slower still for a price,
    # phone number or reference (the two multiply).
    speed = effective_rate(session.policy, FIGURE_SPEECH_SPEED if contains_critical_figure(clause) else None)
    try:
        return await _tts_router.synthesize(lang, clause, speed=speed)
    except UnspeakableTextError as e:
        # KCD-455: a production occurrence is an alert, not routine
        # noise -- distinct log level and reason from an ordinary TTS
        # infra failure below, even though both fall back to the same
        # pre-recorded apology.
        logger.error("[%s] reply blocked, unspeakable spans %s in %r", session.call_id, e.spans, clause)
        return _tts.fallback_audio(fallback_reason or "tts_failure", lang)
    except Exception as e:  # noqa: BLE001 - TTS is the last mile, must not raise past here
        logger.warning("[%s] TTS failed (%s) -- using fallback audio", session.call_id, e)
        return _tts.fallback_audio(fallback_reason or "tts_failure", lang)


async def _speak(session: CallSession, text: str, lang: str | None = None, fallback_reason: str | None = None) -> float:
    """Speak `text` in `lang` (default: the call's current language).
    Returns the total audio duration in seconds.

    KCD-456: automatically slower for a reply carrying a price, phone
    number or reference ID (agent.speech_norm.contains_critical_figure)
    -- decided here, once, rather than at every call site that happens to
    produce such a reply, so no future reply_templates.py addition can
    forget to ask for it.

    KCD-462: a long reply (agent.clause_split.split_into_clauses) is
    synthesized and sent ONE CLAUSE AT A TIME instead of as one TTS call
    for the whole text -- the first clause reaches the caller as soon as
    its own (much shorter) synthesis finishes, not after the entire
    reply renders. A short reply (the common case) comes back as a
    single-item list and behaves exactly as before -- one TTS call, one
    clip. session.hold_gate_for already extends cumulatively for
    multiple clips in a row (see its own docstring), which is what a
    correction-acknowledgement-then-readback turn already relies on
    elsewhere in this file, so streaming several clips here is the same,
    already-proven pattern, not a new one."""
    lang = lang or session.lang
    if fallback_reason in ("llm_failure", "tool_failure"):
        _note(session, "system_failures")
    session.agent_spoke_since_commit = True
    if session.speak_epoch != session.turn_epoch:
        # The caller interrupted the turn this reply belongs to: it is stale, and
        # playing it over them is exactly what barge-in exists to stop.
        logger.info("[%s] reply discarded after interrupt", session.call_id)
        return 0.0
    if "?" in text or "\uff1f" in text:
        session.awaiting_answer = True
    policy = session.policy
    if not policy.emergency and policy.questions_per_turn >= 1:
        # KCD-084/149: one question at a time. Only a SURPLUS question is
        # ever dropped, never a statement, so no fact is lost.
        text = limit_questions(text, policy.questions_per_turn)
    if language_mismatch(lang, text):
        # KCD-087: a reply in a different language from the caller's is a
        # defect, made visible rather than discovered on a live call.
        language_mismatches.record("reply", "mismatch", lang)
        logger.error("[%s] reply language mismatch (expected %s): %r", session.call_id, lang, text[:60])
    # KCD-514: at most one apology per reply. Templates are written to that rule; this catches
    # the ones assembled at run time (a secondary answer appended to a primary one).
    text, stacked = enforce_single_apology(text, lang)
    if stacked:
        apology_events.record("stacked_removed", "reply", lang)
    await session.send_json("AI", text)
    clauses = split_into_clauses(text) or [text]

    total_duration = 0.0
    for i, clause in enumerate(clauses):
        if session.speak_epoch != session.turn_epoch:
            break  # interrupted mid-reply: the rest is discarded, not sent
        # A live call (2026-09-28): the filler ("please hold") is itself spoken through this same
        # function while the REAL answer is still being resolved (see _await_with_filler). Marking
        # and logging it here used to consume and clear session.marks (_mark is first-occurrence-only,
        # _log_timing clears the list) before the real answer's own "reply"/"tts"/"send" marks could
        # ever be recorded -- so on exactly the slow turns worth measuring, the turn-timing log went
        # dark right after the filler and the actual delay (the thing a caller waited through) was
        # never logged at all. The filler's own latency is not interesting (it is pre-cached, near-
        # instant); skip marking it so the marks recorded so far (lid/prep/asr) survive to be logged
        # against the real reply that follows.
        if not session.speaking_filler:
            _mark(session, "reply")
        wav = await _synthesize_one_clause(session, clause, lang, fallback_reason)
        if not session.speaking_filler:
            _mark(session, "tts")
        if session.speak_epoch != session.turn_epoch:
            break  # interrupted while this clause was being synthesised

        # Close the gate BEFORE the bytes leave, never after: the client
        # can start playing the moment they land, and a poll tick that
        # slips in between send and gate is exactly the echo this
        # prevents.
        duration = _wav_duration_s(wav)
        session.hold_gate_for(duration)
        session.register_agent_audio(wav, voice=lang)
        await session.send_audio(wav)
        total_duration += duration
        if i == 0:
            if not session.speaking_filler:
                _mark(session, "send")
                _log_timing(session)
            if not session.reply_noted and not session.speaking_filler and session.turn_started_at is not None:
                session.reply_noted = True
                session.signals.note_reply(time.monotonic() - session.turn_started_at)

        # "Time to first audio" (admission.py's own docstring) means the
        # FIRST clause, not the last -- recorded once, right after it is
        # sent, so streaming a long reply is measured (and shows up as
        # faster) rather than averaged against the clauses still to come.
        if i == 0 and session.turn_started_at is not None and _admission is not None:
            turn_latency_s = time.monotonic() - session.turn_started_at
            _admission.record_turn_latency(turn_latency_s)
            # KCD-469: the SAME measurement, also broken down per
            # language -- admission's own p50/p95 stay a single
            # aggregate signal by design (see its module docstring), so
            # a language-specific regression could hide inside a
            # healthy-looking aggregate without this.
            turn_latency_by_language.record(lang, turn_latency_s)
            session.turn_started_at = None

    return total_duration


async def _queue_callback_request(session: CallSession, phone: str, lang: str) -> None:
    """Record a caller's own request to be rung back, then say so.

    Ported story "Caller asks to be called back". The reason is built by
    agent/callback_request.build_callback_reason(), which prefers the caller's
    own words and otherwise records nothing rather than inventing a generic
    "general enquiry". The call summary rides along exactly as it does for a
    hand-off, so the colleague who makes the call has the context beside the
    number.

    A storage failure never costs the caller their answer: it is suppressed the
    same way every other durable write on a spoken path here is, and the spoken
    line only claims the number was noted, which is true either way.
    """
    reason = build_callback_reason(None) or "caller_requested_callback"
    summary = build_call_summary("callback_requested", lang=lang, phone=phone, call_id=session.call_id)
    with contextlib.suppress(ToolCallError):
        await _tools.request_callback(phone, session.call_id, "asap", reason, summary)
    _rec(session, "escalation", "callback_requested")
    await _speak(session, phrase("callback_noted", lang), lang)


async def _handoff_to_human(session: CallSession, reason: str, languages: tuple[str, ...] | None = None):
    """Tell the caller and the telephony bridge that a person is taking over.

    The `handoff_human` frame is the machine-readable half: the SBC/bridge
    that owns the SIP leg acts on it (this repo has no SIP transfer). The
    spoken notice is the human-readable half, said in every language when
    the caller's is not yet known.

    Also QUEUES a callback (clinic-api/enquiry_service.request_callback) whenever a phone is already known
    (session.identity.phone) -- this is what makes a handoff (including the reask policy's own 3rd-failure handoff,
    agent/reask_policy.DEFAULT_MAX_REASKS) an actual actionable queue entry for a human, not only a spoken notice and
    a websocket frame nothing durable is kept of. Without a phone there is nothing to call back on (no automatic
    caller-ID exists in this stack), so nothing is queued -- silently, never a spoken complaint about it.
    """
    logger.warning("[%s] handoff to human: %s", session.call_id, reason)
    _note(
        session,
        "emergency"
        if reason == "emergency"
        else "asked_for_person"
        if reason in ("caller_requested", "history_requested_staff")
        else "system_handoff",
    )
    session.outcome = "handed_off"
    _rec(session, "escalation", reason)
    phone = session.identity.phone
    if phone:
        summary = build_call_summary(f"handoff:{reason}", lang=session.lang, phone=phone, call_id=session.call_id)
        with contextlib.suppress(ToolCallError):
            await _tools.request_callback(phone, session.call_id, "asap", f"handoff:{reason}", summary)
    with contextlib.suppress(Exception):
        await session.ws.send_text(json.dumps({"type": "handoff_human", "reason": reason}))
    total = 0.0
    for lang in languages or (session.lang,):
        with contextlib.suppress(Exception):
            total += await _speak(session, phrase("handoff", lang), lang)
    # Let the notice finish playing before the socket closes under it.
    await asyncio.sleep(min(total + 0.5, 20.0))
    with contextlib.suppress(Exception):
        await session.ws.close()


async def _reask_or_handoff(
    session: CallSession,
    decision,
    lang: str,
    fallback_reason: str | None = None,
    languages: tuple[str, ...] | None = None,
) -> None:
    """Ask again, kindly, before routing to a person (agent/reask_policy.py).
    A second consecutive failure also marks the caller "confused", which
    slows the agent and lowers its escalation threshold for the rest of the
    call (Appendix C)."""
    reask_outcomes.record(decision.reason or "unclear", decision.action, lang)
    _note(session, "reasks")
    if decision.mark_confused:
        apply_caller_state(session.call_state, caller_state="confused")
    # `languages` is passed when the caller's language could not be identified:
    # the re-ask and the hand-off notice are then said in EVERY active language,
    # not in whichever one the call happened to default to.
    langs = languages or (lang,)
    if decision.action == "handoff":
        for lg in langs:
            await _speak(session, phrase("reask_final", lg), lg)
        await _handoff_to_human(session, f"unintelligible:{decision.reason}", langs)
        return
    for lg in langs:
        text = phrase(decision.phrase_key, lg)
        patience = session.kindness.patience(lg)
        await _speak(session, f"{text} {patience}" if patience else text, lg, fallback_reason=fallback_reason)


async def _retry_on_enhanced_audio(
    session: CallSession, wav_path: str, lang: str | None, result, audio_issues: list[str], duration_s: float | None
):
    """A failed or doubtful turn on a noisy or faint line gets one more try
    on a noise-filtered, level-corrected copy of the clip. Only when the
    first attempt was actually poor -- filtering can hurt a recogniser that
    coped fine -- and never for cross-talk, which filtering cannot fix."""
    if lang is None or not ({"noisy", "too_quiet"} & set(audio_issues)):
        return result
    poor = (
        not result.text.strip()
        or transcript_problem(result.text, lang, duration_s, result.decoder_agreement)
        or is_low_confidence(result.decoder_agreement)
    )
    if not poor:
        return result
    enhanced = await asyncio.to_thread(_enhance_wav_to_path, wav_path)
    if not enhanced:
        return result
    try:
        retry = await _asr_router.transcribe(lang, enhanced)
    except Exception as e:  # noqa: BLE001 - the original result stands
        logger.warning("[%s] enhanced-audio retry failed: %s", session.call_id, e)
        return result
    finally:
        with contextlib.suppress(OSError):
            os.remove(enhanced)
    if retry.text.strip() and (not result.text.strip() or retry.decoder_agreement > result.decoder_agreement):
        logger.info("[%s] enhanced-audio retry improved the transcript", session.call_id)
        return retry
    return result


def _update_caller_state(session: CallSession, text: str, lang: str, analysis: dict | None) -> None:
    """KCD-084: the older-caller decision. An explicit request or self-
    description decides at once; the acoustic estimate needs evidence over
    several clips (agent/senior_voice.SeniorEvidence). Once senior, sticky
    for the call."""
    ev = session.senior_evidence
    cue = explicit_senior_cue(text, lang)
    flipped = ev.note_explicit(cue) if cue else (ev.add(analysis["senior"]) if analysis else False)
    if flipped and not session.call_state.senior:
        apply_caller_state(session.call_state, senior=True)
        senior_detections.record(ev.reason or "unknown", "call", lang)
        logger.info("[%s] senior mode on (%s)", session.call_id, ev.reason)
    # Ported story "Caller is angry about a previous experience" (agent/anger.py).
    # Deterministic, before any model call, in every language at once -- the same
    # discipline as agent/abuse.py and agent/human_request.py, and for the same
    # reason: a policy the model is asked to apply is one it can be talked out of.
    #
    # Setting the state is the WHOLE of the wiring, by design. The caller's next
    # spoken line is already decided from here by machinery this file calls two
    # lines after _update_caller_state() returns: apply_caller_state() re-derives
    # the "angry" row of Appendix C (slower delivery, one question per turn, a
    # human offered, escalation threshold lowered), and select_acknowledgement()
    # speaks the single-apology acknowledgement in the caller's own language.
    # Before this port nothing in the repository ever set "angry", so all of that
    # was unreachable; nothing new had to be written for it here.
    #
    # Sticky for the rest of the call, like senior above: a caller who has told us
    # they are angry is not un-angered by one calm turn, and quietly speeding back
    # up would read as the agent having stopped listening. The acknowledgement
    # itself is still spoken only once (main's `acknowledged_state` guard).
    if session.call_state.caller_state != "angry":
        angry, family = anger.detect_anger(text)
        if angry:
            apply_caller_state(session.call_state, caller_state="angry")
            _note(session, "anger_turns")
            logger.info("[%s] caller expressed anger (%s phrase list)", session.call_id, family)


def _note_stated_age(session: CallSession, slots: dict, lang: str) -> None:
    if stated_age_is_senior(slots.get("patient_age"), slots.get("relationship")):
        if session.senior_evidence.note_explicit("stated_age") and not session.call_state.senior:
            apply_caller_state(session.call_state, senior=True)
            senior_detections.record("stated_age", "call", lang)


_ECHOED_FIELDS = ("doctor_name", "date", "time_slot", "patient_name", "phone", "test_name", "new_date", "new_time_slot")


async def _thank(session: CallSession, lang: str) -> None:
    """A bare "Thank you." -- once per turn, and only when this turn is the caller's answer to a question we asked."""
    if not session.answering or getattr(session, "thanked_turn", None) == session.turn_count:
        return
    session.thanked_turn = session.turn_count
    await _speak(session, thanks_for(lang), lang)


async def _thank_for_answer(session: CallSession, st, changed: list[str], prior_slots: dict, lang: str) -> None:
    """Booking flows: thank the caller when they have just given a detail we asked for. Not on the first request
    ("I want an appointment with Dr Sen" answers the greeting, and is not a reply to a question about a detail)."""
    if (prior_slots or st.asked_fields) and any(f not in prior_slots for f in changed):
        await _thank(session, lang)


async def _echo_new_slots(session: CallSession, changed: list[str], prior_slots: dict, slots: dict, lang: str) -> None:
    """Senior mode's "one question, listen, confirm, next question": say
    the value just collected back before the next question. Also the moment
    a phone number first arrives, so a returning older caller is recognised
    from the record without having to ask for slower speech again."""
    newly = [f for f in changed if f not in prior_slots]
    if "phone" in newly:
        await _remember_senior(session, slots.get("phone"), lang)
    if not session.policy.confirm_each_slot:
        return
    for field in newly:
        value = slots.get(field)
        if value and field in _ECHOED_FIELDS:
            await _speak(session, slot_echo(str(value), lang), lang)
            return


async def _remember_senior(session: CallSession, phone: str | None, lang: str) -> None:
    if session.senior_lookup_done or not phone:
        return
    session.senior_lookup_done = True
    try:
        if await _tools.get_patient_senior(phone) and session.senior_evidence.note_explicit("remembered"):
            apply_caller_state(session.call_state, senior=True)
            senior_detections.record("remembered", "call", lang)
    except ToolCallError as e:
        logger.warning("[%s] senior lookup failed: %s", session.call_id, e)


async def _persist_senior(session: CallSession, phone: str | None) -> None:
    """KCD-084: once a booking is made, remember the delivery mode against
    the patient -- a boolean only, never a score, an age or any audio."""
    if not (session.call_state.senior and phone):
        return
    try:
        await _tools.set_patient_senior(phone, True)
    except ToolCallError as e:
        logger.warning("[%s] senior persist failed: %s", session.call_id, e)


_last_messages_error_logged = False


async def _refresh_messages() -> None:
    """Pull the operator-editable wording from the database (KCD-353/500/513). Bounded and silent on
    failure: the built-in text is spoken whenever the API cannot be reached, so an outage never leaves the
    agent without words."""
    global _last_messages_error_logged
    if _tools is None or not agent_messages.stale():
        return
    try:
        agent_messages.load(await asyncio.wait_for(_tools.agent_messages(), 1.5))
        _last_messages_error_logged = False
    except Exception as e:  # noqa: BLE001
        if not _last_messages_error_logged:
            logger.warning("agent messages not refreshed (%s): built-in wording is used", e)
            _last_messages_error_logged = True


def _note(session: CallSession, name: str, n: int = 1) -> None:
    """Count one signal for the call's implicit happiness score. Never raises: a score is never a reason to fail a turn."""
    try:
        session.signals.note(name, n)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] score signal %s failed: %s", session.call_id, name, e)


def _write_satisfaction(session: CallSession) -> None:
    """At hang-up: score the call from its signals and queue the result with the call record. Counts and flags only."""
    st = session.booking
    if st is not None and topic_flow.worth_suspending(st):
        session.signals.left_mid_task = True  # a booking was still half-done when the line dropped
    result = score_call(session.signals)
    logger.info(
        "[%s] call score: %s (%s) %s",
        session.call_id,
        result.score,
        result.band,
        ", ".join(f"{n} {p:+d}" for n, p in result.reasons) or result.reason,
    )
    _rec(session, "satisfaction", result.to_payload(session.signals))


def _rec(session: CallSession, method: str, *args) -> None:
    """Queue one call-record event and start delivering it. Never raises and never waits: a
    record that cannot be written is retried and, at hang-up, reported (agent/call_record.py) --
    it is never a reason to slow or fail a turn."""
    rec = session.recorder
    if rec is None:
        return
    try:
        getattr(rec, method)(*args)
        asyncio.ensure_future(rec.flush())
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] call record event %s failed: %s", session.call_id, method, e)


def _record_action(session: CallSession, name: str, result: dict, slots: dict) -> None:
    if result.get("success"):
        _note(session, "task_completed")
        _rec(session, "action", name, result.get("confirmation_id"))
        _rec(session, "confirmed", slots)


async def _write_call_event(call_id: str, seq: int, kind: str, payload: dict, caller_phone: str | None) -> dict:
    if _tools is None:
        raise ToolCallError("clinic tools not ready")
    return await _tools.write_call_event(call_id, seq, kind, payload, caller_phone)


async def _handle_unverified_write(session: CallSession, lang: str) -> None:
    """KCD-486: a booking/reschedule write whose post-commit verification
    (clinic-api's _verify_appointment_persisted) did not confirm the
    expected values. Speaks a hold notice -- distinct from both a
    confirmation number (which would be an unverified fact) and the
    generic tool_failure apology (which invites a retry, and the write
    likely already happened, so a retry risks a second, duplicate one) --
    then escalates to a human rather than guessing."""
    await _speak(session, phrase("booking_hold_for_verification", lang), lang)
    await _handoff_to_human(session, "unverified_booking_write")


async def _await_with_filler(session: CallSession, awaitable, lang: str, threshold_s: float = FILLER_THRESHOLD_S):
    """KCD-461: thin wrapper over agent.filler.await_with_filler (the
    tested race logic) supplying "speak the pre-warmed please_wait
    phrase" as the timeout callback -- never synthesized fresh
    (TTSClient's cache already holds it from startup prewarm), so it
    costs no synthesis time at exactly the moment the system is already
    running slow."""

    async def _speak_filler():
        session.speaking_filler = True  # the holding phrase is not the answer: do not time it as one
        try:
            await _speak(session, phrase("please_wait", lang), lang)
        finally:
            session.speaking_filler = False

    # The threshold is counted from when the caller stopped speaking, not from when this stage began: recognition
    # already used some of it. It never drops below FILLER_MIN_WAIT_S, so a semantic-cache hit is not announced.
    started = session.turn_started_at
    if started is not None:
        threshold_s = max(threshold_s - (time.monotonic() - started), FILLER_MIN_WAIT_S)
    return await await_with_filler(awaitable, threshold_s, _speak_filler)


async def _slice_utterance(session: CallSession, start_s: float, end_s: float, seq: int) -> str:
    """Cuts [start_s, end_s+pad] -- both ABSOLUTE call-time offsets -- out
    of the call's decoded WAV into its own small file for ASR."""
    sr = session.audio.sample_rate
    clip = session.audio.slice_tensor(start_s, end_s + UTTERANCE_PAD_S)
    clip_path = os.path.join(session.tmpdir, f"utt{seq}.wav")

    def _write():
        wav = clip.unsqueeze(0)
        out_sr = sr
        if sr != SAMPLE_RATE:
            # Only reachable when the browser refused a 16kHz AudioContext.
            # ASR expects 16k, so convert here rather than letting it
            # silently transcribe pitch-shifted audio.
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            out_sr = SAMPLE_RATE
        torchaudio.save(clip_path, wav, out_sr)

    await asyncio.to_thread(_write)
    return clip_path


# KCD-101: the whole wall-clock a caller may wait for "what do they want" (cache lookup + model, all attempts)
# before the apology is spoken. Equal to the model's own default deadline (agent/llm.py, 12 s, REASONED, not
# measured on real telephony). The holding phrase is spoken once FILLER_THRESHOLD_S has passed without an answer.
INTENT_BUDGET_S = float(os.environ.get("INTENT_TURN_BUDGET_S", "12"))
# The semantic-cache embedding lookup gets this head start on the model; after that both run together.
CACHE_HEAD_START_S = float(os.environ.get("INTENT_CACHE_HEAD_START_S", "0.15"))
# ...and it may never hold the turn for more than this (kept as a ceiling on the head start).
CACHE_LOOKUP_MAX_S = float(os.environ.get("INTENT_CACHE_MAX_S", "1.5"))
# Below this much budget left, asking the model is pointless: go straight to the apology.
MIN_MODEL_BUDGET_S = 0.5


async def _resolve_intent_uncached(session: CallSession, text: str, lang: str, key: str) -> dict:
    """Tier 2 (semantic cache) + tier 3 (LLM), run AFTER fast_path already
    abstained. Split out from _resolve_intent so the filler race below can
    wrap the WHOLE remaining sequence -- see KCD-459 note there."""
    # KCD-101: ONE wall-clock budget covers the cache lookup AND every model attempt. The model call already had
    # its own deadline (KCD-465), but the embedding lookup in front of it did not count against it, so a slow
    # cache plus a slow model could keep a caller waiting past the budget. A slow cache is treated as a miss.
    start = time.monotonic()

    def left() -> float:
        return INTENT_BUDGET_S - (time.monotonic() - start)

    # The cache lookup (an embedding call) and the model run TOGETHER, not one after the other. The lookup gets a
    # short head start; if it has not answered by then the model starts anyway and the lookup keeps going. A hit ends
    # the wait at once and the model's answer is dropped; a miss costs the model turn nothing. Measured on the pod, the
    # serial version spent up to 1.5 s waiting for a lookup that then missed, before the 1.3 s model call even began.
    lookup = asyncio.ensure_future(asyncio.to_thread(_intent_cache.get, key))
    lookup.add_done_callback(lambda f: f.cancelled() or f.exception())  # a failed lookup is a miss, never an error
    try:
        cached, how = await asyncio.wait_for(
            asyncio.shield(lookup), timeout=max(0.02, min(CACHE_HEAD_START_S, CACHE_LOOKUP_MAX_S, left()))
        )
    except TimeoutError:
        cached, how = None, "pending"
    except Exception as e:  # noqa: BLE001 - the semantic cache is optional
        logger.warning("[%s] intent cache lookup failed (%s): treated as a miss", session.call_id, e)
        cached, how = None, "error"
    if cached is not None:
        logger.info("[%s] intent cache %s hit", session.call_id, how)
        return cached

    budget_left = left()
    if budget_left < MIN_MODEL_BUDGET_S:
        raise ExtractionError(f"intent budget of {INTENT_BUDGET_S}s used up before the model was asked")
    # the extractor bounds itself by `budget_left`; the wait_for is the backstop for a thread that does not
    model = asyncio.ensure_future(
        asyncio.wait_for(asyncio.to_thread(extract_intent, text, 2, lang, budget_left), timeout=budget_left + 1.0)
    )
    model.add_done_callback(lambda f: f.cancelled() or f.exception())
    if not lookup.done():
        await asyncio.wait({lookup, model}, return_when=asyncio.FIRST_COMPLETED)
    if lookup.done() and not lookup.cancelled() and lookup.exception() is None and not model.done():
        hit, how = lookup.result()
        if hit is not None:
            model.cancel()  # its thread finishes on its own, bounded by its deadline; the answer is dropped
            logger.info("[%s] intent cache %s hit (while the model was running)", session.call_id, how)
            return hit
    try:
        data, diag = await model
    except TimeoutError:
        raise ExtractionError(f"intent extraction did not complete within the {INTENT_BUDGET_S}s budget") from None
    logger.info(
        "[%s] intent extracted in %.2fs (%d attempt(s))", session.call_id, diag["total_time_s"], diag["attempts"]
    )
    await asyncio.to_thread(_intent_cache.put, key, data)
    return data


async def _resolve_intent(session: CallSession, text: str, lang: str = "bn") -> dict:
    """Semantic cache in front of the LLM. A hit skips Ollama entirely --
    the slowest hop in the turn -- but the clinic lookup that follows still
    runs live, so a cached intent can never serve a stale price."""
    # Tier 1: decide it locally if we can. For a fixed catalogue the
    # entity is a string-matching problem with a 0.32 confidence margin,
    # where the embedding route had 0.03 -- see agent/fast_path.py. This
    # returns None whenever it is not sure, which is the common case for
    # anything except a routine price or availability question. It has a
    # cue table per language (agent/fast_path_cues.py, KCD-095) and abstains
    # for a language it has none for, so the turn goes to the LLM -- safe by
    # construction, just not free.
    await _maybe_reload_fast_path()
    if _fast_path is not None:
        gap = session.turn_count - session.last_enquiry_turn if session.last_enquiry_entities else None
        topic = recent_unique(session.last_enquiry_entities, "test_name", gap)
        hit = await asyncio.to_thread(_fast_path.resolve, text, lang, topic)
        if hit is not None:
            logger.info("[%s] fast path resolved %s (%.2f) -- no LLM call", session.call_id, hit.intent, hit.confidence)
            return hit.as_llm_shape()

    # The cache stores the whole intent object (smalltalk carries reply
    # text in the caller's language), so the language is part of the key.
    key = text if lang == "bn" else f"[{lang}] {text}"

    # KCD-459: the filler race used to wrap ONLY the LLM call, so a slow
    # embedding lookup (agent/semantic_cache.py's L2 tier, a live call to
    # Ollama's embed endpoint) burned into FILLER_THRESHOLD_S's budget
    # silently before the caller ever got the "please hold" filler -- a
    # cold/loaded embed step and a cold LLM call could sum past the
    # threshold with the caller hearing nothing for either half. Wrapping
    # the whole cache-then-LLM sequence means "no more than
    # FILLER_THRESHOLD_S of silence" is an end-to-end guarantee for this
    # tier, not just a promise about the LLM's own share of it.
    return await _await_with_filler(session, _resolve_intent_uncached(session, text, lang, key), lang)


# CodeRabbit-flagged, real bug: a failed hold used to ALWAYS clear
# time_slot, regardless of why hold_slot actually failed ("keep
# doctor/date, re-collect just the time" -- true only for a slot-shaped
# failure). For "doctor_not_found" or a date-shaped failure, the ACTUAL
# problem field was never cleared, so missing_required() still saw
# doctor_name/date as present and the SAME bad hold was retried forever
# on every subsequent turn, having only pointlessly re-asked for a time
# that was never the issue.
_HOLD_FAILURE_FIELD = {
    "doctor_not_found": "doctor_name",
    "doctor_ambiguous": "doctor_name",
    "invalid_date": "date",
    "date_in_past": "date",
    "doctor_not_available_that_day": "date",
}


def _field_to_reclear_on_hold_failure(reason: str | None) -> str:
    return _HOLD_FAILURE_FIELD.get(reason, "time_slot")


# A not-found answer that carries exactly ONE sound-alike suggestion ("did you mean Dr. Mukherjee?") is
# remembered for the turn, so a "yes" runs the lookup with that name -- otherwise the question would lead
# nowhere. Set by the lookups below, read by _finish_enquiry_turn.
_suggested: contextvars.ContextVar[dict | None] = contextvars.ContextVar("suggested", default=None)


def _note_suggestion(intent: str, slot: str, slots: dict, result: dict) -> None:
    names = result.get("did_you_mean") or []
    if not result.get("found") and result.get("needs_confirmation") and not result.get("ambiguous") and len(names) == 1:
        _suggested.set({"intent": intent, "slot": slot, "slots": dict(slots), "name": names[0]})


async def _answer_enquiry_intent(intent: str, slots: dict, lang: str) -> str | None:
    """Tool call + reply-template for one of the stateless enquiry
    intents (agent.enquiry_followup.ENQUIRY_INTENTS). Returns None when
    the intent's required slot is missing -- the caller decides whether
    that means "ask the caller" (the turn's primary question) or
    "silently skip" (a KCD-395 secondary question), never this function.
    The single implementation per intent here is also what a secondary
    question is answered with, so there is exactly one place each
    intent's tool call and reply template are wired together."""
    if intent == "test_rate":
        if not slots.get("test_name"):
            return None
        result = await _tools.get_test_rate(slots["test_name"])
        _note_suggestion(intent, "test_name", slots, result)
        return test_rate_reply(slots, result, lang)
    if intent == "doctor_availability":
        if not slots.get("doctor_name"):
            return None
        result = await _tools.get_doctor_availability(slots["doctor_name"], slots.get("date"))
        _note_suggestion(intent, "doctor_name", slots, result)
        return doctor_availability_reply(slots, result, lang)
    if intent == "test_prep":
        # KCD-382: the model may have put a single test name in "test_names" instead of "test_name" now that
        # "test_prep" is also allowed to fill the plural slot (see agent/llm.py) -- two or more names is handled
        # entirely BEFORE this function is called (main.py's test_prep dispatch branch), so by the time a turn
        # reaches here "test_names" never has more than one entry; this is only the fallback for that one.
        test_name = slots.get("test_name") or next(iter(slots.get("test_names") or []), None)
        if not test_name:
            return None
        result = await _tools.get_test_prep(test_name, lang)
        _note_suggestion(intent, "test_name", slots, result)
        return test_prep_reply({**slots, "test_name": test_name}, result, lang)
    if intent == "clinic_faq":
        if not slots.get("faq_topic"):
            return None
        result = await _tools.get_faq(slots["faq_topic"], lang)
        return clinic_faq_reply(slots, result, lang)
    if intent == "department_query":
        symptom = slots.get("symptom_description")
        if not symptom:
            return None
        result = await _tools.route_department(symptom, lang)
        return department_route_reply(result, symptom, lang)
    return None


async def _handle_multi_test_prep(session: CallSession, test_names: list[str], lang: str) -> None:
    """KCD-382, story "Caller has several tests with conflicting
    preparation": the caller named more than one test in a "test_prep"
    turn (agent/llm.py's "test_names" slot, the same structure "book_test"
    already uses -- see agent/intent_schema.py). Name resolution and the
    strictest-constraint merge both happen in clinic-api (AC1); this
    function only inspects the merged result and either speaks it (AC2,
    agent.reply_templates.merged_test_prep_reply -- one sentence, never a
    paragraph per test) or hands off to a human (AC3), exactly the same
    shape as _handle_unverified_write above: a fixed notice, then
    _handoff_to_human, then return -- never both a spoken instruction and
    a hand-off for the same request, and never a continuation into
    _finish_enquiry_turn's secondary-question/coreference bookkeeping,
    which this story's AC's have no use for.

    `result["escalate_to_human"]` is the ONLY signal this function acts
    on for AC3 -- it is not re-derived here (CLAUDE.md rule 1): it is
    real whenever enquiry_service.merge_prep_instructions() found a
    conflict it could not safely fold into one number OR a resolved test
    whose structured fasting data is still incomplete (see that
    function's own docstring) -- not a model guess at a clinical
    conflict."""
    result = await _tools.merge_test_prep(test_names, lang)
    if result.get("escalate_to_human"):
        logger.info(
            "[%s] test_prep merge escalated for %r: incomplete_data=%s",
            session.call_id,
            test_names,
            result.get("incomplete_data"),
        )
        await _speak(session, phrase("test_prep_conflict_handoff", lang), lang)
        await _handoff_to_human(session, "test_prep_conflict_unresolved", (lang,))
        return
    reply = merged_test_prep_reply(test_names, result, lang)
    await _speak(session, _with_resume(session, reply, lang), lang)
    # KCD-382b: remember what was actually resolved and spoken (never on the escalation path above,
    # where nothing was safely resolved) for a same-call follow-up that names no test.
    session.last_resolved_prep_tests = tuple(test_names)
    session.last_resolved_prep_turn = session.turn_count


def _recover_followup_prep_tests(session: CallSession) -> tuple[str, ...] | None:
    """KCD-382b: the minimum context this story's follow-up case needs -- not general conversation
    memory. Reuses, unchanged, the exact recency rule agent/enquiry_followup.resolve_followup_slot
    already applies to a single test_name follow-up (gap <= FOLLOWUP_MAX_GAP counts as "what we were
    just talking about"), so "how recent is a follow-up" has exactly one definition in this
    codebase, not a second, slightly different one invented here for the plural case."""
    tests = session.last_resolved_prep_tests
    if not tests:
        return None
    if session.turn_count - session.last_resolved_prep_turn > FOLLOWUP_MAX_GAP:
        return None
    return tests


async def _speak_single_test_prep(session: CallSession, test_name: str, lang: str, text: str, data: dict) -> None:
    """KCD-382b: speak ONE already-resolved test's preparation through KCD-382's existing
    single-test path (_answer_enquiry_intent / _finish_enquiry_turn -- the same
    agent.reply_templates.test_prep_reply a directly-named single test already uses), so a test
    resolved from a registered/booking lookup or from this story's follow-up recall is never a
    second implementation of the single-test reply, and still gets _finish_enquiry_turn's own
    secondary-question and coreference bookkeeping for free."""
    slots = {"test_name": test_name}
    reply = await _answer_enquiry_intent("test_prep", slots, lang)
    if reply is None:
        # The resolved name is not in the live catalogue -- never invented, say so plainly rather
        # than guessing at a preparation instruction.
        await _speak(session, phrase("tool_failure", lang), lang)
        return
    await _finish_enquiry_turn(session, text, lang, "test_prep", slots, reply, data)
    session.last_resolved_prep_tests = (test_name,)
    session.last_resolved_prep_turn = session.turn_count


async def _handle_booking_prep_lookup(session: CallSession, phone: str, lang: str) -> None:
    """KCD-382b, PATH B of "Caller has several tests with conflicting preparation": the caller
    confirmed they are registered / have a booking and just gave the phone number it is under.
    Reuses the existing booking-search infrastructure exactly as built (clinic-api/
    patient_context.py::search_bookings, already exposed as _tools.search_bookings) -- including
    its own several-bookings-never-bookings[0] rule (KCD-497) -- to find the caller's TEST
    booking(s) by that phone, then hands the resolved test name(s) to KCD-382's own existing
    single-test path or _handle_multi_test_prep. Never a third implementation of either."""
    try:
        found = await _tools.search_bookings(phone, phone=phone)
    except ToolCallError as e:
        logger.warning("[%s] booking-prep search failed: %s", session.call_id, e)
        await _speak(session, phrase("tool_failure", lang), lang)
        return
    # Only TEST bookings matter here -- an unrelated doctor appointment on the same number is never
    # used for (or counted toward) a test-preparation answer.
    matches = [m for m in (found.get("matches") or []) if m.get("kind") == "test_booking"]
    if not matches:
        # No real test booking on that number: never invented, never guessed -- fall back to the
        # existing generic (caller-names-the-test) path, exactly PATH A's own behaviour.
        session.pending_enquiry = ("test_prep", session.turn_count)
        await _speak(session, phrase("test_prep_no_booking_found", lang), lang)
        await _speak(session, missing_slot_prompt("test_prep", "test_name", lang), lang)
        return
    if len(matches) > 1:
        # KCD-497's own rule, reused: several bookings are never collapsed to the first -- the
        # caller is asked which one, by confirmation number, the same bearer-token identifier
        # every other lookup path in this codebase already uses.
        await _speak(session, multiple_test_bookings_reply(matches, lang), lang)
        return
    confirmation_id = matches[0]["confirmation_id"]
    try:
        tests_result = await _tools.booking_test_names(confirmation_id)
    except ToolCallError as e:
        logger.warning("[%s] booking_test_names failed for %s: %s", session.call_id, confirmation_id, e)
        await _speak(session, phrase("tool_failure", lang), lang)
        return
    test_names = [n for n in (tests_result.get("test_names") or []) if n]
    if not test_names:
        # A confirmation id with no confirmed test rows left (e.g. all cancelled) -- same honest
        # fallback as "no booking found" above, never an invented test.
        session.pending_enquiry = ("test_prep", session.turn_count)
        await _speak(session, phrase("test_prep_no_booking_found", lang), lang)
        await _speak(session, missing_slot_prompt("test_prep", "test_name", lang), lang)
        return
    if len(test_names) > 1:
        await _handle_multi_test_prep(session, test_names, lang)
    else:
        await _speak_single_test_prep(session, test_names[0], lang, "", {})


def _mentions_abdomen(test_name: str) -> bool:
    return "abdomen" in test_name.lower()


async def _queue_prescription_callback(session: CallSession, reason: str, text: str, lang: str) -> None:
    """Every time the caller is asked to send a prescription over WhatsApp, a callback is queued with a factual
    summary (agent/call_summary.py) so the human colleague who opens the photo has the call's context next to it --
    KCD-lay-terms. Only queued when a phone is actually known (session.identity.phone): there is no automatic
    caller-ID in this stack (see _handoff_to_human's own docstring), so without one there is nothing to call back on,
    and a tool failure here must never interrupt the call -- the caller was already told what to do."""
    phone = session.identity.phone
    if not phone:
        return
    summary = build_call_summary(reason, lang=lang, transcript=text, phone=phone, call_id=session.call_id)
    try:
        await _tools.request_callback(phone, session.call_id, "asap", reason, summary)
    except ToolCallError as e:
        logger.warning("[%s] could not queue prescription callback (%s): %s", session.call_id, reason, e)


async def _handle_lay_term(session: CallSession, category: str, text: str, lang: str) -> None:
    """A rural lay term for a test CATEGORY (agent/lay_terms.py) -- "রক্ত পরীক্ষা", "পেটের ছবি", "মাথার ছবি" -- answered
    from the REAL catalogue, never a hardcoded test name. Deterministic, no model call. See agent/lay_terms.py's
    module docstring for why HEAD never names a specific test: the catalogue has none today, and naming one that
    does not exist would be exactly the fabricated fact CLAUDE.md's truth boundary exists to prevent."""
    reason = f"lay_term_{category}"
    if category == BLOOD:
        cat = await _cached_catalogue()
        names = sorted({t["name"] for t in cat.get("tests", []) if t.get("sample_type") == "Blood"})
        reply = blood_test_list_reply(names, lang)
    elif category == ABDOMEN:
        cat = await _cached_catalogue()
        matches = sorted(
            t["name"]
            for t in cat.get("tests", [])
            if t.get("sample_type") == "Imaging" and _mentions_abdomen(t["name"])
        )
        if matches:
            result = await _tools.get_test_rate(matches[0])
            reply = test_rate_reply({"test_name": matches[0]}, result, lang)
        else:  # the catalogue changed and no longer has an abdomen imaging test: honest, not a guess
            reply = phrase("head_imaging_not_listed", lang)
    else:  # HEAD
        reply = phrase("head_imaging_not_listed", lang)
    await _speak(session, f"{reply} {phrase('send_prescription_whatsapp', lang)}", lang)
    await _queue_prescription_callback(session, reason, text, lang)


async def _finish_enquiry_turn(
    session: CallSession, text: str, lang: str, intent: str, slots: dict, base_reply: str, data: dict
) -> None:
    """Shared tail for every stateless enquiry branch (KCD-395/KCD-396):
    answers `secondary_intent` if the model extracted one this turn,
    remembers this turn's entities for the NEXT turn's coreference
    resolution, then speaks the combined reply.

    A secondary lookup that fails (ToolCallError) never discards the
    primary answer the caller already got -- it is reported as "couldn't
    check that" addendum instead, same as this call's own primary
    ToolCallError handling one level up in _dispatch_turn, just scoped
    so a second-question failure cannot blow away a first-question
    success."""
    turn_entities = [(intent, slots)]
    secondary_intent = data.get("secondary_intent")
    if secondary_intent:
        secondary_slots = data.get("secondary_slots") or {}
        secondary_slots, _ = resolve_followup_slot(
            secondary_intent, secondary_slots, text, lang, session.last_enquiry_entities
        )
        addendum = None
        try:
            reply2 = await _answer_enquiry_intent(secondary_intent, secondary_slots, lang)
        except ToolCallError as e:
            logger.warning("[%s] secondary-question lookup failed (%s) -- keeping the first answer", session.call_id, e)
            reply2, addendum = None, phrase("tool_failure", lang)
        if reply2:
            base_reply = f"{base_reply} {reply2}"
            turn_entities.append((secondary_intent, secondary_slots))
        else:
            # A second question was asked but could not be answered this
            # turn (missing slot, or the lookup itself failed) -- KCD-395
            # requires it be explicitly addressed, never silently dropped.
            base_reply = f"{base_reply} {addendum or phrase('unclear', lang)}"
    session.last_enquiry_entities = entities_from_turn(*turn_entities)
    session.last_enquiry_turn = session.turn_count
    suggestion, _ = _suggested.get(), _suggested.set(None)
    if suggestion is not None and session.pending_entity is None:
        replay = {
            **data,
            "intent": suggestion["intent"],
            "secondary_intent": None,
            "slots": {**suggestion["slots"], suggestion["slot"]: suggestion["name"]},
        }
        session.pending_entity = entity_text.PendingEntity(
            suggestion["intent"], suggestion["slot"], suggestion["name"], replay
        )
    base_reply = _with_resume(session, base_reply, lang)  # KCD-104: back to the booking, if one is open
    # An answer to the caller's own question starts with the answer (no "thank you for telling me"); KCD-512: a
    # senior's warm closing after.
    base_reply = session.kindness.decorate(base_reply, lang)
    _note(session, "answers_given")
    await _speak(session, base_reply, lang)


def _mark(session: CallSession, name: str) -> None:
    """Record when a stage of THIS turn finished (first occurrence only). Read by _log_timing at first audio."""
    marks = getattr(session, "marks", None)
    if marks is not None and all(n != name for n, _t in marks):
        marks.append((name, time.monotonic()))


def _log_timing(session: CallSession) -> None:
    """One line per turn: where the time from "the turn started" to "the first audio left" went. Latency was being
    felt and reported without a breakdown; this is the breakdown, per stage, in the log."""
    marks, session.marks = getattr(session, "marks", None), None
    if not marks or len(marks) < 2:
        return
    parts, prev = [], marks[0][1]
    for name, t in marks[1:]:
        parts.append(f"{name} {int((t - prev) * 1000)} ms")
        prev = t
    logger.info(
        "[%s] turn timing: %s | total %d ms",
        session.call_id,
        " | ".join(parts),
        int((marks[-1][1] - marks[0][1]) * 1000),
    )


async def _route_and_transcribe(session: CallSession, utterance_wav: str):
    """LID -> routing decision -> ASR. Returns (language, ASRResult), or
    (None, None) when the router says a human should take the call."""
    if _lid is None or len(_languages_active) < 2:
        return "bn", await _asr_router.transcribe("bn", utterance_wav)

    # Language ID (CPU, ~0.19 s) and recognition used to run one after the other. They now overlap: the recognisers
    # that are almost always needed -- the language this call has been in, and English (cheap, and never ruled out by
    # language ID) -- start at the SAME moment as language ID, and any other one is started the moment it says so.
    # A recogniser nobody needs is simply not run (agent/lang_select.engines_needed): Hindi's ~0.6 s of GPU time was
    # being spent on every Bengali turn for a result that could never win.
    prior = session.lang_router.previous_language or "bn"
    tasks: dict[str, asyncio.Future] = {}
    took: dict[str, float] = {}

    def start(lang: str) -> None:
        if lang in tasks or lang not in _languages_active:
            return

        async def run(lang=lang):
            t0 = time.perf_counter()
            try:
                return await _asr_router.transcribe(lang, utterance_wav)
            finally:
                took[lang] = (time.perf_counter() - t0) * 1000

        tasks[lang] = asyncio.ensure_future(run())
        tasks[lang].add_done_callback(lambda f: f.cancelled() or f.exception())  # an unused failure is not an error

    for lang in dict.fromkeys((prior, "en")):
        start(lang)
    try:
        lid = await asyncio.to_thread(_lid.identify_path, utterance_wav)
    except Exception as e:  # noqa: BLE001 - a LID fault must degrade the turn, not kill it
        logger.warning("[%s] LID failed (%s) -- treating as unknown", session.call_id, e)
        lid = LIDResult(language="unknown", confidence=0.0)
    _mark(session, "lid")

    async def outcomes_for(langs: list[str]) -> list[tuple[str, object]]:
        for lang in langs:
            start(lang)
        done = await asyncio.gather(*[tasks[lang] for lang in langs if lang in tasks], return_exceptions=True)
        good, order = [], [lang for lang in langs if lang in tasks]
        for lang, out in zip(order, done):
            if isinstance(out, BaseException):
                logger.warning("[%s] ASR %s failed: %s", session.call_id, lang, out)
            else:
                good.append((lang, out))
        if not good:
            raise next(o for o in done if isinstance(o, BaseException))
        logger.info(
            "[%s] ASR timings: %s", session.call_id, " | ".join(f"{lg} {int(took.get(lg, 0))} ms" for lg, _ in good)
        )
        return good

    # LID's top label alone is not trusted: Indian-accented English is labelled Hindi at ~0.9, or not found at all
    # (measured; see agent/lang_select.py). Unless LID is decisive, run the recognisers that can still matter and keep
    # the one whose decoders agree AND whom language ID believes.
    verify = languages_to_verify(lid.language, lid.scores, _languages_active) if lid.scores else None
    if verify:
        need = engines_needed(lid.language, lid.scores, _languages_active)
        outcomes = await outcomes_for(need)
        lang, result = pick_candidate(outcomes, lid.scores, session.lang_router.previous_language)
        logger.info(
            "[%s] LID %s %s not decisive -> ran %s, chose %s (%s)",
            session.call_id,
            lid.language,
            {k: round(v, 2) for k, v in lid.scores.items()},
            "+".join(lg for lg, _ in outcomes),
            lang,
            ", ".join(f"{lg}:{r.decoder_agreement:.2f}" for lg, r in outcomes),
        )
        session.lang_router.note_response_language(lang)
        return lang, result

    decision = session.lang_router.route(lid)
    logger.info(
        "[%s] LID %s %.2f -> %s %s (%s)",
        session.call_id,
        lid.language,
        lid.confidence,
        decision.action,
        decision.language or "",
        decision.reason,
    )

    if decision.action == "handoff_human":
        return None, None

    if decision.action == "dual_asr" and decision.secondary_language:
        primary, secondary = decision.language, decision.secondary_language
        both = await outcomes_for([primary, secondary])
        if len(both) == 2:
            return pick_candidate(both, lid.scores, session.lang_router.previous_language)
        return both[0]

    return decision.language, (await outcomes_for([decision.language]))[0][1]


async def _dispatch_turn(session: CallSession, utterance_wav: str):
    """One full turn: LID -> ASR -> intent -> tool -> templated reply -> TTS.
    Serialized per-call via session.dispatch_lock so replies never
    interleave, even if the caller starts talking again immediately."""
    async with session.dispatch_lock:
        session.turn_epoch = session.speak_epoch  # this turn is current; an interrupt from now on makes it stale
        session.acks.next_turn()
        session.turn_acknowledged = False
        session.reply_noted = False
        prior_lang = session.lang  # the language the call was in BEFORE this turn's recognition (see _keep_language)
        # An answer to a question we asked: the last thing we said had a question mark, or a flow that asks in commands
        # ("please tell me your full name") is waiting on this very turn.
        session.answering = (
            session.awaiting_answer
            or session.history_state in ("need_phone", "need_name")
            or (session.security is not None and not session.security.finished)
        )
        session.awaiting_answer = False
        session.silence_since = None
        session.turn_started_at = time.monotonic()
        session.marks = [("start", session.turn_started_at)]
        channel_quality = session.call_state.channel_quality
        analysis: dict | None = None
        audio_issues: list[str] = []
        asr_wav = utterance_wav
        temp_wavs: list[str] = []
        try:
            if NEAR_END_ATTENTION != "off" and session.near_end.established:
                attended = await asyncio.to_thread(
                    _attend_wav_to_path, utterance_wav, session.near_end.level_dbfs, session.near_end.f0_hz or 0.0
                )
                if attended:
                    temp_wavs.append(attended)
                    asr_wav = attended
            if CONDITION_INPUT != "off":
                conditioned = await asyncio.to_thread(_condition_wav_to_path, asr_wav, CONDITION_INPUT)
                if conditioned:
                    temp_wavs.append(conditioned)
                    asr_wav = conditioned
            _mark(session, "prep")
            lang, asr_result = await _route_and_transcribe(session, asr_wav)
            _mark(session, "asr")
            # KCD-075/KCD-076: measured inside its own budget slice, on
            # the SAME clip ASR already read, before the finally below
            # deletes it. A slow or failed classification just keeps
            # last turn's label -- never blocks or fails the turn.
            channel_quality = await run_within_budget(
                "channel_quality",
                CHANNEL_QUALITY_BUDGET_S,
                _classify_channel_from_wav_path,
                utterance_wav,
                previous_value=channel_quality,
            )
            # Why is this clip hard to hear, and does the speaker sound older?
            # A slow or failed analysis just means no verdict this turn.
            analysis = await run_within_budget(
                "audio_quality",
                AUDIO_ANALYSIS_BUDGET_S,
                _analyze_utterance_from_wav_path,
                utterance_wav,
                previous_value=None,
            )
            audio_issues = analysis["audio"].issues if analysis else []
            for issue in audio_issues:
                audio_issue_buckets.record(issue, "seen", lang or session.lang)
            # Appendix F channel buckets, on the same vocabulary as the offline evaluation:
            # the live rate of cross_talk / noisy / narrowband turns (KCD-053/055).
            golden_buckets.record("turn", "analysed", lang or session.lang)
            for bucket in channel_buckets(audio_issues, channel_quality):
                golden_buckets.record(bucket, "turn", lang or session.lang)
            asr_result = await _retry_on_enhanced_audio(
                session,
                utterance_wav,
                lang,
                asr_result,
                audio_issues,
                analysis["audio"].duration_s if analysis else None,
            )
        except Exception as e:
            logger.exception("[%s] ASR/LID stage failed: %s", session.call_id, e)
            await _speak(session, phrase("llm_failure", session.lang), session.lang, fallback_reason="llm_failure")
            return
        finally:
            with contextlib.suppress(OSError):
                os.remove(utterance_wav)
            for extra in temp_wavs:
                with contextlib.suppress(OSError):
                    os.remove(extra)

        # KCD-053: a much quieter utterance in a DIFFERENT voice is someone else in the room,
        # not the caller. It never opens a turn; the call stays silent for it. The same voice,
        # merely quieter, is still the caller and goes on to the empathetic re-ask.
        if analysis and analysis.get("near_end"):
            level_dbfs, f0_hz = analysis["near_end"]
            verdict = session.near_end.judge(level_dbfs, f0_hz)
            if verdict.background:
                session.near_end.reject()
                audio_issue_buckets.record("background_utterance", "dropped", lang or session.lang)
                logger.info("[%s] background utterance dropped (%s)", session.call_id, verdict.reason)
                return
            session.near_end.accept(level_dbfs, f0_hz)

        if lang is None:
            # Ask again, kindly, before any hand-off -- an unidentifiable
            # utterance is usually a faint, noisy or mumbled one, not a
            # request for a person (agent/reask_policy.py).
            await _reask_or_handoff(
                session,
                session.reask.decide(language_ambiguous=True, audio_issues=audio_issues),
                session.lang,
                languages=_languages_active,
            )
            return
        detected_lang = lang  # what the CALLER actually spoke this turn -- kept for analytics; never overridden
        _rec(session, "language", detected_lang)
        lang = _resolve_reply_language(session, lang)  # what the AGENT replies in -- sticky; see the function
        confidence = confidence_state(getattr(asr_result, "decoder_used", None), asr_result.decoder_agreement)
        apply_confidence(session.call_state, confidence != VERIFIED)
        apply_channel_quality(session.call_state, channel_quality)
        channel_quality_buckets.record(channel_quality, "seen", lang)
        logger.debug("[%s] call_state %s", session.call_id, session.call_state.to_log_dict())

        text = asr_result.text.strip()
        if not text:
            logger.info("[%s] ASR returned empty text", session.call_id)
            lang = _keep_language(session, prior_lang)
            await _reask_or_handoff(
                session,
                session.reask.decide(asr_empty=True, audio_issues=audio_issues),
                lang,
                fallback_reason="asr_empty",
            )
            return
        # A possible medical emergency outranks everything: it is checked on EVERY turn, in every
        # language, at any ASR confidence, before any re-ask, model call or booking step, and it
        # abandons whatever was in progress (agent/emergency.py).
        if detect_emergency(text):
            logger.warning("[%s] possible emergency in the transcript", session.call_id)
            await session.send_json("User", text)
            session.turn_count += 1
            session.booking, session.pending_entity, session.history_state = None, None, None
            _note(session, "emergency")
            apply_caller_state(session.call_state, "emergency")
            await _speak(session, phrase("emergency_notice", lang), lang)
            await _handoff_to_human(session, "emergency", (lang,))
            session.outcome = "emergency"
            return
        # Swearing at the agent: a calm fixed boundary in the caller's own language, never the model (agent/abuse.py);
        # the third time in one call, a courteous close.
        if abuse.is_abusive(text):
            session.abuse_count += 1
            _note(session, "abuse_turns")
            session.signals.turns += 1
            reply_lang = _abuse_reply_language(text, lang, prior_lang)
            _keep_language(session, reply_lang)
            logger.info("[%s] abusive turn %d", session.call_id, session.abuse_count)
            await session.send_json("User", text)
            session.turn_count += 1
            if abuse.closes_the_call(session.abuse_count):
                session.outcome = "abusive_caller"
                await _end_call(session, reply_lang, abuse.response_key(session.abuse_count))
            else:
                await _speak(session, phrase(abuse.response_key(session.abuse_count), reply_lang), reply_lang)
            return
        # A yes/no to a confirmation is legitimately one short word; the
        # jumbled-transcript checks would misread it as a fragment.
        # The same holds for a dictated phone number, a spelled name or a bare
        # "না" while a booking is collecting slots: all legitimately short or
        # tokenised, so only decoder disagreement is meaningful then.
        in_booking_flow = session.booking is not None and (
            session.booking.stage in ("confirming", "awaiting_charge_confirm") or not session.booking.is_stale()
        )
        problem = transcript_problem(
            text,
            detected_lang,  # the script this text was actually RECOGNISED in, not the (now sticky) reply language
            analysis["audio"].duration_s if analysis else None,
            asr_result.decoder_agreement,
            slot_answer=in_booking_flow
            or session.pending_entity is not None
            or session.awaiting_handoff_offer
            or session.awaiting_clinician_offer  # a bare "yes" to "shall I connect our clinician?"
            or session.awaiting_callback_phone  # a dictated phone number is legitimately just digits
            or session.history_state is not None
            or session.awaiting_draft_answer
            or session.awaiting_pref_answer
            or session.awaiting_resume
            or session.awaiting_close_answer
            or session.awaiting_test_prep_registration_answer  # KCD-382b: a bare yes/no
            or session.awaiting_test_prep_booking_phone,  # KCD-382b: a dictated phone number
        )  # KCD-104; "anything else?" -> a bare no: a bare yes or no to "shall I go back to it" is an answer
        if problem:
            logger.info("[%s] jumbled transcript (%s)", session.call_id, problem)
            lang = _keep_language(session, prior_lang)
            await _reask_or_handoff(
                session, session.reask.decide(transcript_issue=problem, audio_issues=audio_issues), lang
            )
            return
        session.reask.note_success()
        session.signals.turns += 1
        # The CALLER's language history (for the flip-detection signal below and call_score.py's language_flips) --
        # what was actually SAID each turn, not the (now sticky) reply language, which barely varies by design.
        session.lang_history.append(detected_lang)
        if (
            len(session.lang_history) >= 3
            and session.lang_history[-1] == session.lang_history[-3] != session.lang_history[-2]
        ):
            _note(session, "language_flips")  # A, B, A: the bot moved the call to B and the caller went back
        if _caller_thanked(text, detected_lang):  # detecting a thank-you in what was actually said, not the reply lang
            _note(session, "caller_thanked")
        session.silence_prompts = 0
        await session.send_json("User", text)
        session.turn_count += 1
        after_prompt, session.awaiting_close_answer = session.awaiting_close_answer, False
        answer = classify_yes_no(text, lang) if after_prompt else None
        # "end the call", "শেষ করে দিন" (as the answer to "anything else?"): the call ends, decided by words, not the model
        if call_end.wants_to_end(text, lang, after_prompt=after_prompt) or answer == "no":
            _note(session, "ended_by_caller")
            await _end_call(session, lang, "silence_goodbye")
            return
        if answer == "yes":
            await _speak(session, phrase("silence_go_on", lang), lang)
            return
        code_switch_buckets.record(mixture_bucket(text), "seen", lang)

        # Ported story "Caller asks whether their result is dangerous"
        # (agent/clinical_safety.py). Two halves of one exchange, kept together.
        #
        # First: the answer to the offer this guard made on an earlier turn. A
        # "yes" reaches a clinician under this story's own reason code, never
        # the history flow's. Anything that is neither yes nor no lets the offer
        # lapse and the turn is handled normally -- the same courtesy
        # _continue_history_flow() shows its own offer.
        if session.awaiting_clinician_offer:
            session.awaiting_clinician_offer = False
            offer_answer = classify_yes_no(text, lang)
            if offer_answer in ("yes", "no"):
                await session.send_json("User", text)
                session.turn_count += 1
                if offer_answer == "yes":
                    await _handoff_to_human(session, session.clinician_offer_reason, (lang,))
                else:
                    await _speak(session, phrase("silence_go_on", lang), lang)
                return

        # Second: the guard itself. It runs before the fast path, the semantic
        # cache and the model, because "routed to a human BY POLICY rather than
        # by prompt wording" is the acceptance criterion -- a prompt instruction
        # is advice a 7B model can be argued out of, and a third of this
        # module's phrase list exists because an adversarial suite proved
        # exactly that. Checked BEFORE the complaint guard below, which is the
        # order the module this was ported from tested.
        #
        # Nothing here judges the result. The reply is fixed text that says who
        # WILL explain it and offers to connect them: without that offer a
        # frightened caller gets the generic greeting this turn used to fall
        # through to, which is the rebuff the story exists to prevent.
        if is_clinical_interpretation(text):
            logger.info("[%s] clinical interpretation asked for; offering a clinician", session.call_id)
            _note(session, "clinical_questions")
            await session.send_json("User", text)
            session.turn_count += 1
            session.awaiting_clinician_offer = True
            session.clinician_offer_reason = "clinical_interpretation"
            await _speak(session, phrase("clinical_interpretation", lang), lang)
            return

        # Ported story "Caller wants to make a complaint" (agent/complaint.py).
        #
        # Checked BEFORE asks_for_a_person() just below, and before agent/anger.py's
        # check inside _update_caller_state(): "I want to speak to someone about a
        # complaint" matches all three guards, and this is the most specific and the
        # most complete of them -- it keeps the caller's words verbatim AND routes to
        # a person, where the other two only route. It returns, so neither of the
        # other two runs for this turn, which is how the story's "anger enhances the
        # complaint flow, it does not compete with it" is honoured without either
        # module knowing about the other.
        #
        # The reply is fixed text (agent/phrases.py: complaint_acknowledged), never
        # the model: acceptance criterion 5 forbids defending the clinic or
        # explaining the complaint away, and a sentence the model writes is a
        # sentence that can do both.
        #
        # The verbatim text goes to clinic-api's `complaints` table and nowhere else
        # -- the escalation ledger and the call record stay text-free by design
        # (agent/call_record.py). A failure to store it must never cost the caller
        # their acknowledgement or their hand-off, so the tool call is suppressed on
        # error exactly as every other durable write on a spoken path here is.
        if is_complaint(text):
            _, family = detect_complaint(text)
            logger.info("[%s] complaint filed (%s phrase list)", session.call_id, family)
            _note(session, "complaint_filed")
            await session.send_json("User", text)
            session.turn_count += 1
            with contextlib.suppress(ToolCallError):
                await _tools.submit_complaint(session.call_id, text, family or "", lang, session.identity.phone)
            await _speak(session, phrase("complaint_acknowledged", lang), lang)
            await _handoff_to_human(session, "complaint", (lang,))
            return

        # Ported story "Caller wants to speak to a doctor personally"
        # (agent/doctor_personal_request.py). Checked next to asks_for_a_person()
        # below because it is the same KIND of request, and immediately BEFORE it
        # because the two want different answers: a caller asking for the front
        # desk can be handed straight there, and a caller asking for a doctor
        # cannot be -- there is no clinician on this line. The noun lists do not
        # overlap ("doctor" and "clinician" are deliberately absent from
        # agent/human_request.py's front-desk list), so this ordering decides
        # nothing today; it is fixed so that widening either list later cannot
        # silently turn a doctor request into a front-desk hand-off.
        #
        # The reply is fixed text and never names a doctor, including one the
        # caller just named: there is no way to reach a clinician from this call,
        # so anything warmer would be a promise this stack cannot keep.
        if is_doctor_personal_request(text):
            logger.info("[%s] caller asked to speak to a doctor personally", session.call_id)
            await session.send_json("User", text)
            session.turn_count += 1
            await _speak(session, phrase("doctor_personal_request", lang), lang)
            return

        # KCD-353: a request for a person is honoured IMMEDIATELY -- deterministic, before any
        # model call, before intent extraction, with no "did you really mean it".
        if asks_for_a_person(text, lang):
            logger.info("[%s] caller asked for a person", session.call_id)
            await _handoff_to_human(session, "caller_requested", (lang,))
            return

        # Ported story "Caller asks to be called back" (agent/callback_request.py).
        #
        # This stack has NO outbound calling, so the whole feature is: write the
        # request down for a colleague, and say exactly that. The durable half
        # already existed before this merge -- clinic-api's callback_requests
        # table and _tools.request_callback(), used today when a hand-off happens
        # and a number is known -- so what was missing was only the caller-facing
        # half: noticing that the caller ASKED.
        #
        # No promise is made about WHEN. The original module decided that from the
        # clinic's opening hours; this repository stores hours only as localised
        # free text (clinic-api/models.py DepartmentHours), and reading a time
        # window out of prose to promise a callback would be inventing a fact.
        #
        # Checked after agent/doctor_personal_request.py, which is the right
        # precedence: "have the doctor call me" is answered there, because no
        # specific doctor can be promised. The phrases here are the generic ones.
        if session.awaiting_callback_phone:
            session.awaiting_callback_phone = False
            dictated = _parse_spoken_phone(text)
            if dictated:
                await session.send_json("User", text)
                session.turn_count += 1
                session.identity.phone = session.identity.phone or dictated
                await _queue_callback_request(session, dictated, lang)
                return
            # No number heard: the request lapses rather than trapping the caller
            # in a loop, and this turn is handled normally. Same courtesy the
            # clinician offer above shows an answer that is neither yes nor no.

        if asks_for_a_callback(text):
            logger.info("[%s] caller asked to be called back", session.call_id)
            await session.send_json("User", text)
            session.turn_count += 1
            if not CALLBACKS_ENABLED:
                await _speak(session, phrase("callback_unavailable", lang), lang)
                return
            known = session.identity.phone
            if known:
                await _queue_callback_request(session, known, lang)
            else:
                session.awaiting_callback_phone = True
                await _speak(session, phrase("callback_need_number", lang), lang)
            return

        # Ported story "Caller asks to compare two options" (agent/compare_flow.py).
        #
        # Runs BEFORE the fast path, and that placement is the point. The fast
        # path's own `_ambiguous_test()` sees two tests fitting the words almost
        # equally well and abstains by design, so a comparison goes to the model
        # -- which has no compare intent, because this merge leaves the intent set
        # untouched. The caller asked about two things and was told the price of
        # one. This guard answers it from two real lookups instead.
        #
        # Both names are resolved against the live catalogue with the SAME matcher
        # and the SAME floor the fast path uses, and the second is found with
        # `skip_name` so it cannot be the first one again. If only one resolves,
        # the caller is ASKED for both: comparing against a guessed second item
        # would state a difference that was never computed from real data.
        if asks_to_compare(text) and _fast_path is not None:
            first, _form_a, _score_a = _fast_path.catalogue.match(text, "test", lang, COMMIT_FLOOR)
            second, _form_b, _score_b = _fast_path.catalogue.match(
                text, "test", lang, COMMIT_FLOOR, skip_name=first
            )
            if first and second:
                logger.info("[%s] comparing %r with %r", session.call_id, first, second)
                await session.send_json("User", text)
                session.turn_count += 1
                sides = []
                for name in (first, second):
                    answer = None
                    with contextlib.suppress(ToolCallError):
                        answer = await _tools.get_test_rate(name)
                    sides.append({"kind": KIND_TEST, **answer} if answer else {"kind": KIND_NOT_FOUND})
                _note(session, "answers_given")
                await _speak(
                    session, compare_options_reply(build_comparison(sides[0], sides[1]), first, second, lang), lang
                )
                return
            if first or second:
                await session.send_json("User", text)
                session.turn_count += 1
                await _speak(session, phrase("compare_need_two", lang), lang)
                return
            # Neither side resolved: not a comparison this system can make from
            # its own catalogue, so the turn carries on as it did before.

        # Ported story "Caller asks which doctors are available"
        # (agent/doctor_list.py). A plural, unnamed ask -- "which doctors do you
        # have", "কে কে ডাক্তার আছেন", "कौन कौन से डॉक्टर हैं". Served here
        # because it has a real lookup behind it and this repository had neither
        # the intent nor the endpoint: /api/v1/doctors and
        # /api/v1/doctors/by-department were added for it.
        #
        # The WHOLE utterance is handed to the by-department endpoint first. That
        # is deliberate and is why no department-name extractor was ported: the
        # endpoint matches against the real department table in both directions,
        # so "which doctors are in cardiology" resolves on the department name
        # inside the sentence, and "cardio" resolves on being inside the name.
        # Several matches come back `ambiguous` and a near match as
        # `did_you_mean`, both of which the reply template ASKS about -- never a
        # silent guess. A query that names no department at all resolves to
        # nothing, and the caller gets the full list instead, which is what they
        # asked for.
        if asks_for_the_doctor_list(text):
            logger.info("[%s] caller asked which doctors there are", session.call_id)
            await session.send_json("User", text)
            session.turn_count += 1
            by_department = None
            with contextlib.suppress(ToolCallError):
                by_department = await _tools.doctors_by_department(text)
            if by_department and (
                by_department.get("found") or by_department.get("ambiguous") or by_department.get("did_you_mean")
            ):
                _note(session, "answers_given")
                await _speak(session, doctors_by_department_reply(by_department, lang), lang)
                return
            everyone = None
            with contextlib.suppress(ToolCallError):
                everyone = await _tools.list_doctors()
            if everyone is None:
                _note(session, "system_failures")
                await _speak(session, phrase("tool_failure", lang), lang)
                return
            _note(session, "answers_given")
            await _speak(session, doctor_list_reply(everyone, lang), lang)
            return

        # Ported story "Caller states something the agent cannot verify"
        # (agent/unverifiable_claim.py). Runs AFTER every guard above it --
        # emergency, abuse, clinical interpretation, complaint,
        # doctor-personal-request, and the request for a person just above --
        # because each of those is a more specific reading of an utterance that
        # may also contain an assertion. It only ever sees text they have all
        # let through.
        #
        # Only TWO of the original's four claim categories are intercepted here.
        # "My appointment is tomorrow" and "I already paid" are both CHECKABLE in
        # this repository (GET /api/v1/bookings/lookup, GET
        # /api/v1/billing/outstanding), so they are deliberately left to fall
        # through to the real verified lookup -- which is this story's own second
        # acceptance criterion, "the agent checks the system of record where
        # verification is possible". Declining them would lose a capability this
        # system has.
        #
        # A claim about a RESULT is answered with the clinical reply: what a
        # result means is not something this system may state at all
        # (agent/clinical_safety.py), so "a doctor will explain it, shall I
        # connect one" is both the honest answer and the already-tested one.
        claim_category, _claim_phrase = detect_unverifiable_claim(text)
        if claim_category != "none":
            logger.info("[%s] unverifiable claim (%s)", session.call_id, claim_category)
            await session.send_json("User", text)
            session.turn_count += 1
            session.awaiting_clinician_offer = True
            if claim_category == CATEGORY_REPORT_RESULT:
                session.clinician_offer_reason = "clinical_interpretation"
                await _speak(session, phrase("clinical_interpretation", lang), lang)
            else:
                session.clinician_offer_reason = "unverifiable_claim"
                await _speak(session, phrase("cannot_confirm_claim", lang), lang)
            return

        # KCD-353: the disclosure is in the greeting (Bengali); a caller who answers in another
        # language hears it once, in theirs.
        if lang not in session.disclosed_langs:
            session.disclosed_langs.add(lang)
            await _speak(session, disclosure_for(lang), lang)

        # KCD-054: after verification, a different voice revokes it. The agent cannot know who
        # is speaking -- only that the voice differs from the one that was verified -- so it
        # says so plainly and stops disclosing personal details until verification is repeated.
        await _check_speaker_change(session, analysis, lang)

        # KCD-494/495/498/500: a question about the caller's OWN history, or an answer to a step of
        # one already under way, is handled here, deterministically, before any model call.
        if session.security is not None and not session.security.finished:
            if await _continue_security_flow(session, text, lang, analysis):
                return
        if session.awaiting_draft_answer and await _continue_draft_offer(session, text, lang):
            return
        if session.awaiting_pref_answer and await _continue_preference_offer(session, text, lang):
            return
        if session.awaiting_handoff_offer or session.history_state:
            if await _continue_history_flow(session, text, lang):
                return
        history_question = detect_history_question(text, lang)
        if history_question is not None:
            session.pending_history, session.history_attempts = history_question, 0
            await _history_step(session, lang)
            return

        # KCD-084/149/155: caller state -> delivery policy, and the
        # acknowledgement that comes before anything else when the policy
        # calls for one (once per state entry, not every turn).
        _update_caller_state(session, text, lang, analysis)
        ack = select_acknowledgement(session.policy, session.call_state.caller_state, lang)
        if ack and session.acknowledged_state != session.call_state.caller_state:
            session.acknowledged_state = session.call_state.caller_state
            session.turn_acknowledged = True
            await _speak(session, ack, lang)

        # KCD-438: an explicit "speak in Hindi/Bengali/English" request,
        # detected deterministically (no LLM call, same zero-extra-latency
        # reasoning as the yes/no check below). Acknowledged immediately in
        # the requested language. KCD-lay-terms: this is now the (only)
        # way the sticky reply language (_resolve_reply_language) changes
        # mid-call -- an explicit request LOCKS it, unlike ambient per-turn
        # language ID, which no longer moves it at all.
        switch_target = detect_language_switch_request(text, lang)
        if switch_target:
            _lock_reply_language(session, switch_target)
            session.lang_router.note_response_language(switch_target)
            await _speak(session, phrase("language_switched", switch_target), switch_target)
            return

        # Epic E26: a booking/reschedule/cancel/add-test confirmation
        # already in progress is a closed yes/no question -- answered
        # deterministically (agent/booking_flow.classify_yes_no), with NO
        # LLM round trip, both for the zero-extra-latency requirement and
        # because CLAUDE.md's truth boundary keeps exactly this kind of
        # high-stakes binary decision out of the model's hands.
        if session.awaiting_resume and await _continue_resume_offer(session, text, lang):
            return
        if session.booking is not None and session.booking.stage in ("confirming", "awaiting_charge_confirm"):
            # KCD-104: only a yes or a no is decided here. Anything else -- a question, a correction, another
            # task -- used to be answered by reading the confirmation out again; it is now a fresh turn, and the
            # pending confirmation stays pending (see _with_resume and the "unclear" branch below).
            session.no_at_confirm = False
            if await _handle_booking_confirmation_turn(session, text, lang) is not False:
                await _after_task_finished(session, lang)
                return

        # KCD-382b: the two questions the registered/booked-patient preparation flow asks are
        # themselves closed yes/no (or digits-only) questions, answered the same deterministic way
        # as the booking-confirmation and awaiting_callback_phone checks above -- no LLM round trip,
        # and a caller who answers something else entirely just falls through to the normal turn.
        if session.awaiting_test_prep_booking_phone:
            session.awaiting_test_prep_booking_phone = False
            dictated = _parse_spoken_phone(text)
            if dictated:
                session.identity.phone = session.identity.phone or dictated
                await _handle_booking_prep_lookup(session, dictated, lang)
                return
            # No number heard: never guess one -- the request lapses and this turn is handled normally.

        if session.awaiting_test_prep_registration_answer:
            session.awaiting_test_prep_registration_answer = False
            reg_answer = classify_yes_no(text, lang)
            if reg_answer == "yes":
                session.awaiting_test_prep_booking_phone = True
                await _speak(session, phrase("test_prep_ask_booking_phone", lang), lang)
                return
            if reg_answer == "no":
                session.pending_enquiry = ("test_prep", session.turn_count)
                await _speak(session, missing_slot_prompt("test_prep", "test_name", lang), lang)
                return
            # Neither yes nor no heard: the question lapses and this turn is handled normally.

        confirmed_entity = False
        pending, session.pending_entity = session.pending_entity, None
        if pending is not None:
            answer = classify_yes_no(text, lang)
            if answer == "yes":
                data, confirmed_entity = pending.data, True
            elif answer == "no":
                await _speak(session, entity_text.reask(lang), lang)
                return
            # anything else: the question lapses and this turn is handled as a fresh one

        if not confirmed_entity:
            # KCD-lay-terms: a rural lay term for a test category ("রক্ত পরীক্ষা", "পেটের ছবি", "মাথার ছবি"), matched
            # deterministically before any model call -- but never while a booking is already actively in progress
            # (session.booking is not None), so a lay-term-shaped word inside an unrelated answer never hijacks it.
            lay_term = None if session.booking is not None else match_lay_term(text)
            if lay_term is not None:
                await _handle_lay_term(session, lay_term, text, lang)
                return
            try:
                data = await _resolve_intent(session, text, lang)
            except ExtractionError as e:
                logger.error("[%s] intent extraction failed: %s", session.call_id, e)
                await _speak(session, phrase("llm_failure", lang), lang, fallback_reason="llm_failure")
                return

        intent = data["intent"]
        slots = data["slots"]
        pending, session.pending_enquiry = session.pending_enquiry, None
        if intent == "unclear" and pending is not None and session.turn_count - pending[1] <= 1:
            intent = pending[0]  # "the same test", a bare name: the answer to the question we just asked
        _mark(session, "intent")
        _rec(session, "intent", intent)
        _note_stated_age(session, slots, lang)

        if intent == "smalltalk":
            reply = data.get("direct_reply_bn")
            # The ONLY model-composed text a caller ever hears, so it passes the persona
            # (register, no hedging, no reassurance or advice, sentence length) or is replaced.
            # KCD-500: and it may not claim anything about THIS caller's past -- history is rendered
            # from retrieved fields by agent/history_templates.py, never composed.
            chosen = (
                reply
                if (
                    _speakable(reply or "", lang)
                    and persona_clean(reply or "", lang)
                    and not history_text.mentions_personal_history(reply or "", lang)
                )
                else phrase("smalltalk_default", lang)
            )
            await _speak(session, _with_resume(session, chosen, lang), lang)
            return

        if (
            intent == "unclear"
            and session.booking is not None
            and session.booking.stage in ("confirming", "awaiting_charge_confirm")
        ):
            if session.no_at_confirm:
                await _reopen_after_no(session, session.booking, lang)  # a no with nothing to change in it
            else:
                await _speak(
                    session, booking_confirmation_readback(session.booking.slots, session.booking.action, lang), lang
                )
            return

        if intent == "unclear":
            # CodeRabbit-flagged, real gap: the "confirming"/
            # "awaiting_charge_confirm" short-circuit above only covers
            # the yes/no step. During ordinary "collecting"-stage slot
            # filling, a bare answer with no sentence structure to key
            # off ("Ravi Das", a bare 10-digit number) is exactly the
            # kind of turn the LLM classifies "unclear" even when it DID
            # correctly extract the slot into `slots` -- and nothing
            # routed it back to the active booking, silently stalling
            # the guided one-slot-per-turn flow on the minimal answers
            # it is meant to collect. Re-attempt this turn as the active
            # action instead of bailing, as long as the state is not
            # stale (is_stale() -- previously never called anywhere,
            # per CodeRabbit -- guards against resurrecting a booking
            # the caller has plainly moved on from).
            if session.booking is not None and session.booking.stage == "collecting" and not session.booking.is_stale():
                intent = session.booking.action
            else:
                await _speak(session, phrase("unclear", lang), lang)
                return

        # KCD-442/KCD-447: a turn the two ASR decoders disagreed on is not
        # trusted to drive a factual lookup at all -- acting on a misheard
        # test/doctor name would produce a confident, wrong answer about
        # something else, which is worse than asking the caller to repeat
        # themselves. Checked before the tool call, not after: the point
        # is to never RUN the lookup on an unreliable entity, not merely
        # to hedge the reply once it comes back.
        #
        # One gate decides this (agent/action_gate.py), not two separate confidence checks -- it is proven
        # (tests/test_action_gate.py) to give exactly the same answer should_withhold_factual_answer() and
        # needs_entity_readback() gave here before; the entity itself is not yet resolved at this point in the
        # turn (the lookup has not run), so it is not passed -- exactly as those two functions never took it either.
        decoder_used = getattr(asr_result, "decoder_used", None)
        gate_verdict = (
            action_gate.decide(intent, decoder_used=decoder_used, decoder_agreement=asr_result.decoder_agreement)
            if not confirmed_entity
            else None
        )
        if gate_verdict is not None and gate_verdict.action == action_gate.REPEAT:
            logger.info(
                "[%s] withholding %s answer: decoder_agreement=%.2f below floor",
                session.call_id,
                intent,
                asr_result.decoder_agreement,
            )
            insufficient_information.record("low_decoder_agreement", intent, lang)
            # A live call (2026-09-28): the model DID extract a name ("Roy") even though the two decoders
            # disagreed on the sentence as a whole -- the caller heard a blank "sorry, I didn't catch the name,
            # which doctor?" when what they actually needed was "do you mean Roy?", a yes/no that is faster to
            # answer and confirms EXACTLY the uncertain part instead of asking them to repeat everything. This is
            # the same read-back entity_text already does for UNAVAILABLE confidence (see the READ_BACK branch
            # below) -- extended here to LOW confidence too, but ONLY when the model actually named something;
            # nothing is invented, and the lookup still only runs after an explicit yes.
            entity = entity_text.entity_to_confirm(intent, slots)
            if entity is not None:
                slot, value = entity
                session.pending_entity = entity_text.PendingEntity(intent, slot, value, data)
                insufficient_information.record("low_decoder_agreement_readback", intent, lang)
                await _speak(session, entity_text.confirm_question(slot, value, lang), lang)
                return
            # Nothing at all was extracted: the caller said WHAT they want ("the price of ...") but not clearly
            # enough to guess a name from. Ask for the name only, instead of "say that again" for the whole sentence.
            name_slot = {"test_rate": "test_name", "test_prep": "test_name", "doctor_availability": "doctor_name"}.get(
                intent
            )
            if name_slot:
                session.pending_enquiry = (intent, session.turn_count)
                await _speak(
                    session, f"{phrase('name_not_caught', lang)} {missing_slot_prompt(intent, name_slot, lang)}", lang
                )
            else:
                await _speak(session, insufficient_information_reply(lang), lang)
            return
        # Only one decoder produced this text, so nothing vouches for it: read the entity back and
        # run the lookup only after a yes. Never let "no confidence figure" mean "trusted".
        if gate_verdict is not None and gate_verdict.action == action_gate.READ_BACK:
            entity = entity_text.entity_to_confirm(intent, slots)
            if entity is not None:
                slot, value = entity
                session.pending_entity = entity_text.PendingEntity(intent, slot, value, data)
                insufficient_information.record("single_decoder_readback", intent, lang)
                await _speak(session, entity_text.confirm_question(slot, value, lang), lang)
                return

        try:
            if intent == "test_rate":
                slots, _ = _followup(session, intent, slots, text, lang)
                reply = await _answer_enquiry_intent(intent, slots, lang)
                if reply is None:
                    session.pending_enquiry = (intent, session.turn_count)
                    await _speak(session, missing_slot_prompt(intent, "test_name", lang), lang)
                    return
                await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

            elif intent == "doctor_availability":
                slots, _ = _followup(session, intent, slots, text, lang)
                reply = await _answer_enquiry_intent(intent, slots, lang)
                if reply is None:
                    session.pending_enquiry = (intent, session.turn_count)
                    await _speak(session, missing_slot_prompt(intent, "doctor_name", lang), lang)
                    return
                await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

            elif intent == "test_prep":
                slots, _ = _followup(session, intent, slots, text, lang)
                # KCD-382: more than one test named in the same turn goes through the merge path entirely
                # BEFORE _answer_enquiry_intent / _finish_enquiry_turn -- see _handle_multi_test_prep's own
                # docstring for why (a hand-off closes the socket; nothing may speak after it).
                multi_names = [n for n in (slots.get("test_names") or []) if n and n.strip()]
                if len(multi_names) > 1:
                    await _handle_multi_test_prep(session, multi_names, lang)
                    return
                reply = await _answer_enquiry_intent(intent, slots, lang)
                if reply is None:
                    # KCD-382b: nothing named at all. Before falling back to the existing "which
                    # test" ask, try the same-call follow-up recall (a test set already resolved
                    # this call, named directly or via a booking lookup), then -- once, per call --
                    # ask whether the caller is registered/has a booking, so a DB-backed caller is
                    # never made to repeat test names the booking record already has.
                    recovered = _recover_followup_prep_tests(session)
                    if recovered:
                        if len(recovered) > 1:
                            await _handle_multi_test_prep(session, list(recovered), lang)
                        else:
                            await _speak_single_test_prep(session, recovered[0], lang, text, data)
                        return
                    if not session.test_prep_registration_asked:
                        session.test_prep_registration_asked = True
                        session.awaiting_test_prep_registration_answer = True
                        await _speak(session, phrase("test_prep_ask_registered", lang), lang)
                        return
                    session.pending_enquiry = (intent, session.turn_count)
                    await _speak(session, missing_slot_prompt(intent, "test_name", lang), lang)
                    return
                await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)
                resolved_name = slots.get("test_name") or next(iter(slots.get("test_names") or []), None)
                if resolved_name:
                    session.last_resolved_prep_tests = (resolved_name,)
                    session.last_resolved_prep_turn = session.turn_count

            elif intent == "clinic_faq":
                slots, _ = _followup(session, intent, slots, text, lang)
                reply = await _answer_enquiry_intent(intent, slots, lang)
                if reply is None:
                    session.pending_enquiry = (intent, session.turn_count)
                    await _speak(session, missing_slot_prompt(intent, "faq_topic", lang), lang)
                    return
                await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

            elif intent == "book_appointment":
                st = _enter_task(session, "book_appointment")
                prior_slots = dict(st.slots)
                changed = merge_slots(st, slots)
                if st.hold_token and {"doctor_name", "date", "time_slot"} & set(changed):
                    # KCD-104: the caller changed WHICH slot they want; the hold is on the old one. Drop it so
                    # the new doctor/date/time is held below (the old hold expires on its own).
                    st.hold_token, st.hold_doctor_id = None, None
                ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
                if ack:
                    await _speak(session, ack, lang)
                else:
                    await _thank_for_answer(session, st, changed, prior_slots, lang)
                    await _echo_new_slots(session, changed, prior_slots, st.slots, lang)

                # Secure the hold as soon as doctor+date+time are known, even
                # if patient details are still missing -- KCD-376: the
                # concurrency guard must run before the caller has spoken a
                # patient name, not after.
                if (
                    st.hold_token is None
                    and st.slots.get("doctor_name")
                    and st.slots.get("date")
                    and st.slots.get("time_slot")
                ):
                    hold = await _tools.hold_slot(st.slots["doctor_name"], st.slots["date"], st.slots["time_slot"])
                    if not hold.get("success"):
                        await _speak(session, booking_reply(st.slots, hold, lang), lang)
                        st.slots.pop(_field_to_reclear_on_hold_failure(hold.get("reason")), None)
                        return
                    st.hold_token, st.hold_doctor_id = hold["hold_token"], hold["doctor_id"]

                # KCD-368: a caller may spell a name unprompted, or after
                # being offered it below -- either way, letters heard this
                # turn are assembled and used before anything else is
                # asked, so an eager caller is never told to repeat
                # themselves. Confirmed by readback like every other slot.
                if not st.slots.get("patient_name") and slots.get("spelled_letters"):
                    spelled = merge_spelling(st, slots["spelled_letters"])
                    if spelled:
                        st.slots["patient_name"] = spelled.capitalize()
                        await _speak(session, spelling_readback(spelled, lang), lang)
                        return

                missing = missing_required(st)
                if missing:
                    field_name = missing[0]
                    if field_name == "phone" and st.note_retry("phone") >= 2:
                        # KCD-370: a caller who won't give a number still
                        # gets the booking, told plainly what that costs
                        # them, instead of being blocked here forever.
                        st.phone_declined = True
                    elif field_name == "patient_name" and st.note_retry("patient_name") >= 2:
                        # KCD-368: offered proactively after one failed
                        # capture, not only when the caller asks for it.
                        await _speak(session, spelling_prompt(lang), lang)
                        return
                    else:
                        await _speak(session, _next_question(session, st, intent, missing, lang), lang)
                        return
                    missing = missing_required(st)
                    if missing:
                        await _speak(session, _next_question(session, st, intent, missing, lang), lang)
                        return

                # Skipped when no real phone exists to check by -- the
                # "not_provided" sentinel must never be used as a lookup
                # key, or two different callers who both declined to give
                # a number would appear to conflict with EACH OTHER.
                if effective_phone(st) != "not_provided" and not st.slots.get("_conflict_checked"):
                    conflict = await _tools.booking_conflict(
                        effective_phone(st), st.slots["date"], st.slots["time_slot"]
                    )
                    st.slots["_conflict_checked"] = "1"
                    if conflict.get("conflict"):
                        await _speak(session, conflict_reply(conflict["existing"], lang), lang)
                        return

                if is_ready_to_confirm(st):
                    mark_confirming(st)
                    if st.phone_declined:
                        await _speak(session, phrase("no_confirmation_number", lang), lang)
                    await _speak(session, booking_confirmation_readback(st.slots, "book_appointment", lang), lang)

            elif intent == "book_test":
                # A lab test is paid and booked at the counter, never over the call (clinic-api/booking_service.
                # send_payment_link's docstring says the same from the other side: a lab test's confirmation_id is
                # never found there). No booking task is entered -- there is nothing to collect slots for.
                await _speak(session, phrase("lab_test_counter_only", lang), lang)

            elif intent == "reschedule_appointment":
                st = _enter_task(session, "reschedule_appointment")
                prior_slots = dict(st.slots)
                changed = merge_slots(st, slots)
                ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
                if ack:
                    await _speak(session, ack, lang)
                else:
                    await _thank_for_answer(session, st, changed, prior_slots, lang)
                    await _echo_new_slots(session, changed, prior_slots, st.slots, lang)
                if not st.slots.get("confirmation_id") and slots.get("phone"):
                    found = await _tools.lookup_bookings(phone=slots["phone"])
                    bookings = found.get("bookings") or []
                    if len(bookings) > 1:
                        # CodeRabbit-flagged: never silently act on
                        # bookings[0] -- a caller with several bookings
                        # names which one before anything proceeds.
                        await _speak(session, multiple_bookings_reply(bookings, lang), lang)
                        return
                    if bookings:
                        st.slots["confirmation_id"] = bookings[0]["confirmation_id"]
                missing = missing_required(st)
                if missing:
                    await _speak(session, missing_slot_prompt(intent, missing[0], lang), lang)
                    return
                if is_ready_to_confirm(st):
                    mark_confirming(st)
                    await _speak(session, booking_confirmation_readback(st.slots, "reschedule_appointment", lang), lang)

            elif intent == "cancel_appointment":
                st = _enter_task(session, "cancel_appointment")
                prior_slots = dict(st.slots)
                changed = merge_slots(st, slots)
                ack = correction_acknowledgement(changed, prior_slots, st.slots, lang)
                if ack:
                    await _speak(session, ack, lang)
                else:
                    await _thank_for_answer(session, st, changed, prior_slots, lang)
                    await _echo_new_slots(session, changed, prior_slots, st.slots, lang)
                if not st.slots.get("confirmation_id") and slots.get("phone"):
                    found = await _tools.lookup_bookings(phone=slots["phone"])
                    bookings = found.get("bookings") or []
                    if len(bookings) > 1:
                        await _speak(session, multiple_bookings_reply(bookings, lang), lang)
                        return
                    if bookings:
                        st.slots["confirmation_id"] = bookings[0]["confirmation_id"]
                missing = missing_required(st)
                if missing:
                    await _speak(session, missing_slot_prompt(intent, missing[0], lang), lang)
                    return
                if is_ready_to_confirm(st):
                    mark_confirming(st)
                    await _speak(session, booking_confirmation_readback(st.slots, "cancel_appointment", lang), lang)

            elif intent == "add_test_booking":
                # Same counter-only rule as book_test above -- adding a test to an existing booking is still a lab
                # test booking.
                await _speak(session, phrase("lab_test_counter_only", lang), lang)

            elif intent == "lookup_booking":
                if not (slots.get("phone") or slots.get("confirmation_id")):
                    await _speak(session, missing_slot_prompt(intent, "phone", lang), lang)
                    return
                result = await _tools.lookup_bookings(
                    phone=slots.get("phone"), confirmation_id=slots.get("confirmation_id")
                )
                await _speak(
                    session, _with_resume(session, lookup_reply(result.get("bookings") or [], lang), lang), lang
                )

            elif intent == "resend_confirmation":
                confirmation_id = slots.get("confirmation_id")
                if not confirmation_id and slots.get("phone"):
                    found = await _tools.lookup_bookings(phone=slots["phone"])
                    bookings = found.get("bookings") or []
                    if len(bookings) > 1:
                        await _speak(session, multiple_bookings_reply(bookings, lang), lang)
                        return
                    if bookings:
                        confirmation_id = bookings[0]["confirmation_id"]
                if not confirmation_id:
                    await _speak(session, missing_slot_prompt("cancel_appointment", "confirmation_id", lang), lang)
                    return
                result = await _tools.resend_confirmation(confirmation_id)
                await _speak(session, _with_resume(session, resend_reply(result, lang), lang), lang)

            elif intent == "department_query":
                reply = await _answer_enquiry_intent(intent, slots, lang)
                if reply is None:
                    await _speak(session, phrase("unclear", lang), lang)
                    return
                await _finish_enquiry_turn(session, text, lang, intent, slots, reply, data)

        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")


async def _reopen_after_no(session: CallSession, st, lang: str) -> None:
    """The caller refused the confirmation and gave nothing to change: collection reopens, the details captured
    stay, and the first question is asked again (KCD-367)."""
    st.stage = "collecting"
    st.pending_charge_inr = None
    await _speak(
        session,
        missing_slot_prompt(st.action, "doctor_name" if st.action == "book_appointment" else "date", lang),
        lang,
    )


def _followup(session: CallSession, intent: str, slots: dict, text: str, lang: str):
    """The enquiry slots, with the topic of the last few turns filled in when the caller pointed at it or simply
    did not name a different one (agent/enquiry_followup.py). Deterministic; the model decides nothing here."""
    gap = session.turn_count - session.last_enquiry_turn if session.last_enquiry_entities else None
    return resolve_followup_slot(intent, slots, text, lang, session.last_enquiry_entities, gap)


def _next_question(session: CallSession, st, intent: str, missing: list[str], lang: str) -> str:
    """KCD-103: the next question, one field or a group. The caller-state table (`questions_per_turn`) always wins
    over grouping, and a field already asked is asked alone (agent/slot_grouping.py)."""
    fields = slot_grouping.next_fields(st.action, missing, session.policy.questions_per_turn, st.asked_fields)
    grouped = slot_grouping.grouped_prompt(fields, lang) if len(fields) > 1 else None
    if grouped is None:
        fields = fields[:1]
    st.asked_fields.update(fields)
    return grouped or missing_slot_prompt(intent, fields[0], lang)


def _resume_tail(session: CallSession, reply: str, lang: str) -> str:
    """KCD-104: what brings the caller back to a booking left open while they asked something else -- the next
    question while details are still being collected, "shall I confirm it" at the confirmation step. Empty when
    there is no booking worth returning to, when the policy allows no question, or when the reply already asks one."""
    st = session.booking
    if not topic_flow.worth_suspending(st) or not topic_flow.resume_allowed(session.policy.questions_per_turn, reply):
        return ""
    if st.stage in ("confirming", "awaiting_charge_confirm"):
        return topic_flow.confirm_resume_line(lang)
    missing = missing_required(st)
    if not missing:
        return ""
    return topic_flow.resume_line(_next_question(session, st, st.action, missing, lang), lang)


def _with_resume(session: CallSession, reply: str, lang: str) -> str:
    tail = _resume_tail(session, reply, lang)
    return f"{reply} {tail}" if tail else reply


def _enter_task(session: CallSession, action: str):
    """The booking state for `action`. A DIFFERENT task started while one is half-done sets the old one aside
    instead of replacing it (KCD-104); it is offered back once the new task is finished."""
    current = session.booking
    if current is not None and current.action == action:
        return current
    if topic_flow.worth_suspending(current):
        session.suspended = current
    session.booking = new_state(action)
    return session.booking


async def _after_task_finished(session: CallSession, lang: str) -> None:
    """A task just ended (confirmed, refused or aborted). If another was set aside, offer -- never assume -- to
    go back to it."""
    if session.booking is not None or session.suspended is None:
        return
    if not topic_flow.worth_suspending(session.suspended):
        session.suspended = None
        return
    session.awaiting_resume = True
    await _speak(session, topic_flow.offer_resume(session.suspended.action, lang), lang)


async def _continue_resume_offer(session: CallSession, text: str, lang: str) -> bool:
    """The caller's answer to "shall I go back to it?". A yes restores the booking with everything it held; a
    no drops it; anything else lapses the offer and is handled as a fresh turn."""
    session.awaiting_resume = False
    st, session.suspended = session.suspended, None
    if st is None:
        return False
    answer = classify_yes_no(text, lang)
    if answer == "no":
        await _say_history(session, [history_text.ok_anything_else(lang)], lang)
        return True
    if answer != "yes":
        return False
    st.touch()
    session.booking = st
    if st.stage in ("confirming", "awaiting_charge_confirm"):
        await _speak(session, booking_confirmation_readback(st.slots, st.action, lang), lang)
        return True
    missing = missing_required(st)
    if missing:
        await _speak(session, _next_question(session, st, st.action, missing, lang), lang)
    else:
        mark_confirming(st)
        await _speak(session, booking_confirmation_readback(st.slots, st.action, lang), lang)
    return True


async def _handle_booking_confirmation_turn(session: CallSession, text: str, lang: str) -> bool | None:
    """The caller's answer to a "shall I confirm this?" readback --
    resolved deterministically (see the classify_yes_no call site in
    _dispatch_turn), never via the LLM.

    "no" means different things by action: for cancel_appointment there is
    nothing to correct, so "no" simply aborts it (KCD-372's spirit -- the
    caller was asked once with the charge stated, a "no" ends it there).
    For every other action, "no" re-opens collection instead of discarding
    the booking, because the far more common reason a caller says "no"
    here is "something's wrong", not "forget the whole thing" -- and
    KCD-367 explicitly asks for a correction to re-enter slot filling
    rather than restart. An unrecognised answer just repeats the question
    rather than guessing which the caller meant."""
    st = session.booking
    assert st is not None
    answer = classify_yes_no(text, lang)

    if answer is None:
        return False  # KCD-104: not a yes or a no -- the caller handles it as a fresh turn

    if answer == "no":
        _note(session, "corrections")
        if st.action == "cancel_appointment":
            await _speak(session, phrase("cancel_aborted", lang), lang)
            session.booking = None
            return
        if len(text.split()) > 2:
            # KCD-104: "no, make it eleven" is a correction, not just a refusal. Only a BARE no reopens the
            # booking by asking again; anything longer is handled as a fresh turn (the correction is applied and
            # read back; a question is answered; a bare no from the extractor's point of view reopens below).
            session.no_at_confirm = True
            return False
        await _reopen_after_no(session, st, lang)
        return

    # answer == "yes": commit for real.
    try:
        if st.action == "book_appointment":
            phone = effective_phone(st)
            result = await _tools.confirm_booking(
                st.hold_token,
                st.hold_doctor_id,
                st.slots["date"],
                st.slots["time_slot"],
                st.slots["patient_name"],
                phone,
                caller_phone=phone,
                patient_age=st.slots.get("patient_age"),
                relationship=st.slots.get("relationship") or "self",
            )
            if result.get("reason") == "write_unverified":
                session.booking = None
                await _handle_unverified_write(session, lang)
                return
            await _speak(session, booking_reply(st.slots, result, lang), lang)
            if result.get("success"):
                _record_action(session, "booking_confirmed", result, st.slots)
                await _persist_senior(session, phone)
                session.booking = None
                # Only a DOCTOR appointment is paid and booked over the call at all; having reached here, this one
                # just was. The agent calls the clinic API to send a payment link, then ends the call -- it never
                # takes or states a payment itself (CLAUDE.md's truth boundary keeps money out of its hands the same
                # way it keeps every other fact out of the model's). A failure sending the link is logged, never
                # spoken over the confirmation the caller already heard: the appointment IS still confirmed either
                # way, and the closing line says the link was sent regardless -- if it genuinely failed to queue,
                # that is a clinic-api-side follow-up, not something the caller can act on by being told mid-call.
                confirmation_id = result.get("confirmation_id")
                if confirmation_id:
                    with contextlib.suppress(ToolCallError):
                        await _tools.send_payment_link(confirmation_id)
                session.suspended = None  # nothing is offered back after the socket closes below
                await _end_call(session, lang, "payment_link_sent")
                return
            if result.get("reason") != "hold_expired":
                session.booking = None
            else:
                st.hold_token, st.stage = None, "collecting"  # let a retry re-hold

        elif st.action == "book_test":
            phone = effective_phone(st)
            result = await _tools.book_tests(
                st.test_names,
                st.slots["date"],
                st.slots["patient_name"],
                phone,
                caller_phone=phone,
                patient_age=st.slots.get("patient_age"),
                relationship=st.slots.get("relationship") or "self",
            )
            await _speak(session, multi_test_reply(result, lang), lang)
            _record_action(session, "tests_booked", result, st.slots)
            session.booking = None

        elif st.action == "reschedule_appointment":
            result = await _tools.reschedule_appointment(
                st.slots["confirmation_id"], st.slots["new_date"], st.slots["new_time_slot"]
            )
            if result.get("reason") == "write_unverified":
                session.booking = None
                await _handle_unverified_write(session, lang)
                return
            await _speak(session, reschedule_reply(result, lang), lang)
            _record_action(session, "appointment_rescheduled", result, st.slots)
            if result.get("success") or result.get("reason") != "slot_taken":
                session.booking = None
            else:
                st.stage = "collecting"
                st.slots.pop("new_time_slot", None)

        elif st.action == "cancel_appointment":
            confirm_charge = st.stage == "awaiting_charge_confirm"
            result = await _tools.cancel_appointment(st.slots["confirmation_id"], confirm_charge)
            if not result.get("success") and result.get("reason") == "charge_confirmation_required":
                mark_awaiting_charge_confirm(st, result["charge_inr"])
                await _speak(session, cancel_reply(result, lang), lang)
                return
            await _speak(session, cancel_reply(result, lang), lang)
            _record_action(session, "appointment_cancelled", result, st.slots)
            session.booking = None

        elif st.action == "add_test_booking":
            result = await _tools.add_test_to_booking(st.slots["confirmation_id"], st.slots["test_name"])
            await _speak(session, add_test_reply(result, lang), lang)
            _record_action(session, "test_added", result, st.slots)
            session.booking = None

    except ToolCallError as e:
        logger.error("[%s] clinic API call failed during booking confirmation: %s", session.call_id, e)
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        session.booking = None


async def _resync_after_playback(session: CallSession) -> bool:
    """Drop everything captured while the agent was talking, by moving
    processed_until_s to the current end of the decoded buffer. That region
    is muted silence from the client's side; skipping it keeps the turn
    detector from ever analysing it, and -- more importantly -- keeps
    processed_until_s anchored to real time instead of drifting a full
    reply behind, which is what made later turns surface late."""
    buffer_end_s = session.audio.duration_s
    session.processed_until_s = max(session.processed_until_s, buffer_end_s - RESYNC_REWIND_S)
    session.resync_pending = False
    logger.info("[%s] resynced to %.2fs after playback", session.call_id, session.processed_until_s)
    return True


def _next_wake_delay(decision) -> float:
    """KCD-049: sleep until the moment the answer could change, not until the
    next tick of a fixed timer. While the caller has spoken this turn the cadence
    is ACTIVE_POLL_INTERVAL_S; when the detector says "commit in x seconds if
    silence continues", wake exactly then (plus a hair, so the audio that
    settles it has landed)."""
    base = ACTIVE_POLL_INTERVAL_S if decision.had_any_speech else POLL_INTERVAL_S
    if decision.commit_after_s is None:
        return base
    return min(base, max(MIN_WAKE_S, decision.commit_after_s + WAKE_SLACK_S))


async def _speculative_completeness(session: CallSession, speech_end_abs: float) -> str | None:
    """KCD-048: transcribe what has been said so far, in the call's current
    language only (no LID, no router state touched), and say whether it reads as
    finished. Any failure is simply "no verdict" -- the baseline threshold then
    applies, so this can only ever shorten a wait it is confident about."""
    session.spec_count += 1
    wav = None
    try:
        session.utt_seq += 1
        wav = await _slice_utterance(session, session.processed_until_s, speech_end_abs, session.utt_seq)
        asr = await asyncio.wait_for(_asr_router.transcribe(session.lang, wav), SPECULATIVE_ASR_TIMEOUT_S)
        # an emergency phrase in what has been said so far ends the wait at once: the turn is treated as
        # finished so the notice is not delayed by a silence timer (agent/emergency.py)
        verdict = "complete" if detect_emergency(asr.text) else classify_completeness(asr.text, session.lang)
    except Exception as e:  # noqa: BLE001 - speculation is advisory
        logger.info("[%s] speculative transcription unavailable: %s", session.call_id, e)
        return None
    finally:
        if wav:
            with contextlib.suppress(OSError):
                os.remove(wav)
    session.spec_end_abs, session.spec_verdict = speech_end_abs, verdict
    return verdict


async def _decide_turn(session: CallSession, tail, sr: int):
    """Where is the speech (Silero), then has the caller finished
    (agent/endpointing.decide, pure and tested off-pod). With SEMANTIC_ENDPOINTING
    a candidate turn end may trigger one speculative transcription, whose
    completeness verdict is reused for as long as the speech end has not moved."""
    spans, duration_s = await asyncio.to_thread(_turn_detector.spans, tail, sr)
    cfg = session.pause.config(_turn_detector.config) if ADAPTIVE_PAUSE != "off" else _turn_detector.config
    if spans and not session.commit_checked and session.last_commit_end_abs is not None:
        # speech resumed after the last turn was cut: was that cut too early for THIS caller?
        resumed_after = max(0.0, float(spans[0]["start"]) - session.last_confirm_s)
        session.pause.note_turn_cut(resumed_after, session.agent_spoke_since_commit)
        session.commit_checked = True
    speech_end_abs = session.processed_until_s + float(spans[-1]["end"]) if spans else None
    completeness = None
    if (
        SEMANTIC_ENDPOINTING
        and speech_end_abs is not None
        and session.spec_end_abs is not None
        and abs(session.spec_end_abs - speech_end_abs) < SPECULATION_MATCH_TOLERANCE_S
    ):
        completeness = session.spec_verdict
    decision = decide_turn_end(spans, duration_s, cfg, completeness=completeness, semantic=SEMANTIC_ENDPOINTING)
    if decision.speculate and speech_end_abs is not None and session.spec_count < MAX_SPECULATIONS_PER_TURN:
        verdict = await _speculative_completeness(session, speech_end_abs)
        if verdict is not None:
            decision = decide_turn_end(spans, duration_s, cfg, completeness=verdict, semantic=True)
    if decision.utterance_end_s is not None:
        session.pause.observe_utterance(spans, upto_s=decision.utterance_end_s)
        session.last_confirm_s = decision.confirm_s or cfg.silence_confirm_s
        session.last_commit_end_abs = session.processed_until_s + float(decision.utterance_end_s)
        session.agent_spoke_since_commit = False
        session.commit_checked = False
    return decision


async def _interrupt_playback(session: CallSession, reason: str, resume_from_s: float | None = None) -> None:
    """The caller has taken the floor -- by clicking (manual) or by talking over
    the agent (acoustic barge-in). Stop what is playing on the client, discard
    what the interrupted turn has not yet said, open the gate.

    `resume_from_s`, when given, is where the caller's utterance begins on the
    call timeline: the marker is set there and the usual post-playback resync
    (which would skip past their first words) is cancelled."""
    was_speaking = session.agent_speaking
    if was_speaking:
        _note(session, "barge_ins")  # the caller talked over the agent
    session.speak_epoch += 1
    session.release_gate()
    if session.duplex is not None:
        session.duplex.cancel_pending()
    if reason != "manual":
        # A manual interrupt comes FROM the client, which has already stopped
        # its own playback; only a server-detected one needs to tell it to.
        with contextlib.suppress(Exception):
            await session.ws.send_text(json.dumps({"type": "stop_playback"}))
    if resume_from_s is not None:
        session.processed_until_s = max(session.processed_until_s, resume_from_s)
        session.resync_pending = False
    if was_speaking:
        logger.info("[%s] caller interrupt (%s) -- playback stopped, gate released", session.call_id, reason)
        barge_in_interrupts.record(reason, session.lang)


def _mark_verified(session: CallSession, method: str, patient_ref: str, analysis: dict | None) -> None:
    """Record that a check passed (an OTP, a confirmed date of birth -- whatever the clinic
    decides counts) and remember THIS voice as the verified one (KCD-054). The mechanism that
    verifies lives outside this file (KCD-203); it calls this when it succeeds."""
    session.identity.verify(method, patient_ref)
    session.speaker.reset()
    emb = (analysis or {}).get("voice")
    if emb is not None:
        level = ((analysis or {}).get("near_end") or (None,))[0]
        session.speaker.enroll_embedding(emb, level)


async def _check_speaker_change(session: CallSession, analysis: dict | None, lang: str) -> None:
    if not session.identity.is_verified or not session.speaker.enrolled or not analysis:
        return
    emb = analysis.get("voice")
    if emb is None:
        return  # too little voiced speech: abstain, never guess
    level = (analysis.get("near_end") or (None,))[0]
    result = session.speaker.observe_embedding(emb, level)
    if result.verdict == "changed" and session.identity.revoke("speaker_changed"):
        logger.warning("[%s] voice changed after verification -- verification revoked", session.call_id)
        audio_issue_buckets.record("speaker_change", "verification_revoked", lang)
        await _speak(session, phrase("reverify_notice", lang), lang)


async def _say_history(session: CallSession, statements: list[tuple[str, str]], lang: str, warm: bool = False) -> None:
    """Speak history statements and record their provenance ids (KCD-500/501). `warm` marks a real answer
    (not a question): it gets the thanks, and a senior's kind closing (KCD-512/513)."""
    _rec(session, "history", [sid for sid, _ in statements])
    text = " ".join(t for _, t in statements)
    if warm:
        text = session.kindness.decorate(text, lang)
    await _speak(session, text, lang)


def _digits(value) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _caller_phone(session: CallSession) -> str:
    return session.identity.phone or "not_provided"


async def _history_step(session: CallSession, lang: str) -> None:
    """Where the history question stands: find the record, verify who is asking, answer."""
    ident = session.identity
    if ident.patient_ref is None:
        session.history_state = "need_phone"
        await _say_history(session, [history_text.ask_phone(lang)], lang)
        return
    decision = patient_policy.history_gate(ident, ident.patient_ref, lang)
    if decision.allowed:
        await _answer_history(session, lang)
        return
    if SECURITY_QUESTIONS != "off":
        await _start_security_check(session, ident.patient_ref, lang)
        return
    # NOTHING beyond the existence of a record is spoken before verification. The refusal offers a
    # person; the caller's yes/no is handled at the top of the next turn.
    session.history_state, session.pending_history, session.awaiting_handoff_offer = None, None, True
    await _say_history(session, [decision.statement], lang)


async def _start_security_check(session: CallSession, patient_ref: str | None, lang: str) -> None:
    """Begin the security questions (KCD-495). With no record found yet (`patient_ref` None) the same
    answers first LOCATE the record (KCD-497), then verify it."""
    session.security = SecurityCheck(patient_ref)
    session.history_state = "security"
    text = session.security.next_question(lang)
    if patient_ref is None:
        text = f"{history_text.no_record(lang)[1]} {text}"  # the operator's "I cannot find it" wording first
    await _say_history(session, [("security_question", text)], lang)


async def _continue_security_flow(session: CallSession, text: str, lang: str, analysis: dict | None) -> bool:
    """One caller answer to a security question. True if this turn belonged to the check."""
    chk = session.security
    if chk is None or chk.finished:
        return False
    if not chk.absorb(text):
        if chk.gave_up_reading:
            session.security, session.history_state = None, None
            await _handoff_to_human(session, "verification_unclear", (lang,))
            return True
        await _say_history(
            session,
            [("security_not_understood", f"{sec_text.did_not_understand(lang)} {chk.next_question(lang)}")],
            lang,
        )
        return True
    await _thank(session, lang)
    if not chk.ready_to_submit():
        await _say_history(session, [("security_question", chk.next_question(lang))], lang)
        return True
    try:
        if chk.finding:
            found = (
                await _tools.find_patient(session.call_id, chk.find_payload())
                if chk.find_payload()
                else {"status": "insufficient"}
            )
            if found.get("status") == "single":
                chk.patient_ref = str(found["patient_ref"])
                session.identity.claim(chk.patient_ref, session.identity.phone)
            elif found.get("status") == "insufficient":
                await _say_history(session, [("security_question", chk.next_question(lang))], lang)
                return True
            else:
                # none or ambiguous: never say which. The operator's "I cannot find that" wording, then ask again.
                stop = chk.record_find_miss()
                if stop:
                    session.security, session.history_state = None, None
                    await _say_history(session, [history_text.cannot_see(lang)], lang)
                    await _handoff_to_human(session, "history_record_not_found", (lang,))
                    return True
                await _say_history(
                    session, [history_text.cannot_see(lang), ("security_question", chk.next_question(lang))], lang
                )
                return True
        result = await _tools.verify_patient(
            session.call_id, int(chk.patient_ref), chk.payload(), session.identity.phone
        )
    except ToolCallError as e:
        logger.error("[%s] verification failed to run: %s", session.call_id, e)
        session.security, session.history_state = None, None
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return True
    chk.record_result(result)
    if chk.verified:
        await _on_verified(session, chk, analysis, lang)
        return True
    if chk.finished:
        session.security, session.history_state = None, None
        await _say_history(session, [("security_failed", sec_text.failed_text(lang))], lang)
        await _handoff_to_human(session, "verification_failed", (lang,))
        return True
    await _say_history(
        session, [("security_not_matched", f"{sec_text.not_matched(lang)} {chk.next_question(lang)}")], lang
    )
    return True


async def _on_verified(session: CallSession, chk: SecurityCheck, analysis: dict | None, lang: str) -> None:
    """The server said yes (KCD-495): remember this voice, switch to senior care if the registered date
    of birth says so (KCD-512), offer to continue anything unfinished (KCD-496), then answer."""
    ref = chk.patient_ref
    _mark_verified(session, "security_questions", str(ref), analysis)
    _rec(session, "patient", int(ref))
    session.security, session.history_state = None, None
    parts = [("security_verified", sec_text.verified_text(lang))]
    if session.kindness.activate(chk.age_years):
        apply_caller_state(session.call_state, senior=True)  # slower, one question at a time (Appendix C)
        opening = session.kindness.opening(lang)
        if opening:
            parts.append(("senior_opening", opening))
    await _say_history(session, parts, lang)
    await _offer_unfinished(session, lang)
    if session.pending_history is not None:
        await _answer_history(session, lang)
    if not session.awaiting_draft_answer:
        await _offer_preferences(session, lang)


def _draft_offer_text(draft: dict, lang: str) -> str:
    what = {
        "book_appointment": {"bn": "একটা অ্যাপয়েন্টমেন্ট", "hi": "एक अपॉइंटमेंट", "en": "an appointment"},
        "book_test": {"bn": "একটা টেস্ট", "hi": "एक टेस्ट", "en": "a test"},
        "reschedule_appointment": {
            "bn": "অ্যাপয়েন্টমেন্টের সময় বদল",
            "hi": "अपॉइंटमेंट का समय बदलना",
            "en": "changing an appointment",
        },
        "cancel_appointment": {"bn": "অ্যাপয়েন্টমেন্ট বাতিল", "hi": "अपॉइंटमेंट रद्द करना", "en": "cancelling an appointment"},
        "add_test_booking": {"bn": "একটা টেস্ট যোগ করা", "hi": "एक टेस्ट जोड़ना", "en": "adding a test"},
    }
    label = (what.get(draft.get("action")) or what["book_appointment"]).get(lang) or ""
    return {
        "bn": f"আগের বার আপনি {label} করছিলেন, কাজটা শেষ হয়নি। সেটা কি চালিয়ে যাব, নাকি নতুন করে শুরু করব?",
        "hi": f"पिछली बार आप {label} कर रहे थे, वह पूरा नहीं हुआ। क्या उसे जारी रखूँ, या नए सिरे से शुरू करूँ?",
        "en": f"Last time you were doing {label}, and it was not finished. Shall I continue it, or start fresh?",
    }.get(lang, "")


async def _offer_unfinished(session: CallSession, lang: str) -> None:
    """KCD-496: after verification, offer -- never assume -- to continue an unfinished booking."""
    ident = session.identity
    if session.booking is not None or _tools is None:
        return
    try:
        ctx = await _tools.continuity(_caller_phone(session), session.call_id, int(ident.patient_ref))
    except ToolCallError:
        return
    draft = ctx.get("draft")
    if not draft:
        return
    try:
        info = json.loads(draft.get("slots_json") or "{}")
    except ValueError:
        return
    if not info.get("action"):
        return
    session.pending_draft, session.awaiting_draft_answer = info, True
    await _say_history(session, [("continuity_offer", _draft_offer_text(info, lang))], lang)


async def _offer_preferences(session: CallSession, lang: str) -> None:
    """KCD-499: the patient's stored preferences are OFFERED for confirmation (the address is never read
    aloud), and a stored language bias never overrides what the caller actually says."""
    ident = session.identity
    if _tools is None or ident.patient_ref is None:
        return
    try:
        prefs = await _tools.get_preferences(int(ident.patient_ref), _caller_phone(session))
    except ToolCallError:
        return
    offer = patient_policy.preference_offer(prefs.get("preferences") or {}, ident, ident.patient_ref, lang)
    if offer is None:
        return
    session.pending_pref, session.awaiting_pref_answer = offer, True
    await _speak(session, offer.text, lang)


async def _continue_preference_offer(session: CallSession, text: str, lang: str) -> bool:
    if not session.awaiting_pref_answer or session.pending_pref is None:
        return False
    session.awaiting_pref_answer = False
    offer, session.pending_pref = session.pending_pref, None
    answer = classify_yes_no(text, lang)
    if answer == "yes":
        session.confirmed_preferences = dict(offer.fields)
        _rec(session, "confirmed", dict(offer.fields))
        if offer.fields.get("accessibility_mode") == "slower":
            apply_caller_state(session.call_state, senior=True)  # the slower, one-question-at-a-time delivery
        await _speak(session, history_text.ok_anything_else(lang)[1], lang)
        return True
    if answer == "no":
        await _speak(session, history_text.ok_anything_else(lang)[1], lang)
        return True
    return False


async def _continue_draft_offer(session: CallSession, text: str, lang: str) -> bool:
    """The caller's answer to "continue it, or start fresh?"."""
    if not session.awaiting_draft_answer or session.pending_draft is None:
        return False
    session.awaiting_draft_answer = False
    info, session.pending_draft = session.pending_draft, None
    answer = classify_yes_no(text, lang)
    if answer == "yes":
        st = new_state(info["action"])
        merge_slots(st, info.get("slots") or {})
        for t in info.get("test_names") or []:
            if t not in st.test_names:
                st.test_names.append(t)
        session.booking = st
        missing = missing_required(st)
        if missing:
            await _speak(session, missing_slot_prompt(info["action"], missing[0], lang), lang)
        else:
            await _speak(session, phrase("unclear", lang), lang)
        return True
    if answer == "no":
        await _say_history(session, [history_text.ok_anything_else(lang)], lang)
        return True
    return False


async def _answer_history(session: CallSession, lang: str) -> None:
    hq, ident = session.pending_history, session.identity
    now = datetime.datetime.now()
    phone = _caller_phone(session)
    try:
        timeline = await _tools.patient_timeline(int(ident.patient_ref), phone, session.call_id)
        if hq.kind == "last_test":
            catalogue = _fast_path.catalogue if _fast_path is not None else None
            name, _form, score = (
                catalogue.match(hq.test_hint or "", "test", lang, HISTORY_TEST_MATCH_MIN)
                if catalogue
                else (None, None, 0.0)
            )
            if not name or score < HISTORY_TEST_MATCH_MIN:
                session.history_state = "need_test"
                await _say_history(session, [history_text.ask_which_test(lang)], lang)
                return
            status = await _tools.patient_test_status(int(ident.patient_ref), name, phone, session.call_id)
            answer = patient_policy.answer_last_test(
                timeline, name, status if status.get("success") else None, lang, now
            )
        elif hq.kind == "appointments":
            answer = patient_policy.answer_appointments(timeline, lang, now)
        elif hq.kind == "medicines":
            answer = patient_policy.answer_medicines(timeline, lang, now)
        else:
            answer = patient_policy.answer_recent_tests(timeline, lang, now)
    except ToolCallError as e:
        logger.error("[%s] history lookup failed: %s", session.call_id, e)
        session.history_state, session.pending_history = None, None
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return
    session.history_state, session.pending_history = None, None
    await _say_history(session, answer.statements, lang, warm=True)


async def _continue_history_flow(session: CallSession, text: str, lang: str) -> bool:
    """True if this turn belonged to the history flow (and has been handled)."""
    if session.awaiting_handoff_offer:
        session.awaiting_handoff_offer = False
        answer = classify_yes_no(text, lang)
        if answer == "yes":
            await _handoff_to_human(session, "history_requested_staff", (lang,))
            return True
        if answer == "no":
            await _say_history(session, [history_text.ok_anything_else(lang)], lang)
            return True
        return False  # neither: the offer lapses and the turn is handled normally
    state = session.history_state
    if state == "need_test":
        session.pending_history.test_hint, session.history_state = text, None
        await _history_step(session, lang)
        return True
    if state not in ("need_phone", "need_name"):
        return False
    try:
        data = await _resolve_intent(session, text, lang)
    except ExtractionError:
        data = {"slots": {}}
    slots = data.get("slots") or {}
    session.history_attempts += 1
    try:
        if state == "need_phone":
            phone = _digits(slots.get("phone") or slots.get("contact_phone"))
            if len(phone) < 10:
                if session.history_attempts >= 2:
                    session.history_state = None
                    await _handoff_to_human(session, "history_phone_unclear", (lang,))
                else:
                    await _say_history(session, [history_text.ask_phone(lang)], lang)
                return True
            await _thank(session, lang)
            result = await _tools.identify_patient(phone)
            outcome = patient_policy.identify_outcome(result)
            session.identity.phone = session.identity.phone or phone
            if outcome == "new":
                # nobody on this number: the caller's own details can locate the record (KCD-497)
                if SECURITY_QUESTIONS != "off":
                    await _start_security_check(session, None, lang)
                else:
                    session.history_state, session.pending_history = None, None
                    await _say_history(session, [history_text.no_record(lang)], lang)
            elif outcome == "single":
                session.identity.claim(str(result["patient_ref"]), phone)
                _rec(session, "patient", result["patient_ref"])
                session.history_state = None
                await _history_step(session, lang)
            else:
                session.history_state, session.history_attempts = "need_name", 0
                await _speak(session, patient_policy.resolve_question(lang, 0), lang)
            return True
        # state == "need_name": an open question, answered with a NAME (never chosen from a list)
        spoken = slots.get("patient_name") or text
        result = await _tools.resolve_patient(session.identity.phone or "", spoken, slots.get("patient_age"))
        if result.get("status") == "single" and result.get("basis") == "exact":
            session.identity.claim(str(result["patient_ref"]), session.identity.phone)
            _rec(session, "patient", result["patient_ref"])
            session.history_state = None
            await _history_step(session, lang)
        elif session.history_attempts >= 2:
            session.history_state, session.pending_history = None, None
            await _handoff_to_human(session, "history_patient_unclear", (lang,))
        else:
            # none, ambiguous, or found only by SOUND: ask again, never assume
            await _speak(session, patient_policy.resolve_question(lang, 1), lang)
        return True
    except ToolCallError as e:
        logger.error("[%s] patient lookup failed: %s", session.call_id, e)
        session.history_state, session.pending_history = None, None
        await _speak(session, phrase("tool_failure", lang), lang, fallback_reason="tool_failure")
        return True


async def _save_unfinished_draft(session: CallSession) -> None:
    """KCD-496: a booking that was being collected when the call ended is saved, so the next call can
    offer to continue it. Bounded, never raises."""
    st = session.booking
    if st is None or st.stage != "collecting" or not (st.slots or st.test_names) or _tools is None:
        return
    phone = st.slots.get("phone") or st.slots.get("contact_phone") or session.identity.phone
    if not phone:
        return
    try:
        body = json.dumps({"action": st.action, "slots": st.slots, "test_names": st.test_names})
        await asyncio.wait_for(_tools.save_draft_booking(phone, session.call_id, body), 3.0)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] could not save the unfinished booking: %s", session.call_id, e)


def _caller_thanked(text: str, lang: str) -> bool:
    from agent import fast_path_cues as _cues

    table = _cues.table_for(lang)
    return bool(table and table.any(_cues.normalise_cue(text), table.thanks))


def _resolve_reply_language(session: CallSession, detected_lang: str) -> str:
    """The language to REPLY in this turn (KCD-lay-terms, a live call 2026-09-28) -- locked to the first language
    established for the call, not re-derived from this turn's own language ID. `detected_lang` is what THIS turn's
    LID/ASR-verification actually settled on (used to select the ASR engine already, by the time this runs) --
    that recognition is unaffected; only which language the agent ANSWERS in is locked.

    Unlocks (moves session.lang) only for: the first turn (nothing locked yet -- CallSession.lang_locked), or an
    explicit request, handled where it is detected (agent/language_switch.detect_language_switch_request's call
    site calls _lock_reply_language directly, since that decision is made later in the same turn than this
    function runs). A detected different voice after identity verification revokes verification and asks to
    re-verify (agent/speaker_change.py) but does not, on its own, reopen the language lock -- a new voice
    re-verifying is free to explicitly ask for a different language like any caller can."""
    if not session.lang_locked:
        _lock_reply_language(session, detected_lang)
    session.lang_router.note_response_language(session.lang)
    apply_language(session.call_state, session.lang)
    return session.lang


def _lock_reply_language(session: CallSession, lang: str) -> None:
    session.lang, session.lang_locked = lang, True


def _keep_language(session: CallSession, prior_lang: str) -> str:
    """A turn we could not read (empty, jumbled) says nothing about which language the caller speaks: the call stays in the
    language it was in, and is answered in it. Before this, a swear word or a mumble that language ID happened to label
    Hindi moved a Bengali call to Hindi for the re-ask."""
    session.lang = prior_lang
    session.lang_router.note_response_language(prior_lang)
    apply_language(session.call_state, prior_lang)
    return prior_lang


def _abuse_reply_language(text: str, lang: str, prior_lang: str) -> str:
    """The language to answer abuse in: the script it was written in (Bengali or Devanagari), else the call's own."""
    from agent.lang_select import script_share

    for candidate in ("bn", "hi"):
        if script_share(text, candidate) >= 0.5:
            return candidate
    return prior_lang if prior_lang in _languages_active else lang


async def _end_call(session: CallSession, lang: str, key: str) -> None:
    """Say the closing line, let it play, then close the connection."""
    session.turn_epoch = session.speak_epoch
    duration = await _speak(session, phrase(key, lang), lang)
    await asyncio.sleep(min(duration, 8.0) + 0.3)
    with contextlib.suppress(Exception):
        await session.ws.close()


async def _handle_silence(session: CallSession, quiet: bool) -> bool:
    """The caller has said nothing since the agent stopped speaking. After SILENCE_PROMPT_S ask once whether there is
    anything else (one question; the call ends if not); if still nothing SILENCE_CLOSE_S later, end the call.
    True when the call was ended. `quiet` is False whenever there is speech, or the agent is mid-turn."""
    if not quiet or session.dispatch_lock.locked():
        session.silence_since = None
        return False
    now = time.monotonic()
    if session.silence_since is None:
        session.silence_since = now
        return False
    waited = now - session.silence_since
    if session.silence_prompts == 0 and waited >= SILENCE_PROMPT_S:
        session.silence_prompts = 1
        _note(session, "silence_prompts")
        session.silence_since = None
        session.awaiting_close_answer = True
        session.turn_epoch = session.speak_epoch
        logger.info("[%s] %.1fs of silence -- asking whether there is anything else", session.call_id, waited)
        await _speak(session, phrase("silence_prompt", session.lang), session.lang)
        return False
    if session.silence_prompts >= 1 and waited >= SILENCE_CLOSE_S:
        logger.info("[%s] still silent after the prompt -- ending the call", session.call_id)
        session.outcome = "silence_timeout"
        _note(session, "silence_timeout")
        await _end_call(session, session.lang, "idle_close")
        return True
    return False


async def _handle_barge_in(session: CallSession, event) -> None:
    sr = session.audio.sample_rate if hasattr(session, "audio") else 16000
    await _interrupt_playback(session, "acoustic", resume_from_s=max(0.0, event.at_sample / sr - BARGE_IN_PREROLL_S))


async def _turn_poll_loop(session: CallSession):
    """Runs for the lifetime of the call. Every POLL_INTERVAL_S, re-decodes
    the growing buffer -- ALWAYS from byte 0, since that's the only way
    the WebM container stays valid -- then asks the turn detector "is the
    caller done talking yet?" using only the slice of audio past
    session.processed_until_s (a prior turn's already-consumed audio).
    On yes: slice that utterance out for ASR, hand it to _dispatch_turn as
    a background task (so ingestion of the NEXT turn's audio is never
    blocked by this turn's ASR/LLM/TTS work), and advance the marker.
    """
    while True:
        await asyncio.sleep(session.take_wake_delay())

        if time.time() - session.last_activity > IDLE_TIMEOUT_S:
            logger.info("[%s] idle timeout, closing", session.call_id)
            session.turn_epoch = session.speak_epoch
            await _speak(session, phrase("idle_close", session.lang), session.lang)
            with contextlib.suppress(Exception):
                await session.ws.close()
            return

        if time.time() - session.last_heartbeat > HEARTBEAT_INTERVAL_S:
            session.last_heartbeat = time.time()
            with contextlib.suppress(Exception):
                await session.ws.send_text('{"sender":"_ping","text":""}')

        # --- half-duplex gate: never run turn detection on our own voice ---
        backstops_before = session.gate.backstop_releases
        if session.gate.blocked():
            session.silence_since = None  # the agent is talking: the quiet starts when it stops
            continue
        if session.gate.backstop_releases != backstops_before:
            logger.warning("[%s] no playback-done from client, released gate on deadline", session.call_id)

        if session.resync_pending:
            await _resync_after_playback(session)
            continue

        sr = session.audio.sample_rate
        tail = session.audio.tail_tensor(session.processed_until_s)
        if tail.numel() < int(0.2 * sr):
            continue  # not enough new audio to judge yet -- not an error

        result = await _decide_turn(session, tail, sr)
        session.next_wake_s = _next_wake_delay(result)
        if result.utterance_end_s is None:
            if await _handle_silence(session, result.reason in ("no_speech", "too_little_audio")):
                return
            continue
        session.reset_speculation()

        absolute_end_s = session.processed_until_s + result.utterance_end_s
        session.utt_seq += 1
        utterance_wav = await _slice_utterance(
            session,
            session.processed_until_s,
            absolute_end_s,
            session.utt_seq,
        )
        session.processed_until_s = absolute_end_s
        asyncio.create_task(_dispatch_turn(session, utterance_wav))


async def _handle_control(session: CallSession, raw: str):
    """Client -> server control channel.

    "playback_done" is the load-bearing half of the echo gate: the server
    cannot otherwise know when the caller's speaker actually stopped.

    "interrupt" (KCD-464) is the manual stop affordance: static/index.html
    shows a button while agent_speaking, and clicking it stops local
    playback, unmutes the mic and sends this immediately, WITHOUT waiting
    for the queued clips to finish. Deliberately NOT acoustic barge-in --
    this file's own module docstring already explains why that needs an
    echo canceller this system does not have (no reference signal for
    audio synthesized locally and played through Web Audio). This is the
    honest, buildable version of "stop the AI, let the caller talk": the
    caller asks for the floor instead of the system trying to detect it
    from the muted mic, which structurally cannot see it."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[%s] unparseable control frame: %r", session.call_id, raw[:80])
        return
    if msg.get("type") == "playback_done":
        session.release_gate()
    elif msg.get("type") == "hello":
        # The browser may refuse the 16kHz AudioContext we ask for. Trust
        # what the client reports over what we requested: a wrong assumed
        # rate would not fail loudly, it would just make every timestamp
        # and every transcript quietly wrong.
        rate = int(msg.get("sampleRate") or SAMPLE_RATE)
        session.declared_rate = rate
        session.audio.sample_rate = rate
        if rate != SAMPLE_RATE and session.duplex is not None:
            logger.warning("[%s] client at %dHz: echo cancellation needs 16kHz, disabled for this call",
                           session.call_id, rate)
            session.duplex = None
        if rate != SAMPLE_RATE:
            logger.warning("[%s] client capturing at %dHz, not %dHz -- resampling per utterance",
                           session.call_id, rate, SAMPLE_RATE)
        logger.info("[%s] transport: %s @ %dHz", session.call_id,
                    msg.get("format", "pcm_s16le"), rate)
    elif msg.get("type") == "interrupt":
        await _interrupt_playback(session, "manual")


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    session = CallSession(ws)

    # The door. A call the AI cannot serve goes to a person NOW -- never
    # queued behind the GPU. See agent/admission.py for what closes it.
    admission = _admission.try_admit()
    if not admission.admitted:
        logger.warning("[%s] call refused: %s", session.call_id, admission.reason)
        try:
            await _handoff_to_human(session, admission.reason, languages=HANDOFF_ALL_LANGUAGES)
        finally:
            session.cleanup()
        return
    session.admission = admission
    logger.info("[%s] call started (%d active)", session.call_id, _admission.active_calls)
    await _refresh_messages()
    session.recorder = CallRecorder(session.call_id, _write_call_event)
    _rec(session, "start", "voice", disclosure_version_label())
    poll_task = asyncio.create_task(_turn_poll_loop(session))

    try:
        if session.duplex is not None:
            # Tell the client to keep the microphone LIVE during playback and to
            # honour stop_playback: without this it mutes capture as before and
            # the canceller simply sees silence (safe, just no barge-in).
            await session.ws.send_text(json.dumps({"type": "config", "aec": True}))
        await _speak(session, phrase("greeting", "bn"), "bn")
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session.append(message["bytes"])
            elif message.get("text"):
                await _handle_control(session, message["text"])
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[%s] session crashed", session.call_id)
    finally:
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        if session.recorder is not None:
            # KCD-501: everything completed was already written as it happened; this adds the outcome
            # and drains anything still queued, within the story's thirty seconds. A miss is logged.
            outcome = session.outcome if session.turn_count else "abandoned"
            with contextlib.suppress(Exception):
                _write_satisfaction(session)
            try:
                left = await asyncio.wait_for(session.recorder.finish(outcome), CALL_RECORD_DEADLINE_S + 3.0)
                if left:
                    logger.error("[%s] call record: %d events could not be written", session.call_id, left)
            except Exception as e:  # noqa: BLE001
                logger.error("[%s] call record could not be closed: %s", session.call_id, e)
        await _save_unfinished_draft(session)
        _admission.release(admission)
        session.cleanup()
        logger.info("[%s] call ended (%d active)", session.call_id, _admission.active_calls)


app.mount("/", StaticFiles(directory="static/pcm", html=True), name="static")
