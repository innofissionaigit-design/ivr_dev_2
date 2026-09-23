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
import datetime
import difflib
import io
import json
import logging
import os
import tempfile
import time
import uuid
import wave

import torchaudio
from agent.pcm_buffer import PcmCallBuffer, SAMPLE_RATE
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from agent import answer_ledger
from agent import turn_parts
from agent.turn_parts import ANSWERED, INTERACTIVE, UNANSWERABLE
from agent import call_state as call_state_mod
from agent import confidence, outcomes, speakability, tool_outcome, turn_log
from agent.asr import TurnASR
# ADDED BY SOURAV -- real production bug, reported directly by the
# caller: "why voice is giving response only in bengali... when the user
# asks in hindi aur hinglish or english." detect_language() already
# existed but was NEVER CALLED anywhere in this repo -- every reply
# function below already accepts a `language` argument (all default to
# "bengali"), but nothing ever passed anything else. See detect_language()
# below wherever it's called for the full writeup, and its own updated
# docstring in agent/bn_normalize.py for a second, real bug found and
# fixed in the function itself while wiring this in.
from agent.bn_normalize import detect_language
from agent.llm import extract_intent, ExtractionError
from agent.reply_templates import (
    missing_slot_prompt, test_rate_reply, sample_type_reply, test_duration_reply,
    doctor_availability_reply, booking_reply,
    booking_confirm_prompt, heard_confirm_prompt, doctors_by_department_reply,
    booking_correction_prompt, INSUFFICIENT_VERIFIED_INFORMATION_BN,
    date_range_confirm_prompt, UNSPEAKABLE_ESCALATION, with_change_notice,
    near_match_prompt, NEAR_MATCH_UNCLEAR_BN,
    DEFERRED_PART_BN, RESUMING_PART_BN, unanswered_part_prompt,
    # ADDED BY SOURAV -- "Caller asks when a doctor sits" story.
    doctor_schedule_reply, booking_confirmation_prompt,
    # ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story.
    # delivery_declined_reply / otp_disclosure_refusal_reply are the two new
    # reply functions _dispatch_turn/_continue_pending speak directly
    # (every other new reply function is only ever reached indirectly,
    # through agent/report_flow.py's interpret_*() functions -- see that
    # module for why the decision logic itself lives there and not here).
    delivery_declined_reply, otp_disclosure_refusal_reply,
    # ADDED BY SOURAV -- "Caller asks about a health package" combined
    # with "Caller asks opening hours, address or directions".
    health_package_reply, health_packages_list_reply, clinic_info_reply,
    # ADDED BY SOURAV -- "Caller asks how to prepare for a test" story,
    # plus its bundled human_fallback config (see human_fallback_reply's
    # own module-level comment in agent/reply_templates.py for why that
    # part lives here rather than as an actual call transfer).
    test_preparation_reply, human_fallback_reply,
    # ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables (Walk-in
    # Eligibility, Prescription Requirements, Insurance Coverage Policy,
    # Outstanding Balance / Billing stories).
    walkin_eligibility_reply, prescription_requirements_reply,
    insurance_coverage_reply, billing_balance_reply,
    # ADDED BY SOURAV -- "Caller asks something the agent does not cover"
    # story. human_fallback_reply above is reused verbatim for the
    # "connect me to a human" branch -- these two are only the new
    # initial-offer and declined-offer replies.
    out_of_scope_reply, out_of_scope_counter_reply,
    # ADDED BY SOURAV -- "Caller asks whether their result is dangerous"
    # story (Epic: Conversation -- Difficult, Sensitive and Edge Cases).
    # Fixed, non-generative offer/decline pair -- human_fallback_reply
    # above is reused verbatim for the "yes, connect me" branch, same
    # pattern as out_of_scope_reply/out_of_scope_counter_reply just above.
    # See agent/reply_templates.py's own header comment on these two
    # functions, and agent/clinical_safety.py's module docstring, for why
    # this reply is never model-composed.
    clinical_interpretation_reply, clinical_interpretation_decline_reply,
    # ADDED BY SOURAV -- "Caller asks two questions in one breath" story.
    # These three back _resolve_combinable_intent_fragment()'s three
    # non-fabricating fallback fragments below -- every OTHER fragment in
    # a combined reply reuses an existing single-question reply function
    # from this same import block verbatim.
    multi_intent_missing_info_reply, multi_intent_out_of_scope_reply,
    multi_intent_needs_separate_flow_reply,
    # ADDED BY SOURAV -- "Caller asks the agent to compare two options"
    # story. Renders agent/compare_flow.py's build_comparison() output --
    # see that module and this function's own docstring for the
    # arithmetic/clinical-safety design.
    compare_options_reply,
    # ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
    # previous answer" story. See agent/state.py's own module docstring
    # and this function's docstring for when this is spoken.
    ambiguous_reference_reply,
    # ADDED BY SOURAV -- "Caller asks to be called back" story. See
    # agent/callback_flow.py's module docstring and each function's own
    # docstring for when these are spoken.
    callback_unavailable_reply, callback_confirmation_prompt, callback_scheduled_reply,
    # ADDED BY SOURAV -- KCD-448 correction path for request_callback, same
    # discipline as booking_correction_prompt above.
    callback_correction_prompt,
    # ADDED BY SOURAV -- "The agent accepts a correction and restates"
    # story (Epic: Answer Quality and Grounding). One shared acknowledgment
    # function for both booking and callback corrections -- see its own
    # docstring in agent/reply_templates.py.
    correction_acknowledged_reply,
    # ADDED BY SOURAV -- "Caller goes silent" story (Epic: Conversation --
    # Difficult, Sensitive and Edge Cases). Two graduated re-engagement
    # prompts plus the completion-aware graceful close -- see each
    # function's own docstring in agent/reply_templates.py, and
    # agent/silence_flow.py's module docstring for the stage machine in
    # _turn_poll_loop below that picks between them.
    silence_prompt_one, silence_prompt_two, silence_close_reply,
    # ADDED BY SOURAV -- "Caller wants to make a complaint" story. Fixed,
    # non-generative acknowledgment -- see this function's own docstring
    # in agent/reply_templates.py for why AC 2/AC 5 rule out anything else.
    complaint_acknowledged_reply,
    # ADDED BY SOURAV -- "Caller wants to speak to a doctor personally"
    # story. Fixed, non-generative, never names a doctor -- see this
    # function's own docstring in agent/reply_templates.py for why AC 1/2/
    # 3/4 rule out anything model-composed here.
    doctor_personal_request_reply,
)
from agent.compare_flow import build_comparison
# ADDED BY SOURAV -- "The agent accepts a correction and restates" story.
# Pure, transport-agnostic "is this a correction, and to what" detection --
# see this module's own docstring for why it lives here rather than
# duplicated inline in both main.py and main_pcm.py.
from agent.correction_flow import detect_booking_correction, detect_callback_correction
# ADDED BY SOURAV -- "Caller goes silent" story (Epic: Conversation --
# Difficult, Sensitive and Edge Cases). Pure, transport-agnostic "what is
# the next silence action / has anything been left unfinished" logic --
# same drift reason as detect_booking_correction() above: kept here once
# so _turn_poll_loop's independently-maintained copy in main.py and
# main_pcm.py cannot disagree about it. See this module's own docstring.
from agent import silence_flow
# ADDED BY SOURAV -- "Caller asks a follow-up that depends on the previous
# answer" story. Cross-turn entity memory (pronoun/elliptical follow-up
# resolution) -- see agent/state.py's own module docstring for the full
# design and exactly which intents this applies to.
from agent.state import DialogueState, resolve_follow_up, primary_slot_for_intent, kind_for_slot
from agent.fast_path import Catalogue, FastPath, COMMIT_MARGIN
from agent.semantic_cache import SemanticCache, embed as _embed_probe
# story title: The model never originates a fact
# user story: As a clinical lead, I want every price, date and identifier
#   to come from a verified system response, so that a wrong answer is a
#   data bug rather than a model bug.
# acceptance criteria: Every factual sentence is a template substitution
#   from a validated tool response and the model is never shown a figure
#   it could restate. An automated assertion on every commit proves no
#   model-composed span reaches synthesis on a factual intent.
#
# resolve_date is the story's core: the model's date becomes a candidate
# and slot_parse.py becomes the record. See its docstring.
from agent.slot_parse import (
    parse_date, parse_time, parse_phone, is_affirmative, is_negative,
    parse_correction_field,
    # ADDED BY SOURAV -- KCD-448 correction path for request_callback, same
    # discipline as parse_correction_field above.
    parse_callback_correction_field,
    # ADDED BY SOURAV -- report_status/report_send combined story: OTP entry
    # is parsed deterministically here, never sent to the LLM or the
    # semantic cache (see agent/slot_parse.py's parse_otp() docstring and
    # RULE 9 -- the OTP must never appear in an Ollama prompt or a cache key).
    parse_otp, looks_like_otp_disclosure_request,
    # ADDED BY SOURAV -- KCD-383 ("Caller asks whether their report is
    # ready"): detects a caller naming the callback option in reply to
    # the "confirm_delivery" offer, which now names both delivery and a
    # callback (agent/reply_templates.py's report_status_reply()). See
    # agent/slot_parse.py's own docstring for the ordering guarantee.
    looks_like_callback_preference,
)
# story title: The model never originates a fact
# user story: As a clinical lead, I want every price, date and identifier to
#   come from a verified system response, so that a wrong answer is a data bug
#   rather than a model bug.
# acceptance criteria: Every factual sentence is a template substitution from a
#   validated tool response and the model is never shown a figure it could
#   restate. An automated assertion on every commit proves no model-composed
#   span reaches synthesis on a factual intent.
#
# date_calc owns the calendar. The model names what the caller MEANT
# ("next_week"); every digit is computed here. See that module's docstring for
# the division of authority and for why an expression, unlike an ISO date,
# cannot go stale in the semantic cache.
from agent import date_calc
from agent.date_calc import SOURCE_INTERPRETED, SOURCE_PARSED
from agent.tools_client import ClinicToolsClient, ToolCallError
from agent.outcomes import (
    missing_booking_write_fields,
    record_insufficient_verified_information,
    # ADDED BY SOURAV -- "Caller asks how to prepare for a test" story's
    # bundled human_fallback config -- see record_human_handoff()'s own
    # docstring in agent/outcomes.py.
    record_human_handoff,
    # ADDED BY SOURAV -- "Caller asks to be called back" story. Mirrors
    # missing_booking_write_fields() exactly -- see that function's own
    # docstring in agent/outcomes.py.
    missing_callback_write_fields,
    # ADDED BY SOURAV -- "Caller asks for a person immediately" story
    # (Epic: Conversation -- Difficult, Sensitive and Edge Cases).
    # record_turn_attempt() is called once per dispatched turn (see this
    # story's own comment at the top of _dispatch_turn_inner below) so
    # immediate_human_escalation_rate() has a genuine denominator;
    # record_immediate_human_handoff() is the zero-negotiation escalation
    # itself. See agent/outcomes.py's own module comment on this section
    # for why the rate is computed from the existing handoff counter
    # rather than a second, driftable one.
    record_turn_attempt,
    record_immediate_human_handoff,
    immediate_human_escalation_rate,
    # ADDED BY SOURAV -- "Caller goes silent" story (Epic: Conversation --
    # Difficult, Sensitive and Edge Cases). Same ESCALATION_LOG_PATH ledger
    # and _lock as record_human_handoff() above, own "call_abandoned" event
    # name -- see its own docstring in agent/outcomes.py.
    record_call_abandoned,
    # ADDED BY SOURAV -- "Caller wants to make a complaint" story. Own
    # "complaint" intent/reason pair in the same shared ledger -- see its
    # own docstring in agent/outcomes.py for why it never logs the
    # complaint text itself.
    record_complaint_filed,
)
# ADDED BY SOURAV -- "Caller asks to be called back" story. Pure
# availability-check/context-building logic -- see that module's own
# docstring for why "operating hours" means the clinic's own hours and why
# nothing here reads os.environ or datetime directly.
from agent.callback_flow import check_callback_availability, build_callback_reason
from agent.callback_config import CALLBACKS_ENABLED
# ADDED BY SOURAV -- new shared module holding the actual report-flow
# DECISIONS as pure functions, so main.py and main_pcm.py both get
# identical business logic for report_status/report_send without
# hand-duplicating the branching a second time. See agent/report_flow.py's
# module docstring for the full reasoning (it also explains the
# pre-existing test_sample drift this same story restores parity on, just
# below).
from agent.report_flow import (
    interpret_report_status_result, interpret_delivery_request_result,
    interpret_otp_verify_result, match_candidate_report,
)
# ADDED BY SOURAV -- "Caller asks whether their result is dangerous" story
# (Epic: Conversation -- Difficult, Sensitive and Edge Cases). Deterministic,
# pre-LLM, pre-fast-path phrase/pattern detector -- see this module's own
# docstring for why it must run BEFORE _fast_path and BEFORE extract_intent
# inside _resolve_intent(), rather than living as an LLM prompt instruction
# or a fast_path.py catalogue entry: a smalltalk misclassification of a
# safety-panic question must be made structurally impossible, not merely
# unlikely.
from agent.clinical_safety import is_clinical_interpretation
# ADDED BY SOURAV -- "Caller asks for a person immediately" story (Epic:
# Conversation -- Difficult, Sensitive and Edge Cases). Deterministic,
# pre-LLM, pre-fast-path phrase matcher -- see this module's own docstring
# for why it must run BEFORE _fast_path and BEFORE extract_intent inside
# _resolve_intent(), AND again at the very top of _continue_pending(),
# ahead of every in-progress flow's own field parsing.
from agent.human_fast_path import is_immediate_human_request
# ADDED BY SOURAV -- "Caller wants to make a complaint" story. Deterministic,
# pre-LLM, pre-fast-path phrase matcher, structured directly on
# agent/human_fast_path.py's own guard just above -- see this module's own
# docstring for why it must run BEFORE _fast_path and BEFORE extract_intent
# inside _resolve_intent(), AND again at the very top of _continue_pending(),
# ahead of every in-progress flow's own field parsing.
from agent.complaint_flow import is_complaint
# ADDED BY SOURAV -- "Caller wants to speak to a doctor personally" story.
# Pure, transport-agnostic "is this a personal-contact-with-a-doctor
# request" detection -- see that module's own docstring for why this is a
# separate, pre-classifier guard, structured the same way agent/
# complaint_flow.py and agent/human_fast_path.py already are.
from agent.doctor_personal_request import is_doctor_personal_request
# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
from agent import tts as tts_mod
from agent.tts import TTSClient, UnspeakableReply, SPEAKABILITY_ENFORCE
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

CLINIC_API_BASE = os.environ.get("CLINIC_API_BASE", "http://localhost:8080")

# Order also doubles as PRIORITY: the field _next_missing() asks for next
# when several are still empty. doctor_name first because it is almost
# always already known by the time booking starts (named directly, or
# carried over from session.pending after a doctors_by_department /
# doctor_availability turn -- see _continue_pending below).
_BOOKING_FIELDS = ("doctor_name", "date", "time_slot", "patient_name", "phone")

# story title: A thing not existing is never confused with a system being down
# user story: As a caller, I want to know whether my test does not exist or the
#   system cannot be reached, so that I know whether to call back.
# acceptance criteria: The two produce different spoken sentences and different
#   metrics, and the distinction survives every refactor. This behaviour exists
#   today and gains a permanent regression case.
#
# THE SYSTEM-IS-DOWN SENTENCE, and the only one. It was written out at all
# seven ToolCallError handlers, which is seven chances for one of them to drift
# into a different wording -- or, worse, into a not-found wording -- during a
# refactor nobody reviewed carefully. A caller must be able to tell "your test
# does not exist" from "I could not reach the system" without knowing which
# code path they happened to hit, because the two carry opposite instructions:
# one means stop asking, the other means call back.
#
# The not-found sentences deliberately stay in agent/reply_templates.py, where
# every other sentence that NAMES something lives. The separation is the point:
# this file says what the system could not do, that file says what the clinic
# said. tests/test_outcome_distinction.py asserts the two sets never overlap.
SYSTEM_UNREACHABLE_BN = "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।"

# story title: Every critical value is read back before it is used
# user story: As a patient giving a phone number, I want it read back, so that
#   a misheard digit does not send my report to a stranger.
# acceptance criteria: Phone numbers, dates, times and names are confirmed
#   aloud before any write, and a rejection opens a correction path rather than
#   repeating the prompt. Readback is mandatory regardless of confidence for
#   values that affect a write.
#
# Said when the repair ladder runs out -- from confirm_booking, and now also
# from confirm_correction. Named once for the same reason SYSTEM_UNREACHABLE_BN
# is: two copies of a sentence are one refactor away from two different
# sentences, and this one carries a promise -- that nothing was written -- which
# must not drift.
BOOKING_NOT_CONFIRMED_BN = "এখনো কনফার্ম করতে পারলাম না। অ্যাপয়েন্টমেন্টটা করা হয়নি — কাউন্টারে একবার কথা বলে নেবেন।"

# Turns that died on an exception nobody expected. Counted because a metric
# that only tallies TIDY failures reads healthy during exactly the incident it
# exists for. Surfaced at /api/stats beside the per-tool outcomes.
_turn_crashes = 0

# story title: The same question gets the same answer within one call
# user story: As a caller who asks twice, I want the same answer, so that I
#   know which one to believe.
# acceptance criteria: Repeating a question in one call produces an identical
#   factual answer unless the underlying data changed, in which case the
#   change is stated. A test asserts consistency across three repeats with an
#   unchanged backend.
#
# Each AnswerLedger lives and dies with its call, which is the right scope for
# the behaviour and the wrong one for a metric -- nothing would ever read it.
# This is the process-level roll-up, accumulated as each call's verdicts come
# in, on the same argument as _turn_crashes above.
_consistency = {"repeats": 0, answer_ledger.SAME: 0, answer_ledger.CHANGED: 0}

# ADDED BY SOURAV -- "Caller asks to be called back" story. Deliberately
# its OWN tuple/helper, not a repurposing of _BOOKING_FIELDS/_next_missing
# above -- those two are hard-wired to each other (see _next_missing's own
# docstring) and to book_appointment's specific 5-field shape; a request-
# callback flow needs only two fields, and giving it a separate helper
# keeps this story's blast radius off the booking flow's already-tested
# behaviour entirely. Values here are the SLOT keys ("phone" -- the same
# slot key booking/report flows already use), not the scoped `awaiting`
# strings _continue_pending uses for its pending state (those are
# "callback_time_window" and "callback_phone" -- see that function's own
# comment on why "phone" alone would collide with the booking flow's
# existing "phone" awaiting state).
_CALLBACK_FIELDS = ("callback_time_window", "phone")

app = FastAPI()

# ---- process-wide singletons: loaded once, shared by every call ----
_asr: TurnASR | None = None
_turn_detector: TurnDetector | None = None
_tools: ClinicToolsClient | None = None
_tts: TTSClient | None = None
_intent_cache: SemanticCache | None = None
_fast_path: FastPath | None = None


@app.on_event("startup")
async def _startup():
    global _asr, _turn_detector, _tools, _tts, _intent_cache, _fast_path
    logger.info("loading IndicConformer...")
    _asr = await asyncio.to_thread(TurnASR)
    logger.info("loading Silero VAD...")
    _turn_detector = await asyncio.to_thread(TurnDetector)
    _tools = ClinicToolsClient(CLINIC_API_BASE)
    _tts = TTSClient()
    _intent_cache = SemanticCache()

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

    # Load the 74-row catalogue once so the fast path can identify a test
    # or doctor locally. Optional: if the clinic API is not up yet, every
    # turn simply goes to the LLM, which is the behaviour that existed
    # before this path did.
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=10) as c:
            payload = (await c.get(f"{CLINIC_API_BASE}/api/v1/catalogue")).json()
        _fast_path = FastPath(Catalogue(payload))
        # story title: A thing not existing is never confused with a system
        #   being down
        # user story: As a caller, I want to know whether my test does not
        #   exist or the system cannot be reached, so that I know whether to
        #   call back.
        # acceptance criteria: The two produce different spoken sentences and
        #   different metrics, and the distinction survives every refactor.
        #   This behaviour exists today and gains a permanent regression case.
        #
        # An EMPTY catalogue used to log this same line with a 0 in it and
        # carry on. It is not a quiet condition: the clinic API is up and
        # answering, so nothing is "unreachable", and every single caller is
        # about to be told in a well-formed sentence that their test does not
        # exist. That is the confusion this story is named after, arriving from
        # the data side rather than the code side.
        #
        # clinic-api's own /api/health reports these counts. Nobody was looking.
        if not len(_fast_path.catalogue):
            logger.error("CLINIC CATALOGUE IS EMPTY -- the API is up and has no "
                         "rows, so every caller will be told their test does not "
                         "exist. Check the clinic database before taking calls.")
        else:
            logger.info("fast path ready over %d catalogue rows", len(_fast_path.catalogue))
    except Exception as e:  # noqa: BLE001 - degrade to LLM-only, never fail startup
        logger.warning("catalogue unavailable, fast path disabled: %s", e)
        _fast_path = None

    # STORY [Answer Quality and Grounding]
    # As a patient, I want to hear the whole sentence, so that I am
    # not left guessing what the agent tried to say.
    # Before a single caller connects. Deliberately NOT wrapped in a try:
    # a canned line the synthesizer would mangle is a defect in this
    # repository, and the escalation line being sayable is what stops
    # _speak()'s blocked path recursing. Refusing to start is the correct
    # response to either -- and it is checked whether or not enforcement is
    # on, because a literal in the source is not the unknown data shadow
    # mode exists to measure.
    _tts.assert_canned_lines_speakable()
    logger.info("canned lines verified speakable (%d)", len(tts_mod.PREWARM_LINES))

    logger.info("prewarming TTS...")
    await _tts.prewarm()
    logger.info("startup complete -- ready for calls (speakability enforce=%s)",
                SPEAKABILITY_ENFORCE)


@app.on_event("shutdown")
async def _shutdown():
    if _tools:
        await _tools.aclose()
    if _tts:
        await _tts.aclose()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "asr_loaded": _asr is not None,
        "clinic_api_base": CLINIC_API_BASE,
    }


@app.get("/api/stats")
async def stats():
    """Cache effectiveness, for tuning the similarity threshold against
    real traffic rather than against my assumptions about it.

    `speakability` is not a cache statistic and is here anyway, because this
    is the only endpoint anything scrapes. unspeakable_blocked is the number
    an alert rule would watch; while enforce is false it reads as "replies
    that WOULD have been blocked", which is the whole question shadow mode
    exists to answer. Any non-zero value is a missing spoken-form entry --
    agent/turn_log.py's unspeakable_reply rows name which one.
    """
    tts_snapshot = _tts.snapshot() if _tts else None
    return {
        "fast_path": _fast_path.snapshot() if _fast_path else None,
        "intent_cache": _intent_cache.snapshot() if _intent_cache else None,
        "tts_cache": tts_snapshot,
        # STORY [Answer Quality and Grounding]
        # As a patient, I want to hear the whole sentence, so that I am
        # not left guessing what the agent tried to say.
        "speakability": {
            "enforced": SPEAKABILITY_ENFORCE,
            "blocked": tts_snapshot["unspeakable_blocked"] if tts_snapshot else None,
        },
        # story title: A thing not existing is never confused with a system
        #   being down
        # user story: As a caller, I want to know whether my test does not
        #   exist or the system cannot be reached, so that I know whether to
        #   call back.
        # acceptance criteria: The two produce different spoken sentences and
        #   different metrics, and the distinction survives every refactor.
        #   This behaviour exists today and gains a permanent regression case.
        #
        # The "different metrics" half of the criterion. Per tool: answered,
        # not_found, unreachable, and the derived rate. Alertable in BOTH
        # directions -- unreachable rising is the API or the network;
        # not_found_rate rising is either callers asking for things the clinic
        # does not stock, or the catalogue emptying itself, which today tells
        # every caller their test does not exist in a perfectly well-formed
        # sentence while nothing anywhere notices.
        #
        # turn_crashes sits beside them because a turn that died in our own
        # code is the system being down from the caller's seat, and a metric
        # that counts only tidy failures reads healthy during an incident.
        "clinic": {
            "tools": _tools.snapshot() if _tools else None,
            "turn_crashes": _turn_crashes,
        },
        # story title: The same question gets the same answer within one call
        # user story: As a caller who asks twice, I want the same answer, so
        #   that I know which one to believe.
        # acceptance criteria: Repeating a question in one call produces an
        #   identical factual answer unless the underlying data changed, in
        #   which case the change is stated. A test asserts consistency across
        #   three repeats with an unchanged backend.
        #
        # `changed` is the alertable one. On a catalogue nobody is editing it
        # should sit at zero, so a rising rate is either real churn in the
        # clinic's data or entity resolution landing on a different row for
        # the same words -- and the second of those is precisely the wrong-hit
        # auditing E9-S12 asks for and nothing in this repository has ever
        # been able to see. `repeats` is the denominator: a `changed` count
        # means nothing without knowing how many repeats there were at all.
        "consistency": dict(_consistency),
        # ADDED BY SOURAV -- reference-data TTL cache (agent/
        # reference_data_cache.py), for tuning cache_ttl_s against real
        # traffic the same way intent_cache's threshold was tuned above.
        "reference_cache": _tools.reference_cache_snapshot() if _tools else None,
    }


def _wav_duration_s(wav_bytes: bytes) -> float:
    try:
        with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001 - a fallback clip may not be canonical WAV
        return 5.0


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

        # Starts True: the greeting goes out before the caller has said
        # anything, so the gate must already be closed when the first poll
        # tick runs, not opened a moment later by _speak().
        self.agent_speaking = True
        self.speak_deadline = time.time() + PLAYBACK_GUARD_S
        self.resync_pending = False

        # Cross-turn booking state. None outside a booking flow. See
        # _continue_pending's docstring for the shape and why this exists --
        # in short, it is the only thing that survives between turns, since
        # every _resolve_intent call otherwise starts from zero context.
        self.pending: dict | None = None

        # The single structure every downstream layer reads for caller signals
        # (Blueprint 4.5). Replaced wholesale each turn by Call Intelligence,
        # never mutated -- CallSession stays the transport bookkeeper it is,
        # and the caller picture lives in its own frozen object.
        #
        # Today no detector exists, so this is the Appendix C "normal" row on
        # every call with caller_state="unknown". That is deliberate: it is
        # the behaviour the agent already had, so the object changes nothing
        # until something actually detects.
        self.call_state = call_state_mod.build()

        # story title: The same question gets the same answer within one call
        # user story: As a caller who asks twice, I want the same answer, so
        #   that I know which one to believe.
        # acceptance criteria: Repeating a question in one call produces an
        #   identical factual answer unless the underlying data changed, in
        #   which case the change is stated. A test asserts consistency across
        #   three repeats with an unchanged backend.
        #
        # What this caller has already been told. Per-call by construction --
        # it is created here and nothing outlives the session, which is the
        # scope the story asks for ("within one call") and also the only scope
        # that is safe: a process-wide version would compare one caller's
        # answer against another's. Never holds a reply, only the facts a
        # reply was rendered from. See agent/answer_ledger.py.
        self.answer_ledger = answer_ledger.AnswerLedger()

        # story title: A multi-part question is answered in full
        # user story: As a caller who asked two things, I want both answered,
        #   so that I do not have to ask again.
        # acceptance criteria: Every answerable part of a turn is answered in
        #   the order asked, and any part that cannot be answered is
        #   explicitly addressed rather than dropped. Completeness is scored
        #   on a labelled multi-part set.
        #
        # Parts of an earlier turn that a question got in front of, the
        # utterance they came from, and how many turns they have waited.
        # Deliberately NOT stored inside session.pending: that dict is
        # cleared on a dozen paths, every one of which would silently bin a
        # question the caller actually asked.
        self.deferred: dict | None = None

        # Consecutive turns that were echoed back for confirmation because the
        # decoders disagreed. Reset by any turn that proceeds normally. Capped
        # so a caller on a bad line is offered a human instead of being asked
        # "did I hear you right?" indefinitely -- a repair ladder that never
        # ends is a D-grade outcome even though every individual turn is safe.
        self.confirm_attempts = 0

        # ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
        # previous answer" story. A SEPARATE kind of cross-turn memory
        # from `pending` just above: `pending` tracks one IN-PROGRESS flow
        # waiting on one specific missing field; `state` tracks the last
        # entity (test/doctor/package) each already-FINISHED question was
        # actually about, so a brand-new question that refers back with a
        # pronoun ("eta-r jonno ki prescription lagbe?") can resolve
        # without making the caller repeat the name. See agent/state.py's
        # own module docstring for the full design.
        self.state = DialogueState()

        # ADDED BY SOURAV -- "Caller goes silent" story (Epic: Conversation
        # -- Difficult, Sensitive and Edge Cases). Per-call, never global or
        # shared between sessions (a process-wide counter would leak one
        # caller's silence into another caller's call): how many graduated
        # silence prompts the CURRENT silence episode has already used
        # (agent/silence_flow.STAGE_NORMAL/STAGE_PROMPT_1_SENT/
        # STAGE_PROMPT_2_SENT), and the exact text of whichever one was
        # spoken most recently, so a later abandonment log entry can record
        # the preceding prompt without re-rendering or re-guessing it. Both
        # are read and written only by _turn_poll_loop -- see that
        # function's own comment on the IDLE_TIMEOUT_S branch below.
        self.silence_stage = silence_flow.STAGE_NORMAL
        self.last_silence_prompt: str | None = None

    def hold_gate_for(self, audio_duration_s: float):
        """Called before each reply goes out. Extends rather than replaces
        the deadline: replies queue on the client, so a second clip starts
        playing only after the first finishes."""
        base = max(self.speak_deadline, time.time()) if self.agent_speaking else time.time()
        self.agent_speaking = True
        self.speak_deadline = base + audio_duration_s + PLAYBACK_GUARD_S

    def release_gate(self):
        """Playback is over. Don't touch processed_until_s here -- the poll
        loop owns the decoded buffer and does the resync on its next tick."""
        self.agent_speaking = False
        self.resync_pending = True

    async def append(self, chunk: bytes):
        self.last_activity = time.time()
        self.audio.append(chunk)

    async def send_json(self, sender: str, text: str):
        await self.ws.send_text(json.dumps({"sender": sender, "text": text}, ensure_ascii=False))

    async def send_audio(self, wav_bytes: bytes):
        if wav_bytes:
            await self.ws.send_bytes(wav_bytes)

    def cleanup(self):
        import shutil
        with contextlib.suppress(OSError):
            shutil.rmtree(self.tmpdir, ignore_errors=True)


# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
async def _speak(session: CallSession, text_bn: str, fallback_reason: str | None = None):
    """Say one line to the caller, or say why it could not be said.

    SYNTHESIZE FIRST, THEN SEND THE TRANSCRIPT.
    This used to push text_bn to the browser before synthesis, which read as
    snappier -- the line appeared while the vocoder was still working. It also
    meant the pane was a record of what the agent INTENDED to say. Once a reply
    can be blocked and replaced, that stops being a harmless discrepancy: the
    transcript would show the answer while the caller heard a referral to the
    counter, and the one artefact anyone would check afterwards would disagree
    with the call. The cost is that the text now appears with the audio instead
    of a beat before it; correctness of the record wins.
    """
    wav = None
    try:
        # speech_rate is read from the call state rather than hardcoded. This
        # is the one place the object currently changes what a caller HEARS,
        # and it resolves to "default" until a detector sets caller_state or
        # senior -- so it is a no-op today by design, not by accident.
        # language likewise: always None until a detector sets it, and threaded
        # through because the speakability rule is Bengali-only.
        wav = await _tts.synthesize(text_bn,
                                    speech_rate=session.call_state.speech_rate,
                                    language=session.call_state.language)
    # STORY [Answer Quality and Grounding]
    # As a patient, I want to hear the whole sentence, so that I am
    # not left guessing what the agent tried to say.
    except UnspeakableReply as e:
        # A hole was found in a reply we composed. Caught BEFORE the generic
        # handler below on purpose: that one answers with the "system busy"
        # clip, which is right when the vocoder is down and wrong here. This
        # is a defect on our side, not an outage, and the honest response is
        # to send the caller somewhere that can actually answer them.
        logger.error("[%s] reply blocked, dropped=%s -- escalating to counter",
                     session.call_id, list(e.dropped))
        turn_log.record_unspeakable(session.call_id, session.utt_seq,
                                    e.dropped, enforced=True)
        text_bn = UNSPEAKABLE_ESCALATION
        try:
            # Cannot recurse: this line is in PREWARM_LINES and startup asserts
            # every one of them is speakable, so it can never be blocked itself.
            # The broad catch is the belt to that braces -- if it somehow were,
            # the caller still gets the pre-recorded clip rather than silence.
            wav = await _tts.synthesize(text_bn,
                                        speech_rate=session.call_state.speech_rate)
        except Exception:  # noqa: BLE001 - last resort, never raise past here
            logger.exception("[%s] escalation line failed to synthesize", session.call_id)
            wav = _tts.fallback_audio("tool_failure")
    except Exception as e:  # noqa: BLE001 - TTS is the last mile, must not raise past here
        logger.warning("[%s] TTS failed (%s) -- using fallback audio", session.call_id, e)
        wav = _tts.fallback_audio(fallback_reason or "tts_failure")

    # STORY [Answer Quality and Grounding]
    # As a patient, I want to hear the whole sentence, so that I am
    # not left guessing what the agent tried to say.
    if not SPEAKABILITY_ENFORCE:
        # SHADOW MODE. synthesize() counted and logged this, but only the
        # orchestrator knows the call and turn, so the exported row is written
        # here. Guarded so it cannot double up with the enforced branch above:
        # exactly one of the two runs.
        #
        # This whole block is temporary. It exists to answer one question --
        # how often would real traffic have been blocked -- and comes out when
        # SPEAKABILITY_ENFORCE becomes the default.
        _verdict = speakability.check(text_bn, language=session.call_state.language)
        if _verdict.is_blocked:
            turn_log.record_unspeakable(session.call_id, session.utt_seq,
                                        _verdict.dropped, enforced=False)

    await session.send_json("AI", text_bn)

    # Close the gate BEFORE the bytes leave, never after: the client can
    # start playing the moment they land, and a poll tick that slips in
    # between send and gate is exactly the echo this prevents.
    session.hold_gate_for(_wav_duration_s(wav))
    await session.send_audio(wav)


# story title: The same question gets the same answer within one call
# user story: As a caller who asks twice, I want the same answer, so that I
#   know which one to believe.
# acceptance criteria: Repeating a question in one call produces an identical
#   factual answer unless the underlying data changed, in which case the
#   change is stated. A test asserts consistency across three repeats with an
#   unchanged backend.
#
# WHY A WRAPPER AND NOT A LINE INSIDE _speak
# ------------------------------------------
# _speak knows the sentence and nothing else -- not which intent produced it,
# not which clinic response it was rendered from -- and both are needed to
# decide whether this answer contradicts an earlier one. Passing them into
# _speak would push tool responses into the transport layer for the sake of
# one caller in eight.
#
# WHY A WRAPPER AND NOT A LINE AT EACH CALL SITE
# ----------------------------------------------
# There are EIGHT places that speak a factual reply: three in
# _dispatch_turn_inner and five in _continue_pending. Eight places to remember
# is eight places to eventually forget, and the guarantee would then hold on
# the routes someone happened to think about. Same argument tool_outcome.py
# makes about its twelve, and tests/test_answer_consistency.py enforces it
# statically: a reply template handed straight to _speak fails the build.
# story title: Near matches are offered rather than guessed or refused
# user story: As a caller naming something loosely, I want the close matches
#   offered, so that I am not told my test does not exist when it does.
# acceptance criteria: When several catalogue rows fall within the match band
#   the agent offers up to three by name and asks which. Candidates are
#   generated across every supported language and romanised spelling. The
#   did-you-mean path covers the ambiguous case and not only total failure.
async def _offer_near_matches(session: CallSession, intent: str, result: dict,
                              offered_date: str | None) -> bool:
    """-> True if this response was an ambiguity and the turn is now finished.

    An ambiguous response is a QUESTION the clinic asked back, so nothing
    factual is spoken and nothing is recorded as an answer. The caller's
    reply lands in _continue_pending's entity_choice state, which re-runs the
    lookup against the canonical name they chose.

    `offered_date` is carried through so the second lookup asks about the
    same day as the first. Without it, "is Dr Sen in on Tuesday" answered
    with a choice, then resolved, would silently become a question about
    today.

    An EMPTY candidate list is not a bug: clinic-api sends one when several
    rows matched and at least one of them has no Bengali alias, because
    offering a partial list would be a guess wearing a question mark. There
    is nothing to choose from, so no choice state is opened -- the caller is
    asked to name it again and the next turn starts clean.
    """
    if not isinstance(result, dict) or not result.get("ambiguous"):
        return False

    candidates = result.get("candidates") or []
    logger.info("[%s] %s ambiguous (%d candidate(s)) -- offering instead of guessing",
                session.call_id, intent, len(candidates))

    session.pending = {
        "awaiting": "entity_choice", "intent": intent, "slots": {},
        "candidates": candidates, "offered_date": offered_date, "retries": 0,
    } if candidates else None

    await _speak(session, near_match_prompt(candidates))
    return True


async def _speak_fact(session: CallSession, intent: str, slots: dict,
                      result: dict, reply: str,
                      offered_date: str | None = None) -> bool:
    """Speak a factual reply, saying so if it contradicts an earlier one.

    `reply` is always rendered FRESH by the caller from a live clinic
    response. Nothing here substitutes a remembered answer -- the ledger only
    compares, and the most it can do is prepend one sentence. The
    Architecture Plan is explicit on this point ("do not optimise by caching
    replies"), and a reply cache is also the one change that would reintroduce
    the stale price this whole design avoids.

    story title: Near matches are offered rather than guessed or refused
    user story: As a caller naming something loosely, I want the close
        matches offered, so that I am not told my test does not exist when
        it does.
    acceptance criteria: When several catalogue rows fall within the match
        band the agent offers up to three by name and asks which. Candidates
        are generated across every supported language and romanised
        spelling. The did-you-mean path covers the ambiguous case and not
        only total failure.

    -> True when the response was ambiguous and an offer was spoken instead
    of the reply; every call site must then end the turn. The guard lives
    here rather than at the eight call sites for the same reason the
    consistency check does, and the return value exists because the callers
    set session.pending AFTER speaking -- an offer that set the choice state
    from in here would be overwritten a line later by the caller's own
    bookkeeping. tests/test_near_match_offers.py fails the build if a call
    site drops the guard.
    """
    if await _offer_near_matches(session, intent, result, offered_date):
        return True

    verdict, previous = session.answer_ledger.check(intent, slots, result)

    if verdict != answer_ledger.FIRST:
        _consistency["repeats"] += 1
        _consistency[verdict] += 1

    if verdict == answer_ledger.CHANGED:
        # Logged at WARNING, not INFO. On a catalogue nobody is editing this
        # should not happen, and when it does the two candidate causes -- the
        # clinic's data moved, or entity resolution landed on a different row
        # for the same words -- are told apart by whether the leading identity
        # in the two tuples matches. Facts only; no caller data reaches here.
        logger.warning("[%s] %s answer changed within the call: %s -> %s",
                       session.call_id, intent, previous,
                       answer_ledger.facts(intent, result))
        reply = with_change_notice(reply)

    await _speak(session, reply)
    return False


def _record_answer_for_ledger(session: CallSession, intent: str, slots: dict,
                              result: dict, reply: str) -> str:
    """The consistency-check-and-annotate half of _speak_fact() above,
    factored out for _resolve_combinable_intent_fragment() below.

    ADDED BY SOURAV -- KCD-449. A multi-intent turn ("what's the CBC
    rate, and is Dr Sen in today?") resolves each of its questions
    through _resolve_combinable_intent_fragment(), never through the
    solo dispatch branches _speak_fact() already guards -- so a
    test_rate/doctor_availability/doctors_by_department answer given
    inside a combined turn was never recorded in the ledger at all, and
    a repeat of it (asked combined again, or asked plainly on its own
    afterwards) had nothing to be compared against.

    Deliberately does NOT call _offer_near_matches() the way _speak_fact()
    does: _resolve_combinable_intent_fragment()'s own docstring is
    explicit that a multi-intent turn never opens a follow-up pending
    state of any kind, near-match "did you mean" offers included -- this
    keeps that guarantee exactly as narrow as it already was. An
    ambiguous result inside a combined turn still gets whatever
    _resolve_combinable_intent_fragment() already rendered for it (a
    separate, pre-existing limitation of that path, not something this
    consistency fix should touch); AnswerLedger.check() already refuses
    to record an ambiguous result on its own (see its own docstring), so
    nothing here needs to duplicate that guard either.

    Takes `reply` pre-rendered, exactly like _speak_fact(), and returns it
    -- annotated with the change notice when the verdict is CHANGED --
    for the caller to fold into its own joined multi-fragment sentence.
    Never speaks anything itself.
    """
    verdict, previous = session.answer_ledger.check(intent, slots, result)

    if verdict != answer_ledger.FIRST:
        _consistency["repeats"] += 1
        _consistency[verdict] += 1

    if verdict == answer_ledger.CHANGED:
        logger.warning("[%s] %s answer changed within the call (combined turn): %s -> %s",
                       session.call_id, intent, previous,
                       answer_ledger.facts(intent, result))
        reply = with_change_notice(reply)

    return reply


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


async def _resolve_intent(session: CallSession, text: str) -> dict:
    """Semantic cache in front of the LLM. A hit skips Ollama entirely --
    the slowest hop in the turn -- but the clinic lookup that follows still
    runs live, so a cached intent can never serve a stale price."""
    # ADDED BY SOURAV -- "Caller asks for a person immediately" story
    # (Epic: Conversation -- Difficult, Sensitive and Edge Cases). Checked
    # FIRST, even ahead of the clinical-interpretation guard just below --
    # a caller explicitly demanding a human, right now, is the more
    # absolute, zero-tolerance-for-negotiation request of the two, and an
    # utterance that could plausibly read as either is safest resolved as
    # immediate escalation rather than an offer to connect. Same reasoning
    # as clinical_interpretation's own placement: checked before fast_path,
    # before the semantic cache, and before Ollama, so a caller asking for
    # a human can never be misclassified as smalltalk, out_of_scope (which
    # OFFERS a choice rather than escalating outright), or anything else
    # that would ask a follow-up question first. See
    # agent/human_fast_path.py's module docstring for the full reasoning,
    # including why this same check also has to run again inside
    # _continue_pending() for a caller who asks mid-flow.
    if is_immediate_human_request(text):
        logger.info("[%s] immediate-human-request guard fired -- no LLM call",
                    session.call_id)
        slots = {"test_name": None, "doctor_name": None, "date": None,
                  "time_slot": None, "patient_name": None, "phone": None}
        return {
            "intent": "human_direct_request",
            "slots": slots,
            "parts": [{"intent": "human_direct_request", "slots": slots}],
            "direct_reply_bn": None,
        }

    # ADDED BY SOURAV -- "Caller asks whether their result is dangerous"
    # story (Epic: Conversation -- Difficult, Sensitive and Edge Cases).
    # Checked BEFORE fast_path and BEFORE the semantic cache/LLM, not
    # alongside them: a safety-panic question must never be able to land in
    # "smalltalk" (whether via a fast-path greeting hit, a stale semantic
    # cache entry, or an LLM misclassification under pressure) and get the
    # model's own free-composed text spoken with zero downstream check --
    # see agent/clinical_safety.py's module docstring for the full "smalltalk
    # loophole" this closes structurally. `slots` mirrors
    # agent/fast_path.py's _empty_slots() shape -- this intent never carries
    # any slot value -- and `direct_reply_bn` is left None (never a
    # model-or-guard-composed string) so _validate()'s existing strip-for-
    # non-smalltalk rule is not even the thing keeping this reply honest;
    # the reply text itself always comes from
    # agent.reply_templates.clinical_interpretation_reply(), never from here.
    if is_clinical_interpretation(text):
        logger.info("[%s] clinical-interpretation guard fired -- no LLM call",
                    session.call_id)
        slots = {"test_name": None, "doctor_name": None, "date": None,
                  "time_slot": None, "patient_name": None, "phone": None}
        return {
            "intent": "clinical_interpretation",
            "slots": slots,
            "parts": [{"intent": "clinical_interpretation", "slots": slots}],
            "direct_reply_bn": None,
        }

    # ADDED BY SOURAV -- "Caller wants to make a complaint" story. Checked
    # here, after the two guards above, before fast_path/the semantic
    # cache/Ollama -- same "structurally impossible to misclassify" reasoning
    # agent/complaint_flow.py's module docstring gives in full: AC 5's "does
    # NOT attempt to resolve, explain, defend, justify, or argue" is a
    # zero-tolerance policy the model must never get a turn to violate, so a
    # complaint can never be misclassified as smalltalk, out_of_scope, or
    # anything an LLM retry under phrasing pressure might produce. See that
    # module's own docstring for why this also has to run again inside
    # _continue_pending() for a caller who complains mid-flow.
    if is_complaint(text):
        logger.info("[%s] complaint guard fired -- no LLM call", session.call_id)
        slots = {"test_name": None, "doctor_name": None, "date": None,
                  "time_slot": None, "patient_name": None, "phone": None}
        return {
            "intent": "complaint",
            "slots": slots,
            "parts": [{"intent": "complaint", "slots": slots}],
            "direct_reply_bn": None,
        }

    # ADDED BY SOURAV -- "Caller wants to speak to a doctor personally"
    # story. Checked here, after the three guards above, before fast_path/
    # the semantic cache/Ollama -- same "structurally impossible to
    # misclassify" reasoning agent/doctor_personal_request.py's module
    # docstring gives in full: AC 3's "never promises a call from a named
    # doctor it cannot schedule" is a zero-tolerance policy the model must
    # never get a turn to violate (including via the "smalltalk" intent's
    # own unguarded direct_reply_bn -- see agent/clinical_safety.py's
    # docstring on that exact loophole), so this can never be
    # misclassified as book_appointment (the wrong route for someone
    # wanting reassurance now, not a future visit), "unclear", smalltalk,
    # or human_direct_request. See that module's own docstring for why
    # this also has to run again inside _continue_pending() for a caller
    # who asks mid-flow, and for why it deliberately never matches a bare
    # "doctor" mention (doctor_availability/doctor_schedule/book_appointment
    # must keep working exactly as before).
    if is_doctor_personal_request(text):
        logger.info("[%s] doctor-personal-request guard fired -- no LLM call",
                    session.call_id)
        slots = {"test_name": None, "doctor_name": None, "date": None,
                  "time_slot": None, "patient_name": None, "phone": None}
        return {
            "intent": "doctor_personal_request",
            "slots": slots,
            "parts": [{"intent": "doctor_personal_request", "slots": slots}],
            "direct_reply_bn": None,
        }

    # Tier 1: decide it locally if we can. For a fixed catalogue the
    # entity is a string-matching problem with a 0.32 confidence margin,
    # where the embedding route had 0.03 -- see agent/fast_path.py. This
    # returns None whenever it is not sure, which is the common case for
    # anything except a routine price or availability question.
    if _fast_path is not None:
        hit = await asyncio.to_thread(_fast_path.resolve, text)
        if hit is not None:
            logger.info("[%s] fast path resolved %s (%.2f) -- no LLM call",
                        session.call_id, hit.intent, hit.confidence)
            return hit.as_llm_shape()

    cached, how = await asyncio.to_thread(_intent_cache.get, text)
    if cached is not None:
        logger.info("[%s] intent cache %s hit", session.call_id, how)
        return cached

    data, diag = await asyncio.to_thread(extract_intent, text)
    logger.info("[%s] intent extracted in %.2fs (%d attempt(s))",
                session.call_id, diag["total_time_s"], diag["attempts"])
    await asyncio.to_thread(_intent_cache.put, text, data)
    return data


def _next_missing(slots: dict) -> str | None:
    """-> the first still-empty field in _BOOKING_FIELDS order, or None
    once every field a booking needs is filled."""
    for field in _BOOKING_FIELDS:
        if not slots.get(field):
            return field
    return None


def _next_missing_callback(slots: dict) -> str | None:
    """-> the scoped `awaiting` value (NOT the bare slot key -- see
    _CALLBACK_FIELDS' own comment) for the first still-empty field this
    story needs, or None once both are filled. Checks slots["phone"] (the
    real slot key) but returns "callback_phone" (the scoped awaiting
    name) so _continue_pending routes it through this story's own branch
    rather than the pre-existing booking flow's bare "phone" tail."""
    if not slots.get("callback_time_window"):
        return "callback_time_window"
    if not slots.get("phone"):
        return "callback_phone"
    return None


# ADDED BY SOURAV -- "The agent accepts a correction and restates" story.
# Callback's own analogue of _booking_correction_parsers() above -- ONLY
# "phone" (keyed by SLOT key, not the "callback_phone" awaiting name --
# see _next_missing_callback's own comment on that distinction).
#
# "callback_time_window" is deliberately NOT included, for the identical
# reason "patient_name" is excluded from _booking_correction_parsers()
# above: it is free text with no grammar of its own, so a bare non-empty-
# string check cannot tell "the caller is naming this field" from "the
# caller is also giving its new value" -- tests/test_request_callback.py's
# own pre-existing test_naming_time_window_reenters_its_collection_with_
# phone_kept caught exactly this when an earlier version of this
# function DID include it. See agent/correction_flow.py's own module
# docstring (point 2) for the full writeup; the time window keeps the
# deliberate, pre-existing two-step path instead.
def _callback_correction_parsers() -> dict:
    return {"phone": parse_phone}


def _match_offered(text: str, candidates: list[dict]) -> dict | None:
    """-> the canonical `name` (the form book_appointment/get_doctor_
    availability need) of the doctor the caller just named out of a list
    session.pending offered a moment ago, or None if the utterance is not
    confidently one of them.

    Same trust model as fast_path.Catalogue.match: score every candidate
    against every spoken form (English surname AND the seeded Bengali
    alias, since the caller may answer in either script), and only commit
    above a floor rather than always taking the best of a bad field. 0.55
    is fast_path.ENTITY_MATCH_FLOOR -- reused here because the situation is
    the same shape (matching a short spoken name against a small local
    list), just with the candidate list narrowed to what was JUST spoken
    to the caller instead of the whole 74-row catalogue.
    """
    if not candidates:
        return None
    norm_text = text.strip().lower()
    if not norm_text:
        return None
    # story title: Near matches are offered rather than guessed or refused
    # user story: As a caller naming something loosely, I want the close
    #   matches offered, so that I am not told my test does not exist when it
    #   does.
    # acceptance criteria: When several catalogue rows fall within the match
    #   band the agent offers up to three by name and asks which. Candidates
    #   are generated across every supported language and romanised spelling.
    #   The did-you-mean path covers the ambiguous case and not only total
    #   failure.
    #
    # The runner-up is tracked now, and the margin applies here too. This is
    # the turn AFTER an offer -- the caller has just been read two names and
    # answered -- so an answer that fits both of them equally is the one
    # place where guessing would be least forgivable: the whole point of the
    # preceding turn was that the agent had stopped guessing.
    best, best_score, runner_up = None, 0.0, 0.0
    for c in candidates:
        forms = [c["name"], c["name"].split()[-1]]
        if c.get("name_bn"):
            forms.append(c["name_bn"])
        row_best = 0.0
        for form in forms:
            if not form:
                continue
            form_l = form.lower()
            score = difflib.SequenceMatcher(None, form_l, norm_text).ratio()
            if form_l in norm_text or norm_text in form_l:
                score = max(score, 0.85)
            row_best = max(row_best, score)
        if row_best > best_score:
            best, best_score, runner_up = c, row_best, best_score
        elif row_best > runner_up:
            runner_up = row_best
    # Returns the whole candidate, not just the canonical name. The API needs
    # the English `name`; the booking readback needs `name_bn`, because it is
    # SPOKEN. Returning one and looking the other up later is what put an
    # English name into a Bengali sentence -- see booking_confirm_prompt.
    if best_score < 0.55:
        return None
    if (best_score - runner_up) < COMMIT_MARGIN:
        # Two of the offered names fit what the caller just said equally
        # well. Re-asking is the only honest move; picking one would undo
        # the turn that produced the offer.
        return None
    return best


def _match_candidate_name(text: str, candidates: list[str]) -> str | None:
    """ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
    previous answer" story. Same trust model and score floor as
    _match_offered()/agent/report_flow.py's match_candidate_report()
    just above/elsewhere -- matching a short spoken reply against a SMALL
    list just offered to the caller (here: the 2+ ambiguous names
    ambiguous_reference_reply() just spoke, from agent/state.py's
    EntitySlot.names) -- kept as its own function rather than reused
    because those two match against `{"name":..., "name_bn":...}` dicts
    while this one's candidates are already plain strings (agent/state.py
    never tracks a Bengali alias, only the catalogue's own canonical
    name -- see that module's own docstring)."""
    if not candidates:
        return None
    norm_text = text.strip().lower()
    if not norm_text:
        return None
    best_name, best_score = None, 0.0
    for candidate in candidates:
        form_l = candidate.lower()
        score = difflib.SequenceMatcher(None, form_l, norm_text).ratio()
        if form_l in norm_text or norm_text in form_l:
            score = max(score, 0.85)
        if score > best_score:
            best_name, best_score = candidate, score
    return best_name if best_score >= 0.55 else None


# Bare "নাম বলছি" prefixes a caller sometimes leads a name with. Stripped
# rather than relied upon -- most callers just say the name on its own.
_NAME_PREFIXES = ("আমার নাম ", "নাম ", "আমি ")


def _clean_patient_name(text: str) -> str | None:
    """Strip at most ONE leading filler phrase off a caller's spoken
    patient name, e.g. "আমার নাম রাহুল সেন" -> "রাহুল সেন".

    Bug fixed here: this used to re-check ALL of _NAME_PREFIXES in a
    plain `for` loop with no `break`, testing each prefix against the
    ALREADY-stripped text from the previous iteration. Real disfluent
    speech (or ASR output) that happens to start with more than one
    filler phrase in a row -- e.g. "নাম আমি সেন" ("name -- I'm Sen") --
    walked through BOTH matching prefixes one after another
    ("নাম আমি সেন" -> strip "নাম " -> "আমি সেন" -> strip "আমি " -> "সেন"),
    silently eating the caller's first name along with the filler words
    and leaving only the surname. Stopping after the first match means
    at most one filler phrase is ever removed -- the rest of whatever
    the caller said, first name included, is left alone."""
    t = text.strip().strip("।!?., ")
    if not t:
        return None
    for prefix in _NAME_PREFIXES:
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
            break
    return t or None


# ADDED BY SOURAV -- "The agent accepts a correction and restates" story.
# The THREE booking fields with a genuine, structurally-validating parser
# -- used both by agent/correction_flow.detect_booking_correction() (the
# spontaneous, cue-gated mid-collection check) and by the "confirm_
# correction" state's own "field + value in one breath" shortcut below --
# bundled once here rather than re-built at each call site.
#
# "patient_name" is deliberately NOT included, and this is a fix, not an
# oversight: an earlier version of this dict DID include it (mapped to
# _clean_patient_name), and tests/test_booking_readback.py's own pre-
# existing test_the_named_field_is_the_one_re_collected caught the real
# bug that caused -- _clean_patient_name() has no grammar to reject a
# non-name with, so a caller merely NAMING "patient_name" as the field to
# fix ("রোগীর নাম", "the patient's name") was itself accepted as though it
# WERE the new name. See agent/correction_flow.py's own module docstring
# (point 2) for the full writeup; patient_name keeps the deliberate,
# pre-existing two-step path instead (name the field, then a separate
# turn gives the value).
#
# parse_date's `offered_date` context comes from THIS pending dict, same
# as the ordinary "date" collection branch further down uses -- a
# spontaneous date correction is held to the same "hmm/yes confirms the
# date I already said out loud" rule as an original date answer would be.
def _booking_correction_parsers(pending: dict) -> dict:
    return {
        "date": lambda t: parse_date(t, offered_date=pending.get("offered_date")),
        "time_slot": parse_time,
        "phone": parse_phone,
    }


async def _finish_booking(session: CallSession, slots: dict, *, confirmed: bool = False,
                           language: str = "bengali"):
    """All 5 fields are filled -- place the booking and clear pending
    regardless of outcome. Failure here is reported the same way the old
    single-shot book_appointment branch reported it (tool_failure
    fallback audio), just reachable now from either that branch OR from
    the tail of a multi-turn _continue_pending flow.

    `confirmed` must be True. This is the one irreversible thing the agent
    does, and the guard lives HERE rather than at the call sites on purpose:
    there are already two ways in (the book_appointment intent branch and the
    tail of _continue_pending), a third will eventually be added, and a guard
    that has to be remembered at each entry point is a guard that will
    eventually be forgotten at one of them. Refusing inside the function makes
    the write unreachable by omission rather than by discipline.

    UPDATED BY SOURAV -- `language` is new (default "bengali" so every
    pre-existing caller of this function keeps working unchanged): see
    the module-level detect_language import comment above for why this
    is threaded through now. Passed straight to booking_reply() below,
    which already accepted it and already had a full 4-language body --
    nothing in that function needed to change, only this call site. The
    insufficient-verified-information outcome below still only speaks
    INSUFFICIENT_VERIFIED_INFORMATION_BN, which stays Bengali-only by
    design for now (see that constant's own comment in
    agent/reply_templates.py) -- not yet part of this threading.

    "The agent says it cannot confirm rather than guessing": a
    success=True response is trusted only after confirming
    confirmation_id/date/time_slot actually came back non-empty. This is
    the ONE real trigger that story wires up -- deliberately just a
    presence check, not shape validation, not a business-rule check, not
    a database re-query (see agent/outcomes.py's module docstring for
    exactly which sibling stories those belong to instead). A caller is
    never read a confirmation number the code cannot itself verify it
    received.
    """
    if not confirmed:
        # Not an error the caller caused -- most likely a new code path that
        # skipped the readback. Log it loudly, then do the safe thing rather
        # than the convenient one: ask, and write only if they say yes.
        #
        # FIXED BY SOURAV -- KCD-448. This still called the older, Bengali-
        # only booking_confirm_prompt(slots), unchanged since before
        # booking_confirmation_prompt() (the 4-language readback this story
        # built) existed. A caller reaching this rare defensive branch in
        # English/Hinglish/Banglish would have been asked to confirm their
        # own booking in Bengali. `language` is already a parameter here
        # (see this function's own docstring) -- this branch just never
        # used it.
        logger.error("[%s] booking reached _finish_booking unconfirmed -- "
                     "refusing the write and asking the caller", session.call_id)
        session.pending = {
            "awaiting": "confirm_booking", "slots": slots,
            "candidates": None, "offered_date": slots.get("date"), "retries": 0,
        }
        await _speak(session, booking_confirmation_prompt(slots, language=language))
        return

    session.pending = None
    try:
        result = await _tools.book_appointment(
            slots["doctor_name"], slots["date"], slots["time_slot"],
            slots["patient_name"], slots["phone"],
        )
    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, SYSTEM_UNREACHABLE_BN,
                     fallback_reason="tool_failure")
        return
    # story title: The agent says it cannot confirm rather than guessing
    # user story: As a caller, I want to be told plainly when the system
    #   cannot verify something, so that I am not given a confident guess.
    # acceptance criteria: The insufficient-verified-information outcome has
    #   its own template per language, its own metric and its own escalation
    #   path, distinct from not-found and from an infrastructure apology. Its
    #   rate is reported per intent because a rise means a data or
    #   integration problem.
    #
    # THE WRITE HAPPENED. Whether the agent can say what it did is a separate
    # question, and this is where it is asked -- after the POST returned
    # success and before a single one of its values is read aloud.
    #
    # Routed to its own outcome rather than to either neighbour, because the
    # instruction to the caller is different from both: not "that does not
    # exist", not "call back", but "it IS booked, we will follow up, do not
    # rebook". Telling them to call back here is what produces the duplicate
    # appointment.
    # story title: Near matches are offered rather than guessed or refused
    # user story: As a caller naming something loosely, I want the close
    #   matches offered, so that I am not told my test does not exist when it
    #   does.
    # acceptance criteria: When several catalogue rows fall within the match
    #   band the agent offers up to three by name and asks which. Candidates
    #   are generated across every supported language and romanised spelling.
    #   The did-you-mean path covers the ambiguous case and not only total
    #   failure.
    #
    # NO WRITE UNDER AMBIGUITY. This is the worst version of the bug the story
    # names: booking the higher-scoring of two plausible doctors leaves the
    # caller believing they have an appointment, and they do -- with someone
    # else. clinic-api refuses the write and hands back the candidates, so
    # this asks instead of reporting "no such doctor" for a doctor who exists
    # twice over.
    #
    # The booking state is NOT resumed automatically after the choice. The
    # caller picks a doctor, hears their availability, and walks the booking
    # again with the readback intact -- longer, and the only version where
    # every field is re-confirmed against the doctor they actually chose.
    if result.get("reason") == "doctor_ambiguous":
        candidates = result.get("candidates") or []
        logger.info("[%s] booking refused: %r matches %d doctors",
                    session.call_id, slots.get("doctor_name"), len(candidates))
        session.pending = {
            "awaiting": "entity_choice", "intent": "doctor_availability",
            "slots": {}, "candidates": candidates,
            "offered_date": slots.get("date"), "retries": 0,
        } if candidates else None
        await _speak(session, near_match_prompt(candidates))
        return

    unverified = outcomes.missing_booking_write_fields(result)
    if unverified:
        logger.error("[%s] booking write is unverifiable -- missing %s. The "
                     "appointment WAS created; the response did not carry it back.",
                     session.call_id, unverified)
        if _tools is not None:
            _tools.outcomes.record("book_appointment", tool_outcome.INSUFFICIENT)
        turn_log.record_insufficient(session.call_id, session.utt_seq,
                                     "book_appointment", unverified)
        await _speak(session, INSUFFICIENT_VERIFIED_INFORMATION_BN)
        return
    await _speak(session, booking_reply(slots, result, language=language))


# ADDED BY SOURAV -- "Caller asks to be called back" story. Same shape as
# _finish_booking() just above: place the write and clear pending
# regardless of outcome, catch ToolCallError for the infra-apology path,
# and withhold the confirmation via missing_callback_write_fields() rather
# than ever speaking a callback_id the response didn't actually confirm
# (mirrors _finish_booking()'s own missing_booking_write_fields() check --
# see agent/outcomes.py for both). The ONLY caller is the "confirm_callback"
# branch of _continue_pending, below.
async def _finish_callback(session: CallSession, slots: dict, language: str = "bengali"):
    session.pending = None
    try:
        result = await _tools.request_callback(
            slots["phone"], slots["callback_time_window"], slots.get("callback_reason"),
        )
    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                     fallback_reason="tool_failure")
        return

    if result.get("success"):
        missing = missing_callback_write_fields(result)
        if missing:
            logger.error("[%s] callback request reported success but missing %s -- withholding confirmation",
                         session.call_id, missing)
            record_insufficient_verified_information(
                intent="request_callback", field=",".join(missing),
                reason="missing_after_success", call_id=session.call_id,
            )
            # INSUFFICIENT_VERIFIED_INFORMATION_BN is Bengali-only by design
            # (see its own comment in agent/reply_templates.py) -- `language`
            # is accepted here for signature parity with the rest of this
            # story's functions but not yet threaded into this one sentence.
            await _speak(session, INSUFFICIENT_VERIFIED_INFORMATION_BN,
                         fallback_reason="insufficient_verified_information")
            return

    await _speak(session, callback_scheduled_reply(slots, result, language=language))


# ADDED BY SOURAV -- "Caller wants to make a complaint" story. Same shape as
# _finish_callback() just above (place the write, catch ToolCallError for
# the infra-apology path) but with no "confirm_*" pending state leading
# into it: BOTH call sites (the complaint guard inside _continue_pending
# below, and the "complaint" branch of _dispatch_turn_inner's dispatch
# chain) call this directly, on the SAME turn the complaint was said --
# AC 2/AC 5's zero-negotiation contract leaves nothing to confirm first,
# the same reasoning human_direct_request's own branch already follows.
#
# `session.pending` is cleared unconditionally on entry, same as the
# immediate-human-request guard just above in _continue_pending -- a
# complaint said mid-flow abandons that flow outright, with no attempt to
# finish, resume, or ask about it.
#
# `text` is passed through to agent/tools_client.py's submit_complaint()
# EXACTLY as received -- never trimmed, translated, or summarised -- so
# clinic-api's ComplaintRecord.complaint_text ends up holding precisely
# what AC 3 asks for ("captured verbatim"). No confirmation field is ever
# read back to the caller (complaint_acknowledged_reply() names no
# complaint_id or status), so there is no missing_*_write_fields() check
# here the way _finish_callback()/_finish_booking() need one -- the only
# thing that can go wrong is the write itself failing, which the
# ToolCallError branch below already covers honestly.
async def _finish_complaint(session: CallSession, text: str, language: str = "bengali"):
    session.pending = None
    try:
        result = await _tools.submit_complaint(text)
    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                     fallback_reason="tool_failure")
        return

    if not result.get("success"):
        logger.error("[%s] complaint submission reported failure -- withholding acknowledgment",
                     session.call_id)
        await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                     fallback_reason="tool_failure")
        return

    record_complaint_filed(call_id=session.call_id)
    await _speak(session, complaint_acknowledged_reply(language=language))


# ADDED BY SOURAV -- "Caller wants to speak to a doctor personally" story.
# Same shape as _finish_complaint() just above (BOTH call sites -- the
# guard inside _continue_pending above, and the "doctor_personal_request"
# branch of _dispatch_turn_inner's dispatch chain below -- call this
# directly, on the SAME turn the request was made, with no "confirm_*"
# pending state leading into it and none set afterward).
#
# Reuses check_callback_availability()/CALLBACKS_ENABLED/get_clinic_info()
# EXACTLY as the "request_callback" intent's own fresh-dispatch branch
# does further below (same already-cached call, same config switch, same
# pure decision function) -- there is no second callback-availability
# implementation here, per this story's own "reuse the existing callback
# availability logic" instruction. The result is passed straight into
# doctor_personal_request_reply() as a plain bool: that function decides
# the wording, this function only decides the FACT of whether a callback
# is offerable right now.
#
# Deliberately does NOT call request_callback() or write any
# CallbackRequest row itself -- this turn only ANSWERS the caller's
# question about what is possible; the caller's own next ordinary
# utterance ("please call me back" / "book me an appointment with Dr
# Sen") is what actually starts the request_callback/book_appointment
# flow, through those intents' own existing, already-tested dispatch
# branches. See agent/doctor_personal_request.py's own module docstring
# for why reusing those whole flows, rather than building a second
# pending-choice state machine here, is the "reuse it where appropriate"
# this story asks for.
async def _finish_doctor_personal_request(session: CallSession, language: str = "bengali"):
    session.pending = None

    # Same short-circuit as the "request_callback" intent's own
    # fresh-dispatch branch further below: CALLBACKS_ENABLED is checked
    # BEFORE ever calling get_clinic_info() -- a deployment with the
    # feature off entirely has no reason to pay for that round-trip just
    # to learn something a config constant already answers.
    # check_callback_availability(None, ..., callbacks_enabled=False)
    # would reach REASON_DISABLED anyway; skipping straight there avoids
    # the unnecessary call without duplicating its own decision logic.
    if not CALLBACKS_ENABLED:
        await _speak(session, doctor_personal_request_reply(False, language=language))
        return

    try:
        hours_result = await _tools.get_clinic_info()
    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                     fallback_reason="tool_failure")
        return

    hours = hours_result.get("hours") if hours_result.get("found") else None
    availability = check_callback_availability(
        hours, datetime.date.today().weekday(),
        datetime.datetime.now().strftime("%H:%M"), CALLBACKS_ENABLED,
    )
    await _speak(session, doctor_personal_request_reply(availability["available"], language=language))


# ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story
# (previously two separate stories, "is my report ready" / "send my
# report"). These two helpers are the only NEW glue _dispatch_turn and
# _continue_pending need: every actual decision (what to say, what pending
# state comes next) lives in agent/report_flow.py's pure interpret_*()
# functions -- see that module's docstring for why. These two functions
# exist only to do the I/O those pure functions cannot do themselves:
# await the tools client, then hand the response to the right interpret_*()
# call and speak/store whatever it returns.
async def _finish_report_flow(session: CallSession, phone: str, result: dict, flow: str,
                               language: str = "bengali"):
    """Common tail for BOTH a fresh report_status/report_send lookup and a
    caller resolving a "which report?" disambiguation (see the
    "which_report" pending state below, which reconstructs a `result`
    locally from the remembered candidate list rather than re-querying).

    UPDATED BY SOURAV -- `language` is new (default "bengali", so every
    pre-existing caller of this function keeps working unchanged); see
    the module-level detect_language import comment above. Passed
    straight to agent/report_flow.py's interpret_*() functions below,
    which already accepted it -- only this call site needed updating."""
    text, pending = interpret_report_status_result(result, flow, language=language)
    if pending and pending.get("awaiting") == "__request_delivery_now__":
        # flow == "report_send" on a READY + delivery-enabled report: the
        # caller already asked for delivery, so go straight to requesting
        # an OTP rather than asking "shall I send it?" first (that offer
        # question is only for flow == "report_status").
        report_number = pending["report_number"]
        try:
            delivery_result = await _tools.request_report_delivery(phone, report_number)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                         fallback_reason="tool_failure")
            return
        text, pending = interpret_delivery_request_result(delivery_result, report_number, language=language)
    if pending is not None:
        # phone is never something the caller re-supplies mid-flow (RULE 15
        # -- identity was already resolved) -- carry it forward on every
        # pending dict this story introduces so later states never need to
        # re-ask for it.
        pending["phone"] = phone
    session.pending = pending
    if text:
        await _speak(session, text)


async def _handle_report_lookup(session: CallSession, phone: str, test_name: str | None, flow: str,
                                 language: str = "bengali"):
    """Entry point for BOTH the report_status and report_send intents (see
    _dispatch_turn below) once a phone number is in hand, and for the
    "phone" pending state once a caller who was first asked for one gives
    it. `flow` tells report_status and report_send apart -- same lookup,
    different thing to do once a READY+enabled report is found (see
    agent/report_flow.py's interpret_report_status_result).

    UPDATED BY SOURAV -- `language` is new (default "bengali", so every
    pre-existing caller keeps working unchanged); threaded straight
    through to _finish_report_flow. See the module-level detect_language
    import comment above."""
    try:
        result = await _tools.get_report_status(phone, test_name)
    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        session.pending = None
        await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                     fallback_reason="tool_failure")
        return
    await _finish_report_flow(session, phone, result, flow, language=language)


# ADDED BY SOURAV -- KCD-383 ("Caller asks whether their report is
# ready"). agent/reply_templates.py's report_status_reply() now offers
# BOTH delivery and a callback when a report is READY+delivery_enabled;
# this is the glue the "confirm_delivery" branch of _continue_pending
# below calls when the caller's answer to that offer names the callback
# option instead of a plain yes/no (agent/slot_parse.py's
# looks_like_callback_preference()). It reuses the exact SAME
# availability check and slot-filling pending states
# ("callback_time_window"/"callback_phone"/"confirm_callback") the
# "request_callback" intent's own dispatch branch further below already
# owns (agent/callback_flow.py/agent/callback_config.py) -- this is
# "routing to the existing request_callback intent", per the KCD-383 AC,
# not a second, parallel callback pipeline. Deliberately does NOT prefill
# a callback phone number from the report flow's already-resolved
# `phone` (that number identified the PATIENT for RULE 15 purposes; the
# number to actually call back on is a separate fact the caller has not
# yet stated, and this module's "never invent a fact" discipline applies
# here exactly as it does everywhere else) -- the caller is asked for it
# normally, the same as any other fresh request_callback flow.
async def _pivot_report_offer_to_callback(session: CallSession, report_pending: dict,
                                           language: str = "bengali"):
    if not CALLBACKS_ENABLED:
        session.pending = None
        await _speak(session, callback_unavailable_reply("disabled", language=language))
        return

    # Same already-cached get_clinic_info() call the "request_callback"
    # intent branch below uses -- no new tool, no new network round-trip
    # pattern (agent/reference_data_cache.py).
    hours_result = await _tools.get_clinic_info()
    hours = hours_result.get("hours") if hours_result.get("found") else None
    availability = check_callback_availability(
        hours, datetime.date.today().weekday(),
        datetime.datetime.now().strftime("%H:%M"), CALLBACKS_ENABLED,
    )
    if not availability["available"]:
        session.pending = None
        await _speak(session, callback_unavailable_reply(availability["reason"], language=language))
        return

    # Acceptance Criterion 1's "preserving the conversation context and
    # reason" -- grounded in the ONE real fact already in hand (which
    # report the caller was just asking about), never a guessed or
    # generic reason, same discipline build_callback_reason()'s own
    # docstring holds the fresh-intent path to.
    slots = {
        "callback_reason": build_callback_reason(
            None, active_test=report_pending.get("test_name"),
        ),
    }
    missing = _next_missing_callback(slots)
    if missing is None:
        session.pending = {
            "awaiting": "confirm_callback", "slots": slots, "candidates": None,
            "offered_date": None, "retries": 0,
        }
        await _speak(session, callback_confirmation_prompt(slots, language=language))
        return

    session.pending = {
        "awaiting": missing, "slots": slots, "candidates": None,
        "offered_date": None, "retries": 0,
    }
    await _speak(session, missing_slot_prompt("request_callback", missing, language=language))


async def _resolve_comparable_entity(name: str) -> dict:
    """ADDED BY SOURAV -- "Caller asks the agent to compare two options"
    story. agent/llm.py deliberately never classifies whether a caller-
    named term is a TEST or a PACKAGE (see its own CLINICAL SAFETY NOTE
    and the "compare_option_a"/"compare_option_b" slot rule) -- it only
    ever copies the literal span the caller said, same discipline as
    "test_name"/"package_name" elsewhere in this file. Resolving WHICH
    catalogue a name belongs to is this function's only job, done the
    same way a human clinic-desk operator would: try the test catalogue
    first (get_test_rate), and only if that comes back not-found, try the
    package catalogue (search_health_package) -- tests significantly
    outnumber packages in this clinic's catalogue, so this order resolves
    the common case (comparing two tests) in a single tool call.

    Returns the underlying tool response dict, unmodified, plus one added
    key `"kind"`: "test" | "package" | "not_found" -- read by
    agent/compare_flow.py's build_comparison() (which fields it reads
    depends on this tag) and agent/reply_templates.py's
    compare_options_reply() (which alias field to prefer). Never raises
    ToolCallError itself -- a caller of this function (the compare_options
    dispatch branch below) awaits it for BOTH sides before deciding how to
    handle a tool failure, same as every other two-tool-call intent in
    this file.
    """
    test_result = await _tools.get_test_rate(name)
    if test_result.get("found"):
        return {**test_result, "kind": "test"}
    package_result = await _tools.search_health_package(name)
    if package_result.get("found"):
        return {**package_result, "kind": "package"}
    # Neither catalogue has it -- prefer the package lookup's own
    # did_you_mean suggestions (already computed by clinic-api) since a
    # caller who names something unfamiliar in a comparison is at least as
    # likely to mean a package as a plain test; the test lookup's own
    # did_you_mean is not lost, just not the one used here, since this
    # dict only needs to say "not found", not carry both catalogues' near
    # matches.
    return {**package_result, "kind": "not_found"}


# ADDED BY SOURAV -- "Caller asks a follow-up that depends on the previous
# answer" story. Which result-dict field carries the CATALOGUE's own
# canonical name for each trackable slot -- see _remember_primary_entity()
# below for why the canonical name, not the caller's raw words, is what
# gets remembered.
_CANONICAL_NAME_FIELD = {"test_name": "test_name", "doctor_name": "doctor_name", "package_name": "package_name"}


def _session_state(session: CallSession) -> DialogueState | None:
    """ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
    previous answer" story. Every REAL CallSession always has `.state`
    (see its own __init__), but a large number of EXISTING tests written
    before this story build a lightweight `types.SimpleNamespace` fake
    session instead -- with only the specific attributes that story
    needed at the time, never `.state`. Rather than retrofit `.state=...`
    into every one of those pre-existing fakes (an unrelated change to
    18+ test files this story has no reason to touch), every call site
    below goes through this helper and treats "no `.state` attribute at
    all" the same as "follow-up resolution is simply not available this
    turn" -- the exact behaviour those tests already expect and pass
    with today, completely unaffected by this story. A real caller,
    which always has `.state`, is never affected by this fallback.
    """
    return getattr(session, "state", None)


def _remember_primary_entity(session: CallSession, intent: str, slots: dict, result: dict | None) -> None:
    """Called right after a single-primary-entity intent's lookup
    returns (see agent/state.py's primary_slot_for_intent() for exactly
    which intents this applies to), successful or not.

    Only a `result.get("found")` lookup updates agent/state.py's
    DialogueState -- and even then with the CATALOGUE's own canonical
    name (e.g. clinic-api's own `test_name`), never the caller's raw
    spoken words: a later follow-up backfills THIS value straight into
    the next tool call's argument (see main.py's dispatch, right after
    resolve_follow_up()), and the canonical name is guaranteed to still
    match on that next lookup the way an ASR-mangled or partial spoken
    form is not. A not-found result intentionally changes nothing --
    nothing new was actually confirmed to exist this turn, so clobbering
    a still-valid, previously-tracked entity with a miss would make the
    NEXT follow-up resolve to nothing instead of the last real one.
    """
    state = _session_state(session)
    if state is None:
        return
    slot_key = primary_slot_for_intent(intent)
    if slot_key is None or not result or not result.get("found"):
        return
    kind = kind_for_slot(slot_key)
    if kind is None:
        return
    canonical = result.get(_CANONICAL_NAME_FIELD[slot_key]) or slots.get(slot_key)
    if canonical:
        state.mark(kind, canonical)


def _remember_compared_entities(session: CallSession, entity_a: dict, entity_b: dict) -> None:
    """Called after compare_options resolves both sides (dispatch branch
    and its "compare_options_slot" continuation below both call this).

    If both sides turned out to be the SAME kind (two tests, or two
    packages) with DIFFERENT canonical names, that kind becomes AMBIGUOUS
    for the next turn's follow-up (see agent/state.py's mark_ambiguous())
    -- a caller who just asked to compare CBC and Lipid Profile, then
    says "does IT need a prescription?", cannot honestly have either one
    guessed. If both sides are the same kind with the SAME name (a caller
    comparing a test to itself), or only one side resolved to that kind
    at all, there is only one real candidate -- mark() as usual. A
    not_found side never contributes anything to remember, same
    not-found-changes-nothing rule as _remember_primary_entity() above.
    """
    state = _session_state(session)
    if state is None:
        return
    by_kind: dict[str, list[str]] = {}
    for entity in (entity_a, entity_b):
        kind = entity.get("kind")
        if kind not in ("test", "package"):
            continue
        # "test_name" or "package_name" -- the exact field
        # _resolve_comparable_entity() tags each result with.
        name = entity.get(f"{kind}_name")
        if name:
            by_kind.setdefault(kind, []).append(name)
    for kind, names in by_kind.items():
        distinct = list(dict.fromkeys(names))  # de-dupe, order-preserved
        if len(distinct) > 1:
            state.mark_ambiguous(kind, distinct)
        else:
            state.mark(kind, distinct[0])


async def _continue_pending(session: CallSession, text: str) -> bool:
    """The fix for "appointment pipeline breaking": every turn used to be
    classified from a bare transcript with ZERO memory of the turn before
    it (see _resolve_intent / agent/llm.py's docstring -- one utterance
    in, one classification out, nothing carried over). A caller who had
    just been asked "কোন দিন চান?" and replied "আজ" produced a fresh,
    context-free classification of the single word "আজ", which the model
    has no way to recognise as a date answer -- it almost always came
    back "unclear", and the booking that was three-quarters filled a
    moment ago silently died with no record it had ever started.

    This function is the session's memory. While session.pending is set,
    EVERY turn is routed here first (see _dispatch_turn), and is
    interpreted against exactly the one field pending["awaiting"] says was
    just asked for -- using agent/slot_parse.py's local parsers, not
    another LLM call (see that module's docstring for why a fresh
    classification is the wrong tool for a reply this short). The LLM is
    not consulted again until the flow ends, one way or another.

    pending shape: {
        "awaiting": "doctor_choice" | "department_date" | "date" | "time_slot"
                    | "patient_name" | "phone" | "confirm_booking" | "confirm_correction",
        "slots": {<whatever of the 5 booking fields is already known>},
        "candidates": [{"name", "name_bn"}, ...] | None,  # only for "doctor_choice"
        "offered_date": "<iso>" | None,  # the date main.py already SPOKE to
                                          # the caller ("today", or a
                                          # next-available date) -- lets a
                                          # bare "হ্যাঁ" confirm THAT date
                                          # instead of literally "today"
        "retries": int,
    }

    ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story
    adds FOUR more "awaiting" values, handled in their own block below
    (checked BEFORE the universal "না" escape hatch, same reason
    "confirm_booking"/"confirm_correction" already are: that hatch's
    booking-specific wording would be wrong mid-report-flow, and RULE 4/9
    need their own "না means decline delivery, not abandon the call"
    wording and their own OTP-disclosure-attempt handling instead):
        "report_phone"     -- report_status/report_send asked for a phone
                               number (RULE 15, identity-by-phone-first);
                               pending also carries "flow" and "test_name".
                               NOT plain "phone" -- that string is already
                               used by the booking flow below (a caller
                               correcting a booking's phone number
                               re-enters awaiting="phone"); a shared name
                               made a correcting-a-booking's-phone-number
                               caller get routed into the report flow
                               instead (caught by test_booking_readback.py).
        "which_report"     -- RULE 13, caller has more than one report and
                               was asked which; pending carries "flow" and
                               the remembered "candidates" list.
        "confirm_delivery" -- the "shall I send it to your phone, or would
                               you prefer a callback?" offer after a
                               report_status lookup found a READY +
                               delivery-enabled report (RULE 4); pending
                               carries "report_number", "test_name" and
                               "phone". ADDED BY SOURAV -- KCD-383: a
                               caller who answers with a callback
                               preference instead of yes/no is routed into
                               the request_callback flow by
                               _pivot_report_offer_to_callback() (see that
                               function's own docstring above), via
                               looks_like_callback_preference(), checked
                               the same way looks_like_otp_disclosure_
                               request() is below -- only after
                               is_affirmative()/is_negative() have already
                               failed on the same utterance, so a plain
                               "হ্যাঁ"/"না" is never misrouted.
        "otp_code"         -- RULE 4-9, waiting for the caller to speak
                               back the OTP just sent; pending carries
                               "report_number" and "phone". A caller who
                               asks to be TOLD the otp instead of speaking
                               it back is refused (RULE 9 / ATTACK 8) via
                               looks_like_otp_disclosure_request(), not
                               treated as an ordinary unparseable reply.
    All four pending dicts also carry "phone" (see _finish_report_flow /
    _handle_report_lookup above) so none of these states ever needs to
    re-ask for a phone number it already resolved identity with.

    ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables adds TWO
    more "awaiting" values:
        "billing_phone"          -- billing_balance asked for a phone
                                     number (RULE 14/15, same as
                                     "report_phone" above). Own distinct
                                     string for the same reason
                                     "report_phone" isn't just "phone".
        "insurance_coverage_slot" -- insurance_coverage is missing
                                     test_name and/or insurance_provider_
                                     name; pending also carries "slots"
                                     (whichever of the two is already
                                     known) and "missing_field" (which one
                                     this turn's reply fills). Unlike
                                     phone/date/time_slot, both fields are
                                     free-text named entities with no
                                     local grammar -- the caller's
                                     utterance is accepted verbatim for
                                     whichever field is missing, same as
                                     agent/llm.py's own extraction rule
                                     for these two slots ("copy the term
                                     as said, do not normalize").

    ADDED BY SOURAV -- "Caller asks something the agent does not cover"
    story adds ONE more "awaiting" value:
        "out_of_scope_choice" -- the caller was offered a choice (connect
                                  to a human, or contact the counter
                                  themselves) after an "out_of_scope"
                                  intent; carries no lookup state at all
                                  (unlike confirm_delivery above, there is
                                  no report_number/phone to remember), just
                                  "retries", same yes/no/unparseable shape
                                  as confirm_delivery.

    ADDED BY SOURAV -- "Caller asks the agent to compare two options"
    story adds ONE more "awaiting" value:
        "compare_options_slot" -- compare_options is missing
                                   compare_option_a and/or compare_option_b;
                                   pending also carries "slots" (whichever
                                   of the two is already known) and
                                   "missing_field" (which one this turn's
                                   reply fills). Same free-text, no-local-
                                   grammar, accept-verbatim shape as
                                   "insurance_coverage_slot" above, for the
                                   identical reason -- neither slot has a
                                   local grammar to parse against, and
                                   agent/llm.py never classifies which one
                                   is a test versus a package anyway (that
                                   happens downstream, in
                                   _resolve_comparable_entity(), only once
                                   both names are in hand).

    ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
    previous answer" story adds ONE more "awaiting" value:
        "follow_up_clarification" -- a brand-new question's primary
                                       entity slot (test/doctor/package)
                                       was left null AND agent/state.py's
                                       tracked state for that kind was
                                       AMBIGUOUS (2+ different entities
                                       discussed a moment ago -- see that
                                       module's own docstring); pending
                                       carries "intent" and "slots" (the
                                       question to RESUME, exactly as
                                       extracted, once the ambiguity is
                                       resolved), "kind", and "candidates"
                                       (the names ambiguous_reference_
                                       reply() just spoke). Unlike every
                                       other pending state above, this one
                                       does not ask for a brand-new piece
                                       of information the caller never
                                       gave -- it asks them to pick which
                                       of two things they ALREADY said a
                                       moment ago they meant, so the reply
                                       is matched against `candidates`
                                       (via _match_candidate_name(), same
                                       trust model as _match_candidate_
                                       doctor()/agent/report_flow.py's
                                       match_candidate_report()) rather
                                       than accepted verbatim the way
                                       insurance_coverage_slot/compare_
                                       options_slot's free-text fields
                                       are -- a stray word or two around
                                       the real name should not silently
                                       fail to match.

    ADDED BY SOURAV -- "Caller asks to be called back" story adds THREE
    more "awaiting" values:
        "callback_time_window" -- request_callback is missing a time
                                   window; free-text, accepted verbatim
                                   (same as insurance_coverage_slot/
                                   compare_options_slot above -- a time
                                   window like "this evening" has no local
                                   grammar to parse against).
        "callback_phone"       -- request_callback is missing a phone
                                   number. NOT plain "phone" -- same
                                   collision reasoning as "report_phone"/
                                   "billing_phone" above, since the
                                   booking flow's own shared tail further
                                   down already owns bare "phone".
        "confirm_callback"     -- both fields are known; the caller is
                                   read back the number and time window
                                   and asked to confirm before the write
                                   (Answer Quality and Grounding, same
                                   shape as "confirm_booking" above).
    All three carry "slots" (whatever of "callback_time_window"/"phone"/
    "callback_reason" is already known) -- see agent/callback_flow.py's
    own module docstring for the availability check that runs BEFORE any
    of these three states is ever entered, and _next_missing_callback()
    just above _continue_pending's own definition for the field order.

    Returns True when the turn was fully handled here (caller must not
    also run intent extraction on top of it); False to fall through to
    the normal pipeline -- either because there was no pending flow, or
    because this one gave up on it after repeated unparseable replies.
    """
    pending = session.pending
    if pending is None:
        return False

    # ADDED BY SOURAV -- see the module-level detect_language import
    # comment above for the real bug this fixes. Detected fresh from
    # THIS turn's own utterance (not carried over from an earlier turn,
    # and not stored on `pending`) -- a caller can code-switch mid-call,
    # and every reply below should reflect what they just said, not what
    # they said several turns ago when the flow started.
    language = detect_language(text)

    # ADDED BY SOURAV -- "Caller asks for a person immediately" story
    # (Epic: Conversation -- Difficult, Sensitive and Edge Cases). Checked
    # here, FIRST, before `awaiting` is even read -- ahead of every
    # flow-specific branch below, including the universal "না" escape
    # hatch and the "confirm_transcript" echo-back. Without this, a caller
    # mid-booking (or mid-OTP-verification, or being asked to confirm what
    # was heard) who says "just connect me to a person" would have that
    # sentence parsed as an attempted answer to whatever field happened to
    # be pending -- misreading the one sentence that most needs to be
    # heard as itself, and the literal shape of "feeling trapped" this
    # story's own user narrative names. See agent/human_fast_path.py's
    # module docstring for the full reasoning. Whatever flow was in
    # progress is simply abandoned, with no attempt to finish it, resume
    # it, or ask why -- the same zero-negotiation contract the AC asks
    # for, regardless of which pending state this turn interrupts.
    if is_immediate_human_request(text):
        logger.info("[%s] immediate human request interrupted an in-progress "
                    "flow (awaiting=%s) -- abandoning it, escalating now",
                    session.call_id, pending.get("awaiting"))
        session.pending = None
        record_immediate_human_handoff(call_id=session.call_id)
        await _speak(session, human_fallback_reply(language=language))
        return True

    # ADDED BY SOURAV -- "Caller wants to make a complaint" story. Checked
    # here, immediately after the immediate-human-request guard above and
    # still before `awaiting` is even read -- ahead of every flow-specific
    # branch below. Without this, a caller mid-booking (or mid-OTP-
    # verification, or being asked to confirm what was heard) who suddenly
    # says "actually I want to file a complaint about..." would have that
    # sentence parsed as an attempted answer to whatever field was pending,
    # silently swallowing the complaint into the wrong flow -- see
    # agent/complaint_flow.py's own module docstring for the full
    # reasoning. Whatever flow was in progress is abandoned outright, with
    # no attempt to finish, resume, or ask about it -- AC 5 leaves nothing
    # to do but record this complaint on this turn.
    if is_complaint(text):
        logger.info("[%s] complaint interrupted an in-progress flow "
                    "(awaiting=%s) -- abandoning it, recording complaint now",
                    session.call_id, pending.get("awaiting"))
        await _finish_complaint(session, text, language=language)
        return True

    # ADDED BY SOURAV -- "Caller wants to speak to a doctor personally"
    # story. Checked here, immediately after the complaint guard above and
    # still before `awaiting` is even read -- ahead of every flow-specific
    # branch below. Without this, a caller mid-booking (or mid-OTP-
    # verification) who suddenly says "actually I want to speak to a
    # doctor personally" would have that sentence parsed as an attempted
    # answer to whatever field was pending, and the booking/report/
    # callback flow already in progress would keep going as though nothing
    # had been said -- exactly the "left waiting for something that will
    # not happen" this story's own user narrative names, just from the
    # other direction (a flow silently continuing instead of a promise
    # silently going unfulfilled). Whatever flow was in progress is
    # abandoned outright, with no attempt to finish, resume, or ask about
    # it -- same zero-negotiation shape as the complaint guard just above.
    if is_doctor_personal_request(text):
        logger.info("[%s] doctor-personal-request interrupted an in-progress "
                    "flow (awaiting=%s) -- abandoning it, answering now",
                    session.call_id, pending.get("awaiting"))
        await _finish_doctor_personal_request(session, language=language)
        return True

    awaiting = pending["awaiting"]

    # NOTE: confirm_booking/confirm_correction are handled below, together
    # with the universal "না" escape hatch -- see that hatch's own comment
    # for why confirm_correction is deliberately NOT excluded from it (a
    # considered divergence from dev_sourav, where both states skip the
    # hatch and are handled in an earlier, language-aware pair of branches
    # here instead). That earlier pair duplicated this behaviour with a
    # regression -- it never reproduced the immediate-abandon-on-"না"
    # behaviour tests/test_booking_readback.py pins for confirm_correction
    # -- so it was removed rather than kept as a second, unreachable
    # implementation; its one real improvement, threading `language`
    # through the replies below, was folded into the surviving branches
    # instead.

    # ADDED BY SOURAV -- report_status/report_send combined story's four new
    # pending states. Checked here, BEFORE the universal booking escape
    # hatch just below, for the same reason "confirm_booking"/
    # "confirm_correction" already are (see this function's docstring):
    # a "না" here means something specific to a report flow, not
    # "abandon the appointment" (there is no appointment in this flow).
    #
    # "report_phone", NOT "phone": the pre-existing booking flow already
    # uses the bare string "phone" as an awaiting value (see the shared
    # date/time_slot/patient_name/phone tail further down, and its
    # "confirm_correction" re-entry) -- a caller correcting a BOOKING's
    # phone number was briefly being routed into this report-flow handler
    # instead, because this check ran first and matched on the same
    # string. Caught by test_booking_readback.py's correction-round-trip
    # test failing once these states were wired in. See
    # agent/report_flow.py's AWAITING_REPORT_PHONE comment for the same
    # note from the other side of this collision.
    if awaiting == "report_phone":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        phone = parse_phone(text)
        if phone is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt(pending["flow"], "phone", language=language))
            return True
        await _handle_report_lookup(session, phone, pending.get("test_name"), pending["flow"],
                                     language=language)
        return True

    # ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables.
    # Outstanding Balance / Billing story. Own distinct string, NOT
    # "phone" or "report_phone" -- same collision reasoning as
    # "report_phone" above: a caller correcting a BOOKING's phone number,
    # or resuming a report flow's phone ask, must never be routed here
    # instead just because the bare string matched.
    if awaiting == "billing_phone":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        phone = parse_phone(text)
        if phone is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("billing_balance", "phone", language=language))
            return True
        # FIXED BY SOURAV -- Phase 2 end-to-end testing caught a real bug
        # here: this branch spoke the real answer but never cleared
        # session.pending, so the call stayed stuck in "awaiting a phone
        # number" afterward -- the caller's NEXT utterance, whatever it
        # was, would have been misinterpreted as another phone attempt
        # instead of a fresh question. Must be cleared BEFORE the tool
        # call, same ordering as the insurance_coverage_slot branch below,
        # so a slow/failing tool call never leaves pending in a stale
        # state either.
        session.pending = None
        result = await _tools.get_patient_billing(phone)
        await _speak(session, billing_balance_reply(result, language=language))
        return True

    # ADDED BY SOURAV -- Phase 1: Insurance Coverage Policy story. Unlike
    # phone/date/time_slot, "test_name" and "insurance_provider_name" are
    # free-text named entities with no local grammar to parse (per agent/
    # llm.py's own slot rule: copy the term as said, do not normalize --
    # the actual alias/fuzzy matching happens downstream in clinic-api's
    # _find_lab_test()/_find_insurance_provider()), so accepting the
    # caller's utterance verbatim for whichever field pending["missing_
    # field"] names IS the correct local equivalent of the LLM's own
    # extraction rule for these two slots, not a shortcut.
    if awaiting == "insurance_coverage_slot":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        value = text.strip()
        if not value:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("insurance_coverage", pending["missing_field"],
                                                        language=language))
            return True
        pending["slots"][pending["missing_field"]] = value
        pending["retries"] = 0
        still_missing = next(
            (f for f in ("test_name", "insurance_provider_name") if not pending["slots"].get(f)), None,
        )
        if still_missing:
            pending["missing_field"] = still_missing
            await _speak(session, missing_slot_prompt("insurance_coverage", still_missing, language=language))
            return True
        final_slots = pending["slots"]
        session.pending = None
        result = await _tools.get_insurance_coverage(
            final_slots["test_name"], final_slots["insurance_provider_name"],
        )
        await _speak(session, insurance_coverage_reply(final_slots, result, language=language))
        return True

    if awaiting == "compare_options_slot":
        # Mirrors "insurance_coverage_slot" immediately above exactly --
        # same two-free-text-slots shape, same accept-verbatim discipline
        # (agent/llm.py never classifies test-vs-package for these two
        # slots either; see this file's own _resolve_comparable_entity()
        # for where that resolution actually happens, only once both
        # names are known).
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        value = text.strip()
        if not value:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("compare_options", pending["missing_field"],
                                                        language=language))
            return True
        pending["slots"][pending["missing_field"]] = value
        pending["retries"] = 0
        still_missing = next(
            (f for f in ("compare_option_a", "compare_option_b") if not pending["slots"].get(f)), None,
        )
        if still_missing:
            pending["missing_field"] = still_missing
            await _speak(session, missing_slot_prompt("compare_options", still_missing, language=language))
            return True
        final_slots = pending["slots"]
        session.pending = None
        name_a, name_b = final_slots["compare_option_a"], final_slots["compare_option_b"]
        entity_a, entity_b = await _resolve_comparable_entity(name_a), await _resolve_comparable_entity(name_b)
        comparison = build_comparison(entity_a, entity_b)
        await _speak(session, compare_options_reply(name_a, name_b, entity_a, entity_b, comparison, language=language))
        _remember_compared_entities(session, entity_a, entity_b)
        return True

    if awaiting == "follow_up_clarification":
        # ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
        # previous answer" story. Unlike every OTHER pending state above,
        # this is not asking for a brand-new piece of information -- it
        # is asking the caller to pick which of two things they ALREADY
        # named a moment ago they meant (see agent/state.py's own
        # docstring on ambiguity, and this function's docstring above),
        # so the reply is matched against the small `candidates` list
        # rather than accepted verbatim.
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        matched = _match_candidate_name(text, pending["candidates"])
        if not matched:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, ambiguous_reference_reply(pending["kind"], pending["candidates"], language=language))
            return True
        session.pending = None
        # The caller just resolved the ambiguity themselves -- collapsing
        # the tracked state back down to this one name means the NEXT
        # follow-up (AC3's "three consecutive follow-ups without
        # re-prompting") does not need to ask again.
        state = _session_state(session)
        if state is not None:
            state.mark(pending["kind"], matched)
        resolved_slots = dict(pending["slots"])
        resolved_slots[primary_slot_for_intent(pending["intent"])] = matched
        # Reuses the exact same per-intent lookup-and-reply logic a solo
        # turn for this intent would use -- see that function's own
        # docstring; every intent agent/state.py can ever produce an
        # ambiguous_kind for is one _resolve_combinable_intent_fragment
        # already knows how to answer.
        reply = await _resolve_combinable_intent_fragment(session, pending["intent"], resolved_slots, language)
        if reply:
            await _speak(session, reply)
        return True

    # ADDED BY SOURAV -- "Caller asks to be called back" story. Three new
    # scoped states: "callback_time_window" and "callback_phone" (NOT
    # "phone" -- same collision reasoning as "report_phone"/"billing_phone"
    # above, since the pre-existing booking flow's own shared tail further
    # down already treats bare "phone" as ITS awaiting value) collect the
    # two fields this story needs, then "confirm_callback" reads them back
    # before the write -- same "every critical value is read back before
    # it is used" discipline as "confirm_booking" above.
    if awaiting == "callback_time_window":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        # ADDED BY SOURAV -- "The agent accepts a correction and restates"
        # story. Same spontaneous-cross-field check as the booking flow's
        # own generic tail -- the caller is asked for the time window but
        # may instead be correcting the phone number they already gave a
        # moment ago ("actually, my number is..."). See agent/
        # correction_flow.py's own module docstring for the cue-gated,
        # never-guess design.
        correction = detect_callback_correction(
            text, pending["slots"], awaiting, _callback_correction_parsers())
        if correction is not None:
            field, new_value = correction
            logger.info("[%s] spontaneous callback correction: %s -> %r (still awaiting %s)",
                        session.call_id, field, new_value, awaiting)
            pending["slots"][field] = new_value
            ack = correction_acknowledged_reply(field, pending["slots"], language=language)
            await _speak(session, ack + " " + missing_slot_prompt('request_callback', 'callback_time_window', language=language))
            return True
        window = text.strip()
        if not window:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("request_callback", "callback_time_window", language=language))
            return True
        was_correction = pending.pop("_correcting_field", None) == "callback_time_window"
        pending["slots"]["callback_time_window"] = window
        pending["retries"] = 0
        prefix = ""
        if was_correction:
            prefix = correction_acknowledged_reply("callback_time_window", pending["slots"], language=language) + " "
        missing = _next_missing_callback(pending["slots"])
        if missing is None:
            pending["awaiting"] = "confirm_callback"
            await _speak(session, prefix + callback_confirmation_prompt(pending["slots"], language=language))
            return True
        pending["awaiting"] = missing
        await _speak(session, prefix + missing_slot_prompt("request_callback", missing, language=language))
        return True

    if awaiting == "callback_phone":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        # ADDED BY SOURAV -- "The agent accepts a correction and restates"
        # story. Mirrors the "callback_time_window" branch above: the
        # caller is asked for the phone number but may be correcting the
        # time window they already gave.
        correction = detect_callback_correction(
            text, pending["slots"], awaiting, _callback_correction_parsers())
        if correction is not None:
            field, new_value = correction
            logger.info("[%s] spontaneous callback correction: %s -> %r (still awaiting %s)",
                        session.call_id, field, new_value, awaiting)
            pending["slots"][field] = new_value
            ack = correction_acknowledged_reply(field, pending["slots"], language=language)
            await _speak(session, ack + " " + missing_slot_prompt('request_callback', 'callback_phone', language=language))
            return True
        phone = parse_phone(text)
        if phone is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("request_callback", "callback_phone", language=language))
            return True
        was_correction = pending.pop("_correcting_field", None) == "phone"
        pending["slots"]["phone"] = phone
        pending["retries"] = 0
        prefix = ""
        if was_correction:
            prefix = correction_acknowledged_reply("phone", pending["slots"], language=language) + " "
        missing = _next_missing_callback(pending["slots"])
        if missing is None:
            pending["awaiting"] = "confirm_callback"
            await _speak(session, prefix + callback_confirmation_prompt(pending["slots"], language=language))
            return True
        pending["awaiting"] = missing
        await _speak(session, prefix + missing_slot_prompt("request_callback", missing, language=language))
        return True

    if awaiting == "confirm_callback":
        if is_affirmative(text):
            await _finish_callback(session, pending["slots"], language=language)
            return True

        if is_negative(text):
            # story title: Every critical value is read back before it is used
            # acceptance criteria: Phone numbers, dates, times and names are
            #   confirmed aloud before any write, and a rejection opens a
            #   correction path rather than repeating the prompt. Readback is
            #   mandatory regardless of confidence for values that affect a
            #   write.
            #
            # FIXED BY SOURAV -- KCD-448. This used to abandon the whole
            # callback outright here ("ঠিক আছে, তাহলে থাক") -- not even a
            # repeat of the prompt, the stronger failure the AC names: one
            # misheard digit in either of only two fields threw both away,
            # and the caller had to redial and give both again. Same
            # correction-path discipline "confirm_booking" above already
            # has: ask WHICH of the two values is wrong, keep the other one,
            # and re-collect only the disputed field.
            pending["awaiting"] = "confirm_callback_correction"
            pending["retries"] = 0
            await _speak(session, callback_correction_prompt(language=language))
            return True

        # ADDED BY SOURAV -- "The agent accepts a correction and restates"
        # story. Same direct-correction shortcut as "confirm_booking"
        # above: the caller need not say "no" first if they just correct
        # the readback outright.
        correction = detect_callback_correction(
            text, pending["slots"], None, _callback_correction_parsers())
        if correction is not None:
            field, new_value = correction
            logger.info("[%s] correcting callback %s directly from the readback "
                        "(no menu needed)", session.call_id, field)
            pending["slots"][field] = new_value
            pending["retries"] = 0
            ack = correction_acknowledged_reply(field, pending["slots"], language=language)
            await _speak(session, ack + " " + callback_confirmation_prompt(pending['slots'], language=language))
            return True

        # Neither a clear yes nor a clear no -- bounded retries of the
        # SAME confirmation, same posture as "confirm_booking" above.
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False
        await _speak(session, callback_confirmation_prompt(pending["slots"], language=language))
        return True

    # story title: Every critical value is read back before it is used
    # acceptance criteria: ...a rejection opens a correction path rather
    #   than repeating the prompt.
    #
    # ADDED BY SOURAV -- KCD-448. This turn is the caller's answer to
    # callback_correction_prompt() above: which of the two callback values
    # is wrong. Mirrors "confirm_correction" above field for field --
    # re-collecting ONE field and returning to the readback is what makes
    # this a correction rather than a restart, and a caller saying "না"
    # here is not naming a field (the agent has already listed both
    # options), so, like "confirm_correction" above, this state is
    # deliberately NOT covered by the universal "না" escape hatch further
    # down; it is its own abandon instead.
    if awaiting == "confirm_callback_correction":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True

        field = parse_callback_correction_field(text)
        if field is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                logger.info("[%s] callback correction abandoned -- no field named",
                            session.call_id)
                await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
                return True
            await _speak(session, callback_correction_prompt(language=language))
            return True

        logger.info("[%s] correcting callback %s", session.call_id, field)
        # The disputed value is cleared, not merely left to be overwritten --
        # a stale, known-wrong value has no business sitting in slots while
        # it is being re-collected. "callback_phone" is the awaiting STATE
        # name (see parse_callback_correction_field's own docstring); the
        # slots dict itself still keys the value as plain "phone".
        slot_key = "phone" if field == "callback_phone" else field
        pending["slots"].pop(slot_key, None)

        # ADDED BY SOURAV -- "The agent accepts a correction and restates"
        # story. If the caller names the field AND gives its new value in
        # the same breath ("the phone number -- actually it's..."), apply
        # it immediately with an explicit acknowledgment instead of a
        # second round trip.
        parser = _callback_correction_parsers().get(slot_key)
        new_value = parser(text) if parser else None
        if new_value is not None:
            pending["slots"][slot_key] = new_value
            pending["retries"] = 0
            missing = _next_missing_callback(pending["slots"])
            ack = correction_acknowledged_reply(slot_key, pending["slots"], language=language)
            if missing is None:
                pending["awaiting"] = "confirm_callback"
                await _speak(session, ack + " " + callback_confirmation_prompt(pending['slots'], language=language))
            else:
                pending["awaiting"] = missing
                await _speak(session, ack + " " + missing_slot_prompt('request_callback', missing, language=language))
            return True

        pending["awaiting"] = field
        pending["retries"] = 0
        # Marks this re-collection as a CORRECTION (keyed by SLOT name,
        # matching what the "callback_time_window"/"callback_phone" tails
        # above check), so the acknowledgment is spoken once the new
        # value actually arrives instead of silently repeating the exact
        # same question a first-time collection would ask.
        pending["_correcting_field"] = slot_key
        await _speak(session, missing_slot_prompt("request_callback", field, language=language))
        return True

    if awaiting == "which_report":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        candidates = pending.get("candidates") or []
        report_number = match_candidate_report(text, candidates)
        if report_number is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, "দুঃখিত, কোন টেস্টের রিপোর্টের কথা বলছেন, আরেকটু স্পষ্ট করে বলবেন?")
            return True
        # Reconstruct a report_status-shaped result LOCALLY from the
        # candidate the caller just picked, rather than re-querying
        # clinic-api a second time -- the candidate list came from that
        # same lookup moments ago and already carries every field
        # interpret_report_status_result needs (status, delivery_enabled,
        # report_number, test_name). Considered tradeoff, not an oversight:
        # a report's status could in principle change in the few seconds
        # between the ambiguous listing and this answer, same as any
        # read-then-act gap; request_report_delivery() re-checks
        # eligibility server-side regardless (RULE 3/16 defense in depth),
        # so this can never cause an unauthorized delivery, only a stale
        # status read in an already-rare multi-report case.
        chosen = next(c for c in candidates if c["report_number"] == report_number)
        result = {"patient_found": True, "found": True, **chosen}
        await _finish_report_flow(session, pending.get("phone"), result, pending["flow"],
                                   language=language)
        return True

    if awaiting == "confirm_delivery":
        if is_affirmative(text):
            phone = pending.get("phone")
            report_number = pending["report_number"]
            try:
                delivery_result = await _tools.request_report_delivery(phone, report_number)
            except ToolCallError as e:
                logger.error("[%s] clinic API call failed: %s", session.call_id, e)
                session.pending = None
                await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                             fallback_reason="tool_failure")
                return True
            text_out, new_pending = interpret_delivery_request_result(
                delivery_result, report_number, language=language)
            if new_pending is not None:
                new_pending["phone"] = phone
            session.pending = new_pending
            await _speak(session, text_out)
            return True
        if is_negative(text):
            session.pending = None
            await _speak(session, delivery_declined_reply(language=language))
            return True
        # ADDED BY SOURAV -- KCD-383: checked only AFTER is_affirmative()/
        # is_negative() have already failed on this same utterance (same
        # ordering guarantee looks_like_otp_disclosure_request() below
        # follows), so an unambiguous "হ্যাঁ"/"না" is never misrouted by a
        # stray word -- see agent/slot_parse.py's own docstring.
        if looks_like_callback_preference(text):
            await _pivot_report_offer_to_callback(session, pending, language=language)
            return True
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False
        await _speak(session, "রিপোর্টটা কি আপনার ফোনে পাঠাব, নাকি কল ব্যাক করব?")
        return True

    if awaiting == "otp_code":
        if is_negative(text):
            session.pending = None
            await _speak(session, delivery_declined_reply(language=language))
            return True
        otp = parse_otp(text)
        if otp is None:
            # RULE 9 / ATTACK 8: checked only AFTER parse_otp() already
            # failed on this same utterance, so "the otp is 482913" (which
            # DOES contain the word "otp" but is also a valid code) is
            # handled as a normal OTP attempt above, never misclassified
            # as a disclosure request.
            if looks_like_otp_disclosure_request(text):
                await _speak(session, otp_disclosure_refusal_reply(language=language))
                return True
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, "দুঃখিত, ওটিপিটা ঠিকমতো বুঝতে পারিনি, আবার বলবেন?")
            return True
        phone = pending.get("phone")
        report_number = pending["report_number"]
        try:
            result = await _tools.verify_report_otp(phone, report_number, otp)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                         fallback_reason="tool_failure")
            return True
        text_out, new_pending = interpret_otp_verify_result(result, report_number, language=language)
        if new_pending is not None:
            new_pending["phone"] = phone
        session.pending = new_pending
        await _speak(session, text_out)
        return True

    # ADDED BY SOURAV -- "Caller asks something the agent does not cover"
    # story. Checked BEFORE the universal "না" escape hatch just below,
    # same reason confirm_booking/confirm_delivery/etc. all are: that
    # hatch's fixed "appointment bad thak" wording would be wrong here (no
    # appointment was ever in progress), and a plain "না" in this state
    # means "no, I'll contact the counter myself" -- a real, distinct
    # answer with its own reply, not an abandonment.
    if awaiting == "out_of_scope_choice":
        if is_affirmative(text):
            session.pending = None
            # Same honest "logged for a human, no real transfer capability"
            # handling as the "unclear" intent branch above -- see
            # agent/outcomes.record_human_handoff()'s own docstring.
            # intent="out_of_scope" here (not "unclear") so the two stay
            # distinguishable in the shared escalation ledger.
            record_human_handoff("out_of_scope", call_id=session.call_id)
            await _speak(session, human_fallback_reply(language=language))
            return True
        if is_negative(text):
            session.pending = None
            await _speak(session, out_of_scope_counter_reply(language=language))
            return True
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False  # give a fresh LLM classification a chance instead
        await _speak(session, out_of_scope_reply(language=language))
        return True

    if awaiting == "clinical_interpretation_choice":
        # ADDED BY SOURAV -- "Caller asks whether their result is
        # dangerous" story (Epic: Conversation -- Difficult, Sensitive and
        # Edge Cases). Mirrors "out_of_scope_choice" immediately above,
        # including its retry-then-fall-through-to-a-fresh-classification
        # shape -- the AC's "routed to a human by policy" is this affirmative
        # branch reusing human_fallback_reply()/record_human_handoff()
        # verbatim, the exact same escalation path "out_of_scope" and
        # "unclear" already use, just tagged with its own intent string so
        # the three stay distinguishable in the shared escalation ledger.
        if is_affirmative(text):
            session.pending = None
            record_human_handoff("clinical_interpretation", call_id=session.call_id)
            await _speak(session, human_fallback_reply(language=language))
            return True
        if is_negative(text):
            session.pending = None
            await _speak(session, clinical_interpretation_decline_reply(language=language))
            return True
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False  # give a fresh LLM classification a chance instead
        await _speak(session, clinical_interpretation_reply(language=language))
        return True

    # Universal escape hatch, checked before any field-specific parsing:
    # a caller mid-flow who says "না" / "থাক" is abandoning the booking,
    # not answering whichever question was pending.
    # story title: Every critical value is read back before it is used
    # user story: As a patient giving a phone number, I want it read back, so
    #   that a misheard digit does not send my report to a stranger.
    # acceptance criteria: Phone numbers, dates, times and names are confirmed
    #   aloud before any write, and a rejection opens a correction path rather
    #   than repeating the prompt. Readback is mandatory regardless of
    #   confidence for values that affect a write.
    #
    # The hatch now SKIPS confirm_booking, and that exclusion is the story.
    # A "no" answering "did I get this right?" does not mean "cancel my
    # appointment" -- it means one of the five values is wrong. Letting the
    # universal hatch see it first threw the whole booking away at the exact
    # moment the caller was trying to repair it.
    #
    # confirm_correction is deliberately NOT excluded, which is a considered
    # divergence from dev_sourav, where both states skip the hatch. By the time
    # the agent has asked "which one should I fix -- doctor, date, time, name,
    # or phone?", a caller answering "no" is not naming a field; the likeliest
    # reading is that they have given up, and that is what the hatch does.
    if is_negative(text) and awaiting != "confirm_booking":
        session.pending = None
        await _speak(session, "ঠিক আছে, অ্যাপয়েন্টমেন্ট বাদ থাক। আর কিছু জানতে চান?")
        return True

    if awaiting == "confirm_booking":
        # The whole booking has been read back to the caller and this turn is
        # their answer. ONLY an explicit affirmative writes.
        #
        # The negative case never reaches here -- is_negative() above already
        # cancels the flow, which is the correct outcome for "না" / "থাক".
        # Everything that is neither yes nor no falls through to a re-ask:
        # silence, a restatement of the details, a half-heard grunt. None of
        # those are consent, and treating an ambiguous reply as one would give
        # back exactly the guess-becomes-a-booking failure this state exists to
        # prevent.
        if is_affirmative(text):
            slots = pending["slots"]
            session.pending = None
            logger.info("[%s] booking confirmed by caller", session.call_id)
            await _finish_booking(session, slots, confirmed=True, language=language)
            return True

        if is_negative(text):
            # THE CORRECTION PATH. Before this, a rejection re-asked "just say
            # yes or no" twice and then abandoned the booking: the caller said
            # something was wrong and the agent's reply was to ask the same
            # question again, then hang up on it. The criterion names that
            # exact behaviour as the thing not to do.
            #
            # Nothing is discarded -- the four correct values stay in
            # pending["slots"], and only the named one is re-collected.
            pending["awaiting"] = "confirm_correction"
            pending["retries"] = 0
            logger.info("[%s] readback rejected -- opening the correction path",
                        session.call_id)
            await _speak(session, booking_correction_prompt(language=language))
            return True

        # ADDED BY SOURAV -- "The agent accepts a correction and restates"
        # story. A caller does not always say "no" first -- hearing the
        # readback, they may just correct it directly ("actually, my phone
        # number is..."). Checked only once neither yes nor no matched, so
        # a plain "না"/"no" still opens the existing, separately-tested
        # two-step "which one is wrong?" menu untouched; this is a
        # SHORTCUT for when the caller volunteers the field and the new
        # value together, needing no menu at all. `awaiting=None` because
        # every field is already known here -- none of the five is "the
        # current field" the way it is mid-collection.
        correction = detect_booking_correction(
            text, pending["slots"], None, _booking_correction_parsers(pending))
        if correction is not None:
            field, new_value = correction
            logger.info("[%s] correcting %s directly from the readback (no menu needed)",
                        session.call_id, field)
            pending["slots"][field] = new_value
            pending["retries"] = 0
            ack = correction_acknowledged_reply(field, pending["slots"], language=language)
            await _speak(session, ack + " " + booking_confirmation_prompt(pending['slots'], language=language))
            return True

        pending["retries"] += 1
        if pending["retries"] > 2:
            # Three unclear answers to a yes/no question is a handoff, not a
            # fourth attempt. Nothing has been written, and saying so plainly
            # is a B-grade outcome; looping again would trend towards D.
            session.pending = None
            logger.info("[%s] booking abandoned -- no clear confirmation", session.call_id)
            await _speak(session, BOOKING_NOT_CONFIRMED_BN)
            return True
        await _speak(session, "শুধু বলুন — হ্যাঁ, নাকি না?")
        return True

    # story title: Every critical value is read back before it is used
    # user story: As a patient giving a phone number, I want it read back, so
    #   that a misheard digit does not send my report to a stranger.
    # acceptance criteria: Phone numbers, dates, times and names are confirmed
    #   aloud before any write, and a rejection opens a correction path rather
    #   than repeating the prompt. Readback is mandatory regardless of
    #   confidence for values that affect a write.
    #
    # The caller rejected the readback and has been asked which single value
    # is wrong. This turn is that answer.
    #
    # Re-collecting ONE field and returning to the readback is what makes this
    # a correction rather than a restart: the tail of this function fills the
    # named field, finds nothing missing, and routes straight back to
    # confirm_booking -- so the corrected booking is read back IN FULL and
    # still needs an explicit affirmative. A correction never shortens the
    # path to the write.
    if awaiting == "confirm_correction":
        field = parse_correction_field(text)
        if field is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                logger.info("[%s] correction abandoned -- no field named",
                            session.call_id)
                await _speak(session, BOOKING_NOT_CONFIRMED_BN)
                return True
            await _speak(session, booking_correction_prompt(language=language))
            return True

        logger.info("[%s] correcting %s", session.call_id, field)
        # FIXED BY SOURAV -- KCD-448 test cleanup. The disputed value used
        # to be left sitting in pending["slots"] until overwritten by the
        # caller's next answer -- harmless in practice (nothing reads it
        # while `awaiting` is pointed at re-collecting it), but a value
        # already known to be wrong has no business surviving in the
        # confirmed data even briefly. Cleared here instead, matching
        # test_naming_phone_reenters_phone_collection_with_others_kept's
        # own expectation.
        pending["slots"].pop(field, None)

        # ADDED BY SOURAV -- "The agent accepts a correction and restates"
        # story. If the caller names the field AND gives its new value in
        # the same breath ("the date -- actually make it Friday"), apply
        # it immediately with an explicit acknowledgment rather than
        # asking a question they have already answered. Not attempted for
        # "doctor_name": validating a doctor needs an async catalogue
        # call, which this synchronous field-naming turn cannot make --
        # see the dedicated `awaiting == "doctor_name"` branch below,
        # which is where a doctor correction is actually validated.
        parser = _booking_correction_parsers(pending).get(field)
        new_value = parser(text) if parser else None
        if new_value is not None:
            pending["slots"][field] = new_value
            pending["retries"] = 0
            missing = _next_missing(pending["slots"])
            ack = correction_acknowledged_reply(field, pending["slots"], language=language)
            if missing is None:
                pending["awaiting"] = "confirm_booking"
                await _speak(session, ack + " " + booking_confirmation_prompt(pending['slots'], language=language))
            else:
                pending["awaiting"] = missing
                await _speak(session, ack + " " + missing_slot_prompt('book_appointment', missing, language=language))
            return True

        pending["awaiting"] = field
        pending["retries"] = 0
        # Marks this re-collection as a CORRECTION, not an original ask,
        # so the generic date/time_slot/patient_name/phone tail (and the
        # doctor_name branch) below knows to acknowledge once the new
        # value actually arrives, instead of silently repeating the exact
        # same question a first-time collection would ask.
        pending["_correcting_field"] = field
        await _speak(session, missing_slot_prompt("book_appointment", field, language=language))
        return True

    # ADDED BY SOURAV -- "The agent accepts a correction and restates"
    # story. FIXES A REAL DEAD END: booking_correction_prompt() above
    # names "the doctor" as a correctable field, and parse_correction_
    # field() (agent/slot_parse.py) dutifully recognises it -- but until
    # this branch existed, nothing ever matched `awaiting == "doctor_name"`
    # in the generic date/time_slot/patient_name/phone tail further down,
    # so whatever the caller said next fell straight through to "value is
    # None" every single time. Three retries later the WHOLE booking --
    # the four still-correct values included -- was silently discarded for
    # a fresh LLM classification, and the caller's request to fix the
    # doctor was never actually honoured. See agent/correction_flow.py's
    # own module docstring for the full writeup of this gap.
    #
    # Unlike date/time_slot/patient_name/phone, a doctor's name cannot be
    # accepted as free text -- it has to be validated against the real
    # catalogue, the same way every other doctor-name entry point in this
    # codebase already does (get_doctor_availability() doubles as both
    # "is this doctor real" and "are they sitting on this date").
    if awaiting == "doctor_name":
        # No explicit is_negative() check here -- unlike the other named
        # states above this point in the function, this one is reached
        # AFTER the universal "না" escape hatch (it sits between
        # "confirm_booking" and here), which already catches a plain
        # rejection for any awaiting value other than "confirm_booking".
        candidate_name = text.strip()
        if not candidate_name:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("book_appointment", "doctor_name", language=language))
            return True
        date_iso = pending.get("offered_date") or pending["slots"].get("date") or datetime.date.today().isoformat()
        try:
            result = await _tools.get_doctor_availability(candidate_name, date_iso)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, SYSTEM_UNREACHABLE_BN, fallback_reason="tool_failure")
            return True
        if not result.get("found"):
            pending["retries"] += 1
            if pending["retries"] > 2:
                # Same graceful-degradation shape as every other repair
                # ladder in this function: nothing was written, and a
                # fourth attempt at the same unresolved name is a worse
                # outcome than letting a fresh classification try.
                session.pending = None
                logger.info("[%s] doctor correction abandoned -- name never resolved",
                            session.call_id)
                return False
            await _speak(session, "দুঃখিত, এই নামে কোনো ডাক্তার খুঁজে পাচ্ছি না। আবার নাম বলবেন?")
            return True

        was_correction = pending.pop("_correcting_field", None) == "doctor_name"
        pending["slots"]["doctor_name"] = result.get("doctor_name") or candidate_name
        pending["slots"]["doctor_name_bn"] = result.get("doctor_name_bn")
        pending["retries"] = 0
        prefix = ""
        if was_correction:
            prefix = correction_acknowledged_reply("doctor_name", pending["slots"], language=language) + " "
        missing = _next_missing(pending["slots"])
        if missing is None:
            pending["awaiting"] = "confirm_booking"
            await _speak(session, prefix + booking_confirmation_prompt(pending["slots"], language=language))
            return True
        pending["awaiting"] = missing
        await _speak(session, prefix + missing_slot_prompt("book_appointment", missing, language=language))
        return True

    # story title: Near matches are offered rather than guessed or refused
    # user story: As a caller naming something loosely, I want the close
    #   matches offered, so that I am not told my test does not exist when it
    #   does.
    # acceptance criteria: When several catalogue rows fall within the match
    #   band the agent offers up to three by name and asks which. Candidates
    #   are generated across every supported language and romanised spelling.
    #   The did-you-mean path covers the ambiguous case and not only total
    #   failure.
    #
    # The turn after an offer. The caller has been read up to three names and
    # has said one of them; this resolves which, then re-runs the SAME lookup
    # against the canonical name rather than against their words -- so the
    # second attempt cannot be ambiguous for the same reason the first was.
    #
    # Separate from doctor_choice, which looks nearly identical and is not:
    # that state follows a department LISTING, where every candidate is a
    # correct answer and the caller is choosing who to see. Here the
    # candidates are competing readings of one thing the caller already said,
    # and only one of them is what they meant.
    if awaiting == "entity_choice":
        intent = pending.get("intent")
        chosen = _match_offered(text, pending.get("candidates") or [])
        if chosen is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                # Two failed attempts at the same choice. Drop the state and
                # let a fresh classification try, rather than asking a third
                # time -- a repair ladder with no end is its own bad outcome,
                # the same cap doctor_choice uses.
                session.pending = None
                return False
            await _speak(session, NEAR_MATCH_UNCLEAR_BN)
            return True

        date_iso = pending.get("offered_date")
        session.pending = None
        logger.info("[%s] %s disambiguated to %r", session.call_id, intent,
                    chosen.get("name"))

        try:
            if intent == "test_rate":
                result = await _tools.get_test_rate(chosen["name"])
            elif intent == "doctor_availability":
                result = await _tools.get_doctor_availability(
                    chosen["name"], date_iso or datetime.date.today().isoformat())
            elif intent == "doctor_schedule":
                # ADDED BY SOURAV -- KCD-385. Without this explicit case,
                # a caller who answers "did you mean Dr X or Dr Y?" for a
                # doctor_schedule question fell into the trailing `else`
                # below and got asked about DEPARTMENTS instead of their
                # chosen doctor's schedule -- the `else` was written back
                # when this state only ever served test_rate/doctor_
                # availability/doctors_by_department, before doctor_
                # schedule's own near-match wiring existed.
                result = await _tools.get_doctor_schedule(chosen["name"])
            else:
                result = await _tools.get_doctors_by_department(chosen["name"], date_iso)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            await _speak(session, SYSTEM_UNREACHABLE_BN,
                         fallback_reason="tool_failure")
            return True

        # Three near-identical calls rather than one over a `reply` local,
        # and deliberately so: tests/test_fact_provenance.py requires the
        # sentence reaching _speak_fact to be a reply_templates CALL at the
        # call site, not a name that a local could have been reassigned to.
        # Collapsing these three would pass a variable and blind that gate --
        # which is the regression it exists to catch, so the duplication is
        # the cheaper half of the trade.
        #
        # A canonical name resolving to another ambiguity would mean two rows
        # share a name outright, a catalogue defect rather than a caller one.
        # _speak_fact would offer again and return True; session.pending was
        # cleared above, so it asks once and stops rather than looping.
        spoken_name = chosen.get("name_bn") or chosen["name"]
        if intent == "test_rate":
            asked = {"test_name": spoken_name}
            if await _speak_fact(session, intent, asked, result,
                                 test_rate_reply(asked, result),
                                 offered_date=date_iso):
                return True
        elif intent == "doctor_availability":
            asked = {"doctor_name": spoken_name}
            if await _speak_fact(session, intent, asked, result,
                                 doctor_availability_reply(asked, result),
                                 offered_date=date_iso):
                return True
        elif intent == "doctor_schedule":
            # ADDED BY SOURAV -- KCD-385. No `offered_date` -- this
            # intent is deliberately date-free (see doctor_schedule_
            # reply()'s own docstring), so there is nothing to carry
            # forward the way test_rate/doctor_availability carry a date
            # into their own choice states.
            asked = {"doctor_name": spoken_name}
            if await _speak_fact(session, intent, asked, result,
                                 doctor_schedule_reply(asked, result)):
                return True
        else:
            asked = {"department": spoken_name}
            if await _speak_fact(session, intent, asked, result,
                                 doctors_by_department_reply(asked, result),
                                 offered_date=date_iso):
                return True
        return True

    if awaiting == "doctor_choice":
        match = _match_offered(text, pending.get("candidates") or [])
        if match is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False  # give a fresh LLM classification a chance instead
            await _speak(session, "দুঃখিত, ডাক্তারের নামটা একটু স্পষ্ট করে বলবেন?")
            return True

        date_iso = pending.get("offered_date") or datetime.date.today().isoformat()
        try:
            result = await _tools.get_doctor_availability(match["name"], date_iso)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, SYSTEM_UNREACHABLE_BN,
                         fallback_reason="tool_failure")
            return True

        # story title: The same question gets the same answer within one call
        # user story: As a caller who asks twice, I want the same answer, so that
        #   I know which one to believe.
        # acceptance criteria: Repeating a question in one call produces an
        #   identical factual answer unless the underlying data changed, in which
        #   case the change is stated. A test asserts consistency across three
        #   repeats with an unchanged backend.
        #
        # Every factual reply goes through _speak_fact rather than _speak, so the
        # consistency check cannot be forgotten on one route. The reply itself is
        # still rendered here, fresh, from this turn's live clinic response.
        asked = {"doctor_name": match.get("name_bn") or match["name"]}
        if await _speak_fact(session, "doctor_availability", asked, result,
                             doctor_availability_reply(asked, result, language=language),
                             offered_date=date_iso):
            return True

        offered = None
        if result.get("found"):
            offered = result.get("date") if result.get("available") else result.get("next_available_date")
        if offered:
            # doctor_availability_reply() just asked "today or another
            # day" (or, if not available today, "want that next date
            # instead?") -- stay in the flow so the caller's answer to
            # THAT question is picked up as the "date" field next.
            session.pending = {
                "awaiting": "date",
                # doctor_name is the canonical English label the booking API
                # needs; doctor_name_bn is the one that gets SPOKEN in the
                # readback. Both are carried from here on -- see
                # booking_confirm_prompt for what happens when they are not.
                "slots": {
                    "doctor_name": result.get("doctor_name") or match["name"],
                    "doctor_name_bn": result.get("doctor_name_bn") or match.get("name_bn"),
                },
                "candidates": None, "offered_date": offered, "retries": 0,
            }
        else:
            session.pending = None
        return True

    if awaiting == "confirm_date":
        # story title: The model never originates a fact
        # user story: As a clinical lead, I want every price, date and identifier
        #   to come from a verified system response, so that a wrong answer is a
        #   data bug rather than a model bug.
        # acceptance criteria: Every factual sentence is a template substitution
        #   from a validated tool response and the model is never shown a figure
        #   it could restate. An automated assertion on every commit proves no
        #   model-composed span reaches synthesis on a factual intent.
        #
        # The caller said something covering several days ("আগামী সপ্তাহে"),
        # date_calc worked out which days those are, and the agent read the
        # range back. This turn is the answer.
        #
        # হ্যাঁ  -> look up the FIRST day of the range. clinic-api answers
        #          "in that day?" and, when not, "next available day" computed
        #          from its own schedule table -- which is the honest answer to
        #          "is Dr Sen in next week?" and needs no new API surface.
        # anything else -> the range was wrong, so ask for one specific day.
        #          Deliberately NOT a re-ask of the same question: the caller
        #          already said no to it once.
        start = pending.get("offered_date")
        resume = pending.get("resume_intent")
        if is_affirmative(text) and start and resume:
            if resume == "doctor_availability":
                doctor_name = pending["slots"]["doctor_name"]
                try:
                    result = await _tools.get_doctor_availability(doctor_name, start)
                except ToolCallError as e:
                    logger.error("[%s] clinic API call failed: %s", session.call_id, e)
                    session.pending = None
                    await _speak(session, SYSTEM_UNREACHABLE_BN,
                                 fallback_reason="tool_failure")
                    return True
                asked = {"doctor_name": doctor_name}
                if await _speak_fact(session, "doctor_availability", asked, result,
                                     doctor_availability_reply(asked, result),
                                     offered_date=start):
                    return True
                offered = None
                if result.get("found"):
                    offered = result.get("date") if result.get("available") else result.get("next_available_date")
                session.pending = {
                    "awaiting": "date",
                    "slots": {
                        "doctor_name": result.get("doctor_name") or doctor_name,
                        "doctor_name_bn": result.get("doctor_name_bn"),
                    },
                    "candidates": None, "offered_date": offered, "retries": 0,
                } if offered else None
                return True

            department = pending["slots"]["department"]
            try:
                result = await _tools.get_doctors_by_department(department, start)
            except ToolCallError as e:
                logger.error("[%s] clinic API call failed: %s", session.call_id, e)
                session.pending = None
                await _speak(session, SYSTEM_UNREACHABLE_BN,
                             fallback_reason="tool_failure")
                return True
            asked = {"department": department}
            if await _speak_fact(session, "doctors_by_department", asked, result,
                                 doctors_by_department_reply(asked, result),
                                 offered_date=start):
                return True
            if result.get("found") and result.get("doctors"):
                session.pending = {
                    "awaiting": "doctor_choice", "slots": {},
                    "candidates": [
                        {"name": d["name"], "name_bn": d.get("doctor_name_bn")}
                        for d in result["doctors"]
                    ],
                    "offered_date": start, "retries": 0,
                }
            else:
                session.pending = None
            return True

        ask_state = ("availability_date" if resume == "doctor_availability"
                     else "department_date")
        pending["awaiting"] = ask_state
        pending["offered_date"] = None
        pending["retries"] = 0
        await _speak(session, missing_slot_prompt(resume or "book_appointment", "date"))
        return True

    if awaiting == "availability_date":
        # story title: The model never originates a fact
        # user story: As a clinical lead, I want every price, date and identifier
        #   to come from a verified system response, so that a wrong answer is a
        #   data bug rather than a model bug.
        # acceptance criteria: Every factual sentence is a template substitution
        #   from a validated tool response and the model is never shown a figure
        #   it could restate. An automated assertion on every commit proves no
        #   model-composed span reaches synthesis on a factual intent.
        #
        # The caller said something date-shaped the local parser could not
        # resolve, so rather than state the model's guess as fact the agent
        # asked which day they meant. This is that answer. Mirrors
        # "department_date" below exactly, one intent over: parse it locally,
        # re-run the same lookup, and never fall back to the model's original
        # guess -- an unparseable answer re-asks and then gives up to a fresh
        # classification, which is the same trust model every other state here
        # uses.
        value = parse_date(text, offered_date=pending.get("offered_date"))
        if value is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("doctor_availability", "date"))
            return True

        doctor_name = pending["slots"]["doctor_name"]
        try:
            result = await _tools.get_doctor_availability(doctor_name, value)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, SYSTEM_UNREACHABLE_BN,
                         fallback_reason="tool_failure")
            return True

        asked = {"doctor_name": doctor_name}
        if await _speak_fact(session, "doctor_availability", asked, result,
                             doctor_availability_reply(asked, result),
                             offered_date=value):
            return True

        offered = None
        if result.get("found"):
            offered = result.get("date") if result.get("available") else result.get("next_available_date")
        session.pending = {
            "awaiting": "date",
            "slots": {
                "doctor_name": result.get("doctor_name") or doctor_name,
                "doctor_name_bn": result.get("doctor_name_bn"),
            },
            "candidates": None, "offered_date": offered, "retries": 0,
        } if offered else None
        return True

    if awaiting == "department_date":
        # Mirrors "doctor_choice" above, one level up: the caller was just
        # told nobody in this department sits TODAY and asked for another
        # day. Parse that reply as a date and re-run the same department
        # lookup with it, rather than dropping back to a cold LLM
        # classification of a bare date phrase (see this function's
        # docstring for why that silently loses context).
        value = parse_date(text, offered_date=pending.get("offered_date"))
        if value is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("doctors_by_department", "date", language=language))
            return True

        department = pending["slots"]["department"]
        try:
            result = await _tools.get_doctors_by_department(department, value)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, SYSTEM_UNREACHABLE_BN,
                         fallback_reason="tool_failure")
            return True

        asked = {"department": department}
        if await _speak_fact(session, "doctors_by_department", asked, result,
                             doctors_by_department_reply(asked, result, language=language),
                             offered_date=value):
            return True

        if result.get("found") and result.get("doctors"):
            session.pending = {
                "awaiting": "doctor_choice",
                "slots": {},
                "candidates": [
                    {"name": d["name"], "name_bn": d.get("doctor_name_bn")}
                    for d in result["doctors"]
                ],
                "offered_date": value, "retries": 0,
            }
        elif result.get("found"):
            # Still nobody that day either -- stay in the same state and
            # let the caller name yet another day, capped by the shared
            # retries counter above so this cannot loop forever.
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
            else:
                pending["awaiting"] = "department_date"
        else:
            session.pending = None
        return True

    # Remaining states (date / time_slot / patient_name / phone) all share
    # the same shape: parse the ONE field awaited, fill it in, ask for the
    # next missing one or finish the booking.
    #
    # ADDED BY SOURAV -- "The agent accepts a correction and restates"
    # story. Before treating this turn as an answer to whatever field is
    # currently awaited, check whether it is instead a correction to a
    # DIFFERENT field collected earlier in this same booking -- e.g. the
    # caller is asked for the time slot but says "actually, make the date
    # Friday instead." Gated behind an explicit correction cue (see
    # agent/correction_flow.py's own module docstring for why: these are
    # the same deterministic parsers used for original collection, and
    # running them unconditionally against every turn would risk silently
    # reinterpreting an ordinary answer as a correction to something
    # else). The field still being asked for is left untouched -- the
    # caller is asked for it again in the SAME breath as the
    # acknowledgment, so nothing about the current question is lost.
    correction = detect_booking_correction(
        text, pending["slots"], awaiting, _booking_correction_parsers(pending))
    if correction is not None:
        field, new_value = correction
        logger.info("[%s] spontaneous correction: %s -> %r (still awaiting %s)",
                    session.call_id, field, new_value, awaiting)
        pending["slots"][field] = new_value
        ack = correction_acknowledged_reply(field, pending["slots"], language=language)
        await _speak(session, ack + " " + missing_slot_prompt('book_appointment', awaiting, language=language))
        return True

    value = None
    if awaiting == "date":
        value = parse_date(text, offered_date=pending.get("offered_date"))
    elif awaiting == "time_slot":
        value = parse_time(text)
    elif awaiting == "phone":
        value = parse_phone(text)
    elif awaiting == "patient_name":
        value = _clean_patient_name(text)

    if value is None:
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False
        await _speak(session, missing_slot_prompt("book_appointment", awaiting, language=language))
        return True

    # ADDED BY SOURAV -- "The agent accepts a correction and restates"
    # story. True only when this exact field's re-collection was opened
    # BY a correction (see the "confirm_correction" state above, which
    # sets this marker only on its no-immediate-value fallback path) --
    # an ordinary, first-time collection of this field never sets it, so
    # this acknowledgment is never spoken for a plain original answer.
    was_correction = pending.pop("_correcting_field", None) == awaiting
    pending["slots"][awaiting] = value
    pending["retries"] = 0
    prefix = ""
    if was_correction:
        prefix = correction_acknowledged_reply(awaiting, pending["slots"], language=language) + " "
    missing = _next_missing(pending["slots"])
    if missing is None:
        # Every field is filled, but nothing is written yet. Read the whole
        # thing back and wait for a yes -- see the "confirm_booking" state
        # above for why an affirmative is required rather than assumed.
        pending["awaiting"] = "confirm_booking"
        await _speak(session, prefix + booking_confirmation_prompt(pending["slots"], language=language))
        return True
    pending["awaiting"] = missing
    await _speak(session, prefix + missing_slot_prompt("book_appointment", missing, language=language))
    return True


# story title: A thing not existing is never confused with a system being down
# user story: As a caller, I want to know whether my test does not exist or the
#   system cannot be reached, so that I know whether to call back.
# acceptance criteria: The two produce different spoken sentences and different
#   metrics, and the distinction survives every refactor. This behaviour exists
#   today and gains a permanent regression case.
#
# THE THIRD OUTCOME, WHICH SHOULD NOT EXIST.
# _dispatch_turn used to be the whole turn with no outer handler, fired by
# asyncio.create_task() with nothing attached to it. Anything uncaught -- a
# KeyError in a template, a torchaudio failure slicing the clip, a bug in code
# not yet written -- became an unretrieved task exception and the caller heard
# NOTHING. Neither sentence: dead air, which this module's own docstring
# promises never happens.
#
# An unexpected exception is the SYSTEM failing, so it maps to the
# system-unreachable side of the distinction this story is about. It is never
# "your test does not exist": we do not know that, and telling a caller to stop
# asking because our code raised would be the exact confusion the story names.
# story title: A multi-part question is answered in full
# user story: As a caller who asked two things, I want both answered, so that
#   I do not have to ask again.
# acceptance criteria: Every answerable part of a turn is answered in the
#   order asked, and any part that cannot be answered is explicitly addressed
#   rather than dropped. Completeness is scored on a labelled multi-part set.
#
# How long a deferred part may wait. Two turns is one clarification plus its
# answer; past that the caller has moved on, and reviving a question they
# asked four turns ago reads as the agent losing the thread rather than
# keeping it. The queue is never silently binned -- see _drain_deferred.
MAX_DEFERRED_TURNS = 2


# story title: A multi-part question is answered in full
# user story: As a caller who asked two things, I want both answered, so that
#   I do not have to ask again.
# acceptance criteria: Every answerable part of a turn is answered in the
#   order asked, and any part that cannot be answered is explicitly addressed
#   rather than dropped. Completeness is scored on a labelled multi-part set.
async def _run_parts(session: CallSession, text: str, parts: list[dict]) -> bool:
    """Answer the parts in the order the caller asked them.

    -> True if a queue was created on THIS turn, which tells the caller not
    to immediately try to drain it.

    TWO RULES CARRY THIS FUNCTION.

    STOP AT THE FIRST INTERACTIVE PART. Not "skip it and do the rest": the
    criterion says in the order asked, and answering the second question
    while the first is still waiting on a clarification reorders the
    conversation from the caller's side. Whatever follows is deferred, which
    is a promise -- _drain_deferred keeps it.

    DISCARD A SOFT CONTINUATION WHEN ANOTHER PART FOLLOWS. Several answers
    leave session.pending set opportunistically: doctor_availability ends by
    asking "today or another day?" and stays in the flow so a bare date is
    understood next turn. That convenience belongs to the LAST thing said. If
    part two is about to speak, part one's open flow would catch the caller's
    reply to a question they have already stopped thinking about -- so it is
    dropped, deliberately, rather than left to misread the next utterance.
    """
    for index, part in enumerate(parts):
        remaining = parts[index + 1:]
        outcome = await _answer_part(session, text, part)

        if outcome == INTERACTIVE:
            if remaining:
                session.deferred = {"parts": remaining, "text": text, "age": 0}
                logger.info("[%s] deferring %d part(s) behind a question",
                            session.call_id, len(remaining))
                await _speak(session, DEFERRED_PART_BN)
                return True
            return False

        # ANSWERED or UNANSWERABLE: carry on. UNANSWERABLE has already said
        # something about itself inside _answer_part -- that is the whole
        # point of it being a third outcome rather than a silent skip.
        if remaining and session.pending is not None:
            session.pending = None

    return False


# story title: A multi-part question is answered in full
# user story: As a caller who asked two things, I want both answered, so that
#   I do not have to ask again.
# acceptance criteria: Every answerable part of a turn is answered in the
#   order asked, and any part that cannot be answered is explicitly addressed
#   rather than dropped. Completeness is scored on a labelled multi-part set.
async def _drain_deferred(session: CallSession, text: str) -> None:
    """Answer what was put aside, once the question in front of it is done.

    A queue that is only ever created is not a deferral, it is a drop with
    better manners. This is the half that makes DEFERRED_PART_BN a true
    sentence.
    """
    queue = session.deferred
    if not queue:
        return

    if session.pending is not None:
        # Still mid-flow. Wait, but not forever.
        queue["age"] += 1
        if queue["age"] <= MAX_DEFERRED_TURNS:
            return
        session.deferred = None
        logger.info("[%s] dropping %d deferred part(s) after %d turns",
                    session.call_id, len(queue["parts"]), queue["age"])
        # Said, not silently binned. The caller asked; they are owed the
        # information that it went unanswered even when the answer is that
        # too much has happened since.
        for part in queue["parts"]:
            await _speak(session, unanswered_part_prompt(turn_parts.subject_of(part)))
        return

    session.deferred = None
    logger.info("[%s] resuming %d deferred part(s)", session.call_id, len(queue["parts"]))
    await _speak(session, RESUMING_PART_BN)
    # The ORIGINAL utterance, not this turn's. date_calc and slot_parse read
    # the raw words, and "আগামীকাল" said once for two questions means it for
    # both -- re-resolving the second part against the caller's answer to the
    # first ("হ্যাঁ") would lose the day entirely.
    await _run_parts(session, queue["text"], queue["parts"])


# story title: A multi-part question is answered in full
# user story: As a caller who asked two things, I want both answered, so that
#   I do not have to ask again.
# acceptance criteria: Every answerable part of a turn is answered in the
#   order asked, and any part that cannot be answered is explicitly addressed
#   rather than dropped. Completeness is scored on a labelled multi-part set.
#
# LIFTED VERBATIM out of _dispatch_turn_inner, which is why it reads like a
# chain rather than like a function: every branch, comment and ordering
# decision below predates this story and none of them changed. What changed
# is that each of the thirteen `return` statements -- each of which used to
# mean "the turn is over" -- now names WHICH KIND of over it was, so a caller
# with a second question can be told apart from a caller who is owed an
# answer to the first.
#
# `text` is still passed in whole, not just the part: date_calc.resolve and
# slot_parse's parsers read the raw utterance, and a caller who says "আগামীকাল"
# once for two questions means it for both.
async def _answer_part(session: CallSession, text: str, part: dict) -> str:
    """Answer one part of a turn. -> ANSWERED | INTERACTIVE | UNANSWERABLE."""
    intent = part["intent"]
    slots = part["slots"]

    if intent == "smalltalk":
        await _speak(session, part.get("direct_reply_bn") or "নমস্কার, কী সাহায্য করতে পারি?")
        return ANSWERED

    if intent == "unclear":
        await _speak(session, "দুঃখিত, বুঝতে পারিনি। আবার একটু বলবেন?")
        return UNANSWERABLE

    try:
        if intent == "test_rate":
            if not slots.get("test_name"):
                await _speak(session, missing_slot_prompt(intent, "test_name"))
                return INTERACTIVE
            result = await _tools.get_test_rate(slots["test_name"])
            # story title: The same question gets the same answer within one call
            # user story: As a caller who asks twice, I want the same answer, so that
            #   I know which one to believe.
            # acceptance criteria: Repeating a question in one call produces an
            #   identical factual answer unless the underlying data changed, in which
            #   case the change is stated. A test asserts consistency across three
            #   repeats with an unchanged backend.
            #
            # Every factual reply goes through _speak_fact rather than _speak, so the
            # consistency check cannot be forgotten on one route. The reply itself is
            # still rendered here, fresh, from this turn's live clinic response.
            if await _speak_fact(session, intent, slots, result,
                                 test_rate_reply(slots, result)):
                return INTERACTIVE

        elif intent == "doctor_availability":
            if not slots.get("doctor_name"):
                await _speak(session, missing_slot_prompt(intent, "doctor_name"))
                return INTERACTIVE
            # Default to TODAY, not "whenever next available": a bare
            # "ডাক্তার সেন আছেন?" with no date mentioned is a caller
            # asking about right now, and the reply text below already
            # said " আজ" (today) for exactly this case -- the old code
            # passed date=None through to the API, which answers a
            # different question ("when next"), so a doctor who simply
            # wasn't in today got reported by their NEXT sitting date
            # instead of "not today, but they're on Tuesdays" etc.
            # story title: The model never originates a fact
            # user story: As a clinical lead, I want every price, date and
            #   identifier to come from a verified system response, so that
            #   a wrong answer is a data bug rather than a model bug.
            # acceptance criteria: Every factual sentence is a template
            #   substitution from a validated tool response and the model is
            #   never shown a figure it could restate. An automated
            #   assertion on every commit proves no model-composed span
            #   reaches synthesis on a factual intent.
            #
            # The model said what the caller MEANT; date_calc did the
            # calendar. Four outcomes, and they are genuinely different:
            #
            #   range      -> say which days it computed and ask. Safe to
            #                 read the dates out loud precisely because
            #                 code produced them.
            #   unmapped   -> the caller named a day nothing could express.
            #                 Ask which one. Never today: that was the old
            #                 silent wrong answer.
            #   single day -> answer it.
            #   absent     -> no day was mentioned; today is the question
            #                 the caller actually asked.
            span = date_calc.resolve(text, slots.get("date_expr"))
            if span.needs_confirmation:
                logger.info("[%s] %s -> %s..%s, confirming the range",
                            session.call_id, span.expression, span.start, span.end)
                session.pending = {
                    "awaiting": "confirm_date",
                    "slots": {"doctor_name": slots["doctor_name"]},
                    "candidates": None, "offered_date": span.start, "retries": 0,
                    "resume_intent": "doctor_availability", "span_end": span.end,
                }
                await _speak(session, date_range_confirm_prompt(span.start, span.end))
                return INTERACTIVE
            if span.source == date_calc.SOURCE_UNMAPPED:
                logger.info("[%s] caller named a day the vocabulary cannot express "
                            "-- asking instead of assuming", session.call_id)
                session.pending = {
                    "awaiting": "availability_date",
                    "slots": {"doctor_name": slots["doctor_name"]},
                    "candidates": None, "offered_date": None, "retries": 0,
                }
                await _speak(session, missing_slot_prompt(intent, "date"))
                return INTERACTIVE
            if span.source == SOURCE_INTERPRETED:
                logger.info("[%s] date interpreted: %s -> %s",
                            session.call_id, span.expression, span.start)
            date_iso = span.start or datetime.date.today().isoformat()
            result = await _tools.get_doctor_availability(slots["doctor_name"], date_iso)
            if await _speak_fact(session, intent, slots, result,
                                 doctor_availability_reply(slots, result),
                                 offered_date=date_iso):
                return INTERACTIVE

            # Keep the flow open for "yes, book that day" / "another
            # day" -- doctor_availability_reply() just asked exactly
            # that question. See _continue_pending's "date" state.
            offered = None
            if result.get("found"):
                offered = result.get("date") if result.get("available") else result.get("next_available_date")
            session.pending = {
                "awaiting": "date",
                "slots": {
                    "doctor_name": result.get("doctor_name") or slots["doctor_name"],
                    "doctor_name_bn": result.get("doctor_name_bn"),
                },
                "candidates": None, "offered_date": offered, "retries": 0,
            } if offered else None

        elif intent == "doctors_by_department":
            if not slots.get("department"):
                await _speak(session, missing_slot_prompt(intent, "department"))
                return INTERACTIVE
            # Default to TODAY when the caller didn't name a date, same
            # reasoning as doctor_availability above: "অর্থোতে কারা
            # আছেন" (who's in ortho) is almost always asking who is
            # actually in the chamber right now, not for a roster of
            # every doctor the department has ever employed regardless
            # of whether they sit this week. Only an EXPLICIT date
            # bypasses this (used as-is below).
            # story title: The model never originates a fact
            # user story: As a clinical lead, I want every price, date and
            #   identifier to come from a verified system response, so that
            #   a wrong answer is a data bug rather than a model bug.
            # acceptance criteria: Every factual sentence is a template
            #   substitution from a validated tool response and the model is
            #   never shown a figure it could restate. An automated
            #   assertion on every commit proves no model-composed span
            #   reaches synthesis on a factual intent.
            #
            # The model said what the caller MEANT; date_calc did the
            # calendar. Four outcomes, and they are genuinely different:
            #
            #   range      -> say which days it computed and ask. Safe to
            #                 read the dates out loud precisely because
            #                 code produced them.
            #   unmapped   -> the caller named a day nothing could express.
            #                 Ask which one. Never today: that was the old
            #                 silent wrong answer.
            #   single day -> answer it.
            #   absent     -> no day was mentioned; today is the question
            #                 the caller actually asked.
            span = date_calc.resolve(text, slots.get("date_expr"))
            if span.needs_confirmation:
                logger.info("[%s] %s -> %s..%s, confirming the range",
                            session.call_id, span.expression, span.start, span.end)
                session.pending = {
                    "awaiting": "confirm_date",
                    "slots": {"department": slots["department"]},
                    "candidates": None, "offered_date": span.start, "retries": 0,
                    "resume_intent": "doctors_by_department", "span_end": span.end,
                }
                await _speak(session, date_range_confirm_prompt(span.start, span.end))
                return INTERACTIVE
            if span.source == date_calc.SOURCE_UNMAPPED:
                logger.info("[%s] caller named a day the vocabulary cannot express "
                            "-- asking instead of assuming", session.call_id)
                session.pending = {
                    "awaiting": "department_date",
                    "slots": {"department": slots["department"]},
                    "candidates": None, "offered_date": None, "retries": 0,
                }
                await _speak(session, missing_slot_prompt(intent, "date"))
                return INTERACTIVE
            if span.source == SOURCE_INTERPRETED:
                logger.info("[%s] date interpreted: %s -> %s",
                            session.call_id, span.expression, span.start)
            date_iso = span.start or datetime.date.today().isoformat()
            result = await _tools.get_doctors_by_department(slots["department"], date_iso)
            if await _speak_fact(session, intent, slots, result,
                                 doctors_by_department_reply(slots, result),
                                 offered_date=date_iso):
                return INTERACTIVE

            # Continue straight into booking: offer the doctors just
            # listed as candidates, so the caller's very next utterance
            # -- which may be nothing but a bare doctor name -- is
            # matched against THIS list rather than sent to the LLM with
            # no context to interpret it against. See _continue_pending's
            # "doctor_choice" state.
            if result.get("found") and result.get("doctors"):
                session.pending = {
                    "awaiting": "doctor_choice",
                    "slots": {},
                    "candidates": [
                        {"name": d["name"], "name_bn": d.get("doctor_name_bn")}
                        for d in result["doctors"]
                    ],
                    "offered_date": date_iso,
                    "retries": 0,
                }
            elif result.get("found"):
                # Department exists but nobody sits that day --
                # doctors_by_department_reply() just told the caller
                # exactly that and invited another day ("অন্য কোনো
                # দিনের কথা জিজ্ঞেস করতে পারেন"). Stay in the flow so the
                # caller's next utterance is interpreted as THAT date
                # instead of needing to restate the whole department
                # question from scratch -- see _continue_pending's
                # "department_date" state.
                session.pending = {
                    "awaiting": "department_date",
                    "slots": {"department": slots["department"]},
                    "candidates": None, "offered_date": None, "retries": 0,
                }
            else:
                session.pending = None

        elif intent == "book_appointment":
            # Merge onto whatever session.pending already knows (e.g. a
            # doctor_name carried over from a doctor_availability or
            # doctors_by_department turn moments ago) rather than
            # requiring every field in one utterance -- that all-or-
            # nothing check was the other half of "pipeline breaking":
            # a caller who gave the doctor and date in one sentence and
            # the time in the next used to have the doctor/date silently
            # discarded the moment ANY field was still missing.
            merged = dict(session.pending["slots"]) if session.pending else {}
            for field in _BOOKING_FIELDS:
                if slots.get(field):
                    merged[field] = slots[field]

            # story title: The model never originates a fact
            # user story: As a clinical lead, I want every price, date and identifier
            #   to come from a verified system response, so that a wrong answer is a
            #   data bug rather than a model bug.
            # acceptance criteria: Every factual sentence is a template substitution
            #   from a validated tool response and the model is never shown a figure
            #   it could restate. An automated assertion on every commit proves no
            #   model-composed span reaches synthesis on a factual intent.
            #
            # The booking path is the one place that ALREADY had a
            # verifier: every field below is read back in full and an
            # explicit হ্যাঁ is required before _finish_booking writes
            # anything, so a mis-resolved date here is caught by the one
            # party who knows what "কাল" meant. That readback is not
            # touched by this story and must not be weakened by it.
            #
            # What this adds is removing the model from the loop wherever
            # a deterministic parser can do the same job on the same
            # words -- a date, a time and a phone number are all things
            # slot_parse.py resolves in code. Only fields the model
            # claimed from THIS utterance are corrected; a value carried
            # over from an earlier turn was already parsed locally by
            # _continue_pending and must not be re-derived from a
            # transcript that no longer mentions it.
            for field, parser in (("time_slot", parse_time), ("phone", parse_phone)):
                if not slots.get(field):
                    continue
                parsed = parser(text)
                if parsed and parsed != merged.get(field):
                    logger.warning("[%s] %s disagreement: model=%s parsed=%s -- using parsed",
                                   session.call_id, field, merged.get(field), parsed)
                    merged[field] = parsed

            # The date is not merged from `slots` at all any more, because
            # `slots["date"]` now holds the caller's WORDS ("১৫ তারিখ"),
            # not a calendar date -- llm.py stopped producing those. Only a
            # value date_calc computed may be stored, or the API would be
            # handed a Bengali phrase and _next_missing() would report the
            # field as filled while holding something unusable.
            #
            # A RANGE is dropped rather than confirmed here: a booking is
            # one slot on one day, so "আগামী সপ্তাহে অ্যাপয়েন্টমেন্ট চাই"
            # has to become a specific day, and leaving the field empty
            # makes _next_missing() ask for exactly that. The range
            # confirmation belongs to the two read-only intents, which can
            # actually answer about a span.
            if slots.get("date") or slots.get("date_expr"):
                span = date_calc.resolve(text, slots.get("date_expr"))
                if span.start and not span.is_range:
                    merged["date"] = span.start
                else:
                    merged.pop("date", None)

            missing = _next_missing(merged)
            if missing is None:
                # Everything arrived in one utterance. That is the case
                # MOST in need of a readback, not least: five fields pulled
                # from a single sentence of phone audio is where a
                # mishearing is likeliest and least visible. Route it
                # through the same confirmation state as the slow path.
                #
                # FIXED BY SOURAV -- KCD-448. This called the older,
                # Bengali-only booking_confirm_prompt(merged), the same
                # staleness fixed in _finish_booking's own defensive branch
                # (see that function's comment) -- _answer_part() has no
                # `language` parameter of its own (this whole multi-part-
                # turn path predates the language-detection work; see the
                # module-level detect_language import comment), but `text`
                # -- this part's own utterance -- is right here, so it is
                # detected locally rather than left Bengali-only.
                session.pending = {
                    "awaiting": "confirm_booking", "slots": merged,
                    "candidates": None,
                    "offered_date": merged.get("date"), "retries": 0,
                }
                await _speak(session, booking_confirmation_prompt(
                    merged, language=detect_language(text)))
                return INTERACTIVE

            session.pending = {
                "awaiting": missing, "slots": merged, "candidates": None,
                "offered_date": (session.pending or {}).get("offered_date"), "retries": 0,
            }
            await _speak(session, missing_slot_prompt(intent, missing))
            return INTERACTIVE

    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, SYSTEM_UNREACHABLE_BN,
                     fallback_reason="tool_failure")
        return UNANSWERABLE

    return ANSWERED


async def _dispatch_turn(session: CallSession, utterance_wav: str):
    global _turn_crashes
    try:
        await _dispatch_turn_inner(session, utterance_wav)
    except Exception:  # noqa: BLE001 - a turn must not die silently
        _turn_crashes += 1
        logger.exception("[%s] turn crashed -- answering as unreachable", session.call_id)
        with contextlib.suppress(Exception):
            await _speak(session, SYSTEM_UNREACHABLE_BN, fallback_reason="tool_failure")


async def _dispatch_turn_inner(session: CallSession, utterance_wav: str):
    """One full turn: ASR -> intent -> tool -> templated reply -> TTS.
    Serialized per-call via session.dispatch_lock so replies never
    interleave, even if the caller starts talking again immediately."""
    async with session.dispatch_lock:
        try:
            asr_result = await _asr.transcribe_utterance(utterance_wav)
        finally:
            with contextlib.suppress(OSError):
                os.remove(utterance_wav)

        text = asr_result.text.strip()

        # ADDED BY SOURAV -- "Caller asks for a person immediately" story
        # (Epic: Conversation -- Difficult, Sensitive and Edge Cases). The
        # denominator for immediate_human_escalation_rate() -- counted
        # once per dispatched turn, BEFORE the empty-ASR check just below,
        # so a turn nothing could be transcribed from still belongs in
        # "total turns attempted" rather than being quietly excluded from
        # the rate's own base.
        record_turn_attempt()

        if not text:
            logger.info("[%s] ASR returned empty text", session.call_id)
            await _speak(session, "দুঃখিত, শুনতে পাইনি। আবার বলবেন?", fallback_reason="asr_empty")
            return
        await session.send_json("User", text)

        # ADDED BY SOURAV -- production bug: replies were spoken only in
        # Bengali no matter what language the caller actually used (see
        # detect_language()'s own docstring in agent/bn_normalize.py for
        # the full writeup). Detected fresh from THIS turn's own utterance
        # -- not carried over from a previous turn -- since a caller can
        # code-switch mid-call and every reply should reflect what they
        # just said. Every reply_templates.py function below already
        # accepted a language= argument; it was simply never passed.
        language = detect_language(text)

        # How much did the two decoders agree about what was said? Logged on
        # every turn -- not for debugging, but because the floors in
        # agent/confidence.py are REASONED and cannot become measured until
        # there is a body of these lines to sweep a threshold against. No
        # transcript is logged; only the score, the decoder and the zone.
        turn_zone = confidence.zone(asr_result)
        # "n/a" rather than a number when the decoders were not compared --
        # %.2f would raise on None, and printing 0.00 there would be the same
        # lie the sentinel used to tell.
        #
        # FIXED BY SOURAV -- pre-existing bug, unrelated to KCD-379, found
        # while verifying that story: these four fields were read with a
        # direct attribute access, unlike agent/confidence.py's own
        # zone() (called just above), which reads decoder_used and
        # decoder_agreement off the same asr_result with getattr(...,
        # None). Any ASR result object that doesn't carry the full
        # decoder-agreement dataclass shape -- every dispatch-level test's
        # FakeASRResult in this suite predates that shape and defines only
        # `.text` -- raised AttributeError here on every real turn,
        # crashing the turn as "unreachable" (see main_pcm.py's own
        # try/except around _dispatch_turn_inner) before a single reply
        # template was ever reached. Defaults mirror agent/asr.py's real
        # ASRResult dataclass field defaults exactly (decoder_agreement:
        # None, ctc_words/rnnt_words: 0) so a value actually produced by
        # the real ASR pipeline is completely unaffected by this change --
        # only a fixture/object missing the field now degrades gracefully
        # instead of crashing.
        _agree = getattr(asr_result, "decoder_agreement", None)
        logger.info("[%s] asr agreement=%s decoder=%s words=%d/%d zone=%s",
                    session.call_id,
                    "n/a" if _agree is None else f"{_agree:.2f}",
                    getattr(asr_result, "decoder_used", None),
                    getattr(asr_result, "ctc_words", 0),
                    getattr(asr_result, "rnnt_words", 0), turn_zone)
        # Structured export for the correlation study. Signal and join key
        # only -- never the transcript. See agent/turn_log.py.
        turn_log.record(session.call_id, session.utt_seq, asr_result, turn_zone,
                        call_state=session.call_state)

        if turn_zone == confidence.REJECT:
            # Both decoders produced text and disagreed about nearly all of it.
            # Acting on either version is a guess, so the turn buys nothing and
            # is not worth reading back either -- ask again instead. Any booking
            # in progress is left intact: the caller has not withdrawn it, this
            # one utterance was simply not understood.
            logger.info("[%s] turn rejected on low decoder agreement", session.call_id)
            await _speak(session, "দুঃখিত, ভালো করে শুনতে পাইনি। আরেকবার বলবেন?")
            return

        # ---- answer to a "did I hear you right?" echo -----------------------
        # Handled HERE rather than in _continue_pending, and before its
        # universal is_negative() escape, because "না" means two different
        # things in the two places. In a booking flow it abandons the booking;
        # here it means "you misheard me", and must leave any booking in
        # progress exactly as it was.
        resumed_from_confirm = False
        if session.pending and session.pending.get("awaiting") == "confirm_transcript":
            echoed = session.pending
            session.pending = echoed.get("resume")   # put the real flow back
            if is_affirmative(text):
                # Confirmed. Continue this turn with what was originally heard,
                # not with the word "yes".
                logger.info("[%s] caller confirmed the transcript", session.call_id)
                text = echoed["heard"]
                resumed_from_confirm = True
            elif is_negative(text):
                logger.info("[%s] caller rejected the transcript", session.call_id)
                await _speak(session, "ঠিক আছে, আরেকবার বলবেন?")
                return
            else:
                # Neither yes nor no -- callers usually just say the thing
                # again rather than answering. Treat this utterance as a fresh
                # turn: it carries its own confidence score and gets judged on
                # its own merits below.
                logger.info("[%s] transcript echo answered with a restatement",
                            session.call_id)

        # ---- criterion 1: below the floor, check before acting -------------
        # The decoders disagreed enough that acting on this transcript would be
        # a guess. Reads are included, not just writes: a wrong price spoken
        # confidently is the same class of failure as a wrong booking, and
        # inside a booking flow this is the ONLY place a misheard field is
        # caught -- the final readback faithfully reads back whatever was
        # captured, so a name misheard three turns earlier is confirmed by a
        # caller who hears their own answer echoed correctly.
        skip_zones = ("confirm_transcript", "confirm_booking")
        already_confirming = bool(session.pending) and \
            session.pending.get("awaiting") in skip_zones
        if turn_zone == confidence.CONFIRM and not resumed_from_confirm \
                and not already_confirming:
            session.confirm_attempts += 1
            if session.confirm_attempts > 2:
                # Three in a row means the line, not the utterance, is the
                # problem. Offer a human rather than ask a fourth time.
                logger.info("[%s] repeated low-agreement turns -- offering handoff",
                            session.call_id)
                session.confirm_attempts = 0
                await _speak(session, "লাইনটা পরিষ্কার শোনা যাচ্ছে না। "
                                      "কাউন্টারে একবার কথা বলে নিলে ভালো হয়।")
                return
            logger.info("[%s] echoing transcript for confirmation (attempt %d)",
                        session.call_id, session.confirm_attempts)
            session.pending = {
                "awaiting": "confirm_transcript", "slots": {}, "candidates": None,
                "offered_date": None, "retries": 0,
                "heard": text,             # replayed verbatim once confirmed
                "resume": session.pending,  # the flow this interrupted
            }
            await _speak(session, heard_confirm_prompt(text))
            return

        # The turn is trusted from here on.
        session.confirm_attempts = 0

        # A booking (or the doctor-choice / date-confirm step just before
        # one) already in progress owns this turn -- see _continue_pending's
        # docstring for why intent extraction must NOT also run on top of it.
        if await _continue_pending(session, text):
            # story title: A multi-part question is answered in full
            # The flow that was holding the turn may have just finished, and
            # something the caller asked before it is still waiting.
            await _drain_deferred(session, text)
            return

        try:
            data = await _resolve_intent(session, text)
        except ExtractionError as e:
            logger.error("[%s] intent extraction failed: %s", session.call_id, e)
            await _speak(session, "একটু সমস্যা হচ্ছে, একটু ধরুন।", fallback_reason="llm_failure")
            return

        # story title: A multi-part question is answered in full
        # user story: As a caller who asked two things, I want both answered,
        #   so that I do not have to ask again.
        # acceptance criteria: Every answerable part of a turn is answered in
        #   the order asked, and any part that cannot be answered is
        #   explicitly addressed rather than dropped. Completeness is scored
        #   on a labelled multi-part set.
        #
        # normalise() always returns at least one part and guarantees that
        # parts[0] is the top-level intent and slots -- so a model that never
        # emits `parts` produces exactly the single-part turn this agent had
        # before the story, through the same code path.
        #
        # ADDED BY SOURAV's "Caller asks two questions in one breath" story
        # originally dispatched multiple intents via its own `intents` array
        # and _dispatch_multi_intent_turn(); that array never reaches this
        # branch (agent/llm.py emits `parts`, not `intents` -- see this
        # branch's own "parts" wrapper story above), so multi-part turns are
        # routed through turn_parts/_run_parts() here instead. A single-part
        # turn falls through to the full single-intent chain below rather
        # than into _answer_part() (which backs _run_parts() and still only
        # covers the intents this branch had before dev_sourav's stories
        # landed) -- porting those stories into _answer_part() so a
        # multi-part turn can reach them too is follow-up work, not part of
        # this merge.
        parts = turn_parts.normalise(data)
        if turn_parts.is_multi(parts):
            logger.info("[%s] %d-part turn: %s", session.call_id, len(parts),
                        [p["intent"] for p in parts])
            if not await _run_parts(session, text, parts):
                await _drain_deferred(session, text)
            return

        intent = data["intent"]
        slots = data["slots"]

        # ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
        # previous answer" story. Runs BEFORE any of the per-intent
        # branches below, for every intent (a no-op for the many intents
        # agent/state.py does not apply to at all -- see
        # primary_slot_for_intent()'s own comment). Two outcomes:
        #   - `slots` comes back with the intent's primary entity slot
        #     silently filled in from the last thing actually discussed,
        #     when the caller left it null this turn (a pronoun/elliptical
        #     follow-up) and exactly one candidate is tracked -- every
        #     branch below runs completely unaware anything was backfilled.
        #   - `ambiguous_kind` is set instead when that slot is null AND
        #     more than one different entity of that kind was discussed a
        #     moment ago (compare_options naming two tests, say) -- the
        #     turn stops HERE, asks which one was meant, and remembers
        #     enough (intent + already-known slots) to resume the exact
        #     same question once the caller answers (see
        #     _continue_pending's new "follow_up_clarification" state).
        _state = _session_state(session)
        ambiguous_kind = None
        if _state is not None:
            slots, ambiguous_kind = resolve_follow_up(_state, intent, slots)
        if ambiguous_kind:
            candidates = list(_state.slot_for(ambiguous_kind).names)
            session.pending = {
                "awaiting": "follow_up_clarification", "intent": intent, "slots": slots,
                "kind": ambiguous_kind, "candidates": candidates, "retries": 0,
            }
            await _speak(session, ambiguous_reference_reply(ambiguous_kind, candidates, language=language))
            return

        if intent == "smalltalk":
            await _speak(session, data.get("direct_reply_bn") or "নমস্কার, কী সাহায্য করতে পারি?")
            return

        if intent == "unclear":
            # UPDATED BY SOURAV -- wires up the business's own
            # human_fallback config (lab_tests_with_fallback_config sample
            # file's voice_agent_config.human_fallback block):
            # trigger_condition "query_unresolved_or_low_confidence" maps
            # onto this codebase's existing "unclear" intent (see
            # agent/llm.py's own docstring for exactly when the classifier
            # returns it) -- the one real, already-existing signal for
            # "the caller's query could not be resolved". This branch used
            # to speak a single fixed Bengali-only "sorry, please repeat"
            # line with no language selection at all; it now speaks the
            # business's own per-language "connecting you to an expert"
            # script instead, and records the handoff to the same
            # escalation ledger agent/outcomes.py already maintains -- see
            # human_fallback_reply()'s and record_human_handoff()'s own
            # docstrings for why the config's requested action
            # ("transfer_to_human_agent") is honestly logged rather than
            # literally transferred: this codebase has no telephony
            # transfer capability of any kind to actually do that.
            record_human_handoff(intent, call_id=session.call_id)
            await _speak(session, human_fallback_reply(language=language))
            return

        if intent == "out_of_scope":
            # ADDED BY SOURAV -- "Caller asks something the agent does not
            # cover" story. Distinct from "unclear" just above: the
            # classifier understood EXACTLY what the caller wants here
            # (see agent/llm.py's own "out_of_scope" vs "unclear"
            # distinction) -- it is simply not a service any intent above
            # covers. Rather than apologize-and-connect immediately (the
            # "unclear" path) or silently force it into a lookalike real
            # intent, this offers the caller an explicit choice and acts
            # on whichever they pick next turn -- see _continue_pending's
            # "out_of_scope_choice" branch below.
            await _speak(session, out_of_scope_reply(language=language))
            session.pending = {"awaiting": "out_of_scope_choice", "retries": 0}
            return

        if intent == "clinical_interpretation":
            # ADDED BY SOURAV -- "Caller asks whether their result is
            # dangerous" story (Epic: Conversation -- Difficult, Sensitive
            # and Edge Cases). Mirrors the "out_of_scope" branch immediately
            # above: a fixed, code-level offer is spoken (never anything the
            # model composed -- see agent/reply_templates.py's
            # clinical_interpretation_reply()), and the caller's yes/no is
            # resolved next turn by the "clinical_interpretation_choice"
            # branch in _continue_pending below. There is deliberately no
            # `try`/tools-client lookup here: unlike every intent in the
            # block below, this one never depends on a clinic-api result --
            # LabReport has no clinical-value column for any lookup to
            # return (see clinic-api/models.py), so there is nothing to
            # fetch and nothing that could leak a clinical judgment even by
            # accident.
            await _speak(session, clinical_interpretation_reply(language=language))
            session.pending = {"awaiting": "clinical_interpretation_choice", "retries": 0}
            return

        if intent == "human_direct_request":
            # ADDED BY SOURAV -- "Caller asks for a person immediately"
            # story (Epic: Conversation -- Difficult, Sensitive and Edge
            # Cases). Deliberately the SIMPLEST branch in this whole
            # if/elif chain: no offer, no choice, no `session.pending` set
            # afterward -- the AC's "no retention attempt and no question
            # about why" means there is nothing left to ask. This escalates
            # on the SAME turn, unconditionally, the moment the guard in
            # _resolve_intent() (or _continue_pending(), for a caller
            # mid-flow) fires -- contrast with "out_of_scope" and
            # "clinical_interpretation" just above, which both OFFER a
            # connection and wait for a yes/no next turn. human_fallback_
            # reply() is reused verbatim (the same "connecting you now"
            # line "unclear" already speaks) rather than a new template,
            # because it already says nothing but "connecting you" -- no
            # question, no negotiation -- which is exactly this AC's bar.
            record_immediate_human_handoff(call_id=session.call_id)
            await _speak(session, human_fallback_reply(language=language))
            return

        if intent == "complaint":
            # ADDED BY SOURAV -- "Caller wants to make a complaint" story.
            # Reached when the guard fires inside _resolve_intent() above
            # for a FRESH turn (no in-progress flow) -- the mid-flow case
            # is handled by _continue_pending()'s own guard instead, which
            # never falls through to here. Same zero-negotiation shape as
            # "human_direct_request" just above: no offer, no choice, no
            # `session.pending` left set afterward. `text` here is this
            # turn's own ASR transcript (set at the top of this function,
            # still in scope) -- passed straight through to
            # _finish_complaint() so the stored complaint is exactly what
            # the caller said, never a field extracted or rewritten by the
            # classifier.
            await _finish_complaint(session, text, language=language)
            return

        if intent == "doctor_personal_request":
            # ADDED BY SOURAV -- "Caller wants to speak to a doctor
            # personally" story. Reached when the guard fires inside
            # _resolve_intent() above for a FRESH turn (no in-progress
            # flow) -- the mid-flow case is handled by _continue_pending()'s
            # own guard instead, which never falls through to here. Same
            # zero-negotiation shape as "complaint" just above: no
            # `session.pending` left set afterward -- the caller's own next
            # ordinary utterance (an appointment request or a callback
            # request) is handled entirely by those intents' own existing
            # branches, not by anything special retained here.
            await _finish_doctor_personal_request(session, language=language)
            return

        try:
            if intent == "test_rate":
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_test_rate(slots["test_name"])
                # FIXED BY SOURAV -- KCD-449. This was the one ordinary,
                # plain-single-question dispatch path for test_rate that
                # spoke straight from _speak(), never touching the answer
                # ledger at all -- see agent/answer_ledger.py's own module
                # docstring and tests/test_answer_consistency.py's
                # AST-based safeguard, which fails the build on exactly
                # this shape of miss (a route that forgot the gate).
                # Every OTHER test_rate call site (the near-match retry,
                # the multi-part single-utterance path) already routed
                # through _speak_fact(); this plain path is also the
                # single most common way a caller ever asks this question,
                # so it was also the one most likely to be asked twice
                # with nothing recorded to compare the second time against.
                if await _speak_fact(session, intent, slots, result,
                                     test_rate_reply(slots, result, language=language)):
                    return
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "test_sample":
                # UPDATED BY SOURAV -- restores parity with main_pcm.py,
                # which already had this branch (Story 5, "caller asks what
                # sample is needed") while main.py never did. Found while
                # wiring the report_status/report_send combined story:
                # main_pcm.py is a GENERATED file (see its own module
                # docstring and tools/make_pcm_variant.py) meant to be
                # produced FROM main.py, but this branch was added by hand
                # directly to main_pcm.py at some point without re-running
                # the generator off an updated main.py -- so a caller on
                # the WAV transport (main.py, ports 8080/8100 per
                # deploy/start_all.sh) asking only about sample type hit no
                # matching branch at all, even though agent/llm.py
                # classifies "test_sample" correctly on either transport.
                # Restoring it here BEFORE regenerating main_pcm.py from
                # this file closes that gap for good: from now on
                # main_pcm.py is only ever produced by re-running that
                # script against this file, so the two cannot drift apart
                # on this branch (or the three this story adds) again.
                # Same tool call as test_rate -- clinic-api's test lookup
                # already returns sample_type on every call, nothing new
                # was added to the API for this -- only the reply function
                # differs, so a caller who asked ONLY about the sample
                # hears just that, not the bundled rate+sample+duration
                # answer test_rate gives.
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_test_rate(slots["test_name"])
                await _speak(session, sample_type_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "test_duration":
                # ADDED BY SOURAV -- fixes a real production bug, reported
                # directly from a live call transcript:
                #   [User] How long does it take to get the urine test report?
                #   [AI]   Urine test rate is 200 taka.
                # A caller asking about REPORT TURNAROUND TIME was being
                # misclassified as "test_rate" and answered with the
                # test's PRICE instead. Root cause: an earlier story
                # ("Caller asks the price of a test") narrowed
                # test_rate_reply() to speak ONLY the price, but
                # agent/llm.py's intent prompt was never updated to match
                # -- it kept telling the classifier that "how long results
                # take" belongs to test_rate. See test_duration_reply()'s
                # own docstring for the full writeup, including a second,
                # related bug found and fixed in agent/fast_path.py.
                # Same tool call as test_rate/test_sample -- clinic-api's
                # test lookup already returns report_time_hours on every
                # call, nothing new was added to the API for this -- only
                # the reply function differs, so a caller who asked ONLY
                # about turnaround time hears just that, never the price
                # or the sample.
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_test_rate(slots["test_name"])
                await _speak(session, test_duration_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "test_preparation":
                # ADDED BY SOURAV -- "Caller asks how to prepare for a
                # test" story. Unlike test_rate/test_sample/test_duration
                # just above, this calls a DEDICATED new endpoint
                # (get_test_preparation -> GET /api/v1/tests/preparation)
                # rather than reusing get_test_rate's response, since
                # preparation data (fasting rules, medication holds,
                # per-language ready-to-speak scripts) is not part of that
                # payload at all -- see clinic-api/main.py's
                # _test_preparation_reply_dict() for the response shape.
                # Same test_name-required gate as those three: "how do I
                # prepare" has no "list every test's prep instructions"
                # analog the way health_package's bare "what packages do
                # you have" does, so a missing test_name always re-prompts
                # rather than trying to answer something unbounded.
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_test_preparation(slots["test_name"])
                await _speak(session, test_preparation_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "walkin_eligibility":
                # ADDED BY SOURAV -- Phase 1: Database Schema & Policy
                # Tables. Same single-required-slot gate as test_rate/
                # test_preparation above.
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_walkin_policy(slots["test_name"])
                await _speak(session, walkin_eligibility_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "prescription_requirements":
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_prescription_policy(slots["test_name"])
                await _speak(session, prescription_requirements_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "insurance_coverage":
                # Two required slots, not one -- ask for whichever is
                # still missing, mirroring book_appointment's own
                # merge-onto-pending pattern below (see agent/
                # semantic_cache.py's _is_l2_eligible for why this intent
                # is excluded from L2 entirely, the same reason
                # book_appointment is). A caller who names only the test
                # ("amar CBC insurance-e cover hobe?") gets asked for
                # their insurer next turn, and vice versa; either slot
                # already known this turn is kept.
                merged = dict(session.pending["slots"]) if session.pending else {}
                for field in ("test_name", "insurance_provider_name"):
                    if slots.get(field):
                        merged[field] = slots[field]
                missing = next((f for f in ("test_name", "insurance_provider_name") if not merged.get(f)), None)
                if missing:
                    session.pending = {
                        "awaiting": "insurance_coverage_slot", "slots": merged,
                        "missing_field": missing, "retries": 0,
                    }
                    await _speak(session, missing_slot_prompt(intent, missing, language=language))
                    return
                session.pending = None
                result = await _tools.get_insurance_coverage(merged["test_name"], merged["insurance_provider_name"])
                await _speak(session, insurance_coverage_reply(merged, result, language=language))

            elif intent == "compare_options":
                # ADDED BY SOURAV -- "Caller asks the agent to compare two
                # options" story. Same two-required-slots merge-onto-
                # pending pattern as insurance_coverage just above, for
                # the identical reason (either name can be missing this
                # turn). Once both names are in hand, EACH is resolved
                # independently via _resolve_comparable_entity() (tries
                # the test catalogue, then the package catalogue -- see
                # that function's own docstring) before
                # agent/compare_flow.py's build_comparison() computes the
                # actual price/component comparison in code -- this
                # branch itself does no arithmetic and makes no
                # recommendation; it only fetches both sides' live data
                # and hands it to compare_flow/reply_templates.
                merged = dict(session.pending["slots"]) if session.pending else {}
                for field in ("compare_option_a", "compare_option_b"):
                    if slots.get(field):
                        merged[field] = slots[field]
                missing = next((f for f in ("compare_option_a", "compare_option_b") if not merged.get(f)), None)
                if missing:
                    session.pending = {
                        "awaiting": "compare_options_slot", "slots": merged,
                        "missing_field": missing, "retries": 0,
                    }
                    await _speak(session, missing_slot_prompt(intent, missing, language=language))
                    return
                session.pending = None
                name_a, name_b = merged["compare_option_a"], merged["compare_option_b"]
                entity_a, entity_b = await _resolve_comparable_entity(name_a), await _resolve_comparable_entity(name_b)
                comparison = build_comparison(entity_a, entity_b)
                await _speak(session, compare_options_reply(name_a, name_b, entity_a, entity_b, comparison, language=language))
                _remember_compared_entities(session, entity_a, entity_b)

            elif intent == "billing_balance":
                # ADDED BY SOURAV -- Phase 1: Outstanding Balance / Billing
                # story. Identity resolved by PHONE (RULE 14/15), same
                # gate as report_status/report_send below -- deliberately
                # NOT behind OTP (see clinic-api/models.py's
                # PatientBilling docstring for that scoping decision).
                phone = parse_phone(slots.get("phone") or "")
                if not phone:
                    session.pending = {"awaiting": "billing_phone", "retries": 0}
                    await _speak(session, missing_slot_prompt(intent, "phone", language=language))
                    return
                result = await _tools.get_patient_billing(phone)
                await _speak(session, billing_balance_reply(result, language=language))

            elif intent == "report_status":
                # ADDED BY SOURAV -- "Lab Report Status & Secure Delivery"
                # combined story. Identity is resolved by PHONE, never by
                # name (RULE 14/15) -- if the caller's utterance didn't
                # carry one, ask for it and park in the "phone" pending
                # state above rather than guessing or proceeding without it.
                phone = parse_phone(slots.get("phone") or "")
                if not phone:
                    session.pending = {
                        "awaiting": "report_phone", "flow": "report_status",
                        "test_name": slots.get("test_name"), "retries": 0,
                    }
                    await _speak(session, missing_slot_prompt(intent, "phone", language=language))
                    return
                await _handle_report_lookup(session, phone, slots.get("test_name"), "report_status", language=language)

            elif intent == "report_send":
                # ADDED BY SOURAV -- same identity-by-phone gate as
                # report_status just above; the two intents share
                # _handle_report_lookup/_finish_report_flow and differ only
                # in `flow`, which agent/report_flow.py's
                # interpret_report_status_result() uses to decide whether a
                # READY+enabled report gets the "shall I send it?" offer
                # (report_status) or goes straight to requesting delivery
                # (report_send, since the caller already asked for it).
                phone = parse_phone(slots.get("phone") or "")
                if not phone:
                    session.pending = {
                        "awaiting": "report_phone", "flow": "report_send",
                        "test_name": slots.get("test_name"), "retries": 0,
                    }
                    await _speak(session, missing_slot_prompt(intent, "phone", language=language))
                    return
                await _handle_report_lookup(session, phone, slots.get("test_name"), "report_send", language=language)

            elif intent == "doctor_availability":
                if not slots.get("doctor_name"):
                    await _speak(session, missing_slot_prompt(intent, "doctor_name", language=language))
                    return
                # Default to TODAY, not "whenever next available": a bare
                # "ডাক্তার সেন আছেন?" with no date mentioned is a caller
                # asking about right now, and the reply text below already
                # said " আজ" (today) for exactly this case -- the old code
                # passed date=None through to the API, which answers a
                # different question ("when next"), so a doctor who simply
                # wasn't in today got reported by their NEXT sitting date
                # instead of "not today, but they're on Tuesdays" etc.
                date_iso = slots.get("date") or datetime.date.today().isoformat()
                result = await _tools.get_doctor_availability(slots["doctor_name"], date_iso)
                # FIXED BY SOURAV -- KCD-449. Same miss as test_rate just
                # above: this plain single-question path spoke straight
                # from _speak(), bypassing the answer ledger entirely,
                # while every OTHER doctor_availability call site (the
                # near-match retry, the "date" follow-up, the multi-part
                # single-utterance path) already used _speak_fact().
                # offered_date is threaded through exactly as those other
                # call sites do, so a later disambiguation is compared
                # against the same day this turn asked about. A True
                # return means _speak_fact() already spoke a "did you
                # mean...?" offer and opened its OWN pending state for it
                # -- return immediately, before the "keep the flow open
                # for booking" logic below would otherwise clobber it.
                if await _speak_fact(session, intent, slots, result,
                                     doctor_availability_reply(slots, result, language=language),
                                     offered_date=date_iso):
                    return
                _remember_primary_entity(session, intent, slots, result)

                # Keep the flow open for "yes, book that day" / "another
                # day" -- doctor_availability_reply() just asked exactly
                # that question. See _continue_pending's "date" state.
                offered = None
                if result.get("found"):
                    offered = result.get("date") if result.get("available") else result.get("next_available_date")
                session.pending = {
                    "awaiting": "date",
                    "slots": {"doctor_name": result.get("doctor_name") or slots["doctor_name"]},
                    "candidates": None, "offered_date": offered, "retries": 0,
                } if offered else None

            elif intent == "doctor_schedule":
                # ADDED BY SOURAV -- "Caller asks when a doctor sits" story.
                # Deliberately DATE-FREE, unlike doctor_availability just
                # above: this intent exists exactly for the caller who has
                # NOT named a day and wants the doctor's general recurring
                # weekly schedule instead (see agent/llm.py's SYSTEM_PROMPT
                # for how the two are told apart at classification time,
                # and clinic-api/main.py::doctor_schedule()'s docstring for
                # the response shape). No `date` slot is read or passed
                # here at all -- even if the LLM happened to also extract
                # one from the same utterance, it is not used, since a
                # date would silently turn this back into the OTHER
                # question this intent exists to be distinct from.
                #
                # Same known limitation as doctor_availability just above,
                # not introduced here: no pending state is opened when
                # doctor_name is missing, so the caller's next utterance
                # (e.g. a bare doctor's name in reply to the prompt) goes
                # through a fresh LLM classification rather than a
                # targeted single-slot fill. Flagged, not fixed -- fixing
                # it would mean touching doctor_availability's identical
                # gap too, which is out of this story's scope.
                if not slots.get("doctor_name"):
                    await _speak(session, missing_slot_prompt(intent, "doctor_name", language=language))
                    return
                result = await _tools.get_doctor_schedule(slots["doctor_name"])
                # UPDATED BY SOURAV -- KCD-385 near-match fix. clinic-api's
                # doctor_schedule() endpoint now resolves the name through
                # the same _resolve_doctor() ranker doctor_availability
                # uses (see that endpoint's own docstring), so a loosely
                # named/misspelled doctor can come back
                # {"ambiguous": true, "candidates": [...]} instead of a
                # flat not-found. Before this fix this branch always spoke
                # straight from `result`: an ambiguous response has no
                # "found" key at all, so doctor_schedule_reply() read it
                # as a plain not-found and told the caller "we don't have
                # a doctor named X" about a doctor we clearly have
                # candidates for. Routing through _speak_fact() is what
                # actually speaks the "did you mean...?" offer and opens
                # the entity_choice pending state instead -- see
                # _continue_pending's own "doctor_schedule" case for what
                # happens when the caller answers it.
                #
                # UPDATE -- KCD-449: doctor_availability's own first-turn
                # dispatch just above had this identical ambiguous-
                # response gap (it spoke straight from `result` the same
                # way, unwrapped) until it was wrapped in _speak_fact()
                # too, as part of wiring it into the answer-consistency
                # ledger -- see that branch's own "FIXED BY SOURAV --
                # KCD-449" comment. That gap is closed now, as a side
                # effect of this story's fix, not merely flagged.
                if await _speak_fact(session, intent, slots, result,
                                     doctor_schedule_reply(slots, result, language=language)):
                    return
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "doctors_by_department":
                if not slots.get("department"):
                    await _speak(session, missing_slot_prompt(intent, "department", language=language))
                    return
                # Default to TODAY when the caller didn't name a date, same
                # reasoning as doctor_availability above: "অর্থোতে কারা
                # আছেন" (who's in ortho) is almost always asking who is
                # actually in the chamber right now, not for a roster of
                # every doctor the department has ever employed regardless
                # of whether they sit this week. Only an EXPLICIT date
                # bypasses this (used as-is below).
                date_iso = slots.get("date") or datetime.date.today().isoformat()
                result = await _tools.get_doctors_by_department(slots["department"], date_iso)
                # FIXED BY SOURAV -- KCD-449. Same miss as test_rate and
                # doctor_availability above: this plain single-question
                # path spoke straight from _speak(), bypassing the answer
                # ledger entirely. Routed through _speak_fact() the same
                # way every other doctors_by_department call site already
                # is, with offered_date threaded through so a later
                # disambiguation compares against the same day this turn
                # asked about. A True return means an offer was already
                # spoken and its own pending state opened -- return
                # immediately, before the "continue straight into
                # booking" logic below would otherwise clobber it.
                if await _speak_fact(session, intent, slots, result,
                                     doctors_by_department_reply(slots, result, language=language),
                                     offered_date=date_iso):
                    return

                # Continue straight into booking: offer the doctors just
                # listed as candidates, so the caller's very next utterance
                # -- which may be nothing but a bare doctor name -- is
                # matched against THIS list rather than sent to the LLM with
                # no context to interpret it against. See _continue_pending's
                # "doctor_choice" state.
                if result.get("found") and result.get("doctors"):
                    session.pending = {
                        "awaiting": "doctor_choice",
                        "slots": {},
                        "candidates": [
                            {"name": d["name"], "name_bn": d.get("doctor_name_bn")}
                            for d in result["doctors"]
                        ],
                        "offered_date": date_iso,
                        "retries": 0,
                    }
                elif result.get("found"):
                    # Department exists but nobody sits that day --
                    # doctors_by_department_reply() just told the caller
                    # exactly that and invited another day ("অন্য কোনো
                    # দিনের কথা জিজ্ঞেস করতে পারেন"). Stay in the flow so the
                    # caller's next utterance is interpreted as THAT date
                    # instead of needing to restate the whole department
                    # question from scratch -- see _continue_pending's
                    # "department_date" state.
                    session.pending = {
                        "awaiting": "department_date",
                        "slots": {"department": slots["department"]},
                        "candidates": None, "offered_date": None, "retries": 0,
                    }
                else:
                    session.pending = None

            elif intent == "book_appointment":
                # Merge onto whatever session.pending already knows (e.g. a
                # doctor_name carried over from a doctor_availability or
                # doctors_by_department turn moments ago) rather than
                # requiring every field in one utterance -- that all-or-
                # nothing check was the other half of "pipeline breaking":
                # a caller who gave the doctor and date in one sentence and
                # the time in the next used to have the doctor/date silently
                # discarded the moment ANY field was still missing.
                merged = dict(session.pending["slots"]) if session.pending else {}
                for field in _BOOKING_FIELDS:
                    if slots.get(field):
                        merged[field] = slots[field]

                missing = _next_missing(merged)
                if missing is None:
                    # A caller who gave all 5 fields in one breath still
                    # gets the pre-write readback -- this is the SAME gap
                    # the multi-turn flow had (see _continue_pending's
                    # "confirm_booking" state): a single-shot utterance is
                    # exactly as capable of a misheard phone digit as one
                    # collected field-by-field.
                    session.pending = {
                        "awaiting": "confirm_booking", "slots": merged, "candidates": None,
                        "offered_date": (session.pending or {}).get("offered_date"), "retries": 0,
                    }
                    await _speak(session, booking_confirmation_prompt(merged, language=language))
                    return

                session.pending = {
                    "awaiting": missing, "slots": merged, "candidates": None,
                    "offered_date": (session.pending or {}).get("offered_date"), "retries": 0,
                }
                await _speak(session, missing_slot_prompt(intent, missing, language=language))

            elif intent == "request_callback":
                # ADDED BY SOURAV -- "Caller asks to be called back" story
                # (Evidence: "No outbound capability"). Availability is
                # checked FIRST, before asking for a single detail --
                # Acceptance Criterion 3: a caller is never walked through
                # collecting a time window and phone number only to be
                # refused at the very end. CALLBACKS_ENABLED is checked
                # BEFORE ever calling get_clinic_info() -- a deployment
                # that has turned the feature off entirely has no reason
                # to pay for that round-trip (cached or not) just to
                # decide something a config constant already answered.
                if not CALLBACKS_ENABLED:
                    await _speak(session, callback_unavailable_reply("disabled", language=language))
                    return

                # get_clinic_info() is the SAME already-cached call
                # clinic_info's own branch above makes (agent/
                # reference_data_cache.py) -- no new tool, no new network
                # round-trip pattern.
                hours_result = await _tools.get_clinic_info()
                hours = hours_result.get("hours") if hours_result.get("found") else None
                availability = check_callback_availability(
                    hours, datetime.date.today().weekday(),
                    datetime.datetime.now().strftime("%H:%M"), CALLBACKS_ENABLED,
                )
                if not availability["available"]:
                    await _speak(session, callback_unavailable_reply(availability["reason"], language=language))
                    return

                # Merge onto whatever session.pending already knows, same
                # "don't discard a field the caller already gave" reasoning
                # as book_appointment just above -- a caller who names a
                # time window AND a phone number in one breath should never
                # be asked for either again.
                merged = dict(session.pending["slots"]) if session.pending else {}
                for field in (*_CALLBACK_FIELDS, "callback_reason"):
                    if slots.get(field):
                        merged[field] = slots[field]

                # Acceptance Criterion 1's "preserving the conversation
                # context and reason" -- resolved ONCE, here, on the turn
                # that actually opens this flow (session.state reflects
                # whatever was discussed earlier in THIS call right now;
                # nothing about it changes while the rest of this flow
                # collects the remaining fields over the next turns, so
                # there is no benefit to re-resolving it later, only risk
                # of it drifting from what was true when the caller asked).
                # Always stored, even as null (build_callback_reason()'s own
                # "no reason given" case) -- see agent/callback_flow.py's
                # own docstring for why that null is never papered over
                # with an invented generic reason.
                if "callback_reason" not in merged:
                    state = _session_state(session)
                    merged["callback_reason"] = build_callback_reason(
                        slots.get("callback_reason"),
                        active_test=state.slot_for("test").primary if state else None,
                        active_doctor=state.slot_for("doctor").primary if state else None,
                        active_package=state.slot_for("package").primary if state else None,
                    )

                missing = _next_missing_callback(merged)
                if missing is None:
                    # Same "every critical value is read back before it is
                    # used" discipline as book_appointment's own
                    # confirm_booking state just above.
                    session.pending = {
                        "awaiting": "confirm_callback", "slots": merged, "candidates": None,
                        "offered_date": None, "retries": 0,
                    }
                    await _speak(session, callback_confirmation_prompt(merged, language=language))
                    return

                session.pending = {
                    "awaiting": missing, "slots": merged, "candidates": None,
                    "offered_date": None, "retries": 0,
                }
                await _speak(session, missing_slot_prompt("request_callback", missing, language=language))

            elif intent == "health_package":
                # ADDED BY SOURAV -- "Caller asks about a health package"
                # story. Deliberately NEVER re-prompts for a missing
                # "package_name" the way every single-entity intent above
                # does for its own required slot -- see agent/llm.py's own
                # comment on VALID_INTENTS: a caller who names no package
                # at all is asking a complete, different, equally valid
                # question ("what packages do you have"), backed by its
                # own clinic-api list endpoint, not an incomplete
                # extraction waiting on a re-prompt.
                if slots.get("package_name"):
                    result = await _tools.search_health_package(slots["package_name"])
                    await _speak(session, health_package_reply(slots, result, language=language))
                    _remember_primary_entity(session, intent, slots, result)
                else:
                    result = await _tools.get_health_packages()
                    await _speak(session, health_packages_list_reply(result, language=language))

            elif intent == "clinic_info":
                # ADDED BY SOURAV -- "Caller asks opening hours, address or
                # directions" story. No slot is required to call the tool
                # (clinic-api/models.py's ClinicInfo is a singleton table) --
                # "info_topic" only narrows which part of the already-
                # fetched answer gets SPOKEN, in clinic_info_reply() itself.
                # "today_weekday" resolves "which day" here, in dispatch,
                # the same way doctor_availability's date_iso default does
                # just above -- reply_templates.py never imports datetime
                # itself (see clinic_info_reply()'s own docstring).
                result = await _tools.get_clinic_info()
                info_slots = {
                    "info_topic": slots.get("info_topic"),
                    "today_weekday": datetime.date.today().weekday(),
                }
                await _speak(session, clinic_info_reply(info_slots, result, language=language))

        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                         fallback_reason="tool_failure")


# ADDED BY SOURAV -- "Caller asks two questions in one breath" story. Sets
# of intents that never get a real inline answer in a COMBINED turn,
# regardless of slot completeness -- see _resolve_combinable_intent_fragment's
# own docstring just below for why each is excluded rather than composed.
_MULTI_INTENT_NEEDS_SEPARATE_FLOW = {
    "book_appointment", "report_status", "report_send",
    # ADDED BY SOURAV -- "Caller asks whether their result is dangerous"
    # story. Even without this explicit entry, the function's own
    # defensive fallback below (any intent not otherwise handled) already
    # routes this here -- but it is listed explicitly, matching this
    # codebase's "no implicit intent" documentation style, and because a
    # safety-critical intent should never depend on falling through to a
    # catch-all to behave correctly.
    "clinical_interpretation",
    # ADDED BY SOURAV -- "Caller asks for a person immediately" story.
    # Same reasoning as clinical_interpretation just above -- listed
    # explicitly even though the guard's own whole-utterance short-circuit
    # in _resolve_intent() means a solo "give me a human" turn never
    # actually reaches the multi-intent LLM path at all.
    "human_direct_request",
    # ADDED BY SOURAV -- "Caller wants to make a complaint" story. Same
    # reasoning as human_direct_request just above -- listed explicitly
    # even though agent/complaint_flow.py's own whole-utterance guard in
    # _resolve_intent() means a complaint phrased in this module's own
    # phrase set never actually reaches the multi-intent LLM path at all.
    # Defense-in-depth for the case the guard's literal phrase list misses
    # but the classifier itself still tags one PART of a combined
    # utterance as "complaint" (e.g. "book me an appointment and also I
    # want to raise a complaint about..."): AC 5 rules out ever composing
    # an inline "answer" to a complaint, so it must always be routed here
    # rather than fragment-resolved like an ordinary lookup.
    "complaint",
    # ADDED BY SOURAV -- "Caller wants to speak to a doctor personally"
    # story. Same reasoning as complaint just above -- listed explicitly
    # even though agent/doctor_personal_request.py's own whole-utterance
    # guard in _resolve_intent() means this phrasing never actually
    # reaches the multi-intent LLM path at all. Defense-in-depth for the
    # case the guard's phrase/regex coverage misses but the classifier
    # itself still tags one PART of a combined utterance this way: AC 3
    # rules out ever composing an inline "answer" that might name a
    # doctor or promise a callback, so it must always be routed to the
    # fixed template rather than fragment-resolved like an ordinary
    # lookup.
    "doctor_personal_request",
}
_MULTI_INTENT_NO_FRAGMENT = {"smalltalk", "unclear"}


async def _resolve_combinable_intent_fragment(session: CallSession, intent: str, slots: dict, language: str) -> str | None:
    """ADDED BY SOURAV -- "Caller asks two questions in one breath" story.
    Resolves ONE intent (one entry of a multi-question turn's "intents"
    array) into its own spoken fragment for _dispatch_multi_intent_turn()
    below to join with the others, in order.

    UPDATED BY SOURAV -- KCD-449. Takes `session` now, purely to reach
    session.answer_ledger: the three ledgered intents (test_rate,
    doctor_availability, doctors_by_department) route their fragment
    through _record_answer_for_ledger() below before returning it -- see
    that function's own docstring for why this path needed its own,
    narrower version of what _speak_fact() does for the solo dispatch
    branches.

    Returns None ONLY for "smalltalk"/"unclear" -- see
    _MULTI_INTENT_NO_FRAGMENT above: neither is really a second QUESTION
    the caller needs an honest answer or acknowledgment for (a bare "ভালো
    আছেন?" tacked onto a real question is not something Criterion 2's
    "explicitly acknowledge the unanswerable one" was written for), so
    these two are the sole, deliberate exception to "nothing is silently
    dropped." Every other intent returns a real, non-empty fragment, one
    of:
      - The SAME reply_templates function a solo turn for that intent
        would use, called exactly the same way (a "found but not yet
        reviewed" policy row -- Criterion 2's "unreviewed" example --
        already gets an honest sentence for free from that same function,
        e.g. walkin_eligibility_reply(); zero new code needed for that
        case specifically).
      - multi_intent_missing_info_reply() when an otherwise-combinable
        intent is missing a slot it needs (test_name, doctor_name,
        department, or -- for insurance_coverage/billing_balance -- either
        of their two/one required fields). Deliberately generic rather
        than a targeted per-field re-prompt: a combined turn does not open
        a SECOND pending state on top of whatever the turn's other
        question may already need to report, so there is nowhere to
        attach a targeted follow-up question to (see
        _dispatch_multi_intent_turn's own docstring).
      - multi_intent_out_of_scope_reply() for "out_of_scope" -- a
        non-interactive acknowledgment, unlike out_of_scope_reply()'s
        solo interactive yes/no offer (again: no second pending state).
      - multi_intent_needs_separate_flow_reply() for book_appointment/
        report_status/report_send (_MULTI_INTENT_NEEDS_SEPARATE_FLOW) --
        ALWAYS, even when every slot they'd need already happens to be
        present. These three are multi-turn, sometimes security-sensitive
        (OTP) flows in their own right, not something to fold into a
        shared reply alongside an unrelated question.

    Deliberately narrower than the solo dispatch branches in _dispatch_turn's
    own if/elif chain above: this NEVER opens a follow-up pending state of
    any kind, even for an intent that would open one solo (doctor_
    availability's "book this day?" offer, doctors_by_department's
    candidate list, insurance_coverage's/billing_balance's slot-fill
    prompt) -- composing two independently-stateful sub-conversations into
    one reply is a different, larger problem than "answer both questions
    honestly in one turn."
    """
    if intent in _MULTI_INTENT_NO_FRAGMENT:
        return None

    if intent == "out_of_scope":
        return multi_intent_out_of_scope_reply(language=language)

    if intent in _MULTI_INTENT_NEEDS_SEPARATE_FLOW:
        return multi_intent_needs_separate_flow_reply(language=language)

    if intent == "test_rate":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_test_rate(slots["test_name"])
        # FIXED BY SOURAV -- KCD-449. See _record_answer_for_ledger()'s
        # own docstring.
        return _record_answer_for_ledger(session, intent, slots, result,
                                         test_rate_reply(slots, result, language=language))

    if intent == "test_sample":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_test_rate(slots["test_name"])
        return sample_type_reply(slots, result, language=language)

    if intent == "test_duration":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_test_rate(slots["test_name"])
        return test_duration_reply(slots, result, language=language)

    if intent == "test_preparation":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_test_preparation(slots["test_name"])
        return test_preparation_reply(slots, result, language=language)

    if intent == "walkin_eligibility":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_walkin_policy(slots["test_name"])
        return walkin_eligibility_reply(slots, result, language=language)

    if intent == "prescription_requirements":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_prescription_policy(slots["test_name"])
        return prescription_requirements_reply(slots, result, language=language)

    if intent == "insurance_coverage":
        if not slots.get("test_name") or not slots.get("insurance_provider_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_insurance_coverage(slots["test_name"], slots["insurance_provider_name"])
        return insurance_coverage_reply(slots, result, language=language)

    if intent == "billing_balance":
        phone = parse_phone(slots.get("phone") or "")
        if not phone:
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_patient_billing(phone)
        return billing_balance_reply(result, language=language)

    if intent == "doctor_availability":
        if not slots.get("doctor_name"):
            return multi_intent_missing_info_reply(language=language)
        date_iso = slots.get("date") or datetime.date.today().isoformat()
        result = await _tools.get_doctor_availability(slots["doctor_name"], date_iso)
        # FIXED BY SOURAV -- KCD-449. See _record_answer_for_ledger()'s
        # own docstring.
        return _record_answer_for_ledger(session, intent, slots, result,
                                         doctor_availability_reply(slots, result, language=language))

    if intent == "doctor_schedule":
        if not slots.get("doctor_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_doctor_schedule(slots["doctor_name"])
        return doctor_schedule_reply(slots, result, language=language)

    if intent == "doctors_by_department":
        if not slots.get("department"):
            return multi_intent_missing_info_reply(language=language)
        date_iso = slots.get("date") or datetime.date.today().isoformat()
        result = await _tools.get_doctors_by_department(slots["department"], date_iso)
        # FIXED BY SOURAV -- KCD-449. See _record_answer_for_ledger()'s
        # own docstring.
        return _record_answer_for_ledger(session, intent, slots, result,
                                         doctors_by_department_reply(slots, result, language=language))

    if intent == "health_package":
        if slots.get("package_name"):
            result = await _tools.search_health_package(slots["package_name"])
            return health_package_reply(slots, result, language=language)
        result = await _tools.get_health_packages()
        return health_packages_list_reply(result, language=language)

    if intent == "clinic_info":
        result = await _tools.get_clinic_info()
        info_slots = {
            "info_topic": slots.get("info_topic"),
            "today_weekday": datetime.date.today().weekday(),
        }
        return clinic_info_reply(info_slots, result, language=language)

    # Defensive: every member of VALID_INTENTS (agent/llm.py) is handled
    # explicitly somewhere above -- this is unreachable in practice, but
    # falls back to the same honest "ask that one separately" fragment
    # book_appointment/report_status/report_send get, rather than ever
    # silently dropping an intent this function does not recognize.
    return multi_intent_needs_separate_flow_reply(language=language)


def _remember_multi_intent_entities(session: CallSession, intents_list: list[dict]) -> None:
    """ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
    previous answer" story. A multi-intent turn naming two DIFFERENT
    tests in the same breath (e.g. "CBC-r rate koto, ar Lipid Profile-er
    sample ki lagbe?") is exactly the "multiple entities discussed
    previously" ambiguity Acceptance Criterion 2 describes -- handled
    here, once, after the whole turn's fragments are resolved, so a bare
    pronoun in the NEXT turn cannot silently be guessed as one or the
    other (see agent/state.py's own docstring).

    Deliberately uses the CALLER'S OWN WORDS for each name, not a tool
    result's canonical spelling: _resolve_combinable_intent_fragment()
    above returns only a rendered reply string, not the resolved entity
    data, so reconfirming a canonical name here would mean a second,
    duplicate tool call purely to remember it. This is a smaller
    guarantee than the single-intent path's _remember_primary_entity()
    (which only ever remembers a name a lookup just confirmed exists) --
    flagged, not silently equated with it -- but clinic-api's own lookups
    already tolerate the caller's raw phrasing fine on their own, so a
    follow-up backfilled from a multi-intent turn's memory is no worse
    off than the ORIGINAL turn's own lookup was.
    """
    state = _session_state(session)
    if state is None:
        return
    by_kind: dict[str, list[str]] = {}
    for item in intents_list:
        slot_key = primary_slot_for_intent(item.get("intent"))
        if slot_key is None:
            continue
        name = (item.get("slots") or {}).get(slot_key)
        if not name:
            continue
        by_kind.setdefault(kind_for_slot(slot_key), []).append(name)

    for kind, names in by_kind.items():
        distinct = list(dict.fromkeys(names))  # de-dupe, order-preserved
        if len(distinct) > 1:
            state.mark_ambiguous(kind, distinct)
        else:
            state.mark(kind, distinct[0])


async def _dispatch_multi_intent_turn(session: CallSession, intents_list: list[dict], language: str) -> None:
    """ADDED BY SOURAV -- "Caller asks two questions in one breath" story.
    Only ever called from _dispatch_turn above, and only when
    `len(intents_list) > 1` -- a single-entry list (the overwhelming
    majority of turns: a fast_path hit, a plain single-question LLM
    extraction, or any pre-this-story test mock) takes the ORIGINAL,
    completely untouched if/elif chain in _dispatch_turn instead. Nothing
    about that existing path changes for this story.

    Story Criterion 1 (Order Preservation): `intents_list` is iterated
    below in order, exactly as agent/llm.py's SYSTEM_PROMPT_TEMPLATE
    instructs the model to return it, and each fragment is appended to
    `fragments` in that same order -- no sorting or reordering anywhere.
    The joined reply is strictly in the order asked, by construction.

    Story Criterion 2 (Honest Partial Handling): every intent in
    `intents_list` produces either a real, grounded fragment or one of the
    three honest, non-fabricating fallback fragments -- see
    _resolve_combinable_intent_fragment() above for exactly which and why.
    Nothing is silently dropped; smalltalk/unclear are the sole,
    deliberate exception (see that function's own docstring).

    Tool-failure isolation ("individual tool calls executed independently
    without one tool failure short-circuiting the second"): each intent's
    fragment is resolved inside its OWN try/except ToolCallError in the
    loop below -- unlike the solo dispatch's single try/except wrapping
    its ENTIRE if/elif chain -- so intent #1's clinic-api call failing can
    never prevent intent #2's from running at all.
    """
    fragments: list[str] = []
    for item in intents_list:
        intent = item.get("intent")
        slots = item.get("slots") or {}
        try:
            fragment = await _resolve_combinable_intent_fragment(session, intent, slots, language)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed for intent %s (multi-intent turn): %s",
                         session.call_id, intent, e)
            fragment = "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।"
        if fragment:
            fragments.append(fragment)

    if not fragments:
        # Defensive: every entry was smalltalk/unclear. agent/llm.py's own
        # prompt instructs the model to never split one real question into
        # several entries, so a genuine multi-question turn should not
        # reach this -- but rather than speak nothing at all, fall back to
        # the same honest "connecting you to an expert" handling the solo
        # "unclear" path gives above.
        record_human_handoff("unclear", call_id=session.call_id)
        await _speak(session, human_fallback_reply(language=language))
        return

    await _speak(session, " ".join(fragments))
    _remember_multi_intent_entities(session, intents_list)


async def _resync_after_playback(session: CallSession) -> bool:
    """Drop everything captured while the agent was talking, by moving
    processed_until_s to the current end of the decoded buffer. That region
    is muted silence from the client's side; skipping it keeps the turn
    detector from ever analysing it, and -- more importantly -- keeps
    processed_until_s anchored to real time instead of drifting a full
    reply behind, which is what made later turns surface late."""
    buffer_end_s = session.audio.duration_s
    session.processed_until_s = max(session.processed_until_s,
                                    buffer_end_s - RESYNC_REWIND_S)
    session.resync_pending = False
    logger.info("[%s] resynced to %.2fs after playback", session.call_id, session.processed_until_s)
    return True


async def _turn_poll_loop(session: CallSession):
    """Runs for the lifetime of the call. Every POLL_INTERVAL_S, re-decodes
    the growing buffer -- ALWAYS from byte 0, since that's the only way
    the WebM container stays valid -- then asks the turn detector "is the
    caller done talking yet?" using only the slice of audio past
    session.processed_until_s (a prior turn's already-consumed audio).
    On yes: slice that utterance out for ASR, hand it to _dispatch_turn as
    a background task (so ingestion of the NEXT turn's audio is never
    blocked by this turn's ASR/LLM/TTS work), and advance the marker.

    ADDED BY SOURAV -- "Caller goes silent" story (Epic: Conversation --
    Difficult, Sensitive and Edge Cases). The IDLE_TIMEOUT_S branch just
    below used to close the call outright on the first silence, with no
    warning (this story's own Repo Evidence: "closes after 90 s with a
    single message and no counting"). It now runs two graduated
    re-engagement prompts first -- see agent/silence_flow.py's module
    docstring for the exact stage machine and why it lives in its own
    pure module rather than inline here a second time (once for main.py,
    once again for main_pcm.py).
    """
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)

        if time.time() - session.last_activity > IDLE_TIMEOUT_S:
            # Two graduated prompts, then a completion-aware close -- see
            # this function's own docstring above and
            # agent/silence_flow.next_silence_action()'s docstring for the
            # stage machine. session.last_activity is reset to time.time()
            # after each prompt is spoken (never after the abandon branch,
            # which ends the call) so every stage gets its own FULL
            # IDLE_TIMEOUT_S window -- the timeout's value and how it is
            # measured are otherwise unchanged.
            action = silence_flow.next_silence_action(session.silence_stage)

            if action == silence_flow.ACTION_PROMPT_1:
                logger.info("[%s] silence: sending prompt 1", session.call_id)
                session.last_silence_prompt = silence_prompt_one()
                await _speak(session, session.last_silence_prompt)
                session.silence_stage = silence_flow.STAGE_PROMPT_1_SENT
                session.last_activity = time.time()
                continue

            if action == silence_flow.ACTION_PROMPT_2:
                logger.info("[%s] silence: sending prompt 2", session.call_id)
                session.last_silence_prompt = silence_prompt_two()
                await _speak(session, session.last_silence_prompt)
                session.silence_stage = silence_flow.STAGE_PROMPT_2_SENT
                session.last_activity = time.time()
                continue

            # ACTION_ABANDON: silent through both prompts. Say plainly what
            # was and was not completed (never inventing a completed
            # action the caller only started -- see
            # silence_flow.has_unfinished_business()'s own docstring for
            # exactly which session state this is allowed to read), log
            # the abandonment exactly once, then close. This `return` is
            # the only exit from this branch, so _turn_poll_loop never
            # ticks again for this call afterward -- there is no path that
            # could log a second call_abandoned event for the same call.
            #
            # ADDED BY SOURAV -- bugfix (validation report Bug #2):
            # next_close_state() replaces a bare has_unfinished_business()
            # call here so a caller who never said a word (utt_seq == 0)
            # is not told "everything you asked has been taken care of" --
            # see that function's own docstring for the three-way split.
            logger.info("[%s] silence: abandoning after both prompts", session.call_id)
            close_state = silence_flow.next_close_state(
                session.utt_seq, session.pending, session.deferred,
            )
            await _speak(session, silence_close_reply(close_state))
            record_call_abandoned(
                turn_index=session.utt_seq,
                preceding_prompt=session.last_silence_prompt,
                call_id=session.call_id,
            )
            with contextlib.suppress(Exception):
                await session.ws.close()
            return

        if time.time() - session.last_heartbeat > HEARTBEAT_INTERVAL_S:
            session.last_heartbeat = time.time()
            with contextlib.suppress(Exception):
                await session.ws.send_text('{"sender":"_ping","text":""}')

        # --- half-duplex gate: never run turn detection on our own voice ---
        if session.agent_speaking:
            if time.time() < session.speak_deadline:
                continue
            logger.warning("[%s] no playback-done from client, releasing gate on deadline",
                           session.call_id)
            session.release_gate()

        if session.resync_pending:
            await _resync_after_playback(session)
            continue

        sr = session.audio.sample_rate
        tail = session.audio.tail_tensor(session.processed_until_s)
        if tail.numel() < int(0.2 * sr):
            continue  # not enough new audio to judge yet -- not an error

        result = await asyncio.to_thread(_turn_detector.poll, tail, sr)
        if result.utterance_end_s is None:
            continue

        # ADDED BY SOURAV -- "Caller goes silent" story. A confirmed
        # utterance end here is the turn detector (VAD) reporting actual
        # caller speech, not merely audio bytes arriving (CallSession.
        # append() already bumps last_activity on every raw chunk
        # regardless of content) and not the background noise/silence the
        # turn detector already filters out before ever returning an
        # utterance_end_s at all. This is therefore the correct point --
        # and the only point in this loop -- to end a silence episode:
        # whichever graduated prompt was most recently sent, the caller
        # has now actually answered it.
        session.silence_stage = silence_flow.STAGE_NORMAL

        absolute_end_s = session.processed_until_s + result.utterance_end_s
        session.utt_seq += 1
        utterance_wav = await _slice_utterance(
            session, session.processed_until_s, absolute_end_s, session.utt_seq,
        )
        session.processed_until_s = absolute_end_s
        # Second layer: the wrapper above handles everything it can while the
        # session is alive, but a failure in the wrapper itself -- or a
        # cancellation -- would still be swallowed by asyncio. Retrieving the
        # exception is what turns "silently discarded" into "in the log".
        task = asyncio.create_task(_dispatch_turn(session, utterance_wav))
        task.add_done_callback(_log_task_failure)


def _log_task_failure(task: asyncio.Task) -> None:
    """Retrieve a background turn's exception so asyncio cannot discard it."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("dispatch task failed after its own handler: %r", exc)


async def _handle_control(session: CallSession, raw: str):
    """Client -> server control channel. Only one message today, but it is
    the load-bearing half of the echo gate: the server cannot otherwise
    know when the caller's speaker actually stopped."""
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
        if rate != SAMPLE_RATE:
            logger.warning("[%s] client capturing at %dHz, not %dHz -- resampling per utterance",
                           session.call_id, rate, SAMPLE_RATE)
        logger.info("[%s] transport: %s @ %dHz", session.call_id,
                    msg.get("format", "pcm_s16le"), rate)


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    session = CallSession(ws)
    logger.info("[%s] call started", session.call_id)
    poll_task = asyncio.create_task(_turn_poll_loop(session))

    try:
        await _speak(session, "নমস্কার, কলকাতা কেয়ার ডায়াগনস্টিকসে স্বাগতম। কীভাবে সাহায্য করতে পারি?")
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
        session.cleanup()
        logger.info("[%s] call ended", session.call_id)


app.mount("/", StaticFiles(directory="static/pcm", html=True), name="static")
