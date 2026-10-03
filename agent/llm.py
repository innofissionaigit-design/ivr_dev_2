"""Node 2: intent + slot extraction via Qwen2.5, over Ollama.

Deliberately NOT vLLM, and deliberately NOT the model's native tool-calling
format. Two reasons, both grounded in what's already running on this
account rather than assumed:

1. VRAM. vLLM serving Qwen2.5-7B-Instruct at fp16 needs ~17GB just for the
   model (measured: https://www.spheron.network/tools/gpu-recommender/
   Qwen/Qwen2.5-7B-Instruct/). The already-deployed pipeline runs the same
   model through Ollama as a Q4_K_M GGUF -- 4.68GB on disk -- specifically
   so IndicConformer (~1.2GB VRAM) and this service's TTS model can share
   the same 24GB card comfortably. Introducing vLLM here would mean running
   two different serving stacks for the same model on the same box, for no
   benefit at this call volume.

2. voicerx/extract.py already hardened a JSON-mode + strict-schema pattern
   against a *reproduced* failure: a looser prompt caused the model to
   invent a drug name ("Naloxone") that was never said. Ollama's function
   tool_calls) is comparatively unproven on this stack. The pattern below
   -- classify intent, extract slots as literal spans, then have CODE (not
   the model) fill the reply from the API response -- is the same
   discipline, applied to prices and appointment slots instead of drugs.
   Quoting a wrong price with total confidence is this system's version of
   that bug, so the model is never allowed to state a number on its own;
   see main.py's _compose_reply().
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any

from agent.enquiry_followup import ENQUIRY_INTENTS  # noqa: F401  (re-exported: callers import it from here)
from agent.intent_schema import FAQ_TOPICS, VALID_INTENTS, normalize_age, parse_extraction  # noqa: F401

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
logger = logging.getLogger("llm")

# How the model is asked (agent/llm.py: build_prompt). "classic" is the original prompt, unchanged. "fast" asks for a COMPACT
# answer (only the slots that have a value, not sixteen "null"s) and keeps everything that differs between calls -- the
# language and today's date -- at the END, so the long instructions are the same text on every call and the server can
# reuse its cached copy of them. Default "classic": "fast" is switched on only after tools/intent_ab.py shows on the pod
# that it is no less accurate.
INTENT_PROMPT_VARIANT = os.environ.get("INTENT_PROMPT_VARIANT", "classic")
# The context window Ollama allocates. Unset (0) = the model's own default (32,768 on the pod, far more than a
# ~2,000-token prompt plus a short answer needs); a smaller value such as 4096 makes a smaller KV cache.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "0"))

# VALID_INTENTS, FAQ_TOPICS, SLOT_KEYS and the answer schema live in agent/intent_schema.py


SYSTEM_PROMPT_TEMPLATE = """You are the intent-and-slot extractor for a diagnostic clinic's phone assistant. You will be given ONE caller utterance in {language_name}, transcribed by automatic speech recognition from live phone audio -- it may contain ASR errors, missing punctuation, or code-switched English words written in the caller's own script.

Today's date is {today_iso} ({today_weekday}), Asia/Kolkata.

YOUR ONLY JOB is to classify intent and pull out slots that are LITERALLY present in the utterance. You do NOT know test prices, doctor schedules, test preparation instructions, or clinic FAQ answers -- do not guess or state any of those; that data comes from a separate lookup after you run.

INTENTS (exactly one):
- "test_rate": caller is asking the price/rate of a diagnostic test.
- "doctor_availability": caller is asking whether/when a named doctor is available.
- "book_appointment": caller wants to book a NEW appointment with a doctor.
- "book_test": caller wants to book a lab test (or several), as opposed to just asking its price or prep.
- "reschedule_appointment": caller wants to MOVE an appointment they already have to a different date/time. Fill "confirmation_id" if they state one, otherwise leave it null -- the system looks it up by phone instead.
- "cancel_appointment": caller wants to cancel an appointment or test booking they already have.
- "lookup_booking": caller is asking what they already have booked ("what's my appointment", "do I have anything booked").
- "add_test_booking": caller wants to add a test to a booking they already have, rather than book a new, separate one.
- "resend_confirmation": caller wants the confirmation message sent again (they lost it, didn't get it, etc.).
- "department_query": caller describes a symptom or problem and needs to be routed to the right department, without naming a doctor or department themselves. Fill "symptom_description" with what they said, verbatim.
- "test_prep": caller is asking how to prepare for a test, or for SEVERAL tests in the same turn (fasting, before/after instructions). Several tests: fill "test_names", not "test_name" -- see the slot rule below.
- "clinic_faq": caller is asking a general clinic question with no specific test or doctor -- hours, location, payment methods, insurance, parking, report collection, contact number, or home sample collection. Fill "faq_topic" with exactly one of: {faq_topics}. If the question doesn't clearly match one of those topics, use "unclear" instead of guessing a topic.
- "smalltalk": greeting, thanks, or anything with no clinic-data lookup needed. You MAY write a short, warm reply yourself for this case only, in {language_name}, in that language's own script.
- "unclear": you cannot confidently tell what the caller wants, or the utterance is empty/garbled ASR noise.

SECOND QUESTION IN THE SAME TURN:
A caller sometimes asks two separate, answerable questions in one breath
("what's the CBC rate, and is Dr. Sen available tomorrow?"). When -- and
ONLY when -- the utterance clearly contains a SECOND, DISTINCT question,
fill "secondary_intent" and "secondary_slots" the same way you filled
"intent" and "slots" for the first one. Leave both null for an ordinary
single-question turn; never split one question into two, and never use
this for anything except "test_rate", "doctor_availability", "test_prep",
"clinic_faq" or "department_query" -- if the second thing the caller said
is a booking/reschedule/cancel action or is itself unclear, leave
"secondary_intent" null rather than guessing.

SLOT RULES:
- Only fill a slot if the caller's words support it. Leave it null rather than inferring.
- "date" / "new_date": resolve relative time words (Bengali আজ/কাল/পরশু and "আগামী <weekday>", Hindi आज/कल/परसों and "अगले <weekday>", English today/tomorrow/day after tomorrow/"this <weekday>"/"next <weekday>") to an ISO yyyy-mm-dd using today's date above. If no date is mentioned for an availability/booking request, leave it null -- do not assume "today". If the resolved date is clearly in the past, still return it as stated -- the system rejects and corrects past dates itself; you must never silently roll a date forward.
- "test_name" / "doctor_name": copy the term as the caller said it (in {language_name} script, or English if they said it in English), do not translate or normalize it -- the lookup service handles matching.
- "test_names": for "book_test", or for "test_prep" when the caller names MORE THAN ONE test, a list of every test name mentioned this turn, each copied as the caller said it, same rule as "test_name" (e.g. "sugar ar lipid profile test er age ki korte hobe" -> intent "test_prep", test_names: ["Blood Sugar", "Lipid Profile"]). A single test named for "test_prep" still goes in "test_name", not a one-item "test_names".
- "faq_topic": only for "clinic_faq" -- one of the fixed topic keys above, never free text.
- "phone": the number to send the confirmation to, only if explicitly spoken, digits only.
- "contact_phone": a phone number the caller gives that is DIFFERENT from the number they are calling from, for the confirmation message -- only when they say so explicitly (e.g. "send it to a different number").
- "relationship": only when the caller says who the patient is to them and it is not themselves, e.g. "mother", "son", "wife" -- copy the relationship word as said, in {language_name}. Leave null if they are booking for themselves.
- "patient_age": only if an age is explicitly spoken, as a number.
- "confirmation_id" / "new_time_slot": only if explicitly spoken/known this turn.
- "spelled_letters": if the caller is spelling a name letter by letter (e.g. "R, A, V, I"), a list of the individual letters in order, lowercase. Otherwise null.
- "symptom_description": only for "department_query" -- the caller's own words describing the problem, never your own paraphrase or a diagnosis.
- Never invent a patient name, phone number, confirmation number, or date that was not said.

Output ONLY a single valid JSON object, no other text, in exactly this shape:
{{
  "intent": "test_rate" | "doctor_availability" | "book_appointment" | "book_test" | "reschedule_appointment" | "cancel_appointment" | "lookup_booking" | "add_test_booking" | "resend_confirmation" | "department_query" | "test_prep" | "clinic_faq" | "smalltalk" | "unclear",
  "slots": {{
    "test_name": string or null,
    "test_names": array of strings or null,
    "doctor_name": string or null,
    "date": string or null,
    "time_slot": string or null,
    "new_date": string or null,
    "new_time_slot": string or null,
    "confirmation_id": string or null,
    "patient_name": string or null,
    "patient_age": number or null,
    "phone": string or null,
    "contact_phone": string or null,
    "relationship": string or null,
    "spelled_letters": array of strings or null,
    "symptom_description": string or null,
    "faq_topic": string or null
  }},
  "secondary_intent": "test_rate" | "doctor_availability" | "test_prep" | "clinic_faq" | "department_query" | null,
  "secondary_slots": {{
    "test_name": string or null,
    "doctor_name": string or null,
    "date": string or null,
    "faq_topic": string or null,
    "symptom_description": string or null
  }} or null,
  "direct_reply_bn": string or null
}}

"direct_reply_bn" must be null for every intent except "smalltalk" -- for every other intent, the reply is composed later from real clinic data, not from you."""


_FAST_OUTPUT_BLOCK = """Output ONLY a single valid JSON object, no other text. Include ONLY the fields that have a value: leave out every null slot, and leave out "secondary_intent", "secondary_slots" and "direct_reply_bn" unless they apply. Shape:
{"intent": "<exactly one intent from the list above>", "slots": {"<slot name>": <value>, ...}}
Slot names, and the type of each value: test_name, doctor_name, date, time_slot, new_date, new_time_slot, confirmation_id, patient_name, phone, contact_phone, relationship, symptom_description, faq_topic are strings; patient_age is a number; test_names and spelled_letters are arrays of strings.
For a second question add "secondary_intent" ("test_rate", "doctor_availability", "test_prep", "clinic_faq" or "department_query") and "secondary_slots" (any of test_name, doctor_name, date, faq_topic, symptom_description). "direct_reply_bn" (a string) is only for "smalltalk"; for every other intent the reply is composed later from real clinic data, not from you."""


def _fast_static_prompt() -> str:
    """The instructions with everything that varies between calls taken out (the language, today's date) and the answer
    shape made compact. Built from SYSTEM_PROMPT_TEMPLATE so the rules exist in one place."""
    t = SYSTEM_PROMPT_TEMPLATE
    t = t.replace(
        "ONE caller utterance in {language_name}", "ONE caller utterance in the caller's language (named at the end)"
    )
    t = t.replace("Today's date is {today_iso} ({today_weekday}), Asia/Kolkata.\n\n", "")
    t = t.replace(
        "in {language_name}, in that language's own script", "in the caller's language, in that language's own script"
    )
    t = t.replace("(in {language_name} script, or English", "(in the caller's own script, or English")
    t = t.replace("as said, in {language_name}.", "as said, in the caller's language.")
    a = t.index("Output ONLY a single valid JSON object")
    t = t[:a] + _FAST_OUTPUT_BLOCK
    t = t.replace("{faq_topics}", ", ".join(FAQ_TOPICS))
    assert "{language_name}" not in t and "{today_iso}" not in t, "the fast prompt still has a per-call placeholder"
    return t


_FAST_STATIC = _fast_static_prompt()


def build_prompt(transcript: str, lang: str, now: datetime.datetime, variant: str = "classic") -> str:
    """The prompt for one caller utterance. `classic` is the original text, byte for byte."""
    language = _LANGUAGE_NAMES.get(lang, "Bengali")
    if variant == "fast":
        return (
            f"{_FAST_STATIC}\n\nCALLER LANGUAGE: {language}. Today's date is {now.strftime('%Y-%m-%d')} "
            f"({now.strftime('%A')}), Asia/Kolkata.\n\nCALLER UTTERANCE (ASR output):\n{transcript}\n\nJSON:"
        )
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        today_iso=now.strftime("%Y-%m-%d"),
        today_weekday=now.strftime("%A"),
        faq_topics=", ".join(FAQ_TOPICS),
        language_name=language,
    )
    return f"{system_prompt}\n\nCALLER UTTERANCE ({language}, ASR output):\n{transcript}\n\nJSON:"


class ExtractionError(Exception):
    pass


def _call_ollama(prompt: str, timeout_s: float = 90) -> str:
    # 90s, not 20s: a cold-loaded Qwen2.5:7b (Ollama unloaded it after its
    # default 5-minute idle timeout) measured at 47s just to answer "Say
    # OK" on this pod. The real fix is OLLAMA_KEEP_ALIVE keeping the model
    # resident (see setup docs) so this path is rarely hit in practice --
    # this margin is a backstop for whenever it still is.
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            # Keep Qwen resident. Ollama unloads a model after 5 idle minutes and a
            # cold 7B load measured 74 s on this pod -- the first caller after any
            # quiet spell would wait that long for one intent.
            "keep_alive": -1,
            "options": _model_options(),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    _last_call.stats = summarise_timing(body)
    return body.get("response", "")


_last_call = threading.local()  # the timing of this thread's last model call, read by extract_intent


def _model_options() -> dict:
    options: dict = {"temperature": 0.0}
    if OLLAMA_NUM_CTX > 0:
        options["num_ctx"] = OLLAMA_NUM_CTX
    return options


def summarise_timing(body: dict) -> dict:
    """Where the model's time went, from what Ollama reports (all durations are nanoseconds): reading the prompt (prefill),
    writing the answer (decode), loading the model, and how many tokens each was. Tells us which to shorten."""
    ms = lambda key: round(int(body.get(key) or 0) / 1e6)  # noqa: E731
    out_tokens, decode_ms = int(body.get("eval_count") or 0), ms("eval_duration")
    return {
        "prompt_tokens": int(body.get("prompt_eval_count") or 0),
        "prefill_ms": ms("prompt_eval_duration"),
        "output_tokens": out_tokens,
        "decode_ms": decode_ms,
        "load_ms": ms("load_duration"),
        "total_ms": ms("total_duration"),
        "tokens_per_s": round(out_tokens / (decode_ms / 1000), 1) if decode_ms else None,
    }


# Kept under their old names: tests and callers import them from here.
_normalize_age = normalize_age


def _validate(data: dict) -> tuple[bool, list[str]]:
    """Validate and normalise one model answer IN PLACE (the schema is agent/intent_schema.py). (ok, problems): not ok when
    the answer cannot be used at all (an unknown intent, or `slots` that is not an object); a slot the model left out is a
    problem worth reporting but not a reason to reject it."""
    model, errors = parse_extraction(data)
    if model is None:
        return False, errors
    data.update(model.to_dict())
    return True, errors


_LANGUAGE_NAMES = {"bn": "Bengali", "hi": "Hindi", "en": "English"}

# KCD-465: "a deadline budget rather than a retry count bounds every
# stage... no path can leave the caller waiting past the total budget."
# REASONED, not measured against real telephony: agent/asr.py's own
# module docstring measures a cold Qwen load at ~74s, and this session's
# earlier work margined _call_ollama's per-attempt timeout at 90s for
# exactly that case -- but 90s (let alone up to 3 attempts of it) is
# nowhere near a bound a caller on the line should ever wait behind.
# DEFAULT_DEADLINE_S is the whole-operation ceiling every retry shares;
# a caller hits the spoken "llm_failure" apology at a predictable time
# instead of an unbounded one, exactly the acceptance criterion's own
# wording. Recalibrate against real telephony latency once measured.
DEFAULT_DEADLINE_S = 12.0
_MAX_ATTEMPTS_BACKSTOP = 5  # defence in depth only -- see extract_intent's docstring


def extract_intent(
    transcript_bn: str, max_retries: int = 2, lang: str = "bn", deadline_s: float = DEFAULT_DEADLINE_S
) -> tuple[dict, dict]:
    """Returns (parsed JSON dict, diagnostics dict).

    `transcript_bn` keeps its historical name for callers; it is the caller's
    utterance in `lang`. Only the prompt's language wording changes -- the
    schema, the slot rules and the never-state-a-fact rule are identical
    for every language.

    `max_retries` is kept only as a defensive attempt-count backstop
    (capped at _MAX_ATTEMPTS_BACKSTOP regardless of its value) against a
    pathological case where every attempt somehow returns instantly --
    `deadline_s` is what actually bounds how long this function may run,
    and each attempt's own network timeout is clamped to whatever of the
    deadline remains, so one slow attempt cannot by itself consume the
    whole budget meant to cover retries too."""
    variant = INTENT_PROMPT_VARIANT
    prompt = build_prompt(transcript_bn, lang, datetime.datetime.now(), variant)

    diagnostics: dict[str, Any] = {"attempts": 0, "total_time_s": 0.0, "errors": [], "deadline_s": deadline_s}
    last_error: Exception | ExtractionError = ExtractionError("no attempt was made")
    start = time.time()
    max_attempts = min(max_retries + 2, _MAX_ATTEMPTS_BACKSTOP)

    for attempt in range(1, max_attempts + 1):
        remaining = deadline_s - (time.time() - start)
        if remaining <= 0:
            break
        diagnostics["attempts"] = attempt
        t0 = time.time()
        try:
            _last_call.stats = None
            raw = _call_ollama(prompt, timeout_s=remaining)
            diagnostics["total_time_s"] += time.time() - t0
            timing = getattr(_last_call, "stats", None)
            if timing:
                diagnostics["model_timing"] = timing
                logger.info(
                    "intent model: %s prompt tokens read in %d ms, %s tokens written in %d ms (%s tok/s), total %d ms [%s, ctx %s]",
                    timing["prompt_tokens"],
                    timing["prefill_ms"],
                    timing["output_tokens"],
                    timing["decode_ms"],
                    timing["tokens_per_s"],
                    timing["total_ms"],
                    variant,
                    OLLAMA_NUM_CTX or "default",
                )
            model, errors = parse_extraction(json.loads(raw), slots_optional=variant == "fast")
            if model is None:
                raise ValueError(f"schema validation failed: {errors}")
            return model.to_dict(), diagnostics
        except Exception as e:  # noqa: BLE001 - retry on anything, log it
            diagnostics["total_time_s"] += time.time() - t0
            last_error = e
            diagnostics["errors"].append(f"attempt {attempt}: {type(e).__name__}: {e}")

    raise ExtractionError(
        f"intent extraction did not complete within its {deadline_s}s deadline "
        f"({diagnostics['attempts']} attempt(s), last error: {last_error})"
    )
