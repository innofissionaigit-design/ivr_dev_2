# Checks to run when the pod is live

Everything below was accepted as "do it when the pod is up". None of it can be measured off-pod; each
item says what to run, what to look at, and what result means what. The first run on a live pod is recorded in "Results" at the end.

## 1. English recogniser: one decoder or two (extra confirmation turn?)

```bash
curl -s -X POST localhost:8003/transcribe -H "Content-Type: application/json" -d '{"wav_path":"/workspace/sample_en.wav"}'
```
Read `decoder_used` in the reply.
- `rnnt` (with a `decoder_agreement` between 0 and 1): both decoders ran. Nothing changes.
- `fastconformer` or `ctc_fallback`: only one decoder produced the text, so every English factual question
  (test price, doctor, preparation, booking lookup) is read back once before it is looked up
  ("Do you mean CBC?"). Decide: accept the extra turn, or give the English service a second decoder.

Then make one English call: "what is the price of a CBC test". If the agent asks "Do you mean ...?" first,
you are on the single-decoder path.

## 2. TTS: the one-at-a-time-per-language ceiling

Run the load test (KCD-236) and log, per language, at 10, 20 and 30 simultaneous calls: time waiting for
the lock, time rendering, p95 time to first audio. Set a number it must stay under (for example p95 wait
under 100 ms at 30 calls). If it is exceeded, run one TTS worker per language or per GPU; do not remove
the lock (the test in `tests/test_tts_concurrency.py` fails without it).

## 3. Service token and the kill switch

1. `ls -l /workspace/.clinic_api_token` exists (mode 600); `CLINIC_API_REQUIRE_TOKEN=1` in `deploy/env.sh`.
2. `curl -i localhost:8080/api/v1/tests/search?name=CBC` must return **401**; with
   `-H "Authorization: Bearer $(cat /workspace/.clinic_api_token)"` it must return **200**.
3. The clinic API startup log must NOT contain "served WITHOUT authentication".
4. The internal ports (8080, 8002, 8003) must not be reachable from outside the pod; only 8100/8101 are.
5. Kill switch: `touch /workspace/.ai_bypass`; the next call must be handed to a person at once
   (`/api/admission` shows `closed_reason: bypass`); remove the file and confirm calls are admitted again.
6. `lookup_booking` by a stated phone number is accepted for the pilot with the kill switch as the
   mitigation. To turn that off later, route booking lookups through the security questions (the same
   `SecurityCheck` flow as history).

## 4. Voice (KCD-511)

```bash
python tools/audition_voices.py --url http://localhost:8002 --out voices/
```
Play the files and choose. Set `TTS_SPEAKER=<name>` in `deploy/env.sh`, restart the TTS service. A person
has to judge how old a voice sounds; nothing can measure it.

## 5. Recognition and audio (needs recordings)

- Endpointing (KCD-047): at least 200 labelled narrowband calls, `python tools/calibrate_endpointing.py`.
  The adaptive pause (`ADAPTIVE_PAUSE`) is reasoned; this replaces its constants with measured ones.
- Attention and noise (KCD-053/055/057): `python tools/conditioning_eval.py CLIPS_DIR --engine en=http://localhost:8003`,
  once with `NEAR_END_ATTENTION=on` and once `off`, per language, at the four levels and noise profiles.
  Keep it on only for a condition where the word error rate drops.
- Echo and barge-in (KCD-051/052): `python tools/echo_eval.py --pairs RECORDINGS_DIR`.

## 6. Messages and sample data on the pod

```bash
python tools/check_messages.py --url http://localhost:8080 --token-file /workspace/.clinic_api_token
python tools/demo_patient_history.py --url http://localhost:8080 --token-file /workspace/.clinic_api_token
```
The demo shows the sample patients through the security questions and history. To stop the sample rows
being created on a database that will hold real patients, set `CLINIC_SEED_SAMPLE_PATIENTS=0` in
`deploy/env.sh` before the first start.

---

## Results: first run on a live pod (2026-09-25)

Pod `pfup5m0iyql2l9` (RunPod EU-RO-1, RTX PRO 4500 Blackwell, 32 GB, 28 vCPU, 62 GB), running the code at
`voicebot/main` 8b67ccf plus the fixes listed under "Found on the pod". Run by Claude over SSH on the direct TCP
port. Synthetic input only; nothing here used a real caller's audio.

| # | Item | Result |
|---|---|---|
| 1 | English recogniser, one decoder or two | `decoder_used = rnnt`, `decoder_agreement = 0.8` on a synthetic English clip, three runs, identical text. Both decoders ran, so English factual questions do **not** take the extra confirmation turn. One clean synthetic clip; a real accented caller may differ, so re-check on recordings. |
| 2 | TTS per-language lock | Confirmed serialised: a single synthesis is about 25-30 ms; ten requests at the same instant give p95 about 0.5-0.6 s, twenty about 0.9-1.0 s, thirty about 1.4-1.5 s, in all three languages, with no errors. That is a **same-instant burst of one short sentence**, the worst case, not 30 calls' steady state. **Not yet decided:** the number this must stay under (the item asks for one). Reasoned only: 28 calls each needing a sentence every 5-10 s is 3-6 syntheses per second against a capacity of roughly 20-30 per second per language, so the lock is not the first limit; a real load test (KCD-236) with multi-clause replies is what settles it. |
| 3.1 | Token file exists | Exists (41 bytes). **Mode is 666, not 600**: the `/workspace` network volume ignores `chmod` (a test file also stayed 666), the same limitation as `chown`. The pod has one user, root, but the token is not confidential from anything else that can read the volume. Mitigation open: keep it off the shared volume if the volume is ever shared. |
| 3.2 | 401 without, 200 with the token | 401 and 200 as required. |
| 3.3 | No "served WITHOUT authentication" in the startup log | Zero occurrences. |
| 3.4 | Internal ports not reachable from outside | 8080, 8002, 8003 and 11434 bind 127.0.0.1 only. 8100 and 8101 bind 0.0.0.0, but RunPod's HTTP proxy routes only 8888 on this pod (the other ports return 404 from outside); the public URL is `deploy/tcp_forward.py 8888 8101`. |
| 3.5 | Kill switch | `touch /workspace/.ai_bypass` gave `open:false, closed_reason:"bypass"`; removing it reopened admission. |
| 3.6 | `lookup_booking` by stated phone, kill switch as the mitigation | Kill switch verified (3.5). The lookup-by-stated-phone acceptance is unchanged; no separate test was run. |
| 4 | Voice audition | Files written to `/workspace/voices/` (bn, hi, en; female and male). **A person still has to listen and choose**; nothing was judged. |
| 5 | Recordings (endpointing, attention/noise, echo) | **Not run**: needs recordings. |
| 6 | Messages and sample data | `check_messages.py`: 9 of 9 pass. `demo_patient_history.py`: all four sample patients pass the security questions, the refused-before-verification read, the wrong-answer attempt, the history answers in three languages and the "cannot find" wording. **The sample patients are now in the pod's persistent `clinic.db`** (`CLINIC_SEED_SAMPLE_PATIENTS` defaults to 1 on an empty registry); set it to 0 before this database ever holds real patients. |
| - | `tests/test_smoke.py` against the live services | 32 passed (after the token change to the test and the fallback audio below). |

### Found on the pod, and what was done

1. **The fast path was silently OFF on `main`.** `main.py` fetched `/api/v1/catalogue` without the service token,
   got a 401, took the error body for a catalogue and logged "fast path ready over 0 catalogue rows"; being
   non-`None` it was never retried, so every turn paid a model call. Fixed: the token is sent, a non-success
   reply or a body that is not a catalogue is refused (so it degrades to "no fast path yet" and is retried).
   Verified on the pod: "fast path ready over 74 catalogue rows".
2. **The first caller after a restart would hit the model's cold start.** Only `bge-m3` was resident. The
   startup warm-up existed but called the extractor with the per-turn 12 s deadline (KCD-465), so on a 47-74 s cold
   load it timed out, the client disconnected and Ollama abandoned the load (both entrypoints logged "intent model
   warmup failed ... 12.0s deadline"). Fixed: a separate `INTENT_WARMUP_DEADLINE_S` (default 300 s) used by
   `_warm_intent_model()`, with a test. Warmed by hand on the pod meanwhile; **the fixed code is verified by test, not
   yet by a cold restart of the pod.**
3. **The fallback audio did not exist** (`static/fallback_audio/*.wav` is git-ignored and recorded by hand per
   `setup_addon.sh`). They are what a caller hears when TTS itself is down. Synthesised once from the live TTS in
   Bengali (the three lines in `setup_addon.sh`) and in Hindi and English; the Hindi and English wording is
   a draft for native review.
4. **`speakable()` rejected every Bengali reply ending in a danda** (U+0964 is inside the Devanagari block it
   counted as the wrong script). Fixed in `agent/lang_select.py` with a test.
5. **`/speakers` listed a stray `"` speaker** from the Bengali checkpoint's own table. Filtered.
6. **`tests/test_smoke.py` sent no token** to a clinic API that now requires one. A marked, deliberate change: the
   four calls carry the token; no assertion was changed.
7. Packaging note: `git archive` on Windows converts line endings (`core.autocrlf=true`) and breaks the shell
   scripts on the pod; use `git -c core.autocrlf=false archive`.

### Still to do on the pod
A cold-restart check that the warm-up now leaves the 7B model resident (item 2); real recordings for item 5; the ear-choice for item 4;
the load-test number for item 2; a real caller's call through the public URL.



---

## Latency, measured on the pod (2026-09-25, second live conversation)

The caller reported "more than a second" and asked for under 500 ms, and for a repeated question to be served from a
cache with no model and no synthesis. p50 turn latency on the pod was 3.5 s (p95 4.0 s). Every turn now logs one line
(`turn timing: prep | lid | asr | intent | reply | tts | send | total`), and recognisers log their own times
(`ASR timings`), so this is measured, not assumed. Measured with `tools`-style synthetic caller speech through the real
WebSocket, so recogniser accuracy on it is not representative; the STAGE TIMES are.

| Stage | Before | Now | What changed |
|---|---|---|---|
| Language ID (CPU) + recognition | 0.19 s, then 0.64 s in sequence (three recognisers) | about 0.12 s and 0.12 s, **overlapping**: total ~0.15-0.25 s | The call's language and English start with language ID; Hindi runs only when language ID points to it |
| Semantic-cache lookup before the model | up to 1.5 s spent waiting, then a miss | 0 s on a miss (the model starts after a 0.15 s head start), instant on a hit | Lookup and model run together |
| Clinic lookup | 8 ms | 8 ms | unchanged (the truth boundary: prices are read live) |
| Speech synthesis of an answer | 20-85 ms | 0 ms for every catalogue answer | The audio of every test price, preparation and FAQ answer is pre-rendered and pinned in the TTS cache; keyed by the exact text, so a changed price is a different clip |
| The intent model (non-routine turns only) | 1.3-1.5 s | 1.3-1.6 s | **Not changed.** |

Server-side, from the start of a turn to the first audio out, on turns the fast path serves (no model): **160 ms**
(a parking question) and **349 ms** (a price question). On a turn that needs the model it is about 1.5-2 s plus the
recognition time; the holding phrase ("একটু দেখছি।") is now spoken after 0.9 s instead of 2.5 s of silence.

What is NOT under 500 ms, and why:
* A turn that reaches the model is bound by the model (1.3-1.6 s for the 7B intent extractor: a long JSON reply). A
  smaller model, a shorter output schema or a constrained decoder could cut that; each changes accuracy and needs
  the golden-set comparison first, so it was not done.
* What the CALLER perceives includes the turn detector's wait after they stop speaking (tail guard 0.3 s plus a 0.2 s
  poll), which is outside the numbers above.
* The first call after a restart pays kernel warm-up (the first Bengali recognition took 0.6 s, later ones 0.12 s).

Should catalogue answers always be kept in cache? The rendered AUDIO: yes, pinned, rebuilt every 10 minutes from live
data, bounded (3,000 clips), cleared of clips for answers that no longer exist. The FACT (a price, a fasting rule): no
-- it is read from the clinic API on every turn, so a price change is spoken at once; only a clip whose exact text is
still current is ever reused. Not pre-recorded by a person: prices change and a recording cannot.


---

## Third live conversation (2026-09-25): filler, thanks, silence, follow-up

Owner's instructions: a holding phrase ("আচ্ছা, বলছি।" in three languages) only after 0.7 s and never otherwise; no "thank you
for telling me" on an answer to the caller's own question, only a bare "ধন্যবাদ।" after they answer a question we asked;
five seconds of silence -> ask whether there is anything else. Also: "do I need to fast?" straight after a price no longer
waits for the model (the fast path takes the recent test as the topic when the question names none).

Pod-only, to check on a call:
* the filler is heard only on a model turn (about 1.5 s) and not on a price/preparation/FAQ or follow-up turn;
* "ধন্যবাদ।" after a name / number / day is given, and nowhere else; the seeded `thanks_ack` rows were moved to the new
  wording at start-up (`GET /api/v1/agent/messages`);
* silence: whether 5 s after the reply is right on a real line (the client's `playback_done` starts the count), whether a
  caller thinking of a phone number is cut in on too early, and that the call really closes after the second silence;
* "না" / "no" to "anything else?" ends the call after the goodbye has played (a 0.3 s margin after the clip's length).
Not fixed, found while testing: "blood sugar preparation" (no "fasting"/"PP" said) is matched to the PP test at 0.88,
because a partial name reaches the commit floor against the longer form; the reply names the test, but it is a wrong-entity
risk for a preparation answer and wants an ambiguity margin between the two best tests.


---

## Code hygiene pass and doctors' whole names (2026-09-25)

* `tools/hygiene_scan.py` (AST scan, run over every `.py` file and asserted clean by
  `tests/test_idempotency_hygiene_names.py`): mutable default arguments, dataclass/pydantic mutable field defaults,
  `x in ("abc")` (a substring test, not a one-element tuple), `("abc")` constants meant as tuples,
  `.startswith([...])` / `isinstance(x, [...])`. One finding in the whole repository (a pydantic `payload: dict = {}`),
  fixed with `Field(default_factory=dict)`.
* Request models: every write endpoint of the clinic API takes a pydantic model. The one exception was
  `POST /bookings/resend` (a bare query parameter); it now takes `ResendRequest` and still accepts `?confirmation_id=`.
* Idempotent writes (`clinic-api/idempotency.py`): a write carrying `Idempotency-Key` runs once and a retry with the same
  key and body gets the stored answer (a different body under the same key is a 422). Applied to 16 write endpoints
  (bookings, cancel, reschedule, tests, SMS, resend, draft, callbacks, out-of-scope, report OTP/deliver, senior mode,
  preferences). Without a key, the endpoints with a natural identity de-duplicate on it: the same SMS to the same number
  for the same booking within 60 s, the same callback, the same out-of-scope question on a call, an OTP re-requested within
  60 s (gets the code already issued). Reads that use POST (resolve, find, verify, search) are deliberately NOT
  idempotency-keyed: `verify` counts attempts. The agent's client now sends a key on its writes.
  Not covered: two truly concurrent requests with one key can both run before either stores (documented in the module);
  a retry with NO key after the 60 s window is a new operation. Not verified on a real flaky line.
* `doctors.full_name` (new column, migration + backfill at start-up, never overwrites an entered value): the whole name of
  each of the 32 doctors, exposed in the catalogue as `full_name`. The seeded given names are FICTIONAL placeholders (the seed
  only held initials); replace them with the real names by editing the column. Spoken replies still use the short name.

---

## Implicit happiness score (2026-09-26)

Every call is scored 0-100 at hang-up from how it went, with no question asked (`agent/call_score.py`, model `implicit-v1`):
re-asks, system failures, swearing, silence prompts, language flips (moved and moved back), talking over the bot, slow replies
(> 1.5 s to first audio), corrections at read-back, a request for a person, a system hand-off, a silence timeout and a booking
left half-done take points off; answers given, a completed task, the caller's thanks and a polite end add them. The score, band
(happy / neutral / unhappy / very_unhappy), the reasons and the counts are stored with the call record
(`call_records.satisfaction_*`, event kind `satisfaction`; no text, no phone number) and reported by
`GET /api/v1/calls/satisfaction/summary[?since=&language=]`: mean, median, band shares, by language, top causes, unscored calls.
Too-short calls and emergencies are not scored.

It is a PROXY from behaviour, not what the caller felt. Every weight is REASONED, not measured. To make it real: ask the
one-question survey on a sample of good calls and refit the weights to those answers; until then read it as a ranking of calls.
Pod-only to check: that the numbers look sane on real calls (a smooth call should land in `happy`, a call with swearing or a
hand-off in `unhappy`), that `slow_replies` is not inflated by the hold-line, and the per-language means.


---

## Intent-model latency: measure first, then switch (2026-09-26)

Model turns cost about 1.3-1.6 s and nobody knew whether that was reading the prompt or writing the answer. Now every model turn logs
one line (`intent model: N prompt tokens read in X ms, M tokens written in Y ms (Z tok/s) ...`, from Ollama's own timings).

Two switches exist, both OFF by default, so nothing changes until they are tried on the pod:
* `INTENT_PROMPT_VARIANT=fast`: asks for a compact answer (only the slots that have a value, not sixteen "null"s) and keeps the language and the date at the
  END of the prompt, so the ~6,600 characters of instructions are identical on every call and Ollama can reuse its cached copy.
* `OLLAMA_NUM_CTX=4096`: a 4,096-token context instead of the model's 32,768 (the prompt is about 2,000 tokens).

Procedure on the pod: read the timing log for a few real calls; run `python tools/intent_ab.py --variants classic,fast --num-ctx 0,4096 --repeats 5`;
switch on `fast` (and the smaller context) ONLY if it is no less accurate. The 28 sentences in that tool are a smoke test, not the golden set.
Not done, and needs that golden set first: a smaller model (Qwen2.5 3B/1.5B) or speculative decoding.
