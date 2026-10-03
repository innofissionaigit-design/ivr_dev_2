# ADR 0001 — Pilot production architecture: one L4-class GPU, three routed ASR/TTS pairs, LID before ASR

**Status:** Accepted for the pilot. Supersedes the multi-GPU sizing implied by
`D:\Kolkata-Care-Voice-Agent-Architecture-and-Implementation-Plan.docx` and by
`Kolkata-Care-Voice-Agent-Backlog-Sprints-and-Gates.xlsx` epic E01's "two on-prem
GPUs" framing, **for the pilot workload only**. Does not change the
`Production-Grade Solution Blueprint`'s governance, truth-boundary or
deployment-mode (A/B/C) design — see [Reconciliation](#reconciliation-with-the-blueprint).

**Date:** 2026-09-15
**Driver:** actual near-term workload — **6,000–7,000 calls/day, a ~10-hour
operating window, ~20–30 simultaneous calls** — is far below the 100-concurrent
peak the earlier design was sized for.

Cost figures from the source discussion are intentionally omitted here; this
ADR is architecture and capacity only. Cost planning lives wherever the
business side tracks it, not in the repo.

---

## 1. Decision

Build the pilot around **one NVIDIA L4-class 24 GB GPU**, not six GPUs and not
three ASR GPUs plus three TTS GPUs. All three ASR checkpoints and all three
TTS checkpoints stay **resident** on that one GPU simultaneously; for any given
utterance, only the selected ASR and the selected TTS actually run. VAD and
language identification run on CPU. A second GPU is a burst/peak option, not a
standing commitment — bring it up only if the load test in §7 proves the
single GPU cannot hold the latency target at 30 concurrent calls.

```text
                    PSTN / Mobile caller
                            |
                     Carrier SIP trunk
                            |
                  +-------------------+
                  | SBC pair          |
                  | FreeSWITCH /      |
                  | Kamailio+rtpengine|
                  | G.711 8 kHz       |
                  +---------+---------+
                            |
                     8k -> 16k ONCE
                            |
              +-------------v-------------+
              | VAD + AUDIO LANGUAGE ID   |
              | CPU only                  |
              | output: bn / hi / en      |
              +-------------+-------------+
                            |
              +-------------v--------------+
              |         ASR ROUTER          |
              |                             |
              | bn -> IndicConformer Bengali|
              | hi -> IndicConformer Hindi  |
              | en -> FastConformer English |
              +-------------+--------------+
                            |
                     transcript + lang
                            |
             transliteration / aliases / normalise
                            |
              +-------------v--------------+
              | Deterministic fast path     |
              | + exact cache               |
              +-------+-------------+-------+
                      | hit         | miss/booking
                      |             v
                      |       Qwen2.5-3B Q4
                      |       intent + slots only
                      |             |
                      +------+------+
                             v
                      Clinic/HIS API
                     LIVE factual lookup
                             |
                   template(intent, lang)
                             |
                    verbalize(text, lang)
                             |
              +--------------v-------------+
              |          TTS ROUTER         |
              | bn -> Bengali FastPitch/HFG |
              | hi -> Hindi FastPitch/HFG   |
              | en -> English FastPitch/HFG |
              +--------------+-------------+
                             |
                     16/22k -> G.711 8k
                             |
                            SBC
                             |
                           caller

         Low confidence / failure / user asks person
                             |
                             v
                    Human contact centre
```

This is the architecture the codebase should be built toward from here.
`agent/lid.py`, `agent/asr_router.py` and `agent/tts_router.py` (added
alongside this ADR) are the first pieces of it — see §8.

---

## 2. Language identification moves before ASR

The Architecture & Implementation Plan's Section 10 identified language from
the ASR transcript. That only works with a single multilingual ASR. This
pilot routes **audio -> one of three ASRs**, so the language decision has to
happen **before** ASR runs — there is no transcript yet to read.

- First implementation to try: **SpeechBrain VoxLingua107 ECAPA-TDNN**,
  restricted to bn/hi/en, 16 kHz mono, ~86 MB, CPU-only.
- Start LID on the first useful chunk of voiced audio, not at end-of-turn —
  by the time VAD confirms the turn is complete, the routing decision should
  already be sitting there. LID should contribute close to nothing to
  end-of-turn latency.
- Language is **per utterance, never per call** (matrix-language requirement,
  unchanged from Section 10): a caller may use Bengali grammar with an
  English test name, switch to Hindi next turn, then read a phone number in
  English.
- Do **not** routinely run all three ASRs on low-confidence turns — that
  destroys the routing's cost/latency advantage. Order of preference:
  1. previous turn's language as a prior,
  2. if still ambiguous, run the **top two** ASRs for that one turn,
  3. after repeated ambiguity in the same call, hand off to a human.
- Confidence thresholds are **reasoned, not measured** until calibrated
  against real Kolkata G.711 call recordings. Do not trust the default in
  `agent/lid.py` (`DEFAULT_CONFIDENCE_FLOOR = 0.55`) past the pilot without
  that calibration — same discipline the rest of this codebase already
  applies (see `agent/fast_path.py`'s docstring on measured-vs-reasoned).

Implemented in `agent/lid.py` as two halves: `LanguageIdentifier` (the
model boundary — **unverified**, no pod was available to exercise it) and
`ASRLanguageRouter` (the routing policy above — **verified**, unit tested
without a GPU in `tests/test_lid.py`, passing today). Do not conflate the
two when reviewing or extending this module.

---

## 3. Exact model selection

| Language | Model | Why |
|---|---|---|
| Bengali | `ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large` | Already in production (`agent/asr.py`). 120M-parameter hybrid CTC/RNNT Conformer, MIT-licensed. |
| Hindi | `ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large` | Same 120M Conformer-Large architecture and hybrid CTC/RNNT decoding as the Bengali checkpoint — same AI4Bharat NeMo fork, same known workarounds apply (see `agent/asr.py`'s module docstring). |
| English | `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi` | ~114M parameters, hybrid Transducer/CTC, cache-aware streaming. Architecturally close to the existing ASR rather than a jump to Whisper. |

Rejected: AI4Bharat's newer 600M-parameter multilingual IndicConformer (still
has no English), and Whisper Large (the production addendum already
identifies concurrent-stream capacity, not accuracy, as the binding
constraint — a small Conformer-class model fits that better than a large
general-purpose one).

**Decoder-work saving to apply once agreement statistics exist:** the
Bengali ASR currently runs both CTC and RNNT on every utterance
(`agent/asr.py`, `_transcribe_clip`) even though `decoder_agreement` is
computed and never consumed downstream. Keep both during the pilot to
collect agreement data; once there is enough, switch production to
**RNNT first, accept on good confidence, CTC only on uncertain cases** —
same models, less GPU work per turn. Do not make this change until the
agreement data exists to justify the confidence threshold.

| Language | TTS |
|---|---|
| Bengali | Existing AI4Bharat Indic-TTS FastPitch + HiFi-GAN (unchanged) |
| Hindi | AI4Bharat Indic-TTS Hindi FastPitch + HiFi-GAN |
| English | NVIDIA FastPitch + NVIDIA HiFi-GAN |

Rejected for the pilot: Indic Parler-TTS (0.9B-class; the final signal is
G.711 8 kHz narrowband, so spending materially more GPU for studio-quality
audio that gets downsampled away has little return here).

Keep the exact-text WAV cache aggressive: pre-render greetings, apologies,
transfer messages and other static prompts in all three languages, and cache
the **rendered reply**, not the underlying fact — the cache key changes
automatically whenever a database field does, because the rendered text
changes. Continue toward first-clause streaming (`tts_server.py` already
computes clause boundaries; nothing streams them yet — tracked separately,
not part of this ADR).

---

## 4. Qwen2.5-3B, and where it runs — the decision the user asked us to make

**Decision: ASR and TTS on the GPU. Qwen2.5-3B Q4 starts on CPU, promoted
to GPU only if a load test shows CPU can't hold the latency budget.**

**Bring-up sequencing update (2026-09-15):** the previous pod is
unrecoverable; a new pod is being created from scratch. For the *initial
bring-up only*, all three components — ASR, TTS, and Qwen — go on the one
GPU together. This does not reverse the decision above; it defers the
CPU-Qwen step rather than skipping it:

1. **Bring-up:** everything on the GPU (six speech models + Qwen2.5-3B
   AWQ via vLLM, or Qwen via Ollama if vLLM setup isn't ready yet).
   Simplest possible path to a working end-to-end pilot on a fresh pod —
   one fewer moving part while `agent/lid.py`, `agent/asr_router.py` and
   `agent/tts_router.py` get their first live verification (§8).
2. **Measure**, once the stack is up: Qwen Tier-3 p95 GPU-resident, ASR/TTS
   latency, fast-path serve rate (now higher — see §8, the FAQ/prep
   extension).
3. **Only then** decide whether moving Qwen to CPU is worth doing, using
   the same load test in §7. There is no cost pressure forcing that move
   this pilot (cost is explicitly out of scope for this ADR); it remains
   available as a lever if GPU contention between ASR/TTS and Qwen ever
   shows up under load, not as a default.

This is a firm recommendation, not the reverse pairing (ASR/TTS on CPU,
Qwen on GPU) that was raised as an open question. Reasoning:

- ASR and TTS sit directly in the per-turn critical path for **every**
  turn, and their footprint is small: two ~120M Conformer ASR models, one
  ~114M Conformer ASR model, and three FastPitch+HiFi-GAN pairs are a few
  GB combined, nowhere near the 24 GB budget. Running six small models on
  CPU to save that little VRAM would blow the latency budget on every
  single turn, for a workload where turn latency is already the thing
  most under pressure (see `HANDOVER.md` §3 — 5.2 s today, target ≤1.0–1.2 s
  p95).
- Qwen2.5-3B Q4 is invoked far less often — only on a fast-path miss
  (bookings, ambiguous dates, genuinely open extraction), and the backlog's
  own target is a 65–75% fast-path serve rate. At 20–30 concurrent calls
  that is roughly 5–10 simultaneous Tier-3 requests, which a modern
  16+ vCPU instance running llama.cpp/Ollama can plausibly hold within
  budget for a 3B Q4 model producing a short JSON response.
- Keeping Qwen off the GPU by default keeps the GPU's load profile flat
  and predictable (ASR + TTS only), which makes capacity planning for the
  one-GPU baseline tractable. A bursty, booking-heavy period that pushes
  Tier-3 load up does not compete with ASR/TTS for VRAM or scheduling.

**The test that decides, not a guess:** run the Phase-1-style load test
(30 concurrent calls, sustained one hour, real G.711 recordings — see §7)
with Qwen on CPU and measure Tier-3 p95 against the latency budget
(`docs adr §5` / Blueprint Appendix B). If it passes with headroom, stop —
CPU is the answer. If it fails, move the same Qwen2.5-3B AWQ build onto the
L4 behind vLLM (continuous batching + PagedAttention), sharing the GPU with
ASR/TTS rather than adding a second card for this alone. Only reach for a
second GPU if neither placement holds the target under real load.

Whichever placement wins, the truth boundary is unchanged and
non-negotiable: **Qwen identifies intent and extracts slots; it never
originates a price, appointment time or clinical fact.** The clinic/HIS API
remains the only source of truth (`agent/reply_templates.py`).

---

## 5. Concurrency sizing (Little's Law, no cost)

| Average AI call duration | 6,000/day | 6,500/day | 7,000/day |
|---:|---:|---:|---:|
| 2.0 min | 20.0 | 21.7 | 23.3 |
| 2.5 min | 25.0 | 27.1 | 29.2 |
| 3.0 min | 30.0 | 32.5 | 35.0 |
| 4.0 min | 40.0 | 43.3 | 46.7 |

"Peak is 20–30 calls" is internally consistent only if average handle time
stays near 2–2.5 minutes and traffic is not very spiky. If AHT drifts to 3
minutes at 7,000 calls/day, the **average** concurrency is already 35; peak
is higher than that.

**Sizing decision:** AI admission cap around **28–30** concurrent calls;
telephony layer sized nearer **40–50** concurrent sessions. Above the AI
cap: burst GPU if warmed, otherwise straight to the human queue — never an
unbounded queue against the GPU. This is the same admission-control model
the Blueprint already specifies as the primary protection against GPU
overload (Appendix A/B), applied at this workload's actual numbers.

---

## 6. Container/process layout

Two isolated NeMo installs are required regardless of GPU count — the
Bengali/Hindi IndicConformer checkpoints need **AI4Bharat's NeMo fork**;
the English FastConformer checkpoint needs **mainline NVIDIA NeMo**; mixing
both in one Python environment is the exact class of bug `agent/asr.py`'s
docstring already documents for the Bengali case. Isolate by service, not
by giant shared environment:

```text
indic-asr-service       Bengali + Hindi / pinned AI4Bharat NeMo fork
english-asr-service     English / pinned NVIDIA NeMo

indic-tts-service       Bengali + Hindi
english-tts-service     English

qwen-service            CPU initially (Ollama); GPU only if the load test requires it
orchestrator            CPU — main_pcm.py, agent/{lid,asr_router,tts_router,fast_path,...}
```

**Deviation from the source discussion, and why:** it suggested Docker or
Podman containers for this isolation. This repo's actual RunPod pattern —
proven on the sibling `clinical-emr-adapter` project's pilot pod — is a
**native stack with no Docker**: separate Python virtualenvs per service,
supervised by a single restart-proof bootstrap script, in the same spirit as
this repo's own `deploy/start_all.sh` (launches every service with `setsid`
and `</dev/null`, kills by port never by process-name pattern). A RunPod pod
is already a container; nesting Docker inside it adds real operational risk
(base-image Docker support is inconsistent across RunPod templates, and this
account has already lost time once to a stale offline cache after a base
image moved — `HANDOVER.md` §10.3) for no isolation benefit a virtualenv
does not already give. **Use separate virtualenvs per service, not
containers**, unless a concrete dependency conflict proves a venv
insufficient.

Avoid Kubernetes at this scale, per the source discussion — agreed, for the
same reason: one pod, five services, a supervisor script is legible; a
scheduler is not proportionate.

---

## 7. What the load test must measure before any second GPU is bought

Sustain 30 concurrent calls for one hour, then a 40-call overload, then a
simulated GPU failure — using **real 8 kHz G.711 recordings**, not clean
microphone audio (resampling 8 kHz to 16 kHz makes the tensor shape
compatible; it does not restore frequency content the phone network already
removed). Measure and record:

- per-language WER, doctor/test entity accuracy, LID confusion matrix
- intent accuracy, p50/p95/p99 turn latency, first-audio time
- GPU VRAM headroom, queue depth, fast-path serve rate, human-overflow rate
- **Qwen Tier-3 p95 on CPU specifically** — the number that decides §4

Multilingual gate (unchanged from the Blueprint): **≥90% intent/entity
accuracy in every language-mixture bucket**, not only in aggregate.

Engineering latency targets for this pilot (targets, not yet measured
claims — do not present the current prototype's 5.2 s as the production
expectation):

| Stage | Target |
|---|---:|
| Turn endpoint after speech | 300–600 ms |
| Audio LID | hidden during caller speech |
| ASR finalisation | ~150–500 ms |
| Fast-path intent | negligible |
| Qwen Tier-3 | benchmark-dependent; minority of turns |
| Clinic/HIS API | <100 ms ideally |
| Cached TTS | effectively immediate |
| New TTS, first clause | ~150–400 ms |
| **p95 first audible reply** | **≤1.0–1.2 s** |
| p99 | preferably <2 s |

---

## 8. What this ADR actually changes in the repository

Added alongside this ADR, all unit-tested without a GPU or a pod
(`tests/test_lid.py`, `tests/test_routers.py` — 16 tests, passing):

- `agent/lid.py` — `LIDResult`, `LanguageIdentifier` (model boundary,
  **unverified**), `SpeechBrainVoxLingua107LID` (**unverified** stub with
  the right shape), `ASRLanguageRouter` (**verified** routing policy: §2).
- `agent/asr_router.py` — `ASRRouter`, `PILOT_ASR_MODELS` (the §3 table as
  data), dependency-injected engines, **verified** routing logic.
- `agent/tts_router.py` — `TTSRouter`, `PILOT_TTS_MODELS`, same pattern.
- `pytest.ini` — registers the four CI Sanity Gate markers
  (degradation/latency/security/hallucination) so they exist as a target
  even before Epic E24 builds real suites under them.

**Not done in this pass, and why:** loading the actual Hindi/English ASR
and TTS checkpoints, wiring Ollama/vLLM for Qwen placement, and the
SpeechBrain LID model itself all require the pod — none of it can be
honestly verified without live GPU/model access, and no pod was running
when this ADR was written (see `HANDOVER.md` §12). These are the first
concrete stories for Epic E04 (Call Intelligence), E12 (Multilingual) and
E11 (Synthesis) once a pod is available — see the RunPod readiness note
delivered alongside this ADR.

**Added 2026-09-15 — the semantic-cache/fast-path FAQ and prep extension.**
Before backlog execution starts, the deterministic layer was widened so
two more common categories never reach Qwen: test preparation
(`test_prep`) and a fixed set of clinic FAQ topics (`clinic_faq` — hours,
location, payment methods, insurance, parking, report collection, contact
number, home collection). Same discipline as `test_rate` and
`doctor_availability` throughout — fast_path decides the entity/topic
locally, the actual answer is still fetched live from clinic-api on every
turn (never cached as a value), and a semantic-cache hit on any of these
intents is still entity-guarded (`agent/semantic_cache.py`'s
`_ENTITY_SLOTS` now includes `faq_topic`). Concretely:

- `clinic-api/models.py` — `LabTest.fasting_required` / `.prep_instructions_bn`; new `FAQ` table.
- `clinic-api/seed.py` — prep text for 8 tests (rest get a generic default) + 8 FAQ topics, fictional, same convention as the existing seed data.
- `clinic-api/main.py` — `GET /api/v1/tests/prep`, `GET /api/v1/faq`; `/api/v1/catalogue` now also returns `faq_topics` (topic + keyword set only — the answer text is still fetched live via `/api/v1/faq`, mirroring test_rate/doctor_availability's cached-routing-never-cached-value rule).
- `agent/tools_client.py` — `get_test_prep()`, `get_faq()`.
- `agent/fast_path.py` — `Catalogue` generalised to also hold `faq_topics`; `_PREP_CUES`; `FAQ_COMMIT_FLOOR` (reasoned, not measured — no real FAQ-question call data exists yet); two new `resolve()` branches.
- `agent/reply_templates.py`, `agent/llm.py` (Tier-3 fallback: `test_prep`/`clinic_faq` intents, `faq_topic` slot, `FAQ_TOPICS` enum), `main.py`/`main_pcm.py` dispatch (regenerated via `tools/make_pcm_variant.py`, verified byte-identical on the shared half).

**Verified**, without a pod: 33 tests passing locally —
`tests/test_fast_path_faq_prep.py` (pure routing logic, including a
regression guard that a `clinic_faq` match can never steal a
`doctor_availability` question) and `tests/test_clinic_api_new_endpoints.py`
(clinic-api's new endpoints exercised end-to-end against a throwaway local
SQLite file via FastAPI's `TestClient` — genuinely running code, not
inferred behaviour). **Not verified:** Qwen actually classifying
`test_prep`/`clinic_faq` from a real ASR transcript — that needs the pod
and a live Ollama/Qwen instance.

---

## Reconciliation with the Blueprint

This ADR only revises **capacity and placement** for the pilot workload. It
does not change:

- the truth boundary (the model never originates a fact — unchanged);
- the deployment-mode framing (Mode A/B/C) — this pilot is a **self-hosted,
  no-vendor-API call path**, i.e. the Mode A shape, just hosted on rented
  GPU infrastructure rather than hospital-owned metal. Read every "on-prem"
  statement in the Blueprint's Part 3 as "Mode A, hosted" for the duration
  of the pilot, and revisit data residency (`HANDOVER.md` §10.5 — the EU-CZ-1
  mistake) before any real patient data touches this pod;
- the governance model (§4.2/4.12) — a second GPU or a Qwen placement change
  is exactly the kind of candidate change the release-gate process governs,
  not an exemption from it.

Epic E01's "two on-prem GPUs, two sites" framing should be read as the
**eventual production target**, not the pilot's starting point. Do not
close E01 stories against the two-GPU assumption without first confirming
against this ADR which stories are pilot-scoped and which are
production-scoped — that reconciliation is tracked as a follow-up, not done
in this pass.
