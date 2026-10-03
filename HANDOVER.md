# HANDOVER — Kolkata Care Voice Agent

**For:** an engineer or agent session picking this up cold and writing code.
**Written:** 2026-09-08, from a session that deployed, tested and audited this system end to end.
**Repo state at handover:** `main` @ `569f0c4`, working tree clean, in sync with `origin`.

> **Update, 2026-09-15 — read this first, everything below is otherwise still accurate.**
> The workload target was pinned down (6,000–7,000 calls/day, ~10h window,
> 20–30 concurrent) and the production shape changed from the two-GPU plan
> implied by Epic E01 to one L4-class GPU with three routed ASR/TTS pairs
> (bn/hi/en) and LID *before* ASR — see `docs/adr/0001-pilot-single-l4-architecture.md`,
> which is now authority order #2 (`CLAUDE.md` §2). `agent/lid.py`,
> `agent/asr_router.py`, `agent/tts_router.py` exist and are unit tested
> (routing logic only — model loading is unverified, no pod). The
> deterministic layer was also widened: `test_prep` and `clinic_faq` are
> new intents that never reach Qwen, same discipline as `test_rate` and
> `doctor_availability` (ADR 0001 §8). `CLAUDE.md`, `.claude/agents/*`, and
> `scripts/gate.sh` were added — read `CLAUDE.md` before writing code. The
> previous pod referenced below is gone; a new one is being provisioned.
> None of this changes §1–§3 or §6's infrastructure traps — they're still
> exactly as true as they were.

Read sections 1–4 before touching anything. They will save you a week.

---

## 1. The one rule you must not break

> **The LLM may decide what the caller wants. It may never decide what is true.**

Every price, date, chamber time and confirmation number is a **template substitution
from a verified system response** — see `agent/reply_templates.py`. The model is never
asked to produce a number, so it cannot hallucinate one into a patient's ear.

This is the single most valuable property of the codebase. Two production bugs were
already reproduced and fixed because of it:

- A garbled fragment once became **"Naloxone"**, a real but wrong drug name, in the
  sibling project. That is why `agent/llm.py` runs strict JSON mode with a validated schema.
- **"Doctor Nobody"** once fuzzy-matched to a real doctor and returned that doctor's real
  schedule. Fixed by matching on surname only — `clinic-api/main.py`, `FUZZY_SURNAME_FLOOR = 0.60`.

If you find yourself letting the model emit a fact, stop and reconsider the design.

Two supporting rules from the Blueprint that generalise it:

- **The one-import rule.** Nothing in `agent/` imports the orchestrator. Keep it. It is the
  seam that makes every component testable and replaceable.
- **No vendor SDK outside `providers/`.** (That directory does not exist yet — see §8.)
  The moment a vendor name appears in `agent/`, the boundary has leaked.

---

## 2. Sources of truth, and which one wins

| Source | What it is | Authority |
|---|---|---|
| **`D:\Kolkata-Care-Voice-Agent-Production-Grade-Solution-Blueprint.docx`** | On-prem hospital platform design, 12 controls, Phases 0–7, Appendices A–K | **Highest. If anything below conflicts with the Blueprint, the Blueprint wins.** |
| `D:\Kolkata-Care-Voice-Agent-Architecture-and-Implementation-Plan.docx` | Earlier architecture plan | Superseded where the Blueprint disagrees |
| `D:\Kolkata-Care-IVR-Gap-Analysis-and-Solutions.docx` | Every backlog story assessed against this exact code, with a solution each | Use as the per-story "how do I start" reference |
| `D:\Kolkata-Care-Voice-Agent-User-Stories-EN-HI-BN.docx` | 153 stories in English / हिन्दी / বাংলা | Requirements |
| `D:\Kolkata-Care-Voice-Agent-User-Stories-CodeMixed.docx` | Code-mixed variants | Requirements |
| `D:\Kolkata-Care-Voice-Agent-Finetuning-Plan-2500-Hours.docx` | Historical-call adaptation programme | **Do not start this yet — see §9** |
| `D:\Kolkata-Care-Voice-Agent-Backlog-Sprints-and-Gates.xlsx` | Sprint plan and gates | Planning |
| `github.com/sayantanIFAI/ivr` | The code | Reality |

**Reconciliation note.** The Blueprint reframes the target from *"a self-hosted voice LLM"*
to *"an on-premise, governed, deterministic contact-centre platform with a bounded,
interchangeable AI layer (OpenAI / Google / Anthropic), that also runs air-gapped."*
The gap analysis and story documents were written before that reframe, so they assume a
self-hosted-only stack. **Their findings about this code are still accurate; their
architectural assumption is not.** Where they say "self-hosted", read "Mode A adapter
behind `providers/`".

---

## 3. What actually exists today — honestly

**~3,950 lines of Python. A working bench prototype. Not a product.**

The gap analysis scored all 153 backlog stories against this code:
**1 built · 39 partial · 113 absent.**

### It works

A caller can open a browser, speak Bengali, and get a spoken Bengali answer about a test
price or a doctor's availability. Verified live end to end during the deploy session.

### It cannot

- **Be reached by telephone.** Browser WebSocket only. No SIP, no number, no PSTN.
- **Tell the truth.** Every rate and doctor is **fictional seed data** — 8 departments,
  32 invented doctors, ~34 invented prices from `clinic-api/seed.py`. No EMR/HIS/LIS.
- **Handle more than one call.** Module-level singletons in one process. No worker pool.
- **Protect anything.** No auth on any endpoint, no OTP, no audit trail, no consent capture.
- **Speak anything but Bengali.**
- **Be interrupted.** No barge-in; the line is half-duplex by design.

### Measured numbers — trust these, they were taken on real hardware

| Metric | Value | Where it comes from |
|---|---|---|
| Turn latency, end of speech → first audio | **≈ 5.2 s** | 1.0 s VAD silence + ~0.25 s poll + ~1.0 s ASR + 1.5 s LLM + 1.5 s TTS |
| …with a `fast_path` hit | ≈ 3.7 s | LLM stage skipped |
| Human conversational turn gap | ~200 ms | For scale: you are 5–25× too slow |
| Intent extraction, warm | 1.5 s | Measured |
| **Intent extraction, cold** | **73 s** | Measured. See §7 — this is a severe defect |
| VRAM, full stack | 9.2 GB / 24 GB | Measured on RTX 3090 |
| Cold boot to serving | ~4 min | Full stack restore |

---

## 4. Run it

### The pod

Development ran on RunPod. **Pods are ephemeral — the ID, IP and SSH port change every
time one is recreated.** Get current values from the RunPod console; do not trust any
hardcoded here.

```bash
# Pattern. Substitute the current IP and port.
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
```

**SSH rules that matter** (violating them wasted hours):

- **Never use `ssh.runpod.io`.** Use the direct TCP endpoint only.
- **Never pass `-t` or `-tt`.** The direct port needs no PTY, and forcing one breaks scripts.
- Always pass `-p <PORT>` explicitly.

### Bring the stack up

```bash
bash /workspace/kolkata-care-voice-agent/deploy/start_all.sh
bash /workspace/kolkata-care-voice-agent/deploy/status.sh   # poll until all UP
```

`start_all.sh` is well written and worth reading before you replace it. Two hard-won
details are encoded in it:

- Every service is launched with `setsid` and `</dev/null` so it survives the SSH channel
  closing. A plain background job dies with the session and silently leaves services down.
- It **kills by port, never by process-name pattern** — because a pattern broad enough to
  match the service also matched the launching SSH command and killed the launcher.

### Ports

| Port | Service |
|---|---|
| 8100 | Voice agent (WebM transport) — **the one to expose** |
| 8101 | Voice agent (raw PCM transport) |
| 8080 | clinic-api |
| 8002 | TTS server |
| 11434 | Ollama |

**Port 8100 must be added to "Expose HTTP ports" in the RunPod console**, otherwise the
proxy URL 404s. Public URL pattern:

```
https://<POD_ID>-8100.proxy.runpod.net
```

### Tests

A smoke suite is at **`tests/test_smoke.py`** (31 tests: pure functions, clinic API, LLM,
TTS, fallback audio, full pipeline, transport + WebSocket). It was written during the
deploy session, passed 31/31 against live services, and **was previously lost because it
only ever lived on a pod.** It is now committed here. Run it on the pod, not locally —
it needs the live services:

```bash
source /workspace/env.sh
cd /workspace/kolkata-care-voice-agent
/workspace/venv/bin/python3 -m pytest tests/test_smoke.py -v
```

Two of those tests are the **anti-hallucination regression guards** described in §1.
Treat them as load-bearing. Every future wrong-answer incident should add a case here.

---

## 5. Code map

| File | Lines | What it is |
|---|---|---|
| `main.py` | 527 | WebSocket orchestrator, WebM transport, turn loop |
| `main_pcm.py` | 542 | Raw PCM variant. No ffmpeg, works on Safari, avoids an O(n²) re-decode. **Prefer this as the base for telephony** |
| `agent/fast_path.py` | 284 | Deterministic intent + entity matcher that bypasses the 7B model. **Read its docstring first** |
| `agent/semantic_cache.py` | 308 | Two-tier cache: exact match, then bge-m3 similarity with an entity guard |
| `agent/bn_normalize.py` | 226 | Bengali number/date/time verbalisation. Without it prices are silent |
| `agent/tts.py` | 164 | TTS client, prewarming, pre-recorded fallbacks |
| `agent/llm.py` | 149 | Ollama JSON-mode extraction. 5 intents. Retries on malformed output |
| `agent/asr.py` | 137 | IndicConformer wrapper, dual CTC/RNNT decode |
| `agent/vad_stream.py` | 127 | Silero VAD turn detection on a live growing buffer |
| `agent/reply_templates.py` | 105 | **The truth boundary.** Replies assembled from API data only |
| `agent/tools_client.py` | 86 | **The integration seam.** Its docstrings are the spec for the real backend |
| `clinic-api/main.py` | 332 | FastAPI: test search, doctor availability, appointments |
| `clinic-api/seed.py` | 197 | **Fictional data.** Delete from every production path |
| `tts_server.py` | 174 | FastPitch + HiFi-GAN with clause-level prosody splitting |

### The finding in `fast_path.py` worth internalising

Measured against bge-m3 on real Bengali clinic questions:

```
same question, reworded / ASR-garbled     cosine 0.78 – 0.82
DIFFERENT test, same sentence frame       cosine 0.7492   <- only 0.03 apart
same entity, character-level              0.696 – 1.000
different entity, character-level         0.231 – 0.381   <- 0.32 apart
```

The embedding separates *which test* by 0.03. Character overlap separates it by 0.32 — an
order of magnitude better — because the embedding is dominated by the sentence frame while
the part that decides which price a patient is quoted is a handful of characters.

**Conclusion, and it is correct: entity identification was never a language-modelling
problem.** `fast_path` abstains between `ENTITY_MATCH_FLOOR = 0.55` and
`COMMIT_FLOOR = 0.72`, and always abstains on booking. **Do not widen it by lowering the
floors** — that trades latency for wrong answers. Abstention is a first-class result here.

---

## 6. Live defects — verified, with locations

All four below were re-verified against `569f0c4` at the time of writing.

| # | Defect | Where | Note |
|---|---|---|---|
| 1 | **Latin text is silently dropped by the Bengali TTS tokenizer** | `agent/tts.py:114-116`, `agent/reply_templates.py:69` | Detected but only warned about. See below — this is the one to fix first |
| 2 | **73-second cold start** | `agent/llm.py` | Mitigated but not closed: `deploy/env.sh` now sets `OLLAMA_KEEP_ALIVE=-1`, so the 5-minute idle unload is gone. A genuine cold boot still pays it. Needs a startup warm-up plus a readiness gate so no call routes to a cold worker |
| 3 | **Default password shipped in a public repo** | `clinic-api/setup_db.sh:13` | `DB_PASS="${DB_PASS:-kcd_app_pw}"`. It is an overridable shell default, not a live credential — and the stack has since moved off Postgres to SQLite, so this path is largely vestigial. Still: remove the default and require the variable |
| 4 | **README/code drift** | `README.md:74,157` vs `agent/vad_stream.py:45` | README says 0.8 s silence threshold; the code default is `silence_confirm_s = 1.0`. The 5.2 s latency figure above uses the code value |

**Two things I checked and did *not* find, contrary to earlier review notes:**

- **Transcripts are not logged.** Every log line in `main.py` / `main_pcm.py` carries
  `call_id` plus metadata — intent name, timing, confidence — and never the caller's words.
  An earlier audit claimed otherwise; it is wrong against this head.
- **`deploy/env.sh` contains no credential.** It deliberately does *not* set `DATABASE_URL`
  and documents why.

### Defect 2, in detail — it is the instructive one

`agent/reply_templates.py` builds: `"… স্যাম্পল: Blood। …"`

`"Blood"` is Latin script. The Bengali TTS tokenizer drops it. **The caller hears a spoken
colon followed by silence.** Confirmed in the TTS logs: `Character 'b' not found in the
vocabulary. Discarding it.`

The codebase **already detects this** — `bn_normalize.unspeakable_spans()` exists precisely
for it, and `tts.py` logs a warning. But it only logs, and synthesises anyway.

**The fix is small:** make an unspeakable span either rewrite from a translation table or
block the reply, instead of warning. The detection is built and correct; only the
enforcement is missing.

---

## 7. Start here — Phase 0

Per the Blueprint: **"nothing else counts until this passes."**

1. **Rotate the leaked credential** and move all secrets into a vault. Add a `gitleaks`
   pre-commit hook and a CI job that fails on any match.
2. **Connect a real source of truth** behind the existing `agent/tools_client.py` contract.
   Keep `clinic-api` as the anti-corruption layer — no hospital field names may leak into
   `agent/`. Delete seeded data from every production path.
3. **Tool gateway** with authentication, authorisation, schema validation, idempotency keys,
   WORM audit logging and a static allowlist. **Identity and PHI checks live in the gateway,
   not in the model.**
4. **Deterministic policy engine** — emergency vs distress state machine, clinical-advice
   boundary in code.
5. **Handoff service** with a context packet.
6. **Kill switch** — all calls to humans within 30 s, no deploy required.
7. **Commit the test suite into CI** (it is now at `tests/test_smoke.py`) and add a case for
   every incident.

**Phase 0 gate:** zero fictional values reach a caller; every failure path grades B or
better; identity and PHI checks are enforced in the gateway rather than the model.

Then Phase 1 (telephony) → 2 (vendor boundary) → 3 (multilingual) → 4 (latency) →
5 (governance) → 6 (fine-tuning) → 7 (scale). **Do not start a phase whose predecessor
gate has not passed.**

### Two cheap wins you can take on day one

Neither is on the critical path, but both are small and visible:

- **Defect 2 above.** Enforce `unspeakable_spans` instead of logging it.
- **First-clause streaming TTS.** `tts_server.py` already computes clause boundaries for
  prosody; nothing streams them. `agent/tts.py` awaits the whole WAV and `main.py` sends one
  frame. Forwarding chunks as they render removes ~1.2 s of the 5.2 s turn for very little work.

---

## 8. The restructure the Blueprint mandates

Current layout is flat. The Blueprint (Part 7) specifies:

```
agent/  voice/  providers/  tools/  policy/  responses/  handoff/
governance/  datasets/  training/  observability/  deploy/  tests/
```

`providers/` is **the only place a vendor name may appear**. `agent/llm.py` moves behind an
adapter in Phase 2.

**Do this migration deliberately, not incidentally.** Moving files while also changing
behaviour makes both impossible to review. One commit that only moves things, then commits
that change things.

---

## 9. Do not start fine-tuning yet

There is a complete 2,500-hour programme in
`D:\Kolkata-Care-Voice-Agent-Finetuning-Plan-2500-Hours.docx`. It is Phase **6**.

The Blueprint is explicit that **the evaluation laboratory must exist first** — otherwise
you cannot prove an improvement, and you will have spent months on an unfalsifiable claim.
Build the locked golden set (Phase 5) before the first fine-tune. The same corpus serves
both, so build it once.

Three findings from that plan worth knowing now, because they change what you commission:

- **Consent is the programme-ending risk.** Recordings captured for quality monitoring are
  not automatically lawful training data under DPDP purpose limitation. Settle it in week one.
- **Do not transcribe all 2,500 hours.** Full verbatim runs ₹1.2–2.25 crore. Pseudo-label
  everything, hand-transcribe a stratified ~300 h gold subset (₹14–27 lakh). **300 hours of
  good labels beats 2,500 hours of noisy ones.**
- **This corpus cannot improve the TTS voice.** 8 kHz noisy multi-speaker telephony is close
  to the worst possible vocoder training data. A better voice needs 5–20 studio hours with
  one voice artist. Nothing in the corpus substitutes for that.

---

## 10. Infrastructure traps that will cost you a day each

These were all hit for real during deployment.

1. **RunPod wipes the container disk on every restart.** Only `/workspace` survives.
   Anything installed with `apt` or `curl | sh` — Postgres, Ollama, ffmpeg — is gone.
2. **`/workspace` is a MooseFS mount that forces `777` and refuses `chown`, so Postgres
   cannot run on it** — it will not start unless its data directory is owned by the
   `postgres` user, a check with no override. Postgres could therefore only live on the
   container disk, which is wiped on restart, so the one component whose job is to remember
   things was the only one that could not survive a reboot. It was lost three times.
   **This is already solved: `clinic-api/db.py` defaults to SQLite on `/workspace`** and its
   docstring explains the whole reasoning, including why journal mode is deliberately left
   at DELETE rather than WAL (WAL needs a shared-memory index file, which is exactly what a
   network filesystem is worst at). `DATABASE_URL` still overrides, so pointing at a real
   Postgres for production is a one-line change. **Do not "fix" this back to Postgres on the
   volume.**
3. **A cached `.deb` set goes stale when RunPod updates the base image.** An offline restore
   that worked last month failed when the image moved from CUDA 13.0 to 13.2 and libc moved
   with it. Keep the network fallback path.
4. **Two different boot scripts exist.** `deploy/start_all.sh` (in this repo, better) and
   `/workspace/pod_scripts/boot_all.sh` (on the pod, written during deployment, handles the
   ollama/postgres restore). Know which one you are running.
5. **Data residency.** Development pods ran in **`EU-CZ-1`, a European datacentre**, while
   processing Bengali patient names, phone numbers and voice. The Blueprint requires
   on-premise in India. Do not stand up another EU pod with real data.

---

## 11. Decisions that need a human, not an agent

Do not guess at these; they change the architecture.

- **Which deployment mode per site** — air-gapped (Mode A) / private endpoint (B) / hybrid (C).
- **Which vendor(s)** to implement behind `providers/`, and whether the self-hosted stack
  stays a first-class candidate in the bake-off. (The Blueprint says it should.)
- **The consent basis** for using historical recordings for training (§9).
- **Which hospital systems** are the systems of record, and who owns those integrations.
- **The telephony carrier and SBC** choice for the DMZ.

---

## 12. State of this handover

- `tests/test_smoke.py` is **newly added and uncommitted**. It is the only executable
  artefact this handover adds. Review it, then commit it.
- This file is likewise uncommitted.
- Nothing else in the working tree was modified.
- No RunPod pod is currently running; the live URLs from earlier sessions are dead.
  `deploy/start_all.sh` restores a stack in about four minutes on a fresh pod.

```bash
git add HANDOVER.md tests/test_smoke.py
git commit -m "Add handover notes and restore the smoke test suite"
```
