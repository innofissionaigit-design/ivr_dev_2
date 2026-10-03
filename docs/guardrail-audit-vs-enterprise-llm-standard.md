# Audit: this codebase against the "enterprise Python + LLM" standard (2026-09-26)

Checked against the 12 guardrails and the architecture in the reviewer's standard ("the LLM is an untrusted probabilistic
dependency"). Every row below is from the code or a command run today, not from what a document claims. **Verdict: the
safety design is strong; the engineering hygiene around it is not yet production grade.** Both halves are stated plainly.

## Where it is genuinely strong (the part that matters most)

| Standard says | What the code does |
|---|---|
| The LLM never owns money, data, irreversible actions | The model only picks an intent and pulls out words the caller said. Every price, time, doctor, confirmation number is read from the clinic API and put into a fixed template (`agent/reply_templates.py`). The model is never shown a number it could repeat. (CLAUDE.md rule 1.) |
| Never execute model output | No `eval`, `exec`, shell from model text, generated SQL or dynamic imports (grep: only a constant `__import__("torch")` and an ffmpeg subprocess with fixed arguments). |
| Capability-based tools | The model cannot name a tool. The orchestrator maps a validated intent to one fixed function in `agent/tools_client.py` (`get_test_rate(name)`, not `execute_sql(query)`). |
| Two-phase sensitive actions | Booking, cancelling, rescheduling: the bot reads the details back and needs a deterministic yes (`classify_yes_no`, no model). A cancellation charge is stated and confirmed first. |
| Authorization outside the prompt | Personal history is released only after the SERVER verifies the security answers (`patients/verify`); the model never decides who someone is. Proxy access is checked in `booking_service.authorize_disclosure`. |
| Untrusted content is data | The transcript is passed to the extractor as data with "literal spans only"; `tests/test_security_input.py` and `tests/test_orchestrator_security.py` cover injection and identity attacks. |
| Deterministic fallbacks | Fast path (no model) for routine questions, fixed apology and fallback audio when the model or TTS is down, deadline-bounded retries (`INTENT_BUDGET_S`), idempotent writes (new: `Idempotency-Key` plus natural de-duplication). |
| Validate the model's output | `agent/llm.py:_validate` rejects an unknown intent or malformed slots and retries within a deadline; a rejected reply is an apology, never a guess. |
| Test depth | 1,933 tests pass (run today in 8 parallel shards in about 5 minutes). |

## Guardrail by guardrail (1-12)

| # | Guardrail | Status | Evidence |
|---|---|---|---|
| 1 | Strict typing (mypy/pyright strict in CI) | **Not met** | Type hints are used throughout, but there is no `pyproject.toml`, no mypy/pyright config, and neither tool is installed. `scripts/gate.sh` calls `mypy` without strict mode and skips it when missing. Most internal data (intents, slots, the clinic API's answers) travels as plain `dict`. |
| 2 | Validate every boundary into typed schemas | **Partly** | The clinic API validates every request with pydantic (checked by a test that reads the OpenAPI schema; fixed today: `resend`, one mutable default). But the LLM's output and the API answers the agent reads are validated by hand-written `dict` checks, not schema objects. |
| 3 | Structured LLM output only | **Partly** | JSON mode plus a strict validator and reject/retry, but the schema is code in `_validate`, not a declared model. |
| 4 | Never execute model output | **Met** | See above. |
| 5 | Capability tools, least privilege, re-authorize | **Partly** | Capability-based: yes. Least privilege: no. One shared service token opens every clinic-API route; there are no per-tool scopes. |
| 6 | Authorization outside the prompt / RAG ACLs | **Met, not exercised** | Identity is server-side. There is no RAG or vector store, so tenant-isolated retrieval does not apply today (single tenant). The semantic cache stores intents only. |
| 7 | Separate instructions from untrusted content | **Partly** | Data/instruction split and injection tests exist; no dedicated adversarial suite or eval set is run as a release gate. |
| 8 | Resource budgets | **Partly** | Per-call wall-clock budget, attempt cap, per-attempt timeout clamp, prompt/reply length limits, admission control on concurrent calls. Missing: per-user/tenant rate limits and a cost budget (self-hosted model, so cost is GPU time, not tokens). |
| 9 | Fallbacks, timeouts, breakers, backoff | **Partly** | Timeouts and bounded retries: yes. No circuit breaker and no exponential backoff with jitter on the model call. Non-idempotent writes are now safe to retry on the server side; the agent client does not retry them. |
| 10 | Two-phase sensitive actions | **Met** | See above. |
| 11 | Version prompts and models; regression evals; canary/rollback | **Partly / not met** | Versioned: disclosure text, editable messages, the score model (`implicit-v1`), gazetteer/cue data. Not versioned: the extraction prompt (inline template) and model id are constants. The golden-set regression suites named in `pytest.ini` are still "not yet implemented" (Epic E24). No canary; deployment is a tar over ssh to a pod. |
| 12 | Everything goes through CI | **Not met** | There is no CI pipeline in the repo (no `.github`). `scripts/gate.sh --full` exists but does not pass today (below). No lockfile (`pylock.toml`), `uvicorn[standard]` is unpinned, no dependency audit, SBOM or provenance. |

## What `scripts/gate.sh` would say today (measured)

- `ruff format --check`: **209 of 210 files would be reformatted.** The repository has never been ruff-formatted.
- `ruff check`: **508 findings.** Largest groups: 101 "redefined while unused" (mostly pytest fixtures imported by name, a known false-positive pattern),
  77 unsorted imports, 57 calls in default arguments (FastAPI `Depends(...)`, the accepted idiom), 38 unused imports, 27 `date.today()` without a timezone.
- `check_boundaries.py`: **3 false positives**: the word "OpenAI" appears in a comment in `clinic-api/main.py` and in two test files ("an external OpenAI review").
  Not a vendor SDK. I have not changed tests to satisfy a check, so this needs an owner decision (fix the checker to ignore comments).
- mypy, pyright, Hypothesis (property tests), pip-audit, bandit, gitleaks: **not installed / not run**.

Files written this session (13 new modules) were brought to ruff-clean and ruff-formatted today. The older 200-odd files were not
touched: reformatting them all is one large, mechanical diff that should be its own reviewed change.

## Architecture against the standard

- **One LLM gateway:** partly. There is one module (`agent/llm.py`) and no vendor SDK anywhere (Ollama over HTTP), but no `providers/` layer
  even though CLAUDE.md names one, and no central budget/redaction/tracing wrapper.
- **Clean layering:** `agent/` is pure and does not import the orchestrator (enforced by `scripts/check_boundaries.py`). But `main.py` is a ~3,400-line
  orchestrator that mixes transport, turn logic and flows; `main_pcm.py` is a generated copy of it. That is the largest reuse and readability debt.
- **Observability:** per-turn stage timing, call ids in every log line, per-call records with events, and now a per-call implicit happiness score.
  No OpenTelemetry traces or metrics export; no test that logs stay free of personal data beyond the security suite.
- **Property-based tests:** none (Hypothesis is not used); the scoring, matching and parsing code would benefit most.

## Recommended order (smallest safe steps first)

1. `pyproject.toml` with ruff and mypy config; install mypy, Hypothesis, pip-audit, gitleaks; make the checker ignore the word "OpenAI" in comments.
2. One reviewed mechanical commit: `ruff format` + safe `ruff check --fix` over the old files. Then turn the gate on in CI (GitHub Actions) and pin dependencies with a lockfile.
3. Replace the hand-written LLM and API-answer `dict` validation with pydantic models (`LLMDecision`-style), starting with `agent/llm.py:_validate` and `agent/tools_client.py`.
4. Scope the service token per route group; add circuit breaker and jittered backoff around the model call; put the extraction prompt and model id under a version string that is logged per call.
5. Build the golden-set regression and adversarial suites the backlog already names (Epic E24) as release gates; add Hypothesis tests for the scorer, gazetteer and slot merging.
6. Split `main.py` into a transport layer and the turn/flow modules; then retire the generated `main_pcm.py` copy.
7. OpenTelemetry traces; a canary/rollback path for the pod deploy.

Honest summary: the design rule that matters most (the model cannot state a fact, move money or touch data) is enforced in code and tests.
Typing, schemas at every boundary, CI, dependency locking, supply-chain controls and file structure are behind the standard, and the list above closes them.


---

## Status after the follow-up work (2026-09-26, later the same day)

| Item from the fix plan | Now |
|---|---|
| `pyproject.toml`, ruff and mypy configured | Done. Ruff rules are a modest set (pyflakes, import order, bugbear, pyupgrade); the only ignores are B008 (FastAPI `Depends`) and B905, plus the test/tool `sys.path` idiom (E402, F811, one-line setups), each with its reason in the file. |
| Boundary checker | Fixed to look at real imports (`import openai`, `from anthropic import ...`, `importlib.import_module("openai")`), not the word in a comment. Tested both ways (`tests/test_check_boundaries.py`). |
| One-time reformat of the old files | Done as its own mechanical commit: `ruff format` and safe `ruff check --fix` over `agent`, `tools`, `clinic-api`, `tests`, `scripts`, then `main.py`. `ruff format --check` and `ruff check` now pass on all 222 files. The full suite (1,965 tests then) passed unchanged in a clean environment. |
| Strict typing | `mypy agent` now reports **zero errors** across all 82 files (it was 86, then 42). The 42 were fixed without changing behaviour: the 11 "might be `None`" spots were narrowed with guards that only differ from the old code in states that could not happen (each has a test), and every changed function is tested side by side against its pre-change version (`tests/test_typing_guards.py`, frozen copies in `tests/_legacy_typing/`). 32 modules also pass the strict flags (`disallow_untyped_defs`, `disallow_any_generics`, ...) and are listed in `pyproject.toml`; the other 49 still need their functions annotated before they can join. CI runs `mypy agent` and blocks on it. |
| Property-based tests | Added (Hypothesis): the call score, the end-of-call and abuse matching, the idempotency fingerprint, the tuple/list scanner, cue removal, and the model-output parser. On its first run Hypothesis found two real defects, both fixed: cue removal was not stable (`how price much`), and a model answer whose `intent` was a list crashed the old validator (unhashable). |
| Pydantic for the model's output | `agent/intent_schema.py`: `IntentExtraction` and `Slots` replace the hand-written dict checks. Every slot is typed and normalised, an invented FAQ topic is nulled, only smalltalk may carry a model-written reply, a second question must be an enquiry intent with slots. The old `_validate` and `_normalize_age` remain as thin wrappers, so nothing else changed. |
| Pydantic for the API's answers | `agent/api_models.py`: every read and write the agent makes is checked where it enters (`found` / `success` / `status` / `matched` / `conflict` / `verified` must be real booleans, prices real numbers, slot lists lists of strings). A malformed answer becomes a tool failure, never a spoken fact. The new tests found a third defect: a non-JSON body (a proxy error page) raised `JSONDecodeError` instead of the tool-failure error. Fixed for every client method. |
| CI | `.github/workflows/ci.yml`: static checks, the suite in 8 shards, and an informational dependency audit. Validated as YAML and every command run locally; **never run on GitHub**, so its first run is its real test. |
| Dependency audit | `pip-audit` (run today): 29 known vulnerabilities in the pins: 28 in `torch==2.1.0`, 1 in `fastapi==0.104.1`; `uvicorn[standard]` is unpinned. Not changed: the speech stack on the pod depends on those versions; upgrading needs a pod test. |

Still open from the plan: scoping the service token per route group, a circuit breaker and jittered backoff around the model call,
a versioned extraction prompt logged per call, the golden-set and adversarial release suites (Epic E24), splitting `main.py`,
OpenTelemetry, a lockfile, and a canary/rollback path for the pod.
