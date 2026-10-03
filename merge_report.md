# Merge report — `merged_code 1` → `multilingual_ivr`

**Date:** 2026-10-01
**Target:** `C:\Users\Interviewer\Desktop\multilingual_IVR\multilingual_ivr`
**Foundation:** `voicebot-main`, copied byte-identical (300 files, MD5-verified before any edit). Nothing removed, nothing rewritten — additions only.
**Features ported from:** `merged_code 1\merged_code\merged_code`
**Governing brief:** `Kolkata-Care-Voice-Agent-Trilingual-Upgrade-Prompt-No-Git-Access.md`

---

## 1. Headline

| | Baseline (untouched `voicebot-main`) | After this merge |
|---|---|---|
| tests passed | 2,418 | **3,118** |
| tests failed | 17 | **0** |
| errors | 259 | **63 — every one `No module named 'nemo'`** |

The 63 remaining errors are the ASR library, which was deliberately **not**
installed locally as instructed. Nothing else fails.

| | |
|---|---|
| Features integrated and **live** (wired into both orchestrators) | **8** |
| Features integrated as **tested code, not wired** | **1** (`answer_ledger`) |
| `merged_code 1` modules **already covered** by `voicebot-main` | **21** |
| Deliberately not ported, with reasons | **3** |
| New tests, all passing | **487** |
| Pre-existing defects found and fixed | **1** (it was blocking 16 test files) |
| Bugs my own port introduced, caught by tests, fixed | **3** |

---

## 2. Why the merge runs in this direction

The upgrade prompt describes turning a Bengali-only bot trilingual. **That work
is already done in `voicebot-main`** — it has `agent/lid.py`, `asr_router.py`,
`tts_router.py`, `english_asr_server.py`, `tts_server.py`,
`reply_templates_i18n.py`, and the supplied architecture diagram describes that
repository. `merged_code 1` is the opposite shape: Bengali-only (28 intents, a
7,985-line `main.py`) but far richer in caller-facing behaviour.

So `voicebot-main` is the foundation and `merged_code 1`'s **behaviour** is what
moves across. The prompt is the acceptance checklist each ported feature must
satisfy: §1's truth boundary, §6.5's "thread the detected language through",
§6.6's speakability check, §7's per-language templates.

### Owner's decisions

| Question | Decision |
|---|---|
| Scope | All three phases |
| Hindi wording | Ships **enabled**, not behind an off-by-default toggle |
| New intents | **Guards only** — `agent/intent_schema.py` and the LLM prompt untouched, so Qwen's accuracy on the existing 14 intents cannot regress |
| Orchestrator | Wire it, following `voicebot-main`'s own trilingual guard pattern |
| Test depth | After the first three modules: lighter tests, more modules |
| Automation `.sh` files | Not wanted — none were ported (see §8) |

---

## 3. The method, and why a name-by-name comparison was not enough

A function-level inventory said 910 of `merged_code 1`'s 1,192 names were absent
from the merge. That number is **misleading**, and acting on it would have
duplicated working code. It mixes four very different things:

| Group | Names | Meaning |
|---|---|---|
| Ported, renamed to house style | ~282 | `_EN_ANGER` → `_EN`. Feature works. |
| **Already in `voicebot-main` under other names** | ~600 | `slot_parse`(104)→`intent_schema`; `reply_templates`(160)→its own trilingual pair; `main.py`(107)→its own orchestrator; `outcomes`(32)→`outcome_metrics`; … |
| Genuinely missing | ~250 | narrowed to **two real features** once checked against the code (§3.2) |
| Automation, excluded | 53 | `vast_push.py`(16), `gate-report.py`(32), `listening_stimuli.py`(5) |

So every candidate was checked **against the running code**, not the file list.

### 3.1 Twenty-one modules were already implemented here

| `merged_code 1` module | Already here as |
|---|---|
| `correction_flow.py` | `booking_flow.reopen_for_correction` + `correction_acknowledgement` |
| `silence_flow.py` | `main.py`'s `silence_prompts` / `SILENCE_PROMPT_S` path |
| `symptom_routing.py` | the `department_query` intent + `enquiry_followup.py` |
| `human_fast_path.py` | `human_request.asks_for_a_person` (already trilingual) |
| `speakability.py` | `pronunciation.py`, `speech_norm.py`, `lang_select.py` |
| `confidence.py` | `confidence_gate.py` + `action_gate.py` |
| `slot_parse.py` | `intent_schema.parse_extraction` (pydantic-typed) |
| `outcomes.py` | `outcome_metrics.py` + `call_record.py` |
| `tool_contract.py` | `tools_client.validated()` + `api_models.py` |
| `tool_outcome.py` | same |
| `state.py` | `call_state.py` |
| `turn_parts.py` | `clause_split.py` |
| `turn_log.py` | `call_record.add(kind, payload)` |
| `reference_data_cache.py` | `fast_path.py`'s catalogue cache |
| `match_band.py` | the `ambiguous` / `did_you_mean` / `needs_confirmation` banding in `clinic-api/main.py` |
| `report_flow.py`, `report_access_control.py`, `otp_messaging_config.py`, `report_delivery_config.py` | `eq.request_report_otp()` + `eq.deliver_report(confirmation_id, otp_code)`, `POST /api/v1/reports/request-otp`, `POST /api/v1/reports/deliver` |
| `charging_for_cancellation.py` | `booking_service.cancellation_charge()` with `CancellationPolicy`, `refund_eligible`, `free_window_hours`, `charge_percent` |
| `company_config.py` | `voicebot-main`'s own config for its own report/notification design |
| `responses/speech_policy.py` | `agent/speech_policy.py` |

Reports + OTP, cancellation charges, billing, walk-in, prescription policy and
packages are all confirmed working: **194 clinic-api endpoint tests pass against
the seeded database.**

### 3.2 The single most valuable discovery

`voicebot-main` already contained a **complete, trilingual `"angry"`
caller-state policy** that **nothing ever set**:

- `agent/speech_policy.py` `_ROWS["angry"]` — `escalation_threshold="low"`,
  `speech_rate="slow-normal"`, `offer_human=True`, `acknowledge_first=True`
- `agent/acknowledgement.py` `_ACK["angry"]` — written in **bn, hi and en**
- `agent/call_state.py` `VALID_CALLER_STATES` — lists `"angry"`

`grep` for `"angry"` returned those three definition sites **and nothing else**.
The same is true of `"distressed"`. An entire feature, in three languages, was
unreachable.

That turned the anger port from "copy 220 lines of flow logic" into "supply the
detector and set the state" — smaller, safer, reusing six already-tested
mechanisms, and **the Hindi wording for that path did not have to be invented**.

---

## 4. What was integrated

### 4.1 `agent/anger.py` — caller anger ✅ wired

Nothing detected plain anger: `abuse.py` catches swearing only,
`human_request.py` needs an explicit request for a person. "This is ridiculous,
I am fed up" fell through to the model.

Only the four phrase lists were ported; all seven acceptance criteria are met by
the existing machinery in §3.2. **New here:** a Devanagari list.

Wired inside `_update_caller_state()`, two lines before
`select_acknowledgement()`. Sticky for the call, like `senior` — a caller who
has said they are angry is not un-angered by one calm turn — while the
acknowledgement is still spoken once via the existing `acknowledged_state`
guard. `anger_turns` added to `call_score.py` at `(-10, -30)`: less harsh than
`abuse_turns`, because anger is usually the clinic's own earlier failure being
reported, not the caller mistreating the agent.

### 4.2 `agent/complaint.py` — complaints ✅ wired

`voicebot-main` has no `complaint` intent, so an undetected complaint is not an
error — it is classified `smalltalk`, `clinic_faq` or `unclear` and answered as
a cheerful enquiry. The caller is simply not heard, silently.

| File | Addition |
|---|---|
| `clinic-api/models.py` | `ComplaintRecord` (`complaints` table) |
| `clinic-api/enquiry_service.py` | `file_complaint()`, deduped by `(call_id, text)` |
| `clinic-api/main.py` | `POST /api/v1/complaints` |
| `agent/tools_client.py` | `submit_complaint()` |
| `agent/phrases.py` | `complaint_acknowledged` in bn/hi/en |
| `agent/call_score.py` | `complaint_filed` flag at `-15` |

**No migration needed** — `create_all()` at startup creates a brand-new table;
only new *columns* need the `*_migrate.py` ALTER path. The test fixture proves
this rather than leaving it as a claim.

**Where the verbatim text goes.** Criterion 3 wants the caller's own words kept;
this repository forbids caller free text in `logs/` and the call record
(`call_record.py` keeps counts, flags and ids, "never any text"). Both hold
because the verbatim complaint goes to **one narrow-access store** — the
`complaints` table — while the ledger and call record keep carrying only the
reason code. `CallbackRequest.call_summary` already set that precedent here.

### 4.3 `agent/clinical_safety.py` — "is my result dangerous?" ✅ wired

**The original's stated reason does not apply here.** It argued the danger is the
model inventing clinical advice via `smalltalk`. `voicebot-main` already passes
that text through `_speakable()`, `persona_clean()` and
`mentions_personal_history()`, and `agent/persona.py` already blocks reassurance
("don't worry", "nothing serious") **in all three languages**.

**The real gap, measured against this code:** "is my result dangerous?" names no
test, doctor or date, so there is nothing to look up, and `agent/emergency.py`
does not fire — correctly; verified for seven such utterances across the three
languages. The turn resolves to `smalltalk` or `unclear`, so **a frightened
caller gets a generic greeting or is asked to repeat themselves.** That is the
"procedural limit that feels like a rebuff" the story exists to prevent.

All four detection categories ported (danger/panic, normal-vs-abnormal,
roleplay/hypothetical, forced binary), plus the gap-tolerant bypass regexes and
the lab-parameter-next-to-a-number rule. **New here:** Devanagari lists, a
Devanagari bypass regex, Devanagari clinical terms with Indic-digit matching.
`_bn_bounded` was reimplemented locally as `_bounded(word, text, char_class)`
using the `(?<![ঀ-৿])` idiom `agent/persona.py` already uses, rather than
porting `slot_parse.py` (1,474 lines) for a two-line helper.

Carries its own `awaiting_clinician_offer` flag rather than reusing
`awaiting_handoff_offer`, because that one is answered by
`_continue_history_flow()` and hands off as `history_requested_staff` — a
clinical question filed as a history request goes to the wrong desk. The flag is
also added to the `slot_answer=` exemption list so a bare "yes" is not rejected
as a jumbled transcript.

### 4.4 `agent/doctor_personal_request.py` — asking for a doctor ✅ wired

`human_request.py`'s noun list is the **front desk** and contains neither
"doctor" nor "clinician", so "connect me to a doctor" matched nothing. The two
need different answers: the front desk can be reached, a clinician cannot — so
the reply declines the direct connection and offers the appointment that is the
real route. A doctor's name is **never captured**: the reply never repeats one,
so there is nothing a mis-extraction could disclose.

### 4.5 `agent/unverifiable_claim.py` — a claim never becomes a fact ✅ wired

**Only half the original was ported, deliberately.** It intercepted four claim
categories and answered all four "I cannot confirm that", justifying the
appointment case explicitly: *"no GET endpoint anywhere in clinic-api for
reading back an existing appointment by phone"*.

**That is false here.** This repository has `GET /api/v1/bookings/lookup` (with a
`lookup_booking` intent already reaching it) and
`GET /api/v1/billing/outstanding`. Porting those two faithfully would answer "I
cannot confirm that" to a caller whose appointment or balance the system can look
up — **violating the story's own second criterion** ("the agent checks the system
of record where verification is possible") and losing a working capability.

| Category | Here |
|---|---|
| `DOCTOR_APPROVAL` | **intercepted** — nothing in this schema can ever check it |
| `REPORT_RESULT` | **intercepted** — answered with the clinical reply |
| `APPOINTMENT_EXISTING` | **falls through** to `GET /api/v1/bookings/lookup` |
| `PAYMENT_ALREADY` | **falls through** to `GET /api/v1/billing/outstanding` |

The excluded two are named in the module's own `VERIFIABLE_HERE` dict, so the
decision is discoverable from the code. `_REAL_REQUEST_MARKERS` is the narrowing
rule: an utterance that also asks for a real check falls through regardless of
what it asserts.

### 4.6 `agent/callback_request.py` + `callback_config.py` — "call me back" ✅ wired

The durable half already existed — `callback_requests` and
`_tools.request_callback()`, used on hand-off when a number is known. What was
missing was noticing that the caller **asked**.

**The detector is new code, not a port.** In `merged_code 1` this was an **LLM
intent**, and "guards only" rules out adding one, so all four phrase lists were
written here.

**`check_callback_availability()` is deliberately NOT ported.** It needs the
clinic's hours as structured `{weekday: {closed, open, close}}`. **No such data
exists here:** `DepartmentHours` stores `hours_bn`/`hours_hi`/`hours_en` as
free-text strings, and clinic-wide hours are a FAQ answer string. Reading a time
window out of prose to promise a callback would be **inferring a fact from
text**, which CLAUDE.md rule 1 forbids. So **no line promises a time.**

Wiring: detect → if disabled, say so honestly; if a number is known, queue it;
otherwise ask for one and read the next turn for digits
(`agent/spoken_codes.parse_phone`).

### 4.7 `agent/doctor_list.py` — "which doctors do you have" ✅ wired

A plural, unnamed ask. `doctor_availability` needs a NAMED doctor and a caller
asking for the list cannot name one; `department_query` takes a SYMPTOM and
cannot answer "what departments do you have". There was no intent **and no
endpoint**.

| File | Addition |
|---|---|
| `clinic-api/main.py` | `GET /api/v1/departments`, `GET /api/v1/doctors`, `GET /api/v1/doctors/by-department`, `_doctor_listing()` |
| `agent/tools_client.py` | `list_departments()`, `list_doctors()`, `doctors_by_department()` |
| `agent/reply_templates.py` + `_i18n.py` | `doctor_list_reply`, `doctors_by_department_reply`, `_spoken_doctor`, `MAX_DOCTORS_SPOKEN` |

**No new tables** — `Doctor` and `Department` already carry trilingual names
(`full_name_bn`, `full_name_hi`, `aliases_bn`, `aliases_hi`).

**Why this is ~150 lines and not the original's 423.** The original carried its
own department-name extractor: plural detection, reduplicated question words,
Bengali case endings, fuzzy department scoring with its own floor and margin, a
`_NOT_A_DEPARTMENT` stop list — all to pull a department name out of the
sentence *before* asking the API, i.e. a text matcher in `agent/` guessing at a
name it had no list to check itself against.

None of it is needed. The endpoint matches the query against the real department
table **in both directions**, so the whole utterance is handed over and the
clinic's own data decides: "which doctors are in cardiology" resolves on the
name inside the sentence, "cardio" on being inside the name. Several matches
come back `ambiguous`, a near match as `did_you_mean`, and neither resolves
silently — the "Doctor Nobody" rule applied to departments.

With 32 doctors seeded, reading every name down a phone line is a recital nobody
can hold in their head, so the full-list reply **names the departments** and asks
which; a named department lists its own four.

### 4.8 `agent/compare_flow.py` — comparing two options ✅ wired

`clinic-api` already compares a package against buying its tests separately.
Nothing compared two arbitrary things — and worse, there was an active misfire:
`fast_path._ambiguous_test()` sees two tests fitting almost equally well and
abstains by design, so a comparison went to the model, which has no compare
intent, and became `test_rate` for **one** of the two. The caller asked about two
things and was told the price of one. This guard runs **before** the fast path.

Both names are resolved with the **same matcher and floor the fast path uses**,
the second via `skip_name` so it cannot be the first again. If only one resolves
the caller is asked for both — comparing against a guessed second item would
state a difference never computed from real data.

**Money is `Decimal`, never `float`.** This is the first place in the repository
that does arithmetic on a live price. `float("450.00") == float("450.0")` eats a
digit that matters, and binary floating point gives `0.3 - 0.1 =
0.19999999999999998`, which would be spoken with total confidence. Both cases are
asserted in the tests.

**No recommendation, structurally.** `build_comparison()` has no "better" field,
because neither catalogue exposes a clinical or suitability axis — a
recommendation would be invented rather than retrieved. A caller who asks "which
is better" is answered with both prices and the difference, asserted in all three
languages.

### 4.9 `agent/answer_ledger.py` — a repeat must not contradict ⚠️ NOT WIRED

A caller told 450 then 500 in one call hears no acknowledgement; from their seat a
live update and a system contradicting itself sound identical. Not a reply cache
— every repeat still does its own live lookup, as `semantic_cache.py` requires.

**Ported and unit-tested, but nothing calls it** — stated in its own docstring as
well as here. The place to call `check()` is `_answer_enquiry_intent()`, which
takes `(intent, slots, lang)` and returns a string: no session, so it cannot
reach the per-call ledger, and the `result` it needs never leaves it. Giving it
either changes a signature used at **12 call sites across the two orchestrators
and faked in 9 test files**, six of them pre-existing. That refactor deserves its
own review.

### 4.10 Deliberately not ported

| Module | Why |
|---|---|
| `date_calc.py` | Would be a genuine truth-boundary win — `llm.py`'s prompt still asks the 7B model to *"resolve relative time words … to an ISO yyyy-mm-dd"*. But its design needs the model to name an **expression** (`date_expr`), a **prompt change the owner excluded**; used instead as a silent corrector it would alter the most heavily tested flow (booking) for no test-visible gain. And the fast path is already correct per language: `fast_path_cues.py` resolves relative days in code, and Hindi's table is deliberately minimal (`{"आज": 0}`) because both कल and परसों are genuinely ambiguous in Hindi. That restraint is right and must not be "fixed". |
| `deploy/*vast*`, `scripts/gate-report.py`, `tools/listening_stimuli.py` | Automation, not wanted. This deployment is RunPod (`docs/adr/0001`). |
| `clinic-api/migrations.py` | An appointment-status migration for a schema this repository does not have; it has its own `booking_migrate.py` / `enquiry_migrate.py`. |

### 4.11 Guard order

```
emergency                  (existing) — outranks everything
abuse                      (existing) — swearing has its own calm boundary
transcript checks          (existing)
clinical_safety            (NEW)  — health question before service question
complaint                  (NEW)  — records verbatim AND routes
doctor_personal_request    (NEW)  — a doctor, not the front desk
asks_for_a_person          (existing)
callback_request           (NEW)  — generic callback, after the doctor one
compare_flow               (NEW)  — before the fast path, which abstains on two tests
doctor_list                (NEW)
unverifiable_claim         (NEW)  — last: every guard above is more specific
… _update_caller_state → anger    (NEW)
```

Verified identical in `main.py` and `main_pcm.py` by importing both through
`tests/_pod_stubs.py` and comparing each guard's position in the real
`_dispatch_turn` source.

Two orderings are load-bearing and asserted in tests:

- **clinical before complaint** — the order the original tested. "I want to file
  a complaint, is my result dangerous" takes the clinical path.
- **complaint before `asks_for_a_person`** — a **deliberate divergence** from
  `merged_code 1`. Exactly one utterance collides: "I want to speak to someone
  about a complaint". Under the original order it is routed as a bare
  person-request and **the complaint is never recorded** — the call still looks
  fine, which is why this is worth changing.

---

## 5. The pre-existing defect that was fixed

### `@idempotent` broke FastAPI's annotation resolution — 16 test files unblocked

`clinic-api/idempotency.py`'s decorator wraps an endpoint in a function defined
in *that* module, so `wrapper.__globals__` belongs to it. `clinic-api/main.py`
uses `from __future__ import annotations`, so every annotation there is a plain
string. FastAPI resolved `"BookingRequest"` against the wrong namespace and
raised `PydanticUndefinedAnnotation` **at import**: the app could not be
imported at all.

**Impact before the fix:** 16 test files / ~259 errors could not run — including
`test_every_write_endpoint_with_a_body_declares_a_schema`, the test that audits
a new endpoint's schema — and `scripts/gate.sh` could not run either.

It was fixed because it blocked verification of this merge's own new endpoints.

**The fix, and the wrong turn on the way to it.** Setting
`wrapper.__annotations__` to resolved types was tried first and **does not
work**: `functools.wraps` sets `__wrapped__`, and `inspect.signature()` follows
that to the original function, whose annotations are still the raw strings. The
fix is an explicit `__signature__` built from `typing.get_type_hints(fn)`, which
resolves against `main.py`'s own globals and takes precedence over
`__wrapped__`. `include_extras=True` keeps `Query(...)`/`Depends(...)` intact,
and an unresolvable hint leaves the signature untouched, so the change can only
improve on the previous behaviour.

**Result:** the app imports (62 routes), and **363 previously-unrunnable tests
now pass** — including the migration tests (`test_enquiry_migration`,
`test_i18n_migration`, `test_patient_context_service`) that had been written off
as needing Postgres. They were the same bug.

This is also why `POST /api/v1/complaints` is **not** `@idempotent`: the defect
was not worked around, it was simply not re-created, and dedup is done in the
service layer as `request_callback()` already does.

---

## 6. Bugs this port introduced, caught by tests, fixed

All three were caught by the repository's own suites, not by inspection.

### 6.1 My clinical_safety guard broke add-test booking *(live flow)*

`tests/test_lay_term_and_payment_flow.py` failed: "please add ESR to
confirmation KCD-1" fired the lab-value rule, because `ESR` is a clinical term
and `KCD-1` supplied a digit inside the 25-character window. A caller adding a
test to a booking was answered "a doctor will explain your result" — a working
flow broken by a guard meant to sit quietly in front of it.

Fixed with two defences **scoped to the number heuristic only**, so the
adversarial phrase lists are untouched: a digit inside an alphanumeric
identifier is not a measurement, and an explicit booking cue means the figure
belongs to a reference, a date or a time. Both directions are now tested — the
booking turns are quiet, and real lab values still fire in all three languages.

### 6.2 My doctor_list regex suppressed 8 of 15 genuine requests — all non-English

A "was a doctor named?" regex matched a doctor title followed by any word. In
Bengali and Hindi the words after a title are ordinary grammar — "ডাক্তার আছেন"
("are there doctors"), "डॉक्टर हैं" — so it read almost every genuine list
request in those languages as naming a doctor and silently dropped it.
**Precisely the trilingual failure this merge exists to avoid.**

The regex was deleted, not patched: the phrase lists already pair a plural or
interrogative sense with a doctor word, so a named-doctor question matches none
of them — verified across all three scripts, including
"ডাক্তার সেনের কাছে অ্যাপয়েন্টমেন্ট" and "डॉक्टर सेन के साथ अपॉइंटमेंट".

### 6.3 Two ratchet tests broken by new wording — neither ratchet was touched

`CLAUDE.md` §4.4 forbids editing a test or threshold to go green.

- **`test_the_long_sentence_backlog_is_exactly_its_recorded_size`** — my Bengali
  complaint line was 18 words against a cap of 15, the Hindi 22 against 18. Both
  sentences were split.
- **`test_reply_templates_do_not_exceed_the_known_artefact_backlog`** — my
  ambiguous-department line used a colon, which reads badly in TTS (copied from
  an existing template that is itself one of the known artefacts). Reworded to a
  comma in both Bengali and Hindi.

A third check found a bug in the repository, not in my text:
`agent/apology.py` counted **two** apologies in a Bengali line containing one,
because its marker `সরি` is matched as a bare substring and also matches inside
the ordinary adverb **সরাসরি** ("directly"). My wording avoids the collision;
the matcher bug is reported in §9 and raised as a separate task, because
`enforce_single_apology()` uses the same markers to *remove* a second apology
and can therefore strip a real one.

---

## 7. Test results

### 7.1 New tests — 487, all passing

| File | Tests |
|---|---|
| `tests/test_anger.py` | 36 |
| `tests/test_orchestrator_anger.py` | 11 |
| `tests/test_complaint.py` | 46 |
| `tests/test_orchestrator_complaint.py` | 15 |
| `tests/test_clinical_safety.py` | 103 |
| `tests/test_orchestrator_clinical_safety.py` | 15 |
| `tests/test_doctor_personal_request.py` | 38 |
| `tests/test_unverifiable_claim.py` | 43 |
| `tests/test_callback_request.py` | 55 |
| `tests/test_answer_ledger.py` | 19 |
| `tests/test_doctor_list.py` | 59 |
| `tests/test_compare_flow.py` | 47 |

```bash
python -m pytest tests/test_anger.py tests/test_complaint.py tests/test_clinical_safety.py \
  tests/test_doctor_personal_request.py tests/test_unverifiable_claim.py \
  tests/test_callback_request.py tests/test_answer_ledger.py tests/test_doctor_list.py \
  tests/test_compare_flow.py tests/test_orchestrator_anger.py \
  tests/test_orchestrator_complaint.py tests/test_orchestrator_clinical_safety.py -q
# 487 passed
```

### 7.2 Why the first three features also have orchestrator tests

Before the port the `"angry"` policy was perfect, fully trilingual, and
**completely dead**, because nothing set it. A port that added a perfect
detector and wired it to nothing would reproduce that bug — **and every unit
test would still pass.**

So those features have a second file driving the **real `_dispatch_turn`** from
`main_pcm.py` via the repository's own off-pod harness (`tests/_pod_stubs.py`,
which stubs torch/NeMo/SpeechBrain at import). Those tests assert on what the
fake WebSocket actually received — the sentence a real caller would hear. Each
also carries a
`test_an_ordinary_question_is_completely_unaffected_by_this_port` case.

From feature 4 onward the owner chose lighter tests for broader coverage, so
those have unit tests plus a static verification that the guard is reached in
the real `_dispatch_turn` source, in both orchestrators. `doctor_list`'s
endpoints are additionally tested against a **real seeded SQLite database**.

### 7.3 Whole suite

Run in two halves, `tests/test_smoke.py` excluded (it needs a live clinic-api,
agent and TTS — a pod suite, not a unit suite).

| | Baseline | After |
|---|---|---|
| passed | 2,418 | **3,118** (1,773 + 1,345) |
| failed | 17 | **0** |
| errors | 259 | **63** (61 + 2) |

Every one of the 63 is `ModuleNotFoundError: No module named 'nemo'` — the ASR
library, deliberately not installed. No models were installed locally.

### 7.4 Regression — proven against a pristine copy

This is not a git repository, so `voicebot-main` was copied untouched to a
scratch directory and the **identical pytest command** run against both.

| Suite | Pristine | After the merge |
|---|---|---|
| 14 files covering everything touched | **470 passed, 15 errors** | **470 passed, 15 errors** |
| The 3 files that failed at baseline | **6 failed, 38 passed** | **identical at the time; all now pass** after §5 |

---

## 8. Hindi: what was written, and how far it is trustworthy

Hindi replies are **enabled**, per the owner's decision.

| Path | Hindi source | Reviewed? |
|---|---|---|
| anger acknowledgement + offer of a person | **already in the repo** (`_ACK["angry"]`) | pre-existing |
| anger detection (22 phrases) | written here | **no** |
| complaint detection (23) + acknowledgement | written here | **no** |
| clinical-safety detection (4 categories) + reply | written here | **no** |
| doctor-personal-request detection (14) + reply | written here | **no** |
| unverifiable-claim detection + decline | written here | **no** |
| callback detection (13) + 3 lines | written here | **no** |
| doctor-list detection (14) + 2 replies | written here | **no** |
| compare detection (13) + reply | written here | **no** |

Every drafted line passes the repository's own automated wording checks —
`agent/persona.py` (register, no hedging, no reassurance, sentence length),
`agent/apology.py`, `tests/test_spoken_text_lint.py`'s artefact ratchet, and the
invariant that bn/hi/en carry identical phrase keys (now **40 each**, up from
32). Those checks are **not** a substitute for a native speaker, and every
module docstring says so in the same "REASONED, not measured" language the rest
of the repository uses.

**The honest limit, as upgrade-prompt §8.5 requires:** the code paths exist and
are tested offline. **No Hindi or Bengali audio has been through them.** The
adversarial suite the original closed its clinical-safety gaps against — a
hundred prompts per language — was never run in Hindi, because it never had a
Hindi list.

### One known limitation, pre-existing and not introduced here

Department names are stored only as `Department.name` (Latin) with no localised
column, so a Bengali or Hindi reply names a department in Latin script — which
that voice silently drops. `agent/reply_templates.py:390`'s existing
`department_route_reply` already does exactly this, so the new templates are
consistent with current behaviour rather than copying a flaw silently. The fix
is localised department-name columns, named in §10.

---

## 9. Defects reported, not fixed

### Bengali apology marker matches inside ordinary words *(medium)*

`agent/apology.py` — `_MARKERS["bn"]` matches `সরি` unbounded, so the ordinary
adverb `সরাসরি` ("directly") reads as an apology.
`enforce_single_apology()` uses the same markers to remove a second apology, so a
correctly-worded reply containing such a word can have its **real** apology
stripped. `দুঃখিত` is duplicated in the alternation, and the Hindi markers
`क्षमा`/`खेद` should be checked for the same class of bug. Raised as a task.

---

## 10. Files changed

### New
```
agent/anger.py                      agent/complaint.py
agent/clinical_safety.py            agent/doctor_personal_request.py
agent/unverifiable_claim.py         agent/callback_request.py
agent/callback_config.py            agent/answer_ledger.py
agent/doctor_list.py                agent/compare_flow.py

tests/test_anger.py                 tests/test_orchestrator_anger.py
tests/test_complaint.py             tests/test_orchestrator_complaint.py
tests/test_clinical_safety.py       tests/test_orchestrator_clinical_safety.py
tests/test_doctor_personal_request.py
tests/test_unverifiable_claim.py    tests/test_callback_request.py
tests/test_answer_ledger.py         tests/test_doctor_list.py
tests/test_compare_flow.py          merge_report.md
```

### Modified
| File | Change |
|---|---|
| `main.py`, `main_pcm.py` | 11 imports, 3 session fields, 2 `slot_answer` entries, 8 guard blocks, 1 helper. **Kept byte-identical to each other in every patched region** |
| `agent/phrases.py` | 8 new keys in bn/hi/en (32 → 40) |
| `agent/reply_templates.py`, `agent/reply_templates_i18n.py` | `doctor_list_reply`, `doctors_by_department_reply`, `compare_options_reply`, `_spoken_doctor`, `MAX_DOCTORS_SPOKEN` |
| `agent/call_score.py` | `anger_turns`, `clinical_questions` counters; `complaint_filed` flag |
| `agent/tools_client.py` | `submit_complaint()`, `list_departments()`, `list_doctors()`, `doctors_by_department()` |
| `clinic-api/models.py` | `ComplaintRecord` |
| `clinic-api/enquiry_service.py` | `file_complaint()` + import |
| `clinic-api/main.py` | `ComplaintBody` + 4 endpoints + `_doctor_listing()`, bidirectional department matching |
| `clinic-api/idempotency.py` | **the §5 fix** |

### Untouched on purpose
`agent/intent_schema.py` and `agent/llm.py`'s prompt — per the "guards only"
decision. Every integrated feature reaches the caller **before** the model runs.

---

## 11. How to verify, and what remains

```bash
# the ported features (fast, no models, no services)
python -m pytest tests/test_anger.py tests/test_complaint.py tests/test_clinical_safety.py \
  tests/test_doctor_personal_request.py tests/test_unverifiable_claim.py \
  tests/test_callback_request.py tests/test_answer_ledger.py tests/test_doctor_list.py \
  tests/test_compare_flow.py tests/test_orchestrator_anger.py \
  tests/test_orchestrator_complaint.py tests/test_orchestrator_clinical_safety.py -q
```

```bash
# everything except the pod suite
python -m pytest -q --ignore=tests/test_smoke.py
# expect 0 failures; the only errors are "No module named 'nemo'"
```

`bash scripts/gate.sh --full` is what `CLAUDE.md` §4 requires before human
review. **It can now run** (§5 unblocked it) and has not been run here — it also
wants `ruff`, `mypy` and `gitleaks`, which are not installed locally.

### Before this ships

1. A **native speaker** reviews every Hindi line in §8.
2. The **adversarial suite is re-run per language** against
   `agent/clinical_safety.py`, in Hindi especially.
3. One **real call per language** confirms Bengali behaves exactly as it did
   before this merge (upgrade-prompt §8.4).
4. `bash scripts/gate.sh --full` is run and `gate-report.md` attached.

### Open work

| Item | Note |
|---|---|
| Wire `answer_ledger` | §4.9 — needs the `_answer_enquiry_intent` refactor |
| `date_calc` | §4.10 — needs the `date_expr` prompt change |
| Structured clinic hours | would let `check_callback_availability()` be ported honestly (§4.6) |
| Localised department names | the §8 limitation, pre-existing |
| Fix `agent/apology.py` | §9 |
