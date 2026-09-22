# E4-S3 — Caller moves an existing appointment: implementation plan

**Story.** As a patient whose plans changed, I want to move my appointment
without cancelling it, so that I do not lose my place entirely.

**Acceptance criteria.**
1. The booking is found by contact number, name or reference.
2. The new slot is swapped atomically, holding the old one until the new
   commits, and a failed swap leaves the original intact.
3. Confirmation is sent on both channels.

**Owner:** Rajarshee · **Reviewer:** Saurav · **Evidence:** "no reschedule capability"

**The property that matters most:** a failed move leaves the original
appointment exactly as it was. A caller told "that time is taken" still has
their Tuesday appointment.

---

## 0. Where this plan comes from

This is the plan from `ivr-rajarshee_dev_2`, which was built against the
`569f0c4` baseline and tested there (92 offline tests), **ported onto this
tree**. This tree already contains rows 2–15 and the `ADDED BY SOURAV`
stories, so the "MERGE" notes from that plan are carried out here, not
postponed. Section 2 lists every place this plan differs from the original
and why. Everything else (data model, swap algorithm, dialogue, decisions
D1–D6) is unchanged.

## 1. Decisions carried over (need a named owner's sign-off)

| # | Decision | Where it lives |
|---|---|---|
| D1 | Lookup needs **any 2 of {phone, name, reference}** to agree on the same future appointment. A single factor is refused with the same answer whether or not anything would match. This stands in for OTP, which doesn't exist for bookings yet. | `MIN_IDENTITY_FACTORS = 2` |
| D2 | The confirmation ID is **kept** across a move. The ID has the *original* date in it, so no one may read a date off an ID. | PATCH endpoint |
| D3 | No minimum notice. Any future date is allowed, including today. "Past" is checked at date level only. | PATCH endpoint |
| D4 | Same doctor only. Changing the doctor = cancel + new booking (cancellation doesn't exist yet). | PATCH uses `appointment.doctor_id` |
| D5 | "Both channels" = voice + SMS. SMS is queued in an outbox; no sender exists (E1-S10 absent). | `notification_outbox` |
| D6 | At most 3 matching appointments are offered, earliest first. | `_continue_reschedule` |

## 2. Reconciliation with this tree (new in this port)

| # | Difference from the original plan | Why |
|---|---|---|
| R1 | **Availability pre-check is scoped to the reference:** new `GET /api/v1/appointments/{reference}/availability?date=`, instead of `get_doctor_availability(doctor_name)`. | In this tree, doctor lookup goes through `match_band` and can come back *ambiguous*. The seed has both "Dr. Sen" and "Dr. A. Sen". Re-matching by name could report another doctor's hours, which is the same class of bug the DO-NOT against re-matching on the move exists for. The new endpoint uses `appointment.doctor_id` and returns no doctor or patient fields. |
| R2 | **Write-failure outcomes are their own exception classes:** `ToolWriteNotApplied(ToolCallError)` (provably not written) and `ToolOutcomeUnknown(ToolCallError)` (sent, answer lost). A final `except ToolCallError` still speaks `SYSTEM_UNREACHABLE_BN`. | `tests/test_outcome_distinction.py` requires every `except ToolCallError` handler to speak exactly `SYSTEM_UNREACHABLE_BN`. A write has two more truthful outcomes than a read. Naming them as distinct exceptions keeps the gate's meaning ("the system-down sentence never drifts") and still lets the caller hear "your appointment is unchanged" or "I can't confirm either way". |
| R3 | **Read failures (lookup, availability) speak `SYSTEM_UNREACHABLE_BN`**, not the reschedule "not changed" line. | Same gate. It's also true: no write was attempted. |
| R4 | **Templates have no colon, bracket or pipe.** They use `_spoken_list` for alternatives, and "নম্বর হলো …" instead of "নম্বর …:". | `tests/test_spoken_punctuation.py` (E12-S3) fails a build where any of those survive `verbalize()`. The original templates had three label colons. |
| R5 | **Only the `_RE_CONF_ID` suffix fix is needed in `bn_normalize`.** Grouped `spell_out()` (row 15) and single-letter reporting in `unspeakable_spans()` already exist here. | Verified: today `KCD-20260915-4F0C` verbalizes to `…পাঁচ-চারFশূন্যC`. Two letters are dropped and a digit is read as a number, on **every booking confirmation**, not only reschedules. |
| R6 | **`parse_phone` is extended, not rewritten.** Bengali and transliterated single-digit words are resolved token by token, on top of the existing English digit-word pass. `parse_otp` is untouched. | This tree's `parse_phone` already has English digit words and a pinned test file (`test_slot_parse_phone.py`). Every one of those cases must still pass. |
| R7 | **Confidence gate:** `resched_confirm` joins `skip_zones` in `_dispatch_turn_inner`. | Rows 2/3 echo low-agreement turns. A yes/no to the readback must not be re-echoed (same rule as `confirm_booking`). The echo's own "না" is handled before `_continue_pending`, so it never reaches the reschedule "না". An echo "হ্যাঁ" replays the original transcript into the flow. |
| R8 | **Write guard:** `_commit_reschedule(..., confirmed=True)` refuses and re-reads-back when not confirmed. | One write-guard convention with `_finish_booking` (row 2). |
| R9 | **Multi-part turns:** `_answer_part` gains a `reschedule_appointment` branch that opens the flow and returns `INTERACTIVE`. `_MULTI_INTENT_NEEDS_SEPARATE_FLOW` gains it too. | Otherwise a reschedule asked alongside another question would fall off the end of `_answer_part` and be silently dropped. |
| R10 | **Tool contracts and metrics:** `tool_contract` gets `find_appointments`, `appointment_availability` and `reschedule_appointment`. `tool_outcome.TOOLS` counts them. | Same boundary every other tool in this tree goes through. A malformed 200 on the PATCH is reported as *outcome unknown*, not "unreachable". The write may have committed. |
| R11 | **Test fixtures load clinic-api under a private module name** (`importlib`, not `import main`). | This tree's clinic-api tests pop and re-import `main`/`db`/`models`, so a shared name would collide with the orchestrator's `main`. |

## 3. Build order

Each step keeps the tree importable and the existing suite no worse than
its current baseline (287 failed / 3823 passed / 1 collection error).

1. **Schema** (`clinic-api/models.py`): two new append-only tables, and no new column on `appointments`. `create_all()` creates missing tables but never adds columns, and there's no migration tool (E0-S4).
   - `AppointmentChange` (`appointment_changes`): the history row, which is also the idempotency record. Fields: `idempotency_key` UNIQUE, and `result_json`, the exact response.
   - `NotificationOutbox` (`notification_outbox`): `kind`, `channel="sms"`, `template_id="reschedule_confirmation_v1"`, `payload_json` (no phone, no name), `status="pending"`.
2. **Booking race fix** (`book_appointment`): wrap the commit, turning `IntegrityError` into `slot_taken` plus free alternatives. New helper `_free_slots()`.
3. **Lookup:** `GET /api/v1/appointments/lookup?phone=&name=&reference=` (D1). Returns only `{reference, doctor_name, doctor_name_bn, date, time_slot}` per match. Never phone, never patient name.
4. **Availability for an appointment (R1):** `GET /api/v1/appointments/{reference}/availability?date=`. Returns `{found, date, available, chamber_hours, next_available_date}`.
5. **The swap:** `PATCH /api/v1/appointments/{reference}`.
   - Validation order: idempotent replay → not_found → past → conflict (expected slot ≠ current) → invalid_date → past → same_slot → doctor_not_available_that_day → slot_taken (outside hours).
   - Then one transaction: a conditional `UPDATE … WHERE id AND date=expected AND time_slot=expected`, the history row and the outbox row, then COMMIT.
   - Error handling: `IntegrityError` on `uq_doctor_slot` → rollback → `slot_taken` + alternatives. Any other error → rollback → re-raise (500).
6. **Client** (`agent/tools_client.py`): `find_appointments`, `get_appointment_availability`, `reschedule_appointment`.
   - The write has `WRITE_TIMEOUT_S = 8`, at most 2 attempts, and the same idempotency key on the retry.
   - Error mapping: connect error or HTTP status error → `ToolWriteNotApplied`; lost answer twice → `ToolOutcomeUnknown`; malformed or contract-violating 200 → `ToolOutcomeUnknown`.
   - Exception messages carry the exception **type only**, because httpx messages include the URL, and the URL carries phone and name.
7. **Contracts and metrics** (R10).
8. **Understanding.**
   - `llm.py`: `book_appointment` becomes "book or confirm a NEW appointment", and a new `reschedule_appointment` intent is added with all slots null. Its slots are ignored downstream.
   - `fast_path._BOOK_CUES`: add the 9 move cues, so the fast path abstains.
   - `semantic_cache`: no L2 for `reschedule_appointment`.
9. **Parsers** (`slot_parse.py`): extend `parse_phone` (R6), and add `parse_reference` and `parse_ordinal`.
10. **Speakability** (`bn_normalize._RE_CONF_ID`): the suffix `(?:-[0-9A-Z]{2,})?` (R5).
11. **Templates** (`reply_templates.py`): reschedule section, Bengali, no label punctuation (R4). The sentence "the old appointment is as it was" appears **only** where it's provably true.
12. **Dialogue** (`main.py`): the flow runs on `session.pending` with `flow="reschedule"`.
    - Routing: in `_continue_pending`, before any other state and before the universal "না" escape.
    - Entry: branches in `_dispatch_turn_inner` and `_answer_part` (R9).
    - `_continue_reschedule` state machine: `resched_ident → resched_name → [resched_pick] → resched_date → resched_time → resched_confirm → commit`.
    - Wiring: `skip_zones` (R7), write guard (R8), exceptions (R2/R3).
13. **`main_pcm.py`**: regenerate with `tools/make_pcm_variant.py`, then keep LF line endings.
14. **Tests** (pure; no GPU, audio or network; clinic-api in-process on a temp SQLite file):
    - `tests/test_reschedule_api.py`: lookup privacy, swap, rollback on injected crash, idempotent retry, conflict, every validation branch, 5-way concurrent move, 5-way concurrent booking.
    - `tests/test_reschedule_agent.py`: client retry/unknown/not-applied taxonomy against the real app over `httpx.ASGITransport`, parsers, speakability of every template branch, the tiers that must not answer, contracts.
    - `tests/test_reschedule_dialogue.py`: `main.py` end to end against the real clinic-api, plus every branch (no, slot taken, unknown, not applied, two matches, reference instead of phone, doctor not sitting, confidence echo, multi-part entry).
    - Cases for the new templates are added to the `tests/test_spoken_punctuation.py` gate.

## 4. The atomic swap, precisely

```
result = {success, reference, doctor_name, doctor_name_bn,
          old_date, old_time_slot, new_date, new_time_slot, date, time_slot}
BEGIN
  n = UPDATE appointments SET date=:new_date, time_slot=:new_slot
      WHERE id=:id AND date=:expected_date AND time_slot=:expected_slot
  if n != 1: ROLLBACK; replay stored result for key if any, else "conflict"
  INSERT appointment_changes (… idempotency_key UNIQUE, result_json)
  INSERT notification_outbox (… status='pending')
COMMIT
IntegrityError (uq_doctor_slot, or a concurrent same-key insert):
  ROLLBACK; replay stored result for key if any, else "slot_taken" + free slots
anything else: ROLLBACK; raise
```

The row keeps its old slot until COMMIT. `uq_doctor_slot` makes the new slot
exclusive. The `WHERE` on the expected slot turns a concurrent change into a
visible `conflict` instead of a silent overwrite. History and outbox land
with the move or not at all. This is portable to Postgres with no
`SELECT … FOR UPDATE`.

## 5. What the caller hears on each failure

| Situation | Written? | Caller hears |
|---|---|---|
| "না"/"থাক" at any step | no | "nothing changed", with the old slot if it was found |
| lookup / availability unreachable | no | `SYSTEM_UNREACHABLE_BN` |
| slot taken / doctor doesn't sit / same slot / past / not found | no | the reason + "your earlier appointment is as it was" + a next step |
| conflict (changed since readback) | no, but the old slot is gone | "it changed in between, nothing done, please check at the counter". **No** "unchanged" claim. |
| connect refused / HTTP error on the PATCH | no | "couldn't move it, your earlier appointment is as it was" |
| answer lost twice / malformed 200 | unknown | "can't confirm whether it changed, please check at the counter". Claims neither. |
| success | yes | new slot + "reference stays the same". **No** "SMS sent" claim. |

## 6. Status the story will close at

**PARTIAL: AC1 met (tightened by D1), AC2 met, AC3 half-built.** Voice
confirmation is met. The SMS confirmation is only *queued* in
`notification_outbox`. There's no sender until E1-S10 (SMS gateway +
DLT-registered template) exists. The story can't be closed until then.

## 6a. Result of this implementation (offline only)

- **New tests:** 114, all passing, plus 26 new cases in the E12-S3 punctuation gate.
  - `test_reschedule_api.py`: 24 passed
  - `test_reschedule_agent.py`: 71 passed (including 6 regression cases for the `parse_phone` conflict below)
  - `test_reschedule_dialogue.py`: 19 passed
- **Whole suite**, excluding `test_multi_intent_dispatch.py`, which already failed to collect: **277 failed / 3973 passed**, against 287 / 3797 on an untouched copy of the tree.
  - **0 newly failing.**
  - **10 newly passing**: the existing confirmation-ID fidelity tests in `test_number_fidelity*.py`, repaired by the `_RE_CONF_ID` suffix fix (R5).
- **Static gates:** already failing before this change. The reschedule code adds no offender to `test_outcome_distinction`, `test_fact_provenance` or `test_answer_consistency`. Its public templates are all covered by `test_spoken_punctuation`.
- **`main_pcm.py`:** regenerated. The generator verified the reasoning half is byte-identical, and line endings stayed LF.
- **Production-conflict check:**
  - **Old vs new on 4,500 inputs** (every string in the old tests, agent and seed, plus caller sentences):
    - Fast-path routing: unchanged.
    - Speech: only confirmation IDs changed, now read in full. This also fixes callback IDs.
    - `parse_phone`: one real conflict, now **fixed (R7)**. `"9876543210 এক মিনিট দাঁড়ান"` had become `8765432101`, another person's number, in flows that identify the caller by phone alone. Rule: numerals said in full win, and a number read as words must be exactly 10 digits (or 0 / 91 + 10).
  - **Upgrade:** a database made by the old clinic-api, opened by the new one, gains the two tables and keeps all its data. An old booking can be moved.
  - **Routes and state names:** no duplicate routes, no clashing state names.
- **Not run:** anything live (pod, real ASR/TTS, Qwen), `official_doc/test_smoke.py`, and Postgres. Local package versions differ from `requirements.txt`: fastapi 0.141, httpx 0.28, SQLAlchemy 2.0.54, pydantic 2.13.

## 7. Not done here (follow-ups)

1. Live test on the pod: book → call back → move; "না" at readback; a taken slot.
2. Measure Qwen's book-vs-reschedule separation on a labelled set of 50+ utterances, including code-mixed and indirect ones ("বৃহস্পতিবার আসতে পারব না").
3. What the ASR actually emits for spoken phone numbers and references (digits vs digit words vs pairs).
4. SMS sender over `notification_outbox` (closes AC3).
5. Calibrate `NAME_MATCH_FLOOR = 0.75` (reasoned).
6. Re-run the API tests against Postgres if `DATABASE_URL` points at it.
7. Hindi- and English-matrix callers: reschedule is Bengali-only, like booking's readback.
8. The outbox has no retention rule (E7-S5), and lookup is a linear scan (fine for a prototype).
