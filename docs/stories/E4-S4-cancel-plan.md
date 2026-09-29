# E4-S4 -- Caller cancels an appointment: implementation plan

**Epic:** Conversation: Booking, Rescheduling and Cancellation
**Story:** Caller cancels an appointment
**User story:** As a patient who cannot attend, I want to cancel and be told any
charge clearly, so that I am not surprised by a deduction later.
**Acceptance criteria:**
1. Cancellation applies the configured window rules and states refund
   eligibility from policy, never improvised.
2. A cancellation within a charging window is confirmed explicitly with the
   charge stated before it is applied.

**Owner:** Rajarshee. **Reviewer:** Saurav.
**Built on:** this tree (ivr_stagging) with E4-S3 (reschedule) already in it.
An earlier version of this story was built on a different tree
(Desktop\kolkata care\ivr-dev_sourav); it was the reference for this plan, but
line numbers and some design points differ (section 3).

---

## 1. What the caller experiences

```
CALLER  আমার অ্যাপয়েন্টমেন্টটা বাতিল করতে চাই
          -> Qwen classifies the turn as cancel_appointment (the ONLY component
             that decides "wants to cancel"; see section 2)
AGENT   আপনি কি অ্যাপয়েন্টমেন্টটা বাতিল করতে চান, নাকি বাতিল না করে অন্য দিনে বা
        অন্য সময়ে সরাতে চান? বাতিল করতে চাইলে হ্যাঁ বলুন, না চাইলে না বলুন।
        অন্য দিনে সরাতে চাইলে বলুন, সরাতে চাই।              <- NEW choice step
CALLER  হ্যাঁ                 -> cancel flow continues
        সরাতে চাই            -> hands over to the E4-S3 reschedule flow
        না                   -> "nothing was cancelled"
AGENT   (phone or confirmation number) -> (patient name) -> [which one?]
          -> clinic-api computes the charge from cancellation_rules.txt
AGENT   (no charge)  ... কোনো চার্জ লাগবে না ... বাতিল করব?        -> "হ্যাঁ" is enough
        (charge)     ... 320 টাকা চার্জ লাগবে ... চার্জ মেনে বাতিল করতে চাইলে বলুন,
                     হ্যাঁ, বাতিল করুন।                         -> ONLY explicit consent
CALLER  হ্যাঁ, বাতিল করুন
          -> clinic-api recomputes the charge; if it still matches what the
             caller heard, cancels and records the charge in one transaction
AGENT   আপনার অ্যাপয়েন্টমেন্ট বাতিল হয়েছে ... 320 টাকা বাতিলের চার্জ নথিভুক্ত হয়েছে ...
```

At the readback or the charge question the caller can still say "সরাতে চাই";
the same appointment is handed to the reschedule flow without asking for the
phone and name again.

## 2. Who decides what

| Decision | Decided by | Where |
|---|---|---|
| "the caller wants to cancel" | Qwen (LLM), intent `cancel_appointment` | agent/llm.py |
| cancel, move, or keep | the caller, parsed locally, no LLM | slot_parse.parse_cancel_choice |
| which appointment | clinic-api, two matching factors (E4-S3 D1) | GET /appointments/lookup |
| the charge and the refund eligibility | the admin's rules file, applied by clinic-api | clinic-api/cancellation_rules.txt |
| agreement to the charge | the caller, exact phrases only | slot_parse.is_explicit_consent |
| whether it is still allowed at commit | clinic-api recomputes | POST /appointments/{ref}/cancel |

The fast path and the fuzzy (L2) semantic cache are barred from this intent:
the fast path abstains on cancel words; L2 never reuses a cancel extraction.

## 3. Design decisions, and what differs from the earlier implementation

- **D1 -- The rules are a plain-text file an admin edits**
  (`clinic-api/cancellation_rules.txt`), not JSON. One line per window:
  `0 to 12 hours | charge 320 taka | refund none`. clinic-api reads it on EVERY
  quote and every cancel, so a saved edit applies to the next caller with no
  restart. Bengali digits are accepted.
  `python clinic-api/charging_for_cancellation.py` checks the file and prints
  what callers will hear, or every problem with its line number.
- **D2 -- A file that is wrong switches the feature OFF, never guesses.**
  Missing file, a line not understood, a gap or overlap between windows, a
  fractional charge, an empty `approved_by`, or an `effective_from` in the
  future: all make clinic-api answer `policy_unavailable`. The caller is told
  nothing was cancelled and is sent to the counter. **The shipped file has
  `approved_by` empty**: the numbers in it are examples until the clinic
  approves them (Blueprint Phase 0 gate: no fictional value reaches a caller).
- **D3 -- Choice step first** (the user's instruction for this build). A
  cancel request is first answered with "cancel, or move to another day?".
  "Move" starts the E4-S3 flow unchanged.
- **D4 -- Explicit consent for a charge.** A bare "হ্যাঁ", "হুম", "ওকে" or
  "ঠিক আছে" is NOT agreement to pay; the charge is restated and asked again.
  The phrase list (`_EXPLICIT_CONSENT`) needs a human sign-off.
- **D5 -- "Applied" means RECORDED.** There is no payment system in this code:
  an appointment has no fee and no payment record. The charge is written as
  `charge_status = "pending_collection"`; the caller is told the charge is
  recorded and to ask at the counter how to pay. No sentence says money was
  deducted or refunded. How the clinic collects it is undecided.
- **D6 -- Boundary crossed mid-call:** clinic-api recomputes at commit. If the
  window, the charge or the rules version differs from what the caller heard,
  nothing is written (`quote_changed`), the new charge is stated, and a fresh
  answer is required -- an earlier "yes" never carries over.
- **D7 -- A cancelled appointment is kept, not deleted**, marked
  `status = 'cancelled'`. Foreign keys are ON and E4-S3's history and outbox
  rows point at appointments. The full-table unique constraint `uq_doctor_slot`
  becomes a partial unique index over ACTIVE rows only, so the slot is free
  again. Existing databases are converted once by `clinic-api/migrations.py`
  (create_all never alters an existing table).
- **D8 -- After the start time: refused** by default (`after_start = not
  cancellable`); the admin may instead write a charge for it.
- **D9 -- Identification steps are duplicated, not shared, with E4-S3.** The
  earlier implementation moved them into one helper. In this tree a helper
  that speaks a template passed in as a parameter would violate the
  fact-provenance gate (every spoken span must be a literal or a named
  reply_templates call), and a helper that only returns a status would still
  need the reschedule flow rewritten. Duplicating ~40 lines leaves the
  reschedule code byte-for-byte unchanged, which is the safer trade here.
- **D10 -- Failure wording follows this tree's gates**
  (tests/test_outcome_distinction.py): every `except ToolCallError` speaks
  `SYSTEM_UNREACHABLE_BN`. Only the two narrower write failures speak their own
  sentence: `ToolWriteNotApplied` -> "nothing was cancelled",
  `ToolOutcomeUnknown` -> "cannot confirm either way".

## 4. Files

| File | Change |
|---|---|
| clinic-api/cancellation_rules.txt | NEW. The admin's rules. |
| clinic-api/charging_for_cancellation.py | NEW. Pure parser, validator and calculator; a checker when run directly. |
| clinic-api/migrations.py | NEW. Versioned migration 0001_appointment_status. |
| clinic-api/models.py | Appointment.status, cancelled_at; uq_doctor_slot -> uq_doctor_slot_active; NEW AppointmentCancellation. |
| clinic-api/main.py | Migration at startup; booking, free slots, lookup, availability and reschedule ignore cancelled rows; NEW GET cancellation-quote, POST cancel. |
| agent/tools_client.py | NEW get_cancellation_quote, cancel_appointment. The reschedule write loop is shared through _keyed_write with identical messages. |
| agent/tool_contract.py, agent/tool_outcome.py | Contracts and outcome classes for the two new tools. |
| agent/llm.py | Intent `cancel_appointment`. |
| agent/fast_path.py, agent/semantic_cache.py | Cancel cues abstain; never L2-cached. |
| agent/slot_parse.py | NEW parse_cancel_choice, is_explicit_consent. |
| agent/reply_templates.py | NEW cancellation section (Bengali, no label colons). |
| main.py (+ regenerated main_pcm.py) | NEW cancel flow; routing; dispatch; multi-part branch; "cancel_confirm" skips the echo gate. |
| tests/ | NEW cancel test files; cases added to test_spoken_punctuation.py. |

## 5. Verification

- Before and after: full suite, compared test by test against the list of
  failures in the tree before this story (277 known failures, none caused
  by this work). Pass condition: no test that passed before fails after.
- E4-S3's deleted tests (backed up in the session scratchpad) are run once
  against the new tree to prove the reschedule flow still behaves, then
  removed again.
- An upgrade smoke test: the OLD clinic-api creates a database and books; the
  NEW clinic-api starts on the same file, migrates it, and cancels.

## 6. Not done here

- No live call, no Qwen measurement of cancel vs reschedule vs book.
- Replies are Bengali only (like E4-S3).
- The Postgres branch of the migration is untested.
- Charges are recorded, never collected (D5).

## 7a. Later on 2026-09-19 -- tests deleted, feature switched ON

- **The story's test files were deleted on request** (tests/cancel_fixtures.py,
  test_cancellation_rules.py, test_cancel_api.py, test_cancel_agent.py,
  test_cancel_dialogue.py; backup in the session scratchpad,
  `deleted_cancel_tests`). The 35 cases in the existing
  tests/test_spoken_punctuation.py were kept. After the deletion: 277 failed
  (the same pre-existing list) / 3894 passed; all 47 production .py files
  byte-identical; nothing outside tests/ referenced the deleted files.
- **Switched ON for now:** `approved_by` in clinic-api/cancellation_rules.txt
  now reads "Rajarshee (switched on for now, 2026-09-19 -- example charges,
  clinic sign-off pending)". The charges are still the example numbers (D2).
  Verified with the checker (ON, exit 0), an upgrade smoke test and a
  dialogue smoke test through main.py, both with the shipped file.
- Full write-up: row 23 of raj_ivr_story_with_implementation.xlsx, with the
  switch report in the new column T "Switch".

## 7. Results (2026-09-19, offline)

**New tests: 191, all passing.**

| File | Tests | Covers |
|---|---|---|
| tests/test_cancellation_rules.py | 48 | the rules file: every boundary to the minute, ordinary wordings (₹, Rs, commas, Bengali digits), 17 kinds of wrong file each refused with its reason, the checker's ON/OFF output |
| tests/test_cancel_api.py | 25 | quote, cancel, consent enforced by the server, boundary crossed mid-call, rules edited mid-call, retry-once, concurrency (4 threads, 1 winner), crash rollback, freed slot re-bookable, cancelled appointment invisible to lookup / availability / reschedule, the migration of a pre-story database |
| tests/test_cancel_agent.py | 91 | cancel/keep/move parsing, explicit consent, every sentence speakable with no digits, Latin letters or colons left, no "deducted"/"refunded" wording, the fast path and L2 cache abstain, contracts, the client's three write outcomes, reschedule's messages unchanged |
| tests/test_cancel_dialogue.py | 27 | main.py end to end on the real client and clinic-api: the choice step, move -> reschedule (before and after identification), charged and free cancels, grunts never charge, a free yes never carried into a charge, every refusal and failure line, two matches, the low-confidence echo |

Plus 35 cancel cases in the existing tests/test_spoken_punctuation.py gate,
all passing.

**Full suite** (`python -m pytest tests --ignore=tests/test_multi_intent_dispatch.py`):
277 failed, 4085 passed, 1 skipped. Before this story: 277 failed, 3859
passed. The 226 extra passes are exactly the 191 new tests plus the 35 new
punctuation cases. Compared test by test, the 277 failures are the same list
as before (none caused by this work) and no test that passed before fails now.

**E4-S3 still works:** its 114 deleted tests (scratchpad backup) were run
once against this tree: 112 passed. The 2 that failed are a source-level
check that assumed reschedule was the ONLY code catching ToolOutcomeUnknown /
ToolWriteNotApplied; the cancel flow now catches them too and speaks its own
sentences, which is correct. The files were removed again afterwards.

**Gates:** the 3 new `except ToolCallError` handlers speak
SYSTEM_UNREACHABLE_BN with the tool_failure clip; no new offender in the
fact-provenance or spoken-punctuation gates (both already failed on older
code before this story).

**Upgrade smoke test** (two processes): the OLD clinic-api created a
database and booked; the NEW one started on the same file:
`schema_migrations = [0001_appointment_status]`, index
`uq_doctor_slot_active`, empty foreign_key_check; without consent ->
`charge_not_confirmed`; with consent -> success, 320, `pending_collection`;
the freed slot booked again (rows in that slot: cancelled, active); a second
start applied nothing. With no usable rules file the same run answered
`policy_unavailable` and wrote nothing.

**Not run:** anything on the pod, real ASR/TTS, Qwen on cancel utterances,
the Postgres migration branch, scripts/gate.sh.
