# Caller asks for the earliest available appointment -- implementation plan

**Epic:** Conversation: Booking, Rescheduling and Cancellation
**Story:** Caller asks for the earliest available appointment
**User story:** As a caller who wants to be seen soon, I want the earliest slot
offered directly, so that I do not have to guess dates until one is free.
**Acceptance criteria:**
1. The agent offers the earliest available slot with its day and time, plus
   the next two alternatives.
2. Availability is read live.
3. If nothing is available within a configured horizon the agent says so and
   offers a callback when a slot opens.

**Owner:** Rajarshee. **Reviewer:** Saurav.

**Status:** BUILT 2026-09-21 -- see section 7. Not deployed.

---

## 0. What the official docs say

`official_doc/KolkataCareVoiceAgentUserStoriesCodeMixed.docx` (175 stories,
180 tables) has **no story with this title or wording** ("earliest",
"soonest", "configured horizon", "when a slot opens" -- all searched). The
other five documents in `official_doc/` do not mention it either. This is a
custom story, named by slug like the previous one
(`doctor-only-booking-plan.md`). It touches these backlog stories:

| Backlog story | What it requires | How this plan relates |
|---|---|---|
| **E8-S1** Hospital system connector for doctors and slots (P0) | *"Availability is read live from the hospital system. Staleness is bounded and stated. The connector fails closed and never serves a stale slot as current."* | AC2 is this story's rule applied to one question. There is no hospital connector in this tree; "live" here means **read from clinic-api's database at the moment of the question, never cached** (section 1, F5). When E8-S1 lands, the same endpoint reads the connector instead. |
| **E4-S1** Booking with slot locking (P0) | *"A slot is held with a time-to-live when offered, committed on confirmation and released on abandonment."* | NOT built in this tree. An offered earliest slot is not held; another caller can take it before the write. The existing write already refuses a taken slot and returns the free ones (section 5, R1). |
| **E12-S5** Dead ends always offer a next step (P0) | *"No reply ends on a bare apology. A not-found offers the nearest alternatives, a transfer to a human, or a callback."* | AC3: "nothing within the horizon" must offer a callback, not end. |
| **E6-S5** Callback when no agent is free / **E1-S8** Missed call to callback | *"A callback slot is offered and honoured... stated honestly rather than optimistically."* | The callback is recorded by the **existing** "Caller asks to be called back" flow. The system has **no outbound dialer** (agent/callback_flow.py); a human places the call. The wording must say so. |
| **E13-S1** Ask for several fields in one turn | grouped questions | After a slot is taken, the rest (name + phone) is asked with the grouped question built for "Caller names only a doctor". |

## 1. What the code does today (checked, not assumed)

| # | Finding | Where | Effect |
|---|---|---|---|
| F1 | **No "earliest slot" anywhere.** The closest is `_next_available_date()`: the first day in 14 whose weekday the doctor **sits** -- it does NOT look at bookings. A fully booked day counts as "available". | clinic-api/main.py:607-613 | "Next available" can name a day with no free slot left. |
| F2 | Free slots are computed **for one day only** (`_free_slots(db, doctor, date, sched, limit=3)`: the doctor's 15-minute slots minus active appointments, earliest first). Nothing walks forward across days. | clinic-api/main.py:887-894 | The data needed exists; it is not combined across days. |
| F3 | **Past times today are never excluded.** `_generate_slots` / `_free_slots` return the whole chamber session; `book_appointment` also accepts a time that has already passed today. | clinic-api/main.py:783-790, 793-866 | An "earliest" answer at 11:40 could offer 10:00 today. Must be fixed inside the new query (the booking write's own gap is flagged, not fixed -- R4). |
| F4 | "When is Dr Sen next available" is classified `doctor_availability` (agent/llm.py:180), and the dispatcher **defaults a missing date to TODAY** (main.py:4341-4370). So "earliest" today is answered as "is he in today?", with no time and no alternatives. | agent/llm.py:180, main.py:4341 | AC1 is not met on any route. |
| F5 | Availability is already **never cached**: `get_doctor_availability` is deliberately excluded from `reference_data_cache` ("a specific day's slot state... always live"), and the semantic cache stores only the intent extraction, never a tool result. | agent/tools_client.py:243-255, agent/reference_data_cache.py:38-45 | AC2 holds if the new call follows the same rule (not added to the reference cache). |
| F6 | The **horizon is a hard-coded 14** (`horizon_days: int = 14`), not configurable. | clinic-api/main.py:608 | AC3's "configured horizon" needs a setting. |
| F7 | A **callback flow exists**: `request_callback` intent; checks `CALLBACKS_ENABLED` (env, agent/callback_config.py) and the clinic's opening hours first; collects `callback_time_window` + `callback_phone`; confirms; `POST /api/v1/callbacks` stores a pending row (`callback_requests`: phone, free-text time_window, reason). A human makes the call; nothing moves it past "pending". No doctor column, no "slot opened" trigger. | main.py:4480-4540, 2684-2740; agent/callback_flow.py; clinic-api/main.py:1941-1990; models.py:1216 | AC3's callback can reuse this flow as-is: time window fixed to "when a slot opens", reason naming the doctor, only the phone asked. |
| F8 | After a slot is chosen, the **existing booking flow** already carries the caller to the write: `_next_booking_step` asks name + phone together, readback, explicit "হ্যাঁ", `_finish_booking` (which refuses a taken slot and returns free alternatives). | main.py `_open_booking`, `_next_booking_step`, `_continue_pending` tail | No new booking path is needed. |
| F9 | A picker for "which of these?" already exists: `parse_ordinal(text, count)` ("প্রথমটা", "দ্বিতীয়", "শেষেরটা", "২"). | agent/slot_parse.py:614 | Choosing among the three offered slots needs no new parser. |

## 2. Design principle (from the previous story)

Rajarshee's rule for "Caller names only a doctor": **attach the feature to
the existing flow; do not build a parallel branch; keep the pipeline light.**
So:
- one new **read** endpoint in clinic-api (the only genuinely new capability:
  free slots across days);
- the agent's **existing booking flow** carries everything after the offer;
- the **existing callback flow** records the "call me when a slot opens"
  request;
- no new intent, no new write endpoint, no schema change.

## 3. Decisions (recommended -- need sign-off where marked)

| # | Decision | Why |
|---|---|---|
| D1 | **A doctor is required.** "Earliest appointment" with no doctor asks "which doctor?" (the existing question). Earliest-across-a-department is NOT in this story. **[sign-off]** | Every slot belongs to a doctor; the booking write needs one. A department-wide "earliest with anyone" is a different, larger feature (it must rank doctors). |
| D2 | **"Earliest" is detected by code on the caller's words**, not by a new model intent: a small phrase matcher `wants_earliest(text)` in agent/slot_parse.py ("সবচেয়ে তাড়াতাড়ি", "যত তাড়াতাড়ি সম্ভব", "সবচেয়ে আগে", "প্রথম খালি", "কবে খালি", "earliest", "soonest", "as soon as possible", "jaldi se jaldi", "sabse pehle", "joto taratari"...). It fires only on a `book_appointment` or `doctor_availability` turn with a doctor and **no explicit date**. One line is added to agent/llm.py's book_appointment description so such sentences keep classifying as a booking. | A new intent touches the validator, the semantic-cache defining-slot map, the multi-intent dispatcher and every test that enumerates intents -- heavy for one phrase. The model already classifies these as booking/availability; code only needs to notice "earliest", which is also the project's rule that code, not the model, decides facts. |
| D3 | **"Next two alternatives" = the next two free slots after the earliest, in time order** (they may be on the same day: 10:00, 10:15, 10:30). **[sign-off]** Alternative: the earliest free slot on each of the next three sitting days (more choice of day). | Literal reading of the AC. The alternative is one flag in the endpoint (`spread=day`) if preferred. |
| D4 | **Horizon is a setting**, `EARLIEST_SLOT_HORIZON_DAYS`, read by clinic-api from the environment, **default 14** (the value `_next_available_date` already uses), listed in clinic-api/company_config.py's index. A minimum lead time `EARLIEST_SLOT_LEAD_MINUTES` (default 30) drops today's slots that start too soon to reach. **[sign-off on both numbers]** | Same convention as `CALLBACKS_ENABLED` / `DATABASE_URL`. (If the clinic prefers an admin-editable .txt like cancellation_rules.txt, the loader is the only difference.) |
| D5 | **The callback is recorded, not placed.** The reply says a staff member will call when a slot opens; `callback_time_window` = "when a slot opens"; `reason` = "earliest slot with <doctor>: none within N days (asked <date>)". Only the phone is asked (reused if the caller already gave it). If callbacks are disabled or the clinic is closed now, the existing `callback_unavailable_reply` is spoken, plus "please call again / ask for another day" -- never a bare refusal. **[sign-off: no automatic "slot opened" trigger]** | There is no dialer (F7). A trigger that watches cancellations (E4-S4) and flags matching pending callbacks is a sensible follow-up, but it adds a schema column and a worker -- not light. |
| D6 | **An offered slot is not held** (E4-S1 is not built). If it is taken before the write, the existing refusal + `alternative_slots` reply handles it. | Holding needs a hold table + TTL + release -- that is E4-S1 itself. |
| D7 | **"Live" = read from clinic-api's database on every question, never cached** (not added to reference_data_cache; the agent calls the endpoint every time). The reply does not claim "live from the hospital system". | Honest about E8-S1 not existing (F5). |

## 4. Design

### 4.1 What the caller hears

```
CALLER ডাক্তার সেনের কাছে সবচেয়ে তাড়াতাড়ি কবে অ্যাপয়েন্টমেন্ট পাব?
         -> book_appointment (or doctor_availability), doctor_name only
         -> wants_earliest(text) = True, no explicit date
         -> clinic-api GET /doctors/earliest-slots (live, 3 slots)
AGENT  ডাঃ সেনের সবচেয়ে আগের খালি সময় সেপ্টেম্বর মাসের বাইশ তারিখ, সকাল দশটা।
       এছাড়া ওই দিন সকাল সোয়া দশটা, আর সকাল সাড়ে দশটাও খালি আছে।
       কোনটা নেব? প্রথমটা বললেই হবে, বা অন্য দিন বলুন।
CALLER প্রথমটা   (or "হ্যাঁ" = the earliest, or "সাড়ে দশটা", or "দ্বিতীয়টা")
         -> date + time filled -> the EXISTING booking flow from here
AGENT  রোগীর নাম আর কনফার্মেশনের জন্য একটা ফোন নম্বর বলবেন?     (existing grouped question)
CALLER রাহুল দাস, ৯৮৭৬৫৪৩২১০
AGENT  বুক করার আগে একবার শুনে নিন। ... সব ঠিক আছে তো?                (existing readback)
CALLER হ্যাঁ     -> existing write (refuses a taken slot, offers free ones)
```

Nothing within the horizon:

```
AGENT  ডাঃ সেনের সামনের চোদ্দো দিনে কোনো খালি সময় নেই। কোনো সময় খালি হলে
       আমাদের একজন কর্মী আপনাকে ফোন করে জানাবেন -- ফোন নম্বরটা দেবেন?
CALLER ৯৮৭৬৫৪৩২১০
         -> EXISTING callback flow: confirm_callback -> POST /api/v1/callbacks
AGENT  (existing callback confirmation + reference number)
```

### 4.2 clinic-api (the one new capability)

`GET /api/v1/doctors/earliest-slots?name=<doctor>&limit=3` -- read only.

```
verdict, doctor, offered = _resolve_doctor(db, name)
ambiguous -> _ambiguous_reply(...)          (same shape as /doctors/availability)
not found -> {"found": false, "query": name}
now = datetime.now(); horizon = EARLIEST_SLOT_HORIZON_DAYS; lead = EARLIEST_SLOT_LEAD_MINUTES
for offset in range(horizon):
    day = today + offset
    sched = _schedule_for_weekday(db, doctor.id, day.weekday()); if not sched: continue
    for t in _free_slots(db, doctor.id, day.isoformat(), sched, limit=None):
        if offset == 0 and t < (now + lead).strftime("%H:%M"): continue
        collect {"date": day.isoformat(), "time_slot": t, "chamber_hours": f"{start}-{end}"}
        stop at `limit`
-> {"found": true, "doctor_name", "doctor_name_bn", "horizon_days": horizon,
    "slots": [...], "as_of": now.isoformat(timespec="seconds")}
   (slots == [] means nothing free within the horizon)
```

- `_free_slots` gains `limit: int | None = 3` (None = all). The two existing
  callers pass nothing, so their behaviour is unchanged.
- Reads `Appointment.status == "active"` through `_free_slots`, so a
  cancelled appointment's slot is free again (E4-S4) -- live by construction.
- `as_of` is returned (E8-S1's "staleness stated"); it is logged, not spoken.

### 4.3 Agent

| File | Change |
|---|---|
| agent/tool_contract.py | Contract for `doctor_earliest_slots` (found / ambiguous / slots list shape; each slot has ISO date + HH:MM) -- the same validation every tool response already passes through. |
| agent/tools_client.py | `get_earliest_slots(doctor_name, limit=3)` -- NOT cached (same note as `get_doctor_availability`). |
| agent/slot_parse.py | `wants_earliest(text)` -- the phrase matcher (D2), Bengali-aware boundaries like `_match_day_word`. |
| agent/llm.py | One line in the `book_appointment` description: "asking for the earliest / soonest appointment with a doctor is book_appointment with doctor_name and no date". |
| agent/reply_templates.py | `booking_earliest_slots_prompt(slots, result, language)` (earliest + two alternatives + "which one?"; 1 or 2 slots worded naturally) and `booking_earliest_none_prompt(slots, result, language)` (none within N days + callback offer). 4 languages each; every date/time read off `result`. |
| main.py `_open_booking` | Doctor found, no date, and `merged["want_earliest"]`: call `get_earliest_slots`; slots -> speak the earliest prompt, `pend("date", offered_slots=[...], offered_date=slots[0].date)`; none -> speak the none prompt and hand to the callback flow (below). Without `want_earliest` the path is exactly as today (the doctor-offer question). |
| main.py `_booking_slots_from_turn` | Sets `merged["want_earliest"] = True` when `wants_earliest(text)` and no date was said. Not sent to the booking API (it reads only its five fields). |
| main.py doctor_availability branch | Same check: an "earliest" question with no date goes to `_open_booking` (so both intents give the same answer) instead of defaulting to today. |
| main.py `_continue_pending`, the existing `date` tail | When `pending["offered_slots"]` is set: `parse_ordinal(text, n)` -> that slot; a bare "হ্যাঁ" -> the first; a time equal to one offered -> that one; otherwise the answer is read as a normal day/time (the caller asked for another day), exactly as today. The chosen slot fills `date` + `time_slot`, then the existing `_open_booking` re-check (sitting + hours) and `_next_booking_step` (name + phone, readback). **No new state.** |
| main.py callback hand-off | Reuses the existing callback code: `CALLBACKS_ENABLED` + `check_callback_availability` first; available -> `session.pending = {"awaiting": "callback_phone", "slots": {"callback_time_window": "<when a slot opens>", "callback_reason": <D5 text>, "phone": <if already known>}}` -> the existing `callback_phone` / `confirm_callback` / `_finish_callback` steps. If the phone is already known, straight to `confirm_callback`. Not available -> existing `callback_unavailable_reply` + "অন্য কোনো ডাক্তার বা পরে ফোন করে দেখতে পারেন" (no bare refusal). |
| main.py `except ToolCallError` | The new call's handler speaks only `SYSTEM_UNREACHABLE_BN` with `fallback_reason="tool_failure"` (outcome-distinction gate). |
| main_pcm.py | Regenerated with `python tools/make_pcm_variant.py`. |
| tests/test_spoken_punctuation.py | One case per new template (kept after the story's own tests are deleted). |

### 4.4 What is deliberately NOT built

- No new intent; no new dialogue state (the `date` state carries the offered slots).
- No slot hold / TTL (E4-S1).
- No automatic "slot opened" callback trigger, no schema change to `callback_requests` (D5).
- No department-wide earliest (D1).
- No hospital-system connector (E8-S1).

## 5. Risks and open questions

- **R1 -- the offered slot can be taken** before the caller confirms (no hold). The existing write refuses and offers free alternatives; the caller hears that at the end, not at the offer. E4-S1 fixes it properly.
- **R2 -- phrase coverage.** `wants_earliest` only knows the phrases listed; ASR spellings of "তাড়াতাড়ি" vary. Build the list from real transcripts before relying on it; a miss falls back to today's behaviour (the doctor-offer question), never to a wrong answer.
- **R3 -- three slots on the same day** (D3) may feel like one option to the caller. Listen to it; switch to one-per-day if so.
- **R4 -- the booking write accepts a past time today** (F3). The new query never offers one, but the write itself is not fixed by this story -- flag as a separate fix.
- **R5 -- the pod clock and timezone** decide "now" and "today" for the lead-time cut; a UTC pod would offer slots that already passed in Kolkata. Check `TZ` on the pod.
- **R6 -- "when a slot opens" is a promise a human keeps.** Staff need to see these pending callbacks (reason names the doctor). Agree the operating process before switching on.
- **R7 -- long reply.** Three dates/times in one sentence at TTS pace; the second and third are said as times only when on the same day as the first to keep it short.

## 6. Build steps

1. clinic-api: `EARLIEST_SLOT_HORIZON_DAYS` / `EARLIEST_SLOT_LEAD_MINUTES` (env, defaults 14 / 30), listed in company_config.py; `_free_slots(limit=None)`; the new endpoint.
2. agent/tool_contract.py contract; agent/tools_client.py `get_earliest_slots` (uncached).
3. agent/slot_parse.py `wants_earliest`; agent/llm.py one line.
4. agent/reply_templates.py two templates (4 languages).
5. main.py: `_booking_slots_from_turn` flag; `_open_booking` earliest branch; doctor_availability routing; `date` tail offered-slot choice; callback hand-off.
6. Regenerate main_pcm.py.
7. Punctuation-gate cases for the two templates.
8. Tests in their own deletable file, against a real in-process clinic-api on a seeded temp database:
   - earliest = first free slot, skipping a booked 10:00 and a fully booked day; next two in order;
   - today's past slots and slots inside the lead time skipped (frozen clock);
   - a cancelled appointment's slot is offered again (live read);
   - horizon from the env var; nothing within it -> `slots: []`;
   - agent: offer spoken with day + time + two alternatives; "প্রথমটা" / "হ্যাঁ" / "দ্বিতীয়টা" / "সাড়ে দশটা" pick the right slot; "অন্য দিন, বুধবার" falls back to the normal day path; then name + phone -> readback -> row in the database;
   - none within horizon -> callback offered -> phone -> confirm -> row in `callback_requests` with the time window and reason; callbacks disabled / clinic closed -> the existing unavailable reply, no dead end;
   - unknown / ambiguous doctor -> existing replies; clinic-api down -> SYSTEM_UNREACHABLE_BN;
   - regression: doctor-only, one-sentence booking, reschedule, cancel unchanged;
   - full suite vs the current baseline (277 failed / 3906 passed / 1 skipped), no new gate offender.
9. Update this plan with results; then (on request) delete the story's tests and write the spreadsheet row.

## 7. What was built (2026-09-21)

Built as planned, ON the booking flow (Rajarshee: "keep it light like
before"), with the defaults of section 3 (doctor required; next two = next
free slots in time order; horizon 14 days; lead time 30 minutes; callback
recorded for a staff member). Plus **fix 1** (asked for separately) and one
pre-existing bug that fix 1 exposed.

| File | Lines | Change |
|---|---|---|
| clinic-api/main.py | 1992 -> 2053 | `EARLIEST_SLOT_HORIZON_DAYS` / `EARLIEST_SLOT_LEAD_MINUTES` (env, defaults 14 / 30, next to `SLOT_STEP_MIN`); `_free_slots(limit=None)`; NEW `GET /api/v1/doctors/earliest-slots` (read only). |
| agent/tool_contract.py | 206 -> 214 | contract `doctor_earliest_slots`. |
| agent/tools_client.py | 884 -> 909 | `get_earliest_slots()` -- never cached. |
| agent/slot_parse.py | 794 -> 815 | `wants_earliest()`. |
| agent/llm.py | 430 -> 430 | one sentence added to the book_appointment line. |
| agent/reply_templates.py | 3474 -> 3555 | `booking_earliest_slots_prompt`, `booking_earliest_none_prompt` (4 languages). |
| main.py | 4990 -> 5145 | `_booking_slots_from_turn` sets `want_earliest`; NEW `_offer_earliest()` + `_pick_offered_slot()`; `_open_booking` calls `_offer_earliest` for a doctor-only turn that asked for the earliest; the existing field tail accepts a pick among `offered_slots` (date / time_slot states); both doctor_availability branches route an "earliest" question to `_open_booking`; **fix 1** in `_finish_booking`; **bug fix** below. |
| main_pcm.py | 5005 -> 5160 | regenerated. |
| tests/test_spoken_punctuation.py | 349 -> 362 | 4 cases (`earliest/*`). |
| tests/test_earliest_appointment.py | NEW, 454 | 20 tests; its own fixtures, deletable as one file. |

Deviation from D4: the two settings live in clinic-api/main.py beside
`SLOT_STEP_MIN`, not in company_config.py (that file indexes URLs).

**Fix 1.** When the write comes back `slot_taken` with free alternatives,
`_finish_booking` now keeps the booking open (existing `time_slot` state,
doctor / date / name / phone kept, the alternatives as `offered_slots`), so
"প্রথমটা" or a time goes to a fresh readback and the write -- no restart.

**Pre-existing bug fixed (found by the fix-1 test).** `_finish_booking` ran
`outcomes.missing_booking_write_fields(result)` on EVERY response. A refusal
has no confirmation_id, so every refused booking (slot taken, day not sat)
was spoken as INSUFFICIENT_VERIFIED_INFORMATION_BN: "আপনার অ্যাপয়েন্টমেন্টটা
হয়ে গেছে ... আবার বুক করার দরকার নেই" -- the caller was told they were booked
when nothing was written. The function's own docstring says it is for a
response "that otherwise reported success=True"; the call now only runs on
success. One line; present in this tree before this story.

**Verification.** 20/20 story tests against a real in-process clinic-api
(seeded temp SQLite): earliest + next two in order; booking and then
cancelling the first slot moves it out and back (live read); a fully booked
day skipped; nothing today inside the lead time; horizon setting; the offer
spoken (book_appointment AND doctor_availability); "দ্বিতীয়টা" -> name +
phone -> readback -> the row, "after 4 caller turn(s)"; "হ্যাঁ" = the
earliest; saying an offered time; another weekday -> normal day path;
nothing within the horizon -> callback -> row in callback_requests (window
"সময় খালি হওয়ার দিন", reason naming Dr. A. Sen); callbacks off / outside
hours -> said, no dead end; slot taken before the write -> alternatives ->
booked with name + phone kept; clinic-api down -> SYSTEM_UNREACHABLE_BN;
unknown doctor, doctor-only without "earliest", one-sentence booking --
unchanged.
Full suite: 277 failed / 3930 passed / 1 skipped -- the identical 277 as
before (none new, none fixed); +24 = 20 story tests + 4 gate cases. Static
gate messages unchanged (no new offender). Production files changed: exactly
the 8 above.

**Noted, not changed.** The existing callback confirmation wraps the window
as "{window}-এর মধ্যে", which reads a little stiffly with "সময় খালি হওয়ার
দিন" ("... দিন-এর মধ্যে কল ব্যাক করব"). Listen on the pod; a tweak would
touch the existing callback template.

## 8. Acceptance-criteria audit (2026-09-21, same day)

Each clause checked against every way a caller can reach it; four gaps found
and closed:

| Gap | Fix |
|---|---|
| "Day" was only the date ("বাইশ তারিখ") | the offer now names the weekday too: "মঙ্গলবার, সেপ্টেম্বর মাসের বাইশ তারিখে, সময় সকাল দশটা" (`_weekday_of`, from the date clinic-api returned) |
| "সবচেয়ে তাড়াতাড়ি যেটা আছে" said in ANSWER to the day question was not understood | the field tail offers the earliest slots when the answer asks for them and names no day |
| an unclear reply to the offer dropped the offered slots | the offer is kept until a slot is picked or a day/time is given |
| an "earliest" question the model files as `doctor_schedule` got the weekly schedule | routed to the same offer (main dispatcher) |

| AC clause | Where in code | Proven by |
|---|---|---|
| earliest available slot | clinic-api `doctor_earliest_slots` (first free slot, today's past + lead-time slots excluded, fully booked days skipped) | earliest_is_the_first_free_slot..., a_fully_booked_day_is_skipped, nothing_today_starts_inside_the_lead_time |
| with its day and time | `booking_earliest_slots_prompt` (weekday + date + time; verbalised) | ac1_offer_names_the_day_and_time_plus_two_alternatives |
| plus the next two alternatives | `limit=3`, in time order | ac1_..., earliest_question_offers_three_slots |
| availability read live | endpoint reads the tables on every call; `get_earliest_slots` never cached | availability_is_read_live_book_and_cancel, ac2_a_second_caller_sees_the_first_callers_booking |
| nothing within a configured horizon -> says so | `EARLIEST_SLOT_HORIZON_DAYS`; `booking_earliest_none_prompt` | horizon_is_a_setting, none_within_horizon_* |
| offers a callback when a slot opens | `_offer_earliest` -> existing callback flow (window "সময় খালি হওয়ার দিন", reason names the doctor) | none_within_horizon_offers_a_callback_and_records_it, ac3_phone_already_known..., callbacks_off / outside_hours (no dead end) |
| every entry route | book_appointment, doctor_availability, doctor_schedule, no doctor then the doctor, the answer to the day question | route_* tests |

Tests: 27/27 (tests/test_earliest_appointment.py). Full suite 277 failed /
3937 passed / 1 skipped -- the identical 277, none new; gate messages
unchanged.

Not covered (by design): an "earliest" request inside a COMBINED
multi-question answer (`_resolve_combinable_intent_fragment`, which only
returns a text fragment and opens no booking) still answers availability
for today.
