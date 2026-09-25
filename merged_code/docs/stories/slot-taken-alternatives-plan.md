# Requested slot is already taken -- implementation plan

**Epic:** Conversation: Booking, Rescheduling and Cancellation
**Story:** Requested slot is already taken
**User story:** As a caller whose preferred time is gone, I want alternatives
immediately, so that the call does not stall on a no.
**Acceptance criteria:**
1. A taken slot produces the two nearest alternatives on the same day and the
   same time on the nearest other day, in one sentence.
2. The caller may accept one by saying which.
3. No alternative is offered that is not actually free at the moment of
   speaking.

**Owner:** Rajarshee. **Reviewer:** Saurav.
**Status:** BUILT 2026-09-21 -- see section 7. Not deployed. Kept light, on the existing booking flow.

---

## 0. Checked against the official docs (2026-09-21)

All six documents in `official_doc/` were searched (alternative, clash, taken,
slot hold / time-to-live, double-booking, nearest, dead end). There is **no
backlog story with this title**; it is a custom story. What the docs DO say,
and what it changes in this plan:

| Document / story | What it says | Effect on this plan |
|---|---|---|
| **User Stories E4-S1** "Booking with slot locking" (P0) + **Gap Analysis E4-S1: PARTIAL** | *"A slot is held with a time-to-live when offered, committed on confirmation and released on abandonment."* Gap Analysis: *"no slot hold, no time-to-live and no locking, so two concurrent callers can be offered and given the same slot. Solution: a slot hold table with an expiry ... SELECT FOR UPDATE."* | **Not built here** (still out of scope -- it is its own P0 story). AC3 of this story ("free at the moment of speaking") is met by computing from the live tables in the same request. But: `_slot_alternatives` becomes the ONE place that decides "free", so when E4-S1 adds holds, a slot held by another caller is excluded there and nowhere else (**D8**). |
| **Architecture & Implementation Plan**, reply_templates table | *"booking_reply(...) -- Confirmation number, or alternative slots on a clash."* clinic-api table: *"book_appointment(req) -- validates the slot against generated 15-minute slots, checks clashes, writes."* | Keep the documented homes: the write still validates + checks clashes; **booking_reply stays the function that offers alternatives on a clash** -- its slot_taken branch uses the new one-sentence wording instead of a separate path (**D9**). The architecture doc's line stays true. |
| **User Stories E12-S5** "Dead ends always offer a next step" (P0) + **E12-S12** tone tests | *"No reply ends on a bare apology. A not-found offers the nearest alternatives, a transfer to a human, or a callback."* | **Found a dead end in today's code:** a taken slot with nothing free nearby says *"ওই সময়টা বুক হয়ে গেছে, এবং কাছাকাছি কোনো সময় ফাঁকা নেই।"* and stops. D7 changes: nothing free that day or at that time on another day -> offer the doctor's **earliest free slots** (the earliest-slot story's offer), and if even that is empty, its callback. |
| **User Stories E13-S4** "Confirm the whole booking before committing" (P0) | Read back the complete appointment and require assent before the write. | A picked alternative never skips the readback -- it re-enters the existing flow (already the plan; now stated as a rule). |
| **User Stories E13-S2 / E13-S9** never repeat a prompt; cap questions before escalating | | Repeated "taken" rounds are capped by the existing retries counter (3), then the existing "not booked" ending. |
| **User Stories E2-S4E** the reply mirrors the caller's language mixture | | The sentence exists in all four languages, chosen from the caller's own turn (detect_language), like every booking reply. |
| **Blueprint 4.9 / E4-S2** idempotent writes: *"call_id + action_id ... No double-booked appointment"* | The booking write has **no idempotency key** (reschedule and cancel do). | Not changed here, but a RISK: a retried "হ্যাঁ" after a network blip would hit its own booking as "taken" and be offered alternatives. Listed under Risks; belongs to E4-S2. |
| **User Stories E8-S1** availability read live | | Same as the earliest-slot story: live = clinic-api's tables on every question, never cached. |

---

## 1. What the code does today (checked by running it)

Seeded clinic-api, Dr. A. Sen already booked at 10:00 and 11:15 on Tue 22 Sep:

```
caller asks 11:15 (taken)        -> offered 10:15, 10:30, 10:45
caller asks 11:10 (not a slot)   -> offered 10:00, 10:15, 10:30   <- 10:00 is BOOKED
```

| # | Finding | Where |
|---|---|---|
| F1 | Alternatives are the first 3 free slots from the **start** of the session, not the nearest to the time asked. | clinic-api/main.py:846-848, and `_free_slots` on the race branch (:868) |
| F2 | No "same time on another day" alternative anywhere. | -- |
| F3 | **Bug:** a time that is not one of the doctor's 15-minute slots gets `valid_slots[:3]` with **no check for bookings** -- a booked slot is offered. | clinic-api/main.py:838 |
| F4 | "Taken" is discovered only at the **write**, after name, phone and the readback's "হ্যাঁ". The earlier check in `_open_booking` asks only "does he sit that day / is it inside his hours". | main.py `_open_booking`; clinic-api `/doctors/availability` |
| F5 | Already built (previous story, fix 1): after a refused write the booking stays open and "প্রথমটা" / a time picks from the offered slots (`_pick_offered_slot`), name and phone kept. Same-day only. | main.py `_finish_booking`, `_pick_offered_slot` |
| F6 | `parse_date("বৃহস্পতিবারেরটা")` -> None (the possessive ending defeats the word boundary); "তৃতীয়টা" -> ordinal 2 and "এগারোটা" -> 11:00 already work. | agent/slot_parse.py |

## 2. Decisions (light)

| # | Decision |
|---|---|
| D1 | **One clinic-api helper** decides the alternatives for every "taken" answer: `_slot_alternatives(db, doctor, date, time)`. It is used by the booking write's refusals AND by the pre-readback check, so both always say the same thing. |
| D2 | **"Nearest on the same day"** = the 2 free slots closest in minutes to the time asked (a tie goes to the earlier); spoken in time order. Today: nothing already passed or inside the 30-minute lead. |
| D3 | **"Same time on the nearest other day"** = the nearest other day (either side of the asked day, never in the past, within `EARLIEST_SLOT_HORIZON_DAYS`) on which the doctor sits and that exact time is a free slot. If the asked time is not a slot (11:10), the same rule uses the nearest slot time (11:15). None found -> only the same-day ones are offered. |
| D4 | **Checked when the time is given**, not only at the write: `_open_booking` already asks clinic-api about the day; it now passes the time too and gets "free / taken + alternatives" back. So the caller hears the alternatives before being asked for name and phone. The write still re-checks (nothing is held -- E4-S1). |
| D5 | **Reschedule is NOT changed** (it keeps its earliest-first alternatives). Say if it should follow. |
| D6 | **A plain "হ্যাঁ" does not pick** here (three different options) -- it re-asks "which one?". |
| D7 | **Nothing free that day and that time on no other day** -> NOT the existing dead end (E12-S5): offer the doctor's earliest free slots (earliest-slot story's `_offer_earliest`), which itself ends in a callback offer if the whole horizon is full. |
| D8 | **One definition of "free"** (`_slot_alternatives`), so a future slot hold (E4-S1) plugs in at one place. |
| D9 | **booking_reply keeps its documented job** ("alternative slots on a clash"): its slot_taken branch speaks the new one-sentence wording (it calls `booking_slot_taken_prompt`), and the pre-readback check uses the same function -- one wording for both moments. |

## 3. What changes

### clinic-api/main.py
- NEW `_slot_alternatives(db, doctor_id, date_iso, time_slot)` -> `{"alternative_slots": [HH:MM, HH:MM], "other_day_slot": {"date", "time_slot"} | None}`. Reads ACTIVE appointments at call time (reuses `_schedule_for_weekday`, `_generate_slots`, `_free_slots(limit=None)`).
- `book_appointment`: the three refusal branches (time not a slot -- **F3 fix**; slot taken; race on insert) return `_slot_alternatives(...)`. `alternative_slots` keeps its name and type (a list of times on that day), so `booking_reply` and every existing reader keep working; `other_day_slot` is a new key.
- `GET /api/v1/doctors/availability`: optional `time=HH:MM`. When given and the doctor sits that day: `slot_free: true|false`, and when false the same `alternative_slots` + `other_day_slot`. Without `time` the response is exactly as today.

### agent
- `tools_client.get_doctor_availability(name, date, time=None)` -- one optional parameter; still never cached.
- `reply_templates.booking_slot_taken_prompt(slots, result, language)` -- ONE sentence, 4 languages, every value read off `result`:
  "সাড়ে এগারোটা বুক হয়ে গেছে। সেদিন এগারোটা বা পৌনে বারোটা খালি আছে, অথবা বৃহস্পতিবার, চব্বিশ তারিখ, একই সময়ে সাড়ে এগারোটায়। কোনটা নেব?"
  (1 or 0 same-day, or no other day -> the sentence shortens accordingly.)
- `main.py _open_booking`: pass the time; if `slot_free` is false -> drop the time, keep everything else, `awaiting="time_slot"`, `offered_slots` = the same-day ones + the other-day one (each with its date), speak `booking_slot_taken_prompt`.
- `reply_templates.booking_reply`: its slot_taken branch returns `booking_slot_taken_prompt(...)` (D9); the "nothing nearby" dead end is removed from the booking path (D7).
- `main.py _finish_booking` (fix 1 path): unchanged call to `booking_reply` (now the new sentence); `offered_slots` gains the other-day slot; no alternatives at all -> `_offer_earliest` (D7).
- `main.py _pick_offered_slot`: also accept the other day by its weekday name with any ending ("বৃহস্পতিবারেরটা", "বৃহস্পতিবার") and "পরের দিনেরটা / অন্য দিনেরটা" when exactly one offered slot is on another day; a plain "হ্যাঁ" with several options -> no pick (D6).
- Everything after the pick is the existing flow: `_open_booking` re-check (so a slot taken in the meantime produces fresh alternatives -- AC3), name + phone, readback, write.
- `main_pcm.py` regenerated; punctuation-gate cases for the new template.

### Not changed
Intent, dialogue states, database tables, the write's double-booking guard (unique index), reschedule, cancellation, earliest-slot and callback flows.

## 4. How each criterion is met

| AC | Mechanism |
|---|---|
| two nearest on the same day + same time on the nearest other day | `_slot_alternatives` (D2, D3) |
| in one sentence | `booking_slot_taken_prompt` |
| accept one by saying which | `_pick_offered_slot`: "প্রথমটা/দ্বিতীয়টা/তৃতীয়টা", the time, the weekday name, "পরের দিনেরটা" |
| nothing offered that is not free at the moment of speaking | computed from the live tables in the same request that produces the sentence; the pre-readback check (D4) and the write both recompute; F3 bug fixed |

## 5. Risks

- Two callers can still be offered the same alternative (no hold, E4-S1); the loser hears fresh alternatives at the re-check or the write.
- The pre-readback check adds a little to one existing clinic-api read (same call, one more query).
- A long sentence when all three alternatives exist -- listen on the pod.
- No idempotency key on the booking write (E4-S2 / Blueprint 4.9): a retried confirmation would be treated as a clash with the caller's own booking.
- Until E4-S1 exists, "free at the moment of speaking" is true when spoken but not guaranteed at the caller's pick; the re-check and the write catch it.
- The caller naming a time that is not in the offer (e.g. "বারোটা") is read as a normal time answer and checked again.

## 6. Tests (own file, deletable)

clinic-api: nearest two on each side of the asked time; ties; the asked time not a slot (11:10) never offers a booked slot (F3); other day nearest either side, skipping days he does not sit and days that time is booked; nothing today inside the lead; `/doctors/availability?time=` free and taken shapes; no `time` -> unchanged response.
Agent: a sentence naming a taken time -> the one-sentence offer BEFORE name/phone; "দ্বিতীয়টা", the time, "বৃহস্পতিবারেরটা", "পরের দিনেরটা" each pick right, then name + phone -> readback -> the row; "হ্যাঁ" re-asks; an alternative taken before the pick -> fresh alternatives; taken between readback and write -> the same sentence, name/phone kept -> booked; day full -> existing path; clinic-api down -> SYSTEM_UNREACHABLE_BN; one-sentence, doctor-only and earliest flows unchanged. Full suite vs baseline (277 failed / 3910 passed / 1 skipped), no new gate offender.

## 7. What was built (2026-09-21)

As planned (D1-D9), with no new intent, state or table.

| File | Change |
|---|---|
| clinic-api/main.py | NEW `_hhmm_minutes`, `_slot_alternatives` (the one definition of what may be offered); the write's three refusal branches return it (**F3 bug fixed** -- a time that is not a slot no longer offers a booked one); `/doctors/availability` takes an optional `time` -> `slot_free`, and when false the alternatives. Without `time` the response is unchanged. |
| agent/tools_client.py | `get_doctor_availability(name, date, time=None)`. |
| agent/reply_templates.py | NEW `booking_slot_taken_prompt` (ONE sentence, 4 languages, every alternative clinic-api returned is spoken); `booking_reply`'s slot_taken branch delegates to it (D9). |
| main.py | imports (`unicodedata` + the template); `_open_booking` passes the time and, when not free, speaks the offer BEFORE name/phone (capped at 3 rounds; nothing at all -> the earliest-slot offer, D7); `_pick_offered_slot` also takes the other day by weekday name with any ending or "পরের দিনেরটা", and a plain "হ্যাঁ" does not pick among taken alternatives (D6); `_taken_offer`; `_finish_booking` offers the other day too, or the earliest slots when there is nothing. |
| main_pcm.py | regenerated. |
| tests/test_spoken_punctuation.py | 3 cases (taken/*). |
| tests/test_slot_taken_alternatives.py | NEW, 17 tests, deletable as one file. |

What the caller hears (real run, 11:00 booked by someone else):
"সকাল এগারোটা বুক হয়ে গেছে, তবে সেদিন সকাল পৌনে এগারোটা বা সকাল সোয়া এগারোটা খালি
আছে, অথবা বৃহস্পতিবার, সেপ্টেম্বর মাসের চব্বিশ তারিখে একই সময়ে সকাল এগারোটা খালি আছে,
কোনটা নেব?" -> "দ্বিতীয়টা" -> name + phone -> readback -> "হ্যাঁ" -> booked 11:15.

One correction during the build: the first template cut the alternatives
to two, and the existing number-fidelity test (`test_booking_reply_preserves_
alternative_slots`) failed -- the agent must never drop a value clinic-api
returned. clinic-api decides the count; the template now speaks them all.

**Verification.** 17/17 story tests (real in-process clinic-api, seeded
SQLite): nearest two either side; booked neighbours skipped with the tie
rule; a non-slot time never offers a booked one; same time on the nearest
other day, and the next one when that is booked too; the free / no-time
response shapes; the one-sentence offer before name and phone; picks by
ordinal, by time, by weekday name ("…বারেরটা"), by "পরের দিনেরটা"; a plain
"হ্যাঁ" re-asks; an alternative taken before the pick -> fresh alternatives;
taken between readback and write -> same sentence, details kept, other day
picked and booked; nothing near and no other day -> "taken" + earliest
offer -> callback (no dead end); the 3-round cap; clinic-api down ->
SYSTEM_UNREACHABLE_BN; a free time goes straight on as before.
Full suite: 277 failed / 3930 passed / 1 skipped -- the identical 277, none
new; gate messages unchanged. Production files changed: clinic-api/main.py,
agent/tools_client.py, agent/reply_templates.py, main.py, main_pcm.py.
