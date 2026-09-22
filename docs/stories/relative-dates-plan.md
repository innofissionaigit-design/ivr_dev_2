# Caller says tomorrow, day after, or next Monday -- implementation plan

**Epic:** Conversation: Booking, Rescheduling and Cancellation
**Story:** Caller says tomorrow, day after, or next Monday
**User story:** As a caller who speaks in relative dates, I want them
understood, so that I do not have to work out a calendar date.
**Acceptance criteria:**
1. Relative expressions resolve against the call date in every supported
   language, including tomorrow, day after tomorrow, this coming weekday and
   next week.
2. The resolved absolute date is always read back before use.
3. Ambiguous expressions are clarified rather than assumed.

**Rajarshee's rule (the design):** *"If the LLM and the parser give the same
response, confirm the date with the caller; otherwise ask the date. It is a
universal rule -- for appointments AND availability."*

**Owner:** Rajarshee. **Reviewer:** Saurav.
**Status:** BUILT 2026-09-21 -- see section 7. Not deployed. Kept light: one decision
function, used everywhere a date is read; existing states and templates reused.

---

## 0. Checked against the official docs

No backlog story has this title (custom story). What the docs say:

| Document / story | What it says | Effect |
|---|---|---|
| **Architecture plan + Blueprint -- "the one rule"** | *"The model may decide what the caller wants. It may never decide what is true. Every ... date ... is substituted from a verified system response."* | Kept: the model only NAMES a meaning (`date_expr`); code computes every date on both sides of the comparison. The rule compares two code-computed dates, never a model-written date. |
| **E2-S7 Readback of critical values** (P0; Gap Analysis: PARTIAL) | *"Phone numbers, dates, appointment times and patient names are confirmed aloud before any write."* | This story extends date confirmation from "before a write" to "before ANY use", including availability answers (AC2). |
| **E13-S4 Confirm the whole booking** | The complete appointment read back once. | Unchanged; when the readback is the very next question, it doubles as the date confirmation (keeps the one-sentence booking at two turns -- E13-S7). |
| **E13-S3 Disambiguate when several entities match** | Offer the close matches and ask which. | Same pattern for dates: when the two readings differ, both dates are offered ("২৮ সেপ্টেম্বর, নাকি ৫ অক্টোবর?"). |
| **Architecture plan §6.2 / §9.2 fast path** | The fast path serves availability only when *"any date present is a simple relative day word"*; it abstains on digits or weekday names. | Still true. A fast-path turn never consulted the model, so it is treated as "parser only" (D2). |
| **Architecture plan, llm.py section** | *"SYSTEM_PROMPT_TEMPLATE injects today's date and weekday for relative-date resolution."* | **Out of date** -- the code removed today's date from the prompt ("The model never originates a fact"). Noted as doc drift, not changed. |

## 1. What the code does today (checked by running it)

Today = Monday 21 September 2026.

| # | Finding | Where |
|---|---|---|
| F1 | Tomorrow / day after tomorrow resolve in every script (কাল, আগামীকাল, পরশু, tomorrow, kal, kalke, parso, porshu). | agent/slot_parse.py `parse_date` |
| F2 | A Bengali weekday resolves to the coming one, **ignoring the qualifier**: "আগামী / সামনের / পরের / এই সোমবার" -> all 28 Sep. | `parse_date` |
| F3 | **Latin-script weekdays are not read by code** ("monday", "somvaar", "sombar", "next monday") -- they depend on the model, whose prompt maps "আগামী শনিবার" -> `next_saturday`, which the engine treats as **skip a week**. So "আগামী সোমবার" = 28 Sep but "agami sombar"/"next monday" = **5 Oct**. Same phrase, different date by script; nobody is asked. | agent/llm.py:206, agent/date_calc.py `_next_weekday` |
| F4 | **BUG -- single-question availability / department turns ignore the model's date.** The inline branches of `_dispatch_turn_inner` use `slots["date"]` raw (now the caller's WORDS or null) and never `date_expr`: "ডাঃ সেন কাল আছেন?" (not caught by the fast path) is answered for **today**; "ডাঃ সেন ২৫ তারিখে আছেন?" sends "২৫ তারিখ" to clinic-api, which cannot parse it and answers `found: false` -> the caller hears **the doctor does not exist**. The multi-part path (`_answer_part`) does it right (date_calc.resolve, range confirmation). | main.py:4588 (availability), 4649 (department); also `_resolve_combinable_intent_fragment` 4956 / 4969 |
| F5 | A single day in an availability question is answered straight away -- never read back first. Only a RANGE is confirmed (`date_range_confirm_prompt`, state `confirm_date`). | `_answer_part` |
| F6 | Booking rechecks only a date calculated from a day WORD (E13-S7 `confirm_booking_date`); an explicit "২৫ তারিখ" is confirmed only inside the final readback. "Next week" while booking is dropped and the caller gets the generic day question. | main.py `_booking_slots_from_turn`, `_next_booking_step` |
| F7 | Reusable pieces already exist: `date_calc.resolve_expression`; states `confirm_date` (confirm, then resume availability / department), `availability_date` / `department_date` (ask the day), `confirm_booking_date` (booking); `booking_date_check_prompt`, `date_range_confirm_prompt`; `parse_ordinal`; `multi_intent_needs_separate_flow_reply`. | -- |

## 2. The rule, as code

ONE pure function in `agent/date_calc.py` (the module that owns the calendar):

```
decide(transcript, date_expr, date_words, llm_used, today) -> DateDecision
    parser = slot_parse.parse_date(transcript)                 # code, caller's words
    model  = resolve_expression(date_expr)  or  parse_date(date_words)   # model's MEANING, code's date
```

| Situation | Decision | What the caller hears |
|---|---|---|
| no day said by anyone | **absent** | nothing new (availability: today, as now; booking: the day question) |
| a range ("next week") | **range** | the range read back + "which day?": "আগামী সপ্তাহ মানে ২৮ সেপ্টেম্বর থেকে ৪ অক্টোবর -- কোন দিন চান?" |
| parser and model give the **same** date | **confirm** | "আগামী সোমবার মানে সোমবার, ২৮ সেপ্টেম্বর। ঠিক আছে?" |
| they give **different** dates | **ask**, offering both | "আপনি কি সোমবার ২৮ সেপ্টেম্বর বলছেন, নাকি সোমবার ৫ অক্টোবর?" |
| only one of them gives a date | **ask**, offering that one | "আপনি কি মঙ্গলবার, ২২ সেপ্টেম্বর বলছেন, নাকি অন্য কোনো দিন?" |
| the model said "a day nothing can express" (`other`) | **ask** | the existing "which day?" question |
| the model was **not** consulted (fast path; an answer inside a flow) | parser only -> **confirm** if it read a date | same confirm sentence |

"Ask" and "confirm" both end with the caller choosing or approving a date that CODE produced; the answer is read by the existing parsers (`parse_date`, `parse_ordinal` for "প্রথমটা/দ্বিতীয়টা", `is_affirmative`/`is_negative`).

## 3. Decisions

| # | Decision |
|---|---|
| D1 | **One function, every flow.** `date_calc.decide` is called wherever a spoken date becomes a calendar date: booking, availability, doctors-by-department, both dispatchers. No flow resolves a date any other way. |
| D2 | **When the model was not consulted** (fast path, or the caller's answer inside a flow -- the model is never called mid-flow), the parser alone decides and a read date is CONFIRMED. Treating "no model" as disagreement would make every fast-path "কাল" a question. **[sign-off]** |
| D3 | **Code reads Latin-script weekdays too** (monday, somvaar, sombar, ... -> the coming one), so Latin callers can reach "agree". The parser keeps ignoring qualifiers ("আগামী", "next", "পরের") on purpose: where the model reads a qualifier as "a week later", the two DISAGREE and the caller is asked -- that is exactly how "next Monday" gets clarified instead of assumed. |
| D4 | **The model prompt is aligned** so a plain coming weekday agrees: "আগামী / সামনের / এই শনিবার", "this saturday", "is shanivaar" -> `saturday`; only an explicit "the week after" ("পরের সপ্তাহের শনিবার", "saturday of next week", "agle hafte shanivaar") -> `next_saturday`. English "next Saturday" is left to the model; if it says `next_saturday` the two disagree and the caller chooses. |
| D5 | **Explicit dates are covered too** ("২৫ তারিখ" -> "২৫ তারিখ মানে শুক্রবার, ২৫ সেপ্টেম্বর। ঠিক আছে?") -- the rule is universal. |
| D6 | **Booking keeps two turns for the one-sentence case**: when the readback is the very next question, it IS the confirmation (it names the day and date) -- as E13-S7 already does. |
| D7 | **Availability costs one more turn** when a day is named (confirm, then answer) -- the price of AC2 "always read back before use". Asking about today with no day named is unchanged. **[sign-off]** |
| D8 | **Reschedule is not changed**: its readback ("... সরিয়ে ... করা হবে। ঠিক আছে?") already reads the new date back before the write. |
| D9 | **Combined multi-question answers** (`_resolve_combinable_intent_fragment`) cannot ask or confirm inside a joined sentence: a question that names a day returns the existing "please ask that one separately" fragment instead of a guessed day. |
| D10 | "No" to a confirmation -> the day is asked again (existing states); a new day goes through the same rule. Retries capped by the existing counters. |

## 4. What changes (light)

| File | Change |
|---|---|
| agent/date_calc.py | NEW `DateDecision` + `decide(...)` (section 2). Pure; no I/O. |
| agent/slot_parse.py | Latin weekday names in `_match_day_word` (-> the coming weekday). |
| agent/llm.py | The `date_expr` examples (D4). |
| agent/reply_templates.py | NEW `date_check_prompt(date_iso, said, language)` ("{what they said} মানে {weekday}, {date}। ঠিক আছে?") and `date_ask_prompt(candidates, language)` (0, 1 or 2 dates), 4 languages. `date_range_confirm_prompt` gains the "which day?" ending for booking. Existing templates unchanged. |
| main.py -- availability + department (both dispatchers) | Replace the raw `slots["date"]` (F4) and the `_answer_part` span logic with `decide`: absent -> today; confirm -> existing `confirm_date` state (resume the intent on "হ্যাঁ"); ask -> existing `availability_date` / `department_date` state with the candidate dates; range -> existing range confirmation. **Fixes F4.** |
| main.py -- booking | `_booking_slots_from_turn` uses `decide`; confirm -> the existing `confirm_booking_date` recheck (now also for explicit dates, using `date_check_prompt` when no day word was said); ask -> the `date` state with `date_candidates` (the tail accepts "প্রথমটা/দ্বিতীয়টা" among them); range -> read back + which day. Answers inside the flow (parser only) are confirmed as in D2, except a date the AGENT offered and the caller took with "হ্যাঁ". |
| main.py -- fragments | D9. |
| main.py -- `_resolve_intent` | marks a fast-path result so `decide` knows the model was not used (D2). |
| main_pcm.py | regenerated. |
| tests/test_spoken_punctuation.py | cases for the two new templates. |

No new intent, no new dialogue state, no clinic-api change, no table.

## 5. Risks

- **More questions.** Availability with a named day gains a turn (D7); a phrase where the model and parser disagree gains a question. That is the AC's price; the plan keeps booking's one-sentence path at two turns.
- **The model is untested here** (no Qwen in this environment): how often it agrees with the parser on real sentences must be measured on the pod. Disagreement is always SAFE (the caller is asked), never a wrong date.
- **"kal" in Hindi** means both yesterday and tomorrow; in a booking/availability question both sides read "tomorrow" and the confirmation names the date, so a caller who meant otherwise can say "না".
- **Docs drift** (§0): the architecture plan still says the prompt carries today's date.

## 6. Tests (own file, deletable)

`decide` unit table (fixed "today" = Monday and = Wednesday): agree / disagree / one-sided / range / other / fast path / parser-only, in Bengali, English, Hinglish, Banglish. Agent end to end against the real clinic-api: availability "কাল" -> confirm -> "হ্যাঁ" -> the answer for TOMORROW (F4); "২৫ তারিখে" -> confirm -> the right answer (not "doctor not found"); "next monday" with the model saying `next_monday` -> both dates offered -> "দ্বিতীয়টা" -> answered for 5 Oct; "আগামী সপ্তাহে" -> range + which day; the same for booking, ending in the row in the database; department questions; "না" to a confirmation -> the day asked again; the one-sentence booking still two turns; multi-part and combined answers; clinic-api down -> SYSTEM_UNREACHABLE_BN. Full suite vs the baseline (277 failed / 3913 passed / 1 skipped), no new gate offender.

## 7. What was built (2026-09-21)

As planned, with no new intent, dialogue state, clinic-api change or table.

| File | Change |
|---|---|
| agent/date_calc.py | NEW `DateDecision`, `decide()` -- the one rule. `resolve()` / `resolve_expression()` unchanged. |
| agent/slot_parse.py | `_WEEKDAYS_LATIN` (English / Hinglish / Banglish weekday names) read by `_match_day_word`. |
| agent/llm.py | the `date_expr` examples: a coming weekday -> `saturday`; only "the week after" -> `next_saturday`. |
| agent/reply_templates.py | NEW `date_ask_prompt`, `date_range_pick_prompt`; `booking_date_check_prompt` reads a calendar date back with its weekday ("শুক্রবার, সেপ্টেম্বর মাসের পঁচিশ তারিখ। ঠিক আছে?"). |
| main.py | NEW `_pick_date_candidate`, `_date_check_slots`, `_date_gate` (availability / department: both dispatchers -- **fixes F4**), `_date_answer` (the answer to "which day?" is confirmed or picked); booking uses `decide` in `_booking_slots_from_turn`, `_next_booking_step` asks the open date question, the field tail takes a pick / a weekday inside "next week" and rechecks a calendar date the caller gave; the doctor-only offer stands aside while a date question is open; "না" to a date check asks the day again (`confirm_date` excluded from the universal "না" hatch); `confirm_date` added to the confirming states; combined answers send a dated question to its own turn. |
| main_pcm.py | regenerated. |
| tests/test_spoken_punctuation.py | 5 cases (dates/*). |
| tests/test_relative_dates.py | NEW, 27 tests, deletable as one file. |

**Heard in the real run** (today = Monday 21 September):
"ডাঃ সেন কাল আছেন?" -> "আগামীকাল মানে মঙ্গলবার, সেপ্টেম্বর মাসের বাইশ তারিখ। ঠিক আছে?" -> "হ্যাঁ" -> clinic-api is asked about 22 Sep
(before this story: about TODAY). "Dr Sen next monday e achen?" (model: `next_monday`) -> "আপনি কি সোমবার, সেপ্টেম্বর মাসের আটাশ
তারিখের কথা বলছেন, নাকি সোমবার, অক্টোবর মাসের পাঁচ তারিখের?" -> "দ্বিতীয়টা" -> 5 Oct.

**A test-environment lesson.** The callback tests depend on the real clock
(clinic opening hours), so the morning baseline did not hold at 21:30: a
callback test "failed" identically with and without this change (proven by
swapping the original files back). The comparison was therefore re-run
against a fresh baseline taken at the same time, with the original files.

**Verification.** 27/27 story tests (real in-process clinic-api): the rule
as a 14-row table (Bengali, English, Hinglish, Banglish, fast path; Monday
and Wednesday as "today"); availability "কাল" -> confirm -> answered for
tomorrow; a calendar date -> confirm -> answered (no longer "no such
doctor"); "next monday" -> both dates -> the pick answered; parser-only ->
offered; "না" -> the day asked again -> the new day rechecked; next week ->
range read back; no day -> today straight away (unchanged); department;
clinic-api down; booking: tomorrow rechecked, "next <weekday>" -> asked ->
picked with no extra recheck, "next week" + a weekday -> that weekday IN next
week, one-sentence booking still two turns with the weekday in the readback.
Full suite: 277 failed / 3945 passed / 1 skipped -- the identical 277 as the
same-clock baseline; gate messages unchanged. Production files changed:
agent/date_calc.py, agent/slot_parse.py, agent/llm.py, agent/reply_templates.py,
main.py, main_pcm.py.

**Noticed, not changed:** after a confirmed future date the existing
`doctor_availability_reply` still ends "আজকের জন্যই অ্যাপয়েন্টমেন্ট করবেন, নাকি অন্য কোনো
দিনের জন্য?" ("today's") -- pre-existing wording in that template.

## 8. Official-docs alignment check and the E13-S9 fix (2026-09-21)

Checked the build against the official docs: aligned on the one rule
(the model names meaning, code computes dates), E2-S7, E13-S4, E13-S3,
E12-S5 and the fast path's scope. Two gaps were found; one is fixed here:

**E13-S9 "Cap the number of questions before escalating" -- FIXED.** A caller
who kept rejecting the date check looped forever (check -> "না" -> which
day? -> new day -> check ...), because each round reset the retry counter.
Now the THIRD rejection hands the call to a human, the same honest
escalation the unclear / out-of-scope paths use (`record_human_handoff` +
`human_fallback_reply`: "আমি আপনাকে আমাদের একজন এক্সপার্টের সাথে কানেক্ট করে দিচ্ছি।").
- availability / department: `date_rejections` travels on the pending state
  (`_date_answer` carries it to the next confirmation; the `confirm_date`
  rejection branch counts);
- booking: counted on the booking's slots (the pending dict is rebuilt when
  the doctor's day is re-checked); both a plain "না" and "না, <another day>"
  count; "the same day again" is a yes and does not;
- a "হ্যাঁ" within the cap carries on exactly as before.
(Also: "না, <a calendar date>" in booking is now rechecked like any other
calendar date the caller gives.)
Tests: +3 (two rejections continue, the third escalates -- availability and
booking; a yes after two rejections still answers). 30/30. Full suite 277
failed / 3948 passed / 1 skipped -- the identical 277; gate messages unchanged.

**Still open (not asked to fix):** E2-S4E -- the "next week" range
confirmation reuses `date_range_confirm_prompt`, which exists only in
Bengali.
