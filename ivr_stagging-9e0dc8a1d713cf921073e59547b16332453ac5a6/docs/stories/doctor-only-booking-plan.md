# Caller names only a doctor -- implementation plan

**Epic:** Conversation: Booking, Rescheduling and Cancellation
**Story:** Caller names only a doctor
**User story:** As a caller who knows the doctor but nothing else, I want to be
guided through the rest, so that I do not need to know how the clinic is
organised.
**Acceptance criteria:**
1. The agent confirms the doctor and offers the next available sittings.
2. The agent collects date, time, patient and contact in grouped questions.
3. If the doctor has no sitting in the requested window the agent says so
   and offers the nearest alternative rather than a bare refusal.

**Owner:** Rajarshee. **Reviewer:** Saurav.

**Backlog mapping:** this exact title and wording do not appear in
`official_doc/KolkataCareVoiceAgentUserStoriesCodeMixed.docx` -- I checked
(the 175-story backlog, all 180 tables) and found no row matching "names
only a doctor", "guided through the rest", "next available sitting(s)" or
"nearest alternative". This is a custom story, written directly for this
tree. It sits closest to two backlog stories already used as reference
points on this epic:
- **E13-S1** "Ask for several fields in one turn" -- *"Related fields are
  grouped into a single natural question... Booking completes in at most
  three question turns"* -- this is AC2, almost verbatim.
- **E13-S6** "Question ordering by information gain" -- *"a doctor who does
  not exist or a date the clinic is closed is caught before name and phone
  are collected"* -- the fail-fast shape behind AC1 and AC3.

Because there is no official ID, the plan file is named by slug
(`doctor-only-booking-plan.md`), not `E##-S#-...`, to avoid implying a
backlog number that does not exist.

---

## 1. What the code does today (checked, not assumed)

The shared booking helpers built for E13-S7 (`docs/stories/E13-S7-one-sentence-booking-plan.md`)
already do most of the work this story needs -- **just not for the
doctor-only case**:

| # | Finding | Where | Effect on a caller who names only a doctor |
|---|---|---|---|
| F1 | `_open_booking` calls `_tools.get_doctor_availability(doctor, merged.get("date"))`. When no date was given, `merged.get("date")` is `None`, and clinic-api's own `doctor_availability` endpoint (clinic-api/main.py:647-661) **already** computes the answer to "when is this doctor next available" for exactly that case -- it returns `available: true`, `date: <next sitting day>`, `chamber_hours: "..."`. But `_open_booking`'s own branching only *reads* `result` when `merged.get("date")` is truthy (main.py:1298, 1305) -- so this computed answer is thrown away. | `_open_booking`, main.py:1262-1318 | The caller who says only "ডাঃ সেনের কাছে অ্যাপয়েন্টমেন্ট করতে চাই" gets no acknowledgement that the doctor exists or is in soon -- the flow falls straight through to `_next_missing`, which returns `"date"`, and the caller is asked a bare "আজকের জন্য চান, নাকি অন্য কোনো দিনের জন্য?" with nothing about the doctor at all. |
| F2 | The exact fact-plus-question this story wants (confirm the doctor, state chamber hours or the next sitting day, ask today-or-another-day) **already exists** as `doctor_availability_reply()` (agent/reply_templates.py:940) -- but it is wired only to the standalone `doctor_availability` intent (main.py's `awaiting == "date"` branch at line ~3243, used when the caller *asked* "is Dr Sen in"), never to the `book_appointment` intent's doctor-only path. | agent/reply_templates.py:940-986 | The identical sentence the caller would hear by asking "is Dr Sen in?" first is not spoken when they instead go straight to booking with the doctor's name only -- two callers with the same information end up with two different experiences. |
| F3 | Once the doctor is resolved, every remaining field is asked **one at a time**: `_next_missing` (main.py:939) returns exactly one of `doctor_name`, `date`, `time_slot`, `patient_name`, `phone`, in that fixed order, and `missing_slot_prompt` (agent/reply_templates.py:363) has one question per field, nothing grouped. `_continue_pending`'s tail for these states (main.py:3357-3392) parses the reply with exactly one parser (`parse_date` **or** `parse_time` **or** `parse_phone` **or** `_clean_patient_name`), never two. | `_next_missing`, `missing_slot_prompt`, `_continue_pending` tail | Doctor-only entry costs the maximum turn count: date, then time, then patient name, then phone -- four separate questions, each answerable in the same breath as its natural pair. |
| F4 | A day the doctor does not sit, or a time outside chamber hours, already gets the "nearest alternative" treatment, not a bare refusal: `booking_day_unavailable_prompt` offers `next_available_date` and lets a bare "হ্যাঁ" take it; `booking_time_outside_hours_prompt` states the real chamber hours (main.py:1298-1310, agent/reply_templates.py:3311-3348). This is F5 from the E13-S7 plan, already built and tested. | `_open_booking` | **AC3 is already satisfied for the case where a date/time was given and rejected.** The only gap is the doctor-only entry point covered by F1/F2 above, where no date was offered in the first place for there to be a rejection of. |
| F5 | When a doctor genuinely has no `DoctorSchedule` row inside the 14-day search horizon, `doctor_availability_reply`'s fallback branch says "এখন কোনো নির্দিষ্ট দিন বসছেন না। আমাদের কাউন্টারে খোঁজ নিতে পারেন" (agent/reply_templates.py:981-986) -- a next step, not a bare apology. No change needed here; noted for completeness against AC3. | agent/reply_templates.py:940 | Already compliant. |

**The property that matters most:** the caller who gives only a doctor's
name should hear, in one turn, everything a caller who *asked* "is Dr Sen
in?" would hear -- then be asked for the rest two fields at a time, not one.

## 2. Decisions

| # | Decision | Reasoning |
|---|---|---|
| D1 | Reuse the **existing** clinic-api response (`get_doctor_availability` with `date=None`) for the doctor-only confirm-and-offer step. No new clinic-api endpoint. | The endpoint already computes exactly this (`_next_available_date`, F1). A new endpoint would duplicate `_next_available_date`'s horizon-search logic and create a second source of truth for "when is this doctor next in". |
| D2 | Grouping is **pairwise and conditional**, not all-or-nothing: `date`+`time_slot` are asked together **only when both are still empty**; `patient_name`+`phone` likewise. If exactly one of a pair is already known (e.g. the caller gave a time but not a date), the existing single-field prompt for the one missing field is used unchanged. | Preserves every already-tested single-field continuation path (date-only, time-only, name-only, phone-only) byte-for-byte. Only the *both-empty* case is new, so nothing that currently passes can regress. |
| D3 | The grouped reply is parsed with the **same local parsers** already used elsewhere (`parse_date`, `parse_time`, `_clean_patient_name`, `find_phone_in_sentence`/`parse_phone`), run independently against the caller's one utterance -- exactly the technique `_booking_slots_from_turn` already uses for a whole one-sentence booking. No LLM re-extraction inside a continuation; `_continue_pending` never calls the model mid-flow today and this does not start. | Consistent with the existing trust model (agent/llm.py's role stops at the opening classification); avoids adding a new place where the model could originate a date/time/name fact. |
| D4 | If only one half of a grouped reply parses, keep it and fall back to asking the other half **alone**, via the existing single-field state (`awaiting="date"` or `"time_slot""`/`"patient_name"`/`"phone"`) and its existing retry counter. If a bare affirmative ("হ্যাঁ") answers the grouped date+time question, it takes the **offered date** exactly as the plain "date" tail already does (`took_offer`, main.py:3383) and still asks for the time alone. If neither half parses, the SAME grouped question is re-asked, capped by the existing `retries > 2` -> abandon pattern used throughout `_continue_pending`. | No new retry/abandonment policy; reuses the one everywhere else in this function. |
| D5 | Two new **states** are added to the `awaiting` vocabulary: `"date_time"` and `"patient_contact"`. Both are added to `_BOOKING_STATES` (main.py:1115) and to `skip_zones` alongside `"confirm_booking_date"` (per the E13-S7 pattern), so the turn counter and the AST "no repeated prompt within a call" gate (E13-S2, test_outcome_distinction-style checks) both see them correctly. | Mirrors exactly how `"confirm_booking_date"` was added for E13-S7; no new mechanism invented. |
| D6 | Three new reply_templates functions, each in the existing 4-language shape (english/hinglish/banglish/bengali) used by every neighbour in that file: `booking_confirm_doctor_prompt(slots, result, language)`, `booking_ask_date_time_prompt(language)`, `booking_ask_patient_contact_prompt(language)`. No existing template's wording changes. | Keeps the "every spoken span is a literal or a named reply_templates call" gate satisfied, and keeps every currently-passing `test_spoken_punctuation` case untouched. |
| D7 | Test file for this story lives on its own (`tests/test_doctor_only_booking.py`, plus a `tests/doctor_only_fixtures.py`-style module if fixtures are needed), matching the pattern the user asked for since E4-S3: deletable as a set once the story is verified, without touching `conftest.py` or any other story's tests. | Standing instruction; see memory `ivr-story-workflow`. |

## 3. Design (first draft -- SUPERSEDED by section 7)

> Rajarshee asked for a lighter build: once the doctor is named, the
> EXISTING booking path carries the caller through. So no `date_time` /
> `patient_contact` states and no `_next_missing_group` were built. What
> was actually built is section 7.

### 3.1 The new turn shape

```
CALLER  ডাঃ সেনের কাছে একটা অ্যাপয়েন্টমেন্ট করতে চাই
          -> book_appointment, slots = {doctor_name: "সেন"}, nothing else
          -> _open_booking: doctor resolves via get_doctor_availability(name, date=None)
AGENT   হ্যাঁ, ডাঃ সেন সেপ্টেম্বর মাসের বাইশ তারিখে চেম্বারে থাকবেন, সময় সকাল দশটা
        থেকে দুপুর একটা। ওই দিনেই করব, নাকি অন্য কোনো দিনে, এবং কোন সময়ে চান?
          -> booking_confirm_doctor_prompt(); awaiting="date_time", offered_date=result["date"]
CALLER  ওই দিনেই, সকাল সাড়ে দশটায়                      (both parse: date=offered, time=10:30)
          -> or: হ্যাঁ                                    (bare affirmative -> takes offered date only)
          -> or: বুধবার                                  (date parses, time doesn't -> ask time alone)
AGENT   [if time still missing] কোন সময়ে আসতে চান?        (existing missing_slot_prompt, unchanged)
          -> awaiting="patient_contact" once date+time are both known
AGENT   রোগীর নাম আর একটা ফোন নম্বর বলবেন, যাতে কনফার্ম করতে পারি?
          -> booking_ask_patient_contact_prompt(); awaiting="patient_contact"
CALLER  রাহুল দাস, নম্বর ৯৮৭৬৫৪৩২১০
          -> both parse -> _next_booking_step -> readback (booking_confirmation_prompt, unchanged)
```

Doctor-not-found, doctor-ambiguous, day-unavailable and time-outside-hours
all keep their current behaviour exactly (F4/F5) -- this story only adds the
confirm-and-offer step for the *found, date-not-yet-given* case, and groups
the two field-pairs that follow it.

### 3.2 Where each change lands

**`_open_booking`** (main.py:1246-1318) -- after the doctor resolves
(`result.get("found")` true, not ambiguous) and *before* the existing
`if merged.get("date") and not result.get("available")` check:

```python
if not merged.get("date"):
    offered_date = result.get("date") if result.get("available") else result.get("next_available_date")
    pend("date_time" if not merged.get("time_slot") else "date")
    await _speak(session, booking_confirm_doctor_prompt(merged, result, language=language))
    return
```

`booking_confirm_doctor_prompt` internally branches on whether
`slots.get("time_slot")` is already set (same style as
`doctor_availability_reply`'s own `if result.get("available")` branch) to
choose its trailing clause: the combined "ওই দিনেই করব... এবং কোন সময়ে চান"
when time is also missing, or a plain date-only trailing clause
("ওই দিনেই করব, নাকি অন্য কোনো দিন?") when a time was already given and only
the date needs settling. One template function, one `_speak` call, no
branching at the call site -- consistent with the gate that spoken spans are
literals or a named `reply_templates` call.

**`_next_missing`** (main.py:939-945) gets a sibling, not a replacement:

```python
def _next_missing_group(slots: dict) -> str | None:
    """Like _next_missing, but the two adjacent pairs in _BOOKING_FIELDS
    collapse to one grouped state when BOTH members are still empty."""
    if not slots.get("doctor_name"):
        return "doctor_name"
    if not slots.get("date") and not slots.get("time_slot"):
        return "date_time"
    if not slots.get("date"):
        return "date"
    if not slots.get("time_slot"):
        return "time_slot"
    if not slots.get("patient_name") and not slots.get("phone"):
        return "patient_contact"
    if not slots.get("patient_name"):
        return "patient_name"
    if not slots.get("phone"):
        return "phone"
    return None
```

`_next_booking_step` (main.py:1168-1184) calls `_next_missing_group` instead
of `_next_missing`, and gets one extra branch for the two grouped names:

```python
if missing == "date_time":
    await _speak(session, booking_ask_date_time_prompt(language=language)); return
if missing == "patient_contact":
    await _speak(session, booking_ask_patient_contact_prompt(language=language)); return
```

(`_next_missing` itself stays exactly as-is -- `_open_booking`'s existing
"`if _next_missing(merged) is None`" completeness check at line 1315 does
not need to know about grouping, only about full-vs-not-full.)

**`_continue_pending`** (main.py:3357-3392) gets two new `awaiting`
branches, inserted alongside the existing single-field tail:

```python
if awaiting == "date_time":
    date_val = parse_date(text, offered_date=pending.get("offered_date"))
    time_val = parse_time(text)
    took_offer = (date_val == pending.get("offered_date") and is_affirmative(text))
    if date_val is None and time_val is None:
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None; return False
        await _speak(session, booking_ask_date_time_prompt(language=language)); return True
    if date_val is not None:
        _apply_booking_date(pending["slots"], date_val, None if took_offer else spoken_day_word(text))
    if time_val is not None:
        pending["slots"]["time_slot"] = time_val
    pending["retries"] = 0
    await _next_booking_step(session, pending, language); return True

if awaiting == "patient_contact":
    name_val = _clean_patient_name(text)
    phone_val = find_phone_in_sentence(text) or (parse_phone(text) if text.strip() else None)
    if name_val is None and phone_val is None:
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None; return False
        await _speak(session, booking_ask_patient_contact_prompt(language=language)); return True
    if name_val is not None:
        pending["slots"]["patient_name"] = name_val
    if phone_val is not None:
        pending["slots"]["phone"] = phone_val
    pending["retries"] = 0
    await _next_booking_step(session, pending, language); return True
```

These sit directly above the existing "Remaining states... all share the
same shape" block (line 3357), which keeps handling the single-field
fallbacks (`date`, `time_slot`, `patient_name`, `phone`) completely
unchanged -- a half-answered grouped question routes into exactly those
same states for its second half.

**`_BOOKING_STATES`** (main.py:1115-1117) and the `skip_zones` set gain
`"date_time"` and `"patient_contact"`, next to `"confirm_booking_date"`.

### 3.3 New reply_templates functions (agent/reply_templates.py)

All three follow the existing 4-language pattern (english / hinglish /
banglish / bengali) and existing helpers (`_spoken_doctor_name`) already
used by `doctor_availability_reply` and `booking_day_unavailable_prompt`:

1. `booking_confirm_doctor_prompt(slots, result, language)` -- states the
   doctor is confirmed and either their chamber hours today/that day or the
   next available date, then asks the combined date+time question (or the
   date-only trailing clause when time is already known). FACT_KEYS
   (`date`, `chamber_hours`, `next_available_date`) read off `result` only,
   matching the existing gate.
2. `booking_ask_date_time_prompt(language)` -- literal, no facts: "কবে, কোন
   সময়ে আসতে চান, একটু বলবেন?" (+ english/hinglish/banglish).
3. `booking_ask_patient_contact_prompt(language)` -- literal, no facts:
   "রোগীর নাম আর একটা ফোন নম্বর বলবেন, যাতে কনফার্মেশন পাঠাতে পারি?"
   (+ english/hinglish/banglish).

## 4. Build steps

1. Add the three templates to `agent/reply_templates.py` (section 3.3).
2. Add `_next_missing_group` beside `_next_missing` in `main.py`; switch
   `_next_booking_step` to call it and handle the two grouped branches.
3. Add the doctor-only confirm-and-offer branch to `_open_booking` (3.2).
4. Add the `"date_time"` / `"patient_contact"` branches to
   `_continue_pending`'s tail (3.2), placed before the existing single-field
   block so the two behave as a strict superset of it.
5. Add `"date_time"` and `"patient_contact"` to `_BOOKING_STATES` and to
   `skip_zones`.
6. Regenerate `main_pcm.py` with `python tools/make_pcm_variant.py` --
   never hand-edited.
7. Add one `test_spoken_punctuation` case per new template (3 cases), in
   the existing file, per the standing gate rule -- these stay after the
   story's own tests are deleted, same as E13-S7's 7 cases.
8. New, deletable test file(s): `tests/test_doctor_only_booking.py` (+ a
   fixtures module if the existing `cancel_fixtures.py` / one-sentence
   fixtures cannot be reused as-is). Covers:
   - doctor named alone -> confirm+offer spoken, correct next date/hours.
   - doctor named alone, doctor has no schedule at all -> F5's existing
     "check at our counter" fallback, unchanged.
   - doctor + time (no date) -> date-only trailing clause, not the grouped one.
   - grouped date+time answered together -> both filled, one turn.
   - grouped date+time answered with only a date -> time asked alone next.
   - grouped date+time answered with a bare "হ্যাঁ" -> offered date taken,
     time still asked alone.
   - grouped date+time, neither parses, twice -> abandons per existing cap.
   - grouped patient+contact answered together -> both filled, one turn.
   - grouped patient+contact, only a name given -> phone asked alone next.
   - doctor ambiguous / not found / day unavailable / time outside hours --
     regression only, confirming F4/F5 behaviour is untouched.
   - full baseline diff: production `.py` files unchanged except the ones
     listed in step 1-5; suite compared against the current 277
     failed / 3901 passed / 1 skipped baseline, `--ignore=tests/test_multi_intent_dispatch.py`,
     grepping `^(FAILED|ERROR) tests`.
9. Update `docs/stories/doctor-only-booking-plan.md` (this file) with a
   results section once built, and the `ivr-story-workflow` memory.

## 5. Risks / open questions

- **R1 -- sentence length.** `booking_confirm_doctor_prompt`'s combined
  case ("confirmed + hours/next date + asks date and time together") is the
  longest single spoken sentence added by this story. Worth listening to
  the rendered audio (same method as E13-S7's `onesentence_conversation.txt`
  dump) before calling it done, to check it does not run together or get
  misheard at TTS pace.
- **R2 -- "requested window" is read as a requested date/time, not a
  time-of-day band.** Nothing in `agent/slot_parse.py` currently models
  "morning/afternoon/evening" as a slot (checked; only one incidental
  string match, not a real concept) -- so a caller who says "সকালের দিকে"
  without a specific time still falls through to the plain time-outside-
  hours / grouped-time-ask paths, not a new "window" concept. Flag this to
  Saurav in review in case "requested window" was meant more broadly.
- **R3 -- shared surnames.** Doctor disambiguation (রায়/বসু, from
  E13-S7's session) is unchanged by this story; a doctor-only caller who
  names an ambiguous surname still hits the existing `near_match_prompt`.
- **R4 -- grouped parsing false-accepts.** Running `parse_date` and
  `parse_time` independently against the same sentence for the "date_time"
  state could, in principle, let a phone-number-shaped or date-shaped
  fragment inside a *time* answer be misread as a date (or vice versa).
  `_booking_slots_from_turn` already takes this same risk for the opening
  one-sentence case with no reported false-accepts in the E13-S7 test
  suite; worth a dedicated adversarial case in step 8's test file
  (e.g. a sentence containing two numbers, only one of which is a time).

## 6. Non-goals

- No new clinic-api endpoint (D1).
- No change to the write path, idempotency, or the confirm-before-write
  guard in `_finish_booking` -- this story only changes what is *asked*
  before that point.
- No change to reschedule (E4-S3) or cancellation (E4-S4) flows.
- No change to the fully-specified one-sentence path (E13-S7) except that
  it now shares `_next_missing_group` -- a booking that already has every
  field after the opening sentence hits `_next_missing_group`'s final
  `return None` exactly as `_next_missing` did, so its behaviour is
  unchanged.

## 7. What was built (2026-09-21) -- the light version

Once the doctor is named, the **existing** booking path does the rest
(`_open_booking` -> field questions -> recheck of a calculated day ->
readback -> `_finish_booking`). No new state, no new endpoint, no new
clinic-api call shape.

| Where | Change |
|---|---|
| agent/reply_templates.py (end of file) | 3 new templates, 4 languages each: `booking_doctor_offer_prompt` (doctor confirmed + next sitting + chamber hours + "that day or another? and what time?"; the time clause is dropped when the time is already known; if no sitting in 14 days it says so and asks for another day), `booking_date_time_prompt`, `booking_patient_contact_prompt`. |
| main.py `_open_booking` | Doctor found and no day given -> speak `booking_doctor_offer_prompt`, `awaiting="date"`, `offered_date` = the next sitting (the `date` clinic-api already returns when asked with no date). |
| main.py `_next_booking_step` | Still `_next_missing`. Only the WORDING changes: `date` with no time -> the day+time question; `patient_name` with no phone -> the name+phone question. Otherwise `missing_slot_prompt` as before. |
| main.py `_continue_pending` field tail | Same states. The `date` answer also reads a time (`parse_time`); "হ্যাঁ, দশটায়" takes the offered day + the time (`_starts_affirmative`). The `patient_name` answer also reads a numeral phone (`_split_name_and_phone`; with no numeral phone the name is read exactly as before). Whatever half is found is kept; only the missing half is asked next. A new day or time, when the doctor is known, goes back through `_open_booking`, so a day he does not sit or a time outside hours gets the nearest alternative (existing `booking_day_unavailable_prompt` / `booking_time_outside_hours_prompt`) BEFORE the readback, not a refused write after it. |
| main_pcm.py | Regenerated (`tools/make_pcm_variant.py`). |
| tests/test_spoken_punctuation.py | 5 cases (`doctoronly/*`) for the 3 templates, KEPT (the gate needs them). |

What the caller hears (real templates, real `verbalize()`, real clinic-api
on a seeded DB -- 4 caller turns, booked):

```
CALLER ডাক্তার সেনের কাছে একটা অ্যাপয়েন্টমেন্ট করতে চাই
AGENT  হ্যাঁ, ডাঃ সেন। ওঁর পরের বসার দিন সেপ্টেম্বর মাসের বাইশ তারিখ, চেম্বারের সময় সকাল
       দশটা থেকে দুপুর বারোটা পর্যন্ত। ওই দিনে করব, নাকি অন্য কোনো দিন? আর কোন সময়ে চান?
CALLER হ্যাঁ, ১০:১৫ এ
AGENT  রোগীর নাম আর কনফার্মেশনের জন্য একটা ফোন নম্বর বলবেন?
CALLER রাহুল দাস, নম্বর ৯৮৭৬৫৪৩২১০
AGENT  বুক করার আগে একবার শুনে নিন। ডাঃ সেন, সেপ্টেম্বর মাসের বাইশ তারিখ, সময় সকাল সোয়া দশটা,
       রোগীর নাম রাহুল দাস, ফোন নম্বর নয় আট সাত ছয় পাঁচ চার তিন দুই এক শূন্য। সব ঠিক আছে তো?
CALLER হ্যাঁ                     -> written; "after 4 caller turn(s)" logged
```

A day he does not sit: *"ডাঃ সেন ওই দিন বসেন না। পরের দিন সেপ্টেম্বর মাসের চব্বিশ তারিখে
বসবেন। ওই দিনে করব, নাকি অন্য কোনো দিন বলবেন?"* -> "হ্যাঁ" takes it.

**Verification.** 9 end-to-end scenarios (doctor only; full guided booking
to the DB; day he does not sit -> alternative; time outside hours; only a
time answered; name then phone; phone then name; one-sentence booking
unchanged; unknown doctor unchanged) -- all pass. They were run from a
scratch file and NOT added to the repo. Full suite: 277 failed / 3906
passed / 1 skipped -- the identical 277 failures as before the story (none
new, none fixed), +5 passes = the new punctuation cases. Gate failure
messages are unchanged (no new offender).

**Known limits.** (1) One next sitting is offered, not a list (clinic-api
returns one; listing more would need a second call). (2) The phone is
split off a name answer only when said in numerals; digit WORDS in a name
answer are read as before. (3) "requested window" = a day or a time, not
"morning/evening" (R2). (4) A day re-checked via `_open_booking` costs one
extra clinic-api read per day answer.
