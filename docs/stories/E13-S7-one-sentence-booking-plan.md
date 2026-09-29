# E13-S7 -- Caller gives everything in one sentence: implementation plan

**Epic:** Conversation: Booking, Rescheduling and Cancellation
**Story:** Caller gives everything in one sentence
**User story:** As a caller who already knows what I want, I want to say it all
at once and only confirm, so that a simple booking takes one exchange rather
than five.
**Acceptance criteria:**
1. A caller who states doctor, day, time and patient name in the opening
   utterance is asked only to confirm.
2. Every slot is filled from that single turn, and no question already
   answered is asked again.
3. The confirmation reads back all four values.
4. Median turns for this scenario is two.

**Repo evidence given:** "main_pcm.py books when five slots are present, filled
one per turn."
**Backlog mapping:** E13-S7 "Accept information given ahead of the question"
(*a caller who states everything in the first sentence is asked only to
confirm*); it leans on E13-S4 (the whole booking read back, phone included)
and E13-S6 (a doctor who does not exist or a day they do not sit is caught
before the rest is collected).

---

## 1. What the code does today (checked, not assumed)

A single-sentence path EXISTS in both dispatchers, but it does not work for
the sentences real callers say:

| # | Finding | Where | Effect on a caller |
|---|---|---|---|
| F1 | The main dispatch copies the model's `date` slot straight in. Since the "model never originates a fact" story, `date` holds the caller's WORDS ("১৫ তারিখ") and relative days come back in `date_expr` ("tomorrow"), which this path ignores. | main.py `_dispatch_turn_inner`, `elif intent == "book_appointment"` | "কাল সকাল দশটায় ডাক্তার সেনের কাছে ..." -> asked "কোন দিন?" again. "১৫ তারিখে ..." -> the words "১৫ তারিখ" are read back and sent to the booking API, which cannot parse them. |
| F2 | The same path does not check the time or phone with the local parsers. | same | Model digit slips reach the readback unchecked. |
| F3 | The multi-part path (`_answer_part`) DOES resolve the date and check time/phone -- but prefers `parse_phone(whole sentence)`, which takes the LAST ten digits of the sentence. | `_answer_part` | "আমার নম্বর ৯৮৭৬৫৪৩২১০, কাল সকাল ১০:৩০ এ ..." -> phone read back as 5432101030 (measured). |
| F4 | There is no step that reads an answer to "which doctor?". `_next_missing` can return `doctor_name`, and the correction path can re-open it, but the field tail only parses date / time / name / phone. | `_continue_pending` tail | If the doctor is the missing field, the caller is asked three times and the booking silently dies. |
| F5 | The doctor is never checked before the readback. A surname shared by two doctors (রায়: Dr. N. Roy / Dr. P. Ray; বসু: Dr. A. Basu / Dr. T. Bose), a name that does not exist, or a day the doctor does not sit, is discovered only AFTER the caller said "yes" -- the write is refused and the booking has to be walked again. | `_finish_booking` | Wasted confirmation; for ambiguity, the whole booking is restarted from the doctor. |
| F6 | The two dispatchers read back with different templates (`booking_confirm_prompt` vs `booking_confirmation_prompt`). | both | Two wordings of the same question. |
| F7 | Nothing measures turns per booking. | -- | AC4 cannot be checked. |

The readback itself (`confirm_booking`), the correction path, and the write
(`_finish_booking`, with its `confirmed=True` guard and verification of the
returned booking) are correct and are NOT changed.

## 2. What will change

One shared path for "a turn that is a booking request", used by BOTH
dispatchers:

```
opening sentence
  -> _booking_slots_from_turn(): every field that can be read from THIS sentence
       doctor   : the caller's words (clinic-api does the matching)
       date     : date_calc.resolve(sentence, date_expr) -> one calendar day, or empty
       time     : parse_time(sentence) / the model's value, stored only as HH:MM
       name     : the model's value, filler ("আমার নাম") stripped
       phone    : find_phone_in_sentence() -- exactly one clean 10-digit run --
                  else the model's digits; never "the last ten digits of the sentence"
  -> _open_booking():
       doctor named? ask clinic-api ONCE (doctor availability):
         two doctors fit      -> offer them, ask ONLY the doctor
         no such doctor       -> say so, ask ONLY the doctor
         found                -> keep the canonical name + the Bengali name (for the readback)
         does not sit that day-> say so + their next day, ask ONLY the day
         time outside hours   -> say the hours, ask ONLY the time
       a field still empty?   -> ask ONLY that field (the existing prompts)
       nothing missing        -> the EXISTING readback (confirm_booking)  <- turn 1 ends here
  caller: "হ্যাঁ"             -> the EXISTING write (_finish_booking)     <- turn 2
```

- **F1/F2/F3:** fixed by `_booking_slots_from_turn` + a new
  `slot_parse.find_phone_in_sentence`.
- **F4:** a new `doctor_name` step in `_continue_pending`: the answer (or the
  pick from the offered doctors) goes back through `_open_booking`, so the
  doctor is checked the same way and the readback follows when nothing else
  is missing. This also repairs "the doctor is wrong" in the correction path.
- **F5:** the one clinic-api read in `_open_booking` (no new endpoint).
- **F6:** both dispatchers use `booking_confirmation_prompt` (4 languages).
- **F7:** every booking step counts caller turns on the pending dict; the
  write logs `booking confirmed after N caller turn(s)`, so the median can be
  computed from logs.
- **The model prompt** gets one line telling Qwen that a caller may give
  doctor, day, time, name and phone in one sentence and every slot must then
  be filled from it.

## 3. Decisions

- **D1 -- The phone number stays required.** The criteria name four values;
  the booking needs five. clinic-api requires a phone, and E4-S3/E4-S4 find a
  booking again by phone + name. There is no caller ID in this system (the
  browser/WebSocket call carries none). So a caller who says the four values
  is asked ONLY for the phone, then confirms: three turns. A caller who also
  says the phone is asked only to confirm: two turns. The readback reads all
  five (E13-S4). If the clinic later gets caller ID, the phone can be
  pre-filled and the four-value case also becomes two turns.
- **D2 -- The doctor is checked before the readback, on the opening turn and
  on a doctor answer.** Not in the later date/time/name/phone steps: those
  are unchanged, and the write still re-checks everything.
- **D3 -- Time is checked against the doctor's chamber hours only** (start
  inclusive, end exclusive). Whether the exact slot is free is still decided
  at the write, where it cannot race.
- **D4 -- A value is stored only if it is usable.** A time that is not HH:MM,
  a phone that is not ten digits, a date that is a range: left empty, so that
  field -- and only that field -- is asked.
- **D5 -- "Ask only the missing field" means one field per question**, in the
  existing order doctor, date, time, name, phone. Grouping several into one
  question is E13-S1, a different story.

## 4. Files

| File | Change |
|---|---|
| agent/slot_parse.py | NEW `find_phone_in_sentence()` |
| agent/reply_templates.py | NEW `booking_doctor_not_found_prompt`, `booking_day_unavailable_prompt`, `booking_time_outside_hours_prompt` (4 languages, no label colons) |
| agent/llm.py | one sentence in the `book_appointment` description |
| main.py (+ regenerated main_pcm.py) | NEW `_booking_slots_from_turn`, `_time_within_hours`, `_open_booking`; both booking branches use them; NEW `doctor_name` step; turn counting + log |
| tests/ | NEW tests for this story; the three new templates added to test_spoken_punctuation.py |

## 5. What must not break

- The readback, correction path and write: unchanged code.
- The date / time / name / phone steps: unchanged (only a turn counter added).
- Static gates: every `except ToolCallError` speaks `SYSTEM_UNREACHABLE_BN`;
  every spoken span is a literal or a named template call; fact keys
  (`chamber_hours`, `next_available_date`) are read only off `result`; no
  label colons.
- Full suite compared test by test with the tree before this story.

## 8. Later on 2026-09-19 -- tests deleted, spreadsheet written

- **tests/test_one_sentence_booking.py was deleted on request** (backup in the
  session scratchpad, `deleted_onesentence_tests`). The 7 cases in the
  existing tests/test_spoken_punctuation.py were kept.
- **After the deletion:** 277 failed / 3901 passed / 1 skipped (the same 277 as
  before this story); all 47 production .py files byte-identical; nothing
  outside tests/ references the deleted file (only this document).
- **Full write-up:** row 24 of raj_ivr_story_with_implementation.xlsx, columns
  G-T (column T "Switch": this story has none).

## 7. Follow-up (asked for on 2026-09-19): day words everywhere, and a recheck

**Request:** "আজ, আজকে, কাল, কালকে, আগামীকাল, পরশু, পরশুদিন" must be
understood in this story AND the normal (field-by-field) path, and the agent
must recheck a calculated date with the caller; on approval, the rest of the
pipeline is unchanged.

**What was already there:** the calendar engine (agent/date_calc.py +
slot_parse.parse_date) already understood those Bengali words (whole-word,
so "সকাল"/"বিকাল"/"গতকাল" are not "কাল"). What was missing was the recheck,
and the main booking path not using the engine (fixed in section 2).

**Added:**
- **Latin-script forms** in slot_parse: today/aaj/aj/ajke, tomorrow/kal/kaal/
  kalke/agamikal, day after tomorrow/porshu/porshudin/parso/parson. These are
  whole-word matches, so "kalyan" is not "kal".
- **`slot_parse.spoken_day_word()`:** what the caller's word meant ("tomorrow",
  or a weekday). It shares one matcher with parse_date, so the word that is
  rechecked is always the word the date was computed from.
- **The recheck rule** (main.py `_apply_booking_date`, `_next_booking_step`, state
  `confirm_booking_date`):
  - More still to ask: the agent asks FIRST, "আগামীকাল মানে রবিবার, সেপ্টেম্বর মাসের
    কুড়ি তারিখ। ঠিক আছে?"
  - Nothing left to ask: the READBACK names it ("ডাঃ সেন, আগামীকাল রবিবার,
    সেপ্টেম্বর মাসের কুড়ি তারিখ, …") and its "হ্যাঁ" is the approval. The
    one-sentence case stays at two turns.
  - "হ্যাঁ": the pipeline goes on unchanged (next field / readback / write).
  - "না": only the day is asked again, never "booking dropped".
  - A different day ("না, পরশু"): calculated and rechecked again.
  - "হ্যাঁ, কাল" (the same day again): counts as yes.
  - 3 unclear answers: the booking stops with the usual not-confirmed line.
- **The same rule on every route:** the whole sentence, the answer to "কোন
  দিন?", a correction, a model expression ("next_saturday"). An explicit
  date ("২৫ তারিখে") or "হ্যাঁ" to a day the agent itself offered is not a
  calculation and is not rechecked separately (the readback still reads it).
- **New template `booking_date_check_prompt`** (4 languages), and
  `booking_confirmation_prompt` naming a calculated day. The readback's output is
  unchanged when no day word was used.

**Verified:** tests/test_one_sentence_booking.py now has 65 passing tests (+2
skipped on a Saturday only: tomorrow is a Sunday, when no doctor sits); 7
cases in test_spoken_punctuation.py. Full suite 277 failed / 3966 passed /
3 skipped -- the 277 are exactly the list from before this story, none new.
The new main.py code passes the gate rules (ToolCallError ->
SYSTEM_UNREACHABLE_BN; every spoken span a literal, constant or named
template). main_pcm.py regenerated and verified.

## 6. Results (2026-09-19, offline)

**The two-turn conversation** (real templates, run through the real
`verbalize()`):
```
CALLER  ২২ তারিখে ১০:১৫ এ ডাক্তার সেনের কাছে রাহুল দাসের নামে অ্যাপয়েন্টমেন্ট চাই, নম্বর ৯৮৭৬৫৪৩২১০
AGENT   বুক করার আগে একবার শুনে নিন। ডাঃ সেন, সেপ্টেম্বর মাসের বাইশ তারিখ, সময় সকাল
        সোয়া দশটা, রোগীর নাম রাহুল দাস, ফোন নম্বর নয় আট সাত ছয় পাঁচ চার তিন দুই এক শূন্য।
        সব ঠিক আছে তো?
CALLER  হ্যাঁ
AGENT   (the existing booking confirmation, from the committed clinic-api response)
log     booking confirmed by caller after 2 caller turn(s)
```

**New tests:** tests/test_one_sentence_booking.py, 40 tests, all passing
(self-contained: its clinic-api fixtures are in the same file). Real
main.py dispatcher, real client, in-process clinic-api; bookings read off the
database. Covers:
- **Two turns:** all five values give the readback on turn 1 and the write on turn 2.
- **Days:** a weekday word via `date_expr` and a spoken calendar date are both resolved.
- **Both routes:** a multi-part turn gives the same readback.
- **Only the missing field is asked:** phone (3 turns), time, name, doctor (the old dead end), and a date range.
- **The doctor is checked before the readback:**
  - two doctors fit the name (রায়): those two are offered;
  - no such doctor: that is said, and 3 misses end the booking;
  - a day the doctor does not sit: only the day is asked, and "হ্যাঁ" takes the offered day;
  - a time outside the chamber hours: only the time is asked;
  - clinic-api unreachable: the shared sentence is spoken.
- **The existing pipeline:** the correction path fixes only the doctor and then books; nothing is written without an explicit yes.
- **The phone in a whole sentence (9 cases):**
  - the number said before the time;
  - digit words;
  - spaces, +91 and a leading 0;
  - two numbers, no number, and a date plus a time.
- **Only usable values stored; carried values kept; hours boundaries.**
- **New questions:** all four languages; Bengali speakable.
- **The model-prompt line.**

Plus 4 cases in the existing tests/test_spoken_punctuation.py (all passing).

**Full suite** (`python -m pytest tests --ignore=tests/test_multi_intent_dispatch.py`):
277 failed / 3938 passed. Before: 277 failed / 3894 passed. The 44 extra
passes are the 40 new tests plus the 4 punctuation cases. The 277 failures
are exactly the list from before this story; none are new.

**Gates:** the new code's `except ToolCallError` speaks
SYSTEM_UNREACHABLE_BN with the tool_failure clip; every spoken span in it is
a literal, a module constant or a named template call; `chamber_hours` and
`next_available_date` are read only off `result` (test_fact_provenance's
fact-keys test passes). main_pcm.py regenerated: "reasoning half verified
byte-identical", LF only.

**Known limits:**
- **Surnames shared in the seed:** two doctors share each of the surnames রায় and বসু, with
  the same Bengali alias. The offer reads "রায় নাকি রায়". This is the catalogue's
  problem, not the flow's, and it already exists for availability questions.
- **Earlier questions still take only their own field:** a booking sentence said in answer to an
  earlier "which day?" (a pending availability flow) is still read as that one answer.
- **Later answers are not re-checked:** the day and hours check runs on the opening sentence
  and on a doctor answer, not on later date or time answers (D2). The write
  still refuses a bad day or slot.
- **Not tested live:** Qwen on real sentences, real ASR, the pod.
