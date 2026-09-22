"""No field label, colon or bracket is ever spoken.

story title: Answers sound like a person, not a database row
user story: As a patient, I want to hear a sentence, so that the agent sounds
    like someone at the counter.
acceptance criteria: No field label, colon or bracket is ever spoken and every
    structured value renders as a natural clause in the reply language. An
    automated check fails a build containing a spoken punctuation artefact.

Ported from dev_sourav's TestNoSpokenPunctuationArtifact, Bengali only.

THE CHECK RUNS THROUGH verbalize(), NOT ON THE RAW TEMPLATE, and that is the
whole design. A raw template contains colons that are perfectly fine -- a
chamber hour is "18:00-20:00" and a time slot is "18:30" -- because
bn_normalize.verbalize() turns those into Bengali words before synthesis. The
defect was never the character; it was the character SURVIVING to the
synthesiser. So the assertion is made on the string TTS actually receives,
which is the only place the question has an answer.

Confusing those two is the most likely way someone breaks this later: banning
":" from templates would fail on values that are already handled, and banning
it from nothing would miss the labels. The gate sits at exactly one point.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.bn_normalize import verbalize  # noqa: E402
from agent.reply_templates import (  # noqa: E402
    ANSWER_CHANGED_BN, DEFERRED_PART_BN, NEAR_MATCH_UNCLEAR_BN,
    RESUMING_PART_BN, UNANSWERED_PART_BN, UNSPEAKABLE_ESCALATION,
    unanswered_part_prompt,
    near_match_prompt, booking_confirm_prompt,
    booking_correction_prompt, booking_reply, date_range_confirm_prompt,
    doctor_availability_reply, doctors_by_department_reply, heard_confirm_prompt,
    missing_slot_prompt, test_rate_reply as rate_reply, with_change_notice,
    # story title: Caller moves an existing appointment (E4-S3)
    reschedule_prompt, reschedule_found_prompt, reschedule_pick_prompt,
    reschedule_not_found_reply, reschedule_day_unavailable_reply,
    reschedule_time_prompt, reschedule_confirm_prompt, reschedule_reply,
    reschedule_kept_reply, reschedule_not_changed_reply,
    reschedule_outcome_unknown_reply,
    # story title: Caller cancels an appointment (E4-S4)
    cancel_choice_prompt, cancel_prompt, cancel_pick_prompt, cancel_not_found_reply,
    cancel_confirm_prompt, cancel_charge_prompt, cancel_charge_retry_prompt,
    cancel_quote_changed_prompt, cancel_unavailable_reply, cancel_reply,
    cancel_kept_reply, cancel_not_cancelled_reply, cancel_outcome_unknown_reply,
    # story title: Caller gives everything in one sentence (E13-S7)
    booking_doctor_not_found_prompt, booking_day_unavailable_prompt,
    booking_time_outside_hours_prompt, booking_date_check_prompt,
    booking_confirmation_prompt,
    # story title: Caller names only a doctor
    booking_doctor_offer_prompt, booking_date_time_prompt, booking_patient_contact_prompt,
    # story title: Caller asks for the earliest available appointment
    booking_earliest_slots_prompt, booking_earliest_none_prompt,
    # story title: Requested slot is already taken
    booking_slot_taken_prompt,
    # story title: Caller says tomorrow, day after, or next Monday
    date_ask_prompt, date_range_pick_prompt,
)

# Characters that mean something on a page and nothing in a sentence. A caller
# hears them as a stumble, a mispronunciation, or -- with the Bengali
# tokenizer -- as nothing at all, which is worse.
FORBIDDEN = (":", "：", "[", "]", "{", "}", "<", ">", "|")

FOUND_TEST = {"found": True, "test_name": "Uric Acid", "test_name_bn": "ইউরিক অ্যাসিড",
              "rate_inr": 250, "sample_type": "Blood", "report_time_hours": 12}
FOUND_DOCTOR = {"found": True, "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
                "date": "2026-09-14", "available": True,
                "chamber_hours": "18:00-20:00", "next_available_date": None}
BOOKING = {"doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন", "date": "2026-09-14",
           "time_slot": "18:30", "patient_name": "রিয়া দাস", "phone": "9876543210"}


def _replies():
    """Every reply function, every branch, with realistic values.

    Named so a failure says which sentence broke rather than which index.
    """
    yield "rate/found", rate_reply({"test_name": "ইউরিক অ্যাসিড"}, FOUND_TEST)
    yield "rate/not-found", rate_reply({"test_name": "কিছু"},
                                       {"found": False, "query": "কিছু"})
    yield "rate/suggestions-1", rate_reply(
        {"test_name": "কিছু"},
        {"found": False, "query": "কিছু", "did_you_mean_bn": ["লিপিড প্রোফাইল"]})
    yield "rate/suggestions-3", rate_reply(
        {"test_name": "কিছু"},
        {"found": False, "query": "কিছু",
         "did_you_mean_bn": ["লিপিড প্রোফাইল", "এলএফটি", "সিবিসি"]})

    yield "doctor/available", doctor_availability_reply({"doctor_name": "সেন"}, FOUND_DOCTOR)
    yield "doctor/not-that-day", doctor_availability_reply(
        {"doctor_name": "সেন"},
        dict(FOUND_DOCTOR, available=False, chamber_hours=None,
             next_available_date="2026-09-17"))
    yield "doctor/no-fixed-day", doctor_availability_reply(
        {"doctor_name": "সেন"},
        dict(FOUND_DOCTOR, available=False, chamber_hours=None,
             next_available_date=None))
    yield "doctor/not-found", doctor_availability_reply(
        {"doctor_name": "ঘোষ"}, {"found": False, "query": "ঘোষ"})

    dept = {"found": True, "department": "Cardiology", "department_bn": "কার্ডিওলজি",
            "date": "2026-09-14"}
    one = {"name": "Dr. A Sen", "doctor_name_bn": "সেন"}
    two = {"name": "Dr. B Roy", "doctor_name_bn": "রায়"}
    three = {"name": "Dr. C Bose", "doctor_name_bn": "বসু"}
    yield "dept/1", doctors_by_department_reply({}, dict(dept, doctors=[one]))
    yield "dept/2", doctors_by_department_reply({}, dict(dept, doctors=[one, two]))
    yield "dept/3", doctors_by_department_reply({}, dict(dept, doctors=[one, two, three]))
    yield "dept/none-that-day", doctors_by_department_reply({}, dict(dept, doctors=[]))
    yield "dept/none-unfiltered", doctors_by_department_reply(
        {}, {"found": True, "department": "Cardiology", "department_bn": "কার্ডিওলজি",
             "date": None, "doctors": []})
    yield "dept/not-found", doctors_by_department_reply(
        {"department": "নিউরো"}, {"found": False, "query": "নিউরো"})

    yield "booking/success", booking_reply(
        {"doctor_name": "সেন"},
        {"success": True, "confirmation_id": "KCD-20260914-4A2F",
         "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
         "date": "2026-09-14", "time_slot": "18:30"})
    yield "booking/slot-taken-alts", booking_reply(
        {}, {"success": False, "reason": "slot_taken",
             "alternative_slots": ["17:30", "18:15", "19:00"]})
    yield "booking/slot-taken-none", booking_reply(
        {}, {"success": False, "reason": "slot_taken", "alternative_slots": []})
    yield "booking/doctor-not-found", booking_reply(
        {"doctor_name": "ঘোষ"}, {"success": False, "reason": "doctor_not_found"})
    yield "booking/other", booking_reply({}, {"success": False, "reason": "missing_field"})

    yield "booking/readback", booking_confirm_prompt(BOOKING)
    yield "booking/correction", booking_correction_prompt()
    yield "date/range", date_range_confirm_prompt("2026-09-14", "2026-09-20")
    yield "asr/echo", heard_confirm_prompt("কাল ডাক্তার সেন আছেন")
    yield "escalation", UNSPEAKABLE_ESCALATION

    # story title: The same question gets the same answer within one call
    # user story: As a caller who asks twice, I want the same answer, so that
    #   I know which one to believe.
    # acceptance criteria: Repeating a question in one call produces an
    #   identical factual answer unless the underlying data changed, in which
    #   case the change is stated. A test asserts consistency across three
    #   repeats with an unchanged backend.
    #
    # The JOINED sentence, not the preamble alone. What the caller hears when
    # a figure moves is one utterance, and a punctuation artefact at the seam
    # would belong to neither half on its own.
    yield "changed/preamble", ANSWER_CHANGED_BN
    yield "changed/rate", with_change_notice(
        rate_reply({"test_name": "ইউরিক অ্যাসিড"}, FOUND_TEST))
    yield "changed/doctor", with_change_notice(
        doctor_availability_reply({"doctor_name": "সেন"}, FOUND_DOCTOR))

    # story title: Near matches are offered rather than guessed or refused
    # user story: As a caller naming something loosely, I want the close
    #   matches offered, so that I am not told my test does not exist when it
    #   does.
    # acceptance criteria: When several catalogue rows fall within the match
    #   band the agent offers up to three by name and asks which. Candidates
    #   are generated across every supported language and romanised spelling.
    #   The did-you-mean path covers the ambiguous case and not only total
    #   failure.
    #
    # One, two and the capped three, plus the re-ask. The three-candidate
    # case is the one worth having: _spoken_list switches from "নাকি" to a
    # comma list there, and a comma list is exactly the database-row reading
    # E12-S3 bans.
    _sugar = [{"name": "Blood Sugar Fasting", "name_bn": "সুগার ফাস্টিং"},
              {"name": "Blood Sugar PP", "name_bn": "সুগার পিপি"}]
    yield "near/1", near_match_prompt(_sugar[:1])
    yield "near/2", near_match_prompt(_sugar)
    yield "near/3", near_match_prompt(
        _sugar + [{"name": "HbA1c", "name_bn": "এইচবিএ১সি"}])
    yield "near/unclear", NEAR_MATCH_UNCLEAR_BN

    # story title: A multi-part question is answered in full
    # user story: As a caller who asked two things, I want both answered, so
    #   that I do not have to ask again.
    # acceptance criteria: Every answerable part of a turn is answered in the
    #   order asked, and any part that cannot be answered is explicitly
    #   addressed rather than dropped. Completeness is scored on a labelled
    #   multi-part set.
    #
    # The subject-carrying form quotes the CALLER'S words, so it is the one
    # that can pick up whatever the ASR produced -- including a stray colon
    # or bracket out of a code-switched utterance.
    yield "multipart/unanswered", UNANSWERED_PART_BN
    yield "multipart/unanswered-named", unanswered_part_prompt("ইউরিক অ্যাসিড")
    yield "multipart/deferred", DEFERRED_PART_BN
    yield "multipart/resuming", RESUMING_PART_BN
    yield "near/no-spoken-name", near_match_prompt(
        [{"name": "Some Test", "name_bn": None}])

    # story title: Caller moves an existing appointment (E4-S3)
    # Every branch. The originals carried three label colons ("পেয়েছি:",
    # "ফাঁকা আছে:", "থাকছে:") that this gate would have failed.
    _appt = {"reference": "KCD-20260915-4F0C", "doctor_name": "Dr. A Sen",
             "doctor_name_bn": "সেন", "date": "2026-09-15", "time_slot": "10:00"}
    for field in ("phone", "name", "pick", "date", "time_slot", "confirm", "other"):
        yield f"resched/prompt/{field}", reschedule_prompt(field)
    yield "resched/found", reschedule_found_prompt(_appt)
    yield "resched/pick-2", reschedule_pick_prompt([_appt, _appt])
    yield "resched/pick-3", reschedule_pick_prompt([_appt, _appt, _appt])
    yield "resched/not-found", reschedule_not_found_reply()
    yield "resched/day-off", reschedule_day_unavailable_reply(
        _appt, {"next_available_date": "2026-09-19"})
    yield "resched/day-off-none", reschedule_day_unavailable_reply(
        _appt, {"next_available_date": None})
    yield "resched/time", reschedule_time_prompt(
        _appt, "2026-09-17", {"chamber_hours": "10:00-12:00"})
    yield "resched/readback", reschedule_confirm_prompt(_appt, "2026-09-17", "10:30")
    yield "resched/success", reschedule_reply(
        {"success": True, **_appt, "new_date": "2026-09-17", "new_time_slot": "10:30"}, _appt)
    for reason, extra in (("slot_taken", {"alternative_slots": ["10:45", "11:00"]}),
                          ("slot_taken", {"alternative_slots": []}),
                          ("doctor_not_available_that_day", {"next_available_date": "2026-09-19"}),
                          ("same_slot", {}), ("conflict", {}), ("past", {})):
        yield (f"resched/{reason}{'-alts' if extra.get('alternative_slots') else ''}",
               reschedule_reply({"success": False, "reason": reason, **extra}, _appt))
    yield "resched/kept", reschedule_kept_reply(_appt)
    yield "resched/kept-none", reschedule_kept_reply(None)
    yield "resched/not-changed", reschedule_not_changed_reply()
    yield "resched/unknown", reschedule_outcome_unknown_reply()

    # story title: Caller cancels an appointment (E4-S4)
    # Every branch, for a free, a part-refund and a no-refund window.
    yield "cancel/choice", cancel_choice_prompt()
    for field in ("choice", "phone", "name", "pick", "confirm", "other"):
        yield f"cancel/prompt/{field}", cancel_prompt(field)
    yield "cancel/pick-2", cancel_pick_prompt([_appt, _appt])
    yield "cancel/pick-3", cancel_pick_prompt([_appt, _appt, _appt])
    yield "cancel/not-found", cancel_not_found_reply()
    for name, quote in (("free", {"charge_inr": 0, "refund_eligibility": "full",
                                  "refund_percent": None}),
                        ("part", {"charge_inr": 150, "refund_eligibility": "partial",
                                  "refund_percent": 50}),
                        ("late", {"charge_inr": 320, "refund_eligibility": "none",
                                  "refund_percent": None})):
        yield f"cancel/confirm/{name}", cancel_confirm_prompt(_appt, quote)
        yield f"cancel/charge/{name}", cancel_charge_prompt(_appt, quote)
        yield f"cancel/retry/{name}", cancel_charge_retry_prompt(quote)
        yield f"cancel/changed/{name}", cancel_quote_changed_prompt(_appt, quote)
        yield f"cancel/success/{name}", cancel_reply({**_appt, **quote})
    for reason in ("already_cancelled", "after_start", "policy_unavailable", "conflict",
                   "not_found", None):
        yield f"cancel/unavailable/{reason}", cancel_unavailable_reply(reason)
    yield "cancel/kept", cancel_kept_reply(_appt)
    yield "cancel/kept-none", cancel_kept_reply(None)
    yield "cancel/not-cancelled", cancel_not_cancelled_reply()
    yield "cancel/unknown", cancel_outcome_unknown_reply()

    # story title: Caller gives everything in one sentence (E13-S7)
    # The three "ask that one field again" questions, Bengali.
    _doc = {"doctor_name": "Dr. A. Sen", "doctor_name_bn": "সেন"}
    _avail = dict(_doc, next_available_date="2026-09-24", chamber_hours="10:00-13:00")
    yield "onesentence/doctor-not-found", booking_doctor_not_found_prompt()
    yield "onesentence/day-off", booking_day_unavailable_prompt(_doc, _avail)
    yield "onesentence/day-off-none", booking_day_unavailable_prompt(
        _doc, dict(_avail, next_available_date=None))
    yield "onesentence/outside-hours", booking_time_outside_hours_prompt(_doc, _avail)
    # A date calculated from "কাল" / a weekday, rechecked, and the readback
    # that names it.
    _calc = dict(_doc, date="2026-09-20", date_said="tomorrow", date_weekday=6)
    yield "onesentence/date-check", booking_date_check_prompt(_calc)
    yield "onesentence/date-check-weekday", booking_date_check_prompt(
        dict(_calc, date_said="sunday"))
    yield "onesentence/readback-day-word", booking_confirmation_prompt(
        dict(_calc, time_slot="10:15", patient_name="রাহুল দাস", phone="9876543210"))

    # story title: Caller names only a doctor
    _next = dict(_doc, available=True, date="2026-09-22", chamber_hours="10:00-13:00")
    yield "doctoronly/offer", booking_doctor_offer_prompt(_doc, _next)
    yield "doctoronly/offer-time-known", booking_doctor_offer_prompt(
        dict(_doc, time_slot="10:30"), _next)
    yield "doctoronly/offer-none", booking_doctor_offer_prompt(
        _doc, dict(_doc, available=False, date=None))
    yield "doctoronly/date-time", booking_date_time_prompt()
    yield "doctoronly/patient-contact", booking_patient_contact_prompt()

    # story title: Caller asks for the earliest available appointment
    _early = dict(_doc, found=True, horizon_days=14, slots=[
        {"date": "2026-09-22", "time_slot": "10:00"}, {"date": "2026-09-22", "time_slot": "10:15"},
        {"date": "2026-09-24", "time_slot": "10:00"}])
    yield "earliest/three", booking_earliest_slots_prompt(_doc, _early)
    yield "earliest/one", booking_earliest_slots_prompt(_doc, dict(_early, slots=_early["slots"][:1]))
    yield "earliest/none", booking_earliest_none_prompt(_doc, dict(_early, slots=[]))
    yield "earliest/none-no-phone", booking_earliest_none_prompt(
        _doc, dict(_early, slots=[]), ask_phone=False)

    # story title: Requested slot is already taken
    _asked = dict(_doc, time_slot="11:15")
    _taken = {"success": False, "reason": "slot_taken", "alternative_slots": ["11:00", "11:30"],
              "other_day_slot": {"date": "2026-09-26", "time_slot": "11:15"}}
    yield "taken/all-three", booking_slot_taken_prompt(_asked, _taken)
    yield "taken/same-day-only", booking_slot_taken_prompt(_asked, dict(_taken, other_day_slot=None))
    yield "taken/other-day-only", booking_slot_taken_prompt(_asked, dict(_taken, alternative_slots=[]))

    # story title: Caller says tomorrow, day after, or next Monday
    yield "dates/ask-none", date_ask_prompt(())
    yield "dates/ask-one", date_ask_prompt(("2026-09-22",))
    yield "dates/ask-two", date_ask_prompt(("2026-09-28", "2026-10-05"))
    yield "dates/range-pick", date_range_pick_prompt("2026-09-28", "2026-10-04")
    yield "dates/check-explicit", booking_date_check_prompt(
        {"date": "2026-09-25", "date_said": "date", "date_weekday": 4})

    for intent in ("test_rate", "doctor_availability", "doctors_by_department",
                   "book_appointment"):
        for field in ("test_name", "doctor_name", "department", "date",
                      "time_slot", "patient_name", "phone"):
            yield f"prompt/{intent}/{field}", missing_slot_prompt(intent, field)


CASES = list(_replies())


@pytest.mark.parametrize("name,reply", CASES, ids=[c[0] for c in CASES])
def test_no_punctuation_artefact_reaches_the_synthesiser(name, reply):
    spoken = verbalize(reply)
    offenders = [ch for ch in FORBIDDEN if ch in spoken]
    assert not offenders, (
        f"{name}: {offenders} survives verbalize() and would be spoken.\n"
        f"  template: {reply}\n"
        f"  spoken:   {spoken}"
    )


def test_the_gate_covers_every_public_reply_function():
    """A new reply function that nobody adds a case for is the way this gate
    quietly stops covering things. Fails if reply_templates grows one."""
    import agent.reply_templates as rt

    public = {n for n in dir(rt)
              if not n.startswith("_") and callable(getattr(rt, n))
              and getattr(rt, n).__module__ == rt.__name__}
    exercised = {
        "test_rate_reply", "doctor_availability_reply", "doctors_by_department_reply",
        "booking_reply", "booking_confirm_prompt", "booking_correction_prompt",
        "date_range_confirm_prompt", "heard_confirm_prompt", "missing_slot_prompt",
        "with_change_notice", "near_match_prompt", "unanswered_part_prompt",
        "reschedule_prompt", "reschedule_found_prompt", "reschedule_pick_prompt",
        "reschedule_not_found_reply", "reschedule_day_unavailable_reply",
        "reschedule_time_prompt", "reschedule_confirm_prompt", "reschedule_reply",
        "reschedule_kept_reply", "reschedule_not_changed_reply",
        "reschedule_outcome_unknown_reply",
        "cancel_choice_prompt", "cancel_prompt", "cancel_pick_prompt",
        "cancel_not_found_reply", "cancel_confirm_prompt", "cancel_charge_prompt",
        "cancel_charge_retry_prompt", "cancel_quote_changed_prompt",
        "cancel_unavailable_reply", "cancel_reply", "cancel_kept_reply",
        "cancel_not_cancelled_reply", "cancel_outcome_unknown_reply",
        "booking_doctor_not_found_prompt", "booking_day_unavailable_prompt",
        "booking_time_outside_hours_prompt", "booking_date_check_prompt",
        "booking_confirmation_prompt",
        "booking_doctor_offer_prompt", "booking_date_time_prompt",
        "booking_patient_contact_prompt",
        "booking_earliest_slots_prompt", "booking_earliest_none_prompt",
        "booking_slot_taken_prompt",
        "date_ask_prompt", "date_range_pick_prompt",
    }
    assert public <= exercised, (
        f"reply function(s) {sorted(public - exercised)} have no case in this gate"
    )


def test_a_raw_value_colon_is_not_the_defect():
    """Guards the distinction the docstring makes. A time still contains a
    colon in the TEMPLATE and must keep doing so -- verbalize() is what turns
    it into words. Someone "fixing" this by banning colons from templates
    would break the times and fix nothing."""
    raw = booking_confirm_prompt(BOOKING)
    assert "18:30" in raw, "the raw template no longer carries the time"
    assert ":" not in verbalize(raw), "but it must not survive to the synthesiser"


def test_the_artefact_would_actually_be_caught():
    """The gate, pointed at a sentence that has the defect. If this stops
    failing, the check has stopped checking."""
    assert ":" in verbalize("স্যাম্পল: রক্ত।"), "a label colon must survive verbalize"
