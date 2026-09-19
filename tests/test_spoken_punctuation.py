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
    # ADDED BY SOURAV -- KCD-454. reply_templates.py grew a lot of public
    # functions after this gate's original corpus was written, and none of
    # them ever got a case here -- see test_the_gate_covers_every_public_
    # reply_function()'s own docstring for why that is exactly the failure
    # mode this second test exists to catch. Every one of these 33 names is
    # a function this file did not previously import at all.
    ambiguous_reference_reply, billing_balance_reply, booking_confirmation_prompt,
    callback_confirmation_prompt, callback_correction_prompt, callback_scheduled_reply,
    callback_unavailable_reply, clinic_info_reply, compare_options_reply,
    delivery_blocked_reply, delivery_declined_reply, doctor_schedule_reply,
    health_package_reply, health_packages_list_reply, human_fallback_reply,
    insurance_coverage_reply, multi_intent_missing_info_reply,
    multi_intent_needs_separate_flow_reply, multi_intent_out_of_scope_reply,
    otp_disclosure_refusal_reply, otp_requested_reply, otp_verify_reply,
    out_of_scope_counter_reply, out_of_scope_reply, patient_not_found_reply,
    prescription_requirements_reply, report_ambiguous_reply, report_not_found_reply,
    report_status_reply, sample_type_reply,
    # Aliased for the same reason test_rate_reply already was above --
    # a name starting with "test_" gets collected by pytest itself as a
    # test FUNCTION, not spoken as data, and both of these have required
    # positional params pytest would then fail to find fixtures for.
    test_duration_reply as duration_reply,
    test_preparation_reply as preparation_reply,
    walkin_eligibility_reply,
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

    # ADDED BY SOURAV -- KCD-454. One case per branch for every reply
    # function that grew up after this gate's original corpus was
    # written (see the import block's own comment above). Same "named so
    # a failure says which sentence broke" discipline as every case above.
    yield "sample/not-found", sample_type_reply(
        {"test_name": "কিছু"}, {"found": False, "query": "কিছু"})
    yield "sample/missing", sample_type_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid",
         "test_name_bn": "ইউরিক অ্যাসিড", "sample_type": None})
    yield "sample/found", sample_type_reply({"test_name": "ইউরিক অ্যাসিড"}, FOUND_TEST)

    yield "duration/not-found", duration_reply(
        {"test_name": "কিছু"}, {"found": False, "query": "কিছু"})
    yield "duration/missing", duration_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid",
         "test_name_bn": "ইউরিক অ্যাসিড", "report_time_hours": None})
    yield "duration/found", duration_reply({"test_name": "ইউরিক অ্যাসিড"}, FOUND_TEST)

    yield "preparation/not-found", preparation_reply(
        {"test_name": "কিছু"}, {"found": False, "query": "কিছু"})
    yield "preparation/unavailable", preparation_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "advisory_available": False})
    yield "preparation/available", preparation_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "advisory_available": True,
         "advisory_script_bn": "{test_name}-এর জন্য ৮ ঘণ্টা উপবাস প্রয়োজন।"})

    yield "schedule/not-found", doctor_schedule_reply(
        {"doctor_name": "ঘোষ"}, {"found": False, "query": "ঘোষ"})
    yield "schedule/on-leave-with-date", doctor_schedule_reply(
        {"doctor_name": "সেন"},
        {"found": True, "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
         "on_leave": True, "leave_return_date": "2026-09-20"})
    yield "schedule/on-leave-no-date", doctor_schedule_reply(
        {"doctor_name": "সেন"},
        {"found": True, "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
         "on_leave": True, "leave_return_date": None})
    yield "schedule/none", doctor_schedule_reply(
        {"doctor_name": "সেন"},
        {"found": True, "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
         "on_leave": False, "schedule": []})
    yield "schedule/weekly", doctor_schedule_reply(
        {"doctor_name": "সেন"},
        {"found": True, "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
         "on_leave": False, "schedule": [
             {"start_time": "10:00", "end_time": "13:00", "weekday": 0},
             {"start_time": "10:00", "end_time": "13:00", "weekday": 2},
             {"start_time": "16:00", "end_time": "18:00", "weekday": 4}]})

    yield "booking/prompt", booking_confirmation_prompt(BOOKING)

    yield "patient/not-found", patient_not_found_reply()
    yield "report/not-found", report_not_found_reply()
    yield "report/ambiguous", report_ambiguous_reply(
        {"candidates": [{"test_name": "CBC"}, {"test_name": "Lipid Profile"}]})

    yield "status/ready-with-delivery", report_status_reply(
        {"test_name": "সিবিসি", "status": "READY", "delivery_enabled": True})
    yield "status/ready-no-delivery", report_status_reply(
        {"test_name": "সিবিসি", "status": "READY", "delivery_enabled": False})
    yield "status/not-ready", report_status_reply(
        {"test_name": "সিবিসি", "status": "NOT_READY"})
    yield "status/processing", report_status_reply(
        {"test_name": "সিবিসি", "status": "PROCESSING"})
    yield "status/cancelled", report_status_reply(
        {"test_name": "সিবিসি", "status": "CANCELLED"})

    yield "delivery/disabled", delivery_blocked_reply("DELIVERY_DISABLED")
    yield "delivery/not-ready", delivery_blocked_reply("NOT_READY")
    yield "delivery/processing", delivery_blocked_reply("PROCESSING")
    yield "delivery/cancelled", delivery_blocked_reply("CANCELLED")
    yield "delivery/declined", delivery_declined_reply()

    yield "otp/requested-masked", otp_requested_reply({"masked_phone": "6543"})
    yield "otp/requested-unmasked", otp_requested_reply({})
    yield "otp/disclosure-refusal", otp_disclosure_refusal_reply()
    yield "otp/delivery-sent", otp_verify_reply(
        {"reason": "DELIVERY_SENT", "signed_link_expires_minutes": 15})
    yield "otp/invalid", otp_verify_reply({"reason": "OTP_INVALID"})
    yield "otp/expired", otp_verify_reply({"reason": "OTP_EXPIRED"})
    yield "otp/already-used", otp_verify_reply({"reason": "OTP_ALREADY_USED"})
    yield "otp/max-attempts", otp_verify_reply({"reason": "OTP_MAX_ATTEMPTS"})
    yield "otp/not-requested", otp_verify_reply({"reason": "OTP_NOT_REQUESTED"})
    yield "otp/delivery-failed", otp_verify_reply({"reason": "DELIVERY_FAILED"})

    yield "package/not-found", health_package_reply(
        {"package_name": "কিছু"}, {"found": False, "query": "কিছু"})
    yield "package/found", health_package_reply(
        {"package_name": "ফুল বডি চেকআপ"},
        {"found": True, "package_name": "Full Body Checkup",
         "package_name_bn": "ফুল বডি চেকআপ", "price_inr": 2500,
         "description": "A comprehensive annual health screening.",
         "tests": ["CBC", "Lipid Profile"], "tests_bn": ["সিবিসি", "লিপিড প্রোফাইল"]})
    yield "packages/empty", health_packages_list_reply({"packages": []})
    yield "packages/list", health_packages_list_reply({"packages": [
        {"package_name": "Full Body Checkup", "package_name_bn": "ফুল বডি চেকআপ",
         "price_inr": 2500},
        {"package_name": "Cardiac Package", "package_name_bn": "কার্ডিয়াক প্যাকেজ",
         "price_inr": 4000}]})

    yield "clinic/not-found", clinic_info_reply({}, {"found": False})
    _clinic_found = {"found": True, "hours": {"monday": {"open": "09:00", "close": "18:00"}},
                      "address": "42 Lake View Road, Kolkata, West Bengal 700029",
                      "directions": "Near Lake View Crossing, on the ground floor."}
    yield "clinic/hours", clinic_info_reply(
        {"info_topic": "hours", "today_weekday": 0}, _clinic_found)
    yield "clinic/closed-today", clinic_info_reply(
        {"info_topic": "hours", "today_weekday": 0},
        dict(_clinic_found, hours={"monday": {"closed": True}}))
    yield "clinic/address", clinic_info_reply({"info_topic": "address"}, _clinic_found)
    yield "clinic/directions", clinic_info_reply({"info_topic": "directions"}, _clinic_found)
    yield "clinic/all-topics", clinic_info_reply(
        {"info_topic": None, "today_weekday": 0}, _clinic_found)

    yield "human-fallback", human_fallback_reply()
    yield "out-of-scope/offer", out_of_scope_reply()
    yield "out-of-scope/counter", out_of_scope_counter_reply()

    yield "walkin/not-found", walkin_eligibility_reply(
        {"test_name": "কিছু"}, {"found": False, "query": "কিছু"})
    yield "walkin/policy-unavailable", walkin_eligibility_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "policy_available": False})
    yield "walkin/eligible-with-hours", walkin_eligibility_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "policy_available": True,
         "walkin_eligible": True, "walkin_hours": "09:00-17:00"})
    yield "walkin/eligible-no-hours", walkin_eligibility_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "policy_available": True,
         "walkin_eligible": True, "walkin_hours": None})
    yield "walkin/not-eligible", walkin_eligibility_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "policy_available": True,
         "walkin_eligible": False})

    yield "prescription/not-found", prescription_requirements_reply(
        {"test_name": "কিছু"}, {"found": False, "query": "কিছু"})
    yield "prescription/policy-unavailable", prescription_requirements_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "policy_available": False})
    yield "prescription/required-with-channels", prescription_requirements_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "policy_available": True,
         "prescription_required": True,
         "prescription_channels": ["whatsapp_photo", "counter_in_person"]})
    yield "prescription/required-no-channels", prescription_requirements_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "policy_available": True,
         "prescription_required": True, "prescription_channels": []})
    yield "prescription/not-required", prescription_requirements_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "policy_available": True,
         "prescription_required": False})

    yield "insurance/test-not-found", insurance_coverage_reply(
        {"test_name": "কিছু"}, {"test_found": False, "query": "কিছু"})
    yield "insurance/provider-not-found", insurance_coverage_reply(
        {"test_name": "ইউরিক অ্যাসিড", "insurance_provider_name": "কিছু"},
        {"test_found": True, "test_name": "Uric Acid", "provider_found": False,
         "query_provider": "কিছু"})
    yield "insurance/policy-unavailable", insurance_coverage_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"test_found": True, "test_name": "Uric Acid", "provider_found": True,
         "provider_name": "Star Health", "policy_available": False})
    yield "insurance/covered", insurance_coverage_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"test_found": True, "test_name": "Uric Acid", "provider_found": True,
         "provider_name": "Star Health", "policy_available": True,
         "coverage_status": "COVERED", "pre_auth_required": False})
    yield "insurance/not-covered", insurance_coverage_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"test_found": True, "test_name": "Uric Acid", "provider_found": True,
         "provider_name": "Star Health", "policy_available": True,
         "coverage_status": "NOT_COVERED", "pre_auth_required": False})
    yield "insurance/partial-preauth", insurance_coverage_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"test_found": True, "test_name": "Uric Acid", "provider_found": True,
         "provider_name": "Star Health", "policy_available": True,
         "coverage_status": "PARTIAL", "pre_auth_required": True})

    yield "billing/patient-not-found", billing_balance_reply({"patient_found": False})
    yield "billing/no-record", billing_balance_reply(
        {"patient_found": True, "found": False})
    yield "billing/due-date", billing_balance_reply(
        {"patient_found": True, "found": True, "outstanding_amount": 1250,
         "due_date": "2026-09-30"})
    yield "billing/no-due-date", billing_balance_reply(
        {"patient_found": True, "found": True, "outstanding_amount": 1250,
         "due_date": None})

    yield "multi/missing-info", multi_intent_missing_info_reply()
    yield "multi/out-of-scope", multi_intent_out_of_scope_reply()
    yield "multi/needs-separate-flow", multi_intent_needs_separate_flow_reply()

    yield "compare/not-found", compare_options_reply(
        "কিছু", "অন্য কিছু", {}, {}, {"a_found": False, "b_found": False})
    yield "compare/cheaper-test", compare_options_reply(
        "সিবিসি", "লিপিড প্রোফাইল",
        {"kind": "test", "test_name_bn": "সিবিসি", "test_name": "CBC"},
        {"kind": "test", "test_name_bn": "লিপিড প্রোফাইল", "test_name": "Lipid Profile"},
        {"a_found": True, "b_found": True, "price_delta": "150", "cheaper": "a",
         "both_packages": False, "identical_tests": None, "test_count_delta": None,
         "more_tests_side": None, "extra_tests_a": [], "extra_tests_b": []})
    yield "compare/packages-extra-tests", compare_options_reply(
        "ফুল বডি চেকআপ", "বেসিক চেকআপ",
        {"kind": "package", "package_name_bn": "ফুল বডি চেকআপ",
         "package_name": "Full Body Checkup"},
        {"kind": "package", "package_name_bn": "বেসিক চেকআপ",
         "package_name": "Basic Checkup"},
        {"a_found": True, "b_found": True, "price_delta": "500", "cheaper": "b",
         "both_packages": True, "identical_tests": False, "test_count_delta": 3,
         "more_tests_side": "a",
         "extra_tests_a": [{"name": "Vitamin D", "name_bn": "ভিটামিন ডি"},
                            {"name": "Vitamin B12", "name_bn": "ভিটামিন বি১২"},
                            {"name": "Iron", "name_bn": "আয়রন"}],
         "extra_tests_b": []})

    yield "ambiguous/test", ambiguous_reference_reply("test", ["সিবিসি", "লিপিড প্রোফাইল"])

    yield "callback/unavailable-disabled", callback_unavailable_reply("disabled")
    yield "callback/unavailable-outside-hours", callback_unavailable_reply("outside_hours")
    yield "callback/unavailable-hours-unknown", callback_unavailable_reply("hours_unknown")
    yield "callback/confirmation", callback_confirmation_prompt(
        {"callback_time_window": "সকাল ১০টা থেকে দুপুর ১টা", "phone": "9876543210"})
    yield "callback/correction", callback_correction_prompt()
    yield "callback/scheduled-success", callback_scheduled_reply(
        {"callback_time_window": "সকাল ১০টা থেকে দুপুর ১টা"},
        {"success": True, "callback_id": "CB-20260915-A1B2"})
    yield "callback/scheduled-failure", callback_scheduled_reply({}, {"success": False})

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
        # ADDED BY SOURAV -- KCD-454. The 33 functions the import block
        # above grew to cover.
        "ambiguous_reference_reply", "billing_balance_reply",
        "booking_confirmation_prompt", "callback_confirmation_prompt",
        "callback_correction_prompt", "callback_scheduled_reply",
        "callback_unavailable_reply", "clinic_info_reply", "compare_options_reply",
        "delivery_blocked_reply", "delivery_declined_reply", "doctor_schedule_reply",
        "health_package_reply", "health_packages_list_reply", "human_fallback_reply",
        "insurance_coverage_reply", "multi_intent_missing_info_reply",
        "multi_intent_needs_separate_flow_reply", "multi_intent_out_of_scope_reply",
        "otp_disclosure_refusal_reply", "otp_requested_reply", "otp_verify_reply",
        "out_of_scope_counter_reply", "out_of_scope_reply", "patient_not_found_reply",
        "prescription_requirements_reply", "report_ambiguous_reply",
        "report_not_found_reply", "report_status_reply", "sample_type_reply",
        "test_duration_reply", "test_preparation_reply", "walkin_eligibility_reply",
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
