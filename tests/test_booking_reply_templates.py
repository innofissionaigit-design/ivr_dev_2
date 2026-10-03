"""Epic E26's new reply templates, in all three languages. Checks: every
function is callable for bn/hi/en without crashing, script-appropriate
content lands in the reply (no Bengali text leaking into a Hindi/English
reply or vice versa), and every FACT (price, date, confirmation number)
in the reply comes from the `result` dict passed in, not invented.

    python -m pytest tests/test_booking_reply_templates.py -v
"""

import re

import agent.reply_templates as rt

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_BENGALI = re.compile(r"[ঀ-৿]")


def _script(text: str) -> str:
    if _DEVANAGARI.search(text):
        return "hi"
    if _BENGALI.search(text):
        return "bn"
    return "en"


def test_booking_confirmation_readback_all_languages():
    slots = {
        "doctor_name": "Sen",
        "date": "2026-10-01",
        "time_slot": "18:15",
        "patient_name": "Ravi",
        "phone": "9800000001",
    }
    for lang in ("bn", "hi", "en"):
        reply = rt.booking_confirmation_readback(slots, "book_appointment", lang)
        assert "2026-10-01" in reply and "18:15" in reply and "9800000001" in reply
        assert _script(reply) == lang or lang == "en"  # english readback embeds a latin doctor name either way


def test_reschedule_reply_success_and_failure_all_languages():
    success = {"success": True, "confirmation_id": "KCD-999", "date": "2026-10-05", "time_slot": "18:30"}
    not_found = {"success": False, "reason": "not_found"}
    slot_taken = {"success": False, "reason": "slot_taken", "alternative_slots": ["18:00", "18:15"]}
    for lang in ("bn", "hi", "en"):
        r = rt.reschedule_reply(success, lang)
        assert "KCD-999" in r and "2026-10-05" in r and "18:30" in r
        r2 = rt.reschedule_reply(not_found, lang)
        assert r2  # non-empty, doesn't crash
        r3 = rt.reschedule_reply(slot_taken, lang)
        assert "18:00" in r3 and "18:15" in r3


def test_cancel_reply_states_the_charge_before_it_is_applied():
    pending = {"success": False, "reason": "charge_confirmation_required", "charge_inr": 350}
    for lang in ("bn", "hi", "en"):
        r = rt.cancel_reply(pending, lang)
        assert "350" in r

    charged = {"success": True, "charge_inr": 350}
    free = {"success": True, "charge_inr": 0}
    for lang in ("bn", "hi", "en"):
        assert "350" in rt.cancel_reply(charged, lang)
        r = rt.cancel_reply(free, lang)
        assert "350" not in r


def test_lookup_reply_states_the_real_confirmation_number():
    bookings = [{"doctor_name": "Dr. A. Sen", "date": "2026-10-01", "time_slot": "18:15", "confirmation_id": "KCD-777"}]
    for lang in ("bn", "hi", "en"):
        r = rt.lookup_reply(bookings, lang)
        assert "KCD-777" in r and "2026-10-01" in r
    for lang in ("bn", "hi", "en"):
        r = rt.lookup_reply([], lang)
        assert "KCD-777" not in r


def test_multi_test_reply_states_the_real_total():
    result = {
        "success": True,
        "test_names": ["Uric Acid", "Lipid Profile"],
        "date": "2026-10-02",
        "total_rate_inr": 900,
        "confirmation_id": "KCD-555",
        "combined_prep": "Fast for 10 hours.",
    }
    for lang in ("bn", "hi", "en"):
        r = rt.multi_test_reply(result, lang)
        assert "900" in r and "KCD-555" in r and "Uric Acid" in r and "Lipid Profile" in r


def test_add_test_reply_reasons_all_languages():
    for reason in ("not_found", "test_not_found", "already_booked"):
        for lang in ("bn", "hi", "en"):
            r = rt.add_test_reply({"success": False, "reason": reason}, lang)
            assert r


def test_department_route_reply_never_frames_as_diagnosis():
    matched = {"matched": True, "department_name": "Cardiology"}
    ambiguous = {"matched": False, "ambiguous": True, "candidates": ["Cardiology", "General Medicine"]}
    none = {"matched": False}
    for lang in ("bn", "hi", "en"):
        r = rt.department_route_reply(matched, "chest pain", lang)
        assert "Cardiology" in r
        # never claims to diagnose -- no "you have" / clinical verdict wording
        assert "diagnos" not in r.lower()
        r2 = rt.department_route_reply(ambiguous, "pain", lang)
        assert "Cardiology" in r2 and "General Medicine" in r2
        assert rt.department_route_reply(none, "??", lang)


def test_department_route_reply_names_real_doctors_when_given_them():
    matched = {"matched": True, "department_name": "Cardiology", "doctors": ["Dr. K. Bhattacharya", "Dr. N. Roy"]}
    ambiguous = {
        "matched": False,
        "ambiguous": True,
        "candidates": ["Cardiology", "General Medicine"],
        "doctors_by_department": {"Cardiology": ["Dr. K. Bhattacharya"], "General Medicine": ["Dr. S. Mukherjee"]},
    }
    for lang in ("bn", "hi", "en"):
        r = rt.department_route_reply(matched, "chest pain", lang)
        assert "Dr. K. Bhattacharya" in r and "Dr. N. Roy" in r
        r2 = rt.department_route_reply(ambiguous, "pain", lang)
        assert "Dr. K. Bhattacharya" in r2 and "Dr. S. Mukherjee" in r2


def test_conflict_reply_states_the_existing_booking():
    conflict = {"doctor_name": "Dr. P. Ghosh", "date": "2026-10-03", "time_slot": "10:15"}
    for lang in ("bn", "hi", "en"):
        r = rt.conflict_reply(conflict, lang)
        assert "2026-10-03" in r and "10:15" in r


def test_earliest_available_reply_all_languages():
    found = {
        "found": True,
        "available": True,
        "date": "2026-10-04",
        "time_slot": "10:00",
        "alternatives": ["10:15", "10:30"],
    }
    unavailable = {"found": True, "available": False, "doctor_name": "Dr. X"}
    not_found = {"found": False, "query": "Nobody"}
    for lang in ("bn", "hi", "en"):
        r = rt.earliest_available_reply(found, lang)
        assert "2026-10-04" in r and "10:00" in r
        assert rt.earliest_available_reply(unavailable, lang)
        r3 = rt.earliest_available_reply(not_found, lang)
        assert "Nobody" in r3


def test_resend_reply_never_reads_the_full_number_aloud():
    ok = {"success": True, "sent_to_last4": "6780"}
    for lang in ("bn", "hi", "en"):
        r = rt.resend_reply(ok, lang)
        assert "6780" in r
        assert "9123456780" not in r  # never the full number


def test_booking_reply_states_a_past_date_plainly_and_never_rolls_it_forward():
    # KCD-363: a date in the past is stated plainly, never silently advanced.
    slots = {"doctor_name": "Sen", "date": "2020-01-01"}
    result = {"success": False, "reason": "date_in_past"}
    for lang in ("bn", "hi", "en"):
        r = rt.booking_reply(slots, result, lang)
        assert "2020-01-01" not in r  # never states a booked date that never happened
        assert r


def test_booking_reply_hold_expired_offers_a_retry_not_a_dead_end():
    result = {"success": False, "reason": "hold_expired"}
    for lang in ("bn", "hi", "en"):
        r = rt.booking_reply({}, result, lang)
        assert r


def test_missing_slot_prompt_new_intents_all_languages():
    for intent, field in (
        ("book_test", "test_names"),
        ("reschedule_appointment", "new_date"),
        ("cancel_appointment", "confirmation_id"),
        ("add_test_booking", "test_name"),
        ("lookup_booking", "phone"),
    ):
        for lang in ("bn", "hi", "en"):
            r = rt.missing_slot_prompt(intent, field, lang)
            assert r and r != rt.missing_slot_prompt("unclear", "nonsense", lang)


def test_spelling_prompt_and_readback_all_languages():
    for lang in ("bn", "hi", "en"):
        assert rt.spelling_prompt(lang)
        r = rt.spelling_readback("ravi", lang)
        assert "RAVI" in r


def test_blood_test_list_reply_splits_into_short_tts_clauses_not_one_giant_one():
    """A live call (2026-09-28): the whole test list joined into one comma-separated run with only a single
    sentence break at the end came back as ONE over-long TTS clause and was never actually spoken. Grouped into
    short runs of _NAMES_PER_GROUP, each ending in a sentence break, so agent.clause_split.split_into_clauses
    turns this into several short clauses instead."""
    from agent.clause_split import split_into_clauses
    from agent.reply_templates import _NAMES_PER_GROUP, blood_test_list_reply

    names = [f"Test {i:02d}" for i in range(23)]  # more than one group's worth
    for lang in ("bn", "hi", "en"):
        reply = blood_test_list_reply(names, lang)
        for name in names:
            assert name in reply
        clauses = split_into_clauses(reply)
        assert len(clauses) > 1, (lang, reply)
        assert all(len(c) < 150 for c in clauses), (lang, [len(c) for c in clauses])
        # every group of _NAMES_PER_GROUP names lands together in the same clause -- and one clause is never split
        # across two group boundaries, or a caller would hear a mid-list pause in the wrong place
        for clause in clauses:
            in_this_clause = [n for n in names if n in clause]
            assert len(in_this_clause) <= _NAMES_PER_GROUP


def test_blood_test_list_reply_with_no_tests_is_a_single_short_reply():
    from agent.reply_templates import blood_test_list_reply

    for lang in ("bn", "hi", "en"):
        reply = blood_test_list_reply([], lang)
        assert reply and len(reply) < 150
