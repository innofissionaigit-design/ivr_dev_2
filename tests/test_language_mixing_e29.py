"""Epic E29: properties of the booking flow that must hold regardless of
which language each turn was spoken in.

    python -m pytest tests/test_language_mixing_e29.py -v
"""

from agent.booking_flow import is_ready_to_confirm, merge_slots, missing_required, new_state


def test_language_switch_mid_booking_does_not_lose_captured_slots():
    # KCD-432: a caller who starts in Hindi and finishes in Bengali must
    # not have to repeat anything. agent/booking_flow.BookingState never
    # stores or checks a language at all -- slots are language-agnostic
    # values by construction -- so this is really a property test proving
    # that design holds, not something a language field could accidentally
    # reset.
    state = new_state("book_appointment")
    # Turn 1, spoken in Hindi: doctor name and date given.
    merge_slots(state, {"doctor_name": "Sen", "date": "2026-10-01"})
    # Turn 2, caller switches to Bengali mid-call: time and name given.
    merge_slots(state, {"time_slot": "18:15", "patient_name": "Ravi"})
    # Turn 3, back to Hindi: phone number given.
    merge_slots(state, {"phone": "9800000001"})

    assert state.slots == {
        "doctor_name": "Sen",
        "date": "2026-10-01",
        "time_slot": "18:15",
        "patient_name": "Ravi",
        "phone": "9800000001",
    }
    assert is_ready_to_confirm(state)


def test_language_switch_after_confirmation_still_reopens_correctly():
    # The correction-reopens-collection behaviour (KCD-367) must also
    # survive a language switch at the exact moment of the correction.
    state = new_state("book_appointment")
    merge_slots(
        state,
        {
            "doctor_name": "Sen",
            "date": "2026-10-01",
            "time_slot": "18:15",
            "patient_name": "Ravi",
            "phone": "9800000001",
        },
    )
    from agent.booking_flow import mark_confirming

    mark_confirming(state)

    # Correction given in a different language from the one the booking
    # was started in -- the slot merge itself has no notion of language,
    # so nothing about this path needs to special-case it.
    changed = merge_slots(state, {"doctor_name": "Ghosh"})
    assert changed == ["doctor_name"]
    assert state.stage == "collecting"
    assert missing_required(state) == []
