"""agent/home_collection_flow.py: the cross-turn state machine for "Caller asks whether a test can
be collected at home" (KCD-387 full flow). Pure and offline -- no network, no database, no GPU.

    python -m pytest tests/test_home_collection_flow.py -v
"""

from agent.home_collection_flow import (
    HomeCollectionState,
    address_display,
    address_postal_code_conflict,
    merge_postal_code,
    merge_test_names,
    missing_address_fields,
    new_state,
    resolve_slot_choice,
)


def test_new_state_starts_with_nothing_known():
    st = new_state()
    assert st.test_names == [] and st.postal_code is None and st.date is None
    assert st.hold_token is None and st.quote is None


def test_test_names_accumulate_across_turns_never_overwrite():
    st = new_state()
    assert merge_test_names(st, ["Uric Acid"]) is True
    assert merge_test_names(st, ["Lipid Profile"]) is True
    assert st.test_names == ["Uric Acid", "Lipid Profile"]
    # The same name again changes nothing and is not duplicated.
    assert merge_test_names(st, ["Uric Acid"]) is False
    assert st.test_names == ["Uric Acid", "Lipid Profile"]


def test_merge_test_names_ignores_blank_and_none():
    st = new_state()
    assert merge_test_names(st, [None, "", "  "]) is False
    assert st.test_names == []
    assert merge_test_names(st, None) is False


def test_postal_code_overwrites_a_correction():
    st = new_state()
    assert merge_postal_code(st, "700091") is True
    assert st.postal_code == "700091"
    assert merge_postal_code(st, "700091") is False  # same value: not a change
    assert merge_postal_code(st, "700019") is True
    assert st.postal_code == "700019"
    assert merge_postal_code(st, None) is False
    assert st.postal_code == "700019"  # never cleared by a null


def test_missing_address_fields_requires_address_line_and_postal_code():
    st = new_state()
    assert set(missing_address_fields(st)) == {"address_line", "postal_code"}
    st.postal_code = "700091"
    assert missing_address_fields(st) == ["address_line"]
    st.address_line = "12 Lake Road"
    assert missing_address_fields(st) == []


def test_missing_address_fields_never_requires_a_landmark():
    st = new_state()
    st.postal_code, st.address_line = "700091", "12 Lake Road"
    assert missing_address_fields(st) == []  # landmark is optional (section 21: "where needed")


def test_address_display_is_one_natural_clause_not_a_label_value_dump():
    st = new_state()
    st.address_line, st.landmark, st.postal_code = "12 Lake Road", "near the bridge", "700091"
    text = address_display(st)
    assert text == "12 Lake Road, near the bridge, 700091"
    assert ":" not in text


def test_address_display_skips_an_empty_landmark():
    st = new_state()
    st.address_line, st.postal_code = "12 Lake Road", "700091"
    assert address_display(st) == "12 Lake Road, 700091"


def test_address_postal_code_conflict_catches_a_mismatched_embedded_pincode():
    st = new_state()
    st.postal_code = "700091"
    st.address_line = "12 Lake Road, 400001"
    assert address_postal_code_conflict(st) == "400001"


def test_address_postal_code_conflict_is_none_when_they_agree():
    st = new_state()
    st.postal_code = "700091"
    st.address_line = "12 Lake Road, 700091"
    assert address_postal_code_conflict(st) is None


def test_address_postal_code_conflict_is_none_when_the_address_names_no_pincode():
    st = new_state()
    st.postal_code = "700091"
    st.address_line = "12 Lake Road, near the big banyan tree"
    assert address_postal_code_conflict(st) is None


def test_address_postal_code_conflict_ignores_a_run_that_is_not_six_digits():
    # A 7-digit or 5-digit run in the address (a flat number, a landline fragment) is not a
    # pincode and must never be flagged as a conflict.
    st = new_state()
    st.postal_code = "700091"
    st.address_line = "Flat 12345678, Lake Road"
    assert address_postal_code_conflict(st) is None


def test_resolve_slot_choice_auto_picks_the_only_option():
    one = [{"start_time": "08:00", "end_time": "10:00"}]
    assert resolve_slot_choice(one, "", None, "en") == one[0]


def test_resolve_slot_choice_by_start_time():
    slots = [{"start_time": "08:00", "end_time": "10:00"}, {"start_time": "10:00", "end_time": "12:00"}]
    assert resolve_slot_choice(slots, "the 10:00 one please", None, "en") == slots[1]


def test_resolve_slot_choice_by_ordinal_word_in_english():
    slots = [{"start_time": "08:00", "end_time": "10:00"}, {"start_time": "10:00", "end_time": "12:00"}]
    assert resolve_slot_choice(slots, "the second one please", None, "en") == slots[1]
    assert resolve_slot_choice(slots, "the first one please", None, "en") == slots[0]


def test_resolve_slot_choice_by_ordinal_word_in_bengali():
    slots = [{"start_time": "08:00", "end_time": "10:00"}, {"start_time": "10:00", "end_time": "12:00"}]
    assert resolve_slot_choice(slots, "দ্বিতীয়টা", None, "bn") == slots[1]


def test_resolve_slot_choice_returns_none_rather_than_guess():
    slots = [{"start_time": "08:00", "end_time": "10:00"}, {"start_time": "10:00", "end_time": "12:00"}]
    assert resolve_slot_choice(slots, "I'm not sure", None, "en") is None


def test_resolve_slot_choice_with_no_slots_offered_is_none():
    assert resolve_slot_choice([], "the first one", None, "en") is None


def test_is_stale_after_the_idle_timeout():
    st = HomeCollectionState(last_updated=0.0)  # epoch zero -- certainly stale against monotonic time
    assert st.is_stale() is True
    st.touch()
    assert st.is_stale() is False
