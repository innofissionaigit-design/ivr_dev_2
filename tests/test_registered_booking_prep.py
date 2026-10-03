"""KCD-382b: the registered/booked-patient preparation flow, through the REAL orchestrator, off-pod.

This file exists separately from tests/test_orchestrator_multi_test_prep.py (KCD-382's own merge wiring,
already proven there) for the same reason that file exists separately from test_enquiry_service.py: neither
the merge logic nor the single-test reply template can prove the thing that decides whether THIS story
actually works from the caller's side -- the deterministic, pre-LLM registered/booking decision in main.py's
`_dispatch_turn` (the `awaiting_test_prep_registration_answer` / `awaiting_test_prep_booking_phone` checks),
and the booking-lookup wiring in `_handle_booking_prep_lookup` that hands its result to KCD-382's existing
single-test and merge paths rather than reimplementing either.

Each test below is labelled with the letter (A-R) it covers from the KCD-382b validation brief:

    A unregistered caller                         J/K/L/M/N  2/3/4/5/6+ booked tests via the booking lookup
    B registered caller + valid identity          O incomplete preparation data -> safe human handoff
    C registered + one booked test                P all real language codes (en, hi, bn)
    D registered + multiple booked tests          Q existing single-test regression (direct name, unaffected)
    E asks about one test from several bookings   R existing direct named multi-test regression (unaffected)
    F common preparation for several bookings
    G invalid identity / no matching patient
    H no booking on a real number
    I follow-up after booking lookup / after a named multi-test turn

    python -m pytest tests/test_registered_booking_prep.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _pod_stubs import pod_stubs  # noqa: E402
from _synth_voices import FATHER, utterance  # noqa: E402
from test_orchestrator_persona import FakeTTS, FakeWS, _wav  # noqa: E402

from agent.phrases import phrase  # noqa: E402
from agent.reply_templates import missing_slot_prompt  # noqa: E402
from agent.tools_client import ToolCallError  # noqa: E402


class ASRResult:
    def __init__(self, text, agreement=0.9):
        self.text, self.decoder_agreement = text, agreement


class FakeTools:
    """Every clinic-API call KCD-382 and KCD-382b's orchestrator wiring can make, each recorded
    separately so a test can assert not just what was SAID but which real lookup produced it --
    reusing the exact technique tests/test_orchestrator_multi_test_prep.py already proved out for
    merge_test_prep/get_test_prep, extended here with search_bookings/booking_test_names (KCD-382b's
    own two new calls)."""

    def __init__(self):
        self.merge_calls, self.merge_result = [], {"found": False}
        self.prep_calls, self.prep_result = [], {"found": False}
        self.search_calls, self.search_result = [], {"status": "none", "matches": []}
        self.search_error = None
        self.booking_tests_calls, self.booking_tests_result = [], {"test_names": []}
        self.booking_tests_error = None

    async def merge_test_prep(self, test_names, lang="bn"):
        self.merge_calls.append((list(test_names), lang))
        return self.merge_result

    async def get_test_prep(self, test_name, lang="bn"):
        self.prep_calls.append((test_name, lang))
        return self.prep_result

    async def search_bookings(self, caller_phone, **kw):
        self.search_calls.append((caller_phone, kw))
        if self.search_error is not None:
            raise self.search_error
        return self.search_result

    async def booking_test_names(self, confirmation_id):
        self.booking_tests_calls.append(confirmation_id)
        if self.booking_tests_error is not None:
            raise self.booking_tests_error
        return self.booking_tests_result


def booking_match(confirmation_id, test_name, date="2026-10-05"):
    return {
        "kind": "test_booking",
        "confirmation_id": confirmation_id,
        "date": date,
        "time_slot": None,
        "doctor_name": None,
        "test_name": test_name,
        "name_basis": None,
    }


@pytest.fixture(scope="module")
def m():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


@pytest.fixture
def env(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_admission", None)
    monkeypatch.setattr(m, "_tts_router", FakeTTS())
    monkeypatch.setattr(m, "CONDITION_INPUT", "off")
    tools = FakeTools()
    monkeypatch.setattr(m, "_tools", tools)
    state = {
        "text": "do I need to fast",
        "lang": "en",
        "intent": "test_prep",
        "slots": {},
        "reply": "No special preparation is needed.",  # used only by the single-test regression tests
        "handoffs": [],
        "answer_calls": [],
    }

    async def route(session, wav):
        return state["lang"], ASRResult(state["text"])

    async def resolve(session, text, lang):
        return {
            "intent": state["intent"],
            "slots": dict(state["slots"]),
            "secondary_intent": None,
            "direct_reply_bn": None,
        }

    async def answer(intent, slots, lang):
        test_name = slots.get("test_name") or next(iter(slots.get("test_names") or []), None)
        if not test_name:
            return None  # exactly the real function's own "nothing to look up" behaviour
        state["answer_calls"].append((intent, dict(slots), lang))
        return state["reply"]

    async def handoff(session, reason, languages=None):
        state["handoffs"].append(reason)

    monkeypatch.setattr(m, "_route_and_transcribe", route)
    monkeypatch.setattr(m, "_resolve_intent", resolve)
    monkeypatch.setattr(m, "_answer_enquiry_intent", answer)
    monkeypatch.setattr(m, "_handoff_to_human", handoff)
    session = m.CallSession(FakeWS())
    session.release_gate()
    session.disclosed_langs.update({"en", "bn", "hi"})
    session.call_state = m.new_call_state()
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)

    class Driver:
        pass

    d = Driver()
    d.session, d.state, d.tools, d.voice = session, state, tools, voice

    async def turn(**overrides):
        state.update(overrides)
        session.turn_epoch = session.speak_epoch
        path = _wav(tmp_path, voice, f"u{len(session.ws.texts)}.wav")
        before = len(session.ws.spoken())
        await m._dispatch_turn(session, path)
        return session.ws.spoken()[before:]

    d.turn = turn
    yield d


def text_of(said):
    return " ".join(said)


# ===================================================================================== A: unregistered caller


@pytest.mark.asyncio
async def test_A_unregistered_caller_falls_through_to_the_existing_generic_ask(env):
    said = await env.turn(text="do I need to fast", intent="test_prep", slots={})
    assert "registered" in text_of(said).lower()
    assert env.tools.search_calls == [] and env.tools.booking_tests_calls == []
    said = await env.turn(text="no", intent="unclear", slots={})
    assert missing_slot_prompt("test_prep", "test_name", "en") in text_of(said)
    assert env.tools.search_calls == [], "PATH A must never perform a booking lookup"


@pytest.mark.asyncio
async def test_A_a_vague_date_alone_does_not_imply_a_booking_exists(env):
    """"My test is tomorrow" is not evidence of a DB booking -- the brief's own explicit warning.
    The registration question is still asked first, exactly as for any other nameless test_prep turn."""
    said = await env.turn(text="my test is tomorrow, what should I eat", intent="test_prep", slots={})
    assert "registered" in text_of(said).lower()
    assert env.tools.search_calls == []


# ============================================================================ B/C: registered + one booked test


@pytest.mark.asyncio
async def test_B_C_registered_with_a_valid_phone_and_one_booked_test_is_answered_from_the_booking(env):
    await env.turn(text="do I need to fast", intent="test_prep", slots={})
    said = await env.turn(text="yes I am registered", intent="unclear", slots={})
    assert phrase("test_prep_ask_booking_phone", "en") in text_of(said)
    env.tools.search_result = {"status": "single", "matches": [booking_match("KCD-T1", "Complete Blood Count (CBC)")]}
    env.tools.booking_tests_result = {"test_names": ["Complete Blood Count (CBC)"]}
    env.state["reply"] = "No special preparation is needed for the CBC."
    said = await env.turn(text="9876543210", intent="unclear", slots={})
    assert env.tools.search_calls == [("9876543210", {"phone": "9876543210"})]
    assert env.tools.booking_tests_calls == ["KCD-T1"]
    assert env.tools.merge_calls == [], "exactly one resolved test must use the single-test path, not the merge"
    assert len(env.state["answer_calls"]) == 1
    assert env.state["answer_calls"][0][1] == {"test_name": "Complete Blood Count (CBC)"}
    assert "No special preparation is needed for the CBC." in text_of(said)


# ================================================================================= D: registered + several tests


@pytest.mark.asyncio
async def test_D_registered_with_a_valid_phone_and_several_booked_tests_goes_through_the_merge_path(env):
    await env.turn(text="do I need to fast", intent="test_prep", slots={})
    await env.turn(text="yes", intent="unclear", slots={})
    env.tools.search_result = {
        "status": "single",
        "matches": [booking_match("KCD-T2", "Blood Sugar Fasting")],
    }
    env.tools.booking_tests_result = {"test_names": ["Blood Sugar Fasting", "Lipid Profile"]}
    env.tools.merge_result = {
        "found": True,
        "test_names": ["Blood Sugar Fasting", "Lipid Profile"],
        "fasting_hours": 11,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": False,
        "not_found": [],
    }
    said = await env.turn(text="9876543210", intent="unclear", slots={})
    assert env.tools.merge_calls == [(["Blood Sugar Fasting", "Lipid Profile"], "en")]
    assert "11" in text_of(said)
    assert env.state["answer_calls"] == [], "two resolved tests must use the merge path, not the single-test one"


# ==================================================== E: a directly-named test is never pulled into the booking


@pytest.mark.asyncio
async def test_E_naming_one_test_directly_never_touches_the_booking_lookup_even_with_bookings_available(env):
    """The caller has (per the fake tool, were it ever asked) several booked tests, but THIS turn names one
    test by itself -- exactly case 3 of the brief's one-vs-multiple table: answer only the one asked about,
    never auto-merge, and never even consult the registered/booking machinery for a turn that already named
    its own test."""
    env.state["reply"] = "No special preparation is needed for the CBC."
    said = await env.turn(
        text="do I need to fast for the CBC", intent="test_prep", slots={"test_name": "Complete Blood Count (CBC)"}
    )
    assert "No special preparation is needed for the CBC." in text_of(said)
    assert env.tools.search_calls == [] and env.tools.booking_tests_calls == []
    assert not env.session.test_prep_registration_asked


# ============================================================================ F: one common rule for several


@pytest.mark.asyncio
async def test_F_a_common_preparation_question_merges_all_the_bookings_relevant_tests(env):
    await env.turn(text="what do I need to do tonight", intent="test_prep", slots={})
    await env.turn(text="yes, I have a booking", intent="unclear", slots={})
    env.tools.search_result = {"status": "single", "matches": [booking_match("KCD-T3", "Blood Sugar Fasting")]}
    env.tools.booking_tests_result = {
        "test_names": ["Blood Sugar Fasting", "Lipid Profile", "Complete Blood Count (CBC)"]
    }
    env.tools.merge_result = {
        "found": True,
        "test_names": ["Blood Sugar Fasting", "Lipid Profile", "Complete Blood Count (CBC)"],
        "fasting_hours": 11,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": False,
        "not_found": [],
    }
    said = await env.turn(text="9876543210", intent="unclear", slots={})
    assert env.tools.merge_calls == [
        (["Blood Sugar Fasting", "Lipid Profile", "Complete Blood Count (CBC)"], "en")
    ]
    # ONE merged sentence, not three separate replies.
    assert len([t for t in said if "11" in t]) == 1


# ======================================================================== G: invalid identity / no such patient


@pytest.mark.asyncio
async def test_G_a_phone_matching_no_patient_falls_back_to_the_generic_path_never_inventing_a_difference(env):
    """search_bookings's own contract (clinic-api/patient_context.py) never discloses whether a number
    belongs to nobody or to a patient with no bookings -- both come back as zero matches, by design, so
    this story never invents a distinction the underlying lookup deliberately does not expose."""
    await env.turn(text="do I need to fast", intent="test_prep", slots={})
    await env.turn(text="yes", intent="unclear", slots={})
    env.tools.search_result = {"status": "none", "matches": []}
    said = await env.turn(text="9000000000", intent="unclear", slots={})
    assert phrase("test_prep_no_booking_found", "en") in text_of(said)
    assert missing_slot_prompt("test_prep", "test_name", "en") in text_of(said)
    assert env.tools.booking_tests_calls == []


# ===================================================================================== H: no booking on a number


@pytest.mark.asyncio
async def test_H_a_real_patient_with_an_unrelated_appointment_but_no_test_booking(env):
    """An unrelated doctor appointment on the same number must never be used for, or counted toward, a
    test-preparation answer -- only `kind == "test_booking"` matches count."""
    await env.turn(text="do I need to fast", intent="test_prep", slots={})
    await env.turn(text="yes", intent="unclear", slots={})
    env.tools.search_result = {
        "status": "single",
        "matches": [
            {
                "kind": "appointment",
                "confirmation_id": "KCD-APPT-1",
                "date": "2026-10-05",
                "time_slot": "10:00",
                "doctor_name": "Dr. A. Sen",
                "test_name": None,
                "name_basis": None,
            }
        ],
    }
    said = await env.turn(text="9876543210", intent="unclear", slots={})
    assert phrase("test_prep_no_booking_found", "en") in text_of(said)
    assert env.tools.booking_tests_calls == []


@pytest.mark.asyncio
async def test_H_several_test_bookings_are_never_collapsed_to_the_first(env):
    """KCD-497's never-bookings[0] rule, reused for test bookings: several matches must ask which one by
    confirmation number, never silently pick the first."""
    await env.turn(text="do I need to fast", intent="test_prep", slots={})
    await env.turn(text="yes", intent="unclear", slots={})
    env.tools.search_result = {
        "status": "several",
        "matches": [
            booking_match("KCD-T10", "Complete Blood Count (CBC)", date="2026-10-04"),
            booking_match("KCD-T11", "Thyroid Profile (T3 T4 TSH)", date="2026-10-09"),
        ],
    }
    said = await env.turn(text="9876543210", intent="unclear", slots={})
    assert env.tools.booking_tests_calls == [], "a several-match result must never be resolved automatically"
    assert "confirmation" in text_of(said).lower()


# =========================================================================== I: follow-up / same-call context


@pytest.mark.asyncio
async def test_I_a_vague_follow_up_after_a_directly_named_multi_test_turn_is_answered_without_re_asking(env):
    """Brief Example 1: Turn 1 names three tests directly; Turn 2 ("what do I need to do tonight?") must
    reuse the same resolved set, never re-asking "which test" nor the registered/booking question."""
    env.tools.merge_result = {
        "found": True,
        "test_names": ["Blood Sugar Fasting", "Lipid Profile", "Complete Blood Count (CBC)"],
        "fasting_hours": 11,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": False,
        "not_found": [],
    }
    await env.turn(
        text="what is the prep for blood sugar fasting, lipid profile and CBC",
        intent="test_prep",
        slots={"test_names": ["Blood Sugar Fasting", "Lipid Profile", "Complete Blood Count (CBC)"]},
    )
    assert env.tools.merge_calls == [(["Blood Sugar Fasting", "Lipid Profile", "Complete Blood Count (CBC)"], "en")]
    said = await env.turn(text="so what do I need to do tonight", intent="test_prep", slots={})
    assert "registered" not in text_of(said).lower()
    assert missing_slot_prompt("test_prep", "test_name", "en") not in text_of(said)
    assert len(env.tools.merge_calls) == 2 and env.tools.merge_calls[1][0] == env.tools.merge_calls[0][0]


@pytest.mark.asyncio
async def test_I_a_vague_follow_up_after_a_booking_lookup_reuses_it_without_a_second_lookup(env):
    """Brief Example 2: Turn 1 "yes, registered"; Turn 2 gives the phone; Turn 3's vague question must use
    the booking already retrieved THIS call -- no second search_bookings/booking_test_names call, and no
    re-asking of the registered/phone questions."""
    await env.turn(text="do I need to fast", intent="test_prep", slots={})
    await env.turn(text="yes, I am registered", intent="unclear", slots={})
    env.tools.search_result = {"status": "single", "matches": [booking_match("KCD-T4", "Blood Sugar Fasting")]}
    env.tools.booking_tests_result = {"test_names": ["Blood Sugar Fasting", "Lipid Profile"]}
    env.tools.merge_result = {
        "found": True,
        "test_names": ["Blood Sugar Fasting", "Lipid Profile"],
        "fasting_hours": 11,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": False,
        "not_found": [],
    }
    await env.turn(text="9876543210", intent="unclear", slots={})
    assert len(env.tools.search_calls) == 1 and len(env.tools.booking_tests_calls) == 1
    said = await env.turn(text="what do I need to do tonight", intent="test_prep", slots={})
    assert len(env.tools.search_calls) == 1, "a follow-up must reuse the resolved booking, never re-query it"
    assert len(env.tools.booking_tests_calls) == 1
    assert "registered" not in text_of(said).lower()
    assert len(env.tools.merge_calls) == 2


@pytest.mark.asyncio
async def test_I_too_long_a_gap_since_the_resolved_booking_asks_again_rather_than_assuming(env):
    """The same FOLLOWUP_MAX_GAP rule agent/enquiry_followup.py already applies to a single test_name
    follow-up -- not a new, looser threshold invented for the plural/booking case. A MULTI-test
    resolution is used here deliberately: _handle_multi_test_prep never calls _finish_enquiry_turn, so
    the pre-existing SINGULAR follow-up mechanism (session.last_enquiry_turn) has nothing of its own to
    recall here -- only this story's own last_resolved_prep_tests/turn cache could possibly answer the
    vague turn below, so rolling back only ITS turn number is a clean, unconfounded test of ITS gap rule.

    The registration question itself is asked at most ONCE per call (session.test_prep_registration_asked),
    so once the caller has already answered "yes" this call, a stale cache falls through to the ordinary
    "which test" ask, not back to "are you registered" again -- re-asking a question already answered this
    call would be its own kind of broken."""
    await env.turn(text="do I need to fast", intent="test_prep", slots={})
    await env.turn(text="yes", intent="unclear", slots={})
    env.tools.search_result = {"status": "single", "matches": [booking_match("KCD-T5", "Blood Sugar Fasting")]}
    env.tools.booking_tests_result = {"test_names": ["Blood Sugar Fasting", "Lipid Profile"]}
    env.tools.merge_result = {
        "found": True,
        "test_names": ["Blood Sugar Fasting", "Lipid Profile"],
        "fasting_hours": 11,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": False,
        "not_found": [],
    }
    await env.turn(text="9876543210", intent="unclear", slots={})
    assert env.session.last_resolved_prep_tests == ("Blood Sugar Fasting", "Lipid Profile")
    env.session.last_resolved_prep_turn -= 20  # long enough ago that it is not assumed
    said = await env.turn(text="what do I need to do tonight", intent="test_prep", slots={})
    assert missing_slot_prompt("test_prep", "test_name", "en") in text_of(said), (
        "too stale to reuse -- falls through to the ordinary ask, not a second registration question"
    )
    assert len(env.tools.merge_calls) == 1, "the stale cache must not be merged again"


# ============================================================ J-N: the full test_names collection, via booking


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tests",
    [
        pytest.param(["Blood Sugar Fasting", "Lipid Profile"], id="J-2"),
        pytest.param(["Blood Sugar Fasting", "Lipid Profile", "Complete Blood Count (CBC)"], id="K-3"),
        pytest.param(
            ["Blood Sugar Fasting", "Lipid Profile", "Complete Blood Count (CBC)", "Kidney Function Test (KFT)"],
            id="L-4",
        ),
        pytest.param(
            [
                "Blood Sugar Fasting",
                "Lipid Profile",
                "Complete Blood Count (CBC)",
                "Kidney Function Test (KFT)",
                "Liver Function Test (LFT)",
            ],
            id="M-5",
        ),
        pytest.param(
            [
                "Blood Sugar Fasting",
                "Lipid Profile",
                "Complete Blood Count (CBC)",
                "Kidney Function Test (KFT)",
                "Liver Function Test (LFT)",
                "Thyroid Profile (T3 T4 TSH)",
            ],
            id="N-6",
        ),
    ],
)
async def test_JKLMN_the_full_booked_test_set_reaches_the_merge_unh_truncated(env, tests):
    await env.turn(text="do I need to fast", intent="test_prep", slots={})
    await env.turn(text="yes", intent="unclear", slots={})
    env.tools.search_result = {"status": "single", "matches": [booking_match("KCD-SIZE", tests[0])]}
    env.tools.booking_tests_result = {"test_names": list(tests)}
    env.tools.merge_result = {
        "found": True,
        "test_names": list(tests),
        "fasting_hours": 11,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": False,
        "not_found": [],
    }
    await env.turn(text="9876543210", intent="unclear", slots={})
    assert env.tools.merge_calls == [(list(tests), "en")], (
        f"expected all {len(tests)} booked tests to reach the merge call unchanged"
    )


# =============================================================== O: incomplete data -> safe human hand-off


@pytest.mark.asyncio
async def test_O_incomplete_structured_data_from_a_booking_hands_off_instead_of_guessing(env):
    await env.turn(text="do I need to fast", intent="test_prep", slots={})
    await env.turn(text="yes", intent="unclear", slots={})
    env.tools.search_result = {"status": "single", "matches": [booking_match("KCD-T6", "USG Whole Abdomen")]}
    env.tools.booking_tests_result = {"test_names": ["USG Whole Abdomen", "Blood Sugar Fasting"]}
    env.tools.merge_result = {
        "found": True,
        "test_names": ["USG Whole Abdomen", "Blood Sugar Fasting"],
        "fasting_hours": None,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": True,
        "incomplete_data": ["USG Whole Abdomen"],
        "not_found": [],
    }
    said = await env.turn(text="9876543210", intent="unclear", slots={})
    assert phrase("test_prep_conflict_handoff", "en") in text_of(said)
    assert env.state["handoffs"] == ["test_prep_conflict_unresolved"]
    assert not any("fast for at least" in line.lower() for line in said)


# ============================================================================= P: all real language codes

# The actual words a real caller would use in each of this repo's three supported language codes --
# agent/audio_quality.py's wrong_script check (part of the pre-existing jumbled-transcript guard, nothing
# to do with this story) rejects a Latin-script sentence claimed as "hi" or "bn", exactly as it should for
# real speech, so each language's own turn is worded in that language's own script here.
_PREP_QUESTION = {
    "en": "do I need to fast",
    "hi": "क्या मुझे उपवास करना होगा",
    "bn": "আমাকে কি উপবাস করতে হবে",
}
_YES = {"en": "yes", "hi": "हाँ", "bn": "হ্যাঁ"}
_NO = {"en": "no", "hi": "नहीं", "bn": "না"}


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["en", "hi", "bn"])
async def test_P_path_a_in_every_real_language_code(env, lang):
    said = await env.turn(text=_PREP_QUESTION[lang], intent="test_prep", slots={}, lang=lang)
    assert phrase("test_prep_ask_registered", lang) in text_of(said)
    said = await env.turn(text=_NO[lang], intent="unclear", slots={}, lang=lang)
    assert missing_slot_prompt("test_prep", "test_name", lang) in text_of(said)


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["en", "hi", "bn"])
async def test_P_path_b_booking_lookup_in_every_real_language_code(env, lang):
    await env.turn(text=_PREP_QUESTION[lang], intent="test_prep", slots={}, lang=lang)
    said = await env.turn(text=_YES[lang], intent="unclear", slots={}, lang=lang)
    assert phrase("test_prep_ask_booking_phone", lang) in text_of(said)
    env.tools.search_result = {"status": "single", "matches": [booking_match("KCD-T7", "Complete Blood Count (CBC)")]}
    env.tools.booking_tests_result = {"test_names": ["Complete Blood Count (CBC)"]}
    env.state["reply"] = "reply"
    said = await env.turn(text="9876543210", intent="unclear", slots={}, lang=lang)
    assert env.tools.booking_tests_calls == ["KCD-T7"]
    assert "reply" in text_of(said)


def test_P_the_three_new_phrase_keys_keep_phrase_parity_across_languages():
    from agent.phrases import PHRASES

    for key in ("test_prep_ask_registered", "test_prep_ask_booking_phone", "test_prep_no_booking_found"):
        assert key in PHRASES["en"] and key in PHRASES["hi"] and key in PHRASES["bn"]


# ============================================================ Q/R: existing behaviour is completely unaffected


@pytest.mark.asyncio
async def test_Q_a_single_named_test_is_unaffected_by_this_story(env):
    env.state["reply"] = "No special preparation is needed for the CBC."
    said = await env.turn(
        text="what is the preparation for a CBC",
        intent="test_prep",
        slots={"test_name": "Complete Blood Count (CBC)"},
    )
    assert env.tools.search_calls == [] and env.tools.merge_calls == []
    assert len(env.state["answer_calls"]) == 1
    assert "No special preparation is needed for the CBC." in text_of(said)


@pytest.mark.asyncio
async def test_R_a_directly_named_multi_test_turn_is_unaffected_by_this_story(env):
    env.tools.merge_result = {
        "found": True,
        "test_names": ["Blood Sugar Fasting", "Lipid Profile"],
        "fasting_hours": 11,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": False,
        "not_found": [],
    }
    said = await env.turn(
        text="what is the preparation for blood sugar fasting and lipid profile",
        intent="test_prep",
        slots={"test_names": ["Blood Sugar Fasting", "Lipid Profile"]},
    )
    assert env.tools.search_calls == [] and env.tools.booking_tests_calls == []
    assert not env.session.test_prep_registration_asked
    assert "11" in text_of(said)
