"""Context between turns for test and doctor questions, from the first live conversation (2026-09-25):

    caller: ক্রিয়েটিং টেস্টের দাম কত            -> the price of Serum Creatinine
    caller: ফাস্টিং করতে হবে                        -> "which test?"   (it had just been talking about ONE test)
    caller: এই সমে টেস্ট টাই                       -> "there is no test called 'সমে টেস্ট'"   ("this same test")

    python -m pytest tests/test_followup_context.py -v

The rule is deterministic (no model decides which test): a test or doctor question with nothing named, within the last
few turns of an answer about exactly ONE entity, is about that entity; a pointing-only value ("this same test") is a
reference, never a name to look up. The reply always names what it answers about, so a wrong assumption is heard.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_orchestrator_booking_flow import env, m  # noqa: F401  (the harness: real dispatch, fake tools)

from agent.enquiry_followup import FOLLOWUP_MAX_GAP, EnquiryEntity, is_reference_only, resolve_followup_slot

ONE = [EnquiryEntity("test_name", "Serum Creatinine")]
TWO = [EnquiryEntity("test_name", "Serum Creatinine"), EnquiryEntity("test_name", "Uric Acid")]


# =========================================================================================== a reference is not a name


@pytest.mark.parametrize(
    "value",
    [
        "সমে টেস্ট",
        "এই সমে টেস্ট টাই",
        "সেম টেস্ট",
        "এই টেস্ট",
        "ওই টেস্টটা",
        "সেই টেস্ট",
        "same test",
        "this test",
        "the same",
        "that one",
        "the test",
        "यही टेस्ट",
        "वही टेस्ट",
        "सेम टेस्ट",
    ],
)
def test_a_value_that_only_points_is_a_reference_not_a_name(value):
    assert is_reference_only(value)


@pytest.mark.parametrize(
    "value", ["ক্রিয়াটিনিন", "ইউরিক অ্যাসিড টেস্ট", "সেন", "দে", "CBC", "lipid profile test", "एचबीए वन सी", "", None]
)
def test_a_real_name_or_nothing_is_not_a_reference(value):
    assert not is_reference_only(value)


def test_the_second_line_of_the_live_conversation_uses_the_test_just_discussed():
    slots, resolved = resolve_followup_slot(
        "test_prep", {"test_name": None}, "ফাস্টিং করতে হবে", "bn", ONE, turns_since_last=1
    )
    assert resolved and slots["test_name"] == "Serum Creatinine"


@pytest.mark.parametrize("said", ["সমে টেস্ট", "এই সমে টেস্ট টাই", "same test", "यही टेस्ट"])
def test_the_third_line_a_pointing_value_is_replaced_by_the_test_just_discussed_at_any_distance(said):
    slots, resolved = resolve_followup_slot("test_prep", {"test_name": said}, "x", "bn", ONE, turns_since_last=50)
    assert resolved and slots["test_name"] == "Serum Creatinine"


# ================================================================================= what must still ask, never guess


def test_a_real_name_is_never_replaced():
    slots, resolved = resolve_followup_slot("test_prep", {"test_name": "ESR"}, "x", "en", ONE, turns_since_last=1)
    assert not resolved and slots["test_name"] == "ESR"


def test_with_nothing_remembered_it_asks():
    assert (
        resolve_followup_slot("test_prep", {"test_name": None}, "do I need to fast", "en", [], turns_since_last=None)[1]
        is False
    )


def test_too_long_ago_and_no_pointer_it_asks():
    assert (
        resolve_followup_slot("test_prep", {}, "do I need to fast", "en", ONE, turns_since_last=FOLLOWUP_MAX_GAP + 1)[1]
        is False
    )
    assert (
        resolve_followup_slot("test_prep", {}, "do I need to fast", "en", ONE, turns_since_last=FOLLOWUP_MAX_GAP)[1]
        is True
    )


def test_two_remembered_tests_is_ambiguous_so_it_asks_even_when_pointing():
    assert (
        resolve_followup_slot("test_prep", {"test_name": "same test"}, "x", "en", TWO, turns_since_last=1)[1] is False
    )
    assert resolve_followup_slot("test_prep", {}, "do I need to fast", "en", TWO, turns_since_last=1)[1] is False


def test_a_test_is_never_filled_into_a_doctor_question_or_the_reverse():
    assert (
        resolve_followup_slot("doctor_availability", {}, "when does he sit", "en", ONE, turns_since_last=1)[1] is False
    )


def test_a_faq_question_with_no_topic_is_not_a_follow_up():
    last = [EnquiryEntity("faq_topic", "hours")]
    assert resolve_followup_slot("clinic_faq", {}, "hmm", "en", last, turns_since_last=1)[1] is False


def test_a_doctor_follow_up_uses_the_doctor_just_discussed():
    last = [EnquiryEntity("doctor_name", "Dr. A. Sen")]
    slots, resolved = resolve_followup_slot("doctor_availability", {}, "and tomorrow", "en", last, turns_since_last=1)
    assert resolved and slots["doctor_name"] == "Dr. A. Sen"


def test_without_the_turn_gap_the_old_pointer_only_behaviour_is_unchanged():
    assert resolve_followup_slot("test_prep", {}, "do I need to fast", "en", ONE)[1] is False
    assert resolve_followup_slot("test_prep", {}, "is fasting needed for that one", "en", ONE)[1] is True


# ================================================================================ the live conversation, end to end


class Spy:
    """Records what the orchestrator asked the clinic lookup layer to answer."""

    def __init__(self, env):
        self.calls = []
        env.state["reply"] = "answer"

    async def answer(self, intent, slots, lang):
        subject = slots.get("test_name") or slots.get("doctor_name") or slots.get("faq_topic")
        if not subject:
            return None  # like the real one: nothing to look up, so the agent asks
        self.calls.append((intent, dict(slots)))
        return f"reply for {subject}"


@pytest.fixture
def spy(m, env, monkeypatch):
    s = Spy(env)
    monkeypatch.setattr(m, "_answer_enquiry_intent", s.answer)
    return s


@pytest.mark.asyncio
async def test_the_live_conversation_the_test_just_answered_about_is_used_twice(m, env, spy):
    from agent.reply_templates import missing_slot_prompt

    env.state["lang"] = "bn"  # a Bengali call, as it was
    await env.say("ক্রিয়েটিং টেস্টের দাম কত", "test_rate", {"test_name": "Serum Creatinine"})
    said = await env.say("ফাস্টিং করতে হবে", "test_prep", {})
    assert spy.calls[-1] == ("test_prep", {"test_name": "Serum Creatinine"})
    assert missing_slot_prompt("test_prep", "test_name", "bn") not in " ".join(said)  # no "which test?"
    assert "Serum Creatinine" in " ".join(said)  # and it says which one
    said = await env.say("এই সমে টেস্ট টাই", "test_prep", {"test_name": "সমে টেস্ট"})
    assert spy.calls[-1] == ("test_prep", {"test_name": "Serum Creatinine"})  # not looked up as "সমে টেস্ট"
    assert all("সমে" not in t for t in said)


@pytest.mark.asyncio
async def test_with_no_earlier_answer_the_agent_still_asks_which_test(m, env, spy):
    # KCD-382b: the FIRST time in a call that a test_prep turn names nothing and nothing is
    # remembered, the agent asks whether the caller is registered/has a booking before falling
    # back to "which test" -- see PATH A/B of the registered/booked-patient preparation story.
    said = await env.say(
        "do I need to fast",
        "test_prep",
        {},
    )
    assert not spy.calls and "registered" in " ".join(said).lower()
    # PATH A: caller says no -- never asked for a phone, falls straight through to the existing
    # generic "which test" ask, exactly as before this story.
    said = await env.say("no", "unclear", {})
    assert not spy.calls and "which test" in " ".join(said).lower()


@pytest.mark.asyncio
async def test_a_bare_pointing_answer_the_model_could_not_label_answers_the_question_just_asked(m, env, spy):
    """We asked "which test?"; the caller says only "the same test" and the extractor returns 'unclear'."""
    await env.say("price of creatinine", "test_rate", {"test_name": "Serum Creatinine"})
    env.session.last_enquiry_turn -= 20  # long enough ago that it is not assumed
    said = await env.say("do I need to fast", "test_prep", {})
    # KCD-382b: first ask in this call is the registered/booking question, not "which test" yet.
    assert "registered" in " ".join(said).lower()
    said = await env.say("no", "unclear", {})  # PATH A: not registered -- the existing ask follows
    assert "which test" in " ".join(said).lower() and env.session.pending_enquiry is not None
    said = await env.say("the same test", "unclear", {})
    assert spy.calls[-1] == ("test_prep", {"test_name": "Serum Creatinine"})


@pytest.mark.asyncio
async def test_after_the_question_lapses_a_bare_unclear_turn_is_not_pulled_back_into_it(m, env, spy):
    await env.say("price of creatinine", "test_rate", {"test_name": "Serum Creatinine"})
    env.session.last_enquiry_turn -= 20
    await env.say("do I need to fast", "test_prep", {})  # asks "which test?"
    await env.say("hmm", "smalltalk", {}, direct_reply_bn="Okay.")  # a different turn: consumes it
    n = len(spy.calls)
    await env.say("blah", "unclear", {})
    assert len(spy.calls) == n
