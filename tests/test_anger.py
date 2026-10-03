"""agent/anger.py -- the detector, and the dead policy it was written to light up.

WHY THESE TESTS EXIST, AND WHAT EACH GROUP IS ACTUALLY PROVING:

agent/anger.py is deliberately only a detector: the seven acceptance criteria of
the story it was ported for are satisfied by machinery this repository already
had (agent/speech_policy.py's "angry" row, agent/acknowledgement.py's "angry"
entry, agent/apology.py's single-apology rule, agent/call_state.py's log dict).
Before the port, nothing in the repository ever set `caller_state="angry"`, so
all of that was unreachable.

That makes the risk here unusual, and it is what these tests are aimed at. A
test that only checked "angry sentence -> True" would pass while the feature
stayed completely dead, because the detector's output has to reach a caller
through four other modules to change anything. So the groups below prove, in
order: (1) the detector fires on real anger in every language this system
serves, including the Hindi written new for this repo; (2) it does NOT fire on
ordinary unhappy-but-calm speech, since a false positive hijacks the turn;
(3) the state it sets actually derives the policy the story demanded, with the
numbers checked against the story's text rather than against the code; and
(4) the apology the caller hears is exactly one, in each language, which is the
single criterion most likely to regress silently when wording is edited later.
"""

from __future__ import annotations

import pytest

from agent import abuse, anger
from agent.acknowledgement import acknowledgement_for, select_acknowledgement
from agent.apology import count_apologies
from agent.call_state import VALID_CALLER_STATES, apply_caller_state, new_call_state
from agent.speech_policy import derive_policy

LANGS = ("bn", "hi", "en")


# ---------------------------------------------------------------- detection

# One real utterance per phrase family. The `latin` family matters on its own
# because the recogniser transliterates as often as it writes native script.
@pytest.mark.parametrize(
    "text,family",
    [
        ("I am very angry about this", "en"),
        ("this is absolutely ridiculous", "en"),
        ("you people are useless", "en"),
        ("mujhe bahut gussa aa raha hai", "latin"),
        ("ami khub raag korchi", "latin"),
        ("bas bahut ho gaya", "latin"),
        ("আমি খুব রেগে গেছি", "bn"),
        ("এটা একদম বিরক্তিকর", "bn"),
        ("আপনারা কোনো কাজের না", "bn"),
        ("मुझे बहुत गुस्सा आ रहा है", "hi"),
        ("यह बिलकुल बकवास है", "hi"),
        ("आप लोग बेकार हैं", "hi"),
    ],
)
def test_real_anger_is_detected_in_every_language_and_reports_which_list_matched(text, family):
    matched, tag = anger.detect_anger(text)
    assert matched is True
    assert tag == family
    assert anger.is_anger(text) is True


def test_anger_is_detected_whatever_language_the_rest_of_the_turn_is_in():
    """A Bengali-speaking caller swearing off in English must still be heard.

    This is the same assumption agent/abuse.py already makes for swearing, and
    it is why detect_anger() takes no `lang` argument: the detected language of
    the turn and the language the anger is expressed in are routinely different,
    and gating the lists by the turn's language would silently drop the most
    common real case.
    """
    assert anger.is_anger("দেখুন, this is ridiculous, আমি কিছু বুঝতে পারছি না") is True
    assert anger.is_anger("सुनिए, i am fed up with this") is True


@pytest.mark.parametrize("lang", LANGS)
def test_every_language_has_its_own_phrase_list(lang):
    """Guards against the port's own worst failure mode: shipping a trilingual
    system whose anger detection only covers one language. The Hindi list in
    particular did not exist in the module this was ported from."""
    families = dict(anger._FAMILIES)
    assert lang in families
    assert len(families[lang]) >= 10, f"{lang} list is too thin to be real coverage"


# ------------------------------------------------------------ no false fire

# Unhappy, confused or merely reporting a problem -- none of these is anger, and
# all of them must keep their normal flow. A false positive here does not just
# add a sentence: it slows the whole call down and offers a human to a caller
# who only asked a question.
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I have a problem with my report",
        "this test is bad for me",
        "আমি একটু বিরক্ত হয়েছি কিন্তু ঠিক আছে" ,
        "amar report ta kothay",
        "मुझे रिपोर्ट चाहिए",
        "can you tell me the CBC rate",
        "ডক্টর সেন কি আজ আছেন",
        "I am not happy with the timing, can I change it",
    ],
)
def test_ordinary_or_merely_unhappy_speech_is_not_anger(text):
    assert anger.is_anger(text) is False
    assert anger.detect_anger(text) == (False, None)


def test_anger_and_abuse_stay_separate_concerns():
    """They are two different states with two different responses -- a calm
    boundary for swearing, a slowed turn plus a human for anger -- so no phrase
    may sit on both lists. main.py checks abuse first; this test makes sure that
    ordering is a real decision about distinct inputs and not a tie-break
    between two lists that overlap.
    """
    for _tag, phrases in anger._FAMILIES:
        for p in phrases:
            assert not abuse.is_abusive(p), f"{p!r} is on both the anger and the abuse list"


# -------------------------------------------- the policy the detector unlocks


def test_angry_is_a_valid_caller_state_and_setting_it_derives_the_story_policy():
    """The four numbers asserted here are the story's acceptance criteria 1-3,
    not a restatement of speech_policy.py. If someone later softens the "angry"
    row, this fails and names the criterion that was lost."""
    assert "angry" in VALID_CALLER_STATES
    state = new_call_state()
    apply_caller_state(state, "angry")
    assert state.caller_state == "angry"

    policy = derive_policy("angry")
    assert policy.escalation_threshold == "low"  # AC 1: anger lowers the threshold
    assert policy.speech_rate < 1.0  # AC 2: delivery slows
    assert policy.offer_human is True  # AC 3: a human is offered explicitly
    assert policy.acknowledge_first is True
    assert policy.questions_per_turn == 1


def test_the_frustration_reaches_the_context_packet():
    """AC 7. The packet is what a human picking the call up actually reads."""
    state = new_call_state()
    apply_caller_state(state, "angry")
    assert state.to_log_dict()["caller_state"] == "angry"


@pytest.mark.parametrize("lang", LANGS)
def test_the_caller_hears_exactly_one_apology_in_their_own_language(lang):
    """AC 4, and the criterion most likely to regress when wording is edited:
    the "angry" acknowledgement already contains an apology, so anything that
    adds a second one anywhere on this path breaks the story. Checked through
    agent/apology.py's own counter rather than by eye."""
    line = acknowledgement_for("angry", lang)
    assert line, f"no angry acknowledgement for {lang}"
    assert count_apologies(line, lang) == 1


@pytest.mark.parametrize("lang", LANGS)
def test_the_angry_policy_actually_selects_that_acknowledgement_and_offers_a_person(lang):
    """End of the chain the detector starts: this is the sentence a real caller
    hears. It proves the port is wired to something live rather than merely
    setting a field nobody reads."""
    spoken = select_acknowledgement(derive_policy("angry"), "angry", lang)
    assert spoken, f"the angry policy selected no acknowledgement for {lang}"
    assert acknowledgement_for("angry", lang) in spoken
    # offer_human=True, so the selected line must end up asking to connect one.
    assert spoken != acknowledgement_for("angry", lang), (
        "offer_human is set for the angry policy, so the spoken line should carry "
        "the offer of a person as well as the acknowledgement"
    )


def test_a_neutral_call_is_unchanged_by_this_port():
    """The regression guard for every caller who is not angry: the default path
    must not acquire an acknowledgement, a slower rate or a human offer."""
    policy = derive_policy("neutral")
    assert policy.offer_human is False
    assert policy.speech_rate == 1.0
    assert select_acknowledgement(policy, "neutral", "bn") is None
