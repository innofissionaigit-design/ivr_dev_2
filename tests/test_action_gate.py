"""The single action gate (agent/action_gate.py): it agrees with the gates that already exist, and it never says ACT
unless recognition, entity and rules are all certain.

    python -m pytest tests/test_action_gate.py -v
"""

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent import action_gate as g
from agent.confidence_gate import (
    GATED_FACTUAL_INTENTS,
    LOW,
    needs_entity_readback,
    should_withhold_factual_answer,
)
from agent.intent_schema import VALID_INTENTS

DECODERS = [None, "rnnt", "ctc_fallback", "fastconformer", "unknown"]
AGREEMENTS = [0.0, 0.2, 0.49, 0.5, 0.7, 1.0]
READS = sorted(GATED_FACTUAL_INTENTS)
UNGATED = sorted(VALID_INTENTS - GATED_FACTUAL_INTENTS - g.WRITE_INTENTS)


def _decide(intent, decoder, agreement, **kw):
    return g.decide(intent, decoder_used=decoder, decoder_agreement=agreement, **kw)


def test_the_write_intents_are_all_real_intents():
    assert g.WRITE_INTENTS <= VALID_INTENTS
    assert not g.WRITE_INTENTS & GATED_FACTUAL_INTENTS


@pytest.mark.parametrize("intent,decoder,agreement", list(itertools.product(READS, DECODERS, AGREEMENTS)))
def test_for_every_read_it_gives_the_answer_the_existing_gates_give(intent, decoder, agreement):
    verdict = _decide(intent, decoder, agreement, entity=g.EXACT)
    if should_withhold_factual_answer(intent, agreement, decoder):
        assert verdict.action == g.REPEAT
    elif needs_entity_readback(intent, decoder, agreement):
        assert verdict.action == g.READ_BACK
    else:
        assert verdict.action == g.ACT


@pytest.mark.parametrize("intent,decoder,agreement", list(itertools.product(UNGATED, DECODERS, AGREEMENTS)))
def test_an_intent_that_states_no_fact_and_writes_nothing_is_never_held_back_by_confidence(intent, decoder, agreement):
    assert _decide(intent, decoder, agreement).action == g.ACT


@pytest.mark.parametrize(
    "intent,decoder,agreement", list(itertools.product(sorted(g.WRITE_INTENTS), DECODERS, AGREEMENTS))
)
def test_a_write_needs_the_callers_yes_and_a_certain_recognition(intent, decoder, agreement):
    unconfirmed = _decide(intent, decoder, agreement, entity=g.EXACT)
    confirmed = _decide(intent, decoder, agreement, entity=g.EXACT, confirmed=True)
    assert not unconfirmed.may_act
    from agent.confidence_gate import confidence_state

    if confidence_state(decoder, agreement) == LOW:
        # DELIBERATELY stricter than the gates that exist today: they let a low-confidence write reach the confirmation
        # question (which is itself a yes/no); this repeats the turn first. See the module docstring.
        assert unconfirmed.action == g.REPEAT and confirmed.action == g.REPEAT
    else:
        assert unconfirmed.action == g.CONFIRM and confirmed.action == g.ACT


def test_a_sound_alike_is_read_back_never_acted_on():
    for intent in sorted(VALID_INTENTS):
        for confirmed in (False, True):
            assert not _decide(intent, "rnnt", 1.0, entity=g.SUGGESTED, confirmed=confirmed).may_act


def test_an_ambiguous_or_missing_name_is_a_question():
    assert _decide("test_rate", "rnnt", 1.0, entity=g.AMBIGUOUS).action == g.CLARIFY
    assert _decide("book_appointment", "rnnt", 1.0, entity=g.ABSENT, confirmed=True).action == g.CLARIFY


def test_a_business_rule_beats_the_confirmation_question():
    assert _decide("book_appointment", "rnnt", 1.0, entity=g.EXACT, rules_ok=False).action == g.DECLINE


def test_the_rules_apply_in_the_stated_order():
    args = dict(entity=g.AMBIGUOUS, rules_ok=False)
    assert _decide("test_rate", "rnnt", 0.1, failed_attempts=3, **args).action == g.HANDOFF  # 1 beats 2
    assert _decide("test_rate", "rnnt", 0.1, **args).action == g.REPEAT  # 2 beats 3
    assert _decide("test_rate", "rnnt", 1.0, **args).action == g.CLARIFY  # 3 beats 6
    assert _decide("test_rate", "ctc_fallback", 0.0, entity=g.SUGGESTED).reason.startswith(
        "only a sound-alike"
    )  # 4 beats 5


def test_handoff_follows_the_reask_policys_limit():
    from agent.reask_policy import DEFAULT_MAX_REASKS

    assert _decide("test_rate", "rnnt", 1.0, failed_attempts=DEFAULT_MAX_REASKS).action == g.ACT
    assert _decide("test_rate", "rnnt", 1.0, failed_attempts=DEFAULT_MAX_REASKS + 1).action == g.HANDOFF


@given(
    intent=st.sampled_from(sorted(VALID_INTENTS)),
    decoder=st.sampled_from(DECODERS),
    agreement=st.floats(min_value=0.0, max_value=1.0),
    entity=st.sampled_from([g.EXACT, g.AMBIGUOUS, g.SUGGESTED, g.ABSENT, g.NOT_NEEDED]),
    rules_ok=st.booleans(),
    confirmed=st.booleans(),
    failed=st.integers(min_value=0, max_value=5),
)
def test_act_is_only_ever_said_when_everything_is_certain(
    intent, decoder, agreement, entity, rules_ok, confirmed, failed
):
    from agent.confidence_gate import confidence_state

    verdict = _decide(
        intent, decoder, agreement, entity=entity, rules_ok=rules_ok, confirmed=confirmed, failed_attempts=failed
    )
    if verdict.may_act:
        assert failed <= 2 and rules_ok and entity in (g.EXACT, g.NOT_NEEDED)
        assert intent not in g.WRITE_INTENTS or confirmed
        state = confidence_state(decoder, agreement)
        assert state != LOW or (intent not in GATED_FACTUAL_INTENTS | g.WRITE_INTENTS)
        assert not (intent in GATED_FACTUAL_INTENTS and needs_entity_readback(intent, decoder, agreement))
