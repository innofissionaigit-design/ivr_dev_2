"""Unit tests for agent/lid.py's ASRLanguageRouter -- the orchestration
half of LID-before-ASR, which needs no model, no audio and no pod. Runs
anywhere with plain pytest:

    python -m pytest tests/test_lid.py -v

(from the repository root; unlike tests/test_smoke.py this has no
/workspace path dependency and no live services to reach.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.lid import ASRLanguageRouter, LIDResult, RoutingDecision


def test_high_confidence_commits_immediately():
    router = ASRLanguageRouter()
    decision = router.route(LIDResult(language="bn", confidence=0.92))
    assert decision.action == "commit"
    assert decision.language == "bn"


def test_low_confidence_with_no_prior_runs_dual_asr():
    router = ASRLanguageRouter()
    decision = router.route(LIDResult(language="hi", confidence=0.30))
    assert decision.action == "dual_asr"
    assert decision.language is not None


def test_low_confidence_uses_previous_turn_as_prior():
    router = ASRLanguageRouter()
    first = router.route(LIDResult(language="en", confidence=0.95))
    assert first.action == "commit" and first.language == "en"

    second = router.route(LIDResult(language="unknown", confidence=0.10))
    assert second.action == "commit"
    assert second.language == "en", "should fall back to the previous turn's language"


def test_ambiguous_streak_without_prior_hands_off_to_human():
    router = ASRLanguageRouter(max_ambiguous_streak=2)
    # No prior ever established -- every low-confidence turn is ambiguous.
    d1 = router.route(LIDResult(language="unknown", confidence=0.10))
    d2 = router.route(LIDResult(language="unknown", confidence=0.15))
    d3 = router.route(LIDResult(language="unknown", confidence=0.05))
    assert d1.action == "dual_asr"
    assert d2.action == "dual_asr"
    assert d3.action == "handoff_human", "must stop guessing after max_ambiguous_streak"


def test_committing_resets_the_ambiguous_streak():
    router = ASRLanguageRouter(max_ambiguous_streak=1)
    router.route(LIDResult(language="unknown", confidence=0.10))  # streak = 1, dual_asr
    router.route(LIDResult(language="bn", confidence=0.9))  # commits on its own confidence
    assert router._ambiguous_streak == 0, "a high-confidence commit must reset the streak counter"
    # A later low-confidence turn now has "bn" as a prior, so it commits
    # via the prior rather than re-entering the ambiguous-streak path --
    # that path is only reachable again once there is no prior at all.
    decision = router.route(LIDResult(language="unknown", confidence=0.10))
    assert decision.action == "commit" and decision.language == "bn"


def test_dual_asr_prefers_scored_top_two_when_available():
    router = ASRLanguageRouter()
    lid = LIDResult(
        language="unknown",
        confidence=0.20,
        scores={"bn": 0.20, "hi": 0.55, "en": 0.15},
    )
    decision = router.route(lid)
    assert decision.action == "dual_asr"
    assert decision.language == "hi"
    assert decision.secondary_language == "bn"


def test_note_response_language_seeds_the_prior():
    router = ASRLanguageRouter()
    # A clarification turn resolved the language without LID ever
    # committing on its own -- the response layer tells the router.
    router.note_response_language("hi")
    decision = router.route(LIDResult(language="unknown", confidence=0.05))
    assert decision.action == "commit"
    assert decision.language == "hi"


def test_routing_decision_is_a_plain_dataclass():
    # Guards the orchestrator's contract: it pattern-matches on .action.
    d = RoutingDecision(action="commit", language="en")
    assert d.action == "commit"
    assert d.secondary_language is None
    assert d.reason == ""
