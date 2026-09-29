"""Unit tests for agent/confidence.py::zone().

story title: A low-confidence turn is confirmed, never acted upon
user story: As a caller who was misheard, I want the agent to check before
  doing anything, so that a guess never becomes a booking.
acceptance criteria: Below the calibrated agreement threshold the turn takes
  a confirmation path. No write occurs on a low-confidence turn without an
  explicit affirmative on the same call. Both error directions of the
  threshold are published.

Pure function under test, no I/O, no mocking, no services: zone() takes one
ASR result and returns one of PROCEED / CONFIRM / REJECT. Every case below
is a boundary around AGREEMENT_REJECT_FLOOR (0.30) and
AGREEMENT_CONFIRM_FLOOR (0.60), plus the two special signals that are not
plain agreement scores: the ctc_fallback decoder and an absent comparison
(agreement=None).
"""
from __future__ import annotations

import types

import pytest

from agent import confidence


def asr_result(agreement, decoder_used="rnnt"):
    return types.SimpleNamespace(decoder_agreement=agreement, decoder_used=decoder_used)


class TestAgreementBoundaries:
    """The two floors, and one step to either side of each."""

    @pytest.mark.parametrize("agreement", [0.60, 0.80, 1.0])
    def test_at_or_above_confirm_floor_proceeds(self, agreement):
        assert confidence.zone(asr_result(agreement)) == confidence.PROCEED

    @pytest.mark.parametrize("agreement", [0.59, 0.45, 0.30])
    def test_between_floors_confirms(self, agreement):
        assert confidence.zone(asr_result(agreement)) == confidence.CONFIRM

    @pytest.mark.parametrize("agreement", [0.29, 0.15])
    def test_below_reject_floor_rejects(self, agreement):
        assert confidence.zone(asr_result(agreement)) == confidence.REJECT

    def test_exactly_on_confirm_floor_is_inclusive_of_proceed(self):
        # zone() uses `< AGREEMENT_CONFIRM_FLOOR` for CONFIRM, so the floor
        # value itself falls through to PROCEED.
        assert confidence.zone(asr_result(confidence.AGREEMENT_CONFIRM_FLOOR)) == confidence.PROCEED

    def test_exactly_on_reject_floor_is_inclusive_of_confirm(self):
        # Same shape at the lower floor: `< AGREEMENT_REJECT_FLOOR` for
        # REJECT, so the floor value itself is CONFIRM, not REJECT.
        assert confidence.zone(asr_result(confidence.AGREEMENT_REJECT_FLOOR)) == confidence.CONFIRM


class TestZeroAgreementBothDirections:
    """agreement=0.0 means two different things depending on why it is 0.0 --
    real total disagreement (RNNT ran and the words didn't match) versus no
    second opinion at all (RNNT produced nothing, CTC's text was used
    instead). Scoring both the same way is exactly the bug this story
    guards against."""

    def test_zero_agreement_from_a_real_rnnt_comparison_rejects(self):
        result = asr_result(0.0, decoder_used="rnnt")
        assert confidence.zone(result) == confidence.REJECT

    def test_zero_agreement_from_ctc_fallback_confirms_not_rejects(self):
        # decoder_used == "ctc_fallback" short-circuits before the agreement
        # number is even inspected.
        result = asr_result(0.0, decoder_used="ctc_fallback")
        assert confidence.zone(result) == confidence.CONFIRM

    def test_ctc_fallback_confirms_even_with_a_high_agreement_value(self):
        # The decoder_used check runs first regardless of what agreement
        # happens to hold -- ctc_fallback always means "only one opinion",
        # never "proceed".
        result = asr_result(0.95, decoder_used="ctc_fallback")
        assert confidence.zone(result) == confidence.CONFIRM


class TestMissingAgreementSignal:
    """No comparison was made at all (agreement=None) -- treated as doubt,
    not as confidence, so it must never fall through to PROCEED."""

    def test_none_agreement_confirms(self):
        result = asr_result(None, decoder_used="rnnt")
        assert confidence.zone(result) == confidence.CONFIRM

    def test_missing_decoder_agreement_attribute_entirely_confirms(self):
        # zone() reads decoder_agreement via getattr(..., None), so an
        # object that never set the attribute must behave identically to
        # one that explicitly set it to None.
        result = types.SimpleNamespace(decoder_used="rnnt")
        assert confidence.zone(result) == confidence.CONFIRM


class TestReturnValuesAreTheDeclaredConstants:
    def test_zone_never_returns_anything_outside_the_three_constants(self):
        seen = {
            confidence.zone(asr_result(0.0)),
            confidence.zone(asr_result(0.30)),
            confidence.zone(asr_result(0.60)),
            confidence.zone(asr_result(1.0)),
            confidence.zone(asr_result(None)),
            confidence.zone(asr_result(0.0, decoder_used="ctc_fallback")),
        }
        assert seen == {confidence.PROCEED, confidence.CONFIRM, confidence.REJECT}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
