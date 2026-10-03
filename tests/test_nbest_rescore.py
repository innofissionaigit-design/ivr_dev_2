"""agent/nbest_rescore.py: choosing the best of an ASR's N-best hypotheses against the clinic catalogue.

    python -m pytest tests/test_nbest_rescore.py -v

Pure and synthetic: a plain list of candidate transcripts stands in for what real N-best output will look like: no
pod, no model, no recording. tests/test_asr_evaluation.py's small catalogue is reused so the fixtures agree.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent.nbest_rescore import EXACT, NONE, CatalogueRescorer
from agent.transcript_rules import DOCTOR, TEST

CAT = {
    "tests": [
        {"name": "Complete Blood Count (CBC)", "aliases_bn": ["সিবিসি", "সি বি সি"], "aliases_hi": ["सीबीसी"]},
        {"name": "CRP (C-Reactive Protein)", "aliases_bn": ["সিআরপি"], "aliases_hi": []},
    ],
    "doctors": [
        {
            "name": "Dr. A. Sen",
            "full_name": "Dr. Arindam Sen",
            "surname": "Sen",
            "aliases_bn": ["সেন"],
            "aliases_hi": [],
        },
        {
            "name": "Dr. P. Ghosh",
            "full_name": "Dr. Prabir Ghosh",
            "surname": "Ghosh",
            "aliases_bn": ["ঘোষ"],
            "aliases_hi": [],
        },
    ],
}
R = CatalogueRescorer(CAT)


def test_rank_zero_is_kept_when_it_already_names_the_entity_exactly():
    result = R.rescore(["সিবিসি টেস্টের দাম কত", "garbage"], TEST, "bn")
    assert (result.rank, result.tier, result.promoted) == (0, EXACT, False)
    assert result.text == "সিবিসি টেস্টের দাম কত"


def test_a_lower_ranked_hypothesis_with_an_exact_entity_is_promoted_over_one_with_none():
    hyps = ["how much is the price", "how much is the CBC test", "unrelated noise"]
    result = R.rescore(hyps, TEST, "en")
    assert (result.rank, result.tier, result.promoted) == (1, EXACT, True)
    assert result.text == hyps[1]


def test_exact_always_beats_a_mere_suggestion_regardless_of_rank():
    # "সিবি" alone is not a catalogue form (no exact hit); a mangled "সিবিসি" might only be suggested
    hyps = ["সিবি সি টেস্ট", "সিবিসি টেস্টের দাম কত"]  # rank 0: no exact match; rank 1: exact
    result = R.rescore(hyps, TEST, "bn")
    assert result.tier == EXACT and result.rank == 1


def test_a_suggestion_beats_nothing_but_never_beats_an_exact_match_at_a_worse_rank():
    hyps = ["completely unrelated words with nothing test-like", "cbc test please"]
    result = R.rescore(hyps, TEST, "en")
    assert result.rank == 1 and result.tier == EXACT  # rank 1 is exact and wins outright


def test_when_nothing_beats_rank_zero_it_is_returned_unpromoted():
    hyps = ["nothing here at all", "still nothing", "and nothing again"]
    result = R.rescore(hyps, TEST, "en")
    assert (result.rank, result.tier, result.promoted) == (0, NONE, False)


def test_promotion_never_reaches_past_the_configured_rank():
    hyps = ["nothing"] * 5 + ["cbc test"]  # the exact match is at rank 5
    assert R.rescore(hyps, TEST, "en", max_promote_rank=4).promoted is False  # out of reach: rank 0 stands
    result = R.rescore(hyps, TEST, "en", max_promote_rank=5)
    assert result.rank == 5 and result.tier == EXACT  # reachable now


def test_doctors_and_tests_are_scored_independently():
    hyps = ["ডাক্তার সেন কবে বসবেন", "সিবিসি টেস্টের দাম কত"]
    assert R.rescore(hyps, DOCTOR, "bn") == R.rescore(hyps, DOCTOR, "bn")  # deterministic
    doctor_result = R.rescore(hyps, DOCTOR, "bn")
    test_result = R.rescore(hyps, TEST, "bn")
    assert doctor_result.rank == 0 and doctor_result.tier == EXACT
    assert test_result.rank == 1 and test_result.tier == EXACT


def test_empty_hypotheses_is_a_safe_empty_result_not_an_error():
    assert R.rescore([], TEST, "en") == R.rescore((), TEST, "en") == R.__class__(CAT).rescore([], TEST, "en")
    result = R.rescore([], TEST, "en")
    assert (result.text, result.rank, result.tier, result.promoted) == ("", 0, NONE, False)


@given(st.lists(st.text(max_size=30), max_size=8), st.sampled_from([DOCTOR, TEST]), st.sampled_from(["bn", "hi", "en"]))
def test_rescoring_never_raises_and_the_result_is_always_one_of_the_given_hypotheses_or_empty(hyps, kind, lang):
    result = R.rescore(hyps, kind, lang)
    if hyps:
        assert result.text in hyps and 0 <= result.rank < len(hyps)
    else:
        assert result.text == "" and result.rank == 0


@given(st.lists(st.text(max_size=30), min_size=1, max_size=8), st.sampled_from([DOCTOR, TEST]))
def test_the_chosen_hypothesis_is_never_a_worse_tier_than_rank_zero(hyps, kind):
    from agent.nbest_rescore import _TIER_ORDER  # the module's own ordering, used to state the invariant precisely

    result = R.rescore(hyps, kind, "bn")
    rank_zero_tier = R._tier(hyps[0], kind, "bn")
    assert _TIER_ORDER[result.tier] <= _TIER_ORDER[rank_zero_tier]


def test_wrong_kind_is_never_promoted_for_a_hit_of_the_other_kind():
    # a hypothesis that names a DOCTOR exactly must not promote a TEST rescoring, and vice versa
    hyps = ["nothing at all", "ডাক্তার সেন কবে বসবেন"]  # rank 1 exactly names a doctor, no test anywhere
    result = R.rescore(hyps, TEST, "bn")
    assert result.rank == 0 and result.tier == NONE


@pytest.mark.parametrize("kind", [DOCTOR, TEST])
def test_rescoring_is_read_only_the_catalogue_gazetteers_are_not_mutated_by_repeated_calls(kind):
    before = (R._doctor_gaz.stats["queries"], R._test_gaz.stats["queries"])
    R.rescore(["সিবিসি", "ডাক্তার সেন"], kind, "bn")
    after_first = (R._doctor_gaz.stats["queries"], R._test_gaz.stats["queries"])
    R.rescore(["সিবিসি", "ডাক্তার সেন"], kind, "bn")
    after_second = (R._doctor_gaz.stats["queries"], R._test_gaz.stats["queries"])
    # repeatable: the same input runs the same number of underlying gazetteer queries each time
    assert after_second[0] - after_first[0] == after_first[0] - before[0]
    assert after_second[1] - after_first[1] == after_first[1] - before[1]
