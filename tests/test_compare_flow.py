"""agent/compare_flow.py -- the arithmetic, and the recommendation it refuses to make.

Two things carry real risk here and the tests are built around them.

**Money.** This is the FIRST place in the repository that does arithmetic on a
live price; until now every number passed through a template unchanged. The
number-fidelity rule exists because `float("450.00") == float("450.0")` eats a
digit that matters, so the arithmetic is `Decimal` over the exact source string.
The group below proves that with the two cases floats actually get wrong: the
classic 0.1/0.3 binary-fraction error, and a trailing zero.

**Not giving advice.** The story forbids a clinical or purchasing
recommendation, and a caller may ask for one directly ("which is better?"). The
detector matches that phrasing on purpose -- letting it fall through to the model
is how an opinion gets spoken -- so the reply has to answer with prices and
nothing else. That is asserted on the strings, in all three languages.
"""

from __future__ import annotations

import pytest

from agent import persona
from agent.compare_flow import (
    KIND_NOT_FOUND,
    KIND_PACKAGE,
    KIND_TEST,
    asks_to_compare,
    build_comparison,
    detect_comparison_request,
)
from agent.phrases import PHRASES, phrase
from agent.reply_templates import compare_options_reply

LANGS = ("bn", "hi", "en")


def _test(rate):
    return {"kind": KIND_TEST, "found": True, "rate_inr": rate}


def _pkg(price, tests):
    return {
        "kind": KIND_PACKAGE,
        "found": True,
        "bundled_price_inr": price,
        "tests": [{"name": n} for n in tests],
    }


# ---------------------------------------------------------------- detection
@pytest.mark.parametrize(
    "text,family",
    [
        ("which is cheaper, CBC or lipid profile", "en"),
        ("what is the price difference between them", "en"),
        ("compare CBC and ESR", "en"),
        ("which one is better", "en"),
        ("konta sosta", "latin"),
        ("kaun sa sasta hai", "latin"),
        ("fark kitna hai", "latin"),
        ("কোনটা সস্তা", "bn"),
        ("দামের পার্থক্য কত", "bn"),
        ("কোনটা ভালো", "bn"),
        ("कौन सा सस्ता है", "hi"),
        ("कीमत में अंतर कितना है", "hi"),
        ("कौन सा बेहतर है", "hi"),
    ],
)
def test_a_request_to_compare_is_caught_in_every_language(text, family):
    matched, tag = detect_comparison_request(text)
    assert matched is True
    assert tag == family
    assert asks_to_compare(text) is True


@pytest.mark.parametrize("lang", LANGS)
def test_every_language_has_its_own_phrase_list(lang):
    import agent.compare_flow as m

    table = {"bn": m._BN, "hi": m._HI, "en": m._EN}[lang]
    assert len(table) >= 10, f"{lang} list is too thin to be real coverage"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "what is the price of a CBC test",
        "is Dr Sen available tomorrow",
        "I want to book a CBC",
        "আমার রিপোর্ট কোথায়",
        "मुझे रिपोर्ट चाहिए",
    ],
)
def test_an_ordinary_single_question_is_not_a_comparison(text):
    assert asks_to_compare(text) is False


# ------------------------------------------------------- money, exactly
def test_the_difference_is_computed_in_decimal_not_float():
    """The classic binary-fraction error. `float` gives 0.19999999999999998 for
    this, which would then be SPOKEN with complete confidence."""
    out = build_comparison(_test("0.3"), _test("0.1"))
    assert out["price_delta"] == "0.2"


def test_a_trailing_zero_is_preserved_in_the_difference():
    """`float("450.00") == float("450.0")` is True, which is exactly why
    tools_client parses money with `parse_float=str`. Two equal-but-differently-
    written prices have a zero difference, and it keeps its digits."""
    out = build_comparison(_test("450.00"), _test("450.0"))
    assert out["price_delta"] == "0.00"
    assert out["cheaper"] is None


def test_the_cheaper_side_is_identified_both_ways_round():
    assert build_comparison(_test("450"), _test("900"))["cheaper"] == "a"
    assert build_comparison(_test("900"), _test("450"))["cheaper"] == "b"


def test_equal_prices_have_no_cheaper_side():
    """None, not an arbitrary pick: there is no cheaper one, and saying there is
    would be a statement the data does not support."""
    out = build_comparison(_test("450"), _test("450"))
    assert out["cheaper"] is None
    assert out["price_delta"] == "0"


def test_an_int_price_and_a_string_price_compare_correctly():
    """The API sends whole rupees as an int and fractional ones as a string."""
    out = build_comparison(_test(650), _test("199.55"))
    assert out["cheaper"] == "b"
    assert out["price_delta"] == "450.45"


def test_a_missing_or_unparseable_price_is_a_fact_not_a_crash():
    for bad in (None, "", "not a number"):
        out = build_comparison(_test(bad), _test("450"))
        assert out["price_delta"] is None
        assert out["cheaper"] is None


def test_a_side_that_was_not_found_stops_the_comparison():
    out = build_comparison(_test("450"), {"kind": KIND_NOT_FOUND})
    assert out["b_found"] is False
    assert out["price_delta"] is None and out["cheaper"] is None


# ------------------------------------------------- two packages: test counts
def test_two_packages_report_which_has_more_tests_and_which_differ():
    out = build_comparison(_pkg("1200", ["CBC", "ESR", "TSH"]), _pkg("900", ["CBC", "ESR"]))
    assert out["both_packages"] is True
    assert out["identical_tests"] is False
    assert out["test_count_delta"] == 1
    assert out["more_tests_side"] == "a"
    assert [t["name"] for t in out["extra_tests_a"]] == ["TSH"]
    assert out["extra_tests_b"] == []


def test_an_equal_count_of_different_tests_names_no_bigger_side():
    """`more_tests_side` is about the COUNT. Two packages of two different tests
    each have no bigger side, and `identical_tests` is what records that they
    are nonetheless not the same."""
    out = build_comparison(_pkg("900", ["CBC", "TSH"]), _pkg("900", ["ESR", "LFT"]))
    assert out["more_tests_side"] is None
    assert out["test_count_delta"] == 0
    assert out["identical_tests"] is False


def test_two_tests_are_not_treated_as_packages():
    out = build_comparison(_test("450"), _test("900"))
    assert out["both_packages"] is False
    assert out["identical_tests"] is None and out["test_count_delta"] is None


# -------------------------------------------- no recommendation, ever
def test_the_facts_dict_carries_no_judgement_field():
    """Structural, not wording: there is no field a template could render as an
    opinion, because neither catalogue exposes a clinical or suitability axis."""
    out = build_comparison(_pkg("1200", ["CBC"]), _pkg("900", ["ESR"]))
    for forbidden in ("better", "best", "recommended", "recommendation", "suitable", "advice", "worth_it"):
        assert forbidden not in out


@pytest.mark.parametrize("lang", LANGS)
def test_asking_which_is_better_is_answered_with_prices_only(lang):
    """The caller asked for an opinion. The reply gives the arithmetic and asks
    them to choose -- it never ranks the two on anything else."""
    out = build_comparison(_test("450"), _test("900"))
    said = compare_options_reply(out, "CBC", "Lipid Profile", lang)
    assert "450" in said
    assert said.count("?") == 1
    lowered = said.lower()
    for opinion in ("better", "best", "recommend", "should choose", "बेहतर", "सबसे अच्छा", "ভালো হবে", "পরামর্শ"):
        assert opinion not in lowered


@pytest.mark.parametrize("lang", LANGS)
def test_the_reply_names_both_options_and_passes_the_persona(lang):
    out = build_comparison(_test("450"), _test("900"))
    said = compare_options_reply(out, "CBC", "Lipid Profile", lang)
    assert "CBC" in said and "Lipid Profile" in said
    assert persona.violations(said) == [], persona.violations(said)


@pytest.mark.parametrize("lang", LANGS)
def test_an_unfound_option_is_named_and_the_names_are_asked_for_again(lang):
    out = build_comparison(_test("450"), {"kind": KIND_NOT_FOUND})
    said = compare_options_reply(out, "CBC", "Zebra Panel", lang)
    assert "Zebra Panel" in said
    assert said.count("?") == 1
    assert "450" not in said, "a price was quoted for a comparison that could not be made"


@pytest.mark.parametrize("lang", LANGS)
def test_there_is_a_line_for_resolving_only_one_of_the_two(lang):
    """Spoken when the compare cue was heard but only one option matched the
    catalogue. Asking is the only honest move -- comparing against a guessed
    second item would state a difference never computed from real data."""
    assert "compare_need_two" in PHRASES[lang]
    said = phrase("compare_need_two", lang)
    assert said.count("?") == 1
    assert persona.violations(said) == []


def test_phrase_coverage_is_still_identical_across_languages():
    assert sorted(PHRASES["bn"]) == sorted(PHRASES["hi"]) == sorted(PHRASES["en"])
