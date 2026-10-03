"""agent/lay_terms.py: rural lay terms matched to a category, never to a specific test.

python -m pytest tests/test_lay_terms.py -v
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent.lay_terms import ABDOMEN, BLOOD, HEAD, match_lay_term, normalise


@pytest.mark.parametrize(
    "text,expected",
    [
        ("আমার রক্ত পরীক্ষা করাতে হবে", BLOOD),
        ("rokto porikkha korte hobe", BLOOD),
        ("rokto porikha korate chai", BLOOD),
        ("खून की जांच करानी है", BLOOD),
        ("আমার একটা পেটের ছবি করাতে হবে", ABDOMEN),
        ("peter chhobi korate chai", ABDOMEN),
        ("peter porikkha korte hobe", ABDOMEN),
        ("पेट की तस्वीर चाहिए", ABDOMEN),
        ("মাথার ছবি করাতে হবে", HEAD),
        ("mathar chhobi korate chai", HEAD),
        ("mathar porikkha korte hobe", HEAD),
        ("सिर की तस्वीर चाहिए", HEAD),
    ],
)
def test_a_lay_phrase_matches_its_category_in_any_script(text, expected):
    assert match_lay_term(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "সিবিসি টেস্টের দাম কত",  # a real test name, not a lay term
        "আমার নাম রবি দাস",
        "",
        "chhobi",  # too short/generic on its own -- not a recognised phrase
        "hello there",
    ],
)
def test_unrelated_text_matches_nothing(text):
    assert match_lay_term(text) is None


def test_filler_words_around_the_phrase_do_not_block_the_match():
    assert match_lay_term("amar ekta rokto porikkha korte hobe please") == BLOOD


def test_matching_is_case_and_punctuation_insensitive():
    assert match_lay_term("ROKTO, PORIKKHA!!") == BLOOD
    assert match_lay_term("Peter Chhobi.") == ABDOMEN


def test_the_longer_more_specific_phrase_wins_when_more_than_one_matches(monkeypatch):
    import agent.lay_terms as lt

    # An isolated, controlled table: "aa bb" (BLOOD, the longer match) is contained in a sentence alongside the
    # shorter "cc" (HEAD) -- the longer, more specific phrase must win, not whichever happens to be scanned first.
    monkeypatch.setattr(lt, "_PHRASES", {"cc": HEAD, "aa bb": BLOOD})
    assert lt.match_lay_term("xx cc aa bb yy") == BLOOD
    assert lt.match_lay_term("xx cc yy") == HEAD


def test_two_real_categories_in_one_sentence_returns_one_of_them_not_a_crash():
    result = match_lay_term("rokto porikkha ar peter chhobi dutoi lagbe")
    assert result in (BLOOD, ABDOMEN)


def test_normalise_never_raises_on_arbitrary_text():
    for text in ["", "   ", "!!!", "​​", "🩸🩺"]:
        normalise(text)  # must not raise


@given(st.text(max_size=200))
def test_match_lay_term_never_raises(text):
    match_lay_term(text)
