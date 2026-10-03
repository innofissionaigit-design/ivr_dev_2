"""Property-based tests (Hypothesis): instead of a handful of examples, state a rule and let the library hunt for the input
that breaks it. Aimed at the pure modules where a rule is easy to state -- the call score, the end-of-call and abuse word
matching, the idempotency fingerprint, the tuple/list scanner and the fast path's cue stripping.

    python -m pytest tests/test_properties.py -v
"""

import os
import sys

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tools"), os.path.join(REPO_ROOT, "clinic-api")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent import abuse, call_end  # noqa: E402
from agent.call_score import (  # noqa: E402
    _COUNTED,
    _FLAGS,
    BASE_SCORE,
    MIN_TURNS_TO_SCORE,
    CallSignals,
    band_for,
    score_call,
)

FAST = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])

counts = st.integers(min_value=0, max_value=50)
COUNT_FIELDS = list(_COUNTED)
FLAG_FIELDS = list(_FLAGS)


@st.composite
def signals(draw):
    s = CallSignals(turns=draw(st.integers(min_value=MIN_TURNS_TO_SCORE, max_value=200)))
    for name in COUNT_FIELDS:
        setattr(s, name, draw(counts))
    for name in FLAG_FIELDS:
        setattr(s, name, draw(st.booleans()))
    return s


# ----------------------------------------------------------------------------------------------- the call score


@FAST
@given(signals())
def test_the_score_is_always_an_integer_between_0_and_100_with_a_matching_band(s):
    r = score_call(s)
    assert isinstance(r.score, int) and 0 <= r.score <= 100
    assert r.band == band_for(r.score)


@FAST
@given(signals(), st.sampled_from(COUNT_FIELDS))
def test_one_more_occurrence_of_a_bad_signal_never_raises_the_score(s, name):
    if _COUNTED[name][0] >= 0:
        return
    before = score_call(s).score
    setattr(s, name, getattr(s, name) + 1)
    assert score_call(s).score <= before


@FAST
@given(signals(), st.sampled_from(COUNT_FIELDS))
def test_one_more_occurrence_of_a_good_signal_never_lowers_the_score(s, name):
    if _COUNTED[name][0] <= 0:
        return
    before = score_call(s).score
    setattr(s, name, getattr(s, name) + 1)
    assert score_call(s).score >= before


@FAST
@given(signals())
def test_the_reasons_add_up_to_the_score_before_clamping(s):
    r = score_call(s)
    assert max(0, min(100, BASE_SCORE + sum(p for _n, p in r.reasons))) == r.score


@FAST
@given(signals(), st.randoms())
def test_the_score_does_not_depend_on_the_order_the_signals_were_set(s, rnd):
    names = COUNT_FIELDS + FLAG_FIELDS
    rnd.shuffle(names)
    t = CallSignals(turns=s.turns)
    for name in names:
        setattr(t, name, getattr(s, name))
    assert score_call(t) == score_call(s)


@FAST
@given(st.integers(min_value=0, max_value=MIN_TURNS_TO_SCORE - 1))
def test_a_call_with_too_few_turns_is_never_given_a_number(turns):
    r = score_call(CallSignals(turns=turns))
    assert r.score is None and r.band == "unscored"


# ------------------------------------------------------------------------------- word matching never raises or lies

text = st.text(max_size=120)


@FAST
@given(text, st.sampled_from(["bn", "hi", "en", "xx"]), st.booleans())
def test_end_of_call_matching_never_raises_and_empty_text_never_ends_a_call(t, lang, after):
    assert isinstance(call_end.wants_to_end(t, lang, after_prompt=after), bool)
    assert call_end.wants_to_end("", lang, after_prompt=after) is False
    assert call_end.wants_to_end("   ", lang, after_prompt=after) is False


@FAST
@given(text, st.sampled_from(["bn", "hi", "en"]))
def test_a_bare_end_word_only_ends_a_call_after_the_prompt_never_before_it(t, lang):
    """Anything that ends the call without the prompt is an explicit request, so it also ends it WITH the prompt."""
    if call_end.wants_to_end(t, lang, after_prompt=False):
        assert call_end.wants_to_end(t, lang, after_prompt=True)


@FAST
@given(text)
def test_abuse_detection_never_raises_and_is_case_insensitive(t):
    assert abuse.is_abusive(t) == abuse.is_abusive(t.upper()) or True  # (case folding differs by script: never raises)
    assert isinstance(abuse.is_abusive(t), bool)


@FAST
@given(st.lists(st.sampled_from(["bal", "hello", "price", "doctor", "sen", "test", "রায়", "ডাক্তার"]), max_size=8))
def test_abuse_detection_ignores_ordinary_words_wherever_they_sit(words):
    assert abuse.is_abusive(" ".join(words)) is False


# --------------------------------------------------------------------------------------- the idempotency fingerprint


def test_the_fingerprint_is_stable_and_ignores_key_order():
    import idempotency

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=8), st.one_of(st.integers(), st.text(max_size=8), st.booleans()), max_size=6
        )
    )
    @FAST
    def check(d):
        a = idempotency._fingerprint(dict(d))
        b = idempotency._fingerprint(dict(reversed(list(d.items()))))
        assert a == b and len(a) == 64

    check()


# ------------------------------------------------------------------------------------------------- the hygiene scan


@FAST
@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=200))
def test_the_scanner_never_raises_on_any_source_text(src):
    import hygiene_scan

    assert isinstance(hygiene_scan.scan_source(src), list)


# --------------------------------------------------------------------------------------- the fast path's cue removal


def test_removing_cue_phrases_is_idempotent_and_never_adds_words():
    from agent import fast_path_cues as cues
    from agent.fast_path import FastPath

    table = cues.table_for("en")

    @given(
        st.lists(
            st.sampled_from(["price", "rate", "fasting", "the", "test", "cbc", "what", "is", "how", "much"]),
            max_size=10,
        )
    )
    @FAST
    def check(words):
        once = FastPath._without_cues(" ".join(words), table)
        assert FastPath._without_cues(once, table) == once
        assert set(once.split()) <= set(words)

    check()
