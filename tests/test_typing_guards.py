"""The 42 mypy errors in agent/ were fixed without changing what the code does. This file is the evidence.

For every function whose body changed, the version from BEFORE the change is kept (tests/_legacy_typing/old_*.py, copied
from git) and the two are run side by side on generated inputs: they must return the same thing. Where the old code could
only crash (a value that "might be None" and, in an unreachable state, was), the new code's behaviour is tested directly.

    python -m pytest tests/test_typing_guards.py -v
"""

import os
import sys
import types

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _legacy_typing import (  # noqa: E402
    old_apology,
    old_barge_in,
    old_endpoint_calibration,
    old_language_policy,
    old_messages,
    old_near_end,
    old_pause_profile,
    old_security_input,
    old_speaker_change,
)

from agent import (  # noqa: E402
    apology,
    barge_in,
    endpoint_calibration,
    language_policy,
    lid,
    messages,
    near_end,
    pause_profile,
    security_input,
    speaker_change,
)
from agent.endpointing import EndpointConfig  # noqa: E402

FAST = settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def outcome(fn, *args):
    """(value, None) or (None, the exception's type): two implementations agree if they succeed alike or fail alike."""
    try:
        return fn(*args), None
    except Exception as e:  # noqa: BLE001
        return None, type(e)


# ------------------------------------------------------------------------------ security_input: number words -> digits

VOCAB = (
    sorted(security_input._WORDS)[:80] + sorted(security_input._FUSED_HUNDREDS) + sorted(security_input._HUNDRED_WORDS)
)
VOCAB += sorted(security_input._THOUSAND_WORDS) + ["5", "12", "1985", "007", "abc", "phone", "ই", "-", "9."]


@FAST
@given(st.lists(st.sampled_from(VOCAB), max_size=14))
def test_number_words_are_composed_exactly_as_before(tokens):
    assert outcome(security_input._to_numbers, list(tokens)) == outcome(old_security_input._to_numbers, list(tokens))


@pytest.mark.parametrize(
    "tokens",
    [
        sorted(security_input._FUSED_HUNDREDS)[:1] + ["5"],  # a fused hundred, then a number
        sorted(security_input._FUSED_HUNDREDS)[:1] + ["abc"],  # ...then a word
        sorted(security_input._FUSED_HUNDREDS)[:1],  # ...then nothing
        ["twenty", "five"],
        ["eighty", "five", "hundred", "thirty"],
    ],
)
def test_the_branches_the_narrowing_touched_still_agree(tokens):
    assert outcome(security_input._to_numbers, tokens) == outcome(old_security_input._to_numbers, tokens)


# ---------------------------------------------------------------------------------- language_policy / max(key=dict.get)


@FAST
@given(st.text(max_size=80))
def test_the_dominant_script_is_unchanged(text):
    assert language_policy.dominant_script(text) == old_language_policy.dominant_script(text)


@FAST
@given(st.dictionaries(st.text(min_size=1, max_size=4), st.integers(min_value=0, max_value=5), min_size=1, max_size=8))
def test_a_lambda_key_picks_the_same_winner_as_dict_get_including_on_ties(d):
    assert max(d, key=lambda k: d[k]) == max(d, key=d.get)
    assert min(d, key=lambda k: d[k]) == min(d, key=d.get)


def test_the_serve_rate_gap_is_unchanged():
    from agent.fast_path import serve_rate_gap

    by_lang = {
        "bn": {"served": 9, "abstained": 1},
        "hi": {"served": 5, "abstained": 5},
        "en": {"served": 8, "abstained": 2},
    }
    gap = serve_rate_gap(by_lang, margin=0.2, min_turns=5)
    assert gap["widest"] == ["bn", "hi"] and gap["gap"] == 0.4 and gap["defect"] is True


# ------------------------------------------------------------------------------------------------------ apology

APOLOGY_PARTS = [
    "Sorry",
    "Sorry,",
    "I am sorry.",
    "Hello.",
    "Your price is fixed.",
    "দুঃখিত।",
    "माफ़ कीजिए।",
    "Okay",
    "sorry sorry.",
]


@FAST
@given(st.lists(st.sampled_from(APOLOGY_PARTS), max_size=7), st.sampled_from(["en", "bn", "hi"]))
def test_at_most_one_apology_is_enforced_exactly_as_before(parts, lang):
    text = " ".join(parts)
    assert outcome(apology.enforce_single_apology, text, lang) == outcome(
        old_apology.enforce_single_apology, text, lang
    )


def test_two_apologies_in_the_first_sentence_never_hit_the_missing_match_guard():
    assert apology.enforce_single_apology("Sorry, sorry. Please wait.", "en") == old_apology.enforce_single_apology(
        "Sorry, sorry. Please wait.", "en"
    )


# ------------------------------------------------------------------------------------------------- messages.load


PAYLOADS = [
    None,
    [],
    "messages",
    5,
    {},
    {"messages": None},
    {"messages": []},
    {"messages": {"greeting": None}},
    {
        "messages": {"greeting": {"en": {"text": "Hello", "version": 3}, "bn": {"text": "  ", "version": 1}}},
        "version": 7,
    },
    {"messages": {"k": {"en": {"text": "x"}}}},
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_message_payloads_load_exactly_as_before_and_a_malformed_one_never_empties_the_cache(payload):
    for mod in (messages, old_messages):
        mod._cache.clear()
        mod._cache[("seed", "en")] = ("kept", 1)
    new = outcome(messages.load, payload)
    old = outcome(old_messages.load, payload)
    assert new == old and messages._cache == old_messages._cache
    if not (isinstance(payload, dict) and isinstance(payload.get("messages"), dict)):
        assert messages._cache == {("seed", "en"): ("kept", 1)}


# ---------------------------------------------------------------------------------------------- pause_profile


def _profile(mod, gaps, bump):
    p = mod.PauseProfile()
    p.gaps, p.bump = list(gaps), bump
    return p


@FAST
@given(st.lists(st.floats(min_value=0.05, max_value=4.0), max_size=14), st.sampled_from([1.0, 1.15, 1.3, 1.6]))
def test_the_per_caller_endpoint_config_and_snapshot_are_unchanged(gaps, bump):
    base = EndpointConfig()
    new, old = _profile(pause_profile, gaps, bump), _profile(old_pause_profile, gaps, bump)
    assert new.config(base) == old.config(base)
    assert new.snapshot() == old.snapshot()


def test_a_profile_with_no_gaps_reports_none_not_a_number():
    p = pause_profile.PauseProfile()
    assert (
        p.snapshot()["p90_s"] is None
        and p.snapshot()["median_s"] is None
        and p.config(EndpointConfig()) == EndpointConfig()
    )


# ----------------------------------------------------------------------------------------------- near_end


@FAST
@given(
    st.lists(st.booleans(), min_size=1, max_size=60),
    st.lists(st.floats(min_value=80.0, max_value=300.0), min_size=60, max_size=60),
)
def test_overlap_segments_are_unchanged(mask, f0):
    track = (None, np.array(f0[: len(mask)]), np.array(mask), None)
    results = []
    for mod in (near_end, old_near_end):
        original = mod._voiced_track
        mod._voiced_track = lambda samples, sr, track=track: track
        try:
            results.append(outcome(mod.overlap_segments, np.zeros(16000, dtype=np.float32), 16000))
        finally:
            mod._voiced_track = original
    assert results[0] == results[1]


@FAST
@given(
    st.integers(min_value=0, max_value=3),
    st.one_of(st.none(), st.floats(min_value=-90.0, max_value=0.0)),
    st.one_of(st.none(), st.floats(min_value=0.0, max_value=300.0)),
    st.floats(min_value=-90.0, max_value=0.0),
    st.floats(min_value=-5.0, max_value=400.0),
)
def test_the_background_verdict_is_unchanged(accepted, level, f0, heard_level, heard_f0):
    new, old = near_end.NearEndProfile(), old_near_end.NearEndProfile()
    for prof in (new, old):
        prof.accepted, prof.level_dbfs, prof.f0_hz = accepted, level, f0
    a, b = outcome(new.judge, heard_level, heard_f0), outcome(old.judge, heard_level, heard_f0)
    if b[1] is None:  # wherever the old code answered, the new code answers the same
        assert a[0].background == b[0].background and a[0].reason == b[0].reason
    else:  # ...and where it could only crash (a level of None), it now says why
        assert a[1] is None and a[0].reason == "no_profile_yet"


# ---------------------------------------------------------------------------------------------- barge_in


@FAST
@given(
    st.lists(
        st.tuples(st.floats(min_value=1e-4, max_value=0.6), st.floats(min_value=1e-4, max_value=0.6)),
        min_size=2,
        max_size=25,
    )
)
def test_the_barge_in_noise_floor_and_evidence_are_unchanged(levels):
    rng = np.random.default_rng(3)
    new, old = barge_in.BargeInDetector(), old_barge_in.BargeInDetector()
    for cleaned_amp, echo_amp in levels:
        cleaned = (rng.standard_normal(640) * cleaned_amp).astype(np.float32)
        echo = (rng.standard_normal(640) * echo_amp).astype(np.float32)
        assert new._frame_hit(cleaned, echo) == old._frame_hit(cleaned, echo)
        assert new._floor_db == old._floor_db


# ------------------------------------------------------------------------------------------ speaker_change


@FAST
@given(
    st.floats(min_value=6.0, max_value=9.0),
    st.floats(min_value=6.0, max_value=9.0),
    st.lists(st.floats(min_value=-60.0, max_value=-10.0), max_size=4),
    st.one_of(st.none(), st.floats(min_value=-60.0, max_value=-10.0)),
)
def test_pitch_corroboration_is_unchanged_when_there_is_a_reference_voice(ref_f0, new_f0, levels, level):
    new, old = speaker_change.SpeakerChangeDetector(), old_speaker_change.SpeakerChangeDetector()
    for det in (new, old):
        det._ref = types.SimpleNamespace(log2_f0=ref_f0)
        det._levels = list(levels)
    emb = types.SimpleNamespace(log2_f0=new_f0)
    assert new._corroborated(emb, level) == old._corroborated(emb, level)


def test_with_no_reference_voice_nothing_is_corroborated_instead_of_a_crash():
    emb = types.SimpleNamespace(log2_f0=7.0)
    assert speaker_change.SpeakerChangeDetector()._corroborated(emb, -30.0) is False
    with pytest.raises(AttributeError):  # what the old code did in that (unreachable) state
        old_speaker_change.SpeakerChangeDetector()._corroborated(emb, -30.0)


# ------------------------------------------------------------------------------------------ endpoint_calibration


@FAST
@given(
    st.lists(st.tuples(st.floats(min_value=0.0, max_value=4.0), st.floats(min_value=0.0, max_value=4.0)), max_size=4),
    st.floats(min_value=0.5, max_value=4.5),
)
def test_a_commit_time_and_an_utterance_end_are_none_together_and_the_evaluation_is_unchanged(raw_spans, turn_end):
    spans = [{"start": min(a, b), "end": max(a, b)} for a, b in raw_spans]
    sr, samples = 16000, np.zeros(16000 * 5, dtype=np.float32)
    for mod in (endpoint_calibration, old_endpoint_calibration):
        rec = mod.Recording(samples, sr, turn_end, spans=spans)
        t, end = mod.commit_time(rec, EndpointConfig())
        assert (t is None) == (end is None)
    new_rec = endpoint_calibration.Recording(samples, sr, turn_end, spans=spans)
    old_rec = old_endpoint_calibration.Recording(samples, sr, turn_end, spans=spans)
    a, b = endpoint_calibration.evaluate([new_rec], 0.8), old_endpoint_calibration.evaluate([old_rec], 0.8)
    assert (a.n, a.false_cut_rate, a.false_wait_rate) == (b.n, b.false_cut_rate, b.false_wait_rate)


# ------------------------------------------------------------------------------------------------------ lid


def test_language_id_names_the_problem_when_its_model_is_not_loaded():
    ident = lid.SpeechBrainVoxLingua107LID.__new__(lid.SpeechBrainVoxLingua107LID)
    ident._model = None
    with pytest.raises(RuntimeError, match="not loaded"):
        ident._require_model()
    ident._model = object()
    assert ident._require_model() is ident._model


@FAST
@given(st.dictionaries(st.sampled_from(["bn", "hi", "en"]), st.floats(min_value=0.0, max_value=1.0), min_size=1))
def test_the_language_id_winner_is_the_same_as_with_dict_get(scores):
    assert max(scores, key=lambda k: scores[k]) == max(scores, key=scores.get)
