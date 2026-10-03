"""KCD-162: agent/prosody.py -- clause-level prosody with punctuation-aware
pauses. This is the logic that used to sit inside tts_server.py, which
loads three GPU synthesizers at import and so could never be tested off a
pod. Pure numpy, no GPU, no models.

    python -m pytest tests/test_prosody.py -v
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.prosody import (
    BOUNDS,
    ProsodyParams,
    assemble,
    normalize_peak,
    pause_kind_for,
    resolve_params,
    split_for_prosody,
    trim_silence,
)

SR = 1000  # 1 sample = 1 ms, so pause lengths are exact integers in the assertions


# ------------------------------------------------------------- splitting


def test_english_sentences_split_on_full_stops():
    # The earlier splitter had no "." at all, so an English reply of
    # several sentences was rendered as one flat pass with no pause.
    chunks = split_for_prosody("Your price is 650 rupees. The report takes one day. Thank you.")
    assert [c for c, _ in chunks] == ["Your price is 650 rupees.", "The report takes one day.", "Thank you."]
    assert all(kind == "sentence" for _, kind in chunks)


def test_bengali_danda_and_question_mark_split_sentences():
    chunks = split_for_prosody("আপনার রেট পাঁচশো টাকা। রিপোর্ট কালকে পাবেন। আর কিছু জানতে চান?")
    assert len(chunks) == 3
    assert all(kind == "sentence" for _, kind in chunks)


def test_an_abbreviation_full_stop_is_not_a_sentence_end():
    chunks = split_for_prosody("Dr. Sen sits on Monday. Please come early.")
    assert [c for c, _ in chunks] == ["Dr. Sen sits on Monday.", "Please come early."]


@pytest.mark.parametrize("text", ["The rate is 2.5 percent.", "Room a.b is free."])
def test_a_full_stop_inside_a_token_is_not_a_sentence_end(text):
    assert len(split_for_prosody(text)) == 1


def test_a_long_sentence_is_broken_at_clause_boundaries_with_shorter_pauses():
    long_sentence = (
        "Please bring your previous reports, your prescription and your identity card, "
        "and arrive fifteen minutes early, because the counter closes quickly."
    )
    assert len(long_sentence) > 90
    chunks = split_for_prosody(long_sentence)
    assert len(chunks) > 1
    kinds = [k for _, k in chunks]
    assert kinds[-1] == "sentence"  # the real sentence end
    assert set(kinds[:-1]) == {"clause"}  # commas earn the shorter pause


def test_max_chunk_chars_is_honoured():
    sentence = "one, two, three, four, five, six, seven."
    assert len(split_for_prosody(sentence, max_chunk_chars=90)) == 1
    assert len(split_for_prosody(sentence, max_chunk_chars=20)) > 1


def test_pause_kind_comes_from_the_punctuation_that_actually_ended_the_chunk():
    assert pause_kind_for("Hello there.") == "sentence"
    assert pause_kind_for("আপনার নাম কী?") == "sentence"
    assert pause_kind_for("first part,") == "clause"
    # A chunk cut with no terminator used to be labelled "sentence" and
    # given a full sentence's pause; it now earns the shortest one.
    assert pause_kind_for("no punctuation at all") == "none"


def test_text_with_no_punctuation_is_one_chunk_with_the_minimal_pause():
    assert split_for_prosody("just some words") == [("just some words", "none")]


def test_empty_text_produces_no_chunks():
    assert split_for_prosody("   ") == []


# --------------------------------------------------------------- params


def test_defaults_match_the_values_that_were_previously_hardcoded():
    p = ProsodyParams()
    assert (p.pause_sentence_s, p.pause_clause_s, p.pause_none_s) == (0.28, 0.14, 0.06)
    assert p.trim_threshold == 0.012 and p.target_peak == 0.89 and p.max_chunk_chars == 90


def test_every_parameter_can_be_overridden_per_request():
    overrides = {
        "pause_sentence_s": 0.5,
        "pause_clause_s": 0.2,
        "pause_none_s": 0.1,
        "trim_threshold": 0.02,
        "target_peak": 0.7,
        "max_chunk_chars": 60,
        "lead_in_s": 0.1,
    }
    p = resolve_params(ProsodyParams(), overrides)
    for key, value in overrides.items():
        assert getattr(p, key) == value
    assert set(overrides) == set(BOUNDS), "a parameter added without a bound would be unvalidated"


def test_none_overrides_leave_the_default_and_the_base_is_never_mutated():
    base = ProsodyParams()
    p = resolve_params(base, {"pause_sentence_s": None, "target_peak": 0.5})
    assert p.pause_sentence_s == base.pause_sentence_s and p.target_peak == 0.5
    assert base.target_peak == 0.89


@pytest.mark.parametrize(
    "key,bad",
    [
        ("pause_sentence_s", -0.1),
        ("pause_sentence_s", 5.0),
        ("target_peak", 1.5),
        ("target_peak", 0.0),
        ("max_chunk_chars", 5),
        ("trim_threshold", 0.9),
    ],
)
def test_out_of_range_values_are_rejected_not_silently_clamped(key, bad):
    with pytest.raises(ValueError, match=key):
        resolve_params(ProsodyParams(), {key: bad})


def test_an_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        resolve_params(ProsodyParams(), {"pause_lenght_s": 0.3})


# ---------------------------------------------------------------- audio


def _tone(n: int, amp: float) -> np.ndarray:
    return np.full(n, amp, dtype=np.float32)


def test_ragged_padding_is_trimmed_off_each_chunk():
    padded = np.concatenate([np.zeros(300), _tone(100, 0.5), np.zeros(200)]).astype(np.float32)
    assert trim_silence(padded, 0.012).size == 100


def test_an_all_silent_chunk_trims_to_nothing():
    assert trim_silence(np.zeros(500, dtype=np.float32), 0.012).size == 0


def test_pauses_are_chosen_by_the_punctuation_that_ended_each_chunk():
    params = ProsodyParams(lead_in_s=0.0)
    out = assemble([(_tone(100, 0.5), "sentence"), (_tone(100, 0.5), "clause"), (_tone(100, 0.5), "none")], SR, params)
    # 3 x 100 samples of speech + 280 + 140 + 60 samples of pause
    assert out.size == 300 + 280 + 140 + 60


def test_pause_lengths_follow_per_request_overrides():
    params = resolve_params(ProsodyParams(lead_in_s=0.0), {"pause_sentence_s": 0.5})
    out = assemble([(_tone(100, 0.5), "sentence")], SR, params)
    assert out.size == 100 + 500


def test_the_lead_in_is_prepended():
    out = assemble([(_tone(100, 0.5), "none")], SR, ProsodyParams(lead_in_s=0.04))
    assert out.size == 40 + 100 + 60
    assert np.all(out[:40] == 0)


def test_the_whole_reply_is_normalised_to_one_peak_so_the_speaker_does_not_appear_to_move():
    quiet, loud = _tone(100, 0.1), _tone(100, 0.6)
    out = assemble([(quiet, "sentence"), (loud, "sentence")], SR, ProsodyParams())
    assert float(np.max(np.abs(out))) == pytest.approx(0.89, abs=1e-4)
    # relative dynamics are preserved: the quiet chunk stays 6x quieter
    assert float(np.max(np.abs(out[40:140]))) == pytest.approx(0.89 / 6, rel=1e-3)


def test_target_peak_is_overridable_per_request():
    params = resolve_params(ProsodyParams(), {"target_peak": 0.5})
    out = assemble([(_tone(100, 0.3), "sentence")], SR, params)
    assert float(np.max(np.abs(out))) == pytest.approx(0.5, abs=1e-4)


def test_silent_chunks_are_skipped_and_an_all_silent_reply_is_a_short_silence():
    assert assemble([(np.zeros(400, dtype=np.float32), "sentence")], SR, ProsodyParams()).size == 200
    out = assemble([(np.zeros(400), "sentence"), (_tone(100, 0.5), "none")], SR, ProsodyParams(lead_in_s=0.0))
    assert out.size == 100 + 60


def test_pauses_can_be_disabled_entirely():
    out = assemble(
        [(_tone(100, 0.5), "sentence"), (_tone(100, 0.5), "sentence")], SR, ProsodyParams(lead_in_s=0.0), pauses=False
    )
    assert out.size == 200


def test_normalize_peak_leaves_silence_alone():
    assert float(np.max(np.abs(normalize_peak(np.zeros(10, dtype=np.float32), 0.89)))) == 0.0
