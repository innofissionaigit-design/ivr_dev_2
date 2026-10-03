"""KCD-051: agent/echo_cancel.py, on SYNTHETIC echo scenes with known ground truth.

What these prove, and what they do not
--------------------------------------
Every number here is MEASURED on synthetic scenes (tests/_synth_echo.py): a
decaying random echo path after a bulk delay, synthetic speech as the far end.
They show the algorithm converges, tracks the delay, leaves a caller intact and
is not wrecked by one talking over it. They do NOT show what it does on a real
handset: ERLE on real speakers and rooms is what tools/echo_eval.py is for,
on a pod, with recorded audio. Known weak spot, measured and deliberately not
hidden: when the far-end VOICE changes (the synthetic scenes concatenate
different pitches) the filter has never seen the newly excited frequencies and
cancellation dips to 0-7 dB for two to three seconds. The agent's own TTS is one
voice per language, so this is milder in practice, but it is a limit.

    python -m pytest tests/test_echo_cancel.py -v
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _synth_echo import SR, caller, echo_scene, run_pipeline

from agent.echo_cancel import EchoPipeline, erle_db

CALLER_AT_S = 9.0


def _corr(x, y):
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y) + 1e-12))


@pytest.fixture(scope="module")
def scenes():
    """Four echo paths (delays 120-270 ms, two with a distorting speaker), each
    run once with a caller talking over the agent at 9 s. The window BEFORE the
    caller serves the single-talk figures, so one run answers both questions."""
    out = []
    for seed in range(4):
        far, mic, echo = echo_scene(seed, drive=2.0 if seed % 2 else 0.0)
        c = caller(seed, dur=1.5)
        a = int(CALLER_AT_S * SR)
        with_caller = mic.copy()
        with_caller[a : a + c.size] += c
        pipe, cleaned, _ = run_pipeline(with_caller, far)
        out.append(dict(seed=seed, far=far, mic=with_caller, echo=echo, caller=c, at=a, pipe=pipe, cleaned=cleaned))
    return out


def test_cancellation_reaches_at_least_15_db_in_the_median_scene(scenes):
    erles = [erle_db(s["mic"][7 * SR : 9 * SR], s["cleaned"][7 * SR : 9 * SR]) for s in scenes]
    assert np.median(erles) >= 15.0, erles  # measured: 21-27 dB


def test_it_recovers_after_a_change_of_far_end_voice(scenes):
    # the synthetic scenes change voice at 4 s and 8 s; by the end of the caller
    # segment's aftermath the filter has re-converged in every scene
    for s in scenes:
        after = slice(s["at"] + s["caller"].size + SR // 2, s["at"] + s["caller"].size + 2 * SR)
        assert erle_db(s["mic"][after], s["cleaned"][after]) >= 6.0, s["seed"]


def test_a_caller_talking_over_the_agent_is_kept_and_the_filter_is_not_damaged(scenes):
    for s in scenes:
        a, n = s["at"], s["caller"].size
        seg = slice(a, a + n)
        raw = _corr(s["mic"][seg], s["caller"])
        cleaned = _corr(s["cleaned"][seg], s["caller"][: s["cleaned"][seg].size])
        assert cleaned >= raw - 0.01, (s["seed"], raw, cleaned)  # never worse than the raw microphone
        before = erle_db(s["mic"][a - SR : a], s["cleaned"][a - SR : a])
        after = erle_db(
            s["mic"][a + n + SR // 2 : a + n + 3 * SR // 2], s["cleaned"][a + n + SR // 2 : a + n + 3 * SR // 2]
        )
        assert after >= before - 3.0, (s["seed"], before, after)  # the caller did not wreck the filter


def test_a_converged_filter_leaves_the_caller_almost_untouched(scenes):
    # the two scenes that had converged by 9 s (ERLE > 20 dB): the caller comes
    # through with correlation 0.999 against 0.91-0.93 on the raw microphone
    converged = [
        s for s in scenes if erle_db(s["mic"][s["at"] - SR : s["at"]], s["cleaned"][s["at"] - SR : s["at"]]) > 20.0
    ]
    assert converged
    for s in converged:
        seg = slice(s["at"], s["at"] + s["caller"].size)
        assert _corr(s["cleaned"][seg], s["caller"][: s["cleaned"][seg].size]) >= 0.98


@pytest.mark.parametrize("true_delay_ms", [120, 250, 400])
def test_the_bulk_delay_is_measured_not_assumed(true_delay_ms):
    far, mic, _ = echo_scene(1, delay_ms=true_delay_ms)
    pipe, _, _ = run_pipeline(mic, far)
    # the filter is started `lead_s` (32 ms) BEFORE the direct path
    assert abs(pipe.stats().delay_ms + 32.0 - true_delay_ms) <= 25.0
    assert pipe.stats().delay_confidence >= 6.0


def test_a_short_delay_needs_no_bulk_alignment():
    far, mic, _ = echo_scene(1, delay_ms=40)
    pipe, cleaned, _ = run_pipeline(mic, far)
    assert pipe.stats().delay_ms <= 10.0  # inside the filter's own reach
    assert erle_db(mic[-4 * SR :], cleaned[-4 * SR :]) >= 10.0


def test_a_distorting_speaker_still_cancels_substantially():
    ratios = []
    for seed in range(3):
        far, mic, _ = echo_scene(seed, drive=3.0)
        _, cleaned, _ = run_pipeline(mic, far)
        ratios.append(erle_db(mic[-3 * SR :], cleaned[-3 * SR :]))
    assert np.median(ratios) >= 10.0, ratios  # measured 17-24 dB


def test_with_no_far_end_audio_the_microphone_passes_through():
    rng = np.random.default_rng(0)
    mic = (0.1 * rng.standard_normal(3 * SR)).astype(np.float32)
    pipe = EchoPipeline()
    cleaned = pipe.process(mic)
    assert cleaned.size == mic.size - (mic.size % pipe.block)
    assert np.allclose(cleaned, mic[: cleaned.size], atol=1e-6)


def test_the_output_does_not_depend_on_how_the_stream_is_chunked():
    far, mic, _ = echo_scene(0, dur_each=2.0, voices=2)
    a = run_pipeline(mic, far, chunk=1600)[1]
    b = run_pipeline(mic, far, chunk=333)[1]
    n = min(a.size, b.size)
    assert n > 3 * SR
    assert np.allclose(a[:n], b[:n], atol=1e-4)


def test_the_state_it_reports_is_honest_about_what_it_has_measured():
    far, mic, _ = echo_scene(0)
    pipe, cleaned, _ = run_pipeline(mic, far)
    st = pipe.stats()
    assert st.blocks == cleaned.size // pipe.block
    assert 0 <= st.double_talk_blocks <= st.blocks
    assert st.erle_db > 8.0  # it knows it has converged
