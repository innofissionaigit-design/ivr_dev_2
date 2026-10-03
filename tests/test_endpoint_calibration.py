"""KCD-047: agent/endpoint_calibration.py -- the instrument that replaces the
REASONED silence threshold with a MEASURED one.

The corpus here is SYNTHETIC, with the true turn end known by construction, so
what is tested is the arithmetic (false-cut / false-wait rates, the interval, the
choice of threshold, the refusal to call thin data MEASURED). It does not, and
cannot, say what the right threshold is for Kolkata callers: that needs the two
hundred real, human-annotated recordings the story asks for.

    python -m pytest tests/test_endpoint_calibration.py -v
"""

import json
import os
import sys
import wave

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _synth_speech import SR, speech

from agent.endpoint_calibration import (
    MIN_RECORDINGS,
    Recording,
    calibrate,
    evaluate,
    wilson_interval,
    write_report,
)
from agent.endpointing import EndpointConfig, config_from_json


def _recording(seed, max_pause=0.9):
    """1-3 phrases separated by pauses (the within-turn pauses that a too-eager
    threshold cuts at), then 2.5 s of quiet. turn_end_s is the end of the last phrase."""
    rng = np.random.default_rng(seed)
    parts, t = [np.zeros(int(0.3 * SR), np.float32)], 0.3
    for k in range(int(rng.integers(1, 4))):
        dur = float(rng.uniform(0.8, 1.6))
        parts.append(
            speech(f0=float(rng.uniform(100, 200)), dur=dur, amp=0.3, syll_hz=3.0, breath=0.02, seed=int(seed * 10 + k))
        )
        t += dur
        end_of_last = t
        pause = float(rng.uniform(0.15, max_pause))
        parts.append(np.zeros(int(pause * SR), np.float32))
        t += pause
    parts.append(np.zeros(int(2.5 * SR), np.float32))
    return Recording(np.concatenate(parts), SR, end_of_last)


@pytest.fixture(scope="module")
def corpus():
    return [_recording(s) for s in range(120)]


def test_a_generous_threshold_never_cuts_and_a_tiny_one_cuts_often(corpus):
    generous = evaluate(corpus, 1.2)
    tiny = evaluate(corpus, 0.3)
    assert generous.false_cut_rate == 0.0
    assert tiny.false_cut_rate > 0.25


def test_false_cuts_fall_and_waits_grow_as_the_threshold_rises(corpus):
    rows = [evaluate(corpus, c) for c in (0.3, 0.6, 0.9, 1.2)]
    cuts = [r.false_cut_rate for r in rows]
    waits = [r.wait_p50_s for r in rows]
    assert cuts == sorted(cuts, reverse=True)
    assert waits == sorted(waits)
    # where nothing is being cut (0.9 s and 1.2 s), the threshold IS the extra wait
    assert cuts[-1] == 0
    assert waits[-1] - waits[-2] == pytest.approx(0.3, abs=0.06)


def test_the_chosen_threshold_is_the_shortest_that_clears_the_target(corpus):
    report = calibrate(corpus, target_false_cut=0.05, min_recordings=100)
    chosen = report["chosen_silence_confirm_s"]
    assert report["status"] == "MEASURED" and report["config"] == {"silence_confirm_s": chosen}
    by = {round(r["silence_confirm_s"], 2): r for r in report["table"]}
    assert by[round(chosen, 2)]["false_cut_rate"] <= 0.05
    shorter = [r for r in report["table"] if r["silence_confirm_s"] < chosen - 1e-9]
    assert all(r["false_cut_ci95"][1] > 0.05 for r in shorter)  # every shorter one fails at 95% confidence


def test_thin_data_is_never_called_measured():
    few = [_recording(s) for s in range(30)]
    report = calibrate(few)
    assert report["status"] == "INSUFFICIENT_DATA"
    assert report["config"] == {}  # nothing to load: the REASONED value stays
    assert "Nothing was changed" in report["note"]
    assert report["required_recordings"] == MIN_RECORDINGS == 200


def test_the_wilson_interval_is_honest_at_small_counts():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and 0.02 < hi < 0.05  # zero cuts in 100 does NOT prove a zero rate
    lo, hi = wilson_interval(5, 100)
    assert lo < 0.05 < hi
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_a_calibration_report_round_trips_into_the_detector_config(corpus, tmp_path):
    report = calibrate(corpus, target_false_cut=0.05, min_recordings=100)
    f = tmp_path / "cal.json"
    write_report(report, str(f))
    cfg = config_from_json(str(f))
    assert cfg.silence_confirm_s == report["chosen_silence_confirm_s"]
    assert cfg.tail_guard_s == EndpointConfig().tail_guard_s  # untouched fields keep their values


def test_the_command_line_loader_reads_wav_and_json_pairs(tmp_path):
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
    import calibrate_endpointing as cli

    for i in range(3):
        rec = _recording(i)
        with wave.open(str(tmp_path / f"c{i}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((rec.samples * 32767).astype(np.int16).tobytes())
        (tmp_path / f"c{i}.json").write_text(json.dumps({"turn_end_ms": rec.turn_end_s * 1000, "language": "bn"}))
    with wave.open(str(tmp_path / "unannotated.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"\x00\x00" * 100)
    recs = cli.load_recordings(str(tmp_path), "energy")
    assert len(recs) == 3 and recs[0].language == "bn"  # the unannotated file was skipped
    assert recs[0].turn_end_s == pytest.approx(_recording(0).turn_end_s, abs=1e-3)
