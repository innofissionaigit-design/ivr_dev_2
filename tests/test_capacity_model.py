"""KCD-019: the sizing model's arithmetic, and its refusal to invent numbers.

python -m pytest tests/test_capacity_model.py -v
"""

import json
import math
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import capacity_model as cm


def test_erlang_b_matches_the_textbook_values():
    assert cm.erlang_b(1.0, 1) == pytest.approx(0.5)
    assert cm.erlang_b(10.0, 10) == pytest.approx(0.2146, abs=1e-4)
    assert cm.erlang_b(5.0, 10) == pytest.approx(0.0184, abs=1e-4)
    assert cm.erlang_b(5.0, 0) == 1.0


def test_the_cap_is_the_smallest_that_meets_the_blocking_target():
    for offered in (8.0, 20.0, 35.0):
        cap = cm.cap_for_blocking(offered, 0.02)
        assert cm.erlang_b(offered, cap) <= 0.02 < cm.erlang_b(offered, cap - 1)


def test_the_workload_follows_littles_law():
    out = cm.size(cm.default_inputs())
    w = out["workload"]
    # 7000 calls over 10 h = 700/h, busiest hour 1.5x = 1050/h, 2.5 min each -> 43.75 Erlangs
    assert w["offered_load_erlangs_at_peak"] == pytest.approx(43.8, abs=0.1)
    assert w["ai_cap_for_blocking_target"] > w["offered_load_erlangs_at_peak"]


def test_with_nothing_measured_no_gpu_count_is_stated():
    out = cm.size(cm.default_inputs())
    assert out["gpus_per_dc"] is None and out["calls_per_gpu"] is None
    assert "asr_rtf" in out["undetermined"] and "tts_x_realtime" in out["undetermined"]
    text = cm.report(out)
    assert "UNDETERMINED" in text and "GPU per data centre" not in text
    assert "INDICATIVE" in text  # Appendix H shown, labelled as such


def _measured(**kw):
    inp = cm.default_inputs()
    base = {"asr_rtf": 0.05, "tts_x_realtime": 20.0, "llm_tokens_per_s": 400.0}
    base.update(kw)
    return cm.apply_measured(inp, base)


def test_a_measured_sizing_is_arithmetic_on_the_inputs():
    out = cm.size(_measured())
    # per call-second: ASR 0.05*0.40 = 0.02 ; TTS 0.45/20 = 0.0225 ; both on the GPU = 0.0425
    assert out["per_call_demand"]["asr_gpu_s_per_call_s"] == pytest.approx(0.02)
    assert out["per_call_demand"]["tts_gpu_s_per_call_s"] == pytest.approx(0.0225)
    assert out["calls_per_gpu"] == pytest.approx(0.70 / 0.0425, abs=0.1)
    target = out["target_concurrency"]
    per_dc = math.ceil(target * 0.5 / (0.70 / 0.0425)) + 1
    assert out["gpus_per_dc"] == per_dc and out["gpus_total"] == 2 * per_dc
    assert out["headroom_calls"] >= 0  # sized to carry the target, plus spares


def test_a_slower_recogniser_needs_more_gpus_never_fewer():
    fast = cm.size(_measured(asr_rtf=0.03))["gpus_per_dc"]
    slow = cm.size(_measured(asr_rtf=0.30))["gpus_per_dc"]
    assert slow > fast


def test_the_llm_counts_against_the_gpu_only_when_it_shares_it():
    off = cm.size(_measured(llm_on_gpu=0))["calls_per_gpu"]
    on = cm.size(_measured(llm_on_gpu=1))["calls_per_gpu"]
    assert on < off


def test_an_llm_on_the_gpu_with_no_measured_tokens_per_s_is_undetermined():
    inp = cm.apply_measured(cm.default_inputs(), {"asr_rtf": 0.05, "tts_x_realtime": 20.0, "llm_on_gpu": 1})
    out = cm.size(inp)
    assert out["gpus_per_dc"] is None and "llm_tokens_per_s" in out["undetermined"]


def test_losing_a_data_centre_lowers_capacity_and_the_report_says_so():
    out = cm.size(_measured())
    assert out["capacity_after_dc_loss_calls"] < out["target_concurrency"] * 1.01
    assert "capacity falls to" in out["at_capacity"] and "person" in out["at_capacity"]
    assert "never queued" in out["at_capacity"]


def test_active_standby_sizes_the_full_load_in_each_site():
    aa = cm.size(_measured())["gpus_per_dc"]
    inp = _measured()
    inp["active_active"] = cm.Input(0, cm.STATED)
    assert cm.size(inp)["gpus_per_dc"] > aa


def test_an_unknown_input_name_is_refused_not_ignored():
    with pytest.raises(KeyError):
        cm.apply_measured(cm.default_inputs(), {"asr_rft": 0.1})


def test_the_template_file_is_valid_and_leaves_everything_unmeasured():
    with open(os.path.join(REPO_ROOT, "docs", "capacity-measurements.template.json"), encoding="utf-8") as f:
        data = json.load(f)
    out = cm.size(cm.apply_measured(cm.default_inputs(), data))
    assert out["gpus_per_dc"] is None


def test_every_default_input_declares_where_it_came_from():
    for name, i in cm.default_inputs().items():
        assert i.source in (cm.MEASURED, cm.STATED, cm.REASONED), name
        assert i.note, f"{name} has no note"
