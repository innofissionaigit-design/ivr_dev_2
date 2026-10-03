"""tools/conditioning_eval.py's own correctness (KCD-055/057): WER arithmetic, the degradations,
and the evaluation wiring, with a stand-in recogniser. The tool's REAL result -- does the pod's
ASR hear better after conditioning -- can only come from a pod; nothing here claims it.

    python -m pytest tests/test_conditioning_eval.py -v
"""

import asyncio
import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests"), os.path.join(REPO_ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import conditioning_eval as ce
from _synth_voices import FATHER, utterance


def test_wer_counts_substitutions_insertions_and_deletions():
    assert ce.wer("book a cbc test", "book a cbc test") == 0.0
    assert ce.wer("book a cbc test", "book a lipid test") == pytest.approx(0.25)  # 1 substitution / 4
    assert ce.wer("book a cbc test", "book a test") == pytest.approx(0.25)  # 1 deletion
    assert ce.wer("book a test", "please book a cbc test") == pytest.approx(2 / 3)  # 2 insertions / 3
    assert ce.wer("book a test", "") == 1.0
    assert ce.wer("", "") == 0.0


def test_wer_ignores_case_and_punctuation_and_handles_indic_text():
    assert ce.wer("Book a CBC test.", "book a cbc test") == 0.0
    assert ce.wer("আমার টেস্ট কবে হয়েছিল", "আমার টেস্ট কবে হয়েছিল।") == 0.0
    assert ce.wer("मेरा टेस्ट कब हुआ", "मेरा टेस्ट कब") == pytest.approx(0.25)


def test_degrade_sets_the_requested_peak_level_and_snr():
    x = utterance(FATHER, dur=2.0, seed=3, amp=0.3)
    rng = np.random.default_rng(0)
    y = ce.degrade(x, -30.0, None, rng)
    assert 20 * np.log10(np.max(np.abs(y))) == pytest.approx(-30.0, abs=0.1)
    noisy = ce.degrade(x, None, 10.0, np.random.default_rng(0))
    resid = noisy - x
    active = x[np.abs(x) > 0.05 * np.max(np.abs(x))]  # SNR is defined over the speech-active samples
    snr = 10 * np.log10(np.mean(active**2) / np.mean(resid**2))
    assert snr == pytest.approx(10.0, abs=0.5)


def _write_clips(tmp_path, n=12, lang="en"):
    for i in range(n):
        x = utterance(FATHER, dur=2.0, seed=i, amp=0.3)
        ce.write_wav(str(tmp_path / f"c{i}.wav"), x)
        (tmp_path / f"c{i}.txt").write_text("book a cbc test", encoding="utf-8")
        (tmp_path / f"c{i}.json").write_text(json.dumps({"lang": lang}), encoding="utf-8")


def test_clips_load_in_pairs_and_skip_a_wav_without_a_reference(tmp_path):
    _write_clips(tmp_path, 3)
    ce.write_wav(str(tmp_path / "orphan.wav"), np.zeros(1600, dtype=np.float32))
    clips = ce.load_clips(str(tmp_path))
    assert [c.name for c in clips] == ["c0", "c1", "c2"] and clips[0].reference == "book a cbc test"


def test_a_wrong_format_wav_is_refused(tmp_path):
    import wave

    p = str(tmp_path / "bad.wav")
    with wave.open(p, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)
    with pytest.raises(ValueError):
        ce.read_wav(p)


def _level_sensitive_asr():
    """A stand-in recogniser that only hears audio above a level floor -- so a conditioner that
    normalises level should HELP it on quiet conditions and change nothing on normal ones."""

    async def transcribe(lang, path):
        x = ce.read_wav(path)
        rms_db = 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9)
        return "book a cbc test" if rms_db > -38.0 else "book"

    return transcribe


def test_the_evaluation_reports_conditioning_helping_where_it_helps(tmp_path):
    _write_clips(tmp_path, 12)
    clips = ce.load_clips(str(tmp_path))
    out = asyncio.run(ce.evaluate(clips, _level_sensitive_asr()))
    quiet = out["table"][("en", "very_quiet_-42dBFS")]
    normal = out["table"][("en", "as_recorded")]
    assert quiet["wer_raw"] > quiet["wer_conditioned"] and ce.verdict(quiet["change"], quiet["n"]) == "HELPS"
    assert normal["change"] == pytest.approx(0.0, abs=0.01)
    assert len(out["rows"]) == len(clips) * len(ce.CONDITIONS)


def test_a_conditioner_that_makes_things_worse_is_reported_as_hurting(tmp_path):
    _write_clips(tmp_path, 12)
    clips = ce.load_clips(str(tmp_path))

    def ruin(x, sr):
        from agent.conditioning import ConditioningReport

        return np.zeros_like(x), ConditioningReport(True, "test", 0.0, None, None, None)

    out = asyncio.run(ce.evaluate(clips, _level_sensitive_asr(), conditioner=ruin))
    r = out["table"][("en", "as_recorded")]
    assert ce.verdict(r["change"], r["n"]) == "HURTS"


def test_too_few_clips_are_not_turned_into_a_verdict():
    assert ce.verdict(-0.5, 3) == "too few clips to say"
    assert ce.verdict(0.0, 50) == "no clear effect"


def test_the_table_renders_every_row(tmp_path):
    _write_clips(tmp_path, 10)
    out = asyncio.run(ce.evaluate(ce.load_clips(str(tmp_path)), _level_sensitive_asr()))
    text = ce.render(out)
    for cname, _, _ in ce.CONDITIONS:
        assert cname in text
