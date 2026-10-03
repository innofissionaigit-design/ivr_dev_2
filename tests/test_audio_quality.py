"""agent/audio_quality.py: audibility, cross-talk, noise filtering and
jumbled-transcript detection, validated against SYNTHETIC signals whose
ground truth is known. This is the only validation possible off-pod --
there is no real call audio locally -- so what these tests prove is that
the algorithm separates the cases it was designed to separate, not that
its thresholds are right for Kolkata handsets. See the module docstring.

    python -m pytest tests/test_audio_quality.py -v
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.audio_quality import CROSSTALK_RATIO, assess, enhance, transcript_problem

SR = 16000


def voice(f0=120.0, dur=3.0, amp=0.3, seed=0, syll_hz=3.5, drift=0.02):
    """A crude but honest voiced-speech stand-in: harmonics of a slowly
    drifting f0, shaped by a syllable-rate envelope that dips to silence
    between syllables (so there are real pauses to measure a noise floor in)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    f0_t = f0 * (1.0 + drift * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28)))
    phase = 2 * np.pi * np.cumsum(f0_t) / SR
    sig = np.zeros_like(t)
    for k in range(1, 26):
        if k * f0 > 3400:
            break
        sig += np.sin(k * phase + rng.uniform(0, 6.28)) / k
    env = np.clip(np.sin(2 * np.pi * syll_hz * t - np.pi / 2) * 1.4, 0.0, 1.0) ** 0.8
    sig = sig * env
    return (amp * sig / np.max(np.abs(sig))).astype(np.float32)


def noise(dur=3.0, amp=0.05, seed=1):
    return (amp * np.random.default_rng(seed).standard_normal(int(dur * SR))).astype(np.float32)


def whisper(dur=3.0, amp=0.15, seed=2, syll_hz=3.5):
    t = np.arange(int(dur * SR)) / SR
    env = np.clip(np.sin(2 * np.pi * syll_hz * t - np.pi / 2) * 1.4, 0.0, 1.0) ** 0.8
    return (amp * np.random.default_rng(seed).standard_normal(t.size) * env).astype(np.float32)


def add_noise_at_snr(clean, snr_db, seed=3):
    p_sig = float(np.mean(clean[np.abs(clean) > 0.05 * np.max(np.abs(clean))] ** 2))
    n = np.random.default_rng(seed).standard_normal(clean.size).astype(np.float32)
    n *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / np.mean(n**2))
    return clean + n


def corr(a, b):
    n = min(a.size, b.size)
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------- assess


def test_clean_speech_has_no_issues():
    a = assess(voice(), SR)
    assert a.ok, a.to_log_dict()
    assert a.voiced_ratio > 0.5 and a.two_pitch_ratio < 0.1
    assert a.intelligibility > 0.6


def test_a_faint_voice_is_flagged_too_quiet_not_noisy():
    a = assess(voice(amp=0.004), SR)
    assert "too_quiet" in a.issues
    assert "noisy" not in a.issues and "crosstalk" not in a.issues


def test_a_normal_voice_is_not_flagged_too_quiet():
    assert "too_quiet" not in assess(voice(amp=0.2), SR).issues


@pytest.mark.parametrize("snr_db", [0, 3, 6])
def test_background_noise_is_flagged(snr_db):
    a = assess(add_noise_at_snr(voice(), snr_db), SR)
    assert "noisy" in a.issues, a.to_log_dict()
    assert a.snr_db < 10.0


def test_mild_noise_is_tolerated():
    a = assess(add_noise_at_snr(voice(), 30), SR)
    assert "noisy" not in a.issues


def test_two_people_talking_at_once_is_flagged_as_crosstalk():
    mixed = voice(120.0, seed=10) + 0.8 * voice(205.0, seed=11, syll_hz=4.1)
    a = assess(mixed, SR)
    assert "crosstalk" in a.issues, a.to_log_dict()
    assert a.two_pitch_ratio >= CROSSTALK_RATIO


def test_a_weaker_background_talker_is_still_detected():
    mixed = voice(118.0, seed=20, amp=0.4) + voice(190.0, seed=21, amp=0.2, syll_hz=4.6)
    assert "crosstalk" in assess(mixed, SR).issues


def test_one_voice_is_never_flagged_as_crosstalk_even_with_pitch_drift():
    for f0 in (95.0, 120.0, 180.0, 240.0):
        assert "crosstalk" not in assess(voice(f0, drift=0.04, seed=int(f0)), SR).issues, f0


def test_noise_is_not_mistaken_for_a_second_talker():
    a = assess(add_noise_at_snr(voice(), 8), SR)
    assert "crosstalk" not in a.issues, a.to_log_dict()


def test_whispering_or_mumbling_without_voicing_is_flagged():
    a = assess(whisper(), SR)
    assert "unvoiced_mumble" in a.issues, a.to_log_dict()
    assert a.voiced_ratio < 0.2


def test_clipping_is_flagged():
    a = assess(np.clip(voice(amp=1.0) * 6.0, -1.0, 1.0), SR)
    assert "clipped" in a.issues


def test_a_fragment_too_short_to_judge_is_reported_as_such():
    assert assess(voice(dur=0.3), SR).issues == ["too_short"]


def test_silence_is_no_speech():
    assert assess(np.zeros(SR * 2, dtype=np.float32), SR).issues == ["no_speech"]


def test_intelligibility_orders_clean_above_noisy_above_faint_noisy():
    clean = assess(voice(), SR).intelligibility
    noisy = assess(add_noise_at_snr(voice(), 3), SR).intelligibility
    worst = assess(add_noise_at_snr(voice(amp=0.01), 0), SR).intelligibility
    assert clean > noisy > worst


# --------------------------------------------------------------- enhance


def test_enhance_makes_noisy_speech_closer_to_the_clean_signal():
    clean = voice()
    noisy = add_noise_at_snr(clean, 3)
    better = enhance(noisy, SR)
    assert corr(better, clean) > corr(noisy, clean) + 0.05


def test_enhance_lifts_the_measured_snr():
    noisy = add_noise_at_snr(voice(), 3)
    assert assess(enhance(noisy, SR), SR).snr_db > assess(noisy, SR).snr_db + 4.0


def test_enhance_leaves_clean_speech_essentially_intact():
    clean = voice(amp=0.3)
    assert corr(enhance(clean, SR), clean) > 0.97


def test_enhance_brings_a_faint_voice_up_to_a_usable_level_without_clipping():
    faint = voice(amp=0.01)
    out = enhance(faint, SR)
    assert "too_quiet" in assess(faint, SR).issues
    assert "too_quiet" not in assess(out, SR).issues
    assert float(np.max(np.abs(out))) <= 0.98


def test_enhance_removes_low_frequency_rumble():
    t = np.arange(SR * 3) / SR
    rumble = (0.2 * np.sin(2 * np.pi * 30 * t)).astype(np.float32)
    out = enhance(voice() + rumble, SR)
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(out.size, 1 / SR)
    assert spec[(freqs > 20) & (freqs < 45)].max() < 0.05 * spec.max()


def test_enhance_of_too_short_audio_is_a_safe_copy():
    x = voice(dur=0.1)
    assert np.array_equal(enhance(x, SR), x)


# ---------------------------------------------------- transcript-side


@pytest.mark.parametrize(
    "text,lang,dur,expected",
    [
        ("", "bn", 3.0, "empty"),
        ("   ", "en", 1.0, "empty"),
        ("ও", "bn", 4.0, "fragment"),
        ("hello there how are you", "bn", 2.0, "wrong_script"),
        ("a b c d e f g", "en", 3.0, "shattered"),
        ("the the the the the the", "en", 3.0, "repetitive"),
        ("I want to book an appointment", "en", 3.0, None),
        ("আমি কালকে ডাক্তার সেনের অ্যাপয়েন্টমেন্ট চাই", "bn", 3.0, None),
        ("হ্যাঁ", "bn", 1.0, None),
        ("ok", "en", 1.0, None),
    ],
)
def test_transcript_problem(text, lang, dur, expected):
    assert transcript_problem(text, lang, dur) == expected


def test_decoders_disagreeing_badly_is_a_transcript_problem():
    assert transcript_problem("some words here today", "en", 3.0, decoder_agreement=0.1) == "decoders_disagree"
    assert transcript_problem("some words here today", "en", 3.0, decoder_agreement=0.0) is None
    assert transcript_problem("some words here today", "en", 3.0, decoder_agreement=0.8) is None
