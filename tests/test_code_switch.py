"""KCD-065: agent/code_switch.py. Pure, offline text analysis.

python -m pytest tests/test_code_switch.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.code_switch import has_code_switch, mixture_bucket, script_spans, switch_points


def test_single_script_bengali_has_no_switch():
    assert has_code_switch("লিপিড প্রোফাইলের রেট কত?") is False
    assert mixture_bucket("লিপিড প্রোফাইলের রেট কত?") == "bn"


def test_english_loanword_inside_bengali_is_a_switch():
    text = "CBC টেস্টের রেট কত?"
    assert has_code_switch(text) is True
    spans = script_spans(text)
    assert spans[0].script == "en"
    assert all(s.script == "bn" for s in spans[1:])  # every Bengali word is its own span


def test_switch_points_are_real_character_offsets():
    text = "CBC টেস্টের রেট কত?"
    points = switch_points(text)
    assert len(points) == 1
    assert text[points[0]] not in ("C", "B")  # points at the FIRST Bengali char


def test_three_way_mixture_bucket():
    text = "CBC টেস্ট के बारे में बताइए"
    bucket = mixture_bucket(text)
    assert bucket == "bn+en+hi"


def test_pure_english_utterance():
    assert mixture_bucket("what is the price of CBC") == "en"
    assert has_code_switch("what is the price of CBC") is False


def test_empty_text_has_no_switch_and_none_bucket():
    assert has_code_switch("") is False
    assert mixture_bucket("") == "none"


def test_switching_back_and_forth_counts_every_switch():
    text = "আমার CBC আর লিপিড দুটোই"  # bn -> en -> bn
    points = switch_points(text)
    assert len(points) == 2
