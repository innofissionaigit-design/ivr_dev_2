"""KCD-462: agent/clause_split.py. Pure, offline text splitting.

python -m pytest tests/test_clause_split.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.clause_split import MIN_CHARS_TO_SPLIT, split_into_clauses


def test_short_text_is_not_split():
    text = "নমস্কার।"
    assert split_into_clauses(text) == [text]


def test_empty_text_returns_empty_list():
    assert split_into_clauses("") == []
    assert split_into_clauses("   ") == []


def test_a_long_bengali_multi_sentence_reply_splits_on_danda():
    text = "এই টেস্টের জন্য খালি পেটে আসতে হবে। " * 3 + "প্রেসক্রিপশন সাথে আনবেন।"
    assert len(text) >= MIN_CHARS_TO_SPLIT
    clauses = split_into_clauses(text)
    assert len(clauses) == 4
    for c in clauses:
        assert c.endswith("।")  # punctuation stays attached to its own clause


def test_a_long_english_reply_splits_on_sentence_punctuation():
    text = "Please come on an empty stomach for this test. " * 2 + "Bring your prescription with you. Water is allowed."
    clauses = split_into_clauses(text)
    assert len(clauses) == 4
    assert clauses[0].endswith(".")
    assert clauses[-1] == "Water is allowed."


def test_rejoining_clauses_reproduces_the_original_text():
    text = "এই টেস্টের জন্য খালি পেটে আসতে হবে। প্রেসক্রিপশন সাথে আনবেন। জল খাওয়া যাবে।"
    clauses = split_into_clauses(text)
    assert " ".join(clauses) == text


def test_long_text_with_no_sentence_punctuation_is_not_split():
    text = "কোন বিরতিচিহ্ন ছাড়া একটানা অনেক লম্বা একটা বাক্য এখানে লেখা হয়েছে পরীক্ষার জন্য"
    assert len(text) >= MIN_CHARS_TO_SPLIT
    assert split_into_clauses(text) == [text]
