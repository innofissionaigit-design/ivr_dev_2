"""agent/phrase_proximity.py: a fixed multi-word phrase matched with a word or two allowed between its words.

    python -m pytest tests/test_phrase_proximity.py -v

Found on a live call: "বুকে ব্যথা" (chest pain, literal adjacent words) did not match "বুকে খুব ব্যথা করছে" (chest
REALLY hurts) -- an ordinary way to say it, with "খুব" inserted. This is the fix, and the test that pins it.
"""

import filecmp
import os

from hypothesis import given
from hypothesis import strategies as st

from agent.phrase_proximity import any_phrase_matches, phrase_matches

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_live_call_regression_case():
    assert phrase_matches("বুকে ব্যথা", "আমার বুকে খুব ব্যথা করছে")
    assert phrase_matches("chest pain", "I have chest area pain")


def test_the_literal_adjacent_phrase_still_matches():
    assert phrase_matches("বুকে ব্যথা", "আমার বুকে ব্যথা করছে")
    assert phrase_matches("chest pain", "I have chest pain")


def test_a_different_body_part_or_wrong_order_does_not_match():
    assert not phrase_matches("বুকে ব্যথা", "আমার হাতে ব্যথা করছে")
    assert not phrase_matches("বুকে ব্যথা", "ব্যথা আর বুকে কিছু নেই")  # wrong order


def test_more_than_the_gap_limit_does_not_match_unless_widened():
    # 3 words between "বুকে" and "ব্যথা"; the default gap is 2
    text = "আমার বুকে সেই থেকে অনেকটা ব্যথা করছে"
    assert not phrase_matches("বুকে ব্যথা", text)
    assert phrase_matches("বুকে ব্যথা", text, max_gap_words=3)


def test_a_single_word_phrase_is_a_plain_substring_check():
    assert phrase_matches("জ্বর", "আমার জ্বর হয়েছে")
    assert not phrase_matches("জ্বর", "আমার সর্দি হয়েছে")


def test_empty_phrase_or_text_never_matches():
    assert not phrase_matches("", "some text")
    assert not phrase_matches("chest pain", "")
    assert not phrase_matches("", "")


def test_any_phrase_matches_is_true_if_any_one_of_the_list_matches():
    assert any_phrase_matches(["জ্বর", "বুকে ব্যথা"], "আমার বুকে খুব ব্যথা")
    assert not any_phrase_matches(["জ্বর", "বুকে ব্যথা"], "আমার হাতে ব্যথা")
    assert not any_phrase_matches([], "anything")


def test_agent_and_clinic_api_copies_are_byte_identical():
    a = os.path.join(REPO_ROOT, "agent", "phrase_proximity.py")
    b = os.path.join(REPO_ROOT, "clinic-api", "phrase_proximity.py")
    assert filecmp.cmp(a, b, shallow=False), "agent/phrase_proximity.py and clinic-api/phrase_proximity.py drifted"


@given(st.text(max_size=15), st.text(max_size=200))
def test_phrase_matches_never_raises(phrase, text):
    phrase_matches(phrase, text)


def _reference_match(words: list[str], text: str, max_gap_words: int) -> bool:
    """A from-scratch reimplementation of the same rule, independent of the module's own regex-building code, so a
    bug shared by both would have to be the same bug reasoned about twice, not one algorithm trusting itself."""
    if not words:
        return False
    if len(words) == 1:
        return words[0] in text  # a single word is a plain substring check, same as phrase_matches itself
    text_words = text.split()
    i = 0
    prev_end = -1
    for w in words:
        found_at = next((j for j in range(i, len(text_words)) if text_words[j] == w), None)
        if found_at is None:
            return False
        if prev_end >= 0 and (found_at - prev_end - 1) > max_gap_words:
            return False
        prev_end = found_at
        i = found_at + 1
    return True


@given(
    st.lists(st.text(alphabet="abcde", min_size=1, max_size=3), min_size=1, max_size=4),
    st.text(alphabet="abcde ", max_size=40),
)
def test_matching_agrees_with_an_independent_reimplementation(words, text):
    phrase = " ".join(words)
    assert phrase_matches(phrase, text) == _reference_match(words, text, 2)
