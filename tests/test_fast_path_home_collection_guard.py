"""KCD-387 full flow: the FastPath guard in agent/fast_path.py's resolve() that stops an
ACTIONABLE home-collection utterance (a caller naming a test, or giving a pincode) from being
answered by the old, fixed, context-free "home_collection" FAQ sentence before the LLM -- and
therefore before the real "home_collection" intent -- ever gets a chance to run.

This is the exact risk the spec's FINAL RULE worked example calls out: a caller going through the
CURRENT voice orchestrator with an utterance like "Amar Uric Acid test ta bari theke collect kora
jabe? Amar pincode 700091" must NOT receive the generic FAQ answer. test_orchestrator_home_collection.py
proves the full multi-turn flow once the "home_collection" intent is reached (it mocks
_resolve_intent directly, by that suite's own established convention); this file proves the
upstream gate that decides whether the LLM-free fast path would have intercepted the turn BEFORE
that intent classification ever ran -- a bare "do you do home collection?" still gets the fast,
free FAQ answer (no test, no pincode: a real general question), while anything with a test name or
a pincode abstains and falls through to the model.

Pure and offline -- no network, no database, no GPU.

    python -m pytest tests/test_fast_path_home_collection_guard.py -v
"""

import os
import sys

# ADDED BY SOURAV: guarded, not unconditional -- an unconditional sys.path.insert(0, REPO_ROOT) at
# module level (several pre-existing test files use that form) can outrun a clinic-api test's own
# "if CLINIC_API_DIR not in sys.path" guard when pytest collects the whole tests/ directory in one
# process, since collection imports every test module before any test runs; this was found and
# root-caused during this story's own full-suite regression sweep (see the final report's
# regression section) and is avoided here rather than repeated in a new file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.fast_path import Catalogue, FastPath

PAYLOAD = {
    "tests": [
        {"name": "Uric Acid", "aliases_bn": ["ইউরিক এসিড"], "aliases_en": ["uric acid"]},
    ],
    "doctors": [],
    "faq_topics": [
        {
            "topic": "home_collection",
            "keywords_bn": ["বাড়ি থেকে", "বাড়িতে এসে", "হোম কালেকশন"],
            "keywords_en": ["home collection", "collect from home", "collect at home"],
        },
        {"topic": "hours", "keywords_bn": ["সময়", "কখন খোলে"], "keywords_en": ["what time", "hours"]},
    ],
}


def _fast_path() -> FastPath:
    return FastPath(Catalogue(PAYLOAD))


def test_a_bare_home_collection_question_with_no_test_or_pincode_still_gets_the_fast_faq_answer():
    fp = _fast_path()
    result = fp.resolve("do you do home collection", "en")
    assert result is not None
    assert result.intent == "clinic_faq"
    assert result.slots.get("faq_topic") == "home_collection"


def test_naming_a_test_alongside_home_collection_words_abstains_to_reach_the_real_intent():
    fp = _fast_path()
    result = fp.resolve("can uric acid be collected at home", "en")
    assert result is None, "an actionable mention must fall through to the LLM's home_collection intent, never the FAQ"


def test_giving_a_six_digit_pincode_alongside_home_collection_words_abstains_too():
    fp = _fast_path()
    result = fp.resolve("home collection available for pincode 700091", "en")
    assert result is None


def test_a_five_or_seven_digit_run_is_not_mistaken_for_a_pincode():
    # Neither a 5-digit nor a 7-digit run is a real Indian PIN code -- the guard must not
    # over-fire on an unrelated number (a flat number) and silently swallow a genuinely generic
    # FAQ question into an abstain. ("charge"/"price"-style wording is avoided here so the
    # example exercises only this guard, not the separate, unrelated rate/price cue-word branch
    # that runs earlier in resolve().)
    fp = _fast_path()
    assert fp.resolve("do you have home collection for flat number 12345", "en") is not None
    assert fp.resolve("home collection for house number 1234567 please", "en") is not None


def test_other_faq_topics_are_completely_unaffected_by_this_guard():
    fp = _fast_path()
    result = fp.resolve("what time do you open", "en")
    assert result is not None
    assert result.slots.get("faq_topic") == "hours"


def test_the_guard_is_scoped_to_the_home_collection_topic_object_directly():
    # Exercises FastPath._home_collection_is_actionable in isolation, independent of resolve()'s
    # own FAQ-matching confidence floor -- the two tests above already prove it end to end.
    fp = _fast_path()
    assert fp._home_collection_is_actionable("pincode 700091", "en") is True
    assert fp._home_collection_is_actionable("uric acid", "en") is True
    assert fp._home_collection_is_actionable("do you offer this service", "en") is False
