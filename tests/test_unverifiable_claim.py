"""agent/unverifiable_claim.py -- a caller's assertion never becomes a system fact.

The interesting tests here are the ones about what this guard must NOT do. Its
story has two criteria pulling against each other: never let an assertion become
a fact, but *check the system of record wherever verification is possible*. The
module this was ported from satisfied the first and, in this repository, would
break the second -- it declined four claim categories, two of which voicebot-main
can actually look up (`GET /api/v1/bookings/lookup`,
`GET /api/v1/billing/outstanding`).

So the groups below are: the two genuinely uncheckable categories are caught; the
two checkable ones **fall through** to their real lookups; any utterance that
also asks for a real check falls through regardless of what it asserts; and the
replies decline without echoing the claim back as true.
"""

from __future__ import annotations

import pytest

from agent import persona
from agent.phrases import PHRASES, phrase
from agent.unverifiable_claim import (
    CATEGORY_DOCTOR_APPROVAL,
    CATEGORY_REPORT_RESULT,
    NONE,
    VERIFIABLE_HERE,
    asks_for_a_real_check,
    detect_unverifiable_claim,
    is_unverifiable_claim,
)

LANGS = ("bn", "hi", "en")


# ------------------------------- caught: nothing here can ever check these
@pytest.mark.parametrize(
    "text,category",
    [
        ("the doctor already approved", CATEGORY_DOCTOR_APPROVAL),
        ("my doctor approved this", CATEGORY_DOCTOR_APPROVAL),
        ("doctor already said yes", CATEGORY_DOCTOR_APPROVAL),
        ("doctor ne pehle hi haan bol diya", CATEGORY_DOCTOR_APPROVAL),
        ("ডাক্তার আগে থেকেই অনুমোদন দিয়েছেন", CATEGORY_DOCTOR_APPROVAL),
        ("डॉक्टर ने पहले ही मंज़ूरी दे दी है", CATEGORY_DOCTOR_APPROVAL),
        ("my report is normal", CATEGORY_REPORT_RESULT),
        ("my report came back normal", CATEGORY_REPORT_RESULT),
        ("mera report normal hai", CATEGORY_REPORT_RESULT),
        ("আমার রিপোর্ট নরমাল", CATEGORY_REPORT_RESULT),
        ("मेरी रिपोर्ट नॉर्मल है", CATEGORY_REPORT_RESULT),
    ],
)
def test_an_uncheckable_assertion_is_caught_and_categorised(text, category):
    cat, matched = detect_unverifiable_claim(text)
    assert cat == category
    assert matched, "the matched phrase should be reported for auditing"
    assert is_unverifiable_claim(text) is True


@pytest.mark.parametrize("lang", LANGS)
def test_both_categories_cover_every_language(lang):
    """The Devanagari phrases are new in this repository."""
    import agent.unverifiable_claim as m

    def has(lang_, phrases):
        if lang_ == "bn":
            return any(any("ঀ" <= c <= "৿" for c in p) for p in phrases)
        if lang_ == "hi":
            return any(any("ऀ" <= c <= "ॿ" for c in p) for p in phrases)
        return any(p.isascii() for p in phrases)

    for category, phrases in m.CLAIM_PATTERNS.items():
        assert has(lang, phrases), f"{category} has no {lang} phrasing"


# ---------------- NOT caught: this repository can verify these itself
@pytest.mark.parametrize(
    "text",
    [
        "my appointment is tomorrow",
        "my appointment is today",
        "i already have an appointment",
        "i already paid",
        "i have already paid",
        "payment is already done",
        "mera appointment kal hai",
        "আমার অ্যাপয়েন্টমেন্ট কাল আছে",
    ],
)
def test_a_claim_this_system_can_check_is_left_to_the_real_lookup(text):
    """The load-bearing divergence from the module this was ported from. These
    were declined there because that codebase had no lookup for them. This one
    does, and declining them would break the story's own second criterion --
    "the agent checks the system of record where verification is possible" --
    as well as losing a working capability."""
    assert detect_unverifiable_claim(text)[0] == NONE
    assert is_unverifiable_claim(text) is False


def test_the_checkable_categories_are_documented_in_the_code():
    """So the decision is discoverable from the module, not only from a report."""
    assert "APPOINTMENT_EXISTING_CLAIM" in VERIFIABLE_HERE
    assert "PAYMENT_ALREADY_CLAIM" in VERIFIABLE_HERE
    assert "bookings/lookup" in VERIFIABLE_HERE["APPOINTMENT_EXISTING_CLAIM"]
    assert "billing/outstanding" in VERIFIABLE_HERE["PAYMENT_ALREADY_CLAIM"]


# ------------- NOT caught: the caller is asking for a real check anyway
@pytest.mark.parametrize(
    "text",
    [
        "my report is normal, can you check",
        "the doctor already approved, please confirm",
        "my doctor approved this, can you check my booking",
        "my report is normal but I want to book another test",
        "ডাক্তার আগে থেকেই অনুমোদন দিয়েছেন, একটু দেখুন",
        "डॉक्टर ने पहले ही मंज़ूरी दे दी है, ज़रा देखिए",
    ],
)
def test_an_assertion_that_also_asks_for_a_check_falls_through(text):
    """The narrowing rule. Without it this guard becomes a second verification
    system that answers "I cannot confirm" to someone who asked to be checked --
    which the story forbids explicitly."""
    assert asks_for_a_real_check(text) is True
    assert detect_unverifiable_claim(text)[0] == NONE


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "what is the price of a CBC test",
        "is my report ready",
        "is Dr Sen available tomorrow",
        "I want to book a CBC",
        "আমার রিপোর্ট কি হয়ে গেছে",
        "मुझे रिपोर्ट चाहिए",
    ],
)
def test_ordinary_business_and_plain_questions_are_untouched(text):
    """"Is my report ready" is a QUESTION with a real lookup behind it, and must
    keep reaching it. Only declarative assertions are in scope."""
    assert detect_unverifiable_claim(text)[0] == NONE


# ---------------------------------------------------------------- the replies
@pytest.mark.parametrize("lang", LANGS)
def test_the_decline_exists_in_all_three_languages_and_offers_a_person(lang):
    assert "cannot_confirm_claim" in PHRASES[lang]
    text = phrase("cannot_confirm_claim", lang)
    assert persona.violations(text) == [], persona.violations(text)
    assert text.count("?") == 1, "a person is not actually being offered"


def test_phrase_coverage_is_still_identical_across_languages():
    assert sorted(PHRASES["bn"]) == sorted(PHRASES["hi"]) == sorted(PHRASES["en"])


def test_the_english_decline_never_echoes_the_claim_back_as_true():
    """Acceptance criterion 4. It is a fixed phrase, so it cannot contain the
    caller's words -- this pins the absence of any agreeing language a future
    editor might add."""
    text = phrase("cannot_confirm_claim", "en").lower()
    assert "cannot confirm" in text
    for agreeing in ("that's right", "that is right", "as you said", "since you", "i see that you have", "correct"):
        assert agreeing not in text


def test_a_result_claim_is_answered_with_the_clinical_reply_not_a_bare_decline():
    """A claim about what a result MEANS is clinical territory, so it reuses the
    already-tested clinical reply: a doctor will explain, and here is one. The
    two replies are genuinely different lines, which is why the category matters."""
    assert phrase("clinical_interpretation", "en") != phrase("cannot_confirm_claim", "en")
    clinical = phrase("clinical_interpretation", "en").lower()
    assert "doctor" in clinical and "connect" in clinical
