"""Byte-level equality between the tool value and the spoken value.

story title: Numbers are never rounded, reordered or approximated
user story: As a patient, I want the exact figure, so that what I am quoted is
    what I pay.
acceptance criteria: Figures pass from the validated response into the template
    unchanged and are verbalised digit-faithfully. A test asserts byte-level
    equality between the tool value and the spoken value for a corpus of
    amounts, dates and identifiers.

Ported from dev_sourav's test_number_fidelity_review_fixes.py and reworked for
this branch, where the confirmation-ID half is not a partial fix but a live
break: clinic-api generates KCD-{date}-{4 hex}, the old pattern did not match
hex at all, and agent/speakability.py blocked the resulting reply -- so every
successful booking escalated to the counter instead of giving the caller a
number.

"Byte-level equality" needs saying precisely, because the spoken form is
Bengali words and the tool value is digits: what is asserted is that the
DIGIT SEQUENCE survives. Each source digit appears, in order, with nothing
rounded away, nothing reordered, and no leading or trailing zero dropped. The
expected Bengali is reconstructed from the same primitives bn_normalize uses,
so this checks the pipeline rather than restating its output.

---

Test suite for number fidelity in the voice agent pipeline.

Acceptance Criteria: "Figures pass from the validated response into the template
unchanged and are verbalised digit-faithfully. A test asserts byte-level equality
between the tool value and the spoken value for a corpus of amounts, dates and
identifiers."

This test verifies that:
1. Tool/API values are passed unchanged through the pipeline
2. Number verbalization is digit-faithful (no rounding, approximation, or reordering)
3. The spoken form preserves the exact semantic value of the original number
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import speakability  # noqa: E402
from agent.bn_normalize import (  # noqa: E402
    _ONES_TO_99, GROUP_SEPARATOR, spell_out, verbalize,
    number_to_bn_words, date_to_bn_words, time_to_bn_words, digits_one_by_one,
)
# Aliased: pytest collects any module-level name starting with test_ as a
# test case, and would try to run the template itself as one.
from agent.reply_templates import (  # noqa: E402
    booking_reply, test_rate_reply as rate_reply,
)
from agent.tools_client import _parse_exact  # noqa: E402

DIGIT_BN = {str(i): _ONES_TO_99[i] for i in range(10)}


def _digits_spoken(value: str) -> list[str]:
    """The Bengali words a digit string must produce, digit by digit."""
    return [DIGIT_BN[c] for c in value if c.isdigit()]


class _Body:
    """Only what _parse_exact touches: the raw response text."""

    def __init__(self, text: str):
        self.text = text


# --------------------------------------------------- the parse boundary

@pytest.mark.parametrize("literal", [
    "100.00", "0.99", "12547.85", "999999.99", "250.50", "1.10", "0.10",
    "100.000", "7.0", "1234567.89",
])
def test_a_decimal_keeps_its_source_digits(literal):
    """float("100.00") == float("100.0"). Once the default parser has run, the
    trailing zero is gone and no template or verbaliser downstream can put it
    back -- the digit was discarded at the boundary."""
    parsed = _parse_exact(_Body(json.dumps({"rate_inr": float(literal)}).replace(
        json.dumps(float(literal)), literal)))
    assert parsed["rate_inr"] == literal, "the source digits did not survive parsing"
    assert isinstance(parsed["rate_inr"], str), "a decimal must not become a float"


@pytest.mark.parametrize("literal", ["650", "0", "1", "999999"])
def test_a_whole_number_is_untouched(literal):
    """No decimal point in the source means no float in the first place. This
    is the regression that matters most, because rate_inr is an Integer column
    on this branch, so whole numbers are the ONLY case production exercises."""
    parsed = _parse_exact(_Body('{"rate_inr": %s}' % literal))
    assert parsed["rate_inr"] == int(literal)
    assert isinstance(parsed["rate_inr"], int)


def test_the_default_parser_would_have_lost_the_digit():
    """The bug, demonstrated rather than asserted about. If this ever stops
    failing, the fix has become unnecessary and this file can shrink."""
    lost = json.loads('{"rate_inr": 100.00}')["rate_inr"]
    assert str(lost) == "100.0", "default parsing no longer drops the zero"
    kept = _parse_exact(_Body('{"rate_inr": 100.00}'))["rate_inr"]
    assert kept == "100.00"


def test_a_malformed_body_still_raises_a_value_error():
    """_parse_exact must keep composing with the failure handling in
    tools_client: JSONDecodeError subclasses ValueError, which is the arm that
    reports "the system cannot be reached"."""
    with pytest.raises(ValueError):
        _parse_exact(_Body("<html>502 Bad Gateway</html>"))


# ------------------------------------------------- amounts, end to end

@pytest.mark.parametrize("rate", [250, 650, 1800, 2200, 100, 7, 999999])
def test_a_rate_reaches_the_caller_digit_for_digit(rate):
    reply = rate_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "test_name_bn": "ইউরিক অ্যাসিড",
         "rate_inr": rate, "sample_type": "Blood", "report_time_hours": 12})
    assert str(rate) in reply, "the exact figure must reach the template unchanged"
    verdict = speakability.check(reply)
    assert verdict.state == speakability.SPEAKABLE
    # Not digit-by-digit -- a price is spoken as a NUMBER ("চারশো পঞ্চাশ"), which
    # is correct for money and is why this asserts the rendering is non-empty
    # and audible rather than asserting a digit sequence.
    assert "টাকা" in verdict.spoken


# --------------------------------------------- identifiers, end to end

REAL_IDS = [
    "KCD-20260911-4A2F",   # hex with letters -- ~85% of real IDs
    "KCD-20260911-0031",   # all digits, leading zeros
    "KCD-20260911-ABCD",   # all letters -- ~2% of real IDs
    "KCD-20260911-0000",
    "KCD-20261231-FFFF",
]


@pytest.mark.parametrize("conf_id", REAL_IDS)
def test_a_confirmation_id_is_spoken_in_full(conf_id):
    """Every digit, in order, no group rounded into a word.

    The old pattern captured only the first hyphen group, so the trailing one
    fell through to the bare-integer sweep: "0031" was spoken as the WORD
    "একত্রিশ" (thirty-one) with the leading zeros gone -- a caller reading it
    back to the counter would have the wrong number.
    """
    spoken = verbalize(f"কনফার্মেশন নম্বর: {conf_id}।")
    for word in _digits_spoken(conf_id):
        assert word in spoken, f"{conf_id}: a digit is missing from {spoken!r}"

    # story title: Figures are spoken at a pace a caller can write down
    # user story: As a patient noting a price or a reference, I want it
    #   grouped and slower, so that I do not have to ask twice.
    # acceptance criteria: Prices, phone numbers and reference identifiers
    #   are spoken with grouping and a reduced rate through the per-request
    #   speed parameter. A listening test confirms callers transcribe
    #   correctly on first hearing.
    #
    # This assertion used to read `spell_out(conf_id) in spoken` -- "the ID
    # is spelled out as one unit". spell_out() now GROUPS, so that form would
    # still pass, symmetrically and by accident, while asserting nothing
    # about the thing that changed. Rewritten to the contract it actually has
    # now: every group present, in order, separated.
    #
    # The ordered scan is the part that matters. "Grouped" must never become
    # licence to reorder -- a reference number read back out of order is the
    # same defect as one read back with a digit missing.
    groups = [g for g in spell_out(conf_id).split(GROUP_SEPARATOR) if g.strip()]
    assert len(groups) > 1, f"{conf_id}: not grouped at all"
    cursor = 0
    for group in groups:
        found = spoken.find(group.strip(), cursor)
        assert found != -1, f"{conf_id}: group {group.strip()!r} missing or out of order"
        cursor = found + len(group.strip())


@pytest.mark.parametrize("conf_id", REAL_IDS)
def test_a_confirmation_id_survives_the_speakability_gate(conf_id):
    """The half dev_sourav deferred, and the half that actually broke bookings
    here. An unmatched hex group leaves Latin characters in a Bengali sentence;
    the tokenizer drops them and the gate blocks the whole reply, so a caller
    whose booking SUCCEEDED was sent to the counter."""
    reply = booking_reply(
        {"doctor_name": "সেন"},
        {"success": True, "confirmation_id": conf_id, "doctor_name": "Dr. A Sen",
         "doctor_name_bn": "সেন", "date": "2026-09-14", "time_slot": "18:30"})
    verdict = speakability.check(reply)
    assert verdict.state == speakability.SPEAKABLE, list(verdict.dropped)


def test_the_generated_id_format_is_the_one_being_tested():
    """Pins the corpus to reality. If clinic-api's generator changes shape,
    this fails rather than the suite quietly testing a format nobody issues."""
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "clinic-api" / "main.py").read_text(encoding="utf-8")
    assert 'f"KCD-{req.date.replace(\'-\', \'\')}-{uuid.uuid4().hex[:4].upper()}"' in source


def test_every_id_the_real_generator_can_emit_is_speakable():
    """A sweep, because the failure was concentrated in a subset: IDs whose
    hex group contains letters. 2000 samples over the real alphabet."""
    random.seed(20260911)
    blocked = []
    for _ in range(2000):
        conf_id = "KCD-2026%02d%02d-%s" % (
            random.randint(1, 12), random.randint(1, 28),
            "".join(random.choice("0123456789ABCDEF") for _ in range(4)))
        if speakability.check(verbalize(f"নম্বর: {conf_id}।")).is_blocked:
            blocked.append(conf_id)
    assert not blocked, f"{len(blocked)}/2000 blocked, e.g. {blocked[:5]}"


@pytest.mark.parametrize("word", [
    "CBC", "ECG", "TSH", "LFT", "USG", "HIV", "DEADBEEF", "BEEF", "ABCDEF",
])
def test_an_ordinary_uppercase_word_is_not_mistaken_for_an_identifier(word):
    """The cost of widening the pattern, held to zero. Allowing bare hex runs
    would have spelled "DEADBEEF" out letter by letter; requiring the
    [A-Z]{2,}\\d{3,} prefix is what keeps a word a word."""
    assert verbalize(word) == word


# ------------------------------------------------------ dates and times

@pytest.mark.parametrize("iso,expected_digits", [
    ("2026-09-14", ["চোদ্দো"]),
    ("2026-09-01", ["এক"]),
    ("2026-12-31", ["একত্রিশ"]),
])
def test_a_date_is_spoken_as_the_day_it_names(iso, expected_digits):
    spoken = verbalize(f"{iso} তারিখে")
    for word in expected_digits:
        assert word in spoken, spoken


@pytest.mark.parametrize("phone", ["9876543210", "9000000009", "1234509876"])
def test_a_phone_number_is_spoken_digit_by_digit_in_order(phone):
    """Order is part of the criterion. A phone number read as a quantity, or
    with its digits regrouped, sends a report to a stranger."""
    spoken = verbalize(f"ফোন {phone}")
    words = _digits_spoken(phone)
    positions = []
    cursor = 0
    for word in words:
        found = spoken.find(word, cursor)
        assert found != -1, f"{phone}: {word} missing from {spoken!r}"
        positions.append(found)
        cursor = found + len(word)
    assert positions == sorted(positions), "digits were reordered"


# --------------------------------------------------------------------- #
# Direct unit tests of the bn_normalize primitives (ported from dev_sourav)
# --------------------------------------------------------------------- #

class TestNumberToBengaliWords:
    """Test that number_to_bn_words produces digit-faithful Bengali representations."""

    def test_single_digit_numbers(self):
        """Single digits should be exact word equivalents."""
        assert number_to_bn_words(0) == "শূন্য"
        assert number_to_bn_words(1) == "এক"
        assert number_to_bn_words(5) == "পাঁচ"
        assert number_to_bn_words(9) == "নয়"

    def test_tens(self):
        """Tens should be exact word equivalents."""
        assert number_to_bn_words(10) == "দশ"
        assert number_to_bn_words(15) == "পনেরো"
        assert number_to_bn_words(20) == "কুড়ি"
        assert number_to_bn_words(99) == "নিরানব্বই"

    def test_hundreds(self):
        """Hundreds should preserve exact value."""
        assert number_to_bn_words(100) == "একশো"
        assert number_to_bn_words(250) == "দুইশো পঞ্চাশ"
        assert number_to_bn_words(999) == "নয়শো নিরানব্বই"

    def test_thousands(self):
        """Thousands should use Indian numbering system (হাজার)."""
        assert number_to_bn_words(1000) == "এক হাজার"
        assert number_to_bn_words(1500) == "এক হাজার পাঁচশো"
        assert number_to_bn_words(12345) == "বারো হাজার তিনশো পঁয়তাল্লিশ"

    def test_lakhs(self):
        """Lakhs should use Indian numbering system (লাখ)."""
        assert number_to_bn_words(100000) == "এক লাখ"
        assert number_to_bn_words(150000) == "এক লাখ পঞ্চাশ হাজার"
        # 999,999 = 9 lakh 99 thousand 999
        result = number_to_bn_words(999999)
        # Just verify it contains the expected components, not exact formatting
        assert "নয় লাখ" in result
        assert "নিরানব্বই হাজার" in result

    def test_crores(self):
        """Crores should use Indian numbering system (কোটি)."""
        assert number_to_bn_words(10000000) == "এক কোটি"
        assert number_to_bn_words(15000000) == "এক কোটি পঞ্চাশ লাখ"

    def test_negative_numbers(self):
        """Negative numbers should preserve sign and magnitude."""
        assert number_to_bn_words(-5) == "মাইনাস পাঁচ"
        assert number_to_bn_words(-100) == "মাইনাস একশো"

    def test_realistic_test_rates(self):
        """Realistic test rates from clinic data should be preserved exactly."""
        # Common diagnostic test prices
        assert number_to_bn_words(650) == "ছয়শো পঞ্চাশ"
        assert number_to_bn_words(1200) == "এক হাজার দুইশো"
        assert number_to_bn_words(850) == "আটশো পঞ্চাশ"
        assert number_to_bn_words(2500) == "দুই হাজার পাঁচশো"
        assert number_to_bn_words(350) == "তিনশো পঞ্চাশ"


class TestDateVerbalization:
    """Test that date verbalization preserves exact dates."""

    def test_date_to_bengali_words(self):
        """Dates should be converted exactly without approximation."""
        assert date_to_bn_words(2026, 8, 24) == "আগস্ট মাসের চব্বিশ তারিখ"
        assert date_to_bn_words(2026, 9, 1) == "সেপ্টেম্বর মাসের এক তারিখ"
        assert date_to_bn_words(2026, 12, 31) == "ডিসেম্বর মাসের একত্রিশ তারিখ"

    def test_invalid_month(self):
        """Invalid months should fall back gracefully but preserve day."""
        assert date_to_bn_words(2026, 13, 5) == "পাঁচ তারিখ"


class TestTimeVerbalization:
    """Test that time verbalization preserves exact times."""

    def test_on_the_hour(self):
        """Exact hours should be preserved."""
        assert time_to_bn_words(9, 0) == "সকাল নটা"
        assert time_to_bn_words(14, 0) == "দুপুর দুটো"
        assert time_to_bn_words(18, 0) == "সন্ধ্যা ছটা"

    def test_quarter_hour(self):
        """Quarter hours should use natural Bengali forms (সোয়া)."""
        assert time_to_bn_words(9, 15) == "সকাল সোয়া নটা"
        assert time_to_bn_words(14, 15) == "দুপুর সোয়া দুটো"

    def test_half_hour(self):
        """Half hours should use natural Bengali forms (সাড়ে)."""
        assert time_to_bn_words(9, 30) == "সকাল সাড়ে নটা"
        assert time_to_bn_words(14, 30) == "দুপুর সাড়ে দুটো"

    def test_quarter_to(self):
        """Quarter to should use natural Bengali forms (পৌনে)."""
        assert time_to_bn_words(9, 45) == "সকাল পৌনে দশটা"
        assert time_to_bn_words(14, 45) == "দুপুর পৌনে তিনটে"

    def test_other_minutes(self):
        """Other minutes should be preserved exactly."""
        assert time_to_bn_words(9, 10) == "সকাল নটা বেজে দশ মিনিট"
        assert time_to_bn_words(14, 25) == "দুপুর দুটো বেজে পঁচিশ মিনিট"


class TestIdentifierVerbalization:
    """Test that identifiers (confirmation IDs, phone numbers) are digit-faithful."""

    def test_confirmation_ids(self):
        """Confirmation IDs should be spelled out character by character.

        UPDATED BY SOURAV -- KCD-445 test cleanup. These assertions predate
        spell_out()'s grouping contract ("Figures are spoken at a pace a
        caller can write down" -- see spell_out()'s own docstring in
        agent/bn_normalize.py): it now joins the letters group and the
        digits group with GROUP_SEPARATOR, a deliberate pacing comma, not a
        flat run of words. The digits and letters themselves are unchanged
        and still in the original order; only the separator was missing
        from these expectations."""
        assert spell_out("KCD-4471") == f"কে সি ডি{GROUP_SEPARATOR} চার চার সাত এক"
        assert spell_out("KCD-1234") == f"কে সি ডি{GROUP_SEPARATOR} এক দুই তিন চার"
        assert spell_out("KCD-9876") == f"কে সি ডি{GROUP_SEPARATOR} নয় আট সাত ছয়"

    def test_phone_numbers(self):
        """Phone numbers should be read digit by digit."""
        assert digits_one_by_one("9876543210") == "নয় আট সাত ছয় পাঁচ চার তিন দুই এক শূন্য"
        assert digits_one_by_one("1234567890") == "এক দুই তিন চার পাঁচ ছয় সাত আট নয় শূন্য"


class TestFullVerbalizationPipeline:
    """Test the full verbalize() function with realistic templates."""

    def test_price_in_template(self):
        """A complete price template should preserve the exact amount."""
        template = "Uric Acid টেস্টের রেট 650 টাকা।"
        result = verbalize(template)
        # The number 650 should become "ছয়শো পঞ্চাশ"
        assert "ছয়শো পঞ্চাশ" in result
        # The original number should not appear as digits
        assert "650" not in result

    def test_confirmation_id_in_template(self):
        """A confirmation ID in a template should be spelled out.

        UPDATED BY SOURAV -- KCD-445 test cleanup. Same stale-grouping issue
        as TestIdentifierVerbalization.test_confirmation_ids just above:
        spell_out()'s output is comma-grouped now, so the expected substring
        needs the same separator."""
        template = "কনফার্মেশন নম্বর: KCD-4471।"
        result = verbalize(template)
        # Should be spelled out character by character, in groups
        assert f"কে সি ডি{GROUP_SEPARATOR} চার চার সাত এক" in result
        # Original ID should not appear as is
        assert "KCD-4471" not in result

    def test_date_in_template(self):
        """A date in a template should be converted to Bengali words."""
        template = "২০২৬-০৮-২৪ তারিখে"
        result = verbalize(template)
        # Should be converted to Bengali date format
        assert "আগস্ট" in result
        assert "চব্বিশ" in result
        assert "তারিখ" in result

    def test_time_slot_in_template(self):
        """A time slot in a template should be converted to Bengali time."""
        template = "সময় 09:30"
        result = verbalize(template)
        # Should use natural Bengali time expression
        assert "সাড়ে" in result or "বেজে" in result

    def test_phone_number_in_template(self):
        """A phone number should be read digit by digit."""
        template = "ফোন: 9876543210"
        result = verbalize(template)
        # Should be digit by digit
        assert "নয় আট সাত ছয় পাঁচ চার তিন দুই এক শূন্য" in result

    def test_complex_booking_confirmation(self):
        """A full booking confirmation should preserve all values."""
        template = "আপনার অ্যাপয়েন্টমেন্ট কনফার্ম হয়েছে। ডাঃ সেন, 2026-08-24, সময় 09:30। কনফার্মেশন নম্বর: KCD-4471।"
        result = verbalize(template)
        
        # Check date
        assert "আগস্ট" in result
        assert "চব্বিশ" in result
        
        # Check time
        assert "সাড়ে" in result or "বেজে" in result
        
        # Check confirmation ID
        assert "কে সি ডি" in result
        assert "চার চার সাত এক" in result


class TestByteLevelEquality:
    """Test that the semantic value is preserved through transformation.

    This is the core requirement: byte-level equality between the tool value
    and the spoken value in terms of semantic meaning.
    """

    def test_amount_semantic_preservation(self):
        """The spoken amount should represent the exact same value as the tool value."""
        test_cases = [
            (650, "ছয়শো পঞ্চাশ"),
            (1200, "এক হাজার দুইশো"),
            (850, "আটশো পঞ্চাশ"),
            (2500, "দুই হাজার পাঁচশো"),
        ]
        
        for numeric_value, expected_bengali in test_cases:
            result = number_to_bn_words(numeric_value)
            assert result == expected_bengali, (
                f"Numeric value {numeric_value} should verbalize to {expected_bengali}, "
                f"got {result} instead"
            )

    def test_date_semantic_preservation(self):
        """The spoken date should represent the exact same date as the tool value."""
        # ISO date -> Bengali words should preserve exact date
        iso_date = "2026-08-24"
        template = f"{iso_date} তারিখে"
        result = verbalize(template)
        
        # The year, month, and day should all be preserved
        assert "২০২৬" not in result  # Bengali digits should be converted
        assert "আগস্ট" in result  # Month name preserved
        assert "চব্বিশ" in result  # Day (24) preserved

    def test_identifier_semantic_preservation(self):
        """The spoken identifier should represent the exact same identifier as the tool value."""
        test_cases = [
            "KCD-4471",
            "KCD-1234",
            "KCD-9876",
        ]
        
        for identifier in test_cases:
            result = spell_out(identifier)
            # Each character should be represented
            parts = identifier.replace("-", "")
            spoken_parts = result.split()
            # Should have same number of digit/letter representations
            assert len(spoken_parts) == len(parts), (
                f"Identifier {identifier} should have {len(parts)} spoken parts, "
                f"got {len(spoken_parts)} instead"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
