"""Unit tests for the test_prep and clinic_faq additions to
agent/fast_path.py -- pure string-matching logic, no network, no GPU, no
pod. Uses a small fixed payload of the same shape
clinic-api's /api/v1/catalogue actually returns (verified against a live
local instance in tests/test_clinic_api_new_endpoints.py).

    python -m pytest tests/test_fast_path_faq_prep.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.fast_path import Catalogue, FastPath

PAYLOAD = {
    "tests": [
        {"name": "Lipid Profile", "aliases_bn": ["লিপিড প্রোফাইল", "কোলেস্টেরল টেস্ট"]},
        {"name": "Complete Blood Count (CBC)", "aliases_bn": ["সিবিসি", "সি বি সি"]},
    ],
    "doctors": [
        {"name": "Dr. A. Sen", "surname": "Sen", "aliases_bn": ["সেন"]},
    ],
    "faq_topics": [
        {"topic": "hours", "keywords_bn": ["সময়", "কখন খোলে", "কখন বন্ধ", "ক্লিনিকের সময়"]},
        {"topic": "parking", "keywords_bn": ["পার্কিং", "গাড়ি রাখার জায়গা"]},
        {"topic": "location", "keywords_bn": ["কোথায়", "ঠিকানা", "লোকেশন"]},
    ],
}


def _fast_path() -> FastPath:
    return FastPath(Catalogue(PAYLOAD))


def test_catalogue_length_includes_faq_topics():
    cat = Catalogue(PAYLOAD)
    assert len(cat) == 2 + 1 + 3  # tests + doctors + faq_topics


def test_test_prep_resolves_for_known_test():
    fp = _fast_path()
    result = fp.resolve("লিপিড প্রোফাইল টেস্টের আগে কী প্রস্তুতি নিতে হবে")
    assert result is not None
    assert result.intent == "test_prep"
    assert result.slots["test_name"]
    assert result.confidence >= 0.72


def test_test_prep_fasting_phrase_also_routes_to_prep_not_rate():
    fp = _fast_path()
    result = fp.resolve("সিবিসি টেস্টের আগে কি খালি পেটে থাকতে হবে")
    assert result is not None
    assert result.intent == "test_prep"


def test_rate_question_is_unaffected_by_the_prep_addition():
    fp = _fast_path()
    result = fp.resolve("সিবিসি টেস্টের রেট কত")
    assert result is not None
    assert result.intent == "test_rate"


def test_clinic_faq_resolves_hours():
    fp = _fast_path()
    result = fp.resolve("আপনাদের ক্লিনিকের সময় কী")
    assert result is not None
    assert result.intent == "clinic_faq"
    assert result.slots["faq_topic"] == "hours"


def test_clinic_faq_resolves_parking_not_hours():
    fp = _fast_path()
    result = fp.resolve("গাড়ি রাখার জায়গা আছে কি")
    assert result is not None
    assert result.intent == "clinic_faq"
    assert result.slots["faq_topic"] == "parking"


def test_clinic_faq_does_not_steal_availability_questions():
    fp = _fast_path()
    result = fp.resolve("ডক্টর সেন কবে চেম্বারে বসবেন")
    assert result is not None
    assert result.intent == "doctor_availability", "an availability question must never be misrouted to clinic_faq"


def test_unrecognised_faq_style_question_abstains():
    fp = _fast_path()
    result = fp.resolve("আপনাদের হাসপাতালের মালিক কে")
    assert result is None


def test_prep_plus_rate_cue_together_abstains_as_ambiguous():
    # Contrived, but exercises the "more than one cue fired" guard now that
    # there are three cue families (rate/avail/prep) instead of two.
    fp = _fast_path()
    result = fp.resolve("সিবিসি টেস্টের রেট কত আর প্রস্তুতি কী")
    assert result is None


def test_empty_slots_always_carries_faq_topic_key():
    fp = _fast_path()
    result = fp.resolve("সিবিসি টেস্টের রেট কত")
    assert "faq_topic" in result.slots
    assert result.slots["faq_topic"] is None
