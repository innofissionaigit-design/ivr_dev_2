"""The model's answer and the clinic API's answers are parsed into declared schemas (pydantic) at the boundary:
agent/intent_schema.py for what the intent model may say, agent/api_models.py for what the clinic API may answer.

    python -m pytest tests/test_schemas.py -v
"""

import asyncio
import os
import sys

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.api_models import ApiShapeError, DoctorAnswer, SuccessAnswer, TestAnswer, validated  # noqa: E402
from agent.enquiry_followup import ENQUIRY_INTENTS  # noqa: E402
from agent.intent_schema import FAQ_TOPICS, SLOT_KEYS, VALID_INTENTS, parse_extraction  # noqa: E402
from agent.llm import _validate  # noqa: E402
from agent.tools_client import ClinicToolsClient, ToolCallError  # noqa: E402

FAST = settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def full(**over):
    slots = dict.fromkeys(SLOT_KEYS)
    slots.update(over)
    return slots


# ==================================================================================== the intent model's answer


def test_a_well_formed_answer_becomes_a_typed_object_and_the_same_plain_dict():
    model, errors = parse_extraction({"intent": "test_rate", "slots": full(test_name="CBC"), "direct_reply_bn": None})
    assert model is not None and errors == []
    d = model.to_dict()
    assert d["intent"] == "test_rate" and d["slots"]["test_name"] == "CBC" and d["secondary_intent"] is None
    assert set(d["slots"]) == set(SLOT_KEYS)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "test_rate",
        42,
        {},
        {"intent": "delete_all_bookings", "slots": full()},
        {"intent": "test_rate"},
        {"intent": "test_rate", "slots": "CBC"},
        {"intent": "test_rate", "slots": ["CBC"]},
        {"intent": ["test_rate"], "slots": full()},
    ],
)
def test_an_answer_that_cannot_be_used_is_rejected_not_repaired(bad):
    model, errors = parse_extraction(bad)
    assert model is None and errors


def test_a_slot_the_model_left_out_is_reported_but_does_not_reject_the_answer():
    model, errors = parse_extraction({"intent": "test_rate", "slots": {"test_name": "CBC"}})
    assert model is not None and "slots.faq_topic: missing" in errors and model.slots.test_name == "CBC"


@pytest.mark.parametrize(
    "slot,value,expected",
    [
        ("test_names", "CBC", ["CBC"]),
        ("test_names", ["CBC", "ESR"], ["CBC", "ESR"]),
        ("test_names", [1, 2, 3], None),
        ("test_names", {"a": 1}, None),
        ("spelled_letters", "ravi", ["r", "a", "v", "i"]),
        ("patient_age", "72", 72),
        ("patient_age", 72.0, 72),
        ("patient_age", "seventy", None),
        ("patient_age", 0, None),
        ("patient_age", 200, None),
        ("patient_age", True, None),
        ("phone", 9830011111, "9830011111"),
        ("phone", ["9830011111"], None),
        ("date", {"day": 1}, None),
        ("doctor_name", True, None),
        ("faq_topic", "hours", "hours"),
        ("faq_topic", "the_secret_topic", None),
    ],
)
def test_every_slot_is_normalised_never_trusted(slot, value, expected):
    model, _ = parse_extraction({"intent": "clinic_faq", "slots": full(**{slot: value})})
    assert model is not None and getattr(model.slots, slot) == expected


def test_only_smalltalk_may_carry_a_reply_the_model_wrote():
    keep, _ = parse_extraction({"intent": "smalltalk", "slots": full(), "direct_reply_bn": "নমস্কার"})
    strip, _ = parse_extraction({"intent": "book_test", "slots": full(), "direct_reply_bn": "your test costs 250 taka"})
    assert keep is not None and keep.direct_reply_bn == "নমস্কার"
    assert strip is not None and strip.direct_reply_bn is None


def test_a_second_question_is_kept_only_when_it_is_a_stateless_enquiry_with_slots():
    ok, _ = parse_extraction(
        {
            "intent": "test_rate",
            "slots": full(test_name="CBC"),
            "secondary_intent": "doctor_availability",
            "secondary_slots": full(doctor_name="Sen"),
        }
    )
    assert ok is not None and ok.secondary_intent == "doctor_availability" and ok.secondary_slots.doctor_name == "Sen"
    for secondary in ("book_appointment", "unclear", "cancel_appointment", "nonsense", None):
        dropped, _ = parse_extraction(
            {"intent": "test_rate", "slots": full(), "secondary_intent": secondary, "secondary_slots": full()}
        )
        assert dropped is not None and dropped.secondary_intent is None and dropped.secondary_slots is None
    no_slots, _ = parse_extraction({"intent": "test_rate", "slots": full(), "secondary_intent": "test_prep"})
    assert no_slots is not None and no_slots.secondary_intent is None


def test_an_invented_secondary_faq_topic_is_nulled_like_the_primary_one():
    model, _ = parse_extraction(
        {
            "intent": "test_rate",
            "slots": full(),
            "secondary_intent": "clinic_faq",
            "secondary_slots": full(faq_topic="x"),
        }
    )
    assert model is not None and model.secondary_slots.faq_topic is None


def test_the_old_validate_function_still_works_in_place():
    data = {"intent": "book_test", "slots": full(test_names="CBC"), "direct_reply_bn": "costs 250"}
    ok, _ = _validate(data)
    assert ok and data["slots"]["test_names"] == ["CBC"] and data["direct_reply_bn"] is None
    bad = {"intent": "nonsense", "slots": full()}
    ok, errors = _validate(bad)
    assert not ok and any("invalid intent" in e for e in errors)


# Any JSON-shaped value the model could conceivably emit: the parser returns a typed answer or nothing, never raises.
json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=20),
    lambda kids: st.lists(kids, max_size=4) | st.dictionaries(st.text(max_size=12), kids, max_size=6),
    max_leaves=25,
)


@FAST
@given(json_values)
def test_any_json_at_all_is_parsed_or_rejected_never_an_exception(value):
    model, errors = parse_extraction(value)
    assert (model is None) == any(
        "invalid intent" in e or "slots: expected" in e or "expected a JSON" in e for e in errors
    )


@FAST
@given(
    st.sampled_from(sorted(VALID_INTENTS)),
    st.dictionaries(st.sampled_from(SLOT_KEYS), json_values, max_size=len(SLOT_KEYS)),
    json_values,
    st.sampled_from([*sorted(ENQUIRY_INTENTS), "book_appointment", "unclear", None, "x"]),
    json_values,
)
def test_whatever_the_slots_hold_the_parsed_answer_has_the_declared_types(intent, slots, reply, secondary, sec_slots):
    model, _ = parse_extraction(
        {
            "intent": intent,
            "slots": slots,
            "direct_reply_bn": reply,
            "secondary_intent": secondary,
            "secondary_slots": sec_slots,
        }
    )
    assert model is not None
    s = model.slots
    for k in ("test_name", "doctor_name", "date", "phone", "faq_topic"):
        assert getattr(s, k) is None or isinstance(getattr(s, k), str)
    assert s.test_names is None or all(isinstance(x, str) for x in s.test_names)
    assert s.spelled_letters is None or all(isinstance(x, str) for x in s.spelled_letters)
    assert s.patient_age is None or (isinstance(s.patient_age, int) and 0 < s.patient_age < 130)
    assert s.faq_topic is None or s.faq_topic in FAQ_TOPICS
    assert intent == "smalltalk" or model.direct_reply_bn is None
    assert (model.secondary_intent is None) == (model.secondary_slots is None)
    assert model.secondary_intent is None or model.secondary_intent in ENQUIRY_INTENTS


# ====================================================================================== the clinic API's answers


@pytest.mark.parametrize(
    "model,payload",
    [
        (
            TestAnswer,
            {"found": True, "test_name": "CBC", "rate_inr": 350, "sample_type": "Blood", "report_time_hours": 24},
        ),
        (TestAnswer, {"found": False, "query": "x", "did_you_mean": ["CBC"], "ambiguous": True, "extra_field": 1}),
        (DoctorAnswer, {"found": True, "doctor_name": "Dr. A. Sen", "available": True, "chamber_hours": "18:00-20:00"}),
        (SuccessAnswer, {"success": True, "confirmation_id": "KCD-1", "hold_token": "h", "doctor_id": 7}),
        (SuccessAnswer, {"success": False, "reason": "slot_taken", "alternative_slots": ["17:30"]}),
    ],
)
def test_a_real_shaped_answer_passes_and_keeps_exactly_its_own_keys(model, payload):
    assert validated(model, payload, "t") == payload


@pytest.mark.parametrize(
    "model,payload",
    [
        (TestAnswer, {"error": "boom"}),
        (TestAnswer, "<html>502 Bad Gateway</html>"),
        (TestAnswer, [{"found": True}]),
        (TestAnswer, {"found": "yes"}),
        (TestAnswer, {"found": True, "rate_inr": "350"}),
        (TestAnswer, {"found": True, "did_you_mean": "CBC"}),
        (SuccessAnswer, {"success": "true"}),
        (SuccessAnswer, {"success": True, "alternative_slots": [1, 2]}),
        (SuccessAnswer, {"success": True, "doctor_id": "7"}),
        (DoctorAnswer, {"found": None}),
    ],
)
def test_an_answer_of_the_wrong_shape_is_rejected(model, payload):
    with pytest.raises(ApiShapeError):
        validated(model, payload, "t")


def test_the_error_names_the_bad_fields_and_never_the_values():
    with pytest.raises(ApiShapeError) as e:
        validated(TestAnswer, {"found": True, "rate_inr": "9830011111"}, "get_test_rate")
    assert "rate_inr" in str(e.value) and "9830011111" not in str(e.value)


def _client_answering(payload, status=200):
    client = ClinicToolsClient("http://clinic.test")

    def handler(_request):
        if isinstance(payload, str):
            return httpx.Response(status, text=payload)
        return httpx.Response(status, json=payload)

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://clinic.test")
    return client


def test_the_client_returns_the_answer_unchanged_when_it_has_the_right_shape():
    body = {"found": True, "test_name": "CBC", "rate_inr": 350, "sample_type": "Blood", "report_time_hours": 24}
    assert asyncio.run(_client_answering(body).get_test_rate("CBC")) == body


@pytest.mark.parametrize("payload", [{"error": "boom"}, {"found": "yes"}, "not json at all", [1, 2, 3]])
def test_the_client_turns_a_malformed_answer_into_a_tool_failure_never_a_fact(payload):
    with pytest.raises(ToolCallError):
        asyncio.run(_client_answering(payload).get_test_rate("CBC"))


def test_the_merge_test_prep_client_returns_the_answer_unchanged_when_it_has_the_right_shape():
    """KCD-382: agent.tools_client.ClinicToolsClient.merge_test_prep() is a thin, validated pass-through to
    clinic-api's merge endpoint -- it must not drop or alter the escalation/incomplete-data keys."""
    body = {
        "found": True,
        "test_names": ["USG Whole Abdomen"],
        "fasting_hours": None,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "escalate_to_human": True,
        "incomplete_data": ["USG Whole Abdomen"],
        "not_found": [],
    }
    assert asyncio.run(_client_answering(body).merge_test_prep(["USG Whole Abdomen"])) == body


def test_the_merge_test_prep_client_turns_a_malformed_answer_into_a_tool_failure_never_a_fact():
    with pytest.raises(ToolCallError):
        asyncio.run(_client_answering({"escalate_to_human": "yes"}).merge_test_prep(["CBC", "Lipid Profile"]))


def test_every_write_the_client_makes_is_checked_too():
    client = _client_answering({"nope": True})
    for call in (
        lambda: client.hold_slot("Sen", "2026-10-05", "10:00"),
        lambda: client.cancel_appointment("KCD-1"),
        lambda: client.lookup_bookings(phone="9830011111"),
        lambda: client.booking_conflict("9830011111", "2026-10-05", "10:00"),
        lambda: client.route_department("chest pain"),
        lambda: client.identify_patient("9830011111"),
        lambda: client.verify_patient("c1", 1, {"dob": "1985-06-30"}),
    ):
        with pytest.raises(ToolCallError):
            asyncio.run(call())
