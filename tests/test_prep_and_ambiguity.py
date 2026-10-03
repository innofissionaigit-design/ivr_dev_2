"""Fourth live-call findings (2026-09-25):

* "blood sugar preparation" (no "fasting", no "PP") was answered for the PP test: the fast path took the closest form
  and never asked whether another test fit as well. Now it abstains, and the model's turn reaches the clinic API,
  which answers "which one -- Blood Sugar Fasting or Blood Sugar PP?" (KCD-446).
* the preparation answer comes from the lab-test table: its text, else its fasting_required column -- a test the table
  marks as needing fasting is never told "no special preparation" because the text column was left empty.

  python -m pytest tests/test_prep_and_ambiguity.py -v
"""

import datetime
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.fast_path import AMBIGUITY_MARGIN, Catalogue, FastPath
from agent.reply_templates import merged_test_prep_reply  # noqa: E402
from agent.reply_templates import test_prep_reply as prep_reply  # (not test_*: pytest would collect it)

TODAY = datetime.date(2026, 9, 25)


@pytest.fixture(scope="module")
def fp():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        return FastPath(Catalogue(client.get("/api/v1/catalogue").json()), today=TODAY)


@pytest.mark.parametrize(
    "text", ["blood sugar preparation", "blood sugar test rate", "what is the blood sugar test price"]
)
def test_a_name_that_two_tests_fit_is_not_picked_between(fp, text):
    assert fp.resolve(text, "en") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("how much is a cbc", "cbc"),
        ("kidney function test price", "kidney function test"),
        ("lipid profile rate", "lipid profile"),
    ],
)
def test_a_name_that_is_one_tests_is_still_served_with_no_model(fp, text, expected):
    hit = fp.resolve(text, "en")
    assert hit is not None and hit.slots["test_name"] == expected


def test_the_margin_is_a_stated_number():
    assert 0.0 < AMBIGUITY_MARGIN < 0.5


def test_the_clinic_api_asks_which_one_when_two_tests_fit(fp):
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        body = client.get("/api/v1/tests/prep", params={"name": "blood sugar", "lang": "en"}).json()
    assert body["found"] is False and body["ambiguous"] is True
    assert set(body["did_you_mean"]) == {"Blood Sugar Fasting", "Blood Sugar PP"}


@pytest.mark.parametrize("lang,needle", [("bn", "উপবাস"), ("hi", "खाली पेट"), ("en", "Fasting is required")])
def test_a_test_the_table_marks_as_fasting_is_never_told_no_preparation_needed(lang, needle):
    reply = prep_reply(
        {"test_name": "Uric Acid"},
        {"found": True, "test_name": "Uric Acid", "fasting_required": True, "prep_instructions": ""},
        lang,
    )
    assert needle in reply and "No special" not in reply and "বিশেষ কোনো" not in reply and "ख़ास" not in reply


@pytest.mark.parametrize(
    "lang,needle", [("bn", "বিশেষ কোনো প্রস্তুতির প্রয়োজন নেই"), ("hi", "ख़ास तैयारी"), ("en", "No special preparation")]
)
def test_a_test_the_table_marks_as_no_fasting_with_no_text_says_so(lang, needle):
    reply = prep_reply(
        {"test_name": "X"}, {"found": True, "test_name": "X", "fasting_required": False, "prep_instructions": ""}, lang
    )
    assert needle in reply


def test_the_tables_own_text_is_spoken_when_it_has_one():
    reply = prep_reply(
        {"test_name": "X"},
        {"found": True, "test_name": "X", "fasting_required": True, "prep_instructions": "Do not eat for eight hours."},
        "en",
    )
    assert "Do not eat for eight hours." in reply


# ---------------------------------------------------------------------------------------------------- KCD-382
# merged_test_prep_reply: story "Caller has several tests with conflicting preparation" (AC2 -- one coherent
# spoken instruction, never a separate paragraph per test). These are unit tests of the template function
# alone: it is handed exactly what clinic-api/enquiry_service.py::merge_prep_instructions() would return
# (forwarded verbatim by agent.tools_client.merge_test_prep), so no clinic app or orchestrator is needed here.


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_the_merged_reply_is_a_single_sentence_not_one_per_test(lang):
    result = {
        "found": True,
        "test_names": ["Blood Sugar Fasting", "Lipid Profile"],
        "fasting_hours": 11,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "not_found": [],
    }
    reply = merged_test_prep_reply(["Blood Sugar Fasting", "Lipid Profile"], result, lang)
    assert isinstance(reply, str) and reply.count(".") <= 2 and "\n" not in reply
    assert "11" in reply  # the strictest merged number is actually spoken, not re-derived or omitted


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_the_merged_reply_names_the_strictest_water_rule(lang):
    forbids_water = merged_test_prep_reply(
        ["A", "B"],
        {
            "found": True,
            "test_names": ["A", "B"],
            "fasting_hours": 8,
            "water_allowed_while_fasting": False,
            "prescription_required": False,
            "not_found": [],
        },
        lang,
    )
    allows_water = merged_test_prep_reply(
        ["A", "B"],
        {
            "found": True,
            "test_names": ["A", "B"],
            "fasting_hours": 8,
            "water_allowed_while_fasting": True,
            "prescription_required": False,
            "not_found": [],
        },
        lang,
    )
    assert forbids_water != allows_water


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_a_merged_result_with_no_fasting_requirement_says_so_without_inventing_a_number(lang):
    result = {
        "found": True,
        "test_names": ["CBC", "Thyroid Profile"],
        "fasting_hours": None,
        "water_allowed_while_fasting": True,
        "prescription_required": False,
        "not_found": [],
    }
    reply = merged_test_prep_reply(["CBC", "Thyroid Profile"], result, lang)
    # No digit anywhere in the spoken reply -- there is no fasting window to state.
    assert not any(ch.isdigit() for ch in reply)


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_an_unresolved_test_name_is_never_guessed_into_a_reply(lang):
    """CLAUDE.md rule 1 + the story's "do not invent medical preparation rules": a name the merge endpoint
    could not resolve must make the caller re-say the names, never be silently dropped from the sentence."""
    result = {"found": True, "test_names": ["CBC"], "fasting_hours": None, "not_found": ["Not A Real Test"]}
    reply = merged_test_prep_reply(["CBC", "Not A Real Test"], result, lang)
    assert "Not A Real Test" in reply


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_nothing_resolving_is_also_never_guessed(lang):
    reply = merged_test_prep_reply(["Not A Real Test"], {"found": False}, lang)
    assert "Not A Real Test" in reply
