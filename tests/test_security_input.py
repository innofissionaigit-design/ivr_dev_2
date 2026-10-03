"""Reading what a caller SAYS for the security questions (agent/security_input.py) and the question
conversation (agent/security_check.py). Parsing only: whether an answer is RIGHT is decided by the server
(tests/test_e33_registry.py).

    python -m pytest tests/test_security_input.py -v
"""

import datetime
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import security_input as si
from agent.security_check import MAX_PARSE_FAILURES, SecurityCheck

TODAY = datetime.date(2026, 9, 24)


def dob(text):
    d = si.parse_dob(text, TODAY)
    return None if d is None else (d.year, d.month, d.day)


# Hypothesis (tests/test_nbest_rescore.py's fuzzing, which reaches this module through the number parser) found that
# str.isdigit() is true for some Unicode "digits" int() cannot parse (superscripts, circled numbers, ...); a real ASR
# never emits one (agent/security_input.py's own _norm() already turns every digit it recognises into plain "0"-"9"
# first), but the parser must not crash on it either.
@pytest.mark.parametrize("token", ["²", "②"])  # superscript 2, circled 2: isdigit() true, int() raises
def test_a_unicode_digit_int_cannot_parse_is_not_a_number_not_a_crash(token):
    assert si._value_of(token) is None
    assert si._to_numbers([token]) == [token]


# ------------------------------------------------------------------------------ dates of birth


@pytest.mark.parametrize(
    "text,expected",
    [
        ("12 May 1980", (1980, 5, 12)),
        ("12/05/1980", (1980, 5, 12)),
        ("12-5-1980", (1980, 5, 12)),
        ("5-12-1980", (1980, 12, 5)),  # day first: the 5th of December
        ("May 12, 1980", (1980, 5, 12)),
        ("12th of May 1980", (1980, 5, 12)),
        ("1st january 2001", (2001, 1, 1)),
        ("twelfth of May nineteen eighty", (1980, 5, 12)),
        ("fourteen march nineteen forty eight", (1948, 3, 14)),
        ("the second of november two thousand ten", (2010, 11, 2)),
        ("twenty one november two thousand ten", (2010, 11, 21)),
        ("my date of birth is 3 august 1962", (1962, 8, 3)),
    ],
)
def test_english_dates(text, expected):
    assert dob(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("১২/০৫/১৯৮০", (1980, 5, 12)),
        ("১২ মে ১৯৮০", (1980, 5, 12)),
        ("বারো মে উনিশশো আশি", (1980, 5, 12)),
        ("চোদ্দো মার্চ উনিশশো আটচল্লিশ", (1948, 3, 14)),
        ("একুশ নভেম্বর দুই হাজার দশ", (2010, 11, 21)),
        ("ছয় ফেব্রুয়ারি উনিশশো পঁচাত্তর", (1975, 2, 6)),
    ],
)
def test_bengali_dates(text, expected):
    assert dob(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("१२ मई १९८०", (1980, 5, 12)),
        ("12 मई 1980", (1980, 5, 12)),
        ("बारह मई उन्नीस सौ अस्सी", (1980, 5, 12)),
        ("चौदह मार्च उन्नीस सौ अड़तालीस", (1948, 3, 14)),
        ("इक्कीस नवंबर दो हज़ार दस", (2010, 11, 21)),
    ],
)
def test_hindi_dates(text, expected):
    assert dob(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "",
        "12 May",
        "May 1980",
        "12 May 80",  # no year, or a two-digit year: 80 could be 1980 or 2080
        "31 february 1980",
        "30 february 1980",
        "12 May 2090",  # not a real date / in the future
        "12 13 1980",
        "15 May 1850",  # month 13 / before 1900
    ],
)
def test_a_date_that_is_not_all_there_is_refused_never_guessed(text):
    assert dob(text) is None


# ------------------------------------------------------------------------------ patient id


@pytest.mark.parametrize(
    "text,expected",
    [
        ("KCP-100001", "KCP100001"),
        ("kcp 100001", "KCP100001"),
        ("my id is KCP 100002", "KCP100002"),
        ("K C P one zero zero zero zero one", "KCP100001"),
        ("100001", "100001"),
    ],
)
def test_patient_ids(text, expected):
    assert si.parse_patient_id(text) == expected


@pytest.mark.parametrize("text", ["hello", "", "one two", "my number is nine"])
def test_something_that_is_not_an_id_is_refused(text):
    assert si.parse_patient_id(text) is None


def test_a_name_lead_in_is_tidied_and_an_address_keeps_its_words():
    assert si.clean_name("my name is Asha Saha") == "asha saha"
    assert si.clean_name("আমার নাম আশা সাহা") == "আশা সাহা"
    assert "lake" in si.address_text("Lake Town, Block A 700089") and "700089" in si.address_text(
        "Lake Town, Block A 700089"
    )


# ------------------------------------------------------------------------------ the conversation


def test_the_first_question_asks_for_a_strong_fact_with_an_introduction_once():
    chk = SecurityCheck("7")
    q1 = chk.next_question("en")
    assert q1.startswith("For your safety") and "date of birth" in q1 and "patient ID" in q1
    assert chk.expecting == ["dob", "patient_id"]
    assert not chk.next_question("en").startswith("For your safety")


def test_either_strong_fact_satisfies_the_first_question_then_the_name_is_asked():
    a = SecurityCheck("7")
    a.next_question("en")
    assert a.absorb("12 May 1980", TODAY) and a.answers == {"dob": "1980-05-12"}
    assert "full name" in a.next_question("en")
    b = SecurityCheck("7")
    b.next_question("en")
    assert b.absorb("KCP-100001") and b.answers == {"patient_id": "KCP100001"}


def test_it_submits_only_when_two_facts_including_a_strong_one_are_in():
    c = SecurityCheck("7")
    c.next_question("en")
    c.absorb("12 May 1980", TODAY)
    assert not c.ready_to_submit()
    c.next_question("en")
    c.absorb("Asha Saha")
    assert c.ready_to_submit() and c.payload() == {"dob": "1980-05-12", "name": "asha saha"}


def test_a_single_word_is_not_taken_as_a_full_name_or_an_address():
    c = SecurityCheck("7")
    c.answers["dob"] = "1980-05-12"
    c.next_question("en")
    assert c.expecting == ["name"] and not c.absorb("Asha")
    c.next_question("en")
    assert c.parse_failures == 1


def test_it_gives_up_reading_after_repeated_failures():
    c = SecurityCheck("7")
    c.next_question("en")
    for _ in range(MAX_PARSE_FAILURES + 1):
        c.absorb("mumble mumble")
    assert c.gave_up_reading


def test_after_a_failed_pair_it_asks_for_a_further_fact_then_a_repeat_and_stops_at_three():
    c = SecurityCheck("7")
    c.next_question("en")
    c.answers.update({"dob": "1980-05-12", "name": "asha saha"})
    c.record_result({"verified": False, "attempts_left": 2})
    assert "address" in c.next_question("en").lower()
    c.answers["address"] = "gariahat ballygunge"
    c.record_result({"verified": False, "attempts_left": 1})
    assert not c.finished
    assert "patient id" in c.next_question("en").lower()
    c.record_result({"verified": False, "attempts_left": 0, "locked": True})
    assert c.finished and not c.verified


def test_a_pass_records_the_age_and_seniority_the_server_computed():
    c = SecurityCheck("7")
    c.record_result({"verified": True, "age_years": 78, "is_senior": True, "attempts_left": 2})
    assert c.verified and c.finished and c.age_years == 78 and c.is_senior


def test_find_mode_locates_by_id_or_by_dob_with_name_and_stops_after_two_misses():
    c = SecurityCheck(None)
    assert c.finding
    c.answers.update({"dob": "1980-05-12", "name": "x y"})
    assert c.find_payload() == {"dob": "1980-05-12", "name": "x y"}
    assert c.record_find_miss() is False and "dob" not in c.answers
    c.answers.update({"dob": "1980-05-13", "name": "x y"})
    assert c.record_find_miss() is True
    c2 = SecurityCheck(None)
    c2.answers["patient_id"] = "KCP100001"
    assert c2.find_payload() == {"patient_id": "KCP100001"}


def test_every_question_exists_in_all_three_languages_and_is_short():
    from agent import security_check as sc

    for table in (
        sc.INTRO,
        sc.DID_NOT_UNDERSTAND,
        sc.NOT_MATCHED,
        sc.VERIFIED,
        sc.FAILED,
        sc.FIND_BY_DETAILS,
        *sc.QUESTION.values(),
    ):
        assert set(table) == {"bn", "hi", "en"}
        for lang, text in table.items():
            assert all(
                len(sentence.split()) <= 14
                for sentence in text.replace("?", ".").split("।")
                for sentence in sentence.split(".")
            )
