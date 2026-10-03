"""agent/doctor_list.py and the two endpoints added for it.

Three things are worth testing here and they are quite different from each other:

  1. **The detector, in every script.** This is the one that went wrong during
     the port: a "was a doctor named?" regex suppressed 8 of 15 genuine requests,
     all of them non-English, because in Bengali and Hindi the words after the
     title are ordinary grammar ("ডাক্তার আছেন", "डॉक्टर हैं") rather than a
     surname. The regex is gone and the phrase lists carry the whole job, so the
     false-negative cases below are the regression guard for that mistake, and
     the quiet cases prove the lists are specific enough to replace it.

  2. **The endpoints, against a real seeded database.** Including the
     no-guessing behaviour: a department that half-matches is ASKED about, and
     two that match equally are `ambiguous`, never resolved to the first row.
     This is the "Doctor Nobody" rule applied to departments.

  3. **The replies, in all three languages.** With 32 seeded doctors, reading
     every name down a phone line is not an answer; the full-list reply names the
     DEPARTMENTS instead. And a Bengali or Hindi voice silently drops Latin
     script, so each language's reply must use that language's own name column.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

from agent.doctor_list import asks_for_the_doctor_list, detect_doctor_list_request
from agent.reply_templates import MAX_DOCTORS_SPOKEN, doctor_list_reply, doctors_by_department_reply

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")
LANGS = ("bn", "hi", "en")


@pytest.fixture()
def clinic_client():
    """A fresh clinic-api app + freshly seeded SQLite file, as
    tests/test_clinic_api_new_endpoints.py does it."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("main", "db", "models", "seed", "booking_service", "booking_migrate", "enquiry_migrate"):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient

    import main as clinic_main  # noqa: PLC0415

    with TestClient(clinic_main.app) as client:
        yield client
    try:
        os.remove(db_path)
    except OSError:
        pass


# ================================================================ 1. detector
@pytest.mark.parametrize(
    "text,family",
    [
        ("which doctors do you have", "en"),
        ("who are your doctors", "en"),
        ("list of doctors please", "en"),
        ("which departments do you have", "en"),
        ("ke ke doctor achen", "latin"),
        ("kaun kaun se doctor hain", "latin"),
        ("doctor ki list dijiye", "latin"),
        ("কে কে ডাক্তার আছেন", "bn"),
        ("সব ডাক্তারের নাম বলুন", "bn"),
        ("ডাক্তারদের তালিকা দিন", "bn"),
        ("কোন কোন বিভাগ আছে", "bn"),
        ("कौन कौन से डॉक्टर हैं", "hi"),
        ("डॉक्टरों की सूची दीजिए", "hi"),
        ("कौन से विभाग हैं", "hi"),
    ],
)
def test_a_plural_unnamed_ask_for_doctors_is_caught(text, family):
    matched, tag = detect_doctor_list_request(text)
    assert matched is True
    assert tag == family
    assert asks_for_the_doctor_list(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "কে কে ডাক্তার আছেন",
        "ডাক্তারদের তালিকা দিন",
        "कौन कौन से डॉक्टर हैं",
        "डॉक्टरों की सूची दीजिए",
        "ke ke doctor achen",
        "kaun kaun se doctor hain",
    ],
)
def test_a_non_english_request_is_not_mistaken_for_naming_a_doctor(text):
    """THE regression guard for the bug this port made and fixed. An earlier
    "was a doctor named?" regex matched the ordinary words that follow a title in
    Bengali and Hindi, and silently suppressed exactly these requests -- the
    trilingual failure mode this whole merge exists to avoid."""
    assert asks_for_the_doctor_list(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        # a NAMED doctor: availability/schedule/booking, all with their own paths
        "is Dr Sen available tomorrow",
        "which day is Dr Sen in",
        "I want to book with Dr Sen",
        "ডক্টর সেন কি আজ আছেন",
        "ডাক্তার সেনের কাছে অ্যাপয়েন্টমেন্ট চাই",
        "डॉक्टर सेन कल हैं क्या",
        "डॉक्टर सेन के साथ अपॉइंटमेंट",
        "Dr Sen er appointment chai",
        # unrelated
        "what is the price of a CBC test",
        "I want to speak to a doctor",
        "which test is cheaper",
    ],
)
def test_a_named_doctor_or_unrelated_turn_is_left_alone(text):
    """The phrase lists pair a plural/interrogative sense with a doctor word, so
    a question about ONE named doctor matches none of them. This is what makes
    the removed regex unnecessary."""
    assert asks_for_the_doctor_list(text) is False


@pytest.mark.parametrize("lang", LANGS)
def test_every_language_has_its_own_phrase_list(lang):
    import agent.doctor_list as m

    table = {"bn": m._BN, "hi": m._HI, "en": m._EN}[lang]
    assert len(table) >= 10, f"{lang} list is too thin to be real coverage"


# =============================================================== 2. endpoints
def test_the_department_list_endpoint_reports_every_department_with_its_headcount(clinic_client):
    body = clinic_client.get("/api/v1/departments").json()
    assert body["found"] is True
    assert len(body["departments"]) >= 2
    assert all(row["doctor_count"] >= 1 for row in body["departments"])
    assert "General Medicine" in {row["name"] for row in body["departments"]}


def test_the_all_doctors_endpoint_returns_speakable_names_for_every_language(clinic_client):
    """A Bengali or Hindi voice silently drops Latin script, so every row has to
    carry a name in each script -- otherwise the caller hears a sentence with the
    name missing from it."""
    body = clinic_client.get("/api/v1/doctors").json()
    assert body["found"] is True and body["count"] >= 8
    for doc in body["doctors"]:
        assert doc["name"] and doc["department"]
        assert doc["full_name_bn"] or doc["alias_bn"], f"{doc['name']} has no Bengali form"
        assert doc["full_name_hi"] or doc["alias_hi"], f"{doc['name']} has no Hindi form"


def test_an_exact_department_name_resolves_to_its_own_doctors(clinic_client):
    body = clinic_client.get("/api/v1/doctors/by-department", params={"department": "Cardiology"}).json()
    assert body["found"] is True
    assert body["department"] == "Cardiology"
    assert body["count"] >= 1
    assert all(d["department"] == "Cardiology" for d in body["doctors"])


def test_a_whole_sentence_resolves_on_the_department_name_inside_it(clinic_client):
    """Why no department-name extractor was ported: the endpoint matches in both
    directions, so the agent can hand over the raw utterance."""
    body = clinic_client.get(
        "/api/v1/doctors/by-department", params={"department": "which doctors are in cardiology"}
    ).json()
    assert body["found"] is True and body["department"] == "Cardiology"


def test_a_short_form_resolves_on_being_inside_the_department_name(clinic_client):
    body = clinic_client.get("/api/v1/doctors/by-department", params={"department": "cardio"}).json()
    assert body["found"] is True and body["department"] == "Cardiology"


def test_a_department_that_does_not_exist_is_never_guessed_at(clinic_client):
    """The "Doctor Nobody" rule, applied to departments: a name that matches
    nothing resolves to nothing, and the reply asks."""
    body = clinic_client.get("/api/v1/doctors/by-department", params={"department": "Astrology"}).json()
    assert body["found"] is False
    assert "doctors" not in body


def test_a_general_question_names_no_department_so_the_full_list_is_served(clinic_client):
    """"which doctors do you have" must NOT resolve to a department -- that is
    what makes the orchestrator fall through to the full list."""
    body = clinic_client.get(
        "/api/v1/doctors/by-department", params={"department": "which doctors do you have"}
    ).json()
    assert body["found"] is False


def test_an_empty_department_is_refused_rather_than_matching_everything(clinic_client):
    body = clinic_client.get("/api/v1/doctors/by-department", params={"department": "  "}).json()
    assert body["found"] is False


# ================================================================= 3. replies
FEW = {
    "found": True,
    "count": 2,
    "department": "Cardiology",
    "doctors": [
        {
            "name": "Dr. A. Sen",
            "full_name": "Dr. Arup Sen",
            "full_name_bn": "ডক্টর অরূপ সেন",
            "full_name_hi": "डॉक्टर अरूप सेन",
            "department": "Cardiology",
        },
        {
            "name": "Dr. N. Roy",
            "full_name": "Dr. Nita Roy",
            "full_name_bn": "ডক্টর নীতা রায়",
            "full_name_hi": "डॉक्टर नीता रॉय",
            "department": "Cardiology",
        },
    ],
}
MANY = {
    "found": True,
    "count": MAX_DOCTORS_SPOKEN + 4,
    "doctors": [
        {
            "name": f"Dr. X{i}",
            "full_name": f"Dr. X{i}",
            "full_name_bn": f"ডক্টর এক্স{i}",
            "full_name_hi": f"डॉक्टर एक्स{i}",
            "department": ["Cardiology", "Orthopaedics", "ENT"][i % 3],
        }
        for i in range(MAX_DOCTORS_SPOKEN + 4)
    ],
}


@pytest.mark.parametrize("lang,expected", [("bn", "ডক্টর অরূপ সেন"), ("hi", "डॉक्टर अरूप सेन"), ("en", "Dr. Arup Sen")])
def test_each_language_speaks_the_name_in_its_own_script(lang, expected):
    """The reply must never hand a Bengali or Hindi voice a Latin name: that
    voice drops it silently and the caller hears a gap where the name was."""
    assert expected in doctor_list_reply(FEW, lang)
    assert expected in doctors_by_department_reply(FEW, lang)


@pytest.mark.parametrize("lang", LANGS)
def test_a_long_list_names_the_departments_instead_of_reciting_every_doctor(lang):
    """32 doctors are seeded. Reading them all down a phone line is a recital
    nobody can hold in their head, so the full-list reply switches to departments
    and asks which -- one question, as the delivery policy requires."""
    said = doctor_list_reply(MANY, lang)
    assert "Orthopaedics" in said and "ENT" in said
    assert "X1" not in said, "individual doctor names were recited for a long list"
    assert said.count("?") == 1


@pytest.mark.parametrize("lang", LANGS)
def test_a_short_list_reads_the_doctors_and_asks_one_question(lang):
    said = doctor_list_reply(FEW, lang)
    assert said.count("?") == 1


@pytest.mark.parametrize("lang", LANGS)
def test_an_unresolved_department_is_asked_about_never_answered(lang):
    near = {"found": False, "query": "cardio", "did_you_mean": ["Cardiology"]}
    said = doctors_by_department_reply(near, lang)
    assert "Cardiology" in said and said.count("?") == 1

    ambiguous = {"found": False, "query": "ortho", "ambiguous": True, "did_you_mean": ["Orthopaedics", "Orthodontics"]}
    said2 = doctors_by_department_reply(ambiguous, lang)
    assert "Orthopaedics" in said2 and "Orthodontics" in said2 and said2.count("?") == 1


@pytest.mark.parametrize("lang", LANGS)
def test_a_tool_failure_says_so_and_never_invents_a_roster(lang):
    """A not-found reply says it cannot see the list and names NOBODY.

    Asserted against the real names from the fixtures above rather than against
    the word "doctor": the fallback line legitimately contains that word in
    every language ("I cannot see the list of doctors", "डॉक्टरों की सूची"), and
    an assertion on the word alone fails for the wrong reason while catching
    nothing. What must never appear is a NAME.
    """
    said = doctor_list_reply({"found": False}, lang)
    assert said.strip()
    names = [
        n
        for doc in FEW["doctors"] + MANY["doctors"]
        for n in (doc["name"], doc["full_name"], doc["full_name_bn"], doc["full_name_hi"])
    ]
    for name in names:
        assert name not in said, f"a doctor name appeared in a not-found reply: {name!r}"
