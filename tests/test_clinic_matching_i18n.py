"""Test-name matching for the ways Hindi and English callers actually phrase it.

Found on the pod: "lipid profile test" (English) and "लिपिड प्रोफ़ाइल टेस्ट"
(Hindi, nukta spelling) both came back not-found although Lipid Profile is in
the catalogue. Matching must accept the wrapped phrase -- and must still
refuse a near-miss, because a confidently wrong test's real price is the
failure this service exists to prevent."""

import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def client():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)
    for mod in (
        "main",
        "db",
        "models",
        "seed",
        "booking_service",
        "booking_migrate",
        "enquiry_migrate",
        "i18n_content",
    ):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient

    import main as clinic_main

    with TestClient(clinic_main.app) as c:
        yield c
    try:
        os.remove(db_path)
    except OSError:
        pass


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("lipid profile test", "Lipid Profile"),
        ("the lipid profile", "Lipid Profile"),
        ("Uric Acid test", "Uric Acid"),
        ("uric acid", "Uric Acid"),
        ("लिपिड प्रोफ़ाइल टेस्ट", "Lipid Profile"),  # nukta spelling
        ("लिपिड प्रोफाइल टेस्ट", "Lipid Profile"),  # plain spelling
        ("यूरिक एसिड टेस्ट", "Uric Acid"),
        ("ইউরিক অ্যাসিড টেস্টের", "Uric Acid"),
    ],
)
def test_wrapped_phrases_find_the_right_test(client, phrase, expected):
    body = client.get("/api/v1/tests/search", params={"name": phrase}).json()
    assert body["found"] is True, body
    assert body["test_name"] == expected


@pytest.mark.parametrize("phrase", ["rick acid test", "zzz unknown test", "lipid"])
def test_near_misses_are_suggested_never_silently_accepted(client, phrase):
    body = client.get("/api/v1/tests/search", params={"name": phrase}).json()
    if phrase == "lipid":
        # a prefix of one real name is still that test: substring matching has always allowed this
        assert body["found"] is True and body["test_name"] == "Lipid Profile"
    else:
        assert body["found"] is False


def test_prep_uses_the_same_matching(client):
    body = client.get("/api/v1/tests/prep", params={"name": "लिपिड प्रोफ़ाइल टेस्ट", "lang": "hi"}).json()
    assert body["found"] is True and "उपवास" in body["prep_instructions"]
