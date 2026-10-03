"""Epic E29 (KCD-434): a romanised spelling of a Bengali/Hindi test alias
resolves to the correct catalogue row, at the HTTP layer.

    python -m pytest tests/test_test_matching_e29.py -v
"""

import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def clinic_client():
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
        "phonetic_match",
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


def test_romanised_spelling_of_a_bengali_alias_is_suggested_never_resolved(clinic_client):
    # SPEC CHANGE (OpenAI review): a sound-alike used to resolve straight to a test and its price.
    # "CBC" and "CRP" are one phoneme apart, so it is now a suggestion the caller must confirm.
    # Seeded: Complete Blood Count (CBC), Bengali alias "সিবিসি" (sibisi).
    # A caller who types/says the Latin transliteration ("sibisi") rather
    # than the native script must still reach the same row.
    result = clinic_client.get("/api/v1/tests/search", params={"name": "sibisi"}).json()
    assert result["found"] is False
    assert "Complete Blood Count (CBC)" in result["did_you_mean"]
    assert "price" not in result and "test_name" not in result


def test_short_romanised_query_does_not_produce_a_false_match(clinic_client):
    # Guards the length floor added alongside the phonetic fallback: a
    # short, low-information query must not collide with an unrelated
    # test just because its folded key happens to be short too.
    result = clinic_client.get("/api/v1/tests/search", params={"name": "xz"}).json()
    assert result["found"] is False
