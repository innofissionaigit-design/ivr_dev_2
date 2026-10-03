"""KCD-096: the phonetic-fold gazetteer with an index. Pure and offline except for the two tests that use the real
seeded clinic API.

    python -m pytest tests/test_gazetteer.py -v

Every number here is on SYNTHETIC input (tools/gazetteer_eval.py says how it is generated); these tests guard the
behaviour and the direction of the comparison, they do not measure real callers.
"""

import os
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "tools"), os.path.join(REPO_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.append(_p)

import gazetteer_eval as ev  # noqa: E402

from agent.gazetteer import Gazetteer, normalise, vowelled_key  # noqa: E402

TITLES = frozenset(ev.TITLES)


def _doctors():
    rows = {
        "Dr. S. Mukherjee": ["Mukherjee", "মুখার্জী", "मुखर्जी"],
        "Dr. A. Sen": ["Sen", "সেন", "सेन"],
        "Dr. K. Bhattacharya": ["Bhattacharya", "ভট্টাচার্য"],
        "Dr. R. Chowdhury": ["Chowdhury", "চৌধুরী"],
        "Dr. N. Roy": ["Roy", "রায়"],
        "Dr. P. Ray": ["Ray"],
        "Dr. A. Kar": ["Kar"],
        "Dr. D. Das": ["Das", "দাস"],
    }
    return [(n, f) for n, forms in rows.items() for f in forms]


@pytest.fixture(scope="module")
def doctors():
    return Gazetteer(_doctors(), drop_words=TITLES)


def _names(g, query, **kw):
    return [s.canonical for s in g.suggest(query, **kw)]


# ============================================================================ the tiers


def test_the_written_form_is_an_exact_hit(doctors):
    top = doctors.suggest("Dr. Mukherjee")[0]
    assert (top.canonical, top.basis) == ("Dr. S. Mukherjee", "exact")


def test_a_mispronounced_surname_is_found_by_sound_not_by_spelling(doctors):
    top = doctors.suggest("Mukharji")[0]
    assert top.canonical == "Dr. S. Mukherjee" and top.basis == "sound"


def test_a_dropped_consonant_is_found_by_the_one_edit_tier(doctors):
    top = doctors.suggest("Bhatacharjo")[0]
    assert top.canonical == "Dr. K. Bhattacharya" and top.basis == "sound_near"


def test_short_surnames_are_matched_by_sound_too(doctors):
    """Sen/Das/Roy fold to one or two consonant classes, which the old rule refused to trust at all."""
    assert _names(doctors, "Shen")[0] == "Dr. A. Sen"
    assert _names(doctors, "Sain")[0] == "Dr. A. Sen"
    assert _names(doctors, "Dass")[0] == "Dr. D. Das"
    assert _names(doctors, "ষেন")[0] == "Dr. A. Sen"  # Bengali sibilant confusion, same script


def test_vowel_quality_still_separates_short_names(doctors):
    """Sen is not Son: the consonants alone would call them equal."""
    assert vowelled_key("sen") != vowelled_key("son")
    assert doctors.suggest("Son")[0].basis == "spelling"  # offered on spelling, never as a sound match


def test_a_name_in_another_script_matches_when_the_skeleton_is_long_enough(doctors):
    assert _names(doctors, "Chowdhary")[0] == "Dr. R. Chowdhury"
    g = Gazetteer([("Dr. K. Bhattacharya", "ভট্টাচার্য")])
    assert [s.canonical for s in g.suggest("Bhattacharya")] == ["Dr. K. Bhattacharya"]


def test_ai_and_ay_are_one_vowel():
    assert vowelled_key("sain") == vowelled_key("sen") == vowelled_key("sayn")


# ======================================================================= the "Doctor Nobody" class of bug


@pytest.mark.parametrize("query", ["Nobody", "Doctor Nobody", "Dr Nobody", "Xyz", "Qwerty", "hello", "", "   ", "123"])
def test_a_nonsense_name_suggests_no_real_doctor(doctors, query):
    assert doctors.suggest(query) == []


def test_a_sound_alike_is_only_ever_a_suggestion_object_never_a_resolution(doctors):
    """The return type carries no 'resolved' flag on purpose: callers get names to read back."""
    s = doctors.suggest("Mukharji")[0]
    assert set(s.__dataclass_fields__) == {"canonical", "form", "score", "basis"}


def test_one_consonant_off_on_a_short_name_is_not_a_sound_match():
    """Kar vs Ray fold to different first consonants and must stay apart."""
    g = Gazetteer([("Dr. A. Kar", "Kar"), ("Dr. P. Ray", "Ray")])
    assert [s.canonical for s in g.suggest("Ray")] == ["Dr. P. Ray"]
    assert [s.canonical for s in g.suggest("Kar")] == ["Dr. A. Kar"]


def test_a_test_code_is_not_matched_by_any_four_letter_word():
    """ESR/TSH/CRP scored 0.57 against ordinary words on the plain ratio; the short-form floor stops that."""
    g = Gazetteer([("ESR", "ESR"), ("TSH", "TSH"), ("CRP", "CRP")])
    assert g.suggest("Iyer") == [] and g.suggest("Singh") == [] and g.suggest("Blorp") == []
    assert [s.canonical for s in g.suggest("esr")] == ["ESR"]


# ================================================================================== the list is short and honest


def test_two_names_that_really_tie_come_back_together():
    g = Gazetteer([("Dr. A. Sen", "Sen"), ("Dr. P. Sengupta", "Sengupta"), ("Dr. D. Das", "Das")])
    names = _names(g, "Senn")
    assert "Dr. A. Sen" in names and "Dr. D. Das" not in names


def test_weak_spelling_neighbours_do_not_pad_a_confident_answer(doctors):
    assert _names(doctors, "Mukharji") == ["Dr. S. Mukherjee"]


def test_each_entity_is_listed_once_however_many_of_its_forms_match(doctors):
    names = _names(doctors, "Mukherjee", limit=5)
    assert names.count("Dr. S. Mukherjee") == 1


def test_neighbours_is_untruncated_and_carries_the_similarity(doctors):
    got = {c: r for c, _f, r in doctors.neighbours("Mukharji")}
    assert got["Dr. S. Mukherjee"] > 0.6
    assert len(got) >= 1


def test_normalise_keeps_indic_combining_marks_and_drops_titles():
    assert normalise("Dr. Sen", TITLES) == "sen"
    assert normalise("ডাক্তার সেন", TITLES) == "সেন"
    # the Devanagari nukta is written both ways by the ASR
    assert normalise("मुखर्जी") == normalise("मुखर्जी")
    assert normalise("ज़हर") == normalise("जहर")


def test_flap_letters_fold_to_the_same_form_in_either_normalisation():
    decomposed = "ব" + chr(0x09A1) + chr(0x09BC) + "ুয়া"
    precomposed = "ব" + chr(0x09DC) + "ুয়া"
    assert normalise(decomposed) == normalise(precomposed)


def test_an_empty_gazetteer_and_duplicate_entries_are_harmless():
    assert Gazetteer([]).suggest("anything") == []
    g = Gazetteer([("A", "Sen"), ("A", "sen"), ("A", "SEN")])
    assert len(g) == 1


# ============================================================================================ an index, not a scan


def test_a_lookup_scores_a_bounded_shortlist_not_every_row():
    entries = ev.synthetic_catalogue(6000)
    g = Gazetteer(entries)
    import random

    rng = random.Random(3)
    for canonical, form in rng.sample(entries, 25):
        vs = [v for vv in ev.variants(form, rng).values() for v in vv]
        g.suggest(rng.choice(vs) if vs else form)
    per_query = g.stats["scored"] / g.stats["queries"]
    assert per_query < 0.05 * len(g), f"{per_query:.0f} scored per query of {len(g)} forms is not an index"


def test_the_answer_for_a_planted_mispronunciation_survives_catalogue_growth():
    """The same query returns the same right answer at 100 forms and at 8000: growth does not push it out."""
    small = ev.synthetic_catalogue(100)
    big = ev.synthetic_catalogue(8000)
    target_canonical, target_form = small[17]
    import random

    for entries in (small, big):
        g = Gazetteer(entries)
        query = ev.variants(target_form, random.Random(5))["in_class"] or [target_form]
        got = [s.canonical for s in g.suggest(query[0], limit=3)]
        assert target_canonical in got


def test_match_latency_stays_inside_the_fast_path_budget_at_ten_thousand_forms():
    """Blueprint 4.3: 150 ms for the fast tier. Measured about 10-20 ms here; asserted at a third of the budget
    so a slow CI machine does not make this a flaky test, and a scan (about 230 ms) could never pass it."""
    entries = ev.synthetic_catalogue(10000)
    g = Gazetteer(entries)
    import random

    rng = random.Random(9)
    queries = []
    for _c, f in rng.sample(entries, 40):
        vs = [v for vv in ev.variants(f, rng).values() for v in vv]
        queries.append(rng.choice(vs) if vs else f)
    best_p95 = []
    for _attempt in range(3):
        times = []
        for q in queries:
            t = time.perf_counter()
            g.suggest(q)
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        best_p95.append(times[int(len(times) * 0.95) - 1])
    assert min(best_p95) < 50.0, best_p95


# ======================================================================== the F-score comparison (synthetic)


@pytest.fixture(scope="module")
def catalogue():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        return client.get("/api/v1/catalogue").json()


@pytest.mark.parametrize("which", ["doctors", "tests"])
def test_entity_f_score_improves_in_the_mispronunciation_bucket(catalogue, which):
    doctors, tests = ev.entries_from_catalogue(catalogue)
    report = ev.evaluate(doctors if which == "doctors" else tests)
    new, old = report["mispronunciation_bucket"]["new"], report["mispronunciation_bucket"]["old"]
    assert new["f1"] >= old["f1"] + 0.10, (which, new, old)
    assert new["recall"] >= old["recall"], (which, new, old)


@pytest.mark.parametrize("which", ["doctors", "tests"])
def test_like_for_like_with_one_suggestion_each_it_is_no_worse(catalogue, which):
    """The comparison that does not depend on how many names each method is willing to list."""
    doctors, tests = ev.entries_from_catalogue(catalogue)
    report = ev.evaluate(doctors if which == "doctors" else tests)
    assert report["top1"]["new"]["f1"] >= report["top1"]["old"]["f1"], report["top1"]


@pytest.mark.parametrize("which", ["doctors", "tests"])
def test_no_family_is_left_worse_than_the_scan(catalogue, which):
    """Recall is not bought by giving up a family, including the one the fold does not model."""
    doctors, tests = ev.entries_from_catalogue(catalogue)
    report = ev.evaluate(doctors if which == "doctors" else tests)
    for fam, r in report["families"].items():
        if fam == "negative":
            continue
        assert r["new"]["f1"] >= r["old"]["f1"], (which, fam, r)


@pytest.mark.parametrize("which", ["doctors", "tests"])
def test_nonsense_names_get_no_more_suggestions_than_the_scan_gave(catalogue, which):
    doctors, tests = ev.entries_from_catalogue(catalogue)
    neg = ev.evaluate(doctors if which == "doctors" else tests)["negatives"]
    assert neg["new_false_suggestions"] <= neg["old_false_suggestions"], neg


def test_the_documented_regression_names_never_suggest_a_real_doctor(catalogue):
    doctors, _tests = ev.entries_from_catalogue(catalogue)
    g = Gazetteer(doctors, drop_words=TITLES)
    for q in ("Doctor Nobody", "Nobody", "Xyz"):
        assert g.suggest(q) == [], q


# ==================================================================== clinic-api uses it (real seeded API)


@pytest.fixture
def clinic():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        yield client


def test_the_api_reads_back_a_mispronounced_doctor_as_a_suggestion_to_confirm(clinic):
    body = clinic.get("/api/v1/doctors/availability", params={"name": "Mukharji"}).json()
    assert body["found"] is False and body["did_you_mean"] == ["Dr. S. Mukherjee"]
    assert body["needs_confirmation"] is True
    assert body["did_you_mean_bn"] and body["did_you_mean_hi"]


def test_the_api_matches_a_short_surname_by_sound_but_never_resolves_it(clinic):
    body = clinic.get("/api/v1/doctors/availability", params={"name": "Shen"}).json()
    assert body["found"] is False and "Dr. A. Sen" in body["did_you_mean"]


def test_the_api_gives_nothing_for_doctor_nobody(clinic):
    body = clinic.get("/api/v1/doctors/availability", params={"name": "Doctor Nobody"}).json()
    assert body["found"] is False and body["did_you_mean"] == []


def test_the_api_suggests_a_test_for_a_mispronounced_name(clinic):
    body = clinic.get("/api/v1/tests/search", params={"name": "Lipit Profeel"}).json()
    assert body["found"] is False and "Lipid Profile" in body["did_you_mean"]


def test_the_api_finds_the_bracketed_code_of_a_test_by_sound(clinic):
    body = clinic.get("/api/v1/tests/search", params={"name": "sibisi"}).json()
    assert body.get("found") or "Complete Blood Count (CBC)" in body.get("did_you_mean", [])


def test_a_new_row_is_suggested_at_once_and_an_alias_edit_after_invalidation(clinic):
    """The built index is reused, but a change in the number of rows rebuilds it immediately."""
    main = sys.modules["main"]
    from db import SessionLocal
    from models import Doctor

    before = clinic.get("/api/v1/doctors/availability", params={"name": "Bandopadhyay"}).json()
    assert "Dr. Z. Bandopadhyay" not in before.get("did_you_mean", [])
    db = SessionLocal()
    try:
        db.add(
            Doctor(
                name="Dr. Z. Bandopadhyay", qualifications="MBBS", department_id=1, aliases_bn="ব্যানার্জী", aliases_hi=""
            )
        )
        db.commit()
    finally:
        db.close()
    after = clinic.get("/api/v1/doctors/availability", params={"name": "Bandopadhyay"}).json()
    assert after.get("found") or "Dr. Z. Bandopadhyay" in after.get("did_you_mean", [])
    main.invalidate_gazetteers()
    assert main._gazetteers == {}


# ========================================================================================== drift guard


def test_the_two_copies_of_the_gazetteer_are_byte_identical():
    a = open(os.path.join(REPO_ROOT, "agent", "gazetteer.py"), "rb").read()
    b = open(os.path.join(REPO_ROOT, "clinic-api", "gazetteer.py"), "rb").read()
    assert a == b, "clinic-api/gazetteer.py must be a copy of agent/gazetteer.py"


def test_the_two_copies_of_the_fold_agree_on_every_form_in_the_catalogue(catalogue):
    """gazetteer.py imports whichever phonetic_match is next to it; both must fold the same way."""
    import importlib.util

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    agent_fold = load(os.path.join(REPO_ROOT, "agent", "phonetic_match.py"), "pm_agent")
    clinic_fold = load(os.path.join(REPO_ROOT, "clinic-api", "phonetic_match.py"), "pm_clinic")
    doctors, tests = ev.entries_from_catalogue(catalogue)
    for _c, form in doctors + tests:
        assert agent_fold.phonetic_key(form) == clinic_fold.phonetic_key(form), form
