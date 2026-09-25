"""ADDED BY SOURAV -- "Caller describes symptoms and asks what is wrong"
story.

story title: Caller describes symptoms and asks what is wrong
user story: As a caller, I want to be routed to the right department
    without being diagnosed, so that the agent stays inside what a phone
    line can safely promise.
acceptance criteria: Symptom -> department routing is deterministic/code-
    based, never something the model invents. The agent never states or
    implies a diagnosis. The reply is explicitly framed as administrative
    routing. Existing clinical-safety handling remains intact and runs
    before symptom routing.

Structured the same way tests/test_clinical_interpretation_safety.py is
(this story's own closest sibling, per the investigation that preceded
this implementation): reply-function checks, a deterministic-resolver
adversarial suite, a "never reaches the classifier" structural proof, a
dispatch-level check with a fake tools client, a pending-flow interrupt
check, and transport parity.

STORY 6 VALIDATION CLEANUP (this revision): three new test classes cover
the three limitations fixed in agent/symptom_routing.py (see that
module's own docstring for the full reasoning) --
TestDepartmentAlignment (every mapped department is a real seeded one),
TestBengaliScriptSymptomRouting (Bengali-script phrases resolve the same
as their romanised equivalents), and TestColdFalsePositiveMatching (the
specific "cold weather"/"cold water"/"cold drinks" false-positive gap is
closed while "I have a cold" still routes). TestClinicalSafetyBoundary
adds the exact adversarial phrases from that cleanup request that must
never be converted into a symptom-routing response.
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import re
import sys
import types

import pytest

import main_pcm
from agent.symptom_routing import resolve_symptom_department, COMMIT, NONE, SYMPTOM_DEPARTMENT_MAP
from agent.clinical_safety import is_clinical_interpretation
from agent.reply_templates import symptom_routing_reply, doctors_by_department_reply

_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLINIC = ROOT / "clinic-api"


def _load(name: str):
    """clinic-api is not an importable package -- the hyphen makes it not an
    identifier -- so its pure modules are loaded by path. Same pattern
    tests/test_near_match_offers.py already uses for exactly this reason."""
    spec = importlib.util.spec_from_file_location(name, CLINIC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_literal(name: str):
    """Lift one top-level literal out of seed.py without importing it (it
    pulls in SQLAlchemy and a live database at import time) -- same helper
    tests/test_near_match_offers.py already uses."""
    source = (CLINIC / "seed.py").read_text(encoding="utf-8")
    match = re.search(name + r"\s*=\s*[\[{].*?\n[\]}]", source, re.S)
    assert match, f"{name} is gone from seed.py"
    namespace: dict = {}
    exec(match.group(0), namespace)  # noqa: S102 - our own source, read by path
    return namespace[name]


SEEDED_DEPARTMENTS = _seed_literal("DEPARTMENTS")


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- #
# agent/symptom_routing.py -- the deterministic resolver itself
# --------------------------------------------------------------------- #

class TestResolveSymptomDepartment:
    @pytest.mark.parametrize("text,expected_department", [
        ("I have a skin rash", "Dermatology"),
        ("mujhe skin pe rash hai", "Dermatology"),
        ("I have chest pain", "Cardiology"),
        ("seene mein dard ho raha hai", "Cardiology"),
        ("I have ear pain", "ENT"),
        ("gala kharab hai", "ENT"),
        ("my knee pain is bad", "Orthopaedics"),
        ("kamar mein dard hai", "Orthopaedics"),
        ("I have a fever", "General Medicine"),
        ("bukhar hai mujhe", "General Medicine"),
        # Folded into General Medicine in the Story 6 validation cleanup --
        # these symptom groups used to point at "Ophthalmology" /
        # "Gastroenterology" / "Neurology", none of which are seeded
        # departments (see agent/symptom_routing.py's own docstring,
        # section "1. DEPARTMENT ALIGNMENT"). They now resolve to the
        # clinic's actual general/undifferentiated department instead of a
        # destination that could never resolve downstream.
        ("I have blurred vision", "General Medicine"),
        ("pet mein dard hai", "General Medicine"),
        ("I have a headache", "General Medicine"),
    ])
    def test_commits_to_the_right_department(self, text, expected_department):
        verdict, department = resolve_symptom_department(text)
        assert verdict == COMMIT
        assert department == expected_department

    def test_no_symptom_phrase_is_never_intercepted(self):
        for text in [
            "which department should I go to?",
            "what doctor should I see for this?",
            "what is the rate for CBC",
            "is Dr. Sen available tomorrow",
            "can I book an appointment",
            "",
            "   ",
        ]:
            verdict, department = resolve_symptom_department(text)
            assert verdict == NONE
            assert department is None

    def test_none_and_empty_text_never_crash(self):
        assert resolve_symptom_department(None) == (NONE, None)
        assert resolve_symptom_department("") == (NONE, None)

    def test_two_different_departments_in_one_utterance_is_never_guessed(self):
        # Chest pain (Cardiology) and knee pain (Orthopaedics) named
        # together -- never picked between, per this module's own "never
        # guessed" contract.
        verdict, department = resolve_symptom_department(
            "I have chest pain and also my knee pain is bad")
        assert verdict == NONE
        assert department is None

    @pytest.mark.parametrize("text", [
        "I have chest pain, what disease do I have?",
        "I have a headache, which disease is this?",
        "mera pet mein dard hai, kaunsi bimari hai yeh?",
        "chest pain hai, diagnose me please",
    ])
    def test_a_symptom_plus_a_diagnosis_request_is_never_intercepted(self, text):
        # AC10's "symptom + request for diagnosis" case: this module must
        # decline to act so the utterance is left for the layers that
        # actually exist to handle a diagnosis request (agent/
        # clinical_safety.py, already run first in production; the
        # classifier's own "clinical_interpretation" intent as its second
        # layer) -- never quietly answered with a department listing that
        # ignores the diagnosis question entirely.
        verdict, department = resolve_symptom_department(text)
        assert verdict == NONE
        assert department is None

    def test_a_symptom_plus_danger_wording_is_left_to_clinical_safety_first(self):
        # AC10's "symptom + urgent/danger wording" case. This module WOULD
        # commit on "chest pain" alone -- proving that is the point: it is
        # main.py/main_pcm.py's _resolve_intent() GUARD ORDER (see
        # TestGuardOrderInResolveIntent below), not this function in
        # isolation, that guarantees agent/clinical_safety.py wins. Checked
        # here so a future reordering of that chain is caught structurally.
        text = "I have chest pain, am I dying?"
        assert is_clinical_interpretation(text) is True
        verdict, department = resolve_symptom_department(text)
        assert verdict == COMMIT
        assert department == "Cardiology"

    def test_every_departments_prototype_vocabulary_is_nonempty(self):
        # Sanity check on the data itself, not the logic -- catches an
        # accidental empty list surviving a future edit to the map.
        for department, phrases in SYMPTOM_DEPARTMENT_MAP.items():
            assert phrases, f"{department} has no vocabulary entries"


# --------------------------------------------------------------------- #
# Story 6 validation cleanup, item 1: department alignment. Every mapped
# department must be one clinic-api/seed.py actually seeds -- no unknown
# department may remain (see agent/symptom_routing.py's own docstring).
# --------------------------------------------------------------------- #

class TestDepartmentAlignment:
    def test_every_mapped_department_is_a_real_seeded_department(self):
        unknown = set(SYMPTOM_DEPARTMENT_MAP) - set(SEEDED_DEPARTMENTS)
        assert not unknown, f"these mapped departments do not exist in clinic-api/seed.py: {unknown}"

    def test_the_three_previously_unseeded_departments_are_gone(self):
        # "Ophthalmology" / "Gastroenterology" / "Neurology" were never
        # seeded departments -- confirmed directly against seed.py's own
        # DEPARTMENTS literal, not just asserted from memory.
        for stale_name in ("Ophthalmology", "Gastroenterology", "Neurology"):
            assert stale_name not in SEEDED_DEPARTMENTS, (
                f"test assumption wrong: {stale_name} IS now a seeded department")
            assert stale_name not in SYMPTOM_DEPARTMENT_MAP

    @pytest.mark.parametrize("text,expected_department", [
        # Symptom groups that used to point at the three unseeded
        # departments now resolve to General Medicine instead.
        ("I have eye pain", "General Medicine"),
        ("aankh lal hai", "General Medicine"),
        ("I have stomach pain", "General Medicine"),
        ("acidity hai", "General Medicine"),
        ("I have a migraine", "General Medicine"),
        ("chakkar aa raha hai", "General Medicine"),
    ])
    def test_folded_symptom_groups_still_route_somewhere_real(self, text, expected_department):
        verdict, department = resolve_symptom_department(text)
        assert verdict == COMMIT
        assert department == expected_department
        assert department in SEEDED_DEPARTMENTS


# --------------------------------------------------------------------- #
# Story 6 validation cleanup, item 2: Bengali-script coverage. Bengali-
# script phrases must resolve exactly like their romanised (Banglish)
# equivalents, and existing English/Hinglish/Banglish phrases must keep
# working unchanged.
# --------------------------------------------------------------------- #

class TestBengaliScriptSymptomRouting:
    @pytest.mark.parametrize("text,expected_department", [
        ("আমার গায়ে র‍্যাশ হয়েছে", "Dermatology"),
        ("আমার বুকে ব্যথা করছে", "Cardiology"),
        ("আমার কানে ব্যথা করছে", "ENT"),
        ("আমার কোমরে ব্যথা করছে", "Orthopaedics"),
        ("আমার জ্বর হয়েছে", "General Medicine"),
        ("আমার সর্দি হয়েছে", "General Medicine"),
        # folded groups (Ophthalmology/Gastroenterology/Neurology origin)
        ("আমার চোখে ব্যথা করছে", "General Medicine"),
        ("আমার পেটে ব্যথা করছে", "General Medicine"),
        ("আমার মাথা ব্যথা করছে", "General Medicine"),
    ])
    def test_bengali_script_commits_to_the_right_department(self, text, expected_department):
        verdict, department = resolve_symptom_department(text)
        assert verdict == COMMIT
        assert department == expected_department

    def test_bengali_script_diagnosis_marker_is_never_intercepted(self):
        # Bengali-script equivalent of AC10's "symptom + request for
        # diagnosis" case -- must decline exactly like the English/Latin
        # markers already tested above.
        verdict, department = resolve_symptom_department("আমার বুকে ব্যথা, কি রোগ হয়েছে?")
        assert verdict == NONE
        assert department is None

    def test_bengali_script_does_not_fire_inside_an_unrelated_longer_word(self):
        # The word-boundary-safety reason this module uses _bn_bounded()
        # for Bengali script instead of a bare substring check (see that
        # module's own docstring): "চোখ" (eye) must not fire merely because
        # it is a substring of some unrelated longer word containing it.
        # "সচোখে" is a nonsense probe word chosen only to contain "চোখ" as
        # a substring with no word boundary on either side.
        verdict, department = resolve_symptom_department("সচোখেসামগ্রী কিনতে হবে")
        assert verdict == NONE
        assert department is None

    @pytest.mark.parametrize("romanised,script", [
        ("I have chest pain", "আমার বুকে ব্যথা করছে"),
        ("kamar mein dard hai", "আমার কোমরে ব্যথা করছে"),
    ])
    def test_bengali_script_and_its_romanised_equivalent_agree(self, romanised, script):
        r_verdict, r_department = resolve_symptom_department(romanised)
        s_verdict, s_department = resolve_symptom_department(script)
        assert r_verdict == s_verdict == COMMIT
        assert r_department == s_department


# --------------------------------------------------------------------- #
# Story 6 validation cleanup, item 3: false-positive matching. A short,
# common word like "cold" must not fire inside an unrelated sentence, but
# a genuine "I have a cold" statement must still be caught.
# --------------------------------------------------------------------- #

class TestColdFalsePositiveMatching:
    @pytest.mark.parametrize("text", [
        "cold weather",
        "cold water",
        "I don't like cold drinks",
        "it's really cold today",
        "can I get a cold drink please",
        "the weather has turned cold",
    ])
    def test_unrelated_cold_mentions_are_never_intercepted(self, text):
        verdict, department = resolve_symptom_department(text)
        assert verdict == NONE, f"{text!r} should not trigger symptom routing"
        assert department is None

    @pytest.mark.parametrize("text", [
        "I have a cold",
        "I've had a cold since yesterday",
        "I have a bad cold",
        "I am down with a cold",
        "mujhe sardi hai",
        "amar sordi hoyeche",
        "আমার সর্দি হয়েছে",
    ])
    def test_genuine_cold_statements_still_route_to_general_medicine(self, text):
        verdict, department = resolve_symptom_department(text)
        assert verdict == COMMIT
        assert department == "General Medicine"


# --------------------------------------------------------------------- #
# main.py/main_pcm.py's _resolve_intent() -- guard order and short-circuit
# --------------------------------------------------------------------- #

class _ExplodingFastPath:
    def resolve(self, text):
        raise AssertionError("fast_path.resolve() must not run for a symptom-routing turn")


class _ExplodingCache:
    def get(self, text):
        raise AssertionError("semantic cache must not run for a symptom-routing turn")

    def put(self, text, data):
        raise AssertionError("semantic cache must not run for a symptom-routing turn")


def _exploding_extract_intent(text):
    raise AssertionError("extract_intent() (the LLM) must not run for a symptom-routing turn")


class TestGuardOrderInResolveIntent:
    @pytest.fixture(autouse=True)
    def _wire_exploding_stand_ins(self, monkeypatch):
        monkeypatch.setattr(main_pcm, "_fast_path", _ExplodingFastPath())
        monkeypatch.setattr(main_pcm, "_intent_cache", _ExplodingCache())
        monkeypatch.setattr(main_pcm, "extract_intent", _exploding_extract_intent)

    def test_short_circuits_before_any_classifier_runs(self):
        session = types.SimpleNamespace(call_id="test-symptom-routing-call")
        data = run(main_pcm._resolve_intent(session, "I have chest pain"))
        assert data["intent"] == "symptom_department_routing"
        assert data["slots"]["department"] == "Cardiology"
        assert data["parts"] == [{"intent": "symptom_department_routing", "slots": data["slots"]}]
        assert data["direct_reply_bn"] is None

    def test_clinical_interpretation_still_wins_over_symptom_routing(self):
        # The central boundary claim of this story: a caller who names a
        # symptom AND asks a danger/diagnosis question is classified
        # "clinical_interpretation", never "symptom_department_routing" --
        # is_clinical_interpretation() is checked earlier in
        # _resolve_intent() and returns first.
        session = types.SimpleNamespace(call_id="test-symptom-routing-call")
        data = run(main_pcm._resolve_intent(session, "I have chest pain, am I dying?"))
        assert data["intent"] == "clinical_interpretation"

    def test_a_bare_routing_question_reaches_the_fast_path(self):
        # Negative control: no symptom named, so neither guard fires and
        # this reaches the (exploding) fast path -- proves the stand-ins
        # are actually wired up.
        session = types.SimpleNamespace(call_id="test-symptom-routing-call")
        with pytest.raises(AssertionError):
            run(main_pcm._resolve_intent(session, "which department should I go to?"))


# --------------------------------------------------------------------- #
# Story 6 validation cleanup, item 4: preserve the existing safety
# boundary. The exact adversarial phrases from the cleanup request must
# never be converted into a plain symptom-routing response -- either
# because agent/clinical_safety.py (untouched by this story) already
# catches them, or because they name no symptom phrase at all so this
# module's own resolver has nothing to commit to either way. Either
# reason is an acceptable pass here: what must never happen is
# resolve_symptom_department() committing to a department for one of
# these.
# --------------------------------------------------------------------- #

class TestClinicalSafetyBoundary:
    @pytest.mark.parametrize("text", [
        "What disease do I have?",
        "What is wrong with me?",
        "Is this a heart problem?",
        "Is this serious?",
    ])
    def test_bare_diagnosis_questions_never_commit_to_a_department(self, text):
        verdict, department = resolve_symptom_department(text)
        assert verdict == NONE
        assert department is None

    @pytest.mark.parametrize("text", [
        "What is wrong with me?",
        "Is this serious?",
    ])
    def test_the_ones_clinical_safety_already_covers_still_fire_it(self, text):
        # Not every phrase in this class's own list is (or needs to be)
        # covered by agent/clinical_safety.py -- see this class's own
        # docstring above -- but the ones that plainly are must keep
        # firing it, unmodified, exactly as before this cleanup pass.
        assert is_clinical_interpretation(text) is True

    @pytest.mark.parametrize("text", [
        "I have chest pain, what disease do I have?",
        "seene mein dard hai, kaunsi bimari hai yeh?",
    ])
    def test_symptom_plus_own_diagnosis_marker_never_commits(self, text):
        # AC10's "symptom + request for diagnosis" case where the wording
        # matches this module's OWN _DIAGNOSIS_REQUEST_MARKERS -- the
        # resolver itself declines, the same behaviour
        # TestResolveSymptomDepartment::test_a_symptom_plus_a_diagnosis_
        # request_is_never_intercepted already covers with a different set
        # of examples.
        verdict, department = resolve_symptom_department(text)
        assert verdict == NONE
        assert department is None

    @pytest.mark.parametrize("text", [
        "I have a headache, what is wrong with me?",
        "I have chest pain, is this serious?",
    ])
    def test_symptom_plus_clinical_safety_phrase_is_protected_by_guard_order(self, text, monkeypatch):
        # These two do NOT match any of this module's own
        # _DIAGNOSIS_REQUEST_MARKERS ("wrong with me" / "is this serious"
        # are agent/clinical_safety.py's phrases, not this module's) -- so,
        # exactly like the pre-existing
        # test_a_symptom_plus_danger_wording_is_left_to_clinical_safety_
        # first above, resolve_symptom_department() WOULD commit on the
        # symptom alone. That is expected and fine: what actually protects
        # these is agent/clinical_safety.py already having caught them,
        # checked FIRST by _resolve_intent()'s guard order, so this
        # module's own resolver is never even consulted in production for
        # these two phrases. Verified at both levels here.
        assert is_clinical_interpretation(text) is True
        verdict, _department = resolve_symptom_department(text)
        assert verdict == COMMIT  # true in isolation; never reached in production -- see below

        monkeypatch.setattr(main_pcm, "_fast_path", _ExplodingFastPath())
        monkeypatch.setattr(main_pcm, "_intent_cache", _ExplodingCache())
        monkeypatch.setattr(main_pcm, "extract_intent", _exploding_extract_intent)
        session = types.SimpleNamespace(call_id="test-symptom-routing-call")
        data = run(main_pcm._resolve_intent(session, text))
        assert data["intent"] == "clinical_interpretation"

    def test_full_dispatch_still_prefers_clinical_interpretation_for_a_boundary_phrase(self, monkeypatch):
        # End-to-end structural proof at the _resolve_intent() level (not
        # just the resolver function in isolation): a symptom plus a
        # clinical-safety phrase must resolve to "clinical_interpretation",
        # never "symptom_department_routing", with the classifier itself
        # never invoked.
        monkeypatch.setattr(main_pcm, "_fast_path", _ExplodingFastPath())
        monkeypatch.setattr(main_pcm, "_intent_cache", _ExplodingCache())
        monkeypatch.setattr(main_pcm, "extract_intent", _exploding_extract_intent)
        session = types.SimpleNamespace(call_id="test-symptom-routing-call")
        data = run(main_pcm._resolve_intent(session, "I have chest pain, is this serious?"))
        assert data["intent"] == "clinical_interpretation"


# --------------------------------------------------------------------- #
# agent/reply_templates.symptom_routing_reply() -- AC2/AC3/AC11
# --------------------------------------------------------------------- #

_FOUND_WITH_DOCTORS = {
    "found": True, "department": "Cardiology", "department_bn": "কার্ডিওলজি",
    "doctors": [{"name": "Dr. K. Bhattacharya", "doctor_name_bn": "ভট্টাচার্য"},
                {"name": "Dr. N. Roy", "doctor_name_bn": "রায়"}],
}
_FOUND_NO_DOCTORS = {
    "found": True, "department": "Cardiology", "department_bn": "কার্ডিওলজি", "doctors": [],
}
_NOT_FOUND = {"found": False, "query": "Neurology"}
_ALL_RESULT_SHAPES = [_FOUND_WITH_DOCTORS, _FOUND_NO_DOCTORS, _NOT_FOUND]

# AC2/AC11: phrasing this reply (and, as a spot check, the pre-existing
# templates this story relies on) must never contain, in English -- the
# language every "never say X" example in the acceptance criteria itself
# was given in.
_FORBIDDEN_DIAGNOSTIC_PATTERNS = [
    re.compile(r"\byou have\b", re.IGNORECASE),
    re.compile(r"\bthis sounds like\b", re.IGNORECASE),
    re.compile(r"\byou are suffering from\b", re.IGNORECASE),
    re.compile(r"\byou('re| are) diagnosed\b", re.IGNORECASE),
    re.compile(r"\byou might have\b", re.IGNORECASE),
    re.compile(r"\bi think you have\b", re.IGNORECASE),
]


def _assert_no_diagnostic_language(text: str):
    for pattern in _FORBIDDEN_DIAGNOSTIC_PATTERNS:
        assert not pattern.search(text), f"diagnostic phrasing found: {pattern.pattern!r} in {text!r}"


class TestSymptomRoutingReplyLinguisticSafety:
    @pytest.mark.parametrize("result", _ALL_RESULT_SHAPES)
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_returns_nonempty_text(self, result, language):
        text = symptom_routing_reply({"department": "Cardiology"}, result, language=language)
        assert isinstance(text, str)
        assert text.strip()

    def test_default_language_is_bengali(self):
        assert symptom_routing_reply({"department": "Cardiology"}, _FOUND_WITH_DOCTORS) == \
            symptom_routing_reply({"department": "Cardiology"}, _FOUND_WITH_DOCTORS, language="bengali")

    @pytest.mark.parametrize("result", _ALL_RESULT_SHAPES)
    def test_every_language_is_textually_distinct(self, result):
        replies = {symptom_routing_reply({"department": "Cardiology"}, result, language=l)
                   for l in _LANGUAGES}
        assert len(replies) == len(_LANGUAGES)

    @pytest.mark.parametrize("result", _ALL_RESULT_SHAPES)
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_never_contains_forbidden_diagnostic_phrasing(self, result, language):
        # Only the English-language reply is checked against the English
        # forbidden-phrase list (AC2's own examples are English) -- see
        # this test class's own docstring note above.
        if language == "english":
            _assert_no_diagnostic_language(
                symptom_routing_reply({"department": "Cardiology"}, result, language=language))

    def test_found_with_doctors_states_this_is_routing_not_diagnosis(self):
        text = symptom_routing_reply({"department": "Cardiology"}, _FOUND_WITH_DOCTORS, language="english")
        assert "not a diagnosis" in text.lower()
        assert "routing" in text.lower()

    def test_names_the_department_when_found(self):
        # _spoken_department() (pre-existing, unmodified helper this reply
        # reuses -- see agent/reply_templates.py's own docstring on it)
        # prefers the seeded Bengali alias over the English name whenever
        # clinic-api supplies one, the SAME way doctors_by_department_reply()
        # already does in every language branch -- not something this
        # story changes. Checked against whichever form that helper
        # actually returns, rather than hard-coding an assumption about
        # which one it is.
        from agent.reply_templates import _spoken_department
        expected = _spoken_department({"department": "Cardiology"}, _FOUND_WITH_DOCTORS)
        text = symptom_routing_reply({"department": "Cardiology"}, _FOUND_WITH_DOCTORS, language="english")
        assert expected in text

    def test_not_found_never_names_the_inferred_department(self):
        # The caller never said "Neurology" -- speaking it back as though
        # they had would be confusing, not helpful. See this reply
        # function's own docstring.
        text = symptom_routing_reply({"department": "Neurology"}, _NOT_FOUND, language="english")
        assert "Neurology" not in text
        assert "not a diagnosis" in text.lower()

    def test_is_distinct_from_doctors_by_department_reply(self):
        # AC2: this is a different kind of moment from a caller who named
        # the department themselves, and must not share that reply's
        # wording verbatim.
        slots = {"department": "Cardiology"}
        assert symptom_routing_reply(slots, _FOUND_WITH_DOCTORS, language="english") != \
            doctors_by_department_reply(slots, _FOUND_WITH_DOCTORS, language="english")

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_is_spoken_punctuation_clean(self, language):
        from agent.bn_normalize import verbalize
        for result in _ALL_RESULT_SHAPES:
            spoken = verbalize(
                symptom_routing_reply({"department": "Cardiology"}, result, language=language),
                language=language)
            for ch in (":", "：", "[", "]", "{", "}"):
                assert ch not in spoken


class TestExistingTemplatesSpotCheckedForDiagnosticLanguage:
    # AC3's "check the existing reply templates" -- a spot check of the
    # templates this story's own investigation and implementation lean on
    # most heavily (doctors_by_department_reply, whose doctor-listing
    # shape this story's own reply mirrors) rather than a full
    # introspection of every reply function in the module, which is out
    # of proportion for this story -- see the final report's own note on
    # this being a deliberately scoped check, not a repo-wide linter.
    def test_doctors_by_department_reply_has_no_diagnostic_language(self):
        result = {"found": True, "department": "Cardiology", "department_bn": "কার্ডিওলজি",
                  "doctors": [{"name": "Dr. K. Bhattacharya", "doctor_name_bn": "ভট্টাচার্য"}]}
        text = doctors_by_department_reply({"department": "Cardiology"}, result, language="english")
        _assert_no_diagnostic_language(text)

    def test_clinical_interpretation_reply_has_no_diagnostic_language(self):
        from agent.reply_templates import clinical_interpretation_reply
        _assert_no_diagnostic_language(clinical_interpretation_reply(language="english"))


# --------------------------------------------------------------------- #
# main_pcm.py dispatch -- initial "symptom_department_routing" branch
# --------------------------------------------------------------------- #

class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-symptom-routing-call", pending=pending,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
        call_state=main_pcm.call_state_mod.build(), utt_seq=1,
        confirm_attempts=0,
    )


class FakeASRResultChestPain:
    text = "I have chest pain"
    decoder_agreement = 0.95
    decoder_used = "rnnt"
    ctc_words = 4
    rnnt_words = 4


class FakeToolsClient:
    def __init__(self, response):
        self.calls = []
        self._response = response

    async def get_doctors_by_department(self, department, date=None):
        self.calls.append((department, date))
        return self._response


class TestSymptomRoutingIntentDispatch:
    def _dispatch(self, monkeypatch, tmp_path, tools_response):
        spoken = []
        fake_tools = FakeToolsClient(tools_response)

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        async def fake_resolve_intent(session, text):
            return {"intent": "symptom_department_routing",
                    "slots": {"department": "Cardiology"}}

        class FakeASR:
            async def transcribe_utterance(self, wav_path):
                return FakeASRResultChestPain()

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        monkeypatch.setattr(main_pcm, "_asr", FakeASR())
        monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
        monkeypatch.setattr(main_pcm, "_tools", fake_tools)

        session = make_session()
        wav_path = tmp_path / "utt.wav"
        wav_path.write_bytes(b"")
        run(main_pcm._dispatch_turn(session, str(wav_path)))
        return spoken, session, fake_tools

    def test_calls_the_existing_doctors_by_department_tool_with_the_resolved_department(self, monkeypatch, tmp_path):
        _, _, fake_tools = self._dispatch(monkeypatch, tmp_path, _FOUND_WITH_DOCTORS)
        assert fake_tools.calls == [("Cardiology", None)]

    def test_speaks_the_routing_reply(self, monkeypatch, tmp_path):
        spoken, _, _ = self._dispatch(monkeypatch, tmp_path, _FOUND_WITH_DOCTORS)
        assert spoken == [symptom_routing_reply({"department": "Cardiology"},
                                                  _FOUND_WITH_DOCTORS, language="english")]

    def test_does_not_open_a_doctor_choice_pending_state(self, monkeypatch, tmp_path):
        # Deliberately a single-turn answer -- see _finish_symptom_routing()'s
        # own docstring for why this does not continue into booking the
        # way "doctors_by_department" does.
        _, session, _ = self._dispatch(monkeypatch, tmp_path, _FOUND_WITH_DOCTORS)
        assert session.pending is None

    def test_department_not_offered_is_answered_honestly(self, monkeypatch, tmp_path):
        spoken, session, fake_tools = self._dispatch(monkeypatch, tmp_path, _NOT_FOUND)
        assert fake_tools.calls == [("Cardiology", None)]
        assert spoken == [symptom_routing_reply({"department": "Cardiology"},
                                                  _NOT_FOUND, language="english")]
        assert session.pending is None


# --------------------------------------------------------------------- #
# main_pcm.py _continue_pending -- interrupting an in-progress flow
# --------------------------------------------------------------------- #

class TestSymptomRoutingInterruptsPendingFlow:
    def test_abandons_an_in_progress_booking_flow_and_routes_instead(self, monkeypatch):
        spoken = []
        fake_tools = FakeToolsClient(_FOUND_WITH_DOCTORS)

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        monkeypatch.setattr(main_pcm, "_tools", fake_tools)

        # An in-progress booking flow, mid-way through collecting a phone
        # number -- same shape as the pending states the existing
        # immediate-human-request/complaint/doctor-personal-request
        # interrupt tests use.
        pending = {"awaiting": "phone", "slots": {"doctor_name": "Dr. Sen", "date": "2026-10-01",
                                                    "time_slot": "10:00", "patient_name": "Test"}}
        session = make_session(pending=dict(pending))
        handled = run(main_pcm._continue_pending(session, "I have chest pain"))

        assert handled is True
        assert session.pending is None
        assert fake_tools.calls == [("Cardiology", None)]
        assert spoken == [symptom_routing_reply({"department": "Cardiology"},
                                                  _FOUND_WITH_DOCTORS, language="english")]


# --------------------------------------------------------------------- #
# Transport parity
# --------------------------------------------------------------------- #

class TestTransportParity:
    def test_main_dot_py_has_the_symptom_routing_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "symptom_department_routing"' in src
        assert "resolve_symptom_department" in src
        assert "symptom_routing_reply" in src
        assert "_finish_symptom_routing" in src

    def test_main_pcm_dot_py_has_the_symptom_routing_branch(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "symptom_department_routing"' in src
        assert "resolve_symptom_department" in src
        assert "symptom_routing_reply" in src
        assert "_finish_symptom_routing" in src

    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb


# --------------------------------------------------------------------- #
# agent/llm.py -- deliberately NOT a registered LLM intent (AC4/AC constraint
# 4: the model must never decide symptom routing)
# --------------------------------------------------------------------- #

class TestSymptomRoutingIsNeverDelegatedToTheModel:
    def test_symptom_department_routing_is_not_a_valid_llm_intent(self):
        # Deliberate design choice, not an oversight: unlike
        # "clinical_interpretation" (registered in VALID_INTENTS as a
        # defense-in-depth SECOND layer, because that guard's own
        # docstring accepts the model may occasionally need to catch
        # phrasing its literal phrase lists miss), this intent must NEVER
        # be something the model itself classifies or fills a department
        # slot for -- see agent/symptom_routing.py's own module docstring.
        from agent.llm import VALID_INTENTS
        assert "symptom_department_routing" not in VALID_INTENTS

    def test_clinical_interpretation_prompt_now_disambiguates_bare_symptom_routing(self):
        from agent.llm import SYSTEM_PROMPT_TEMPLATE
        assert "naming a symptom" in SYSTEM_PROMPT_TEMPLATE.lower() or \
            "symptom by itself" in SYSTEM_PROMPT_TEMPLATE.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
