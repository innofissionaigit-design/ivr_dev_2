"""KCD-511 / KCD-512 / KCD-513 / KCD-514 / KCD-516: the persona, register, acknowledgement
and apology rules, checked against every caller-facing template on every commit.

Native-speaker sign-off is NOT something a test can give (agent/persona.py
REVIEW_STATUS says every language is still pending). What these do is stop the voice
eroding: any template edit that breaks the documented persona fails here, and each
check is proved to FIRE on a known-bad example so it cannot quietly stop working.

    python -m pytest tests/test_persona_and_apology.py -v
"""

import ast
import os
import re
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import persona
from agent.apology import (
    CAUSE_OF_PHRASE_KEY,
    CAUSES,
    NOT_EXIST,
    apology_for,
    count_apologies,
    ends_on_bare_apology,
    enforce_single_apology,
)
from agent.phrases import PHRASES
from agent.speech_norm import unspeakable_spans, verbalize
from agent.spoken_text_lint import has_spoken_artifact
from agent.turn_ack import _ACK, MAX_ACK_WORDS, MIN_TURNS_BETWEEN, AckTracker, validate_table

SPOKEN_SOURCES = [
    "agent/phrases.py",
    "agent/reply_templates.py",
    "agent/reply_templates_i18n.py",
    "agent/acknowledgement.py",
    "agent/reask_policy.py",
    "agent/apology.py",
    "agent/turn_ack.py",
    "agent/disclosure.py",
    "agent/history_templates.py",
    "agent/patient_context.py",
]
_LANG_SCRIPT = {"bn": re.compile(r"[ঀ-৿]"), "hi": re.compile(r"[ऀ-ॿ]")}
_ASKS_TO_SWITCH = {
    "en": re.compile(
        r"\b(please )?(speak|say (it|that)|talk|repeat (it|that)) in (english|hindi|bengali|bangla)\b", re.I
    ),
    "bn": re.compile(r"(বাংলা|হিন্দি|ইংরেজি)(তে|য়)\s*(বলুন|বলবেন|বলতে পারেন)"),
    "hi": re.compile(r"(हिंदी|हिन्दी|बंगाली|बांग्ला|अंग्रेज़ी|अंग्रेजी)\s*में\s*(बोलिए|बताइए|बोलें|बोलेंगे)"),
}


def _fstring_text(node):
    """An f-string as the text a caller would hear, with {} where a value goes."""
    return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)


def _spoken_literals():
    """Every string literal (f-strings whole, with {} for values) in the
    caller-facing sources that could be spoken. Docstrings, code-like strings and
    this test's own regex sources (agent/apology.py, agent/turn_ack.py hold patterns
    and tables, checked by their own tests) are skipped."""
    out = []
    for rel in SPOKEN_SOURCES:
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        docs, inside_fstring = set(), set()
        for n in ast.walk(tree):
            if (
                isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.body
                and isinstance(n.body[0], ast.Expr)
                and isinstance(getattr(n.body[0], "value", None), ast.Constant)
                and isinstance(n.body[0].value.value, str)
            ):
                docs.add(n.body[0].value.lineno)
            if isinstance(n, ast.JoinedStr):
                inside_fstring.update(id(v) for v in n.values)
        for n in ast.walk(tree):
            if isinstance(n, ast.JoinedStr):
                t, line = _fstring_text(n), n.lineno
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in inside_fstring:
                t, line = n.value, n.lineno
            else:
                continue
            if line in docs or len(t) < 6 or not t.strip():
                continue
            if rel in ("agent/apology.py", "agent/turn_ack.py", "agent/history_templates.py") and re.search(
                r"[\|^$*?()\[\]]", t
            ):
                continue  # a regex source, not speech
            is_indic = any(p.search(t) for p in _LANG_SCRIPT.values())
            is_english_sentence = bool(re.search(r"[A-Za-z]{3,} [A-Za-z]{2,}", t)) and not re.search(
                r"[_=/\()]|^\s*[a-z_]+$", t
            )
            if is_indic or is_english_sentence:
                out.append((rel, line, t))
    return out


LITERALS = _spoken_literals()


def test_the_scan_actually_finds_the_templates():
    assert len(LITERALS) > 300  # a scan that finds nothing would pass every check below
    assert {rel for rel, _, _ in LITERALS} >= {
        "agent/phrases.py",
        "agent/reply_templates.py",
        "agent/reply_templates_i18n.py",
    }


# ============================================== KCD-511/512/516: persona on every template

# RATCHET, in the same spirit as tests/test_spoken_text_lint.py: two templates read a
# booking back with five values in one sentence (Bengali 17 words, Hindi 23, against
# caps of 15 and 18). Splitting them changes what a caller hears in their own
# language, which is a native reviewer's call, not this file's. The count may only
# go DOWN: fixing one fails the test until this number is lowered with it.
KNOWN_LONG_SENTENCE_TEMPLATES = 2


def test_no_template_breaks_the_persona_beyond_the_known_long_readbacks():
    bad = [
        (rel, ln, persona.violations(t), t[:70])
        for rel, ln, t in LITERALS
        if set(persona.violations(t)) - {"sentence_too_long"}
    ]
    assert not bad, chr(10).join(map(str, bad))


def test_the_long_sentence_backlog_is_exactly_its_recorded_size():
    long = [(rel, ln) for rel, ln, t in LITERALS if "sentence_too_long" in persona.violations(t)]
    assert len(long) == KNOWN_LONG_SENTENCE_TEMPLATES, long


def test_register_is_respectful_in_every_language():
    for rel, ln, t in LITERALS:
        assert "informal_register" not in persona.violations(t), (rel, ln, t)


def test_no_template_asks_the_caller_to_change_language():
    for rel, ln, t in LITERALS:
        for lang, pat in _ASKS_TO_SWITCH.items():
            assert not pat.search(t), (rel, ln, t)


def test_no_spoken_field_label_or_bracket_in_the_shared_phrases():
    for lang, table in PHRASES.items():
        for key, text in table.items():
            assert not has_spoken_artifact(text), (lang, key, text)


def test_every_shared_phrase_is_speakable_with_no_surviving_latin_or_digit_span():
    for lang in ("bn", "hi"):
        for key, text in PHRASES[lang].items():
            assert unspeakable_spans(verbalize(text, lang), lang) == [], (lang, key, text)


def test_every_reply_key_exists_in_all_three_languages():
    keys = {lang: set(t) for lang, t in PHRASES.items()}
    assert keys["bn"] == keys["hi"] == keys["en"], {k: keys["bn"] ^ v for k, v in keys.items()}


@pytest.mark.parametrize(
    "text,lang,expected",
    [
        ("তুমি কী চাও?", "bn", "informal_register"),
        ("আপনি আজ এসো", "bn", "informal_register"),
        ("तुम कल आओ।", "hi", "informal_register"),
        ("It is probably ready tomorrow.", "en", "hedge"),
        ("Don't worry, it is nothing serious.", "en", "reassurance"),
        ("You should take the medicine after food.", "en", "clinical_direction"),
        ("As an AI language model I cannot say.", "en", "self_reference_as_model"),
        ("शायद कल मिल जाएगा।", "hi", "hedge"),
        ("চিন্তা করবেন না।", "bn", "reassurance"),
    ],
)
def test_every_check_fires_on_a_known_bad_example(text, lang, expected):
    assert expected in persona.violations(text, lang)


def test_a_sentence_over_the_cap_is_flagged():
    long_en = " ".join(["word"] * (persona.MAX_SENTENCE_WORDS["en"] + 1)) + "."
    assert "sentence_too_long" in persona.violations(long_en, "en")
    assert "sentence_too_long" not in persona.violations(
        " ".join(["word"] * persona.MAX_SENTENCE_WORDS["en"]) + ".", "en"
    )


def test_clean_text_in_all_three_languages_passes():
    assert persona.is_clean("আপনার অ্যাপয়েন্টমেন্টটা কালকে সকাল দশটায়।")
    assert persona.is_clean("आपका अपॉइंटमेंट कल सुबह दस बजे है।")
    assert persona.is_clean("Your appointment is tomorrow at ten in the morning.")


def test_every_language_is_still_pending_native_review_and_says_so():
    assert persona.pending_review() == ["bn", "en", "hi"]
    assert persona.PERSONA_VERSION


def test_the_persona_document_exists_and_states_its_own_status():
    with open(os.path.join(REPO_ROOT, "docs", "persona.md"), encoding="utf-8") as f:
        doc = f.read()
    assert "DRAFT" in doc and "not** been signed off" in doc
    for section in ("Register", "What the agent never says", "Apologies", "Acknowledging"):
        assert section in doc


# ====================================================== KCD-514: apologies, calibrated and never stacked


def test_no_single_template_string_stacks_apologies():
    bad = []
    for rel, ln, t in LITERALS:
        lang = persona.detect_language(t)
        if count_apologies(t, lang) > 1:
            bad.append((rel, ln, t[:80]))
    assert not bad, "\n".join(map(str, bad))


def test_the_four_causes_have_distinct_wording_in_every_language():
    for lang in ("bn", "hi", "en"):
        texts = [apology_for(c, lang) for c in CAUSES]
        assert len(set(texts)) == 4, (lang, texts)
        for t in texts:
            assert count_apologies(t, lang) == 1 and persona.is_clean(t, lang), (lang, t)


def test_cannot_check_and_does_not_exist_are_never_interchangeable():
    for lang in ("bn", "hi", "en"):
        assert apology_for("cannot_check", lang) != apology_for(NOT_EXIST, lang)


def test_existing_failure_phrases_map_to_the_right_cause_and_carry_one_apology():
    assert CAUSE_OF_PHRASE_KEY["tool_failure"] == "cannot_check" and CAUSE_OF_PHRASE_KEY["asr_empty"] == "not_heard"
    for lang in ("bn", "hi", "en"):
        for key in CAUSE_OF_PHRASE_KEY:
            if key in PHRASES[lang]:
                assert count_apologies(PHRASES[lang][key], lang) <= 1, (lang, key)


@pytest.mark.parametrize(
    "text,lang",
    [
        ("Sorry, I could not hear that. Sorry, please repeat it.", "en"),
        ("Sorry, sorry, I could not hear you.", "en"),
        ("দুঃখিত, শুনতে পাইনি। দুঃখিত, আবার বলবেন?", "bn"),
        ("माफ़ कीजिए, सुन नहीं पाई। माफ़ कीजिए, क्या आप फिर बताएँगे?", "hi"),
    ],
)
def test_stacked_apologies_are_reduced_to_one_at_run_time(text, lang):
    assert count_apologies(text, lang) >= 2
    fixed, removed = enforce_single_apology(text, lang)
    assert count_apologies(fixed, lang) == 1 and removed >= 1


def test_only_the_apology_is_removed_never_the_content():
    text = "Sorry, I could not hear that. Sorry, could you repeat the doctor's name?"
    fixed, _ = enforce_single_apology(text, "en")
    assert "could you repeat the doctor's name" in fixed.lower()
    assert fixed.lower().startswith("sorry, i could not hear that.")


def test_a_reply_with_one_or_no_apology_is_untouched():
    for text in ("Dr. Sen is available on Tuesday.", "Sorry, I cannot check that right now."):
        assert enforce_single_apology(text, "en") == (text, 0)


def test_a_fact_carrying_sentence_is_never_dropped_with_a_stacked_apology():
    text = "Sorry, that test is not in our list. Sorry. The CBC costs 350 rupees."
    fixed, _ = enforce_single_apology(text, "en")
    assert "350 rupees" in fixed


def test_a_reply_ending_on_a_bare_apology_is_detected():
    assert ends_on_bare_apology("The report is not ready. Sorry.", "en")
    assert ends_on_bare_apology("রিপোর্ট তৈরি হয়নি। দুঃখিত।", "bn")
    assert not ends_on_bare_apology("Sorry, I cannot check that. Please try the counter.", "en")


def test_no_template_ends_on_a_bare_apology():
    bad = [(rel, ln, t[-60:]) for rel, ln, t in LITERALS if ends_on_bare_apology(t, persona.detect_language(t))]
    assert not bad, "\n".join(map(str, bad))


# ================================================================ KCD-513: acknowledge before answering


def test_every_acknowledgement_is_safe_by_construction():
    assert validate_table() == []
    for lang, variants in _ACK.items():
        for a in variants:
            assert len(a.split()) <= MAX_ACK_WORDS and not re.search(r"\d", a)
            assert persona.is_clean(a, lang)


def test_the_validator_rejects_an_acknowledgement_that_could_carry_a_fact(monkeypatch):
    import agent.turn_ack as ta

    monkeypatch.setitem(ta._ACK, "en", ["Sure, that is 350 rupees."])
    problems = ta.validate_table()
    assert problems and any("digit" in p or "longer" in p for p in problems)


def test_a_substantive_reply_opens_with_an_acknowledgement():
    t = AckTracker()
    t.next_turn()
    text, used = t.decorate("The CBC costs three hundred and fifty rupees.", "en")
    assert used and text.startswith(_ACK["en"][0]) and text.endswith("rupees.")


def test_it_is_suppressed_on_repeat_turns_and_returns_after_the_gap():
    t = AckTracker()
    used = []
    for _ in range(6):
        t.next_turn()
        used.append(t.decorate("The doctor is available tomorrow morning.", "en")[1])
    assert used == [True, False, True, False, True, False]  # never on consecutive replies
    assert MIN_TURNS_BETWEEN == 2


def test_the_variant_rotates_so_it_is_not_a_tic():
    t = AckTracker()
    heard = []
    for _ in range(7):
        t.next_turn()
        text, used = t.decorate("The doctor is available tomorrow morning.", "en")
        if used:
            heard.append(text.split(".")[0])
    assert len(set(heard)) >= 3


@pytest.mark.parametrize(
    "reply,kwargs",
    [
        ("Sorry, I cannot check that right now.", {}),  # an apology opens it
        ("Yes.", {}),  # too short to need one
        ("Sure. The fee is five hundred rupees.", {}),  # already acknowledged
        (
            "The fee is five hundred rupees.",
            {"already_acknowledged": True},
        ),  # the distress acknowledgement opened the turn
        ("The fee is five hundred rupees.", {"substantive": False}),
        ("দুঃখিত, দেখতে পারছি না।", {}),
    ],
)
def test_no_acknowledgement_in_front_of_these(reply, kwargs):
    t = AckTracker()
    t.next_turn()
    text, used = t.decorate(reply, "bn" if "দুঃখিত" in reply else "en", **kwargs)
    assert used is False and text == reply


def test_the_acknowledgement_never_changes_the_reply_it_precedes():
    t = AckTracker()
    t.next_turn()
    reply = "Dr. Sen sits on Tuesday from ten to one. The fee is five hundred rupees."
    text, used = t.decorate(reply, "en")
    assert used and text.endswith(reply)
