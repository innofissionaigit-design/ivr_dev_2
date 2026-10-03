"""The tools for scoring a recogniser before any real recording exists: transcript rules, WER/CER, critical-entity
accuracy, the manifest checks, the confusable-names report, and the command that ties them together.

    python -m pytest tests/test_asr_evaluation.py -v

Everything runs on a tiny catalogue written in this file; nothing needs the pod, a model or the clinic API.
"""

import json
import os
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.append(_p)

import asr_eval  # noqa: E402
import confusable_pairs  # noqa: E402

from agent import asr_manifest as am  # noqa: E402
from agent.asr_metrics import (  # noqa: E402
    Edits,
    align,
    bootstrap_interval,
    char_edits,
    score_entities,
    word_edits,
)
from agent.catalogue_forms import lab_test_code  # noqa: E402
from agent.confusables import (  # noqa: E402
    SAME_SOUND,
    SAME_WRITTEN,
    find_confusables,
    shared_words,
)
from agent.transcript_rules import (  # noqa: E402
    DOCTOR,
    NUMBER,
    TEST,
    YES_NO,
    AliasTable,
    Entity,
    normalize_text,
    numbers_in,
)

CAT = {
    "tests": [
        {"name": "Complete Blood Count (CBC)", "aliases_bn": ["সিবিসি", "সি বি সি"], "aliases_hi": ["सीबीसी"]},
        {"name": "CRP (C-Reactive Protein)", "aliases_bn": ["সিআরপি"], "aliases_hi": []},
        {"name": "TSH", "aliases_bn": ["টিএসএইচ"], "aliases_hi": []},
        {"name": "Thyroid Profile (T3 T4 TSH)", "aliases_bn": ["থাইরয়েড প্রোফাইল"], "aliases_hi": []},
        {"name": "Blood Sugar Fasting", "aliases_bn": ["সুগার ফাস্টিং"], "aliases_hi": []},
        {"name": "Blood Sugar PP", "aliases_bn": ["সুগার পিপি"], "aliases_hi": []},
    ],
    "doctors": [
        {
            "name": "Dr. A. Sen",
            "full_name": "Dr. Arindam Sen",
            "surname": "Sen",
            "aliases_bn": ["সেন"],
            "aliases_hi": [],
        },
        {"name": "Dr. N. Roy", "full_name": "Dr. Nirmal Roy", "surname": "Roy", "aliases_bn": ["রায়"], "aliases_hi": []},
        {"name": "Dr. P. Ray", "full_name": "Dr. Partha Ray", "surname": "Ray", "aliases_bn": ["রায়"], "aliases_hi": []},
    ],
}
TABLE = AliasTable(CAT)


# ------------------------------------------------------------------------------------------- transcript rules


def test_one_test_is_one_token_however_it_was_said():
    spoken = ["সিবিসি টেস্টের দাম কত", "সি বি সি টেস্টের দাম কত", "CBC টেস্টের দাম কত", "cbc টেস্টের দাম কত"]
    assert len({TABLE.normalize(s) for s in spoken}) == 1
    assert TABLE.normalize(spoken[0]) == "TEST:CBC টেস্টের দাম কত"


def test_the_longest_form_wins_and_a_test_is_not_split():
    # "T3 T4 TSH" is the Thyroid Profile; the TSH inside it must not also be read as the TSH test
    found = TABLE.entities("thyroid profile t3 t4 tsh", "en")
    assert [(e.kind, e.value) for e in found if e.kind == TEST] == [(TEST, "Thyroid Profile (T3 T4 TSH)")]
    assert [e.value for e in TABLE.entities("tsh", "en") if e.kind == TEST] == ["TSH"]


def test_a_surname_shared_by_two_doctors_is_ambiguous_and_left_as_said():
    found = [e for e in TABLE.entities("ডাক্তার রায় কবে বসবেন", "bn") if e.kind == DOCTOR]
    assert len(found) == 1 and found[0].value is None
    assert set(found[0].candidates) == {"Dr. N. Roy", "Dr. P. Ray"}
    assert "রায়" in TABLE.normalize("ডাক্তার রায় কবে বসবেন")  # not resolved


def test_a_title_is_not_part_of_the_doctor_and_the_whole_name_matches():
    assert [(e.kind, e.value) for e in TABLE.entities("Doctor Arindam Sen please", "en") if e.kind == DOCTOR] == [
        (DOCTOR, "Dr. A. Sen")
    ]
    assert TABLE.normalize("ডাক্তার সেন").split()[0] == "ডাক্তার"  # the title stays in the text


def test_two_tests_with_one_code_fall_back_to_their_names():
    cat = {"tests": [{"name": "HIV Test (ELISA)"}, {"name": "HIV Test (Rapid)"}], "doctors": []}
    table = AliasTable(cat)
    assert table.normalize("hiv test elisa") == "TEST:HIV_Test_(ELISA)"
    assert table.normalize("hiv test rapid") == "TEST:HIV_Test_(Rapid)"


@pytest.mark.parametrize(
    "name,code",
    [
        ("Complete Blood Count (CBC)", "CBC"),
        ("TMT (Treadmill Test)", "TMT"),
        ("HIV Test (ELISA)", "HIV"),
        ("HbA1c", "HbA1c"),
        ("Vitamin D (25-OH)", "Vitamin D (25-OH)"),
        ("2D Echocardiography", "2D Echocardiography"),
        ("Blood Sugar PP", "Blood Sugar PP"),
    ],
)
def test_lab_test_code(name, code):
    assert lab_test_code(name) == code


def test_numbers_are_read_from_digits_and_from_words_in_three_scripts():
    assert numbers_in("book on 25") == [25]
    assert numbers_in("twenty five") == [25]
    assert numbers_in("২৫") == [25]
    assert numbers_in("no numbers here") == []
    assert normalize_text("২৫০ টাকা") == "250 টাকা"


def test_yes_and_no_are_scored_for_short_references_only():
    assert [(e.kind, e.value) for e in TABLE.entities("হ্যাঁ", "bn", reference=True) if e.kind == YES_NO] == [
        (YES_NO, "yes")
    ]
    long_ref = "আমি জানতে চাই যে আপনারা কি রবিবার খোলা থাকেন হ্যাঁ"
    assert not [e for e in TABLE.entities(long_ref, "bn", reference=True) if e.kind == YES_NO]


@given(st.text(max_size=60))
def test_the_comparison_form_is_stable_and_normalizing_never_raises(text):
    TABLE.normalize(text)  # any text at all: no exception
    assert normalize_text(normalize_text(text)) == normalize_text(text)


# ----------------------------------------------------------------------------------------------- metrics


def test_align_splits_the_edits_and_counts_them_correctly():
    e = align("a b c d".split(), "a x c".split())
    assert (e.substitutions, e.deletions, e.insertions, e.reference_length) == (1, 1, 0, 4)
    assert align([], ["x"]).insertions == 1
    assert align(["x"], []).deletions == 1


def test_corpus_wer_is_total_edits_over_total_words_not_a_mean_of_rates():
    short = word_edits("a", "b")  # 1 of 1
    long = word_edits("a b c d e f g h i j", "a b c d e f g h i j")  # 0 of 10
    assert (short + long).rate == pytest.approx(1 / 11)  # a mean of rates would say 0.5


def test_character_error_rate_ignores_spaces():
    assert char_edits("সি বি সি", "সিবিসি").errors == 0


_WORDS = st.lists(st.sampled_from(list("abcde")), max_size=12)


@given(_WORDS, _WORDS)
def test_align_is_a_distance(ref, hyp):
    forward, back = align(ref, hyp), align(hyp, ref)
    assert forward.errors == back.errors
    assert forward.errors >= abs(len(ref) - len(hyp))
    assert forward.errors <= max(len(ref), len(hyp))
    assert (align(ref, ref).errors == 0) and (forward.errors == 0) == (ref == hyp)
    assert forward.reference_length == len(ref)
    assert forward.insertions - forward.deletions == len(hyp) - len(
        ref
    )  # what is added less what is dropped is the change in length


def test_the_interval_resamples_callers_and_is_reproducible():
    by_caller = {f"c{i}": Edits(1 if i % 3 == 0 else 0, 0, 0, 10) for i in range(12)}
    lo, hi = bootstrap_interval(by_caller)
    assert (lo, hi) == bootstrap_interval(by_caller)
    assert 0.0 <= lo <= hi <= 0.1
    assert bootstrap_interval({"only": Edits(1, 0, 0, 10)}) == (0.1, 0.1)  # nothing to resample


def _e(kind, value, candidates=()):
    return Entity(kind, value, str(value), tuple(candidates))


def test_a_wrong_entity_is_counted_apart_from_a_missed_one():
    s = score_entities([_e(TEST, "CBC")], [_e(TEST, "CRP")])[TEST]
    assert (s.said, s.correct, s.missed, s.wrong) == (1, 0, 1, 1)


def test_entities_are_matched_as_multisets():
    s = score_entities([_e(NUMBER, "5")], [_e(NUMBER, "5"), _e(NUMBER, "5")])[NUMBER]
    assert (s.correct, s.wrong) == (1, 1)


def test_an_ambiguous_hearing_is_neither_missed_nor_wrong_when_it_could_be_the_entity():
    s = score_entities([_e(DOCTOR, "Dr. N. Roy")], [_e(DOCTOR, None, ["Dr. N. Roy", "Dr. P. Ray"])])[DOCTOR]
    assert (s.correct, s.missed, s.wrong, s.ambiguous) == (0, 0, 0, 1)
    off = score_entities([_e(DOCTOR, "Dr. A. Sen")], [_e(DOCTOR, None, ["Dr. N. Roy", "Dr. P. Ray"])])[DOCTOR]
    assert (off.missed, off.wrong, off.ambiguous) == (1, 0, 1)


def test_accuracy_over_nothing_said_is_undefined_not_zero():
    assert score_entities([], [])[TEST].accuracy is None


# ---------------------------------------------------------------------------------------------- manifest


def _row(**kw):
    base = {
        "id": "u1",
        "audio": "a.wav",
        "speaker": "c1",
        "split": "test",
        "source": "real",
        "language": "bn",
        "verbatim": "নমস্কার",
        "consent": True,
        "locked": True,
    }
    return {**base, **kw}


def _parse(*rows):
    return am.parse_manifest(json.dumps(r, ensure_ascii=False) for r in rows)


def test_a_valid_row_parses():
    rows, problems = _parse(_row())
    assert len(rows) == 1 and not problems


@pytest.mark.parametrize(
    "bad",
    [
        {"consent": False},  # a real recording needs consent
        {"locked": True, "split": "dev"},  # locked means test
        {"locked": True, "source": "synthetic"},  # synthetic never enters the locked set
        {"tags": ["not_a_tag"]},
        {"split": "validation"},
        {"speaker": ""},
        {"unexpected": 1},  # extra="forbid"
    ],
)
def test_a_bad_row_is_reported_and_skipped(bad):
    rows, problems = _parse(_row(**bad))
    assert not rows and len(problems) == 1


def test_a_duplicate_id_is_reported():
    rows, problems = _parse(_row(), _row())
    assert len(rows) == 1 and "duplicate" in problems[0]


def test_a_caller_in_two_splits_is_reported():
    rows, _ = _parse(_row(), _row(id="u2", split="train", locked=False))
    assert "c1" in am.speaker_split_problems(rows)[0]
    rows, _ = _parse(_row(), _row(id="u2", speaker="c2", split="train", locked=False))
    assert not am.speaker_split_problems(rows)


def test_the_locked_digest_changes_when_the_locked_set_does():
    rows, _ = _parse(_row(), _row(id="u2", speaker="c2"))
    digest = am.locked_digest(rows)
    assert not am.locked_problems(rows, digest)
    fewer = rows[:1]
    assert "changed" in am.locked_problems(fewer, digest)[0]
    assert "no locked" in am.locked_problems([], None)[0]
    unlocked_extra, _ = _parse(_row(), _row(id="u3", speaker="c3", locked=False))
    assert am.locked_digest(unlocked_extra) == am.locked_digest(rows[:1])  # an unlocked row does not move the digest


# ------------------------------------------------------------------------------------------- confusables


def test_two_doctors_with_one_written_surname_are_the_strongest_finding():
    pairs = find_confusables([("Dr. N. Roy", "রায়"), ("Dr. P. Ray", "রায়"), ("Dr. A. Sen", "সেন")])
    assert [(p.a, p.b, p.basis, p.risk) for p in pairs] == [("Dr. N. Roy", "Dr. P. Ray", SAME_WRITTEN, "high")]


def test_sugar_fasting_and_sugar_pp_are_confusable_and_share_a_word():
    entries = [("Blood Sugar Fasting", "Blood Sugar Fasting"), ("Blood Sugar PP", "Blood Sugar PP"), ("CBC", "CBC")]
    pairs = find_confusables(entries)
    assert any({p.a, p.b} == {"Blood Sugar Fasting", "Blood Sugar PP"} and p.basis == SAME_SOUND for p in pairs)
    assert shared_words(entries)["sugar"] == ["Blood Sugar Fasting", "Blood Sugar PP"]


def test_the_word_test_and_spoken_letters_identify_nothing():
    entries = [("A", "সিবিসি টেস্ট"), ("B", "সিআরপি টেস্ট")]
    assert "টেস্ট" not in shared_words(entries)
    assert not shared_words([("A", "বি ভিটামিন"), ("B", "বি সিরাম")])  # a lone spoken letter is not a shared word


_FORMS = st.lists(
    st.tuples(st.sampled_from(["A", "B", "C", "D"]), st.text(alphabet="abcdeklmnrst ", min_size=1, max_size=10)),
    max_size=10,
)


@given(_FORMS)
def test_confusables_never_pair_an_entity_with_itself_and_ignore_input_order(entries):
    pairs = find_confusables(entries)
    assert all(p.a < p.b for p in pairs)
    assert {(p.a, p.b, p.basis) for p in pairs} == {(p.a, p.b, p.basis) for p in find_confusables(reversed(entries))}


def test_the_report_is_built_from_a_catalogue():
    report = confusable_pairs.build_report(CAT)
    doctor_pairs = report["doctors"]["pairs"]
    assert any({p["a"], p["b"]} == {"Dr. N. Roy", "Dr. P. Ray"} and p["risk"] == "high" for p in doctor_pairs)
    text = confusable_pairs.render_markdown(report, "test")
    assert "Names that can be mistaken for each other" in text and "Dr. N. Roy" in text


# ------------------------------------------------------------------------------------------- the command


def _write(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(path)


def _run(tmp_path, manifest, hyps, *extra, capsys):
    cat = tmp_path / "cat.json"
    cat.write_text(json.dumps(CAT, ensure_ascii=False), encoding="utf-8")
    args = [
        "--manifest",
        _write(tmp_path, "m.jsonl", manifest),
        "--hyp",
        _write(tmp_path, "h.jsonl", hyps),
        "--catalogue",
        str(cat),
    ]
    code = asr_eval.main([*args, *extra])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_the_command_scores_and_separates_real_from_synthetic(tmp_path, capsys):
    manifest = [
        _row(id="u1", verbatim="সিবিসি টেস্টের দাম কত"),
        _row(
            id="s1",
            speaker="tts",
            source="synthetic",
            consent=False,
            locked=False,
            verbatim="how much is the CBC test",
            language="en",
        ),
    ]
    hyps = [{"id": "u1", "hyp": "সি বি সি টেস্টের দাম কত"}, {"id": "s1", "hyp": "how much is the CRP test"}]
    code, out, _ = _run(tmp_path, manifest, hyps, capsys=capsys)
    assert code == 0
    real, synthetic = out.split("SYNTHETIC utterances")
    assert "REAL utterances" in real and "NOT a benchmark" in synthetic
    assert "WRONG" in real
    # the spelled-out CBC is the same test: the real utterance has a correct test and no wrong one
    real_test = next(line for line in real.splitlines() if line.strip().startswith("test "))
    assert real_test.split()[1:6] == ["1", "1", "0", "0", "0"]
    synth_test = next(line for line in synthetic.splitlines() if line.strip().startswith("test "))
    assert synth_test.split()[1:6] == ["1", "0", "1", "1", "0"]  # CBC said, CRP heard: missed AND wrong


def test_the_command_refuses_when_a_caller_is_in_two_splits(tmp_path, capsys):
    manifest = [_row(id="u1"), _row(id="u2", split="train", locked=False)]
    code, _out, err = _run(tmp_path, manifest, [{"id": "u1", "hyp": "x"}], capsys=capsys)
    assert code == 2 and "exactly one split" in err


def test_the_command_refuses_when_a_hypothesis_is_missing(tmp_path, capsys):
    code, _out, err = _run(
        tmp_path, [_row(id="u1"), _row(id="u2", speaker="c2")], [{"id": "u1", "hyp": "x"}], capsys=capsys
    )
    assert code == 2 and "no hypothesis" in err


def test_the_command_refuses_when_the_locked_set_changed(tmp_path, capsys):
    lock = str(tmp_path / "lock.sha")
    two = [_row(id="u1"), _row(id="u2", speaker="c2")]
    hyps = [{"id": "u1", "hyp": "x"}, {"id": "u2", "hyp": "y"}]
    assert _run(tmp_path, two, hyps, "--lock", lock, capsys=capsys)[0] == 0  # records the digest
    assert _run(tmp_path, two, hyps, "--lock", lock, capsys=capsys)[0] == 0  # same set: fine
    code, _out, err = _run(tmp_path, two[:1], hyps[:1], "--lock", lock, capsys=capsys)
    assert code == 2 and "locked test set has changed" in err
