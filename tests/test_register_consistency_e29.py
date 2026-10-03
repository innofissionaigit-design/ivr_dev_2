"""KCD-440: every template uses the respectful second person consistently.

This cannot self-certify the acceptance criterion in full -- "reviewed and
signed off by a native speaker per language" needs an actual native
speaker, which no amount of testing substitutes for. What this DOES check,
automatically, on every run: no INFORMAL pronoun (Bengali তুমি/তুই,
Hindi तू/तुम) has crept into caller-facing text, across every phrase and
reply-template source file. A mixed register within one call is called
out in the story as a defect in its own right -- this is the mechanical
half of catching that.

    python -m pytest tests/test_register_consistency_e29.py -v
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_SOURCE_FILES = [
    REPO_ROOT / "agent" / "phrases.py",
    REPO_ROOT / "agent" / "reply_templates.py",
    REPO_ROOT / "agent" / "reply_templates_i18n.py",
]

_INFORMAL_BENGALI = re.compile(r"তুমি|তুই|তোমার|তোমাকে|তোর|তোকে")
# Word-boundary guarded: তুম/तुम etc. can appear as a SUBSTRING of an
# unrelated formal word in Devanagari conjuncts, so these are checked as
# whole tokens only.
_INFORMAL_HINDI = re.compile(r"(?<![ऀ-ॿ])(तू|तुम|तुझे|तेरा|तेरी|तुझसे)(?![ऀ-ॿ])")


def test_no_informal_pronoun_in_any_caller_facing_template():
    violations = []
    for path in _SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        for m in _INFORMAL_BENGALI.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            violations.append(f"{path.name}:{line_no}: informal Bengali {m.group()!r}")
        for m in _INFORMAL_HINDI.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            violations.append(f"{path.name}:{line_no}: informal Hindi {m.group()!r}")
    assert not violations, "\n".join(violations)
