"""KCD-454: "No field label, colon or bracket is ever spoken... An
automated check fails a build containing a spoken punctuation artefact."

This is that automated check. It scans agent/reply_templates.py and
agent/reply_templates_i18n.py's own SOURCE for string literals carrying
a colon or bracket (agent/spoken_text_lint.py), which covers every reply
branch without needing to invoke each function with fixture data.

RATCHET, not a free pass: reply_templates.py currently ships a real
backlog of "label: value" phrasing (test_rate_reply's "স্যাম্পল: {sample}"
etc.) that has NOT been reworded yet -- deliberately, since the exact
replacement wording is the caller's own call, not this session's. The
baseline counts below are that backlog's exact size today. This test
still does its job either way:
  - a NEW artefact anywhere pushes a file's count above its baseline and
    fails the build, same as the story asks for.
  - fixing an item in the backlog lowers the real count below the
    baseline and ALSO fails -- on purpose, so the fix has to lower the
    baseline in the same change, not silently leave slack for a future
    regression to hide in.

    python -m pytest tests/test_spoken_text_lint.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.spoken_text_lint import find_artifacts, has_spoken_artifact, scan_source_for_artifacts

# Pending-rework backlog size, per file -- see module docstring. Lower
# this (never raise it) as reply_templates.py/reply_templates_i18n.py
# get reworded to natural clauses.
_KNOWN_BACKLOG = {
    "agent/reply_templates.py": 13,  # lowered from 14: the "sample: value" phrasing became a sentence (sample_wording.py)
    "agent/reply_templates_i18n.py": 26,  # lowered from 28: Hindi and English "sample: value" became sentences
}


def test_has_spoken_artifact_detects_colon_and_bracket():
    assert has_spoken_artifact("কনফার্মেশন নম্বর: 12345") is True
    assert has_spoken_artifact("[not found]") is True
    assert has_spoken_artifact("আপনার রেট পাঁচশো টাকা।") is False


def test_find_artifacts_lists_every_offending_character():
    assert set(find_artifacts("label: [x]")) == {":", "[", "]"}


def test_regex_pattern_literals_are_not_flagged_as_spoken_text():
    # "[A-Za-z]" is source code for re.compile, never spoken to a caller.
    source = 'import re\n_RE = re.compile(r"[A-Za-z]")\n'
    assert scan_source_for_artifacts(source) == []


def test_docstrings_are_not_flagged_as_spoken_text():
    source = '''
def f(x):
    """Explains something: with a colon in the prose."""
    return "ok"
'''
    assert scan_source_for_artifacts(source) == []


@pytest.mark.parametrize("relpath", sorted(_KNOWN_BACKLOG))
def test_reply_templates_do_not_exceed_the_known_artefact_backlog(relpath):
    source = open(os.path.join(REPO_ROOT, relpath), encoding="utf-8").read()
    hits = scan_source_for_artifacts(source)
    baseline = _KNOWN_BACKLOG[relpath]
    assert len(hits) == baseline, (
        f"{relpath}: {len(hits)} spoken-punctuation artefacts found, expected exactly {baseline}. "
        f"If you fixed one, lower _KNOWN_BACKLOG[{relpath!r}] to match. "
        f"If this is higher, a NEW artefact was introduced -- fix the reply text, not this number."
    )
