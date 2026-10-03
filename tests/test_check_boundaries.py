"""scripts/check_boundaries.py enforces CLAUDE.md's vendor-boundary rule. It must fire on a real SDK import and stay quiet
about the word in prose: it used to report "an external OpenAI review" in a comment as a leaked vendor SDK, three false
positives that made the gate fail for no reason (2026-09-26).

    python -m pytest tests/test_check_boundaries.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import check_boundaries as cb  # noqa: E402


@pytest.mark.parametrize(
    "source,module",
    [
        ("import openai\n", "openai"),
        ("import openai.types as t\n", "openai.types"),
        ("from openai import OpenAI\n", "openai"),
        ("from anthropic import Anthropic\n", "anthropic"),
        ("import google.generativeai as genai\n", "google.generativeai"),
        ("from google.cloud import aiplatform\n", "google.cloud"),
        ("import vertexai\n", "vertexai"),
        ("mod = __import__('openai')\n", "openai"),
        ("import importlib\nmod = importlib.import_module('anthropic')\n", "anthropic"),
        ("def f():\n    import openai\n    return openai\n", "openai"),
    ],
)
def test_a_real_vendor_sdk_import_is_reported(source, module):
    assert cb.vendor_uses(source), source
    assert any(module.split(".")[0] in found for found in cb.vendor_uses(source))


@pytest.mark.parametrize(
    "source",
    [
        "# (OpenAI review; CLAUDE.md 'Doctor Nobody')\nx = 1\n",
        '"""Docstring: an external OpenAI review and an Anthropic paper."""\n',
        'note = "the OpenAI review"\n',
        "def openai_style(): ...\n",
        "import json\nfrom agent import llm\n",
        "openai_compatible = True\n",
        "from mypackage.anthropic_notes import x\n",
    ],
)
def test_the_word_in_prose_or_a_name_is_not_a_vendor_sdk(source):
    assert cb.vendor_uses(source) == []


def test_a_file_that_does_not_parse_is_still_checked_by_its_import_lines():
    assert cb.vendor_uses("import openai\nthis is not python (\n") == ["openai"]
    assert cb.vendor_uses("# openai\nthis is not python (\n") == []


def test_the_repository_itself_passes_both_boundary_rules():
    assert cb.check_one_import_rule() == []
    assert cb.check_vendor_boundary() == []
