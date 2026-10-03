#!/usr/bin/env python3
"""Enforces CLAUDE.md's two non-negotiable structural rules, mechanically:

  1. Nothing under agent/ may import the orchestrator (main / main_pcm).
  2. No vendor SDK name may appear outside providers/ (which does not
     exist yet -- see docs/adr/0001 and Blueprint Part 7 / Epic E07).

Deliberately dependency-free (stdlib only) so it runs in every gate.sh
call site with nothing to install.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENT_DIR = REPO_ROOT / "agent"
PROVIDERS_DIR = REPO_ROOT / "providers"

FORBIDDEN_ORCHESTRATOR_IMPORTS = re.compile(r"^\s*(from|import)\s+(main|main_pcm)\b", re.MULTILINE)

# Names that, if imported or referenced outside providers/, indicate the
# vendor boundary has leaked. Extend this list as real providers land
# (Epic E07); it starts narrow and explicit on purpose -- a broad regex
# here just trains everyone to ignore false positives.
VENDOR_SDK_MODULES = ("openai", "anthropic", "google.generativeai", "google.cloud.aiplatform", "vertexai")
VENDOR_SDK_MARKERS = [rf"{re.escape(m)}" for m in VENDOR_SDK_MODULES]  # kept for callers that read the old name
VENDOR_PATTERN = re.compile("|".join(VENDOR_SDK_MARKERS), re.IGNORECASE)


def _is_vendor(module: str) -> bool:
    return any(module == m or module.startswith(m + ".") for m in VENDOR_SDK_MODULES)


def vendor_uses(source: str) -> list[str]:
    """The vendor SDK modules this source actually IMPORTS: `import openai`, `from anthropic import X`, and the dynamic
    forms `__import__("openai")` / `importlib.import_module("openai")`. The bare word in a comment, a docstring or a
    string ("an external OpenAI review") is not an SDK and is not reported -- a check that fires on prose only teaches
    people to ignore it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # cannot parse: fall back to import-shaped lines only
        return [
            m.group(1)
            for m in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", source, re.MULTILINE)
            if _is_vendor(m.group(1))
        ]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if _is_vendor(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if _is_vendor(node.module) or any(_is_vendor(f"{node.module}.{a.name}") for a in node.names):
                found.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in ("__import__", "import_module") and _is_vendor(node.args[0].value):
                found.append(node.args[0].value)
    return found


def check_one_import_rule() -> list[str]:
    violations = []
    if not AGENT_DIR.exists():
        return violations
    for path in AGENT_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if FORBIDDEN_ORCHESTRATOR_IMPORTS.search(text):
            violations.append(f"{path.relative_to(REPO_ROOT)}: imports the orchestrator")
    return violations


def check_vendor_boundary() -> list[str]:
    violations = []
    for path in REPO_ROOT.rglob("*.py"):
        if PROVIDERS_DIR in path.parents:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if "site-packages" in path.parts or "venv" in path.parts or ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for module in vendor_uses(text):
            violations.append(f"{path.relative_to(REPO_ROOT)}: imports vendor SDK '{module}' outside providers/")
    return violations


def main() -> int:
    violations = check_one_import_rule() + check_vendor_boundary()
    if violations:
        print("Boundary violations found:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("Boundary rules OK: no orchestrator import in agent/, no vendor SDK outside providers/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
