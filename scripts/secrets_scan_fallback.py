#!/usr/bin/env python3
"""A minimal, dependency-free secrets scan for when gitleaks isn't
installed. This is a FALLBACK, not a replacement -- install gitleaks (or
whatever Epic E20/E24 standardises on) for real coverage; this only
catches the shapes a handful of regexes can catch.

Known finding this exists to prevent a repeat of:
clinic-api/setup_db.sh once shipped `DB_PASS="${DB_PASS:-kcd_app_pw}"` --
an overridable shell default, not a live credential, but exactly the shape
this scan should flag for a human to confirm.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic_api_key_assignment": re.compile(
        r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
    "private_key_block": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "hardcoded_db_default": re.compile(r"DB_PASS\s*=\s*[\"']?\$\{?DB_PASS:-[^}\"']+"),
}

SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".jpeg", ".wav", ".pdf", ".zip", ".md"}
# .md is skipped for THIS fallback scan only: docs (HANDOVER.md, ADRs) discuss
# known issues like the setup_db.sh default in prose, which reads as a false
# positive to a regex that can't tell prose from code. Real secrets scanning
# of docs is gitleaks's job (Epic E20/E24), not this stdlib fallback's.
SKIP_FILES = {pathlib.Path(__file__).resolve()}  # don't flag this file's own patterns


def iter_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() in SKIP_FILES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        yield path


def main() -> int:
    findings = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {name}")

    if findings:
        print("Possible secrets found (fallback scan -- verify each by hand):")
        for f in findings:
            print(f"  - {f}")
        return 1
    print("Fallback secrets scan: no matches. This is NOT a substitute for gitleaks/E20's real scan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
