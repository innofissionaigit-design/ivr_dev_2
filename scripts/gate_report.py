#!/usr/bin/env python3
"""Turns gate.sh's raw check results into gate-report.json and
gate-report.md -- the machine-readable evidence
Pre-Human-Review-Quality-Gate-Blueprint.docx section 2.6/4.4 requires.
"Claude says it passed" is never evidence; this file is.

Usage: gate_report.py <status-file> --mode --fast|--full
           --out-json gate-report.json --out-md gate-report.md

<status-file> is tab-separated lines: name<TAB>pass|fail|skip<TAB>detail
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def parse_status_file(path: str) -> dict[str, dict[str, str]]:
    checks: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            name = parts[0]
            status = parts[1] if len(parts) > 1 else "unknown"
            detail = parts[2] if len(parts) > 2 else ""
            checks[name] = {"status": status, "detail": detail}
    return checks


def build_report(checks: dict[str, dict[str, str]], mode: str) -> dict:
    mandatory_statuses = [c["status"] for c in checks.values()]
    any_fail = "fail" in mandatory_statuses
    any_skip = "skip" in mandatory_statuses

    if any_fail:
        overall = "RED"
    elif any_skip:
        overall = "GREEN_WITH_GAPS"
    else:
        overall = "GREEN"

    changed_files = git("diff", "--name-only", "HEAD").splitlines()
    if not changed_files:
        changed_files = git("diff", "--name-only", "--cached", "HEAD").splitlines()

    high_risk_globs = ("agent/reply_templates.py", "agent/tools_client.py", "agent/llm.py", "clinic-api/", "deploy/")
    high_risk_files = [f for f in changed_files if f.startswith(high_risk_globs)]

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_by": "scripts/gate.sh",
        "checks": checks,
        "files_changed": len(changed_files),
        "high_risk_files": high_risk_files,
        "gate_config_touched": any(f.startswith("scripts/gate") or f == "pytest.ini" for f in changed_files),
        "tests_weakened": False,  # requires a human/AI-review judgment call; see code-reviewer.md
        "overall_status": overall,
        "note": (
            "GREEN_WITH_GAPS means every implemented check passed but at "
            "least one check is not yet implemented (see 'skip' entries) -- "
            "never present this as equivalent to full GREEN in a PR description."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# Gate report -- {report['overall_status']}",
        "",
        f"- commit: `{report['commit']}` on `{report['branch']}`",
        f"- mode: `{report['mode']}`",
        f"- generated: {report['generated_at']}",
        f"- files changed: {report['files_changed']}",
    ]
    if report["high_risk_files"]:
        lines.append(f"- **high-risk files touched:** {', '.join(report['high_risk_files'])}")
    if report["gate_config_touched"]:
        lines.append(
            "- ⚠️ **gate config or pytest.ini was touched in this diff -- "
            "requires explicit human sign-off, per CLAUDE.md section 4.4.**"
        )
    lines += ["", "| Check | Status | Detail |", "|---|---|---|"]
    for name, info in sorted(report["checks"].items()):
        icon = {"pass": "✅", "fail": "❌", "skip": "⚪"}.get(info["status"], "?")
        detail = info["detail"].replace("|", "/")[:200]
        lines.append(f"| {name} | {icon} {info['status']} | {detail} |")
    lines += ["", report["note"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("status_file")
    ap.add_argument("--mode", default="--fast")
    ap.add_argument("--out-json", default="gate-report.json")
    ap.add_argument("--out-md", default="gate-report.md")
    args = ap.parse_args()

    checks = parse_status_file(args.status_file)
    report = build_report(checks, args.mode)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))

    print(f"Wrote {args.out_json} and {args.out_md} (overall: {report['overall_status']})")
    return 0 if report["overall_status"] != "RED" else 1


if __name__ == "__main__":
    sys.exit(main())
