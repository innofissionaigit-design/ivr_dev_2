#!/usr/bin/env bash
# The single source of truth for "is this change ready for review."
# Same script for: the PostToolUse hook (--fast), the Stop hook and a
# developer's own pre-PR run (--full), and CI once Epic E24 lands the
# pipeline. One definition, every call site --
# Pre-Human-Review-Quality-Gate-Blueprint.docx section 2.4.
#
# Degrades honestly: a check whose tooling isn't installed yet, or whose
# suite doesn't exist yet (the four CI Sanity Gate markers, pre-Epic-E24),
# is reported as "skipped: <reason>" in gate-report.json/.md -- never
# silently treated as a pass. See CLAUDE.md section 4-5 before changing
# this file's pass/fail semantics.
#
# NEVER edit this file, or a test's assertions, to make a run go green.
# Fix the code. If you believe the gate itself is wrong, raise it with a
# human -- do not route around it (CLAUDE.md section 4, item 4).
set -uo pipefail

MODE="${1:---fast}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
STATUS_FILE="$(mktemp)"
trap 'rm -f "$STATUS_FILE"' EXIT

overall_pass=true

# check <name> <shell-command...>
# Records pass/fail/skip to STATUS_FILE as "name<TAB>status<TAB>detail".
check() {
  local name="$1"; shift
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ]; then
    printf '%s\tpass\t\n' "$name" >> "$STATUS_FILE"
    echo "[pass] $name"
  else
    overall_pass=false
    printf '%s\tfail\t%s\n' "$name" "$(echo "$out" | tail -5 | tr '\n' '|')" >> "$STATUS_FILE"
    echo "[FAIL] $name"
    echo "$out" | tail -20 | sed 's/^/       /'
  fi
}

skip() {
  local name="$1" reason="$2"
  printf '%s\tskip\t%s\n' "$name" "$reason" >> "$STATUS_FILE"
  echo "[skip] $name -- $reason"
}

echo "== gate.sh $MODE  ($(date -u +%Y-%m-%dT%H:%M:%SZ)) =="

# --- format / lint / typecheck -------------------------------------------
if command -v ruff >/dev/null 2>&1; then
  check "format" ruff format --check agent tools clinic-api tests
  check "lint" ruff check agent tools clinic-api tests
else
  skip "format" "ruff not installed -- pip install ruff (tracked: Epic E24)"
  skip "lint" "ruff not installed -- pip install ruff (tracked: Epic E24)"
fi

if command -v mypy >/dev/null 2>&1; then
  check "typecheck" mypy agent --ignore-missing-imports
else
  skip "typecheck" "mypy not installed -- pip install mypy (tracked: Epic E24)"
fi

# --- one-import / vendor-boundary check -----------------------------------
# Cheap, dependency-free, and important enough to run even with nothing
# else installed: nothing in agent/ may import the orchestrator, and no
# vendor SDK name may appear outside providers/ (CLAUDE.md section 1).
check "boundary_rules" "$PY" scripts/check_boundaries.py

# --- unit/integration tests that need no pod, no GPU, no live pod services
# (test_clinic_api_new_endpoints.py runs clinic-api against a throwaway
# local SQLite file via FastAPI's TestClient -- that's "local", not "the
# pod", so it belongs here, not in the pod-only smoke_suite below).
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  check "unit_and_local_integration" "$PY" -m pytest \
    tests/test_lid.py tests/test_routers.py \
    tests/test_fast_path_faq_prep.py tests/test_clinic_api_new_endpoints.py -q
else
  skip "unit_and_local_integration" "pytest not installed -- pip install pytest pytest-asyncio httpx fastapi sqlalchemy"
fi

# --- pod-only suite: skip off-pod, never fake a result --------------------
if [ -n "${VOICE_AGENT_ON_POD:-}" ]; then
  check "smoke_suite" "$PY" -m pytest tests/test_smoke.py -q
else
  skip "smoke_suite" "needs live services on the pod -- set VOICE_AGENT_ON_POD=1 and run there (HANDOVER.md section 4)"
fi

# --- secrets scan (cheap regex fallback if no dedicated tool) -------------
if command -v gitleaks >/dev/null 2>&1; then
  check "secrets_scan" gitleaks detect --no-banner --source . -v
else
  check "secrets_scan_fallback" "$PY" scripts/secrets_scan_fallback.py
fi

# --- the four CI Sanity Gate markers --------------------------------------
# Registered in pytest.ini. Real suites are Epic E24; until they exist,
# report the honest gap rather than a false pass. pytest exits 5 for
# "no tests collected" -- that is a skip, not a failure, until E24 adds
# tests under these markers; any other non-zero exit is a real failure.
for marker in degradation latency security hallucination; do
  if "$PY" -c "import pytest" >/dev/null 2>&1; then
    out="$("$PY" -m pytest -m "$marker" -q 2>&1)"; rc=$?
    if [ "$rc" -eq 5 ]; then
      skip "marker_${marker}" "0 tests collected -- suite not yet implemented (Epic E24)"
    elif [ "$rc" -eq 0 ]; then
      printf 'marker_%s\tpass\t\n' "$marker" >> "$STATUS_FILE"
      echo "[pass] marker_${marker}"
    else
      overall_pass=false
      printf 'marker_%s\tfail\t%s\n' "$marker" "$(echo "$out" | tail -5 | tr '\n' '|')" >> "$STATUS_FILE"
      echo "[FAIL] marker_${marker}"
      echo "$out" | tail -20 | sed 's/^/       /'
    fi
  else
    skip "marker_${marker}" "pytest not installed"
  fi
done

# --- full mode only: everything else that matters before a PR ------------
if [ "$MODE" = "--full" ]; then
  if [ -f main.py ]; then
    check "orchestrator_parses" "$PY" -c \
      "import ast; ast.parse(open('main.py', encoding='utf-8').read())"
  fi
fi

echo "=================================================="
"$PY" scripts/gate_report.py "$STATUS_FILE" --mode "${MODE#--}" \
  --out-json gate-report.json --out-md gate-report.md

if [ "$overall_pass" = true ]; then
  echo "RESULT: green (mandatory checks pass; see gate-report.md for skips)"
  exit 0
else
  echo "RESULT: red -- see gate-report.md"
  exit 1
fi
