# CLAUDE.md — Kolkata Care Voice Agent

Read this before touching code. It is short on purpose; `HANDOVER.md` and the
documents it points to carry the detail. This file states what every session
— human or agent — must not skip.

## 1. The one rule

> **The LLM may decide what the caller wants. It may never decide what is true.**

Every price, date, chamber time and confirmation number is a **template
substitution from a verified system response** (`agent/reply_templates.py`).
The model is never shown a number it could misremember or invent. Two real
bugs are the reason this rule exists — a garbled fragment becoming
**"Naloxone"** in the sibling project, and **"Doctor Nobody"** once
fuzzy-matching to a real doctor's real schedule. If a change lets the model
emit a fact, stop and redesign; do not ship it.

Two rules that generalise it, also non-negotiable:

- **One-import rule.** Nothing in `agent/` imports the orchestrator
  (`main.py` / `main_pcm.py`). It is the seam that keeps every module
  testable and replaceable.
- **No vendor SDK outside `providers/`.** The moment a vendor name appears
  in `agent/`, the vendor boundary has leaked.

## 2. Authority order when documents disagree

1. `D:\Kolkata-Care-Voice-Agent-Production-Grade-Solution-Blueprint.docx` —
   governance, the twelve controls, Phases 0–7. Highest authority.
2. `docs/adr/*.md` in this repo — accepted decisions that refine the
   Blueprint for this pilot (e.g. `0001-pilot-single-l4-architecture.md`).
   An ADR may revise capacity/placement; it may never revise the truth
   boundary or the governance model without a new, explicit ADR saying so.
3. `Kolkata-Care-Voice-Agent-Backlog-Sprints-and-Gates.xlsx` — the epics,
   stories, Definition of Done, and CI Sanity Gates. This is the actual
   work breakdown; "epic-wise development" means working this backlog,
   epic by epic, against its Definition of Done sheet.
4. `HANDOVER.md` in this repo — what is actually true about the code today,
   verified against a live pod. Re-verify anything it claims before relying
   on it if more than a few weeks have passed since its date.
5. The code (`github.com/sayantanIFAI/ivr`) — reality. When in doubt about
   what the system does today, read the code, don't infer it from a doc.

## 3. Before you write any code

- Read `HANDOVER.md` in full. It names live defects, infrastructure traps
  that already cost a day each, and exactly where Phase 0 starts.
- Check whether a `docs/adr/*.md` already covers the decision you're about
  to make. If your change contradicts one, write a new ADR that supersedes
  it — don't silently diverge.
- If you are about to touch `agent/asr.py`, `agent/vad_stream.py`, or
  anything RunPod/`/workspace`-related: **read `deploy/env.sh` and
  `deploy/start_all.sh` first.** They encode real, previously-hit failures
  (RunPod wipes everything outside `/workspace`; Postgres cannot run on the
  `/workspace` MooseFS mount; kill by port, never by process-name pattern).
  Do not re-derive these the hard way.
- Never assume a claim about live behaviour (latency, VRAM, "it works") is
  current. `HANDOVER.md` labels numbers as measured or reasoned for exactly
  this reason — a reasoned number is a to-do, not a setting. If you can
  check it against the live pod or a local test, check it before stating it.

## 4. Before requesting human review, you MUST

This mirrors `Pre-Human-Review-Quality-Gate-Blueprint.docx` and the
project's own **Definition of Done** / **CI Sanity Gates** sheets in the
backlog spreadsheet — read those two sheets if this summary is not enough.

1. Run `bash scripts/gate.sh --full` (Git Bash on Windows works fine off-pod;
   this is the same script that runs in the PostToolUse/Stop hooks and,
   once Epic E24 lands, in CI — one definition, every call site). All
   checks that are implemented must pass; checks not yet implemented are
   reported as such, never silently skipped.
2. If a check fails: diagnose, fix, re-run. **Maximum 3 remediation cycles
   per failing check.**
3. On the 4th failure: **STOP.** Post the diff and the failure log. Do not
   keep iterating, and do not lower a threshold or weaken a test to get to
   green.
4. **Never** edit `scripts/gate*`, CI configs, or test files to make a
   check pass. Never add a lint-ignore / type-ignore / threshold change to
   go green. Fix the code, not the gauge — this is enforced on the diff in
   CI once Epic E24 lands the pipeline; treat it as already in force.
5. Paste `gate-report.md` into the pull request description.

Claude's or any session's statement that the gate passed is **not
evidence**. CI is the authoritative verifier once it exists; until then,
the local `gate.sh --full` run is the closest thing to authoritative and
must be shown, not asserted.

## 5. The four CI Sanity Gate markers

Registered in `pytest.ini`, target state per the backlog's **CI Sanity
Gates** sheet — most have no real tests yet (tracked as Epic E24) and
`gate.sh` reports that honestly rather than treating "0 collected" as a
pass:

- **degradation** — per-bucket accuracy against the locked golden set;
  strong English must never compensate for weak Bengali.
- **latency** — per-stage timing against the Appendix B budget on
  deterministic stubs (no GPU, no network).
- **security** — no secret-shaped literal, no personal identifier in a log
  or provider payload, no unauthenticated path to patient data, zero
  successful adversarial injections.
- **hallucination** — every factual intent is a template substitution, no
  model-composed span; unknown doctor / unknown test produce the not-found
  path, never a guess. `tests/test_smoke.py` already carries two guards in
  this family (the "Naloxone-class" regression tests) — extend that file
  rather than starting a parallel suite.

## 6. Engineering roles available in this repo

Four subagents live in `.claude/agents/`, grounded in this codebase's own
conventions (not generic personas):

- **senior-python-architect** — design and restructuring decisions, ADRs,
  epic-boundary calls. Use before a change that crosses `agent/`, `tools/`,
  `policy/`, `responses/` boundaries or touches the vendor boundary.
- **python-developer** — implements one story against its acceptance
  criteria, in this codebase's house style (see `agent/asr.py`,
  `agent/lid.py` for the pattern: `from __future__ import annotations`,
  dataclasses, a module docstring that explains *why*, comments only where
  the reason is non-obvious).
- **python-debugger** — root-causes a defect against the live system, not
  the docstring. Verifies on the pod (or with a reproducing local test)
  before proposing a fix.
- **code-reviewer** — the fresh-context, pre-human-review pass (G2 in
  `Pre-Human-Review-Quality-Gate-Blueprint.docx`): re-derives correctness
  from the spec and the diff alone, blocks on unresolved Critical/High,
  never approves release and never re-runs the deterministic gate's job.

Use `senior-python-architect` before starting a new epic's first story,
`python-developer` for the implementation, `python-debugger` when
something that used to pass stops passing, and `code-reviewer` before
opening the pull request — in addition to, not instead of, a human peer
review and CodeRabbit.

## 7. Repository state note

No RunPod pod is running as of the last handover (`HANDOVER.md` §12).
`agent/lid.py`, `agent/asr_router.py` and `agent/tts_router.py`
(`docs/adr/0001-pilot-single-l4-architecture.md`) were written and unit
tested **without** a pod — their routing logic is verified; their model
loading paths (SpeechBrain, the Hindi/English ASR and TTS checkpoints) are
not, and are marked as such in their docstrings. Verify those on first pod
connect before building further on top of them.
