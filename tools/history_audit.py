"""Audit a call's spoken history against the records it was drawn from (KCD-500, KCD-501).

Every historical sentence the agent speaks is a template filled from a retrieved field, and
the call record (agent/call_record.py) stores the id of the field each one came from. This
tool checks that claim after the fact -- the audit the hallucination story asks for:

  * every statement id is either a fixed "cannot see / need to verify / ask" label, or the id
    of an event that is actually in the timeline that was retrieved for that call;
  * a `due:` statement exists only if a clinician-approved interval was retrieved;
  * nothing is recorded as spoken that the timeline could not have produced.

An ORPHAN is a statement id with no origin: the agent said something about the caller's
history that no retrieved field supports. A call with any orphan fails the audit.

    python tools/history_audit.py CALL_EVENTS.json TIMELINE.json [--statuses STATUS.json]

CALL_EVENTS.json   the call's record: a list of {"seq", "kind", "payload"}
TIMELINE.json      the response of GET /patients/{ref}/timeline for that call
STATUS.json        optional list of test-status responses retrieved during the call

Exit status 0 when every call is clean, 1 otherwise. `audit()` is the importable form.
"""

import argparse
import json
import sys

# Fixed statements that carry no patient data. They are safe by construction and need no origin.
FIXED_LABELS = frozenset(
    {
        "cannot_see",
        "cannot_confirm",
        "ambiguous",
        "needs_verification",
        "no_record",
        "ask_phone",
        "ask_which_test",
        "ok_anything_else",
        # the security-question conversation (agent/security_check.py): fixed sentences, no patient data
        "security_question",
        "security_verified",
        "security_failed",
        "security_not_understood",
        "security_not_matched",
        "senior_opening",
        "continuity_offer",
    }
)


def audit(events: list[dict], timeline: dict | None, statuses: list[dict] | None = None) -> dict:
    """-> {"spoken": n, "grounded": n, "fixed": n, "orphans": [ids], "clean": bool}"""
    known = {e["id"] for e in (timeline or {}).get("events", [])}
    due_known = any(s.get("known") and s.get("due") is not None for s in (statuses or []))
    spoken = grounded = fixed = 0
    orphans: list[str] = []
    for ev in events:
        if ev.get("kind") != "history":
            continue
        for sid in (ev.get("payload") or {}).get("statements", []):
            spoken += 1
            if sid in FIXED_LABELS:
                fixed += 1
            elif sid in known:
                grounded += 1
            elif sid in ("due:yes", "due:no") and due_known:
                grounded += 1
            else:
                orphans.append(sid)
    return {"spoken": spoken, "grounded": grounded, "fixed": fixed, "orphans": orphans, "clean": not orphans}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events")
    ap.add_argument("timeline")
    ap.add_argument("--statuses")
    args = ap.parse_args(argv)
    with open(args.events, encoding="utf-8") as f:
        events = json.load(f)
    with open(args.timeline, encoding="utf-8") as f:
        timeline = json.load(f)
    statuses = None
    if args.statuses:
        with open(args.statuses, encoding="utf-8") as f:
            statuses = json.load(f)
    result = audit(events, timeline, statuses)
    print(json.dumps(result, indent=2))
    return 0 if result["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
