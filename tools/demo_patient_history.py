"""Run the sample patients through the registry, the security questions and the history answers, and PRINT
what happens -- so the data and the rules can be seen working (KCD-493..500, KCD-512).

    python tools/demo_patient_history.py                 # a throw-away clinic API with the sample data
    python tools/demo_patient_history.py --url http://localhost:8080 [--token-file /workspace/.clinic_api_token]

For each sample patient: the answers a caller would give; whether the server accepts them; the age and
whether the registry says senior; what the agent would say in Bengali, Hindi and English about tests,
medicines and appointments (rendered by the same templates the agent uses, from the retrieved events);
what the agent says when nothing is found; a refused read before verification; and a wrong-answer
attempt. Nothing here is a real person: see clinic-api/patient_seed.py.
"""

import argparse
import datetime
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tests")):
    if p not in sys.path:
        sys.path.append(p)

from agent import history_templates as ht
from agent import patient_context as pc

# what each sample caller says (patient id, date of birth, name)
CALLERS = [
    ("KCP-100004", "Debashis Mondal", "1985-06-30", "Debashis Mondal"),
    ("KCP-100001", "Asha Saha (a senior)", "1948-03-14", "Asha Saha"),
    ("KCP-100005", "Sunita Devi (a senior)", "1962-01-09", "Sunita Devi"),
    ("KCP-100006", "Ravi Kumar", "1990-12-05", "Ravi Kumar"),
]
NOW = datetime.datetime.now()


class Api:
    """The few calls the demo needs, over a TestClient or a live URL."""

    def __init__(self, client, prefix=""):
        self.c, self.prefix = client, prefix

    def get(self, path, **params):
        return self.c.get(self.prefix + path, params=params).json()

    def post(self, path, **body):
        return self.c.post(self.prefix + path, json=body).json()


def show(title):
    print(f"\n=== {title}")


def run(api: Api) -> int:
    failures = 0
    for uid, label, dob, name in CALLERS:
        show(f"{label}  [{uid}]")
        call = f"demo-{uid}"
        found = api.post("/api/v1/patients/find", call_id=call, patient_id=uid)
        ref = found["patient_ref"]
        print(f"  find by patient id      -> {found}")
        refused = api.get(f"/api/v1/patients/{ref}/timeline", caller_phone="-", call_id=call)
        print(f"  history BEFORE verifying -> {refused}")
        failures += refused.get("reason") != "not_verified"
        bad = api.post("/api/v1/patients/verify", call_id=call + "-bad", patient_ref=ref, dob="1970-01-01", name=name)
        print(f"  wrong date of birth      -> {bad}   (never says which answer was wrong)")
        ok = api.post("/api/v1/patients/verify", call_id=call, patient_ref=ref, dob=dob, name=name)
        print(f"  right answers            -> {ok}")
        failures += not ok.get("verified")
        tl = api.get(f"/api/v1/patients/{ref}/timeline", caller_phone="-", call_id=call)
        for lang, lab in (("en", "English"), ("bn", "Bengali"), ("hi", "Hindi")):
            print(f"  [{lab}]")
            for fn, what in (
                (lambda: pc.answer_recent_tests(tl, lang, NOW), "tests done"),
                (lambda: pc.answer_medicines(tl, lang, NOW), "medicines"),
                (lambda: pc.answer_appointments(tl, lang, NOW), "next appointment"),
            ):
                print(f"    {what:17s} {fn().text}")
        st = api.get(f"/api/v1/patients/{ref}/test-status", test_name="HbA1c", caller_phone="-", call_id=call)
        if st.get("known"):
            due = ht.due_statement(st, "en")
            print(
                f"  HbA1c last done {st['last_performed_on']}; "
                f"{due[1] if due else 'no clinician-approved interval, so nothing is said about need'}"
            )
        for i, ev in enumerate(
            [
                ("start", {}),
                ("patient", {"patient_ref": ref}),
                ("intent", {"intent": "book_test"}),
                ("action", {"name": "tests_booked", "ref": "KCD-DEMO"}),
                ("end", {"outcome": "completed"}),
            ],
            1,
        ):
            api.post(f"/api/v1/calls/{call}/events", seq=i, kind=ev[0], payload=ev[1])
        ctx = api.get("/api/v1/continuity", caller_phone="0", patient_ref=ref)
        print(
            f"  next call's context (kept one day) -> {ctx['cached'][0]['actions']} outcome={ctx['cached'][0]['outcome']}"
        )
    show("nothing found")
    print("  ", ht.CANNOT_FIND["en"])
    print("  ", ht.CANNOT_FIND["bn"])
    print("  ", ht.CANNOT_FIND["hi"])
    show("audit trail (newest 8)")
    for e in api.get("/api/v1/audit", limit=8)["entries"]:
        print(f"  {e['at'][11:19]} {e['action']:26s} {e['outcome']:7s} call={e['call_id']}")
    print("\nAll checks in this demo passed." if not failures else f"\n{failures} check(s) FAILED.")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--token-file")
    args = ap.parse_args(argv)
    if args.url:
        import httpx

        headers = {}
        if args.token_file:
            with open(args.token_file, encoding="utf-8") as f:
                headers["Authorization"] = f"Bearer {f.read().strip()}"
        with httpx.Client(base_url=args.url, headers=headers, timeout=30.0) as client:
            return run(Api(client))
    from _clinic_app import clinic_app

    with clinic_app() as (_app, client):
        return run(Api(client))


if __name__ == "__main__":
    sys.exit(main())
