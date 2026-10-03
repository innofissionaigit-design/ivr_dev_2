"""Check the wording an operator has put in the database BEFORE callers hear it (KCD-353/500/513).

The agent speaks whatever is in the `agent_messages` table (clinic-api/agent_messages.py). The operator
owns that wording, and this is the check that keeps a well-meant edit from breaking the voice's rules:

    * register     the respectful second person in every language (no "tumi"/"tum" forms);
    * apology      at most ONE apology in a message, and never a bare apology as the last thing said;
    * sentence     no sentence longer than the persona's cap for its language;
    * never-say    no hedging on a fact ("probably"), no reassurance about health ("don't worry"),
                   no clinical direction ("you should take"), no talk about being a language model;
    * shape        no digits, colons or brackets inside a message that is spoken (a number must come
                   from a live answer, not from a message), and a disclosure must say it is automated
                   and offer staff.

    python tools/check_messages.py --url http://localhost:8080 [--token-file /workspace/.clinic_api_token]

Exit status 0 when every message passes, 1 otherwise. Run it after changing a message and before
relying on it.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import re

from agent.apology import count_apologies, ends_on_bare_apology
from agent.persona import violations

AUTOMATED = {"bn": "স্বয়ংক্রিয়", "hi": "स्वचालित", "en": "automated"}
STAFF = {"bn": "স্টাফ", "hi": "स्टाफ़", "en": "staff"}


def check_message(key: str, lang: str, text: str) -> list[str]:
    """Problems with one message; empty means it passes."""
    problems = [f"persona: {v}" for v in violations(text, lang)]
    n = count_apologies(text, lang)
    if n > 1:
        problems.append(f"{n} apologies in one message (at most one)")
    if ends_on_bare_apology(text, lang):
        problems.append("ends on a bare apology, leaving the caller with no next step")
    if re.search(r"\d|[:\[\]{}]", text) and key not in ("emergency_notice", "emergency_hint"):
        problems.append("contains a digit, colon or bracket (numbers come from live answers, not messages)")
    if key == "disclosure":
        if AUTOMATED[lang] not in text:
            problems.append("a disclosure must say the assistant is automated")
        if STAFF[lang].split("़")[0] not in text:
            problems.append("a disclosure must offer staff")
    return problems


def check_all(messages: dict) -> dict[tuple[str, str], list[str]]:
    """{(key, lang): problems} for every message that has any."""
    bad = {}
    for key, by_lang in messages.items():
        for lang, entry in by_lang.items():
            probs = check_message(key, lang, entry["text"] if isinstance(entry, dict) else entry)
            if probs:
                bad[(key, lang)] = probs
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--token-file")
    args = ap.parse_args(argv)
    import httpx

    headers = {}
    if args.token_file:
        with open(args.token_file, encoding="utf-8") as f:
            headers["Authorization"] = f"Bearer {f.read().strip()}"
    data = httpx.get(f"{args.url}/api/v1/agent/messages", headers=headers, timeout=15.0).json()
    bad = check_all(data["messages"])
    total = sum(len(v) for v in data["messages"].values())
    for (key, lang), probs in sorted(bad.items()):
        print(f"FAIL {key}/{lang}")
        for p in probs:
            print(f"   - {p}")
    print(f"{total - len(bad)} of {total} messages pass" + ("" if not bad else f"; {len(bad)} need changing"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
