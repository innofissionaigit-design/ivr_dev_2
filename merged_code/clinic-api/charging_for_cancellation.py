"""Cancellation charging rules -- read from the admin's text file, applied
here, and nowhere else.

story title: Caller cancels an appointment (E4-S4)
user story: As a patient who cannot attend, I want to cancel and be told any
  charge clearly, so that I am not surprised by a deduction later.
acceptance criteria: Cancellation applies the configured window rules and
  states refund eligibility from policy, never improvised. A cancellation
  within a charging window is confirmed explicitly with the charge stated
  before it is applied.

THE RULES LIVE IN cancellation_rules.txt, NEXT TO THIS FILE, written by an
admin in plain words ("0 to 12 hours | charge 320 taka | refund none"). The
file's own header explains the format. Its path can be overridden with the
CANCELLATION_RULES_PATH environment variable.

Why this lives in clinic-api and not in the agent: the charge the caller
HEARS and the charge that is RECORDED must come from one calculation, in the
service that performs the write. A copy in the agent would be a second
source of truth that could drift from the one that charges.

Why it is pure (no database, no web framework, no clock): `now` is always
passed in, so every window boundary can be tested to the minute.

A FILE THAT IS WRONG NEVER PRODUCES A CHARGE. Anything this module cannot
read with certainty -- an unknown line, a gap between windows, a fractional
charge, no approver -- raises PolicyError, and clinic-api then refuses to
cancel by phone (the caller is sent to the counter). A guessed charge would
be exactly the "surprise deduction" the story exists to prevent.

There is no payment system in this codebase. A "charge" is an amount the
rules say applies; "refund eligibility" is what the rules say the caller is
entitled to. Neither moves money.

Run this file directly to check the rules file:
    python charging_for_cancellation.py [path]
"""
from __future__ import annotations

import datetime
import os
import re
import sys
from dataclasses import asdict, dataclass

# India has no daylight saving, so a fixed +05:30 offset is exact and needs
# no tzdata package (absent on Windows). Every window is measured in clinic
# time: a server on UTC with a naive clock would put every window 5.5 hours
# off, i.e. quote the wrong charge.
CLINIC_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30), name="IST")

REFUND_ELIGIBILITY = ("full", "partial", "none")
AFTER_START_ID = "after_start"
DEFAULT_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "cancellation_rules.txt")


class PolicyError(ValueError):
    """The rules file is missing, unreadable, inconsistent or unapproved."""


@dataclass(frozen=True)
class Window:
    id: str
    min_hours_before: float            # inclusive
    max_hours_before: float | None     # exclusive; None = no upper bound
    charge_inr: int
    refund_eligibility: str            # "full" | "partial" | "none"
    refund_percent: int | None = None  # only for "partial"


@dataclass(frozen=True)
class Policy:
    version: str
    effective_from: str
    approved_by: str
    windows: tuple[Window, ...]
    after_start: Window | None         # None = not cancellable after the start


@dataclass(frozen=True)
class Quote:
    cancellable: bool
    reason: str | None                 # None when cancellable, else "after_start"
    window_id: str | None
    hours_before: float
    charge_inr: int
    refund_eligibility: str | None
    refund_percent: int | None
    policy_version: str

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ parsing

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_SETTING_KEYS = ("version", "effective_from", "approved_by", "after_start")
_SETTING = re.compile(r"^(?P<key>[A-Za-z][A-Za-z _]*?)\s*=\s*(?P<value>.*)$")
_RANGE = re.compile(
    r"^(?P<lo>\d+(?:\.\d+)?)\s*"
    r"(?:(?:to|-)\s*(?P<hi>\d+(?:\.\d+)?)|(?P<open>\+|or\s+more|and\s+more|or\s+above|and\s+above))"
    r"\s*(?:hours?|hrs?|h)?(?:\s+before)?$",
    re.IGNORECASE)
_CHARGE = re.compile(
    r"^charge\s+(?:₹|rs\.?\s*)?(?P<amount>[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(?:taka|tk|rs\.?|rupees?|inr|টাকা)?$",
    re.IGNORECASE)
_REFUND = re.compile(
    r"^refund\s+(?:(?P<word>full|none|no)|(?P<pct>[0-9]+(?:\.[0-9]+)?)\s*(?:%|percent|per\s*cent))$",
    re.IGNORECASE)
_NOT_CANCELLABLE = {"not cancellable", "not cancelable", "not_cancellable", "no"}

_RULE_SHAPE = ("<from> to <to> hours | charge <amount> taka | refund <full / none / NN percent>")


def _num(value: str) -> float | int:
    f = float(value)
    return int(f) if f.is_integer() else f


def _label(n: float | int) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)


def _charge_and_refund(charge_text: str, refund_text: str, where: str,
                       problems: list[str]) -> tuple[int, str, int | None] | None:
    ok = True
    charge = None
    m = _CHARGE.match(charge_text)
    if not m:
        problems.append(f"{where}: cannot read the charge {charge_text!r} -- write e.g. 'charge 320 taka'")
        ok = False
    else:
        amount = m["amount"].replace(",", "")
        if "." in amount and not float(amount).is_integer():
            problems.append(f"{where}: charge must be whole taka, not {m['amount']}")
            ok = False
        else:
            charge = int(float(amount))

    eligibility, percent = None, None
    m = _REFUND.match(refund_text)
    if not m:
        problems.append(f"{where}: cannot read the refund {refund_text!r} -- write 'refund full', "
                        f"'refund none' or e.g. 'refund 50 percent'")
        ok = False
    elif m["word"]:
        eligibility = "full" if m["word"].lower() == "full" else "none"
    else:
        pct = float(m["pct"])
        if not pct.is_integer() or not 0 < pct < 100:
            hint = " (use 'refund none')" if pct == 0 else " (use 'refund full')" if pct == 100 else ""
            problems.append(f"{where}: a part refund must be a whole number from 1 to 99 percent, "
                            f"not {m['pct']}{hint}")
            ok = False
        else:
            eligibility, percent = "partial", int(pct)
    return (charge, eligibility, percent) if ok else None


def parse_rules_text(text: str, *, require_approval: bool = True) -> Policy:
    """Read the admin's rules. Raises PolicyError listing EVERY problem, each
    with its line number, so one run of the checker shows all of them."""
    problems: list[str] = []
    settings: dict[str, tuple[int, str]] = {}
    windows: list[tuple[int, Window]] = []
    unreadable_rules = 0

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].translate(_BN_DIGITS).strip()
        if not line:
            continue
        where = f"line {lineno}"

        m = _SETTING.match(line)
        key = m and m["key"].strip().lower().replace(" ", "_")
        if m and key in _SETTING_KEYS:
            if key in settings:
                problems.append(f"{where}: {key} is set twice (also on line {settings[key][0]})")
            settings[key] = (lineno, m["value"].strip())
            continue

        parts = [p.strip() for p in line.split("|")]
        unreadable_rules += 1          # undone below once the line is read
        if len(parts) == 1:
            problems.append(f"{where}: not understood: {raw.strip()!r}. A rule looks like: {_RULE_SHAPE}")
            continue
        if len(parts) != 3:
            problems.append(f"{where}: a rule needs exactly three parts separated by '|': {_RULE_SHAPE}")
            continue
        rng = _RANGE.match(parts[0])
        if not rng:
            problems.append(f"{where}: cannot read the hours {parts[0]!r} -- write e.g. "
                            f"'0 to 12 hours' or '24 or more hours'")
            continue
        lo = _num(rng["lo"])
        hi = None if rng["open"] else _num(rng["hi"])
        if hi is not None and hi <= lo:
            problems.append(f"{where}: '{parts[0]}' ends before it starts")
            continue
        money = _charge_and_refund(parts[1], parts[2], where, problems)
        if money is None:
            continue
        charge, eligibility, percent = money
        window_id = f"{_label(lo)}h+" if hi is None else f"{_label(lo)}-{_label(hi)}h"
        windows.append((lineno, Window(window_id, lo, hi, charge, eligibility, percent)))
        unreadable_rules -= 1

    def setting(name: str) -> str:
        return settings.get(name, (0, ""))[1]

    version, approved_by = setting("version"), setting("approved_by")
    effective_from = setting("effective_from")
    if not version:
        problems.append("version is missing -- add e.g. 'version = 2026-09-19-a'")
    if require_approval and not approved_by:
        # Blueprint Phase 0 gate: no unapproved value may reach a caller.
        problems.append("approved_by is empty -- phone cancellation stays OFF until someone "
                        "who approves these charges types their name there")
    try:
        datetime.date.fromisoformat(effective_from)
    except ValueError:
        problems.append(f"effective_from must be a date written YYYY-MM-DD, not {effective_from!r}")

    after_start = None
    after_raw = setting("after_start") or "not cancellable"
    if after_raw.lower() not in _NOT_CANCELLABLE:
        parts = [p.strip() for p in after_raw.split("|")]
        where = f"line {settings['after_start'][0]} (after_start)"
        if len(parts) != 2:
            problems.append(f"{where}: write 'not cancellable' or e.g. 'charge 500 taka | refund none'")
        else:
            money = _charge_and_refund(parts[0], parts[1], where, problems)
            if money is not None:
                after_start = Window(AFTER_START_ID, 0, None, *money)

    # Coverage is only meaningful once every rule line was read: a line
    # already reported as unreadable would show up again as a "gap".
    if not unreadable_rules:
        _check_coverage(windows, problems)

    if problems:
        raise PolicyError("; ".join(problems))
    return Policy(version=version, effective_from=effective_from, approved_by=approved_by,
                  windows=tuple(w for _, w in windows), after_start=after_start)


def _check_coverage(windows: list[tuple[int, Window]], problems: list[str]) -> None:
    """Every moment before the start has exactly one rule: windows start at
    0 hours, follow on with no gap and no overlap, and the last is open."""
    if not windows:
        problems.append(f"there are no rules -- add at least one line like: {_RULE_SHAPE}")
        return
    ordered = sorted(windows, key=lambda lw: lw[1].min_hours_before)
    if ordered[0][1].min_hours_before != 0:
        problems.append(f"line {ordered[0][0]}: the first window must start at 0 hours")
    for (prev_line, prev), (next_line, nxt) in zip(ordered, ordered[1:]):
        if prev.max_hours_before is None:
            problems.append(f"line {prev_line}: only the LAST window may be open-ended "
                            f"('or more'), but line {next_line} starts after it")
        elif nxt.min_hours_before < prev.max_hours_before:
            problems.append(f"lines {prev_line} and {next_line} overlap")
        elif nxt.min_hours_before > prev.max_hours_before:
            problems.append(f"gap between lines {prev_line} and {next_line}: nothing covers "
                            f"{_label(prev.max_hours_before)} to {_label(nxt.min_hours_before)} hours")
    if ordered[-1][1].max_hours_before is not None:
        problems.append(f"line {ordered[-1][0]}: the last window must be open-ended, e.g. "
                        f"'{_label(ordered[-1][1].max_hours_before)} or more hours | ...'")


def load_policy(path: str, *, require_approval: bool = True) -> Policy:
    try:
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
    except FileNotFoundError as e:
        raise PolicyError(f"rules file not found: {path}") from e
    except (OSError, UnicodeDecodeError) as e:
        raise PolicyError(f"rules file unreadable: {type(e).__name__}") from e
    return parse_rules_text(text, require_approval=require_approval)


# -------------------------------------------------------------- applying

def is_in_force(policy: Policy, now: datetime.datetime) -> bool:
    """False before effective_from (in clinic time)."""
    return now.astimezone(CLINIC_TZ).date() >= datetime.date.fromisoformat(policy.effective_from)


def appointment_start(date_iso: str, time_slot: str) -> datetime.datetime:
    """The appointment's start as an aware datetime in clinic time."""
    day = datetime.date.fromisoformat(date_iso)
    clock = datetime.datetime.strptime(time_slot, "%H:%M").time()
    return datetime.datetime.combine(day, clock, tzinfo=CLINIC_TZ)


def quote_cancellation(policy: Policy, start: datetime.datetime, now: datetime.datetime) -> Quote:
    """The one calculation both the spoken quote and the recorded charge use.

    A window's lower bound is inclusive and its upper bound exclusive, so a
    call exactly 12h00m before the start falls in the window that starts at
    12. Compared as timedeltas, not floats, so a boundary is never missed by
    rounding.
    """
    if start.tzinfo is None or now.tzinfo is None:
        # A naive clock is exactly the bug that quotes the wrong window on a
        # UTC server. Refuse rather than guess.
        raise ValueError("start and now must be timezone-aware")

    remaining = start - now
    hours_before = round(remaining.total_seconds() / 3600, 2)

    if remaining <= datetime.timedelta(0):
        if policy.after_start is None:
            return Quote(cancellable=False, reason="after_start", window_id=None,
                         hours_before=hours_before, charge_inr=0, refund_eligibility=None,
                         refund_percent=None, policy_version=policy.version)
        return _quote_for(policy, policy.after_start, hours_before)

    for w in policy.windows:
        lower = datetime.timedelta(hours=w.min_hours_before)
        upper = None if w.max_hours_before is None else datetime.timedelta(hours=w.max_hours_before)
        if remaining >= lower and (upper is None or remaining < upper):
            return _quote_for(policy, w, hours_before)

    # Unreachable for a parsed policy: coverage from 0 up is checked.
    raise PolicyError("no window covers this time")


def _quote_for(policy: Policy, w: Window, hours_before: float) -> Quote:
    return Quote(cancellable=True, reason=None, window_id=w.id, hours_before=hours_before,
                 charge_inr=w.charge_inr, refund_eligibility=w.refund_eligibility,
                 refund_percent=w.refund_percent, policy_version=policy.version)


# ------------------------------------------------------- the admin checker

def _refund_words(w: Window) -> str:
    if w.refund_eligibility == "partial":
        return f"eligible for a {w.refund_percent} percent refund"
    return "eligible for a full refund" if w.refund_eligibility == "full" else "no refund"


def describe(policy: Policy) -> list[str]:
    """Plain-English lines for the admin: what a caller is told, per window."""
    lines = []
    for w in policy.windows:
        when = (f"{_label(w.min_hours_before)} hours or more before"
                if w.max_hours_before is None else
                f"from {_label(w.min_hours_before)} up to {_label(w.max_hours_before)} hours before")
        charge = "no charge" if w.charge_inr == 0 else f"charge {w.charge_inr} taka"
        lines.append(f"  {when}: {charge}, {_refund_words(w)}")
    if policy.after_start is None:
        lines.append("  after the appointment time: cannot be cancelled by phone")
    else:
        w = policy.after_start
        charge = "no charge" if w.charge_inr == 0 else f"charge {w.charge_inr} taka"
        lines.append(f"  after the appointment time: {charge}, {_refund_words(w)}")
    return lines


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else os.environ.get("CANCELLATION_RULES_PATH", DEFAULT_RULES_PATH)
    print(f"Checking {path}")
    try:
        policy = load_policy(path, require_approval=False)
    except PolicyError as e:
        print("\nNOT USABLE -- phone cancellation is OFF until these are fixed:")
        for problem in str(e).split("; "):
            print(f"  - {problem}")
        return 1
    print(f"\nRules version {policy.version}, effective from {policy.effective_from}.")
    print("What callers will be told:")
    print("\n".join(describe(policy)))
    now = datetime.datetime.now(CLINIC_TZ)
    if not policy.approved_by:
        print("\nOFF: approved_by is empty. Type the approver's name there to switch it on.")
        return 1
    if not is_in_force(policy, now):
        print(f"\nNOT YET: these rules start on {policy.effective_from}. Until then, "
              f"nothing is cancelled by phone.")
        return 1
    print(f"\nON: approved by {policy.approved_by}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
