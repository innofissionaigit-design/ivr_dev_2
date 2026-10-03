"""The "implicit happiness" score: how well did this call go for the caller, judged from what the bot already sees, for
EVERY call, without asking anyone anything.

Asking every caller to rate the call reaches a small share of them (a few percent by post-call SMS, more inside the call)
and only the ones who choose to answer. The bot, however, sees the shape of every call: whether it had to ask again,
whether it was slow, whether the caller swore, asked for a person, talked over it, or went quiet, and whether the thing
they called about actually got done. This module turns those counts into one 0-100 number and says WHY, so a low
score can be traced to a cause and a good one is not a mystery.

What this is NOT: a measure of what the caller FELT. It is a proxy built from behaviour, and it says so
(`SCORE_MODEL_VERSION`, `basis: "behaviour"`). Every weight below is REASONED, not measured: they are the right
order of magnitude and the right sign, and nothing more. The way to make them real is to compare the score with the
caller's own answer on the calls where one is asked (the planned one-question survey) and refit; until then, read the
score as a ranking of calls (this one went worse than that one), not as a percentage of happy callers.

Pure and deterministic: the same signals always give the same score, in the same order of reasons. No text of any
conversation goes in or comes out -- only counts and flags -- so the score can be stored with the call record without
adding anything personal to it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

SCORE_MODEL_VERSION = "implicit-v1"
BASE_SCORE = 80

# A reply slower than this to the first audio counts as a slow one (seconds). REASONED: the owner's target is 0.5 s; a
# caller notices past about a second and a half.
SLOW_REPLY_S = 1.5
MIN_TURNS_TO_SCORE = 2  # a call with fewer caller turns says too little to score

# (points per occurrence, the most this signal can take away or add in total). Negative = the call went worse.
_COUNTED: dict[str, tuple[int, int]] = {
    "reasks": (
        -6,
        -24,
    ),  # "I didn't hear / understand that" -- the caller had to repeat themselves
    "system_failures": (
        -10,
        -30,
    ),  # the bot could only say "a problem, please wait" / "I can't check that"
    "abuse_turns": (-12, -36),  # the caller swore at the bot
    # Ported story "Caller is angry about a previous experience" (agent/anger.py): the caller said plainly that
    # they were angry or fed up. Scored a little less harshly than being sworn at, deliberately -- anger is a
    # worse call for the caller either way, but it is often the clinic's own earlier failure being reported, not
    # the caller mistreating the agent, and the agent handling it correctly should not be penalised as if it were.
    "anger_turns": (-10, -30),
    # Ported story "Caller asks whether their result is dangerous" (agent/clinical_safety.py). Scored only
    # LIGHTLY negative, and the reasoning matters: routing this to a clinician is the agent behaving exactly as
    # designed, so a heavy penalty would mark correct conduct as a bad call. But the caller still rang with a
    # question this service could not answer and left needing someone else, which is a worse call FOR THEM than
    # one that ended in an answer -- and that, not the agent's conduct, is what this score measures.
    "clinical_questions": (-3, -9),
    "silence_prompts": (
        -3,
        -6,
    ),  # the caller went quiet long enough to be asked if they were still there
    "language_flips": (
        -5,
        -15,
    ),  # the bot changed the call's language and had to be corrected
    "barge_ins": (-3, -12),  # the caller talked over the bot
    "slow_replies": (-3, -15),  # a reply took longer than SLOW_REPLY_S
    "corrections": (-4, -12),  # "no, that's wrong" at a read-back
    "answers_given": (6, 12),  # a question the caller asked was answered
}
# (points, applies-once) flags
_FLAGS: dict[str, int] = {
    "asked_for_person": -15,  # the caller asked for a human
    # Ported story "Caller wants to make a complaint" (agent/complaint.py). Its own flag rather than being folded
    # into system_handoff, so a call the caller COMPLAINED about stays distinguishable in the score from one the
    # agent merely could not finish -- they need different follow-up from a human, and a single shared code would
    # make the two impossible to tell apart afterwards.
    "complaint_filed": -15,
    "system_handoff": -10,  # the bot itself had to hand the call to a person
    "silence_timeout": -15,  # the call was ended because the caller said nothing
    "left_mid_task": -15,  # a booking was still half-done at hang-up
    "task_completed": 12,  # a booking / cancellation / change was completed
    "caller_thanked": 6,  # the caller said thank you
    "ended_by_caller": 4,  # the caller ended it politely, in their own words
}
BANDS = ((80, "happy"), (60, "neutral"), (40, "unhappy"), (0, "very_unhappy"))


@dataclass
class CallSignals:
    """Counters and flags for one call. Cheap to update from anywhere in a turn (see main._note)."""

    turns: int = 0
    reasks: int = 0
    system_failures: int = 0
    abuse_turns: int = 0
    anger_turns: int = 0
    clinical_questions: int = 0
    silence_prompts: int = 0
    language_flips: int = 0
    barge_ins: int = 0
    slow_replies: int = 0
    corrections: int = 0
    answers_given: int = 0
    asked_for_person: bool = False
    system_handoff: bool = False
    silence_timeout: bool = False
    left_mid_task: bool = False
    task_completed: bool = False
    caller_thanked: bool = False
    complaint_filed: bool = False
    ended_by_caller: bool = False
    emergency: bool = False
    reply_ms: list[int] = field(default_factory=list)  # first-audio time of each turn, milliseconds

    def note(self, name: str, n: int = 1) -> None:
        """Count (int fields) or raise (bool fields) a signal; an unknown name is a programming error, not ignored."""
        cur = getattr(self, name)
        if isinstance(cur, bool):
            setattr(self, name, True)
        elif isinstance(cur, int):
            setattr(self, name, cur + n)
        else:
            raise TypeError(f"signal {name!r} is not a counter or a flag")

    def note_reply(self, seconds: float) -> None:
        self.reply_ms.append(int(seconds * 1000))
        if seconds > SLOW_REPLY_S:
            self.slow_replies += 1


@dataclass(frozen=True)
class ScoreResult:
    score: int | None  # 0-100, or None when the call cannot be scored (see `reason`)
    band: str  # happy | neutral | unhappy | very_unhappy | unscored
    reasons: tuple[tuple[str, int], ...]  # (signal, points) for every signal that moved the score, biggest first
    reason: str = ""  # why it is unscored
    version: str = SCORE_MODEL_VERSION
    basis: str = "behaviour"

    def to_payload(self, signals: CallSignals) -> dict[str, Any]:
        """What is stored with the call record: the number, the band, the reasons and the counts. No text."""
        d = dataclasses.asdict(signals)
        replies = d.pop("reply_ms")
        d["median_reply_ms"] = sorted(replies)[len(replies) // 2] if replies else None
        return {
            "score": self.score,
            "band": self.band,
            "reasons": [list(r) for r in self.reasons],
            "reason": self.reason,
            "version": self.version,
            "basis": self.basis,
            "signals": d,
        }


def band_for(score: int) -> str:
    return next(name for floor, name in BANDS if score >= floor)


def score_call(s: CallSignals) -> ScoreResult:
    if s.emergency:
        return ScoreResult(None, "unscored", (), reason="emergency")  # a satisfaction number means nothing here
    if s.turns < MIN_TURNS_TO_SCORE:
        return ScoreResult(None, "unscored", (), reason="too_short")
    moved: list[tuple[str, int]] = []
    for name, (per, limit) in _COUNTED.items():
        n = getattr(s, name)
        if not n:
            continue
        pts = max(per * n, limit) if per < 0 else min(per * n, limit)
        moved.append((name, pts))
    for name, pts in _FLAGS.items():
        if getattr(s, name):
            moved.append((name, pts))
    total = BASE_SCORE + sum(p for _n, p in moved)
    score = max(0, min(100, total))
    # biggest movement first; ties by name so the order never changes between runs
    ordered = tuple(sorted(moved, key=lambda r: (-abs(r[1]), r[0])))
    return ScoreResult(score, band_for(score), ordered)
