"""KCD-149: speech policy derived from call state.

Blueprint 4.5 and Appendix C: a deterministic table maps the caller's
state to HOW the agent talks -- speech rate, sentence length, questions
per turn, confirmation style, interruption tolerance, escalation
threshold. Empathy is driven by these rules, not by asking a model to "be
empathetic". The values below are Appendix C's table row for row.

This system's replies are templates (agent/reply_templates.py), and the
model only ever extracts intent, so "hard constraints for the planner"
means two concrete things here: the rate goes to synthesis
(main.py's _speak), and the question count / sentence length / echo rules
are enforced on the reply before it is spoken (check_reply). A future
planner that DOES compose wording inherits the same constraints through
hard_constraints().

Precedence when several states apply at once (an older caller who is also
distressed): the most conservative value of every field wins -- slowest
rate, shortest sentences, fewest questions, the strongest confirmation,
the lowest escalation threshold. Emergency overrides everything and stops
the normal flow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Appendix C "speech_rate": a multiplier on the voice's normal rate
# (1.0 = default; see agent/prosody.resolve_length_scale). The Blueprint
# gives words, not numbers, so these are REASONED, not measured -- a
# to-do for the first listening test on a real handset, not a setting.
_RATE = {"default": 1.0, "slow-normal": 0.93, "slow": 0.85}

# Words per sentence ceiling for each Appendix C "sentence_len". Same
# caveat: reasoned.
_SENTENCE_WORDS = {"default": None, "short": 12, "very_short": 8}

# Appendix C "confirmation" column, ordered weakest to strongest.
_CONFIRMATION_ORDER = (
    "implicit",
    "explicit_high_frequency",
    "explicit_acknowledge",
    "explicit_offer_human",
    "explicit_repeat_back",
)
_LEVEL_ORDER = ("normal", "high", "very_high")
_ESCALATION_ORDER = ("normal", "lowered", "low")  # "low" threshold = escalate soonest


@dataclass(frozen=True)
class SpeechPolicy:
    speech_rate: float = 1.0
    sentence_length: str = "default"  # default | short | very_short
    questions_per_turn: int = 2  # Appendix C: "up to 2" for normal
    confirmation: str = "implicit"
    interruption_tolerance: str = "normal"  # normal | high | very_high
    escalation_threshold: str = "normal"  # normal | lowered | low
    acknowledge_first: bool = False  # acknowledge the feeling before anything else
    offer_human: bool = False
    emergency: bool = False  # stop the normal flow entirely

    @property
    def max_sentence_words(self) -> int | None:
        return _SENTENCE_WORDS[self.sentence_length]

    @property
    def confirm_each_slot(self) -> bool:
        """Senior mode's "one question, listen, confirm, next question":
        echo every value back as it is collected instead of waiting for
        one summary at the end."""
        return self.confirmation != "implicit"

    def hard_constraints(self) -> dict:
        """What a wording-composing planner must stay inside."""
        return {
            "speech_rate": self.speech_rate,
            "max_sentence_words": self.max_sentence_words,
            "questions_per_turn": self.questions_per_turn,
            "confirmation": self.confirmation,
            "acknowledge_first": self.acknowledge_first,
            "offer_human": self.offer_human,
            "emergency": self.emergency,
        }


# Appendix C, row for row.
_ROWS: dict[str, SpeechPolicy] = {
    "normal": SpeechPolicy(),
    "senior": SpeechPolicy(
        speech_rate=_RATE["slow"],
        sentence_length="short",
        questions_per_turn=1,
        confirmation="explicit_high_frequency",
        interruption_tolerance="high",
        escalation_threshold="lowered",
    ),
    "distressed": SpeechPolicy(
        speech_rate=_RATE["slow"],
        sentence_length="short",
        questions_per_turn=1,
        confirmation="explicit_acknowledge",
        interruption_tolerance="very_high",
        escalation_threshold="low",
        acknowledge_first=True,
    ),
    "angry": SpeechPolicy(
        speech_rate=_RATE["slow-normal"],
        sentence_length="short",
        questions_per_turn=1,
        confirmation="explicit_offer_human",
        interruption_tolerance="high",
        escalation_threshold="low",
        acknowledge_first=True,
        offer_human=True,
    ),
    "confused": SpeechPolicy(
        speech_rate=_RATE["slow"],
        sentence_length="very_short",
        questions_per_turn=1,
        confirmation="explicit_repeat_back",
        interruption_tolerance="high",
        escalation_threshold="lowered",
    ),
    "emergency": SpeechPolicy(emergency=True, questions_per_turn=0, acknowledge_first=True, escalation_threshold="low"),
}

_SENTENCE_RANK = ("default", "short", "very_short")


def _most_conservative(a: SpeechPolicy, b: SpeechPolicy) -> SpeechPolicy:
    def stronger(order, x, y):
        return x if order.index(x) >= order.index(y) else y

    return SpeechPolicy(
        speech_rate=min(a.speech_rate, b.speech_rate),
        sentence_length=stronger(_SENTENCE_RANK, a.sentence_length, b.sentence_length),
        questions_per_turn=min(a.questions_per_turn, b.questions_per_turn),
        confirmation=stronger(_CONFIRMATION_ORDER, a.confirmation, b.confirmation),
        interruption_tolerance=stronger(_LEVEL_ORDER, a.interruption_tolerance, b.interruption_tolerance),
        escalation_threshold=stronger(_ESCALATION_ORDER, a.escalation_threshold, b.escalation_threshold),
        acknowledge_first=a.acknowledge_first or b.acknowledge_first,
        offer_human=a.offer_human or b.offer_human,
        emergency=a.emergency or b.emergency,
    )


def derive_policy(caller_state: str = "neutral", senior: bool = False) -> SpeechPolicy:
    """Call state -> the Appendix C parameters. `caller_state` is one of
    agent.call_state.VALID_CALLER_STATES ("neutral" is the normal row); an
    unknown value falls back to normal rather than raising -- a detector
    bug must never take a call down."""
    if caller_state == "emergency":
        return _ROWS["emergency"]
    key = "normal" if caller_state == "neutral" else caller_state
    policy = _ROWS.get(key, _ROWS["normal"])
    if senior and key != "senior":
        policy = _most_conservative(policy, _ROWS["senior"])
    return policy


# ---------------------------------------------------- enforcement on a reply

_RE_QUESTION = re.compile(r"[?؟]")


def count_questions(text: str) -> int:
    return len(_RE_QUESTION.findall(text))


def longest_sentence_words(text: str) -> int:
    sentences = re.split(r"[।.?!\n]+", text)
    return max((len(s.split()) for s in sentences if s.strip()), default=0)


def check_reply(text: str, policy: SpeechPolicy) -> list[str]:
    """Constraint violations in `text`, empty if it complies. Used by the
    test suite over every reply template (a senior caller must never hear
    several questions in one reply) and as a runtime guard."""
    problems: list[str] = []
    if policy.emergency:
        return problems
    if count_questions(text) > policy.questions_per_turn:
        problems.append(f"{count_questions(text)} questions, policy allows {policy.questions_per_turn}")
    cap = policy.max_sentence_words
    if cap is not None and longest_sentence_words(text) > cap:
        problems.append(f"sentence of {longest_sentence_words(text)} words exceeds {cap}")
    return problems


def limit_questions(text: str, max_questions: int) -> str:
    """Runtime enforcement of "never several questions in one reply": keep
    everything up to and including the last permitted question and drop
    the sentences that ask more. Statements are never dropped -- only a
    surplus question is -- so no fact is lost."""
    if max_questions <= 0 or count_questions(text) <= max_questions:
        return text
    # Sentence-level split, so a statement is never carried away with the
    # surplus question that follows it. Whitespace is required after the
    # terminator, so "2.5" or "KCD-1.2" is not cut.
    parts = re.split(r"(?<=[।.!?؟])\s+", text)
    kept, asked = [], 0
    for part in parts:
        if not part.strip():
            continue
        if _RE_QUESTION.search(part):
            asked += 1
            if asked > max_questions:
                continue
        kept.append(part.strip())
    return " ".join(kept)


def effective_rate(policy: SpeechPolicy, figure_rate: float | None = None) -> float:
    """The rate for one clause: the policy's, and never faster than a
    figure's own slower rate (KCD-157) when the clause carries a price,
    phone number or reference. The two multiply, so a senior caller hears
    a price slower still."""
    return policy.speech_rate * (figure_rate if figure_rate is not None else 1.0)


def as_call_state_fields(policy: SpeechPolicy) -> dict:
    """The CallState fields this policy sets (agent/call_state.py)."""
    return {
        "speech_rate": policy.speech_rate,
        "response_length": "normal" if policy.sentence_length == "default" else "short",
        "one_question_at_a_time": policy.questions_per_turn <= 1,
        "interruption_tolerance": "patient" if policy.interruption_tolerance != "normal" else "normal",
        "escalation_threshold": "lowered" if policy.escalation_threshold != "normal" else "normal",
    }
