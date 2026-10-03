"""Ask again, kindly, before routing to a human.

A caller who is faint, mumbling, in a noisy room, or talking over someone
else is not asking to be transferred -- they are asking to be heard. Today
a turn the system cannot use either gets one flat "please repeat" or (for
an ambiguous language) an immediate hand-off. This module owns the
decision instead:

    proceed   the turn is usable -- reset the failure count, carry on.
    reask     say WHY, in the caller's language and without blaming them
              (phrases reask_*), and listen again.
    handoff   after `max_reasks` consecutive failures, apologise
              (reask_final) and route to a person.

It only acts when the turn actually FAILED (empty ASR, a jumbled
transcript, an unidentifiable language). Acoustic trouble alone -- mild
noise the recogniser coped with -- never triggers a re-ask: nagging a
caller whose words were understood is its own kind of failure. When the
turn did fail, the acoustic assessment (agent/audio_quality.py) decides
WHICH kind of re-ask is most useful ("speak up" is no help to someone in a
noisy room).

A second consecutive failure marks the caller as "confused" -- Appendix C's
"confusion / speech difficulty" row -- which slows the agent, shortens its
sentences and lowers its escalation threshold for the rest of the call
(agent/speech_policy.py). That is the deterministic empathy Blueprint 4.5
asks for, driven by a rule, not by asking a model to be kind.
"""

from __future__ import annotations

from dataclasses import dataclass

# Acoustic issue -> the phrase that addresses it. Order = priority when
# several apply: a faint voice is fixed by speaking up, so it outranks
# noise; cross-talk is named before generic noise because "move away from
# the other person" is a different instruction.
_ISSUE_PHRASE = (
    ("too_quiet", "reask_low_volume"),
    ("crosstalk", "reask_crosstalk"),
    ("noisy", "reask_noisy"),
    ("unvoiced_mumble", "reask_mumbled"),
)
_MUMBLE_TRANSCRIPT_ISSUES = frozenset({"shattered", "repetitive", "decoders_disagree", "fragment", "wrong_script"})

DEFAULT_MAX_REASKS = 2


@dataclass(frozen=True)
class ReaskDecision:
    action: str  # "proceed" | "reask" | "handoff"
    reason: str | None = None  # the issue that drove it
    phrase_key: str | None = None  # agent.phrases key to speak first
    mark_confused: bool = False  # feed caller_state="confused"
    attempt: int = 0  # consecutive failures so far


class ReaskTracker:
    """One per call. Not thread-safe by design -- a call's turns are
    already serialised by CallSession.dispatch_lock."""

    def __init__(self, max_reasks: int = DEFAULT_MAX_REASKS):
        self.max_reasks = max_reasks
        self.consecutive = 0
        self.total = 0

    def note_success(self) -> ReaskDecision:
        self.consecutive = 0
        return ReaskDecision("proceed")

    def decide(
        self,
        *,
        asr_empty: bool = False,
        audio_issues: list[str] | None = None,
        transcript_issue: str | None = None,
        language_ambiguous: bool = False,
    ) -> ReaskDecision:
        failed = asr_empty or language_ambiguous or transcript_issue is not None
        if not failed:
            return self.note_success()

        self.consecutive += 1
        self.total += 1
        reason, key = self._explain(asr_empty, audio_issues or [], transcript_issue)

        if self.consecutive > self.max_reasks:
            return ReaskDecision("handoff", reason, "reask_final", True, self.consecutive)
        return ReaskDecision("reask", reason, key, self.consecutive >= 2, self.consecutive)

    @staticmethod
    def _explain(asr_empty: bool, audio_issues: list[str], transcript_issue: str | None) -> tuple[str, str]:
        for issue, key in _ISSUE_PHRASE:
            if issue in audio_issues:
                return issue, key
        if transcript_issue in _MUMBLE_TRANSCRIPT_ISSUES:
            return transcript_issue, "reask_mumbled"
        return ("empty" if asr_empty else (transcript_issue or "unclear")), "reask_generic"
