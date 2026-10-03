"""How long THIS caller pauses, learned as the call goes (KCD-047).

A single fixed silence threshold is wrong for someone: too short and a slow, deliberate speaker
is cut off mid-sentence; too long and a brisk one waits for nothing. Instead of a constant, the
agent measures each caller's own rhythm and waits accordingly.

WHAT IT MEASURES. In every utterance the voice detector already splits the speech into spans; the
quiet BETWEEN consecutive spans of one utterance is the caller's own pause between words, phrases
and sentences -- the thing the agent must wait through. Those gaps are collected for the call. Their
upper tail (the 90th percentile, over the most recent gaps) is how long this caller pauses WITHIN a
thought; the agent then waits a margin beyond that before deciding the caller has finished:

    silence needed to commit   =  p90 of this caller's gaps  x  margin      (clamped)
    after a completed phrase   =  their median gap            x  margin      (clamped, never below a floor)
    after an unfinished one    =  never less than 1.5 x the commit wait

A SECOND SIGNAL corrects mistakes: if the caller starts speaking again within a moment of a turn
being cut and before the agent has said anything, the cut was too early -- that is direct evidence
the wait was too short for this person, and the multiplier is raised; a long run of clean turns
lets it relax back. So the estimate is pulled by the pauses it sees and pushed by the cuts it got
wrong.

WHAT IT IS AND IS NOT. It adapts the REASONED constants in agent/endpointing.EndpointConfig to each
caller. It does not replace the measured calibration KCD-047 asks for (tools/calibrate_endpointing.py
on real annotated recordings), and its own constants (margin, clamps) are REASONED, validated on
synthetic speakers with different rhythms (tests/test_pause_profile.py). Until it has enough gaps
(MIN_GAPS) it returns the base configuration unchanged, so a call always starts from the safe default.
Pure numpy/Python; the clock and audio are injected.
"""

from __future__ import annotations

import collections
import dataclasses

import numpy as np

from agent.endpointing import EndpointConfig

MIN_GAPS = 4  # do not adapt on fewer observed gaps than this
MAX_GAPS_KEPT = 40  # the most recent gaps; a caller's rhythm can change during a call
MAX_GAP_S = 3.0  # a longer silence inside "one utterance" is not a pause, it is the end
MIN_GAP_S = 0.08  # below the detector's own hangover it is not a gap
MARGIN = 1.3  # wait this much longer than the caller's own long pauses
COMPLETE_MARGIN = 1.2
SILENCE_MIN_S, SILENCE_MAX_S = 0.6, 2.6
COMPLETE_MIN_S, COMPLETE_MAX_S = 0.35, 1.2
INCOMPLETE_FACTOR = 1.5
FALSE_CUT_WINDOW_S = 1.2  # speech resuming this soon after a cut means the cut was early
FALSE_CUT_STEP = 1.15
RELAX_AFTER_CLEAN_TURNS = 5
RELAX_STEP = 0.95
BUMP_MIN, BUMP_MAX = 1.0, 1.8


@dataclasses.dataclass
class PauseProfile:
    gaps: collections.deque = dataclasses.field(default_factory=lambda: collections.deque(maxlen=MAX_GAPS_KEPT))
    bump: float = 1.0  # raised by early cuts, relaxed by clean turns
    false_cuts: int = 0
    clean_turns: int = 0
    utterances: int = 0

    # ---------------------------------------------------------------- observing
    def observe_utterance(self, spans: list[dict], upto_s: float | None = None) -> int:
        """Record the gaps between consecutive spans of ONE finished utterance. `upto_s` bounds the
        spans to those inside the utterance that was cut. Returns how many gaps were recorded."""
        inside = [s for s in spans if upto_s is None or float(s["end"]) <= upto_s + 1e-6]
        n = 0
        for a, b in zip(inside, inside[1:]):
            gap = float(b["start"]) - float(a["end"])
            if MIN_GAP_S <= gap <= MAX_GAP_S:
                self.gaps.append(gap)
                n += 1
        self.utterances += 1
        return n

    def note_turn_cut(self, resumed_after_s: float | None, agent_spoke_between: bool) -> None:
        """Called when a turn was committed and later speech was seen. `resumed_after_s` is the quiet
        between the cut and the next speech (None if nothing followed yet)."""
        if resumed_after_s is not None and resumed_after_s <= FALSE_CUT_WINDOW_S and not agent_spoke_between:
            self.false_cuts += 1
            self.clean_turns = 0
            self.bump = min(BUMP_MAX, self.bump * FALSE_CUT_STEP)
        else:
            self.clean_turns += 1
            if self.clean_turns >= RELAX_AFTER_CLEAN_TURNS and self.bump > BUMP_MIN:
                self.bump = max(BUMP_MIN, self.bump * RELAX_STEP)
                self.clean_turns = 0

    # ---------------------------------------------------------------- deciding
    @property
    def adapted(self) -> bool:
        return len(self.gaps) >= MIN_GAPS

    def p90(self) -> float | None:
        return float(np.percentile(np.asarray(self.gaps), 90)) if self.gaps else None

    def median(self) -> float | None:
        return float(np.median(np.asarray(self.gaps))) if self.gaps else None

    def config(self, base: EndpointConfig) -> EndpointConfig:
        """The endpoint configuration for THIS caller. Base unchanged until enough gaps were seen
        (unless early cuts have already been noticed, which alone raises the wait)."""
        if not self.adapted and self.bump == 1.0:
            return base
        if self.adapted:
            silence = _clamp(self.p90() * MARGIN * self.bump, SILENCE_MIN_S, SILENCE_MAX_S)
            complete = _clamp(self.median() * COMPLETE_MARGIN * self.bump, COMPLETE_MIN_S, COMPLETE_MAX_S)
        else:
            silence = _clamp(base.silence_confirm_s * self.bump, SILENCE_MIN_S, SILENCE_MAX_S)
            complete = _clamp(base.complete_confirm_s * self.bump, COMPLETE_MIN_S, COMPLETE_MAX_S)
        complete = min(complete, silence)  # finished never waits longer than the baseline
        incomplete = max(silence * INCOMPLETE_FACTOR, base.incomplete_confirm_s if not self.adapted else 0.0)
        return dataclasses.replace(
            base,
            silence_confirm_s=silence,
            complete_confirm_s=complete,
            incomplete_confirm_s=min(incomplete, SILENCE_MAX_S * INCOMPLETE_FACTOR),
        )

    def snapshot(self) -> dict:
        return {
            "gaps": len(self.gaps),
            "p90_s": None if not self.gaps else round(self.p90(), 3),
            "median_s": None if not self.gaps else round(self.median(), 3),
            "bump": round(self.bump, 3),
            "false_cuts": self.false_cuts,
            "adapted": self.adapted,
        }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
