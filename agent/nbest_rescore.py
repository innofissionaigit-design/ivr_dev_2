"""Rescore an ASR's N-best hypotheses for one turn against the clinic catalogue.

Ready to plug in the moment the pod gives real N-best output from NeMo's RNNT beam search -- agent/asr.py's TurnASR
today returns only the single best transcript per decoder, never a ranked list. This module and its tests need
neither a live model nor a recording: they take a plain list of candidate transcripts, best first (the shape N-best
comes in), and score each one exactly the way the catalogue lookup itself will (agent/gazetteer.py,
agent/transcript_rules.py). Once real N-best output exists, feeding it through CatalogueRescorer.rescore() instead of
trusting hypothesis 0 is a one-line change at the call site, not a rewrite of this module.

The rule, and why it is deliberately conservative:

  * it is only applied when the intent already says which KIND of entity the turn is expected to name (a doctor or a
    test) -- rescoring blind, for every kind on every turn, would "find" an entity in an utterance that was never
    about one (see CLAUDE.md's "Doctor Nobody");
  * a hypothesis that names the entity EXACTLY always wins over one that does not -- but only within the first
    `max_promote_rank` hypotheses. A rule that can reach arbitrarily far down the N-best list to justify a much
    weaker acoustic hypothesis defeats the purpose of ranking hypotheses by likelihood at all;
  * a hypothesis that only SUGGESTS an entity (a sound-alike or spelling neighbour, agent/gazetteer.py's non-exact
    tiers) is preferred over one with no candidate at all, but never over one with an exact match, and it never
    invents an entity a hypothesis does not contain;
  * the decoder's own top hypothesis (rank 0) is the answer whenever nothing beats it, so rescoring can only ever
    match or improve on not rescoring at all -- it can never make a turn's transcript worse.

It never resolves an entity to a specific record -- agent/gazetteer.py's own rule stands (a suggestion is read back to
the caller, never acted on). This only chooses which TRANSCRIPT is handed to the rest of the turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent.catalogue_forms import TITLES, doctor_forms, lab_test_forms
from agent.gazetteer import Gazetteer
from agent.transcript_rules import DOCTOR, TEST, AliasTable

EXACT = "exact"  # the hypothesis names the entity's written form, the same rule the live lookup uses
SUGGESTED = "suggested"  # only a sound-alike or spelling neighbour (agent/gazetteer.py)
NONE = "none"  # no candidate at all
_TIER_ORDER = {EXACT: 0, SUGGESTED: 1, NONE: 2}

# How far down the N-best list a hypothesis may be promoted from. REASONED, not measured: there is no real N-best
# output yet to calibrate against (see the module docstring); revisit once there is.
DEFAULT_MAX_PROMOTE_RANK = 4


@dataclass(frozen=True)
class Rescored:
    text: str  # the chosen hypothesis
    rank: int  # its position in the ORIGINAL list (0 = the decoder's own top choice)
    tier: str  # EXACT | SUGGESTED | NONE, for the chosen hypothesis
    promoted: bool  # True if this was not rank 0


class CatalogueRescorer:
    """Built once per catalogue (the same shape as agent/fast_path.Catalogue) and reused across turns; rebuilding it
    per turn would repeat the gazetteer's index build for nothing."""

    def __init__(self, cat: dict[str, Any]) -> None:
        self._table = AliasTable(cat)
        self._doctor_gaz = Gazetteer(doctor_forms(cat), drop_words=TITLES)
        self._test_gaz = Gazetteer(lab_test_forms(cat))

    def _tier(self, text: str, kind: str, lang: str) -> str:
        if any(e.kind == kind and e.value is not None for e in self._table.entities(text, lang)):
            return EXACT
        gaz = self._doctor_gaz if kind == DOCTOR else self._test_gaz
        return SUGGESTED if gaz.suggest(text) else NONE

    def rescore(
        self,
        hypotheses: Sequence[str],
        kind: str,
        lang: str = "bn",
        max_promote_rank: int = DEFAULT_MAX_PROMOTE_RANK,
    ) -> Rescored:
        """`kind`: agent.transcript_rules.DOCTOR or TEST -- which entity this turn is expected to name. `hypotheses`
        empty returns an empty result at rank 0, never an error: an empty N-best is itself a signal the caller already
        has (agent/asr.py's ASRResult.text == "")."""
        if not hypotheses:
            return Rescored("", 0, NONE, False)
        best_rank, best_text, best_tier = 0, hypotheses[0], self._tier(hypotheses[0], kind, lang)
        for rank, text in enumerate(hypotheses[: max_promote_rank + 1]):
            tier = self._tier(text, kind, lang)
            if _TIER_ORDER[tier] < _TIER_ORDER[best_tier]:
                best_rank, best_text, best_tier = rank, text, tier
        return Rescored(best_text, best_rank, best_tier, best_rank != 0)


__all__ = ["DOCTOR", "TEST", "CatalogueRescorer", "Rescored", "EXACT", "SUGGESTED", "NONE"]
