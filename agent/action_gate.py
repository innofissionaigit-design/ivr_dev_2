"""One place that says what the agent may do next: act, or first ask, read back, confirm, repeat or hand over.

Until now the same idea was spread over several modules: what a low-confidence turn may do (agent/confidence_gate.py),
when an entity is read back before a lookup (agent/entity_confirmation.py), the deterministic yes before any write
(agent/booking_flow.py), and how many times to ask again before a human takes over (agent/reask_policy.py). Each is right
on its own. This module states the RULE they share, so it can be read, tested and later called from one place:

    the agent acts automatically only when the recognition, the entity it named and the business rules are ALL certain;
    otherwise it asks, reads back, confirms, or hands over. Never a guess.

It is a pure function of facts the caller already has. It does not read audio, call a model or touch the network, and
nothing in the running agent calls it yet. `tests/test_action_gate.py` checks it against the existing gates on every
combination: for every READ it gives exactly the answer they give. For a WRITE it is deliberately one step stricter (a
low-confidence turn is repeated instead of moving on to the confirmation question); that difference is marked in the test.
Wiring it in is therefore a decision about that one step, not a refactor.

The order below is the rule, strongest reason first:

  1. too many failed attempts on this turn's question                          -> HANDOFF
  2. the recognition is LOW confidence (the two decoders disagreed)            -> REPEAT
  3. the entity is ambiguous or was never named                                -> CLARIFY   ("which one?")
  4. the entity was only a sound-alike or spelling neighbour                   -> READ_BACK ("do you mean X?")
  5. the recognition is UNAVAILABLE (one decoder, nothing to check it against) -> READ_BACK
  6. a business rule forbids it (no slot, already cancelled)                   -> DECLINE
  7. a write the caller has not yet confirmed with a deterministic yes         -> CONFIRM
  8. otherwise                                                                 -> ACT
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.confidence_gate import GATED_FACTUAL_INTENTS, LOW, UNAVAILABLE, confidence_state
from agent.reask_policy import DEFAULT_MAX_REASKS

ACT = "act"
CONFIRM = "confirm"
READ_BACK = "read_back"
CLARIFY = "clarify"
REPEAT = "repeat"
DECLINE = "decline"
HANDOFF = "handoff"

# How the entity a turn named was resolved. Only EXACT may be acted on.
EXACT = "exact"  # matched on its written form (the clinic API's rule: never on sound alone)
AMBIGUOUS = "ambiguous"  # more than one entry fits (two doctors called Roy)
SUGGESTED = "suggested"  # only a sound-alike or spelling neighbour: a candidate to read back, not an answer
ABSENT = "absent"  # the caller named nothing the catalogue has
NOT_NEEDED = "not_needed"  # the intent names no entity

WRITE_INTENTS = frozenset(
    {"book_test", "add_test_booking", "book_appointment", "cancel_appointment", "reschedule_appointment"}
)


@dataclass(frozen=True)
class Verdict:
    action: str  # ACT | CONFIRM | READ_BACK | CLARIFY | REPEAT | DECLINE | HANDOFF
    reason: str  # the first rule that applied, in words the log can carry

    @property
    def may_act(self) -> bool:
        return self.action == ACT


def decide(
    intent: str,
    *,
    decoder_used: str | None,
    decoder_agreement: float,
    entity: str = NOT_NEEDED,
    rules_ok: bool = True,
    confirmed: bool = False,
    failed_attempts: int = 0,
    max_attempts: int = DEFAULT_MAX_REASKS,
) -> Verdict:
    """What to do with this turn. `confirmed` is whether the caller has already given a deterministic yes to the read-back
    of THIS action (agent/booking_flow.classify_yes_no); it matters only for a write."""
    if failed_attempts > max_attempts:
        return Verdict(HANDOFF, "too many failed attempts")
    gated = intent in GATED_FACTUAL_INTENTS or intent in WRITE_INTENTS
    state = confidence_state(decoder_used, decoder_agreement)
    if gated and state == LOW:
        return Verdict(REPEAT, "the recognisers disagreed")
    if entity in (AMBIGUOUS, ABSENT):
        return Verdict(CLARIFY, "the name is ambiguous" if entity == AMBIGUOUS else "no such name")
    if entity == SUGGESTED:
        return Verdict(READ_BACK, "only a sound-alike was found")
    if intent in GATED_FACTUAL_INTENTS and state == UNAVAILABLE:
        return Verdict(READ_BACK, "one decoder only: nothing to check it against")
    if not rules_ok:
        return Verdict(DECLINE, "a business rule forbids it")
    if intent in WRITE_INTENTS and not confirmed:
        return Verdict(CONFIRM, "a write needs the caller's yes")
    return Verdict(ACT, "recognition, entity and rules are certain")
