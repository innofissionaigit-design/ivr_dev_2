"""KCD-442/KCD-447: what a low-ASR-confidence turn is allowed to do.

agent/asr.py already computes `ASRResult.decoder_agreement` (CTC vs RNNT
agreement, 0.0-1.0) but nothing downstream ever reads it once language
routing is done -- exactly the gap both stories point at (see their Repo
Evidence). This module is the one place that decides what "low
confidence" means and what it implies, so main.py's dispatch consults a
single flag instead of re-deriving the threshold at each call site.

Two distinct consequences, matching the two stories:
  - KCD-442: a low-confidence turn asking a FACTUAL question (test_rate,
    doctor_availability, test_prep, clinic_faq) does not run the lookup
    at all -- a misheard entity name silently returning a confident,
    wrong answer for a DIFFERENT test/doctor is exactly the kind of
    quietly-wrong fact CLAUDE.md's truth boundary exists to prevent.
    agent/reply_templates.py's insufficient_information_reply() is
    spoken instead, distinct from "not found" (KCD-443, already Done)
    and from an infrastructure apology.
  - KCD-447: a low-confidence turn is never enough on its own to commit a
    WRITE (a booking hold/confirm/cancel/reschedule). Booking actions in
    this codebase already never write without passing through
    booking_flow.mark_confirming + an explicit classify_yes_no "yes" on
    a LATER turn (pinned locally by tests/test_booking_flow.py's KCD-484
    tests -- the state machine never reaches a write stage on its own and
    classify_yes_no never reads an uncertain answer as yes; the commit
    itself lives in main.py and is verified on-pod only); this module's role for
    writes is only to make that guarantee explicit and testable, not to
    add a second gate on top of it.

THRESHOLD is REASONED, not measured -- there is no real-call corpus
locally to calibrate the boundary between "ASR was genuinely unsure" and
"ASR was fine", the same "measured vs reasoned" caveat fast_path.py's
FAQ_COMMIT_FLOOR already carries. Recalibrate against real call audio
before trusting this past the pilot.
"""

from __future__ import annotations

# Below this, the two decoders disagreed on more than half the turn's
# words. agent/asr.py's dual-decoder agreement check already treats 0.0
# (CTC-fallback-only, RNNT unavailable) as "trust it anyway" for routing
# purposes -- this threshold is deliberately ABOVE that special case, so
# a genuine low-agreement turn (both decoders ran, and disagreed a lot)
# is caught, not the "only one decoder was even available" case.
LOW_CONFIDENCE_FLOOR = 0.5

# Read-only factual lookups a low-confidence turn must not silently act
# on -- acting on a misheard test/doctor name would produce a confident,
# wrong answer about something else entirely. Booking actions are
# deliberately excluded: they already require a separate confirm turn
# regardless of confidence (see module docstring).
GATED_FACTUAL_INTENTS = frozenset(
    {
        "test_rate",
        "doctor_availability",
        "test_prep",
        "clinic_faq",
        # an external "no guessing" review: every fact retrieval, and every lookup that discloses a
        # booking, is gated -- not just four intents.
        "department_query",
        "lookup_booking",
        "resend_confirmation",
    }
)

# --- three states, not a float convention -----------------------------------------------------
# `decoder_agreement == 0.0` is ambiguous: agent/asr.py uses it for "only the CTC decoder produced
# text" (nothing to compare with), and a genuine total disagreement also rounds to 0.0. A float
# convention that reads the first as "trust it" gives one decoder the authority of two agreeing
# ones. The decoder that actually produced the text is recorded (`decoder_used`); this uses it.
VERIFIED = "verified"  # both decoders ran and agree
LOW = "low"  # both decoders ran and disagree
UNAVAILABLE = "unavailable"  # only one decoder produced the text: nothing to check it against


def confidence_state(decoder_used: str | None, decoder_agreement: float) -> str:
    """VERIFIED / LOW / UNAVAILABLE for one recognised turn.

    Only an "rnnt" result carries an agreement measurement (the CTC decoder ran too). "ctc_fallback",
    a single-decoder model ("fastconformer") and an unknown decoder are UNAVAILABLE. With no decoder
    recorded at all (a caller that predates `decoder_used`) the old float rule applies unchanged."""
    if decoder_used is None:
        return LOW if is_low_confidence(decoder_agreement) else VERIFIED
    if decoder_used == "rnnt":
        return LOW if decoder_agreement < LOW_CONFIDENCE_FLOOR else VERIFIED
    return UNAVAILABLE


def is_low_confidence(decoder_agreement: float) -> bool:
    """decoder_agreement == 0.0 without a genuine disagreement measurement
    (agent/asr.py's CTC-fallback path) is NOT treated as low confidence
    here -- that path already carries its own explicit "no RNNT decoder
    to compare against" meaning, distinct from "compared and disagreed"."""
    return 0.0 < decoder_agreement < LOW_CONFIDENCE_FLOOR


def should_withhold_factual_answer(intent: str, decoder_agreement: float, decoder_used: str | None = None) -> bool:
    """LOW confidence on a gated intent: the lookup does not run. (UNAVAILABLE is handled by reading
    the entity back -- agent/entity_confirmation.py -- not by refusing.)"""
    if intent not in GATED_FACTUAL_INTENTS:
        return False
    return confidence_state(decoder_used, decoder_agreement) == LOW


def needs_entity_readback(intent: str, decoder_used: str | None, decoder_agreement: float) -> bool:
    """UNAVAILABLE confidence on a gated intent: ask "do you mean X?" before the lookup runs."""
    return intent in GATED_FACTUAL_INTENTS and confidence_state(decoder_used, decoder_agreement) == UNAVAILABLE
