"""KCD-061: the call state object -- Blueprint 4.5's single, versioned
carrier of caller signals, read by every downstream policy/response
layer instead of each one re-deriving its own picture from scratch.

Written ONLY by call intelligence (this session's LID/ASR-confidence
work, and whatever Epic E?? empathy detectors land later -- see
CALLER_STATE_DEFAULT's docstring for what is real today vs. still a
placeholder). Read by policy/response layers (agent/reply_templates.py,
main.py's dispatch) -- never mutated by them.

Versioned (SCHEMA_VERSION) so a logged/serialized CallState from an
older build is recognisable rather than silently misread by a newer
reader with different field meanings.

No personal data lives here by construction -- every field is a signal
about HOW the caller is being served (language, confidence, a caller-
state category), never WHO they are (no name, phone, or transcript).
to_log_dict() exists anyway, and is the only sanctioned way to log a
CallState, so a future field addition cannot silently start leaking PII
into logs the way an ad-hoc dict.items() dump could.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SCHEMA_VERSION = 1

# The caller-state categories Appendix C's delivery-parameter table is
# keyed on. "neutral" is the only one any detector in this codebase can
# actually set today -- distress/anger/senior detection is Epic
# "Empathy, Distress and Emergency" (KCD-188/189/190), not yet built.
# Every OTHER field below that would ordinarily be DERIVED from those
# states (speech_rate, response_length, ...) is therefore fixed at its
# neutral-caller default for now: present in the schema (so a reader
# never has to special-case a missing field), honestly inert (so
# nothing here claims an adaptation that is not actually happening).
CALLER_STATE_NEUTRAL = "neutral"
VALID_CALLER_STATES = (CALLER_STATE_NEUTRAL, "distressed", "angry", "confused", "senior", "emergency")


@dataclass
class CallState:
    schema_version: int = SCHEMA_VERSION

    # ---- language (real: set every turn from agent/lid.py's per-utterance
    # routing decision, via main.py's session.lang) ----
    language: str = "bn"

    # ---- confidence / confirmation (real: agent/confidence_gate.py) ----
    # True exactly when the current turn's decoder_agreement is below
    # confidence_gate.LOW_CONFIDENCE_FLOOR -- the same signal that
    # withholds a factual answer (KCD-442) and would gate a write
    # (KCD-447), surfaced here so a policy/response layer can read ONE
    # flag instead of re-deriving it from a raw ASR field.
    confirmation_required: bool = False

    # ---- caller state (placeholder until Epic "Empathy, Distress and
    # Emergency" lands real detectors -- see module docstring) ----
    caller_state: str = CALLER_STATE_NEUTRAL
    senior: bool = False

    # ---- channel (real: agent/channel_quality.py, run per turn through
    # agent/detector_budget.py) ----
    channel_quality: str = "clean_16k"

    # ---- delivery parameters (Appendix C), all at their neutral-caller
    # default until something in caller_state actually drives them ----
    speech_rate: float = 1.0  # 1.0 = normal TTS rate
    response_length: str = "normal"  # "normal" | "short" (fewer clauses)
    one_question_at_a_time: bool = True  # already this codebase's baseline behaviour
    interruption_tolerance: str = "normal"  # "normal" | "patient" (wider endpointing)
    escalation_threshold: str = "normal"  # "normal" | "lowered" (offer a human sooner)

    def to_log_dict(self) -> dict:
        """The only sanctioned way to log a CallState -- see module
        docstring for why this exists as its own method rather than a
        bare asdict() call site scattered wherever logging happens."""
        return asdict(self)


def new_call_state() -> CallState:
    return CallState()


def apply_language(state: CallState, language: str) -> None:
    state.language = language


def apply_confidence(state: CallState, requires_confirmation: bool) -> None:
    state.confirmation_required = requires_confirmation


def apply_channel_quality(state: CallState, channel_quality: str) -> None:
    state.channel_quality = channel_quality


def apply_caller_state(state: CallState, caller_state: str | None = None, senior: bool | None = None) -> None:
    """KCD-149: record the caller's state and re-derive every delivery
    parameter from Appendix C's table (agent/speech_policy.py) -- the only
    place those fields are written, so they can never disagree with the
    state that is supposed to drive them. `None` leaves that signal as it
    was (senior is sticky for the call: once detected it is not silently
    cleared by a later turn that merely lacks evidence)."""
    from agent.speech_policy import as_call_state_fields, derive_policy

    if caller_state is not None:
        state.caller_state = caller_state if caller_state in VALID_CALLER_STATES else CALLER_STATE_NEUTRAL
    if senior is not None:
        state.senior = senior
    for name, value in as_call_state_fields(derive_policy(state.caller_state, state.senior)).items():
        setattr(state, name, value)
