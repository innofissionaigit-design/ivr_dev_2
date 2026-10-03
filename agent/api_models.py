"""The shape of what the clinic API may answer, checked where the answer enters the agent (agent/tools_client.py).

The agent states facts from these answers (a price, a time, a confirmation number), so an answer of the wrong shape must
not be spoken from: it is a tool failure ("I can't check that right now"), never a half-understood reply. Each model:

  * requires the ONE field the agent branches on (`found`, `success`, `status`, `matched`, `conflict`, `verified`) with the
    right type, so a body like {"error": "..."} or an HTML error page is rejected instead of read as "not found";
  * types the fields the reply templates rely on (a price is a number, a list of slots is a list of strings);
  * allows every other field (`extra="allow"`), because the API adds fields and the agent must not break when it does.

`validated(model, payload, what)` returns a plain dict of exactly the keys the API sent (unset fields are not invented),
so nothing downstream changes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt, ValidationError

# Strict types: a discriminator or a number that arrives as the WRONG type (the string "yes", the string "250") is a bad
# answer, not something to coerce into a fact.
Number = StrictInt | StrictFloat


class ApiShapeError(ValueError):
    """The clinic API answered, but not in the shape this client requires."""


class _Answer(BaseModel):
    model_config = ConfigDict(extra="allow")


class FoundAnswer(_Answer):
    found: StrictBool


class TestAnswer(FoundAnswer):
    __test__ = False  # not a pytest class, whatever the name says
    test_name: str | None = None
    rate_inr: Number | None = None
    sample_type: str | None = None
    report_time_hours: Number | None = None
    query: str | None = None
    did_you_mean: list[str] | None = None
    did_you_mean_bn: list[str] | None = None
    did_you_mean_hi: list[str] | None = None
    ambiguous: StrictBool | None = None
    needs_confirmation: StrictBool | None = None


class DoctorAnswer(FoundAnswer):
    doctor_name: str | None = None
    date: str | None = None
    available: StrictBool | None = None
    chamber_hours: str | None = None
    next_available_date: str | None = None
    query: str | None = None
    did_you_mean: list[str] | None = None
    did_you_mean_bn: list[str] | None = None
    did_you_mean_hi: list[str] | None = None
    ambiguous: StrictBool | None = None
    needs_confirmation: StrictBool | None = None


class PrepAnswer(FoundAnswer):
    test_name: str | None = None
    fasting_required: StrictBool | None = None
    prep_instructions: str | None = None
    ambiguous: StrictBool | None = None
    did_you_mean: list[str] | None = None


class PrepMergeAnswer(FoundAnswer):
    """POST /api/v1/tests/prep/merge -- clinic-api/enquiry_service.py::merge_prep_instructions(), KCD-382. The
    agent never computes the merge itself (CLAUDE.md rule 1): `fasting_hours` is the strictest MAX() the API
    already took, and `escalate_to_human` (set whenever `incomplete_data` names a resolved test the merge could
    not safely fold in) is the one signal the voice layer is allowed to act on without re-deriving it."""

    test_names: list[str] | None = None
    fasting_hours: Number | None = None
    water_allowed_while_fasting: StrictBool | None = None
    prescription_required: StrictBool | None = None
    escalate_to_human: StrictBool | None = None
    incomplete_data: list[str] | None = None
    not_found: list[str] | None = None


class FaqAnswer(FoundAnswer):
    topic: str | None = None
    answer: str | None = None


class LookupAnswer(FoundAnswer):
    bookings: list[dict[str, Any]] = []


class SuccessAnswer(_Answer):
    """A write: it either succeeded or says why not."""

    success: StrictBool
    reason: str | None = None
    confirmation_id: str | None = None
    hold_token: str | None = None
    doctor_id: StrictInt | None = None
    doctor_name: str | None = None
    date: str | None = None
    time_slot: str | None = None
    alternative_slots: list[str] | None = None
    charge_inr: Number | None = None


class ConflictAnswer(_Answer):
    conflict: StrictBool
    existing: dict[str, Any] | None = None


class RouteAnswer(_Answer):
    matched: StrictBool


class IdentifyAnswer(_Answer):
    status: Literal["new", "single", "ambiguous"]
    record_exists: StrictBool | None = None
    count: StrictInt | None = None
    patient_ref: int | str | None = None


class FindAnswer(_Answer):
    status: str
    patient_ref: int | str | None = None


class VerifyAnswer(_Answer):
    verified: StrictBool
    attempts_left: StrictInt | None = None
    locked: StrictBool | None = None


def validated(model: type[BaseModel], payload: Any, what: str) -> dict[str, Any]:
    """`payload` checked against `model`, as the dict of the keys the API actually sent. Raises ApiShapeError (a
    ValueError) with the field names that were wrong -- never the values, which may be personal."""
    try:
        return model.model_validate(payload).model_dump(exclude_unset=True)
    except ValidationError as e:
        bad = ", ".join(sorted({".".join(str(p) for p in err["loc"]) or "<body>" for err in e.errors()}))
        raise ApiShapeError(f"{what}: unexpected answer shape (problem fields: {bad})") from e
