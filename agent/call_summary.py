"""A short, factual note of what a call was about, for the callback/prescription record a human colleague reads.

Built ENTIRELY from facts the call already has -- the caller's own words, quoted verbatim, and slots already
collected -- never composed or paraphrased by the model. CLAUDE.md's truth boundary is about facts told TO the
caller, but the same discipline applies to a record written ABOUT the caller: a human deciding what to do from this
note must be able to trust every word in it is something that was actually said or already known, not an inference.

    build_call_summary("lay_term_blood", lang="bn", transcript="আমার রক্ত পরীক্ষা করাতে হবে", phone="9830012345")
      -> "reason: lay_term_blood | language: bn | caller said: \"আমার রক্ত পরীক্ষা করাতে হবে\" | phone: 9830012345"

Kept to plain key: value pairs, one line, so it reads the same whether a human or a script opens it later.
"""

from __future__ import annotations

MAX_TRANSCRIPT_CHARS = 300  # long enough for a sentence or two, short enough that this never becomes a transcript dump


def build_call_summary(
    reason: str,
    *,
    lang: str,
    transcript: str = "",
    phone: str | None = None,
    patient_name: str | None = None,
    call_id: str | None = None,
) -> str:
    """`reason` is a short machine-readable tag the caller chooses (e.g. "lay_term_blood", "test_not_found",
    "handoff:decoders_disagree") -- this function does not interpret it, only records it, so it can never be wrong
    about why the summary was written."""
    clipped = transcript.strip()
    if len(clipped) > MAX_TRANSCRIPT_CHARS:
        clipped = clipped[:MAX_TRANSCRIPT_CHARS].rstrip() + "..."
    parts = [f"reason: {reason}", f"language: {lang}"]
    if clipped:
        parts.append(f'caller said: "{clipped}"')
    if patient_name:
        parts.append(f"name: {patient_name}")
    if phone:
        parts.append(f"phone: {phone}")
    if call_id:
        parts.append(f"call_id: {call_id}")
    return " | ".join(parts)
