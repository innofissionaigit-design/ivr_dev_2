"""The record kept for every recorded utterance, and the checks that keep an evaluation honest.

One JSON object per line (a "manifest"). It names the audio, who spoke, the verbatim transcript, and what made the
utterance hard; the normalized transcript and the critical entities are derived from the verbatim text
(agent/transcript_rules.py), never typed twice. Three checks matter more than the fields:

  * SPEAKER SPLIT: a caller must be in exactly one of train / dev / test. Splitting by utterance puts the same voice on
    both sides and makes a model look better than it is.
  * LOCKED TEST SET: the test rows are fixed by a digest of their ids. If the set changes, the digest changes, and a
    before/after comparison on it is no longer a comparison.
  * REAL AND SYNTHETIC NEVER MIX: a real row needs the caller's consent on record; a synthetic row (text-to-speech) is
    labelled, kept in its own group, and never contributes to a real-caller figure.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Split = Literal["train", "dev", "test"]
Source = Literal["real", "synthetic"]
Language = Literal["bn", "hi", "en", "mixed"]

# What made an utterance hard, so a failure is never averaged away (a rural-accent WER hidden inside an urban average).
TAGS = frozenset(
    {
        "rural_accent",
        "strong_dialect",
        "noise",
        "speakerphone",
        "low_volume",
        "overlap",
        "clipped",
        "code_switch",
        "telephone_8k",
        "elderly",
        "child",
    }
)


class ManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    audio: str = Field(min_length=1, description="path of the ORIGINAL audio, not a cleaned copy")
    speaker: str = Field(min_length=1, description="an opaque caller id, never a name or a phone number")
    split: Split
    source: Source
    language: Language
    verbatim: str = Field(description="exactly what was said, in the script it was said in")
    tags: list[str] = Field(default_factory=list)
    locked: bool = Field(default=False, description="part of the locked test set")
    consent: bool = Field(default=False, description="the caller was told the call is recorded and agreed")
    original_sample_rate: int | None = Field(default=None, gt=0)
    duration_s: float | None = Field(default=None, gt=0)
    entities: list[dict[str, str]] | None = Field(
        default=None,
        description="optional: the entities said, [{'kind': 'doctor', 'value': 'Dr. A. Sen'}], when the text alone is ambiguous",
    )

    @model_validator(mode="after")
    def _consistent(self) -> ManifestRow:
        unknown = sorted(set(self.tags) - TAGS)
        if unknown:
            raise ValueError(f"unknown tag(s) {unknown}; allowed: {sorted(TAGS)}")
        if self.source == "real" and not self.consent:
            raise ValueError("a real recording needs consent on record")
        if self.locked and self.split != "test":
            raise ValueError("a locked row must be in the test split")
        if self.locked and self.source != "real":
            raise ValueError("the locked test set is real callers only; synthetic rows never enter it")
        return self


def parse_manifest(lines: Iterable[str]) -> tuple[list[ManifestRow], list[str]]:
    """(rows, problems). A bad line is reported with its number and skipped; the rest are still read."""
    rows: list[ManifestRow] = []
    problems: list[str] = []
    ids: set[str] = set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = ManifestRow.model_validate(json.loads(line))
        except (ValueError, ValidationError) as exc:  # ValidationError is a ValueError; both named for the reader
            problems.append(f"line {number}: {str(exc).splitlines()[0] if str(exc) else 'invalid'}")
            continue
        if row.id in ids:
            problems.append(f"line {number}: duplicate id {row.id!r}")
            continue
        ids.add(row.id)
        rows.append(row)
    return rows, problems


def speaker_split_problems(rows: Iterable[ManifestRow]) -> list[str]:
    """Every caller that appears in more than one split."""
    splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        splits[r.speaker].add(r.split)
    return [
        f"speaker {s!r} is in {sorted(v)}: a caller must be in exactly one split"
        for s, v in sorted(splits.items())
        if len(v) > 1
    ]


def locked_digest(rows: Iterable[ManifestRow]) -> str:
    """A digest of the locked test set's ids. Adding, removing or swapping a row changes it."""
    ids = sorted(r.id for r in rows if r.locked)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def locked_problems(rows: Iterable[ManifestRow], expected_digest: str | None) -> list[str]:
    """The locked set must be non-empty, its rows must all be test rows, and (once a digest has been recorded) unchanged."""
    rows = list(rows)
    locked = [r for r in rows if r.locked]
    problems: list[str] = []
    if not locked:
        problems.append("no locked test rows: mark the fixed benchmark with locked=true")
    elif expected_digest is not None and locked_digest(rows) != expected_digest:
        problems.append(
            "the locked test set has changed since its digest was recorded; a before/after comparison is void"
        )
    return problems
