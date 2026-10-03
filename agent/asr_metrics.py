"""How wrong a recogniser is, in numbers that can be compared before and after a change.

Pure and dependency-free (no jiwer, no numpy): the arithmetic is small and every figure here is one a reviewer should be
able to recompute by hand.

  * WER / CER: word and character error rate over a WHOLE corpus, i.e. total edits divided by total reference length. The
    mean of per-utterance rates is not used: one short utterance with one error would weigh as much as a long one.
  * A confidence interval by resampling CALLERS, not utterances: utterances from one caller are not independent, and a
    test set of a few callers is much less certain than its utterance count suggests. The interval is what stops a
    two-point WER "improvement" on ten callers being reported as one.
  * Critical entities (doctor, test, number, yes/no) counted separately from WER: a transcript can have a low WER and
    still have the one wrong word that books the wrong test. For each kind: how many were said, how many came back right,
    how many were missed, how many WRONG ones were heard (the dangerous kind), and how many came back as a name that
    belongs to several entries (which the agent must ask about, so they are neither right nor wrong).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from agent.transcript_rules import KINDS, Entity


@dataclass(frozen=True)
class Edits:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """errors / reference length. Zero-length reference: 0.0 if nothing was recognised, else 1.0 (all insertions)."""
        if self.reference_length == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_length

    def __add__(self, other: Edits) -> Edits:
        return Edits(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )


def align(ref: Sequence[str], hyp: Sequence[str]) -> Edits:
    """Minimum edits turning `ref` into `hyp` (Levenshtein), split into substitutions, deletions and insertions.
    Ties are broken in the order match, substitution, deletion, insertion, so the split is reproducible."""
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            cost[i][j] = min(sub, cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            subs += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return Edits(subs, dels, ins, n)


def word_edits(ref: str, hyp: str) -> Edits:
    """Edits between two NORMALIZED transcripts (agent/transcript_rules.py), split on spaces."""
    return align(ref.split(), hyp.split())


def char_edits(ref: str, hyp: str) -> Edits:
    """The same, per character with spaces ignored (the customary CER for languages written without fixed word breaks)."""
    return align(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))


def bootstrap_interval(
    by_group: dict[str, Edits], resamples: int = 1000, seed: int = 20260927, level: float = 0.95
) -> tuple[float, float]:
    """The interval for a corpus error rate, resampling whole GROUPS (callers) with replacement. Fewer than two groups
    gives (rate, rate): there is nothing to resample, and the caller should say so rather than print a tight interval."""
    groups = list(by_group.values())
    total = sum(groups, Edits())
    if len(groups) < 2:
        return total.rate, total.rate
    rng = random.Random(seed)
    rates = sorted(sum((rng.choice(groups) for _ in groups), Edits()).rate for _ in range(resamples))
    lo = rates[int(((1 - level) / 2) * resamples)]
    hi = rates[min(resamples - 1, int((1 - (1 - level) / 2) * resamples))]
    return lo, hi


# ------------------------------------------------------------------------------------------------ critical entities


@dataclass
class EntityScore:
    said: int = 0  # entities in the reference
    correct: int = 0  # said, and the recognised transcript has it
    missed: int = 0  # said, and the recognised transcript does not
    wrong: int = 0  # NOT said, but the recognised transcript has it: the wrong doctor, the wrong test, a wrong number
    ambiguous: int = 0  # the recognised transcript names something that belongs to several entries

    def __iadd__(self, other: EntityScore) -> EntityScore:
        self.said += other.said
        self.correct += other.correct
        self.missed += other.missed
        self.wrong += other.wrong
        self.ambiguous += other.ambiguous
        return self

    @property
    def accuracy(self) -> float | None:
        """correct / said; None when nothing of this kind was said (a rate over zero is not zero)."""
        return None if self.said == 0 else self.correct / self.said


def score_entities(ref: Iterable[Entity], hyp: Iterable[Entity]) -> dict[str, EntityScore]:
    """Per kind, compare the entities said with the entities recognised. Values are matched as multisets: saying "CBC" once
    and hearing it twice is one correct and one wrong."""
    out = {k: EntityScore() for k in KINDS}
    for kind in KINDS:
        want = Counter(e.value for e in ref if e.kind == kind and e.value is not None)
        heard_all = [e for e in hyp if e.kind == kind]
        got = Counter(e.value for e in heard_all if e.value is not None)
        ambiguous = [e for e in heard_all if e.value is None]
        s = out[kind]
        s.said = sum(want.values())
        s.correct = sum((want & got).values())
        s.ambiguous = len(ambiguous)
        missing = want - got
        # an ambiguous name that could be one of the missing entities is not a miss: the agent asks, the caller settles it
        for e in ambiguous:
            for value in e.candidates:
                if missing.get(value, 0) > 0:
                    missing[value] -= 1
                    break
        s.missed = sum(missing.values())
        s.wrong = sum((got - want).values())
    return out


@dataclass
class CorpusScore:
    """Everything the evaluation reports for one group of utterances."""

    utterances: int = 0
    words: Edits = field(default_factory=Edits)
    chars: Edits = field(default_factory=Edits)
    entities: dict[str, EntityScore] = field(default_factory=lambda: {k: EntityScore() for k in KINDS})
    by_caller: dict[str, Edits] = field(default_factory=dict)

    def add(self, caller: str, words: Edits, chars: Edits, entities: dict[str, EntityScore]) -> None:
        self.utterances += 1
        self.words += words
        self.chars += chars
        for kind, s in entities.items():
            self.entities.setdefault(kind, EntityScore())
            self.entities[kind] += s
        self.by_caller[caller] = self.by_caller.get(caller, Edits()) + words
