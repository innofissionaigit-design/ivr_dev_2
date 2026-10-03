"""The transcription rulebook, as code: one verbatim transcript in, one normalized transcript and the critical entities out.

Written for scoring speech recognition against a human transcript (agent/asr_metrics.py, tools/asr_eval.py); it changes
nothing in a live call. The rules (docs/transcription-rulebook.md) in one paragraph:

  VERBATIM is what was said, in the script it was said in: Bengali in Bengali script, Hindi in Devanagari, an English word
  in Latin. Nothing is romanised, corrected or completed; a spelled-out test ("সি বি সি") is written as heard.
  NORMALIZED is derived from it, never typed: the comparison form of agent/gazetteer.py (NFC, flap letters folded, nukta
  dropped, lower case, punctuation gone) with Bengali and Devanagari digits made ASCII, and each doctor or test the caller
  named replaced by ONE token naming the catalogue entry (`TEST:CBC`, `DOCTOR:Dr._A._Sen`), so "সিবিসি", "সি বি সি" and
  "CBC" compare equal. A name that belongs to more than one entry (the Bengali "রায়" is both Roy and Ray) is left as
  said and reported as ambiguous: it is not resolved here, exactly as the live agent does not resolve it.

Entity matching is exact on the normalized written form, longest form first: it never guesses from sound, so a
recognition error on a name shows up as a MISSED entity (or a wrong one), never as a silent repair.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from agent.booking_flow import classify_yes_no
from agent.catalogue_forms import TITLES, doctor_forms, lab_test_code, lab_test_forms
from agent.gazetteer import normalise
from agent.security_input import _to_numbers, _tokens

DOCTOR = "doctor"
TEST = "test"
NUMBER = "number"
YES_NO = "yes_no"
KINDS = (DOCTOR, TEST, NUMBER, YES_NO)

_ASCII_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯०१२३४५६७८९", "01234567890123456789")
# A reference this short is treated as an answer to a yes/no question and scored as one.
YES_NO_MAX_TOKENS = 4


def normalize_text(text: str) -> str:
    """The comparison form of a transcript (no entities replaced): see the module docstring."""
    return normalise(text).translate(_ASCII_DIGITS)


@dataclass(frozen=True)
class Entity:
    kind: str  # DOCTOR | TEST | NUMBER | YES_NO
    value: str | None  # the catalogue entry, the number, "yes" / "no"; None when the name is ambiguous
    surface: str  # what was said (normalized)
    candidates: tuple[str, ...] = ()  # the entries an ambiguous name could be


def _slug(text: str) -> str:
    return text.replace(" ", "_")


class AliasTable:
    """Every spoken form of every doctor and test in a catalogue, matched exactly against normalized text."""

    def __init__(self, cat: dict[str, Any]) -> None:
        self._by_tokens: dict[tuple[str, ...], dict[tuple[str, str], None]] = defaultdict(dict)
        codes: dict[str, list[str]] = defaultdict(list)
        names = {t["name"] for t in cat.get("tests", [])}
        for name in names:
            codes[lab_test_code(name)].append(name)
        # a code two tests share names neither of them: fall back to the whole name
        self._token_of_test = {
            name: (lab_test_code(name) if len(codes[lab_test_code(name)]) == 1 else name) for name in names
        }
        for kind, forms in ((DOCTOR, doctor_forms(cat)), (TEST, lab_test_forms(cat))):
            for canonical, form in forms:
                toks = tuple(normalize_text(_drop_titles(form) if kind == DOCTOR else form).split())
                if toks:
                    self._by_tokens[toks][(kind, canonical)] = None
        self._longest = max((len(k) for k in self._by_tokens), default=1)

    def token_for(self, entity: Entity) -> str:
        """The single token an entity is replaced by in the normalized transcript."""
        if entity.kind == TEST and entity.value is not None:
            return "TEST:" + _slug(self._token_of_test.get(entity.value, entity.value))
        return f"{entity.kind.upper()}:{_slug(entity.value or '?')}"

    def _scan(self, tokens: list[str]) -> list[tuple[int, int, Entity]]:
        """(start, end, entity) for each doctor or test named, longest form first, left to right."""
        found: list[tuple[int, int, Entity]] = []
        i = 0
        while i < len(tokens):
            for n in range(min(self._longest, len(tokens) - i), 0, -1):
                hits = self._by_tokens.get(tuple(tokens[i : i + n]))
                if not hits:
                    continue
                surface = " ".join(tokens[i : i + n])
                by_kind: dict[str, list[str]] = defaultdict(list)
                for kind, canonical in hits:
                    by_kind[kind].append(canonical)
                for kind, canonicals in by_kind.items():
                    canonicals = sorted(set(canonicals))
                    if len(canonicals) == 1:
                        found.append((i, i + n, Entity(kind, canonicals[0], surface)))
                    else:
                        found.append((i, i + n, Entity(kind, None, surface, tuple(canonicals))))
                i += n - 1
                break
            i += 1
        return found

    def entities(self, text: str, lang: str = "bn", *, reference: bool = False) -> list[Entity]:
        """Every critical entity in a transcript: doctors and tests named, numbers said, a yes or a no.

        `reference=True` (the human transcript): a yes/no is scored only when the utterance is short enough to be an
        answer (YES_NO_MAX_TOKENS words); a recognised transcript is always asked."""
        norm = normalize_text(text)
        tokens = norm.split()
        out = [e for _s, _e, e in self._scan(tokens)]
        out += [Entity(NUMBER, str(n), str(n)) for n in numbers_in(text)]
        if not reference or len(tokens) <= YES_NO_MAX_TOKENS:
            answer = classify_yes_no(norm, lang) if tokens else None
            if answer:
                out.append(Entity(YES_NO, answer, answer))
        return out

    def normalize(self, text: str) -> str:
        """The normalized transcript: comparison form, with each unambiguous doctor or test one token."""
        tokens = normalize_text(text).split()
        out: list[str] = []
        cursor = 0
        for start, end, entity in self._scan(tokens):
            if entity.value is None:  # ambiguous: left as said
                continue
            out += tokens[cursor:start]
            out.append(self.token_for(entity))
            cursor = end
        out += tokens[cursor:]
        return " ".join(out)


def _drop_titles(form: str) -> str:
    return " ".join(w for w in form.replace(".", " ").split() if w.lower() not in TITLES)


def numbers_in(text: str) -> list[int]:
    """The whole numbers said in a transcript, in order, whether written as digits or as words in the three languages
    (the same parser the identity questions use: agent/security_input.py)."""
    return [n for n in _to_numbers(_tokens(text)) if isinstance(n, int)]


def by_kind(entities: Iterable[Entity]) -> dict[str, list[Entity]]:
    out: dict[str, list[Entity]] = {k: [] for k in KINDS}
    for e in entities:
        out.setdefault(e.kind, []).append(e)
    return out
