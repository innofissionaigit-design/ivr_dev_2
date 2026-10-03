"""The spoken and written forms of every doctor and test in the clinic catalogue, from the one place that knows the shape.

`/api/v1/catalogue` answers with each test's name and its Bengali and Hindi aliases and each doctor's surname, whole name
and aliases. The confusable-pairs report (tools/confusable_pairs.py) and the transcript rules (agent/transcript_rules.py)
both need "every form of every entity"; this is that, once. Pure: it reads a dict, it calls nothing.
"""

from __future__ import annotations

from typing import Any

# The words a caller puts before a doctor's name; they identify nobody, so a comparison drops them.
TITLES = frozenset({"dr", "doctor", "doc", "ডাক্তার", "ডক্টর", "ডাঃ", "डॉक्टर", "डॉ", "डाक्टर"})


def doctor_forms(cat: dict[str, Any]) -> list[tuple[str, str]]:
    """(canonical doctor name, one form of it): the surname, the whole name in three scripts, every alias."""
    out: list[tuple[str, str]] = []
    for d in cat.get("doctors", []):
        name = str(d["name"])
        out.append((name, str(d.get("surname") or name.split()[-1])))
        forms = [d.get("full_name"), d.get("full_name_bn"), d.get("full_name_hi")]
        forms += list(d.get("aliases_bn", [])) + list(d.get("aliases_hi", []))
        out += [(name, str(f)) for f in forms if f]
    return out


def lab_test_forms(cat: dict[str, Any]) -> list[tuple[str, str]]:
    """(canonical test name, one form of it): the name, the part before and inside the brackets, every alias."""
    out: list[tuple[str, str]] = []
    for t in cat.get("tests", []):
        name = str(t["name"])
        forms = {name, name.split("(")[0].strip()}
        if "(" in name:
            forms.add(name.split("(")[1].rstrip(") ").strip())
        forms |= set(t.get("aliases_bn", [])) | set(t.get("aliases_hi", []))
        out += [(name, f) for f in sorted(forms) if f]
    return out


def _code_like(word: str) -> bool:
    return len(word) >= 2 and word.isalnum() and word.upper() == word and any(c.isalpha() for c in word)


def lab_test_code(name: str) -> str:
    """The short code a test is known by, when it has one: "Complete Blood Count (CBC)" -> "CBC", "TMT (Treadmill Test)" ->
    "TMT", "HIV Test (ELISA)" -> "HIV", "HbA1c" -> "HbA1c". Otherwise the whole name. Codes are a convenience for reading a
    transcript; agent/transcript_rules.py checks they are unique in the catalogue and falls back to the name if not."""
    head, _, rest = name.partition("(")
    words, inner = head.split(), rest.rstrip(") ").strip()
    if not words:
        return name.strip()
    if rest and _code_like(words[0]):
        return words[0]
    if len(words) == 1:
        return words[0]
    if _code_like(inner) and not inner[0].isdigit():
        return inner
    return name.strip()
