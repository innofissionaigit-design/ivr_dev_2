"""Which names in the clinic catalogue can be mistaken for each other, worked out from the catalogue alone.

No recordings are needed for this. It uses the same folding the running system uses (agent/gazetteer.py), so the answer
is "which pairs would the live matcher itself find close", not a separate opinion. Two kinds of finding:

  * confusable pairs   two DIFFERENT entities with a spoken form that is written the same, sounds the same, or is one
                       consonant apart. Wherever one of these exists the agent must ask "which one?" and must never
                       resolve on its own;
  * shared words       a word (`sugar`, `dengue`, `usg`) that appears in the names of two or more entities. A caller
                       who says only that word has not said which test they mean.

It never decides anything at run time; it is a report for the people who own the catalogue and for choosing what to
test first. REASONED, NOT MEASURED: "risk" is a rule of thumb about how a phone line mangles speech, not a rate.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from agent.gazetteer import _within_one_edit, normalise, script_of, vowelled_key
from agent.phonetic_match import phonetic_key

# strongest evidence first; `risk` follows from it
SAME_WRITTEN = "same_written_form"  # the two entities share a spoken form outright
SAME_SOUND = "same_sound"  # same consonant skeleton (three or more classes)
SAME_SOUND_WEAK = "same_sound_weak"  # same skeleton, but too short or too long to be strong evidence (see _compare)
SAME_SOUND_SHORT = "same_sound_short"  # a short name: same consonants AND the same coarse vowels
ONE_APART = "one_consonant_apart"  # skeletons differ by one dropped, added or changed consonant
SPELLING = "similar_spelling"  # same script, and the characters are close

_ORDER = {SAME_WRITTEN: 0, SAME_SOUND: 1, SAME_SOUND_SHORT: 2, SAME_SOUND_WEAK: 3, ONE_APART: 4, SPELLING: 5}
_RISK = {
    SAME_WRITTEN: "high",
    SAME_SOUND: "high",
    SAME_SOUND_SHORT: "high",
    SAME_SOUND_WEAK: "medium",
    ONE_APART: "medium",
    SPELLING: "low",
}

MIN_SKELETON = 3  # a shorter skeleton is not evidence by itself (agent/gazetteer.py)
NEAR_MIN_SKELETON = 4
SPELLING_FLOOR = 0.75  # same-script similarity a merely similar spelling must reach to be listed
# The skeleton keeps at most six classes (agent/phonetic_match.py), so two long, unrelated names can share one. Within a
# script, a same-sound pair is listed only if the characters are also this alike; across scripts the skeleton is all
# there is to compare, so it stands alone.
SAME_SOUND_FLOOR = 0.5
# A skeleton of exactly three classes is shared by many unrelated words, and one of six is the cap (the rest of the
# word is not in it), so neither is strong on its own; they count as strong only when the characters agree too.
STRONG_SKELETON = range(4, 6)
STRONG_RATIO = 0.7
MIN_WORD_CHARS = 3  # "বি", "सी", "का": a spoken letter or a particle, shared by many names and identifying none

# words that carry no identity when a caller wraps a name in a sentence
GENERIC_WORDS = frozenset(
    {"test", "tests", "টেস্ট", "टेस्ट", "profile", "প্রোফাইল", "प्रोफाइल", "the", "a", "of", "for", "and", "&"}
)


@dataclass(frozen=True)
class Confusable:
    a: str  # canonical name of one entity
    b: str  # canonical name of the other (a < b)
    form_a: str  # the spoken form of `a` that came closest to `b`
    form_b: str
    basis: str  # SAME_WRITTEN | SAME_SOUND | ...
    similarity: float  # same-script character similarity, 0.0 across scripts

    @property
    def risk(self) -> str:
        return _RISK[self.basis]


def _prepared(entries: Iterable[tuple[str, str]], drop_words: frozenset[str]) -> list[tuple[str, str, str, str, str]]:
    """(canonical, raw form, normalised, skeleton, vowelled key) for each distinct (entity, form)."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str, str, str]] = []
    for canonical, form in entries:
        n = normalise(form, drop_words)
        if not n or (canonical, n) in seen:
            continue
        seen.add((canonical, n))
        out.append((canonical, form, n, phonetic_key(n), vowelled_key(n)))
    return out


def _compare(x: tuple[str, str, str, str, str], y: tuple[str, str, str, str, str]) -> tuple[str, float] | None:
    """(basis, similarity) if the two forms could be mistaken for each other, else None."""
    _cx, _fx, nx, sx, vx = x
    _cy, _fy, ny, sy, vy = y
    same_script = script_of(nx) == script_of(ny)
    ratio = difflib.SequenceMatcher(None, nx, ny).ratio() if same_script else 0.0
    if nx == ny:
        return SAME_WRITTEN, 1.0
    if len(sx) >= MIN_SKELETON and sx == sy and (not same_script or ratio >= SAME_SOUND_FLOOR):
        strong = len(sx) in STRONG_SKELETON or (same_script and ratio >= STRONG_RATIO)
        return (SAME_SOUND if strong else SAME_SOUND_WEAK), ratio
    if len(sx) < MIN_SKELETON and len(sy) < MIN_SKELETON and vx and vx == vy:
        return SAME_SOUND_SHORT, ratio
    if len(sx) >= NEAR_MIN_SKELETON and len(sy) >= NEAR_MIN_SKELETON and same_script and _within_one_edit(sx, sy):
        return ONE_APART, ratio
    if same_script and ratio >= SPELLING_FLOOR:
        return SPELLING, ratio
    return None


def find_confusables(entries: Iterable[tuple[str, str]], drop_words: Iterable[str] = ()) -> list[Confusable]:
    """Every pair of DIFFERENT entities with a pair of forms that could be mistaken for each other, one row per pair of
    entities (the strongest evidence between them), strongest first.

    `entries`: (canonical name, one spoken or written form of it), any number of forms per entity."""
    rows = _prepared(entries, frozenset(w.lower() for w in drop_words) | GENERIC_WORDS)
    best: dict[tuple[str, str], Confusable] = {}
    for i, x in enumerate(rows):
        for y in rows[i + 1 :]:
            if x[0] == y[0]:
                continue
            hit = _compare(x, y)
            if hit is None:
                continue
            basis, ratio = hit
            (a, fa), (b, fb) = sorted([(x[0], x[1]), (y[0], y[1])])
            found = Confusable(a, b, fa, fb, basis, round(ratio, 3))
            old = best.get((a, b))
            if old is None or (_ORDER[found.basis], -found.similarity) < (_ORDER[old.basis], -old.similarity):
                best[(a, b)] = found
    return sorted(best.values(), key=lambda c: (_ORDER[c.basis], -c.similarity, c.a, c.b))


def shared_words(
    entries: Iterable[tuple[str, str]],
    drop_words: Iterable[str] = (),
    generic: frozenset[str] = GENERIC_WORDS,
) -> dict[str, list[str]]:
    """word -> the entities whose spoken forms contain it, for words that appear in two or more entities.

    Only whole words count; a word shorter than MIN_WORD_CHARS is a spoken letter or a particle and is ignored."""
    drop = frozenset(w.lower() for w in drop_words)
    by_word: dict[str, set[str]] = defaultdict(set)
    for canonical, form in entries:
        for word in normalise(form, drop).split():
            if len(word) >= MIN_WORD_CHARS and word not in generic:
                by_word[word].add(canonical)
    return {w: sorted(c) for w, c in sorted(by_word.items()) if len(c) >= 2}
