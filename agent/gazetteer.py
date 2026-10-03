"""A phonetic-fold gazetteer with an index (KCD-096).

WHAT WAS WRONG. Finding "who might the caller have meant" was a linear scan of every doctor and test row, once
with a consonant-skeleton comparison and once with `difflib.get_close_matches` over every name and alias, on every
lookup. Two things follow from that: the cost grows with the catalogue (a scan of a few thousand rows is felt on a
phone call), and the comparison was character-shaped, so a mispronunciation that keeps the consonants but drifts
the spelling ("Mukharji" for "Mukherjee") was found only when the skeleton matched EXACTLY and was at least three
classes long. "Sen", "Das", "Roy", "Kar" fold to one or two classes, so short names -- the commonest Bengali
surnames -- were never matched by sound at all.

WHAT THIS DOES. Every spoken form (a name, a surname, each Bengali and Hindi alias) is folded to phonemes ONCE, at
build time, and put in four inverted indexes. A query is folded the same way and looks up candidates, never scans:

    exact        the normalised written form itself                        (dict)
    sound        the same consonant skeleton (agent/phonetic_match.py)     (dict)
    sound, near  a skeleton one edit away: a dropped or added consonant    (symmetric-delete index)
    sound, short a one- or two-consonant name matched on consonants AND
                 coarse vowel quality ("Sen" = "Sain", not "Son")          (dict)
    spelling     shared character pairs, then difflib on the few sharing   (bigram index, top K)

Only the shortlist is scored, so the work per lookup follows the number of forms that SHARE something with the
query, not the size of the catalogue; `stats` counts how many were scored so that can be checked, not assumed.

WHAT IT IS NOT. It never resolves anything. The return value is a ranked list of "the caller might have meant"
for the agent to READ BACK; a sound-alike is never treated as the entity (CLAUDE.md, "Doctor Nobody"; the
clinic API resolves only on a written-form match). A fold throws information away, so every tier past `exact`
still has to clear a second, independent gate: a same-script character-similarity floor, or (across scripts,
where characters share nothing) a skeleton of at least three classes. A nonsense name that lands on nothing
returns an empty list.

REASONED, NOT MEASURED: every floor below was set from the synthetic mispronunciation set in
tools/gazetteer_eval.py, not from real callers. Recalibrate against real call transcripts before trusting them.

This file is byte-identical in agent/ and clinic-api/ (clinic-api is a separate service that never imports
agent/); tests/test_gazetteer.py fails if the two copies drift.
"""

from __future__ import annotations

import difflib
import heapq
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

try:  # clinic-api ships its own copy of the fold
    from phonetic_match import phonetic_key
except ImportError:  # the agent side
    from agent.phonetic_match import phonetic_key

# Same-script character similarity a sound-alike (skeleton-equal) form must ALSO reach. Loose on purpose: the
# skeleton is the evidence, this only stops two unrelated words that fold to the same three classes.
SOUND_MIN_RATIO = 0.40
# A skeleton one edit away is weaker evidence, so it needs a stronger second gate. Sits just above the 0.44 that
# the documented "doctor nobody" vs "Roy" false positive scored.
SOUND_NEAR_MIN_RATIO = 0.45
# The plain spelling tier -- difflib's own default cutoff for get_close_matches, kept so this is not a stricter
# or looser rule than the scan it replaces.
SPELLING_MIN_RATIO = 0.50
# A skeleton shorter than this is not evidence on its own (agent/phonetic_match.py's own rule), except through
# the vowel-aware key below.
MIN_SKELETON = 3
# One-edit skeleton matches only for skeletons at least this long: one edit on a three-class key is a third of it.
NEAR_MIN_SKELETON = 4
# How many forms sharing character pairs are scored with difflib. The scan scored all of them.
SHORTLIST = 40
# A character pair that occurs in more than this share of all forms carries no information; it is not looked up.
STOP_GRAM_SHARE = 0.03
STOP_GRAM_FLOOR = 60
# When a sound-based candidate exists, a mere spelling neighbour is listed only if it is this close. Without a
# sound-based candidate the spelling tier stands alone at SPELLING_MIN_RATIO. This is what keeps "Mukharji" from
# being read back as "Mukherjee, Kar or Halder": three names is a question the caller cannot answer.
SPELLING_WITH_SOUND_MIN_RATIO = 0.70
# A form this short (a test code such as ESR or TSH) matches almost any four-letter word at ratio 0.57, so the
# spelling tier needs more than the general floor when either side is short.
SHORT_FORM_CHARS = 5
SPELLING_SHORT_MIN_RATIO = 0.65
# Past the first suggestion, a further one is listed only if it is a genuine tie: the same tier of evidence and
# a similarity within this margin of the first. On the synthetic set nearly every right answer is the first
# suggestion and nearly every wrong one is a second or third; a list of three names is a question the caller
# cannot answer. Two doctors who really do share a skeleton still come back together.
TIE_MARGIN = 0.08

_TIERS = {"exact": 0, "sound": 1, "sound_short": 2, "sound_near": 3, "spelling": 4}

_FLAPS = (
    (chr(0x09A1) + chr(0x09BC), "র"),
    (chr(0x09A2) + chr(0x09BC), "র"),  # Bengali DDA/DDHA + nukta -> RA
    (chr(0x0921) + chr(0x093C), "र"),
    (chr(0x0922) + chr(0x093C), "र"),
)  # Devanagari DDA/DDHA + nukta -> RA
_BN_VOWELS = {
    **dict.fromkeys("াঅআ", "A"),
    **dict.fromkeys("িীেৈইঈএঐ", "I"),
    **dict.fromkeys("ুূোৌউঊওঔ", "U"),
}
_HI_VOWELS = {
    **dict.fromkeys("ाअआ", "A"),
    **dict.fromkeys("िीेैइईएऐ", "I"),
    **dict.fromkeys("ुूोौउऊओऔ", "U"),
}
_LATIN_VOWELS = {"a": "A", "e": "I", "i": "I", "y": "I", "o": "U", "u": "U"}
_BENGALI_RANGE = re.compile(r"[ঀ-৿]")
_DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")


def normalise(text: str, drop_words: frozenset[str] = frozenset()) -> str:
    """Comparison form: NFC, flap letters folded (they decompose under NFC, see phonetic_match.py), the Devanagari
    nukta dropped (Hindi ASR writes a word both ways), lower case, punctuation removed, title words removed."""
    t = unicodedata.normalize("NFC", text or "")
    for src, dst in _FLAPS:
        t = t.replace(src, dst)
    t = t.replace("़", "").lower().replace("-", " ")
    kept = "".join(
        ch if (ch.isalnum() or unicodedata.category(ch) in ("Mn", "Mc") or ch.isspace()) else " " for ch in t
    )
    words = [w for w in kept.split() if w not in drop_words]
    return " ".join(words)


def script_of(norm: str) -> str:
    if _DEVANAGARI_RANGE.search(norm):
        return "hi"
    if _BENGALI_RANGE.search(norm):
        return "bn"
    return "latin"


@lru_cache(maxsize=4096)
def _consonant(ch: str) -> str:
    return phonetic_key(ch)


def vowelled_key(norm: str) -> str:
    """Consonant classes interleaved with three coarse vowel qualities (open A, front I, back U), adjacent
    repeats collapsed. The evidence for a name too short for a consonant skeleton to identify: "Sen" -> 2I5."""
    out: list[str] = []
    for ch in norm:
        if ch.isspace():
            continue
        cls = _BN_VOWELS.get(ch) or _HI_VOWELS.get(ch) or _LATIN_VOWELS.get(ch) or _consonant(ch)
        if not cls:
            continue
        if cls in "AIU" and out and out[-1] in "AIU":
            out[-1] = cls  # a run of vowels is one syllable nucleus: the glide wins (ai, ay, ei -> I)
        elif not out or out[-1] != cls:
            out.append(cls)
    return "".join(out)


def _within_one_edit(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long_ = (a, b) if la < lb else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1 :]


def _deletes(key: str) -> set[str]:
    return {key[:i] + key[i + 1 :] for i in range(len(key))}


def _bigrams(norm: str) -> set[str]:
    padded = f"^{norm}$"
    return {padded[i : i + 2] for i in range(len(padded) - 1)}


@dataclass(frozen=True)
class Suggestion:
    canonical: str  # the entity the form belongs to (a test or doctor name)
    form: str  # the spoken form that matched
    score: float  # 0..1, for ordering and the eval only; never a resolution threshold
    basis: str  # exact | sound | sound_short | sound_near | spelling


class Gazetteer:
    def __init__(self, entries: Iterable[tuple[str, str]], drop_words: Iterable[str] = ()):
        """`entries`: (canonical name, one spoken/written form of it), any number of forms per canonical name."""
        self._drop = frozenset(w.lower() for w in drop_words)
        self._canon: list[str] = []
        self._raw: list[str] = []
        self._norm: list[str] = []
        self._skel: list[str] = []
        self._vkey: list[str] = []
        self._script: list[str] = []
        self._by_norm: dict[str, list[int]] = defaultdict(list)
        self._by_skel: dict[str, list[int]] = defaultdict(list)
        self._by_del: dict[str, list[int]] = defaultdict(list)
        self._by_vkey: dict[str, list[int]] = defaultdict(list)
        self._by_gram: dict[str, list[int]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for canonical, form in entries:
            n = normalise(form, self._drop)
            if not n or (canonical, n) in seen:
                continue
            seen.add((canonical, n))
            i = len(self._canon)
            skel = phonetic_key(n)
            self._canon.append(canonical)
            self._raw.append(form)
            self._norm.append(n)
            self._skel.append(skel)
            self._vkey.append(vowelled_key(n))
            self._script.append(script_of(n))
            self._by_norm[n].append(i)
            if skel:
                self._by_skel[skel].append(i)
                if len(skel) >= NEAR_MIN_SKELETON:
                    for d in _deletes(skel) | {skel}:
                        self._by_del[d].append(i)
            if self._vkey[i]:
                self._by_vkey[self._vkey[i]].append(i)
            for g in _bigrams(n):
                self._by_gram[g].append(i)
        self.stats = {"queries": 0, "scored": 0, "last_scored": 0}

    def __len__(self) -> int:
        return len(self._canon)

    # ------------------------------------------------------------------ lookup

    def suggest(self, query: str, limit: int = 3) -> list[Suggestion]:
        """Ranked, de-duplicated by entity. Empty when nothing clears its gate."""
        found = self._collect(query)
        return self._rank(found, limit)

    def neighbours(self, query: str) -> list[tuple[str, str, float]]:
        """Every entity with a form that cleared a gate, WITHOUT the tie truncation `suggest` applies:
        (canonical, form, same-script character similarity; 0.0 across scripts). For a caller that has its own
        rule for what counts as "in the running" and wants to apply it to a shortlist rather than to every row."""
        found = self._collect(query)
        best: dict[str, tuple[str, float]] = {}
        for i, (_tier, ratio) in found.items():
            if self._canon[i] not in best or ratio > best[self._canon[i]][1]:
                best[self._canon[i]] = (self._raw[i], ratio)
        return [(c, f, r) for c, (f, r) in best.items()]

    def _collect(self, query: str) -> dict[int, tuple[int, float]]:
        q = normalise(query, self._drop)
        scored = 0
        found: dict[int, tuple[int, float]] = {}  # item -> (tier, ratio)
        if q:
            qskel, qv, qscript = phonetic_key(q), vowelled_key(q), script_of(q)
            for i in self._by_norm.get(q, ()):
                found[i] = (_TIERS["exact"], 1.0)
            if len(qskel) >= MIN_SKELETON:
                for i in self._by_skel.get(qskel, ()):
                    scored += self._offer(found, i, q, qscript, "sound", SOUND_MIN_RATIO)
            elif qv:
                for i in self._by_vkey.get(qv, ()):
                    if len(self._skel[i]) < MIN_SKELETON:
                        scored += self._offer(found, i, q, qscript, "sound_short", SOUND_MIN_RATIO)
            if len(qskel) >= NEAR_MIN_SKELETON:
                near: set[int] = set()
                for d in _deletes(qskel) | {qskel}:
                    near.update(self._by_del.get(d, ()))
                for i in near:
                    if len(self._skel[i]) >= NEAR_MIN_SKELETON and _within_one_edit(qskel, self._skel[i]):
                        scored += self._offer(found, i, q, qscript, "sound_near", SOUND_NEAR_MIN_RATIO)
            scored += self._spelling(found, q, qscript)
        self.stats["queries"] += 1
        self.stats["scored"] += scored
        self.stats["last_scored"] = scored
        return found

    def _offer(self, found, i, q, qscript, basis, floor) -> int:
        """Score one candidate; keep it if it clears the second gate. Returns 1 (a candidate was scored)."""
        same_script = self._script[i] == qscript
        ratio = difflib.SequenceMatcher(None, q, self._norm[i]).ratio() if same_script else 0.0
        if same_script and ratio < floor:
            return 1
        if basis == "sound_near" and not same_script:
            return 1  # a near skeleton is weak evidence; with no shared characters there is nothing to back it
        tier = _TIERS[basis]
        if i not in found or (tier, -ratio) < (found[i][0], -found[i][1]):
            found[i] = (tier, ratio)
        return 1

    def _spelling(self, found, q: str, qscript: str) -> int:
        n_items = max(1, len(self._canon))
        stop = max(STOP_GRAM_FLOOR, int(n_items * STOP_GRAM_SHARE))
        votes: dict[int, int] = defaultdict(int)
        for g in _bigrams(q):
            posting = self._by_gram.get(g, ())
            if len(posting) > stop:
                continue
            for i in posting:
                votes[i] += 1
        scored = 0
        for i, _v in heapq.nlargest(SHORTLIST, votes.items(), key=lambda kv: (kv[1], -kv[0])):
            if self._script[i] != qscript:
                continue
            scored += 1
            ratio = difflib.SequenceMatcher(None, q, self._norm[i]).ratio()
            floor = (
                SPELLING_SHORT_MIN_RATIO if min(len(q), len(self._norm[i])) < SHORT_FORM_CHARS else SPELLING_MIN_RATIO
            )
            if ratio >= floor and (i not in found or (_TIERS["spelling"], -ratio) < (found[i][0], -found[i][1])):
                found[i] = (_TIERS["spelling"], ratio)
        return scored

    def _rank(self, found: dict[int, tuple[int, float]], limit: int) -> list[Suggestion]:
        basis = {v: k for k, v in _TIERS.items()}
        out: list[Suggestion] = []
        seen: set[str] = set()
        has_sound = any(t < _TIERS["spelling"] for t, _r in found.values())
        top_tier, top_ratio = -1, 0.0
        for i, (tier, ratio) in sorted(found.items(), key=lambda kv: (kv[1][0], -kv[1][1], kv[0])):
            if self._canon[i] in seen:
                continue
            if tier == _TIERS["spelling"] and has_sound and ratio < SPELLING_WITH_SOUND_MIN_RATIO:
                continue
            if out and not (tier == top_tier and top_ratio - ratio <= TIE_MARGIN):
                continue
            if not out:
                top_tier, top_ratio = tier, ratio
            seen.add(self._canon[i])
            out.append(
                Suggestion(
                    self._canon[i],
                    self._raw[i],
                    round(1.0 - tier * 0.1 - (1.0 - ratio) * 0.05, 3),
                    basis[tier],
                )
            )
            if len(out) >= limit:
                break
        return out

    def shortlist(self, query: str) -> list[str]:
        """Just the canonical names worth scoring further, for a caller that has its own commit rule (the fast
        path's character-ratio floor). Never a decision."""
        return [s.canonical for s in self.suggest(query, limit=SHORTLIST)]
