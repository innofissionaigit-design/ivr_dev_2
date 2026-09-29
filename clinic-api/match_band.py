"""Which catalogue rows the caller might have meant, and whether to pick one.

story title: Near matches are offered rather than guessed or refused
user story: As a caller naming something loosely, I want the close matches
    offered, so that I am not told my test does not exist when it does.
acceptance criteria: When several catalogue rows fall within the match band
    the agent offers up to three by name and asks which. Candidates are
    generated across every supported language and romanised spelling. The
    did-you-mean path covers the ambiguous case and not only total failure.

WHAT WAS WRONG, PRECISELY
-------------------------
Every lookup in main.py resolved to ONE row and threw the rest away, by one of
two mechanisms, and only the second of them is the one people expect:

  1. `.first()` on a substring query, and the equivalent `return` inside the
     alias loop. No scoring happens at all. "Blood Sugar" matches both
     "Blood Sugar Fasting" and "Blood Sugar PP"; row order decided, silently.
     Same for "সুগার", "ভিটামিন", "Vitamin".
  2. argmax above a floor, in _find_doctor and _find_department. Here there
     ARE scores, and the runner-up was discarded before anyone could ask
     whether it was close.

A margin check alone would have fixed only the second. Both funnel through
this module now, so a substring hit is a score like any other and the same
question -- is second place close? -- gets asked on every path.

THE THREE VERDICTS
------------------
  COMMIT      one row is clearly best. Answer it.
  AMBIGUOUS   several rows are plausible, or the best one is plausible but
              not clear. Offer up to three by name and ask which.
  NONE        nothing is close enough to be worth saying out loud.

AMBIGUOUS WITH A SINGLE CANDIDATE IS NOT A CONTRADICTION. It is the old
did-you-mean, and merging the two is the point of the story's third clause:
one row at 0.69 is a question ("did you mean X?"), not an answer, and it was
already treated that way. What is new is that TWO rows at 0.90 are also a
question rather than a coin flip.

THE FLOORS
----------
BAND_FLOOR is 0.50 to match difflib.get_close_matches(cutoff=0.5), which is
what generated the old suggestions. Deliberately unchanged so that nothing
that produced a suggestion before stops producing one now.

COMMIT_FLOOR is fast_path.COMMIT_FLOOR. It is STRICTER than the 0.60 that
_find_doctor and _find_department used to commit at, and that is the intended
direction: the band between 0.50 and 0.72 used to be a silent commit for
doctors and is now a question. The Doctor Nobody incident that set
FUZZY_SURNAME_FLOOR at 0.60 (a wrong doctor matched at 0.522) is exactly this
band, and offering the name for confirmation is a better answer to it than
either committing or refusing.

MARGIN is 0.08, and it is REASONED, NOT MEASURED -- no real call audio exists
to calibrate it against, the same admission FUZZY_SURNAME_FLOOR makes. It sits
above the 0.03 that semantic_cache.py measured between two DIFFERENT tests in
the same sentence frame, and far below the 0.315 that character overlap
separates the same distinction by. Tune it from the golden set (E9-S2) when
there is one.

WHY SCORING IS TIERED RATHER THAN PURE difflib
-----------------------------------------------
An exact alias must beat a containment, or "সেন" would be as ambiguous as it
is decisive: it EQUALS Dr Sen's alias and is CONTAINED IN Dr Sengupta's, and a
flat 1.0 for both would ask the caller to choose between a doctor they named
exactly and one they did not. Equality 1.0, containment 0.90, everything else
difflib -- so "সেন" clears the margin and commits, while "সুগার", contained in
two aliases and equal to neither, does not.

KNOWN GAP, NOT CLOSED HERE -- ROMANISED SPELLINGS
--------------------------------------------------
The criterion asks for candidates "across every supported language and
romanised spelling". This module scores whatever forms it is given, and
clinic-api hands it every form the catalogue holds -- so the coverage question
is a DATA question, not a code one.

Today: departments carry romanised aliases ("cardio", "heart", "gyne"); lab
tests and doctors carry Bengali script plus the English canonical name only.
Adding "thairoid" or "uric acid" to those rows means editing seed.py, and
seed() is destructive (drop_all then reload) with no rate history to survive
it, since E3-S5 versioning is absent. That is an operational decision rather
than a deploy, and it was deliberately left out of this change. The gap is
therefore: a caller whose romanised pronunciation lands below 0.50 against
both the Bengali alias and the English name is still told nothing matched.
"""
from __future__ import annotations

import difflib
import unicodedata

COMMIT = "commit"
AMBIGUOUS = "ambiguous"
NONE = "none"

# Worth offering out loud. = difflib.get_close_matches(cutoff=0.5), the value
# that generated the old did-you-mean list, kept so nothing regresses.
BAND_FLOOR = 0.50

# Worth answering without asking. = agent/fast_path.py COMMIT_FLOOR.
COMMIT_FLOOR = 0.72

# How far clear of second place "clearly best" has to be. Reasoned, not
# measured -- see the module docstring.
MARGIN = 0.08

# The criterion's cap. Three is also about as many names as a caller can hold
# in their head from one spoken sentence.
MAX_OFFERED = 3

EXACT_SCORE = 1.0
CONTAINS_SCORE = 0.90

# Below this length a containment test stops meaning anything -- a two-letter
# fragment is inside half the catalogue. Those fall through to difflib, which
# penalises the length mismatch instead.
_MIN_CONTAINS_LEN = 3


class Candidate:
    """One catalogue row that might be what the caller said."""

    __slots__ = ("key", "score", "form")

    def __init__(self, key, score: float, form: str):
        self.key = key          # opaque to this module; the caller's own row
        self.score = score
        self.form = form        # the spelling that actually matched

    def __repr__(self):  # pragma: no cover - diagnostics only
        return f"Candidate({self.form!r}, {self.score:.2f})"


# WHAT CONTAINMENT MEANS IN BENGALI
#
# `q in f or f in q`, guarded only by the QUERY's length, let a two-letter
# form sit "inside" any long sentence. Live, against the seeded catalogue:
#   "বাচ্চাদের ডাক্তার" (children's doctor) -> Dr A. Dey, because দে is inside
#       বাচ্চাদের -- a Dermatologist booked for a child, silently;
#   "চোখের ডাক্তার দেখাতে চাই" -> Dr A. Dey, because দে starts দেখাতে (to show);
#   and "করতে" / "করে" (to do) contain Dr A. Kar's কর, "বসেন" (sits) Dr Sen's সেন.
#
# Bengali inflects with a SUFFIX and never a prefix, so a form counts as
# contained only where it STARTS a word and is followed by nothing, or by a
# case ending, before the word ends. "সেনের", "কার্ডিওলজিতে", "সুগার ফাস্টিং"
# still match; "বসেন", "বাচ্চাদের", "appointment" (ent) and "smear" (ear) do not.
#
# A two-letter form takes only the endings a NAME takes (possessive, dative,
# honorific). With the full set, the locative "ে" and infinitive "তে" would
# read the verbs "করে"/"করতে" as Dr Kar, and "দেন" would be close behind.
_CASE_ENDINGS = frozenset(unicodedata.normalize("NFC", e) for e in (
    "", "র", "ের", "এর", "ে", "কে", "তে", "য়", "টা", "টি", "ই", "ও", "রা", "দের",
    "বাবু", "বাবুর", "বাবুকে", "s", "'s", "er", "e", "ji",
))
# Not even the possessive "র": "দে" + "র" IS "দের", the plural-possessive
# suffix, and the ASR splits words -- "বাচ্চা দের ডাক্তার" would have booked a
# dermatologist for a child. What is left cannot be read as anything else.
_NAME_ENDINGS = frozenset(unicodedata.normalize("NFC", e) for e in (
    "", "কে", "বাবু", "বাবুর", "বাবুকে", "'s", "ji",
))
_SHORT_FORM_LEN = 2
# A form this short (সেন, বসু, রায়, sen, roy) is compared fuzzily only against
# a single WORD of the query that starts with the same letter and is within one
# letter of its length -- the shape of an ASR slip ("সেণ" for "সেন", 0.67, a
# fair question). Against a whole sentence, or a longer word that merely ends
# in it, it matched verbs: "বসেন" (sits) vs "সেন" is 0.86 and COMMITTED Dr Sen.
# And it only ever reaches the band, never the commit floor: a three-letter
# near-miss is a question, not an answer.
_FUZZY_SHORT_LEN = 3
_SHORT_SLIP_MIN = 0.66
_BN_RANGE = ("ঀ", "৿")

# Words every test name carries. In the fuzzy fallback they made two
# unrelated tests look alike: "হার্ট টেস্ট" vs "লিভার টেস্ট" scored 0.73 --
# over the commit floor -- on the strength of "টেস্ট" alone, and a caller
# asking a heart test's price was told the Liver Function Test's. Ignored in
# difflib only; an exact or contained match still sees the whole phrase.
_GENERIC_TERMS = frozenset(unicodedata.normalize("NFC", w) for w in (
    "টেস্ট", "টেস্টের", "টেস্টটা", "test", "tests", "পরীক্ষা", "পরীক্ষার",
))


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or _BN_RANGE[0] <= ch <= _BN_RANGE[1]


def _contained(short: str, long: str) -> bool:
    """`short` inside `long` as a word, or a word plus one allowed ending."""
    endings = _NAME_ENDINGS if len(short) <= _SHORT_FORM_LEN else _CASE_ENDINGS
    start = long.find(short)
    while start != -1:
        if start == 0 or not _is_word_char(long[start - 1]):
            end = start + len(short)
            tail_end = end
            while tail_end < len(long) and not long[tail_end].isspace():
                tail_end += 1
            if long[end:tail_end] in endings:
                return True
        start = long.find(short, start + 1)
    return False


def _without_generic(text: str, ignore: frozenset = frozenset()) -> str:
    drop = _GENERIC_TERMS | ignore
    kept = [w for w in text.split() if w not in drop]
    return " ".join(kept) if kept else text


def score(query: str, form: str, ignore: frozenset = frozenset()) -> float:
    """How well one spelling of one row matches what the caller said.

    `ignore`: extra filler words for this lookup's fuzzy fallback on top of
    _GENERIC_TERMS -- "ডাক্তার" for departments, where "চোখের ডাক্তার" and
    "ত্বকের ডাক্তার" otherwise look alike on the strength of it. Not for
    doctors: there the word is part of how a name is said."""
    q = unicodedata.normalize("NFC", (query or "").strip().lower())
    f = unicodedata.normalize("NFC", (form or "").strip().lower())
    if not q or not f:
        return 0.0
    if q == f:
        return EXACT_SCORE
    short, long = (q, f) if len(q) <= len(f) else (f, q)
    if _contained(short, long):
        return CONTAINS_SCORE
    # A form of two letters carries too little to be matched fuzzily at all:
    # "দে" vs "দেখাতে" is 0.50, inside the band, and would OFFER a
    # dermatologist to a caller who said "to show". Exact or contained only.
    if len(f) <= _SHORT_FORM_LEN:
        return 0.0
    if len(f) <= _FUZZY_SHORT_LEN and " " not in f:
        best = 0.0
        for word in q.split():
            if word[:1] == f[:1] and abs(len(word) - len(f)) <= 1:
                best = max(best, difflib.SequenceMatcher(None, word, f).ratio())
        # One letter off ("সেণ" for "সেন") is 0.67; two ("বসেন" vs "বসু") is
        # 0.57 and is a different word, not a slip.
        if best < _SHORT_SLIP_MIN:
            return 0.0
        return min(best, COMMIT_FLOOR - 0.01)
    return difflib.SequenceMatcher(None, _without_generic(q, ignore),
                                   _without_generic(f, ignore)).ratio()


def rank(query: str, rows, ignore: frozenset = frozenset()) -> list[Candidate]:
    """-> every row that scored anything, best first, one entry per row.

    `rows` is an iterable of (key, forms): the key is whatever the caller
    wants back -- an ORM object, an id, a dict -- and forms is every spelling
    of it, in every script the catalogue holds. Scoring the English name and
    the Bengali aliases through the same function is what lets a caller be
    understood in either, and is why this takes forms rather than a name.

    One entry per row, keeping its best-matching form: a test with four
    aliases must not out-rank one with a single alias by sheer count, and the
    form that won is kept because it is useful in a log line.
    """
    ranked: list[Candidate] = []
    for key, forms in rows:
        # Keyed by POSITION, not by the key object: a row's identity is the
        # caller's business, and hashing or id()-ing an ORM instance here
        # would quietly assume something about it that this module does not
        # need to know.
        winner: Candidate | None = None
        for form in forms or ():
            s = score(query, form, ignore)
            if winner is None or s > winner.score:
                winner = Candidate(key, s, form)
        if winner is not None:
            ranked.append(winner)

    # Stable: equal scores keep catalogue order, so an offer reads the same
    # way twice. Two rows tying is the case this whole module exists for, and
    # a nondeterministic order there would make the offer itself flicker.
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked


def decide(ranked: list[Candidate]) -> tuple[str, list[Candidate]]:
    """-> (verdict, the candidates that verdict is about).

    COMMIT returns exactly one. AMBIGUOUS returns up to MAX_OFFERED. NONE
    returns an empty list -- deliberately, so a caller cannot accidentally
    speak a candidate that did not clear the band.
    """
    in_band = [c for c in ranked if c.score >= BAND_FLOOR]
    if not in_band:
        return NONE, []

    best = in_band[0]
    second = in_band[1].score if len(in_band) > 1 else 0.0

    if best.score >= COMMIT_FLOOR and (best.score - second) >= MARGIN:
        return COMMIT, [best]

    return AMBIGUOUS, in_band[:MAX_OFFERED]
