"""Word-gap-tolerant matching for a fixed multi-word phrase against free text.

Found on a real call: "বুকে ব্যথা" ("chest pain", the literal two words adjacent) is what agent/emergency.py and
clinic-api's department-routing keywords matched on, as a plain substring/adjacent-regex. A caller who says
"বুকে খুব ব্যথা করছে" ("my chest REALLY hurts") -- an entirely ordinary way to say it, with an intensifier inserted
between the two words -- matched NEITHER check: not an emergency, not routed to Cardiology. Two different
safety/quality gaps, one root cause.

The fix: every word of a phrase must still appear, in order, but up to `max_gap_words` OTHER words are allowed
between each consecutive pair. A one-word phrase is unaffected (still a plain substring check).

This file is byte-identical in agent/ and clinic-api/ (clinic-api is a separate service that never imports agent/;
same convention as agent/gazetteer.py's own copy); tests/test_phrase_proximity.py fails if the two drift.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

DEFAULT_MAX_GAP_WORDS = 2


@lru_cache(maxsize=4096)
def _compiled(phrase: str, max_gap_words: int) -> re.Pattern[str]:
    words = [re.escape(w) for w in phrase.split() if w]
    if not words:
        return re.compile(r"(?!x)x")  # matches nothing: an empty phrase is never "found"
    if len(words) == 1:
        return re.compile(words[0])
    gap = rf"(?:\s+\S+){{0,{max_gap_words}}}\s+"
    return re.compile(words[0] + "".join(gap + w for w in words[1:]))


def phrase_matches(phrase: str, text: str, max_gap_words: int = DEFAULT_MAX_GAP_WORDS) -> bool:
    """True if every word of `phrase`, in order, appears in `text` with at most `max_gap_words` other words between
    each consecutive pair. Word order in `phrase` must be preserved; words are matched as literal text, not stems --
    this is proximity tolerance, not a fuzzy or phonetic match (agent/gazetteer.py already owns that, for names)."""
    if not phrase or not text:
        return False
    return _compiled(phrase, max_gap_words).search(text) is not None


def any_phrase_matches(phrases: Iterable[str], text: str, max_gap_words: int = DEFAULT_MAX_GAP_WORDS) -> bool:
    return any(phrase_matches(p, text, max_gap_words) for p in phrases)
