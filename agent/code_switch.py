"""KCD-065: code-switch detection within one utterance.

A Kolkata caller routinely mixes scripts mid-sentence ("CBC টেস্টের রেট
কত?" -- an English loanword inside a Bengali sentence, or the reverse).
LID (agent/lid.py) runs on the WHOLE utterance's audio, once, to pick an
ASR engine -- it has no notion of a switch happening partway through.
This module works on the ASR TRANSCRIPT text instead, purely for the
signal itself (logged and exported per Appendix F's mixture buckets),
never to re-route or re-segment the turn -- a switch point found here is
marked, never treated as an end of turn or a recognition failure (the
acceptance criterion is explicitly about what must NOT happen with it,
which is why there is no "split the utterance here" function in this
file: nothing downstream acts on a switch point except to log it).

Deterministic, script-range based -- same technique and same character
ranges reply_templates_i18n.py's _DEVANAGARI/_LATIN patterns and
speech_norm.py already use for a different purpose (choosing how to
verbalise a span). No LID model call, no LLM call: this is a per-turn
signal that has to be cheap enough to sit inside KCD-076's latency
budget alongside the other call-intelligence detectors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BENGALI = re.compile(r"[ঀ-৿]+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]+")
_LATIN = re.compile(r"[A-Za-z]+")

_SCRIPT_PATTERNS = (("bn", _BENGALI), ("hi", _DEVANAGARI), ("en", _LATIN))


@dataclass(frozen=True)
class ScriptSpan:
    script: str  # "bn" | "hi" | "en"
    start: int  # character offset into the original text
    end: int
    text: str


def script_spans(text: str) -> list[ScriptSpan]:
    """Every contiguous script run in `text`, in the order it appears.
    Whitespace/punctuation between runs is not itself a span -- only
    script-bearing characters are."""
    spans = []
    for script, pattern in _SCRIPT_PATTERNS:
        for m in pattern.finditer(text):
            spans.append(ScriptSpan(script, m.start(), m.end(), m.group(0)))
    spans.sort(key=lambda s: s.start)
    return spans


def switch_points(text: str) -> list[int]:
    """Character offsets where the script changes from the previous
    script-bearing span to a DIFFERENT one -- a code-switch point, per
    KCD-065. A single stray Latin digit/acronym inside an otherwise
    single-script sentence still counts (it is a real switch, even if
    brief); this module does not judge significance, only marks
    position -- filtering by length/significance is a caller concern."""
    spans = script_spans(text)
    points = []
    for prev, cur in zip(spans, spans[1:]):
        if cur.script != prev.script:
            points.append(cur.start)
    return points


def has_code_switch(text: str) -> bool:
    return bool(switch_points(text))


def mixture_bucket(text: str) -> str:
    """Which scripts appear at all, as a stable, sorted key ("bn+en",
    "bn+en+hi", ...) -- the per-mixture-bucket grouping Appendix F asks
    code-switch accuracy to be reported against, rather than only in
    aggregate. A single-script utterance's bucket is just that script."""
    scripts = {s.script for s in script_spans(text)}
    if not scripts:
        return "none"
    return "+".join(sorted(scripts))
