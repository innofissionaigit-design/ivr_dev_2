"""What this caller has already been told, so a repeat cannot contradict it.

Ported from `merged_code/agent/answer_ledger.py` (story: "The same question gets
the same answer within one call" -- a repeat must produce an identical factual
answer unless the data changed, in which case the change is STATED).

WHAT THIS IS NOT: A REPLY CACHE.

Nothing here caches a reply or serves a remembered one. Every repeat still runs
its own clinic lookup and renders its own sentence from that response -- which is
also this repository's existing rule: `agent/semantic_cache.py` caches the
INTENT and never the answer, precisely so a price that changed between two turns
still reaches the caller live.

This module only WATCHES. It records what the clinic said, and when the same
question comes round again it compares. The answer the caller hears is always
the fresh one; the only thing the record can add is a sentence saying the figure
moved.

That is the half the caching rule never addressed. "A repeat is never stale"
says nothing about the caller who wrote down 450 and is now told 500 with no
acknowledgement that anything happened. Silence there is the worse failure of
the two, because from the caller's seat a live update and a system contradicting
itself sound exactly alike -- which is the whole of "so that I know which one to
believe".

WHY THE CLINIC CANNOT SIMPLY BE ASKED:

`clinic-api` stores one current rate per test with no `valid_from`/`valid_to`,
so a price change overwrites history and no field anywhere reports one.
Observing across turns is not a shortcut here; it is the only mechanism
available.

!!! NOT WIRED INTO THE ORCHESTRATOR YET -- READ THIS BEFORE USING IT !!!

This module is complete and unit-tested, but **nothing calls it**. It is here
because the logic is non-trivial and correct, not because the feature is live.

The reason it is not wired is specific. The place to call `check()` is where a
ledgered intent's `result` dict exists, which is `_answer_enquiry_intent()` in
`main.py` / `main_pcm.py`. That function takes `(intent, slots, lang)` and
returns a string -- it has no session, so it cannot reach the per-call ledger,
and the `result` it would need to record never leaves it. Giving it either the
session or a `(reply, result)` return type changes a signature used at **12 call
sites across the two orchestrators and faked in 9 test files**, six of them
pre-existing. That is a refactor of a central function with a wide blast radius,
and it deserves its own reviewed change rather than riding along inside a merge
whose whole point was not to break the working flow.

Until that happens, a caller who is told 450 and then 500 in one call still
hears no acknowledgement. The gap is unchanged by this port; what this port adds
is the tested mechanism to close it.

ADAPTED FOR THIS REPOSITORY:

The original ledgered three intents. `doctors_by_department` does not exist in
this repository's intent set (`agent/intent_schema.py`), so this port covers the
two that do: `test_rate` and `doctor_availability`. The result field names are
unchanged -- they are the same in both codebases, and are pinned here against
`agent/api_models.py`'s `TestAnswer` and `DoctorAnswer`.

One trilingual adjustment: for a not-found answer the original compared the
spoken Bengali suggestions (`did_you_mean_bn`). This repository speaks whichever
language the caller is using, so the comparison uses the language-neutral
`did_you_mean` list instead -- a genuine change in what the catalogue suggests
is the same fact in every language, and comparing the localised list would
report a "change" merely because the reply language changed mid-call.
"""

from __future__ import annotations

import re

FIRST = "first"
SAME = "same"
CHANGED = "changed"

# Only intents whose answer is a FACT the caller could write down. A greeting or
# an FAQ paragraph has nothing to contradict.
LEDGERED_INTENTS = ("test_rate", "doctor_availability")

# Which response field carries the catalogue's own name for the thing. The
# caller's name for the same thing arrives in the slot of the same name, which
# is why one map serves both -- they are read separately below, because only one
# of the two is authoritative.
_CANONICAL_FIELD = {
    "test_rate": "test_name",
    "doctor_availability": "doctor_name",
}

# A call has tens of turns, not thousands, so this is a bound against a
# pathological session rather than a working eviction policy.
MAX_ENTRIES = 64

_RE_WS = re.compile(r"\s+")
_RE_STRIP = re.compile("[।?!,.'\"‌‍]+")


def _fold(value) -> str:
    """Loose normalisation, for key comparison only and never for anything spoken.

    Deliberately a local copy of the same few lines in
    `semantic_cache.normalize_text` rather than an import: that one is tuned for
    embedding lookup and is free to change for reasons unrelated to this, and
    sharing it would couple a cache-tuning decision to whether two answers are
    judged to contradict each other.
    """
    return _RE_WS.sub(" ", _RE_STRIP.sub(" ", str(value).strip().lower())).strip()


def _text(value):
    """Compare a clinic value as the STRING it arrived as.

    Never `float()`. `tools_client` parses numbers with `parse_float=str` exactly
    so a rate keeps the digits the clinic sent, and `float("450.00") ==
    float("450.0")` would erase a real change at precisely the boundary the
    number-fidelity rule exists to protect.

    `bool` passes through rather than being stringified, so `available` stays a
    two-valued thing instead of becoming "True"/"False" text.
    """
    if value is None or isinstance(value, bool):
        return value
    return str(value)


def identity_keys(intent: str, slots: dict, result: dict) -> list[tuple]:
    """Every key under which this answer should be findable, best first.

    The "id" key is the catalogue's identity and is authoritative. The "said"
    key is the caller's own words, and exists so that a not-found answer --
    which carries no catalogue identity at all, only the query echoed back --
    can still be recognised when the same thing is asked again.

    The date is part of the key for the dated intent: "is Dr Sen in today" asked
    at 23:59 and again at 00:01 are different questions, and a date-less ledger
    would report a contradiction between two perfectly correct answers.

    One known limit, stated rather than papered over: a not-found availability
    response carries no date, so it cannot sit on the same axis as a dated
    answer, and a doctor who appears in the catalogue mid-call is not compared
    against the earlier "no such doctor". `test_rate` carries no date and does
    not have this gap.
    """
    field = _CANONICAL_FIELD.get(intent)
    if field is None:
        return []

    date = _text(result.get("date")) if intent != "test_rate" else None

    keys: list[tuple] = []
    canonical = result.get(field)
    if canonical:
        keys.append((intent, "id", _fold(canonical), date))
    for spoken in (slots.get(field), result.get("query")):
        if spoken:
            key = (intent, "said", _fold(spoken), date)
            if key not in keys:
                keys.append(key)
    return keys


def facts(intent: str, result: dict) -> tuple:
    """The values this intent's template actually speaks, and nothing else.

    A field the caller never hears must not be able to trigger a change
    announcement, so each tuple mirrors what its template in
    `agent/reply_templates.py` puts into the sentence.

    The resolved identity leads every tuple. It is redundant whenever the entry
    was found by its "id" key, and load-bearing whenever it was found by the
    caller's words: the same words resolving to a different row is a changed
    answer even in the rare case where both rows quote the same price.
    """
    field = _CANONICAL_FIELD.get(intent)
    if field is None:
        raise ValueError(f"no facts defined for intent {intent!r}")

    canonical = result.get(field)
    ident = _fold(canonical) if canonical else None
    found = bool(result.get("found"))

    if intent == "test_rate":
        if not found:
            # The suggestions are spoken, so a change in them changes the
            # answer. Sorted, because the clinic promises no order and a
            # reshuffle is not something to announce. The language-neutral list
            # is used so switching reply language mid-call is not a "change".
            suggestions = result.get("did_you_mean") or result.get("did_you_mean_bn") or ()
            return (ident, False, tuple(sorted(str(s) for s in suggestions)))
        return (
            ident,
            True,
            _text(result.get("rate_inr")),
            _text(result.get("sample_type")),
            _text(result.get("report_time_hours")),
        )

    if not found:
        return (ident, False)
    return (
        ident,
        True,
        bool(result.get("available")),
        _text(result.get("chamber_hours")),
        _text(result.get("next_available_date")),
    )


class AnswerLedger:
    """One per call. Dies with the call.

    Not thread-safe, and not by oversight: every touch happens on the asyncio
    event loop inside the session's dispatch lock, and a lock here would buy
    nothing but the impression that this is shared with something.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES):
        self._max = max_entries
        self._entries: dict[int, dict] = {}  # slot id -> {facts, asked}
        self._index: dict[tuple, int] = {}  # any identity key -> slot id
        self._next_id = 0
        self.stats = {"repeats": 0, SAME: 0, CHANGED: 0}

    def check(self, intent: str, slots: dict, result: dict) -> tuple[str, tuple | None]:
        """-> (verdict, the facts said last time), and record this answer.

        Recording is a side effect rather than a second call on purpose: a check
        that CAN be performed without recording is a check somebody eventually
        performs without recording, and the entry is then missing for the turn
        after.
        """
        if intent not in LEDGERED_INTENTS or not isinstance(result, dict):
            return FIRST, None

        # An ambiguous response is a QUESTION, not an answer, and recording it
        # would be worse than not recording it: the entry's facts would be
        # (None, False), and the disambiguated answer one turn later would then
        # be reported as a change and prefixed with "that has changed since I
        # told you" -- announcing a move that never happened, on the turn where
        # the caller is least sure of themselves.
        if result.get("ambiguous"):
            return FIRST, None

        keys = identity_keys(intent, slots, result)
        if not keys:
            # Nothing nameable was resolved, so there is nothing this answer
            # could be consistent WITH. Silence beats a guessed identity.
            return FIRST, None

        current = facts(intent, result)
        slot_id = next((self._index[key] for key in keys if key in self._index), None)

        if slot_id is None:
            slot_id = self._next_id
            self._next_id += 1
            self._entries[slot_id] = {"facts": current, "asked": 1}
            verdict, previous = FIRST, None
        else:
            entry = self._entries[slot_id]
            previous = entry["facts"]
            verdict = SAME if previous == current else CHANGED
            # Overwritten, so a third ask after a change says SAME rather than
            # announcing the same move twice. The caller is told once.
            entry["facts"] = current
            entry["asked"] += 1
            self.stats["repeats"] += 1
            self.stats[verdict] += 1

        for key in keys:
            self._index[key] = slot_id
        self._evict()
        return verdict, previous

    def _evict(self) -> None:
        while len(self._entries) > self._max:
            oldest = min(self._entries)
            del self._entries[oldest]
            self._index = {k: v for k, v in self._index.items() if v != oldest}

    def snapshot(self) -> dict:
        """Same shape as the other `snapshot()` methods in this package.

        `changed` is the alertable number. On a catalogue nobody is editing it
        should be zero; a rising count is either real churn in the clinic's data
        or entity resolution landing on a different row for the same words, and
        both are things somebody should be told about rather than things a caller
        should discover.
        """
        return {**self.stats, "entries": len(self._entries)}
