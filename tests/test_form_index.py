"""KCD-096: agent/form_index.py -- exact pruning and the bigram shortlist that keep fast-path matching fast as the
catalogue grows. Pure and offline.

    python -m pytest tests/test_form_index.py -v

The two mechanisms are tested separately because one is exact and one is not: pruning must never change a decision
at or above the commit floor (a property test, not an example); the shortlist may only ever turn "answer" into
"abstain", never into a different answer.
"""

import difflib
import os
import random
import sys
import time
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.append(_p)

import gazetteer_eval as ev  # noqa: E402

from agent.form_index import SHORT_FORM_CHARS, FormTable, _bag_bound  # noqa: E402

FLOOR = 0.72
_ALPHABETS = ["abcdefghij ", "abc ", "সেনমুখার্জীচৌধুরী ", "सेनमुखर्जीचौधरी ", "ab12 "]


def _random_string(rng, alphabet, lo=1, hi=18):
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi))).strip() or "a"


# ================================================================================= the bound is a real bound


def test_the_letter_bag_bound_never_falls_below_the_real_ratio():
    """The whole exactness argument rests on this: bound >= ratio for every pair. Tried on 6,000 random pairs over
    Latin, digit, Bengali and Devanagari alphabets, including pairs that are identical, empty-adjacent and tiny."""
    rng = random.Random(1)
    for _ in range(6000):
        alphabet = rng.choice(_ALPHABETS)
        a, b = _random_string(rng, alphabet), _random_string(rng, alphabet)
        real = difflib.SequenceMatcher(None, a, b).ratio()
        bound = _bag_bound(Counter(a), len(a), Counter(b), len(b))
        assert bound >= real - 1e-12, (a, b, bound, real)


# ================================================================================ pruning changes no decision


def _table_and_utterances():
    rng = random.Random(2)
    entries = ev.synthetic_catalogue(300)
    rows = {}
    for canonical, form in entries:
        rows.setdefault(canonical, []).append(form)
    for c, _f in entries[:40]:
        rows[c].append(rows[c][0] + " profile")
    utterances = []
    for canonical, form in rng.sample(entries, 120):
        variants = [v for vv in ev.variants(form, rng).values() for v in vv] or [form]
        for v in variants[:3]:
            frame = rng.choice(["price of {} test", "{}", "when does dr {} sit", "what should i do before {}"])
            utterances.append(frame.format(v))
    utterances += ["hello there", "thank you very much", "i want the price of something else entirely", ""]
    return list(rows.items()), utterances


def test_pruning_returns_the_same_answer_at_or_above_the_floor_and_nothing_below_it():
    rows, utterances = _table_and_utterances()
    table = FormTable(rows, index_min_forms=10**9)  # scan only: isolates the pruning
    checked = above = 0
    for u in utterances:
        words = u.split()
        full = table.best(words, floor=0.0)
        pruned = table.best(words, floor=FLOOR)
        checked += 1
        if full[2] >= FLOOR:
            above += 1
            assert pruned == full, (u, full, pruned)
        else:
            assert pruned == (None, None, 0.0), (u, full, pruned)
    assert above >= 100, f"only {above} of {checked} utterances reached the floor: the test is not exercising it"


def test_floor_zero_is_the_original_scan_including_a_low_best_score():
    table = FormTable([("Lipid Profile", ["lipid profile"]), ("Uric Acid", ["uric acid"])])
    name, form, score = table.best(["something", "entirely", "different"], floor=0.0)
    assert 0.0 <= score < FLOOR  # reported, not hidden, when no threshold was asked for
    assert table.best(["something", "entirely", "different"], floor=FLOOR) == (None, None, 0.0)


def test_a_tie_between_two_forms_goes_to_the_earlier_row_as_it_always_did():
    table = FormTable([("First", ["cbc"]), ("Second", ["cbc"])])
    assert table.best(["cbc"], floor=FLOOR)[0] == "First"


# ================================================================================ the shortlist only loses answers


def test_the_shortlist_never_changes_an_answer_it_only_ever_drops_one():
    rows, utterances = _table_and_utterances()
    scan = FormTable(rows, index_min_forms=10**9)
    fast = FormTable(rows, index_min_forms=0)  # force the shortlist on
    assert fast.indexed and not scan.indexed
    dropped = same = 0
    for u in utterances:
        words = u.split()
        a, b = scan.best(words, floor=FLOOR), fast.best(words, floor=FLOOR)
        if a == b:
            same += 1
        else:
            assert b == (None, None, 0.0), f"the shortlist produced a DIFFERENT answer for {u!r}: {a} vs {b}"
            dropped += 1
    # Measured on this corpus; if it ever grows, the parity is weakening and the threshold needs a look.
    assert dropped <= 0.02 * len(utterances), f"{dropped} of {len(utterances)} answers lost to the shortlist"
    assert same > 100


def test_a_short_form_is_never_dropped_by_the_shortlist():
    """A four-letter code with one letter changed still scores 0.75, and the shortlist must not lose it."""
    long_rows = [(f"Row {i}", [f"profile number {i} of the catalogue"]) for i in range(60)]
    table = FormTable(long_rows + [("TSH", ["tsh1"])], index_min_forms=0)
    assert len("tsh1") < SHORT_FORM_CHARS
    assert table.best(["price", "of", "tsx1", "please"], floor=FLOOR)[0] == "TSH"


def test_a_long_form_with_two_letters_wrong_survives_the_shortlist():
    rows = [(f"Row {i}", [f"catalogue entry {i} uric acid panel"]) for i in range(80)]
    rows.append(("Uric Acid", ["uric acid profile"]))
    table = FormTable(rows, index_min_forms=0)
    # two substitutions in a 16-character form: ratio 0.875
    assert table.best(["the", "price", "of", "uryc", "ecid", "profile"], floor=0.0)[2] < 1.0
    assert table.best(["uryc", "acod", "profile"], floor=FLOOR)[0] == "Uric Acid"


# ============================================================================================ exact-only short forms


def test_a_short_form_must_match_exactly_when_the_language_asks_for_it():
    lenient = FormTable([("Dr. Das", ["das"])], exact_below_chars=0)
    strict = FormTable([("Dr. Das", ["das"])], exact_below_chars=5)
    assert lenient.best(["in", "two", "days"], floor=FLOOR)[0] == "Dr. Das"  # 'days' ~ 'das': the old behaviour
    assert strict.best(["in", "two", "days"], floor=FLOOR) == (None, None, 0.0)  # what English and Hindi use
    assert strict.best(["doctor", "das", "today"], floor=FLOOR)[0] == "Dr. Das"


# ========================================================================================= scale and edge cases


def test_a_five_thousand_form_table_answers_well_inside_the_fast_path_budget():
    """Measured: about 4,900 ms per turn before (Bengali, 5,000 forms). Asserted at a small fraction of the 150 ms
    budget so a slow machine does not flake this."""
    entries = ev.synthetic_catalogue(5000)
    rows = [(c, [f]) for c, f in entries]
    table = FormTable(rows)
    assert table.indexed
    rng = random.Random(4)
    words_list = []
    for _c, f in rng.sample(entries, 30):
        vs = [v for vv in ev.variants(f, rng).values() for v in vv] or [f]
        words_list.append(f"price of {rng.choice(vs)} test".split())
    best_p95 = []
    for _attempt in range(3):
        times = []
        for words in words_list:
            t = time.perf_counter()
            table.best(words, floor=FLOOR)
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        best_p95.append(times[int(len(times) * 0.95) - 1])
    assert min(best_p95) < 50.0, best_p95
    per_lookup = table.stats["compared"] / table.stats["lookups"]
    assert per_lookup < 0.05 * len(table), f"{per_lookup:.0f} full comparisons per lookup on {len(table)} forms"


def test_an_empty_table_and_an_empty_utterance_are_harmless():
    assert FormTable([]).best(["anything"], floor=FLOOR) == (None, None, 0.0)
    assert FormTable([("A", ["abc"])]).best([], floor=FLOOR) == (None, None, 0.0)
    assert FormTable([("A", [])]).best(["abc"], floor=FLOOR) == (None, None, 0.0)
