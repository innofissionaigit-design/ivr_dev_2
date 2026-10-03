"""Comparing two options: the arithmetic, and nothing resembling a recommendation.

Ported from `merged_code/agent/compare_flow.py` (story: "Caller asks the agent to
compare two options"). Acceptance criteria: computed strictly in code from live
database values and stated clearly; never a clinical or purchasing
recommendation.

WHY THIS IS NEEDED HERE, AND WHAT CURRENTLY HAPPENS WITHOUT IT:

`clinic-api` already compares a health package against buying its tests
separately (`enquiry_service.compare_package_vs_separate`). Nothing compares two
arbitrary things -- "is a CBC cheaper than a lipid profile" has no path at all.

Worse, it has a path that actively misfires. `agent/fast_path.py`'s
`_ambiguous_test()` sees two tests fitting the words almost equally well and
abstains, by design, so the turn goes to the model -- and the model has no
compare intent (this merge leaves the intent set untouched), so a comparison
becomes `test_rate` for ONE of the two tests, or `unclear`. The caller asked
about two things and is told the price of one. That is why the guard that uses
this module runs BEFORE the fast path.

WHY DECIMAL, NEVER FLOAT:

This repository has a hard-won rule that a money figure is never altered by one
digit end to end: `agent/tools_client.py` parses numbers with `parse_float=str`
precisely because `float("450.00") == float("450.0")` silently eats a trailing
zero that matters for money.

Until this module, nothing in the pipeline did ARITHMETIC on a price -- every
number passed through a template unchanged. A price DIFFERENCE is the first real
arithmetic on a live money value, so it is done with `decimal.Decimal` over the
exact string or int the API sent, never `float`. Base-10 money arithmetic in
binary floating point can produce an inexact result (the classic 0.1 + 0.2
problem) which would then be spoken as a wrong number with complete confidence
-- the exact class of bug the number-fidelity rule exists to prevent.

WHAT THIS MODULE DELIBERATELY DOES NOT DO:

`build_comparison()` returns factual fields only: found-ness, the price delta,
which side is cheaper, and for two packages the test-count delta and which
tests differ. There is no "which is better" field, no adjective and no ranking
beyond the arithmetic ones -- a price is either lower or it is not, which is
arithmetic rather than a judgement.

Nothing here weighs a clinical or suitability axis, because neither the lab-test
nor the package catalogue exposes one: there is structurally nothing for this
function, or the model upstream, to base a medical recommendation on even if
asked. `agent/reply_templates.compare_options_reply()` renders this dict and
adds no facts of its own.
"""

from __future__ import annotations

import unicodedata
from decimal import Decimal

KIND_TEST = "test"
KIND_PACKAGE = "package"
KIND_NOT_FOUND = "not_found"


# ---------------------------------------------------------------- detection
# A request to compare, or to put two prices side by side. "which is better" is
# deliberately INCLUDED: a caller may well ask it, and the right response is to
# answer with the facts rather than to let the turn fall through to a model that
# might oblige with an opinion. The reply template states prices and nothing else.
_EN = (
    "which is cheaper",
    "which one is cheaper",
    "which is cheapest",
    "what is cheaper",
    "which costs less",
    "which one costs less",
    "which is more expensive",
    "which one is more expensive",
    "which is costlier",
    "cheaper or",
    "which is better",
    "which one is better",
    "difference between",
    "difference in price",
    "compare",
    "comparison between",
    "price difference",
    "how much more",
    "how much less",
)
_LATIN = (
    "konta sosta",
    "kon ta sosta",
    "konta kom",
    "kontar dam kom",
    "kontay kom",
    "kon ta beshi",
    "konta valo",
    "kon ta bhalo",
    "difference koto",
    "tulona",
    "kaun sasta",
    "kaun sa sasta",
    "konsa sasta",
    "kaun mehnga",
    "kaun sa mehnga",
    "kaun behtar",
    "kaun sa behtar",
    "antar kitna",
    "fark kitna",
    "tulna",
)
_BN = (
    "কোনটা সস্তা",
    "কোনটার দাম কম",
    "কোনটা কম",
    "কোনটা বেশি",
    "কোনটার দাম বেশি",
    "কোনটা ভালো",
    "দামের পার্থক্য",
    "পার্থক্য কত",
    "তুলনা",
    "কোনটা সবচেয়ে সস্তা",
)
_HI = (
    "कौन सस्ता",
    "कौन सा सस्ता",
    "कौन ज़्यादा",
    "कौन सा ज़्यादा",
    "कौन महंगा",
    "कौन सा महंगा",
    "कौन बेहतर",
    "कौन सा बेहतर",
    "कीमत में अंतर",
    "अंतर कितना",
    "फ़र्क़ कितना",
    "फर्क कितना",
    "तुलना",
)

_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("en", _EN),
    ("latin", _LATIN),
    ("bn", _BN),
    ("hi", _HI),
)


def detect_comparison_request(text: str) -> tuple[bool, str | None]:
    """True, plus which phrase family matched, when the caller is asking for two
    options to be compared. The tag is for logging only."""
    if not text or not text.strip():
        return False, None
    raw = unicodedata.normalize("NFC", text)
    lowered = raw.lower()
    for tag, phrases in _FAMILIES:
        if any((p in lowered) if p.isascii() else (p in raw) for p in phrases):
            return True, tag
    return False, None


def asks_to_compare(text: str) -> bool:
    matched, _ = detect_comparison_request(text)
    return matched


# --------------------------------------------------------------- arithmetic
def _to_decimal(raw) -> Decimal | None:
    """Exact base-10 parse of a money value.

    `raw` is a plain int (whole rupees, e.g. 650) or a str preserving the exact
    source digits ("199.55", "250.0"). `Decimal(str(raw))` is exact for both:
    unlike `float()` it parses the base-10 text directly rather than
    approximating it in base 2.

    Returns None for a missing or unparseable value rather than raising -- an
    entity that was found but carries no price is a fact `build_comparison()`
    handles honestly, not a crash.
    """
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except Exception:  # noqa: BLE001  -- a bad stored value must never 500 a call
        return None


def _price_of(entity: dict) -> Decimal | None:
    """Whichever price field this side's kind actually carries.

    "test" reads `rate_inr`, the same field `test_rate_reply()` already speaks;
    "package" reads `bundled_price_inr`, the field this repository's package
    endpoint returns. Neither is renamed or reshaped here.
    """
    kind = entity.get("kind")
    if kind == KIND_TEST:
        return _to_decimal(entity.get("rate_inr"))
    if kind == KIND_PACKAGE:
        return _to_decimal(entity.get("bundled_price_inr"))
    return None


def _package_tests(entity: dict) -> list[dict]:
    return [t for t in (entity.get("tests") or []) if isinstance(t, dict)]


def _names(tests: list[dict]) -> set[str]:
    return {str(t.get("name") or "").strip().lower() for t in tests if t.get("name")}


def build_comparison(entity_a: dict, entity_b: dict) -> dict:
    """-> a dict of FACTS ONLY, for `compare_options_reply()` to render.

    `entity_a` / `entity_b` are each an already-resolved lookup tagged with
    `"kind"`. This function performs NO I/O: the orchestrator owns both lookups,
    the same shape as every other interpret-a-result function here.

    Fields:
      a_found / b_found, a_kind / b_kind,
      price_delta       digit-exact absolute difference as a string, only when
                        BOTH sides have a price,
      cheaper           "a" | "b" | None (None when equal, or when either side
                        has no price),
      both_packages,
      identical_tests   only meaningful when both_packages,
      test_count_delta  only when both_packages,
      more_tests_side   "a" | "b" | None (None on an equal COUNT, even if the
                        actual tests differ -- see identical_tests),
      extra_tests_a / extra_tests_b   tests one package has and the other does
                        not, only when both_packages.
    """
    a_kind, b_kind = entity_a.get("kind"), entity_b.get("kind")
    out: dict = {
        "a_found": a_kind != KIND_NOT_FOUND,
        "b_found": b_kind != KIND_NOT_FOUND,
        "a_kind": a_kind,
        "b_kind": b_kind,
        "price_delta": None,
        "cheaper": None,
        "both_packages": a_kind == KIND_PACKAGE and b_kind == KIND_PACKAGE,
        "identical_tests": None,
        "test_count_delta": None,
        "more_tests_side": None,
        "extra_tests_a": [],
        "extra_tests_b": [],
    }
    if not (out["a_found"] and out["b_found"]):
        return out

    price_a, price_b = _price_of(entity_a), _price_of(entity_b)
    if price_a is not None and price_b is not None:
        out["price_delta"] = str(abs(price_a - price_b))
        if price_a < price_b:
            out["cheaper"] = "a"
        elif price_b < price_a:
            out["cheaper"] = "b"
        # equal prices leave `cheaper` None: there is no cheaper one.

    if out["both_packages"]:
        tests_a, tests_b = _package_tests(entity_a), _package_tests(entity_b)
        names_a, names_b = _names(tests_a), _names(tests_b)
        out["identical_tests"] = names_a == names_b
        out["test_count_delta"] = abs(len(tests_a) - len(tests_b))
        if len(tests_a) > len(tests_b):
            out["more_tests_side"] = "a"
        elif len(tests_b) > len(tests_a):
            out["more_tests_side"] = "b"
        out["extra_tests_a"] = [t for t in tests_a if str(t.get("name") or "").strip().lower() not in names_b]
        out["extra_tests_b"] = [t for t in tests_b if str(t.get("name") or "").strip().lower() not in names_a]
    return out
