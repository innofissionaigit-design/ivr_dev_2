"""agent/answer_ledger.py -- a repeat must not contradict what was already said.

NOTE: the module under test is deliberately NOT wired into the orchestrator (see
its docstring for why -- the wiring needs a signature change to
`_answer_enquiry_intent` touching 12 call sites and 9 test files). These tests
therefore cover the mechanism, not a live behaviour, and are written so that the
session which does the wiring inherits a proven component rather than having to
re-derive it.

What the groups below are actually protecting:

  * the verdicts themselves -- first / same / changed -- against the real result
    shapes `agent/api_models.py` defines, since a field-name mismatch would make
    this silently never fire;
  * the cases where announcing a change would be WRONG, which is where this kind
    of watcher does damage: an ambiguous answer is a question and not an answer,
    a date-scoped question asked on two different days is two questions, and a
    change already announced once must not be announced again;
  * number fidelity: "450.00" and "450.0" must compare as different, because the
    clinic sent different digits and `float()` would erase exactly the change
    this exists to catch.
"""

from __future__ import annotations

from agent.answer_ledger import CHANGED, FIRST, SAME, AnswerLedger, facts, identity_keys

RATE = {"found": True, "test_name": "CBC", "rate_inr": "450", "sample_type": "blood", "report_time_hours": "24"}
AVAIL = {
    "found": True,
    "doctor_name": "Dr. A. Sen",
    "date": "2026-10-02",
    "available": True,
    "chamber_hours": "10:00-13:00",
    "next_available_date": None,
}


def _rate(**over):
    return {**RATE, **over}


def _avail(**over):
    return {**AVAIL, **over}


# ------------------------------------------------------------- the verdicts
def test_the_first_time_a_question_is_asked_there_is_nothing_to_compare():
    led = AnswerLedger()
    assert led.check("test_rate", {"test_name": "CBC"}, _rate()) == (FIRST, None)


def test_asking_twice_with_an_unchanged_backend_is_the_same_answer():
    """The story's own acceptance criterion names three repeats, so three are
    asked here rather than two."""
    led = AnswerLedger()
    led.check("test_rate", {"test_name": "CBC"}, _rate())
    for _ in range(3):
        verdict, previous = led.check("test_rate", {"test_name": "CBC"}, _rate())
        assert verdict == SAME
        assert previous == facts("test_rate", _rate())


def test_a_price_that_moved_between_two_turns_is_reported_as_changed():
    led = AnswerLedger()
    led.check("test_rate", {"test_name": "CBC"}, _rate())
    verdict, previous = led.check("test_rate", {"test_name": "CBC"}, _rate(rate_inr="500"))
    assert verdict == CHANGED
    assert "450" in previous, "the earlier answer must be reported so the change can be stated"


def test_an_availability_answer_that_flips_is_reported_as_changed():
    led = AnswerLedger()
    led.check("doctor_availability", {"doctor_name": "Sen"}, _avail())
    verdict, _ = led.check("doctor_availability", {"doctor_name": "Sen"}, _avail(available=False))
    assert verdict == CHANGED


def test_a_field_the_caller_never_hears_cannot_announce_a_change():
    """`facts()` mirrors what the template speaks. An internal field moving is
    not something to tell a caller about."""
    led = AnswerLedger()
    led.check("test_rate", {"test_name": "CBC"}, _rate())
    verdict, _ = led.check("test_rate", {"test_name": "CBC"}, _rate(internal_row_version="77"))
    assert verdict == SAME


def test_an_unledgered_intent_is_ignored_entirely():
    led = AnswerLedger()
    assert led.check("clinic_faq", {"faq_topic": "hours"}, {"found": True, "answer": "8-8"}) == (FIRST, None)
    assert led.check("smalltalk", {}, {}) == (FIRST, None)


# --------------------------------------- where announcing a change would be wrong
def test_an_ambiguous_answer_is_a_question_and_is_never_recorded():
    """The worst failure this module could have. An ambiguous response has no
    resolved identity, so recording it would make the disambiguated answer one
    turn later read as a change -- announcing a move that never happened, on the
    turn where the caller is least sure of themselves."""
    led = AnswerLedger()
    ambiguous = {"found": False, "ambiguous": True, "query": "blood sugar", "did_you_mean": ["FBS", "PPBS"]}
    assert led.check("test_rate", {"test_name": "blood sugar"}, ambiguous) == (FIRST, None)
    # the real answer that follows is a FIRST, not a CHANGED
    verdict, _ = led.check("test_rate", {"test_name": "blood sugar"}, _rate(test_name="FBS"))
    assert verdict == FIRST


def test_the_same_dated_question_on_two_different_dates_is_two_questions():
    """"Is Dr Sen in today" asked at 23:59 and again at 00:01 are different
    questions; a date-less ledger would report a contradiction between two
    perfectly correct answers."""
    led = AnswerLedger()
    led.check("doctor_availability", {"doctor_name": "Sen"}, _avail(date="2026-10-02"))
    verdict, _ = led.check("doctor_availability", {"doctor_name": "Sen"}, _avail(date="2026-10-03", available=False))
    assert verdict == FIRST


def test_a_change_is_announced_once_and_not_again():
    """The entry is overwritten on a change, so a third ask says SAME. The
    caller is told once; repeating it would sound like a second move."""
    led = AnswerLedger()
    led.check("test_rate", {"test_name": "CBC"}, _rate())
    assert led.check("test_rate", {"test_name": "CBC"}, _rate(rate_inr="500"))[0] == CHANGED
    assert led.check("test_rate", {"test_name": "CBC"}, _rate(rate_inr="500"))[0] == SAME


def test_an_answer_with_nothing_nameable_resolved_is_not_recorded():
    """There is nothing such an answer could be consistent WITH, and a guessed
    identity is worse than silence."""
    led = AnswerLedger()
    assert led.check("test_rate", {}, {"found": False}) == (FIRST, None)


def test_a_reshuffle_of_suggestions_is_not_a_change():
    """The clinic promises no order, so sorting them means a reshuffle does not
    get announced -- but a genuinely different suggestion still does."""
    led = AnswerLedger()
    nf = {"found": False, "query": "sugar", "did_you_mean": ["FBS", "PPBS"]}
    led.check("test_rate", {"test_name": "sugar"}, nf)
    assert led.check("test_rate", {"test_name": "sugar"}, {**nf, "did_you_mean": ["PPBS", "FBS"]})[0] == SAME
    assert led.check("test_rate", {"test_name": "sugar"}, {**nf, "did_you_mean": ["FBS", "HbA1c"]})[0] == CHANGED


def test_switching_reply_language_mid_call_is_not_a_change():
    """A trilingual adjustment to the original, which compared the spoken
    Bengali suggestions. Comparing the localised list would report a "change"
    merely because the caller switched language."""
    led = AnswerLedger()
    nf = {"found": False, "query": "sugar", "did_you_mean": ["FBS"], "did_you_mean_bn": ["এফবিএস"]}
    led.check("test_rate", {"test_name": "sugar"}, nf)
    assert led.check("test_rate", {"test_name": "sugar"}, {**nf, "did_you_mean_hi": ["एफबीएस"]})[0] == SAME


# ------------------------------------------------------------ number fidelity
def test_a_rate_is_compared_as_the_digits_the_clinic_sent():
    """`float("450.00") == float("450.0")` would erase a real change at exactly
    the boundary the number-fidelity rule exists to protect."""
    led = AnswerLedger()
    led.check("test_rate", {"test_name": "CBC"}, _rate(rate_inr="450.00"))
    assert led.check("test_rate", {"test_name": "CBC"}, _rate(rate_inr="450.0"))[0] == CHANGED


def test_available_stays_a_two_valued_thing():
    """`available` must not become the strings "True"/"False", or a None and a
    False would compare equal."""
    f_true = facts("doctor_availability", _avail(available=True))
    f_none = facts("doctor_availability", _avail(available=None))
    assert f_true != f_none
    assert True in f_true


# ------------------------------------------------------- identity and bookkeeping
def test_an_answer_is_findable_by_the_catalogue_name_and_by_the_callers_words():
    keys = identity_keys("test_rate", {"test_name": "cbc test"}, _rate())
    kinds = {k[1] for k in keys}
    assert "id" in kinds and "said" in kinds


def test_the_callers_own_words_find_an_earlier_not_found_answer_again():
    """A not-found answer carries no catalogue identity, only the echoed query,
    so the caller's words are the only axis it can sit on."""
    led = AnswerLedger()
    nf = {"found": False, "query": "xyz test", "did_you_mean": []}
    led.check("test_rate", {"test_name": "xyz test"}, nf)
    assert led.check("test_rate", {"test_name": "xyz test"}, nf)[0] == SAME


def test_the_same_words_resolving_to_a_different_row_is_a_changed_answer():
    """Load-bearing even when both rows quote the same price: the caller asked
    one thing and was told about two different tests."""
    led = AnswerLedger()
    led.check("test_rate", {"test_name": "sugar"}, _rate(test_name="FBS", rate_inr="200"))
    verdict, _ = led.check("test_rate", {"test_name": "sugar"}, _rate(test_name="PPBS", rate_inr="200"))
    assert verdict == CHANGED


def test_the_ledger_is_bounded_against_a_pathological_call():
    led = AnswerLedger(max_entries=4)
    for i in range(20):
        led.check("test_rate", {"test_name": f"t{i}"}, _rate(test_name=f"T{i}"))
    assert led.snapshot()["entries"] <= 4


def test_the_snapshot_reports_what_somebody_should_be_alerted_about():
    led = AnswerLedger()
    led.check("test_rate", {"test_name": "CBC"}, _rate())
    led.check("test_rate", {"test_name": "CBC"}, _rate())
    led.check("test_rate", {"test_name": "CBC"}, _rate(rate_inr="500"))
    snap = led.snapshot()
    assert snap["repeats"] == 2 and snap[SAME] == 1 and snap[CHANGED] == 1
