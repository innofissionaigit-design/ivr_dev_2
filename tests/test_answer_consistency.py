# MERGE NOTE (sourav) -- this test file differed between dev_sourav and
# dev_rajarshee. Merged 3-way against their common ancestor (6cbeb0b ==
# test): every change from each branch touches a different part of the
# file, so it merged with NO conflicts -- rule 1, both sides kept whole.
# Changed regions vs the ancestor: 1 from dev_sourav, 0 from
# dev_rajarshee. Checked after merging: parses, and no test function or
# class name is defined twice (which would silently drop a test).
"""Asking twice does not produce two answers.

story title: The same question gets the same answer within one call
user story: As a caller who asks twice, I want the same answer, so that I
    know which one to believe.
acceptance criteria: Repeating a question in one call produces an identical
    factual answer unless the underlying data changed, in which case the
    change is stated. A test asserts consistency across three repeats with
    an unchanged backend.

THE TEST THE CRITERION NAMES IS test_three_repeats_with_an_unchanged_backend.
Everything else here exists because that one test, on its own, would pass
against an implementation that is wrong in both directions.

It would pass a reply cache -- which the Architecture Plan forbids in as many
words ("None. Do not optimise by caching replies"), because a cached reply is
how a caller gets told yesterday's price. test_nothing_is_ever_served_from_the
_ledger is the one that fails that implementation.

And it would pass a string comparison, which is the natural way to write this
and is wrong: reply_templates._spoken_test_name echoes the caller's own words
when a test has no Bengali alias, so two repeats of one question legitimately
produce two different sentences carrying one identical fact.
test_the_same_question_worded_three_ways is the one that fails THAT.

Everything here is pure -- no GPU, no model, no network.
"""
from __future__ import annotations

import ast
import asyncio
import pathlib
import sys
import unicodedata

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from agent import answer_ledger, speakability  # noqa: E402
from agent.bn_normalize import verbalize  # noqa: E402
from agent.reply_templates import (  # noqa: E402
    ANSWER_CHANGED_BN, doctor_availability_reply, doctors_by_department_reply,
    test_rate_reply as rate_reply,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"

# A test WITHOUT a Bengali alias, on purpose. That is the case where the
# spoken sentence echoes the caller's own words, and therefore the case where
# a sentence-comparing implementation breaks -- see the module docstring.
UNALIASED = {"found": True, "test_name": "Uric Acid", "rate_inr": "450",
             "sample_type": "রক্ত", "report_time_hours": "24"}
ALIASED = dict(UNALIASED, test_name_bn="ইউরিক অ্যাসিড")

IN_CHAMBER = {"found": True, "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
              "date": "2026-09-14", "available": True,
              "chamber_hours": "বিকেল ৫টা থেকে ৮টা", "next_available_date": None}

DEPARTMENT = {"found": True, "department": "Cardiology", "department_bn": "হৃদরোগ",
              "date": "2026-09-14",
              "doctors": [{"name": "Dr. A Sen", "doctor_name_bn": "সেন"},
                          {"name": "Dr. B Roy", "doctor_name_bn": "রায়"}]}


def _says(phrase: str, sentence: str) -> bool:
    """Substring test that survives Unicode normalisation -- Bengali য় exists
    as one codepoint and as two, and this repository contains both."""
    return unicodedata.normalize("NFC", phrase) in unicodedata.normalize("NFC", sentence)


def _changed(sentence: str) -> bool:
    return _says(ANSWER_CHANGED_BN, sentence)


class _Session:
    call_id = "iv03"
    utt_seq = 1

    def __init__(self):
        self.answer_ledger = answer_ledger.AnswerLedger()


@pytest.fixture
def call(monkeypatch):
    """A session, a list of what was said, and a fresh process counter."""
    said: list[str] = []

    async def _say(session, text, fallback_reason=None):
        said.append(text)

    monkeypatch.setattr(main, "_speak", _say)
    monkeypatch.setattr(main, "_consistency",
                        {"repeats": 0, answer_ledger.SAME: 0,
                         answer_ledger.CHANGED: 0})
    return _Session(), said


def _ask(session, said, intent, slots, result, template):
    """One turn: render the reply fresh, exactly as main.py does, and speak
    it through the gate."""
    asyncio.run(main._speak_fact(session, intent, slots, result,
                                 template(slots, result)))
    return said[-1]


# ------------------------------------------------ the criterion, literally

def test_three_repeats_with_an_unchanged_backend_are_identical(call):
    """The acceptance criterion, word for word."""
    session, said = call
    slots = {"test_name": "ইউরিক অ্যাসিড"}

    answers = [_ask(session, said, "test_rate", slots, ALIASED, rate_reply)
               for _ in range(3)]

    assert answers[0] == answers[1] == answers[2]
    assert not any(_changed(a) for a in answers), (
        "nothing changed, so nothing should have been announced")


def test_the_same_question_worded_three_ways_is_still_one_answer(call):
    """The test that fails a string-comparing implementation.

    Three ASR renderings of one question, against a test with no Bengali
    alias, so the reply echoes the caller's words and the three SENTENCES
    genuinely differ. The fact behind them never moved, and announcing a
    change here would be telling a caller the price moved because they
    pronounced it differently.
    """
    session, said = call
    answers = [_ask(session, said, "test_rate", {"test_name": said_as},
                    UNALIASED, rate_reply)
               for said_as in ("ইউরিক অ্যাসিড", "ইউরিক এসিদ", "ইউরিক অ্যাসিড টেস্ট")]

    assert len(set(answers)) > 1, (
        "the fixture stopped exercising the echo path -- this test now proves "
        "nothing, because the three sentences came out identical anyway")
    assert not any(_changed(a) for a in answers)
    assert main._consistency[answer_ledger.SAME] == 2


# ---------------------------------------------- ...unless the data changed

def test_a_changed_rate_is_stated(call):
    session, said = call
    slots = {"test_name": "ইউরিক অ্যাসিড"}

    first = _ask(session, said, "test_rate", slots, ALIASED, rate_reply)
    second = _ask(session, said, "test_rate", slots, ALIASED, rate_reply)
    third = _ask(session, said, "test_rate", slots,
                 dict(ALIASED, rate_inr="500"), rate_reply)

    assert not _changed(first) and not _changed(second)
    assert _changed(third)
    assert "500" in third, "the fresh figure is spoken"
    assert "450" not in third, "the superseded figure is not"


def test_the_change_is_stated_once_not_on_every_later_repeat(call):
    """The caller is told the figure moved. They are not told it again on
    every subsequent ask, which would sound like it kept moving."""
    session, said = call
    slots = {"test_name": "ইউরিক অ্যাসিড"}
    moved = dict(ALIASED, rate_inr="500")

    _ask(session, said, "test_rate", slots, ALIASED, rate_reply)
    announced = _ask(session, said, "test_rate", slots, moved, rate_reply)
    after = _ask(session, said, "test_rate", slots, moved, rate_reply)

    assert _changed(announced)
    assert not _changed(after)


def test_nothing_is_ever_served_from_the_ledger(call):
    """The Architecture Plan's constraint, as a test.

    Its failure-mode row for this exact scenario prescribes, in full: "None.
    Do not optimise by caching replies." What the caller hears is always the
    sentence rendered from THIS turn's response; the ledger may prepend to it
    and may never substitute for it.
    """
    session, said = call
    slots = {"test_name": "ইউরিক অ্যাসিড"}
    moved = dict(ALIASED, rate_inr="500")

    _ask(session, said, "test_rate", slots, ALIASED, rate_reply)
    second = _ask(session, said, "test_rate", slots, moved, rate_reply)

    assert second.endswith(rate_reply(slots, moved)), (
        "the spoken reply is not the freshly rendered one")
    assert "500" in second and "450" not in second, (
        "the old figure was read out -- a stale price reached the caller")


def test_a_trailing_zero_is_a_change(call):
    """450.00 -> 450.0 is a real change to what the caller hears, and
    float() would erase it. tools_client parses with parse_float=str for this
    reason; the ledger must compare the same way."""
    session, said = call
    slots = {"test_name": "ইউরিক অ্যাসিড"}

    _ask(session, said, "test_rate", slots, dict(ALIASED, rate_inr="450.00"), rate_reply)
    second = _ask(session, said, "test_rate", slots,
                  dict(ALIASED, rate_inr="450.0"), rate_reply)

    assert _changed(second)


def test_the_changed_reply_is_sayable(call):
    """A change statement the tokenizer drops is worse than no change
    statement: the caller would hear the new figure with a hole in front of
    it and no idea anything had moved."""
    session, said = call
    slots = {"test_name": "ইউরিক অ্যাসিড"}
    _ask(session, said, "test_rate", slots, ALIASED, rate_reply)
    changed = _ask(session, said, "test_rate", slots,
                   dict(ALIASED, rate_inr="500"), rate_reply)

    verdict = speakability.check(changed)
    assert verdict.state == speakability.SPEAKABLE, verdict.dropped
    assert ":" not in verbalize(changed)


# -------------------------------------------- what counts as one question

def test_another_question_in_between_disturbs_nothing(call):
    session, said = call
    uric = {"test_name": "ইউরিক অ্যাসিড"}
    sugar = {"test_name": "সুগার"}

    _ask(session, said, "test_rate", uric, ALIASED, rate_reply)
    _ask(session, said, "test_rate", sugar,
         {"found": True, "test_name": "Blood Sugar", "test_name_bn": "সুগার",
          "rate_inr": "150", "sample_type": "রক্ত", "report_time_hours": "6"},
         rate_reply)
    again = _ask(session, said, "test_rate", uric, ALIASED, rate_reply)

    assert not _changed(again)
    assert main._consistency[answer_ledger.CHANGED] == 0


def test_a_not_found_that_later_resolves_is_a_change(call):
    """Two sentences that contradict each other outright -- "no such test",
    then a price for it. The caller's own words are what tie them together,
    because a not-found response carries no catalogue identity at all."""
    session, said = call
    slots = {"test_name": "ইউরিক অ্যাসিড"}

    _ask(session, said, "test_rate", slots,
         {"found": False, "query": "ইউরিক অ্যাসিড"}, rate_reply)
    found = _ask(session, said, "test_rate", slots, ALIASED, rate_reply)

    assert _changed(found)


def test_availability_is_keyed_by_the_resolved_date(call):
    """"Is Dr Sen in?" asked about two different days is two questions. A
    date-less ledger would call the second answer a contradiction, which is
    also what would happen to a caller who asks "আজ" either side of
    midnight."""
    session, said = call
    slots = {"doctor_name": "সেন"}

    _ask(session, said, "doctor_availability", slots, IN_CHAMBER,
         doctor_availability_reply)
    other_day = _ask(session, said, "doctor_availability", slots,
                     dict(IN_CHAMBER, date="2026-09-15", available=False,
                          chamber_hours=None, next_available_date="2026-09-17"),
                     doctor_availability_reply)

    assert not _changed(other_day)
    assert main._consistency["repeats"] == 0, "two dates are two questions"


def test_a_slot_taken_between_two_asks_is_stated(call):
    session, said = call
    slots = {"doctor_name": "সেন"}

    _ask(session, said, "doctor_availability", slots, IN_CHAMBER,
         doctor_availability_reply)
    gone = _ask(session, said, "doctor_availability", slots,
                dict(IN_CHAMBER, available=False, chamber_hours=None,
                     next_available_date="2026-09-17"),
                doctor_availability_reply)

    assert _changed(gone)


def test_a_reordered_roster_is_not_a_change(call):
    """The clinic promises no ordering. A reshuffle is not news."""
    session, said = call
    slots = {"department": "কার্ডিওলজি"}
    flipped = dict(DEPARTMENT, doctors=list(reversed(DEPARTMENT["doctors"])))

    _ask(session, said, "doctors_by_department", slots, DEPARTMENT,
         doctors_by_department_reply)
    again = _ask(session, said, "doctors_by_department", slots, flipped,
                 doctors_by_department_reply)

    assert not _changed(again)


def test_a_doctor_leaving_the_roster_is_a_change(call):
    session, said = call
    slots = {"department": "কার্ডিওলজি"}
    shorter = dict(DEPARTMENT, doctors=DEPARTMENT["doctors"][:1])

    _ask(session, said, "doctors_by_department", slots, DEPARTMENT,
         doctors_by_department_reply)
    again = _ask(session, said, "doctors_by_department", slots, shorter,
                 doctors_by_department_reply)

    assert _changed(again)


# --------------------------------------------------- the ledger's own rules

def test_booking_is_never_ledgered():
    """A booking is a write, not a question. Asking twice is a second
    booking, and that is E4's idempotency story -- see the module docstring
    in agent/answer_ledger.py."""
    ledger = answer_ledger.AnswerLedger()
    result = {"success": True, "confirmation_id": "KCD-20260914-4A2F",
              "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন"}

    for _ in range(3):
        verdict, previous = ledger.check("book_appointment", {}, result)
        assert verdict == answer_ledger.FIRST
        assert previous is None
    assert ledger.snapshot()["entries"] == 0


def test_an_unresolvable_answer_is_not_compared():
    """Nothing nameable came back, so there is nothing this answer could be
    consistent with. A guessed identity would be worse than no check."""
    ledger = answer_ledger.AnswerLedger()
    for _ in range(2):
        verdict, _prev = ledger.check("test_rate", {}, {"found": False})
        assert verdict == answer_ledger.FIRST


def test_the_ledger_holds_no_caller_data():
    """Its PHI position is structural, not filtered: only catalogue
    identities and clinic facts ever enter it."""
    ledger = answer_ledger.AnswerLedger()
    slots = {"test_name": "ইউরিক অ্যাসিড", "patient_name": "রিয়া দাস",
             "phone": "9876543210"}
    ledger.check("test_rate", slots, ALIASED)

    blob = repr(ledger.__dict__)
    assert "রিয়া দাস" not in blob
    assert "9876543210" not in blob


def test_the_ledger_is_bounded():
    """A pathological call must not grow it without limit."""
    ledger = answer_ledger.AnswerLedger(max_entries=4)
    for n in range(20):
        ledger.check("test_rate", {}, dict(ALIASED, test_name=f"Test {n}"))
    assert ledger.snapshot()["entries"] == 4
    assert all(v in ledger._entries for v in ledger._index.values()), (
        "the key index still points at evicted entries")


def test_the_metric_counts_repeats_and_verdicts(call):
    session, said = call
    slots = {"test_name": "ইউরিক অ্যাসিড"}

    _ask(session, said, "test_rate", slots, ALIASED, rate_reply)          # first
    _ask(session, said, "test_rate", slots, ALIASED, rate_reply)          # same
    _ask(session, said, "test_rate", slots,
         dict(ALIASED, rate_inr="500"), rate_reply)                        # changed

    assert main._consistency == {"repeats": 2, answer_ledger.SAME: 1,
                                 answer_ledger.CHANGED: 1}
    assert session.answer_ledger.snapshot() == {
        "repeats": 2, answer_ledger.SAME: 1, answer_ledger.CHANGED: 1,
        "entries": 1}


# ------------------------------------------------- survives every refactor

def _tree() -> ast.Module:
    return ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))


FACTUAL_TEMPLATES = {"test_rate_reply", "doctor_availability_reply",
                     "doctors_by_department_reply",
                     # ADDED BY SOURAV -- KCD-385: doctor_schedule_reply now
                     # reaches the caller through _speak_fact too (see
                     # main.py/main_pcm.py's own "UPDATED BY SOURAV --
                     # KCD-385" comment on the doctor_schedule dispatch
                     # branch), so it belongs in the same mechanically
                     # enforced allow-list as the other three factual routes.
                     "doctor_schedule_reply"}


def _calls_to(tree: ast.Module, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == name]


def test_no_factual_reply_reaches_speak_directly():
    """The "survives every refactor" clause.

    Eight call sites speak a price or a roster. A consistency check applied
    at seven of them is not a guarantee, it is a coincidence -- and the one
    that got missed would be the one nobody tested. The rule is mechanical
    instead: these three templates reach the caller through _speak_fact or
    they do not reach the caller.
    """
    offenders = []
    for call in _calls_to(_tree(), "_speak"):
        if len(call.args) < 2:
            continue
        arg = call.args[1]
        if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                and arg.func.id in FACTUAL_TEMPLATES):
            offenders.append((call.lineno, arg.func.id))

    assert not offenders, (
        "factual reply spoken without the consistency check at "
        f"{offenders} -- use _speak_fact"
    )


def test_every_factual_route_still_goes_through_the_gate():
    """The other half. Deleting the call sites would also satisfy the test
    above."""
    calls = _calls_to(_tree(), "_speak_fact")
    assert len(calls) >= 8, (
        f"expected the eight factual routes, found {len(calls)} -- a route "
        f"was removed or stopped using _speak_fact")

    for call in calls:
        assert len(call.args) == 5, (
            f"_speak_fact at line {call.lineno} has the wrong arity")
        reply = call.args[4]
        assert (isinstance(reply, ast.Call) and isinstance(reply.func, ast.Name)
                and reply.func.id in FACTUAL_TEMPLATES), (
            f"_speak_fact at line {call.lineno} is not passed a freshly "
            f"rendered reply template -- this is the property "
            f"tests/test_fact_provenance.py exempts the wrapper on")


# ------------------------------------- the second route: combined turns
#
# ADDED BY SOURAV -- KCD-449. "What's the CBC rate, and is Dr Sen in
# today?" answers two of this story's three ledgered questions inside ONE
# turn, through _resolve_combinable_intent_fragment() and
# _record_answer_for_ledger() in main.py, never through _speak_fact().
# Everything above this point exercises only the solo dispatch route; a
# consistency check that only ever ran on _speak_fact() would leave a
# combined turn's answers completely unrecorded, invisible to every test
# above, and invisible to test_no_factual_reply_reaches_speak_directly's
# own AST walk (that test looks for a bare _speak() call, and this route
# never calls _speak() itself -- its caller joins the fragments and
# speaks them together).

def test_the_combined_turn_path_is_consistent_too(call):
    """The criterion test above, exercised through the second route
    instead of the first: three renders of the same combined-turn
    fragment, unchanged backend, identical facts each time."""
    session, _ = call
    slots = {"test_name": "ইউরিক অ্যাসিড"}

    fragments = [
        main._record_answer_for_ledger(session, "test_rate", slots, ALIASED,
                                       rate_reply(slots, ALIASED))
        for _ in range(3)
    ]

    assert fragments[0] == fragments[1] == fragments[2]
    assert not any(_changed(f) for f in fragments)


def test_a_combined_turn_still_announces_a_genuine_change(call):
    """The other half: a fact that actually moves between two combined-
    turn askings is still announced, exactly as it would be on the solo
    path."""
    session, _ = call
    slots = {"doctor_name": "সেন"}

    first = main._record_answer_for_ledger(
        session, "doctor_availability", slots, IN_CHAMBER,
        doctor_availability_reply(slots, IN_CHAMBER))
    changed_result = dict(IN_CHAMBER, available=False, next_available_date="2026-09-16")
    second = main._record_answer_for_ledger(
        session, "doctor_availability", slots, changed_result,
        doctor_availability_reply(slots, changed_result))

    assert not _changed(first)
    assert _changed(second)
    assert main._consistency == {"repeats": 1, answer_ledger.SAME: 0,
                                 answer_ledger.CHANGED: 1}


def test_one_question_is_one_question_whichever_route_asks_it(call):
    """The ledger has no notion of "which code path asked" -- an identity
    key is an identity key, so _speak_fact() (the solo path) and
    _record_answer_for_ledger() (the combined-turn path) must read from
    and write into the very same table. A caller who asks about a
    department plainly, then asks about it again folded into a combined
    turn, still gets caught on a genuine change -- and the reverse order
    works exactly the same way."""
    session, said = call
    slots = {"department": "হৃদরোগ"}

    solo = _ask(session, said, "doctors_by_department", slots, DEPARTMENT,
                doctors_by_department_reply)
    changed_result = dict(DEPARTMENT, doctors=[{"name": "Dr. A Sen", "doctor_name_bn": "সেন"}])
    combined = main._record_answer_for_ledger(
        session, "doctors_by_department", slots, changed_result,
        doctors_by_department_reply(slots, changed_result))

    assert not _changed(solo)
    assert _changed(combined), (
        "a roster change made through the combined-turn path was not "
        "recognised as a repeat of the solo-path question")


def test_no_factual_reply_bypasses_the_ledger_in_a_combined_turn():
    """The combined-turn counterpart of
    test_no_factual_reply_reaches_speak_directly above.

    _resolve_combinable_intent_fragment() never calls _speak() itself, so
    that test's own pattern -- walking the file for a bare _speak() call
    -- cannot see this function at all; a regression here would be
    completely invisible to the safeguard that already exists. This walks
    _resolve_combinable_intent_fragment()'s own `return` statements
    instead: a ledgered template's rendered result must be wrapped in
    _record_answer_for_ledger() before it leaves this function, or a
    combined turn's answer is unrecorded and never compared against a
    repeat -- same failure as the solo path, just on the route the other
    safeguard cannot reach.

    Deliberately narrower than FACTUAL_TEMPLATES above: doctor_schedule_
    reply() is intentionally NOT wrapped in _record_answer_for_ledger()
    here, because "doctor_schedule" is not one of
    answer_ledger.LEDGERED_INTENTS at all (see that module's own docstring
    -- this story's ledger only ever covers test_rate, doctor_availability
    and doctors_by_department) and _resolve_combinable_intent_fragment()
    never opens a near-match "did you mean" offer for ANY intent, ledgered
    or not, so there is nothing for doctor_schedule to be wrapped in on
    this path.
    """
    fn = None
    for node in ast.walk(_tree()):
        if (isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_resolve_combinable_intent_fragment"):
            fn = node
            break
    assert fn is not None, "_resolve_combinable_intent_fragment not found in main.py"

    ledgered_templates = {"test_rate_reply", "doctor_availability_reply",
                          "doctors_by_department_reply"}

    offenders = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            returned = node.value
            if isinstance(returned.func, ast.Name) and returned.func.id in ledgered_templates:
                offenders.append((node.lineno, returned.func.id))

    assert not offenders, (
        "factual reply returned from the combined-turn path without the "
        f"consistency check at {offenders} -- wrap in _record_answer_for_ledger"
    )
