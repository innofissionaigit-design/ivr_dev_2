"""Epic E33, agent side: agent/patient_context.py, agent/history_templates.py,
agent/call_record.py (KCD-494..501).

Pure over already-fetched dicts, so it runs anywhere. What is proved: nothing personal is
spoken before verification; a shared number is settled by an open question that lists
nobody; continuity and preferences are offered and never applied; a stored language bias
never overrides speech; every historical sentence is a template filled from a retrieved
field with a provenance id; a missing, stale or ambiguous field becomes "I cannot see
that"; a test's last date and due-ness may be stated and a result never can; and the
call recorder delivers every event once, in order, within its deadline, through failures.

    python -m pytest tests/test_patient_context.py -v
"""

import asyncio
import datetime
import os
import re
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _virtual_time import virtual_time

from agent import history_templates as ht
from agent import patient_context as pcx
from agent import persona
from agent.call_record import FINISH_DEADLINE_S, CallRecorder
from agent.identity import IdentityState

NOW = datetime.datetime(2030, 1, 7, 10, 0)


def _verified(patient="42"):
    s = IdentityState(clock=lambda: 1000.0)
    s.verify("otp", patient)
    return s


def _tl(events, as_of=NOW):
    return {
        "success": True,
        "as_of": as_of.isoformat(),
        "events": events,
        "sources": {"local": "ok", "his": "not_connected", "lis": "not_connected"},
    }


def _appt(n, date, slot="10:00", status="confirmed", doctor="Dr Sen"):
    return {
        "id": f"appointment:{n}",
        "kind": "appointment",
        "at": date,
        "source": "local",
        "as_of": NOW.isoformat(),
        "fields": {
            "confirmation_id": f"KC{n}",
            "doctor_name": doctor,
            "date": date,
            "time_slot": slot,
            "status": status,
        },
    }


def _perf(n, test, on):
    return {
        "id": f"test_performed:{n}",
        "kind": "test_performed",
        "at": on,
        "source": "lis",
        "as_of": NOW.isoformat(),
        "fields": {"test_name": test, "performed_on": on},
    }


# ================================================================ KCD-495: the gate


def test_nothing_personal_is_allowed_before_verification():
    d = pcx.history_gate(IdentityState(), "42", "en")
    assert not d.allowed and d.reason == "not_verified" and d.statement[0] == "needs_verification"
    claimed = IdentityState()
    claimed.claim("42")
    assert not pcx.history_gate(claimed, "42", "en").allowed  # saying who you are is not enough


def test_a_verification_opens_the_gate_for_that_patient_only():
    ident = _verified("42")
    assert pcx.history_gate(ident, "42", "en").allowed
    assert not pcx.history_gate(ident, "43", "en").allowed


def test_a_revoked_verification_closes_it_again():
    ident = _verified("42")
    ident.revoke("speaker_changed")
    assert not pcx.history_gate(ident, "42", "en").allowed


def test_a_number_with_no_record_says_so_and_nothing_else():
    d = pcx.history_gate(IdentityState(), None, "en", record_exists=False)
    assert d.reason == "no_record" and d.statement[0] == "no_record"


def test_the_refusal_names_no_one_and_offers_a_person():
    for lang in ("bn", "hi", "en"):
        text = ht.needs_verification(lang)[1]
        assert persona.is_clean(text, lang)
    assert "staff" in ht.needs_verification("en")[1].lower()


# ============================================================ KCD-494: shared numbers


def test_a_shared_number_is_settled_by_an_open_question_that_lists_nobody():
    for lang in ("bn", "hi", "en"):
        for attempt in (0, 1):
            q = pcx.resolve_question(lang, attempt)
            assert persona.is_clean(q, lang)
            if attempt == 0:
                assert "?" in q or "।" in q
    # the question can not contain a name because it takes none: it is a constant per language
    assert pcx.resolve_question("en", 0) == pcx.RESOLVE_QUESTION["en"]


def test_a_failed_lookup_is_a_new_caller_never_a_guess():
    assert pcx.identify_outcome({"status": "new"}) == "new"
    assert pcx.identify_outcome({"status": "single", "patient_ref": 1}) == "single"
    assert pcx.identify_outcome({"status": "ambiguous"}) == "ambiguous"
    assert pcx.identify_outcome({}) == "new" and pcx.identify_outcome(None) == "new"
    assert pcx.identify_outcome({"status": "banana"}) == "new"


# ============================================================ KCD-496: continuity


def test_before_verification_the_offer_says_something_was_left_not_what():
    ctx = {"unfinished": True, "draft": {"slots_json": '{"doctor_name": "Dr Sen"}'}}
    oid, text = pcx.continuity_offer(
        ctx, IdentityState(), "42", "en", draft_text="You were booking Dr Sen on the 14th."
    )
    assert oid == "continuity:generic" and "Sen" not in text and "?" in text


def test_after_verification_the_specifics_are_stated_for_the_caller_to_correct():
    ctx = {"unfinished": True}
    oid, text = pcx.continuity_offer(
        ctx, _verified("42"), "42", "en", draft_text="You were booking Dr Sen on the 14th."
    )
    assert oid == "continuity:specific" and "Dr Sen" in text


def test_nothing_unfinished_means_no_offer():
    assert pcx.continuity_offer({"unfinished": False}, _verified(), "42", "en") is None
    assert pcx.continuity_offer({}, _verified(), "42", "en") is None


def test_the_generic_offer_meets_the_persona_in_every_language():
    for lang, text in pcx.CONTINUITY_GENERIC.items():
        assert persona.is_clean(text, lang)


# ============================================================ KCD-499: preferences


def test_a_stored_language_bias_never_overrides_what_the_caller_says():
    assert pcx.effective_language("hi", "bn") == "hi"
    assert pcx.effective_language("en", "bn") == "en"
    assert pcx.effective_language(None, "hi") == "hi"  # nothing identified: the bias may help
    assert pcx.effective_language(None, None) == "bn"
    assert pcx.effective_language("unknown", "en") == "en"
    assert pcx.effective_language("xx", "fr") == "bn"  # an unsupported bias is ignored


def test_preferences_are_offered_only_after_verification():
    prefs = {"branch": "Salt Lake", "delivery_channel": "sms"}
    assert pcx.preference_offer(prefs, IdentityState(), "42", "en") is None
    offer = pcx.preference_offer(prefs, _verified("42"), "42", "en")
    assert offer is not None and "Salt Lake" in offer.text and "yes" in offer.text.lower()


def test_a_preference_is_never_applied_without_a_yes():
    offer = pcx.preference_offer({"branch": "Salt Lake"}, _verified("42"), "42", "en")
    slots = {"doctor_name": "Dr Sen"}
    assert offer.apply(slots, confirmed=False) == slots
    assert offer.apply(slots, confirmed=True) == {"doctor_name": "Dr Sen", "branch": "Salt Lake"}
    assert slots == {"doctor_name": "Dr Sen"}  # the original is untouched


def test_the_collection_address_is_offered_by_name_never_recited():
    offer = pcx.preference_offer({"collection_address": "12 Park Street, Flat 4B"}, _verified("42"), "42", "en")
    assert "Park Street" not in offer.text and "collection address" in offer.text
    assert offer.fields["collection_address"] == "12 Park Street, Flat 4B"  # applied only if confirmed


def test_no_stored_preference_means_no_offer():
    assert pcx.preference_offer({}, _verified("42"), "42", "en") is None
    assert pcx.preference_offer({"blood_group": "O+"}, _verified("42"), "42", "en") is None


# ============================================================== KCD-500: answering from fields


def test_appointments_are_rendered_from_the_timeline_with_provenance_ids():
    tl = _tl(
        [_appt(1, "2030-01-14", "10:00"), _appt(2, "2030-01-10", "09:15"), _appt(3, "2029-12-01", status="confirmed")]
    )
    a = pcx.answer_appointments(tl, "en", NOW)
    assert a.ids == ["appointment:2", "appointment:1"]  # upcoming only, soonest first; the past one is not listed
    assert "2030-01-10" in a.statements[0][1] and "Dr Sen" in a.text


def test_a_cancelled_appointment_is_not_stated_as_upcoming():
    tl = _tl([_appt(1, "2030-01-14", status="cancelled")])
    a = pcx.answer_appointments(tl, "en", NOW)
    assert a.ids == ["cannot_see"]


def test_no_timeline_at_all_says_it_cannot_see_never_that_there_is_nothing():
    for tl in (None, {"success": False, "reason": "not_authorized"}):
        a = pcx.answer_appointments(tl, "en", NOW)
        # SPEC CHANGE (KCD-500, operator wording): the "cannot see" statement is now "Sorry. I am unable to
        # find the details. Could you please guide me...". The rule is unchanged: it says it cannot find
        # it and asks for help; it never says there is nothing.
        assert a.ids == ["cannot_see"] and "unable to find" in a.text and "guide me" in a.text
        assert "no appointment" not in a.text.lower() and "nothing" not in a.text.lower()


def test_a_stale_timeline_is_not_confirmed():
    old = NOW - datetime.timedelta(hours=pcx.STALE_AFTER_HOURS + 1)
    a = pcx.answer_appointments(_tl([_appt(1, "2030-01-14")], as_of=old), "en", NOW)
    assert a.ids == ["cannot_confirm"]
    b = pcx.answer_last_test(_tl([_perf(1, "CBC", "2029-06-01")], as_of=old), "CBC", None, "en", NOW)
    assert b.ids == ["cannot_confirm"]


def test_a_missing_as_of_counts_as_stale():
    tl = _tl([_appt(1, "2030-01-14")])
    tl["as_of"] = None
    assert pcx.answer_appointments(tl, "en", NOW).ids == ["cannot_confirm"]


def test_the_last_date_of_a_test_is_stated_from_the_field():
    tl = _tl([_perf(1, "CBC", "2028-03-01"), _perf(2, "CBC", "2029-06-01"), _perf(3, "Lipid Profile", "2029-09-09")])
    a = pcx.answer_last_test(tl, "CBC", {"known": True, "due": None}, "en", NOW)
    assert a.ids == ["test_performed:2"] and "2029-06-01" in a.text  # the LATEST
    assert "due" not in a.text.lower()  # no approved interval: nothing about need


def test_due_is_stated_only_with_an_approved_interval():
    tl = _tl([_perf(1, "CBC", "2029-06-01")])
    yes = pcx.answer_last_test(tl, "CBC", {"known": True, "due": True, "interval_days": 90}, "en", NOW)
    assert yes.ids == ["test_performed:1", "due:yes"] and "approved" in yes.text
    no = pcx.answer_last_test(tl, "CBC", {"known": True, "due": False, "interval_days": 90}, "en", NOW)
    assert no.ids == ["test_performed:1", "due:no"]
    none = pcx.answer_last_test(tl, "CBC", {"known": True, "due": None}, "en", NOW)
    assert none.ids == ["test_performed:1"]


def test_a_test_that_is_not_in_the_records_is_not_claimed_never_done():
    a = pcx.answer_last_test(_tl([_perf(1, "CBC", "2029-06-01")]), "Thyroid", None, "en", NOW)
    assert a.ids == ["cannot_see"]
    assert (
        "never" not in a.text.lower() and "not been" not in a.text.lower()
    )  # absence of a record is not a fact about the patient


def test_a_name_that_fits_two_tests_is_asked_about_not_picked():
    tl = _tl([_perf(1, "Lipid Profile", "2029-06-01"), _perf(2, "Lipid Profile Extended", "2029-07-01")])
    a = pcx.answer_last_test(tl, "lipid", None, "en", NOW)
    assert a.ids == ["ambiguous"] and "date" in a.text.lower()


def test_no_history_text_contains_a_result_or_advice():
    banned = ("result", "normal", "abnormal", "high", "low", "should", "recommend", "worry", "elevated")
    tl = _tl([_appt(1, "2030-01-14"), _perf(2, "CBC", "2029-06-01")])
    texts = [pcx.answer_appointments(tl, l, NOW).text for l in ("en",)] + [
        pcx.answer_last_test(tl, "CBC", {"known": True, "due": True}, "en", NOW).text,
        ht.cannot_see("en")[1],
        ht.cannot_confirm("en")[1],
        ht.ambiguous("en")[1],
    ]
    for t in texts:
        assert not any(w in t.lower() for w in banned), t


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_every_history_template_meets_the_persona_in_every_language(lang):
    tl = _tl([_appt(1, "2030-01-14"), _perf(2, "CBC", "2029-06-01")])
    texts = [t for _, t in pcx.answer_appointments(tl, lang, NOW).statements]
    texts += [t for _, t in pcx.answer_last_test(tl, "CBC", {"known": True, "due": True}, lang, NOW).statements]
    texts += [pcx.answer_last_test(tl, "CBC", {"known": True, "due": False}, lang, NOW).statements[1][1]]
    texts += [
        ht.cannot_see(lang)[1],
        ht.cannot_confirm(lang)[1],
        ht.ambiguous(lang)[1],
        ht.no_record(lang)[1],
        ht.needs_verification(lang)[1],
        ht.report_ready_statement({"id": "report_ready:1", "fields": {"confirmation_id": "KC1"}}, lang)[1],
    ]
    for t in texts:
        assert persona.violations(t, lang) == [], (lang, t)
        assert (
            ":" not in re.sub(r"\d\d:\d\d", "", t) and "[" not in t
        )  # a time VALUE is spoken as words by the normaliser


def test_the_stale_constants_agree_between_the_service_and_the_agent():
    import importlib.util

    importlib.util.spec_from_file_location("svc_ctx", os.path.join(REPO_ROOT, "clinic-api", "patient_context.py"))
    src = open(os.path.join(REPO_ROOT, "clinic-api", "patient_context.py"), encoding="utf-8").read()
    assert f"STALE_AFTER_HOURS = {pcx.STALE_AFTER_HOURS}" in src


# ============================================== the guard on the one text a model writes


@pytest.mark.parametrize(
    "text,lang",
    [
        ("You had a blood test last time you visited.", "en"),
        ("Your last appointment was with Dr Sen.", "en"),
        ("Last time you were here you did an ECG.", "en"),
        ("You've been tested for sugar before.", "en"),
        ("আপনার শেষ টেস্ট গতবার ছিল", "bn"),
        ("आपका पिछला टेस्ट कल था", "hi"),
        ("आप पिछली बार आए थे", "hi"),
    ],
)
def test_a_model_written_history_claim_is_detected(text, lang):
    assert ht.mentions_personal_history(text, lang) and ht.mentions_personal_history(text)


@pytest.mark.parametrize(
    "text,lang",
    [
        ("Hello, how can I help you today?", "en"),
        ("I am well, thank you.", "en"),
        ("The clinic opens at nine in the morning.", "en"),
        ("নমস্কার, কী সাহায্য করতে পারি?", "bn"),
        ("नमस्कार, मैं आपकी क्या मदद कर सकती हूँ?", "hi"),
    ],
)
def test_ordinary_small_talk_is_not_taken_for_a_history_claim(text, lang):
    assert not ht.mentions_personal_history(text, lang)


# ================================================================ KCD-501: the call recorder


class FakeWriter:
    """Records what the 'server' applied, like clinic-api: an event applies only if its seq is higher
    than the last applied for that call, so duplicates are acknowledged and ignored."""

    def __init__(self, fail_first=0, drop_ack_every=0):
        self.applied, self.received = [], []
        self.last_seq = 0
        self.fail_first, self.drop_ack_every, self.calls = fail_first, drop_ack_every, 0

    async def __call__(self, call_id, seq, kind, payload, phone):
        self.calls += 1
        self.received.append((seq, kind))
        if self.fail_first and self.calls <= self.fail_first:
            raise ConnectionError("server unreachable")
        if seq > self.last_seq:
            self.last_seq = seq
            self.applied.append((seq, kind, payload))
        if self.drop_ack_every and self.calls % self.drop_ack_every == 0:
            raise TimeoutError("response lost after the write")  # applied server-side, client never heard
        return {"applied": True}


@pytest.mark.asyncio
async def test_events_are_written_in_order_as_they_happen():
    w = FakeWriter()
    r = CallRecorder("c1", w, "9000000001")
    r.start("voice", "1.0-draft")
    r.language("bn")
    r.intent("book_appointment")
    assert r.pending == 3
    assert await r.flush() == 0
    assert [k for _, k, _ in w.applied] == ["start", "language", "intent"] and [s for s, _, _ in w.applied] == [1, 2, 3]


@pytest.mark.asyncio
async def test_a_failed_write_stays_queued_and_order_is_preserved():
    w = FakeWriter(fail_first=1)
    r = CallRecorder("c2", w)
    r.intent("a")
    r.intent("b")
    assert await r.flush() == 2  # the first failed; the second waited behind it
    assert w.applied == []
    assert await r.flush() == 0
    assert [(s, k) for s, k, _ in w.applied] == [(1, "intent"), (2, "intent")]


@pytest.mark.asyncio
async def test_a_lost_acknowledgement_is_retried_without_doubling_anything():
    w = FakeWriter(drop_ack_every=1)  # every write lands, every ack is lost
    r = CallRecorder("c3", w)
    r.action("booking_confirmed", "KC1")
    await r.flush()
    await r.flush()
    await r.flush()
    assert len(w.applied) == 1  # the server applied it exactly once
    assert w.received.count((1, "action")) >= 2  # though the client had to retry


@pytest.mark.asyncio
async def test_sequence_numbers_are_assigned_once_and_never_reused():
    w = FakeWriter(fail_first=1)
    r = CallRecorder("c4", w)
    a = r.add("intent", {"intent": "x"})
    await r.flush()  # fails
    b = r.add("intent", {"intent": "y"})
    await r.flush()
    assert (a, b) == (1, 2) and [s for s, _, _ in w.applied] == [1, 2]


@virtual_time
async def test_finish_drains_through_transient_failures_inside_the_deadline():
    w = FakeWriter(fail_first=3)
    loop = asyncio.get_running_loop()
    r = CallRecorder("c5", w, clock=loop.time)
    r.action("booking_confirmed", "KC1")
    left = await r.finish("completed")
    assert left == 0 and w.applied[-1][1] == "end" and w.applied[-1][2] == {"outcome": "completed"}
    assert loop.time() < FINISH_DEADLINE_S  # written well within the thirty seconds


@virtual_time
async def test_finish_gives_up_at_its_deadline_and_reports_what_it_could_not_write():
    class Dead:
        async def __call__(self, *a):
            raise ConnectionError("down")

    loop = asyncio.get_running_loop()
    r = CallRecorder("c6", Dead(), clock=loop.time)
    r.action("booking_confirmed", "KC1")
    left = await r.finish("completed")
    assert left == 2 and r.unflushed_at_finish == 2  # the action and the end are still pending -- visibly
    assert FINISH_DEADLINE_S <= loop.time() < FINISH_DEADLINE_S + 5.0


@pytest.mark.asyncio
async def test_a_call_that_drops_has_already_written_what_it_completed():
    w = FakeWriter()
    r = CallRecorder("c7", w)
    r.start()
    r.action("booking_confirmed", "KC1")
    await r.flush()  # flushed as it happened
    # ... the socket closes; finish() is never reached ...
    assert [k for _, k, _ in w.applied] == ["start", "action"] and w.applied[1][2]["ref"] == "KC1"


@pytest.mark.asyncio
async def test_finish_twice_writes_one_end_event():
    w = FakeWriter()
    r = CallRecorder("c8", w)
    await r.finish("completed")
    await r.finish("completed")
    assert [k for _, k, _ in w.applied].count("end") == 1


@pytest.mark.asyncio
async def test_history_statements_are_recorded_only_when_there_are_some():
    w = FakeWriter()
    r = CallRecorder("c9", w)
    r.history([])
    r.history(["appointment:1", "due:no"])
    await r.flush()
    assert [(k, p["statements"]) for _, k, p in w.applied] == [("history", ["appointment:1", "due:no"])]
