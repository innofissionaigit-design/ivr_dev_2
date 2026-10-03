"""KCD-387 full flow: "Caller asks whether a test can be collected at home", through the REAL
orchestrator, off-pod.

    python -m pytest tests/test_orchestrator_home_collection.py -v

Same approach as tests/test_orchestrator_booking_flow.py: the real `_dispatch_turn`, `_speak` and
helpers from main_pcm.py, with the pod-only libraries stubbed and the extractor, ASR and clinic
tools replaced by fakes. What is real is the wiring and the order of the checks -- every number a
caller hears in these tests comes from FakeTools, standing in for the real clinic-api response, the
same substitution the real agent/tools_client.py makes over HTTP.

This file is the proof the story's own "FINAL RULE" asks for: a caller naming a test and a pincode
reaches the full deterministic flow (eligibility -> pincode -> date -> slot -> hold -> quote ->
payment policy -> proceed? -> patient details -> address -> address confirm -> final confirm ->
real booking -> a spoken result with a real booking reference and, only when one was actually
assigned, a real collector's name) through the SAME `_dispatch_turn` a live call runs, not a
backend endpoint called directly.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _pod_stubs import pod_stubs
from _synth_voices import FATHER, utterance
from test_orchestrator_persona import ASRResult, FakeTTS, FakeWS, _wav


class FakeHomeCollectionTools:
    """The clinic API as the home-collection flow uses it (agent/tools_client.py's own 10 new
    methods) -- records what it was asked, answers with the same shapes
    clinic-api/home_collection_service.py's real functions return."""

    def __init__(self):
        self.holds, self.releases, self.books, self.cancels = [], [], [], []
        self.eligibility_by_test = {
            "Uric Acid": {"eligible": True, "charge_inr": 100, "rate_inr": 250},
            "Lipid Profile": {"eligible": True, "charge_inr": 100, "rate_inr": 400},
            "Biopsy": {"eligible": False, "reason": "sample_type"},
        }
        self.serviceable_postal_codes = {"700091", "700019"}
        self.slots_by_date = {"2026-10-05": [{"slot_id": 1, "date": "2026-10-05", "start_time": "08:00", "end_time": "10:00", "remaining_capacity": 2}]}
        self.collector_name = "Ananya Das"
        self.booking_lookup_matches: list[dict] = []
        self.booking_test_names_by_id: dict[str, list[str]] = {}

    async def search_bookings(self, _key, phone=None, **kw):
        return {"matches": self.booking_lookup_matches}

    async def booking_test_names(self, confirmation_id):
        return {"test_names": self.booking_test_names_by_id.get(confirmation_id, [])}

    async def home_collection_eligibility_multi(self, test_names, postal_code):
        results, not_found = [], []
        serviceable = postal_code in self.serviceable_postal_codes
        for name in test_names:
            row = self.eligibility_by_test.get(name)
            if row is None:
                not_found.append(name)
                continue
            if not serviceable:
                results.append({"found": True, "eligible": False, "reason": "area_not_covered", "test_name": name})
            elif not row["eligible"]:
                results.append({"found": True, "eligible": False, "reason": row["reason"], "test_name": name})
            else:
                results.append({"found": True, "eligible": True, "charge_inr": row["charge_inr"], "test_name": name})
        return {"found": True, "results": results, "not_found": not_found}

    async def home_collection_slots(self, postal_code, date):
        return {"found": True, "slots": list(self.slots_by_date.get(date, []))}

    async def home_collection_hold(self, slot_id, caller_phone=None, call_id=None):
        self.holds.append(slot_id)
        return {"success": True, "hold_token": f"hold-{len(self.holds)}", "slot_id": slot_id}

    async def home_collection_release_hold(self, hold_token):
        self.releases.append(hold_token)
        return {"success": True}

    async def home_collection_quote(self, test_names, postal_code):
        per_test = []
        total_rate = 0
        for name in test_names:
            row = self.eligibility_by_test.get(name) or {}
            per_test.append({"test_name": name, "eligible": True, "rate_inr": row.get("rate_inr", 0)})
            total_rate += row.get("rate_inr", 0)
        charge = 100
        return {
            "found": True,
            "quote_available": True,
            "per_test": per_test,
            "eligible_tests": list(test_names),
            "test_charges_inr": total_rate,
            "home_collection_charge_inr": charge,
            "total_inr": total_rate + charge,
        }

    async def home_collection_payment_policy(self, lang="bn"):
        return {"found": True, "policy": "pay_on_collection", "description": "Pay the collector at the time of collection."}

    async def home_collection_book(self, **kw):
        self.books.append(kw)
        return {
            "success": True,
            "booking_reference": f"HC-TEST{len(self.books)}",
            "date": "2026-10-05",
            "start_time": "08:00",
            "end_time": "10:00",
            "address_line": kw["address_line"],
            "pincode": kw["postal_code"],
            "test_charges_inr": 250,
            "home_collection_charge_inr": 100,
            "total_inr": 350,
            "payment_policy": "pay_on_collection",
            "assignment_status": "assigned",
            "collector_name": self.collector_name,
            "dispatch_state": "ASSIGNED",
            "status": "confirmed",
        }

    async def home_collection_booking_status(self, booking_reference):
        return {"found": True}

    async def home_collection_cancel(self, booking_reference):
        self.cancels.append(booking_reference)
        return {"success": True, "booking_reference": booking_reference}

    async def get_preferences(self, patient_ref, caller_phone):
        return {"preferences": {}}


@pytest.fixture(scope="module")
def m():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


@pytest.fixture
def env(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_admission", None)
    monkeypatch.setattr(m, "_tts_router", FakeTTS())
    monkeypatch.setattr(m, "CONDITION_INPUT", "off")
    tools = FakeHomeCollectionTools()
    monkeypatch.setattr(m, "_tools", tools)
    state = {"text": "ok", "lang": "en", "data": {}}

    async def route(session, wav):
        return state["lang"], ASRResult(state["text"])

    async def resolve(session, text, lang):
        return {"secondary_intent": None, "direct_reply_bn": None, **state["data"]}

    monkeypatch.setattr(m, "_route_and_transcribe", route)
    monkeypatch.setattr(m, "_resolve_intent", resolve)
    session = m.CallSession(FakeWS())
    session.release_gate()
    session.disclosed_langs.add("en")
    session.call_state = m.new_call_state()
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)

    class Driver:
        pass

    d = Driver()
    d.session, d.state, d.tools = session, state, tools

    async def say(text, intent="unclear", slots=None, **extra):
        state["text"] = text
        state["data"] = {"intent": intent, "slots": dict(slots or {}), **extra}
        session.turn_epoch = session.speak_epoch
        path = _wav(tmp_path, voice, f"u{len(session.ws.texts)}.wav")
        before = len(session.ws.spoken())
        await m._dispatch_turn(session, path)
        return session.ws.spoken()[before:]

    d.say = say
    yield d
    session.cleanup()


def text_of(said):
    return " ".join(said)


SLOTS = {"test_name": "Uric Acid", "postal_code": "700091"}


async def reach_proceed(env):
    """Drives the flow from nothing through the quote/"proceed?" question -- single test, single
    slot (auto-chosen, nothing to choose), one real date."""
    await env.say(
        "can this be collected at home, my pincode is 700091", "home_collection", dict(SLOTS)
    )
    said = await env.say("5th October", "home_collection", {"date": "2026-10-05"})
    return said


@pytest.mark.asyncio
async def test_a_test_and_pincode_in_one_turn_reaches_the_real_flow_not_the_generic_faq(m, env):
    # This is the story's own "FINAL RULE" worked example: naming a specific test AND a pincode
    # must never be answered by the generic clinic_faq/home_collection sentence.
    said = await env.say(
        "Amar Uric Acid test ta bari theke collect kora jabe? Amar pincode 700091",
        "home_collection",
        dict(SLOTS),
    )
    spoken = text_of(said)
    assert "Uric Acid" in spoken
    assert "বুকিং-এর সময়" not in spoken  # never the fixed FAQ sentence's own wording
    assert env.session.home_collection is not None
    assert env.session.home_collection.eligible_tests == ["Uric Acid"]


@pytest.mark.asyncio
async def test_full_happy_path_books_a_real_appointment_with_a_real_reference_and_collector(m, env):
    await reach_proceed(env)
    st = env.session.home_collection
    assert st.stage == "awaiting_proceed"
    assert st.quote["total_inr"] == 350
    assert env.tools.holds == [1]  # a real slot capacity unit was claimed before "proceed?" was even asked

    said = await env.say("yes", "unclear", {})
    assert "patient" in text_of(said).lower() or "নাম" in text_of(said)

    said = await env.say("Ravi Das", "home_collection", {"patient_name": "Ravi Das"})
    assert "phone" in text_of(said).lower() or "ফোন" in text_of(said)

    said = await env.say("9876543210", "home_collection", {"phone": "9876543210"})
    # No stored address (FakeTools.get_preferences returns {}) -- asked for one fresh.
    assert "address" in text_of(said).lower() or "ঠিকানা" in text_of(said)

    said = await env.say(
        "12 Lake Road, Kolkata", "home_collection", {"address_line": "12 Lake Road, Kolkata"}
    )
    assert "correct" in text_of(said).lower() or "ঠিক আছে তো" in text_of(said)

    said = await env.say("yes", "unclear", {})
    assert "confirm" in text_of(said).lower() or "কনফার্ম করব" in text_of(said)

    said = await env.say("yes", "unclear", {})
    spoken = text_of(said)
    assert env.tools.books, "the real booking endpoint was actually called"
    booked = env.tools.books[0]
    assert booked["test_names"] == ["Uric Acid"]
    assert booked["hold_token"] == "hold-1"
    assert "HC-TEST1" in spoken  # the real booking reference, never invented
    assert "Ananya Das" in spoken  # a real assignment -- only spoken because one actually exists
    assert env.session.home_collection is None  # the flow is over, cleanly


@pytest.mark.asyncio
async def test_an_ineligible_test_never_reaches_a_booking(m, env):
    said = await env.say(
        "can a biopsy be collected at home, pincode 700091",
        "home_collection",
        {"test_name": "Biopsy", "postal_code": "700091"},
    )
    spoken = text_of(said)
    assert "lab" in spoken.lower() or "ল্যাবেই" in spoken
    assert env.session.home_collection is None  # terminal -- nothing left to continue
    assert not env.tools.holds and not env.tools.books


@pytest.mark.asyncio
async def test_an_uncovered_pincode_is_reported_honestly_never_guessed_serviceable(m, env):
    said = await env.say(
        "Uric Acid test, pincode 400001", "home_collection", {"test_name": "Uric Acid", "postal_code": "400001"}
    )
    spoken = text_of(said)
    assert "area" in spoken.lower() or "এলাকায়" in spoken
    assert env.session.home_collection is None
    assert not env.tools.holds


@pytest.mark.asyncio
async def test_a_caller_who_declines_to_proceed_releases_the_held_slot(m, env):
    await reach_proceed(env)
    assert env.tools.holds == [1]
    said = await env.say("no", "unclear", {})
    assert env.tools.releases == ["hold-1"]  # real capacity freed immediately, not left to its TTL
    assert env.session.home_collection is None
    assert not env.tools.books


@pytest.mark.asyncio
async def test_a_caller_who_declines_the_final_confirm_still_releases_the_hold_and_books_nothing(m, env):
    await reach_proceed(env)
    await env.say("yes", "unclear", {})
    await env.say("Ravi Das", "home_collection", {"patient_name": "Ravi Das"})
    await env.say("9876543210", "home_collection", {"phone": "9876543210"})
    await env.say("12 Lake Road, Kolkata", "home_collection", {"address_line": "12 Lake Road, Kolkata"})
    await env.say("yes", "unclear", {})  # address confirmed
    said = await env.say("no", "unclear", {})  # final confirm declined
    assert not env.tools.books
    assert env.tools.releases == ["hold-1"]
    assert env.session.home_collection is None


@pytest.mark.asyncio
async def test_an_address_that_names_a_different_pincode_is_never_silently_trusted(m, env):
    await reach_proceed(env)
    await env.say("yes", "unclear", {})
    await env.say("Ravi Das", "home_collection", {"patient_name": "Ravi Das"})
    await env.say("9876543210", "home_collection", {"phone": "9876543210"})
    said = await env.say(
        "12 Lake Road, 400001", "home_collection", {"address_line": "12 Lake Road, 400001"}
    )
    spoken = text_of(said)
    assert "400001" in spoken and "700091" in spoken
    assert env.session.home_collection.stage == "collect_address"  # asked again, not silently accepted
    assert env.session.home_collection.address_line == ""


@pytest.mark.asyncio
async def test_a_booked_patient_is_resolved_by_phone_without_naming_the_test_again(m, env):
    env.tools.booking_lookup_matches = [{"kind": "test_booking", "confirmation_id": "KCD-500"}]
    env.tools.booking_test_names_by_id["KCD-500"] = ["Uric Acid"]
    said = await env.say(
        "can my booked test be collected at home", "home_collection", {}
    )
    assert "already" in text_of(said).lower() or "বুক করা আছে" in text_of(said) or "रजिस्टर्ड" in text_of(said)
    said = await env.say("yes", "unclear", {})
    assert "phone" in text_of(said).lower() or "নম্বর" in text_of(said)
    said = await env.say("9876500000", "unclear", {})
    spoken = text_of(said)
    # The test name resolved from the booking is never re-spoken here -- the next real question is
    # simply the pincode, same as any other home_collection turn once test_names is known.
    assert "pincode" in spoken.lower() or "পিনকোড" in spoken
    assert env.session.home_collection.test_names == ["Uric Acid"]
    assert env.session.home_collection.from_booking is True


@pytest.mark.asyncio
async def test_multiple_offered_slots_require_an_explicit_choice_never_auto_picked(m, env):
    env.tools.slots_by_date["2026-10-05"] = [
        {"slot_id": 1, "date": "2026-10-05", "start_time": "08:00", "end_time": "10:00", "remaining_capacity": 2},
        {"slot_id": 2, "date": "2026-10-05", "start_time": "10:00", "end_time": "12:00", "remaining_capacity": 1},
    ]
    await env.say("Uric Acid test, pincode 700091", "home_collection", dict(SLOTS))
    said = await env.say("5th October", "home_collection", {"date": "2026-10-05"})
    spoken = text_of(said)
    assert "08:00" in spoken and "10:00" in spoken
    assert not env.tools.holds  # nothing is held until a specific slot is actually chosen
    said = await env.say("the second one", "unclear", {})
    assert env.tools.holds == [2]
    assert "350" in text_of(said) or "100" in text_of(said)


@pytest.mark.asyncio
async def test_a_corrected_pincode_after_eligibility_was_already_checked_releases_the_stale_hold(m, env):
    # 400001 is not in FakeTools.serviceable_postal_codes, so the correction also legitimately ends
    # the flow (the same honest "area not covered" outcome as test_an_uncovered_pincode_is_reported_
    # honestly...) -- what this test is actually proving is that the hold made against the OLD,
    # now-superseded pincode is freed immediately rather than left to its TTL.
    await reach_proceed(env)
    assert env.tools.holds == [1]
    said = await env.say("sorry, the pincode is actually 400001", "home_collection", {"postal_code": "400001"})
    assert env.tools.releases == ["hold-1"]  # the OLD hold, freed immediately -- never just left to expire
    assert "area" in text_of(said).lower() or "এলাকায়" in text_of(said)
    assert env.session.home_collection is None  # re-checked against the corrected pincode, honestly not serviceable
    assert not env.tools.books


@pytest.mark.asyncio
async def test_a_corrected_pincode_that_is_still_serviceable_re_quotes_for_the_new_area(m, env):
    await reach_proceed(env)
    assert env.tools.holds == [1]
    said = await env.say("sorry, the pincode is actually 700019", "home_collection", {"postal_code": "700019"})
    assert env.tools.releases == ["hold-1"]  # the stale hold against 700091 is freed
    spoken = text_of(said)
    assert "Uric Acid" in spoken  # re-checked and still eligible in the corrected area
    st = env.session.home_collection
    assert st is not None and st.postal_code == "700019"
    assert st.hold_token == "hold-2"  # a FRESH hold, not the stale one that was just released
    assert env.tools.holds == [1, 1]  # FakeTools' slot fixture is the same slot id for both test pincodes
