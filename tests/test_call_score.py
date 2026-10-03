"""The implicit happiness score: every call scored from how it went, with no question asked (agent/call_score.py),
recorded with the call (clinic-api), reported by language and cause (GET /api/v1/calls/satisfaction/summary).

    python -m pytest tests/test_call_score.py -v

The weights are REASONED, not measured: these tests pin the SHAPE (signs, bounds, ordering, determinism, no text stored),
not any claim that a 72 means 72% happy.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_orchestrator_booking_flow import book, env, m  # noqa: F401  (the harness: real dispatch, fake tools)

from agent.call_score import BASE_SCORE, MIN_TURNS_TO_SCORE, SLOW_REPLY_S, CallSignals, band_for, score_call

SAY = {
    "bn": "সিবিসি টেস্টের দাম কত",
    "hi": "सीबीसी टेस्ट का रेट कितना है",
    "en": "how much is the CBC",
}  # text in the call's script


def calls(**kw):
    s = CallSignals(turns=kw.pop("turns", 4))
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# ======================================================================================================== the scorer


def test_a_smooth_call_that_got_the_job_done_scores_high():
    r = score_call(calls(answers_given=2, caller_thanked=True, task_completed=True))
    assert r.score >= 90 and r.band == "happy" and r.basis == "behaviour"


def test_a_rough_call_scores_low_and_says_why():
    r = score_call(calls(reasks=3, slow_replies=3, abuse_turns=1, asked_for_person=True))
    assert r.score < 50 and r.band in ("unhappy", "very_unhappy")
    assert {name for name, _ in r.reasons} == {"reasks", "slow_replies", "abuse_turns", "asked_for_person"}
    assert r.reasons[0][1] <= r.reasons[-1][1] or abs(r.reasons[0][1]) >= abs(
        r.reasons[-1][1]
    )  # biggest movement first


def test_every_bad_signal_lowers_the_score_and_every_good_one_raises_it():
    base = score_call(calls()).score
    assert base == BASE_SCORE
    for bad in (
        "reasks",
        "system_failures",
        "abuse_turns",
        "silence_prompts",
        "language_flips",
        "barge_ins",
        "slow_replies",
        "corrections",
    ):
        assert score_call(calls(**{bad: 1})).score < base, bad
    for bad in ("asked_for_person", "system_handoff", "silence_timeout", "left_mid_task"):
        assert score_call(calls(**{bad: True})).score < base, bad
    for good in ("answers_given",):
        assert score_call(calls(**{good: 1})).score > base, good
    for good in ("task_completed", "caller_thanked", "ended_by_caller"):
        assert score_call(calls(**{good: True})).score > base, good


def test_one_signal_can_only_take_so_much_and_the_score_stays_within_0_and_100():
    assert score_call(calls(reasks=50)).score == BASE_SCORE - 24  # capped at its limit
    worst = score_call(
        calls(
            reasks=9,
            system_failures=9,
            abuse_turns=9,
            silence_prompts=9,
            language_flips=9,
            barge_ins=9,
            slow_replies=9,
            corrections=9,
            asked_for_person=True,
            system_handoff=True,
            silence_timeout=True,
            left_mid_task=True,
        )
    )
    best = score_call(calls(answers_given=99, task_completed=True, caller_thanked=True, ended_by_caller=True))
    assert 0 <= worst.score < 10 and 90 < best.score <= 100 and worst.band == "very_unhappy"


def test_the_same_signals_always_give_the_same_score_and_the_same_order_of_reasons():
    a = score_call(calls(reasks=2, barge_ins=2, slow_replies=2, caller_thanked=True))
    b = score_call(calls(slow_replies=2, barge_ins=2, caller_thanked=True, reasks=2))
    assert a == b


def test_a_call_too_short_or_an_emergency_is_not_scored():
    short = score_call(CallSignals(turns=MIN_TURNS_TO_SCORE - 1))
    assert short.score is None and short.band == "unscored" and short.reason == "too_short"
    emergency = score_call(calls(emergency=True))
    assert emergency.score is None and emergency.reason == "emergency"


def test_the_bands_cover_every_score():
    assert [band_for(s) for s in (100, 80, 79, 60, 59, 40, 39, 0)] == [
        "happy",
        "happy",
        "neutral",
        "neutral",
        "unhappy",
        "unhappy",
        "very_unhappy",
        "very_unhappy",
    ]


def test_a_reply_slower_than_the_limit_counts_and_a_quick_one_does_not():
    s = CallSignals()
    s.note_reply(SLOW_REPLY_S - 0.1)
    s.note_reply(SLOW_REPLY_S + 0.1)
    assert s.slow_replies == 1 and len(s.reply_ms) == 2


def test_an_unknown_signal_is_an_error_not_silently_ignored():
    with pytest.raises(AttributeError):
        CallSignals().note("not_a_signal")


def test_the_stored_payload_is_counts_and_flags_only():
    s = calls(reasks=2, caller_thanked=True)
    s.note_reply(0.4)
    s.note_reply(1.9)
    payload = score_call(s).to_payload(s)
    assert payload["basis"] == "behaviour" and payload["version"] == "implicit-v1"
    assert all(isinstance(v, (bool, int, type(None))) for v in payload["signals"].values())
    assert payload["signals"]["median_reply_ms"] == 1900 and "reply_ms" not in payload["signals"]
    json.dumps(payload)  # serialisable


# ============================================================================================ the wiring, real dispatch


@pytest.mark.asyncio
async def test_a_price_question_is_counted_as_an_answer_and_a_slow_one_is_noticed(m, env):
    s = env.session
    await env.say("how much is the CBC", "test_rate", {"test_name": "CBC"})
    assert s.signals.turns == 1 and s.signals.answers_given == 1 and len(s.signals.reply_ms) == 1


@pytest.mark.asyncio
async def test_thanks_from_the_caller_and_a_second_question_are_counted(m, env):
    s = env.session
    await env.say("how much is the CBC", "test_rate", {"test_name": "CBC"})
    await env.say("thank you", "smalltalk", {}, direct_reply_bn="You are welcome. Anything else?")
    assert s.signals.caller_thanked is True and s.signals.turns == 2


@pytest.mark.asyncio
async def test_swearing_is_counted_and_scored_down(m, env):
    s = env.session
    env.state["lang"] = "bn"
    await env.say(SAY["bn"], "test_rate", {"test_name": "CBC"})
    await env.say("তুই বাল")
    assert s.signals.abuse_turns == 1
    assert score_call(s.signals).score < score_call(CallSignals(turns=2, answers_given=1)).score


@pytest.mark.asyncio
async def test_a_reask_is_counted(m, env):
    s = env.session
    await env.say("")
    assert s.signals.reasks == 1


@pytest.mark.asyncio
async def test_asking_for_a_person_is_a_flag(m, env):
    s = env.session
    await env.say("connect me to a person")
    assert s.signals.asked_for_person is True


@pytest.mark.asyncio
async def test_a_language_moved_and_moved_back_is_counted_as_a_flip(m, env):
    s = env.session
    for lang in ("bn", "hi", "bn"):
        env.state["lang"] = lang
        await env.say(SAY[lang], "test_rate", {"test_name": "CBC"})
    assert s.signals.language_flips == 1
    env.state["lang"] = "hi"
    await env.say(SAY["hi"], "test_rate", {"test_name": "CBC"})  # bn hi bn hi: flips again
    assert s.signals.language_flips == 2


@pytest.mark.asyncio
async def test_a_real_change_of_language_is_not_a_flip(m, env):
    s = env.session
    for lang in ("bn", "hi", "hi", "hi"):
        env.state["lang"] = lang
        await env.say(SAY[lang], "test_rate", {"test_name": "CBC"})
    assert s.signals.language_flips == 0


@pytest.mark.asyncio
async def test_a_half_finished_booking_at_hang_up_is_marked_and_the_score_is_queued(m, env):
    s = env.session
    await book(env, doctor_name="Sen")
    await env.say("how much is the CBC", "test_rate", {"test_name": "CBC"})
    queued = []
    s.recorder = type("R", (), {"satisfaction": lambda self, p: queued.append(p), "flush": lambda self: None})()
    m._write_satisfaction(s)
    assert s.signals.left_mid_task is True and len(queued) == 1
    assert queued[0]["score"] is not None and queued[0]["basis"] == "behaviour"
    assert "left_mid_task" in [r[0] for r in queued[0]["reasons"]]


# ======================================================================================= the clinic API stores and reports


@pytest.fixture(scope="module")
def api():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        yield client


def _call(client, call_id, lang, payload=None, outcome="completed"):
    client.post(f"/api/v1/calls/{call_id}/events", json={"seq": 1, "kind": "start", "payload": {"channel": "voice"}})
    client.post(f"/api/v1/calls/{call_id}/events", json={"seq": 2, "kind": "language", "payload": {"language": lang}})
    if payload is not None:
        client.post(f"/api/v1/calls/{call_id}/events", json={"seq": 3, "kind": "satisfaction", "payload": payload})
    client.post(f"/api/v1/calls/{call_id}/events", json={"seq": 4, "kind": "end", "payload": {"outcome": outcome}})


def _payload(**kw):
    s = calls(**kw)
    return score_call(s).to_payload(s)


def test_a_score_is_stored_with_the_call_and_reported(api):
    _call(api, "sc-1", "bn", _payload(answers_given=2, caller_thanked=True))
    _call(api, "sc-2", "bn", _payload(reasks=3, slow_replies=4, abuse_turns=1))
    _call(api, "sc-3", "hi", _payload(answers_given=1))
    _call(api, "sc-4", "hi", score_call(CallSignals(turns=1)).to_payload(CallSignals(turns=1)))  # too short
    body = api.get("/api/v1/calls/satisfaction/summary").json()
    assert body["basis"] == "behaviour" and body["overall"]["calls"] >= 4 and body["overall"]["scored"] >= 3
    assert body["by_language"]["bn"]["scored"] == 2 and body["by_language"]["hi"]["scored"] == 1
    assert body["by_language"]["bn"]["mean"] is not None and body["unscored_reasons"].get("too_short", 0) >= 1
    assert {"reasks", "slow_replies", "abuse_turns"} <= {c["signal"] for c in body["top_causes"]}
    only_hi = api.get("/api/v1/calls/satisfaction/summary", params={"language": "hi"}).json()
    assert only_hi["overall"]["scored"] == 1


def test_a_retried_satisfaction_event_changes_nothing(api):
    _call(api, "sc-5", "bn", _payload(answers_given=1))
    before = api.get("/api/v1/calls/satisfaction/summary").json()["overall"]["scored"]
    api.post("/api/v1/calls/sc-5/events", json={"seq": 3, "kind": "satisfaction", "payload": _payload(reasks=9)})
    assert api.get("/api/v1/calls/satisfaction/summary").json()["overall"]["scored"] == before


def test_free_text_can_never_be_stored_through_the_score(api):
    payload = _payload(reasks=1)
    payload["signals"]["transcript"] = "my name is Ravi Das and my phone is 9830012345"
    payload["signals"]["notes"] = {"phone": "9830012345"}
    payload["reasons"].append(["a long sentence spoken by the caller " * 3, 5])
    _call(api, "sc-6", "bn", payload)
    from db import SessionLocal
    from models import CallRecord

    db = SessionLocal()
    try:
        raw = db.query(CallRecord).filter_by(call_id="sc-6").one().satisfaction_json
    finally:
        db.close()
    assert "Ravi" not in raw and "9830012345" not in raw and "transcript" in raw or "Ravi" not in raw
    assert "Ravi" not in raw and "9830012345" not in raw


def test_a_bad_date_is_a_422_not_a_crash(api):
    assert api.get("/api/v1/calls/satisfaction/summary", params={"since": "not-a-date"}).status_code == 422
