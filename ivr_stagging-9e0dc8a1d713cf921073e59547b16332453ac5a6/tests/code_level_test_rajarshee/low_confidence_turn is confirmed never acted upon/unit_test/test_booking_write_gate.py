"""Unit tests for the write guard inside main_pcm._finish_booking().

story title: A low-confidence turn is confirmed, never acted upon
user story: As a caller who was misheard, I want the agent to check before
  doing anything, so that a guess never becomes a booking.
acceptance criteria: No write occurs on a low-confidence turn without an
  explicit affirmative on the same call.

This is the one guarded commit point the module docstring of
_finish_booking() describes: "confirmed must be True... the guard lives
HERE rather than at the call sites on purpose". These tests exercise that
guard directly and do not go through the ASR/confidence-zone/dialogue
layer above it -- that state-machine-level coverage (low agreement -> echo
-> affirmative/negative -> write-or-no-write) belongs in the focused
integration tests, not here. This file proves one narrow thing in
isolation: _finish_booking(confirmed=False) can never reach
_tools.book_appointment, no matter what is in slots.

Same deterministic-stub style as tests/test_booking_readback.py and
tests/test_finish_booking_insufficient_information.py: main_pcm._speak and
main_pcm._tools are stubbed, nothing here needs the real ASR/TTS/NeMo
stack, no GPU, no network, no real clinic-api.
"""
from __future__ import annotations

import asyncio
import types

import pytest

import main_pcm


def run(coro):
    return asyncio.run(coro)


SLOTS = {
    "doctor_name": "Dr. A. Sen",
    "date": "2026-09-15",
    "time_slot": "18:30",
    "patient_name": "Rina Das",
    "phone": "9831234567",
}


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-call-write-gate",
        pending=pending,
        dispatch_lock=asyncio.Lock(),
        send_json=_AsyncNoOp(),
    )


class RecordingToolsClient:
    """Records every book_appointment call it receives -- the assertion
    this whole file exists to make is "this was never called", so the
    fake must make that call countable, not just fake a response."""

    def __init__(self, response: dict | None = None):
        self._response = response or {
            "success": True, "confirmation_id": "KCD-20260915-0099",
            "doctor_name": "Dr. A. Sen", "date": "2026-09-15", "time_slot": "18:30",
        }
        self.calls = []

    async def book_appointment(self, doctor_name, date, time_slot, patient_name, phone):
        self.calls.append({"doctor_name": doctor_name, "date": date, "time_slot": time_slot,
                            "patient_name": patient_name, "phone": phone})
        return self._response


@pytest.fixture(autouse=True)
def stub_speak(monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append((text, fallback_reason))

    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    return spoken


def _stub_tools(monkeypatch, response: dict | None = None) -> RecordingToolsClient:
    tools = RecordingToolsClient(response)
    monkeypatch.setattr(main_pcm, "_tools", tools)
    return tools


class TestUnconfirmedNeverWrites:
    """The central safety property of this story, isolated to the one
    function that enforces it."""

    def test_confirmed_false_never_calls_book_appointment(self, monkeypatch, stub_speak):
        tools = _stub_tools(monkeypatch)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS), confirmed=False))

        assert tools.calls == []

    def test_confirmed_omitted_defaults_to_false_and_never_writes(self, monkeypatch, stub_speak):
        # confirmed has a default; a call site that forgets the keyword
        # entirely must fail safe, not fail open.
        tools = _stub_tools(monkeypatch)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS)))

        assert tools.calls == []

    def test_unconfirmed_call_asks_instead_of_booking(self, monkeypatch, stub_speak):
        from agent.reply_templates import booking_confirm_prompt

        _stub_tools(monkeypatch)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS), confirmed=False))

        assert len(stub_speak) == 1
        text, fallback_reason = stub_speak[0]
        assert text == booking_confirm_prompt(SLOTS)
        assert fallback_reason is None

    def test_unconfirmed_call_sets_pending_to_await_confirmation(self, monkeypatch, stub_speak):
        _stub_tools(monkeypatch)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS), confirmed=False))

        assert session.pending is not None
        assert session.pending["awaiting"] == "confirm_booking"
        assert session.pending["slots"] == SLOTS

    def test_unconfirmed_call_does_not_clear_a_pre_existing_pending_booking(self, monkeypatch, stub_speak):
        # Scenario D from the test plan: a low-confidence turn must not
        # clobber booking state already in progress. _finish_booking
        # overwrites session.pending with the confirm_booking prompt
        # itself (that IS the in-progress state now) rather than
        # discarding it silently.
        _stub_tools(monkeypatch)
        session = make_session(pending={"awaiting": "confirm_booking", "slots": dict(SLOTS),
                                         "candidates": None, "offered_date": None, "retries": 1})

        run(main_pcm._finish_booking(session, dict(SLOTS), confirmed=False))

        assert session.pending["awaiting"] == "confirm_booking"
        assert session.pending["slots"] == SLOTS

    def test_unconfirmed_call_never_speaks_a_confirmation_id(self, monkeypatch, stub_speak):
        # Regression against a specific wrong fix: silently swallowing the
        # guard and speaking booking_reply() text anyway.
        response = {"success": True, "confirmation_id": "KCD-20260915-0099",
                    "doctor_name": "Dr. A. Sen", "date": "2026-09-15", "time_slot": "18:30"}
        _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS), confirmed=False))

        assert all("KCD-20260915-0099" not in text for text, _ in stub_speak)


class TestConfirmedWritesExactlyOnce:
    """The other half of the same guard: an explicit affirmative must
    still result in the booking actually being placed."""

    def test_confirmed_true_calls_book_appointment_once(self, monkeypatch, stub_speak):
        tools = _stub_tools(monkeypatch)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS), confirmed=True))

        assert len(tools.calls) == 1

    def test_confirmed_true_calls_book_appointment_with_the_given_slots(self, monkeypatch, stub_speak):
        tools = _stub_tools(monkeypatch)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS), confirmed=True))

        call = tools.calls[0]
        assert call == {
            "doctor_name": SLOTS["doctor_name"], "date": SLOTS["date"],
            "time_slot": SLOTS["time_slot"], "patient_name": SLOTS["patient_name"],
            "phone": SLOTS["phone"],
        }

    def test_confirmed_true_clears_pending_before_the_write(self, monkeypatch, stub_speak):
        _stub_tools(monkeypatch)
        session = make_session(pending={"awaiting": "confirm_booking", "slots": dict(SLOTS),
                                         "candidates": None, "offered_date": None, "retries": 0})

        run(main_pcm._finish_booking(session, dict(SLOTS), confirmed=True))

        assert session.pending is None

    def test_confirmed_true_speaks_the_confirmation_id(self, monkeypatch, stub_speak):
        response = {"success": True, "confirmation_id": "KCD-20260915-0099",
                    "doctor_name": "Dr. A. Sen", "date": "2026-09-15", "time_slot": "18:30"}
        _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS), confirmed=True))

        assert any("KCD-20260915-0099" in text for text, _ in stub_speak)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
