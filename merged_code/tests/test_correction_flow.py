"""ADDED BY SOURAV -- "The agent accepts a correction and restates" story
(Epic: Answer Quality and Grounding).

story title: The agent accepts a correction and restates
user story: As a caller correcting the agent, I want the correction taken
    and confirmed, so that I am not arguing with a machine.
acceptance criteria: A correction updates the named value, is acknowledged
    explicitly and the corrected value is restated. The agent never
    defends a previous answer. Corrections are tested at every point in
    every flow.

See agent/correction_flow.py's own module docstring for the full gap
analysis against the pre-existing KCD-448 correction path (booking/
callback readback rejection -> "which field?" -> re-collect -> re-read).
Three real gaps closed by this story, each covered below:

  1. No acknowledgment -- TestCorrectionAcknowledgedReply, and the "ack is
     actually spoken" assertions throughout TestBookingCorrectionDispatch
     / TestCallbackCorrectionDispatch.
  2. Only reachable from the final readback -- TestSpontaneousCorrection*
     (mid-collection, caller corrects an earlier field while a LATER one
     is being asked) and the "direct correction, no 'no' needed" tests at
     confirm_booking/confirm_callback.
  3. The doctor dead end -- TestDoctorNameCorrection proves a doctor
     correction can now actually complete (previously: three retries,
     then the whole booking silently discarded, get_doctor_availability
     never even called -- see agent/correction_flow.py's own docstring
     point 3 and main.py's own "awaiting == doctor_name" branch comment).

Layout:
  - TestHasCorrectionCue / TestDetectBookingCorrection /
    TestDetectCallbackCorrection: pure functions, agent/correction_flow.py.
    No I/O, no main.py/main_pcm.py involved.
  - TestCorrectionAcknowledgedReply: pure template test,
    agent/reply_templates.py.
  - TestBookingCorrectionDispatch / TestCallbackCorrectionDispatch /
    TestDoctorNameCorrection: state-machine regression tests against
    main._continue_pending AND main_pcm._continue_pending, parametrized
    over both transports (same dual-transport discipline as
    tests/test_request_callback.py's own TestCallbackDispatch*).
  - TestTransportParity: the reasoning-half byte-identity check this
    story's implementation depended on throughout, formalized as an
    actual test rather than a one-off manual comparison.

No pytest-asyncio dependency, same reasoning as tests/test_booking_
readback.py's own docstring: async entry points are driven with plain
asyncio.run() inside ordinary sync test functions.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main  # noqa: E402
import main_pcm  # noqa: E402

from agent.bn_normalize import detect_language  # noqa: E402
from agent.correction_flow import (  # noqa: E402
    has_correction_cue, detect_booking_correction, detect_callback_correction,
)
from agent.reply_templates import (  # noqa: E402
    correction_acknowledged_reply, booking_confirmation_prompt,
    callback_confirmation_prompt, missing_slot_prompt, _spoken_doctor_name,
)
from agent.slot_parse import parse_date, parse_time, parse_phone  # noqa: E402
from agent.tools_client import ToolCallError  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# ======================================================================= #
# Part 1 -- agent/correction_flow.py, pure functions
# ======================================================================= #

class TestHasCorrectionCue:
    @pytest.mark.parametrize("text", [
        "actually, make it Friday",
        "no wait, that's wrong",
        "wait, no, not that one",
        "sorry i meant the other date",
        "change that to evening",
        "scratch that, use my other number",
        "nahi ruko, phone number alada",
        "mera matlab kuch aur tha",
        "asol e ami vul bolechi",
        "আসলে ভুল বলেছি",
        "না দাঁড়ান, ওটা ঠিক না",
        "সরি ভুল বললাম",
    ])
    def test_recognised_cues_across_every_supported_language(self, text):
        assert has_correction_cue(text) is True

    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "9123456789",
        "হ্যাঁ",
        "না",
        "18:30 হবে",
        "tomorrow morning please",
        "Dr. Sen is fine",
    ])
    def test_an_ordinary_answer_carries_no_cue(self, text):
        """The whole point of gating on a cue: a plain, correctly-targeted
        answer -- even one that happens to parse as some other field's
        type -- must never be mistaken for a correction. See this
        module's own docstring on why "does this parse as something
        else" is never used alone."""
        assert has_correction_cue(text) is False


_BOOKING_SLOTS = {
    "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
    "date": "2026-09-14", "time_slot": "18:30",
    "patient_name": "রিয়া দাস", "phone": "9876543210",
}

_BOOKING_PARSERS = {"date": parse_date, "time_slot": parse_time, "phone": parse_phone}


class TestDetectBookingCorrection:
    def test_no_cue_means_no_correction_even_though_it_would_parse(self):
        """A plain phone-shaped answer with no cue is never reinterpreted
        as a correction -- the "never guess" trust model this module's
        docstring insists on."""
        result = detect_booking_correction(
            "my phone number is 9123456789", _BOOKING_SLOTS, "time_slot", _BOOKING_PARSERS)
        assert result is None

    def test_cue_plus_a_confidently_parsing_other_field_is_a_correction(self):
        result = detect_booking_correction(
            "actually, make it 2026-09-20", _BOOKING_SLOTS, "time_slot", _BOOKING_PARSERS)
        assert result == ("date", "2026-09-20")

    def test_the_currently_awaited_field_is_never_matched_against_itself(self):
        """`awaiting` is excluded on purpose -- a plain answer to the
        field currently on the table is left to the call site's own
        normal parsing, never treated as "a correction to itself"."""
        result = detect_booking_correction(
            "actually, 2026-09-20", _BOOKING_SLOTS, "date", _BOOKING_PARSERS)
        assert result is None

    def test_a_field_not_yet_known_cannot_be_corrected(self):
        slots = dict(_BOOKING_SLOTS)
        del slots["phone"]
        result = detect_booking_correction(
            "actually my phone is 9123456789", slots, "time_slot", _BOOKING_PARSERS)
        assert result is None

    def test_doctor_name_is_never_auto_detected_even_with_a_parser_supplied(self):
        """Deliberate scope exclusion (module docstring point 1): doctor
        validation needs an async catalogue call a pure detector cannot
        make. Even a caller-supplied parsers dict that DOES map
        doctor_name must never be consulted -- _BOOKING_SPONTANEOUS_FIELDS
        itself never includes it."""
        parsers = dict(_BOOKING_PARSERS, doctor_name=lambda t: "Dr. R Gupta")
        result = detect_booking_correction(
            "actually the doctor should be Dr. Gupta", _BOOKING_SLOTS, "time_slot", parsers)
        assert result is None

    def test_patient_name_is_never_auto_detected_even_with_a_parser_supplied(self):
        """Deliberate scope exclusion (module docstring point 2): free
        text with no grammar to validate a spontaneous correction
        against -- see the real bug this caught, documented at length in
        this module's own docstring and in tests/test_booking_readback.py."""
        parsers = dict(_BOOKING_PARSERS, patient_name=lambda t: (t.strip() or None))
        result = detect_booking_correction(
            "actually the patient name is Rina Das", _BOOKING_SLOTS, "time_slot", parsers)
        assert result is None

    def test_a_missing_parser_for_a_spontaneous_field_is_tolerated(self):
        """A caller can pass a parsers dict that leaves a field out
        entirely (e.g. _booking_correction_parsers() is never asked to
        supply one for a field it does not cover) -- this must not raise."""
        result = detect_booking_correction(
            "actually, make it 2026-09-20", _BOOKING_SLOTS, "time_slot", {})
        assert result is None


_CALLBACK_SLOTS = {"callback_time_window": "evening", "phone": "9876543210"}
_CALLBACK_PARSERS = {"phone": parse_phone}


class TestDetectCallbackCorrection:
    def test_no_cue_means_no_correction(self):
        result = detect_callback_correction(
            "my phone number is 9123456789", _CALLBACK_SLOTS,
            "callback_time_window", _CALLBACK_PARSERS)
        assert result is None

    def test_cue_plus_a_confidently_parsing_phone_is_a_correction(self):
        result = detect_callback_correction(
            "actually, my phone number is 9123456789", _CALLBACK_SLOTS,
            "callback_time_window", _CALLBACK_PARSERS)
        assert result == ("phone", "9123456789")

    def test_the_currently_awaited_callback_phone_state_is_excluded(self):
        result = detect_callback_correction(
            "actually, 9123456789", _CALLBACK_SLOTS, "callback_phone", _CALLBACK_PARSERS)
        assert result is None

    def test_a_field_not_yet_known_cannot_be_corrected(self):
        slots = dict(_CALLBACK_SLOTS)
        del slots["phone"]
        result = detect_callback_correction(
            "actually my phone is 9123456789", slots, "callback_time_window", _CALLBACK_PARSERS)
        assert result is None

    def test_callback_time_window_is_never_auto_detected_even_with_a_parser_supplied(self):
        """Same deliberate scope exclusion as booking's patient_name --
        free text, no grammar, and tests/test_request_callback.py's own
        pre-existing test caught the real bug an earlier version of this
        module had here."""
        parsers = dict(_CALLBACK_PARSERS, callback_time_window=lambda t: (t.strip() or None))
        result = detect_callback_correction(
            "actually the time window is tomorrow evening", _CALLBACK_SLOTS,
            "callback_phone", parsers)
        assert result is None


# ======================================================================= #
# Part 2 -- agent/reply_templates.py::correction_acknowledged_reply
# ======================================================================= #

_ACK_SLOTS = {
    "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
    "date": "2026-09-20", "time_slot": "19:00",
    "patient_name": "Rina Das", "phone": "9123456789",
    "callback_time_window": "tomorrow evening",
}

_ALL_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]
_NON_DOCTOR_FIELDS = ["date", "time_slot", "patient_name", "phone", "callback_time_window"]

# Defensive phrasing this story's AC explicitly forbids ("the agent never
# defends a previous answer") -- if any of these ever show up in an
# acknowledgment, the agent is arguing about who was right, not just
# updating the value.
_DEFENSIVE_MARKERS_BY_LANGUAGE = {
    "english": ("but you said", "already told", "i already said", "you told me"),
    "hinglish": ("aapne bola tha", "pehle bola"),
    "banglish": ("apni bolechilen", "age bolechi"),
    "bengali": ("আপনি বলেছিলেন", "আগে বলেছি", "কিন্তু আপনি"),
}


class TestCorrectionAcknowledgedReply:
    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    @pytest.mark.parametrize("field", _NON_DOCTOR_FIELDS)
    def test_the_new_value_is_restated_verbatim_from_slots(self, language, field):
        reply = correction_acknowledged_reply(field, _ACK_SLOTS, language=language)
        assert str(_ACK_SLOTS[field]) in reply

    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    def test_the_doctor_field_uses_the_spoken_honorific(self, language):
        reply = correction_acknowledged_reply("doctor_name", _ACK_SLOTS, language=language)
        expected_name = _spoken_doctor_name(_ACK_SLOTS, {}, language=language)
        assert expected_name in reply

    def test_the_value_reflects_whatever_is_currently_in_slots_not_a_fixed_string(self):
        """Never a second "value" parameter -- correction_acknowledged_
        reply reads straight off `slots`, so a different phone number in
        slots must produce a different acknowledgment."""
        reply_a = correction_acknowledged_reply(
            "phone", {"phone": "9111111111"}, language="english")
        reply_b = correction_acknowledged_reply(
            "phone", {"phone": "9222222222"}, language="english")
        assert "9111111111" in reply_a
        assert "9222222222" in reply_b
        assert reply_a != reply_b

    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    @pytest.mark.parametrize("field", ["doctor_name"] + _NON_DOCTOR_FIELDS)
    def test_never_defends_a_previous_answer(self, language, field):
        reply = correction_acknowledged_reply(field, _ACK_SLOTS, language=language)
        lowered = reply.lower()
        for marker in _DEFENSIVE_MARKERS_BY_LANGUAGE[language]:
            assert marker.lower() not in lowered, (
                f"{marker!r} reads as defending the earlier answer: {reply!r}")

    def test_it_is_sayable_in_bengali(self):
        # UPDATED BY SOURAV -- agent/speakability.py's check() only ever
        # models Bengali (MODELLED_LANGUAGE = "bn"); passed any other
        # `language` string it reports UNCHECKED rather than a verdict it
        # did not earn (see that function's own docstring), and called
        # with none at all (the no-argument form tests/test_booking_
        # readback.py's own test_the_correction_prompt_is_sayable already
        # uses) it assumes the text IS Bengali. So this only meaningfully
        # checks the bengali branch -- same scope the pre-existing
        # sayability test in this codebase covers, not a gap this test
        # introduces. The phone field is picked because it is the one
        # that spells digits into Bengali words (verbalize()), the exact
        # thing this gate exists to catch a regression in.
        from agent import speakability
        reply = correction_acknowledged_reply("phone", _ACK_SLOTS, language="bengali")
        assert speakability.check(reply).state == speakability.SPEAKABLE


# ======================================================================= #
# Part 3 -- dispatch-level, both transports
# ======================================================================= #

TRANSPORTS = [main, main_pcm]


@pytest.fixture(params=TRANSPORTS, ids=["main", "main_pcm"])
def transport(request):
    return request.param


class _Session:
    call_id = "cf01"
    utt_seq = 1

    def __init__(self, call_state_mod, awaiting, slots):
        self.call_state = call_state_mod.build()
        self.said: list[str] = []
        self.pending = {"awaiting": awaiting, "slots": dict(slots),
                         "candidates": None, "offered_date": None, "retries": 0}


async def _say(session, text, fallback_reason=None):
    session.said.append(text)


class _FakeDoctorTools:
    """Only get_doctor_availability -- the one clinic-api call the new
    "awaiting == doctor_name" branch makes. `responses` lets a test queue
    up a not-found attempt followed by a found one, proving the retry
    ladder actually calls the tool again rather than looping on a cached
    failure."""

    def __init__(self, responses=None):
        self.calls: list[tuple[str, str]] = []
        self._responses = list(responses) if responses is not None else None
        self._default = {"found": True, "doctor_name": "Dr. R Gupta", "doctor_name_bn": "গুপ্ত"}
        self._error = None

    async def get_doctor_availability(self, doctor_name, date):
        self.calls.append((doctor_name, date))
        if self._error:
            raise self._error
        if self._responses is not None:
            return self._responses.pop(0)
        return self._default


@pytest.fixture
def booking(transport, monkeypatch):
    """Wires _speak and _finish_booking as spies and _tools as a fake
    doctor-lookup client, mirroring tests/test_booking_readback.py's own
    `wired` fixture but parametrized over both transports."""
    written: list[dict] = []

    async def _finish(session, slots, *, confirmed=False, language="bengali"):
        written.append({"slots": dict(slots), "confirmed": confirmed, "language": language})
        session.pending = None

    monkeypatch.setattr(transport, "_speak", _say)
    monkeypatch.setattr(transport, "_finish_booking", _finish)
    tools = _FakeDoctorTools()
    monkeypatch.setattr(transport, "_tools", tools)
    return types.SimpleNamespace(transport=transport, written=written, tools=tools)


def _bturn(booking, session, text):
    return run(booking.transport._continue_pending(session, text))


class TestSpontaneousBookingCorrection:
    """Gap 2 from agent/correction_flow.py's docstring: a correction to
    an EARLIER field, made while a LATER one is being asked, used to have
    no path at all. This is "at every point in every flow", not only at
    the final readback."""

    def test_correcting_an_earlier_field_while_a_later_one_is_being_asked(self, booking):
        # date is already known; time_slot is what's currently awaited.
        slots = {"doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
                  "date": "2026-09-14", "phone": "9876543210"}
        session = _Session(booking.transport.call_state_mod, "time_slot", slots)

        assert _bturn(booking, session, "actually, make it 2026-09-20")

        assert session.pending["slots"]["date"] == "2026-09-20"
        # Still awaiting time_slot -- the question on the table is not
        # dropped, only answered in the same breath as the acknowledgment.
        assert session.pending["awaiting"] == "time_slot"
        language = detect_language("actually, make it 2026-09-20")
        expected_ack = correction_acknowledged_reply("date", session.pending["slots"], language=language)
        assert session.said[-1] == (
            f"{expected_ack} {missing_slot_prompt('book_appointment', 'time_slot', language=language)}")
        assert booking.written == []

    def test_a_cue_on_the_currently_awaited_field_still_answers_it_normally(self, booking):
        """The cue-plus-value combination on the field ALREADY being
        asked is not "a correction to itself" -- it just falls through to
        ordinary parsing of that same field, unchanged behaviour."""
        slots = {"doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
                  "date": "2026-09-14", "time_slot": "18:30", "patient_name": "রিয়া দাস"}
        session = _Session(booking.transport.call_state_mod, "phone", slots)

        assert _bturn(booking, session, "actually 9123456789")
        assert session.pending["slots"]["phone"] == "9123456789"
        assert session.pending["awaiting"] == "confirm_booking"
        # No acknowledgment prefix -- this was never opened as a correction
        # (_correcting_field was never set for a first-time collection).
        language = detect_language("actually 9123456789")
        assert session.said[-1] == booking_confirmation_prompt(session.pending["slots"], language=language)


class TestBookingCorrectionDispatch:
    _FULL = {
        "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
        "date": "2026-09-14", "time_slot": "18:30",
        "patient_name": "রিয়া দাস", "phone": "9876543210",
    }

    def test_direct_correction_from_the_readback_needs_no_no_first(self, booking):
        """Gap 2's other half: the caller need not say "না" before
        correcting -- hearing the readback, they can just correct it
        outright."""
        session = _Session(booking.transport.call_state_mod, "confirm_booking", self._FULL)
        assert _bturn(booking, session, "actually my phone number is 9123456789")

        assert session.pending["slots"]["phone"] == "9123456789"
        assert session.pending["awaiting"] == "confirm_booking"
        language = detect_language("actually my phone number is 9123456789")
        expected_ack = correction_acknowledged_reply("phone", session.pending["slots"], language=language)
        assert session.said[-1] == (
            f"{expected_ack} {booking_confirmation_prompt(session.pending['slots'], language=language)}")
        assert booking.written == [], "a correction from the readback must never write by itself"

    def test_field_and_value_in_one_breath_at_the_correction_menu(self, booking):
        """Gap 1: no second round trip needed when the caller names the
        field AND gives its new value in the same utterance."""
        session = _Session(booking.transport.call_state_mod, "confirm_correction", self._FULL)
        assert _bturn(booking, session, "ফোন নম্বরটা আসলে ৯১২৩৪৫৬৭৮৯")

        assert session.pending["slots"]["phone"] == "9123456789"
        assert session.pending["awaiting"] == "confirm_booking"
        language = detect_language("ফোন নম্বরটা আসলে ৯১২৩৪৫৬৭৮৯")
        expected_ack = correction_acknowledged_reply("phone", session.pending["slots"], language=language)
        assert session.said[-1] == (
            f"{expected_ack} {booking_confirmation_prompt(session.pending['slots'], language=language)}")
        assert booking.written == []

    def test_naming_the_field_alone_still_falls_back_to_the_two_step_path(self, booking):
        """patient_name has no parser in _booking_correction_parsers(), so
        naming it alone (no value given yet) must NOT be treated as
        "field and value in one breath" -- it opens the pre-existing
        two-step re-collection instead, marked so the acknowledgment
        fires once the value actually arrives."""
        session = _Session(booking.transport.call_state_mod, "confirm_correction", self._FULL)
        assert _bturn(booking, session, "রোগীর নাম")

        assert session.pending["awaiting"] == "patient_name"
        assert session.pending["_correcting_field"] == "patient_name"
        assert "patient_name" not in session.pending["slots"]
        assert booking.written == []

        # The second turn supplies the new name -- only NOW is it acknowledged.
        assert _bturn(booking, session, "রিনা দাস")
        assert session.pending["slots"]["patient_name"] == "রিনা দাস"
        assert session.pending["awaiting"] == "confirm_booking"
        language = detect_language("রিনা দাস")
        expected_ack = correction_acknowledged_reply("patient_name", session.pending["slots"], language=language)
        assert session.said[-1] == (
            f"{expected_ack} {booking_confirmation_prompt(session.pending['slots'], language=language)}")


class TestDoctorNameCorrection:
    """Gap 3, the real dead end this story fixes: naming "doctor" via the
    correction menu used to lead nowhere -- no branch ever matched
    `awaiting == "doctor_name"`, so get_doctor_availability was never
    even called and the caller's request to fix the doctor could not be
    honoured. See agent/correction_flow.py's own module docstring point 3
    and main.py/main_pcm.py's own `awaiting == "doctor_name"` branch."""

    _FULL = {
        "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
        "date": "2026-09-14", "time_slot": "18:30",
        "patient_name": "রিয়া দাস", "phone": "9876543210",
    }

    def test_a_doctor_correction_can_now_actually_complete(self, booking):
        session = _Session(booking.transport.call_state_mod, "confirm_correction", self._FULL)
        assert _bturn(booking, session, "ডাক্তারের নাম ভুল")
        assert session.pending["awaiting"] == "doctor_name"
        assert session.pending["_correcting_field"] == "doctor_name"
        assert booking.tools.calls == [], "no lookup yet -- only the field was named"

        assert _bturn(booking, session, "Dr. Gupta")
        # THE FIX: the catalogue was actually consulted.
        assert len(booking.tools.calls) == 1
        assert session.pending["slots"]["doctor_name"] == "Dr. R Gupta"
        assert session.pending["slots"]["doctor_name_bn"] == "গুপ্ত"
        assert session.pending["awaiting"] == "confirm_booking"
        language = detect_language("Dr. Gupta")
        expected_ack = correction_acknowledged_reply("doctor_name", session.pending["slots"], language=language)
        assert session.said[-1] == (
            f"{expected_ack} {booking_confirmation_prompt(session.pending['slots'], language=language)}")
        assert booking.written == [], "still needs the final yes"

    def test_an_unresolved_doctor_name_retries_rather_than_silently_failing(self, booking):
        booking.tools._responses = [{"found": False}, {"found": False}]
        session = _Session(booking.transport.call_state_mod, "doctor_name", dict(self._FULL))
        session.pending["_correcting_field"] = "doctor_name"

        assert _bturn(booking, session, "Dr. Nobody")
        assert session.pending is not None, "one miss is retried, not abandoned"
        assert session.pending["awaiting"] == "doctor_name"

        assert _bturn(booking, session, "Dr. Nobody")
        assert len(booking.tools.calls) == 2

    def test_giving_up_after_repeated_misses_never_writes(self, booking):
        booking.tools._responses = [{"found": False}] * 5
        session = _Session(booking.transport.call_state_mod, "doctor_name", dict(self._FULL))
        session.pending["_correcting_field"] = "doctor_name"

        handled = True
        for _ in range(4):
            handled = _bturn(booking, session, "Dr. Nobody")
        assert handled is False
        assert session.pending is None
        assert booking.written == []


# --------------------------------------------------------------------- #
# Callback flow -- same story, scoped to request_callback's two fields.
# --------------------------------------------------------------------- #

@pytest.fixture
def callback(transport, monkeypatch):
    monkeypatch.setattr(transport, "_speak", _say)
    return types.SimpleNamespace(transport=transport)


def _cturn(callback, session, text):
    return run(callback.transport._continue_pending(session, text))


class TestCallbackCorrectionDispatch:
    _FULL = {"callback_time_window": "evening", "phone": "9876543210"}

    def test_spontaneous_correction_of_an_earlier_field_mid_collection(self, callback):
        # phone already known; callback_time_window is what's awaited.
        session = _Session(callback.transport.call_state_mod, "callback_time_window",
                            {"phone": "9876543210"})
        assert _cturn(callback, session, "actually, my phone number is 9123456789")

        assert session.pending["slots"]["phone"] == "9123456789"
        assert session.pending["awaiting"] == "callback_time_window"
        language = detect_language("actually, my phone number is 9123456789")
        expected_ack = correction_acknowledged_reply("phone", session.pending["slots"], language=language)
        assert session.said[-1] == (
            f"{expected_ack} {missing_slot_prompt('request_callback', 'callback_time_window', language=language)}")

    def test_direct_correction_from_the_readback_needs_no_no_first(self, callback):
        session = _Session(callback.transport.call_state_mod, "confirm_callback", self._FULL)
        assert _cturn(callback, session, "actually my phone number is 9123456789")

        assert session.pending["slots"]["phone"] == "9123456789"
        assert session.pending["awaiting"] == "confirm_callback"
        language = detect_language("actually my phone number is 9123456789")
        expected_ack = correction_acknowledged_reply("phone", session.pending["slots"], language=language)
        assert session.said[-1] == (
            f"{expected_ack} {callback_confirmation_prompt(session.pending['slots'], language=language)}")

    def test_field_and_value_in_one_breath_at_the_correction_menu(self, callback):
        session = _Session(callback.transport.call_state_mod, "confirm_callback_correction", self._FULL)
        assert _cturn(callback, session, "ফোন নম্বরটা আসলে ৯১২৩৪৫৬৭৮৯")

        assert session.pending["slots"]["phone"] == "9123456789"
        assert session.pending["awaiting"] == "confirm_callback"
        language = detect_language("ফোন নম্বরটা আসলে ৯১২৩৪৫৬৭৮৯")
        expected_ack = correction_acknowledged_reply("phone", session.pending["slots"], language=language)
        assert session.said[-1] == (
            f"{expected_ack} {callback_confirmation_prompt(session.pending['slots'], language=language)}")

    def test_naming_time_window_alone_still_falls_back_to_the_two_step_path(self, callback):
        """callback_time_window has no parser in _callback_correction_
        parsers() -- naming it alone must not be misread as also
        supplying the new value (the real bug this module's docstring
        documents catching)."""
        session = _Session(callback.transport.call_state_mod, "confirm_callback_correction", self._FULL)
        assert _cturn(callback, session, "সময়টা ঠিক না")

        assert session.pending["awaiting"] == "callback_time_window"
        assert session.pending["_correcting_field"] == "callback_time_window"
        assert "callback_time_window" not in session.pending["slots"]

        assert _cturn(callback, session, "tomorrow morning")
        assert session.pending["slots"]["callback_time_window"] == "tomorrow morning"
        assert session.pending["awaiting"] == "confirm_callback"
        language = detect_language("tomorrow morning")
        expected_ack = correction_acknowledged_reply(
            "callback_time_window", session.pending["slots"], language=language)
        assert session.said[-1] == (
            f"{expected_ack} {callback_confirmation_prompt(session.pending['slots'], language=language)}")


# ======================================================================= #
# Part 4 -- transport parity
# ======================================================================= #

class TestTransportParity:
    def test_the_reasoning_half_is_byte_identical_across_both_transports(self):
        """main.py and main_pcm.py each keep an independently-maintained
        copy of _continue_pending (WAV vs raw PCM transports over
        otherwise-identical turn logic) -- see agent/correction_flow.py's
        own module docstring on why the pure decisions were pulled out
        into that module instead of being written twice. This pins the
        actual textual identity of the shared "reasoning half" so the two
        copies cannot silently drift apart in a future edit."""
        import inspect
        sa = inspect.getsource(main._resolve_intent)
        sb = inspect.getsource(main_pcm._resolve_intent)
        # The reasoning half runs from _resolve_intent through the end of
        # _continue_pending, up to (not including) _resync_after_playback
        # -- same boundary this story's implementation was built and
        # checked against throughout. Comparing the whole source file
        # slice between those two markers is more robust to future
        # additions than comparing a single function.
        start_marker = "async def _resolve_intent("
        end_marker = "async def _resync_after_playback("
        src_a = inspect.getsource(main)
        src_b = inspect.getsource(main_pcm)
        a = src_a[src_a.index(start_marker):src_a.index(end_marker)]
        b = src_b[src_b.index(start_marker):src_b.index(end_marker)]
        assert a == b, "main.py and main_pcm.py's reasoning halves have drifted apart"

    def test_a_spontaneous_correction_produces_the_same_reply_on_both_transports(self, monkeypatch):
        results = {}
        for transport in TRANSPORTS:
            spoken = []

            async def fake_speak(session, text, fallback_reason=None, _spoken=spoken):
                _spoken.append(text)

            monkeypatch.setattr(transport, "_speak", fake_speak)
            slots = {"doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
                      "date": "2026-09-14", "phone": "9876543210"}
            session = _Session(transport.call_state_mod, "time_slot", slots)
            run(transport._continue_pending(session, "actually, make it 2026-09-20"))
            results[transport.__name__] = spoken[-1]

        assert results["main"] == results["main_pcm"]
