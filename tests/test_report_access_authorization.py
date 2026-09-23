"""ADDED BY SOURAV -- Story 5: "Caller asks about another person's
report" (privacy gateway enforcement).

story: As a patient, I want my results disclosed only to me or an
    authorised person, so that a phone line is not the weak point in my
    privacy.
acceptance criteria:
    1. Disclosure requires verification of the caller and an
       authorization check for that patient. Both must be enforced in
       code, not by the LLM.
    2. Unauthorized requests are declined plainly.
    3. Unauthorized attempts are audited.
    4. Report disclosure must happen only after the verification +
       authorization checks pass.
    5. Shared phone numbers should eventually surface a chooser instead
       of guessing (NOT implemented -- no documented policy for this
       exists; see the delivered inspection report).
    6. Authorized-person/proxy access should be supported if the
       existing product policy defines it (NOT implemented -- no such
       policy exists in this repo).

WHAT THIS STORY ACTUALLY BUILT, per the investigation done before this
file: this codebase's identity resolution (a caller-spoken phone number
matched against Patient.phone, which is DB-unique) and "authorization"
were the exact same accidental check, with no code path anywhere that
could express "verified, but not authorized." agent/report_access_
control.py adds that second, separate, code-only check
(authorize_report_access()), threaded through agent/report_flow.py's
existing interpret_*() functions via a "verified_patient_id" carried
forward on every pending dict in a report_status/report_send flow (see
that module's own docstrings). Under today's schema this check cannot
be triggered by any real caller utterance alone (Patient.phone is
unique, so there is exactly one patient per phone) -- these tests
exercise it directly, either by simulating an inconsistency that
SHOULD never happen naturally, or by tampering with a resumed pending
dict, to prove the gate actually fires, denies, and audits when it
does.

Structured on tests/test_complaint_flow.py's own pattern for the
ESCALATION_LOG_PATH isolation fixture (record_report_access_denied
writes to the exact same shared ledger record_complaint_filed() does).
"""
from __future__ import annotations

import asyncio
import json
import types

import pytest

import main_pcm
from agent.bn_normalize import detect_language
from agent.tools_client import ToolCallError
from agent.outcomes import (
    record_report_access_denied,
    report_access_denied_counts,
    REPORT_ACCESS_DENIED_EVENT,
    _reset_for_testing,
)
import agent.outcomes as outcomes_module
from agent.report_access_control import resolve_verified_patient_id, authorize_report_access
from agent.report_flow import (
    REPORT_ACCESS_DENIED_AWAITING,
    interpret_report_status_result,
    interpret_delivery_request_result,
    interpret_otp_verify_result,
)
from agent.reply_templates import report_access_denied_reply


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _isolated_escalation_log(monkeypatch, tmp_path):
    log_path = tmp_path / "escalations.jsonl"
    monkeypatch.setattr(outcomes_module, "ESCALATION_LOG_PATH", str(log_path))
    _reset_for_testing()
    yield log_path
    _reset_for_testing()


# --------------------------------------------------------------------- #
# agent/report_access_control.py -- the two pure gate functions in
# isolation, before anything about main.py's dispatch is involved.
# --------------------------------------------------------------------- #

class TestResolveVerifiedPatientId:
    def test_report_status_success_shape(self):
        assert resolve_verified_patient_id(
            {"patient_found": True, "found": True, "patient_id": 7}) == 7

    def test_delivery_or_otp_success_shape(self):
        assert resolve_verified_patient_id({"success": True, "patient_id": 7}) == 7

    def test_patient_not_found_returns_none(self):
        assert resolve_verified_patient_id({"patient_found": False}) is None

    def test_success_false_returns_none(self):
        assert resolve_verified_patient_id({"success": False, "reason": "OTP_INVALID"}) is None

    def test_missing_patient_id_on_an_otherwise_successful_shape_returns_none(self):
        # A defensive case, not one clinic-api can produce today (see
        # clinic-api/main.py -- patient_id is always set alongside
        # success=True) -- but this function must fail closed, not
        # silently succeed with a placeholder, if that ever changed.
        assert resolve_verified_patient_id({"success": True}) is None


class TestAuthorizeReportAccess:
    def test_true_when_both_ids_present_and_equal(self):
        assert authorize_report_access(7, 7) is True

    def test_false_when_ids_differ(self):
        assert authorize_report_access(7, 9) is False

    def test_false_when_verified_id_missing(self):
        assert authorize_report_access(None, 7) is False

    def test_false_when_target_id_missing(self):
        assert authorize_report_access(7, None) is False

    def test_false_when_both_missing(self):
        assert authorize_report_access(None, None) is False

    def test_never_given_anything_from_the_model(self):
        # Structural reminder, not a real code check: both arguments to
        # this function are always ints resolved by clinic-api, never a
        # string an LLM extracted. See TestModelCannotOverrideAuthorization
        # below for the actual dispatch-level proof.
        assert authorize_report_access("patient_name_from_llm", 7) is False


# --------------------------------------------------------------------- #
# agent/report_flow.py's interpret_*() functions -- the gate wired into
# the three points that ever build a spoken reply about a report.
# AC1/AC4: verification + authorization enforced in code, before
# disclosure, for all three.
# --------------------------------------------------------------------- #

class TestReportStatusResultDeniesOnIdentityMismatch:
    def test_denies_when_verified_and_resolved_ids_differ(self):
        # Simulates a which_report resumption where the identity this
        # flow was verified as (carried on the pending dict from an
        # earlier turn) no longer matches the patient the candidate just
        # chosen actually belongs to -- see main.py's "which_report"
        # continuation, which is the one real call site that can set
        # "verified_patient_id" to something other than this same
        # result's own "patient_id".
        result = {
            "patient_found": True, "found": True, "status": "READY",
            "delivery_enabled": True, "report_number": "RPT-X",
            "patient_id": 9, "verified_patient_id": 1,
        }
        text, pending = interpret_report_status_result(result, "report_status")
        assert text is None  # AC2/C: never report_status_reply() here.
        assert pending["awaiting"] == REPORT_ACCESS_DENIED_AWAITING
        assert pending["patient_id"] == 9

    def test_denies_before_ambiguous_candidates_are_ever_read_aloud(self):
        # AC4: disclosure only after both checks pass -- even the
        # (safe, test-name-only) "which report?" chooser must not be
        # reached on a mismatch.
        result = {
            "patient_found": True, "found": False, "reason": "AMBIGUOUS",
            "candidates": [{"report_number": "RPT-X", "test_name": "CBC", "patient_id": 9}],
            "patient_id": 9, "verified_patient_id": 1,
        }
        text, pending = interpret_report_status_result(result, "report_status")
        assert text is None
        assert pending["awaiting"] == REPORT_ACCESS_DENIED_AWAITING

    def test_established_fresh_when_no_verified_patient_id_is_given(self):
        # The FIRST turn of any report_status/report_send flow: nothing
        # to compare against yet, so this lookup's own patient_id
        # establishes verification -- not a denial.
        result = {
            "patient_found": True, "found": True, "status": "NOT_READY",
            "delivery_enabled": False, "report_number": "RPT-1", "patient_id": 3,
        }
        text, pending = interpret_report_status_result(result, "report_status")
        assert text  # the ordinary NOT_READY status reply, unimpeded.
        assert pending is None


class TestDeliveryRequestResultDeniesOnIdentityMismatch:
    def test_denies_when_verified_patient_id_does_not_match_response(self):
        result = {"success": True, "reason": "OTP_REQUIRED", "masked_phone": "xxxxx01", "patient_id": 9}
        text, pending = interpret_delivery_request_result(result, "RPT-A", verified_patient_id=1)
        assert text is None  # never otp_requested_reply() here.
        assert pending["awaiting"] == REPORT_ACCESS_DENIED_AWAITING
        assert pending["patient_id"] == 9

    def test_skips_the_check_entirely_when_verified_patient_id_not_given(self):
        # Backward compatibility: every pre-existing caller of this
        # function (before Story 5) never passed this parameter -- must
        # keep behaving exactly as before.
        result = {"success": True, "reason": "OTP_REQUIRED", "masked_phone": "xxxxx01", "patient_id": 9}
        text, pending = interpret_delivery_request_result(result, "RPT-A")
        assert text
        assert pending["awaiting"] == "otp_code"


class TestOtpVerifyResultDeniesOnIdentityMismatch:
    def test_denies_when_verified_patient_id_does_not_match_response(self):
        result = {"success": True, "reason": "DELIVERY_SENT", "masked_phone": "xxxxx01",
                  "signed_link_expires_minutes": 15, "patient_id": 9}
        text, pending = interpret_otp_verify_result(result, "RPT-A", verified_patient_id=1)
        assert text is None  # never otp_verify_reply()'s "delivery sent" text.
        assert pending["awaiting"] == REPORT_ACCESS_DENIED_AWAITING

    def test_otp_invalid_is_never_treated_as_a_mismatch(self):
        # A wrong OTP code is a VERIFICATION failure (RULE 8/9's own
        # territory), not this story's authorization gate -- must keep
        # re-prompting exactly as before, never divert into the
        # access-denied sentinel.
        result = {"success": False, "reason": "OTP_INVALID"}
        text, pending = interpret_otp_verify_result(result, "RPT-A", verified_patient_id=1)
        assert pending["awaiting"] == "otp_code"


# --------------------------------------------------------------------- #
# Dispatch-level: main_pcm.py's _apply_report_outcome() is the ONE place
# that turns the "__report_access_denied__" sentinel into (a) the fixed
# decline, never report content, and (b) a best-effort audit event.
# --------------------------------------------------------------------- #

class FakeToolsClient:
    def __init__(self):
        self.get_report_status_calls = []
        self.request_report_delivery_calls = []
        self.verify_report_otp_calls = []
        self.get_report_status_response = None
        self.request_report_delivery_response = None
        self.verify_report_otp_response = None

    async def get_report_status(self, phone, test_name=None):
        self.get_report_status_calls.append((phone, test_name))
        return self.get_report_status_response

    async def request_report_delivery(self, phone, report_number):
        self.request_report_delivery_calls.append((phone, report_number))
        return self.request_report_delivery_response

    async def verify_report_otp(self, phone, report_number, otp_code):
        self.verify_report_otp_calls.append((phone, report_number, otp_code))
        return self.verify_report_otp_response


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-call-1", pending=pending,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
        call_state=main_pcm.call_state_mod.build(), utt_seq=5,
        confirm_attempts=0,
    )


class FakeASRResult:
    text = "ignored -- _resolve_intent is stubbed directly"
    decoder_used = "ctc"
    decoder_agreement = 1.0
    ctc_words = 4
    rnnt_words = 4


class FakeASR:
    async def transcribe_utterance(self, wav_path):
        return FakeASRResult()


@pytest.fixture
def env(monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    tools = FakeToolsClient()
    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    monkeypatch.setattr(main_pcm, "_asr", FakeASR())
    monkeypatch.setattr(main_pcm, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools)


def dispatch_with_intent(monkeypatch, intent, slots, tmp_path):
    async def fake_resolve_intent(session, text):
        return {"intent": intent, "slots": slots}

    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
    session = make_session(pending=None)
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return session


def continue_pending(session, text):
    handled = run(main_pcm._continue_pending(session, text))
    assert handled is True
    return session


class TestGoldenPathStillWorks:
    """Item 1 / 8 / 9 of the testing list: a normal, fully verified and
    authorized caller is completely unaffected by this story -- every
    patient_id lines up at every stage, exactly as clinic-api's own
    RULE 15 already guarantees for a real call."""

    def test_verified_and_authorized_caller_gets_the_report_delivered(
        self, monkeypatch, env, tmp_path
    ):
        env.tools.get_report_status_response = {
            "patient_found": True, "found": True, "status": "READY",
            "delivery_enabled": True, "report_number": "RPT-A", "test_name": "CBC",
            "patient_id": 42,
        }
        session = dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": "CBC", "phone": "9000000001"}, tmp_path,
        )
        assert session.pending["awaiting"] == "confirm_delivery"
        assert session.pending["verified_patient_id"] == 42

        env.tools.request_report_delivery_response = {
            "success": True, "reason": "OTP_REQUIRED", "masked_phone": "xxxxx01", "patient_id": 42,
        }
        continue_pending(session, "হ্যাঁ")
        assert env.tools.request_report_delivery_calls == [("9000000001", "RPT-A")]
        assert session.pending["awaiting"] == "otp_code"
        assert session.pending["verified_patient_id"] == 42

        env.tools.verify_report_otp_response = {
            "success": True, "reason": "DELIVERY_SENT", "masked_phone": "xxxxx01",
            "signed_link_expires_minutes": 15, "patient_id": 42,
        }
        continue_pending(session, "123456")
        assert env.tools.verify_report_otp_calls == [("9000000001", "RPT-A", "123456")]
        assert session.pending is None
        assert "sent" in env.spoken[-1].lower() or "পাঠ" in env.spoken[-1]


class TestUnauthorizedRequestDoesNotDiscloseAnything:
    """Items 2-6 of the testing list."""

    def test_which_report_resumption_with_a_mismatched_identity_is_denied(
        self, monkeypatch, env, tmp_path
    ):
        # A caller's flow was verified/authorized as patient 1 (however
        # that got established); the pending "which_report" state's
        # candidate list somehow carries a DIFFERENT patient's report
        # (never producible by a real caller utterance under today's
        # unique-phone schema -- see agent/report_access_control.py's
        # own docstring -- exercised directly here to prove the gate
        # actually fires).
        session = make_session(pending={
            "awaiting": "which_report", "flow": "report_status",
            "candidates": [{"report_number": "RPT-X", "test_name": "CBC",
                             "status": "READY", "delivery_enabled": True, "patient_id": 9}],
            "phone": "9000000001", "retries": 0, "verified_patient_id": 1,
        })
        continue_pending(session, "CBC")

        # AC2/C: a clear, fixed decline -- never report_status_reply(),
        # never a hint that a report was even found.
        assert session.pending is None
        # detect_language("CBC") -> "english" (Latin text)
        assert env.spoken == [report_access_denied_reply(language="english")]
        assert "READY" not in env.spoken[-1]
        assert "CBC" not in env.spoken[-1]

        # AC1/4 (item 4 of the testing list): no further tool call was
        # ever made trying to deliver or verify OTP for this report.
        assert env.tools.request_report_delivery_calls == []
        assert env.tools.verify_report_otp_calls == []

        # AC3 (item 6): the denial was audited.
        assert report_access_denied_counts("report_status") == 1

    def test_confirm_delivery_yes_is_denied_when_the_delivery_response_identity_mismatches(
        self, monkeypatch, env, tmp_path
    ):
        session = make_session(pending={
            "awaiting": "confirm_delivery", "report_number": "RPT-A", "test_name": "CBC",
            "retries": 0, "phone": "9000000001", "verified_patient_id": 1,
        })
        # Simulates clinic-api ever returning an inconsistent patient_id
        # for this (phone, report) pair -- see the module docstring.
        env.tools.request_report_delivery_response = {
            "success": True, "reason": "OTP_REQUIRED", "masked_phone": "xxxxx01", "patient_id": 9,
        }
        continue_pending(session, "হ্যাঁ")

        assert session.pending is None
        assert env.spoken == [report_access_denied_reply(language="bengali")]
        # The delivery-request call itself already happened (clinic-api
        # needs to be asked in order to learn its response's identity at
        # all) -- but no OTP was ever requested from the caller, and
        # nothing about the report was spoken.
        assert env.tools.verify_report_otp_calls == []
        assert report_access_denied_counts("report_delivery") == 1

    def test_otp_verify_success_is_denied_when_the_response_identity_mismatches(
        self, monkeypatch, env, tmp_path
    ):
        session = make_session(pending={
            "awaiting": "otp_code", "report_number": "RPT-A", "retries": 0,
            "phone": "9000000001", "verified_patient_id": 1,
        })
        env.tools.verify_report_otp_response = {
            "success": True, "reason": "DELIVERY_SENT", "masked_phone": "xxxxx01",
            "signed_link_expires_minutes": 15, "patient_id": 9,
        }
        continue_pending(session, "123456")

        assert session.pending is None
        # detect_language("123456") -> "english"
        assert env.spoken == [report_access_denied_reply(language="english")]
        assert "sent" not in env.spoken[-1].lower()

    def test_audit_event_never_contains_report_content(self, monkeypatch, env, tmp_path, _isolated_escalation_log):
        session = make_session(pending={
            "awaiting": "which_report", "flow": "report_send",
            "candidates": [{"report_number": "RPT-SECRET", "test_name": "HIV Panel",
                             "status": "READY", "delivery_enabled": True, "patient_id": 9}],
            "phone": "9000000001", "retries": 0, "verified_patient_id": 1,
        })
        continue_pending(session, "HIV")

        lines = _isolated_escalation_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == REPORT_ACCESS_DENIED_EVENT
        assert record["intent"] == "report_send"
        assert record["reason"] == "VERIFIED_IDENTITY_MISMATCH"
        assert record["patient_id"] == 9
        assert record["call_id"] == "test-call-1"
        assert record["turn_index"] == 5
        # Never the report number, the test name, or any status word.
        blob = json.dumps(record)
        assert "RPT-SECRET" not in blob
        assert "HIV" not in blob
        assert "READY" not in blob

    def test_audit_write_failure_does_not_grant_access(self, monkeypatch, env, tmp_path):
        # "Do not make audit failure grant access" -- the decline and
        # pending=None must happen even if record_report_access_denied()
        # itself raises.
        def boom(*args, **kwargs):
            raise OSError("disk full, simulated")

        monkeypatch.setattr(main_pcm, "record_report_access_denied", boom)
        session = make_session(pending={
            "awaiting": "which_report", "flow": "report_status",
            "candidates": [{"report_number": "RPT-X", "test_name": "CBC",
                             "status": "READY", "delivery_enabled": True, "patient_id": 9}],
            "phone": "9000000001", "retries": 0, "verified_patient_id": 1,
        })
        continue_pending(session, "CBC")

        assert session.pending is None
        # detect_language("CBC") -> "english"
        assert env.spoken == [report_access_denied_reply(language="english")]

    def test_get_report_status_tool_failure_still_denies_access_not_grants_it(
        self, monkeypatch, env, tmp_path
    ):
        # A DIFFERENT failure mode, already pre-existing and unchanged
        # by this story: a clinic-api outage on the very first lookup
        # must never be silently treated as "found" or "authorized" --
        # confirms this story's gate did not weaken the pre-existing
        # ToolCallError handling.
        class FailingTools(FakeToolsClient):
            async def get_report_status(self, phone, test_name=None):
                raise ToolCallError("clinic-api unreachable")

        monkeypatch.setattr(main_pcm, "_tools", FailingTools())
        session = dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": "CBC", "phone": "9000000001"}, tmp_path,
        )
        assert session.pending is None
        assert len(env.spoken) == 1


class TestModelCannotOverrideAuthorization:
    """Item 7 of the testing list, and this story's AC2/AC1: the LLM's
    own "patient_name" slot extraction (agent/llm.py's _SLOT_KEYS) must
    never influence which patient's report gets resolved or disclosed."""

    def test_a_differently_named_patient_in_slots_has_no_effect_on_which_report_is_resolved(
        self, monkeypatch, env, tmp_path
    ):
        env.tools.get_report_status_response = {
            "patient_found": True, "found": True, "status": "NOT_READY",
            "delivery_enabled": False, "report_number": "RPT-A", "test_name": "CBC",
            "patient_id": 42,
        }
        # The model extracted a DIFFERENT patient's name alongside the
        # caller's own real, correctly-verified phone number -- dispatch
        # must resolve strictly by phone, exactly as before this story.
        session = dispatch_with_intent(
            monkeypatch, "report_status",
            {"test_name": "CBC", "phone": "9000000001", "patient_name": "someone else entirely"},
            tmp_path,
        )
        assert env.tools.get_report_status_calls == [("9000000001", "CBC")]
        assert session.pending is None  # NOT_READY -- ordinary status reply, not a denial.

    def test_report_status_and_report_send_dispatch_never_read_the_patient_name_slot(self):
        # Structural regression guard, same style as this codebase's own
        # test_report_phone_pending_state_does_not_collide_with_booking_
        # phone_correction (tests/test_report_status_send_dispatch.py):
        # asserts against the ACTUAL source text of the two dispatch
        # branches, not just today's observed behavior, so a future edit
        # that starts reading patient_name here fails a test immediately.
        import inspect
        source = inspect.getsource(main_pcm)
        start = source.index('elif intent == "report_status":')
        end = source.index('elif intent == "doctor_availability":')
        branch_source = source[start:end]
        assert "patient_name" not in branch_source


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
