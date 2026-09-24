"""ADDED BY SOURAV -- "Caller states something the agent cannot verify"
story.

story title: Caller states something the agent cannot verify
user story: As a caller asserting a fact about my record, I want the
    agent to check rather than accept it, so that a mistake is not
    compounded.
acceptance criteria: A caller assertion never becomes a system fact. The
    agent checks the system of record where verification is possible.
    Where the system cannot verify the claim, the agent clearly says it
    cannot confirm it and offers a human. A caller claim is never echoed
    back as though it were verified.

Structured the same way tests/test_symptom_routing.py is (this story's
own closest sibling, per the delivered investigation report and the
implementation instructions that followed it): resolver-level checks, a
guard-order/short-circuit structural proof, a reply-template safety
suite, a dispatch-level check with fake collaborators, a pending-flow
interrupt check, a pending-choice (offer/decline) check, an audit-event
check, a "verifiable flow still uses the real DB result" regression
check, and transport parity.

Per this story's own instruction ("Do not run unnecessary broad tests
before the focused suite"), this file is meant to be run on its own
first, before the wider regression list the final report enumerates.
"""
from __future__ import annotations

import asyncio
import json
import types

import pytest

import main_pcm
from agent.unverifiable_claim import (
    detect_unverifiable_claim,
    NONE as _NONE,
    UNVERIFIABLE,
    CATEGORY_APPOINTMENT_EXISTING,
    CATEGORY_REPORT_RESULT,
    CATEGORY_DOCTOR_APPROVAL,
    CATEGORY_PAYMENT_ALREADY,
)
from agent.reply_templates import unverifiable_claim_reply, human_fallback_reply, out_of_scope_counter_reply
from agent.slot_parse import is_affirmative, is_negative
import agent.outcomes as outcomes_module
from agent.outcomes import (
    record_unverifiable_claim,
    unverifiable_claim_counts,
    UNVERIFIABLE_CLAIM_EVENT,
    _reset_for_testing,
)

_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]


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
# agent/unverifiable_claim.py -- the deterministic detector itself
# --------------------------------------------------------------------- #

class TestDetectUnverifiableClaimPositives:
    """The 5 example claims from the implementation instruction, across
    English / Hindi-Hinglish / Bengali-Banglish (romanised) / Bengali
    script -- the exact four-way coverage this story's instruction 1
    requires."""

    @pytest.mark.parametrize("text", [
        "my appointment is tomorrow",
        "mera appointment kal hai",
        "amar appointment kal ache",
        "আমার অ্যাপয়েন্টমেন্ট কাল আছে",
    ])
    def test_appointment_is_tomorrow(self, text):
        verdict, category = detect_unverifiable_claim(text)
        assert verdict == UNVERIFIABLE
        assert category == CATEGORY_APPOINTMENT_EXISTING

    @pytest.mark.parametrize("text", [
        "my report is normal",
        "mera report normal hai",
        "amar report normal ache",
        "আমার রিপোর্ট নরমাল",
    ])
    def test_report_is_normal(self, text):
        verdict, category = detect_unverifiable_claim(text)
        assert verdict == UNVERIFIABLE
        assert category == CATEGORY_REPORT_RESULT

    @pytest.mark.parametrize("text", [
        "the doctor already approved it",
        "doctor ne already approve kar diya",
        "doctor already approve kore diyechen",
        "ডাক্তার আগে থেকেই অনুমোদন দিয়েছেন",
    ])
    def test_doctor_already_approved(self, text):
        verdict, category = detect_unverifiable_claim(text)
        assert verdict == UNVERIFIABLE
        assert category == CATEGORY_DOCTOR_APPROVAL

    @pytest.mark.parametrize("text", [
        "i already paid",
        "maine already payment kar diya hai",
        "ami already payment kore diyechi",
        "আমি আগেই পেমেন্ট করে দিয়েছি",
    ])
    def test_i_already_paid(self, text):
        verdict, category = detect_unverifiable_claim(text)
        assert verdict == UNVERIFIABLE
        assert category == CATEGORY_PAYMENT_ALREADY

    @pytest.mark.parametrize("text", [
        "my appointment is with dr sen",
        "mera appointment doctor ke saath hai",
        "amar appointment doctor er sathe ache",
        "আমার অ্যাপয়েন্টমেন্ট ডাক্তারের সাথে আছে",
    ])
    def test_appointment_is_with_dr_x(self, text):
        verdict, category = detect_unverifiable_claim(text)
        assert verdict == UNVERIFIABLE
        assert category == CATEGORY_APPOINTMENT_EXISTING


class TestDetectUnverifiableClaimNegatives:
    """The guard must NOT treat every normal question as an unverifiable
    claim -- this story's own explicit constraint (instruction 1)."""

    @pytest.mark.parametrize("text", [
        "is my appointment tomorrow?",
        "can you check my appointment",
        "can you confirm my appointment is with Dr Sen",
        "is my report normal?",
        "can you check my report",
        "has the doctor approved it, can you confirm",
        "did I already pay, please check",
        "check korte paren amar bill already paid kina",
        "i want to book an appointment",
        "book korna hai appointment",
        "অ্যাপয়েন্টমেন্ট চাই",
        "চেক করতে পারবেন",
        "what is the rate for CBC",
        "is Dr. Sen available tomorrow",
        "I have chest pain",
        "",
        "   ",
    ])
    def test_real_questions_and_requests_are_never_intercepted(self, text):
        verdict, category = detect_unverifiable_claim(text)
        assert verdict == _NONE
        assert category is None

    def test_none_text_never_crashes(self):
        assert detect_unverifiable_claim(None) == (_NONE, None)

    def test_a_claim_phrase_with_a_real_check_request_falls_through(self):
        # AC/instruction 2's central boundary: a caller who states one of
        # the closed-set claim shapes but ALSO asks the system to check or
        # confirm it must reach the real, verified, tool-backed intent --
        # never be quietly declined instead.
        verdict, category = detect_unverifiable_claim(
            "my appointment is with Dr Sen, can you confirm")
        assert verdict == _NONE
        assert category is None

    def test_the_billing_regression_example_from_the_instruction_falls_through(self):
        # This exact sentence is instruction 9's own worked example for
        # "verifiable flows still use the real DB result even when it
        # contradicts the caller's claim." It must NOT match
        # CATEGORY_PAYMENT_ALREADY's fixed phrase list (which requires an
        # explicit "I already paid" shape), so it reaches billing_balance
        # unchanged -- see TestVerifiableFlowStillUsesDbResult below for
        # the end-to-end proof.
        verdict, category = detect_unverifiable_claim("my bill is already paid")
        assert verdict == _NONE
        assert category is None


# --------------------------------------------------------------------- #
# main_pcm.py's _resolve_intent() -- guard order and short-circuit
# --------------------------------------------------------------------- #

class _ExplodingFastPath:
    def resolve(self, text):
        raise AssertionError("fast_path.resolve() must not run for an unverifiable-claim turn")


class _ExplodingCache:
    def get(self, text):
        raise AssertionError("semantic cache must not run for an unverifiable-claim turn")

    def put(self, text, data):
        raise AssertionError("semantic cache must not run for an unverifiable-claim turn")


def _exploding_extract_intent(text):
    raise AssertionError("extract_intent() (the LLM) must not run for an unverifiable-claim turn")


class TestGuardOrderInResolveIntent:
    @pytest.fixture(autouse=True)
    def _wire_exploding_stand_ins(self, monkeypatch):
        monkeypatch.setattr(main_pcm, "_fast_path", _ExplodingFastPath())
        monkeypatch.setattr(main_pcm, "_intent_cache", _ExplodingCache())
        monkeypatch.setattr(main_pcm, "extract_intent", _exploding_extract_intent)

    def test_short_circuits_before_any_classifier_runs(self):
        session = types.SimpleNamespace(call_id="test-unverifiable-claim-call")
        data = run(main_pcm._resolve_intent(session, "my appointment is tomorrow"))
        assert data["intent"] == "unverifiable_claim"
        assert data["slots"]["claim_category"] == CATEGORY_APPOINTMENT_EXISTING
        assert data["parts"] == [{"intent": "unverifiable_claim", "slots": data["slots"]}]

    def test_direct_reply_bn_is_never_set_for_a_guard_fired_claim(self):
        # AC1/AC4, at the structural level: the model's own free text
        # (direct_reply_bn) is never attached to a guard-fired claim --
        # this is the "smalltalk lane is protected" fix, checked here
        # directly at the point it is produced.
        session = types.SimpleNamespace(call_id="test-unverifiable-claim-call")
        data = run(main_pcm._resolve_intent(session, "i already paid"))
        assert data["direct_reply_bn"] is None

    def test_a_bare_claim_reaches_the_fast_path_only_when_no_claim_is_present(self):
        session = types.SimpleNamespace(call_id="test-unverifiable-claim-call")
        with pytest.raises(AssertionError):
            run(main_pcm._resolve_intent(session, "what is the rate for CBC"))


# --------------------------------------------------------------------- #
# Existing guards keep priority -- this new guard must not break them
# --------------------------------------------------------------------- #

class TestExistingGuardsStillWinWhereTheyShould:
    def test_clinical_interpretation_still_wins_over_a_claim_shaped_utterance(self):
        # A caller who states a claim AND asks a danger/diagnosis question
        # must still be caught by clinical_interpretation first -- this
        # module's own docstring's explicit ordering rule.
        # This phrase matches BOTH this new guard's CATEGORY_REPORT_RESULT
        # claim list AND agent/clinical_safety.py's own "is this serious"
        # phrase -- confirmed directly against both detectors, not assumed.
        text = "my report is normal, right? is this serious?"
        from agent.clinical_safety import is_clinical_interpretation
        from agent.unverifiable_claim import detect_unverifiable_claim as _detect
        assert is_clinical_interpretation(text) is True
        assert _detect(text)[0] == UNVERIFIABLE  # would fire in isolation --
        # proving guard ORDER, not phrase-list luck, is what protects this.

        session = types.SimpleNamespace(call_id="test-unverifiable-claim-call")
        data = run(main_pcm._resolve_intent(session, text))
        assert data["intent"] == "clinical_interpretation"

    def test_symptom_routing_is_unaffected_by_the_new_guard(self):
        session = types.SimpleNamespace(call_id="test-unverifiable-claim-call")
        data = run(main_pcm._resolve_intent(session, "I have chest pain"))
        assert data["intent"] == "symptom_department_routing"

    def test_complaint_is_unaffected_by_the_new_guard(self, monkeypatch):
        from agent.complaint_flow import is_complaint
        assert is_complaint("I want to make a complaint") is True
        session = types.SimpleNamespace(call_id="test-unverifiable-claim-call")
        data = run(main_pcm._resolve_intent(session, "I want to make a complaint"))
        assert data["intent"] == "complaint"


# --------------------------------------------------------------------- #
# agent/reply_templates.unverifiable_claim_reply() -- AC3/AC4
# --------------------------------------------------------------------- #

# Words that would sound like the system agreeing the claim is TRUE or
# FALSE, or attempting a medical read, or naming a date/doctor it never
# looked up -- none of these may ever appear in the fixed reply's English
# rendering (the language every "must not" example in the implementation
# instruction was itself given in).
_FORBIDDEN_CONFIRM_OR_DENY_PATTERNS_ENGLISH = [
    "yes, your appointment", "your appointment is confirmed",
    "that is correct", "that's correct", "you are right", "you're right",
    "that is not correct", "that's not true", "you are wrong",
    "your report is normal", "your report is not normal",
    "the doctor did approve", "the doctor did not approve",
    "your payment has been received", "your payment was not received",
    "tomorrow", "is normal", "is dangerous",
]


class TestUnverifiableClaimReply:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_returns_nonempty_text(self, language):
        text = unverifiable_claim_reply(language=language)
        assert isinstance(text, str)
        assert text.strip()

    def test_default_language_is_bengali(self):
        assert unverifiable_claim_reply() == unverifiable_claim_reply(language="bengali")

    def test_every_language_is_textually_distinct(self):
        replies = {unverifiable_claim_reply(language=l) for l in _LANGUAGES}
        assert len(replies) == len(_LANGUAGES)

    def test_never_contains_forbidden_confirm_or_deny_language(self):
        text = unverifiable_claim_reply(language="english").lower()
        for phrase in _FORBIDDEN_CONFIRM_OR_DENY_PATTERNS_ENGLISH:
            assert phrase not in text, f"forbidden phrase found: {phrase!r} in {text!r}"

    def test_states_it_cannot_confirm(self):
        text = unverifiable_claim_reply(language="english").lower()
        assert "confirm" in text or "not able to" in text

    def test_offers_a_human_or_the_counter(self):
        text = unverifiable_claim_reply(language="english").lower()
        assert "staff" in text or "counter" in text or "connect" in text

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_is_spoken_punctuation_clean(self, language):
        from agent.bn_normalize import verbalize
        spoken = verbalize(unverifiable_claim_reply(language=language), language=language)
        for ch in (":", "：", "[", "]", "{", "}"):
            assert ch not in spoken

    def test_is_distinct_from_out_of_scope_and_clinical_interpretation_replies(self):
        # A different kind of moment from either existing offer-based
        # decline -- must not share wording verbatim.
        from agent.reply_templates import out_of_scope_reply, clinical_interpretation_reply
        text = unverifiable_claim_reply(language="english")
        assert text != out_of_scope_reply(language="english")
        assert text != clinical_interpretation_reply(language="english")


# --------------------------------------------------------------------- #
# agent/outcomes.py -- the dedicated audit event (instruction 6)
# --------------------------------------------------------------------- #

class TestUnverifiableClaimAuditEvent:
    def test_records_a_fixed_reason_code_only_never_caller_text(self, _isolated_escalation_log):
        record_unverifiable_claim(intent="unverifiable_claim",
                                   reason=CATEGORY_PAYMENT_ALREADY,
                                   call_id="call-1", turn_index=3)
        lines = _isolated_escalation_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == UNVERIFIABLE_CLAIM_EVENT
        assert record["intent"] == "unverifiable_claim"
        assert record["reason"] == CATEGORY_PAYMENT_ALREADY
        assert record["call_id"] == "call-1"
        assert record["turn_index"] == 3
        # No field anywhere in the record carries free-form caller speech.
        for value in record.values():
            if isinstance(value, str):
                assert "paid" not in value.lower() or value == CATEGORY_PAYMENT_ALREADY

    def test_bumps_the_per_reason_counter(self):
        record_unverifiable_claim(intent="unverifiable_claim", reason=CATEGORY_REPORT_RESULT)
        record_unverifiable_claim(intent="unverifiable_claim", reason=CATEGORY_REPORT_RESULT)
        record_unverifiable_claim(intent="unverifiable_claim", reason=CATEGORY_APPOINTMENT_EXISTING)
        assert unverifiable_claim_counts(CATEGORY_REPORT_RESULT) == 2
        assert unverifiable_claim_counts(CATEGORY_APPOINTMENT_EXISTING) == 1
        all_counts = unverifiable_claim_counts()
        assert all_counts[CATEGORY_REPORT_RESULT] == 2

    def test_counters_are_kept_separate_from_every_other_ledger_counter(self):
        from agent.outcomes import report_access_denied_counts
        record_unverifiable_claim(intent="unverifiable_claim", reason=CATEGORY_DOCTOR_APPROVAL)
        assert report_access_denied_counts() in ({}, 0) or \
            "DOCTOR_APPROVAL_CLAIM" not in report_access_denied_counts()


# --------------------------------------------------------------------- #
# main_pcm.py dispatch -- fresh-turn "unverifiable_claim" branch
# --------------------------------------------------------------------- #

class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-unverifiable-claim-call", pending=pending,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
        call_state=main_pcm.call_state_mod.build(), utt_seq=1,
        confirm_attempts=0,
    )


class FakeASRResultClaim:
    text = "my appointment is tomorrow"
    decoder_agreement = 0.95
    decoder_used = "rnnt"
    ctc_words = 4
    rnnt_words = 4


class _ExplodingToolsClient:
    """Proves the guard NEVER calls out to clinic-api -- there is no
    lookup capability for these claim categories today (instruction 3 /
    this module's own docstring), so no tool call should ever be
    attempted on this path."""

    def __getattr__(self, name):
        def _explode(*args, **kwargs):
            raise AssertionError(
                f"clinic-api tool method {name!r} must never be called for "
                "an unverifiable-claim turn")
        return _explode


class TestUnverifiableClaimIntentDispatch:
    def _dispatch(self, monkeypatch, tmp_path):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        async def fake_resolve_intent(session, text):
            return {"intent": "unverifiable_claim",
                    "slots": {"claim_category": CATEGORY_APPOINTMENT_EXISTING}}

        class FakeASR:
            async def transcribe_utterance(self, wav_path):
                return FakeASRResultClaim()

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        monkeypatch.setattr(main_pcm, "_asr", FakeASR())
        monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
        monkeypatch.setattr(main_pcm, "_tools", _ExplodingToolsClient())

        session = make_session()
        wav_path = tmp_path / "utt.wav"
        wav_path.write_bytes(b"")
        run(main_pcm._dispatch_turn(session, str(wav_path)))
        return spoken, session

    def test_speaks_the_fixed_unverifiable_claim_reply(self, monkeypatch, tmp_path):
        spoken, _ = self._dispatch(monkeypatch, tmp_path)
        assert spoken == [unverifiable_claim_reply(language="english")]

    def test_opens_the_unverifiable_claim_choice_pending_state(self, monkeypatch, tmp_path):
        _, session = self._dispatch(monkeypatch, tmp_path)
        assert session.pending == {"awaiting": "unverifiable_claim_choice", "retries": 0}

    def test_records_the_audit_event(self, monkeypatch, tmp_path, _isolated_escalation_log):
        self._dispatch(monkeypatch, tmp_path)
        lines = _isolated_escalation_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == UNVERIFIABLE_CLAIM_EVENT
        assert record["reason"] == CATEGORY_APPOINTMENT_EXISTING


# --------------------------------------------------------------------- #
# main_pcm.py _continue_pending -- interrupting an in-progress flow
# --------------------------------------------------------------------- #

class TestUnverifiableClaimInterruptsPendingFlow:
    def test_abandons_an_in_progress_booking_flow_and_answers_instead(self, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        monkeypatch.setattr(main_pcm, "_tools", _ExplodingToolsClient())

        pending = {"awaiting": "phone", "slots": {"doctor_name": "Dr. Sen", "date": "2026-10-01",
                                                    "time_slot": "10:00", "patient_name": "Test"}}
        session = make_session(pending=dict(pending))
        handled = run(main_pcm._continue_pending(session, "my appointment is with Dr Sen"))

        assert handled is True
        assert session.pending == {"awaiting": "unverifiable_claim_choice", "retries": 0}
        assert spoken == [unverifiable_claim_reply(language="english")]

    def test_never_finishes_the_flow_it_interrupted(self, monkeypatch):
        # The interrupted flow's own fields must not leak into anything
        # spoken -- the claim wins outright, same zero-negotiation shape
        # every other interrupt guard in this chain already has.
        monkeypatch.setattr(main_pcm, "_speak", _AsyncNoOp())
        monkeypatch.setattr(main_pcm, "_tools", _ExplodingToolsClient())
        pending = {"awaiting": "billing_phone", "retries": 0}
        session = make_session(pending=dict(pending))
        handled = run(main_pcm._continue_pending(session, "i already paid"))
        assert handled is True
        assert session.pending["awaiting"] == "unverifiable_claim_choice"


# --------------------------------------------------------------------- #
# main_pcm.py _continue_pending -- "unverifiable_claim_choice" handler
# --------------------------------------------------------------------- #

class TestUnverifiableClaimChoiceHandler:
    def test_affirmative_hands_off_to_a_human(self, monkeypatch, _isolated_escalation_log):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        session = make_session(pending={"awaiting": "unverifiable_claim_choice", "retries": 0})
        handled = run(main_pcm._continue_pending(session, "yes"))

        assert handled is True
        assert session.pending is None
        assert spoken == [human_fallback_reply(language="english")]
        lines = _isolated_escalation_log.read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(lines[-1])
        assert record["event"] == "human_handoff"
        assert record["intent"] == "unverifiable_claim"

    def test_negative_reuses_the_out_of_scope_counter_reply(self, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        session = make_session(pending={"awaiting": "unverifiable_claim_choice", "retries": 0})
        handled = run(main_pcm._continue_pending(session, "no"))

        assert handled is True
        assert session.pending is None
        assert spoken == [out_of_scope_counter_reply(language="english")]

    def test_an_unclear_answer_retries_then_falls_through(self, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        session = make_session(pending={"awaiting": "unverifiable_claim_choice", "retries": 0})

        handled = run(main_pcm._continue_pending(session, "what do you mean"))
        assert handled is True
        assert session.pending["retries"] == 1
        assert spoken == [unverifiable_claim_reply(language="english")]

        session.pending["retries"] = 3
        handled = run(main_pcm._continue_pending(session, "what do you mean"))
        assert handled is False
        assert session.pending is None


# --------------------------------------------------------------------- #
# The central safety claim: a factual assertion can never reach smalltalk
# --------------------------------------------------------------------- #

class TestSmalltalkIsProtected:
    def test_unverifiable_claim_is_not_a_registered_llm_intent(self):
        # Same deliberate design choice as symptom_department_routing:
        # never something the model itself classifies -- the deterministic
        # phrase list is the real enforcement (instruction 7).
        from agent.llm import VALID_INTENTS
        assert "unverifiable_claim" not in VALID_INTENTS

    def test_llm_prompt_now_disambiguates_a_bare_factual_claim_from_smalltalk(self):
        from agent.llm import SYSTEM_PROMPT_TEMPLATE
        prompt_lower = SYSTEM_PROMPT_TEMPLATE.lower()
        assert "unclear" in prompt_lower
        assert any(phrase in prompt_lower for phrase in
                   ("already true about their own appointment",
                    "you are not the authority for whether it is true",
                    "never write a reply here that agrees with"))

    @pytest.mark.parametrize("text", [
        "my appointment is tomorrow", "my report is normal",
        "the doctor already approved it", "i already paid",
        "my appointment is with dr sen",
    ])
    def test_every_example_claim_short_circuits_before_the_model_ever_runs(self, text, monkeypatch):
        monkeypatch.setattr(main_pcm, "_fast_path", _ExplodingFastPath())
        monkeypatch.setattr(main_pcm, "_intent_cache", _ExplodingCache())
        monkeypatch.setattr(main_pcm, "extract_intent", _exploding_extract_intent)
        session = types.SimpleNamespace(call_id="test-unverifiable-claim-call")
        data = run(main_pcm._resolve_intent(session, text))
        assert data["intent"] == "unverifiable_claim"
        assert data["intent"] != "smalltalk"
        assert data["direct_reply_bn"] is None


# --------------------------------------------------------------------- #
# Verifiable flows keep using the real system-of-record result
# (instruction 2 / instruction 9's own worked example)
# --------------------------------------------------------------------- #

class FakeBillingToolsClient:
    def __init__(self, response):
        self.calls = []
        self._response = response

    async def get_patient_billing(self, phone):
        self.calls.append(phone)
        return self._response


class TestVerifiableFlowStillUsesDbResult:
    def test_billing_still_speaks_the_real_due_amount_despite_the_callers_claim(self, monkeypatch, tmp_path):
        # The caller says the bill is already paid; the DB says otherwise.
        # This phrase does not match the guard's fixed phrase list (proved
        # above in TestDetectUnverifiableClaimNegatives), so it reaches
        # the existing, unmodified billing_balance dispatch branch, which
        # must speak the DB's real outstanding amount -- never agree with
        # the caller, never stay silent about the discrepancy.
        due_result = {"patient_found": True, "found": True,
                      "outstanding_amount": 500.0, "due_date": "2026-10-01"}
        fake_tools = FakeBillingToolsClient(due_result)
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        async def fake_resolve_intent(session, text):
            return {"intent": "billing_balance", "slots": {"phone": "9876543210"}}

        class FakeASR:
            async def transcribe_utterance(self, wav_path):
                return types.SimpleNamespace(
                    text="my bill is already paid", decoder_agreement=0.95,
                    decoder_used="rnnt", ctc_words=5, rnnt_words=5)

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        monkeypatch.setattr(main_pcm, "_asr", FakeASR())
        monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
        monkeypatch.setattr(main_pcm, "_tools", fake_tools)

        session = make_session()
        wav_path = tmp_path / "utt.wav"
        wav_path.write_bytes(b"")
        run(main_pcm._dispatch_turn(session, str(wav_path)))

        from agent.reply_templates import billing_balance_reply
        assert fake_tools.calls == ["9876543210"]
        assert spoken == [billing_balance_reply(due_result, language="english")]
        # The spoken reply states the real due amount, never that nothing
        # is owed -- proving the DB result won, not the caller's claim.
        assert "500" in spoken[0]


# --------------------------------------------------------------------- #
# No-lookup-exists case: a fixed response only, never an invented answer
# --------------------------------------------------------------------- #

class TestNoLookupExistsProducesOnlyTheFixedResponse:
    def test_an_existing_appointment_claim_never_invents_a_date_or_doctor(self, monkeypatch, tmp_path):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        async def fake_resolve_intent(session, text):
            return {"intent": "unverifiable_claim",
                    "slots": {"claim_category": CATEGORY_APPOINTMENT_EXISTING}}

        class FakeASR:
            async def transcribe_utterance(self, wav_path):
                return FakeASRResultClaim()

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        monkeypatch.setattr(main_pcm, "_asr", FakeASR())
        monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
        monkeypatch.setattr(main_pcm, "_tools", _ExplodingToolsClient())

        session = make_session()
        wav_path = tmp_path / "utt.wav"
        wav_path.write_bytes(b"")
        run(main_pcm._dispatch_turn(session, str(wav_path)))

        assert spoken == [unverifiable_claim_reply(language="english")]
        assert "tomorrow" not in spoken[0].lower()
        assert "dr" not in spoken[0].lower().split()


# --------------------------------------------------------------------- #
# Failure cases: nothing ever turns an unverified claim into a fact
# --------------------------------------------------------------------- #

class TestFailureCasesNeverProduceAConfirmedFact:
    def test_audit_logging_failure_still_speaks_the_fixed_honest_reply(self, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        def boom(*args, **kwargs):
            raise RuntimeError("ledger unavailable")

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        monkeypatch.setattr(main_pcm, "record_unverifiable_claim", boom)

        session = make_session()
        run(main_pcm._finish_unverifiable_claim(session, CATEGORY_REPORT_RESULT, language="english"))

        assert spoken == [unverifiable_claim_reply(language="english")]
        assert session.pending == {"awaiting": "unverifiable_claim_choice", "retries": 0}

    def test_pending_is_still_opened_even_if_the_audit_call_raises(self, monkeypatch):
        # Belt-and-braces: the caller must still be offered a human even
        # when the audit write itself failed -- an audit failure must
        # never cause a claim to become verified, but it also must never
        # cause the honest decline+offer to be skipped.
        monkeypatch.setattr(main_pcm, "_speak", _AsyncNoOp())

        def boom(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(main_pcm, "record_unverifiable_claim", boom)
        session = make_session()
        run(main_pcm._finish_unverifiable_claim(session, CATEGORY_DOCTOR_APPROVAL, language="english"))
        assert session.pending is not None
        assert session.pending["awaiting"] == "unverifiable_claim_choice"

    def test_no_tool_is_ever_called_for_a_guard_fired_claim(self, monkeypatch):
        # There is no lookup capability for any of the four claim
        # categories today (see agent/unverifiable_claim.py's own
        # docstring) -- a "clinic-api unavailable" or "lookup failure"
        # scenario cannot even arise on this path because no lookup is
        # ever attempted. _ExplodingToolsClient proves that directly: if
        # any tool method were called, this test would raise.
        monkeypatch.setattr(main_pcm, "_speak", _AsyncNoOp())
        monkeypatch.setattr(main_pcm, "_tools", _ExplodingToolsClient())
        session = make_session()
        run(main_pcm._finish_unverifiable_claim(session, CATEGORY_PAYMENT_ALREADY, language="english"))
        # No AssertionError from _ExplodingToolsClient means no tool call
        # was attempted -- the only thing this test needs to prove.


# --------------------------------------------------------------------- #
# Transport parity
# --------------------------------------------------------------------- #

class TestTransportParity:
    def test_main_dot_py_has_the_unverifiable_claim_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "unverifiable_claim"' in src
        assert "detect_unverifiable_claim" in src
        assert "unverifiable_claim_reply" in src
        assert "_finish_unverifiable_claim" in src
        assert "record_unverifiable_claim" in src
        assert '"unverifiable_claim_choice"' in src

    def test_main_pcm_dot_py_has_the_unverifiable_claim_branch(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "unverifiable_claim"' in src
        assert "detect_unverifiable_claim" in src
        assert "unverifiable_claim_reply" in src
        assert "_finish_unverifiable_claim" in src
        assert "record_unverifiable_claim" in src
        assert '"unverifiable_claim_choice"' in src

    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
