"""ADDED BY SOURAV -- "Caller wants to make a complaint" story.

story title: Caller wants to make a complaint
user story: As a dissatisfied patient, I want my complaint recorded and
    routed to a person, so that it is not absorbed by a machine.
acceptance criteria:
    1. Complaint is recognised.
    2. Complaint is acknowledged exactly once.
    3. Complaint is captured verbatim.
    4. Complaint is routed to the complaints path/person-handling path.
    5. Agent does NOT attempt to resolve, explain, defend, justify, or
       argue about the complaint.

Structured directly on tests/test_immediate_human_request.py's own shape
(adversarial phrase corpus, a _resolve_intent() short-circuit proof, a
dispatch-level test, a _continue_pending() override test, transport
parity, schema checks) for the same reason agent/complaint_flow.py itself
is structured on agent/human_fast_path.py: this is the same kind of
zero-tolerance, pre-classifier guard story. Two sections that story does
not need at all, because this one does:

1. A verbatim-storage proof (TestVerbatimStorage below) -- this story's
   own AC 3 has no equivalent in human_direct_request's own AC set, which
   never stores any caller text at all. Checked against a FakeToolsClient
   that records exactly what agent/tools_client.submit_complaint() was
   called with, for a battery of shapes (short, long, containing numbers/
   names, mixed with a normal request, after several turns, repeated).

2. A "does not trigger normal automated resolution" proof
   (TestDoesNotResolveOrLookUp below) -- AC 5's own bar. Checked by
   asserting NO OTHER FakeToolsClient method is ever called for a
   complaint turn, only submit_complaint().

WHY THE ADVERSARIAL PROMPTS ARE NOT JUST agent.complaint_flow's OWN PHRASE
TUPLES READ BACK: same discipline test_immediate_human_request.py's own
note explains -- every prompt below was written independently and run
against the real function, so this suite can catch a gap in the module
rather than just confirming the module agrees with itself.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import types

import pytest

import main
import main_pcm
from agent.bn_normalize import detect_language
from agent.complaint_flow import detect_complaint, is_complaint
from agent.outcomes import (
    human_handoff_counts,
    HUMAN_COMPLAINT_INTENT,
    record_complaint_filed,
    _reset_for_testing,
)
import agent.outcomes as outcomes_module
from agent.reply_templates import (
    complaint_acknowledged_reply, human_fallback_reply, out_of_scope_reply,
)

TRANSPORTS = [main, main_pcm]


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
# The adversarial coverage. CORES x CARRIERS, same construction as
# tests/test_immediate_human_request.py's own corpora.
# --------------------------------------------------------------------- #

_EN_CORES = [
    "I want to file a complaint",
    "I want to make a complaint",
    "I want to lodge a complaint",
    "I want to register a complaint",
    "I have a complaint",
    "this is a complaint",
    "I'd like to complain",
    "I'd like to complain about the service",
    "I want to complain about how I was treated",
    "let me file a complaint",
    "filing a complaint",
    "making a complaint about your clinic",
    "I want to report how I was treated",
    "your staff was very rude to me",
    "the doctor was rude to me",
    "nobody helped me at the clinic",
    "I am extremely dissatisfied with the service",
    "I am very unhappy with the service",
    # FIXED BY SOURAV -- final validation pass found this exact literal
    # sentence (the story's own case #1) unrecognised: the module only
    # had "...with THE service", not "...with THIS service", which is
    # at least as natural since a caller is usually describing their
    # own just-had experience. Pinned here as a permanent regression
    # case now that agent/complaint_flow.py covers it.
    "I am very unhappy with this service",
    "this service is unacceptable",
    "I had a terrible experience at your clinic",
    "worst experience I have ever had",
    "I am complaining about the front desk",
    "please register my complaint",
    # FIXED BY SOURAV -- final validation pass found "also"/"and" tacked
    # onto the front of an otherwise-covered phrase ("I ALSO want to
    # file a complaint...") went unrecognised, because the original
    # phrases all assumed "I" was the very first word. Pinned here too.
    "I also want to file a complaint about the billing",
    "and I have a complaint about the wait time",
]
_EN_CARRIERS = ["", "Look, ", "Excuse me, ", "I'm calling because "]

_HI_CORES = [
    "mujhe complaint karni hai",
    "complaint darj karna chahta hoon",
    "meri ek shikayat hai",
    "shikayat darj karna hai",
    "shikayat karna chahta hoon",
    "aapki service se bahut naraz hoon",
    "bahut kharab service thi",
    "staff ne bahut badtameezi ki",
    "mujhe bahut bura anubhav hua",
]
_HI_CARRIERS = ["", "Dekho, ", "Suniye, "]

_BN_CORES = [
    "আমার একটা অভিযোগ আছে",
    "আমি অভিযোগ করতে চাই",
    "একটা কমপ্লেইন করতে চাই",
    "আমার একটা কমপ্লেইন আছে",
    "খুব খারাপ ব্যবহার করেছে",
    "আমি খুব অসন্তুষ্ট",
    "স্টাফ খুব খারাপ ব্যবহার করেছে",
]
_BN_CARRIERS = ["", "শুনুন, ", "দয়া করে, "]

_BANGLISH_CORES = [
    "amar ekta complaint ache",
    "ami complaint korte chai",
    "complaint korte chai",
    "obhijog korte chai",
    "amar ekta obhijog ache",
    "khub kharap byabohar korechilo",
    "khub baje experience hoyeche",
    "apnader service niye amar complaint ache",
]
_BANGLISH_CARRIERS = ["", "Shunun, ", "Please, "]


def _corpus(cores, carriers):
    return [carrier + core for core, carrier in itertools.product(cores, carriers)]


_EN_PROMPTS = _corpus(_EN_CORES, _EN_CARRIERS)
_HI_PROMPTS = _corpus(_HI_CORES, _HI_CARRIERS)
_BN_PROMPTS = _corpus(_BN_CORES, _BN_CARRIERS)
_BANGLISH_PROMPTS = _corpus(_BANGLISH_CORES, _BANGLISH_CARRIERS)


class TestAdversarialCoverage:
    def test_the_corpora_each_have_a_healthy_number_of_prompts(self):
        assert len(_EN_PROMPTS) >= 80
        assert len(_HI_PROMPTS) >= 20
        assert len(_BN_PROMPTS) >= 15
        assert len(_BANGLISH_PROMPTS) >= 20

    @pytest.mark.parametrize("prompt", _EN_PROMPTS)
    def test_english_prompt_is_recognised(self, prompt):
        assert is_complaint(prompt), f"breach: {prompt!r}"

    @pytest.mark.parametrize("prompt", _HI_PROMPTS)
    def test_hinglish_prompt_is_recognised(self, prompt):
        assert is_complaint(prompt), f"breach: {prompt!r}"

    @pytest.mark.parametrize("prompt", _BN_PROMPTS)
    def test_bengali_prompt_is_recognised(self, prompt):
        assert is_complaint(prompt), f"breach: {prompt!r}"

    @pytest.mark.parametrize("prompt", _BANGLISH_PROMPTS)
    def test_banglish_prompt_is_recognised(self, prompt):
        assert is_complaint(prompt), f"breach: {prompt!r}"

    def test_ordinary_catalogue_questions_are_left_alone(self):
        ordinary = [
            "what is the rate for CBC",
            "is Dr. Sen available tomorrow",
            "what time do you open",
            "do I need a prescription for uric acid",
            "can I get my report delivered",
            "রেট কত ইউরিক অ্যাসিডের",
            "why is this taking so long",
        ]
        for text in ordinary:
            assert not is_complaint(text), f"false positive: {text!r}"

    def test_human_direct_request_phrasing_is_left_to_its_own_guard(self):
        # The two stories are related but distinct -- a caller asking to
        # bypass the assistant entirely is not, by itself, a complaint.
        human_only = [
            "connect me to a person",
            "I want to talk to a human",
            "get me an agent",
        ]
        for text in human_only:
            assert not is_complaint(text), f"false positive: {text!r}"

    def test_empty_and_blank_text_is_never_flagged(self):
        assert not is_complaint("")
        assert not is_complaint("   ")
        assert not is_complaint(None)

    def test_bengali_word_boundary_safety(self):
        assert not is_complaint("মানুষটি কেমন আছেন")

    def test_detect_returns_the_matched_language_family(self):
        matched, family = detect_complaint("I want to file a complaint")
        assert matched is True
        assert family == "english"
        matched, family = detect_complaint("shikayat darj karna hai")
        assert matched is True
        assert family == "latin_translit"
        matched, family = detect_complaint("আমার একটা অভিযোগ আছে")
        assert matched is True
        assert family == "bengali"
        matched, family = detect_complaint("what is the rate for CBC")
        assert matched is False
        assert family is None

    # --------------------------------------------------------------- #
    # Edge cases explicitly named in the story's own test list.
    # --------------------------------------------------------------- #

    def test_very_short_complaint(self):
        assert is_complaint("I have a complaint.")
        assert is_complaint("complaint korte chai.")

    def test_long_rambling_complaint_still_recognised(self):
        long_text = (
            "so I came in last Tuesday around 4pm for a blood test and the "
            "person at the desk kept me waiting for over an hour without "
            "any explanation and honestly I want to file a complaint about "
            "how the whole thing was handled because this is not the first "
            "time this has happened to me either"
        )
        assert is_complaint(long_text)

    def test_complaint_containing_numbers_and_names(self):
        assert is_complaint(
            "My name is Rahul Sen, phone 9831012345, and I want to file a "
            "complaint about Dr. Gupta and the front desk staff"
        )

    def test_complaint_mixed_with_a_normal_request(self):
        assert is_complaint(
            "I want to book an appointment with Dr. Sen, and also I want "
            "to file a complaint about how rude the staff was last time"
        )


# --------------------------------------------------------------------- #
# _resolve_intent() -- the actual choke point. Proves the guard fires
# structurally, before the fast path, semantic cache, or LLM ever run.
# --------------------------------------------------------------------- #

class _ExplodingFastPath:
    def resolve(self, text):
        raise AssertionError("fast_path.resolve() must not run for a complaint turn")


class _ExplodingCache:
    def get(self, text):
        raise AssertionError("semantic cache must not run for a complaint turn")

    def put(self, text, data):
        raise AssertionError("semantic cache must not run for a complaint turn")


def _exploding_extract_intent(text):
    raise AssertionError("extract_intent() (the LLM) must not run for a complaint turn")


@pytest.fixture(params=TRANSPORTS, ids=["main", "main_pcm"])
def transport(request):
    return request.param


class TestResolveIntentNeverReachesTheClassifier:
    @pytest.fixture(autouse=True)
    def _wire_exploding_stand_ins(self, transport, monkeypatch):
        monkeypatch.setattr(transport, "_fast_path", _ExplodingFastPath())
        monkeypatch.setattr(transport, "_intent_cache", _ExplodingCache())
        monkeypatch.setattr(transport, "extract_intent", _exploding_extract_intent)
        self.transport = transport

    @pytest.mark.parametrize(
        "prompt", _EN_PROMPTS[:15] + _HI_PROMPTS[:10] + _BN_PROMPTS[:10] + _BANGLISH_PROMPTS[:10]
    )
    def test_short_circuits_before_any_classifier_runs(self, prompt):
        session = types.SimpleNamespace(call_id="test-complaint-guard-call")
        data = run(self.transport._resolve_intent(session, prompt))
        assert data["intent"] == "complaint"
        assert data["parts"] == [{"intent": "complaint", "slots": data["slots"]}]
        assert data["direct_reply_bn"] is None

    def test_an_ordinary_question_still_reaches_the_fast_path(self):
        session = types.SimpleNamespace(call_id="test-complaint-guard-call")
        with pytest.raises(AssertionError):
            run(self.transport._resolve_intent(session, "what is the rate for CBC"))


# --------------------------------------------------------------------- #
# Dispatch -- the "complaint" intent branch, and the FakeToolsClient that
# proves verbatim storage and "no other lookup ran".
# --------------------------------------------------------------------- #

class FakeToolsClient:
    def __init__(self):
        self.calls = []
        self._complaint_exception = None
        self._complaint_result = None

    def _record(self, name, *args):
        self.calls.append((name, args))

    async def submit_complaint(self, complaint_text, phone=None):
        self._record("submit_complaint", complaint_text, phone)
        if self._complaint_exception:
            raise self._complaint_exception
        if self._complaint_result is not None:
            return self._complaint_result
        return {"success": True, "complaint_id": "CMP-20260918-TEST",
                "phone": phone, "status": "pending"}

    # Present so a test can assert these were NEVER called (AC 5: no
    # automated resolution/lookup for a complaint turn).
    async def get_test_rate(self, *a, **k):
        self._record("get_test_rate", *a)
        return {"found": False}

    async def get_clinic_info(self, *a, **k):
        self._record("get_clinic_info", *a)
        return {"found": False}


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


class FakeASRResult:
    text = "I want to file a complaint about the front desk"
    decoder_used = "ctc"
    decoder_agreement = 1.0
    ctc_words = 8
    rnnt_words = 8


class FakeASRResultBanglish:
    text = "amar ekta complaint ache"
    decoder_used = "ctc"
    decoder_agreement = 1.0
    ctc_words = 4
    rnnt_words = 4


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-complaint-call", pending=pending,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
        call_state=main.call_state_mod.build(), utt_seq=1,
        confirm_attempts=0,
    )


@pytest.fixture
def stub(transport, monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    class FakeASR:
        async def transcribe_utterance(self, wav_path):
            return FakeASRResult()

    monkeypatch.setattr(transport, "_speak", fake_speak)
    monkeypatch.setattr(transport, "_asr", FakeASR())
    tools = FakeToolsClient()
    monkeypatch.setattr(transport, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools, transport=transport)


def _dispatch_initial(stub, monkeypatch, asr_text_cls=FakeASRResult, tmp_path=None):
    async def fake_resolve_intent(session, text):
        return {"intent": "complaint", "slots": {}}

    class FakeASR:
        async def transcribe_utterance(self, wav_path):
            return asr_text_cls()

    monkeypatch.setattr(stub.transport, "_asr", FakeASR())
    monkeypatch.setattr(stub.transport, "_resolve_intent", fake_resolve_intent)

    session = make_session()
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(stub.transport._dispatch_turn(session, str(wav_path)))
    return session


class TestComplaintDispatch:
    def test_speaks_the_acknowledgment_exactly_once(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken == [complaint_acknowledged_reply(language="english")]

    def test_does_not_speak_human_fallback_or_out_of_scope_offer(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken != [human_fallback_reply(language="english")]
        assert stub.spoken != [out_of_scope_reply(language="english")]

    def test_sets_no_pending_state_at_all(self, stub, monkeypatch, tmp_path):
        session = _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert session.pending is None

    def test_does_not_enter_out_of_scope_negotiation(self, stub, monkeypatch, tmp_path):
        session = _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert session.pending is None
        assert (session.pending or {}).get("awaiting") != "out_of_scope_choice"

    def test_only_submit_complaint_is_called_never_a_lookup(self, stub, monkeypatch, tmp_path):
        # AC 5: does NOT attempt to resolve/explain/defend/argue -- proven
        # here as "no other tool call happened", not just "the right reply
        # was spoken".
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        called = [c[0] for c in stub.tools.calls]
        assert called == ["submit_complaint"]

    def test_records_the_handoff_on_this_turn(self, stub, monkeypatch, tmp_path):
        assert human_handoff_counts(HUMAN_COMPLAINT_INTENT) == 0
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert human_handoff_counts(HUMAN_COMPLAINT_INTENT) == 1

    def test_escalation_ledger_never_contains_the_complaint_text(self, stub, monkeypatch, tmp_path, _isolated_escalation_log):
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        entry = json.loads(_isolated_escalation_log.read_text(encoding="utf-8").strip())
        assert entry["event"] == "human_handoff"
        assert entry["intent"] == "complaint"
        assert entry["reason"] == "caller_filed_complaint"
        assert entry["call_id"] == "test-complaint-call"
        assert FakeASRResult.text not in json.dumps(entry)

    def test_respects_the_detected_language_of_the_utterance(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, asr_text_cls=FakeASRResultBanglish, tmp_path=tmp_path)
        expected_language = detect_language(FakeASRResultBanglish.text)
        assert stub.spoken == [complaint_acknowledged_reply(language=expected_language)]

    def test_tool_failure_speaks_the_generic_apology_not_a_false_acknowledgment(self, stub, monkeypatch, tmp_path):
        from agent.tools_client import ToolCallError
        stub.tools._complaint_exception = ToolCallError("db down")
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken != [complaint_acknowledged_reply(language="english")]
        assert stub.spoken == ["এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।"]

    def test_write_reporting_failure_also_withholds_the_acknowledgment(self, stub, monkeypatch, tmp_path):
        stub.tools._complaint_result = {"success": False}
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken != [complaint_acknowledged_reply(language="english")]


class TestVerbatimStorage:
    """AC 3: 'captured verbatim'. Proven directly against the fake tools
    client's recorded call arguments -- not just that something got spoken,
    but that submit_complaint() received EXACTLY the caller's own text."""

    def _submitted_text(self, stub, monkeypatch, tmp_path, text):
        class _ASR:
            pass
        _ASR.text = text
        _ASR.decoder_used = "ctc"
        _ASR.decoder_agreement = 1.0
        _ASR.ctc_words = len(text.split())
        _ASR.rnnt_words = len(text.split())
        _dispatch_initial(stub, monkeypatch, asr_text_cls=_ASR, tmp_path=tmp_path)
        calls = [c for c in stub.tools.calls if c[0] == "submit_complaint"]
        assert len(calls) == 1, "expected exactly one submit_complaint() call"
        return calls[0][1][0]

    def test_short_complaint_stored_exactly(self, stub, monkeypatch, tmp_path):
        text = "I have a complaint."
        assert self._submitted_text(stub, monkeypatch, tmp_path, text) == text

    def test_long_complaint_stored_exactly(self, stub, monkeypatch, tmp_path):
        text = (
            "so I came in last Tuesday around 4pm for a blood test and the "
            "person at the desk kept me waiting for over an hour without "
            "any explanation and honestly I want to file a complaint about "
            "how the whole thing was handled"
        )
        assert self._submitted_text(stub, monkeypatch, tmp_path, text) == text

    def test_complaint_with_numbers_and_names_stored_exactly(self, stub, monkeypatch, tmp_path):
        text = ("My name is Rahul Sen, phone 9831012345, and I want to file "
                "a complaint about Dr. Gupta")
        assert self._submitted_text(stub, monkeypatch, tmp_path, text) == text

    def test_complaint_mixed_with_a_normal_request_stored_exactly(self, stub, monkeypatch, tmp_path):
        text = ("I want to book an appointment with Dr. Sen, and also I "
                "want to file a complaint about how rude the staff was")
        assert self._submitted_text(stub, monkeypatch, tmp_path, text) == text

    def test_no_trimming_or_rewriting_of_punctuation_or_case(self, stub, monkeypatch, tmp_path):
        text = "I WANT TO FILE A COMPLAINT!!!  right now??"
        assert self._submitted_text(stub, monkeypatch, tmp_path, text) == text


# --------------------------------------------------------------------- #
# _continue_pending() -- the mid-flow override. A complaint said mid-
# booking, mid-OTP, or during any other pending flow must interrupt it.
# --------------------------------------------------------------------- #

def _continue(stub, monkeypatch, pending, reply_text):
    session = make_session(pending=dict(pending))
    handled = run(stub.transport._continue_pending(session, reply_text))
    return handled, stub.spoken, session


class TestContinuePendingOverride:
    def test_interrupts_a_booking_flow_and_records_the_complaint(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "booking_date", "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending,
                                              "actually I want to file a complaint about last time")
        assert handled is True
        assert spoken == [complaint_acknowledged_reply(language="english")]
        assert session.pending is None
        assert [c[0] for c in stub.tools.calls] == ["submit_complaint"]

    def test_interrupts_an_otp_flow(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "otp_code", "retries": 1}
        handled, spoken, session = _continue(stub, monkeypatch, pending,
                                              "I have a complaint about this whole process")
        assert handled is True
        assert session.pending is None

    def test_interrupts_the_out_of_scope_choice_flow(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "out_of_scope_choice", "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending,
                                              "no, I want to file a complaint instead")
        assert handled is True
        assert session.pending is None

    def test_verbatim_text_at_this_call_site_is_also_exact(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "booking_date", "retries": 0}
        text = "actually never mind the booking, I have a complaint about Dr. Gupta"
        _continue(stub, monkeypatch, pending, text)
        calls = [c for c in stub.tools.calls if c[0] == "submit_complaint"]
        assert len(calls) == 1
        assert calls[0][1][0] == text

    def test_an_ordinary_mid_flow_reply_is_not_swallowed_by_this_guard(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "confirm_booking", "slots": {}, "candidates": None,
                   "offered_date": None, "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending, "না")
        assert [c[0] for c in stub.tools.calls] == []

    def test_repeated_complaint_in_the_same_call_is_acknowledged_each_time(self, stub, monkeypatch, tmp_path):
        # AC 2's "acknowledged exactly once" is per complaint, not per
        # call -- a caller who complains twice gets two acknowledgments,
        # each recorded as its own event, same design as human_direct_
        # request's own re-triggerable-every-time guard.
        pending = {"awaiting": "booking_date", "retries": 0}
        _continue(stub, monkeypatch, pending, "I have a complaint about the wait time")
        session2 = make_session(pending={"awaiting": "booking_date", "retries": 0})
        run(stub.transport._continue_pending(session2, "and I want to file a complaint about the billing too"))
        assert len([c for c in stub.tools.calls if c[0] == "submit_complaint"]) == 2
        assert human_handoff_counts(HUMAN_COMPLAINT_INTENT) == 2

    def test_complaint_after_several_turns_still_recognised(self, stub, monkeypatch, tmp_path):
        session = make_session()
        for utterance in ["what is the rate for CBC", "is Dr Sen available today",
                          "what time do you open"]:
            assert not is_complaint(utterance)
        assert is_complaint("after all that, I want to file a complaint about the wait")


# --------------------------------------------------------------------- #
# Transport parity
# --------------------------------------------------------------------- #

class TestTransportParity:
    def test_main_dot_py_has_the_complaint_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "complaint"' in src
        assert "is_complaint" in src
        assert "record_complaint_filed" in src
        assert "_finish_complaint" in src

    def test_main_pcm_dot_py_has_the_complaint_branch(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "complaint"' in src
        assert "is_complaint" in src
        assert "record_complaint_filed" in src
        assert "_finish_complaint" in src

    def test_both_files_check_the_guard_inside_continue_pending_too(self):
        for path in ("main.py", "main_pcm.py"):
            src = open(path, encoding="utf-8").read()
            continue_pending_start = src.index("async def _continue_pending(")
            dispatch_turn_start = src.index("async def _dispatch_turn(")
            body = src[continue_pending_start:dispatch_turn_start]
            assert "is_complaint" in body, f"{path}: guard missing from _continue_pending"

    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb


# --------------------------------------------------------------------- #
# agent/llm.py -- schema-only checks (defense-in-depth only; agent/
# complaint_flow.py never lets a complaint turn reach the model at all).
# --------------------------------------------------------------------- #

class TestComplaintIntentSchema:
    def test_complaint_is_a_valid_intent(self):
        from agent.llm import VALID_INTENTS
        assert "complaint" in VALID_INTENTS

    def test_is_distinct_from_out_of_scope_and_human_direct_request(self):
        from agent.llm import VALID_INTENTS
        assert {"out_of_scope", "human_direct_request"} <= VALID_INTENTS
        assert "complaint" not in {"out_of_scope", "human_direct_request"}

    def test_prompt_documents_the_intent_and_the_no_argument_rule(self):
        from agent.llm import SYSTEM_PROMPT_TEMPLATE
        assert '"complaint"' in SYSTEM_PROMPT_TEMPLATE
        assert "NO-ARGUMENT NOTE" in SYSTEM_PROMPT_TEMPLATE

    def test_output_schema_enum_lists_the_new_intent(self):
        from agent.llm import SYSTEM_PROMPT_TEMPLATE
        assert '"complaint"' in SYSTEM_PROMPT_TEMPLATE


# --------------------------------------------------------------------- #
# Regression: ordinary, non-complaint requests behave exactly as before.
# --------------------------------------------------------------------- #

class TestNormalRequestsUnaffected:
    def test_ordinary_test_rate_question_still_reaches_the_fast_path(self, transport, monkeypatch):
        monkeypatch.setattr(transport, "_fast_path", _ExplodingFastPath())
        session = types.SimpleNamespace(call_id="test-normal-call")
        with pytest.raises(AssertionError):
            run(transport._resolve_intent(session, "what is the rate for CBC"))

    def test_out_of_scope_phrasing_does_not_trip_the_complaint_guard(self, transport, monkeypatch):
        # "can you arrange an ambulance" is out_of_scope territory, not a
        # complaint -- proven the same way as the fast-path test just
        # above: if the complaint guard wrongly intercepted it, this would
        # never reach (and explode) the fast path.
        monkeypatch.setattr(transport, "_fast_path", _ExplodingFastPath())
        session = types.SimpleNamespace(call_id="test-normal-call")
        with pytest.raises(AssertionError):
            run(transport._resolve_intent(session, "can you arrange an ambulance"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
