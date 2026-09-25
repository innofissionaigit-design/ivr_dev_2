"""ADDED BY SOURAV -- "Caller is angry about a previous experience" story.

story title: Caller is angry about a previous experience
user story: As a frustrated caller, I want to be heard and offered a
    person, so that my frustration is not compounded by a machine.
acceptance criteria:
    1. Anger lowers the escalation threshold.
    2. Delivery slows.
    3. A human is offered explicitly rather than continuing to transact.
    4. The agent apologises once.
    5. The agent does not argue.
    6. The agent does not defend the hospital.
    7. The complaint/frustration is captured in the context packet.

Structured directly on tests/test_complaint_flow.py's and
tests/test_unverifiable_claim.py's own shape (this story's two closest
siblings, per the delivered investigation report and the implementation
instructions that followed it): an adversarial phrase corpus for the
deterministic detector, a _resolve_intent()/_continue_pending() guard-
order/short-circuit structural proof, a reply-template safety suite, a
dispatch-level check with fake collaborators, a pending-choice
(offer/decline/retry) check, an audit-event check, a pending-flow
interrupt check, a combined-guard-overlap check, and transport parity.

Per this story's own instruction ("run the focused new test suite
first"), this file is meant to be run on its own before the wider
regression list the final report enumerates.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import types

import pytest

import main
import main_pcm
from agent.anger_flow import detect_anger, is_anger
from agent.bn_normalize import detect_language
from agent.complaint_flow import is_complaint
from agent.human_fast_path import is_immediate_human_request
from agent.reply_templates import (
    anger_reply, human_fallback_reply, out_of_scope_counter_reply,
    complaint_acknowledged_reply,
)
from agent.slot_parse import is_affirmative, is_negative
from agent.tools_client import ToolCallError
import agent.outcomes as outcomes_module
from agent.outcomes import (
    human_handoff_counts, HUMAN_ANGER_INTENT, HUMAN_COMPLAINT_INTENT,
    anger_detected_counts, ANGER_DETECTED_EVENT, _reset_for_testing,
)

TRANSPORTS = [main, main_pcm]
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


@pytest.fixture(params=TRANSPORTS, ids=["main", "main_pcm"])
def transport(request):
    return request.param


# ===================================================================== #
# 1. ANGER DETECTION -- agent/anger_flow.py, the deterministic detector
# ===================================================================== #

_EN_CORES = [
    "I am very angry", "I'm very angry", "I am so angry", "I am extremely angry",
    "I am furious", "I am fed up with this", "I'm fed up",
    "you people are useless", "your staff are useless",
    "this is extremely frustrating", "this is so frustrating",
    "I am extremely frustrated", "I have had enough of this", "I've had enough",
    "this is ridiculous", "this is absolutely ridiculous",
    "I am sick of this", "I am sick and tired of this",
    "I am losing my patience", "this is infuriating", "I am outraged",
]
_EN_CARRIERS = ["", "Look, ", "Excuse me, ", "I'm calling because "]

_TRANSLIT_CORES = [
    "mujhe bahut gussa aa raha hai", "main bahut gussa hoon", "main bahut naraz hoon",
    "yeh bahut frustrating hai", "bas bahut ho gaya", "yeh bilkul bakwas hai",
    "aap log bekar hain", "mera sabr khatam ho gaya hai",
    "ami khub raag korchi", "ami onek frustrated", "eta khub frustrating",
    "ami ar shoy korte parchi na", "onek hoyeche ekhon", "tomra kono kajer na",
]
_TRANSLIT_CARRIERS = ["", "Dekho, ", "Shunun, "]

_BN_CORES = [
    "আমি খুব রাগান্বিত", "আমার খুব রাগ হচ্ছে", "আমি প্রচণ্ড রেগে আছি",
    "এটা খুবই বিরক্তিকর", "আমি আর সহ্য করতে পারছি না", "অনেক হয়েছে এখন",
    "আপনারা কোনো কাজের না",
]
_BN_CARRIERS = ["", "শুনুন, ", "দয়া করে, "]


def _corpus(cores, carriers):
    return [carrier + core for core, carrier in itertools.product(cores, carriers)]


_EN_PROMPTS = _corpus(_EN_CORES, _EN_CARRIERS)
_TRANSLIT_PROMPTS = _corpus(_TRANSLIT_CORES, _TRANSLIT_CARRIERS)
_BN_PROMPTS = _corpus(_BN_CORES, _BN_CARRIERS)


class TestDetectAngerPositives:
    def test_corpora_have_a_healthy_number_of_prompts(self):
        assert len(_EN_PROMPTS) >= 60
        assert len(_TRANSLIT_PROMPTS) >= 30
        assert len(_BN_PROMPTS) >= 15

    @pytest.mark.parametrize("prompt", _EN_PROMPTS)
    def test_english_prompt_is_recognised(self, prompt):
        assert is_anger(prompt), f"breach: {prompt!r}"

    @pytest.mark.parametrize("prompt", _TRANSLIT_PROMPTS)
    def test_hinglish_banglish_prompt_is_recognised(self, prompt):
        assert is_anger(prompt), f"breach: {prompt!r}"

    @pytest.mark.parametrize("prompt", _BN_PROMPTS)
    def test_bengali_prompt_is_recognised(self, prompt):
        assert is_anger(prompt), f"breach: {prompt!r}"

    def test_detect_returns_the_matched_language_family(self):
        matched, family = detect_anger("I am very angry")
        assert matched is True and family == "english"
        matched, family = detect_anger("main bahut gussa hoon")
        assert matched is True and family == "latin_translit"
        matched, family = detect_anger("আমি খুব রাগান্বিত")
        assert matched is True and family == "bengali"


class TestDetectAngerNegatives:
    """The guard must NOT treat ordinary dissatisfaction or ordinary
    requests as anger -- this story's own explicit constraint."""

    @pytest.mark.parametrize("text", [
        "what is the rate for CBC", "is Dr. Sen available tomorrow",
        "what time do you open", "I want to book an appointment",
        "this is bad", "I have an issue", "I am not happy",
        "why is this taking so long", "I am a little annoyed",
        "can you check my report", "I have a complaint about the wait time",
        "amar ekta complaint ache", "আমার একটা অভিযোগ আছে",
        "connect me to a human",
        "", "   ",
    ])
    def test_ordinary_and_adjacent_phrases_are_never_intercepted(self, text):
        assert not is_anger(text), f"false positive: {text!r}"

    def test_none_text_never_crashes(self):
        assert detect_anger(None) == (False, None)

    def test_bengali_word_boundary_safety(self):
        assert not is_anger("মানুষটি কেমন আছেন")


# ===================================================================== #
# 2/3. SPEECH + ESCALATION -- caller_state activation via _finish_anger()
# ===================================================================== #

class _ExplodingToolsClient:
    def __getattr__(self, name):
        def _explode(*args, **kwargs):
            raise AssertionError(f"clinic-api tool method {name!r} must never be "
                                  "called for an anger turn other than submit_complaint")
        return _explode

    async def submit_complaint(self, complaint_text, phone=None):
        return {"success": True, "complaint_id": "CMP-ANGER-TEST", "phone": phone,
                "status": "pending"}


def make_anger_session(pending=None, transport=main):
    return types.SimpleNamespace(
        call_id="test-anger-call", pending=pending,
        dispatch_lock=asyncio.Lock(), call_state=transport.call_state_mod.build(),
        utt_seq=1, confirm_attempts=0, anger_apologized=False,
        send_json=_AsyncNoOp(),
    )


class TestCallStateActivation:
    def test_finish_anger_sets_caller_state_angry(self, transport, monkeypatch):
        monkeypatch.setattr(transport, "_speak", _AsyncNoOp())
        monkeypatch.setattr(transport, "_tools", _ExplodingToolsClient())
        session = make_anger_session(transport=transport)
        run(transport._finish_anger(session, "I am very angry", language="english"))
        assert session.call_state.caller_state == "angry"

    def test_finish_anger_activates_slow_normal_speech_rate(self, transport, monkeypatch):
        # AC 2: delivery slows. Reuses the existing, already-dormant
        # "angry" row in responses/speech_policy.py -- no new speech-rate
        # mechanism.
        monkeypatch.setattr(transport, "_speak", _AsyncNoOp())
        monkeypatch.setattr(transport, "_tools", _ExplodingToolsClient())
        session = make_anger_session(transport=transport)
        run(transport._finish_anger(session, "I am very angry", language="english"))
        assert session.call_state.speech_rate == "slow_normal"

    def test_finish_anger_sets_escalation_threshold_low(self, transport, monkeypatch):
        # AC 1: the field itself is honestly populated even though (per
        # the investigation) the real behavioural effect is the guard's
        # own immediate-offer eligibility rule, proven separately below.
        monkeypatch.setattr(transport, "_speak", _AsyncNoOp())
        monkeypatch.setattr(transport, "_tools", _ExplodingToolsClient())
        session = make_anger_session(transport=transport)
        run(transport._finish_anger(session, "I am very angry", language="english"))
        assert session.call_state.escalation_threshold == "low"

    def test_the_speech_rate_this_story_writes_is_what_speak_already_reads(self, transport, monkeypatch):
        # main.py's own _speak() already reads session.call_state.speech_rate
        # on every turn (pre-existing, live code path -- see its own
        # docstring) -- this proves, at the source level, that it is
        # reading the SAME attribute this story's _finish_anger() writes,
        # without re-exercising _speak()'s own TTS/gate machinery (out of
        # scope for this story, and already covered by that function's own
        # existing tests).
        import inspect
        src = inspect.getsource(transport._speak)
        assert "session.call_state.speech_rate" in src

        monkeypatch.setattr(transport, "_speak", _AsyncNoOp())
        monkeypatch.setattr(transport, "_tools", _ExplodingToolsClient())
        session = make_anger_session(transport=transport)
        run(transport._finish_anger(session, "I am very angry", language="english"))
        assert session.call_state.speech_rate == "slow_normal"

    def test_senior_and_language_are_carried_forward_not_reset(self, transport, monkeypatch):
        monkeypatch.setattr(transport, "_speak", _AsyncNoOp())
        monkeypatch.setattr(transport, "_tools", _ExplodingToolsClient())
        session = make_anger_session(transport=transport)
        session.call_state = transport.call_state_mod.build(
            caller_state="unknown", senior=True, language="bn-IN")
        run(transport._finish_anger(session, "I am very angry", language="english"))
        assert session.call_state.senior is True
        assert session.call_state.language == "bn-IN"

    def test_non_angry_normal_caller_call_state_is_unaffected(self, transport):
        # Regression: a caller who never triggers the guard keeps the
        # normal row untouched.
        session = make_anger_session(transport=transport)
        assert session.call_state.caller_state == "unknown"
        assert session.call_state.speech_rate == "default"
        assert session.call_state.escalation_threshold == "normal"


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


# ===================================================================== #
# 4. APOLOGY -- once per call, not once per angry utterance
# ===================================================================== #

class TestApologyOnce:
    def test_first_angry_utterance_speaks_the_apology_variant(self, transport, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(transport, "_speak", fake_speak)
        monkeypatch.setattr(transport, "_tools", _ExplodingToolsClient())
        session = make_anger_session(transport=transport)
        run(transport._finish_anger(session, "I am very angry", language="english"))
        assert spoken == [anger_reply(False, language="english")]
        assert spoken != [anger_reply(True, language="english")]

    def test_second_angry_utterance_same_call_does_not_repeat_the_apology(self, transport, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(transport, "_speak", fake_speak)
        monkeypatch.setattr(transport, "_tools", _ExplodingToolsClient())
        session = make_anger_session(transport=transport)
        run(transport._finish_anger(session, "I am very angry", language="english"))
        run(transport._finish_anger(session, "this is ridiculous", language="english"))
        assert spoken[0] == anger_reply(False, language="english")
        assert spoken[1] == anger_reply(True, language="english")
        assert spoken[1] != spoken[0]

    def test_repeated_anger_never_repeats_the_apology_across_many_turns(self, transport, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(transport, "_speak", fake_speak)
        monkeypatch.setattr(transport, "_tools", _ExplodingToolsClient())
        session = make_anger_session(transport=transport)
        for utterance in ["I am very angry", "this is ridiculous", "I am fed up with this"]:
            run(transport._finish_anger(session, utterance, language="english"))
        assert spoken[0] == anger_reply(False, language="english")
        assert spoken[1] == anger_reply(True, language="english")
        assert spoken[2] == anger_reply(True, language="english")

    def test_human_offer_is_still_made_on_every_repeat_never_weakened(self, transport, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(transport, "_speak", fake_speak)
        monkeypatch.setattr(transport, "_tools", _ExplodingToolsClient())
        session = make_anger_session(transport=transport)
        run(transport._finish_anger(session, "I am very angry", language="english"))
        run(transport._finish_anger(session, "this is ridiculous", language="english"))
        # Both replies must still contain an offer -- never silence.
        for text in spoken:
            assert "staff" in text.lower() or "counter" in text.lower() or "connect" in text.lower()

    def test_apology_flag_lives_on_the_session_not_call_state(self, transport, monkeypatch):
        monkeypatch.setattr(transport, "_speak", _AsyncNoOp())
        monkeypatch.setattr(transport, "_tools", _ExplodingToolsClient())
        session = make_anger_session(transport=transport)
        assert session.anger_apologized is False
        run(transport._finish_anger(session, "I am very angry", language="english"))
        assert session.anger_apologized is True
        assert not hasattr(session.call_state, "anger_apologized")


# ===================================================================== #
# 5/6. GUARD ORDER -- anger never reaches the LLM/fast_path/cache, so it
# never argues or defends the hospital (deterministic fast path only)
# ===================================================================== #

class _ExplodingFastPath:
    def resolve(self, text):
        raise AssertionError("fast_path.resolve() must not run for an anger turn")


class _ExplodingCache:
    def get(self, text):
        raise AssertionError("semantic cache must not run for an anger turn")

    def put(self, text, data):
        raise AssertionError("semantic cache must not run for an anger turn")


def _exploding_extract_intent(text):
    raise AssertionError("extract_intent() (the LLM) must not run for an anger turn")


class TestGuardOrderInResolveIntent:
    @pytest.fixture(autouse=True)
    def _wire_exploding_stand_ins(self, transport, monkeypatch):
        monkeypatch.setattr(transport, "_fast_path", _ExplodingFastPath())
        monkeypatch.setattr(transport, "_intent_cache", _ExplodingCache())
        monkeypatch.setattr(transport, "extract_intent", _exploding_extract_intent)
        self.transport = transport

    @pytest.mark.parametrize("prompt", _EN_PROMPTS[:15] + _TRANSLIT_PROMPTS[:10] + _BN_PROMPTS[:10])
    def test_short_circuits_before_any_classifier_runs(self, prompt):
        session = types.SimpleNamespace(call_id="test-anger-guard-call")
        data = run(self.transport._resolve_intent(session, prompt))
        assert data["intent"] == "caller_angry"
        assert data["parts"] == [{"intent": "caller_angry", "slots": data["slots"]}]
        assert data["direct_reply_bn"] is None

    def test_an_ordinary_question_still_reaches_the_fast_path(self):
        session = types.SimpleNamespace(call_id="test-anger-guard-call")
        with pytest.raises(AssertionError):
            run(self.transport._resolve_intent(session, "what is the rate for CBC"))


_FORBIDDEN_ARGUE_OR_DEFEND_ENGLISH = [
    "that is not our fault", "our staff did nothing wrong",
    "we did everything correctly", "you are mistaken", "that is not true",
    "our hospital is one of the best", "we never make mistakes",
    "let me explain why", "actually,", "in our defense",
]


class TestAngerReplyNeverArguesOrDefends:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_returns_nonempty_text(self, language):
        assert anger_reply(False, language=language).strip()
        assert anger_reply(True, language=language).strip()

    def test_default_language_is_bengali(self):
        assert anger_reply(False) == anger_reply(False, language="bengali")

    def test_first_and_repeat_variants_are_textually_distinct(self):
        assert anger_reply(False, language="english") != anger_reply(True, language="english")

    def test_never_contains_forbidden_argue_or_defend_language(self):
        for already in (False, True):
            text = anger_reply(already, language="english").lower()
            for phrase in _FORBIDDEN_ARGUE_OR_DEFEND_ENGLISH:
                assert phrase not in text, f"forbidden phrase found: {phrase!r} in {text!r}"

    def test_offers_a_human_or_the_counter(self):
        for already in (False, True):
            text = anger_reply(already, language="english").lower()
            assert "staff" in text or "counter" in text or "connect" in text

    def test_does_not_claim_a_live_transfer_it_cannot_make(self):
        # Truthful wording -- the same fixed offer sentence every sibling
        # guard-offered choice already speaks, never "transferring you now".
        for already in (False, True):
            text = anger_reply(already, language="english").lower()
            assert "transferring you" not in text
            assert "connecting you now" not in text

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_is_spoken_punctuation_clean(self, language):
        from agent.bn_normalize import verbalize
        for already in (False, True):
            spoken = verbalize(anger_reply(already, language=language), language=language)
            for ch in (":", "：", "[", "]", "{", "}"):
                assert ch not in spoken


# ===================================================================== #
# 3/7. HUMAN OFFER + CONTEXT CAPTURE -- dispatch level
# ===================================================================== #

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
        return {"success": True, "complaint_id": "CMP-ANGER-TEST",
                "phone": phone, "status": "pending"}

    async def get_test_rate(self, *a, **k):
        self._record("get_test_rate", *a)
        return {"found": False}

    async def get_clinic_info(self, *a, **k):
        self._record("get_clinic_info", *a)
        return {"found": False}


class FakeASRResultAngry:
    text = "I am very angry about how I was treated last time"
    decoder_used = "ctc"
    decoder_agreement = 1.0
    ctc_words = 10
    rnnt_words = 10


def make_session(pending=None, transport=main):
    return types.SimpleNamespace(
        call_id="test-anger-call", pending=pending,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
        call_state=transport.call_state_mod.build(), utt_seq=1,
        confirm_attempts=0, anger_apologized=False,
    )


@pytest.fixture
def stub(transport, monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    class FakeASR:
        async def transcribe_utterance(self, wav_path):
            return FakeASRResultAngry()

    monkeypatch.setattr(transport, "_speak", fake_speak)
    monkeypatch.setattr(transport, "_asr", FakeASR())
    tools = FakeToolsClient()
    monkeypatch.setattr(transport, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools, transport=transport)


def _dispatch_initial(stub, monkeypatch, asr_text_cls=FakeASRResultAngry, tmp_path=None):
    async def fake_resolve_intent(session, text):
        return {"intent": "caller_angry", "slots": {}}

    class FakeASR:
        async def transcribe_utterance(self, wav_path):
            return asr_text_cls()

    monkeypatch.setattr(stub.transport, "_asr", FakeASR())
    monkeypatch.setattr(stub.transport, "_resolve_intent", fake_resolve_intent)

    session = make_session(transport=stub.transport)
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(stub.transport._dispatch_turn(session, str(wav_path)))
    return session


class TestAngerIntentDispatch:
    def test_speaks_the_apology_plus_offer_reply(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken == [anger_reply(False, language="english")]

    def test_opens_the_angry_choice_pending_state(self, stub, monkeypatch, tmp_path):
        session = _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert session.pending == {"awaiting": "angry_choice", "retries": 0}

    def test_captures_the_utterance_verbatim_via_submit_complaint(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        calls = [c for c in stub.tools.calls if c[0] == "submit_complaint"]
        assert len(calls) == 1
        assert calls[0][1][0] == FakeASRResultAngry.text

    def test_no_other_tool_is_called(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        called = [c[0] for c in stub.tools.calls]
        assert called == ["submit_complaint"]

    def test_records_the_anger_detected_audit_event(self, stub, monkeypatch, tmp_path, _isolated_escalation_log):
        assert anger_detected_counts() == 0
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert anger_detected_counts() == 1
        entry = json.loads(_isolated_escalation_log.read_text(encoding="utf-8").strip())
        assert entry["event"] == ANGER_DETECTED_EVENT
        assert entry["intent"] == HUMAN_ANGER_INTENT
        assert entry["call_id"] == "test-anger-call"
        assert FakeASRResultAngry.text not in json.dumps(entry)

    def test_detection_alone_does_not_yet_bump_the_handoff_counter(self, stub, monkeypatch, tmp_path):
        # The handoff counter only bumps on a CONFIRMED ("yes") answer to
        # the offer, in the "angry_choice" pending handler below -- kept
        # separate so detection and confirmed escalation are never
        # double-counted under one tag.
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert human_handoff_counts(HUMAN_ANGER_INTENT) == 0

    def test_respects_the_detected_language_of_the_utterance(self, stub, monkeypatch, tmp_path):
        class FakeASRBanglish:
            text = "ami khub frustrated eta niye"
            decoder_used = "ctc"
            decoder_agreement = 1.0
            ctc_words = 5
            rnnt_words = 5
        _dispatch_initial(stub, monkeypatch, asr_text_cls=FakeASRBanglish, tmp_path=tmp_path)
        expected_language = detect_language(FakeASRBanglish.text)
        assert stub.spoken == [anger_reply(False, language=expected_language)]

    def test_tool_failure_still_speaks_the_reply_and_offers_a_human(self, stub, monkeypatch, tmp_path):
        # AC 3 must hold even when the context-capture write fails.
        stub.tools._complaint_exception = ToolCallError("db down")
        session = _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken == [anger_reply(False, language="english")]
        assert session.pending == {"awaiting": "angry_choice", "retries": 0}


# ===================================================================== #
# "angry_choice" pending-state handler
# ===================================================================== #

class TestAngryChoiceHandler:
    def test_affirmative_hands_off_to_a_human_and_records_the_handoff(self, transport, monkeypatch, _isolated_escalation_log):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(transport, "_speak", fake_speak)
        session = make_session(pending={"awaiting": "angry_choice", "retries": 0}, transport=transport)
        handled = run(transport._continue_pending(session, "yes"))

        assert handled is True
        assert session.pending is None
        assert spoken == [human_fallback_reply(language="english")]
        assert human_handoff_counts(HUMAN_ANGER_INTENT) == 1
        record = json.loads(_isolated_escalation_log.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert record["event"] == "human_handoff"
        assert record["intent"] == HUMAN_ANGER_INTENT

    def test_negative_reuses_the_out_of_scope_counter_reply(self, transport, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(transport, "_speak", fake_speak)
        session = make_session(pending={"awaiting": "angry_choice", "retries": 0}, transport=transport)
        handled = run(transport._continue_pending(session, "no"))

        assert handled is True
        assert session.pending is None
        assert spoken == [out_of_scope_counter_reply(language="english")]
        assert human_handoff_counts(HUMAN_ANGER_INTENT) == 0

    def test_no_false_live_transfer_promise_on_the_affirmative_branch(self, transport, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(transport, "_speak", fake_speak)
        session = make_session(pending={"awaiting": "angry_choice", "retries": 0}, transport=transport)
        run(transport._continue_pending(session, "yes"))
        assert spoken == [human_fallback_reply(language="english")]

    def test_an_unclear_answer_retries_without_reapologising_then_falls_through(self, transport, monkeypatch):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        monkeypatch.setattr(transport, "_speak", fake_speak)
        session = make_session(pending={"awaiting": "angry_choice", "retries": 0}, transport=transport)

        handled = run(transport._continue_pending(session, "what do you mean"))
        assert handled is True
        assert session.pending["retries"] == 1
        assert spoken == [anger_reply(True, language="english")]

        session.pending["retries"] = 3
        handled = run(transport._continue_pending(session, "what do you mean"))
        assert handled is False
        assert session.pending is None


# ===================================================================== #
# 9. PENDING-FLOW INTERRUPTION
# ===================================================================== #

def _continue(stub, monkeypatch, pending, reply_text):
    session = make_session(pending=dict(pending), transport=stub.transport)
    handled = run(stub.transport._continue_pending(session, reply_text))
    return handled, stub.spoken, session


class TestAngerInterruptsPendingFlow:
    def test_interrupts_a_booking_flow(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "date", "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending, "I am very angry about this whole process")
        assert handled is True
        assert session.pending == {"awaiting": "angry_choice", "retries": 0}
        assert spoken == [anger_reply(False, language="english")]

    def test_interrupts_a_report_flow(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "report_phone", "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending, "this is ridiculous, I am so angry")
        assert handled is True
        assert session.pending["awaiting"] == "angry_choice"

    def test_interrupts_a_billing_flow(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "billing_phone", "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending, "I am furious about this bill")
        assert handled is True
        assert session.pending["awaiting"] == "angry_choice"

    def test_interrupts_a_doctor_lookup_flow(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "doctor_choice", "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending, "I am fed up with this")
        assert handled is True
        assert session.pending["awaiting"] == "angry_choice"

    def test_does_not_finish_the_flow_it_interrupted(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "confirm_callback", "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending, "I am very angry")
        assert handled is True
        assert [c[0] for c in stub.tools.calls] == ["submit_complaint"]

    def test_an_ordinary_mid_flow_reply_is_not_swallowed_by_this_guard(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "confirm_booking", "slots": {}, "candidates": None,
                   "offered_date": None, "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending, "না")
        assert [c[0] for c in stub.tools.calls] == []


# ===================================================================== #
# 8/COMBINED CASES -- anger overlapping other guards
# ===================================================================== #

class TestCombinedCases:
    def test_anger_plus_complaint_routes_to_complaint_not_anger(self, transport, monkeypatch):
        # Instruction 8: "anger should ENHANCE complaint flow rather than
        # create two competing flows" -- implemented entirely as guard
        # ORDER (is_complaint checked before is_anger), proven here both
        # at the detector level and at _resolve_intent()'s own dispatch
        # decision.
        text = "I am very angry and I want to file a complaint about the front desk"
        assert is_complaint(text) is True
        assert is_anger(text) is True  # would fire in isolation --
        # proving guard ORDER, not phrase-list luck, resolves the overlap.
        session = types.SimpleNamespace(call_id="test-anger-guard-call")
        data = run(transport._resolve_intent(session, text))
        assert data["intent"] == "complaint"

    def test_anger_plus_complaint_never_double_acknowledges_or_double_routes(self, monkeypatch, tmp_path):
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append(text)

        async def fake_resolve_intent(session, text):
            return {"intent": "complaint", "slots": {}}

        class FakeASR:
            async def transcribe_utterance(self, wav_path):
                return types.SimpleNamespace(
                    text="I am so angry, I have a complaint about the wait time",
                    decoder_agreement=1.0, decoder_used="ctc", ctc_words=10, rnnt_words=10)

        tools = FakeToolsClient()
        monkeypatch.setattr(main, "_speak", fake_speak)
        monkeypatch.setattr(main, "_asr", FakeASR())
        monkeypatch.setattr(main, "_resolve_intent", fake_resolve_intent)
        monkeypatch.setattr(main, "_tools", tools)

        session = make_session(transport=main)
        wav_path = tmp_path / "utt.wav"
        wav_path.write_bytes(b"")
        run(main._dispatch_turn(session, str(wav_path)))

        # Exactly one acknowledgement (complaint's own), never anger's own
        # apology on top of it, and no competing pending state opened.
        assert spoken == [complaint_acknowledged_reply(language="english")]
        assert session.pending is None
        assert [c[0] for c in tools.calls] == ["submit_complaint"]

    def test_anger_plus_explicit_human_request_routes_to_human_direct_request(self, transport):
        # is_immediate_human_request() is checked before is_complaint()
        # and is_anger() in _resolve_intent()'s own fixed guard order --
        # an explicit ask for a human is the more absolute of the two and
        # wins outright, same reasoning as complaint-vs-anger above.
        text = "I am so angry, connect me to a human right now"
        assert is_immediate_human_request(text) is True
        session = types.SimpleNamespace(call_id="test-anger-guard-call")
        data = run(transport._resolve_intent(session, text))
        assert data["intent"] == "human_direct_request"

    def test_anger_plus_an_ordinary_request_still_fires_anger_first(self, transport):
        # Per instruction 9, anger stops normal transaction handling
        # outright rather than answering the ordinary part first.
        text = "I am very angry, I want to book an appointment with Dr Sen"
        session = types.SimpleNamespace(call_id="test-anger-guard-call")
        data = run(transport._resolve_intent(session, text))
        assert data["intent"] == "caller_angry"

    def test_repeated_anger_across_several_turns_keeps_offering_a_human(self, stub, monkeypatch, tmp_path):
        session = make_session(transport=stub.transport)
        for utterance in ["I am very angry", "this is ridiculous", "I am fed up with this"]:
            run(stub.transport._finish_anger(session, utterance, language="english"))
        assert session.pending == {"awaiting": "angry_choice", "retries": 0}
        assert len(stub.spoken) == 3
        assert stub.spoken[0] == anger_reply(False, language="english")
        assert all(r == anger_reply(True, language="english") for r in stub.spoken[1:])


# ===================================================================== #
# Regression: existing guards keep priority; complaint/human/clinical
# flows are unaffected by the new guard.
# ===================================================================== #

class TestExistingGuardsStillWinWhereTheyShould:
    def test_clinical_interpretation_still_wins_over_an_angry_sounding_utterance(self, transport):
        text = "I am so angry, is this serious, am I dying"
        from agent.clinical_safety import is_clinical_interpretation
        assert is_clinical_interpretation(text) is True
        session = types.SimpleNamespace(call_id="test-anger-guard-call")
        data = run(transport._resolve_intent(session, text))
        assert data["intent"] == "clinical_interpretation"

    def test_complaint_is_unaffected_by_the_new_guard(self, transport):
        session = types.SimpleNamespace(call_id="test-anger-guard-call")
        data = run(transport._resolve_intent(session, "I want to make a complaint"))
        assert data["intent"] == "complaint"

    def test_ordinary_test_rate_question_still_reaches_the_fast_path(self, transport, monkeypatch):
        monkeypatch.setattr(transport, "_fast_path", _ExplodingFastPath())
        session = types.SimpleNamespace(call_id="test-anger-guard-call")
        with pytest.raises(AssertionError):
            run(transport._resolve_intent(session, "what is the rate for CBC"))

    def test_confirm_attempts_escalation_path_is_untouched_for_non_angry_callers(self):
        # The pre-existing 3-strikes low-ASR-confidence escalation
        # (session.confirm_attempts) is a completely separate mechanism
        # from this story's own guard -- proven here it still exists with
        # its own threshold, unmodified.
        import inspect
        src = inspect.getsource(main)
        assert "session.confirm_attempts > 2" in src


# ===================================================================== #
# agent/llm.py -- schema-only checks (defense-in-depth only; the guard
# never lets an angry turn reach the model at all).
# ===================================================================== #

class TestAngerIsNeverAModelClassifiedIntent:
    def test_caller_angry_is_not_a_registered_llm_intent(self):
        from agent.llm import VALID_INTENTS
        assert "caller_angry" not in VALID_INTENTS

    def test_llm_prompt_now_disambiguates_anger_from_smalltalk(self):
        from agent.llm import SYSTEM_PROMPT_TEMPLATE
        assert "never write your own apology, argument, defence of the hospital" in SYSTEM_PROMPT_TEMPLATE.lower() \
            or "anger_flow" in SYSTEM_PROMPT_TEMPLATE.lower()


# ===================================================================== #
# Transport parity
# ===================================================================== #

class TestTransportParity:
    def test_main_dot_py_has_the_anger_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "caller_angry"' in src
        assert "is_anger" in src
        assert "_finish_anger" in src
        assert "record_anger_detected" in src
        assert '"angry_choice"' in src

    def test_main_pcm_dot_py_has_the_anger_branch(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "caller_angry"' in src
        assert "is_anger" in src
        assert "_finish_anger" in src
        assert "record_anger_detected" in src
        assert '"angry_choice"' in src

    def test_both_files_check_the_guard_inside_continue_pending_too(self):
        for path in ("main.py", "main_pcm.py"):
            src = open(path, encoding="utf-8").read()
            continue_pending_start = src.index("async def _continue_pending(")
            dispatch_turn_start = src.index("async def _dispatch_turn(")
            body = src[continue_pending_start:dispatch_turn_start]
            assert "is_anger" in body, f"{path}: guard missing from _continue_pending"

    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
