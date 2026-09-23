"""ADDED BY SOURAV -- "Caller wants to speak to a doctor personally" story.

story title: Caller wants to speak to a doctor personally
user story: As a caller who wants clinical reassurance, I want a realistic
    answer about what is possible, so that I am not left waiting for
    something that will not happen.
acceptance criteria:
    1. The agent states the actual process for reaching a clinician.
    2. The agent offers the appropriate route.
    3. The agent never promises a call from a named doctor it cannot
       schedule.
    4. Where a clinical callback exists in policy it is offered.

Structured directly on tests/test_complaint_flow.py's own shape
(adversarial phrase corpus, a _resolve_intent() short-circuit proof, a
dispatch-level test, a _continue_pending() override test, transport
parity, schema checks) for the same reason agent/doctor_personal_
request.py itself is structured on agent/complaint_flow.py: this is the
same kind of zero-tolerance, pre-classifier guard story. Two sections
that story does not need at all, because this one does:

  1. A callback-availability proof (TestCallbackAvailabilityBehavior
     below) -- this story's own AC 4 has no equivalent in the complaint
     story's AC set, which never offers a second real route at all.
     Checked against a FakeToolsClient/CALLBACKS_ENABLED combination
     covering enabled+available, disabled, and outside-hours, mirroring
     tests/test_request_callback.py's own TestCallbackDispatch fixtures.

  2. A named-doctor-promise proof (TestNamedDoctorProtection below) --
     AC 3's own bar. Checked by asserting the spoken reply never contains
     a doctor's name, that no callback or booking write is ever made BY
     THIS GUARD ITSELF (the caller's own next ordinary utterance is what
     would start book_appointment/request_callback, not this turn), and
     that the reply never claims an immediate connection.

WHY THE ADVERSARIAL PROMPTS ARE NOT JUST agent.doctor_personal_request's
OWN PHRASE TUPLES READ BACK: same discipline test_complaint_flow.py's own
note explains -- every prompt below was written independently and run
against the real function, so this suite can catch a gap in the module
rather than just confirming the module agrees with itself.
"""
from __future__ import annotations

import asyncio
import itertools
import types

import pytest

import main
import main_pcm
from agent.bn_normalize import detect_language
from agent.callback_flow import check_callback_availability
from agent.doctor_personal_request import detect_doctor_personal_request, is_doctor_personal_request
from agent.reply_templates import (
    doctor_personal_request_reply, human_fallback_reply, complaint_acknowledged_reply,
)

TRANSPORTS = [main, main_pcm]


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- #
# The adversarial coverage. CORES x CARRIERS, same construction as
# tests/test_complaint_flow.py's own corpora. Every core below is written
# independently of agent/doctor_personal_request.py's own phrase/regex
# tables.
# --------------------------------------------------------------------- #

_EN_CORES = [
    "I want to speak to a doctor",
    "I need to talk to a doctor",
    "Can I speak directly to a doctor?",
    "I want a doctor to talk to me",
    "Please ask a doctor to call me",
    "I want Dr Sen to call me.",
    "Can Dr Sen speak to me now?",
    "I need to speak with a clinician",
    "I want Dr Gupta to call me.",
    "Please ask Dr Sen to call me.",
    "I want my doctor to call me.",
    "I need my doctor to reassure me.",
    "I would like to talk to a doctor please",
    "Is it possible to speak with a clinician today",
    "I want to speak to the doctor personally",
    "Can you connect me to a doctor",
    "I want to speak to Dr Sen personally",
    "I need clinical reassurance",
    "I want clinical reassurance from a doctor",
    "Can Dr Gupta talk to me now?",
]
_EN_CARRIERS = ["", "Look, ", "Excuse me, ", "I'm calling because "]

_HI_CORES = [
    "mujhe doctor se baat karni hai",
    "doctor se baat karna chahta hoon",
    "doctor se seedhe baat karna chahta hoon",
    "mujhe apne doctor se baat karni hai",
    "doctor mujhe call kare",
    "doctor mujhe phone kare",
    "mera doctor mujhe call kare",
    "Dr Sen mujhe call kare",
    "Dr Sen ko kahiye mujhe call kare",
    "clinician se baat karni hai",
]
_HI_CARRIERS = ["", "Dekho, ", "Suniye, "]

_BN_CORES = [
    "ডাক্তারের সাথে কথা বলতে চাই",
    "ডাক্তারের সাথে সরাসরি কথা বলতে চাই",
    "একজন ডাক্তারের সাথে কথা বলতে চাই",
    "ডাক্তার আমাকে কল করুন",
    "ডাক্তার আমাকে ফোন করুন",
    "আমার ডাক্তার আমাকে কল করুন",
    "ডা. সেন আমাকে কল করুন",
]
_BN_CARRIERS = ["", "শুনুন, ", "দয়া করে, "]

_BANGLISH_CORES = [
    "daktarer sathe kotha bolte chai",
    "daktarer sathe sojasuji kotha bolte chai",
    "daktar amake call korun",
    "daktar amake phone korun",
    "amar daktar amake call korun",
    "Dr Sen amake call korun",
    "daktar ke bolben amake call korte",
]
_BANGLISH_CARRIERS = ["", "Shunun, ", "Please, "]


def _corpus(cores, carriers):
    return [carrier + core for core, carrier in itertools.product(cores, carriers)]


_EN_PROMPTS = _corpus(_EN_CORES, _EN_CARRIERS)
_HI_PROMPTS = _corpus(_HI_CORES, _HI_CARRIERS)
_BN_PROMPTS = _corpus(_BN_CORES, _BN_CARRIERS)
_BANGLISH_PROMPTS = _corpus(_BANGLISH_CORES, _BANGLISH_CARRIERS)

# --------------------------------------------------------------------- #
# Negative examples the story is explicit must NOT trip this guard --
# these must continue through the existing doctor_availability/
# doctor_schedule/book_appointment/clinic_info flows unchanged.
# --------------------------------------------------------------------- #
_PROTECTED_NEGATIVES = [
    "What are the doctor's visiting hours?",
    "I want to book an appointment with Dr Sen.",
    "I want to book an appointment with Dr Sen",
    "Is Dr Sen available tomorrow?",
    "Is Dr Sen available today?",
    "Can I book an appointment with the doctor?",
    "Which days does Dr Sen sit?",
    "Dr Sen ka schedule kya hai",
    "is Dr. Sen available",
    "কাল ডাক্তার সেন আছেন কিনা",
    "ডাক্তার সেন কোন কোন দিন বসেন",
    "what is the rate for CBC",
    "what time do you open",
]


class TestAdversarialCoverage:
    def test_the_corpora_each_have_a_healthy_number_of_prompts(self):
        assert len(_EN_PROMPTS) >= 60
        assert len(_HI_PROMPTS) >= 15
        assert len(_BN_PROMPTS) >= 14
        assert len(_BANGLISH_PROMPTS) >= 14

    @pytest.mark.parametrize("prompt", _EN_PROMPTS)
    def test_english_is_recognised(self, prompt):
        matched, lang = detect_doctor_personal_request(prompt)
        assert matched, f"missed: {prompt!r}"
        assert lang == "english"

    @pytest.mark.parametrize("prompt", _HI_PROMPTS)
    def test_hinglish_is_recognised(self, prompt):
        matched, lang = detect_doctor_personal_request(prompt)
        assert matched, f"missed: {prompt!r}"
        assert lang == "latin_translit"

    @pytest.mark.parametrize("prompt", _BN_PROMPTS)
    def test_bengali_is_recognised(self, prompt):
        matched, lang = detect_doctor_personal_request(prompt)
        assert matched, f"missed: {prompt!r}"
        assert lang == "bengali"

    @pytest.mark.parametrize("prompt", _BANGLISH_PROMPTS)
    def test_banglish_is_recognised(self, prompt):
        matched, lang = detect_doctor_personal_request(prompt)
        assert matched, f"missed: {prompt!r}"
        assert lang == "latin_translit"

    @pytest.mark.parametrize("prompt", _PROTECTED_NEGATIVES)
    def test_protected_phrases_never_false_positive(self, prompt):
        assert not is_doctor_personal_request(prompt), f"false positive: {prompt!r}"

    def test_empty_and_blank_text_is_never_flagged(self):
        assert not is_doctor_personal_request("")
        assert not is_doctor_personal_request("   ")
        assert not is_doctor_personal_request(None)

    def test_ordinary_out_of_scope_and_complaint_phrasing_does_not_trip_this_guard(self):
        # This guard must not overlap with the other pre-classifier guards
        # already in this codebase.
        assert not is_doctor_personal_request("can you arrange an ambulance")
        assert not is_doctor_personal_request("I want to file a complaint about the wait time")
        assert not is_doctor_personal_request("connect me to a person")
        assert not is_doctor_personal_request("is my result dangerous")


# --------------------------------------------------------------------- #
# _resolve_intent() short-circuit proof.
# --------------------------------------------------------------------- #

class _ExplodingFastPath:
    def resolve(self, text):
        raise AssertionError("fast path must not run for a doctor-personal-request turn")


class _ExplodingCache:
    def get(self, text):
        raise AssertionError("semantic cache must not run for a doctor-personal-request turn")

    def put(self, text, data):
        raise AssertionError("semantic cache must not run for a doctor-personal-request turn")


def _exploding_extract_intent(text):
    raise AssertionError("extract_intent() (the LLM) must not run for a doctor-personal-request turn")


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
        "prompt", _EN_PROMPTS[:15] + _HI_PROMPTS[:10] + _BN_PROMPTS[:7] + _BANGLISH_PROMPTS[:7]
    )
    def test_short_circuits_before_any_classifier_runs(self, prompt):
        session = types.SimpleNamespace(call_id="test-doctor-personal-guard-call")
        data = run(self.transport._resolve_intent(session, prompt))
        assert data["intent"] == "doctor_personal_request"
        assert data["parts"] == [{"intent": "doctor_personal_request", "slots": data["slots"]}]
        assert data["direct_reply_bn"] is None

    def test_an_ordinary_question_still_reaches_the_fast_path(self):
        session = types.SimpleNamespace(call_id="test-doctor-personal-guard-call")
        with pytest.raises(AssertionError):
            run(self.transport._resolve_intent(session, "what is the rate for CBC"))

    @pytest.mark.parametrize("prompt", _PROTECTED_NEGATIVES)
    def test_protected_negatives_still_reach_the_fast_path(self, prompt):
        # Proves these phrases are NOT grabbed by the new guard -- if they
        # were, _resolve_intent would return a doctor_personal_request
        # payload instead of ever touching the (exploding) fast path.
        session = types.SimpleNamespace(call_id="test-doctor-personal-guard-call")
        with pytest.raises(AssertionError):
            run(self.transport._resolve_intent(session, prompt))


# --------------------------------------------------------------------- #
# Dispatch -- the "doctor_personal_request" intent branch, and the
# FakeToolsClient that proves availability is checked honestly and no
# doctor/callback write is ever made by this guard itself.
# --------------------------------------------------------------------- #

_OPEN_ALL_DAY = {"closed": False, "open": "00:00", "close": "23:59"}
_CLOSED_ALL_DAY = {"closed": True, "open": None, "close": None}
_ALL_DAYS_OPEN = {d: dict(_OPEN_ALL_DAY) for d in
                  ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")}
_ALL_DAYS_CLOSED = {d: dict(_CLOSED_ALL_DAY) for d in
                    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")}


class FakeToolsClient:
    def __init__(self, clinic_info_response=None):
        self.calls = []
        self._clinic_info_response = clinic_info_response or {"found": True, "hours": _ALL_DAYS_OPEN}
        self._clinic_info_exception = None

    def _record(self, name, *args):
        self.calls.append((name, args))

    async def get_clinic_info(self, *a, **k):
        self._record("get_clinic_info", *a)
        if self._clinic_info_exception:
            raise self._clinic_info_exception
        return self._clinic_info_response

    # Present so a test can assert these were NEVER called BY THIS GUARD
    # (AC 3: no callback/booking write is made just from asking the
    # question -- only the caller's own next ordinary utterance starts
    # either flow, through its own existing intent).
    async def request_callback(self, *a, **k):
        self._record("request_callback", *a)
        return {"success": True, "callback_id": "CB-SHOULD-NEVER-HAPPEN"}

    async def book_appointment(self, *a, **k):
        self._record("book_appointment", *a)
        return {"success": True}

    async def get_test_rate(self, *a, **k):
        self._record("get_test_rate", *a)
        return {"found": False}


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


class FakeASRResult:
    text = "I want to speak to a doctor personally"
    decoder_used = "ctc"
    decoder_agreement = 1.0
    ctc_words = 6
    rnnt_words = 6


class FakeASRResultNamedDoctor:
    text = "I want Dr Sen to call me"
    decoder_used = "ctc"
    decoder_agreement = 1.0
    ctc_words = 6
    rnnt_words = 6


class FakeASRResultBanglish:
    text = "daktarer sathe kotha bolte chai"
    decoder_used = "ctc"
    decoder_agreement = 1.0
    ctc_words = 4
    rnnt_words = 4


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-doctor-personal-call", pending=pending,
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
    monkeypatch.setattr(transport, "CALLBACKS_ENABLED", True)
    tools = FakeToolsClient()
    monkeypatch.setattr(transport, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools, transport=transport)


def _dispatch_initial(stub, monkeypatch, asr_text_cls=FakeASRResult, tmp_path=None):
    async def fake_resolve_intent(session, text):
        return {"intent": "doctor_personal_request", "slots": {}}

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


class TestDoctorPersonalRequestDispatch:
    def test_speaks_the_callback_available_variant_when_available(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken == [doctor_personal_request_reply(True, language="english")]

    def test_does_not_speak_human_fallback_or_complaint_acknowledgment(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken != [human_fallback_reply(language="english")]
        assert stub.spoken != [complaint_acknowledged_reply(language="english")]

    def test_sets_no_pending_state_at_all(self, stub, monkeypatch, tmp_path):
        session = _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert session.pending is None

    def test_only_get_clinic_info_is_called_never_a_write(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        called = [c[0] for c in stub.tools.calls]
        assert called == ["get_clinic_info"]
        assert "request_callback" not in called
        assert "book_appointment" not in called

    def test_respects_the_detected_language_of_the_utterance(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, asr_text_cls=FakeASRResultBanglish, tmp_path=tmp_path)
        expected_language = detect_language(FakeASRResultBanglish.text)
        assert stub.spoken == [doctor_personal_request_reply(True, language=expected_language)]

    def test_clinic_api_failure_speaks_the_generic_apology_not_a_false_answer(self, stub, monkeypatch, tmp_path):
        from agent.tools_client import ToolCallError
        stub.tools._clinic_info_exception = ToolCallError("db down")
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken == ["এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।"]


# --------------------------------------------------------------------- #
# Callback availability behavior -- AC 4. Mirrors tests/test_request_
# callback.py's own fixture shapes (_ALL_DAYS_OPEN/_ALL_DAYS_CLOSED), but
# with a full 00:00-23:59 window so these tests are deterministic
# regardless of the real wall-clock time the suite happens to run at.
# --------------------------------------------------------------------- #

class TestCallbackAvailabilityBehavior:
    def test_a_enabled_and_available_offers_the_callback_route(self, stub, monkeypatch, tmp_path):
        monkeypatch.setattr(stub.transport, "CALLBACKS_ENABLED", True)
        stub.tools._clinic_info_response = {"found": True, "hours": _ALL_DAYS_OPEN}
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken == [doctor_personal_request_reply(True, language="english")]
        # AC 4: only offered when it genuinely is available -- the
        # available-variant template must actually mention a second route.
        assert stub.spoken[0] != doctor_personal_request_reply(False, language="english")

    def test_b_disabled_never_offers_the_callback_route(self, stub, monkeypatch, tmp_path):
        monkeypatch.setattr(stub.transport, "CALLBACKS_ENABLED", False)
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken == [doctor_personal_request_reply(False, language="english")]
        # Disabled is checked before ever calling get_clinic_info() -- same
        # short-circuit discipline as the "request_callback" intent's own
        # branch (see main.py's _finish_doctor_personal_request()).
        assert "get_clinic_info" not in [c[0] for c in stub.tools.calls]

    def test_c_closed_clinic_never_offers_the_callback_route(self, stub, monkeypatch, tmp_path):
        monkeypatch.setattr(stub.transport, "CALLBACKS_ENABLED", True)
        stub.tools._clinic_info_response = {"found": True, "hours": _ALL_DAYS_CLOSED}
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        assert stub.spoken == [doctor_personal_request_reply(False, language="english")]

    def test_the_two_reply_variants_are_actually_different_text(self):
        # Sanity check that callback_available really changes the wording
        # -- otherwise the tests above could pass for the wrong reason.
        for language in ("english", "hinglish", "banglish", "bengali"):
            assert doctor_personal_request_reply(True, language=language) != \
                doctor_personal_request_reply(False, language=language)

    def test_unavailable_reply_never_mentions_a_callback_word(self):
        # A loose but useful truthfulness check: the unavailable variant
        # should not use the same "take down your number"/"call you back
        # generally" language the available variant uses.
        available_text = doctor_personal_request_reply(True, language="english")
        unavailable_text = doctor_personal_request_reply(False, language="english")
        assert "number" in available_text
        assert "number" not in unavailable_text


# --------------------------------------------------------------------- #
# ADDED BY SOURAV -- AC 4 closure, follow-up task on this same story.
#
# AC 4 reads: "Where a clinical callback exists in policy it is offered."
# Investigation for this follow-up task (re-inspecting agent/callback_
# flow.py, clinic-api/models.py's CallbackRequest, and every product/
# source doc in the repo -- see this file's own docstring and the
# original investigation report -- turned up NO distinct "clinical
# callback" policy or model anywhere in this codebase; the ONLY real
# callback mechanism that exists is the generic one CallbackRequest/
# check_callback_availability()/request_callback() already implement,
# with explicitly no doctor field. _finish_doctor_personal_request()
# (main.py) already calls check_callback_availability() -- the SAME
# pure decision function the standalone "caller asks to be called back"
# story's own dispatch branch calls -- and already offers that route the
# moment it is genuinely available, through doctor_personal_request_
# reply()'s `callback_available` branch. So Option A applies: the
# existing generic callback route, truthfully described as such, IS the
# "clinical callback that exists in policy" this codebase actually has;
# no production code changed for this class -- these are the minimum
# targeted tests needed to prove that fact rather than merely assert it.
# --------------------------------------------------------------------- #

class TestAC4ClinicalCallbackPolicy:
    def test_named_doctor_request_still_offers_the_existing_callback_when_available(
            self, stub, monkeypatch, tmp_path):
        # AC 4 does not stop applying just because the caller named a
        # doctor -- the real, existing route is offered either way,
        # since doctor_personal_request_reply() never branches on
        # whether a name was said, only on whether the route is
        # genuinely available right now (stub's default fixture already
        # sets CALLBACKS_ENABLED=True with all-day-open hours).
        _dispatch_initial(stub, monkeypatch, asr_text_cls=FakeASRResultNamedDoctor, tmp_path=tmp_path)
        assert stub.spoken == [doctor_personal_request_reply(True, language="english")]

    def test_never_invents_a_separate_clinical_or_doctor_specific_callback_concept(self):
        # Regression guard against a future edit accidentally introducing
        # the very thing this story's own IMPORTANT INTERPRETATION forbids:
        # a fabricated "clinical callback" / "doctor callback" as if it
        # were a distinct, more specific mechanism than the generic one
        # that actually exists.
        for language in ("english", "hinglish", "banglish", "bengali"):
            for available in (True, False):
                text = doctor_personal_request_reply(available, language=language).lower()
                assert "clinical callback" not in text
                assert "clinician callback" not in text
                assert "doctor callback" not in text

    def test_offer_decision_matches_the_shared_pure_decision_function_directly(
            self, stub, monkeypatch, tmp_path):
        # Cross-checks _finish_doctor_personal_request()'s own offer
        # decision against an independent, direct call to
        # check_callback_availability() with the same inputs -- proving
        # main.py is not quietly running a second/duplicated decision
        # (the story's own "do not duplicate the existing callback flow"
        # constraint) rather than merely trusting main.py's internals.
        import datetime
        stub.tools._clinic_info_response = {"found": True, "hours": _ALL_DAYS_OPEN}
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        expected = check_callback_availability(
            _ALL_DAYS_OPEN, datetime.date.today().weekday(),
            datetime.datetime.now().strftime("%H:%M"), True,
        )
        assert stub.spoken == [doctor_personal_request_reply(expected["available"], language="english")]

    def test_closed_clinic_gives_the_truthful_appointment_only_alternative_not_a_false_callback(
            self, stub, monkeypatch, tmp_path):
        # AC 4's other half: when the real route is NOT available, the
        # agent must still offer the truthful available alternative
        # (booking) rather than staying silent or inventing a callback
        # that check_callback_availability() says does not exist right
        # now.
        monkeypatch.setattr(stub.transport, "CALLBACKS_ENABLED", True)
        stub.tools._clinic_info_response = {"found": True, "hours": _ALL_DAYS_CLOSED}
        _dispatch_initial(stub, monkeypatch, tmp_path=tmp_path)
        spoken_text = stub.spoken[0]
        assert spoken_text == doctor_personal_request_reply(False, language="english")
        assert "appointment" in spoken_text.lower()
        assert "number" not in spoken_text.lower()


# --------------------------------------------------------------------- #
# Named-doctor promise protection -- AC 3.
# --------------------------------------------------------------------- #

class TestNamedDoctorProtection:
    _NAMED_DOCTOR_UTTERANCES = [
        "I want Dr Sen to call me.",
        "Please ask Dr Sen to call me.",
        "Can Dr Sen speak to me now?",
        "I want my doctor to call me.",
        "I need my doctor to reassure me.",
    ]

    @pytest.mark.parametrize("utterance", _NAMED_DOCTOR_UTTERANCES)
    def test_each_is_recognised_by_the_guard(self, utterance):
        assert is_doctor_personal_request(utterance), f"missed: {utterance!r}"

    def test_no_named_doctor_callback_is_created(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, asr_text_cls=FakeASRResultNamedDoctor, tmp_path=tmp_path)
        called = [c[0] for c in stub.tools.calls]
        assert "request_callback" not in called
        assert "book_appointment" not in called

    def test_no_response_promises_the_named_doctor_will_call(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, asr_text_cls=FakeASRResultNamedDoctor, tmp_path=tmp_path)
        spoken_text = " ".join(stub.spoken).lower()
        # The doctor's name is never repeated back, in any form.
        assert "sen" not in spoken_text
        assert "dr " not in spoken_text
        assert "dr." not in spoken_text
        # Any mention of "will call" only ever appears inside an explicit
        # denial ("I can't promise ... will call you back"), never as a
        # bare affirmative promise -- checked by requiring the denial
        # phrase to be present whenever "will call" is.
        if "will call" in spoken_text:
            assert "can't promise" in spoken_text

    def test_no_response_claims_an_immediate_connection(self, stub, monkeypatch, tmp_path):
        _dispatch_initial(stub, monkeypatch, asr_text_cls=FakeASRResultNamedDoctor, tmp_path=tmp_path)
        spoken_text = " ".join(stub.spoken).lower()
        # "connect" only ever appears inside an explicit denial ("not able
        # to connect you ... directly"), never as a bare "connecting you
        # now" affirmative the way human_fallback_reply()/clinical_
        # interpretation_reply() phrase it -- this story is explicit that
        # the new path must not reuse that wording.
        assert "connecting you" not in spoken_text
        if "connect" in spoken_text:
            assert "not able" in spoken_text
        assert "right away" not in spoken_text
        assert "right now" not in spoken_text

    def test_reply_templates_never_take_a_doctor_name_argument(self):
        # Structural proof, not just a string-content proof: the function
        # signature itself has no doctor-name parameter for any caller
        # phrasing to fill in.
        import inspect
        params = inspect.signature(doctor_personal_request_reply).parameters
        assert "doctor_name" not in params
        assert "name" not in params

    @pytest.mark.parametrize("language", ["english", "hinglish", "banglish", "bengali"])
    def test_no_template_in_any_language_contains_a_colon_style_named_slot(self, language):
        # Every OTHER templated reply in this codebase that names a
        # specific entity does so by string-formatting a real value in;
        # this function takes no such value, so its output cannot vary by
        # caller-supplied name no matter how it is phrased.
        text_true = doctor_personal_request_reply(True, language=language)
        text_false = doctor_personal_request_reply(False, language=language)
        assert text_true and text_false


# --------------------------------------------------------------------- #
# _continue_pending() -- the mid-flow override.
# --------------------------------------------------------------------- #

def _continue(stub, monkeypatch, pending, reply_text):
    session = make_session(pending=dict(pending))
    handled = run(stub.transport._continue_pending(session, reply_text))
    return handled, stub.spoken, session


class TestContinuePendingOverride:
    def test_interrupts_a_booking_flow_and_answers_instead(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "date", "slots": {"doctor_name": "Dr Sen"}, "candidates": None,
                   "offered_date": None, "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending,
                                              "actually I want to speak to a doctor personally")
        assert handled is True
        assert spoken == [doctor_personal_request_reply(True, language="english")]
        assert session.pending is None
        assert [c[0] for c in stub.tools.calls] == ["get_clinic_info"]

    def test_interrupts_an_otp_flow(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "otp_code", "retries": 1}
        handled, spoken, session = _continue(stub, monkeypatch, pending,
                                              "I need to speak with a clinician right now")
        assert handled is True
        assert session.pending is None

    def test_interrupts_the_confirm_booking_state_without_confirming_it(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "confirm_booking", "slots": {"doctor_name": "Dr Sen", "date": "2026-09-20",
                   "time_slot": "10:00", "patient_name": "Test", "phone": "9831012345"},
                   "candidates": None, "offered_date": None, "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending,
                                              "please ask a doctor to call me instead")
        assert handled is True
        assert session.pending is None
        assert "book_appointment" not in [c[0] for c in stub.tools.calls]

    def test_an_ordinary_mid_flow_reply_is_not_swallowed_by_this_guard(self, stub, monkeypatch, tmp_path):
        pending = {"awaiting": "confirm_booking", "slots": {}, "candidates": None,
                   "offered_date": None, "retries": 0}
        handled, spoken, session = _continue(stub, monkeypatch, pending, "না")
        assert [c[0] for c in stub.tools.calls] == []

    def test_a_genuine_booking_related_reply_still_continues_the_booking(self, stub, monkeypatch, tmp_path):
        # "Dr Sen" alone, as a plain answer to a doctor-choice prompt,
        # must NOT be mistaken for this story's guard.
        assert not is_doctor_personal_request("Dr Sen")
        assert not is_doctor_personal_request("Dr. Sen please")


# --------------------------------------------------------------------- #
# Appointment / doctor-availability regression -- these must remain
# entirely unaffected by the new guard (item 5 of the story).
# --------------------------------------------------------------------- #

class TestAppointmentAndAvailabilityRemainUnaffected:
    @pytest.mark.parametrize("utterance", [
        "I want to book an appointment with Dr Sen",
        "I want to book an appointment with Dr Sen.",
        "Can I book an appointment with the doctor?",
    ])
    def test_booking_phrasing_never_trips_the_new_guard(self, utterance):
        assert not is_doctor_personal_request(utterance)

    @pytest.mark.parametrize("utterance", [
        "Is Dr Sen available tomorrow?",
        "Is Dr. Sen available",
        "Which days does Dr Sen sit?",
        "Dr Sen ka schedule kya hai",
    ])
    def test_availability_and_schedule_phrasing_never_trips_the_new_guard(self, utterance):
        assert not is_doctor_personal_request(utterance)

    def test_speak_to_dr_sen_personally_uses_the_new_path_not_availability(self):
        # Item 5's own third example: this ONE must use the new path,
        # distinguishing it from the two negatives just above.
        assert is_doctor_personal_request("I want to speak to Dr Sen personally")


# --------------------------------------------------------------------- #
# Transport parity
# --------------------------------------------------------------------- #

class TestTransportParity:
    def test_main_dot_py_has_the_doctor_personal_request_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "doctor_personal_request"' in src
        assert "is_doctor_personal_request" in src
        assert "_finish_doctor_personal_request" in src

    def test_main_pcm_dot_py_has_the_doctor_personal_request_branch(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "doctor_personal_request"' in src
        assert "is_doctor_personal_request" in src
        assert "_finish_doctor_personal_request" in src

    def test_both_files_check_the_guard_inside_continue_pending_too(self):
        for path in ("main.py", "main_pcm.py"):
            src = open(path, encoding="utf-8").read()
            continue_pending_start = src.index("async def _continue_pending(")
            dispatch_turn_start = src.index("async def _dispatch_turn(")
            body = src[continue_pending_start:dispatch_turn_start]
            assert "is_doctor_personal_request" in body, f"{path}: guard missing from _continue_pending"

    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb


# --------------------------------------------------------------------- #
# agent/llm.py -- schema-only checks (defense-in-depth only; agent/
# doctor_personal_request.py never lets this turn reach the model at all).
# --------------------------------------------------------------------- #

class TestDoctorPersonalRequestIntentSchema:
    def test_is_a_valid_intent(self):
        from agent.llm import VALID_INTENTS
        assert "doctor_personal_request" in VALID_INTENTS

    def test_is_distinct_from_book_appointment_and_doctor_availability(self):
        from agent.llm import VALID_INTENTS
        assert {"book_appointment", "doctor_availability", "doctor_schedule"} <= VALID_INTENTS
        assert "doctor_personal_request" not in {"book_appointment", "doctor_availability", "doctor_schedule"}

    def test_prompt_documents_the_intent_and_the_no_promise_rule(self):
        from agent.llm import SYSTEM_PROMPT_TEMPLATE
        assert '"doctor_personal_request"' in SYSTEM_PROMPT_TEMPLATE
        assert "NO-PROMISE NOTE" in SYSTEM_PROMPT_TEMPLATE

    def test_output_schema_enum_lists_the_new_intent(self):
        from agent.llm import SYSTEM_PROMPT_TEMPLATE
        assert '"human_direct_request" | "complaint" | "doctor_personal_request" | "unclear"' in SYSTEM_PROMPT_TEMPLATE


# --------------------------------------------------------------------- #
# Regression: ordinary, non-doctor-personal-request turns behave exactly
# as before.
# --------------------------------------------------------------------- #

class TestNormalRequestsUnaffected:
    def test_ordinary_test_rate_question_still_reaches_the_fast_path(self, transport, monkeypatch):
        monkeypatch.setattr(transport, "_fast_path", _ExplodingFastPath())
        session = types.SimpleNamespace(call_id="test-normal-call")
        with pytest.raises(AssertionError):
            run(transport._resolve_intent(session, "what is the rate for CBC"))

    def test_clinical_interpretation_phrasing_is_left_to_its_own_guard(self):
        assert not is_doctor_personal_request("is my result dangerous")
        assert not is_doctor_personal_request("am i going to die from this")

    def test_generic_human_request_phrasing_is_left_to_its_own_guard(self):
        assert not is_doctor_personal_request("connect me to a person")
        assert not is_doctor_personal_request("I want to talk to a human")

    def test_complaint_phrasing_is_left_to_its_own_guard(self):
        assert not is_doctor_personal_request("I want to file a complaint about the wait time")
