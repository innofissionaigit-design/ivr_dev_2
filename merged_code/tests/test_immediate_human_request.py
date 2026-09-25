"""ADDED BY SOURAV -- "Caller asks for a person immediately" story (Epic:
Conversation -- Difficult, Sensitive and Edge Cases).

story title: Caller asks for a person immediately
user story: As a caller who wants a person, I want one without negotiating,
    so that I do not feel trapped.
acceptance criteria: A request for a human in any supported language
    escalates on the same turn with no retention attempt and no question
    about why. The phrase set is tested per language and the rate is
    reported as a quality signal rather than something to minimise.

Structured the same way tests/test_clinical_interpretation_safety.py is
(cores x carriers adversarial corpus, a _resolve_intent() short-circuit
proof, a dispatch-level test, transport parity, schema checks) plus two
sections that story does not need at all, because this one is not that
one:

1. No "offer, then wait for yes/no" continuation class. clinical_
   interpretation's AC is "routed to a human by policy" -- an offer first,
   escalation only on confirmation. THIS story's AC is "escalates on the
   same turn with no retention attempt" -- there is no offer, no
   session.pending set afterward, and no follow-up turn to continue. So
   TestImmediateHumanRequestDispatch below asserts session.pending is None
   and the handoff is recorded on the VERY FIRST turn, which would be a bug
   for clinical_interpretation and is the whole point here.

2. A brand-new TestContinuePendingOverride class with no equivalent in any
   sibling test file. This is the one piece of the story the user's own
   implementation sketch never mentioned: _resolve_intent() alone only
   fires for a turn that is not already owned by an in-progress flow --
   main_pcm.py's _dispatch_turn_inner calls _continue_pending(session,
   text) FIRST whenever session.pending is set, and only falls through to
   _resolve_intent() when that returns False. A caller mid-booking, mid-
   OTP, or mid- any other multi-turn flow who suddenly says "just connect
   me to a person" would otherwise have that sentence misread as an
   attempted answer to whatever field was pending -- the literal shape of
   "feeling trapped" this user story names. See agent/human_fast_path.py's
   own module docstring for the full reasoning; this test class is what
   proves that reasoning is actually wired up, not just written down.

3. Rate/metrics tests for the genuinely new agent/outcomes.py
   infrastructure this story adds (record_turn_attempt, total_turns,
   record_immediate_human_handoff, immediate_human_escalation_rate) --
   agent/outcomes.py's own pre-existing module docstring said outright that
   nothing in this codebase counted attempts per intent before this story;
   TestEscalationRateMetric is what proves that gap is now closed, cleanly,
   with a genuine denominator rather than a fabricated one.

WHY THE ADVERSARIAL PROMPTS ARE NOT JUST agent.human_fast_path's OWN
PHRASE TUPLES READ BACK: same discipline as
test_clinical_interpretation_safety.py's own note on this -- every prompt
below was written independently, covering direct requests, insistence
phrasing, "no bot/robot" refusals, and polite/impatient variants, then run
against the real function. Building this suite is in fact how the
"transfer/put me through to a representative" and "can you connect/get me/
transfer me ..." gaps in agent/human_fast_path.py's English tuple were
found -- an early draft of this file caught those misses, and the fix went
into agent/human_fast_path.py itself, not here, so it protects every
future caller rather than just this test run.

Every language's set is builder cores x carriers (a short prefix a real
caller might actually say first -- nothing, "please", a stated refusal to
deal with a bot, or impatience) rather than 100+ hand-typed one-off lines,
so the count is honest and reproducible rather than padded duplication
with the numbers filed off.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import types

import pytest

from agent.bn_normalize import verbalize, detect_language
from agent.human_fast_path import (
    detect_immediate_human_request,
    is_immediate_human_request,
)
from agent.outcomes import (
    human_handoff_counts,
    record_human_handoff,
    record_turn_attempt,
    record_immediate_human_handoff,
    total_turns,
    immediate_human_escalation_rate,
    HUMAN_DIRECT_REQUEST_INTENT,
    _reset_for_testing,
)
import agent.outcomes as outcomes_module
from agent.reply_templates import human_fallback_reply, clinical_interpretation_reply

import main_pcm

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
# The adversarial coverage. Each language's corpus is CORES x CARRIERS:
# CORES are independently-written caller utterances asking for a human,
# CARRIERS are short real-world prefixes a caller might say first. Every
# combination must be recognised -- this is a zero-tolerance guard, not a
# best-effort one.
# --------------------------------------------------------------------- #

_EN_CORES = [
    "I want to talk to a person",
    "I want to talk to a human",
    "I want to speak to a human",
    "connect me to a human please",
    "connect me to a person",
    "connect me to an agent",
    "connect me to a representative",
    "get me an agent",
    "get me a real person",
    "get me a human",
    "can you transfer me to a representative",
    "can you transfer me to an agent",
    "can you connect me to a person",
    "can you get me a human",
    "could you connect me to a person",
    "put me through to someone",
    "put me through to a representative",
    "let me talk to a person",
    "let me speak to a human",
    "I want a real person, not a bot",
    "I want a human",
    "I want an agent",
    "I want an operator",
    "I want a representative",
    "is there a real person I can talk to",
    "give me a human",
    "give me an agent",
    "give me an operator",
    "no bot, please",
    "not a bot, a real person",
    "stop the bot and get me a human",
    "no robot, I need a person",
    "human agent please",
    "real person please",
    "I don't want to talk to a machine, get me a person",
    "just transfer me to a human already",
]
_EN_CARRIERS = ["", "Please, ", "Look, ", "I don't have time for this, "]

_HI_CORES = [
    "mujhe kisi insan se baat karni hai",
    "kisi bande se baat karvao",
    "kisi insan se baat karado",
    "human se baat karado",
    "human se connect karo",
    "agent se baat karado",
    "agent se connect karo",
    "executive se baat karado",
    "representative se baat karado",
    "operator se connect karo",
    "mujhe insan chahiye",
    "mujhe koi insan chahiye",
    "mujhe agent chahiye",
    "kisi se connect karo abhi",
    "kisi ko line pe do",
    "kisi ko phone do",
    "kisi ko de do abhi",
    "real insan se baat karado",
    "real banda chahiye mujhe",
    "bot nahi, ek insan se baat karado",
    "robot nahi, mujhe insan chahiye",
    "kisi insaan se baat karado abhi",
]
_HI_CARRIERS = ["", "Please, ", "Dekho, ", "Mujhe jaldi hai, "]

_BN_CORES = [
    "আমি মানুষের সাথে কথা বলতে চাই",
    "কারো সাথে কথা বলতে চাই",
    "একজন মানুষের সাথে কথা বলিয়ে দিন",
    "মানুষের সাথে কথা বলাও",
    "মানুষ চাই",
    "সত্যিকারের মানুষ চাই",
    "এজেন্টের সাথে কথা বলতে চাই",
    "এজেন্টের সাথে সংযুক্ত করুন",
    "কাউকে লাইনে দিন",
    "কাউকে ফোন দিন",
    "কাউকে দিন এখনই",
    "বট না, একজন মানুষ দিন",
    "রোবট না, আমার মানুষ চাই",
    "কোনো মানুষ দিন",
]
_BN_CARRIERS = ["", "দয়া করে, ", "শুনুন, "]

_BANGLISH_CORES = [
    "manush er sathe kotha bolte chai",
    "kono manush er sathe kotha bolte chai",
    "keu ekjon manush er sathe kotha bolao",
    "manush chai amar",
    "manush ke dao",
    "agent er sathe kotha bolte chai",
    "executive er sathe kotha bolte chai",
    "kauke line e dao",
    "kauke phone dao",
    "ekta manusher sathe kotha bolao",
    "bot na, ekta manush chai",
    "robot na, manush chai",
]
_BANGLISH_CARRIERS = ["", "Please, ", "Shunun, "]


def _corpus(cores, carriers):
    return [carrier + core for core, carrier in itertools.product(cores, carriers)]


_EN_PROMPTS = _corpus(_EN_CORES, _EN_CARRIERS)
_HI_PROMPTS = _corpus(_HI_CORES, _HI_CARRIERS)
_BN_PROMPTS = _corpus(_BN_CORES, _BN_CARRIERS)
_BANGLISH_PROMPTS = _corpus(_BANGLISH_CORES, _BANGLISH_CARRIERS)


class TestAdversarialCoverageProducesZeroBreaches:
    def test_the_corpora_each_have_a_healthy_number_of_prompts(self):
        # Not pinned to the clinical_interpretation story's "100" -- that
        # number is THAT story's own AC wording, not this one's. This
        # story's AC asks for "the phrase set ... tested per language",
        # which this mechanical floor checks without inventing a number
        # the AC never stated.
        assert len(_EN_PROMPTS) >= 100
        assert len(_HI_PROMPTS) >= 60
        assert len(_BN_PROMPTS) >= 30
        assert len(_BANGLISH_PROMPTS) >= 30

    @pytest.mark.parametrize("prompt", _EN_PROMPTS)
    def test_english_prompt_is_recognised(self, prompt):
        assert is_immediate_human_request(prompt), f"breach: {prompt!r}"

    @pytest.mark.parametrize("prompt", _HI_PROMPTS)
    def test_hinglish_prompt_is_recognised(self, prompt):
        assert is_immediate_human_request(prompt), f"breach: {prompt!r}"

    @pytest.mark.parametrize("prompt", _BN_PROMPTS)
    def test_bengali_prompt_is_recognised(self, prompt):
        assert is_immediate_human_request(prompt), f"breach: {prompt!r}"

    @pytest.mark.parametrize("prompt", _BANGLISH_PROMPTS)
    def test_banglish_prompt_is_recognised(self, prompt):
        assert is_immediate_human_request(prompt), f"breach: {prompt!r}"

    def test_ordinary_catalogue_questions_are_left_alone(self):
        ordinary = [
            "what is the rate for CBC",
            "is Dr. Sen available tomorrow",
            "what time do you open",
            "do I need a prescription for uric acid",
            "can I get my report delivered",
            "রেট কত ইউরিক অ্যাসিডের",
            "কাল ডাক্তার সেন আছেন কিনা",
        ]
        for text in ordinary:
            assert not is_immediate_human_request(text), f"false positive: {text!r}"

    def test_doctor_related_requests_are_deliberately_left_alone(self):
        # WHAT THIS MODULE DOES NOT DO, per its own module docstring: a
        # caller asking about a doctor is already served by booking/
        # schedule flows or by clinical_interpretation's own gentler offer
        # -- collapsing those into "connect me to any human, no questions
        # asked" would regress both of those existing stories.
        doctor_requests = [
            "connect me to a doctor please",
            "can I book an appointment with a doctor",
            "is Dr. Sen available",
            "I want to speak to a doctor about my result",
        ]
        for text in doctor_requests:
            assert not is_immediate_human_request(text), f"false positive: {text!r}"

    def test_clinical_interpretation_phrasing_is_left_to_its_own_guard(self):
        # A worried patient asking if their result is dangerous is
        # clinical_interpretation's territory, not this story's -- the two
        # guards must not overlap on their respective core cases.
        clinical_only = [
            "is my result dangerous",
            "am i going to die from this",
            "is this life threatening",
        ]
        for text in clinical_only:
            assert not is_immediate_human_request(text), f"false positive: {text!r}"

    def test_empty_and_blank_text_is_never_flagged(self):
        assert not is_immediate_human_request("")
        assert not is_immediate_human_request("   ")
        assert not is_immediate_human_request(None)

    def test_bengali_word_boundary_safety(self):
        # "মানুষ" ("human"/"person") must not fire merely because it
        # appears as a substring inside an unrelated longer sentence --
        # same _bn_bounded() word-boundary discipline agent/clinical_
        # safety.py and agent/fast_path.py already rely on.
        assert not is_immediate_human_request("মানুষটি কেমন আছেন")

    def test_detect_returns_the_matched_language_family(self):
        matched, family = detect_immediate_human_request("connect me to a human please")
        assert matched is True
        assert family == "english"
        matched, family = detect_immediate_human_request("mujhe insan chahiye")
        assert matched is True
        assert family == "latin_translit"
        matched, family = detect_immediate_human_request("মানুষ চাই")
        assert matched is True
        assert family == "bengali"
        matched, family = detect_immediate_human_request("what is the rate for CBC")
        assert matched is False
        assert family is None


# --------------------------------------------------------------------- #
# main_pcm.py's _resolve_intent() -- the actual choke point. Proves the
# guard fires structurally, ahead of even the clinical-interpretation
# guard, and before the fast path, semantic cache, or LLM ever run.
# --------------------------------------------------------------------- #

class _ExplodingFastPath:
    def resolve(self, text):
        raise AssertionError("fast_path.resolve() must not run for a human_direct_request turn")


class _ExplodingCache:
    def get(self, text):
        raise AssertionError("semantic cache must not run for a human_direct_request turn")

    def put(self, text, data):
        raise AssertionError("semantic cache must not run for a human_direct_request turn")


def _exploding_extract_intent(text):
    raise AssertionError("extract_intent() (the LLM) must not run for a human_direct_request turn")


class TestResolveIntentNeverReachesTheClassifier:
    @pytest.fixture(autouse=True)
    def _wire_exploding_stand_ins(self, monkeypatch):
        monkeypatch.setattr(main_pcm, "_fast_path", _ExplodingFastPath())
        monkeypatch.setattr(main_pcm, "_intent_cache", _ExplodingCache())
        monkeypatch.setattr(main_pcm, "extract_intent", _exploding_extract_intent)

    @pytest.mark.parametrize(
        "prompt", _EN_PROMPTS[:15] + _HI_PROMPTS[:15] + _BN_PROMPTS[:15] + _BANGLISH_PROMPTS[:10]
    )
    def test_short_circuits_before_any_classifier_runs(self, prompt):
        session = types.SimpleNamespace(call_id="test-human-direct-guard-call")
        data = run(main_pcm._resolve_intent(session, prompt))
        assert data["intent"] == "human_direct_request"
        assert data["parts"] == [{"intent": "human_direct_request", "slots": data["slots"]}]
        assert data["direct_reply_bn"] is None

    def test_an_ordinary_question_still_reaches_the_fast_path(self):
        session = types.SimpleNamespace(call_id="test-human-direct-guard-call")
        with pytest.raises(AssertionError):
            run(main_pcm._resolve_intent(session, "what is the rate for CBC"))

    def test_fires_ahead_of_the_clinical_interpretation_guard(self):
        # An utterance that could plausibly read as either is safest
        # resolved as immediate escalation rather than an offer -- the
        # ordering decision recorded in agent/human_fast_path.py's own
        # module docstring. This phrasing legitimately matches BOTH
        # guards' phrase sets (a direct human request plus danger wording);
        # human_direct_request must win.
        session = types.SimpleNamespace(call_id="test-human-direct-guard-call")
        text = "I'm really scared, is my result dangerous, just get me a human"
        data = run(main_pcm._resolve_intent(session, text))
        assert data["intent"] == "human_direct_request"


# --------------------------------------------------------------------- #
# main_pcm.py dispatch -- the "human_direct_request" intent branch.
# UNLIKE clinical_interpretation's dispatch test, this must prove NO
# pending state is ever set and the handoff IS recorded on this very
# first turn -- "escalates on the same turn" is the whole AC, not an
# offer-then-wait shape.
# --------------------------------------------------------------------- #

class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-human-direct-request-call", pending=pending,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
        call_state=main_pcm.call_state_mod.build(), utt_seq=1,
        confirm_attempts=0,
    )


class FakeASRResultEnglish:
    text = "just connect me to a human please"
    decoder_agreement = 0.95
    decoder_used = "rnnt"
    ctc_words = 6
    rnnt_words = 6


class FakeASRResultBanglish:
    text = "manush er sathe kotha bolte chai"
    decoder_agreement = 0.95
    decoder_used = "rnnt"
    ctc_words = 5
    rnnt_words = 5


def _dispatch_initial(monkeypatch, asr_text_cls, tmp_path):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    async def fake_resolve_intent(session, text):
        return {"intent": "human_direct_request", "slots": {}}

    class FakeASR:
        async def transcribe_utterance(self, wav_path):
            return asr_text_cls()

    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    monkeypatch.setattr(main_pcm, "_asr", FakeASR())
    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)

    session = make_session()
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return spoken, session


class TestImmediateHumanRequestDispatch:
    def test_speaks_human_fallback_reply_immediately(self, monkeypatch, tmp_path):
        spoken, session = _dispatch_initial(monkeypatch, FakeASRResultEnglish, tmp_path)
        assert spoken == [human_fallback_reply(language="english")]

    def test_does_not_speak_the_clinical_interpretation_offer(self, monkeypatch, tmp_path):
        # Sanity check that the two escalation-adjacent stories' replies
        # are not accidentally cross-wired.
        spoken, _ = _dispatch_initial(monkeypatch, FakeASRResultEnglish, tmp_path)
        assert spoken != [clinical_interpretation_reply(language="english")]

    def test_sets_no_pending_state_at_all(self, monkeypatch, tmp_path):
        # THE core structural proof of "zero negotiation, same turn": no
        # offer, no follow-up question, nothing left half-finished.
        _, session = _dispatch_initial(monkeypatch, FakeASRResultEnglish, tmp_path)
        assert session.pending is None

    def test_records_the_handoff_on_this_very_first_turn(self, monkeypatch, tmp_path):
        # UNLIKE clinical_interpretation (which only counts a handoff once
        # the caller confirms), this story's AC has no confirmation step to
        # wait for -- the handoff IS the same-turn escalation itself.
        assert human_handoff_counts(HUMAN_DIRECT_REQUEST_INTENT) == 0
        _dispatch_initial(monkeypatch, FakeASRResultEnglish, tmp_path)
        assert human_handoff_counts(HUMAN_DIRECT_REQUEST_INTENT) == 1

    def test_escalation_ledger_gets_the_right_intent_reason_and_call_id(self, monkeypatch, tmp_path, _isolated_escalation_log):
        _dispatch_initial(monkeypatch, FakeASRResultEnglish, tmp_path)
        entry = json.loads(_isolated_escalation_log.read_text(encoding="utf-8").strip())
        assert entry["event"] == "human_handoff"
        assert entry["intent"] == "human_direct_request"
        assert entry["reason"] == "caller_requested_human_directly"
        assert entry["call_id"] == "test-human-direct-request-call"

    def test_respects_the_detected_language_of_the_utterance(self, monkeypatch, tmp_path):
        spoken, _ = _dispatch_initial(monkeypatch, FakeASRResultBanglish, tmp_path)
        expected_language = detect_language(FakeASRResultBanglish.text)
        assert spoken == [human_fallback_reply(language=expected_language)]


# --------------------------------------------------------------------- #
# main_pcm.py _continue_pending() -- the universal mid-flow override.
# This is the gap the user's own implementation sketch never mentioned:
# _resolve_intent() alone is never reached for a turn already owned by an
# in-progress flow, so this guard has to run again, first, inside
# _continue_pending() -- see agent/human_fast_path.py's module docstring.
# --------------------------------------------------------------------- #

def _continue(monkeypatch, pending, reply_text):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    session = make_session(pending=dict(pending))
    handled = run(main_pcm._continue_pending(session, reply_text))
    return handled, spoken, session


class TestContinuePendingOverride:
    def test_interrupts_a_booking_flow_and_escalates_immediately(self, monkeypatch):
        # A caller three questions into booking a test, who suddenly
        # demands a human -- their words must NOT be parsed as an attempt
        # to answer whatever field was pending (here, a date).
        pending = {"awaiting": "booking_date", "retries": 0}
        handled, spoken, session = _continue(monkeypatch, pending, "just get me a human please")
        assert handled is True
        assert spoken == [human_fallback_reply(language="english")]
        assert session.pending is None

    def test_interrupts_an_otp_flow_and_escalates_immediately(self, monkeypatch):
        pending = {"awaiting": "otp_code", "retries": 1}
        handled, spoken, session = _continue(monkeypatch, pending, "connect me to an agent")
        assert handled is True
        assert spoken == [human_fallback_reply(language="english")]
        assert session.pending is None

    def test_interrupts_the_clinical_interpretation_choice_flow_too(self, monkeypatch):
        # Even a flow this same feature-set introduced (the clinical_
        # interpretation offer) must yield to a direct human request --
        # zero exceptions, per the AC's "in any supported language" and
        # "no question about why".
        pending = {"awaiting": "clinical_interpretation_choice", "retries": 0}
        handled, spoken, session = _continue(monkeypatch, pending, "no bot, get me a real person")
        assert handled is True
        assert spoken == [human_fallback_reply(language="english")]
        assert session.pending is None

    def test_records_the_handoff_when_interrupting_a_flow(self, monkeypatch):
        assert human_handoff_counts(HUMAN_DIRECT_REQUEST_INTENT) == 0
        pending = {"awaiting": "booking_date", "retries": 0}
        _continue(monkeypatch, pending, "get me a human")
        assert human_handoff_counts(HUMAN_DIRECT_REQUEST_INTENT) == 1

    def test_an_ordinary_mid_flow_reply_is_not_swallowed_by_this_guard(self, monkeypatch):
        # Negative control: an unrelated in-progress-flow reply must still
        # reach that flow's own field parsing, not this guard.
        pending = {"awaiting": "clinical_interpretation_choice", "retries": 0}
        handled, spoken, session = _continue(monkeypatch, pending, "yes")
        assert handled is True
        assert spoken == [human_fallback_reply(language="english")]
        # Reached via clinical_interpretation's OWN confirm path, not this
        # story's -- distinguished by the ledger's intent field.
        assert human_handoff_counts("clinical_interpretation") == 1
        assert human_handoff_counts(HUMAN_DIRECT_REQUEST_INTENT) == 0

    def test_respects_the_detected_language_of_the_interrupting_utterance(self, monkeypatch):
        pending = {"awaiting": "booking_date", "retries": 0}
        handled, spoken, session = _continue(monkeypatch, pending, "manush er sathe kotha bolte chai")
        expected_language = detect_language("manush er sathe kotha bolte chai")
        assert spoken == [human_fallback_reply(language=expected_language)]


# --------------------------------------------------------------------- #
# Rate/metrics -- agent/outcomes.py's genuinely new turn-attempt/rate
# infrastructure. Fills the gap that module's own pre-existing docstring
# said outright did not exist yet ("NO FAKE RATE ... nothing in this
# codebase counts attempts per intent today").
# --------------------------------------------------------------------- #

class TestEscalationRateMetric:
    def test_total_turns_starts_at_zero_and_counts_every_call(self):
        assert total_turns() == 0
        record_turn_attempt()
        record_turn_attempt()
        record_turn_attempt()
        assert total_turns() == 3

    def test_rate_is_zero_with_no_turns_recorded_yet(self):
        # The zero-turns edge case: never divide by zero, never fabricate
        # a rate before there is a genuine denominator.
        result = immediate_human_escalation_rate()
        assert result["total_turns"] == 0
        assert result["immediate_human_handoffs"] == 0
        assert result["immediate_human_escalation_rate"] == 0.0

    def test_rate_reflects_immediate_handoffs_over_total_turns(self):
        for _ in range(10):
            record_turn_attempt()
        record_immediate_human_handoff(call_id="c1")
        record_immediate_human_handoff(call_id="c2")
        result = immediate_human_escalation_rate()
        assert result["total_turns"] == 10
        assert result["immediate_human_handoffs"] == 2
        assert result["immediate_human_escalation_rate"] == 0.2

    def test_rate_reads_the_existing_handoff_counter_not_a_duplicate_one(self):
        # Anti-duplication design decision: the numerator is
        # human_handoff_counts(HUMAN_DIRECT_REQUEST_INTENT), the SAME
        # counter record_human_handoff() already maintains -- not a
        # second, parallel counter that could silently drift out of sync.
        record_turn_attempt()
        record_immediate_human_handoff(call_id="c1")
        assert human_handoff_counts(HUMAN_DIRECT_REQUEST_INTENT) == 1
        assert immediate_human_escalation_rate()["immediate_human_handoffs"] == 1

    def test_other_intents_handoffs_do_not_pollute_the_rate(self):
        record_turn_attempt()
        record_turn_attempt()
        record_human_handoff(intent="out_of_scope", call_id="c1")
        record_human_handoff(intent="clinical_interpretation", call_id="c2")
        result = immediate_human_escalation_rate()
        assert result["immediate_human_handoffs"] == 0
        assert result["immediate_human_escalation_rate"] == 0.0

    def test_record_human_handoff_default_reason_is_unchanged_for_existing_callers(self):
        # This story widened record_human_handoff()'s signature with a new
        # `reason` parameter -- every EXISTING call site (out_of_scope,
        # clinical_interpretation, ...) must keep getting the same default
        # reason string it always has, unchanged.
        record_human_handoff(intent="out_of_scope", call_id="c1")
        entry = json.loads(open(outcomes_module.ESCALATION_LOG_PATH, encoding="utf-8").read().strip())
        assert entry["reason"] == "query_unresolved_or_low_confidence"

    def test_record_immediate_human_handoff_uses_the_dedicated_reason(self):
        record_immediate_human_handoff(call_id="c1")
        entry = json.loads(open(outcomes_module.ESCALATION_LOG_PATH, encoding="utf-8").read().strip())
        assert entry["reason"] == "caller_requested_human_directly"
        assert entry["intent"] == HUMAN_DIRECT_REQUEST_INTENT


# --------------------------------------------------------------------- #
# Transport parity
# --------------------------------------------------------------------- #

class TestTransportParity:
    def test_main_dot_py_has_the_human_direct_request_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "human_direct_request"' in src
        assert "is_immediate_human_request" in src
        assert "record_immediate_human_handoff" in src
        assert "record_turn_attempt" in src

    def test_main_pcm_dot_py_has_the_human_direct_request_branch(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "human_direct_request"' in src
        assert "is_immediate_human_request" in src
        assert "record_immediate_human_handoff" in src
        assert "record_turn_attempt" in src

    def test_both_files_check_the_guard_inside_continue_pending_too(self):
        # Not just _resolve_intent() -- the mid-flow override is the part
        # of this story most likely to be forgotten in one of the two
        # transports, so it gets its own explicit parity assertion rather
        # than relying solely on the byte-identical check below.
        for path in ("main.py", "main_pcm.py"):
            src = open(path, encoding="utf-8").read()
            continue_pending_start = src.index("async def _continue_pending(")
            dispatch_turn_start = src.index("async def _dispatch_turn(")
            # _continue_pending is defined after _resolve_intent and before
            # _dispatch_turn in both files (confirmed by inspection); the
            # guard call must appear somewhere in _continue_pending's own
            # body.
            body = src[continue_pending_start:dispatch_turn_start]
            assert "is_immediate_human_request" in body, f"{path}: guard missing from _continue_pending"

    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb


# --------------------------------------------------------------------- #
# agent/llm.py -- schema-only checks (defense-in-depth only; agent/
# human_fast_path.py never lets a human_direct_request turn reach the
# model at all -- same discipline as clinical_interpretation's own schema
# tests).
# --------------------------------------------------------------------- #

class TestHumanDirectRequestIntentSchema:
    def test_human_direct_request_is_a_valid_intent(self):
        from agent.llm import VALID_INTENTS
        assert "human_direct_request" in VALID_INTENTS

    def test_is_distinct_from_smalltalk_and_unclear_and_out_of_scope_and_clinical_interpretation(self):
        from agent.llm import VALID_INTENTS
        assert {"smalltalk", "unclear", "out_of_scope", "clinical_interpretation"} <= VALID_INTENTS
        assert "human_direct_request" not in {
            "smalltalk", "unclear", "out_of_scope", "clinical_interpretation",
        }

    def test_prompt_documents_the_intent_and_the_no_retention_rule(self):
        from agent.llm import SYSTEM_PROMPT_TEMPLATE
        assert "human_direct_request" in SYSTEM_PROMPT_TEMPLATE
        assert "NO-RETENTION NOTE" in SYSTEM_PROMPT_TEMPLATE

    def test_output_schema_enum_lists_both_newer_intents(self):
        # FIXED BY SOURAV -- pre-existing bug found while building this
        # story: the JSON output schema's intent enum was missing
        # "clinical_interpretation" entirely (an oversight from that
        # story's own implementation), only caught now. Both are checked
        # together here so this specific regression cannot silently
        # reappear.
        from agent.llm import SYSTEM_PROMPT_TEMPLATE
        assert '"clinical_interpretation"' in SYSTEM_PROMPT_TEMPLATE
        assert '"human_direct_request"' in SYSTEM_PROMPT_TEMPLATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
