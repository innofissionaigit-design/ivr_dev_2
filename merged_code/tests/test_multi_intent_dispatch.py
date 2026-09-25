# MERGE NOTE (sourav) -- this test file differed between dev_sourav and
# dev_rajarshee. Merged 3-way against their common ancestor (6cbeb0b ==
# test): every change from each branch touches a different part of the
# file, so it merged with NO conflicts -- rule 1, both sides kept whole.
# Changed regions vs the ancestor: 0 from dev_sourav, 40 from
# dev_rajarshee. Checked after merging: parses, and no test function or
# class name is defined twice (which would silently drop a test).
"""ADDED BY SOURAV -- "Caller asks two questions in one breath" story.

Covers all three layers this story touches, mirroring
tests/test_phase1_intents_and_dispatch.py's own established 3-place
pattern (see that file's module docstring):

  1. agent/llm.py: _validate() accepts the optional "parts" list
     (TestMultiPartSchema), with the same fatal-vs-tolerated semantics the
     single-intent shape always had. NOTE: this story originally shipped
     an "intents" array with a backward-compatibility mirror; both were
     later replaced by "parts" (see that section's own comment), and the
     mirror's tests went with them.
  2. agent/semantic_cache.py: a multi-part value is never L2 (fuzzy)
     eligible, while an ordinary single-part value's eligibility is
     completely unaffected by this story (TestSemanticCacheMultiPartGuard).
  3. main.py AND main_pcm.py dispatch -- parametrized over BOTH modules
     wherever the assertion is transport-independent, so a gap between the
     two (the exact "test_sample" bug this codebase has already hit once
     for real, per main_pcm.py's own module docstring) cannot slip through
     unnoticed (TestMultiIntentDispatch, TestTransportParity).

Story acceptance criteria under direct test:
  - Criterion 1 (Order Preservation): TestMultiIntentDispatch's
    "..._in_requested_order" tests, run with BOTH orderings of the same
    two intents.
  - Criterion 2 (Honest Partial Handling): TestMultiIntentDispatch's
    "..._explicitly_addressed"/"..._missing_slot"/"..._out_of_scope"/
    "..._needs_separate_flow" tests -- nothing is silently dropped.
  - Tool-failure isolation: test_one_tool_failure_does_not_block_the_other.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from agent.llm import _validate, _SLOT_KEYS as _ALL_SLOT_KEYS
from agent.semantic_cache import SemanticCache
from agent.tools_client import ToolCallError
from agent.reply_templates import (
    # Aliased -- a module-level name starting with "test_" gets collected
    # by pytest as a (fixture-less) test function otherwise.
    test_rate_reply as _test_rate_reply,
    walkin_eligibility_reply,
    multi_intent_missing_info_reply, multi_intent_out_of_scope_reply,
    multi_intent_needs_separate_flow_reply, human_fallback_reply,
)

import main
import main_pcm


def run(coro):
    return asyncio.run(coro)


def _empty_slots(**overrides):
    slots = {k: None for k in _ALL_SLOT_KEYS}
    slots.update(overrides)
    return slots


# --------------------------------------------------------------------- #
# agent/llm.py -- the optional "parts" list
#
# The "intents" array this story originally shipped was REPLACED by
# "parts" (agent/llm.py's _validate(), agent/turn_parts.py). Two things
# changed with it, and the assertions below are written against them:
#
#   1. The top-level {intent, slots} pair is authoritative and never went
#      away, so nothing is mirrored back onto it any more -- normalise()
#      builds the parts list FROM the top level instead, forcing parts[0]
#      to be the head. (The old _apply_backward_compat_mirror() and its
#      TestBackwardCompatMirror went with the "intents" key.)
#   2. "parts" is ADDITIVE: its absence is not an error, and a malformed
#      one is dropped rather than repaired, so a degraded extraction
#      becomes a known-good single-part turn instead of a failed one.
#      Only an invalid intent, a non-dict `slots`, or a non-dict entry
#      inside `parts` is fatal.
# --------------------------------------------------------------------- #

def _part(intent, **slot_overrides):
    return {"intent": intent, "slots": _empty_slots(**slot_overrides)}


def _turn(*parts, direct_reply_bn=None, include_parts=True):
    """A whole extraction: the top-level pair is parts[0] -- the invariant
    turn_parts.normalise() establishes -- plus the optional list."""
    head = parts[0]
    data = {"intent": head["intent"], "slots": head["slots"],
            "direct_reply_bn": direct_reply_bn}
    if include_parts:
        data["parts"] = list(parts)
    return data


class TestMultiPartSchema:
    def test_validate_accepts_a_two_part_turn(self):
        data = _turn(_part("test_rate", test_name="CBC"),
                     _part("walkin_eligibility", test_name="CBC"))
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_accepts_a_one_part_turn(self):
        # The common case wrapped in the new shape -- length-1 is valid too.
        data = _turn(_part("test_rate", test_name="CBC"))
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_still_accepts_a_turn_with_no_parts_key(self):
        # No "parts" key at all -- every pre-this-story caller/test, and
        # every model that ignores the field entirely.
        data = _turn(_part("test_rate", test_name="CBC"), include_parts=False)
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_empty_parts_list_is_tolerated_not_fatal(self):
        # DIFFERS from the retired "intents" shape, which rejected an empty
        # array. `parts` is additive: an empty one just means the top-level
        # pair describes the whole turn.
        data = _turn(_part("test_rate", test_name="CBC"))
        data["parts"] = []
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_non_list_parts_is_dropped_not_fatal(self):
        # DIFFERS from the retired "intents" shape, which rejected this.
        # Dropped rather than repaired, so the log says what the model
        # produced and the fallback is a known-good single-part turn.
        data = _turn(_part("test_rate", test_name="CBC"))
        data["parts"] = "test_rate"
        ok, errors = _validate(data)
        assert ok is True, errors
        assert "parts" not in data

    def test_validate_rejects_invalid_intent_inside_a_part(self):
        data = _turn(_part("test_rate", test_name="CBC"),
                     _part("made_up_intent"))
        ok, errors = _validate(data)
        assert ok is False
        assert any("invalid intent" in e for e in errors)

    def test_validate_tolerates_a_missing_slot_key_inside_one_part(self):
        # RULE (unchanged from the legacy shape): an individual missing
        # slot dict key is never fatal by itself.
        second = _part("walkin_eligibility", test_name="CBC")
        del second["slots"]["insurance_provider_name"]
        data = _turn(_part("test_rate", test_name="CBC"), second)
        ok, errors = _validate(data)
        assert ok is True, errors
        assert any("insurance_provider_name" in e and "missing" in e for e in errors)

    def test_one_bad_part_fails_the_whole_turn(self):
        # Not a partial-success shape -- either the whole extraction is
        # schema-valid or extract_intent() retries the whole turn.
        data = _turn(_part("test_rate", test_name="CBC"),
                     _part("not_a_real_intent"))
        ok, errors = _validate(data)
        assert ok is False

    def test_validate_rejects_non_dict_part(self):
        data = _turn(_part("test_rate", test_name="CBC"))
        data["parts"].append("walkin_eligibility")
        ok, errors = _validate(data)
        assert ok is False
        assert any("expected object" in e for e in errors)

    def test_direct_reply_bn_stripped_when_the_turn_has_more_than_one_part(self):
        # Even when the top-level intent IS smalltalk, direct_reply_bn only
        # survives a turn of exactly one part.
        data = _turn(_part("smalltalk"), _part("test_rate", test_name="CBC"),
                     direct_reply_bn="hi there")
        _validate(data)
        assert data["direct_reply_bn"] is None

    def test_direct_reply_bn_kept_for_a_single_smalltalk_part(self):
        data = _turn(_part("smalltalk"), direct_reply_bn="hi there")
        ok, errors = _validate(data)
        assert ok is True, errors
        assert data["direct_reply_bn"] == "hi there"


# --------------------------------------------------------------------- #
# agent/semantic_cache.py
# --------------------------------------------------------------------- #

class TestSemanticCacheMultiPartGuard:
    def test_multi_part_value_never_l2_eligible(self):
        value = {
            "intent": "test_rate", "slots": {"test_name": "CBC"},
            "parts": [
                {"intent": "test_rate", "slots": {"test_name": "CBC"}},
                {"intent": "walkin_eligibility", "slots": {"test_name": "CBC"}},
            ],
        }
        assert SemanticCache._is_l2_eligible(value) is False

    def test_single_part_value_unaffected(self):
        value = {
            "intent": "test_rate", "slots": {"test_name": "CBC"},
            "parts": [{"intent": "test_rate", "slots": {"test_name": "CBC"}}],
        }
        assert SemanticCache._is_l2_eligible(value) is True

    def test_ordinary_value_with_no_parts_key_unaffected(self):
        # Every value before this story, and every ordinary one-question
        # turn after it.
        value = {"intent": "test_rate", "slots": {"test_name": "CBC"}}
        assert SemanticCache._is_l2_eligible(value) is True



# --------------------------------------------------------------------- #
# main.py / main_pcm.py dispatch -- parametrized over both transports
# --------------------------------------------------------------------- #

class FakeToolsClient:
    def __init__(self, **responses):
        self.calls = {}
        self._responses = responses

    def _record(self, name, *args):
        self.calls.setdefault(name, []).append(args)

    async def get_test_rate(self, test_name):
        self._record("get_test_rate", test_name)
        resp = self._responses.get("test_rate")
        if isinstance(resp, Exception):
            raise resp
        return resp or {
            "found": True, "test_name": test_name, "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }

    async def get_walkin_policy(self, test_name):
        self._record("get_walkin_policy", test_name)
        resp = self._responses.get("walkin_policy")
        if isinstance(resp, Exception):
            raise resp
        return resp or {
            "found": True, "test_name": test_name, "test_name_bn": None,
            "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am",
        }

    async def get_doctor_availability(self, doctor_name, date):
        self._record("get_doctor_availability", doctor_name, date)
        resp = self._responses.get("doctor_availability")
        if isinstance(resp, Exception):
            raise resp
        return resp or {
            "found": True, "doctor_name": doctor_name, "doctor_name_bn": None,
            "available": True, "date": date, "next_available_date": None,
        }


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


class FakeASRResult:
    text = "কিছু একটা বললাম"
    # The confidence gate (agent/confidence.py) reads these off every ASR
    # result: no agreement signal at all is treated as DOUBT, so a fake
    # without them gets echoed back ("did I hear you right?") and never
    # reaches the dispatch these tests are about. Full agreement from two
    # decoders = PROCEED, which is the turn being exercised here.
    decoder_agreement = 1.0
    decoder_used = "rnnt"
    ctc_words = 3
    rnnt_words = 3


class FakeASR:
    async def transcribe_utterance(self, wav_path):
        return FakeASRResult()


def make_session(transport, pending=None):
    """The attributes a dispatch touches on a real CallSession. It cannot be
    built directly here -- its __init__ wants a live WebSocket and a temp
    recording file -- so the fields _dispatch_turn_inner reads are mirrored
    instead, from the SAME modules the transport builds them from."""
    return types.SimpleNamespace(
        call_id="test-call-1", pending=pending, utt_seq=0,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
        call_state=transport.call_state_mod.build(),
        answer_ledger=transport.answer_ledger.AnswerLedger(),
        state=transport.DialogueState(),
        confirm_attempts=0, deferred=None,
        last_activity=0.0, last_heartbeat=0.0, processed_until_s=0.0,
        agent_speaking=False, speak_deadline=0.0, resync_pending=False,
    )


TRANSPORTS = [main, main_pcm]


@pytest.fixture(params=TRANSPORTS, ids=["main", "main_pcm"])
def transport(request):
    return request.param


@pytest.fixture
def stub(transport, monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(transport, "_speak", fake_speak)
    monkeypatch.setattr(transport, "_asr", FakeASR())
    tools = FakeToolsClient()
    monkeypatch.setattr(transport, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools, transport=transport)


def _dispatch(stub, monkeypatch, parts, tmp_path, pending=None):
    async def fake_resolve_intent(session, text):
        # turn_parts.normalise() forces parts[0] to be the top-level pair,
        # so an extraction always agrees with itself here.
        return {
            "parts": parts,
            "intent": parts[0]["intent"], "slots": parts[0]["slots"],
        }

    monkeypatch.setattr(stub.transport, "_resolve_intent", fake_resolve_intent)
    session = make_session(stub.transport, pending=pending)
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(stub.transport._dispatch_turn(session, str(wav_path)))
    return session


_CBC_RATE = {"intent": "test_rate", "slots": _empty_slots(test_name="CBC")}
_CBC_WALKIN = {"intent": "walkin_eligibility", "slots": _empty_slots(test_name="CBC")}


class TestMultiIntentDispatch:
    def test_both_questions_answered_in_requested_order(self, stub, monkeypatch, tmp_path):
        _dispatch(stub, monkeypatch, [_CBC_RATE, _CBC_WALKIN], tmp_path)

        assert stub.tools.calls["get_test_rate"] == [("CBC",)]
        assert stub.tools.calls["get_walkin_policy"] == [("CBC",)]
        assert len(stub.spoken) == 1

        rate_result = run(stub.tools.get_test_rate("CBC")) if False else {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }
        walkin_result = {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am",
        }
        expected = (
            _test_rate_reply(_CBC_RATE["slots"], rate_result, language="bengali") + " " +
            walkin_eligibility_reply(_CBC_WALKIN["slots"], walkin_result, language="bengali")
        )
        assert stub.spoken[0] == expected

    def test_order_preservation_reversed(self, stub, monkeypatch, tmp_path):
        # Criterion 1, other direction: swapping the order the caller asked
        # in must swap the order of the fragments in the reply.
        _dispatch(stub, monkeypatch, [_CBC_WALKIN, _CBC_RATE], tmp_path)

        rate_result = {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }
        walkin_result = {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am",
        }
        expected = (
            walkin_eligibility_reply(_CBC_WALKIN["slots"], walkin_result, language="bengali") + " " +
            _test_rate_reply(_CBC_RATE["slots"], rate_result, language="bengali")
        )
        assert stub.spoken[0] == expected

    def test_one_answered_one_missing_slot_explicitly_addressed(self, stub, monkeypatch, tmp_path):
        incomplete_walkin = {"intent": "walkin_eligibility", "slots": _empty_slots()}
        _dispatch(stub, monkeypatch, [_CBC_RATE, incomplete_walkin], tmp_path)

        assert "get_walkin_policy" not in stub.tools.calls
        assert stub.tools.calls["get_test_rate"] == [("CBC",)]
        assert len(stub.spoken) == 1
        assert multi_intent_missing_info_reply(language="bengali") in stub.spoken[0]
        # Not silently dropped -- the first question's real answer is
        # still there too.
        assert stub.spoken[0].startswith(
            _test_rate_reply(_CBC_RATE["slots"], {
                "found": True, "test_name": "CBC", "test_name_bn": None,
                "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
            }, language="bengali")
        )

    def test_one_answered_one_unreviewed_policy_explicitly_addressed(self, stub, monkeypatch, tmp_path):
        # Criterion 2's own "unreviewed" example: the walk-in policy ROW
        # exists but has never been reviewed for this test -- gets the
        # SAME honest sentence walkin_eligibility_reply() already gives a
        # solo turn, with zero new code for this case.
        stub.tools._responses["walkin_policy"] = {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "policy_available": False, "walkin_eligible": None, "walkin_hours": None,
        }
        _dispatch(stub, monkeypatch, [_CBC_RATE, _CBC_WALKIN], tmp_path)

        unreviewed_fragment = walkin_eligibility_reply(_CBC_WALKIN["slots"], stub.tools._responses["walkin_policy"],
                                                        language="bengali")
        assert unreviewed_fragment in stub.spoken[0]
        assert "policy_available" not in unreviewed_fragment  # sanity: a real sentence, not a dict dump

    def test_one_answered_one_out_of_scope_explicitly_addressed(self, stub, monkeypatch, tmp_path):
        out_of_scope_item = {"intent": "out_of_scope", "slots": _empty_slots()}
        session = _dispatch(stub, monkeypatch, [_CBC_RATE, out_of_scope_item], tmp_path)

        assert len(stub.spoken) == 1
        assert multi_intent_out_of_scope_reply(language="bengali") in stub.spoken[0]
        # Combined-turn out_of_scope is non-interactive -- no pending
        # choice state opened, unlike the solo "out_of_scope" dispatch.
        assert session.pending is None

    def test_one_answered_one_book_appointment_needs_separate_flow(self, stub, monkeypatch, tmp_path):
        # Even with every booking field already present, book_appointment
        # is NEVER answered inline in a combined turn.
        full_booking = {
            "intent": "book_appointment",
            "slots": _empty_slots(doctor_name="Dr Sen", date="2026-09-20", time_slot="10am",
                                   patient_name="Rahul Sen", phone="9000000001"),
        }
        session = _dispatch(stub, monkeypatch, [_CBC_RATE, full_booking], tmp_path)

        assert multi_intent_needs_separate_flow_reply(language="bengali") in stub.spoken[0]
        assert session.pending is None

    def test_smalltalk_contributes_no_fragment(self, stub, monkeypatch, tmp_path):
        smalltalk_item = {"intent": "smalltalk", "slots": _empty_slots()}
        _dispatch(stub, monkeypatch, [_CBC_RATE, smalltalk_item], tmp_path)

        expected = _test_rate_reply(_CBC_RATE["slots"], {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }, language="bengali")
        assert stub.spoken == [expected]

    def test_unclear_contributes_no_fragment_and_order_still_holds(self, stub, monkeypatch, tmp_path):
        unclear_item = {"intent": "unclear", "slots": _empty_slots()}
        _dispatch(stub, monkeypatch, [unclear_item, _CBC_RATE], tmp_path)

        expected = _test_rate_reply(_CBC_RATE["slots"], {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }, language="bengali")
        assert stub.spoken == [expected]

    def test_all_entries_no_fragment_falls_back_to_human_handoff(self, stub, monkeypatch, tmp_path):
        smalltalk_item = {"intent": "smalltalk", "slots": _empty_slots()}
        unclear_item = {"intent": "unclear", "slots": _empty_slots()}
        _dispatch(stub, monkeypatch, [smalltalk_item, unclear_item], tmp_path)

        assert stub.spoken == [human_fallback_reply(language="bengali")]

    def test_one_tool_failure_does_not_block_the_other(self, stub, monkeypatch, tmp_path):
        stub.tools._responses["test_rate"] = ToolCallError("clinic-api unreachable")
        _dispatch(stub, monkeypatch, [_CBC_RATE, _CBC_WALKIN], tmp_path)

        assert len(stub.spoken) == 1
        # First fragment is the apology; second is the real, unaffected answer.
        assert "কাউন্টারে যোগাযোগ করুন" in stub.spoken[0]
        walkin_result = {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am",
        }
        assert stub.spoken[0].endswith(
            walkin_eligibility_reply(_CBC_WALKIN["slots"], walkin_result, language="bengali")
        )
        # The SECOND tool call still ran -- not short-circuited by the first's failure.
        assert stub.tools.calls["get_walkin_policy"] == [("CBC",)]

    def test_single_intent_turn_is_completely_unaffected(self, stub, monkeypatch, tmp_path):
        # The fast_path/solo-extraction backward-compatibility guarantee:
        # a length-1 "parts" list takes the ORIGINAL if/elif chain, not
        # _dispatch_multi_intent_turn at all.
        session = _dispatch(stub, monkeypatch, [_CBC_RATE], tmp_path)

        expected = _test_rate_reply(_CBC_RATE["slots"], {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }, language="bengali")
        assert stub.spoken == [expected]
        assert session.pending is None


# --------------------------------------------------------------------- #
# Transport parity: the exact "test_sample" gap this codebase already hit
# once (main_pcm.py's own module docstring) -- confirm every new symbol
# this story adds exists verbatim in BOTH files, not hand-duplicated and
# drifted.
# --------------------------------------------------------------------- #

class TestTransportParity:
    @pytest.mark.parametrize("marker", [
        "async def _dispatch_multi_intent_turn(",
        "async def _resolve_combinable_intent_fragment(",
        "parts = turn_parts.normalise(data)",
        "_MULTI_INTENT_NEEDS_SEPARATE_FLOW = ",
        "_MULTI_INTENT_NO_FRAGMENT = ",
    ])
    def test_symbol_present_in_both_transports(self, marker):
        import inspect
        main_src = inspect.getsource(main)
        main_pcm_src = inspect.getsource(main_pcm)
        assert marker in main_src, f"{marker!r} missing from main.py"
        assert marker in main_pcm_src, f"{marker!r} missing from main_pcm.py"

    def test_both_transports_agree_on_a_two_intent_turn(self, monkeypatch, tmp_path):
        # Same scenario run through BOTH transports independently -- the
        # actual reply text produced must match, not just the source code.
        results = {}
        for transport in TRANSPORTS:
            spoken = []

            async def fake_speak(session, text, fallback_reason=None, _spoken=spoken):
                _spoken.append(text)

            monkeypatch.setattr(transport, "_speak", fake_speak)
            monkeypatch.setattr(transport, "_asr", FakeASR())
            tools = FakeToolsClient()
            monkeypatch.setattr(transport, "_tools", tools)

            async def fake_resolve_intent(session, text):
                return {
                    "parts": [_CBC_RATE, _CBC_WALKIN],
                    "intent": _CBC_RATE["intent"], "slots": _CBC_RATE["slots"],
                }

            monkeypatch.setattr(transport, "_resolve_intent", fake_resolve_intent)
            session = make_session(transport)
            wav_path = tmp_path / f"utt-{transport.__name__}.wav"
            wav_path.write_bytes(b"")
            run(transport._dispatch_turn(session, str(wav_path)))
            results[transport.__name__] = spoken

        assert results["main"] == results["main_pcm"]
        assert len(results["main"]) == 1
