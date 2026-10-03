"""Regressions from the first live call on the pod (call c7eb4b15, 2026-09-25), through the real orchestrator.

    python -m pytest tests/test_live_call_fixes.py -v

What the caller experienced, and what the log showed:
  1. A Bengali call was answered in Hindi. Language ID said bn 0.95 / hi 0.05, all three recognisers ran, and the
     Hindi one (writing Bengali speech out in Devanagari) reported a higher agreement than the Bengali one.
  2. The agent "could not hear" and made the caller repeat the whole sentence, when the intent ("the price of a
     test") had been understood and only the name was not trusted.
  3. The price reply spoke a label and a colon ("sample: Blood") and, in Bengali, any catalogue category as a sample.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_orchestrator_booking_flow import env, m  # noqa: F401  (the harness: real dispatch, fake tools)

from agent import entity_confirmation as ec
from agent.lid import LIDResult
from agent.reply_templates import test_rate_reply as price_reply
from agent.sample_wording import is_specimen, sample_sentence


class Result:
    def __init__(self, text, agreement, decoder_used="rnnt"):
        self.text, self.decoder_agreement, self.decoder_used = text, agreement, decoder_used


# ================================================================================ 1. language


class FakeLID:
    def __init__(self, language, scores):
        self._r = LIDResult(language=language, confidence=scores[language], scores=scores)

    def identify_path(self, path):
        return self._r


class FakeRouter:
    def __init__(self, results, delays=None):
        self.results, self.ran, self.delays = results, [], delays or {}

    async def transcribe_many(self, languages, path):
        return [(lang, self.results[lang]) for lang in languages]

    async def transcribe(self, language, path):
        import asyncio

        self.ran.append(language)
        if self.delays.get(language):
            await asyncio.sleep(self.delays[language])
        return self.results[language]


@pytest.mark.asyncio
async def test_the_pod_call_a_bengali_caller_is_answered_in_bengali_not_hindi(m, env, monkeypatch):
    """The exact numbers from the log: LID bn 0.95 / en 0.00 / hi 0.05; agreement bn 0.20, hi 0.50, en 0.33."""
    monkeypatch.undo()
    monkeypatch.setattr(m, "_lid", FakeLID("bn", {"bn": 0.95, "en": 0.0, "hi": 0.05}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(
        m,
        "_asr_router",
        FakeRouter(
            {
                "bn": Result("এইচবিএ ওয়ান সি টেস্টের দাম কত", 0.20),
                "hi": Result("एइबी वन एसी ए टेस्टर्ड दाम को तो", 0.50),
                "en": Result("hb a one c test her dam co", 0.33),
            }
        ),
    )
    env.session.lang_router.note_response_language("bn")  # the call opened in Bengali
    lang, result = await m._route_and_transcribe(env.session, "x.wav")
    assert lang == "bn" and result.text.startswith("এইচবিএ")


@pytest.mark.asyncio
async def test_a_call_in_bengali_does_not_flip_to_hindi_on_a_weak_hindi_read_even_with_no_prior_turn(
    m, env, monkeypatch
):
    monkeypatch.undo()
    monkeypatch.setattr(m, "_lid", FakeLID("bn", {"bn": 0.84, "en": 0.0, "hi": 0.15}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(
        m,
        "_asr_router",
        FakeRouter({"bn": Result("বাংলা", 0.30), "hi": Result("हिंदी", 0.80), "en": Result("english", 0.10)}),
    )
    lang, _ = await m._route_and_transcribe(env.session, "x.wav")
    assert lang == "bn"


@pytest.mark.asyncio
async def test_a_genuine_hindi_caller_is_still_answered_in_hindi(m, env, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(m, "_lid", FakeLID("hi", {"bn": 0.02, "en": 0.0, "hi": 0.98}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(
        m,
        "_asr_router",
        FakeRouter({"bn": Result("बंगाली", 0.10), "hi": Result("सीबीसी का रेट", 0.90), "en": Result("x", 0.0)}),
    )
    lang, _ = await m._route_and_transcribe(env.session, "x.wav")
    assert lang == "hi"


# ================================================================ 2. ask for the name, not the whole sentence


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
async def test_when_only_the_name_was_not_trusted_the_agent_asks_for_the_name(m, env, monkeypatch, lang):
    heard = {"bn": "ইউরিন টেস্টের দাম কত", "hi": "यूरिन टेस्ट की कीमत क्या है", "en": "what is the price of a urine test"}[
        lang
    ]

    async def route(session, wav):
        return lang, Result(heard, 0.44)  # a well-formed sentence; the recognisers just disagreed

    monkeypatch.setattr(m, "_route_and_transcribe", route)
    env.session.disclosed_langs.add(lang)
    said = await env.say(heard, "test_rate", {"test_name": "ইউরিন"})
    # DELIBERATE spec change (a live call, 2026-09-28): the model DID extract a name ("ইউরিন") even though the
    # recognisers disagreed -- the agent now reads it back and asks a yes/no ("do you mean X?") instead of a
    # blank "I didn't catch it, which test?" -- see main.py's REPEAT branch and tests/test_no_guess.py's own
    # version of this same change.
    assert said[-1] == ec.confirm_question("test_name", "ইউরিন", lang)
    assert "?" in said[-1] and said[-1].count("?") == 1  # one question: which test


@pytest.mark.asyncio
async def test_a_doctor_question_asks_for_the_doctor_and_an_unnamed_intent_keeps_the_general_reply(m, env, monkeypatch):
    from agent.reply_templates import insufficient_information_reply

    async def route(session, wav):
        return "en", Result("garbled", 0.44)

    monkeypatch.setattr(m, "_route_and_transcribe", route)
    said = await env.say("garbled", "doctor_availability", {"doctor_name": "x"})
    # DELIBERATE spec change: see the read-back comment above -- a name WAS extracted ("x"), so it is read back.
    assert said[-1] == ec.confirm_question("doctor_name", "x", "en")
    said = await env.say("garbled", "clinic_faq", {"faq_topic": "hours"})
    assert said[-1] == insufficient_information_reply("en")  # no name to ask for: the general reply


@pytest.mark.asyncio
async def test_a_trusted_turn_is_still_answered_straight_away(m, env, monkeypatch):
    async def route(session, wav):
        return "en", Result("what is the price of a CBC test", 0.90)

    monkeypatch.setattr(m, "_route_and_transcribe", route)
    said = await env.say("x", "test_rate", {"test_name": "CBC"})
    assert "350" in " ".join(said) or "three hundred" in " ".join(said)


# ================================================================================================ 3. the sample


@pytest.mark.parametrize("sample", ["Blood", "Urine", "Stool", "Serum", "Saliva", "Swab", "Plasma"])
def test_every_specimen_has_a_sentence_in_every_language(sample):
    for lang in ("bn", "hi", "en"):
        s = sample_sentence(sample, lang)
        assert s and ":" not in s and "{" not in s
        assert not any("a" <= c.lower() <= "z" for c in s) or lang == "en"  # no Latin word inside bn/hi speech


@pytest.mark.parametrize("category", ["Imaging", "Cardiac", "Sample (Cervical)", "", None])
def test_a_category_that_is_not_a_specimen_says_nothing_about_a_sample(category):
    assert sample_sentence(category, "bn") == "" and not is_specimen(category)


def test_the_price_reply_is_a_sentence_in_all_three_languages():
    result = {
        "found": True,
        "test_name": "HbA1c",
        "test_name_bn": "এইচবিএ১সি",
        "test_name_hi": "एचबीए वन सी",
        "rate_inr": 650,
        "sample_type": "Blood",
        "report_time_hours": 24,
    }
    bn, hi, en = (price_reply({"test_name": "x"}, result, lang) for lang in ("bn", "hi", "en"))
    assert "রক্তের নমুনা" in bn and "ब्लड का सैंपल" in hi and "A blood sample is needed." in en
    for text in (bn, hi, en):
        assert ":" not in text


def test_bengali_no_longer_reads_a_scan_category_out_as_a_sample():
    result = {
        "found": True,
        "test_name": "Chest X-Ray",
        "test_name_bn": "বুকের এক্স-রে",
        "rate_inr": 400,
        "sample_type": "Imaging",
        "report_time_hours": 4,
    }
    assert "নমুনা" not in price_reply({"test_name": "x"}, result, "bn")


# ================================================================= 4. language ID and recognition overlap; no wasted engine


@pytest.mark.asyncio
async def test_a_bengali_turn_runs_bengali_and_english_and_never_hindi(m, env, monkeypatch):
    monkeypatch.undo()
    router = FakeRouter({"bn": Result("বাংলা", 0.90), "hi": Result("हिंदी", 0.10), "en": Result("x", 0.0)})
    monkeypatch.setattr(m, "_lid", FakeLID("bn", {"bn": 0.95, "en": 0.0, "hi": 0.05}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(m, "_asr_router", router)
    lang, _ = await m._route_and_transcribe(env.session, "x.wav")
    assert lang == "bn" and sorted(router.ran) == ["bn", "en"]  # Hindi's GPU time is not spent


@pytest.mark.asyncio
async def test_recognition_starts_before_language_id_finishes(m, env, monkeypatch):
    """The recognisers for the call's language and for English start at the same instant as language ID."""
    import time

    monkeypatch.undo()
    started = {}

    class SlowLID(FakeLID):
        def identify_path(self, path):
            time.sleep(0.3)
            started["lid_done"] = time.monotonic()
            return self._r

    class Router(FakeRouter):
        async def transcribe(self, language, path):
            started.setdefault(language, time.monotonic())
            return await super().transcribe(language, path)

    monkeypatch.setattr(m, "_lid", SlowLID("bn", {"bn": 0.95, "en": 0.0, "hi": 0.05}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(
        m, "_asr_router", Router({"bn": Result("বাংলা", 0.9), "hi": Result("x", 0.0), "en": Result("x", 0.0)})
    )
    t0 = time.monotonic()
    await m._route_and_transcribe(env.session, "x.wav")
    assert started["bn"] - t0 < 0.1 and started["en"] - t0 < 0.1  # not after the 0.3 s of language ID


@pytest.mark.asyncio
async def test_hindi_is_run_the_moment_language_id_points_to_it(m, env, monkeypatch):
    monkeypatch.undo()
    router = FakeRouter({"bn": Result("বাংলা", 0.1), "hi": Result("सीबीसी का रेट", 0.9), "en": Result("x", 0.0)})
    monkeypatch.setattr(m, "_lid", FakeLID("hi", {"bn": 0.02, "en": 0.0, "hi": 0.98}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(m, "_asr_router", router)
    lang, result = await m._route_and_transcribe(env.session, "x.wav")
    assert lang == "hi" and "hi" in router.ran


@pytest.mark.asyncio
async def test_english_speech_that_language_id_calls_bengali_is_still_found(m, env, monkeypatch):
    """The reason English always runs: bn 0.83 / hi 0.16 / en 0.01 was measured on English audio."""
    monkeypatch.undo()
    router = FakeRouter(
        {
            "bn": Result("হোয়াট ইজ দ্য প্রাইজ", 1.00),
            "hi": Result("x", 0.0),
            "en": Result("what is the price of the uric acid test", 0.78),
        }
    )
    monkeypatch.setattr(m, "_lid", FakeLID("bn", {"bn": 0.83, "hi": 0.16, "en": 0.01}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(m, "_asr_router", router)
    lang, _ = await m._route_and_transcribe(env.session, "x.wav")
    assert lang == "en"


@pytest.mark.asyncio
async def test_a_failed_engine_is_dropped_not_fatal(m, env, monkeypatch):
    monkeypatch.undo()

    class Router(FakeRouter):
        async def transcribe(self, language, path):
            if language == "en":
                raise RuntimeError("english asr is down")
            return await super().transcribe(language, path)

    monkeypatch.setattr(m, "_lid", FakeLID("bn", {"bn": 0.95, "en": 0.0, "hi": 0.05}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(
        m, "_asr_router", Router({"bn": Result("বাংলা", 0.9), "hi": Result("x", 0), "en": Result("x", 0)})
    )
    lang, _ = await m._route_and_transcribe(env.session, "x.wav")
    assert lang == "bn"


# ================================================================================ 5. turn timing survives a filler


@pytest.mark.asyncio
async def test_turn_timing_is_logged_against_the_real_answer_not_swallowed_by_the_filler(m, env, monkeypatch, caplog):
    """A live call (2026-09-28): a slow intent-extraction call speaks the "please hold" filler first, THEN the real
    answer. _mark()/_log_timing() used to fire on whichever _speak() call came first in the turn -- the filler,
    since it is spoken before the slow call returns -- clearing session.marks before the real answer's own
    "reply"/"tts"/"send" marks could ever be recorded. The one "turn timing" line a turn like this produced looked
    fast (just the filler's pre-cached synthesis) and the actual multi-second wait the caller sat through was
    never logged at all -- exactly why the live ~8s gap had no breakdown to point at. Fixed in main.py's _speak()
    by skipping the marks while session.speaking_filler is set, so they survive to be logged against the real reply."""
    import asyncio

    async def slow_resolve(session, text, lang):
        # The real _resolve_intent's own filler-racing wrapper (agent/filler.py, via _await_with_filler), around a
        # stand-in for the slow part (fast_path/cache/model) so this test exercises the genuine race, not a re-
        # implementation of it -- only what a slow lookup returns is faked.
        async def slow():
            await asyncio.sleep(0.9)  # > FILLER_THRESHOLD_S (0.7s): forces the filler to speak first
            return {
                "secondary_intent": None,
                "direct_reply_bn": None,
                "intent": "test_rate",
                "slots": {"test_name": "CBC"},
            }

        return await m._await_with_filler(session, slow(), lang)

    monkeypatch.setattr(m, "_resolve_intent", slow_resolve)
    with caplog.at_level("INFO"):
        said = await env.say("what does the CBC cost", "test_rate", {"test_name": "CBC"})
    assert said[0] == m.phrase("please_wait", "en")  # the filler really did speak first (env's route() reports "en")
    assert said[-1] == env.state["reply"]  # ...and the real answer still followed it

    timing_lines = [r.message for r in caplog.records if r.name == "main" and "turn timing" in r.message]
    assert timing_lines, "no turn-timing line was logged at all -- the filler swallowed it again"
    # The delta leading up to "intent" (when _resolve_intent's awaitable finally returns) must cover the ~0.9s
    # the slow call took -- not 0/near-0, which is what it would be if the filler's own (pre-cached, near-
    # instant) synthesis had already cleared the marks and reset the clock before "intent" was ever recorded.
    assert "intent " in timing_lines[-1]
    intent_ms = int(timing_lines[-1].split("intent ", 1)[1].split(" ms", 1)[0])
    assert intent_ms >= 800
