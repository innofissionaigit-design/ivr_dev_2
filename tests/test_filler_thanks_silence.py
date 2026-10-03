"""The third live conversation (2026-09-25), the owner's three instructions:

    caller: ইউরিক অ্যাসিড টেস্টের দাম কত   AI: "বলার জন্য ধন্যবাদ। ইউরিক অ্যাসিড টেস্টের রেট 250 টাকা ..."
    caller: ফাস্টিং করতে হবে               AI: "একটু দেখছি।"  ...  "বলার জন্য ধন্যবাদ। ইউরিক অ্যাসিড টেস্টের জন্য: ..."

  1. a holding phrase ("আচ্ছা, বলছি।") only when the answer takes MORE THAN 0.7 s from when the caller stopped;
  2. no "thank you for telling me" on an answer to the caller's own question -- only a bare "ধন্যবাদ।" when the caller has
     just answered a question the agent asked;
  3. five seconds of silence after the agent finished: ask whether there is anything else (the call ends if not).
  And the follow-up that took the model 1.5 s ("do I need to fast?" straight after a price) is now decided with no model.

    python -m pytest tests/test_filler_thanks_silence.py -v

Real dispatch and helpers from main_pcm.py with fakes for the pod-only parts (same harness as
tests/test_orchestrator_booking_flow.py). Timing tests use small REAL waits.
"""

import asyncio
import datetime
import os
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_orchestrator_booking_flow import PRICE, book, env, m, text_of  # noqa: F401  (the harness)

from agent import fast_path_cues as cues
from agent.enquiry_followup import EnquiryEntity, recent_unique
from agent.fast_path import Catalogue, FastPath
from agent.phrases import phrase
from agent.turn_ack import THANKS, thanks_for, validate_thanks

TODAY = datetime.date(2026, 9, 25)


# ==================================================================================================== 1. the filler


async def _finishes_after(seconds, value="answer"):
    await asyncio.sleep(seconds)
    return value


@pytest.mark.asyncio
async def test_no_filler_when_the_answer_comes_before_the_threshold(m, env):
    env.session.turn_started_at = time.monotonic()
    out = await m._await_with_filler(env.session, _finishes_after(0.05), "en", threshold_s=0.4)
    assert out == "answer" and env.session.ws.spoken() == []


@pytest.mark.asyncio
async def test_the_filler_is_spoken_once_when_the_answer_is_late(m, env):
    env.session.turn_started_at = time.monotonic()
    out = await m._await_with_filler(env.session, _finishes_after(0.6), "en", threshold_s=0.3)
    assert out == "answer" and env.session.ws.spoken() == [phrase("please_wait", "en")]


@pytest.mark.asyncio
async def test_time_already_spent_on_recognition_counts_towards_the_700_ms(m, env):
    """The turn started 0.45 s ago (recognition): only what is left of the 0.7 s is waited, so the filler comes at
    0.7 s from the caller's silence, not at 0.7 s after recognition finished (which would be 1.15 s)."""
    env.session.turn_started_at = time.monotonic() - 0.45
    t0 = time.monotonic()
    await m._await_with_filler(env.session, _finishes_after(0.6), "en", threshold_s=0.7)
    assert env.session.ws.spoken() == [phrase("please_wait", "en")]
    assert time.monotonic() - t0 < 0.7  # it finished, having spoken the filler at ~0.25 s


@pytest.mark.asyncio
async def test_a_turn_already_slow_still_gives_a_cache_hit_its_moment(m, env):
    """Never below FILLER_MIN_WAIT_S: a lookup that hits at once is not announced by a filler."""
    env.session.turn_started_at = time.monotonic() - 5.0
    out = await m._await_with_filler(env.session, _finishes_after(0.05), "en", threshold_s=0.7)
    assert out == "answer" and env.session.ws.spoken() == []


# ================================================================================================== 2. the thank-you


def test_the_thanks_is_one_word_in_every_language():
    assert THANKS == {"bn": "ধন্যবাদ।", "hi": "धन्यवाद।", "en": "Thank you."}
    assert validate_thanks() == []
    assert thanks_for("bn") == "ধন্যবাদ।"


@pytest.mark.asyncio
async def test_an_answer_to_the_callers_own_question_is_not_thanked(m, env):
    said = await env.say("how much is the CBC", "test_rate", {"test_name": "CBC"})
    assert said == [PRICE]  # the answer, and nothing before it
    said = await env.say("do I need to fast", "test_prep", {"test_name": "CBC"})
    assert "Thank you" not in text_of(said)


@pytest.mark.asyncio
async def test_a_detail_the_caller_was_asked_for_is_thanked_once_and_only_then(m, env):
    said = await book(env, doctor_name="Sen")
    assert "Thank you" not in text_of(said)  # the first request answers "how can I help"
    said = await env.say("tomorrow at ten", "book_appointment", {"date": "2026-10-01", "time_slot": "10:00"})
    assert said[0] == "Thank you." and text_of(said).count("Thank you") == 1
    said = await env.say("Ravi Das 9876543210", "book_appointment", {"patient_name": "Ravi Das", "phone": "9876543210"})
    assert said[0] == "Thank you." and text_of(said).count("Thank you") == 1


@pytest.mark.asyncio
async def test_a_question_of_theirs_in_the_middle_of_a_booking_is_answered_not_thanked(m, env):
    await book(env, doctor_name="Sen")
    said = await env.say("how much is the CBC", "test_rate", {"test_name": "CBC"})
    assert "Thank you" not in text_of(said)


@pytest.mark.asyncio
async def test_no_thanks_when_nothing_was_asked(m, env):
    """The agent's last words were not a question (an answer), so the next thing the caller says is not a reply."""
    await env.say("how much is the CBC", "test_rate", {"test_name": "CBC"})
    env.session.awaiting_answer = False
    env.state["reply"] = "Dr Sen sits on Monday."
    said = await env.say("I want to book", "book_appointment", {"doctor_name": "Sen", "date": "2026-10-01"})
    assert "Thank you" not in text_of(said)


# ============================================================================================ 3. five seconds of silence


@pytest.mark.asyncio
async def test_silence_starts_a_timer_and_speech_or_a_busy_turn_cancels_it(m, env):
    s = env.session
    assert await m._handle_silence(s, True) is False and s.silence_since is not None
    await m._handle_silence(s, False)
    assert s.silence_since is None
    await m._handle_silence(s, True)
    async with s.dispatch_lock:  # the agent is mid-turn: not the caller's silence
        await m._handle_silence(s, True)
    assert s.silence_since is None and s.ws.spoken() == []


@pytest.mark.asyncio
async def test_after_five_seconds_it_asks_once_whether_there_is_anything_else(m, env):
    s = env.session
    s.lang = "en"
    s.silence_since = time.monotonic() - (m.SILENCE_PROMPT_S + 0.1)
    assert await m._handle_silence(s, True) is False
    assert s.ws.spoken() == [phrase("silence_prompt", "en")] and s.silence_prompts == 1 and s.awaiting_close_answer
    assert s.ws.closed is False


@pytest.mark.asyncio
async def test_four_seconds_is_not_enough(m, env):
    s = env.session
    s.silence_since = time.monotonic() - 4.0
    await m._handle_silence(s, True)
    assert s.ws.spoken() == []


@pytest.mark.asyncio
async def test_still_silent_after_the_prompt_the_call_ends_politely(m, env, monkeypatch):
    s = env.session
    s.lang = "en"
    s.silence_prompts = 1
    s.silence_since = time.monotonic() - (m.SILENCE_CLOSE_S + 0.1)
    assert await m._handle_silence(s, True) is True
    assert s.ws.spoken() == [phrase("idle_close", "en")] and s.ws.closed and s.outcome == "silence_timeout"


@pytest.mark.asyncio
async def test_a_no_to_anything_else_says_goodbye_and_ends_the_call(m, env):
    s = env.session
    s.awaiting_close_answer, s.silence_prompts = True, 1
    said = await env.say("no")
    assert said == [phrase("silence_goodbye", "en")] and s.ws.closed and s.awaiting_close_answer is False


@pytest.mark.asyncio
async def test_a_yes_to_anything_else_invites_them_to_go_on(m, env):
    s = env.session
    s.awaiting_close_answer = True
    said = await env.say("yes")
    assert said == [phrase("silence_go_on", "en")] and not s.ws.closed


@pytest.mark.asyncio
async def test_a_real_question_after_the_prompt_is_simply_answered(m, env):
    s = env.session
    s.awaiting_close_answer, s.silence_prompts = True, 1
    said = await env.say("how much is the CBC", "test_rate", {"test_name": "CBC"})
    assert said == [PRICE] and not s.ws.closed and s.silence_prompts == 0 and s.awaiting_close_answer is False


def test_the_silence_wording_is_one_question_per_language():
    for lang in ("bn", "hi", "en"):
        line = phrase("silence_prompt", lang)
        assert line.count("?") == 1  # the policy may drop a second question


# ================================================================== 4. the follow-up that no longer needs the model


@pytest.fixture(scope="module")
def fp():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        return FastPath(Catalogue(client.get("/api/v1/catalogue").json()), today=TODAY)


def test_do_i_need_to_fast_after_a_price_is_served_about_that_test_with_no_model(fp):
    hit = fp.resolve("ফাস্টিং করতে হবে", "bn", topic_test="Uric Acid")
    assert hit is not None and hit.intent == "test_prep" and hit.slots["test_name"] == "Uric Acid"
    hit = fp.resolve("do I need to fast", "en", topic_test="Uric Acid")
    assert hit is not None and hit.intent == "test_prep" and hit.slots["test_name"] == "Uric Acid"
    hit = fp.resolve("क्या फास्टिंग करना पड़ेगा", "hi", topic_test="Uric Acid")
    assert hit is not None and hit.intent == "test_prep" and hit.slots["test_name"] == "Uric Acid"


def test_a_cue_word_alone_never_picks_a_test_by_its_own_spelling(fp):
    """Before: the Hindi "do I need to fast" matched the test called "शुगर फास्टिंग" through the word for fasting and
    was answered for a blood-sugar test, with no topic and no model. Nothing is named, so nothing is looked up."""
    assert fp.resolve("क्या फास्टिंग करना पड़ेगा", "hi") is None
    assert fp.resolve("do I need to fast", "en") is None
    hit = fp.resolve("शुगर फास्टिंग के लिए क्या तैयारी", "hi")  # a test IS named: served as before
    assert hit is not None and hit.slots["test_name"] == "शुगर फास्टिंग"


def test_a_bare_price_question_after_a_test_is_served_about_that_test(fp):
    hit = fp.resolve("দাম কত", "bn", topic_test="Uric Acid")
    assert hit is not None and hit.intent == "test_rate" and hit.slots["test_name"] == "Uric Acid"
    hit = fp.resolve("how much is it", "en", topic_test="Uric Acid")
    assert hit is not None and hit.intent == "test_rate" and hit.slots["test_name"] == "Uric Acid"


def test_without_a_recent_topic_it_abstains_as_before(fp):
    assert fp.resolve("ফাস্টিং করতে হবে", "bn") is None
    assert fp.resolve("do I need to fast", "en", topic_test=None) is None


@pytest.mark.parametrize(
    "text,lang",
    [
        ("ফাস্টিং করতে হবে ব্লাড সুগারের জন্য", "bn"),  # names something else: the model / the lookup decides, not the topic
        ("do I need to fast for the lipid profile", "en"),
        ("क्या लिपिड प्रोफाइल के लिए फास्टिंग करना पड़ेगा", "hi"),
        ("ডাক্তার সেন কবে বসবেন ফাস্টিং", "bn"),  # two things at once
    ],
)
def test_a_word_that_could_be_a_name_is_never_read_as_the_topic(fp, text, lang):
    hit = fp.resolve(text, lang, topic_test="Uric Acid")
    assert hit is None or hit.slots["test_name"] != "Uric Acid"


def test_the_recent_topic_is_one_test_within_three_turns():
    one = [EnquiryEntity("test_name", "Uric Acid")]
    two = one + [EnquiryEntity("test_name", "CBC")]
    assert recent_unique(one, "test_name", 1) == "Uric Acid"
    assert recent_unique(one, "test_name", 3) == "Uric Acid" and recent_unique(one, "test_name", 4) is None
    assert recent_unique(two, "test_name", 1) is None and recent_unique(one, "doctor_name", 1) is None
    assert recent_unique([], "test_name", None) is None


def test_every_cue_table_has_function_words_to_recognise_a_bare_follow_up():
    for lang in cues.languages():
        assert cues.table_for(lang).function_words
