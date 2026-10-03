"""The fourth live call (2026-09-25):

  AI: আর কিছু জানতে চান? না চাইলে আমি কল শেষ করে দেব।   caller: শেষ করে দিন    -> the model asked for a confirmation number
  caller: তুই বাল                                          -> "আপনি কেমন আছেন?", then a re-ask in HINDI

1. "end it" as the answer to "anything else?" ends the call (agent/call_end.py), by words and not the model;
2. swearing gets a calm fixed boundary in the caller's language, and the third time a courteous close (agent/abuse.py);
3. a turn that cannot be read never moves the call to another language.

  python -m pytest tests/test_call_end_and_abuse.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_orchestrator_booking_flow import env, m  # noqa: F401  (the harness: real dispatch, fake tools)

from agent import abuse, call_end
from agent.phrases import PHRASES, phrase

# ======================================================================================== ending the call


@pytest.mark.parametrize(
    "text,lang",
    [
        ("শেষ করে দিন", "bn"),
        ("শেষ করুন", "bn"),
        ("না, আর কিছু না", "bn"),
        ("থাক", "bn"),
        ("खत्म कर दीजिए", "hi"),
        ("बस", "hi"),
        ("nothing else", "en"),
        ("no thanks", "en"),
        ("that's it", "en"),
        ("end it", "en"),
    ],
)
def test_the_answer_to_anything_else_that_means_no_ends_the_call(text, lang):
    assert call_end.wants_to_end(text, lang, after_prompt=True)


@pytest.mark.parametrize(
    "text,lang",
    [
        ("কল কেটে দিন", "bn"),
        ("ফোন রাখছি", "bn"),
        ("कॉल खत्म करो", "hi"),
        ("फोन रखता हूँ", "hi"),
        ("end the call", "en"),
        ("goodbye", "en"),
        ("that's all", "en"),
        ("bye", "en"),
        # A live call (2026-09-28): "কেটে দিন" said mid-conversation, not right after the "anything else?" prompt,
        # was not recognised -- "cut" is unambiguous for ending a call (unlike "শেষ"/finish, which also means
        # finishing a booking), so it belongs here, not gated behind after_prompt.
        ("কেটে দিন", "bn"),
        ("কেটে দাও", "bn"),
        ("কাটো", "bn"),
        ("দাদা একটু কাটো তো", "bn"),  # a whole-token match: "কাটো" inside an ordinary sentence still counts
        ("काट दीजिए", "hi"),
        ("काट दो", "hi"),
        ("काटो", "hi"),
    ],
)
def test_an_explicit_request_ends_the_call_at_any_moment(text, lang):
    assert call_end.wants_to_end(text, lang, after_prompt=False)


@pytest.mark.parametrize("text,lang", [("শেষ করে দিন", "bn"), ("শেষ", "bn"), ("nothing", "en"), ("बस", "hi")])
def test_a_bare_end_word_does_not_hang_up_unless_the_agent_asked(text, lang):
    assert not call_end.wants_to_end(text, lang, after_prompt=False)


@pytest.mark.parametrize(
    "text,lang",
    [
        ("বুকিং শেষ করে দিন", "bn"),
        ("আমি বুকিং শেষ করতে চাই", "bn"),
        ("finish the booking", "en"),
        ("maybe tomorrow", "en"),
        ("what is the price", "en"),
        ("", "en"),
        ("अपॉइंटमेंट बस कर दीजिए", "hi"),
    ],
)
def test_a_task_word_or_an_ordinary_sentence_never_hangs_up_even_after_the_prompt(text, lang):
    assert not call_end.wants_to_end(text, lang, after_prompt=True)


@pytest.mark.asyncio
async def test_end_it_after_the_prompt_says_goodbye_and_closes_without_the_model(m, env):
    s = env.session
    s.awaiting_close_answer, s.silence_prompts = True, 1
    env.state["lang"] = "bn"

    async def no_model(session, text, lang):
        raise AssertionError("the model must not be asked")

    m._resolve_intent = no_model
    said = await env.say("শেষ করে দিন")
    assert said == [phrase("silence_goodbye", "bn")] and s.ws.closed


# ============================================================================================= abuse and slang


@pytest.mark.parametrize(
    "text",
    [
        "তুই বাল",
        "তুমি মাদারচোদ",
        "madarchod",
        "you are a bastard",
        "fuck you",
        "तू भोसड़ी का",
        "tum chutiya ho",
        "শুয়োরের বাচ্চা",
        "কুত্তার বাচ্চা",
    ],
)
def test_swearing_is_recognised_in_all_three_languages_and_in_latin_spelling(text):
    assert abuse.is_abusive(text)


@pytest.mark.parametrize(
    "text",
    [
        "আমাকে বলুন",
        "বালক ছেলের জন্য ডাক্তার",
        "বালি",
        "what is the price",
        "কবে বসছে",
        "ডক্টর পার্থ রায়",
        "সব ঠিক আছে",
        "sale price",
        "साला",
        "hello",
    ],
)
def test_ordinary_words_that_look_like_swearing_are_left_alone(text):
    assert not abuse.is_abusive(text)


@pytest.mark.asyncio
async def test_slang_gets_a_calm_fixed_answer_in_the_callers_language_and_never_reaches_the_model(m, env):
    s = env.session
    env.state["lang"] = "hi"  # language ID mislabelled it; the SCRIPT says Bengali

    async def no_model(session, text, lang):
        raise AssertionError("the model must not be asked")

    m._resolve_intent = no_model
    said = await env.say("তুই বাল")
    assert said == [phrase("abuse_first", "bn")] and s.lang == "bn" and not s.ws.closed


@pytest.mark.asyncio
async def test_the_boundary_gets_plainer_and_the_third_time_the_call_closes_politely(m, env):
    s = env.session
    env.state["lang"] = "bn"
    a = await env.say("তুই বাল")
    b = await env.say("খানকির ছেলে")
    assert a == [phrase("abuse_first", "bn")] and b == [phrase("abuse_second", "bn")] and not s.ws.closed
    c = await env.say("হারামজাদা")
    assert c == [phrase("abuse_final", "bn")] and s.ws.closed and s.outcome == "abusive_caller"


def test_the_abuse_lines_exist_in_every_language_with_at_most_one_question():
    for lang in ("bn", "hi", "en"):
        for key in ("abuse_first", "abuse_second", "abuse_final"):
            line = PHRASES[lang][key]
            assert line.count("?") <= 1 and not abuse.is_abusive(line)


# ========================================================================= an unreadable turn keeps the language


@pytest.mark.asyncio
async def test_a_turn_that_cannot_be_read_never_moves_the_call_to_another_language(m, env):
    """The live call: a Bengali call was re-asked in Hindi because language ID labelled a mumble Hindi."""
    s = env.session
    s.lang = "bn"
    env.state["lang"] = "hi"  # what recognition claimed for an empty transcript
    said = await env.say("")
    assert s.lang == "bn" and said and not any("ऀ" <= ch <= "ॿ" and ch not in "।॥" for ch in " ".join(said)), said
