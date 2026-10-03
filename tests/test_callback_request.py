"""agent/callback_request.py and agent/callback_config.py.

The story behind this port rests on one fact about the deployment: there is no
outbound calling in this stack. So the tests that matter are not really about
recall -- they are about what the agent is allowed to SAY. Three of the groups
below exist to pin that: no line may imply the agent itself will ring, no line
may promise a time (this repository has no structured opening hours to judge one
from, only localised free text), and the reason recorded beside the request must
never be an invented "general enquiry".

The detector is new code rather than a port -- in the module this came from,
"caller asks to be called back" was an LLM intent, and this merge leaves the
intent set and the prompt untouched. So its boundary against
agent/doctor_personal_request.py is tested explicitly: "have the doctor call me"
belongs to that guard, because no specific doctor can be promised a callback.
"""

from __future__ import annotations

import importlib

import pytest

from agent import persona
from agent.callback_request import asks_for_a_callback, build_callback_reason, detect_callback_request
from agent.doctor_personal_request import is_doctor_personal_request
from agent.phrases import PHRASES, phrase

LANGS = ("bn", "hi", "en")
KEYS = ("callback_noted", "callback_need_number", "callback_unavailable")


@pytest.mark.parametrize(
    "text,family",
    [
        ("please call me back", "en"),
        ("can someone call me", "en"),
        ("i want a callback", "en"),
        ("could you call me back tomorrow", "en"),
        ("mujhe wapas call kijiye", "latin"),
        ("baad me call kijiye", "latin"),
        ("amake pore phone korun", "latin"),
        ("আমাকে কল ব্যাক করুন", "bn"),
        ("কেউ আমাকে ফোন করুন", "bn"),
        ("আমাকে পরে ফোন করুন", "bn"),
        ("मुझे वापस कॉल कीजिए", "hi"),
        ("कोई मुझे फ़ोन करे", "hi"),
        ("बाद में कॉल कीजिए", "hi"),
    ],
)
def test_a_callback_request_is_caught_in_every_language(text, family):
    matched, tag = detect_callback_request(text)
    assert matched is True
    assert tag == family
    assert asks_for_a_callback(text) is True


@pytest.mark.parametrize("lang", LANGS)
def test_every_language_has_its_own_phrase_list(lang):
    import agent.callback_request as m

    table = {"bn": m._BN, "hi": m._HI, "en": m._EN}[lang]
    assert len(table) >= 10, f"{lang} list is too thin to be real coverage"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "what is the price of a CBC test",
        "I want to book an appointment",
        "is Dr Sen available tomorrow",
        "my phone number is 9876543210",
        "আমার রিপোর্ট কোথায়",
        "मुझे रिपोर्ट चाहिए",
    ],
)
def test_ordinary_business_is_not_a_callback_request(text):
    assert asks_for_a_callback(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "I want my doctor to call me",
        "ডাক্তার আমাকে কল করুন",
        "डॉक्टर मुझे कॉल करें",
        "please ask Dr Sen to call me",
    ],
)
def test_asking_for_a_doctor_to_ring_belongs_to_the_doctor_guard(text):
    """Checked earlier in the turn, and it must stay that way: a callback from a
    NAMED doctor is the one thing this system must not promise, and the generic
    "a colleague will call you back" would read as exactly that promise."""
    assert is_doctor_personal_request(text) is True


# ------------------------------------------------------- the recorded reason
def test_the_callers_own_words_are_preferred_as_the_reason():
    assert build_callback_reason("I was overcharged for the CBC") == "I was overcharged for the CBC"
    assert build_callback_reason("  spaced  ") == "spaced"


def test_an_entity_already_confirmed_in_the_call_is_used_when_nothing_was_said():
    assert build_callback_reason(None, active_test="CBC") == "Regarding CBC"
    assert build_callback_reason(None, active_doctor="Dr Sen") == "Regarding Dr Sen"
    assert build_callback_reason(None, active_package="Full Body") == "Regarding Full Body"


def test_a_stated_reason_always_beats_a_tracked_entity():
    assert build_callback_reason("billing problem", active_test="CBC") == "billing problem"


def test_no_reason_is_recorded_as_no_reason_and_never_invented():
    """Deliberate: an honest empty reason is preserved. A fabricated "general
    enquiry" would put a sentence nobody said into a durable record."""
    assert build_callback_reason(None) is None
    assert build_callback_reason("") is None
    assert build_callback_reason("   ") is None


# ------------------------------------------------------------- the config
def test_callbacks_are_on_unless_switched_off(monkeypatch):
    """Unset means ON, so an existing deployment behaves as it did before this
    was added."""
    import agent.callback_config as cfg

    monkeypatch.delenv("CALLBACKS_ENABLED", raising=False)
    assert importlib.reload(cfg).CALLBACKS_ENABLED is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "disabled", "OFF", " False "])
def test_the_spellings_of_off_a_person_actually_types_all_work(monkeypatch, value):
    import agent.callback_config as cfg

    monkeypatch.setenv("CALLBACKS_ENABLED", value)
    assert importlib.reload(cfg).CALLBACKS_ENABLED is False


def test_an_unrecognised_value_leaves_the_feature_on(monkeypatch):
    """Fail-open, deliberately: a typo in an env file must not silently remove a
    caller-facing feature. The honest line is spoken only when it is really off."""
    import agent.callback_config as cfg

    monkeypatch.setenv("CALLBACKS_ENABLED", "yes-please")
    assert importlib.reload(cfg).CALLBACKS_ENABLED is True
    monkeypatch.delenv("CALLBACKS_ENABLED", raising=False)
    importlib.reload(cfg)


# ------------------------------------------------- what the agent may say
@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("lang", LANGS)
def test_each_line_exists_in_all_three_languages_and_passes_the_persona(key, lang):
    assert key in PHRASES[lang]
    text = phrase(key, lang)
    assert text.strip()
    assert persona.violations(text) == [], persona.violations(text)


def test_phrase_coverage_is_still_identical_across_languages():
    assert sorted(PHRASES["bn"]) == sorted(PHRASES["hi"]) == sorted(PHRASES["en"])


def test_no_line_claims_the_agent_itself_will_ring_back():
    """The whole point of the story. "I will call you" is a promise this stack
    cannot keep; "a colleague will call you" is the truth."""
    text = phrase("callback_noted", "en").lower()
    assert "colleague" in text
    for false_promise in ("i will call you", "i'll call you", "i will ring you", "we will call you back at"):
        assert false_promise not in text


def test_no_line_promises_a_time():
    """There is no structured opening-hours data in this repository to judge a
    time window from -- only localised free text -- so any specific promise here
    would be an invented fact."""
    text = phrase("callback_noted", "en").lower()
    assert "not able to say exactly when" in text
    for timed in ("within an hour", "today", "in 24 hours", "by this evening", "shortly"):
        assert timed not in text


def test_the_number_request_asks_exactly_one_question():
    for lang in LANGS:
        assert phrase("callback_need_number", lang).count("?") == 1


def test_the_unavailable_line_declines_without_pretending_and_keeps_the_call_going():
    text = phrase("callback_unavailable", "en").lower()
    assert "not able to arrange a callback" in text
    assert "help" in text, "the caller is left with nowhere to go"
