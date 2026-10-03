"""Operator-editable wording (KCD-353/500/513), the simple apologies (KCD-514), "speak a little louder"
(KCD-057), the thank-you, the greeting, and the new history questions (medicines, appointments).

    python -m pytest tests/test_messages_and_wording.py -v
"""

import datetime
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import apology, disclosure, messages, turn_ack
from agent import history_templates as ht
from agent import patient_context as pc
from agent.history_intent import detect_history_question
from agent.persona import is_clean
from agent.phrases import PHRASES, greeting_text, phrase

NOW = datetime.datetime(2026, 9, 24, 10, 0, 0)


@pytest.fixture(autouse=True)
def clean_messages():
    messages.clear()
    yield
    messages.clear()


def payload(**by_key):
    return {
        "messages": {k: {lang: {"text": t, "version": 3} for lang, t in v.items()} for k, v in by_key.items()},
        "version": 3,
    }


# ================================================================ the cache


def test_with_nothing_loaded_the_built_in_text_is_used():
    assert (
        messages.text("disclosure", "en", "fallback") == "fallback"
        and messages.stale()
        and messages.label() == "built-in"
    )


def test_a_loaded_message_overrides_the_default_for_that_key_and_language_only():
    messages.load(payload(disclosure={"en": "Hello, this is the assistant."}))
    assert messages.text("disclosure", "en", "d") == "Hello, this is the assistant."
    assert messages.text("disclosure", "bn", "d") == "d"
    assert messages.text("other", "en", "d") == "d"
    assert messages.version() == 3 and messages.label() == "db:3" and not messages.stale()


def test_a_bad_or_empty_payload_never_empties_a_good_cache():
    messages.load(payload(disclosure={"en": "Kept."}))
    for bad in (None, {}, {"messages": None}, "oops", {"nothing": 1}):
        messages.load(bad)
        assert messages.text("disclosure", "en", "d") == "Kept."
    messages.load({"messages": {"disclosure": {"en": {"text": "   "}}}, "version": 1})  # blank text is not a message
    assert messages.text("disclosure", "en", "d") == "d"


def test_the_cache_goes_stale_and_asks_for_a_refresh():
    messages.load(payload(disclosure={"en": "x"}), now=100.0)
    assert not messages.stale(now=100.0 + messages.REFRESH_AFTER_S - 1) and messages.stale(
        now=100.0 + messages.REFRESH_AFTER_S + 1
    )


# ================================================================ the wording it changes


def test_the_disclosure_follows_the_database():
    messages.load(payload(disclosure={"en": "You are speaking with Sonoscan Vaani, an automated helper."}))
    assert disclosure.disclosure_for("en") == "You are speaking with Sonoscan Vaani, an automated helper."
    assert disclosure.version_label().endswith("db:3")


def test_the_greeting_identity_line_follows_the_database_and_the_112_pointer_appears_only_when_set():
    messages.load(payload(greeting_identity={"en": "You are speaking with Sonoscan Vaani, our automated helper."}))
    assert "our automated helper" in greeting_text("en") and "our automated helper" in phrase("greeting", "en")
    assert "112" not in greeting_text("en")
    messages.load(payload(emergency_hint={"en": "In an emergency, please call 112 directly."}))
    assert "112" in greeting_text("en")


def test_the_greeting_is_only_the_welcome_who_you_are_speaking_with_and_the_question():
    """DELIBERATE SPEC CHANGE (owner's instruction after the first live call, 2026-09-25): the KCD-353 opening that
    also said "I am an automated assistant", "staff at any time" and pointed to 112 is no longer spoken in the
    greeting. The full disclosure text is unchanged and still used elsewhere; see agent/phrases.py."""
    assert phrase("greeting", "bn") == "নমস্কার। আপনি সোনোস্ক্যান বাণীর সঙ্গে কথা বলছেন। বলুন, কীভাবে সাহায্য করতে পারি?"
    en = phrase("greeting", "en")
    assert en.startswith("Hello.") and "Sonoscan Vaani" in en and en.rstrip().endswith("How can I help you?")
    assert "automated" not in en and "112" not in en and "staff" not in en
    hi = phrase("greeting", "hi")
    assert "सोनोस्कैन वाणी" in hi and hi.rstrip().endswith("?") and "112" not in hi


def test_any_phrase_can_be_overridden_by_its_key():
    messages.load(payload(unclear={"en": "Pardon? Please say that again."}))
    assert (
        phrase("unclear", "en") == "Pardon? Please say that again."
        and phrase("unclear", "bn") == PHRASES["bn"]["unclear"]
    )


def test_the_cannot_find_wording_is_the_operators_and_is_what_every_history_gap_says():
    en = ht.CANNOT_FIND["en"]
    assert en == "Sorry. I am unable to find the details. Could you please guide me, so that I can help you?"
    for kind in (None, {"success": False}):
        assert pc.answer_appointments(kind, "en", NOW).text == en
    assert ht.no_record("en")[1] == en and ht.cannot_see("hi")[1] == ht.CANNOT_FIND["hi"]
    messages.load(payload(cannot_find={"en": "Sorry. I cannot find it. Please help me."}))
    assert (
        ht.cannot_see("en")[1] == "Sorry. I cannot find it. Please help me."
        and ht.cannot_see("bn")[1] == ht.CANNOT_FIND["bn"]
    )


# ================================================================ KCD-513: the thank-you


def test_the_thanks_opens_every_substantive_reply_when_the_mode_is_always():
    t = turn_ack.AckTracker(mode="always")
    reply = "The CBC costs three hundred and fifty rupees."
    for _ in range(3):
        out, acked = t.decorate(reply, "en")
        assert acked and out == f"{turn_ack.THANKS['en']} {reply}"
    assert t.spoken == 3


def test_the_thanks_is_not_put_in_front_of_an_apology_a_one_word_reply_or_a_non_answer():
    t = turn_ack.AckTracker(mode="always")
    assert t.decorate("Sorry. I could not hear that.", "en")[1] is False
    assert t.decorate("Yes.", "en")[1] is False
    assert t.decorate("The price is three hundred rupees.", "en", substantive=False)[1] is False
    assert t.decorate("The price is three hundred rupees.", "en", already_acknowledged=True)[1] is False


def test_the_thanks_wording_comes_from_the_database_when_it_is_there():
    messages.load(payload(thanks_ack={"en": "Thanks for letting me know."}))
    out, _ = turn_ack.AckTracker(mode="always").decorate("The price is three hundred rupees.", "en")
    assert out.startswith("Thanks for letting me know.")


def test_the_thanks_is_valid_in_three_languages():
    assert turn_ack.validate_thanks() == [] and set(turn_ack.THANKS) == {"bn", "hi", "en"}
    for lang, t in turn_ack.THANKS.items():
        assert is_clean(t, lang)


def test_the_varied_mode_is_unchanged():
    assert turn_ack.validate_table() == []
    t = turn_ack.AckTracker(mode="varied")
    assert t.decorate("The price is three hundred rupees.", "en")[1] is True
    assert t.decorate("The price is three hundred rupees.", "en")[1] is False


# ================================================================ KCD-514: simple apologies


def sentences(text):
    import re

    return [s for s in re.split(r"(?<=[।.?!])\s+", text.strip()) if s]


def test_every_apology_is_one_word_of_sorry_then_a_short_plain_sentence():
    for cause, table in apology._APOLOGY.items():
        for lang, text in table.items():
            parts = sentences(text)
            assert apology.count_apologies(text, lang) == 1, (cause, lang)
            assert len(parts) == 2 and len(parts[0].split()) <= 2, (
                cause,
                lang,
                text,
            )  # "Sorry." / "দুঃখিত।" / "माफ़ कीजिए।"
            assert len(parts[1].split()) <= 8, (cause, lang, text)


@pytest.mark.parametrize(
    "key",
    [
        "asr_empty",
        "unclear",
        "reask_low_volume",
        "reask_noisy",
        "reask_crosstalk",
        "reask_mumbled",
        "reask_generic",
        "reask_final",
    ],
)
def test_every_reask_phrase_has_at_most_one_apology_and_short_sentences(key):
    for lang in ("bn", "hi", "en"):
        text = PHRASES[lang][key]
        assert apology.count_apologies(text, lang) == 1, (lang, key)
        assert all(len(s.split()) <= 14 for s in sentences(text)), (lang, key, text)


def test_a_leading_apology_ending_in_a_full_stop_is_recognised_so_a_second_one_is_removed():
    stacked = "Sorry. I could not hear that. Sorry, please repeat it."
    out, removed = apology.enforce_single_apology(stacked, "en")
    assert apology.count_apologies(out, "en") == 1 and removed == 1
    bn, removed_bn = apology.enforce_single_apology("দুঃখিত। আমি শুনতে পাইনি। দুঃখিত, আবার বলবেন?", "bn")
    assert apology.count_apologies(bn, "bn") == 1 and removed_bn == 1


# ================================================================ KCD-057: ask, kindly, for a louder voice


def test_the_quiet_line_asks_for_a_louder_voice_in_three_languages_and_blames_nobody():
    assert "louder" in PHRASES["en"]["reask_low_volume"] and "please" in PHRASES["en"]["reask_low_volume"].lower()
    assert "জোরে" in PHRASES["bn"]["reask_low_volume"] and "ज़ोर से" in PHRASES["hi"]["reask_low_volume"]
    from agent.reask_policy import ReaskTracker

    d = ReaskTracker().decide(asr_empty=True, audio_issues=["too_quiet"])
    assert d.action == "reask" and d.phrase_key == "reask_low_volume"


# ================================================================ the new history questions


@pytest.mark.parametrize(
    "text,lang,kind",
    [
        ("when is my next appointment", "en", "appointments"),
        ("do I have an appointment", "en", "appointments"),
        ("আমার পরের অ্যাপয়েন্টমেন্ট কবে", "bn", "appointments"),
        ("मेरा अगला अपॉइंटमेंट कब है", "hi", "appointments"),
        ("what medicines was I prescribed", "en", "medicines"),
        ("my medicines", "en", "medicines"),
        ("আমার ওষুধ কী ছিল", "bn", "medicines"),
        ("मेरी दवाइयाँ कौन सी हैं", "hi", "medicines"),
    ],
)
def test_appointment_and_medicine_questions_are_recognised(text, lang, kind):
    assert detect_history_question(text, lang).kind == kind


def test_a_test_history_question_is_still_a_test_history_question():
    assert detect_history_question("when did I last have my CBC", "en").kind == "last_test"
    assert detect_history_question("what tests have I had", "en").kind == "recent_tests"
    assert detect_history_question("book an appointment with Dr Sen", "en") is None


def med_event(i, name, days_ago, bn=None, hi=None):
    on = (NOW.date() - datetime.timedelta(days=days_ago)).isoformat()
    return {
        "id": f"medicine_prescribed:{i}",
        "kind": "medicine_prescribed",
        "fields": {"medicine_name": name, "medicine_name_bn": bn, "medicine_name_hi": hi, "prescribed_on": on},
    }


def test_medicines_are_stated_as_a_fact_with_a_date_in_the_callers_script_and_nothing_more():
    tl = {
        "success": True,
        "as_of": NOW.isoformat(),
        "events": [
            med_event(1, "Metformin", 70, "মেটফর্মিন", "मेटफॉर्मिन"),
            med_event(2, "Metformin", 200),
            med_event(3, "Amlodipine", 71),
        ],
    }
    en = pc.answer_medicines(tl, "en", NOW)
    assert [i for i, _ in en.statements] == [
        "medicine_prescribed:1",
        "medicine_prescribed:3",
    ]  # latest of each, newest first
    assert "Metformin was prescribed to you on" in en.text
    assert (
        "মেটফর্মিন" in pc.answer_medicines(tl, "bn", NOW).text and "मेटफॉर्मिन" in pc.answer_medicines(tl, "hi", NOW).text
    )
    for word in ("dose", "take", "should", "mg", "safe", "advise"):
        assert word not in en.text.lower()


def test_no_medicine_on_record_or_a_stale_record_says_it_cannot_find_or_confirm():
    fresh = {"success": True, "as_of": NOW.isoformat(), "events": []}
    assert pc.answer_medicines(fresh, "en", NOW).ids == ["cannot_see"]
    stale = {
        "success": True,
        "as_of": (NOW - datetime.timedelta(hours=30)).isoformat(),
        "events": [med_event(1, "X", 5)],
    }
    assert pc.answer_medicines(stale, "en", NOW).ids == ["cannot_confirm"]


def test_appointment_and_test_statements_use_the_callers_script_when_the_record_carries_it():
    ev = {
        "id": "test_performed:9",
        "kind": "test_performed",
        "fields": {
            "test_name": "CBC",
            "test_name_bn": "সিবিসি",
            "test_name_hi": "सीबीसी",
            "performed_on": "2026-08-01",
        },
    }
    assert "সিবিসি" in ht.test_performed_statement(ev, "bn")[1] and "सीबीसी" in ht.test_performed_statement(ev, "hi")[1]
    assert "CBC" in ht.test_performed_statement(ev, "en")[1]


# ================================================================ the persona over every new template


def _all_new_templates():
    from agent import entity_confirmation as ec
    from agent import security_check as scq
    from agent import senior_care as sen
    from agent.phrases import EMERGENCY_HINT

    tables = [
        scq.INTRO,
        scq.DID_NOT_UNDERSTAND,
        scq.NOT_MATCHED,
        scq.VERIFIED,
        scq.FAILED,
        scq.FIND_BY_DETAILS,
        *scq.QUESTION.values(),
        sen.OPENING,
        sen.CLOSING,
        sen.PATIENCE,
        sen.WARM_ACK,
        ec.CONFIRM_NAME,
        ec.CONFIRM_NUMBER,
        ec.REASK,
        EMERGENCY_HINT,
        ht.CANNOT_FIND,
        turn_ack.THANKS,
        disclosure.DISCLOSURE,
    ]
    for table in tables:
        for lang, text in table.items():
            yield lang, text
    for lang in ("bn", "hi", "en"):
        yield lang, PHRASES[lang]["emergency_notice"]


@pytest.mark.parametrize("lang,text", list(_all_new_templates()))
def test_every_new_template_passes_the_persona_register_no_hedging_no_reassurance(lang, text):
    assert is_clean(text.replace("{value}", "X"), lang), (lang, text)


def test_the_operators_cannot_find_wording_and_the_thanks_pass_the_same_checks_the_tool_will_run():
    for lang, text in ht.CANNOT_FIND.items():
        assert is_clean(text, lang) and apology.count_apologies(text, lang) == 1


# ================================================================ tools/check_messages.py


def test_the_message_checker_passes_the_built_in_wording_and_catches_a_bad_edit():
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
    import check_messages as cm

    good = {
        "disclosure": {lg: {"text": t} for lg, t in disclosure.DISCLOSURE.items()},
        "cannot_find": {lg: {"text": t} for lg, t in ht.CANNOT_FIND.items()},
        "thanks_ack": {lg: {"text": t} for lg, t in turn_ack.THANKS.items()},
    }
    assert cm.check_all(good) == {}
    bad = {
        "thanks_ack": {
            "en": {"text": "Sorry, sorry. Don't worry, it is probably nothing, and you should take rest 2 times."}
        },
        "disclosure": {"en": {"text": "Welcome to the clinic."}},
    }
    found = cm.check_all(bad)
    text = " ".join(p for v in found.values() for p in v)
    assert ("thanks_ack", "en") in found and ("disclosure", "en") in found
    assert "apologies" in text and "digit" in text and "automated" in text and "staff" in text
