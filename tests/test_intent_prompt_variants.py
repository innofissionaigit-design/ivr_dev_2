"""How the intent model is asked (agent/llm.py: build_prompt) and what its timing log says. No model is needed: the model
call is replaced by a fake.

    python -m pytest tests/test_intent_prompt_variants.py -v
"""

import datetime
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import intent_ab  # noqa: E402

from agent import llm  # noqa: E402
from agent.intent_schema import FAQ_TOPICS, SLOT_KEYS, VALID_INTENTS  # noqa: E402

NOW = datetime.datetime(2026, 9, 26, 10, 30)


def original_prompt(text: str, lang: str, now: datetime.datetime) -> str:
    """The prompt exactly as extract_intent built it before build_prompt existed."""
    names = {"bn": "Bengali", "hi": "Hindi", "en": "English"}
    name = names.get(lang, "Bengali")
    system = llm.SYSTEM_PROMPT_TEMPLATE.format(
        today_iso=now.strftime("%Y-%m-%d"),
        today_weekday=now.strftime("%A"),
        faq_topics=", ".join(FAQ_TOPICS),
        language_name=name,
    )
    return f"{system}\n\nCALLER UTTERANCE ({name}, ASR output):\n{text}\n\nJSON:"


@pytest.mark.parametrize("lang", ["bn", "hi", "en", "xx"])
def test_the_classic_prompt_is_the_original_byte_for_byte(lang):
    assert llm.build_prompt("দাম কত", lang, NOW, "classic") == original_prompt("দাম কত", lang, NOW)


def test_the_default_variant_is_classic():
    assert llm.INTENT_PROMPT_VARIANT == "classic"


def test_the_fast_prompt_shares_one_long_prefix_across_languages_dates_and_sentences():
    a = llm.build_prompt("one sentence", "bn", NOW, "fast")
    b = llm.build_prompt("quite another", "hi", NOW + datetime.timedelta(days=9), "fast")
    prefix = os.path.commonprefix([a, b])
    assert len(prefix) > 0.9 * len(a)  # only the tail differs: the server can reuse its cached copy
    # they part company exactly where the call begins: nothing before the language name is per-call
    assert prefix.rstrip().endswith("CALLER LANGUAGE:") and "2026-09" not in prefix


def test_the_fast_prompt_keeps_every_rule_and_every_intent_and_asks_for_the_compact_shape():
    p = llm.build_prompt("x", "en", NOW, "fast")
    for intent in VALID_INTENTS:
        assert f'"{intent}"' in p, intent
    for topic in FAQ_TOPICS:
        assert topic in p
    for slot in SLOT_KEYS:
        assert slot in p, slot
    assert "Never invent a patient name" in p and "LITERALLY present" in p  # the truth-boundary rules are still there
    assert "leave out every null slot" in p and "string or null" not in p  # ...and the 16 nulls are not asked for
    assert "{language_name}" not in p and "{today_iso}" not in p and "{faq_topics}" not in p


def test_the_fast_prompt_is_shorter_than_the_classic_one():
    assert len(llm.build_prompt("x", "bn", NOW, "fast")) < len(llm.build_prompt("x", "bn", NOW, "classic"))


def test_the_language_and_date_are_stated_at_the_end_of_the_fast_prompt():
    p = llm.build_prompt("hello", "hi", NOW, "fast")
    tail = p[-200:]
    assert "CALLER LANGUAGE: Hindi" in tail and "2026-09-26 (Saturday)" in tail and p.endswith("hello\n\nJSON:")


# ------------------------------------------------------------------------------------------------ the timing log


def test_ollama_timings_are_summarised_in_milliseconds():
    body = {
        "prompt_eval_count": 1800,
        "prompt_eval_duration": 250_000_000,
        "eval_count": 60,
        "eval_duration": 1_000_000_000,
        "load_duration": 5_000_000,
        "total_duration": 1_300_000_000,
    }
    t = llm.summarise_timing(body)
    assert t == {
        "prompt_tokens": 1800,
        "prefill_ms": 250,
        "output_tokens": 60,
        "decode_ms": 1000,
        "load_ms": 5,
        "total_ms": 1300,
        "tokens_per_s": 60.0,
    }


def test_a_response_with_no_timings_summarises_to_zeros_not_an_error():
    t = llm.summarise_timing({"response": "{}"})
    assert t["prompt_tokens"] == 0 and t["tokens_per_s"] is None


def _fake_model(monkeypatch, answer):
    def fake(prompt, timeout_s=90):
        llm._last_call.stats = llm.summarise_timing(
            {
                "prompt_eval_count": 1700,
                "prompt_eval_duration": 200_000_000,
                "eval_count": 25,
                "eval_duration": 500_000_000,
                "total_duration": 800_000_000,
            }
        )
        fake.prompts.append(prompt)
        return json.dumps(answer)

    fake.prompts = []
    monkeypatch.setattr(llm, "_call_ollama", fake)
    return fake


def test_extract_intent_reports_the_split_and_logs_it(monkeypatch, caplog):
    fake = _fake_model(monkeypatch, {"intent": "test_rate", "slots": {k: None for k in SLOT_KEYS}})
    with caplog.at_level("INFO", logger="llm"):
        data, diag = llm.extract_intent("দাম কত", lang="bn")
    assert (
        data["intent"] == "test_rate"
        and diag["model_timing"]["prefill_ms"] == 200
        and diag["model_timing"]["decode_ms"] == 500
    )
    assert any(
        "1700 prompt tokens read in 200 ms" in r.message and "25 tokens written in 500 ms" in r.message
        for r in caplog.records
    )
    assert len(fake.prompts) == 1


def test_a_compact_answer_is_accepted_in_the_fast_variant_only(monkeypatch):
    monkeypatch.setattr(llm, "INTENT_PROMPT_VARIANT", "fast")
    _fake_model(monkeypatch, {"intent": "test_rate", "slots": {"test_name": "CBC"}})
    data, _ = llm.extract_intent("cbc dam", lang="bn")
    assert (
        data["slots"]["test_name"] == "CBC"
        and data["slots"]["doctor_name"] is None
        and data["secondary_intent"] is None
    )
    monkeypatch.setattr(llm, "INTENT_PROMPT_VARIANT", "fast")
    _fake_model(monkeypatch, {"intent": "smalltalk", "direct_reply_bn": "নমস্কার"})  # no "slots" at all
    data, _ = llm.extract_intent("নমস্কার", lang="bn")
    assert data["intent"] == "smalltalk" and data["slots"]["test_name"] is None and data["direct_reply_bn"] == "নমস্কার"


def test_the_classic_variant_still_needs_slots(monkeypatch):
    monkeypatch.setattr(llm, "INTENT_PROMPT_VARIANT", "classic")
    _fake_model(monkeypatch, {"intent": "smalltalk"})
    with pytest.raises(llm.ExtractionError):
        llm.extract_intent("নমস্কার", lang="bn", deadline_s=1.0)


def test_num_ctx_is_only_sent_when_set(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_NUM_CTX", 0)
    assert llm._model_options() == {"temperature": 0.0}
    monkeypatch.setattr(llm, "OLLAMA_NUM_CTX", 4096)
    assert llm._model_options() == {"temperature": 0.0, "num_ctx": 4096}


# ---------------------------------------------------------------------------------------------- the A/B tool


def test_the_ab_tool_scores_intent_and_slots():
    assert intent_ab.score_case(
        "test_rate", {"test_name": "CBC"}, {"intent": "test_rate", "slots": {"test_name": "the cbc test"}}
    ) == (True, True)
    assert intent_ab.score_case(
        "test_rate", {"test_name": "CBC"}, {"intent": "test_rate", "slots": {"test_name": None}}
    ) == (True, False)
    assert intent_ab.score_case(
        "clinic_faq", {"faq_topic": "hours"}, {"intent": "clinic_faq", "slots": {"faq_topic": "hours"}}
    ) == (True, True)
    assert intent_ab.score_case(
        "clinic_faq", {"faq_topic": "hours"}, {"intent": "clinic_faq", "slots": {"faq_topic": "opening_hours"}}
    ) == (True, False)
    assert intent_ab.score_case("smalltalk", {}, {"intent": "unclear", "slots": {}}) == (False, True)


def test_the_ab_tool_runs_both_variants_against_a_fake_model(monkeypatch):
    cases = [("en", "how much is the CBC test", "test_rate", {"test_name": "CBC"})]
    _fake_model(monkeypatch, {"intent": "test_rate", "slots": {"test_name": "CBC"}})
    classic = intent_ab.run_variant("classic", 0, 2, cases)
    fast = intent_ab.run_variant("fast", 4096, 2, cases)
    assert classic["calls"] == 2 and classic["intent_acc"] == 1.0 and classic["slot_acc"] == 1.0
    assert fast["num_ctx"] == 4096 and fast["prefill_ms"] == 200 and fast["out_tokens"] == 25
    intent_ab.llm.INTENT_PROMPT_VARIANT, intent_ab.llm.OLLAMA_NUM_CTX = "classic", 0
