"""Latency and the answer cache, from the first live conversation (2026-09-25): turn latency was 3.5 s (p50) and the
caller asked for under 500 ms and for the same question to be answered from a cache with no model and no synthesis.

    python -m pytest tests/test_latency_and_cache.py -v

Measured on the pod first (stage by stage): clinic lookup 8 ms; TTS 20-85 ms; the intent model 1.3 s; the semantic-cache
embedding 100-185 ms but the SERIAL wait for it was capped at 1.5 s and was being spent in full before the model call
began; recognition plus language ID 0.85 s. What is asserted here is what this code controls: the cache lookup and the
model run together, the holding phrase comes earlier and says what it means, and the rendered audio of every catalogue
answer is pinned so a repeat question needs no synthesis. Wall-clock timing tests use small REAL waits.
"""

import os
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_orchestrator_booking_flow import env, m  # noqa: F401  (the harness: real dispatch, fake tools)

from agent.phrases import phrase
from agent.tts import TTSClient

GOOD = {"intent": "test_rate", "slots": {"test_name": "CBC"}, "secondary_intent": None, "direct_reply_bn": None}


# ============================================================================ cache lookup and model run together


class Cache:
    def __init__(self, get_s=0.0, hit=None):
        self.get_s, self.hit, self.puts = get_s, hit, []

    def get(self, key):
        time.sleep(self.get_s)
        return (self.hit, "l2") if self.hit else (None, "miss")

    def put(self, key, data):
        self.puts.append(key)


@pytest.fixture
def resolve(m, env, monkeypatch):
    """The REAL `_resolve_intent`, with the fast path off and a fake cache and extractor."""
    real = env  # keep the fixture alive
    monkeypatch.undo()
    monkeypatch.setattr(m, "_fast_path", None)

    async def no_reload():
        return None

    monkeypatch.setattr(m, "_maybe_reload_fast_path", no_reload)
    monkeypatch.setattr(m, "INTENT_BUDGET_S", 3.0)
    monkeypatch.setattr(m, "FILLER_THRESHOLD_S", 5.0)  # no filler in these tests

    async def go(text="x"):
        t0 = time.monotonic()
        try:
            out = await m._resolve_intent(real.session, text, "en")
        except m.ExtractionError as e:
            out = e
        return out, time.monotonic() - t0

    return go


@pytest.mark.asyncio
async def test_a_cache_hit_ends_the_turn_at_once_and_the_model_is_not_waited_for(m, resolve, monkeypatch):
    monkeypatch.setattr(m, "_intent_cache", Cache(get_s=0.05, hit=dict(GOOD)))
    started = []

    def slow_model(text, max_retries=2, lang="bn", deadline_s=12.0):
        started.append(1)
        time.sleep(2.0)
        return dict(GOOD), {"total_time_s": 2.0, "attempts": 1}

    monkeypatch.setattr(m, "extract_intent", slow_model)
    out, took = await resolve()
    assert out["intent"] == "test_rate" and took < 0.4 and not started  # answered inside the head start


@pytest.mark.asyncio
async def test_a_slow_lookup_that_hits_while_the_model_runs_still_wins_and_drops_the_model(m, resolve, monkeypatch):
    monkeypatch.setattr(m, "_intent_cache", Cache(get_s=0.4, hit=dict(GOOD)))  # slower than the head start
    monkeypatch.setattr(m, "CACHE_HEAD_START_S", 0.1)

    def slow_model(text, max_retries=2, lang="bn", deadline_s=12.0):
        time.sleep(1.5)
        return {"intent": "clinic_faq", "slots": {}, "secondary_intent": None, "direct_reply_bn": None}, {
            "total_time_s": 1.5,
            "attempts": 1,
        }

    monkeypatch.setattr(m, "extract_intent", slow_model)
    out, took = await resolve()
    assert out["intent"] == "test_rate" and 0.3 < took < 1.0, took


@pytest.mark.asyncio
async def test_a_cache_miss_costs_the_model_turn_only_the_head_start_not_the_lookup(m, resolve, monkeypatch):
    """The serial version waited for the whole lookup (up to 1.5 s) before starting the model."""
    monkeypatch.setattr(m, "_intent_cache", Cache(get_s=1.0))  # a lookup that misses, slowly
    monkeypatch.setattr(m, "CACHE_HEAD_START_S", 0.1)
    monkeypatch.setattr(
        m, "extract_intent", lambda *a, **k: (time.sleep(0.2), (dict(GOOD), {"total_time_s": 0.2, "attempts": 1}))[1]
    )
    out, took = await resolve()
    assert out["intent"] == "test_rate" and took < 0.6, took  # 0.1 head start + 0.2 model, not 1.2


@pytest.mark.asyncio
async def test_a_failing_cache_is_a_miss_never_an_error(m, resolve, monkeypatch):
    class Broken:
        def get(self, key):
            raise RuntimeError("ollama is down")

        def put(self, key, data):
            pass

    monkeypatch.setattr(m, "_intent_cache", Broken())
    monkeypatch.setattr(m, "extract_intent", lambda *a, **k: (dict(GOOD), {"total_time_s": 0.01, "attempts": 1}))
    out, took = await resolve()
    assert out["intent"] == "test_rate"


@pytest.mark.asyncio
async def test_the_model_answer_is_still_cached_for_next_time(m, resolve, monkeypatch):
    cache = Cache()
    monkeypatch.setattr(m, "_intent_cache", cache)
    monkeypatch.setattr(m, "extract_intent", lambda *a, **k: (dict(GOOD), {"total_time_s": 0.01, "attempts": 1}))
    await resolve("what is the price of a CBC")
    assert cache.puts


# ========================================================================================== the holding phrase


def test_the_holding_phrase_is_the_owners_wording_in_every_language():
    """DELIBERATE SPEC CHANGE (owner, 2026-09-25): the holding phrase is "achha bolchi" ("okay, telling you") in the
    three languages. It replaces "একটু দেখছি।" / "एक पल, देख रही हूँ।" / "Let me check, please." """
    assert phrase("please_wait", "bn") == "আচ্ছা, বলছি।"
    assert phrase("please_wait", "hi") == "अच्छा, बताती हूँ।"
    assert phrase("please_wait", "en") == "Alright, let me tell you."


def test_the_holding_phrase_comes_only_after_700_ms(m):
    """DELIBERATE SPEC CHANGE (owner, 2026-09-25): 0.7 s (was 0.9 s), counted from when the caller stopped speaking."""
    assert m.FILLER_THRESHOLD_S == 0.7


# =============================================================================================== pinned audio clips


class FakeHTTP:
    def __init__(self):
        self.posts = []

    async def post(self, url, json=None, **kw):
        self.posts.append(json)

        class R:
            content = b"RIFF" + str(json["text"]).encode("utf-8")

            def raise_for_status(self):
                pass

        return R()

    async def aclose(self):
        pass


def _client():
    c = TTSClient()
    c._client = FakeHTTP()
    return c


@pytest.mark.asyncio
async def test_a_pinned_clip_is_served_from_the_cache_and_never_evicted():
    c = _client()
    keep: set[str] = set()
    await c.synthesize("The CBC costs 350 rupees.", "en", pin=keep)
    n = len(c._client.posts)
    from agent import tts as tts_mod

    for i in range(tts_mod.AUDIO_CACHE_MAX + 50):  # flood the ordinary cache well past its size
        await c.synthesize(f"filler sentence number {i}.", "en")
    before = len(c._client.posts)
    await c.synthesize("The CBC costs 350 rupees.", "en")
    assert len(c._client.posts) == before  # a hit: no synthesis
    assert n == 1 and c.snapshot()["pinned_clips"] == 1


@pytest.mark.asyncio
async def test_the_key_is_the_exact_text_so_a_changed_price_is_a_different_clip_never_a_stale_one():
    c = _client()
    keep: set[str] = set()
    await c.synthesize("The CBC costs 350 rupees.", "en", pin=keep)
    await c.synthesize("The CBC costs 400 rupees.", "en", pin=keep)  # the price changed
    assert len(c._client.posts) == 2  # so it was rendered, not served stale
    assert "four hundred" in c._client.posts[1]["text"]  # numbers are spoken as words


@pytest.mark.asyncio
async def test_a_clip_for_an_answer_that_no_longer_exists_becomes_evictable():
    c = _client()
    old, new = set(), set()
    await c.synthesize("The CBC costs 350 rupees.", "en", pin=old)
    await c.synthesize("The CBC costs 400 rupees.", "en", pin=new)
    assert c.retain_pinned(new) == 1  # only the current price stays pinned
    assert c.snapshot()["pinned_clips"] == 1


@pytest.mark.asyncio
async def test_the_pinned_set_is_bounded():
    c = _client()
    import agent.tts as tts_mod

    tts_mod.MAX_PINNED_CLIPS, saved = 5, tts_mod.MAX_PINNED_CLIPS
    try:
        keep: set[str] = set()
        for i in range(12):
            await c.synthesize(f"clip {i}.", "en", pin=keep)
        assert c.snapshot()["pinned_clips"] == 5
    finally:
        tts_mod.MAX_PINNED_CLIPS = saved


# ============================================================================= every catalogue answer, rendered ahead

CATALOGUE = {
    "tests": [{"name": "Serum Creatinine"}, {"name": "CBC"}],
    "doctors": [],
    "faq_topics": [{"topic": "hours"}, {"topic": "parking"}],
}


class WarmTools:
    def __init__(self):
        self.rates = {"Serum Creatinine": 250, "CBC": 400}

    async def get_test_rate(self, name):
        return {
            "found": True,
            "test_name": name,
            "test_name_bn": "ক্রিয়াটিনিন" if "Creat" in name else "সিবিসি",
            "test_name_hi": "क्रिएटिनिन" if "Creat" in name else "सीबीसी",
            "rate_inr": self.rates[name],
            "sample_type": "Blood",
            "report_time_hours": 12,
        }

    async def get_test_prep(self, name, lang):
        return {
            "found": True,
            "test_name": name,
            "test_name_bn": "ক্রিয়াটিনিন",
            "test_name_hi": "क्रिएटिनिन",
            "prep_instructions": "No special preparation is needed for this test; you can come in normally."
            if lang == "en"
            else "বিশেষ কোনো প্রস্তুতির প্রয়োজন নেই।"
            if lang == "bn"
            else "किसी ख़ास तैयारी की ज़रूरत नहीं है।",
        }

    async def get_faq(self, topic, lang):
        return {
            "found": True,
            "answer": {
                "hours": "We are open every day from 8 AM to 8 PM.",
                "parking": "The clinic has its own parking, and there is no charge.",
            }[topic]
            if lang == "en"
            else "উত্তর। আরেকটা বাক্য এখানে।"
            if lang == "bn"
            else "उत्तर। यहाँ एक और वाक्य।",
        }


@pytest.fixture
def warm(m, monkeypatch):
    tts = _client()
    tools = WarmTools()
    monkeypatch.setattr(m, "_tts", tts)
    monkeypatch.setattr(m, "_tools", tools)
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))

    async def cat():
        return CATALOGUE

    monkeypatch.setattr(m, "_fetch_catalogue", cat)
    return tts, tools


@pytest.mark.asyncio
async def test_the_warm_up_pins_the_audio_of_every_price_preparation_and_faq_answer_in_every_language(m, warm):
    tts, tools = warm
    n = await m._warm_answers_once()
    assert n > 15 and tts.snapshot()["pinned_clips"] == n
    texts = " ".join(str(p["text"]) for p in tts._client.posts)
    assert "two hundred fifty" in texts and "four hundred rupees" in texts  # the prices, as they are spoken
    assert "open every day" in texts and "parking" in texts  # the FAQ answers
    assert "sample is needed" in texts  # the sentence form, not "Sample: Blood"


@pytest.mark.asyncio
async def test_a_second_round_renders_nothing_it_already_holds(m, warm):
    tts, tools = warm
    await m._warm_answers_once()
    rendered = len(tts._client.posts)
    await m._warm_answers_once()
    assert len(tts._client.posts) == rendered


@pytest.mark.asyncio
async def test_a_price_change_renders_only_the_changed_answer_and_retires_the_old_clip(m, warm):
    tts, tools = warm
    first = await m._warm_answers_once()
    rendered = len(tts._client.posts)
    tools.rates["CBC"] = 450  # the clinic changed a price
    second = await m._warm_answers_once()
    new = tts._client.posts[rendered:]
    assert 0 < len(new) < 12 and any("four hundred fifty" in str(p["text"]) for p in new)  # the price clause
    assert not any("Serum" in str(p["text"]) for p in new)
    assert second == first  # the old price is no longer pinned


@pytest.mark.asyncio
async def test_a_live_question_after_the_warm_up_is_answered_with_no_synthesis_and_no_model(m, env, warm, monkeypatch):
    """The whole point: the same question, asked live, needs the model for nothing and the voice for nothing."""
    from agent.tts_router import TTSRouter

    tts, tools = warm
    monkeypatch.setattr(m, "_tts_router", TTSRouter({lang: tts.for_language(lang) for lang in ("bn", "hi", "en")}))
    await m._warm_answers_once()
    rendered = len(tts._client.posts)
    from agent.reply_templates import test_rate_reply

    env.state["lang"] = "en"

    async def real_answer(intent, slots, lang):
        return test_rate_reply(slots, await tools.get_test_rate(slots["test_name"]), lang)

    monkeypatch.setattr(m, "_answer_enquiry_intent", real_answer)
    said = await env.say("what is the price of the creatinine test", "test_rate", {"test_name": "Serum Creatinine"})
    assert said and "two hundred fifty" in " ".join(said) or "250" in " ".join(said)
    assert len(tts._client.posts) == rendered, "the live answer needed a synthesis that the warm-up did not render"
