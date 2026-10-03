"""KCD-164: exact-text audio cache with pre-warming (agent/tts.py).

"Audio is cached by exact verbalised text with bounded size and eviction,
and the fixed sentence set is pre-warmed at startup, greeting first. Hit
rate is exported. A cache hit costs no synthesis time."

The most important test here is the first one. KCD-462 made main.py's
_speak() synthesize a long reply one clause at a time, each clause under
its own cache key, while prewarm() was still caching WHOLE sentences --
so the warm greeting was never the thing actually asked for, and the
first caller paid a full synthesis for it. No HTTP: the TTS server call is
mocked.

    python -m pytest tests/test_tts_cache.py -v
"""

import os
import sys
from unittest.mock import AsyncMock

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import tts as tts_module
from agent.clause_split import split_into_clauses
from agent.phrases import PHRASES, prewarm_lines
from agent.tts import TTSClient


class _Resp:
    content = b"RIFF....WAVEfmt "

    def raise_for_status(self):
        pass


@pytest.fixture()
def client():
    c = TTSClient(base_url="http://unused.invalid/synthesize")
    c._client.post = AsyncMock(return_value=_Resp())
    return c


def _posted_texts(client):
    return [call.kwargs["json"]["text"] for call in client._client.post.call_args_list]


@pytest.mark.asyncio
async def test_prewarm_warms_exactly_the_clauses_speak_will_ask_for(client):
    # The English greeting is longer than clause_split.MIN_CHARS_TO_SPLIT,
    # so _speak sends it as separate clauses. Warming the whole sentence
    # would cache audio nobody requests.
    greeting = PHRASES["en"]["greeting"]
    clauses = split_into_clauses(greeting)
    assert len(clauses) > 1, "fixture assumption: the greeting really is split by _speak"

    await client.prewarm({"en": [greeting]})
    posts_after_warmup = client._client.post.call_count

    for clause in clauses:  # what main.py's _speak does
        await client.synthesize(clause, "en")
    assert client._client.post.call_count == posts_after_warmup, (
        "every clause of the warmed greeting must be a cache hit"
    )


@pytest.mark.asyncio
async def test_every_fixed_phrase_in_every_language_is_a_hit_after_prewarm(client):
    await client.prewarm(prewarm_lines())
    posts_after_warmup = client._client.post.call_count
    for lang, table in PHRASES.items():
        for line in table.values():
            for clause in split_into_clauses(line):
                await client.synthesize(clause, lang)
    assert client._client.post.call_count == posts_after_warmup


@pytest.mark.asyncio
async def test_a_cache_hit_makes_no_synthesis_call(client):
    await client.synthesize("Hello there.", "en")
    assert client._client.post.call_count == 1
    await client.synthesize("Hello there.", "en")
    await client.synthesize("Hello there.", "en")
    assert client._client.post.call_count == 1


@pytest.mark.asyncio
async def test_the_greeting_of_every_language_is_warmed_before_any_second_line(client):
    await client.prewarm(
        {
            "en": ["English greeting.", "English second."],
            "hi": ["नमस्ते।", "दूसरा।"],
        }
    )
    langs_in_order = [call.kwargs["json"]["lang"] for call in client._client.post.call_args_list]
    assert langs_in_order == ["en", "hi", "en", "hi"]
    assert _posted_texts(client)[0] == "English greeting."


@pytest.mark.asyncio
async def test_a_language_that_fails_to_warm_does_not_block_the_others(client):
    async def post(url, json):
        if json["lang"] == "hi":
            raise RuntimeError("hindi voice down")
        return _Resp()

    client._client.post = AsyncMock(side_effect=post)
    await client.prewarm({"hi": ["नमस्ते।"], "en": ["Hello there."]})
    assert client.snapshot()["cached_clips"] == 1  # English still warmed
    assert client._client.post.call_count == 2  # hi tried once, then skipped


@pytest.mark.asyncio
async def test_warmup_misses_do_not_pollute_the_exported_hit_rate(client):
    await client.prewarm({"en": ["First line.", "Second line."]})
    snap = client.snapshot()
    assert snap["hits"] == 0 and snap["misses"] == 0 and snap["hit_rate"] == 0.0
    assert snap["prewarmed_clips"] == 2

    await client.synthesize("First line.", "en")  # first real request is a hit
    assert client.snapshot()["hit_rate"] == 1.0


@pytest.mark.asyncio
async def test_hit_rate_is_exported_per_the_story(client):
    await client.synthesize("Alpha.", "en")  # miss
    await client.synthesize("Alpha.", "en")  # hit
    await client.synthesize("Beta.", "en")  # miss
    await client.synthesize("Alpha.", "en")  # hit
    snap = client.snapshot()
    assert (snap["hits"], snap["misses"]) == (2, 2)
    assert snap["hit_rate"] == 0.5
    assert snap["max_clips"] == tts_module.AUDIO_CACHE_MAX


@pytest.mark.asyncio
async def test_the_cache_is_bounded_and_evicts_least_recently_used(client, monkeypatch):
    monkeypatch.setattr(tts_module, "AUDIO_CACHE_MAX", 3)
    for word in ("One.", "Two.", "Three."):
        await client.synthesize(word, "en")
    await client.synthesize("One.", "en")  # touch: One is now most recent
    await client.synthesize("Four.", "en")  # evicts Two, the least recently used
    await client.synthesize("Five.", "en")  # evicts Three

    snap = client.snapshot()
    assert snap["cached_clips"] == 3
    assert snap["evictions"] == 2

    calls_before = client._client.post.call_count
    await client.synthesize("One.", "en")  # survived
    assert client._client.post.call_count == calls_before
    await client.synthesize("Two.", "en")  # was evicted -> re-synthesized
    assert client._client.post.call_count == calls_before + 1


@pytest.mark.asyncio
async def test_the_same_text_in_another_language_or_speed_is_a_different_clip(client):
    await client.synthesize("12345", "en")
    await client.synthesize("12345", "hi")
    await client.synthesize("12345", "en", speed=0.8)
    assert client._client.post.call_count == 3


# ------------------------------------------------ KCD-162 per-request tuning


@pytest.mark.asyncio
async def test_prosody_overrides_are_forwarded_to_the_tts_server(client):
    await client.synthesize(
        "Hello there.", "en", prosody={"pause_sentence_s": 0.5, "target_peak": 0.7, "trim_threshold": None}
    )
    payload = client._client.post.call_args.kwargs["json"]
    assert payload["pause_sentence_s"] == 0.5 and payload["target_peak"] == 0.7
    assert "trim_threshold" not in payload, "an unset override must not be sent"


@pytest.mark.asyncio
async def test_default_prosody_adds_nothing_to_the_payload(client):
    await client.synthesize("Hello there.", "en")
    assert set(client._client.post.call_args.kwargs["json"]) == {"text", "lang"}


@pytest.mark.asyncio
async def test_different_prosody_is_a_different_clip_but_the_same_prosody_is_a_hit(client):
    await client.synthesize("Hello there.", "en")
    await client.synthesize("Hello there.", "en", prosody={"pause_sentence_s": 0.5})
    assert client._client.post.call_count == 2
    await client.synthesize("Hello there.", "en", prosody={"pause_sentence_s": 0.5})
    assert client._client.post.call_count == 2
    await client.synthesize("Hello there.", "en", prosody={"pause_sentence_s": None})
    assert client._client.post.call_count == 2, "an all-None override is the default clip, already cached"
