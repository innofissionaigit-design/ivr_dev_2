"""Unit tests for agent/tts.py's KCD-455 (block unspeakable spans) and
KCD-456 (per-request speed) changes. No live TTS server needed -- the
HTTP call is monkeypatched.

    python -m pytest tests/test_tts_client.py -v
"""

import os
import sys
from unittest.mock import AsyncMock

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.tts import FIGURE_SPEECH_SPEED, TTSClient, UnspeakableTextError


class _FakeResponse:
    def __init__(self, content: bytes = b"RIFF....WAVEfmt "):
        self.content = content

    def raise_for_status(self):
        pass


@pytest.fixture()
def client():
    c = TTSClient(base_url="http://unused.invalid/synthesize")
    c._client.post = AsyncMock(return_value=_FakeResponse())
    return c


@pytest.mark.asyncio
async def test_unspeakable_english_word_in_bengali_reply_blocks_instead_of_dropping(client):
    # A Latin word longer than one character inside a Bengali sentence is
    # exactly the measured incident this guards against (an English word
    # after a colon silently vanishing).
    with pytest.raises(UnspeakableTextError) as exc_info:
        await client.synthesize("রিপোর্ট Pending অবস্থায় আছে", "bn")
    assert "Pending" in exc_info.value.spans
    client._client.post.assert_not_called()  # never reached the vocoder


@pytest.mark.asyncio
async def test_ordinary_bengali_reply_is_not_blocked(client):
    wav = await client.synthesize("আপনার রেট পাঁচশো টাকা।", "bn")
    assert wav.startswith(b"RIFF")
    client._client.post.assert_called_once()


@pytest.mark.asyncio
async def test_default_speed_is_not_sent_in_the_payload(client):
    await client.synthesize("আপনার রেট পাঁচশো টাকা।", "bn")
    _, kwargs = client._client.post.call_args
    assert "speed" not in kwargs["json"]


@pytest.mark.asyncio
async def test_slow_speed_is_sent_in_the_payload(client):
    await client.synthesize("আপনার রেট পাঁচশো টাকা।", "bn", speed=FIGURE_SPEECH_SPEED)
    _, kwargs = client._client.post.call_args
    assert kwargs["json"]["speed"] == FIGURE_SPEECH_SPEED


@pytest.mark.asyncio
async def test_normal_and_slow_speed_cache_separately(client):
    await client.synthesize("আপনার রেট পাঁচশো টাকা।", "bn")
    await client.synthesize("আপনার রেট পাঁচশো টাকা।", "bn", speed=FIGURE_SPEECH_SPEED)
    assert client._client.post.call_count == 2  # no cache hit across speeds

    await client.synthesize("আপনার রেট পাঁচশো টাকা।", "bn", speed=FIGURE_SPEECH_SPEED)
    assert client._client.post.call_count == 2  # second slow request DOES hit cache
