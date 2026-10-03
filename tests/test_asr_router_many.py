"""ASRRouter.transcribe_many: verification must survive one engine being down."""

import asyncio
import dataclasses

import pytest

from agent.asr_router import ASRRouter


@dataclasses.dataclass
class R:
    text: str


class Ok:
    def __init__(self, text):
        self.text = text

    async def transcribe_utterance(self, wav):
        return R(self.text)


class Down:
    async def transcribe_utterance(self, wav):
        raise ConnectionError("english-asr down")


def router(**engines):
    return ASRRouter(
        {"bn": engines.get("bn", Ok("b")), "hi": engines.get("hi", Ok("h")), "en": engines.get("en", Ok("e"))}
    )


def test_returns_every_engine_result_in_order():
    out = asyncio.run(router().transcribe_many(["hi", "en", "bn"], "x.wav"))
    assert [lang for lang, _ in out] == ["hi", "en", "bn"]
    assert [r.text for _, r in out] == ["h", "e", "b"]


def test_a_down_engine_is_dropped_not_fatal():
    out = asyncio.run(router(en=Down()).transcribe_many(["hi", "en", "bn"], "x.wav"))
    assert [lang for lang, _ in out] == ["hi", "bn"]


def test_raises_only_when_every_engine_fails():
    r = router(bn=Down(), hi=Down(), en=Down())
    with pytest.raises(ConnectionError):
        asyncio.run(r.transcribe_many(["bn", "hi", "en"], "x.wav"))
