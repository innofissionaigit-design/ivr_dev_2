"""Bengali/Hindi/English TTS via AI4Bharat's FastPitch + HiFi-GAN (Indic-TTS).

One tts_server.py process serves all three languages (bn/hi/en); this client
picks the voice per call with `lang` and verbalizes per language
(agent/speech_norm.py). Everything below written about Bengali holds for the
other two: their tokenizers drop out-of-script characters the same way.


Confirmed real and Bengali-capable: github.com/AI4Bharat/Indic-TTS ships
monolingual FastPitch+HiFi-GAN-V1 checkpoints for 13 Indian languages
including Bengali, hosted on the Bhashini platform, with a `synthesize`
inference module. Run its own server process (see README.md "Deploying
Indic-TTS") and point TTS_URL at it -- this client does not embed the
model itself, the same way tools_client.py does not embed the clinic DB.

This is the one piece of the pipeline with NO precedent in voice-to-rx-repo
-- that system never talks back, so nothing here has been battle-tested on
this account the way ASR/LLM have. Treat the fallback path below as load-
bearing, not decorative: a diagnostics line that goes silent when TTS is
down is worse than one that plays a stiff canned apology.

Two things happen here before a single byte is synthesized:

1. bn_normalize.verbalize() spells every number into Bengali words. This
   is not optional polish -- the tokenizer drops Latin digits outright, so
   without it every price is silence. See that module's docstring for the
   measurement.
2. An exact-text WAV cache. Synthesis is a pure function of the text, and
   reply_templates.py deliberately produces a SMALL set of sentences, so
   the hit rate on greetings, apologies and repeat questions is high. This
   is the cheapest latency win in the whole pipeline.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import threading
from typing import Any

import httpx

from agent.clause_split import split_into_clauses
from agent.speech_norm import unspeakable_spans, verbalize

logger = logging.getLogger("tts")


class UnspeakableTextError(Exception):
    """KCD-455: raised instead of silently synthesizing a reply with a
    hole in it. The measured incident this guards against: the English
    word after a colon vanished from a Bengali reply because the Bengali
    tokenizer drops Latin script outright -- logged at the time, sent to
    the caller anyway. Callers (main.py's _speak) must treat this as its
    own named failure path, the same as an ASR/LLM/tool failure, rather
    than let it reach the vocoder."""

    def __init__(self, spans: list[str]):
        self.spans = spans
        super().__init__(f"unspeakable spans: {spans}")


TTS_URL = os.environ.get("TTS_URL", "http://localhost:8002/synthesize")
TTS_TIMEOUT_S = float(os.environ.get("TTS_TIMEOUT_S", "20"))

# KCD-456: the rate a price, phone number, or reference ID is spoken at --
# slow enough that a caller writing it down does not have to ask twice.
# REASONED, not measured: there is no real-call transcription-accuracy
# corpus locally to calibrate against (the story's own "listening test
# confirms callers transcribe correctly on first hearing" is exactly that
# missing measurement). 0.8 is a conservative first cut, a fifth slower
# than normal speech, pending that measurement.
FIGURE_SPEECH_SPEED = 0.8

FALLBACK_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "fallback_audio")

# Pre-recorded once (see README.md "Recording the fallback set") and
# committed alongside the code, NOT generated at runtime -- if the live TTS
# service is what's broken, asking it to synthesize its own apology is
# exactly the failure this exists to route around.
FALLBACK_FILES = {
    "asr_empty": "sorry_repeat.wav",  # "দুঃখিত, শুনতে পাইনি, আবার বলুন"
    "llm_failure": "system_busy.wav",  # "একটু সমস্যা হচ্ছে, একটু ধরুন"
    "tool_failure": "check_failed.wav",  # "এখনই দেখতে পারছি না, স্টাফের কাছে দিচ্ছি"
    "tts_failure": "system_busy.wav",  # reused -- see note below
}

# Sentences the agent says on fixed paths, synthesized once at startup so
# the caller never waits on the vocoder for them. The greeting especially:
# it is the first thing on every single call. Per language: see
# agent/phrases.py, which owns the text so it exists in exactly one place.
PREWARM_LINES_BN = [
    "নমস্কার, কলকাতা কেয়ার ডায়াগনস্টিকসে স্বাগতম। কীভাবে সাহায্য করতে পারি?",
    "দুঃখিত, শুনতে পাইনি। আবার বলবেন?",
    "দুঃখিত, বুঝতে পারিনি। আবার একটু বলবেন?",
    "একটু সমস্যা হচ্ছে, একটু ধরুন।",
    "কোন টেস্টের রেট জানতে চান, একটু বলবেন?",
    "কোন ডাক্তারের কথা জিজ্ঞেস করছেন?",
    "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
    "লাইনে কোনো সাড়া পাচ্ছি না, কল শেষ করছি। ধন্যবাদ।",
]

# ~400 short clips at 22kHz mono. Bounded so a long-running process can't
# grow without limit on unique test names.
AUDIO_CACHE_MAX = 400
# The pinned clips (every catalogue answer, three languages) are bounded separately: ~1,000 short clips at ~100 KB.
MAX_PINNED_CLIPS = 3000


class TTSClient:
    def __init__(self, base_url: str = TTS_URL, timeout_s: float = TTS_TIMEOUT_S):
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self.base_url = base_url
        self._audio_cache: collections.OrderedDict[str, bytes] = collections.OrderedDict()
        # Clips that are never evicted: the rendered audio of every catalogue answer (see main.py _warm_answers).
        # Keyed by the exact text (and language and speed), so a changed price is a different key: this can hold
        # audio for an answer that is no longer given, never audio that disagrees with the text being spoken.
        self._pinned: dict[str, bytes] = {}
        self._cache_lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.prewarmed_clips = 0

    async def aclose(self):
        await self._client.aclose()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> bytes | None:
        with self._cache_lock:
            wav = self._pinned.get(key)
            if wav is not None:
                self.stats["hits"] += 1
                return wav
            wav = self._audio_cache.get(key)
            if wav is not None:
                self._audio_cache.move_to_end(key)
                self.stats["hits"] += 1
            return wav

    def _cache_put(self, key: str, wav: bytes):
        with self._cache_lock:
            self._audio_cache[key] = wav
            self._audio_cache.move_to_end(key)
            while len(self._audio_cache) > AUDIO_CACHE_MAX:
                self._audio_cache.popitem(last=False)
                self.stats["evictions"] += 1

    def retain_pinned(self, keep: set[str]) -> int:
        """Drop pinned clips that are not in `keep` (the answers of the latest warm-up round); returns how many stay.
        A clip for an answer that has since changed becomes evictable instead of living forever."""
        with self._cache_lock:
            for key in [k for k in self._pinned if k not in keep]:
                self._audio_cache[key] = self._pinned.pop(key)
            return len(self._pinned)

    async def synthesize(
        self, text: str, lang: str = "bn", speed: float = 1.0, prosody: dict | None = None, pin: set[str] | None = None
    ) -> bytes:
        """Returns WAV bytes, or raises. Callers should catch ToolCallError-
        shaped infra failures and UnspeakableTextError separately (see
        main.py's _speak()) and fall back to `fallback_audio()`.

        `speed` (KCD-456): 1.0 is the voice's normal rate; a caller passes
        a lower value for a reply carrying a price, phone number or
        reference the listener needs to write down (see
        agent/tts_router.py's docstring -- delivery parameters are decided
        by the caller of this client, never here).

        `prosody` (KCD-162): optional per-request overrides of the TTS
        server's pause/trim/peak/chunk parameters (agent/prosody.py's
        ProsodyParams fields), forwarded as-is so delivery can be tuned
        against a real handset without a redeploy. Part of the cache key --
        the same text with different pauses is different audio."""
        spoken = verbalize(text, lang)

        # KCD-455: anything the voice cannot pronounce is dropped by its
        # tokenizer entirely -- not mispronounced, ABSENT -- so this must
        # block the reply rather than let it reach the vocoder with a
        # silent hole in it. The measured incident this guards against is
        # in this exception's own docstring.
        leftovers = unspeakable_spans(spoken, lang)
        if leftovers:
            logger.error("[%s] unspeakable spans blocked this reply: %s", lang, leftovers)
            raise UnspeakableTextError(leftovers)

        # The language and speed are both part of the key: identical text
        # can be a valid sentence in two languages (a bare number, an ID)
        # and must not return the other voice's audio, and a slow-rate
        # clip must not be served for a normal-rate request or vice versa.
        prosody = {k: v for k, v in (prosody or {}).items() if v is not None}
        prosody_key = json.dumps(prosody, sort_keys=True) if prosody else ""
        key = self._key(f"{lang}\x00{speed}\x00{prosody_key}\x00{spoken}")
        cached = self._cache_get(key)
        if cached is not None:
            if pin is not None:
                with self._cache_lock:
                    if len(self._pinned) < MAX_PINNED_CLIPS:
                        self._pinned[key] = cached
                        self._audio_cache.pop(key, None)
                pin.add(key)
            return cached

        with self._cache_lock:
            self.stats["misses"] += 1
        payload: dict[str, Any] = {"text": spoken, "lang": lang}
        if speed != 1.0:
            payload["speed"] = speed
        payload.update(prosody)
        r = await self._client.post(self.base_url, json=payload)
        r.raise_for_status()
        wav = r.content
        if pin is not None:
            with self._cache_lock:
                if len(self._pinned) < MAX_PINNED_CLIPS:
                    self._pinned[key] = wav
                else:
                    self._audio_cache[key] = wav
            pin.add(key)
        else:
            self._cache_put(key, wav)
        return wav

    async def prewarm(self, lines_by_lang: dict[str, list[str]] | None = None):
        """Best-effort: a failure here must not stop the app from starting.
        Worst case the first caller pays normal synthesis latency.

        KCD-164 -- three things this gets right that matter:

        * Lines are expanded with the SAME split_into_clauses main.py's
          _speak() uses. _speak synthesizes a long reply one clause at a
          time (KCD-462), each under its own cache key, so warming the
          whole sentence would cache audio that is never asked for -- the
          greeting, the longest and most important line, would still cost
          a full synthesis on the first call.
        * Every language's first line (the greeting) is warmed before any
          language's second, so a slow or failing start-up still leaves
          the first thing every caller hears ready.
        * Hit/miss counters are zeroed afterwards. Warm-up is all misses by
          construction; leaving them in would make the exported hit rate
          look worse than live traffic ever is.
        """
        lines_by_lang = lines_by_lang or {"bn": PREWARM_LINES_BN}
        queue: list[tuple[str, str]] = []
        for rank in range(max((len(v) for v in lines_by_lang.values()), default=0)):
            for lang, lines in lines_by_lang.items():
                if rank < len(lines):
                    queue.extend((lang, clause) for clause in split_into_clauses(lines[rank]))
        failed: set[str] = set()
        for lang, clause in queue:
            if lang in failed:
                continue
            try:
                await self.synthesize(clause, lang)
            except Exception as e:  # noqa: BLE001 - prewarm is advisory only
                logger.warning("prewarm failed [%s] for %r: %s", lang, clause[:32], e)
                failed.add(lang)
        with self._cache_lock:
            self.stats["hits"] = self.stats["misses"] = self.stats["evictions"] = 0
            self.prewarmed_clips = len(self._audio_cache)
        logger.info("TTS prewarm complete (%d clips cached)", self.prewarmed_clips)

    def for_language(self, lang: str) -> LanguageVoice:
        """The TTSEngine (agent/tts_router.py) for one language."""
        return LanguageVoice(self, lang)

    def snapshot(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        return {
            **self.stats,
            "cached_clips": len(self._audio_cache) + len(self._pinned),
            "pinned_clips": len(self._pinned),
            "max_clips": AUDIO_CACHE_MAX,
            "prewarmed_clips": self.prewarmed_clips,
            "hit_rate": round(self.stats["hits"] / total, 3) if total else 0.0,
        }

    @staticmethod
    def fallback_audio(reason: str, lang: str = "bn") -> bytes:
        """reason in FALLBACK_FILES. Reads from disk every call (small
        files, infrequent path) rather than caching, so a corrected
        recording takes effect without a restart.

        Non-Bengali callers get `<name>_<lang>.wav` if it has been recorded,
        else the Bengali clip -- a wrong-language apology beats dead air,
        and the missing file is logged so the gap is visible."""
        filename = FALLBACK_FILES.get(reason, FALLBACK_FILES["llm_failure"])
        candidates = []
        if lang != "bn":
            stem, ext = os.path.splitext(filename)
            candidates.append(f"{stem}_{lang}{ext}")
        candidates.append(filename)
        path = os.path.join(FALLBACK_DIR, candidates[0])
        for name in candidates:
            path = os.path.join(FALLBACK_DIR, name)
            if os.path.exists(path):
                break
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(
                "Fallback audio %s missing -- call will go silent on this failure path. Record it: see README.md.",
                path,
            )
            return b""


class LanguageVoice:
    """A TTSClient bound to one language -- satisfies agent.tts_router.TTSEngine,
    so TTSRouter can be built as {lang: client.for_language(lang)}. All three
    share the one client (and its audio cache and HTTP connection pool); the
    voices themselves already live in the one tts_server.py process."""

    def __init__(self, client: TTSClient, lang: str):
        self._client = client
        self.lang = lang

    async def synthesize(self, text: str, **speech_params) -> bytes:
        return await self._client.synthesize(text, self.lang, **speech_params)
