"""AI4Bharat TTS -- FastAPI wrapper around coqui-tts's Synthesizer.

Matches the contract agent/tts.py's TTSClient expects: POST /synthesize
with {"text": ..., "lang": "bn"} -> raw WAV bytes.

Pilot single-L4 architecture (docs/adr/0001-pilot-single-l4-architecture.md):
one process serves all THREE languages -- bn, hi, en -- because AI4Bharat's
own Indic-TTS v1-checkpoints-release ships an English FastPitch+HiFi-GAN
checkpoint in the exact same coqui-tts format as Bengali and Hindi
(https://github.com/AI4Bharat/Indic-TTS/releases/tag/v1-checkpoints-release,
en.zip). That is simpler than the ADR's original assumption of a second,
NVIDIA-NeMo-based English TTS path, and this file supersedes that
assumption -- no second TTS MODEL FAMILY is needed.

It still needs its own PROCESS and VENV (/workspace/tts_venv), separate
from the orchestrator/ASR venv -- confirmed the hard way, not assumed:
coqui-tts needs transformers>=4.57 + huggingface_hub>=0.34, while the
AI4Bharat NeMo fork used for Bengali/Hindi ASR imports huggingface_hub's
ModelFilter, removed in huggingface_hub 0.24. No single version of either
package satisfies both consumers at once, so this cannot share an
environment with agent/asr.py's bn/hi models no matter how the two
libraries' own version constraints are pinned. See deploy/start_all.sh's
_run_tts.sh for the resulting process layout.

All three Synthesizers are loaded ONCE at module import (not per-request)
-- model load takes several seconds and holds real GPU memory, so this
must be a long-lived process,
not spawned per call.

WHY THIS DOES MORE THAN CALL synthesizer.tts()
----------------------------------------------
A single FastPitch pass over a whole multi-clause reply is what makes this
voice sound like a machine, and it is fixable without changing models:

* No breaths. FastPitch renders one flat prosodic contour across an entire
  utterance. Real speakers stop between clauses. Splitting on Bengali
  sentence and clause boundaries and inserting real silence is the single
  largest naturalness gain available here.
* Ragged padding. Each pass emits its own leading/trailing near-silence of
  arbitrary length, so naive concatenation produces gaps that are too long
  in some places and absent in others. Each chunk is trimmed, then padded
  by an amount chosen for the punctuation that ended it.
* Rushed delivery. length_scale 1.0 is noticeably fast for a service line
  a caller is trying to write a price down from. Slightly above 1 reads as
  measured rather than sluggish.
* Inconsistent level. Peak varies per utterance, which on a phone sounds
  like the speaker keeps moving. Normalizing to a fixed peak fixes it.

Every one of these is tunable per-request (see SynthesizeRequest) so the
settings can be A/B'd against a real handset without a redeploy.
"""
import dataclasses
import io
import os
import threading

# Pure prosody logic (splitting, pauses, trim, normalise) lives in
# agent/prosody.py so it can be unit tested without a GPU; the repo root is
# on PYTHONPATH (deploy/env.sh). Imported BEFORE the chdir below.
from agent.prosody import (
    ProsodyParams, assemble, resolve_length_scale, resolve_params, split_for_prosody,
)

# The AI4Bharat checkpoint's speaker manager resolves a RELATIVE path
# ("models/v1/bn/fastpitch/speakers.pth") baked in at save time, against
# whatever the process's CWD happens to be -- not the checkpoint's own
# location. Pin CWD explicitly so this doesn't depend on how/where this
# script gets launched from.
CHECKPOINTS_ROOT = os.environ.get("TTS_CHECKPOINTS_ROOT", "/workspace/tts_checkpoints")
os.chdir(CHECKPOINTS_ROOT)

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from TTS.utils.synthesizer import Synthesizer

app = FastAPI()

SUPPORTED_LANGUAGES = ("bn", "hi", "en")


def _load_synthesizer(lang: str) -> Synthesizer:
    ckpt = f"{CHECKPOINTS_ROOT}/{lang}"
    return Synthesizer(
        tts_checkpoint=f"{ckpt}/fastpitch/best_model.pth",
        tts_config_path=f"{ckpt}/fastpitch/config.json",
        tts_speakers_file=f"{ckpt}/fastpitch/speakers.pth",
        vocoder_checkpoint=f"{ckpt}/hifigan/best_model.pth",
        vocoder_config=f"{ckpt}/hifigan/config.json",
        use_cuda=True,
    )


# All three loaded and resident on the GPU together -- small footprint,
# and never unloaded/reloaded on a language switch (docs/adr/0001 section
# on why all six speech models stay warm: a model-load stall mid-call is
# worse than the fixed VRAM cost of keeping them all in memory).
SYNTHESIZERS: dict[str, Synthesizer] = {lang: _load_synthesizer(lang) for lang in SUPPORTED_LANGUAGES}

SAMPLE_RATES: dict[str, int] = {
    lang: (synth.output_sample_rate or 22050) for lang, synth in SYNTHESIZERS.items()
}
DEFAULT_SPEAKER = os.environ.get("TTS_SPEAKER", "female")

# >1 slows delivery. 1.08 measured as the point where the line stops
# sounding hurried without starting to drag -- tune via /synthesize's
# `length_scale` (raw) or `speed` (rate multiplier) override before
# changing this default.
DEFAULT_LENGTH_SCALE = float(os.environ.get("TTS_LENGTH_SCALE", "1.08"))

# Server-side defaults for every prosody parameter (agent/prosody.py
# ProsodyParams). Any of them can be overridden on a single /synthesize
# request, so delivery can be A/B tested against a real handset without a
# redeploy (KCD-162).
DEFAULT_PROSODY = ProsodyParams()


# One lock per language's synthesizer. `synthesize` is a plain `def`, so FastAPI runs each request on
# a thread-pool thread, and the per-request speed is written onto the SHARED model object
# (`tts_model.length_scale`) before synthesis: two overlapping requests would otherwise render each
# other's speed (an external review flagged this). The lock covers set-and-synthesize together.
# Cost, stated: requests for the SAME language are rendered one at a time -- a throughput ceiling
# per language that a load test (KCD-236) must measure; the remedy is one worker process per
# synthesizer, not removing the lock.
_SYNTH_LOCKS: dict[str, threading.Lock] = {lang: threading.Lock() for lang in SUPPORTED_LANGUAGES}


def _render(lang: str, text: str, speaker: str, speed: float, pauses: bool,
            params: ProsodyParams) -> tuple[np.ndarray, int]:
    synth = SYNTHESIZERS[lang]
    sample_rate = SAMPLE_RATES[lang]
    chunks = split_for_prosody(text, params.max_chunk_chars) if pauses else [(text, "sentence")]
    rendered = []
    with _SYNTH_LOCKS[lang]:
        if hasattr(synth.tts_model, "length_scale"):
            synth.tts_model.length_scale = speed
        for chunk_text, pause_kind in chunks:
            # split_sentences=False: agent/prosody.py already decided the
            # chunking, and letting Coqui re-split would reintroduce the
            # ragged joins.
            rendered.append((synth.tts(chunk_text, speaker_name=speaker, split_sentences=False), pause_kind))
    return assemble(rendered, sample_rate, params, pauses), sample_rate


class SynthesizeRequest(BaseModel):
    text: str
    lang: str = "bn"
    speaker: str | None = None
    speed: float | None = None      # speaking RATE multiplier: 1.0 normal, <1 slower (KCD-157)
    length_scale: float | None = None   # raw FastPitch duration multiplier, >1 slower; wins over speed
    pauses: bool = True
    # KCD-162: every prosody parameter tunable per request. Bounds are
    # enforced by agent.prosody.resolve_params (a bad value is a 422, not
    # a silent clamp).
    pause_sentence_s: float | None = None
    pause_clause_s: float | None = None
    pause_none_s: float | None = None
    trim_threshold: float | None = None
    target_peak: float | None = None
    max_chunk_chars: int | None = None
    lead_in_s: float | None = None


class UnsupportedLanguage(Exception):
    pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "languages": list(SYNTHESIZERS.keys()),
        "speaker": DEFAULT_SPEAKER,
        "sample_rates": SAMPLE_RATES,
        "length_scale": DEFAULT_LENGTH_SCALE,
        "prosody_defaults": dataclasses.asdict(DEFAULT_PROSODY),
    }


@app.get("/speakers")
def speakers():
    """KCD-511: the voices each language's synthesizer offers, and the one in use. To choose a
    different voice for the agent (for example a mature female voice), audition them with
    tools/audition_voices.py, then set TTS_SPEAKER; no code change."""
    out: dict[str, list[str]] = {}
    for lang, synth in SYNTHESIZERS.items():
        try:
            # the Bengali checkpoint's own speaker table carries a stray '"' entry; only real names are offered
            out[lang] = sorted(n for n in synth.tts_model.speaker_manager.name_to_id.keys()
                               if isinstance(n, str) and n.replace("_", "").replace("-", "").isalnum())
        except Exception:  # noqa: BLE001 - a single-speaker model simply has none to list
            out[lang] = []
    return {"default": DEFAULT_SPEAKER, "speakers": out}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if req.lang not in SYNTHESIZERS:
        # Fail loudly and specifically rather than silently falling back
        # to Bengali -- a caller who switched to Hindi does not want an
        # answer read out in the wrong language because the router or the
        # caller passed an unrecognised code.
        return Response(
            content=f'{{"error":"unsupported lang {req.lang!r}, have {list(SYNTHESIZERS)}"}}',
            media_type="application/json", status_code=422,
        )
    overrides = {f: getattr(req, f) for f in dataclasses.asdict(DEFAULT_PROSODY)}
    try:
        length_scale = resolve_length_scale(DEFAULT_LENGTH_SCALE, req.speed, req.length_scale)
        params = resolve_params(DEFAULT_PROSODY, overrides)
    except ValueError as e:
        return Response(content=f'{{"error":"{e}"}}', media_type="application/json", status_code=422)
    wav, sample_rate = _render(
        req.lang, req.text,
        req.speaker or DEFAULT_SPEAKER,
        length_scale,
        req.pauses,
        params,
    )
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
