"""ASR for one caller turn -- self-contained IndicConformer integration.

This project runs on its OWN pod, independent of voice-to-rx-repo (that
system lives on a different pod entirely -- see its own HANDOFF.md). An
earlier version of this file imported voicerx.asr.ASRNode directly to
reuse that project's already-debugged NeMo integration, but that created
a live dependency on an unrelated codebase (drug gazetteers, prescription
extraction -- none of it relevant here) that isn't even deployed on this
pod. Fixed by inlining just the actual proven ASR logic instead.

The logic below -- NOT the code coupling -- is what's actually worth
reusing, because it encodes real, previously-debugged failures:

  - Mainline NeMo cannot load this checkpoint at all: IndicConformer uses
    a multilingual AGGREGATE tokenizer, and mainline's
    _setup_monolingual_tokenizer raises KeyError: 'dir'. Only AI4Bharat's
    NeMo fork (github.com/AI4Bharat/NeMo, branch nemo-v2) handles it.

  - RNNT's default "greedy_batch" decoding strategy silently returns
    empty text on real audio; "greedy" does not. Confirmed on this exact
    pod: verified 2026-08-24 that "greedy_batch" is the library default
    and must be overridden explicitly via change_decoding_strategy().

  - .transcribe() has been observed to return either a flat list of
    strings or a list of single-item lists depending on internal batching
    state -- unwrap defensively.
"""

from __future__ import annotations

import asyncio
import dataclasses
import glob
import os
import threading

import numpy as np
import torch

# The AI4Bharat NeMo fork (nemo_toolkit 1.23) calls np.sctypes, which NumPy 2.0
# removed. Loading the checkpoint works, so the model "comes up" -- and then
# the FIRST real transcription raises AttributeError inside NeMo's audio
# loader. Restored here rather than downgrading numpy, which torch/scipy/
# speechbrain in this venv were installed against.
if not hasattr(np, "sctypes"):
    # (set through the module's namespace, not `np.sctypes = ...`: numpy 2's type stubs no longer declare this attribute)
    vars(np)["sctypes"] = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from omegaconf import OmegaConf

# HF repo per language -- bn/hi share this fork's checkpoint family; en is
# NOT here on purpose, it needs mainline NeMo (see the module docstring)
# and is loaded by a separate process/venv, never by this class.
MODEL_ID_BY_LANGUAGE = {
    "bn": "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large",
    "hi": "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large",
}


def _resolve_nemo_file(language_id: str = "bn") -> str:
    """Locate the .nemo checkpoint without hardcoding a machine-specific
    HF cache path.

    Resolution order:
      1. VOICE_AGENT_NEMO_FILE_<LANG> (e.g. VOICE_AGENT_NEMO_FILE_BN) --
         the explicit, unambiguous path deploy/env.sh sets once more than
         one IndicConformer checkpoint lives on the box (bn AND hi, since
         the pilot single-L4 architecture routes between them --
         docs/adr/0001-pilot-single-l4-architecture.md).
      2. VOICE_AGENT_NEMO_FILE (unqualified) -- kept for backward
         compatibility with the original Bengali-only deployment.
      3. A glob of the HF cache, FILTERED by language_id. An unqualified
         glob across every "*indicconformer*" directory is only safe when
         exactly one checkpoint exists; with two, `hits[0]` picks
         whichever the filesystem happens to list first, silently loading
         the wrong language's model into the wrong TurnASR instance.
    """
    per_lang = os.environ.get(f"VOICE_AGENT_NEMO_FILE_{language_id.upper()}")
    if per_lang and os.path.exists(per_lang):
        return per_lang

    explicit = os.environ.get("VOICE_AGENT_NEMO_FILE")
    if explicit and os.path.exists(explicit):
        return explicit

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pattern = os.path.join(hf_home, "hub", "**", f"*indicconformer*{language_id}*", "**", "*.nemo")
    hits = glob.glob(pattern, recursive=True) or glob.glob(
        os.path.join(hf_home, "hub", "**", "*indicconformer*", "**", f"*{language_id}*.nemo"),
        recursive=True,
    )
    if hits:
        return hits[0]

    repo = MODEL_ID_BY_LANGUAGE.get(language_id, f"<no known AI4Bharat repo for {language_id!r}>")
    raise FileNotFoundError(
        f"Could not locate an IndicConformer .nemo checkpoint for language_id={language_id!r} "
        f"under {hf_home}. Set VOICE_AGENT_NEMO_FILE_{language_id.upper()} explicitly, or "
        f"download it first: huggingface_hub.snapshot_download('{repo}', token=<HF_TOKEN>) -- "
        f"the AI4Bharat IndicConformer models are gated, accept the licence on "
        f"huggingface.co first."
    )


def _first_text(texts) -> str:
    if not texts:
        return ""
    item = texts[0]
    while isinstance(item, (list, tuple)):
        if not item:
            return ""
        item = item[0]
    return (item or "").strip()


def _word_agreement(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@dataclasses.dataclass
class ASRResult:
    text: str
    decoder_used: str
    decoder_agreement: float = 1.0


class TurnASR:
    """One instance shared across all calls -- the NeMo model is the
    expensive singleton, loaded once at process startup."""

    def __init__(self, nemo_file: str | None = None, language_id: str = "bn", device: str | None = None):
        self.language_id = language_id
        # cur_decoder is flipped per call; concurrent threads must not interleave.
        self._lock = threading.Lock()
        nemo_file = nemo_file or _resolve_nemo_file(language_id)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = nemo_asr.models.ASRModel.restore_from(restore_path=nemo_file)
        self.model = self.model.to(self.device)
        self.model.freeze()

        # The proven fix -- see module docstring for why the library
        # default ("greedy_batch") is not used here.
        rnnt_cfg = OmegaConf.structured(RNNTDecodingConfig(strategy="greedy"))
        self.model.change_decoding_strategy(rnnt_cfg, decoder_type="rnnt")

    def _transcribe_clip(self, clip_path: str) -> tuple[str, str]:
        self.model.cur_decoder = "ctc"
        ctc_texts = self.model.transcribe(
            [clip_path],
            batch_size=1,
            logprobs=False,
            language_id=self.language_id,
        )
        ctc_text = _first_text(ctc_texts)

        self.model.cur_decoder = "rnnt"
        rnnt_texts = self.model.transcribe([clip_path], batch_size=1, language_id=self.language_id)
        rnnt_text = _first_text(rnnt_texts)

        return ctc_text, rnnt_text

    def transcribe_utterance_sync(self, wav_path: str) -> ASRResult:
        """Blocking. Callers MUST run this via asyncio.to_thread -- it
        holds the GIL through GPU inference and would otherwise stall
        every other WebSocket connection's audio handling on this process."""
        with self._lock:
            ctc_text, rnnt_text = self._transcribe_clip(wav_path)

        if rnnt_text:
            return ASRResult(
                text=rnnt_text, decoder_used="rnnt", decoder_agreement=round(_word_agreement(ctc_text, rnnt_text), 2)
            )
        if ctc_text:
            return ASRResult(text=ctc_text, decoder_used="ctc_fallback", decoder_agreement=0.0)
        return ASRResult(text="", decoder_used="none", decoder_agreement=1.0)

    async def transcribe_utterance(self, wav_path: str) -> ASRResult:
        return await asyncio.to_thread(self.transcribe_utterance_sync, wav_path)
