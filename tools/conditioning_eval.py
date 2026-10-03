"""Does input conditioning actually lower the word error rate? (KCD-055 level, KCD-057 noise)

agent/conditioning.py cleans noisy input and normalises level; its synthetic tests prove it
does not damage speech (retention, f0), but whether the RECOGNISER hears better afterwards is
a property of the recogniser and only a pod run can say. This tool is that run:

    for every clip in a directory, at each degradation, transcribe the audio
      (a) as it arrives and (b) after condition(), and report WER for both.

INPUT   a directory holding, per clip:
            NAME.wav     clean 16 kHz mono speech
            NAME.txt     the reference transcript
            NAME.json    optional {"lang": "bn"|"hi"|"en", "bucket": "clean_16k"|...}
        The clips should be REAL recordings of the target speakers and handsets -- degrading
        studio speech tells you about studio speech. By default the tool synthesises the level
        and noise conditions from each clean clip so one recording covers all of them;
        recordings that are ALREADY noisy or quiet should be run with `--no-degrade`.

ENGINE  the recogniser under test, one of:
            --engine en=http://localhost:8003          the HTTP recogniser (English, main.py)
            --engine-module mypkg.mod:factory          a factory returning
                                                       `async def transcribe(lang, wav_path) -> str`
                                                       (use this on the pod for the in-process
                                                       Bengali/Hindi engines)

OUTPUT  a table per (language, condition): WER raw, WER conditioned, and the change. A change
        that is not clearly negative is reported as such -- conditioning is only worth keeping
        for a condition where it helps; the default stays OFF (CONDITION_INPUT) until this
        says it should be on.

    python tools/conditioning_eval.py CLIPS_DIR --engine en=http://localhost:8003
"""

import argparse
import asyncio
import collections
import dataclasses
import importlib
import json
import os
import re
import sys
import tempfile
import unicodedata
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Appended, never inserted at the front: importing this module (tests do) must not push
# ROOT ahead of a directory that also holds a `main.py` (clinic-api/) on sys.path.
for _p in (ROOT, os.path.join(ROOT, "tests")):
    if _p not in sys.path:
        sys.path.append(_p)

from agent.conditioning import condition, mix_at_snr

SR = 16000
# Conditions: (name, peak level dBFS or None, noise SNR dB or None). REASONED spread of a quiet handset,
# a very quiet one, and street-like noise; replace with the conditions the pilot's calls actually show.
CONDITIONS = [
    ("as_recorded", None, None),
    ("quiet_-30dBFS", -30.0, None),
    ("very_quiet_-42dBFS", -42.0, None),
    ("noise_20dB", None, 20.0),
    ("noise_10dB", None, 10.0),
    ("noise_5dB", None, 5.0),
]


# ------------------------------------------------------------------------------- WER


def _tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text or "").lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return t.split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: word-level edit distance / reference length. 1.0 for an empty hypothesis."""
    r, h = _tokens(reference), _tokens(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


# ------------------------------------------------------------------------------- clips


@dataclasses.dataclass
class Clip:
    name: str
    lang: str
    samples: np.ndarray
    reference: str


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit PCM")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: str, x: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())


def load_clips(directory: str) -> list[Clip]:
    clips = []
    for name in sorted(f[:-4] for f in os.listdir(directory) if f.endswith(".wav")):
        ref_path = os.path.join(directory, name + ".txt")
        if not os.path.exists(ref_path):
            continue
        meta = {}
        if os.path.exists(os.path.join(directory, name + ".json")):
            with open(os.path.join(directory, name + ".json"), encoding="utf-8") as f:
                meta = json.load(f)
        with open(ref_path, encoding="utf-8") as f:
            ref = f.read().strip()
        clips.append(Clip(name, meta.get("lang", "en"), read_wav(os.path.join(directory, name + ".wav")), ref))
    return clips


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(f), n).astype(np.float32)
    return x / (np.sqrt(np.mean(x**2)) + 1e-9)


def degrade(x: np.ndarray, level_dbfs: float | None, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    if snr_db is not None:
        y = mix_at_snr(y, pink_noise(len(y), rng), snr_db)
    if level_dbfs is not None:
        peak = float(np.max(np.abs(y))) + 1e-9
        y = y * (10 ** (level_dbfs / 20.0) / peak)
    return y


# ------------------------------------------------------------------------------- evaluation

Transcribe = "async (lang, wav_path) -> str"


async def evaluate(
    clips: list[Clip],
    transcribe,
    conditions=CONDITIONS,
    degrade_clips: bool = True,
    conditioner=condition,
    seed: int = 0,
) -> dict:
    """-> {(lang, condition): {"n", "wer_raw", "wer_conditioned", "change"}}, plus per-clip rows."""
    rng = np.random.default_rng(seed)
    acc: dict[tuple[str, str], list[tuple[float, float]]] = collections.defaultdict(list)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for clip in clips:
            for cname, level, snr in conditions if degrade_clips else [("as_recorded", None, None)]:
                x = degrade(clip.samples, level, snr, rng) if degrade_clips else clip.samples
                y, report = conditioner(x, SR)
                # a fresh path per file: a recogniser (or the OS) may hold or cache by path
                k = len(rows)
                raw_p, cond_p = os.path.join(tmp, f"raw{k}.wav"), os.path.join(tmp, f"cond{k}.wav")
                write_wav(raw_p, x)
                write_wav(cond_p, y)
                w_raw = wer(clip.reference, await transcribe(clip.lang, raw_p))
                w_cond = wer(clip.reference, await transcribe(clip.lang, cond_p))
                acc[(clip.lang, cname)].append((w_raw, w_cond))
                rows.append(
                    {
                        "clip": clip.name,
                        "lang": clip.lang,
                        "condition": cname,
                        "wer_raw": w_raw,
                        "wer_conditioned": w_cond,
                        "suppressed": report.suppressed,
                        "reason": report.reason,
                    }
                )
    table = {}
    for key, vals in acc.items():
        raw = float(np.mean([a for a, _ in vals]))
        cond = float(np.mean([b for _, b in vals]))
        table[key] = {"n": len(vals), "wer_raw": raw, "wer_conditioned": cond, "change": cond - raw}
    return {"table": table, "rows": rows}


def verdict(change: float, n: int, tolerance: float = 0.01) -> str:
    if n < 10:
        return "too few clips to say"
    if change <= -tolerance:
        return "HELPS"
    if change >= tolerance:
        return "HURTS"
    return "no clear effect"


def render(result: dict) -> str:
    lines = [
        "| language | condition | clips | WER raw | WER conditioned | change | verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for (lang, cond), r in sorted(result["table"].items()):
        lines.append(
            f"| {lang} | {cond} | {r['n']} | {r['wer_raw']:.1%} | {r['wer_conditioned']:.1%} | "
            f"{r['change']:+.1%} | {verdict(r['change'], r['n'])} |"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------------------- engines


def http_engine(specs: list[str]):
    import httpx

    urls = dict(s.split("=", 1) for s in specs)

    async def transcribe(lang: str, wav_path: str) -> str:
        if lang not in urls:
            raise SystemExit(f"no --engine for language {lang!r}; use --engine {lang}=URL or --engine-module")
        async with httpx.AsyncClient(base_url=urls[lang].rstrip("/"), timeout=30.0) as c:
            r = await c.post("/transcribe", json={"wav_path": wav_path})
            r.raise_for_status()
            return r.json().get("text", "")

    return transcribe


def module_engine(spec: str):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips_dir")
    ap.add_argument("--engine", action="append", default=[], metavar="LANG=URL")
    ap.add_argument("--engine-module", metavar="MOD:FACTORY")
    ap.add_argument("--no-degrade", action="store_true", help="use the recordings as they are")
    ap.add_argument("--json", metavar="PATH", help="also write every per-clip row here")
    args = ap.parse_args(argv)
    if not args.engine and not args.engine_module:
        ap.error("give --engine LANG=URL or --engine-module MOD:FACTORY")
    transcribe = module_engine(args.engine_module) if args.engine_module else http_engine(args.engine)
    clips = load_clips(args.clips_dir)
    if not clips:
        ap.error(f"no NAME.wav + NAME.txt pairs in {args.clips_dir}")
    result = asyncio.run(evaluate(clips, transcribe, degrade_clips=not args.no_degrade))
    print(render(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result["rows"], f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
