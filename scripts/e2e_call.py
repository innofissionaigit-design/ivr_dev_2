"""Drive a real call through the PCM voice agent from a script.

Speaks each WAV to ws://<host>:<port>/ws/audio exactly the way the browser
does (hello frame, 16 kHz pcm_s16le binary frames, `playback_done` once the
reply audio has arrived), then prints what the caller side saw for every
turn: the transcript the agent heard, the reply it spoke, and the time from
the end of the caller's speech to the first reply audio.

This exercises the WHOLE path -- VAD, language ID, the per-language ASR, the
intent layer, the clinic API, the template, the per-language TTS -- which is
the thing unit tests cannot. Run it in the main venv on the pod:

    python scripts/e2e_call.py --port 8101 a.wav b.wav
    python scripts/e2e_call.py --port 8101 --json a.wav       # machine-readable

The input WAVs are usually synthesized by tts_server.py, so this proves the
plumbing, NOT recognition quality on real callers: clean synthetic speech is
not 8 kHz telephony audio from a Kolkata mobile.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np
import soundfile as sf

RATE = 16000
CHUNK_S = 0.1
TAIL_SILENCE_S = 1.8  # enough for the turn detector to call the utterance finished


def load_16k_mono(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != RATE:
        import torch
        import torchaudio

        mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, RATE).numpy()
    return mono


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


async def stream(ws, samples: np.ndarray, speed: float) -> float:
    """Send `samples` at `speed`x real time. Returns monotonic time when the
    last sample was sent (= end of speech, before the tail silence)."""
    step = int(CHUNK_S * RATE)
    for i in range(0, len(samples), step):
        await ws.send(to_pcm16(samples[i : i + step]))
        await asyncio.sleep(CHUNK_S / speed)
    t_end = time.monotonic()
    silence = np.zeros(step, dtype=np.float32)
    for _ in range(int(TAIL_SILENCE_S / CHUNK_S)):
        await ws.send(to_pcm16(silence))
        await asyncio.sleep(CHUNK_S / speed)
    return t_end


async def collect_turn(ws, timeout_s: float) -> dict:
    """Read frames until a reply's audio has arrived. Returns what was seen."""
    turn = {"user": None, "ai": [], "audio_bytes": 0, "first_audio_at": None, "handoff": None}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
        except TimeoutError:
            break
        if isinstance(msg, bytes):
            if turn["first_audio_at"] is None:
                turn["first_audio_at"] = time.monotonic()
            turn["audio_bytes"] += len(msg)
            return turn  # one reply clip per turn is what we asked for
        data = json.loads(msg)
        if data.get("type") == "handoff_human":
            turn["handoff"] = data.get("reason")
        elif data.get("sender") == "User":
            turn["user"] = data["text"]
        elif data.get("sender") == "AI":
            turn["ai"].append(data["text"])
    return turn


async def run(args) -> list[dict]:
    import websockets

    results = []
    async with websockets.connect(f"ws://{args.host}:{args.port}/ws/audio", max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "sampleRate": RATE, "format": "pcm_s16le"}))
        greeting = await collect_turn(ws, 30)  # greeting text + audio
        await ws.send(json.dumps({"type": "playback_done"}))
        results.append({"turn": "greeting", **{k: greeting[k] for k in ("ai", "audio_bytes", "handoff")}})
        if greeting["handoff"]:
            return results

        for path in args.wavs:
            samples = load_16k_mono(path)
            # After playback_done the server resyncs to the end of the buffer on
            # its NEXT poll tick and discards whatever arrived before that. A
            # real caller cannot be talking yet (the mic is muted during
            # playback), so wait it out or the harness eats its own first words.
            await asyncio.sleep(1.2)
            t_end = await stream(ws, samples, args.speed)
            turn = await collect_turn(ws, args.timeout)
            # the reply is spoken back to the caller; tell the server it finished playing
            await ws.send(json.dumps({"type": "playback_done"}))
            latency = (turn["first_audio_at"] - t_end) if turn["first_audio_at"] else None
            results.append(
                {
                    "turn": path,
                    "heard": turn["user"],
                    "reply": " | ".join(turn["ai"]),
                    "latency_s": round(latency, 2) if latency is not None else None,
                    "reply_audio_bytes": turn["audio_bytes"],
                    "handoff": turn["handoff"],
                }
            )
            if turn["handoff"]:
                break
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--speed", type=float, default=1.0, help="send faster than real time (1.0 = real time)")
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out = asyncio.run(run(args))
    for r in out:
        print(json.dumps(r, ensure_ascii=False) if args.json else "\n".join(f"{k}: {v}" for k, v in r.items()) + "\n")


if __name__ == "__main__":
    main()
