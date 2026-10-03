"""Measure echo cancellation and barge-in (KCD-051, KCD-052). Two modes.

SYNTHETIC -- the story's "two hundred calls" harness, on generated scenes with
known ground truth. Reports how often the agent triggered on its own audio, how
often a caller talking over it was detected, and the detection latency
distribution. Runs anywhere.

    python tools/echo_eval.py --synthetic 200

REAL -- recorded handset audio, which is what the story actually asks for
("ERLE measured on real handsets"). A directory of pairs, recorded on a pod or a
handset rig:

    NAME_far.wav    what the agent played (the far-end reference), 16 kHz mono
    NAME_mic.wav    what the microphone captured meanwhile, 16 kHz mono
    NAME.json       {"echo_only_ms": [start, end],        # a stretch with no caller
                     "caller_onsets_ms": [9000, ...]}     # optional, when a caller talked over it

    python tools/echo_eval.py --pairs RECORDINGS_DIR

Reported per recording and in aggregate: ERLE over the echo-only stretch,
barge-in events during it (the agent must never trigger on its own audio: any
count above zero is a failure), and latency to the first event after each
annotated caller onset. Every synthetic number is a property of the synthetic
scenes; only the REAL mode says anything about a handset.
"""

import argparse
import glob
import json
import os
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tests")):  # appended: importing this must not shadow clinic-api/main.py
    if _p not in sys.path:
        sys.path.append(_p)

from agent.echo_cancel import erle_db
from agent.full_duplex import FullDuplexProcessor

SR = 16000


def run(mic, far, chunk=1600, dur_each=None):
    fd = FullDuplexProcessor()
    if dur_each:
        # the way the server would send it: one clip per voice, so a change of voice is known
        from _synth_echo import voice_segments

        for i, seg in enumerate(voice_segments(far, dur_each)):
            fd.place_reference(seg, voice=f"voice{i}")
    else:
        fd.place_reference(far)
    cleaned, events = [], []
    for i in range(0, mic.size, chunk):
        r = fd.process(mic[i : i + chunk])
        cleaned.append(r.cleaned)
        if r.barge_in is not None:
            events.append(r.barge_in)
    return fd, np.concatenate(cleaned), events


def synthetic(n_calls, verbose=False, voice_changes=False):
    from _synth_echo import caller, echo_scene

    own_audio_triggers = 0
    lat, missed = [], 0
    erles = []
    for k in range(n_calls):
        far, mic, _ = echo_scene(k, dur_each=4.0 + (k % 3), same_voice=not voice_changes)
        seg = (4.0 + (k % 3)) if voice_changes else None  # one clip per voice only when the voice really changes
        # even calls: echo only. odd calls: a caller talks over the agent
        if k % 2:
            c = caller(k, dur=1.5, f0=100 + 12 * (k % 9))
            on = int((7.5 + 0.13 * (k % 11)) * SR)
            if on + c.size > mic.size:
                continue
            m = mic.copy()
            m[on : on + c.size] += c
            fd, cleaned, events = run(m, far, dur_each=seg)
            hits = [e for e in events if e.detected_at_sample >= on]
            if hits:
                lat.append((hits[0].detected_at_sample - on) / SR * 1000.0)
            else:
                missed += 1
            if verbose:
                print(
                    f"call {k}: caller onset {on / SR:.2f}s, first event "
                    f"{f'+{lat[-1]:.0f} ms' if hits else 'MISSED'}, ERLE at onset {fd.echo.canceller.erle_db:.1f} dB"
                )
        else:
            fd, cleaned, events = run(mic, far, dur_each=seg)
            own_audio_triggers += len(events)
            if verbose and events:
                print(
                    f"call {k}: OWN-AUDIO TRIGGER at " + ", ".join(f"{e.detected_at_sample / SR:.2f}s" for e in events)
                )
            erles.append(erle_db(mic[-3 * SR :], cleaned[-3 * SR :]))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{n_calls}", file=sys.stderr, flush=True)
    print(
        f"echo-only calls: {len(erles)}  own-audio triggers: {own_audio_triggers}"
        f"  ERLE last 3 s: median {np.median(erles):.1f} dB, min {np.min(erles):.1f} dB"
    )
    print(f"caller calls: {len(lat) + missed}  detected: {len(lat)}  missed: {missed}")
    if lat:
        print(
            f"detection latency ms: median {np.median(lat):.0f}, p95 {np.percentile(lat, 95):.0f}, max {np.max(lat):.0f}"
            "   (budget: p95 <= 300 ms)"
        )
    return 1 if own_audio_triggers else 0


def _load(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, (
            f"{path}: need 16 kHz mono 16-bit"
        )
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def real(directory):
    failures, lat = 0, []
    for far_path in sorted(glob.glob(os.path.join(directory, "*_far.wav"))):
        name = os.path.basename(far_path)[: -len("_far.wav")]
        mic_path = os.path.join(directory, f"{name}_mic.wav")
        meta_path = os.path.join(directory, f"{name}.json")
        if not (os.path.exists(mic_path) and os.path.exists(meta_path)):
            print(f"skipping {name}: needs _mic.wav and .json")
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        far, mic = _load(far_path), _load(mic_path)
        n = min(far.size, mic.size)
        fd, cleaned, events = run(mic[:n], far[:n])
        line = [name]
        if "echo_only_ms" in meta:
            a, b = (int(x * SR / 1000) for x in meta["echo_only_ms"])
            line.append(f"ERLE {erle_db(mic[a:b], cleaned[a:b]):.1f} dB")
            own = [e for e in events if a <= e.detected_at_sample < b]
            line.append(f"own-audio triggers {len(own)}")
            failures += len(own)
        for onset_ms in meta.get("caller_onsets_ms", []):
            on = int(onset_ms * SR / 1000)
            hits = [e for e in events if e.detected_at_sample >= on]
            if hits:
                lat.append((hits[0].detected_at_sample - on) / SR * 1000.0)
                line.append(f"caller@{onset_ms}ms detected +{lat[-1]:.0f} ms")
            else:
                line.append(f"caller@{onset_ms}ms MISSED")
        print("  ".join(line))
    if lat:
        print(f"detection latency ms: median {np.median(lat):.0f}, p95 {np.percentile(lat, 95):.0f}")
    print("FAIL: the agent triggered on its own audio" if failures else "no own-audio triggers")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", type=int, metavar="N_CALLS")
    g.add_argument("--pairs", metavar="DIR")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--voice-changes",
        action="store_true",
        help="synthetic: change the agent's voice every 4-6 s (the worst case; deployed calls change it only on a language switch)",
    )
    a = ap.parse_args()
    return synthetic(a.synthetic, a.verbose, a.voice_changes) if a.synthetic else real(a.pairs)


if __name__ == "__main__":
    sys.exit(main())
