"""Calibrate agent/endpointing.py's silence threshold against human-marked turn
ends (KCD-047). Run on a machine that has the annotated recordings.

    python tools/calibrate_endpointing.py RECORDINGS_DIR [--out calibration.json]
                                          [--target-false-cut 0.02] [--vad energy|silero]

RECORDINGS_DIR holds NAME.wav + NAME.json pairs; see agent/endpoint_calibration.py
for the schema. The report says MEASURED only with at least 200 recordings; the
`config` block is what agent.endpointing.config_from_json() loads. Both error
rates are printed for every candidate so they can be published.
"""

import argparse
import glob
import json
import os
import sys
import wave

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:  # appended: importing this must not shadow clinic-api/main.py
    sys.path.append(_ROOT)

from agent.endpoint_calibration import Recording, calibrate, write_report


def _load_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, width, raw = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: only 16-bit PCM is supported")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return (x.reshape(-1, ch).mean(axis=1) if ch > 1 else x), sr


def load_recordings(directory, vad):
    silero = None
    if vad == "silero":
        from agent.vad_stream import TurnDetector  # needs torch + the Silero repo (a pod)

        silero = TurnDetector()
    recs = []
    for wav_path in sorted(glob.glob(os.path.join(directory, "*.wav"))):
        meta_path = os.path.splitext(wav_path)[0] + ".json"
        if not os.path.exists(meta_path):
            print(f"skipping {os.path.basename(wav_path)}: no annotation", file=sys.stderr)
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if "turn_end_ms" not in meta:
            print(f"skipping {os.path.basename(wav_path)}: no turn_end_ms", file=sys.stderr)
            continue
        x, sr = _load_wav(wav_path)
        spans = None
        if silero is not None:
            import torch

            spans, _ = silero.spans(torch.from_numpy(x), sr)
        recs.append(
            Recording(x, sr, meta["turn_end_ms"] / 1000.0, meta.get("language", ""), meta.get("channel", ""), spans)
        )
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--out", default="endpoint_calibration.json")
    ap.add_argument("--target-false-cut", type=float, default=0.02)
    ap.add_argument("--vad", choices=["energy", "silero"], default="silero")
    a = ap.parse_args()
    recs = load_recordings(a.directory, a.vad)
    report = calibrate(recs, target_false_cut=a.target_false_cut)
    write_report(report, a.out)
    print(f"{report['status']}: {report['recordings']} recordings (need {report['required_recordings']})")
    print(f"{'silence_s':>9} {'false-cut':>10} {'95% CI':>16} {'false-wait':>11} {'wait p50':>9} {'wait p95':>9}")
    for r in report["table"]:
        lo, hi = r["false_cut_ci95"]
        print(
            f"{r['silence_confirm_s']:>9.2f} {r['false_cut_rate']:>10.1%} {lo:>7.1%}-{hi:<7.1%} "
            f"{r['false_wait_rate']:>11.1%} {r['wait_p50_s']:>8.2f}s {r['wait_p95_s']:>8.2f}s"
        )
    print(report["note"])
    return 0 if report["status"] == "MEASURED" else 2


if __name__ == "__main__":
    sys.exit(main())
