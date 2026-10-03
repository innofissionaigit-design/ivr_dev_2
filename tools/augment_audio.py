"""Apply a telephony-degradation profile (agent/telephony_augment.py) to WAV files on disk.

    python tools/augment_audio.py in/*.wav --profile rural_line_8k --out out/
    python tools/augment_audio.py clean.wav --profile noisy_8k --noise office_hum.wav --seed 7
    python tools/augment_audio.py --list-profiles

Needs no pod and no real call recording: it runs on any WAV file (a public-domain clip, a TTS-generated sentence,
even a generated tone) to build a synthetic regression/stress set before real recordings exist. `--noise` is optional;
without it, synthetic white noise stands in.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from agent.telephony_augment import PROFILES, apply_profile  # noqa: E402


def _mono(data: np.ndarray) -> np.ndarray:
    return data if data.ndim == 1 else data.mean(axis=1).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="input WAV files")
    ap.add_argument("--profile", default="rural_line_8k", choices=sorted(PROFILES))
    ap.add_argument("--out", default=".", help="output directory; each file is written as <name>.<profile>.wav")
    ap.add_argument("--noise", help="a WAV file of background noise; default is synthetic white noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list-profiles", action="store_true")
    args = ap.parse_args(argv)

    if args.list_profiles:
        for name, spec in sorted(PROFILES.items()):
            print(f"{name}: target_sample_rate={spec.target_sample_rate} snr_db={spec.snr_db} codec={spec.codec}")
        return 0
    if not args.files:
        ap.error("give at least one WAV file, or --list-profiles")

    noise = None
    if args.noise:
        noise_data, _sr = sf.read(args.noise, dtype="float32", always_2d=True)
        noise = _mono(noise_data)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    spec = PROFILES[args.profile]
    for path in args.files:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        out, out_rate = apply_profile(_mono(data), sample_rate, spec, rng, noise)
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out, f"{base}.{args.profile}.wav")
        sf.write(out_path, out, out_rate)
        print(f"{path} -> {out_path} ({sample_rate} Hz -> {out_rate} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
