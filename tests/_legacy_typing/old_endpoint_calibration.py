"""KCD-047: measure the silence threshold against human-marked turn ends.

The threshold in agent/endpointing.py (silence_confirm_s = 1.0) was raised from
0.8 "by feel" and is labelled REASONED. The story replaces it with a MEASURED
value, per principle P5: at least two hundred real narrowband recordings, each
with a human-marked `turn_end_ms` (the moment the caller actually finished),
and both error rates published:

  false cut   the detector committed the turn while the caller was still going
              (its committed utterance end is more than `cut_tolerance_s` before
              the human-marked end): the agent talks over a caller who paused.
              The tolerance is 0.3 s because a VAD's end and a human's end never
              agree exactly (a fading final syllable, a trailing breath) -- a cut
              is the loss of a following PHRASE, not of that tail;
  false wait  the agent needed more than `wait_budget_s` after the caller really
              finished before it committed: the caller sits in silence.

This module is the instrument, and it is pure: recordings in, a table of rates
per candidate threshold out, and a chosen value. It cannot fabricate data.
`calibrate()` marks its result MEASURED only when it was given at least
`min_recordings` (200) recordings; below that the status is INSUFFICIENT_DATA and
the caller must keep the REASONED value. tools/calibrate_endpointing.py loads real
recordings; tests/test_endpoint_calibration.py exercises the maths on synthetic
audio whose true end is known by construction.

ANNOTATION SCHEMA (minimal; the fuller Appendix K schema is KCD-304's, this is
the subset this tool needs). For each `NAME.wav` a sidecar `NAME.json`:
    {"turn_end_ms": 4230,             # REQUIRED: when the caller finished
     "language": "bn",                # optional, for per-language rates
     "channel": "handset|speaker",    # optional, for per-channel rates
     "annotator": "id"}               # optional
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence

import numpy as np

from agent.endpointing import EndpointConfig, decide, energy_spans

MIN_RECORDINGS = 200


@dataclasses.dataclass
class Recording:
    samples: np.ndarray
    sr: int
    turn_end_s: float  # human-marked
    language: str = ""
    channel: str = ""
    spans: list | None = None  # precomputed speech spans (e.g. from Silero); else energy_spans


@dataclasses.dataclass
class CandidateResult:
    silence_confirm_s: float
    n: int
    false_cut_rate: float
    false_cut_ci95: tuple[float, float]
    false_wait_rate: float
    wait_p50_s: float
    wait_p95_s: float


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a rate: honest at the small counts where a
    false-cut rate near 1-2% lives."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def commit_time(rec: Recording, cfg: EndpointConfig, step_s: float = 0.02) -> tuple[float | None, float | None]:
    """(time the detector commits, the utterance end it commits) for a recording,
    replaying arrival of audio in `step_s` increments against the causal spans."""
    spans = rec.spans if rec.spans is not None else energy_spans(rec.samples, rec.sr)
    total = rec.samples.size / rec.sr
    t = 0.2
    while t <= total + 1e-9:
        seen = [{"start": s["start"], "end": min(s["end"], t)} for s in spans if s["start"] < t]
        d = decide(seen, t, cfg, semantic=False)
        if d.utterance_end_s is not None:
            return t, d.utterance_end_s
        t += step_s
    return None, None


def evaluate(
    recordings: Sequence[Recording],
    silence_confirm_s: float,
    base: EndpointConfig = EndpointConfig(),
    cut_tolerance_s: float = 0.30,
    wait_budget_s: float = 1.0,
) -> CandidateResult:
    cfg = dataclasses.replace(base, silence_confirm_s=silence_confirm_s)
    cuts, waits, long_waits = 0, [], 0
    for rec in recordings:
        t_commit, utt_end = commit_time(rec, cfg)
        if t_commit is None:
            continue
        if utt_end < rec.turn_end_s - cut_tolerance_s:
            cuts += 1
        wait = max(0.0, t_commit - rec.turn_end_s)
        waits.append(wait)
        long_waits += wait > wait_budget_s
    n = len(waits)
    return CandidateResult(
        silence_confirm_s=silence_confirm_s,
        n=n,
        false_cut_rate=cuts / n if n else 0.0,
        false_cut_ci95=wilson_interval(cuts, n),
        false_wait_rate=long_waits / n if n else 0.0,
        wait_p50_s=float(np.percentile(waits, 50)) if waits else 0.0,
        wait_p95_s=float(np.percentile(waits, 95)) if waits else 0.0,
    )


def calibrate(
    recordings: Sequence[Recording],
    candidates: Sequence[float] = tuple(np.arange(0.4, 1.6, 0.1)),
    target_false_cut: float = 0.02,
    min_recordings: int = MIN_RECORDINGS,
    base: EndpointConfig = EndpointConfig(),
    **kw,
) -> dict:
    """Pick the SHORTEST threshold whose false-cut rate (upper 95% bound) stays
    under the target -- shortest, because false wait is what it trades against.
    Returns a JSON-serialisable report; `status` is MEASURED only with enough data."""
    table = [evaluate(recordings, float(c), base, **kw) for c in candidates]
    ok = [r for r in table if r.false_cut_ci95[1] <= target_false_cut]
    if ok:
        chosen, interval_cleared = min(ok, key=lambda r: r.silence_confirm_s), True
    else:
        # the interval is too wide to clear the target: fall back to the point
        # estimate, and say so in the report
        under = [r for r in table if r.false_cut_rate <= target_false_cut]
        chosen = (
            min(under, key=lambda r: r.silence_confirm_s) if under else max(table, key=lambda r: r.silence_confirm_s)
        )
        interval_cleared = False
    enough = len(recordings) >= min_recordings
    return {
        "status": "MEASURED" if enough else "INSUFFICIENT_DATA",
        "recordings": len(recordings),
        "required_recordings": min_recordings,
        "target_false_cut_rate": target_false_cut,
        "target_cleared_at_95pct_confidence": interval_cleared,
        "chosen_silence_confirm_s": chosen.silence_confirm_s,
        "baseline_silence_confirm_s": base.silence_confirm_s,
        "config": {"silence_confirm_s": chosen.silence_confirm_s} if enough else {},
        "table": [dataclasses.asdict(r) for r in table],
        "note": (
            "Replaces the REASONED threshold."
            if enough
            else f"Only {len(recordings)} recordings; {min_recordings} are required before this may replace the REASONED value. "
            "Nothing was changed."
        ),
    }


def write_report(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
