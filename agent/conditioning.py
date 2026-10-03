"""KCD-055 / KCD-057: condition an utterance before recognition -- noise
suppression and level normalisation -- and measure that it helps rather than
hopes.

PIPELINE (condition())
  1. rumble / DC removal and Wiener-style spectral gating (agent/audio_quality.enhance)
     -- but only WHEN THE CLIP IS NOISY (SNR under NOISY_BELOW_DB). A clean clip
     is left alone: suppression can only hurt it.
  2. THE GUARD. Suppression must never "improve the noise at the cost of the
     speech". After suppressing, the voiced frames of the ORIGINAL are checked in
     the output: they must still be voiced (retention >= MIN_RETENTION) and at the
     same pitch (median relative error <= MAX_F0_ERROR). If suppression ate them,
     the unsuppressed clip is used instead and the report says why. This is the
     off-pod, reference-free stand-in for "entity accuracy did not fall": it
     cannot see a mis-recognised word, but it can see that the harmonic structure
     the recogniser relies on is still there.
  3. level normalisation with a soft limiter (agent/level.py).

MEASUREMENT (bench())
  Recognition accuracy needs an ASR and real recordings; that is
  tools/conditioning_eval.py, run on a pod. What runs anywhere is a PROXY with a
  known clean reference: `speech_frame_accuracy` -- of the frames that are voiced
  in the CLEAN signal, the fraction that are voiced in the test signal at the same
  pitch (within 5%) -- plus the SNR against the clean reference. bench() reports
  both BEFORE and AFTER conditioning for every (noise profile, SNR) bucket, the
  degradation of the worst bucket against the best, and whether any bucket got
  worse (which the story forbids). The proxy is labelled as such wherever it is
  reported; it is not a WER.

All thresholds are REASONED and validated on SYNTHETIC speech and noise
(tests/test_conditioning.py); recalibrate on real narrowband recordings.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np

from agent.audio_quality import VOICED_HARMONICITY, _autocorr, _db, _frames, _pitch_lag, _refine_lag, assess, enhance
from agent.level import LevelReport, normalise

NOISY_BELOW_DB = 18.0  # apply suppression only under this frame-energy SNR (REASONED)
MIN_RETENTION = 0.90  # share of the original's voiced frames still voiced (REASONED)
MAX_F0_ERROR = 0.03  # median relative pitch error allowed (REASONED)
F0_MATCH = 0.05  # a pitch counts as the same within 5%


@dataclasses.dataclass
class ConditioningReport:
    suppressed: bool
    reason: str  # why suppression was or was not applied
    snr_before_db: float
    voiced_retention: float | None
    f0_error: float | None
    level: LevelReport | None


def _voiced_f0(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """(voiced mask, f0 in Hz) per 10 ms hop over ALL frames (0 where unvoiced)."""
    frames = _frames(np.asarray(x, np.float32) - float(np.mean(x)), sr)
    n = frames.shape[0]
    mask, f0 = np.zeros(n, bool), np.zeros(n)
    if n < 5:
        return mask, f0
    e = _db(np.mean(frames**2, axis=1))
    floor, top = float(np.percentile(e, 10)), float(np.percentile(e, 90))
    active = np.where(e >= max(floor + 0.35 * (top - floor), top - 25.0))[0]
    if active.size == 0:
        return mask, f0
    acf = _autocorr(frames[active])
    lags, harm = _pitch_lag(acf, sr)
    for j, i in enumerate(active):
        if harm[j] >= VOICED_HARMONICITY:
            mask[i] = True
            f0[i] = sr / max(_refine_lag(acf[j], int(lags[j])), 1.0)
    return mask, f0


def speech_frame_accuracy(clean: np.ndarray, test: np.ndarray, sr: int = 16000) -> float:
    """PROXY for recognition accuracy (see module docstring): of the frames voiced
    in `clean`, the fraction voiced in `test` at the same pitch."""
    n = min(clean.size, test.size)
    cm, cf = _voiced_f0(clean[:n], sr)
    tm, tf = _voiced_f0(test[:n], sr)
    m = min(cm.size, tm.size)
    cm, cf, tm, tf = cm[:m], cf[:m], tm[:m], tf[:m]
    if not cm.any():
        return 1.0
    ok = cm & tm & (np.abs(tf / np.maximum(cf, 1e-9) - 1.0) <= F0_MATCH)
    return float(ok.sum() / cm.sum())


def snr_vs_clean_db(clean: np.ndarray, test: np.ndarray) -> float:
    """SNR of `test` against a known clean reference, after the best scalar gain."""
    n = min(clean.size, test.size)
    c, o = clean[:n].astype(np.float64), test[:n].astype(np.float64)
    g = float(np.dot(o, c) / max(np.dot(c, c), 1e-12))
    err = o - g * c
    return float(10 * np.log10((np.sum((g * c) ** 2) + 1e-12) / (np.sum(err**2) + 1e-12)))


def preservation(before: np.ndarray, after: np.ndarray, sr: int = 16000) -> tuple[float, float]:
    """(voiced retention, median relative f0 error) of `after` against `before`,
    over the frames that were voiced in `before`. Reference-free."""
    bm, bf = _voiced_f0(before, sr)
    am, af = _voiced_f0(after, sr)
    m = min(bm.size, am.size)
    bm, bf, am, af = bm[:m], bf[:m], am[:m], af[:m]
    if not bm.any():
        return 1.0, 0.0
    retention = float((bm & am).sum() / bm.sum())
    both = bm & am
    err = float(np.median(np.abs(af[both] / np.maximum(bf[both], 1e-9) - 1.0))) if both.any() else 1.0
    return retention, err


def condition(x: np.ndarray, sr: int = 16000, suppress: str = "auto", normalise_level: bool = True):
    """-> (conditioned samples, ConditioningReport). `suppress`: "auto" (only when
    noisy, guarded), "always" (guarded), "never"."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    a = assess(x, sr)
    y, suppressed, reason, retention, f0err = x, False, "not_requested", None, None
    if suppress != "never":
        if suppress == "auto" and a.snr_db >= NOISY_BELOW_DB:
            reason = "clean_enough"
        else:
            cand = enhance(x, sr, target_rms_dbfs=-200.0)  # suppression only; level is handled below
            retention, f0err = preservation(x, cand, sr)
            if retention >= MIN_RETENTION and f0err <= MAX_F0_ERROR:
                y, suppressed, reason = cand, True, "applied"
            else:
                reason = "rejected_it_removed_speech"
    level_report = None
    if normalise_level:
        y, level_report = normalise(y, sr)
    return y, ConditioningReport(suppressed, reason, a.snr_db, retention, f0err, level_report)


# ------------------------------------------------------------------- bench


@dataclasses.dataclass
class BucketRow:
    profile: str
    snr_db: float
    snr_before_db: float
    snr_after_db: float
    accuracy_before: float
    accuracy_after: float
    suppressed_share: float

    @property
    def snr_gain_db(self) -> float:
        return self.snr_after_db - self.snr_before_db

    @property
    def accuracy_change(self) -> float:
        return self.accuracy_after - self.accuracy_before


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    active = clean[np.abs(clean) > 0.05 * np.max(np.abs(clean))]
    ps, pn = float(np.mean(active**2)), float(np.mean(noise**2)) + 1e-12
    return (clean + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))).astype(np.float32)


def bench(
    clean_clips: Sequence[np.ndarray],
    noise_for: Callable[[str, int, int], np.ndarray],
    profiles: Sequence[str],
    snrs: Sequence[float],
    sr: int = 16000,
    conditioner: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Run every (profile, SNR) bucket before and after conditioning. `noise_for(profile, n, seed)`
    supplies noise. Returns the rows and the summary the story asks to be stated."""
    rows: list[BucketRow] = []
    for prof in profiles:
        for snr in snrs:
            sb, sa, ab, aa, sup = [], [], [], [], []
            for k, clean in enumerate(clean_clips):
                noisy = mix_at_snr(clean, noise_for(prof, clean.size, 1000 + k), snr)
                if conditioner is None:
                    out, rep = condition(noisy, sr)
                    sup.append(float(rep.suppressed))
                else:
                    out = conditioner(noisy)
                    sup.append(0.0)
                sb.append(snr_vs_clean_db(clean, noisy))
                sa.append(snr_vs_clean_db(clean, out))
                ab.append(speech_frame_accuracy(clean, noisy, sr))
                aa.append(speech_frame_accuracy(clean, out, sr))
            rows.append(
                BucketRow(
                    prof,
                    float(snr),
                    float(np.mean(sb)),
                    float(np.mean(sa)),
                    float(np.mean(ab)),
                    float(np.mean(aa)),
                    float(np.mean(sup)),
                )
            )
    worst_before = min(rows, key=lambda r: r.accuracy_before)
    worst_after = min(rows, key=lambda r: r.accuracy_after)
    best_after = max(rows, key=lambda r: r.accuracy_after)
    return {
        "metric": "speech_frame_accuracy (PROXY for recognition accuracy; not a WER)",
        "rows": rows,
        "worst_bucket_before": (worst_before.profile, worst_before.snr_db, worst_before.accuracy_before),
        "worst_bucket_after": (worst_after.profile, worst_after.snr_db, worst_after.accuracy_after),
        "worst_bucket_degradation_vs_best": best_after.accuracy_after - worst_after.accuracy_after,
        "buckets_that_got_worse": [
            (r.profile, r.snr_db, round(r.accuracy_change, 3)) for r in rows if r.accuracy_change < -0.02
        ],
    }
