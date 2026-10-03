"""Appendix F channel buckets for live turns, and the rate at which each occurs.

Blueprint Appendix F splits the golden set, and the per-bucket evaluation
dashboard, by (among others) CHANNEL: clean_16k, narrowband_8k, noisy,
cross_talk. Several stories ask for a live rate "exported and a golden-set bucket
per Appendix F" (KCD-053 cross-talk, KCD-055 noise, KCD-057 input level). This is
the one place that maps what agent/audio_quality.py and agent/channel_quality.py
observed on a turn onto those names, so the live counters and the offline
evaluation use the SAME bucket vocabulary and can be compared.

A turn can fall in several buckets at once (a noisy narrowband turn is both), so
the counters are per bucket and a rate is bucket-count / turns-seen, not a
partition.
"""

from __future__ import annotations

from agent.channel_quality import (
    CHANNEL_CLEAN_16K,
    CHANNEL_CROSSTALK,
    CHANNEL_NARROWBAND_8K,
    CHANNEL_NOISY,
)

CHANNEL_BUCKETS = (CHANNEL_CLEAN_16K, CHANNEL_NARROWBAND_8K, CHANNEL_NOISY, CHANNEL_CROSSTALK)
# Additional live-only buckets this codebase reports beside the Appendix F set:
# input LEVEL bands (KCD-057). Not part of Appendix F; named so they cannot be
# mistaken for it.
LEVEL_BUCKET_PREFIX = "level_"

_ISSUE_TO_BUCKET = {"crosstalk": CHANNEL_CROSSTALK, "noisy": CHANNEL_NOISY}


def channel_buckets(audio_issues: list[str], channel_quality: str | None) -> list[str]:
    """The Appendix F channel buckets this turn belongs to, in a stable order."""
    out: list[str] = []
    if channel_quality in CHANNEL_BUCKETS:
        out.append(channel_quality)
    for issue in audio_issues or []:
        bucket = _ISSUE_TO_BUCKET.get(issue)
        if bucket and bucket not in out:
            out.append(bucket)
    return out


def bucket_rate(snapshot: dict, bucket: str, turns_seen: int) -> float:
    """Fraction of turns in `bucket`, from an OutcomeCounter snapshot
    ({bucket: {"turn:lang": count}}). 0.0 when nothing has been seen."""
    if turns_seen <= 0:
        return 0.0
    return sum(snapshot.get(bucket, {}).values()) / turns_seen
