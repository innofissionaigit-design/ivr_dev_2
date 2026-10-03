"""KCD-469: "time to first audio and turn latency are reported per
language and per mixture bucket... a gap beyond a stated margin is a
defect with an owner rather than a characteristic of the segment."

agent/admission.py already tracks turn latency, but only in aggregate
(its own p50/p95 feed latency_shed, a single process-wide signal by
design -- see that module's docstring). This is the PER-LANGUAGE
breakdown that aggregate cannot show: whether Hindi/English turns are
systematically slower than Bengali's, not just whether the process as a
whole is currently over budget.
"""

from __future__ import annotations

import collections
import threading

_WINDOW = 200


def _percentile(values, q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


class PerLanguageLatency:
    def __init__(self, window: int = _WINDOW):
        self._lock = threading.Lock()
        self._by_lang: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=window))

    def record(self, lang: str, seconds: float) -> None:
        with self._lock:
            self._by_lang[lang].append(float(seconds))

    def snapshot(self) -> dict:
        with self._lock:
            out = {}
            for lang, samples in self._by_lang.items():
                xs = list(samples)
                out[lang] = {
                    "samples": len(xs),
                    "p50_s": round(_percentile(xs, 0.50), 3) if xs else None,
                    "p95_s": round(_percentile(xs, 0.95), 3) if xs else None,
                }
            return out


turn_latency_by_language = PerLanguageLatency()
