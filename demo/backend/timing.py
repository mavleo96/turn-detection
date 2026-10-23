"""Audio-clock timing: map stream sample positions to the server time that audio arrived.

All times are `time.monotonic_ns()` on the server. A stream sample index is counted from the first
sample of the session (16 kHz). The browser sends fixed 512-sample frames with a sequence number,
so frame `seq` covers samples [seq*512, (seq+1)*512).
"""

import math

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512


def ns_to_ms(ns: int | float) -> float:
    return ns / 1e6


class FrameClock:
    """Arrival time of every received frame, indexed by sequence number."""

    def __init__(self):
        self._arrivals: list[int] = []

    def record(self, seq: int, t_ns: int) -> None:
        if seq != len(self._arrivals):
            raise ValueError(f"frame seq {seq} out of order (expected {len(self._arrivals)})")
        self._arrivals.append(t_ns)

    @property
    def num_frames(self) -> int:
        return len(self._arrivals)

    def arrival_at(self, sample_end: int) -> int:
        """Time at which all audio before `sample_end` (exclusive) had arrived.

        Positions beyond the received audio (e.g. zero padding added at a flush) map to the last frame.
        """
        if not self._arrivals:
            raise ValueError("no frames received yet")
        seq = max(0, math.ceil(sample_end / FRAME_SAMPLES) - 1)
        return self._arrivals[min(seq, len(self._arrivals) - 1)]


def summarize(values_ms: list[float]) -> dict:
    """Count, mean, p50, p95, min and max of a list of latencies (ms)."""
    if not values_ms:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    a = np.asarray(values_ms, dtype=float)
    return {
        "count": int(a.size),
        "mean": round(float(a.mean()), 1),
        "p50": round(float(np.percentile(a, 50)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
        "min": round(float(a.min()), 1),
        "max": round(float(a.max()), 1),
    }
