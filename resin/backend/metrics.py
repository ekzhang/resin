"""Cumulative execution observations, independent of metric exporters."""

from __future__ import annotations

from resin._resin import RunnerMetrics as RunnerMetrics
from resin._resin import _HistogramAccumulator

FORWARD_CATEGORIES = ("prefill", "target_verify", "draft", "decode", "idle")
CUDA_GRAPH_MODES = (
    "decode_cuda_graph",
    "decode_none",
    "prefill_cuda_graph",
    "prefill_none",
)


class DpSyncMetrics:
    """Bounded host-side accumulator for coordinator collective wait time."""

    def __init__(self) -> None:
        self._histogram = _HistogramAccumulator()

    def observe(self, seconds: float) -> None:
        self._histogram.observe(seconds)

    def attach(self, snapshot: RunnerMetrics) -> RunnerMetrics:
        return self._histogram.attach(snapshot)
