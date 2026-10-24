"""Cumulative execution observations, independent of metric exporters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from resin._resin import Histogram, _HistogramAccumulator

FORWARD_CATEGORIES = ("prefill", "target_verify", "draft", "decode", "idle")
CUDA_GRAPH_MODES = (
    "decode_cuda_graph",
    "decode_none",
    "prefill_cuda_graph",
    "prefill_none",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunnerMetrics:
    """A nonblocking snapshot of cumulative observations for one physical rank.

    Forward categories are exclusive: prefill, decode, target_verify, draft,
    and idle (participating without local requests). Graph counts describe
    executions, not individual regions in a breakable graph; a pass counts as
    captured when any of its configured regions replayed. DP wait uses a bounded
    exponential histogram, independent of the exporter.
    """

    forward_seconds: Mapping[str, float] = field(default_factory=dict)
    cuda_graph_passes: Mapping[str, int] = field(default_factory=dict)
    speculative_drafts: int = 0
    speculative_accepted: int = 0
    speculative_rows: int = 0
    dp_sync: Histogram | None = None


class DpSyncMetrics:
    """Bounded host-side accumulator for coordinator collective wait time."""

    def __init__(self) -> None:
        self._histogram = _HistogramAccumulator()

    def observe(self, seconds: float) -> None:
        self._histogram.observe(seconds)

    def snapshot(self) -> Histogram:
        return self._histogram.snapshot()
