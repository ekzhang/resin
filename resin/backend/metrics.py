"""Cumulative execution observations, independent of metric exporters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

DP_SYNC_BUCKETS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
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
    captured when any of its configured regions replayed. DP buckets are
    cumulative counts at ``DP_SYNC_BUCKETS``; the count includes +Inf.
    """

    forward_seconds: Mapping[str, float] = field(default_factory=dict)
    cuda_graph_passes: Mapping[str, int] = field(default_factory=dict)
    speculative_drafts: int = 0
    speculative_accepted: int = 0
    speculative_rows: int = 0
    dp_sync_seconds: float = 0.0
    dp_sync_count: int = 0
    dp_sync_buckets: tuple[int, ...] = (0,) * len(DP_SYNC_BUCKETS)

    @property
    def dp_sync_bounds(self) -> tuple[float, ...]:
        return DP_SYNC_BUCKETS


class DpSyncMetrics:
    """Bounded host-side accumulator for coordinator collective wait time."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.count = 0
        self.buckets = [0] * len(DP_SYNC_BUCKETS)

    def observe(self, seconds: float) -> None:
        self.seconds += seconds
        self.count += 1
        for index, bound in enumerate(DP_SYNC_BUCKETS):
            self.buckets[index] += seconds <= bound
