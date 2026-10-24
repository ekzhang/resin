"""Cumulative execution observations, independent of metric exporters."""

from resin._resin import RunnerMetrics as RunnerMetrics

FORWARD_CATEGORIES = ("prefill", "target_verify", "draft", "decode", "idle")
CUDA_GRAPH_MODES = (
    "decode_cuda_graph",
    "decode_none",
    "prefill_cuda_graph",
    "prefill_none",
)
