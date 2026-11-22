"""Inkling 2 decode geometry and FP32 sink statistics for paged attention."""

from __future__ import annotations

import mlx.core as mx

from resin_models_mlx.inkling2.kernels import constant, pad_to_device
from resin_models_mlx.paged_attention import _merge, _paged_partitions

PARTITION_TOKENS = 256


def decode_attention(
    queries: mx.array,
    current_keys: mx.array,
    current_values: mx.array,
    units: mx.array,
    pages: mx.array,
    meta: mx.array,
    bias: mx.array,
    layer: int,
    scale: float,
    longest: int,
) -> tuple[mx.array, mx.array]:
    """One new token per row over paged history plus itself.

    ``queries`` is ``[rows, heads, D]``; current keys/values ``[rows, KV, D]``;
    ``units`` the arena as ``[units, 2, page, KV * D]``; ``pages[row, j]`` the
    visit-zero unit of the row's ``j``th retained block, offset by ``layer``
    units; ``meta[row] = (context, first visible position, first block)``;
    ``bias`` ``[rows, heads, extent]``. ``longest`` bounds ``context + 1 -
    first`` over rows and sets the partition count. Returns FP32
    ``[rows, heads, D]`` outputs and ``[rows, heads]`` log-sum-exps.
    """
    _, heads, dim = queries.shape
    kv_heads = current_keys.shape[1]
    group = heads // kv_heads
    if dim % 32 or heads % kv_heads:
        raise ValueError("unsupported Inkling 2 decode attention geometry")
    partitions = max(1, -(-longest // PARTITION_TOKENS))
    partials = _paged_partitions(
        queries,
        current_keys,
        current_values,
        units,
        pad_to_device(pages),
        pad_to_device(meta),
        constant((pages.shape[1], partitions, layer, bias.shape[-1]), "int32"),
        constant((float(scale),) * 8, "float32"),
        partitions=partitions,
        part=PARTITION_TOKENS,
        group=group,
        prefetch=1,
        bias=bias,
    )
    return _merge(*partials, return_lse=True)


__all__ = ["PARTITION_TOKENS", "decode_attention"]
