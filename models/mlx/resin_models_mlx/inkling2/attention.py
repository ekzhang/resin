"""Inkling 2 attention over gathered history, with relative bias and a window.

Logits are ``q·k / sqrt(d) + bias[q_pos - k_pos]`` in FP32, where the bias
applies only within its extent. Local visits additionally drop keys more than
``window - 1`` positions back. The softmax output and its log-sum-exp stay
FP32 for the sink and gate.
"""

from __future__ import annotations

import mlx.core as mx

from resin_models_mlx.paged_attention import _merge, _paged_partitions

F32 = mx.float32
# Bound each block's FP32 logits (and gathered bias) to about this many bytes.
_BLOCK_BYTES = 256 * 1024**2
# Query block size for banded windows: each reads window + block keys.
_BAND_QUERIES = 512


def dense_attention(
    q: mx.array,
    keys: mx.array,
    values: mx.array,
    bias: mx.array,
    scale: float,
    window: int | None,
) -> tuple[mx.array, mx.array]:
    """Attend a prefill chunk to contiguous retained K/V followed by its own K/V.

    Queries are the final T tokens of the L keys. Each block scores its causal
    prefix, restricted to the local band when a window is set. Return FP32
    outputs [B, T, H, D] and log-sum-exps [B, T, H] for the sink and gate.
    """
    batch, tokens, heads, dim = q.shape
    length, kv_heads = keys.shape[1:3]
    history = length - tokens
    groups = heads // kv_heads
    extent = bias.shape[-1]
    keys, values = keys.astype(F32), values.astype(F32)
    span_keys = length if window is None else min(length, window + _BAND_QUERIES)
    block = max(1, _BLOCK_BYTES // (4 * batch * heads * span_keys))
    if window is not None:
        block = min(block, _BAND_QUERIES)
    outputs, statistics = [], []
    for start in range(0, tokens, block):
        stop = min(tokens, start + block)
        span = stop - start
        hi = history + stop
        lo = 0 if window is None else max(0, history + start - window + 1)
        query = q[:, start:stop].astype(F32).reshape(batch, span, kv_heads, groups, dim)
        logits = (
            mx.einsum("btkgd,blkd->btkgl", query, keys[:, lo:hi]).reshape(
                batch, span, heads, hi - lo
            )
            * scale
        )
        # Distant global keys are visible and unbiased for this entire block.
        # Only gather bias and construct the causal mask for the remainder.
        split = max(0, history + start - extent + 1) if window is None else lo
        distance = mx.arange(history + start, hi)[:, None] - mx.arange(split, hi)
        index = mx.clip(distance, 0, extent - 1)
        gathered = mx.take_along_axis(
            bias[:, start:stop],
            mx.broadcast_to(index[None, :, None], (batch, span, heads, hi - split)),
            axis=-1,
        )
        recent = logits[..., split - lo :] + mx.where(
            distance[:, None] < extent, gathered, 0
        )
        visible = distance >= 0
        if window is not None:
            visible = visible & (distance < window)
        recent = mx.where(visible[:, None], recent, -mx.inf)
        logits = (
            mx.concatenate([logits[..., : split - lo], recent], axis=-1)
            if split > lo
            else recent
        )
        lse = mx.logsumexp(logits, axis=-1)
        probabilities = mx.exp(logits - lse[..., None])
        output = mx.einsum(
            "btkgl,blkd->btkgd",
            probabilities.reshape(batch, span, kv_heads, groups, hi - lo),
            values[:, lo:hi],
        ).reshape(batch, span, heads, dim)
        outputs.append(output)
        statistics.append(lse)
    if len(outputs) == 1:
        return outputs[0], statistics[0]
    return mx.concatenate(outputs, axis=1), mx.concatenate(statistics, axis=1)


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
    partials = _paged_partitions(
        queries,
        current_keys,
        current_values,
        units,
        pages,
        meta,
        layer,
        scale,
        bias=bias,
        longest=longest,
    )
    return _merge(*partials, return_lse=True)


__all__ = ["decode_attention", "dense_attention"]
