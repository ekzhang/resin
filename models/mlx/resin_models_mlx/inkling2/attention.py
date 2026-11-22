"""Inkling 2 attention over gathered history, with relative bias and a window.

Logits are ``q·k / sqrt(d) + bias[q_pos - k_pos]`` in FP32, where the bias
applies only within its extent. Local visits additionally drop keys more than
``window - 1`` positions back. The softmax output and its log-sum-exp stay
FP32 for the sink and gate.
"""

from __future__ import annotations

import mlx.core as mx

F32 = mx.float32
# Bound each block's FP32 logits (and gathered bias) to about this many bytes.
_BLOCK_BYTES = 256 * 1024**2
# Query block size for banded windows: each reads window + block keys.
_BAND_QUERIES = 512


def dense_attention(
    q: mx.array,
    keys: mx.array,
    values: mx.array,
    key_positions: mx.array,
    key_valid: mx.array,
    positions: mx.array,
    bias: mx.array,
    scale: float,
    window: int | None,
    contiguous: tuple[int, int] | None = None,
) -> tuple[mx.array, mx.array]:
    """Attend ``q [B, T, H, D]`` to ``keys/values [B, L, KV, D]``.

    ``key_positions``/``key_valid`` are ``[B, L]``; ``positions`` is ``[B, T]``;
    ``bias`` is ``[B, T, H, E]``. Returns FP32 ``[B, T, H, D]`` and ``[B, T, H]``.

    ``contiguous = (first key position, first query position)`` states that
    key ``i`` is at position ``first + i`` and query ``t`` at the second plus
    ``t`` in every row. Each query block then reads only keys it can see: its
    causal prefix, and for a window only the band behind it.
    """
    batch, tokens, heads, dim = q.shape
    length, kv_heads = keys.shape[1], keys.shape[2]
    groups = heads // kv_heads
    extent = bias.shape[-1]
    k32 = keys.astype(F32)
    # Masked probabilities are zero, but zero times a stale NaN is still NaN.
    v32 = mx.where(key_valid[:, :, None, None], values, 0).astype(F32)
    span_keys = (
        length if window is None or contiguous is None else window + _BAND_QUERIES
    )
    block = max(1, min(tokens, _BLOCK_BYTES // max(1, 4 * batch * heads * span_keys)))
    if window is not None and contiguous is not None:
        block = min(block, _BAND_QUERIES)
    outputs, statistics = [], []
    for start in range(0, tokens, block):
        stop = min(tokens, start + block)
        span = stop - start
        lo, hi = 0, length
        if contiguous is not None:
            first_key, first_query = contiguous
            hi = min(length, first_query + stop - first_key)
            if window is not None:
                lo = max(0, first_query + start - (window - 1) - first_key)
        keys_block, values_block = k32[:, lo:hi], v32[:, lo:hi]
        block_positions, block_valid = key_positions[:, lo:hi], key_valid[:, lo:hi]
        width = hi - lo
        query = q[:, start:stop].astype(F32).reshape(batch, span, kv_heads, groups, dim)
        logits = mx.einsum("btkgd,blkd->btkgl", query, keys_block).reshape(
            batch, span, heads, width
        )
        logits = logits * scale
        distance = positions[:, start:stop, None] - block_positions[:, None, :]
        index = mx.clip(distance, 0, extent - 1)
        gathered = mx.take_along_axis(
            bias[:, start:stop].astype(F32),
            mx.broadcast_to(index[:, :, None, :], (batch, span, heads, width)),
            axis=-1,
        )
        inside = (distance >= 0) & (distance < extent)
        logits = logits + mx.where(inside[:, :, None, :], gathered, 0.0)
        mask = block_valid[:, None, :] & (distance >= 0)
        if window is not None:
            mask = mask & (distance <= window - 1)
        logits = mx.where(mask[:, :, None, :], logits, -mx.inf)
        lse = mx.logsumexp(logits, axis=-1)
        probabilities = mx.exp(logits - lse[..., None])
        output = mx.einsum(
            "btkgl,blkd->btkgd",
            probabilities.reshape(batch, span, kv_heads, groups, width),
            values_block,
        ).reshape(batch, span, heads, dim)
        outputs.append(output)
        statistics.append(lse)
    if len(outputs) == 1:
        return outputs[0], statistics[0]
    return mx.concatenate(outputs, axis=1), mx.concatenate(statistics, axis=1)


__all__ = ["dense_attention"]
