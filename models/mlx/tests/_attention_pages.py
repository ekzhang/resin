"""Synthetic dense K/V and independently packed, NaN-poisoned pages for decode."""

import mlx.core as mx
import numpy as np

PAGE = 128


def bf16(values):
    return mx.array(values).astype(mx.bfloat16)


def attention_pages(
    rng, lengths, kv_heads, dim, *, starts=None, layers=1, layer=0, first_page=1
):
    """Scatter history into shuffled pages, leaving out each row's current token."""
    lengths = np.asarray(lengths, dtype=np.int32)
    if starts is None:
        starts = np.zeros_like(lengths)
    counts = np.maximum(1, (lengths + PAGE - 1) // PAGE - starts)
    rows = len(lengths)
    dense = rng.standard_normal((2, rows, max(lengths) + 1, kv_heads, dim), np.float32)
    order = (rng.permutation(counts.sum()) + first_page) * layers
    units = np.full(
        ((len(order) + 1) * layers, 2, PAGE, kv_heads * dim), np.nan, np.float32
    )
    pages = np.zeros((rows, max(counts)), np.int32)
    offset = 0
    for row, count in enumerate(counts):
        pages[row, :count] = order[offset : offset + count]
        offset += count
        for block, base in enumerate(pages[row, :count]):
            start = (starts[row] + block) * PAGE
            stop = min(start + PAGE, lengths[row])
            units[base + layer, :, : stop - start] = dense[:, row, start:stop].reshape(
                2, stop - start, kv_heads * dim
            )
    current = dense[:, np.arange(rows), lengths]
    return bf16(dense), bf16(current), bf16(units), mx.array(pages)


def dense_reference(q, keys, values, positions, bias, scale, window):
    """Unsliced attention over complete sequences indexed by absolute position."""
    batch, tokens, heads, dim = q.shape
    length, kv_heads = keys.shape[1:3]
    groups = heads // kv_heads
    query = q.astype(mx.float32).reshape(batch, tokens, kv_heads, groups, dim)
    scores = (
        mx.einsum("btkgd,blkd->btkgl", query, keys.astype(mx.float32)).reshape(
            batch, tokens, heads, length
        )
        * scale
    )
    distance = positions[:, :, None] - mx.arange(length)
    extent = bias.shape[-1]
    index = mx.broadcast_to(mx.clip(distance, 0, extent - 1)[:, :, None], scores.shape)
    relative = mx.take_along_axis(bias.astype(mx.float32), index, axis=-1)
    scores = scores + mx.where((distance < extent)[:, :, None], relative, 0)
    visible = distance >= 0
    if window is not None:
        visible = visible & (distance < window)
    scores = mx.where(visible[:, :, None], scores, -mx.inf)
    lse = mx.logsumexp(scores, axis=-1)
    weights = mx.exp(scores - lse[..., None]).reshape(
        batch, tokens, kv_heads, groups, length
    )
    output = mx.einsum("btkgl,blkd->btkgd", weights, values.astype(mx.float32))
    return output.reshape(q.shape), lse
