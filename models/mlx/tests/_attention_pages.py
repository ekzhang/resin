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
