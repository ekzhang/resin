"""Inkling 2 paged decode attention against the dense reference attention."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from resin_models_mlx.inkling2.attention import dense_attention
from resin_models_mlx.inkling2.decode_attention import decode_attention

PAGE = 128


def _case(contexts, window, heads, kv_heads, dim, extent, seed):
    rng = np.random.default_rng(seed)
    rows = len(contexts)
    width = kv_heads * dim
    # Retained blocks: a window keeps only its suffix, starting at a nonzero block.
    firsts = [max(0, c - (window - 1)) if window else 0 for c in contexts]
    starts = [f // PAGE for f in firsts]
    blocks = [max(1, -(-c // PAGE) - s) for c, s in zip(contexts, starts, strict=True)]
    table = max(blocks)
    units_count = rows * table + 3
    # NaN everywhere a kernel must never read.
    units = np.full((units_count, 2, PAGE, width), np.nan, dtype=np.float32)
    pages = np.zeros((rows, table), dtype=np.int32)
    order = rng.permutation(units_count - 1) + 1
    history_k, history_v = [], []
    used = 0
    for row, context in enumerate(contexts):
        keys = rng.standard_normal((context, kv_heads, dim)).astype(np.float32)
        values = rng.standard_normal((context, kv_heads, dim)).astype(np.float32)
        history_k.append(keys)
        history_v.append(values)
        for j in range(blocks[row]):
            unit = int(order[used])
            used += 1
            pages[row, j] = unit
            block = starts[row] + j
            lo, hi = block * PAGE, min((block + 1) * PAGE, context)
            if hi > lo:
                units[unit, 0, : hi - lo] = keys[lo:hi].reshape(hi - lo, width)
                units[unit, 1, : hi - lo] = values[lo:hi].reshape(hi - lo, width)
    queries = rng.standard_normal((rows, heads, dim)).astype(np.float32)
    current_k = rng.standard_normal((rows, kv_heads, dim)).astype(np.float32)
    current_v = rng.standard_normal((rows, kv_heads, dim)).astype(np.float32)
    bias = (rng.standard_normal((rows, heads, extent)) * 2).astype(np.float32)
    meta = np.array(
        [[c, f, s] for c, f, s in zip(contexts, firsts, starts, strict=True)],
        dtype=np.int32,
    )
    return dict(
        units=units,
        pages=pages,
        meta=meta,
        queries=queries,
        current_k=current_k,
        current_v=current_v,
        bias=bias,
        history_k=history_k,
        history_v=history_v,
    )


def _dense(case, contexts, window, scale):
    bf16 = lambda x: mx.array(x).astype(mx.bfloat16)  # noqa: E731
    rows = len(contexts)
    longest = max(contexts) + 1
    kv_heads, dim = case["current_k"].shape[1:]
    keys = np.zeros((rows, longest, kv_heads, dim), np.float32)
    values = np.zeros_like(keys)
    valid = np.zeros((rows, longest), bool)
    for row, context in enumerate(contexts):
        keys[row, :context] = case["history_k"][row]
        values[row, :context] = case["history_v"][row]
        keys[row, context] = case["current_k"][row]
        values[row, context] = case["current_v"][row]
        valid[row, : context + 1] = True
    positions = np.array(contexts, np.int32)[:, None]
    return dense_attention(
        bf16(case["queries"])[:, None],
        bf16(keys),
        bf16(values),
        mx.broadcast_to(mx.arange(longest)[None], (rows, longest)).astype(mx.int32),
        mx.array(valid),
        mx.array(positions),
        bf16(case["bias"])[:, None],
        scale,
        window,
    )


@pytest.mark.parametrize(
    "contexts,window",
    [
        ([0], None),
        ([1, 127, 128, 129], None),
        ([700, 3, 1500], None),
        ([1023, 1024, 1025, 2000], 1024),
        ([5, 300], 12),
    ],
)
@pytest.mark.parametrize("geometry", [(24, 8, 128, 1024), (4, 2, 32, 12)])
def test_decode_attention_matches_dense(contexts, window, geometry):
    heads, kv_heads, dim, extent = geometry
    if window is not None:
        window = min(window, extent) if extent < 1024 else window
    scale = dim**-0.5
    case = _case(contexts, window, heads, kv_heads, dim, extent, seed=sum(contexts))
    bf16 = lambda x: mx.array(x).astype(mx.bfloat16)  # noqa: E731
    firsts = case["meta"][:, 1]
    output, lse = decode_attention(
        bf16(case["queries"]),
        bf16(case["current_k"]),
        bf16(case["current_v"]),
        bf16(case["units"]),
        mx.array(case["pages"]),
        mx.array(case["meta"]),
        bf16(case["bias"]),
        0,
        scale,
        int(max(c + 1 - f for c, f in zip(contexts, firsts, strict=True))),
    )
    expected, expected_lse = _dense(case, contexts, window, scale)
    np.testing.assert_allclose(
        np.array(output), np.array(expected[:, 0]), rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(
        np.array(lse), np.array(expected_lse[:, 0]), rtol=2e-4, atol=2e-4
    )
