"""Inkling 2 paged decode attention against the dense reference attention."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from _attention_pages import PAGE, attention_pages, bf16, dense_reference

from resin_models_mlx.inkling2.attention import decode_attention


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
    rows = len(contexts)
    rng = np.random.default_rng(sum(contexts))
    lengths = np.array(contexts, np.int32)
    firsts = (
        np.zeros_like(lengths)
        if window is None
        else np.maximum(0, lengths - window + 1)
    )
    starts = firsts // PAGE
    (keys, values), (key, value), units, pages = attention_pages(
        rng, contexts, kv_heads, dim, starts=starts
    )
    queries = bf16(rng.standard_normal((rows, heads, dim), np.float32))
    bias = bf16(2 * rng.standard_normal((rows, heads, extent), np.float32))
    scale = dim**-0.5
    meta = mx.array(np.stack([lengths, firsts, starts], axis=1))
    output, lse = decode_attention(
        queries,
        key,
        value,
        units,
        pages,
        meta,
        bias,
        0,
        scale,
        int(max(lengths + 1 - firsts)),
    )
    positions = mx.array(lengths)[:, None]
    expected, expected_lse = dense_reference(
        queries[:, None],
        keys,
        values,
        positions,
        bias[:, None],
        scale,
        window,
    )
    np.testing.assert_allclose(
        np.array(output), np.array(expected[:, 0]), rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(
        np.array(lse), np.array(expected_lse[:, 0]), rtol=2e-4, atol=2e-4
    )
