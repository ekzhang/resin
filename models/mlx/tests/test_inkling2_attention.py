"""Inkling 2 paged attention against an independent dense reference."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from _attention_pages import PAGE, attention_pages, bf16, dense_reference

from resin_models_mlx.inkling2.attention import decode_attention, paged_prefill
from resin_models_mlx.metal import device_array


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


@pytest.mark.parametrize(("window", "dim"), [(None, 128), (147, 128), (147, 269)])
def test_paged_prefill_matches_dense_across_page_and_tile_boundaries(window, dim):
    context, tokens, layer = 3 * PAGE + 5, 35, 2
    heads, kv_heads = 6, 2
    scale = dim**-0.5
    first_visible = 0 if window is None else max(0, context - window + 1)
    first_block = first_visible // PAGE
    meta = device_array([[context, first_visible, first_block]])
    rng = np.random.default_rng(0)
    (past_k, past_v), _, units, pages = attention_pages(
        rng,
        [context],
        kv_heads,
        dim,
        starts=np.array([first_block]),
        layers=3,
        layer=layer,
    )
    k, v = bf16(rng.standard_normal((2, 1, tokens, kv_heads, dim), np.float32))
    q = bf16(rng.standard_normal((1, tokens, heads, dim), np.float32))
    bias = bf16(rng.standard_normal((1, tokens, heads, 21), np.float32))
    actual = paged_prefill(q, k, v, bias, units, pages, meta, layer, scale, window)
    expected = dense_reference(
        q,
        mx.concatenate([past_k[:, :context], k], axis=1),
        mx.concatenate([past_v[:, :context], v], axis=1),
        mx.arange(context, context + tokens)[None],
        bias,
        scale,
        window,
    )
    for output, reference in zip(actual, expected, strict=True):
        assert output.dtype == mx.float32 and output.shape == reference.shape
        np.testing.assert_allclose(
            np.asarray(output), np.asarray(reference), rtol=2e-5, atol=2e-5
        )
