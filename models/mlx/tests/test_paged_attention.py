from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from _attention_pages import PAGE, attention_pages, bf16  # noqa: E402

from resin_models_mlx.paged_attention import (  # noqa: E402
    PARTITION_TOKENS,
    paged_decode_attention,
)


@pytest.mark.parametrize(
    "heads,kv_heads,head_dim,lengths",
    [
        (16, 2, 256, [0]),
        (16, 2, 256, [511, 512, 513]),
        (4, 2, 64, [255, 256, 257]),  # Half-sized partitions below four rows.
        (4, 2, 64, [511, 512, 513, 1025]),
        (6, 1, 64, [127, 129]),  # Three query heads per group.
        (10, 2, 96, [3000, 17, 0, 2049]),
        (3, 3, 32, [1, 127, 128, 129]),
        (8, 2, 512, [PARTITION_TOKENS * 4 + 3]),
    ],
)
def test_paged_decode_matches_gathered_attention(heads, kv_heads, head_dim, lengths):
    _check_decode(heads, kv_heads, head_dim, lengths, layer=1)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal attention")
@pytest.mark.parametrize("layout", ["offset", "strided"])
def test_native_geometry_matches_gathered_sdpa_with_poisoned_pages(layout):
    _check_decode(8, 2, 256, [0, 129, 513, 8192], layer=2, layout=layout)


def _check_decode(heads, kv_heads, head_dim, lengths, *, layer, layout=None):
    rows = len(lengths)
    rng = np.random.default_rng(9101 if layout else rows + heads)
    (keys, values), (key, value), units, pages = attention_pages(
        rng,
        lengths,
        kv_heads,
        head_dim,
        layers=3,
        layer=layer,
        first_page=1 if layout else 0,
    )
    queries = bf16(rng.standard_normal((rows, heads, head_dim), np.float32))
    if layout:
        # Match the joined projection's nonzero V offset and row stride.
        padding = mx.full((rows, 4608), mx.nan, mx.bfloat16)
        value = mx.concatenate([padding, value.reshape(rows, -1)], axis=1)[:, 4608:]
        value = value.reshape(rows, kv_heads, head_dim)
        original = np.asarray(units.view(mx.uint16)).copy()
        if layout == "offset":
            units = mx.concatenate(
                [mx.full((1,), mx.nan, mx.bfloat16), units.reshape(-1)]
            )[1:].reshape(units.shape)
        else:
            units = mx.stack(
                [units, mx.full(units.shape, mx.nan, mx.bfloat16)], axis=-1
            )[..., 0]

    scale = head_dim**-0.5
    actual = paged_decode_attention(
        queries, key, value, units, pages, mx.array(lengths, mx.int32), layer, scale
    ).astype(mx.float32)
    expected = []
    for row, length in enumerate(lengths):
        expected.append(
            mx.fast.scaled_dot_product_attention(
                queries[row : row + 1, :, None],
                keys[row, : length + 1].transpose(1, 0, 2)[None],
                values[row, : length + 1].transpose(1, 0, 2)[None],
                scale=scale,
            )[0, :, 0]
        )
    result = np.asarray(actual)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(
        result, np.asarray(mx.stack(expected).astype(mx.float32)), atol=2e-2, rtol=2e-2
    )
    if layout:
        np.testing.assert_array_equal(np.asarray(units.view(mx.uint16)), original)


def test_rejects_head_dims_the_lanes_cannot_cover():
    queries = mx.zeros((1, 2, 80), dtype=mx.bfloat16)
    current = mx.zeros((1, 1, 80), dtype=mx.bfloat16)
    units = mx.zeros((1, 2, PAGE, 80), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="divisible by 32"):
        paged_decode_attention(
            queries,
            current,
            current,
            units,
            mx.zeros((1, 1), dtype=mx.int32),
            mx.array([0], dtype=mx.int32),
            0,
            1.0,
        )
