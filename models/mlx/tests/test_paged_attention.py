from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin_models_mlx.paged_attention import (  # noqa: E402
    PARTITION_TOKENS,
    paged_decode_attention,
)

PAGE = 128


def as_bf16(array: np.ndarray):
    return mx.array(array.astype(np.float32)).astype(mx.bfloat16)


def _reference(queries, keys, values, history, scale):
    """SDPA over each row's gathered history plus its current token."""

    rows = []
    for row, length in enumerate(history):
        k = mx.concatenate([keys[row][:length], keys[row][-1:]], axis=0)
        v = mx.concatenate([values[row][:length], values[row][-1:]], axis=0)
        rows.append(
            mx.fast.scaled_dot_product_attention(
                queries[row : row + 1, :, None, :],
                k.transpose(1, 0, 2)[None],
                v.transpose(1, 0, 2)[None],
                scale=scale,
            )[0, :, 0]
        )
    return mx.stack(rows)


@pytest.mark.parametrize(
    "lengths",
    [
        [0],
        [1, 127, 128, 129],
        [PARTITION_TOKENS - 1, PARTITION_TOKENS, PARTITION_TOKENS + 1],
        [3000, 17, 0, 2049],
    ],
)
@pytest.mark.parametrize("heads,kv_heads,head_dim", [(16, 2, 256), (4, 2, 64)])
def test_paged_decode_matches_gathered_attention(lengths, heads, kv_heads, head_dim):
    rng = np.random.default_rng(len(lengths) + heads)
    rows = len(lengths)
    layers, layer = 3, 1
    width = max(1, -(-max(lengths) // PAGE))
    unit_count = rows * width * layers + 7
    units = rng.standard_normal((unit_count, 2, PAGE, kv_heads * head_dim))
    # Poison every unit, then write real history only where rows own it, so a
    # read beyond a row's length produces NaN.
    units[:] = np.nan
    order = rng.permutation(rows * width) * layers
    pages = order.reshape(rows, width).astype(np.int32)
    keys = rng.standard_normal((rows, width * PAGE + 1, kv_heads, head_dim))
    values = rng.standard_normal((rows, width * PAGE + 1, kv_heads, head_dim))
    for row, length in enumerate(lengths):
        for position in range(length):
            unit = pages[row, position // PAGE] + layer
            units[unit, 0, position % PAGE] = keys[row, position].reshape(-1)
            units[unit, 1, position % PAGE] = values[row, position].reshape(-1)
    queries = rng.standard_normal((rows, heads, head_dim))
    scale = head_dim**-0.5

    actual = paged_decode_attention(
        as_bf16(queries),
        as_bf16(keys[:, -1]),
        as_bf16(values[:, -1]),
        as_bf16(units),
        mx.array(pages),
        mx.array(np.asarray(lengths, dtype=np.int32)),
        layer,
        scale,
    ).astype(mx.float32)
    expected = _reference(
        as_bf16(queries), as_bf16(keys), as_bf16(values), lengths, scale
    ).astype(mx.float32)

    assert not mx.isnan(actual).any().item()
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=2e-2, rtol=2e-2
    )
