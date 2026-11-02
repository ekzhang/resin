"""Actual Qwen3.5 decode layouts against independent dense attention."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin_models_mlx.paged_attention import paged_decode_attention  # noqa: E402

_PAGE = 128
_HEADS, _KV, _DIM = 8, 2, 256


def _hash(array):
    return hashlib.sha256(np.asarray(array.view(mx.uint16)).tobytes()).digest()


@pytest.mark.parametrize(
    "lengths,extra_pages,layout",
    [
        ([0], 0, "contiguous"),
        ([127], 0, "contiguous"),
        ([128], 0, "contiguous"),
        ([129], 0, "contiguous"),
        ([511], 0, "contiguous"),
        ([512], 0, "contiguous"),
        ([513], 0, "contiguous"),
        ([127, 129], 2, "contiguous"),
        ([0, 513], 1, "contiguous"),
        ([0, 127, 128, 129], 2, "contiguous"),
        ([511, 512, 513, 1031], 0, "contiguous"),
        ([8192], 1, "contiguous"),
        ([8191, 8193], 0, "contiguous"),
        ([8192, 17, 0, 7937], 3, "contiguous"),
        ([513], 1, "strided"),
        ([129], 0, "offset"),
        ([3], 0, "thread"),
    ],
)
def test_decode_attention_preserves_paged_and_projection_layouts(
    lengths, extra_pages, layout
):
    rows, layers, layer = len(lengths), 6, 3
    blocks = max(1, -(-max(lengths) // _PAGE)) + extra_pages
    rng = np.random.default_rng(9101 + rows + max(lengths))
    keys = mx.array(
        rng.standard_normal((rows, max(lengths) + 1, _KV, _DIM), dtype=np.float32)
    ).astype(mx.bfloat16)
    values = mx.array(rng.standard_normal(keys.shape, dtype=np.float32)).astype(
        mx.bfloat16
    )
    query = mx.array(
        rng.standard_normal((rows, _HEADS, _DIM), dtype=np.float32)
    ).astype(mx.bfloat16)
    query = mx.concatenate(
        [mx.full((13,), mx.nan, dtype=query.dtype), query.reshape(-1)]
    )[13:].reshape(rows, _HEADS, _DIM)
    key = mx.concatenate(
        [mx.full((7,), mx.nan, dtype=keys.dtype), keys[:, -1].reshape(-1)]
    )[7:].reshape(rows, _KV, _DIM)
    # The loaded model's joined Q/K/V projection leaves this exact V offset
    # and row stride for B2/B4, despite the logical array being [B,2,256].
    value = mx.concatenate(
        [
            mx.full((rows, 4608), mx.nan, dtype=values.dtype),
            values[:, -1].reshape(rows, -1),
        ],
        axis=1,
    )[:, 4608:].reshape(rows, _KV, _DIM)
    order = (rng.permutation(rows * blocks).reshape(rows, blocks) + 2) * layers
    pages = np.zeros((rows, blocks), dtype=np.int32)
    physical = np.full(
        ((rows * blocks + 2) * layers + 7, 2, _PAGE, _KV * _DIM),
        np.nan,
        dtype=np.float32,
    )
    dense_keys, dense_values = (
        np.array(keys.astype(mx.float32)),
        np.array(values.astype(mx.float32)),
    )
    for row, length in enumerate(lengths):
        for block in range(-(-length // _PAGE)):
            first, end = block * _PAGE, min((block + 1) * _PAGE, length)
            pages[row, block] = order[row, block]
            unit = int(pages[row, block]) + layer
            physical[unit, 0, : end - first] = dense_keys[row, first:end].reshape(
                -1, _KV * _DIM
            )
            physical[unit, 1, : end - first] = dense_values[row, first:end].reshape(
                -1, _KV * _DIM
            )
    units = mx.array(physical).astype(mx.bfloat16)
    mx.eval(query, key, value, units)
    original = [_hash(x) for x in (query, key, value, units)]
    if layout == "strided":
        units = mx.stack([units, mx.full(units.shape, mx.nan, mx.bfloat16)], axis=-1)[
            ..., 0
        ]
    elif layout == "offset":
        units = mx.concatenate([mx.full((1,), mx.nan, mx.bfloat16), units.reshape(-1)])[
            1:
        ].reshape(units.shape)
    args = (
        query,
        key,
        value,
        units,
        mx.array(pages),
        mx.array(lengths, dtype=mx.int32),
        layer,
        _DIM**-0.5,
    )

    def run():
        result = paged_decode_attention(*args)
        mx.eval(result)
        return result

    if layout == "thread":
        with ThreadPoolExecutor(max_workers=1) as executor:
            actual = executor.submit(run).result(timeout=30)
    else:
        actual = run()
    native, explicit = [], []
    for row, length in enumerate(lengths):
        k = mx.concatenate([keys[row, :length], key[row : row + 1]])
        v = mx.concatenate([values[row, :length], value[row : row + 1]])
        native.append(
            mx.fast.scaled_dot_product_attention(
                query[row : row + 1, :, None],
                k.transpose(1, 0, 2)[None],
                v.transpose(1, 0, 2)[None],
                scale=_DIM**-0.5,
            )[0, :, 0]
        )
        q32 = query[row].astype(mx.float32).reshape(_KV, _HEADS // _KV, _DIM)
        scores = q32 @ k.astype(mx.float32).transpose(1, 2, 0) * _DIM**-0.5
        explicit.append(
            (
                mx.softmax(scores, axis=-1) @ v.astype(mx.float32).transpose(1, 0, 2)
            ).reshape(_HEADS, _DIM)
        )
    result = np.array(actual.astype(mx.float32))
    assert np.isfinite(result).all()
    for reference in (mx.stack(native), mx.stack(explicit)):
        expected = np.array(reference.astype(mx.float32))
        np.testing.assert_allclose(result, expected, atol=2e-2, rtol=2e-2)
    assert original == [_hash(x) for x in (query, key, value, units)]
