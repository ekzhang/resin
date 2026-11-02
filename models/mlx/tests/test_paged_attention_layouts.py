"""Real Qwen head geometry against independent gathered SDPA."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal attention")

from resin_models_mlx.paged_attention import paged_decode_attention  # noqa: E402


@pytest.mark.parametrize("layout", ["offset", "strided"])
def test_native_geometry_matches_gathered_sdpa_with_poisoned_pages(layout):
    lengths, heads, kv, dim, page, layer = [0, 129, 513, 8192], 8, 2, 256, 128, 2
    rows, blocks = len(lengths), -(-max(lengths) // page)
    rng = np.random.default_rng(9101)
    keys = rng.standard_normal((rows, max(lengths) + 1, kv, dim), dtype=np.float32)
    values = rng.standard_normal(keys.shape, dtype=np.float32)
    query = mx.array(rng.standard_normal((rows, heads, dim), dtype=np.float32)).astype(
        mx.bfloat16
    )
    key, value = (mx.array(x[:, -1]).astype(mx.bfloat16) for x in (keys, values))
    # Preserve the real joined projection's nonzero V offset and row stride.
    value = mx.concatenate(
        [mx.full((rows, 4608), mx.nan, mx.bfloat16), value.reshape(rows, -1)], axis=1
    )[:, 4608:].reshape(rows, kv, dim)
    counts = [-(-length // page) for length in lengths]
    order = (rng.permutation(sum(counts)) * 3 + 3).astype(np.int32)
    physical = np.full((len(order) * 3 + 7, 2, page, kv * dim), np.nan, np.float32)
    pages = np.zeros((rows, blocks), np.int32)
    first = 0
    for row, (length, count) in enumerate(zip(lengths, counts, strict=True)):
        pages[row, :count] = order[first : first + count]
        first += count
        for block in range(count):
            start, stop = block * page, min((block + 1) * page, length)
            for kind, source in enumerate((keys, values)):
                physical[pages[row, block] + layer, kind, : stop - start] = source[
                    row, start:stop
                ].reshape(-1, kv * dim)
    units = mx.array(physical).astype(mx.bfloat16)
    mx.eval(units, value)
    original = np.asarray(units.view(mx.uint16)).copy()
    if layout == "offset":
        units = mx.concatenate([mx.full((1,), mx.nan, mx.bfloat16), units.reshape(-1)])[
            1:
        ].reshape(units.shape)
    else:
        units = mx.stack([units, mx.full(units.shape, mx.nan, mx.bfloat16)], axis=-1)[
            ..., 0
        ]
    actual = paged_decode_attention(
        query,
        key,
        value,
        units,
        mx.array(pages),
        mx.array(lengths, mx.int32),
        layer,
        dim**-0.5,
    )
    expected = []
    for row, length in enumerate(lengths):
        k = mx.concatenate(
            [mx.array(keys[row, :length]).astype(mx.bfloat16), key[row : row + 1]]
        )
        v = mx.concatenate(
            [mx.array(values[row, :length]).astype(mx.bfloat16), value[row : row + 1]]
        )
        expected.append(
            mx.fast.scaled_dot_product_attention(
                query[row : row + 1, :, None],
                k.transpose(1, 0, 2)[None],
                v.transpose(1, 0, 2)[None],
                scale=dim**-0.5,
            )[0, :, 0]
        )
    result = np.asarray(actual.astype(mx.float32))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(
        result, np.asarray(mx.stack(expected).astype(mx.float32)), atol=2e-2, rtol=2e-2
    )
    np.testing.assert_array_equal(np.asarray(units.view(mx.uint16)), original)
