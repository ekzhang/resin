from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.gated_delta import (  # noqa: E402
    normalize_qk as reference_normalize_qk,
)

from resin_models_mlx.qwen3_5.normalization import normalize_qk  # noqa: E402


def _assert_equal_bits(actual, expected):
    assert actual.dtype == expected.dtype
    bits = mx.uint16 if actual.dtype == mx.bfloat16 else mx.uint32
    np.testing.assert_array_equal(
        np.array(actual.view(bits)), np.array(expected.view(bits))
    )


@pytest.mark.parametrize(
    "width,dtype,layout,batch",
    [
        (8, "float32", "contiguous", 1),
        (16, "bfloat16", "inner_stride", 4),
        (32, "float32", "qkv", 2),
        (64, "bfloat16", "contiguous", 1),
        (128, "bfloat16", "qkv", 2),
        (128, "float32", "inner_stride", 4),
        (256, "bfloat16", "inner_stride", 4),
        (256, "float32", "contiguous", 1),
    ],
)
def test_qk_normalization_matches_independent_reference(width, dtype, layout, batch):
    rng = np.random.default_rng(901 + width)
    shape = (batch, 4, 2, width)
    query = rng.normal(size=shape).astype(np.float32)
    key = rng.normal(size=shape).astype(np.float32)
    # Cover ordinary rows, signed zero, tiny values and overflowing sums of
    # squares. Comparisons include the sign of zeros after normalization.
    query[:, 1] = 0
    key[:, 1] = -0.0
    query[:, 2] *= 1e-20
    key[:, 2] *= 1e-20
    query[:, 3] *= 1e20
    key[:, 3] *= 1e20
    selected = getattr(mx, dtype)
    if layout == "qkv":
        # Q/K are slices of one projection, with a nonzero offset for K and
        # holes between token rows. Poison the unused V region.
        packed = np.concatenate(
            [
                query.reshape(batch, 4, -1),
                key.reshape(batch, 4, -1),
                np.full((batch, 4, 2 * width), np.nan, np.float32),
            ],
            axis=-1,
        )
        packed = mx.array(packed).astype(selected)
        q = packed[..., : 2 * width].reshape(shape)
        k = packed[..., 2 * width : 4 * width].reshape(shape)
    elif layout == "inner_stride":
        q_data = np.full((*shape[:-1], 2 * width), np.nan, np.float32)
        k_data = q_data.copy()
        q_data[..., ::2], k_data[..., ::2] = query, key
        q = mx.array(q_data).astype(selected)[..., ::2]
        k = mx.array(k_data).astype(selected)[..., ::2]
    else:
        q, k = mx.array(query).astype(selected), mx.array(key).astype(selected)
    inv_scale = width**-0.5
    expected = reference_normalize_qk(q, k, inv_scale=inv_scale, eps=1e-6)
    actual = normalize_qk(q, k, inv_scale=inv_scale, eps=1e-6)
    for result, reference in zip(actual, expected, strict=True):
        _assert_equal_bits(result, reference)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_qk_normalization_matches_reference_for_whole_prefill_chunk(dtype):
    rng = np.random.default_rng(902)
    packed = mx.array(
        rng.normal(size=(1, 4096, 3 * 2 * 128)).astype(np.float32)
    ).astype(getattr(mx, dtype))
    q = packed[..., :256].reshape(1, 4096, 2, 128)
    k = packed[..., 256:512].reshape(1, 4096, 2, 128)
    expected = reference_normalize_qk(q, k, inv_scale=128**-0.5, eps=1e-6)
    actual = normalize_qk(q, k, inv_scale=128**-0.5, eps=1e-6)
    for result, reference in zip(actual, expected, strict=True):
        _assert_equal_bits(result, reference)
