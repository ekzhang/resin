"""Fused decode convolution and its carry against ordinary MLX operations."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from resin_models_mlx.qwen3_5.fused_conv import conv4_silu_qkv  # noqa: E402


@pytest.mark.parametrize("dtype,tokens", [("bfloat16", 1), ("float32", 3)])
def test_conv_and_carry_match_native_across_steps(dtype, tokens):
    rng = np.random.default_rng(71003)
    dtype = getattr(mx, dtype)
    rows, heads, dim = 4, 16, 128
    width, channels = heads * dim, 3 * heads * dim
    projected = mx.array(
        rng.normal(size=(rows, 4 * tokens, channels + 2080)).astype(np.float32)
    ).astype(dtype)
    weight = mx.array(
        rng.normal(0, 0.3, size=(channels, 4, 1)).astype(np.float32)
    ).astype(dtype)
    carry = mx.array(rng.normal(size=(rows, 3, channels)).astype(np.float32)).astype(
        dtype
    )
    reference_carry = carry
    for step in range(4):
        current = projected[:, step * tokens : (step + 1) * tokens, 16 : 16 + channels]
        window = mx.concatenate([reference_carry, current], axis=1)
        activation = nn.silu(mx.conv1d(window, weight, groups=channels))
        q, k, v = mx.split(activation, [width, 2 * width], axis=-1)
        expected = tuple(x.reshape(rows, tokens, heads, dim) for x in (q, k, v))
        expected += (window[:, -3:],)
        mx.eval(expected)
        actual = conv4_silu_qkv(
            current,
            carry,
            weight,
            key_heads=heads,
            key_dim=dim,
            value_heads=heads,
            value_dim=dim,
        )
        mx.eval(actual)
        unsigned = mx.uint16 if dtype == mx.bfloat16 else mx.uint32
        for value, reference in zip(actual, expected, strict=True):
            assert value.shape == reference.shape and value.dtype == reference.dtype
            np.testing.assert_array_equal(
                np.asarray(value.view(unsigned)), np.asarray(reference.view(unsigned))
            )
        carry, reference_carry = actual[-1], expected[-1]
