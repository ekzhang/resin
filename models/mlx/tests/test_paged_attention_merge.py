"""The measured small-partition merge retains native reduction bits."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal merge")

from resin_models_mlx import paged_attention as attention  # noqa: E402


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_small_partition_merge_matches_native_bits_and_fp64_oracle(dtype):
    rng = np.random.default_rng(18389)
    partials = rng.normal(size=(2, 8, 18, 256)).astype(np.float32)
    maxima = rng.normal(size=(2, 8, 18)).astype(np.float32)
    sums = rng.uniform(0.1, 30, size=maxima.shape).astype(np.float32)
    partials[..., 9:, :], maxima[..., 9:], sums[..., 9:] = 0, -np.inf, 0
    weights = np.exp(maxima.astype(np.float64) - maxima.max(axis=-1, keepdims=True))
    expected = (partials * weights[..., None]).sum(axis=2) / (sums * weights).sum(
        axis=-1
    )[..., None]
    inputs = [mx.array(x) for x in (partials, maxima, sums)]
    native = attention._merge(*inputs)
    np.testing.assert_allclose(np.asarray(native), expected, atol=2e-6, rtol=2e-6)
    fused = attention._fused_merge(*inputs, dtype=dtype)
    assert fused is not None and fused.dtype == dtype
    unsigned = mx.uint32 if dtype == mx.float32 else mx.uint16
    np.testing.assert_array_equal(
        np.asarray(fused.view(unsigned)),
        np.asarray(native.astype(dtype).view(unsigned)),
    )
