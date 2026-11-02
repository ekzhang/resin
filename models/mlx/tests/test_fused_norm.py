"""The retained RMS fusions must preserve native MLX rounding exactly."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin_models_mlx.qwen3_5.fused_norm import (  # noqa: E402
    mixed_rms_norm,
    residual_rms_norm,
)


@pytest.mark.parametrize("width,strided", [(128, True), (1024, False)])
@pytest.mark.parametrize("residual", [False, True])
def test_fused_norm_matches_native(width, strided, residual):
    rng = np.random.default_rng(8721 + width)
    # Poisoned extra columns make this a noncontiguous projection view.
    source = mx.array(rng.normal(size=(4, 3, 2 * width)).astype(np.float32)).astype(
        mx.bfloat16
    )
    source[..., width:] = mx.nan
    x = source[..., :width]
    if not strided:
        x = mx.contiguous(x)
    add = mx.array(rng.normal(size=x.shape).astype(np.float32)).astype(mx.bfloat16)
    scale = mx.array(rng.uniform(-1, 2, width).astype(np.float32))
    if residual:
        summed = x + add
        expected = (summed, mx.fast.rms_norm(summed, scale, 1e-6).astype(x.dtype))
        actual = residual_rms_norm(x, add, scale, 1e-6)
    else:
        expected = (mx.fast.rms_norm(x, scale, 1e-6).astype(x.dtype),)
        actual = (mixed_rms_norm(x, scale, 1e-6),)
    mx.eval(actual, expected)
    for value, reference in zip(actual, expected, strict=True):
        assert value.shape == reference.shape and value.dtype == reference.dtype
        np.testing.assert_array_equal(
            np.asarray(value.view(mx.uint16)), np.asarray(reference.view(mx.uint16))
        )
