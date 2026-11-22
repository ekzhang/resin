"""The retained RMS fusions must preserve native MLX rounding exactly."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin_models_mlx.fused_norm import residual_rms_norm, rms_norm  # noqa: E402


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
        actual = (rms_norm(x, scale, 1e-6),)
    mx.eval(actual, expected)
    for value, reference in zip(actual, expected, strict=True):
        assert value.shape == reference.shape and value.dtype == reference.dtype
        np.testing.assert_array_equal(
            np.asarray(value.view(mx.uint16)), np.asarray(reference.view(mx.uint16))
        )


@pytest.mark.parametrize("input_dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("weighted", [True, False])
def test_single_bf16_rounding(input_dtype, weighted):
    rng = np.random.default_rng(940)
    x = mx.array(rng.normal(size=(5, 2304)).astype(np.float32)).astype(input_dtype)
    scale = (
        mx.array(rng.uniform(-2, 2, 2304).astype(np.float32)).astype(mx.bfloat16)
        if weighted
        else None
    )
    scale32 = None if scale is None else scale.astype(mx.float32)
    expected = mx.fast.rms_norm(x.astype(mx.float32), scale32, 1e-6).astype(mx.bfloat16)
    actual = rms_norm(x, scale, 1e-6)
    assert actual.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.asarray(actual.view(mx.uint16)), np.asarray(expected.view(mx.uint16))
    )
    if weighted and input_dtype == mx.bfloat16:
        double_rounded = mx.fast.rms_norm(x, scale, 1e-6)
        assert np.any(
            np.asarray(double_rounded.view(mx.uint16))
            != np.asarray(expected.view(mx.uint16))
        )
