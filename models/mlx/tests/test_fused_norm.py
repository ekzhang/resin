"""Independent native contracts for RMS/residual Metal fusions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin_models_mlx.qwen3_5.fused_norm import (  # noqa: E402
    mixed_rms_norm,
    residual_rms_norm,
)

_KINDS = ("mixed", "residual")
_MATRIX = [
    (32, 1, "contiguous"),
    (64, 2, "offset"),
    (127, 4, "inner_stride"),
    (128, 16, "q_slice"),
    (129, 1, "transpose"),
    (256, 2, "broadcast"),
    (1024, 1, "q_slice"),
    (4096, 16, "inner_stride"),
]


def _assert_native_equal(actual, expected):
    """Finite values, infinities and signed zeros must match bit for bit.

    NaN payload bits are not part of the native arithmetic contract; their
    locations must agree exactly. No numerical tolerance is used.
    """
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _assert_native_equal(value, reference)
        return
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    a = np.asarray(actual.astype(mx.float32))
    b = np.asarray(expected.astype(mx.float32))
    nan = np.isnan(b)
    np.testing.assert_array_equal(np.isnan(a), nan)
    bits = mx.uint32 if expected.dtype == mx.float32 else mx.uint16
    np.testing.assert_array_equal(
        np.asarray(actual.view(bits))[~nan], np.asarray(expected.view(bits))[~nan]
    )


def _view(data, dtype, layout):
    if layout == "offset":
        padded = np.concatenate([np.full(13, np.nan, np.float32), data.reshape(-1)])
        return mx.array(padded).astype(dtype)[13:].reshape(data.shape)
    if layout in ("q_slice", "inner_stride"):
        padded = np.full((*data.shape[:-1], 2 * data.shape[-1]), np.nan, np.float32)
        if layout == "q_slice":
            padded[..., : data.shape[-1]] = data
            return mx.array(padded).astype(dtype)[..., : data.shape[-1]]
        padded[..., ::2] = data
        return mx.array(padded).astype(dtype)[..., ::2]
    value = mx.array(data).astype(dtype)
    if layout == "transpose":
        return value.swapaxes(0, 1)
    if layout == "broadcast":
        return mx.broadcast_to(value[:1, :1], value.shape)
    return value


def _inputs(width, batch=1, tokens=3, layout="contiguous", *, dtype=mx.bfloat16):
    rng = np.random.default_rng(8721 + width + 13 * batch + tokens)
    shape = (batch, tokens, width)
    x = _view(rng.standard_normal(shape, dtype=np.float32), dtype, layout)
    residual = _view(rng.standard_normal(shape, dtype=np.float32), dtype, layout)
    # Nontrivial negative and positive scales expose the native weighted
    # rounding; HF offsets are converted to FP32 before adding one.
    offset = mx.array(rng.uniform(-0.7, 0.7, width).astype(np.float32)).astype(
        mx.bfloat16
    )
    scale = 1.0 + offset.astype(mx.float32)
    return x, residual, scale


def _reference(kind, x, residual, scale, eps=1e-6):
    if kind == "mixed":
        return mx.fast.rms_norm(x, scale, eps).astype(x.dtype)
    if kind == "residual":
        summed = x + residual
        return summed, mx.fast.rms_norm(summed, scale, eps).astype(summed.dtype)
    raise ValueError(kind)


def _actual(kind, x, residual, scale, eps=1e-6):
    if kind == "mixed":
        return mixed_rms_norm(x, scale, eps)
    if kind == "residual":
        return residual_rms_norm(x, residual, scale, eps)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("width,batch,layout", _MATRIX)
def test_rms_fusions_match_native_bits_for_widths_and_views(kind, width, batch, layout):
    values = _inputs(width, batch, layout=layout)
    _assert_native_equal(_actual(kind, *values), _reference(kind, *values))


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("tokens,width", [(37, 129), (4096, 1024)])
def test_rms_fusions_match_native_bits_for_prefill(kind, tokens, width):
    values = _inputs(width, tokens=tokens, layout="q_slice")
    _assert_native_equal(_actual(kind, *values), _reference(kind, *values))


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("scale_layout", ["strided", "broadcast"])
def test_rms_fusions_preserve_scale_and_leading_broadcasts(kind, scale_layout):
    x, residual, scale = _inputs(129, batch=4, layout="inner_stride")
    if scale_layout == "strided":
        scale = mx.stack([scale, mx.full(scale.shape, mx.nan)], axis=-1).reshape(-1)[
            ::2
        ]
    else:
        scale = mx.broadcast_to(mx.array([1.125], dtype=mx.float32), scale.shape)
    residual = residual[0, 0]
    values = (x, residual, scale)
    _assert_native_equal(_actual(kind, *values), _reference(kind, *values))


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("eps", [0.0, 1e-6, 1e-3])
def test_rms_fusions_preserve_special_values_and_signed_zero(kind, eps):
    width = 129
    x, residual, scale = _inputs(width, tokens=7)
    data = np.asarray(x.astype(mx.float32)).copy()
    data[:, 0] = 0.0
    data[:, 1] = -0.0
    data[:, 2] *= 1e-20
    data[:, 3] *= 1e20
    data[:, 4] = np.inf
    data[:, 5, 0] = np.nan
    data[:, 6] *= -1e20
    x = _view(data, mx.bfloat16, "offset")
    residual = mx.zeros(x.shape, dtype=x.dtype)
    values = (x, residual, scale)
    _assert_native_equal(_actual(kind, *values, eps), _reference(kind, *values, eps))


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    "width,dtype,scale_kind",
    [
        (127, "float16", "normal"),
        (127, "float32", "normal"),
        (7, "bfloat16", "normal"),
        (4097, "bfloat16", "normal"),
        (32, "bfloat16", "none"),
        (32, "bfloat16", "wrong_width"),
    ],
)
def test_native_fallbacks_and_validation(kind, width, dtype, scale_kind):
    x, residual, scale = _inputs(
        width, batch=2, dtype=getattr(mx, dtype), layout="inner_stride"
    )
    scale = (
        None
        if scale_kind == "none"
        else scale[:-1]
        if scale_kind == "wrong_width"
        else scale
    )
    try:
        expected = _reference(kind, x, residual, scale)
        mx.eval(expected)
    except (ValueError, RuntimeError, TypeError) as error:
        with pytest.raises(type(error)):
            mx.eval(_actual(kind, x, residual, scale))
    else:
        _assert_native_equal(_actual(kind, x, residual, scale), expected)


@pytest.mark.parametrize("kind", _KINDS)
def test_rms_fusions_keep_native_cpu_fallback(kind):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        values = _inputs(129, batch=2, layout="inner_stride")
        _assert_native_equal(_actual(kind, *values), _reference(kind, *values))
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("kind", _KINDS)
def test_materialized_norm_inputs_and_cached_kernels_reuse_on_fresh_thread(kind):
    values = _inputs(129, batch=2, layout="offset")
    mx.eval(values, _actual(kind, *values))

    def worker():
        actual, expected = _actual(kind, *values), _reference(kind, *values)
        mx.eval(actual, expected)
        _assert_native_equal(actual, expected)

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result(timeout=30)
