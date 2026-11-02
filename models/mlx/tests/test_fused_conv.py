"""Conv4/QKV arithmetic, carry and actual GDN layouts against native MLX."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

from mlx_lm.models.gated_delta import gated_delta_update  # noqa: E402

from resin_models_mlx.qwen3_5.fused_conv import conv4_silu_qkv  # noqa: E402
from resin_models_mlx.qwen3_5.normalization import normalize_qk  # noqa: E402


def _dimensions(channels):
    return (16, 128, 16, 128) if channels == 6144 else (1, 1, 1, channels - 2)


def _actual(current, previous, weight):
    kh, kd, vh, vd = _dimensions(current.shape[-1])
    return conv4_silu_qkv(
        current,
        previous,
        weight,
        key_heads=kh,
        key_dim=kd,
        value_heads=vh,
        value_dim=vd,
    )


def _reference(current, previous, weight):
    rows, tokens, channels = current.shape
    kh, kd, vh, vd = _dimensions(channels)
    window = mx.concatenate([previous.astype(current.dtype), current], axis=1)
    activation = nn.silu(mx.conv1d(window, weight, groups=channels))
    q, k, v = mx.split(activation, [kh * kd, 2 * kh * kd], axis=-1)
    kernel = weight.shape[1]
    carry = window[:, -(kernel - 1) :] if kernel > 1 else window[:, :0]
    return (
        q.reshape(rows, tokens, kh, kd),
        k.reshape(rows, tokens, kh, kd),
        v.reshape(rows, tokens, vh, vd),
        carry,
    )


def _bits(value):
    return np.asarray(value.view(mx.uint32 if value.dtype == mx.float32 else mx.uint16))


def _exact(actual, expected):
    mx.eval(actual, expected)
    for value, reference in zip(actual, expected, strict=True):
        assert value.shape == reference.shape and value.dtype == reference.dtype
        np.testing.assert_array_equal(_bits(value), _bits(reference))


def _fixture(
    rows=4, channels=6144, layout="offset", *, dtype="bfloat16", tokens=1, kernel=4
):
    rng = np.random.default_rng(71003 + channels + rows)
    dtype = getattr(mx, dtype)
    shape = (rows, tokens, channels)
    values = [
        rng.normal(size=s).astype(np.float32)
        for s in (shape, (rows, kernel - 1, channels), (channels, kernel, 1))
    ]
    if layout == "inner_stride":
        values = [
            mx.stack([mx.array(v), mx.full(v.shape, mx.nan)], axis=-1)
            .reshape(*v.shape[:-1], v.shape[-1] * 2)
            .astype(dtype)[..., ::2]
            for v in values
        ]
    else:
        values = [mx.array(v).astype(dtype) for v in values]
        if layout == "offset":
            padded = mx.full((rows, tokens, channels + 2080), mx.nan, dtype=dtype)
            padded[..., 16 : 16 + channels] = values[0]
            values[0] = padded[..., 16 : 16 + channels]
    return tuple(values)


@pytest.mark.parametrize(
    "rows,channels,layout",
    [
        (1, 128, "contiguous"),
        (4, 6144, "offset"),
        (16, 6144, "contiguous"),
        (2, 127, "inner_stride"),
        (4, 3, "contiguous"),
    ],
)
def test_decode_matches_native_and_does_not_mutate_inputs(rows, channels, layout):
    values = _fixture(rows, channels, layout)
    saved = [_bits(v).copy() for v in values]
    _exact(_actual(*values), _reference(*values))
    for value, original in zip(values, saved, strict=True):
        np.testing.assert_array_equal(_bits(value), original)


@pytest.mark.parametrize(
    "dtype,tokens,kernel",
    [
        ("float32", 1, 4),
        ("float16", 1, 4),
        ("bfloat16", 7, 4),
        ("bfloat16", 1, 3),
        ("bfloat16", 1, 1),
    ],
)
def test_native_fallbacks(dtype, tokens, kernel):
    values = _fixture(2, 127, "contiguous", dtype=dtype, tokens=tokens, kernel=kernel)
    _exact(_actual(*values), _reference(*values))


def test_every_finite_bf16_activation_in_actual_gdn_geometry():
    codes = np.arange(65536, dtype=np.uint16)
    codes = codes[(codes & 0x7F80) != 0x7F80]
    padded = np.zeros(((len(codes) + 6143) // 6144) * 6144, dtype=np.uint16)
    padded[: len(codes)] = codes
    current = mx.array(padded.reshape(-1, 1, 6144)).view(mx.bfloat16)
    previous = mx.zeros((current.shape[0], 3, 6144), mx.bfloat16)
    weight = mx.zeros((6144, 4, 1), mx.bfloat16)
    weight[:, -1] = 1
    assert len(codes) == 65280
    _exact(_actual(current, previous, weight), _reference(current, previous, weight))


def test_carry_feeds_changed_inputs_without_drift():
    current, previous, weight = _fixture(2, 128, tokens=16)
    actual_carry, reference_carry = previous, previous
    for step in range(16):
        actual = _actual(current[:, step : step + 1], actual_carry, weight)
        expected = _reference(current[:, step : step + 1], reference_carry, weight)
        _exact(actual, expected)
        actual_carry, reference_carry = actual[-1], expected[-1]


def test_materialized_inputs_and_cached_kernel_work_on_fresh_thread():
    values = _fixture()
    mx.eval(values, _actual(*values))
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(lambda: _exact(_actual(*values), _reference(*values))).result(
            timeout=30
        )


def test_split_values_feed_normalization_and_gdn_exactly():
    current, previous, weight = _fixture()
    rng = np.random.default_rng(71849)
    state = mx.array(rng.normal(0, 0.02, (4, 16, 128, 128)).astype(np.float32))
    a, b = [
        mx.array(rng.normal(size=(4, 1, 16)).astype(np.float32)).astype(mx.bfloat16)
        for _ in range(2)
    ]
    A_log, dt_bias = [
        mx.array(rng.normal(size=16).astype(np.float32)) for _ in range(2)
    ]
    dt_bias = dt_bias.astype(mx.bfloat16)

    def chain(function):
        q, k, v, carry = function(current, previous, weight)
        q, k = normalize_qk(q, k, inv_scale=128**-0.5, eps=1e-6)
        output, recurrent = gated_delta_update(
            q, k, v, a.astype(mx.float32), b, A_log, dt_bias, state
        )
        return output, recurrent, carry

    _exact(chain(_actual), chain(_reference))


def test_cpu_default_uses_the_native_fallback():
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        values = _fixture(1, 8, "contiguous")
        try:
            expected = _reference(*values)
            mx.eval(expected)
        except (ValueError, RuntimeError) as error:
            with pytest.raises(type(error)):
                mx.eval(_actual(*values))
        else:
            _exact(_actual(*values), expected)
    finally:
        mx.set_default_device(previous_device)


@pytest.mark.parametrize(
    "shapes",
    [
        ((0, 1, 8), (0, 3, 8), (8, 4, 1)),
        ((1, 0, 8), (1, 3, 8), (8, 4, 1)),
        ((1, 1, 8), (1, 2, 8), (8, 4, 1)),
        ((1, 1, 8), (1, 3, 8), (9, 4, 1)),
        ((1, 1, 8), (1, 3, 8), (8, 4, 2)),
    ],
)
def test_invalid_geometry_is_rejected(shapes):
    with pytest.raises(ValueError):
        _actual(*(mx.zeros(shape, mx.bfloat16) for shape in shapes))
