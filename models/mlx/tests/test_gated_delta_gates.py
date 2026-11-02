"""Preserve the original FP32-decay caller contract and official GDN results."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models import gated_delta as official  # noqa: E402

from resin_models_mlx.qwen3_5.gated_delta import (  # noqa: E402
    decay_beta,
    gated_delta_update,
)

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


def _exact(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    left = np.asarray(actual.astype(mx.float32))
    right = np.asarray(expected.astype(mx.float32))
    np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
    unsigned = mx.uint16 if actual.dtype == mx.bfloat16 else mx.uint32
    left_bits, right_bits = (
        np.asarray(actual.view(unsigned)),
        np.asarray(expected.view(unsigned)),
    )
    non_nan = ~np.isnan(left)
    np.testing.assert_array_equal(left_bits[non_nan], right_bits[non_nan])


def _inputs(rows, tokens, dtype, dt_dtype):
    rng = np.random.default_rng(8113 + rows * 31 + tokens)
    a, b = [
        mx.array(rng.normal(size=(rows, tokens, 16)).astype(np.float32)).astype(dtype)
        for _ in range(2)
    ]
    A_log = mx.array(rng.uniform(-8, 1, size=16).astype(np.float32))
    dt_bias = mx.array(rng.normal(size=16).astype(np.float32)).astype(dt_dtype)
    return A_log, a, b, dt_bias


def _original_gates(values):
    A_log, a, b, dt_bias = values
    return official.compute_g(A_log, a.astype(mx.float32), dt_bias), mx.sigmoid(b)


@pytest.mark.parametrize(
    "dtype_name,dt_dtype_name,rows,tokens",
    [
        ("bfloat16", "bfloat16", 1, 1),
        ("bfloat16", "float32", 2, 7),
        ("float32", "bfloat16", 4, 1),
        ("float32", "float32", 4, 7),
        ("bfloat16", "float32", 4, 1),
        ("float32", "bfloat16", 1, 7),
    ],
)
def test_gate_dtypes_across_projection_shapes(dtype_name, dt_dtype_name, rows, tokens):
    values = _inputs(rows, tokens, getattr(mx, dtype_name), getattr(mx, dt_dtype_name))
    for actual, expected in zip(
        decay_beta(*values), _original_gates(values), strict=True
    ):
        _exact(actual, expected)


@pytest.mark.parametrize("dt_dtype_name", ["bfloat16", "float32"])
def test_all_finite_bf16_gate_patterns(dt_dtype_name):
    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[(bits & 0x7F80) != 0x7F80]
    gates = mx.array(bits).view(mx.bfloat16).reshape(1, -1, 16)
    values = (
        mx.linspace(-8, 1, 16),
        gates,
        gates[..., ::-1],
        mx.linspace(-1, 1, 16).astype(getattr(mx, dt_dtype_name)),
    )
    for actual, expected in zip(
        decay_beta(*values), _original_gates(values), strict=True
    ):
        _exact(actual, expected)


def test_gate_signed_zeros_infinities_and_nan_masks():
    edges = mx.array(
        [
            -mx.inf,
            mx.inf,
            mx.nan,
            0.0,
            -0.0,
            1e-38,
            -1e-38,
            30.0,
            -30.0,
            1e10,
            -1e10,
            1.0,
            -1.0,
            0.1,
            -0.1,
            3.0,
        ],
        mx.float32,
    ).reshape(1, 1, 16)
    values = (mx.linspace(-8, 1, 16), edges, edges, mx.zeros(16, mx.bfloat16))
    for actual, expected in zip(
        decay_beta(*values), _original_gates(values), strict=True
    ):
        _exact(actual, expected)


def _recurrent_inputs(rows, tokens, dtype, key_dim=128):
    rng = np.random.default_rng(90017 + rows + tokens + key_dim)
    q, k, v = [
        mx.array(
            rng.normal(0, 0.02, (rows, tokens, 16, key_dim)).astype(np.float32)
        ).astype(dtype)
        for _ in range(3)
    ]
    state = mx.array(
        rng.normal(0, 0.01, (rows, 16, key_dim, key_dim)).astype(np.float32)
    )
    return q, k, v, state


@pytest.mark.parametrize(
    "dtype_name,rows,tokens",
    [
        ("bfloat16", 1, 1),
        ("bfloat16", 4, 7),
        ("float32", 2, 1),
        ("float32", 1, 7),
    ],
)
def test_recurrent_output_and_state_match_original_caller(dtype_name, rows, tokens):
    dtype = getattr(mx, dtype_name)
    A_log, a, b, dt_bias = _inputs(rows, tokens, dtype, mx.bfloat16)
    q, k, v, state = _recurrent_inputs(rows, tokens, dtype)
    expected = official.gated_delta_update(
        q, k, v, a.astype(mx.float32), b, A_log, dt_bias, state
    )
    actual = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state)
    for left, right in zip(actual, expected, strict=True):
        _exact(left, right)


@pytest.mark.parametrize("tokens,initial_state", [(1, False), (7, True)])
def test_masked_native_recurrence_matches_official(tokens, initial_state):
    A_log, a, b, dt_bias = _inputs(2, tokens, mx.bfloat16, mx.float32)
    q, k, v, state = _recurrent_inputs(2, tokens, mx.bfloat16)
    state = state if initial_state else None
    mask = mx.array(np.arange(2 * tokens).reshape(2, tokens) % 3 != 0)
    expected = official.gated_delta_update(
        q, k, v, a.astype(mx.float32), b, A_log, dt_bias, state, mask
    )
    actual = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask)
    for left, right in zip(actual, expected, strict=True):
        _exact(left, right)


@pytest.mark.parametrize("masked", [False, True])
def test_low_dimension_initial_state_uses_reference_ops(masked):
    A_log, a, b, dt_bias = _inputs(1, 3, mx.bfloat16, mx.float32)
    q, k, v, _ = _recurrent_inputs(1, 3, mx.bfloat16, key_dim=16)
    mask = mx.array([[True, False, True]]) if masked else None
    expected = official.gated_delta_update(
        q, k, v, a.astype(mx.float32), b, A_log, dt_bias, None, mask
    )
    actual = gated_delta_update(q, k, v, a, b, A_log, dt_bias, None, mask)
    for left, right in zip(actual, expected, strict=True):
        _exact(left, right)
