"""Partition reduction against an independent FP64 log-sum-exp oracle."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin_models_mlx import paged_attention as attention  # noqa: E402
from resin_models_mlx.paged_attention import _merge  # noqa: E402


def _inputs(partitions):
    rng = np.random.default_rng(18371 + partitions)
    return (
        rng.normal(size=(2, 8, partitions, 256)).astype(np.float32),
        rng.normal(size=(2, 8, partitions)).astype(np.float32),
        rng.uniform(0.1, 30, size=(2, 8, partitions)).astype(np.float32),
    )


def _check(values):
    partials, maxima, sums = [value.astype(np.float64) for value in values]
    with np.errstate(invalid="ignore", divide="ignore"):
        peak = maxima.max(axis=-1, keepdims=True)
        weights = np.where(np.isneginf(maxima), 0, np.exp(maxima - peak))
        expected = (partials * weights[..., None]).sum(axis=2) / (sums * weights).sum(
            axis=-1
        )[..., None]
    actual = np.asarray(_merge(*(mx.array(value) for value in values)))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6, equal_nan=True)


@pytest.mark.parametrize("partitions", [1, 2, 7, 8, 9, 31, 32, 33, 64, 65, 129])
def test_partition_merge_at_reduction_boundaries(partitions):
    _check(_inputs(partitions))


_SPECIALS = (
    "empty_tail",
    "all_empty",
    "extreme",
    "nan",
    "negative_zero",
    "cancellation",
    "poison",
    "zero_denominator",
)


def _special(kind):
    partials, maxima, sums = _inputs(18)
    if kind in ("empty_tail", "poison"):
        partials[..., 9:, :] = np.nan if kind == "poison" else 0
        maxima[..., 9:], sums[..., 9:] = -np.inf, 0
    elif kind == "all_empty":
        partials.fill(0)
        maxima.fill(-np.inf)
        sums.fill(0)
    elif kind == "extreme":
        maxima[..., ::2], maxima[..., 1::2] = 1000, -1000
    elif kind == "nan":
        maxima[..., 1] = np.nan
    elif kind == "negative_zero":
        partials.fill(-0.0)
        maxima.fill(-0.0)
        sums.fill(1)
    elif kind == "cancellation":
        partials[..., ::2, :], partials[..., 1::2, :] = 1e10, -1e10
        maxima.fill(0)
        sums.fill(1)
    else:
        sums.fill(0)
    return partials, maxima, sums


@pytest.mark.parametrize("kind", _SPECIALS)
def test_partition_merge_handles_empty_and_extreme_partitions(kind):
    _check(_special(kind))


def _assert_fused(values, dtype):
    inputs = [mx.array(value) for value in values]
    actual = attention._fused_merge(*inputs, dtype=dtype)
    assert actual is not None and actual.dtype == dtype
    unsigned = mx.uint32 if dtype == mx.float32 else mx.uint16
    np.testing.assert_array_equal(
        np.asarray(actual.view(unsigned)),
        np.asarray(_merge(*inputs).astype(dtype).view(unsigned)),
    )


@pytest.mark.parametrize("partitions", [2, 3, 7, 8, 9, 17, 18, 31])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal merge")
def test_small_fused_merge_preserves_native_reduction_bits(partitions, dtype):
    _assert_fused(_inputs(partitions), getattr(mx, dtype))


@pytest.mark.parametrize("kind", _SPECIALS)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal merge")
def test_small_fused_merge_preserves_special_value_bits(kind, dtype):
    _assert_fused(_special(kind), getattr(mx, dtype))


@pytest.mark.parametrize("partitions", [1, 32, 33, 64, 65, 129])
def test_larger_merges_retain_compiled_native_path(partitions):
    inputs = [mx.array(value) for value in _inputs(partitions)]
    assert attention._fused_merge(*inputs, dtype=mx.bfloat16) is None
    expected = _merge(*inputs).astype(mx.bfloat16)
    actual = attention._merge_output(*inputs, dtype=mx.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(actual.view(mx.uint16)), np.asarray(expected.view(mx.uint16))
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal merge")
def test_warmed_merge_runs_on_a_fresh_thread():
    attention._warm_fused_merge()
    inputs = [mx.array(value) for value in _inputs(18)]
    mx.eval(inputs)

    def run():
        for dtype in (mx.float32, mx.bfloat16):
            unsigned = mx.uint32 if dtype == mx.float32 else mx.uint16
            actual = attention._fused_merge(*inputs, dtype=dtype)
            np.testing.assert_array_equal(
                np.asarray(actual.view(unsigned)),
                np.asarray(_merge(*inputs).astype(dtype).view(unsigned)),
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(run).result(timeout=30)
