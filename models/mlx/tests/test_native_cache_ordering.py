"""Cache readers must survive MLX event/command-encoder boundaries."""

import mlx.core as mx
import numpy as np
import pytest

from resin_models_mlx._native import CacheStorage
from resin_models_mlx._native_cache import _paged_attention_partials


def _assert_bits(actual, expected):
    integer = mx.uint16 if actual.dtype == mx.bfloat16 else mx.uint32
    np.testing.assert_array_equal(
        np.asarray(actual.view(integer)), np.asarray(expected.view(integer))
    )


@pytest.mark.parametrize("write", ["rows", "slice"])
def test_cross_stream_attention_readers_precede_mutation(write):
    rng = np.random.default_rng(9937)
    initial = mx.array(rng.normal(size=(10, 2, 128, 512)).astype(np.float32)).astype(
        mx.bfloat16
    )
    q = mx.array(rng.normal(size=(1, 8, 256)).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array(rng.normal(size=(1, 2, 256)).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.normal(size=(1, 2, 256)).astype(np.float32)).astype(mx.bfloat16)
    pages = mx.full((8, 1), 4, mx.int32)
    lengths = mx.array([3, 0, 0, 0, 0, 0, 0, 0], mx.int32)
    mx.eval(initial, q, k, v, pages, lengths)
    changed_bits = np.asarray(initial.view(mx.uint16)).copy()
    changed_bits[5, 0, 0] = 0
    changed = mx.array(changed_bits).view(mx.bfloat16)

    def read(units):
        return _paged_attention_partials(q, k, v, units, pages, lengths, 1, 256**-0.5)

    expected_before, expected_after = read(initial), read(changed)
    mx.eval(expected_before, expected_after)
    row_bytes = 512 * initial.itemsize
    owner = CacheStorage((10 * 256, row_bytes))
    a, b = mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)
    with mx.stream(a):
        epoch = owner.write_slice(owner.initial_epoch(), 0, initial)
        units = owner.read_epoch(epoch).view(mx.bfloat16).reshape(initial.shape)
        old_a = read(units)
    with mx.stream(b):
        old_b = read(units)
    with mx.stream(a):
        roots = [*old_a, *old_b]
        ordered = mx.depends(epoch, roots)
        if write == "rows":
            epoch = owner.scatter_rows(
                ordered,
                mx.array([5 * 256], mx.int64),
                mx.zeros((1, 512), mx.bfloat16),
                row_bytes=row_bytes,
                after=roots,
            )
        else:
            epoch = owner.write_slice(
                ordered,
                5 * 256 * row_bytes,
                mx.zeros(512, mx.bfloat16),
                after=roots,
            )
        current = owner.read_epoch(epoch).view(mx.bfloat16).reshape(initial.shape)
        new_a = read(current)
    with mx.stream(b):
        new_b = read(current)
    # Evaluating the old partials before the write would hide the regression.
    mx.eval(old_a, old_b, new_a, new_b)
    _assert_bits(current, changed)
    for actual, expected in (
        (old_a, expected_before),
        (old_b, expected_before),
        (new_a, expected_after),
        (new_b, expected_after),
    ):
        for array, reference in zip(actual, expected, strict=True):
            _assert_bits(array, reference)
