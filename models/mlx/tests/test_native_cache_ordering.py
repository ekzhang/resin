"""Cache readers must survive MLX event/command-encoder boundaries."""

import types

import mlx.core as mx
import numpy as np
import pytest

from resin_models_mlx import paged_attention
from resin_models_mlx.native import CacheStorage


def _attention_partials(*args):
    # Retain all three independently fenced outputs of the real reader.
    # A generic slice/cast/reduction does not reproduce the encoder boundary.
    function = paged_attention.paged_decode_attention
    core = types.FunctionType(
        function.__code__,
        dict(
            function.__globals__,
            _merge_output=lambda *arrays, **_kwargs: arrays,
        ),
        "attention_partials",
        function.__defaults__,
    )
    return core(*args)


def _assert_bits(actual, expected):
    integer = mx.uint16 if actual.dtype == mx.bfloat16 else mx.uint32
    np.testing.assert_array_equal(
        np.asarray(actual.view(integer)), np.asarray(expected.view(integer))
    )


@pytest.mark.parametrize("write", ["rows", "slice"])
@pytest.mark.parametrize(
    "snapshot,explicit_readers", [(True, False), (True, True), (False, True)]
)
def test_cross_stream_readers_and_lazy_snapshots_precede_mutation(
    write, snapshot, explicit_readers
):
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
        return _attention_partials(q, k, v, units, pages, lengths, 1, 256**-0.5)

    expected_before, expected_after = read(initial), read(changed)
    mx.eval(expected_before, expected_after)
    owner = CacheStorage((10 * 256, 512), mx.bfloat16)
    a, b = mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)
    with mx.stream(a):
        epoch = owner.write_slice(owner.initial_epoch(), 0, initial)
        before = owner.snapshot(epoch) if snapshot else None
        units = owner.read_epoch(epoch).reshape(initial.shape)
        old_a = read(units)
    with mx.stream(b):
        old_b = read(units)
    with mx.stream(a):
        roots = [*old_a, *old_b]
        ordered = mx.depends(epoch, roots)
        keywords = {"after": roots} if explicit_readers else {}
        if write == "rows":
            epoch = owner.scatter_rows(
                ordered,
                mx.array([5 * 256], mx.int64),
                mx.zeros((1, 512), mx.bfloat16),
                **keywords,
            )
        else:
            epoch = owner.write_slice(
                ordered,
                5 * 256 * 512,
                mx.zeros(512, mx.bfloat16),
                **keywords,
            )
        current = owner.read_epoch(epoch).reshape(initial.shape)
        new_a = read(current)
    with mx.stream(b):
        new_b = read(current)
        after = owner.snapshot(mx.depends(epoch, [*new_a, *new_b]))
    # Evaluate the entire queued chain together: pre-evaluating `before` hides
    # the regression. The old snapshot alone is an automatic read dependency.
    mx.eval(old_a, old_b, new_a, new_b, after, *([] if before is None else [before]))
    if before is not None:
        _assert_bits(before.reshape(initial.shape), initial)
    _assert_bits(after.reshape(initial.shape), changed)
    for actual, expected in (
        (old_a, expected_before),
        (old_b, expected_before),
        (new_a, expected_after),
        (new_b, expected_after),
    ):
        for array, reference in zip(actual, expected, strict=True):
            _assert_bits(array, reference)
    assert owner.allocation_count == 1
