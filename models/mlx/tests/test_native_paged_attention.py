"""Native paged partitions against the independent canonical Metal dispatch."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal attention")

from resin_models_mlx import paged_attention as attention  # noqa: E402
from resin_models_mlx._native_cache import (  # noqa: E402
    _paged_attention_partials,
    _warm_paged_attention,
)

_HEADS, _KV, _DIM, _PAGE = 8, 2, 256, 128


def _same_bits(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    unsigned = mx.uint16 if actual.dtype == mx.bfloat16 else mx.uint32
    np.testing.assert_array_equal(
        np.asarray(actual.view(unsigned)), np.asarray(expected.view(unsigned))
    )


def _canonical_partials(q, k, v, units, pages, lengths, layer, scale):
    """Call the original raw kernel directly, bypassing automatic native routing."""

    rows = q.shape[0]
    partitions = -(-(pages.shape[1] * _PAGE + 1) // 512)
    return attention._kernel()(
        inputs=[
            q,
            k,
            v,
            units,
            pages,
            lengths,
            mx.array([pages.shape[1], partitions, layer], mx.int32),
            mx.array([scale], mx.float32),
        ],
        template=[
            ("D", 256),
            ("HEADS", 8),
            ("KV_HEADS", 2),
            ("GROUP", 2),
            ("HEAD_GROUPS", 2),
            ("PAGE", 128),
            ("PART", 512),
            ("SIMDGROUPS", 4),
            ("PREFETCH", 1),
            ("T", mx.bfloat16),
        ],
        grid=(128, 1, rows * 4 * partitions),
        threadgroup=(128, 1, 1),
        output_shapes=[
            (rows, 8, partitions, 256),
            (rows, 8, partitions),
            (rows, 8, partitions),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


def _offset(value, count):
    return mx.concatenate([mx.full((count,), mx.nan, value.dtype), value.reshape(-1)])[
        count:
    ].reshape(value.shape)


def _fixture(lengths, *, lazy_strided_units=False):
    rows, layers, layer = len(lengths), 3, 2
    counts = [-(-length // _PAGE) for length in lengths]
    blocks = max(1, *counts)
    rng = np.random.default_rng(29101 + sum(lengths) + rows)
    order = (rng.permutation(sum(counts)) * layers + 3).astype(np.int32)
    physical = np.full(
        (len(order) * layers + 7, 2, _PAGE, _KV * _DIM), np.nan, np.float32
    )
    capacity = max(rows, -(-8 // blocks))
    pages = np.zeros((capacity, blocks), np.int32)
    first = 0
    for row, (length, count) in enumerate(zip(lengths, counts, strict=True)):
        pages[row, :count] = order[first : first + count]
        first += count
        for block in range(count):
            size = min(_PAGE, length - block * _PAGE)
            physical[pages[row, block] + layer, :, :size] = rng.normal(
                size=(2, size, _KV * _DIM)
            )

    def random(shape):
        return mx.array(rng.normal(size=shape).astype(np.float32)).astype(mx.bfloat16)

    q, k = (
        _offset(random((rows, _HEADS, _DIM)), 13),
        _offset(random((rows, _KV, _DIM)), 7),
    )
    v = random((rows, _KV, _DIM))
    # Joined projections retain this real byte offset and row stride.
    v = mx.concatenate(
        [mx.full((rows, 4608), mx.nan, mx.bfloat16), v.reshape(rows, -1)], axis=1
    )[:, 4608:].reshape(v.shape)
    units = mx.array(physical).astype(mx.bfloat16)
    if lazy_strided_units:
        units = mx.stack([units, mx.full(units.shape, mx.nan, mx.bfloat16)], axis=-1)[
            ..., 0
        ]
    else:
        units = _offset(units, 1)
    pages = mx.concatenate(
        [mx.zeros((capacity, 1), mx.int32), mx.array(pages)], axis=1
    )[:, 1:]
    history = np.zeros(max(8, rows), np.int32)
    history[:rows] = lengths
    history = mx.concatenate([mx.array([99], mx.int32), mx.array(history)])[1:]
    return q, k, v, units, pages, history, layer, _DIM**-0.5


@pytest.mark.parametrize(
    "lengths,strided",
    [
        ([0], False),
        ([1, 127, 128, 129], False),
        ([511, 512], False),
        ([513], True),
        ([8192, 17, 0, 7937], False),
        ([16384], False),
    ],
)
def test_native_partials_preserve_canonical_f32_bits_for_boundaries_and_layouts(
    lengths, strided
):
    args = _fixture(lengths, lazy_strided_units=strided)
    actual = _paged_attention_partials(*args)
    expected = _canonical_partials(*args)
    assert len(actual) == 3 and all(value.dtype == mx.float32 for value in actual)
    for a, b in zip(actual, expected, strict=True):
        _same_bits(a, b)
    merged = attention._merge(*actual).astype(mx.bfloat16)
    _same_bits(merged, attention._merge(*expected).astype(mx.bfloat16))


def test_materialized_inputs_and_pipeline_work_on_a_fresh_native_thread():
    args = _fixture([3])
    mx.eval(*args[:6])
    assert _warm_paged_attention() is None
    assert _warm_paged_attention() is None
    expected = _canonical_partials(*args)
    mx.eval(expected)

    def worker():
        _warm_paged_attention()
        actual = _paged_attention_partials(*args)
        for a, b in zip(actual, expected, strict=True):
            _same_bits(a, b)

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result(timeout=30)


@pytest.mark.parametrize(
    "index,replacement",
    [
        (0, mx.float32),
        (1, mx.float32),
        (2, mx.float32),
        (3, mx.float32),
        (4, mx.int64),
        (5, None),
        (6, -1),
        (7, float("nan")),
    ],
)
def test_native_attention_rejects_unsupported_host_metadata_without_narrowing(
    index, replacement
):
    values = list(_fixture([0]))
    if index < 5:
        values[index] = values[index].astype(replacement)
    elif index == 5:
        values[index] = mx.array([], mx.int32)
    else:
        values[index] = replacement
    with pytest.raises(ValueError, match="BF16|UNITS|metadata"):
        _paged_attention_partials(*values)


def test_native_attention_and_warmup_reject_cpu_stream():
    args = _fixture([0])
    with mx.stream(mx.cpu):
        with pytest.raises(ValueError, match="GPU"):
            _paged_attention_partials(*args)
        with pytest.raises(ValueError, match="GPU"):
            _warm_paged_attention()


@pytest.mark.parametrize("transform", ["compile", "grad", "vmap"])
def test_native_attention_rejects_mutable_read_transforms(transform):
    args = _fixture([0])

    def call(query):
        return _paged_attention_partials(query, *args[1:])[0].sum()

    with pytest.raises(ValueError, match="compile|grad|vmap|transform"):
        if transform == "compile":
            mx.compile(call)(args[0])
        elif transform == "grad":
            mx.grad(call)(args[0])
        else:
            mx.vmap(call)(mx.stack([args[0], args[0]]))
