"""Public native scatter contracts with the production 512-element row width.

These cases deliberately use width 512, so a rebuilt production extension
selects its row/column dispatch. No artifact owner or private selector is used.
Expected storage bytes and computed-read values come from NumPy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    raise pytest.skip.Exception(
        "Native cache row tests require Metal", allow_module_level=True
    )

from resin_models_mlx.native import CacheStorage  # noqa: E402

WIDTH = 512


def _unsigned(dtype):
    return (np.uint16, mx.uint16) if dtype == mx.bfloat16 else (np.uint32, mx.uint32)


def _raw(rows, dtype, seed=17):
    raw_dtype, _ = _unsigned(dtype)
    bits = (
        (np.arange(rows * WIDTH, dtype=np.uint32) * np.uint32(40503) + np.uint32(seed))
        .astype(raw_dtype)
        .reshape(rows, WIDTH)
    )
    if rows:
        edges = (
            [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 1, 0x7F7F]
            if dtype == mx.bfloat16
            else [
                0,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC00001,
                0x7F800001,
                1,
                0x7F7FFFFF,
            ]
        )
        bits[:, : len(edges)] = np.array(edges, raw_dtype)
        # Raw BF16 patterns repeat, so retain an independent identity per row.
        bits[:, 8] = np.arange(rows, dtype=np.uint32).astype(raw_dtype)
    return bits


def _snapshot_bits(storage, epoch, dtype):
    value = storage.snapshot(epoch)
    assert value.shape[-1] == WIDTH and value.dtype == dtype
    return np.asarray(value.view(_unsigned(dtype)[1]))


def _epoch_is(epoch, value):
    assert epoch.dtype == mx.uint32 and epoch.shape == (8,)
    np.testing.assert_array_equal(np.asarray(epoch), np.full(8, value, np.uint32))


@pytest.mark.parametrize(
    "dtype,index_dtype,count",
    [
        (mx.bfloat16, mx.int32, 0),
        (mx.bfloat16, mx.int64, 1),
        (mx.float32, mx.int32, 3),
        (mx.float32, mx.int64, 2049),
    ],
)
def test_public_512_rows_preserve_raw_bits_and_all_epoch_lanes(
    dtype, index_dtype, count
):
    total = count + 17
    storage = CacheStorage((total, WIDTH), dtype)
    epoch = storage.initial_epoch()
    assert storage.allocation_count == storage.buffer_identity == 0
    indices = np.arange(count, dtype=np.int64)[::-1] + 5
    updates = _raw(count, dtype)
    epoch = storage.scatter_rows(
        epoch, mx.array(indices, index_dtype), mx.array(updates).view(dtype)
    )
    expected = np.zeros((total, WIDTH), updates.dtype)
    expected[indices] = updates
    np.testing.assert_array_equal(_snapshot_bits(storage, epoch, dtype), expected)
    _epoch_is(epoch, 1)
    assert storage.buffer_identity != 0 and storage.allocation_count == 1


def test_real_prefill_49152_bf16_rows_preserve_each_row_and_column():
    # One real 4K chunk: 4096 tokens × six layers × key/value, 50 MiB payload.
    count = 49_152
    total = count + 1
    storage = CacheStorage((total, WIDTH), mx.bfloat16)
    raw = _raw(count, mx.bfloat16)
    indices = np.arange(count, dtype=np.int32)[::-1].copy()
    epoch = storage.scatter_rows(
        storage.initial_epoch(), mx.array(indices), mx.array(raw).view(mx.bfloat16)
    )
    actual = _snapshot_bits(storage, epoch, mx.bfloat16)
    np.testing.assert_array_equal(actual[:count], raw[::-1])
    np.testing.assert_array_equal(actual[count], np.zeros(WIDTH, np.uint16))
    _epoch_is(epoch, 1)
    assert storage.buffer_identity != 0 and storage.allocation_count == 1


@pytest.mark.parametrize(
    "dtype,index_dtype,layout",
    [
        (mx.bfloat16, mx.int32, "offset_inner_stride"),
        (mx.float32, mx.int64, "offset_inner_stride"),
        (mx.bfloat16, mx.int64, "transpose"),
        (mx.float32, mx.int32, "transpose"),
        (mx.bfloat16, mx.int32, "broadcast"),
        (mx.float32, mx.int64, "broadcast"),
    ],
)
def test_512_rows_bounds_and_strided_sources_match_numpy(dtype, index_dtype, layout):
    total, count = 8, 9
    original = _raw(total, dtype, 41)
    payload = _raw(count, dtype, 97)
    indices = np.array([-1, 0, total, 6, 2**31 - 1, -2, 3, 1, 2], np.int64)
    if index_dtype == mx.int64:
        indices[4], indices[5] = 2**32 + 1, -(2**63)
    index_backing = np.full(2 * count + 3, 99, np.int64)
    index_backing[1 : 1 + 2 * count : 2] = indices
    selected_indices = mx.array(index_backing, index_dtype)[1 : 1 + 2 * count : 2]
    if layout == "offset_inner_stride":
        backing = np.full((count + 2, 2 * WIDTH + 9), 0x39A5, payload.dtype)
        backing[1 : count + 1, 3 : 3 + 2 * WIDTH : 2] = payload
        updates = mx.array(backing).view(dtype)[1 : count + 1, 3 : 3 + 2 * WIDTH : 2]
    elif layout == "transpose":
        backing = np.full((WIDTH, count + 2), 0x39A5, payload.dtype)
        backing.T[1 : count + 1] = payload
        updates = mx.array(backing).view(dtype).T[1 : count + 1]
    else:
        payload[:] = payload[:1]
        updates = mx.broadcast_to(mx.array(payload[0]).view(dtype), payload.shape)
    storage = CacheStorage((total, WIDTH), dtype)
    epoch = storage.write_slice(
        storage.initial_epoch(), 0, mx.array(original).view(dtype)
    )
    epoch = storage.scatter_rows(epoch, selected_indices, updates)
    expected = original.copy()
    for row, values in zip(indices, payload, strict=True):
        if 0 <= row < total:
            expected[row] = values
    np.testing.assert_array_equal(_snapshot_bits(storage, epoch, dtype), expected)
    _epoch_is(epoch, 2)
    assert storage.buffer_identity != 0 and storage.allocation_count == 1


@pytest.mark.parametrize(
    "dtype,index_dtype", [(mx.bfloat16, mx.int64), (mx.float32, mx.int32)]
)
def test_512_rows_all_out_of_bounds_still_advances_every_epoch_lane(dtype, index_dtype):
    storage = CacheStorage((4, WIDTH), dtype)
    epoch = storage.scatter_rows(
        storage.initial_epoch(),
        mx.array([-1, 4, 2**31 - 1], index_dtype),
        mx.array(_raw(3, dtype)).view(dtype),
    )
    expected = np.zeros((4, WIDTH), _unsigned(dtype)[0])
    np.testing.assert_array_equal(_snapshot_bits(storage, epoch, dtype), expected)
    _epoch_is(epoch, 1)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_512_rows_changed_counts_and_retained_snapshots_keep_one_buffer(dtype):
    storage = CacheStorage((2066, WIDTH), dtype)
    identity = storage.identity
    epoch = storage.initial_epoch()
    mx.eval(epoch)
    pointer = storage.buffer_identity
    expected = np.zeros((2066, WIDTH), _unsigned(dtype)[0])
    snapshots = []
    for step, count in enumerate((0, 1, 3, 2049, 0), 1):
        indices = np.arange(count, dtype=np.int32)
        updates = _raw(count, dtype, step * 31)
        epoch = storage.scatter_rows(
            epoch, mx.array(indices), mx.array(updates).view(dtype)
        )
        expected[indices] = updates
        snapshot = storage.snapshot(epoch)
        snapshots.append((snapshot, expected.copy()))
    # No explicit snapshot dependency: the public API must order lazy snapshots.
    mx.eval(epoch, [snapshot for snapshot, _ in snapshots])
    for snapshot, expected in snapshots:
        np.testing.assert_array_equal(
            np.asarray(snapshot.view(_unsigned(dtype)[1])), expected
        )
    _epoch_is(epoch, 5)
    assert storage.identity == identity
    assert storage.buffer_identity == pointer != 0 and storage.allocation_count == 1


def _integer_bits(values, dtype):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    # The small integer fixture is exactly representable in BF16 and FP32.
    return (
        np.right_shift(bits, 16).astype(np.uint16)
        if dtype == mx.bfloat16
        else bits.copy()
    )


@pytest.mark.parametrize(
    "dtype,index_dtype", [(mx.bfloat16, mx.int64), (mx.float32, mx.int32)]
)
def test_512_rows_lazy_snapshots_and_computed_read_write_chain(dtype, index_dtype):
    storage = CacheStorage((8, WIDTH), dtype)
    initial = (np.arange(8 * WIDTH).reshape(8, WIDTH) % 17 - 8).astype(np.float32)
    epoch = storage.write_slice(
        storage.initial_epoch(), 0, mx.array(_integer_bits(initial, dtype)).view(dtype)
    )
    values, snapshots, reads = [initial], [], []
    for indices, value in (([0, 6], -7), ([2, 7], 11)):
        snapshots.append(storage.snapshot(epoch))
        read = mx.sum(storage.read_epoch(epoch).astype(mx.float32), axis=-1)
        reads.append(read)
        changed = values[-1].copy()
        changed[indices] = value
        values.append(changed)
        epoch = storage.scatter_rows(
            mx.depends(epoch, [read]),
            mx.array(indices, index_dtype),
            mx.full((2, WIDTH), value, dtype),
        )
    snapshots.append(storage.snapshot(epoch))
    assert storage.allocation_count == storage.buffer_identity == 0
    mx.eval(epoch, snapshots, reads)
    for snapshot, expected in zip(snapshots, values, strict=True):
        np.testing.assert_array_equal(
            np.asarray(snapshot.view(_unsigned(dtype)[1])),
            _integer_bits(expected, dtype),
        )
    for read, expected in zip(reads, values[:-1], strict=True):
        np.testing.assert_array_equal(
            np.asarray(read), expected.sum(axis=-1, dtype=np.float32)
        )
    _epoch_is(epoch, 3)
    pointer = storage.buffer_identity
    # Keep every read and lazy snapshot alive across one more mutation.
    epoch = storage.scatter_rows(
        mx.depends(epoch, reads),
        mx.array([1], index_dtype),
        mx.full((1, WIDTH), -1, dtype),
    )
    final_values = values[-1].copy()
    final_values[1] = -1
    np.testing.assert_array_equal(
        _snapshot_bits(storage, epoch, dtype), _integer_bits(final_values, dtype)
    )
    _epoch_is(epoch, 4)
    assert storage.buffer_identity == pointer != 0 and storage.allocation_count == 1
    np.testing.assert_array_equal(
        np.asarray(snapshots[0].view(_unsigned(dtype)[1])),
        _integer_bits(initial, dtype),
    )


def test_512_rows_materialized_owner_and_inputs_work_on_fresh_thread():
    storage = CacheStorage((4, WIDTH), mx.bfloat16)
    epoch = storage.initial_epoch()
    raw = _raw(2, mx.bfloat16)
    indices, updates = mx.array([0, 3], mx.int32), mx.array(raw).view(mx.bfloat16)
    mx.eval(epoch, indices, updates)
    pointer = storage.buffer_identity

    def worker():
        next_epoch = storage.scatter_rows(epoch, indices, updates)
        expected = np.zeros((4, WIDTH), np.uint16)
        expected[[0, 3]] = raw
        np.testing.assert_array_equal(
            _snapshot_bits(storage, next_epoch, mx.bfloat16), expected
        )
        _epoch_is(next_epoch, 1)
        return storage.buffer_identity

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(worker).result(timeout=30) == pointer
    assert storage.allocation_count == 1
