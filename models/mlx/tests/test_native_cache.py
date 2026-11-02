"""Independent contracts for explicitly ordered mutable Metal cache storage.

``read_epoch`` is private: an unversioned alias must not escape the reader.
Computed read outputs are dependencies of the next epoch before a mutation.
Public snapshots copy data, and an unevaluated snapshot also orders the next
mutation automatically.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    raise pytest.skip.Exception(
        "Native cache tests require Metal", allow_module_level=True
    )

from resin_models_mlx.native import CacheStorage  # noqa: E402


def _bits(value):
    unsigned = mx.uint16 if value.dtype == mx.bfloat16 else mx.uint32
    return np.asarray(value.view(unsigned))


def _assert_bits(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def _after_reads(epoch, *reads):
    return mx.depends(epoch, list(reads))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_storage_initialization_is_lazy_zeroed_and_has_one_stable_buffer(dtype):
    shape = (3, 4, 8)
    storage = CacheStorage(shape, dtype)
    identity = storage.identity
    assert identity != 0
    assert storage.nbytes == np.prod(shape) * (2 if dtype == mx.bfloat16 else 4)
    assert storage.buffer_identity == 0
    assert storage.allocation_count == 0
    epoch = storage.initial_epoch()
    assert epoch.shape == (8,) and epoch.dtype == mx.uint32
    assert storage.buffer_identity == 0
    mx.eval(epoch)
    pointer = storage.buffer_identity
    assert pointer != 0
    assert storage.allocation_count == 1
    first = storage.snapshot(epoch)
    _assert_bits(first, mx.zeros(shape, dtype=dtype))
    update = mx.arange(24).reshape(3, 8).astype(dtype)
    epoch = storage.scatter_rows(
        _after_reads(epoch, first), mx.array([0, 5, 11], mx.int32), update
    )
    latest = storage.snapshot(epoch)
    expected = mx.zeros((12, 8), dtype=dtype)
    expected[mx.array([0, 5, 11])] = update
    _assert_bits(latest, expected.reshape(shape))
    assert storage.identity == identity
    assert storage.buffer_identity == pointer
    assert storage.allocation_count == 1
    # A public snapshot is a separate value, never a live mutable cache alias.
    _assert_bits(first, mx.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_large_storage_geometry_does_not_allocate_or_truncate_byte_count(dtype):
    shape = (65_537, 65_536)
    storage = CacheStorage(shape, dtype)
    assert storage.shape == list(shape)
    assert storage.nbytes == 65_537 * 65_536 * (2 if dtype == mx.bfloat16 else 4)
    assert storage.nbytes > 8 << 30
    assert storage.identity != 0
    assert storage.buffer_identity == storage.allocation_count == 0
    # Deliberately do not request an epoch: this is a metadata-only regression
    # for larger Apple GPUs, with no multi-gigabyte allocation on the test host.


@pytest.mark.parametrize("shape", [(2**30, 2**30, 32), (2**30, 2**30, 8)])
def test_element_or_byte_count_overflow_is_rejected_before_allocation(shape):
    with pytest.raises((ValueError, RuntimeError), match="size_t|overflow"):
        CacheStorage(shape, mx.bfloat16)


@pytest.mark.parametrize(
    "dtype,index_dtype", [(mx.bfloat16, mx.int32), (mx.float32, mx.int64)]
)
def test_large_odd_payloads_preserve_all_bytes_for_zero_slice_scatter_and_snapshot(
    dtype, index_dtype
):
    # This exceeds the native dispatch's one-dimensional width but remains a
    # small 2–4 MiB resource. Every operation must cover its second grid row and
    # correctly guard its non-power-of-two tail.
    count = (1 << 20) + 37
    patch_count = (1 << 20) + 13
    raw_dtype = np.uint16 if dtype == mx.bfloat16 else np.uint32
    original_bits = (np.arange(count, dtype=np.uint32) * 17 + 0x3F00).astype(raw_dtype)
    patch_bits = (np.arange(patch_count, dtype=np.uint32) * 13 + 0x41A0).astype(
        raw_dtype
    )
    row_bits = (np.arange(count, dtype=np.uint32) * 7 + 0x3F40).astype(raw_dtype)
    storage = CacheStorage((count, 1), dtype)
    epoch = storage.initial_epoch()
    zero = storage.snapshot(epoch)
    epoch = storage.write_slice(epoch, 0, mx.array(original_bits).view(dtype))
    original = storage.snapshot(epoch)
    epoch = storage.write_slice(epoch, 11, mx.array(patch_bits).view(dtype))
    patched = storage.snapshot(epoch)
    epoch = storage.scatter_rows(
        epoch,
        mx.arange(count, dtype=index_dtype)[::-1],
        mx.array(row_bits).view(dtype).reshape(count, 1),
    )
    scattered = storage.snapshot(epoch)
    mx.eval(zero, original, patched, scattered, epoch)
    expected_patch = original_bits.copy()
    expected_patch[11 : 11 + patch_count] = patch_bits
    np.testing.assert_array_equal(_bits(zero).reshape(-1), np.zeros(count, raw_dtype))
    np.testing.assert_array_equal(_bits(original).reshape(-1), original_bits)
    np.testing.assert_array_equal(_bits(patched).reshape(-1), expected_patch)
    np.testing.assert_array_equal(_bits(scattered).reshape(-1), row_bits[::-1])
    assert storage.buffer_identity != 0 and storage.allocation_count == 1


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_queued_read_scatter_slice_read_preserves_all_earlier_results(dtype):
    shape = (12, 8)
    storage = CacheStorage(shape, dtype)
    expected = mx.arange(96).reshape(shape).astype(dtype)
    epoch = storage.write_slice(storage.initial_epoch(), 0, expected)
    reads, references = [], []
    for step in range(5):
        # The reduction has its own immutable output; do not retain a bare
        # read_epoch alias and then observe it after another cache mutation.
        read = mx.sum(storage.read_epoch(epoch).astype(mx.float32), axis=1)
        reference = mx.sum(expected.astype(mx.float32), axis=1)
        reads.append(read)
        references.append(reference)
        epoch = _after_reads(epoch, read)
        indices = mx.array([step, step + 6], mx.int64)
        update = mx.full((2, 8), -step - 1, dtype=dtype)
        epoch = storage.scatter_rows(epoch, indices, update)
        expected[indices] = update
        # Slice updates use flat element offsets, including offsets that are
        # neither a row boundary nor a 16-byte boundary.
        patch = mx.arange(7).astype(dtype) + step
        epoch = storage.write_slice(epoch, 3, patch)
        flat = expected.reshape(-1)
        flat[3:10] = patch
        expected = flat.reshape(shape)
    final = storage.snapshot(epoch)
    # Merely building the entire chain must not allocate/evaluate the resource.
    assert storage.buffer_identity == 0
    assert storage.allocation_count == 0
    mx.eval(final, *reads)
    for read, reference in zip(reads, references, strict=True):
        _assert_bits(read, reference)
    _assert_bits(final, expected)
    pointer = storage.buffer_identity
    assert pointer != 0
    assert storage.allocation_count == 1
    # Retain every earlier computed read while issuing another write. There
    # must be no second cache allocation due to these live output tensors.
    epoch = storage.write_slice(
        _after_reads(epoch, final, *reads), 0, mx.zeros(8, dtype)
    )
    _assert_bits(storage.snapshot(epoch)[0], mx.zeros(8, dtype))
    assert storage.buffer_identity == pointer
    assert storage.allocation_count == 1
    for read, reference in zip(reads, references, strict=True):
        _assert_bits(read, reference)


@pytest.mark.parametrize("write", ["scatter", "slice"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_unevaluated_snapshots_automatically_precede_the_next_mutation(write, dtype):
    storage = CacheStorage((4, 8), dtype)
    original = mx.arange(32).reshape(4, 8).astype(dtype)
    epoch = storage.write_slice(storage.initial_epoch(), 0, original)
    # Do not eval these snapshots or add them as explicit depends roots. The
    # public snapshot operation itself must preserve them before the next write.
    before = storage.snapshot(epoch)
    another_before = storage.snapshot(epoch)
    if write == "scatter":
        epoch = storage.scatter_rows(
            epoch, mx.array([0, 3], mx.int32), mx.full((2, 8), -7, dtype=dtype)
        )
        expected = mx.arange(32).reshape(4, 8).astype(dtype)
        expected[mx.array([0, 3])] = -7
    else:
        epoch = storage.write_slice(epoch, 7, mx.full(16, -7, dtype=dtype))
        expected = original.reshape(-1)
        expected[7:23] = -7
        expected = expected.reshape(4, 8)
    mx.eval(epoch)
    _assert_bits(before, original)
    _assert_bits(another_before, original)
    _assert_bits(storage.snapshot(epoch), expected)
    assert storage.allocation_count == 1


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_slice_updates_flatten_strided_inputs_in_logical_order(dtype):
    storage = CacheStorage((4, 16), dtype)
    backing = mx.arange(4 * 12).reshape(4, 12).astype(dtype)
    updates = backing[1::2, 1::2]
    epoch = storage.write_slice(storage.initial_epoch(), 3, updates)
    actual = storage.snapshot(epoch)
    expected = mx.zeros(64, dtype=dtype)
    expected[3:15] = updates.reshape(-1)
    _assert_bits(actual, expected.reshape(4, 16))


def test_scatter_payload_flattens_leading_dimensions_in_logical_order():
    storage = CacheStorage((8, 8), mx.bfloat16)
    updates = mx.arange(24).reshape(3, 2, 4).astype(mx.bfloat16)
    indices = mx.array([0, 3, 7], mx.int32)
    epoch = storage.scatter_rows(storage.initial_epoch(), indices, updates)
    expected = mx.zeros((8, 8), mx.bfloat16)
    expected[indices] = updates.reshape(3, 8)
    _assert_bits(storage.snapshot(epoch), expected)


def test_pending_operations_keep_resource_alive_after_python_owner_is_deleted():
    def build():
        storage = CacheStorage((8, 8), mx.float32)
        epoch = storage.write_slice(
            storage.initial_epoch(), 0, mx.arange(64, dtype=mx.float32)
        )
        before = storage.snapshot(epoch)
        epoch = storage.write_slice(
            _after_reads(epoch, before), 7, mx.full((2, 8), -3.0)
        )
        after = storage.snapshot(epoch)
        return before, after

    before, after = build()
    gc.collect()
    expected_before = mx.arange(64, dtype=mx.float32).reshape(8, 8)
    expected_after = mx.arange(64, dtype=mx.float32)
    expected_after[7:23] = -3.0
    _assert_bits(before, expected_before)
    _assert_bits(after, expected_after.reshape(8, 8))
    _assert_bits(before, expected_before)


@pytest.mark.parametrize("shape", [(), (0, 8), (-1, 8)])
def test_invalid_host_storage_geometry_is_rejected(shape):
    with pytest.raises((ValueError, RuntimeError)):
        CacheStorage(shape, mx.bfloat16)


@pytest.mark.parametrize("dtype", [mx.float16, mx.int32, mx.uint8])
def test_unsupported_storage_dtype_is_rejected(dtype):
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        CacheStorage((4, 8), dtype)


@pytest.mark.parametrize(
    "kind",
    [
        "row_rank",
        "row_type",
        "row_width",
        "row_count",
        "row_data_type",
        "slice_type",
        "slice_negative",
        "slice_end",
        "slice_overflow",
        "epoch_type",
        "epoch_short",
        "epoch_long",
        "epoch_rank",
    ],
)
def test_invalid_operations_are_rejected_before_device_execution(kind):
    storage = CacheStorage((4, 8), mx.bfloat16)
    epoch = storage.initial_epoch()
    indices, updates = mx.array([0, 1], mx.int32), mx.zeros((2, 8), mx.bfloat16)
    scatter, write, snapshot = (
        storage.scatter_rows,
        storage.write_slice,
        storage.snapshot,
    )
    invalid = {
        "row_rank": lambda: scatter(epoch, indices[None], updates),
        "row_type": lambda: scatter(epoch, indices.astype(mx.float32), updates),
        "row_width": lambda: scatter(epoch, indices, updates[:, :7]),
        "row_count": lambda: scatter(epoch, indices, updates[:1]),
        "row_data_type": lambda: scatter(epoch, indices, updates.astype(mx.float32)),
        "slice_type": lambda: write(epoch, 0, updates.astype(mx.float32)),
        "slice_negative": lambda: write(epoch, -1, updates[0]),
        "slice_end": lambda: write(epoch, 32, updates[0, :1]),
        "slice_overflow": lambda: write(epoch, 25, updates[0]),
        "epoch_type": lambda: snapshot(mx.zeros(8, mx.float32)),
        "epoch_short": lambda: snapshot(mx.zeros(7, mx.uint32)),
        "epoch_long": lambda: snapshot(mx.zeros(9, mx.uint32)),
        "epoch_rank": lambda: snapshot(mx.zeros((2, 4), mx.uint32)),
    }
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        invalid[kind]()
    assert storage.buffer_identity == storage.allocation_count == 0


def test_operations_reject_a_cpu_stream_before_allocating_the_resource():
    storage = CacheStorage((4, 8), mx.bfloat16)
    with mx.stream(mx.cpu):
        with pytest.raises((ValueError, RuntimeError), match="GPU"):
            storage.initial_epoch()
    assert storage.allocation_count == 0


@pytest.mark.parametrize("transform", ["compile", "grad", "vmap"])
def test_mutable_resource_operations_reject_function_transforms(transform):
    storage = CacheStorage((4, 8), mx.float32)

    def mutate(value):
        epoch = storage.write_slice(storage.initial_epoch(), 0, value)
        return mx.sum(storage.snapshot(epoch))

    values = mx.ones((4, 8))
    with pytest.raises((ValueError, RuntimeError), match="compile|grad|vmap|transform"):
        if transform == "compile":
            mx.compile(mutate)(values)
        elif transform == "grad":
            mx.grad(mutate)(values)
        else:
            mx.vmap(mutate)(mx.stack([values, values]))
    assert storage.allocation_count == 0


def _cache_geometry():
    from resin_models_mlx.qwen3_5.cache import CacheGeometry

    # Each recurrent layer occupies exactly one 8 KiB unit; both convolution
    # tails fit in a third, padded unit. Two KV layers occupy each KV block.
    return CacheGeometry(
        full_layers=2,
        kv_heads=1,
        head_dim=16,
        recurrent_shape=(2, 8, 16, 16),
        convolution_shape=(2, 3, 160),
        slab_units=4,
        state_units=3,
    )


@pytest.mark.parametrize("input_dtype", [mx.bfloat16, mx.float32])
def test_cache_arena_kv_append_preserves_plan_addresses_and_lazy_old_snapshot(
    input_dtype,
):
    from resin.backend.plans import PagedCachePlan
    from resin_models_mlx.qwen3_5.cache import CacheArena

    geometry = _cache_geometry()
    arena = CacheArena(geometry.layout(2), geometry)
    pointer = arena.buffer_identity
    assert pointer != 0 and arena.allocation_count == 1
    before = arena.snapshot()
    slabs = np.array([1, 1, 2], np.int32)
    slots = np.array([0, 0, 1], np.int32)
    offsets = np.array([127, 0, 1], np.int32)
    group = SimpleNamespace(
        append_slab_ids=slabs,
        append_slotnums=slots,
        append_block_offsets=offsets,
    )
    plan = cast(PagedCachePlan, SimpleNamespace(groups=[group]))
    keys = tuple(
        (mx.arange(48).reshape(3, 1, 16) / 7 + 100 * layer).astype(input_dtype)
        for layer in range(2)
    )
    values = tuple((key + 500).astype(input_dtype) for key in keys)
    arena.write_tokens(plan, keys, values, np.arange(3))
    after = arena.snapshot()
    mx.eval(arena.epoch, before, after, keys, values)
    expected = np.zeros(after.shape, np.uint16)
    for token in range(3):
        for layer in range(2):
            unit = (
                slabs[token] * geometry.slab_units
                + layer * geometry.kv_blocks_per_slab
                + slots[token]
            )
            expected[unit, offsets[token]] = _bits(
                keys[layer][token].astype(mx.bfloat16)
            ).reshape(16)
            expected[unit, 128 + offsets[token]] = _bits(
                values[layer][token].astype(mx.bfloat16)
            ).reshape(16)
    np.testing.assert_array_equal(_bits(after), expected)
    np.testing.assert_array_equal(_bits(before), np.zeros(before.shape, np.uint16))
    # In particular, the reserved null slab and unrelated units remain zero.
    assert not _bits(after)[:4].any()
    assert arena.buffer_identity == pointer and arena.allocation_count == 1


def test_cache_arena_state_checkpoint_is_an_exact_copy_and_restored_reads_are_ordered():
    from resin_models_mlx.qwen3_5.cache import CacheArena
    from resin_models_mlx.qwen3_5.model import LinearState

    geometry = _cache_geometry()
    arena = CacheArena(geometry.layout(2), geometry)
    pointer = arena.buffer_identity

    def state(bias):
        return LinearState(
            tuple(
                mx.arange(2048, dtype=mx.float32).reshape(1, 8, 16, 16) / 4
                + bias
                + layer
                for layer in range(2)
            ),
            tuple(
                (mx.arange(480).reshape(1, 3, 160) / 8 + bias + layer).astype(
                    mx.bfloat16
                )
                for layer in range(2)
            ),
        )

    original = state(0)
    before = arena.snapshot()
    arena.write_states([4], original)
    packed = arena.snapshot()
    restored = arena.read_states(np.array([4], np.int64))
    read_roots = (*restored.recurrent, *restored.convolution)
    # This is the same public ownership boundary used for model read roots:
    # queue their completion before publishing a replacement checkpoint.
    arena.order_after(read_roots)
    replacement = state(17)
    arena.write_states([4], replacement)
    latest = arena.read_states(np.array([4], np.int64))
    mx.eval(
        arena.epoch, before, packed, read_roots, latest.recurrent, latest.convolution
    )
    for actual, expected in zip(
        read_roots, (*original.recurrent, *original.convolution), strict=True
    ):
        _assert_bits(actual, expected)
    for actual, expected in zip(
        (*latest.recurrent, *latest.convolution),
        (*replacement.recurrent, *replacement.convolution),
        strict=True,
    ):
        _assert_bits(actual, expected)
    expected_packed = np.zeros(packed.shape, np.uint16)
    for layer, recurrent in enumerate(original.recurrent):
        expected_packed[4 + layer] = (
            np.asarray(recurrent).view(np.uint16).reshape(256, 16)
        )
    tails = np.concatenate([_bits(value).reshape(-1) for value in original.convolution])
    expected_packed[6].reshape(-1)[: len(tails)] = tails
    np.testing.assert_array_equal(_bits(packed), expected_packed)
    np.testing.assert_array_equal(_bits(before), np.zeros(before.shape, np.uint16))
    assert arena.buffer_identity == pointer and arena.allocation_count == 1
