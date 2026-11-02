"""Core contracts for Engine-owned mutable Metal cache storage."""

import gc
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal cache")

from resin_models_mlx._native import CacheStorage  # noqa: E402


def _bits(value):
    return np.asarray(
        value.view(mx.uint16 if value.dtype == mx.bfloat16 else mx.uint32)
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("index_dtype", [mx.int32, mx.int64])
def test_cache_writes_preserve_bits_lazy_snapshots_and_owner_lifetime(
    dtype, index_dtype
):
    unsigned = np.uint16 if dtype == mx.bfloat16 else np.uint32
    fraction = 7 if dtype == mx.bfloat16 else 23
    sign, infinity = 1 << (fraction + 8), 255 << fraction
    raw = np.arange(1024, dtype=unsigned).reshape(2, 512)
    raw[:, :8] = [
        0,
        sign,
        infinity,
        sign | infinity,
        infinity | (1 << (fraction - 1)) | 1,
        infinity | 1,
        1,
        infinity - 1,
    ]
    storage = CacheStorage((6, 512), dtype)
    epoch = storage.initial_epoch()
    assert storage.buffer_identity != 0 and storage.allocation_count == 1
    epoch = storage.write_slice(epoch, 3, mx.array(raw).view(dtype))
    before = storage.snapshot(epoch)
    updates = np.stack([raw[0], raw[1], raw[0], raw[0]])
    epoch = storage.scatter_rows(
        epoch, mx.array([-1, 1, 6, 4], index_dtype), mx.array(updates).view(dtype)
    )
    after = storage.snapshot(epoch)
    assert storage.buffer_identity != 0 and storage.allocation_count == 1
    # Both snapshots are still lazy; the write must preserve the earlier one.
    expected = np.zeros(6 * 512, unsigned)
    expected[3 : 3 + raw.size] = raw.reshape(-1)
    expected = expected.reshape(6, 512)
    mx.eval(epoch, before, after)
    np.testing.assert_array_equal(_bits(before), expected)
    expected[[1, 4]] = updates[[1, 3]]
    np.testing.assert_array_equal(_bits(after), expected)
    expected_after = expected.copy()
    np.testing.assert_array_equal(np.asarray(epoch), np.full(8, 2, np.uint32))
    assert storage.buffer_identity != 0 and storage.allocation_count == 1
    pointer = storage.buffer_identity
    epoch = storage.write_slice(epoch, 11, mx.zeros(7, dtype))
    latest = storage.snapshot(epoch)
    mx.async_eval(latest)
    assert storage.buffer_identity == pointer and storage.allocation_count == 1
    del storage, epoch
    gc.collect()
    expected.reshape(-1)[11:18] = 0
    np.testing.assert_array_equal(_bits(latest), expected)
    # Holding the old immutable snapshots cannot turn them into arena aliases.
    expected_before = np.zeros(6 * 512, unsigned)
    expected_before[3 : 3 + raw.size] = raw.reshape(-1)
    np.testing.assert_array_equal(_bits(before), expected_before.reshape(6, 512))
    np.testing.assert_array_equal(_bits(after), expected_after)


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
