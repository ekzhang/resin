"""Core contracts for Engine-owned mutable Metal cache storage."""

import gc
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal cache")

from resin_models_mlx._native import CacheStorage  # noqa: E402


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize(("row_bytes", "index_dtype"), [(3, mx.int32), (4, mx.int64)])
def test_cache_writes_preserve_bits_and_owner_lifetime(dtype, row_bytes, index_dtype):
    # Signed zeros, infinities and NaN payloads in both 16- and 32-bit views.
    bits = np.array(
        [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC10001, 0x7F810001], np.uint32
    )
    words = bits.view(np.uint16 if dtype == mx.bfloat16 else np.uint32)
    expected = np.zeros((12, row_bytes), np.uint8)
    storage = CacheStorage(expected.shape)
    pointer = storage.buffer_identity
    assert storage.nbytes == expected.nbytes
    payload = mx.array(bits).view(dtype)[::-1]
    epoch = storage.write_slice(storage.initial_epoch(), 3, payload)
    expected.ravel()[3 : 3 + bits.nbytes] = words[::-1].copy().view(np.uint8)
    np.testing.assert_array_equal(np.asarray(storage.read_epoch(epoch)), expected)

    # Exercise an unaligned input buffer and scatter to the last valid row.
    source = mx.arange(1 + 4 * row_bytes, dtype=mx.uint8)[1:]
    indices = mx.array([-1, 2, 12, 11], index_dtype)
    np.testing.assert_array_equal(np.asarray(epoch), np.ones(8, np.uint32))
    epoch = storage.scatter_rows(epoch, indices, source, row_bytes=row_bytes)
    rows = np.arange(1, 1 + 4 * row_bytes, dtype=np.uint8).reshape(4, row_bytes)
    expected[[2, 11]] = rows[[1, 3]]
    latest = storage.read_epoch(epoch)
    assert latest.dtype == mx.uint8
    with pytest.raises(ValueError, match="row_bytes"):
        storage.scatter_rows(epoch, indices, source, row_bytes=row_bytes + 1)
    with pytest.raises(ValueError, match="outside the allocation"):
        storage.write_slice(epoch, storage.nbytes - 1, source)
    mx.async_eval(latest)
    assert storage.buffer_identity == pointer != 0
    del storage, epoch
    gc.collect()
    np.testing.assert_array_equal(np.asarray(latest), expected)


@pytest.fixture
def arena():
    from resin_models_mlx.qwen3_5.cache import CacheArena, CacheGeometry

    # Each recurrent layer occupies exactly one 8 KiB unit; both convolution
    # tails fit in a third, padded unit. Two KV layers occupy each KV block.
    geometry = CacheGeometry(
        full_layers=2,
        kv_heads=1,
        head_dim=16,
        recurrent_shape=(2, 8, 16, 16),
        convolution_shape=(2, 3, 160),
        slab_units=6,
        state_units=3,
    )
    return CacheArena(geometry.layout(2), geometry)


@pytest.mark.parametrize("input_dtype", [mx.bfloat16, mx.float32])
def test_kv_append_preserves_plan_addresses_and_lazy_snapshot(arena, input_dtype):
    from resin.backend.plans import PagedCachePlan

    geometry = arena.geometry
    pointer = arena.buffer_identity
    before = arena.snapshot()
    slabs = np.array([1, 1, 2], np.int32)
    slots = np.array([0, 0, 1], np.int32)
    offsets = np.array([127, 0, 1], np.int32)
    group = SimpleNamespace(
        blocks_per_slab=geometry.kv_blocks_per_slab,
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
    for layer, (key, value) in enumerate(zip(keys, values, strict=True)):
        units = (
            slabs * geometry.slab_units + layer * geometry.kv_blocks_per_slab + slots
        )
        for start, data in [(0, key), (128, value)]:
            words = data.astype(mx.bfloat16).view(mx.uint16)
            expected[units, start + offsets] = np.asarray(words)[:, 0]
    np.testing.assert_array_equal(np.asarray(after.view(mx.uint16)), expected)
    assert not np.asarray(before.view(mx.uint16)).any()
    assert arena.buffer_identity == pointer != 0


def test_state_snapshots_restore_fp32_and_bf16_bits_across_slots(arena):
    from resin_models_mlx.qwen3_5.model import LinearState

    starts = [arena.state_start(slab, slot) for slab, slot in [(1, 0), (1, 1), (2, 0)]]
    rng = np.random.default_rng(57)

    def make_layers(shape, words, dtype):
        layers, *dims = shape
        bits = rng.integers(
            0, np.iinfo(words).max, (layers, len(starts), *dims), dtype=words
        )
        return tuple(mx.array(layer).view(dtype) for layer in bits)

    state = LinearState(
        make_layers(arena.geometry.recurrent_shape, np.uint32, mx.float32),
        make_layers(arena.geometry.convolution_shape, np.uint16, mx.bfloat16),
    )
    arena.write_states(starts, state)
    restored = arena.read_states(np.array(starts[::-1]))
    for group, original in zip(restored, state, strict=True):
        for result, layer in zip(group, original, strict=True):
            np.testing.assert_array_equal(
                np.asarray(result.view(mx.uint8)),
                np.asarray(layer.view(mx.uint8))[::-1],
            )


def test_real_prefill_49152_rows_preserves_every_row_and_column():
    count, width = 49_152, 512  # 4096 tokens × six layers × key/value.
    rows = np.arange(count, dtype=np.uint16)
    payload = np.bitwise_xor(rows[:, None], np.arange(width, dtype=np.uint16))
    indices = np.arange(count, dtype=np.int32)[::-1].copy()
    row_bytes = width * payload.itemsize
    storage = CacheStorage((count + 1, row_bytes))
    epoch = storage.scatter_rows(
        storage.initial_epoch(),
        mx.array(indices),
        mx.array(payload).view(mx.bfloat16),
        row_bytes=row_bytes,
    )
    actual = np.asarray(storage.read_epoch(epoch).view(mx.uint16))
    np.testing.assert_array_equal(actual[:count], payload[::-1])
    np.testing.assert_array_equal(actual[count], np.zeros(width, np.uint16))
    np.testing.assert_array_equal(np.asarray(epoch), np.full(8, 1, np.uint32))
