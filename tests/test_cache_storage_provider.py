from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resin import CacheGroupLayout, CacheLayout
from resin.backend.cuda.paged_kv import PagedKvArena
from resin.cache_storage import CacheBuffer, MemoryCacheStorage


def transfer(*groups):
    return SimpleNamespace(
        resident_tokens=128,
        groups=tuple(
            SimpleNamespace(
                group_index=index, first_block=7, slab_ids=slabs, slotnums=slots
            )
            for index, slabs, slots in groups
        ),
    )


def test_memory_storage_owns_snapshots_and_repartitions_without_overwriting():
    provider = MemoryCacheStorage(max_bytes=16)
    source = bytearray(b"abcd")
    source_tensor = torch.tensor([5, 6], dtype=torch.uint8)
    operation = provider.store(
        (b"prefix",),
        (
            CacheBuffer(
                (
                    memoryview(source)[:2],
                    np.frombuffer(source, np.uint8)[2:],
                    source_tensor,
                ),
                6,
                "cpu",
            ),
        ),
    )
    source[:] = b"xxxx"
    source_tensor.zero_()
    assert operation.wait().success == (True,)
    assert operation.poll() is operation.wait()
    # A second writer never changes an existing immutable key.
    assert provider.store(
        (b"prefix",), (CacheBuffer((b"zzzzzz",), 6, "cpu"),)
    ).wait().success == (True,)
    left, right = bytearray(3), torch.zeros(3, dtype=torch.uint8)
    loaded = provider.load((b"prefix",), (CacheBuffer((left, right), 6, "cpu"),)).wait()
    assert loaded.success == (True,) and loaded.error is None
    assert left == b"abc" and right.tolist() == [100, 5, 6]


def test_memory_storage_bounds_lru_and_leaves_missing_destinations_untouched():
    provider = MemoryCacheStorage(max_bytes=8)

    def put(key, payload):
        return provider.store(
            (key,), (CacheBuffer((payload,), len(payload), "cpu"),)
        ).wait()

    assert put(b"a", b"aaaa").success == put(b"b", b"bbbb").success == (True,)
    lookup = provider.lookup((b"a",)).poll()
    assert lookup is not None and lookup.success == (True,)
    assert put(b"c", b"cccc").success == (True,)
    assert provider.nbytes == 8
    assert put(b"large", b"x" * 9).success == (False,)
    assert provider.lookup((b"a", b"b", b"c")).wait().success == (True, False, True)
    destinations = (bytearray(b"????"), bytearray(b"???"), bytearray(b"????"))
    result = provider.load(
        (b"a", b"missing", b"c"),
        tuple(CacheBuffer((value,), len(value), "cpu") for value in destinations),
    ).wait()
    assert result.success == (True, False, True)
    assert destinations == (bytearray(b"aaaa"), bytearray(b"???"), bytearray(b"cccc"))
    wrong_size = bytearray(b"???")
    assert provider.load(
        (b"a",), (CacheBuffer((wrong_size,), 3, "cpu"),)
    ).wait().success == (False,)
    assert wrong_size == b"???"
    with pytest.raises(ValueError, match="sum to nbytes"):
        CacheBuffer((bytearray(2),), 3, "cpu")
    with pytest.raises(ValueError, match="writable"):
        provider.load((b"a",), (CacheBuffer((b"xxxx",), 4, "cpu"),))


def test_glm_and_deepseek_expose_exact_borrowed_children_and_stable_signatures():
    from resin_models_cuda.deepseek_v41.cache import (
        CacheArena as DeepArena,
    )
    from resin_models_cuda.deepseek_v41.cache import (
        CacheGeometry as DeepGeometry,
    )
    from resin_models_cuda.glm5_3_flash.cache import CacheArena, CacheGeometry

    geometry = CacheGeometry(
        ("deepseek_sparse_attention", "linear_attention"), 4, 4, 4, 1, 2, 1, 64
    )
    arena = CacheArena("cpu", geometry.layout(3), geometry, max_batch_size=2)
    buffers = arena.cache_storage_buffers(transfer((0, [1], [0]), (1, [2, 2], [2, 3])))
    assert [buffer.nbytes for buffer in buffers] == [
        geometry.slab_bytes,
        geometry.state_bytes,
        geometry.state_bytes,
    ]
    assert geometry.state_bytes != geometry.slab_bytes // geometry.packings[1]
    region = buffers[2].regions[0]
    assert (
        region.data_ptr() - arena.storage.data_ptr()
        == 2 * geometry.slab_bytes + 3 * geometry.state_bytes
    )
    region.fill_(19)
    assert (
        arena.storage[2, 3 * geometry.state_bytes : 4 * geometry.state_bytes]
        .eq(19)
        .all()
    )
    assert arena.storage[2, : 3 * geometry.state_bytes].eq(0).all()
    other = CacheArena("cpu", geometry.layout(5), geometry, max_batch_size=1)
    assert other.cache_storage_signature == arena.cache_storage_signature
    with pytest.raises(ValueError, match="outside"):
        arena.cache_storage_buffers(transfer((0, [0], [0])))

    deep_geometry = DeepGeometry(
        1, (0,), (8,), (0,), (16,), (0,), (1, 2, 4, 8), 1024, 128, 128, 0
    )
    deep = DeepArena(
        "cpu", deep_geometry.layout(4), deep_geometry, dtype=torch.bfloat16
    )
    child = deep.cache_storage_buffers(transfer((3, [4], [7])))[0]
    assert child.nbytes == 128
    child.regions[0].fill_(23)
    assert deep.storage[4, 7 * 128 :].eq(23).all()
    assert (
        child.regions[0].untyped_storage().data_ptr()
        == deep.storage.untyped_storage().data_ptr()
    )


def test_generic_paged_arena_keeps_kv_layer_regions_borrowed():
    layout = CacheLayout(parent_capacity=2, groups=[CacheGroupLayout(4)])
    arena = PagedKvArena(
        "cpu",
        layout,
        layer_count=2,
        page_size=4,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.bfloat16,
    )
    buffer = arena.cache_storage_buffers(transfer((0, [1], [0])))[0]
    assert len(buffer.regions) == 4 and buffer.nbytes == 64
    for index, region in enumerate(buffer.regions):
        assert (
            region.untyped_storage().data_ptr()
            == arena.storage.untyped_storage().data_ptr()
        )
        region.fill_(index + 1)
    assert arena.storage[:, :, 0].eq(0).all()
    assert arena.storage[:, :, 2].eq(0).all()
    assert not arena.storage[:, :, 1].eq(0).any()


@pytest.mark.parametrize(
    "dtypes",
    [(torch.float16, torch.bfloat16), (torch.float8_e4m3fn, torch.float8_e5m2)],
)
def test_qwen_storage_signature_distinguishes_same_size_kv_dtypes(dtypes):
    from resin_models_cuda.qwen3_5.cache import CacheArena, CacheGeometry

    geometry = CacheGeometry(
        full_layers=1,
        linear_layers=1,
        kv_heads=1,
        head_dim=2,
        linear_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        conv_dim=2,
        conv_state_length=1,
        hidden_size=4,
        include_mtp=False,
        kv_element_bytes=dtypes[0].itemsize,
        slab_units=1,
        target_blocks_per_slab=1,
        state_blocks_per_slab=512 * dtypes[0].itemsize // 28,
        mtp_blocks_per_slab=1,
    )
    arena, different_dtype = (
        CacheArena("cpu", geometry.layout(2), geometry, kv_dtype=dtype)
        for dtype in dtypes
    )
    assert arena.cache_storage_block_bytes == different_dtype.cache_storage_block_bytes
    assert arena.cache_storage_signature != different_dtype.cache_storage_signature
    larger = CacheArena("cpu", geometry.layout(3), geometry, kv_dtype=dtypes[0])
    assert larger.cache_storage_signature == arena.cache_storage_signature


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_gemma_arena_exposes_global_children_and_local_parent_without_copies(dtype):
    from resin.models.gemma4.plan import PAGE_SIZE
    from resin_models_cuda.gemma4.cuda import _Gemma4KvArena

    config = SimpleNamespace(
        local_layer_count=2,
        num_key_value_heads=1,
        head_dim=4,
        global_layer_count=1,
        num_global_key_value_heads=1,
        global_head_dim=4,
        history_blocks_per_parent=2,
    )
    groups = [
        CacheGroupLayout(PAGE_SIZE, blocks_per_slab=2),
        CacheGroupLayout(PAGE_SIZE),
    ]
    arena = _Gemma4KvArena(
        "cpu",
        config,
        3,
        dtype=dtype,
        layout=CacheLayout(parent_capacity=3, groups=groups),
    )
    raw = arena.storage.view(torch.uint8)
    raw.zero_()
    global_block, local_block = arena.cache_storage_buffers(
        transfer((0, [1], [1]), (1, [2], [0]))
    )
    assert global_block.nbytes * 2 == local_block.nbytes
    for buffer, value in ((global_block, 17), (local_block, 29)):
        region = buffer.regions[0]
        assert (
            region.untyped_storage().data_ptr()
            == arena.storage.untyped_storage().data_ptr()
        )
        region.fill_(value)
    assert arena.global_keys[:, 3].view(torch.uint8).eq(17).all()
    assert arena.global_values[:, 3].view(torch.uint8).eq(17).all()
    assert arena.global_keys[:, 2].view(torch.uint8).eq(0).all()
    assert arena.local_keys[:, 2].view(torch.uint8).eq(29).all()
    assert arena.local_values[:, 2].view(torch.uint8).eq(29).all()
    assert raw[0].eq(0).all()
    larger = _Gemma4KvArena(
        "cpu",
        config,
        5,
        dtype=dtype,
        layout=CacheLayout(parent_capacity=5, groups=groups),
    )
    assert larger.cache_storage_signature == arena.cache_storage_signature


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_memory_storage_cuda_operations_retain_staging_until_retirement(monkeypatch):
    provider = MemoryCacheStorage(max_bytes=16)
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        source = torch.empty(8, dtype=torch.uint8, device="cuda")
        torch.cuda._sleep(10_000_000)
        source.copy_(torch.arange(8, dtype=torch.uint8, device="cuda"))
        stored = provider.store(
            (b"gpu",), (CacheBuffer((source,), 8, str(source.device)),)
        )
    assert provider._copy_streams[str(source.device)] != producer
    assert stored.wait().success == (True,)
    # A source is borrowed until its copy retires, even on a different stream.
    source.fill_(99)
    with torch.cuda.stream(producer):
        destination = torch.empty_like(source)
        torch.cuda._sleep(10_000_000)
        destination.fill_(55)
        loaded = provider.load(
            (b"gpu",), (CacheBuffer((destination,), 8, str(source.device)),)
        )
    oversized = torch.empty(17, dtype=torch.uint8, device="cuda")
    with monkeypatch.context() as patch:

        def forbid_staging(*args, **kwargs):
            raise AssertionError(
                "existing or oversized objects must not allocate staging"
            )

        patch.setattr(torch, "empty_like", forbid_staging)
        assert provider.store(
            (b"gpu",), (CacheBuffer((source,), 8, str(source.device)),)
        ).wait().success == (True,)
        assert provider.store(
            (b"large",), (CacheBuffer((oversized,), 17, str(source.device)),)
        ).wait().success == (False,)
    # Eviction after load submission cannot invalidate the load's source snapshot.
    provider.store((b"other",), (CacheBuffer((b"x" * 16,), 16, "cpu"),)).wait()
    assert loaded.wait().success == (True,)
    assert loaded.poll() is loaded.wait()
    assert destination.cpu().tolist() == list(range(8))
    destination.fill_(77)  # Reuse is safe only after the load retires.
