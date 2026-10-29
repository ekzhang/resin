"""Exact state transfer and graph ownership for the Engine-owned GLM arena."""

import gc
import weakref
from dataclasses import replace

import pytest
import torch

from resin.backend.cuda.slab import DevicePageGroup
from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_cuda.glm5_3_flash.cache import CacheArena, CacheGeometry

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _geometry():
    return CacheGeometry(
        ("linear_attention", "deepseek_sparse_attention", "linear_attention"),
        hidden_size=32,
        latent_dim=8,
        index_dim=4,
        heads=2,
        head_dim=8,
        conv_history=3,
        state_tokens=64,
        include_mtp=True,
    )


def _fixture(device, batch, *, width=3, first=0, geometry=None, capacity=None):
    geometry = geometry or _geometry()
    parents_per_row = (width + geometry.packings[1] - 1) // geometry.packings[1]
    arena = CacheArena(
        device,
        geometry.layout(capacity or batch * parents_per_row + 3),
        geometry,
        max_batch_size=batch,
    )
    arena.storage.random_(0, 256)
    # Logical checkpoints need distinct (parent,slot) pairs even when the
    # bounded-tail layout leaves only one state slot per physical slab.
    rows = torch.arange(batch, device=device, dtype=torch.int32)
    columns = torch.arange(width, device=device, dtype=torch.int32)
    parents = (
        1 + rows[:, None] * parents_per_row + columns[None] // geometry.packings[1]
    ).contiguous()
    slots = (columns % geometry.packings[1])[None].expand(batch, width).contiguous()
    group = DevicePageGroup(
        parents, slots, torch.full((batch,), first, device=device, dtype=torch.int32)
    )
    return arena, arena._make_graph_cache((group, group))


def _packed_bytes(value):
    # A one-element logical view can be reported contiguous while retaining a
    # non-unit last stride. dtype reinterpretation still requires stride 1.
    # Avoid copying already packed arenas: wide-address tests exceed 8 GiB.
    if not value.is_contiguous() or (value.ndim and value.stride(-1) != 1):
        packed = torch.empty(value.shape, dtype=value.dtype, device=value.device)
        packed.copy_(value)
        value = packed
    if value.ndim == 0:
        value = value.reshape(1)
    assert value.stride(-1) == 1
    return value.view(torch.uint8).reshape(-1)


def _bitwise_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    # CUDA equal materializes a bool per compared byte. Bound this temporary,
    # including for the >8GiB physical-address oracle; never compare an entire
    # arena in one torch.equal call.
    actual = _packed_bytes(actual)
    expected = _packed_bytes(expected)
    chunk = 16 * 2**20
    for start in range(0, actual.numel(), chunk):
        assert torch.equal(
            actual[start : start + chunk], expected[start : start + chunk]
        )


def _reference_gather(cache, contexts):
    recurrent, convolution, hidden = cache.state_at((contexts - 1).clamp_min(0))
    new = contexts == 0
    recurrent.masked_fill_(new[:, None, None, None, None], 0)
    convolution.masked_fill_(new[:, None, None, None, None], 0)
    hidden.masked_fill_(new[:, None], 0)
    return recurrent, convolution, hidden


def _on_storage(cache, storage):
    def view(original):
        return storage.view(original.dtype).as_strided(
            original.shape, original.stride(), original.storage_offset()
        )

    return replace(
        cache,
        recurrent_states=view(cache.recurrent_states),
        convolution_states=view(cache.convolution_states),
        target_hidden_states=view(cache.target_hidden_states),
        index_tail_keys=view(cache.index_tail_keys),
        index_tail_gates=view(cache.index_tail_gates),
        index_tail_ends=view(cache.index_tail_ends),
    )


def _reference_tails(cache, contexts):
    from resin_models_cuda.glm5_3_flash.index_tail import IndexTail

    parents, slots, _ = cache.locations(
        1,
        (contexts - 1).clamp_min(0)[:, None],
        block_tokens=cache.geometry.state_tokens,
    )
    indices = parents[:, 0].long(), slots[:, 0].long()
    keys, gates, ends = (
        cache.index_tail_keys[indices],
        cache.index_tail_gates[indices],
        cache.index_tail_ends[indices],
    )
    fresh = (contexts == 0) | (parents[:, 0] == 0)
    keys[fresh], gates[fresh], ends[fresh] = 0, 0, -1
    return [
        IndexTail(
            keys[:, i].contiguous(), gates[:, i].contiguous(), ends[:, i].contiguous()
        )
        for i in range(len(cache.geometry.full_layers))
    ]


def _reference_store_tails(cache, positions, tails):
    parents, slots, _ = cache.locations(
        1, positions[:, None], block_tokens=cache.geometry.state_tokens
    )
    valid = parents[:, 0] > 0
    indices = parents[valid, 0].long(), slots[valid, 0].long()
    cache.index_tail_keys[indices] = torch.stack([t.keys for t in tails], 1)[valid]
    cache.index_tail_gates[indices] = torch.stack([t.gates for t in tails], 1)[valid]
    cache.index_tail_ends[indices] = torch.stack([t.end for t in tails], 1)[valid]


def _batch(cache, contexts, lengths):
    # Host construction is test metadata preparation, outside CUDA capture.
    device = contexts.device
    counts = torch.tensor(lengths, dtype=torch.int32, device=device)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0).int()))
    positions = torch.cat(
        [
            contexts[row] + torch.arange(length, device=device)
            for row, length in enumerate(lengths)
        ]
    )
    sequences = torch.repeat_interleave(
        torch.arange(len(lengths), device=device), counts
    )
    return cache.batch(positions, sequences, offsets, contexts)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_batch_cache_owns_initial_states_and_preserves_published_prefix(device):
    arena, cache = _fixture(device, 3)
    contexts = torch.tensor([0, 64, 65], device=device, dtype=torch.int64)
    initial = _reference_gather(cache, contexts)
    before = arena.storage.clone()
    batch = _batch(cache, contexts, [3, 1, 0])
    assert not batch.is_decode
    batch.is_decode = True
    saved = []
    for i, layer_id in enumerate(cache.geometry.linear_layers):
        layer = batch.layer(layer_id)
        _bitwise_equal(layer.recurrent, initial[0][:, i])
        _bitwise_equal(layer.convolution, initial[1][:, i])
        saved.append((layer.recurrent, layer.convolution))
        # Distinct allocations model actual KDA/convolution final outputs.
        layer.recurrent = layer.recurrent.clone().contiguous()
        layer.convolution = layer.convolution.clone().contiguous()
        layer.recurrent[:2].view(torch.uint8).bitwise_xor_(i + 1)
        layer.convolution[:2].view(torch.uint8).bitwise_xor_(i + 3)
        if device == "cuda":
            assert saved[-1][0].is_contiguous() and saved[-1][1].is_contiguous()
    hidden = batch.previous_hidden.clone()
    hidden[:2].view(torch.uint8).bitwise_xor_(19)
    expected = before.clone()
    reference = _on_storage(cache, expected)
    reference.store_state(
        batch.lengths - 1,
        torch.stack(
            [batch.layer(i).recurrent for i in cache.geometry.linear_layers], 1
        ),
        torch.stack(
            [batch.layer(i).convolution for i in cache.geometry.linear_layers], 1
        ),
        hidden,
    )
    _reference_store_tails(
        reference,
        batch.lengths - 1,
        [batch.layer(i).index_tail for i in cache.geometry.full_layers],
    )
    batch.store(hidden)
    _bitwise_equal(arena.storage, expected)
    for i, (recurrent, convolution) in enumerate(saved):
        _bitwise_equal(recurrent, initial[0][:, i])
        _bitwise_equal(convolution, initial[1][:, i])
        assert (
            recurrent.untyped_storage().data_ptr()
            != arena.storage.untyped_storage().data_ptr()
        )
    _bitwise_equal(batch.previous_hidden, initial[2])
    # Row1 reads published block0 but writes continuation block1. The old state
    # remains unchanged. Row2 is idle and must preserve its current checkpoint.
    old = _on_storage(cache, before).state_at(torch.tensor([0, 63, 64], device=device))
    current = cache.state_at(torch.tensor([0, 63, 64], device=device))
    for a, b in zip(current, old, strict=True):
        _bitwise_equal(a[1:], b[1:])


@CUDA
@pytest.mark.parametrize("live_state", [False, True])
def test_batch_cache_graph_pool_lifetime_and_changed_page_metadata(live_state):
    arena, cache = _fixture("cuda", 4, first=7)
    cache = replace(
        cache, groups=(cache.groups[0], replace(cache.groups[1], live_state=live_state))
    )
    contexts = torch.tensor([0, 449, 512, 448], device="cuda", dtype=torch.int64)
    # A zero-token row exercises scatter position -1 and null storage. Others
    # have disjoint destinations; conflicting null writers are not a valid lease.
    offsets = torch.tensor([0, 0, 1, 2, 3], device="cuda", dtype=torch.int32)
    positions = contexts[1:].clone()
    sequences = torch.arange(1, 4, device="cuda")
    expected_shape = _reference_gather(cache, contexts)
    observed = tuple(torch.empty_like(t) for t in expected_shape)
    hidden = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")
    retired = []

    def call():
        batch = cache.batch(positions, sequences, offsets, contexts)
        retired.append(weakref.ref(batch))
        initial = []
        for i, layer_id in enumerate(cache.geometry.linear_layers):
            layer = batch.layer(layer_id)
            retired.append(weakref.ref(layer))
            initial.append((layer.recurrent, layer.convolution))
            layer.recurrent = layer.recurrent.clone()
            layer.convolution = layer.convolution.clone()
            layer.recurrent.view(torch.uint8).bitwise_xor_(i + 1)
            layer.convolution.view(torch.uint8).bitwise_xor_(i + 3)
        batch.store(hidden)
        # Read initial states after final commit, as speculative rollback does.
        for i, (r, c) in enumerate(initial):
            observed[0][:, i].copy_(r)
            observed[1][:, i].copy_(c)
        observed[2].copy_(batch.previous_hidden)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    gc.collect()
    assert all(ref() is None for ref in retired)
    pressure = [
        torch.empty(16 * 2**20, device="cuda", dtype=torch.uint8) for _ in range(4)
    ]
    for step in range(4):
        group = cache.groups[1]
        group.slab_ids.copy_(torch.roll(group.slab_ids, 1, 0))
        group.slotnums.add_(1).remainder_(cache.geometry.packings[1])
        contexts.copy_(
            torch.tensor(
                [0, 449 + step, 512 + step, 448 if step == 0 else 576 + step],
                device="cuda",
            )
        )
        positions.copy_(contexts[1:])
        hidden.view(torch.uint8).bitwise_xor_(step + 1)
        initial = _reference_gather(cache, contexts)
        expected = arena.storage.clone()
        r, c = initial[0].clone(), initial[1].clone()
        for i in range(2):
            r[:, i].view(torch.uint8).bitwise_xor_(i + 1)
            c[:, i].view(torch.uint8).bitwise_xor_(i + 3)
        _on_storage(cache, expected).store_state(
            contexts + offsets[1:] - offsets[:-1] - 1, r, c, hidden
        )
        _reference_store_tails(
            _on_storage(cache, expected),
            contexts + offsets[1:] - offsets[:-1] - 1,
            _reference_tails(cache, contexts),
        )
        for allocation in pressure:
            allocation.fill_(step)
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(observed, initial, strict=True):
            _bitwise_equal(actual, reference)
        _bitwise_equal(arena.storage, expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_batch_cache_shifted_window_null_lookup_and_negative_idle_position(device):
    arena, cache = _fixture(device, 3, first=7)
    contexts = torch.tensor([0, 448, 512], device=device, dtype=torch.int64)
    initial = _reference_gather(cache, contexts)
    batch = _batch(cache, contexts, [0, 1, 0])
    assert batch.lengths.tolist() == [0, 449, 512]
    expected = arena.storage.clone()
    for i, layer_id in enumerate(cache.geometry.linear_layers):
        _bitwise_equal(batch.layer(layer_id).recurrent, initial[0][:, i])
        _bitwise_equal(batch.layer(layer_id).convolution, initial[1][:, i])
    # Only row0 writes the reserved null location. The others own distinct
    # parents, including row1's transition from an absent previous checkpoint.
    _on_storage(cache, expected).store_state(batch.lengths - 1, *initial)
    _reference_store_tails(
        _on_storage(cache, expected),
        batch.lengths - 1,
        [batch.layer(i).index_tail for i in cache.geometry.full_layers],
    )
    batch.store(batch.previous_hidden)
    _bitwise_equal(arena.storage, expected)


@CUDA
def test_gather_rejects_arena_and_shared_output_allocations():
    from resin_models_cuda.glm5_3_flash.cache_cuda import gather_state_layer_major

    arena, cache = _fixture("cuda", 1)
    contexts = torch.ones(1, device="cuda", dtype=torch.int64)
    alias = (
        cache.recurrent_states[1, 0].unsqueeze(1),
        cache.convolution_states[1, 0].unsqueeze(1),
        cache.target_hidden_states[1, 0].unsqueeze(0),
    )
    with pytest.raises(ValueError, match="Engine-owned"):
        gather_state_layer_major(cache, contexts, out=alias)
    owned = gather_state_layer_major(cache, contexts)
    sizes = [t.numel() * t.element_size() for t in owned]
    shared = torch.empty(sum(sizes), device="cuda", dtype=torch.uint8)
    split = shared.split(sizes)
    aliases = tuple(
        part.view(t.dtype).reshape(t.shape)
        for part, t in zip(split, owned, strict=True)
    )
    with pytest.raises(ValueError, match="independent storage"):
        gather_state_layer_major(cache, contexts, out=aliases)


@CUDA
@pytest.mark.parametrize("mtp", [False, True])
def test_state_transfer_wide_physical_addresses(mtp):
    if torch.cuda.mem_get_info()[0] < 20 * 2**30:
        pytest.skip("high physical-address fixture requires 20 GiB free CUDA memory")  # ty: ignore[too-many-positional-arguments]
    from resin_models_cuda.glm5_3_flash.cache_cuda import (
        gather_state_layer_major,
        scatter_state_layers,
    )

    geometry = CacheGeometry.from_config(
        Glm53Config(),
        state_tokens=512,
        include_mtp=mtp,
        latent_dtype=torch.bfloat16,
    )
    parent = 2**31 // (geometry.slab_bytes // 4) + 2
    arena, cache = _fixture("cuda", 1, width=2, geometry=geometry, capacity=parent + 1)
    cache.groups[1].slab_ids.fill_(parent)
    contexts = torch.tensor([513], device="cuda", dtype=torch.int64)
    assert parent * cache.recurrent_states.stride(0) > 2**31
    assert parent * cache.convolution_states.stride(0) > 2**32
    initial = gather_state_layer_major(cache, contexts)
    reference = _reference_gather(cache, contexts)
    for i in range(34):
        _bitwise_equal(initial[0][i], reference[0][:, i])
        _bitwise_equal(initial[1][i], reference[1][:, i])
    _bitwise_equal(initial[2], reference[2])
    del reference
    expected = arena.storage.clone()
    final_r = [r.clone() for r in initial[0]]
    final_c = [c.clone() for c in initial[1]]
    final_h = initial[2].clone()
    for i, value in enumerate([*final_r, *final_c, final_h]):
        value.view(torch.uint8).bitwise_xor_(i + 1)
    _on_storage(cache, expected).store_state(
        contexts - 1, torch.stack(final_r, 1), torch.stack(final_c, 1), final_h
    )
    scatter_state_layers(cache, contexts - 1, final_r, final_c, final_h)
    _bitwise_equal(arena.storage, expected)
    # The small raw-index checkpoints use the same >32-bit parent offsets.
    from resin_models_cuda.glm5_3_flash.cache_cuda import (
        gather_index_tails,
        scatter_index_tails,
    )

    tails = gather_index_tails(cache, contexts)
    for actual, reference_tail in zip(
        tails, _reference_tails(cache, contexts), strict=True
    ):
        for value, reference_value in zip(
            (actual.keys, actual.gates, actual.end),
            (reference_tail.keys, reference_tail.gates, reference_tail.end),
            strict=True,
        ):
            _bitwise_equal(value, reference_value)
            value.view(torch.uint8).bitwise_xor_(5)
    _reference_store_tails(_on_storage(cache, expected), contexts - 1, tails)
    scatter_index_tails(cache, contexts - 1, tails)
    _bitwise_equal(arena.storage, expected)


@CUDA
def test_state_transfer_does_not_specialize_on_batch_or_context_values():
    from resin_models_cuda.glm5_3_flash import cache_cuda

    def entries():
        device = torch.cuda.current_device()
        return {
            name: set(getattr(cache_cuda, name).device_caches[device][0])
            for name in (
                "_gather_state",
                "_scatter_state",
                "_gather_index_tails",
                "_scatter_index_tails",
            )
        }

    def execute(batch, width, first, dtype):
        arena, cache = _fixture("cuda", batch, width=width, first=first)
        contexts = torch.full((batch,), first * 64 + 1, device="cuda", dtype=dtype)
        r, c, h = cache_cuda.gather_state_layer_major(cache, contexts)
        expected = _reference_gather(cache, contexts)
        for i in range(2):
            _bitwise_equal(r[i], expected[0][:, i])
            _bitwise_equal(c[i], expected[1][:, i])
        _bitwise_equal(h, expected[2])
        tails = cache_cuda.gather_index_tails(cache, contexts)
        for actual, expected_tail in zip(
            tails, _reference_tails(cache, contexts), strict=True
        ):
            _bitwise_equal(actual.keys, expected_tail.keys)
            _bitwise_equal(actual.gates, expected_tail.gates)
            _bitwise_equal(actual.end, expected_tail.end)
        before = arena.storage.clone()
        cache_cuda.scatter_state_layers(cache, contexts - 1, tuple(r), tuple(c), h)
        cache_cuda.scatter_index_tails(cache, contexts - 1, tails)
        _bitwise_equal(arena.storage, before)

    # Integer dtypes are a bounded ABI distinction, warmed independently.
    for dtype in (torch.int32, torch.int64):
        execute(1, 3, 0, dtype)
    baseline = entries()
    assert all(baseline.values())
    for batch, width, first in [
        (2, 1, 1),
        (4, 2, 7),
        (8, 7, 31),
        (16, 4, 1024),
        (32, 5, 16380),
    ]:
        for dtype in (torch.int32, torch.int64):
            execute(batch, width, first, dtype)
            current = entries()
            assert current == baseline, {
                name: sorted(keys - baseline[name])
                for name, keys in current.items()
                if keys != baseline[name]
            }
