"""Physical layout and checkpoint restoration independent of CUDA kernels."""

from types import SimpleNamespace

import pytest
import torch

from resin.backend.cuda.slab import DevicePageGroup, graph_scratch_parent_count
from resin_models_cuda.glm5_3_flash.cache import CacheArena, CacheGeometry


def geometry(*, mtp=False):
    return CacheGeometry(
        ("linear_attention", "deepseek_sparse_attention"),
        hidden_size=16,
        latent_dim=8,
        index_dim=4,
        heads=2,
        head_dim=8,
        conv_history=3,
        state_tokens=64,
        include_mtp=mtp,
    )


def group(parent, *, slot=0):
    return DevicePageGroup(
        torch.tensor([[parent]], dtype=torch.int32),
        torch.tensor([[slot]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )


def test_state_restoration_and_fresh_request_clear():
    shape = geometry()
    arena = CacheArena("cpu", shape.layout(3), shape, max_batch_size=2)
    cache = arena._make_graph_cache((group(1), group(2)))
    positions = torch.arange(3)
    batch = cache.batch(
        positions,
        torch.zeros(3, dtype=torch.long),
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )
    batch.layer(0).recurrent.fill_(7)
    batch.layer(0).convolution.fill_(3)
    hidden = torch.arange(16, dtype=torch.bfloat16)[None]
    batch.store(hidden)
    resumed = cache.batch(
        torch.tensor([3]), torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([3])
    )
    assert not batch.is_decode and not resumed.is_decode
    assert batch.projection_rows == resumed.projection_rows == 0
    batch.projection_rows = 64
    assert resumed.projection_rows == 0
    # One-row prefill remains prefill; a target decode explicitly opts in.
    batch.is_decode = True
    assert not resumed.is_decode
    torch.testing.assert_close(resumed.layer(0).recurrent, batch.layer(0).recurrent)
    torch.testing.assert_close(resumed.layer(0).convolution, batch.layer(0).convolution)
    torch.testing.assert_close(resumed.previous_hidden, hidden)
    fresh = cache.batch(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([0])
    )
    assert not fresh.layer(0).recurrent.count_nonzero()
    assert not fresh.layer(0).convolution.count_nonzero()
    assert not fresh.previous_hidden.count_nonzero()
    # Fresh masking acts on gathered copies, preserving the published prefix.
    restored, _, _ = cache.state_at(torch.tensor([2]))
    assert restored.eq(7).all()
    assert not arena.storage[1].count_nonzero()


def test_mla_index_views_and_storage_envelope_are_disjoint():
    shape = geometry(mtp=True)
    arena = CacheArena("cpu", shape.layout(2), shape, max_batch_size=2)
    cache = arena._make_graph_cache((group(1), group(2)))
    target_page = int(cache.pages(0, 0, 2)[0, 0])
    draft_page = int(cache.pages(0, 1, 2)[0, 0])
    assert target_page != draft_page
    arena.latent[target_page].fill_(1)
    arena.index_tail_keys[2].fill_(2)
    arena.index_tail_gates[2].fill_(3)
    arena.pooled_keys[target_page].fill_(4)
    assert arena.latent[target_page].eq(1).all()
    assert arena.index_tail_keys[2].eq(2).all()
    assert arena.index_tail_gates[2].eq(3).all()
    assert arena.pooled_keys[target_page].eq(4).all()
    assert not arena.latent[draft_page].count_nonzero()
    assert arena.latent.stride(0) % shape.latent_dim == 0
    rows = (
        (arena.latent.shape[0] - 1) * arena.latent.stride(0)
    ) // shape.latent_dim + 64
    envelope = arena.latent.as_strided((rows, shape.latent_dim), (shape.latent_dim, 1))
    physical = target_page * arena.latent.stride(0) // shape.latent_dim
    torch.testing.assert_close(
        envelope[physical : physical + 64], arena.latent[target_page]
    )


def test_graph_padding_has_owned_scratch_and_reset_preserves_live_pages():
    shape = geometry()
    arena = CacheArena("cpu", shape.layout(3), shape, max_batch_size=4)
    addresses = [
        arena._graph_padding_address(group_index, row)
        for group_index in range(2)
        for row in range(4)
    ]
    assert all(parent > 3 for parent, _ in addresses)
    for group_index in range(2):
        assert len(set(addresses[group_index * 4 : (group_index + 1) * 4])) == 4
    assert arena.storage.shape[0] == 4 + graph_scratch_parent_count(shape.packings, 4)
    arena.storage.fill_(1)
    arena.reset_null()
    assert arena.storage[1:4].eq(1).all()
    assert not arena.storage[0].count_nonzero()
    assert not arena.storage[4:].count_nonzero()


@pytest.mark.parametrize("mtp", [False, True])
def test_mtp_origin_is_layer_local_and_reserves_one_graph_token(mtp):
    shape = geometry(mtp=mtp)
    arena = CacheArena("cpu", shape.layout(3), shape, max_batch_size=2)
    assert arena._graph_widths(64) == ((2, 1) if mtp else (1, 1))
    assert arena._graph_widths(65) == (2, 1)
    state_blocks = arena.state_blocks[1]
    assert state_blocks.shape[1:] == (shape.packings[1], shape.state_bytes)
    assert state_blocks.stride(-1) == 1
    assert (
        state_blocks.untyped_storage().data_ptr()
        == arena.storage.untyped_storage().data_ptr()
    )
    cache = arena._make_graph_cache((group(1), group(2)))
    batch = cache.batch(
        torch.tensor([1]), torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([1])
    )
    assert batch.layer(1).storage_origin == 0
    if mtp:
        assert batch.layer(2).storage_origin == 1


def test_mtp_priming_uses_real_batch_host_offsets_and_previous_chunk_hidden():
    from resin_models_cuda.glm5_3_flash.normalization import rms_norm
    from resin_models_cuda.glm5_3_flash.speculative import prime_mtp

    shape = geometry(mtp=True)
    arena = CacheArena("cpu", shape.layout(3), shape, max_batch_size=1)
    cache = arena._make_graph_cache((group(1), group(2)))
    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=2, rms_norm_eps=1e-5),
        norm=torch.ones(16, dtype=torch.bfloat16),
        embedding=torch.arange(32 * 16, dtype=torch.bfloat16).reshape(32, 16),
        mtp=lambda *args: calls.append(args),
    )
    ids = torch.tensor([2, 3, 4])
    hidden = torch.arange(3 * 16, dtype=torch.bfloat16).reshape(3, 16)
    offsets = torch.tensor([0, 3], dtype=torch.int32)
    batch = cache.batch(
        torch.arange(3),
        torch.zeros(3, dtype=torch.long),
        offsets,
        torch.tensor([0], dtype=torch.int32),
        cu_seqlens_cpu=offsets,
    )
    assert batch.cu_seqlens_cpu is offsets
    prime_mtp(model, batch, ids, hidden, [0, 1, 2])
    embedded, previous, positions, draft_cache = calls.pop()
    torch.testing.assert_close(embedded, model.embedding[ids[1:]])
    torch.testing.assert_close(previous, rms_norm(hidden[:-1], model.norm, 1e-5))
    assert positions.tolist() == [0, 1]
    assert draft_cache.layer(2).cu_seqlens.tolist() == [0, 2]
    assert draft_cache.layer(2).storage_origin == 1

    batch.store(hidden[-1:])
    next_offsets = torch.tensor([0, 2], dtype=torch.int32)
    resumed = cache.batch(
        torch.tensor([3, 4]),
        torch.zeros(2, dtype=torch.long),
        next_offsets,
        torch.tensor([3], dtype=torch.int32),
        cu_seqlens_cpu=next_offsets,
    )
    next_ids = torch.tensor([5, 6])
    next_hidden = hidden[:2] + 7
    prime_mtp(model, resumed, next_ids, next_hidden, [3, 4])
    embedded, previous, positions, draft_cache = calls.pop()
    torch.testing.assert_close(embedded, model.embedding[next_ids])
    expected = torch.cat((hidden[-1:], next_hidden[:1]))
    torch.testing.assert_close(previous, rms_norm(expected, model.norm, 1e-5))
    assert positions.tolist() == [2, 3]
    assert draft_cache.layer(2).context_lengths.tolist() == [2]
    assert draft_cache.layer(2).cu_seqlens.tolist() == [0, 2]
