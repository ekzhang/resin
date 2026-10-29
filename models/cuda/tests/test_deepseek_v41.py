"""V4.1 configuration, shared source-page ownership, and checkpoint transforms."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from resin import Parallelism
from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.backend.cuda.slab import DevicePageGroup
from resin.backend.plans import PagedCachePlan
from resin.models.deepseek_v41.config import DeepSeekV41Config
from resin_models_cuda.deepseek_v41.cache import CacheArena, CacheGeometry
from resin_models_cuda.deepseek_v41.checkpoint import DEFAULT_REVISION
from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner
from resin_models_cuda.deepseek_v41.weights import _attention_shard, _dequantize_output

FIXTURES = Path(__file__).with_name("fixtures") / "deepseek_v41"


def test_official_checkpoint_geometry_and_distinct_draft_experts():
    config = DeepSeekV41Config.from_hugging_face(
        json.loads((FIXTURES / "config.json").read_text())
    )
    config.validate_tensor_parallel(1)
    for tp in (2, 4, 8):
        with pytest.raises(ValueError, match="one KV head"):
            config.validate_tensor_parallel(tp)
    assert config.hidden_size == 5120
    assert config.kv_owner(19) == 14
    assert config.kv_owner(39) == 20
    assert config.index_owner(39) == 36
    assert config.dspark_target_layer_ids == (37, 38, 39)
    assert DEFAULT_REVISION == "1bae0e1088993b9b82de08f59de5be5a34e42170"
    inference = json.loads((FIXTURES / "inference.json").read_text())
    assert inference["n_routed_experts"] == 384
    assert inference["dspark_n_routed_experts"] == 128
    assert inference["dspark_n_activated_experts"] == 3


@pytest.mark.parametrize("runner_type", [DeepSeekV41CudaRunner, DeepSeekV41CudaRunner])
@pytest.mark.parametrize("argument", ["tensor_parallel", "parallel_groups"])
def test_deepseek_rejects_query_only_tp_before_cuda_or_checkpoint_access(
    monkeypatch, runner_type, argument
):
    def unexpected(*args, **kwargs):
        raise AssertionError(
            "unsupported attention TP must fail before CUDA or checkpoint access"
        )

    from resin.backend.cuda.runner import CudaRunner

    monkeypatch.setattr(CudaRunner, "__init__", unexpected)
    monkeypatch.setattr(runner_type, "_checkpoint", unexpected)
    value = (
        TensorParallelContext(
            TensorParallelLayout(size=2), Mock(spec=torch.distributed.ProcessGroup)
        )
        if argument == "tensor_parallel"
        else SimpleNamespace(topology=Parallelism(world_size=2, attn_tp=2))
    )
    with pytest.raises(ValueError, match="attn_tp"):
        runner_type(".", max_seq_len=128, device="cuda:0", **{argument: value})
    assert not runner_type.parallelism_capabilities.attn_tp
    runner_type.parallelism_capabilities.validate(
        Parallelism(world_size=8, attn_dp=4, ep=4), model="DeepSeek"
    )


@pytest.mark.parametrize(
    "parallelism",
    [
        None,
        Parallelism(world_size=4, attn_dp=4, ep=2),
        Parallelism(world_size=4, attn_dp=2, ep=4),
    ],
)
def test_mega_moe_rejects_unsupported_mesh_before_cuda(monkeypatch, parallelism):
    from resin.backend.cuda.runner import CudaRunner

    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported mega-MoE mesh reached CUDA initialization")

    monkeypatch.setattr(CudaRunner, "__init__", unexpected)
    with pytest.raises(ValueError, match="explicit parallel groups|ADP=EP"):
        DeepSeekV41CudaRunner(
            ".",
            max_seq_len=128,
            device="cuda:0",
            expert_backend="deepgemm",
            parallel_groups=None
            if parallelism is None
            else cast(Any, SimpleNamespace(topology=parallelism)),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"compress_ratios": [0, 4, 128] * 15},
        {"kv_source_layer_ids": [2, 14, 8, 20]},
        {"kv_source_layer_ids": [8, 14, 20]},
        {"engram_num_embeddings": [1]},
    ],
)
def test_incompatible_source_geometry_fails_before_allocating_weights(change):
    values = json.loads((FIXTURES / "config.json").read_text())
    values["text_config"].update(change)
    with pytest.raises(ValueError):
        DeepSeekV41Config.from_hugging_face(values)


def _args():
    return SimpleNamespace(
        n_layers=5,
        n_mtp_layers=1,
        dspark_block_size=4,
        compress_ratios=(0, 2, 2, 1, 1, 0),
        kv_source_layers=(1, 3),
        window_size=128,
        head_dim=512,
    )


def _cache(batch_size=1):
    geometry = CacheGeometry.build(_args(), prefill_chunk_size=32)
    arena = CacheArena("cpu", geometry.layout(8), geometry, dtype=torch.bfloat16)
    groups = tuple(
        DevicePageGroup(
            torch.tensor([[i + 1, i + 5]], dtype=torch.int32).repeat(batch_size, 1),
            torch.zeros(batch_size, 2, dtype=torch.int32),
            torch.zeros(batch_size, dtype=torch.int32),
        )
        for i in range(4)
    )
    return arena, arena._make_graph_cache(groups)


def test_shared_kv_and_index_pages_alias_only_their_declared_source():
    arena, cache = _cache()
    positions = torch.tensor([[0, 1, 2, 127, 128, 129]])
    assert torch.equal(
        cache.compressed_rows(1, 2, positions), cache.compressed_rows(2, 2, positions)
    )
    assert torch.equal(cache.index_rows(1, positions), cache.index_rows(2, positions))
    assert torch.equal(
        cache.compressed_rows(3, 1, positions), cache.compressed_rows(4, 1, positions)
    )
    assert not torch.equal(
        cache.compressed_rows(1, 2, positions), cache.compressed_rows(3, 1, positions)
    )
    # Shared source KV and its keys occupy disjoint bytes in the same slab.
    main = cache.compressed_rows(1, 2, positions).long()
    index = cache.index_rows(1, positions).long()
    packed = cache.compressed_cache(1)
    pages, offsets = (
        main.div(packed.shape[1], rounding_mode="floor"),
        main % packed.shape[1],
    )
    packed[pages, offsets, 0] = 2
    index_values, index_scales = cache.index_parts(1)
    index_pages, index_offsets = index // packed.shape[1], index % packed.shape[1]
    index_values[index_pages, index_offsets] = 3
    index_scales[index_pages, index_offsets] = 4
    assert (packed[pages, offsets, 0] == 2).all()
    assert packed.storage_offset() == cache.compressed_cache(2).storage_offset()
    assert packed.storage_offset() != cache.compressed_cache(3).storage_offset()
    assert (index_values[index_pages, index_offsets] == 3).all()
    assert (index_scales[index_pages, index_offsets] == 4).all()
    assert arena.geometry.bytes_per_parent < 2 << 20


def test_compressor_and_ngram_rows_use_disjoint_slabs_and_exact_token_offsets():
    _, cache = _cache()
    positions = torch.tensor([[0, 1, 2, 127, 128, 129]])
    values, scores = cache.compressor_rows(1, positions)
    cache.compressor_values[values] = 7
    cache.compressor_values[scores] = 11
    token_rows = cache.token_rows(positions)
    cache.token_values[token_rows] = positions + 123
    assert cache.token_values[token_rows].tolist() == (positions + 123).tolist()
    assert (cache.compressor_values[values] == 7).all()
    assert (cache.compressor_values[scores] == 11).all()
    assert torch.equal(values[:, 1:3] - values[:, :2], torch.tensor([[2, 2]]))


def test_window_guard_bands_preserve_draft_and_ngram_predecessors():
    geometry = CacheGeometry.build(_args(), prefill_chunk_size=32)
    groups = geometry.layout(32).groups
    assert len(groups) == 4
    assert groups[0].read_window_tokens == 129
    assert groups[2].read_window_tokens == 2
    assert groups[3].read_window_tokens == 4
    assert all(
        group.window_tokens - group.read_window_tokens + 1 == 32
        for group in (groups[0], groups[2], groups[3])
    )
    assert geometry.compressed_offsets[1] == geometry.compressed_offsets[2]
    assert geometry.compressed_offsets[3] == geometry.compressed_offsets[4]


def test_pinned_cache_packing_does_not_inflate_parent_or_b256_budget():
    args = SimpleNamespace(**json.loads((FIXTURES / "inference.json").read_text()))
    geometry = CacheGeometry.build(args, prefill_chunk_size=128, max_batch_size=256)
    assert geometry.packings == (1, 16, 1, 2048)
    assert geometry.bytes_per_parent == 3_244_032
    assert geometry.graph_scratch_parents == 530
    layout = geometry.layout(4096)
    # The first speculative decode crosses the 4096-token compressed-page edge.
    resident = layout.estimated_parent_count([4101] * 256)
    assert resident == 1809
    assert (
        resident + 1 + geometry.graph_scratch_parents
    ) * geometry.bytes_per_parent == 7_591_034_880


@pytest.mark.parametrize(
    "group", range(4), ids=["window", "compressed_and_index", "compressor", "tokens"]
)
def test_pinned_packed_children_hold_every_physical_view_without_overlap(group):
    args = SimpleNamespace(**json.loads((FIXTURES / "inference.json").read_text()))
    geometry = CacheGeometry.build(args, prefill_chunk_size=128)
    arena = CacheArena("cpu", geometry.layout(3), geometry, dtype=torch.bfloat16)
    packing = geometry.packings[group]
    table = DevicePageGroup(
        torch.ones(packing, 1, dtype=torch.int32),
        torch.arange(packing, dtype=torch.int32)[:, None],
        torch.zeros(packing, dtype=torch.int32),
    )
    cache = arena._make_graph_cache((table,) * 4)
    positions = torch.arange(128)[None].expand(packing, -1)
    spans = []

    def record(indices, width):
        starts = indices.flatten().unique().long() * width
        spans.append(torch.stack((starts, starts + width), -1))

    def record_packed(view, indices):
        indices = indices.flatten().unique().long()
        starts = (
            view.storage_offset()
            + indices.div(view.shape[1], rounding_mode="floor") * view.stride(0)
            + indices.remainder(view.shape[1]) * view.stride(1)
        )
        spans.append(torch.stack((starts, starts + view.shape[-1]), -1))

    if group == 0:
        for layer in range(geometry.attention_layers):
            record_packed(
                cache.window_cache(layer), cache.window_rows(layer, positions)
            )
        payload_bytes = 43 * 128 * 528
    elif group == 1:
        for layer in geometry.kv_sources:
            record_packed(
                cache.compressed_cache(layer),
                cache.compressed_rows(layer, geometry.ratios[layer], positions),
            )
            record_packed(cache.index_cache(layer), cache.index_rows(layer, positions))
        payload_bytes = (3 * 64 + 128) * (288 + 68)
    elif group == 2:
        for layer in geometry.compressor_sources:
            for rows in cache.compressor_rows(layer, positions):
                record(rows, 2048)
        payload_bytes = 3 * 128 * 2 * 2048
    else:
        record(cache.token_rows(positions), 8)
        payload_bytes = 128 * 8
    ranges = torch.cat(spans)
    ranges = ranges[ranges[:, 0].argsort()]
    assert (ranges[1:, 0] >= ranges[:-1, 1]).all()
    parent_bytes = geometry.bytes_per_parent
    offsets = ranges - parent_bytes  # All children use physical parent 1.
    stride = parent_bytes // packing
    assert stride % (2048 if group == 2 else 8) == 0
    assert offsets.min() >= 0 and offsets.max() <= parent_bytes
    slots = offsets[:, 0] // stride
    assert (offsets[:, 1] <= (slots + 1) * stride).all()
    assert torch.equal(slots.unique(), torch.arange(packing))
    payloads = torch.zeros(packing, dtype=torch.int64).index_add_(
        0, slots, ranges[:, 1] - ranges[:, 0]
    )
    assert payloads.eq(payload_bytes).all()


def test_batched_graph_capture_reserves_disjoint_writable_scratch_children():
    geometry = CacheGeometry.build(_args(), prefill_chunk_size=32, max_batch_size=4)
    arena = CacheArena("cpu", geometry.layout(8), geometry, dtype=torch.bfloat16)
    cache = arena.graph_cache(4, 256)
    capture = arena.graph_shape_capture_inputs(4, 256)
    storage_pointer = arena.storage.data_ptr()
    for name, target in arena.graph_device_inputs(cache).items():
        assert isinstance(target, torch.Tensor)
        target.copy_(torch.tensor(capture[name]).view_as(target))
    # Each family has private writable children for all four batch rows, even
    # with identical dummy positions and different balanced-expert outputs.
    addresses = [
        arena._graph_padding_address(group, row)
        for group in range(4)
        for row in range(4)
    ]
    by_group = [addresses[group * 4 : (group + 1) * 4] for group in range(4)]
    assert all(len(set(group)) == 4 for group in by_group)
    assert all(parent > arena.layout.parent_capacity for parent, _ in addresses)
    assert (
        len({parent for parent, _ in addresses})
        == geometry.graph_scratch_parents - geometry.guard_parents
    )
    assert (
        arena.storage.shape[0]
        == arena.layout.parent_capacity + 1 + geometry.graph_scratch_parents
    )
    assert {parent for parent, _ in by_group[0]}.isdisjoint(
        parent for parent, _ in by_group[1]
    )
    positions = torch.tensor([[2, 3]]).expand(4, -1)
    rows = (
        (cache.compressor_values, cache.compressor_rows(1, positions)[0]),
        (cache.token_values, cache.token_rows(positions)),
    )
    arena.storage.zero_()
    for pool, indices in rows:
        values = torch.arange(1, 5, dtype=pool.dtype)[:, None].expand_as(indices)
        unique, inverse = indices.flatten().unique(return_inverse=True)
        # Compressor/KV ratio2 shares consecutive positions within a row;
        # no physical address may be shared by different batch rows.
        owners = torch.arange(4)[:, None].expand_as(indices).flatten()
        assert all(
            owners[inverse == i].unique().numel() == 1 for i in range(len(unique))
        )
        values = values.flatten()
        if pool.ndim == 2:
            values = values[:, None].expand(-1, pool.shape[1])
        pool.index_copy_(0, indices.flatten().long(), values)
        actual = pool[indices.long()]
        expected = (
            torch.arange(1, 5, dtype=pool.dtype)
            .reshape(4, *([1] * (actual.ndim - 1)))
            .expand_as(actual)
        )
        torch.testing.assert_close(actual, expected)
    for packed, indices in (
        (cache.window_cache(0), cache.window_rows(0, positions)),
        (cache.compressed_cache(1), cache.compressed_rows(1, 2, positions)),
    ):
        pages, offsets = (
            indices.long().div(packed.shape[1], rounding_mode="floor"),
            indices.long() % packed.shape[1],
        )
        values = torch.arange(1, 5, dtype=torch.uint8)[:, None, None].expand(
            4, 2, packed.shape[-1]
        )
        packed[pages, offsets, 0] = values
        assert torch.equal(packed[pages, offsets, 0], values)
    assert not arena.storage[: arena.layout.parent_capacity + 1].count_nonzero()
    arena.reset_scratch()
    assert not arena.storage.count_nonzero()
    assert arena.storage.data_ptr() == storage_pointer


def test_graph_replay_padding_uses_its_own_reserved_scratch_children():
    geometry = CacheGeometry.build(_args(), prefill_chunk_size=32, max_batch_size=4)
    arena = CacheArena("cpu", geometry.layout(8), geometry, dtype=torch.bfloat16)
    cache = arena.graph_cache(4, 256)
    plan = cast(
        PagedCachePlan,
        SimpleNamespace(
            rows=1,
            groups=tuple(
                SimpleNamespace(
                    block_slabs=lambda row: np.array([1], dtype=np.uint32),
                    block_slots=lambda row: np.array([0], dtype=np.uint32),
                    block_table_start_indices=np.array([0], dtype=np.uint32),
                )
                for _ in range(4)
            ),
        ),
    )
    replay = arena.graph_plan_inputs(cache, plan)
    for group in range(4):
        parents, slots = (
            np.asarray(replay[f"cache.{group}.parent_ids"]),
            np.asarray(replay[f"cache.{group}.child_slots"]),
        )
        assert parents[0, 0] == 1
        assert (parents[0, 1:] == 0).all()  # Absent live history stays invalid.
        for row in range(1, 4):
            parent, slot = arena._graph_padding_address(group, row)
            assert (parents[row] == parent).all()
            assert (slots[row] == slot).all()
    with pytest.raises(ValueError, match="scratch capacity"):
        arena.graph_shape_capture_inputs(5, 256)


def test_block32_grouped_output_weight_dequantization():
    weight = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64) / 256
    scales = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    expected = (
        weight * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
    ).bfloat16()
    torch.testing.assert_close(
        _dequantize_output(weight, scales), expected, rtol=0, atol=0
    )
    assert _attention_shard(weight, "mtp.2.markov_head.embed.weight", 1, 2).shape == (
        32,
        64,
    )
    assert _attention_shard(weight, "mtp.2.markov_head.head.weight", 1, 2).shape == (
        32,
        64,
    )


def test_runner_owns_its_graph_and_distributed_lifecycle():
    from resin.backend.cuda.runner import CudaRunner

    assert DeepSeekV41CudaRunner.__bases__ == (CudaRunner,)
    for name in (
        "forward_distributed",
        "_new_decode_graph_step",
        "_new_dspark_graph_step",
        "_accept_speculative",
        "_graph_forward",
    ):
        assert name in DeepSeekV41CudaRunner.__dict__


def test_token_history_addresses_remain_int64_above_sixteen_gibibytes():
    from dataclasses import replace

    from resin_models_cuda.deepseek_v41.cache import DeviceCache

    geometry = CacheGeometry.build(_args(), prefill_chunk_size=32)
    # No giant allocation: only compute the address of a high physical slab.
    geometry = replace(geometry, parent_bytes=8192 * 1024, packings=(1, 8, 2, 4096))
    groups = tuple(
        DevicePageGroup(
            torch.tensor([[4096]], dtype=torch.int32),
            torch.tensor([[7]], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        )
        for _ in range(4)
    )
    cache = DeviceCache(geometry, groups, torch.empty(0, dtype=torch.uint8))
    actual = cache.token_rows(torch.tensor([[0, 127]]))
    expected = 4096 * 8192 * 128 + 7 * (8192 * 128 // 4096)
    assert actual.dtype == torch.int64
    assert actual.tolist() == [[expected, expected + 127]]
    assert actual.min() > 2**32
