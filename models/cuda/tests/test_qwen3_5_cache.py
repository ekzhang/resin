from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_qwen3_5_config import _qwen38

from resin import CacheGroupFamily
from resin.backend.cuda.parallel import TensorParallelLayout
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_cuda.qwen3_5.cache import CacheArena, CacheGeometry


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_tp_cache_restores_local_recurrent_state_and_replicated_mtp_hidden(
    size, kv_dtype
):
    from test_qwen3_5_model import _tiny_config

    from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner
    from resin_models_cuda.qwen3_5.parallel import shard_attention_config

    config = replace(
        _tiny_config(),
        num_attention_heads=8,
        num_key_value_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
    )
    local = shard_attention_config(config, TensorParallelLayout(size=size))
    geometry = CacheGeometry.from_config(
        local, include_mtp=True, kv_element_bytes=kv_dtype.itemsize
    )
    arena = CacheArena("cpu", geometry.layout(2), geometry, kv_dtype=kv_dtype)
    cache = arena.graph_cache(batch_size=1, sequence_capacity=4097)
    for group in cache.groups:
        group.slab_ids.zero_()
        group.slotnums.zero_()
        group.first_blocks.zero_()
    cache.groups[1].slab_ids.fill_(1)
    assert arena.keys.shape[-2] == config.num_key_value_heads // size
    assert arena.values.shape == arena.keys.shape
    assert len(cache.groups) == 3  # Target KV, recurrent state, and MTP KV.

    pool = cache.recurrent_pool(torch.tensor([4095]), torch.tensor([4096]))
    assert pool.states is arena.recurrent_states
    assert pool.source.item() == arena.recurrent_states.stride(0)
    assert pool.destination.item() == arena.recurrent_states.stride(0)
    assert pool.layer_stride == arena.recurrent_states.stride(2)
    assert pool.convolution_source is not None
    assert pool.convolution_destination is not None
    assert pool.convolution_source.item() == arena.convolution_states.stride(0)
    assert pool.convolution_destination.item() == arena.convolution_states.stride(0)
    assert pool.convolution_layer_stride == arena.convolution_states.stride(2)

    recurrent = torch.randn(
        1,
        config.linear_layer_count,
        config.linear_num_value_heads // size,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    )
    convolution = torch.randn(
        1,
        config.linear_layer_count,
        config.linear_conv_dim // size,
        config.linear_conv_kernel_dim - 1,
        dtype=torch.bfloat16,
    )
    hidden = torch.randn(1, config.hidden_size, dtype=torch.bfloat16)
    runner = Qwen3_5CudaRunner.__new__(Qwen3_5CudaRunner)
    for position, multiplier in ((4095, 1), (4096, 2)):
        runner._store_state(
            cache,
            torch.tensor([position]),
            recurrent * multiplier,
            convolution * multiplier,
            hidden * multiplier,
        )
    # The live state remains in one private slot across checkpoint boundaries.
    # TP partitions heads/channels but keeps the MTP hidden row whole.
    for context, multiplier in ((4096, 2), (4097, 2), (0, 0)):
        restored = runner._initial_state(cache, torch.tensor([context]))
        for actual, expected in zip(
            restored, (recurrent, convolution, hidden), strict=True
        ):
            torch.testing.assert_close(actual, expected * multiplier)


def test_qwen38_cache_geometry_accounts_for_fp32_recurrent_state() -> None:
    config = Qwen35Config.from_hugging_face(_qwen38())
    geometry = CacheGeometry.from_config(config, include_mtp=True)

    assert geometry.recurrent_bytes == 144 * 1024**2
    assert geometry.state_bytes == 153_954_304
    assert geometry.slab_bytes == 152 * 1024**2
    assert geometry.slab_units == 304
    assert geometry.target_blocks_per_slab == 19
    assert geometry.state_blocks_per_slab == 1
    assert geometry.mtp_blocks_per_slab == 304
    layout = geometry.layout(10)
    assert layout.groups[1].family is CacheGroupFamily.STATE
    assert len(layout.groups) == 3


def test_qwen35_cache_omits_mtp_group_when_speculation_is_disabled() -> None:
    config = Qwen35Config.from_hugging_face(_qwen38())
    geometry = CacheGeometry.from_config(config, include_mtp=False)
    assert len(geometry.layout(10).groups) == 2


def test_qwen38_fp8_kv_halves_each_attention_unit() -> None:
    config = Qwen35Config.from_hugging_face(_qwen38())
    bf16 = CacheGeometry.from_config(config, include_mtp=True)
    fp8 = CacheGeometry.from_config(config, include_mtp=True, kv_element_bytes=1)

    assert fp8.unit_bytes * 2 == bf16.unit_bytes
    assert fp8.target_blocks_per_slab == 37
    assert fp8.slab_bytes == 148 * 1024**2


def test_mixed_dtype_state_views_share_the_byte_arena() -> None:
    values = _qwen38()
    text = values["text_config"]  # type: ignore[assignment]
    text.update(  # type: ignore[union-attr]
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    config = Qwen35Config.from_hugging_face(values)
    geometry = CacheGeometry.from_config(config, include_mtp=True)
    arena = CacheArena(torch.device("cpu"), geometry.layout(2), geometry)

    assert arena.storage.dtype is torch.uint8
    assert arena.keys.dtype is torch.bfloat16
    assert arena.recurrent_states.dtype is torch.float32
    assert arena.convolution_states.dtype is torch.bfloat16
    assert arena.target_hidden_states.dtype is torch.bfloat16
    arena.recurrent_states[1, 0, 0, 0, 0, 0] = 7.0
    arena.convolution_states[1, 0, 0, 0, 0] = 3.0
    arena.target_hidden_states[1, 0, 0] = 5.0
    assert arena.recurrent_states[1, 0, 0, 0, 0, 0].item() == 7.0
    assert arena.convolution_states[1, 0, 0, 0, 0].item() == 3.0
    assert arena.target_hidden_states[1, 0, 0].item() == 5.0


def test_graph_cache_uses_fixed_tables_and_null_capture_inputs() -> None:
    values = _qwen38()
    text = values["text_config"]  # type: ignore[assignment]
    text.update(  # type: ignore[union-attr]
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    config = Qwen35Config.from_hugging_face(values)
    geometry = CacheGeometry.from_config(config, include_mtp=True)
    arena = CacheArena(torch.device("cpu"), geometry.layout(2), geometry)

    cache = arena.graph_cache(batch_size=3, sequence_capacity=257)
    assert [tuple(group.slab_ids.shape) for group in cache.groups] == [
        (3, 3),
        (3, 1),
        (3, 3),
    ]
    inputs = arena.graph_shape_capture_inputs(3, 257)
    target_slabs = inputs["cache.0.slabs"]
    state_slabs = inputs["cache.1.slabs"]
    assert isinstance(target_slabs, tuple) and isinstance(state_slabs, tuple)
    assert len(target_slabs) == 9
    assert len(state_slabs) == 3
    assert set(inputs.values()) == {(0,) * 9, (0,) * 3}
