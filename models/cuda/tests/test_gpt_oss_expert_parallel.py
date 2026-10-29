"""GPT-OSS keeps global routing separate from local expert storage/execution."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.topology import ParallelGroups, RankGroup
from resin.models.gpt_oss.config import GptOssConfig
from resin_models_cuda.gpt_oss.flashinfer import (
    FlashInferSM90MXFP4Kernel,
    MXFP4Config,
)
from resin_models_cuda.gpt_oss.mxfp4 import (
    EXPERT_CHECKPOINT_NAMES,
    KernelExperts,
    shard_checkpoint_tensor,
    shard_model_config,
)


def _args():
    return SimpleNamespace(
        num_local_experts=8,
        num_experts_per_tok=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        swiglu_limit=7.0,
    )


@pytest.mark.parametrize("name", EXPERT_CHECKPOINT_NAMES)
def test_all_expert_checkpoint_tensors_are_sharded_on_the_expert_axis(name):
    tensor = torch.arange(8 * 16).view(8, 4, 4)
    shard = shard_checkpoint_tensor(
        f"model.layers.0.mlp.experts.{name}",
        tensor,
        args=_args(),
        tensor_parallel=TensorParallelLayout(),
        expert_parallel=TensorParallelLayout(rank=1, size=2),
    )
    torch.testing.assert_close(shard, tensor[4:])
    assert shard.is_contiguous()


def test_expert_and_attention_shards_use_independent_axes():
    args = _args()
    tensor = torch.arange(8 * 128 * 32).view(8, 128, 2, 16)
    expert_shard = shard_checkpoint_tensor(
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        tensor,
        args=args,
        tensor_parallel=TensorParallelLayout(rank=0, size=2),
        expert_parallel=TensorParallelLayout(rank=1, size=2),
    )
    torch.testing.assert_close(expert_shard, tensor[4:, :64])
    query = torch.arange(32 * 32).view(32, 32)
    query_shard = shard_checkpoint_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        query,
        args=args,
        tensor_parallel=TensorParallelLayout(rank=0, size=2),
        expert_parallel=TensorParallelLayout(rank=1, size=2),
    )
    torch.testing.assert_close(query_shard, query[:16])
    router = torch.arange(8 * 32).view(8, 32)
    assert (
        shard_checkpoint_tensor(
            "model.layers.0.mlp.router.weight",
            router,
            args=args,
            tensor_parallel=TensorParallelLayout(),
            expert_parallel=TensorParallelLayout(rank=1, size=2),
        )
        is router
    )


def _config():
    return GptOssConfig.from_dict(
        {
            "model_type": "gpt_oss",
            "vocab_size": 43,
            "hidden_size": 32,
            "intermediate_size": 96,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "head_dim": 8,
            "sliding_window": 32,
        }
    )


def test_local_model_heads_and_expert_channels_use_different_tp_sizes():
    args = _config()
    # Uneven aligned expert shards are legal; their width is unrelated to the
    # number of local attention heads or the expert-owner axis.
    local = shard_model_config(
        args,
        TensorParallelLayout(rank=1, size=2),
        expert_tensor_parallel=TensorParallelLayout(rank=1, size=3),
    )
    assert local == replace(
        args, num_attention_heads=2, num_key_value_heads=1, intermediate_size=32
    )
    legacy = shard_model_config(args, TensorParallelLayout(rank=1, size=2))
    assert legacy.intermediate_size == 32
    full_attention = shard_model_config(
        args,
        TensorParallelLayout(),
        expert_tensor_parallel=TensorParallelLayout(rank=0, size=2),
    )
    assert full_attention.num_attention_heads == args.num_attention_heads
    assert full_attention.intermediate_size == 64


@pytest.mark.parametrize("queries,kv_heads,tp", [(4, 1, 2), (4, 2, 4), (6, 3, 2)])
def test_attention_tp_rejects_query_shards_that_would_replicate_kv(
    queries, kv_heads, tp
):
    args = replace(_config(), num_attention_heads=queries, num_key_value_heads=kv_heads)
    layout = TensorParallelLayout(size=tp)
    with pytest.raises(ValueError, match="num_key_value_heads=.*must be divisible"):
        shard_model_config(args, layout)
    for projection in ("k_proj", "v_proj"):
        with pytest.raises(ValueError, match="num_key_value_heads=.*must be divisible"):
            shard_checkpoint_tensor(
                f"model.layers.0.self_attn.{projection}.weight",
                torch.empty(kv_heads * args.head_dim, args.hidden_size),
                args=args,
                tensor_parallel=layout,
            )


@pytest.mark.parametrize("projection", ["k_proj", "v_proj"])
def test_attention_tp_partitions_complete_kv_heads_without_replication(projection):
    args = _config()
    weight = torch.arange(args.num_key_value_heads * args.head_dim * args.hidden_size)
    weight = weight.reshape(args.num_key_value_heads * args.head_dim, args.hidden_size)
    shards = [
        shard_checkpoint_tensor(
            f"model.layers.0.self_attn.{projection}.weight",
            weight,
            args=args,
            tensor_parallel=TensorParallelLayout(rank=rank, size=2),
        )
        for rank in range(2)
    ]
    assert all(shard.shape[0] == args.head_dim for shard in shards)
    torch.testing.assert_close(torch.cat(shards), weight)


@pytest.mark.parametrize("name", EXPERT_CHECKPOINT_NAMES)
def test_composed_ep_and_expert_tp_reconstruct_all_checkpoint_weights(name):
    args = _config()
    shapes = {
        "gate_up_proj_blocks": (8, 192, 1, 16),
        "gate_up_proj_scales": (8, 192, 1),
        "gate_up_proj_bias": (8, 192),
        "down_proj_blocks": (8, 32, 3, 16),
        "down_proj_scales": (8, 32, 3),
        "down_proj_bias": (8, 32),
    }
    tensor = torch.arange(int(torch.tensor(shapes[name]).prod())).view(shapes[name])
    owners = []
    for ep_rank in range(2):
        channels = [
            shard_checkpoint_tensor(
                f"model.layers.0.mlp.experts.{name}",
                tensor,
                args=args,
                tensor_parallel=TensorParallelLayout(rank=1, size=2),
                expert_parallel=TensorParallelLayout(rank=ep_rank, size=2),
                expert_tensor_parallel=TensorParallelLayout(rank=rank, size=2),
            )
            for rank in range(2)
        ]
        assert all(shard.is_contiguous() for shard in channels)
        if name == "down_proj_bias":
            owners.append(sum(channels))
            assert torch.count_nonzero(channels[1]) == 0
        else:
            owners.append(torch.cat(channels, dim=1 if name.startswith("gate") else 2))
    torch.testing.assert_close(torch.cat(owners), tensor)


def test_output_biases_belong_to_their_own_tp_group():
    args = _config()
    bias = torch.arange(32, dtype=torch.float32)
    layouts = dict(
        args=args,
        tensor_parallel=TensorParallelLayout(rank=1, size=2),
        expert_tensor_parallel=TensorParallelLayout(rank=0, size=2),
    )
    attention = shard_checkpoint_tensor("layer.self_attn.o_proj.bias", bias, **layouts)
    experts = shard_checkpoint_tensor(
        "layer.mlp.experts.down_proj_bias", bias, **layouts
    )
    torch.testing.assert_close(attention, torch.zeros_like(bias))
    torch.testing.assert_close(experts, bias)


def test_sharded_expert_projections_match_complete_weighted_moe():
    args = _config()
    generator = torch.Generator().manual_seed(94)
    hidden = torch.randn(5, args.hidden_size, generator=generator) * 0.1
    gate_up = torch.randn(8, 192, 32, generator=generator) * 0.1
    down = torch.randn(8, 32, 3, 32, generator=generator) * 0.1
    gate_bias = torch.randn(8, 192, generator=generator) * 0.1
    down_bias = torch.randn(8, 32, generator=generator) * 0.1
    indices = torch.tensor([[0, 7], [4, 1], [6, 3], [2, 5], [7, 6]])
    weights = torch.softmax(torch.randn(5, 2, generator=generator), dim=-1)

    def project(gate_up, gate_bias, down, down_bias, expert_start=0):
        output = torch.zeros_like(hidden)
        for row, (experts, probabilities) in enumerate(zip(indices, weights)):
            for expert, probability in zip(experts, probabilities):
                local = int(expert) - expert_start
                if not 0 <= local < gate_up.shape[0]:
                    continue
                pair = torch.nn.functional.linear(
                    hidden[row], gate_up[local], gate_bias[local]
                )
                gate = pair[::2].clamp(max=args.swiglu_limit)
                up = pair[1::2].clamp(-args.swiglu_limit, args.swiglu_limit)
                activated = gate * torch.sigmoid(1.702 * gate) * (up + 1)
                output[row] += probability * torch.nn.functional.linear(
                    activated, down[local].flatten(1), down_bias[local]
                )
        return output

    expected = project(gate_up, gate_bias, down, down_bias)
    actual = torch.zeros_like(hidden)
    for ep_rank in range(2):
        for expert_tp_rank in range(2):
            layouts = dict(
                args=args,
                tensor_parallel=TensorParallelLayout(rank=1, size=2),
                expert_parallel=TensorParallelLayout(rank=ep_rank, size=2),
                expert_tensor_parallel=TensorParallelLayout(
                    rank=expert_tp_rank, size=2
                ),
            )
            actual += project(
                *(
                    shard_checkpoint_tensor(
                        f"layer.mlp.experts.{name}", tensor, **layouts
                    )
                    for name, tensor in (
                        ("gate_up_proj_blocks", gate_up),
                        ("gate_up_proj_bias", gate_bias),
                        ("down_proj_blocks", down),
                        ("down_proj_bias", down_bias),
                    )
                ),
                expert_start=ep_rank * 4,
            )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_checkpoint_prepares_independent_attention_ep_and_expert_tp_shards(
    tmp_path, monkeypatch
):
    from resin_models_cuda.gpt_oss import checkpoint, mxfp4

    args = _config()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt_oss",
                **{
                    name: getattr(args, name)
                    for name in (
                        "vocab_size",
                        "hidden_size",
                        "intermediate_size",
                        "num_hidden_layers",
                        "num_attention_heads",
                        "num_key_value_heads",
                        "num_local_experts",
                        "num_experts_per_tok",
                        "head_dim",
                        "sliding_window",
                    )
                },
                "quantization_config": {"quant_method": "mxfp4"},
            }
        )
    )
    groups = SimpleNamespace(
        attn_tp=RankGroup((0, 1), object(), 0),
        attn_replication=RankGroup((0,), object(), 0),
        ep=RankGroup((1, 3), object(), 1),
        expert_tp=RankGroup((2, 3), object(), 1),
    )
    dispatches = []
    prepared = []
    validated_devices = []

    class Dispatch:
        def __init__(self, local, *, num_experts, groups, replication):
            dispatches.append((num_experts, groups))
            assert replication == (groups.attn_tp, groups.attn_replication)

    class Kernel:
        def forward_partial(self, *args):
            raise AssertionError("weight preparation must not execute experts")

        def validate_device(self, device):
            validated_devices.append(device)
            assert prepared == []

        def validate(self, config):
            assert config.num_experts == 4
            assert config.intermediate_size == 32

        def prepare_weights(self, weights, config):
            prepared.append(weights)
            assert weights.gate_up.shape == (4, 64, 16)
            assert weights.down.shape == (4, 32, 16)
            assert torch.count_nonzero(weights.down_bias) == 0
            return weights

    def load_weights(root, model, *, transform_tensor, install_expert_layer, **kwargs):
        assert validated_devices == [torch.device("cpu")]
        model.to_empty(device="cpu")
        tensors = {
            "gate_up_proj_blocks": torch.zeros(8, 192, 1, 16, dtype=torch.uint8),
            "gate_up_proj_scales": torch.zeros(8, 192, 1, dtype=torch.uint8),
            "gate_up_proj_bias": torch.ones(8, 192),
            "down_proj_blocks": torch.zeros(8, 32, 3, 16, dtype=torch.uint8),
            "down_proj_scales": torch.zeros(8, 32, 3, dtype=torch.uint8),
            "down_proj_bias": torch.ones(8, 32),
        }
        install_expert_layer(
            0,
            {
                name: transform_tensor(f"model.layers.0.mlp.experts.{name}", value)
                for name, value in tensors.items()
            },
        )

    monkeypatch.setattr(mxfp4, "ExpertParallelDispatch", Dispatch)
    monkeypatch.setattr(checkpoint, "load_gpt_oss_weights", load_weights)
    model, _ = checkpoint.load_model(
        tmp_path,
        device="cpu",
        kernel_moe=Kernel(),
        parallel_groups=cast(ParallelGroups, groups),
    )
    assert dispatches == [(8, groups)]
    assert validated_devices == [torch.device("cpu")]
    assert len(prepared) == 1
    assert model.args.intermediate_size == 32
    assert model.args.num_attention_heads == 2
    assert model.args.num_key_value_heads == 1
    assert model.model.layers[0].mlp.router.weight.shape == (8, 32)


def test_ep_local_adapter_preserves_global_routing_weights_and_skips_empty(monkeypatch):
    from resin_models_cuda.gpt_oss import mxfp4

    monkeypatch.setattr(
        mxfp4, "ExpertParallelDispatch", lambda *args, **kwargs: object()
    )
    args = _args()
    local_args = SimpleNamespace(**vars(args))
    local_args.num_local_experts = 4
    config = MXFP4Config.from_model(local_args)
    groups = SimpleNamespace(
        ep=RankGroup((0, 1), object(), 1),
        attn_tp=RankGroup((1,), object(), 0),
        attn_replication=RankGroup((1,), object(), 0),
    )
    calls = []
    storage = object()

    class Kernel:
        def forward_partial(self, hidden, indices, route_weights, weights):
            assert weights is storage
            calls.append((indices, route_weights))
            result = torch.zeros_like(hidden, dtype=torch.float32)
            for expert in range(config.num_experts):
                # CUTLASS's W4 expert map stops at the first matching slot.
                # A foreign zero-weight expert-0 placeholder would conceal
                # the real owned expert-0 route later in the same row.
                matches = indices == expert
                first = matches.int().argmax(1)
                scales = route_weights.gather(1, first[:, None])
                scales = scales * matches.any(1)[:, None]
                result += hidden * scales * (expert + 5)
            return result

    module = KernelExperts(
        args, Kernel(), config, parallel_groups=cast(ParallelGroups, groups)
    )
    module._prepared_weights = storage
    module._device = torch.device("cpu")
    hidden = torch.ones(3, 32, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 4], [7, 3], [5, 6]])
    weights = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
    output = module.forward_local(hidden, indices, weights)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output[:, 0], torch.tensor([4.0, 3.2, 6.7]))
    assert calls[0][0].tolist() == [[4, 0], [3, 4], [1, 2]]
    torch.testing.assert_close(
        calls[0][1],
        torch.tensor([[0.0, 0.8], [0.4, 0.0], [0.3, 0.7]]),
    )
    assert torch.equal(weights, torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]]))
    empty = module.forward_local(hidden[:0], indices[:0], weights[:0])
    assert empty.shape == (0, 32) and len(calls) == 1
    assert empty.dtype == torch.float32

    # Both the torch router (int64/BF16) and idle ranks use one wire format.
    routed = []

    def dispatch(hidden, indices, weights):
        routed.append((hidden, indices, weights))
        return hidden

    monkeypatch.setattr(module, "_dispatch", dispatch)
    module(hidden, indices, weights.bfloat16())
    module.forward_empty()
    for _, expert_ids, probabilities in routed:
        assert expert_ids.dtype == torch.int32
        assert probabilities.dtype == torch.float32
    assert routed[1][0].shape == (0, 32)
    assert routed[1][1].shape == routed[1][2].shape == (0, 2)


def test_ep_requires_prepared_weights_before_local_execution_or_dispatch(monkeypatch):
    from resin_models_cuda.gpt_oss import mxfp4

    def dispatch(*args, **kwargs):
        raise AssertionError("unprepared experts must not enter a collective")

    monkeypatch.setattr(
        mxfp4, "ExpertParallelDispatch", lambda *args, **kwargs: dispatch
    )
    args = _args()
    config = replace(MXFP4Config.from_model(args), num_experts=4)
    groups = SimpleNamespace(
        ep=RankGroup((0, 1), object(), 0),
        attn_tp=RankGroup((0,), object(), 0),
        attn_replication=RankGroup((0,), object(), 0),
    )
    module = KernelExperts(
        args,
        FlashInferSM90MXFP4Kernel(),
        config,
        parallel_groups=cast(ParallelGroups, groups),
    )
    hidden = torch.ones(1, args.hidden_size, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 7]])
    weights = torch.tensor([[0.25, 0.75]])
    with pytest.raises(RuntimeError, match="weights have not been loaded"):
        module.forward_local(hidden, indices, weights)
    with pytest.raises(RuntimeError, match="weights have not been loaded"):
        module(hidden, indices, weights)
