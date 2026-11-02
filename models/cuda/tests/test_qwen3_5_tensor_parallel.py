"""Qwen target and bundled-MTP FFNs use the same dense TP preparation."""

import copy
import json
from dataclasses import asdict, replace
from typing import cast

import pytest
import torch
from test_qwen3_5_model import _tiny_config
from torch.distributed import ProcessGroup

from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.backend.loading import prepare_model
from resin_models_cuda.dense_parallel import TensorParallelSum, shard_dense_mlp
from resin_models_cuda.nvfp4 import Nvfp4LinearWeights
from resin_models_cuda.qwen3_5.model import (
    Qwen35ForConditionalGeneration,
    Qwen35GatedDeltaNet,
    Qwen35MLP,
)


def test_checkpoint_shards_on_cpu_before_allocating_gpu(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_qwen3_5_config import _qwen38

    from resin_models_cuda.qwen3_5.checkpoint import load_model

    config = _tiny_config()
    raw = _qwen38()
    raw["text_config"].update(asdict(config))
    (tmp_path / "config.json").write_text(json.dumps(raw))
    original = Qwen35ForConditionalGeneration(config, include_mtp=True).bfloat16()
    state = original.state_dict()
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(state, "model.safetensors")})
    )
    transferred = []

    def transfer_local_model(model, device):
        assert torch.device(device).type == "cuda"
        tensors = (*model.parameters(), *model.buffers())
        assert all(tensor.device.type == "cpu" for tensor in tensors)
        assert model.language_model.inv_freq.isfinite().all()
        for module in model.modules():
            if isinstance(module, Qwen35MLP):
                assert module.gate_proj.weight.shape[0] == config.intermediate_size // 2
        assert model.language_model.config.num_key_value_heads == 1
        transferred.append(model)
        return model

    monkeypatch.setattr(Qwen35ForConditionalGeneration, "to", transfer_local_model)
    parallel = TensorParallelContext(
        TensorParallelLayout(rank=1, size=2), cast(ProcessGroup, object())
    )
    loaded, loaded_config, _ = load_model(
        tmp_path,
        device="cuda",
        include_mtp=True,
        include_vision=False,
        tensor_parallel=parallel,
        attention_parallel=parallel,
    )
    assert transferred == [loaded]
    assert loaded_config == config
    torch.testing.assert_close(loaded.lm_head.weight, original.lm_head.weight)


@pytest.mark.parametrize("quantized", [False, True])
def test_dense_tp_prepares_target_and_mtp_without_sharding_attention(quantized):
    torch.manual_seed(57)
    config = replace(_tiny_config(), intermediate_size=256)
    model = Qwen35ForConditionalGeneration(
        config, include_mtp=True, quantized=quantized
    )
    if not quantized:
        model.bfloat16()
    mlps = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, Qwen35MLP)
    }
    assert len(mlps) == config.num_hidden_layers + 1
    assert "mtp.layers.0.mlp" in mlps

    ranks = []
    for rank in range(2):
        parallel = TensorParallelContext(
            TensorParallelLayout(rank=rank, size=2), cast(ProcessGroup, object())
        )
        prepared = prepare_model(
            copy.deepcopy(model),
            lambda module: (
                shard_dense_mlp(module, parallel)
                if isinstance(module, Qwen35MLP)
                else module
            ),
        )
        ranks.append(prepared)
        for name in mlps:
            wrapped = prepared.get_submodule(name)
            assert isinstance(wrapped, TensorParallelSum)
            assert isinstance(wrapped.local, Qwen35MLP)
            assert isinstance(
                wrapped.local.gate_proj, (torch.nn.Linear, Nvfp4LinearWeights)
            )
            assert isinstance(
                wrapped.local.down_proj, (torch.nn.Linear, Nvfp4LinearWeights)
            )
            assert wrapped.local.gate_proj.out_features == config.intermediate_size // 2
            assert wrapped.local.down_proj.in_features == config.intermediate_size // 2
        # Full attention's heads and the output vocabulary remain complete.
        for key in (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "lm_head.weight",
        ):
            torch.testing.assert_close(
                prepared.state_dict()[key],
                model.state_dict()[key],
                rtol=0,
                atol=0,
                equal_nan=True,
            )

    if not quantized:
        for tokens in (1, 7, 64):
            hidden = torch.randn(2, tokens, config.hidden_size, dtype=torch.bfloat16)
            for name, original in mlps.items():
                partials = [
                    cast(TensorParallelSum, rank.get_submodule(name)).local(hidden)
                    for rank in ranks
                ]
                combined = partials[0] + partials[1]
                torch.testing.assert_close(
                    combined, original(hidden), rtol=0.02, atol=0.002
                )


def _attention_ranks(model):
    from resin_models_cuda.qwen3_5.parallel import prepare_attention_parallel

    ranks = []
    for rank in range(2):
        local = copy.deepcopy(model)
        prepare_attention_parallel(
            local,
            TensorParallelContext(
                TensorParallelLayout(rank=rank, size=2), cast(ProcessGroup, object())
            ),
        )
        ranks.append(local)
    return ranks


def test_attention_tp_rejects_partial_kv_or_recurrent_heads():
    from resin_models_cuda.qwen3_5.parallel import shard_attention_config

    config = _tiny_config()
    with pytest.raises(ValueError, match="num_key_value_heads"):
        shard_attention_config(config, TensorParallelLayout(size=4))
    with pytest.raises(ValueError, match="linear_num_key_heads"):
        shard_attention_config(
            replace(config, num_key_value_heads=4), TensorParallelLayout(size=4)
        )


@pytest.mark.parametrize("tokens", [1, 7])
def test_target_and_mtp_attention_tp_preserves_head_gate_order(monkeypatch, tokens):
    monkeypatch.setattr(TensorParallelContext, "sum", lambda self, tensor: tensor)
    torch.manual_seed(83)
    model = Qwen35ForConditionalGeneration(_tiny_config(), include_mtp=True)
    ranks = _attention_ranks(model)
    hidden = torch.randn(2, tokens, model.config.hidden_size)
    positions = torch.arange(tokens)[None].expand(2, -1)
    cos, sin = model.language_model.rotary(positions, hidden.dtype)
    for name in ("model.language_model.layers.3.self_attn", "mtp.layers.0.self_attn"):
        reference = model.get_submodule(name)
        kwargs = dict(
            cache=None,
            attention=None,
            sequence_lengths=positions[:, -1] + 1,
            max_sequence_length=tokens,
        )
        expected = reference(hidden, positions, cos, sin, **kwargs)
        actual = sum(
            local.get_submodule(name)(hidden, positions, cos, sin, **kwargs)
            for local in ranks
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for rank in ranks:
        assert rank.config == model.config
        assert rank.language_model.config.num_key_value_heads == 1
        assert rank.mtp.config.num_key_value_heads == 1


@pytest.mark.parametrize("value_heads", [2, 4, 6])
def test_delta_tp_preserves_packed_heads_and_continued_states(monkeypatch, value_heads):
    monkeypatch.setattr(TensorParallelContext, "sum", lambda self, tensor: tensor)
    torch.manual_seed(84)
    config = replace(_tiny_config(), linear_num_value_heads=value_heads)
    model = Qwen35ForConditionalGeneration(config, include_mtp=False)
    ranks = _attention_ranks(model)
    reference = cast(Qwen35GatedDeltaNet, model.language_model.layers[0].linear_attn)
    local_modules = [local.language_model.layers[0].linear_attn for local in ranks]
    recurrent = (
        torch.randn(
            2, value_heads, config.linear_key_head_dim, config.linear_value_head_dim
        )
        * 0.1
    )
    convolution = torch.randn(
        2, config.linear_conv_dim, config.linear_conv_kernel_dim - 1
    )
    widths = (config.linear_key_dim, config.linear_key_dim, config.linear_value_dim)
    states = [
        (
            recurrent[:, rank * value_heads // 2 : (rank + 1) * value_heads // 2],
            torch.cat(
                [
                    component.chunk(2, dim=1)[rank]
                    for component in convolution.split(widths, dim=1)
                ],
                dim=1,
            ),
        )
        for rank in range(2)
    ]
    for tokens in (5, 1, 3):
        hidden = torch.randn(2, tokens, config.hidden_size)
        expected, recurrent, convolution = reference(hidden, recurrent, convolution)
        outputs = [
            module(hidden, *state)
            for module, state in zip(local_modules, states, strict=True)
        ]
        torch.testing.assert_close(
            outputs[0][0] + outputs[1][0], expected, rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            torch.cat([output[1] for output in outputs], dim=1), recurrent
        )
        local_widths = tuple(width // 2 for width in widths)
        packed = [output[2].split(local_widths, dim=1) for output in outputs]
        full_convolution = torch.cat(
            [torch.cat([rank[part] for rank in packed], dim=1) for part in range(3)],
            dim=1,
        )
        torch.testing.assert_close(full_convolution, convolution)
        states = [(output[1], output[2]) for output in outputs]


def test_fp8_projection_shards_keep_scales_and_packed_qkv_rows():
    from resin_models_cuda.qwen3_5.quantization import ModelOptFp8Linear

    torch.manual_seed(85)
    model = Qwen35ForConditionalGeneration(
        _tiny_config(), include_mtp=True, quantized=True
    )
    for module in model.modules():
        if isinstance(module, ModelOptFp8Linear):
            module.weight.copy_(
                torch.randn_like(module.weight, dtype=torch.float32).to(
                    module.weight.dtype
                )
            )
            module.input_scale.fill_(0.125)
            module.weight_scale.fill_(0.25)
    ranks = _attention_ranks(model)
    reference = cast(Qwen35GatedDeltaNet, model.language_model.layers[0].linear_attn)
    assert isinstance(reference.in_proj_qkv, ModelOptFp8Linear)
    widths = (
        model.config.linear_key_dim,
        model.config.linear_key_dim,
        model.config.linear_value_dim,
    )
    for rank, local in enumerate(ranks):
        delta = local.language_model.layers[0].linear_attn
        expected = torch.cat(
            [
                part.chunk(2, dim=0)[rank]
                for part in reference.in_proj_qkv.weight.split(widths)
            ],
            dim=0,
        )
        torch.testing.assert_close(
            delta.in_proj_qkv.weight.view(torch.uint8),
            expected.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        for module in delta.modules():
            if isinstance(module, ModelOptFp8Linear):
                assert module.input_scale.item() == 0.125
                assert module.weight_scale.item() == 0.25
        # The bundled MTP attention remains BF16 while using the same head split.
        assert local.mtp.layers[0].self_attn.q_proj.weight.shape[0] == 32
