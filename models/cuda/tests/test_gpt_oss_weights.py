from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.models.gpt_oss.config import GptOssConfig
from resin_models_cuda.gpt_oss.model import SinkAttention
from resin_models_cuda.gpt_oss.mxfp4 import CheckpointShards
from resin_models_cuda.gpt_oss.weights import load_gpt_oss_weights

safetensors = pytest.importorskip("safetensors.torch")


def _args(*, layers: int = 2, bias: bool = True):
    return GptOssConfig.from_dict(
        {
            "model_type": "gpt_oss",
            "vocab_size": 5,
            "hidden_size": 4,
            "intermediate_size": 32,
            "num_hidden_layers": layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "head_dim": 2,
            "sliding_window": 128,
            "attention_bias": bias,
        }
    )


def _model(args, installed):
    class Experts(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def load_weights(self, tensors):
            installed.append((self.index, tensors))

    with torch.device("meta"):
        model = torch.nn.Module()
        body = torch.nn.Module()
        layers = torch.nn.ModuleList()
        body.add_module("embed_tokens", torch.nn.Embedding(5, 4))
        body.add_module("layers", layers)
        model.add_module("model", body)
        model.add_module("lm_head", torch.nn.Linear(4, 5, bias=False))
        for index in range(args.num_hidden_layers):
            layer = torch.nn.Module()
            mlp = torch.nn.Module()
            mlp.add_module("experts", Experts(index))
            layer.add_module("self_attn", SinkAttention(args, index))
            layer.add_module("mlp", mlp)
            layers.append(layer)
    return model


def _tensors(args):
    tensors = {
        "model.embed_tokens.weight": torch.arange(20, dtype=torch.float32).view(5, 4),
        "lm_head.weight": torch.arange(20, 40, dtype=torch.float32).view(5, 4),
    }
    for index in range(args.num_hidden_layers):
        prefix = f"model.layers.{index}"
        for part, dims, offset in (("q", 8, 0), ("k", 4, 32), ("v", 4, 48)):
            key = f"{prefix}.self_attn.{part}_proj"
            tensors[f"{key}.weight"] = (
                torch.arange(offset, offset + dims * 4).float().view(dims, 4)
            )
            if args.attention_bias:
                tensors[f"{key}.bias"] = torch.arange(offset, offset + dims).float()
        tensors[f"{prefix}.self_attn.o_proj.weight"] = (
            torch.arange(32).float().view(4, 8)
        )
        if args.attention_bias:
            tensors[f"{prefix}.self_attn.o_proj.bias"] = torch.ones(4)
        tensors[f"{prefix}.self_attn.sinks"] = torch.arange(4).float()
        expert = f"{prefix}.mlp.experts"
        tensors[f"{expert}.gate_up_proj_blocks"] = torch.full(
            (2, 64, 1, 16), index, dtype=torch.uint8
        )
        tensors[f"{expert}.gate_up_proj_scales"] = torch.zeros(
            2, 64, 1, dtype=torch.uint8
        )
        tensors[f"{expert}.gate_up_proj_bias"] = torch.zeros(2, 64)
        tensors[f"{expert}.down_proj_blocks"] = torch.zeros(
            2, 4, 1, 16, dtype=torch.uint8
        )
        tensors[f"{expert}.down_proj_scales"] = torch.zeros(2, 4, 1, dtype=torch.uint8)
        tensors[f"{expert}.down_proj_bias"] = torch.zeros(2, 4)
    return tensors


def _load(root: Path, shards: CheckpointShards, installed: list):
    model = _model(shards.model_args, installed)
    load_gpt_oss_weights(root, model, device=torch.device("cpu"), shards=shards)
    return model


def test_streams_ordinary_tensors_and_one_expert_layer_at_a_time(tmp_path: Path):
    args = _args()
    tensors = _tensors(args)
    safetensors.save_file(tensors, tmp_path / "model.safetensors")

    installed = []
    model = _load(tmp_path, CheckpointShards(args), installed)

    torch.testing.assert_close(
        model.model.embed_tokens.weight, tensors["model.embed_tokens.weight"].bfloat16()
    )
    assert [
        (index, weights["gate_up_proj_blocks"][0, 0, 0, 0].item())
        for index, weights in installed
    ] == [(0, 0), (1, 1)]
    assert installed[0][1]["gate_up_proj_blocks"].dtype == torch.uint8
    assert installed[0][1]["gate_up_proj_bias"].dtype == torch.float32


@pytest.mark.parametrize("bias", [False, True])
def test_packed_qkv_loads_rank_local_projection_parts_from_sharded_index(
    tmp_path: Path, bias: bool
):
    args = _args(layers=1, bias=bias)
    tensors = _tensors(args)
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    files = {name: first if ".k_proj." in name else second for name in tensors}
    for file in (first, second):
        safetensors.save_file(
            {name: tensor for name, tensor in tensors.items() if files[name] == file},
            file,
        )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: file.name for name, file in files.items()}})
    )

    model = _load(
        tmp_path,
        CheckpointShards(args, TensorParallelLayout(rank=1, size=2)),
        [],
    )
    prefix = "model.layers.0.self_attn"
    attention = model.model.layers[0].self_attn
    for name, target in attention.qkv_proj.named_parameters():
        expected = torch.cat(
            [
                tensors[f"{prefix}.{part}_proj.{name}"][start:]
                for part, start in (("q", 4), ("k", 2), ("v", 2))
            ]
        )
        torch.testing.assert_close(target, expected.bfloat16())
    assert (attention.qkv_proj.bias is not None) == bias
    torch.testing.assert_close(
        attention.o_proj.weight, tensors[f"{prefix}.o_proj.weight"][:, 4:].bfloat16()
    )
    if bias:
        torch.testing.assert_close(
            attention.o_proj.bias, torch.zeros(4, dtype=torch.bfloat16)
        )


def test_missing_tensor_is_rejected_before_assignment(tmp_path: Path):
    args = _args(layers=1)
    tensors = _tensors(args)
    del tensors["lm_head.weight"]
    model = _model(args, [])
    safetensors.save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="missing tensors"):
        load_gpt_oss_weights(
            tmp_path, model, device=torch.device("cpu"), shards=CheckpointShards(args)
        )
    assert model.model.embed_tokens.weight.is_meta


@pytest.mark.parametrize(
    "key", ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"]
)
def test_shape_mismatch_is_rejected(tmp_path: Path, key: str):
    args = _args(layers=1)
    tensors = _tensors(args)
    tensors[key] = torch.ones(1, 4)
    safetensors.save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="shape mismatch"):
        _load(tmp_path, CheckpointShards(args), [])
