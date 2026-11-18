"""Checkpoint copies preserve values without materializing a device-sized source."""

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from resin_models_cuda.deepseek_v41.weights import load_weights


def checkpoint(tmp_path, *, columns=4):
    values = {
        "layers.0.engram.embed.weight": torch.arange(9 * columns)
        .reshape(9, columns)
        .float(),
        "packed": torch.arange(-4, 4, dtype=torch.int8).reshape(2, 4),
    }
    save_file(values, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in values}})
    )
    return values


def model():
    class Embedding(nn.Module):
        start: int = 0
        stop: int = 9

    result = nn.Module()
    layer = nn.Module()
    engram, embedding = nn.Module(), Embedding()
    embedding.register_buffer("weight", torch.full((9, 4), -1, dtype=torch.bfloat16))
    engram.add_module("embed", embedding)
    layer.add_module("engram", engram)
    result.layers = nn.ModuleList([layer])
    result.register_buffer("packed", torch.empty(2, 4, dtype=torch.float4_e2m1fn_x2))
    return result


def forbid_source_device_copy(monkeypatch):
    original = torch.Tensor.to

    def cast_only(tensor, *args, **kwargs):
        if "device" in kwargs or (args and isinstance(args[0], (str, torch.device))):
            raise AssertionError(
                "checkpoint must copy directly into its existing target"
            )
        result = original(tensor, *args, **kwargs)
        assert result.device.type == "cpu"
        return result

    monkeypatch.setattr(torch.Tensor, "to", cast_only)


def test_loader_copies_cpu_casts_and_fp4_bytes_into_existing_storage(
    tmp_path, monkeypatch
):
    values, target = checkpoint(tmp_path), model()
    pointers = {name: tensor.data_ptr() for name, tensor in target.state_dict().items()}
    forbid_source_device_copy(monkeypatch)
    report = load_weights(tmp_path, target, rank=0, tensor_parallel_size=1)
    loaded = target.state_dict()
    assert pointers == {name: tensor.data_ptr() for name, tensor in loaded.items()}
    assert torch.equal(
        loaded["layers.0.engram.embed.weight"].float(),
        values["layers.0.engram.embed.weight"],
    )
    assert torch.equal(loaded["packed"].view(torch.int8), values["packed"])
    assert report == {"tensors": 2, "bytes": 80}


def test_loader_rejects_shape_mismatch_before_copy(tmp_path, monkeypatch):
    checkpoint(tmp_path, columns=5)
    target = model()
    forbid_source_device_copy(monkeypatch)
    with pytest.raises(ValueError, match="checkpoint shape.*does not match model"):
        load_weights(tmp_path, target, rank=0, tensor_parallel_size=1)
    assert target.layers[0].engram.embed.weight.eq(-1).all()


def test_v41_loader_uses_embedding_ownership_independent_of_attention_rank(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from resin_models_cuda.deepseek_v41.engram import ParallelEngramEmbedding

    embedding = ParallelEngramEmbedding(
        13, 32, SimpleNamespace(group=SimpleNamespace(size=4, local_rank=3))
    )
    target, layer, engram = nn.Module(), nn.Module(), nn.Module()
    engram.add_module("embed", embedding)
    layer.add_module("engram", engram)
    target.layers = nn.ModuleList([layer])
    values = {
        "layers.0.engram.embed.weight": torch.arange(13 * 32)
        .reshape(13, 32)
        .remainder(16)
        .to(torch.float8_e4m3fn),
        "layers.0.engram.embed.scale": torch.full(
            (13, 1), 2, dtype=torch.float8_e8m0fnu
        ),
    }
    save_file(values, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in values}})
    )
    forbid_source_device_copy(monkeypatch)
    report = load_weights(tmp_path, target, rank=0, tensor_parallel_size=1)
    assert (embedding.start, embedding.stop, embedding.weight.shape) == (
        12,
        16,
        (4, 32),
    )
    torch.testing.assert_close(
        embedding.weight[:1].float(), values[next(iter(values))][12:].float()
    )
    assert embedding.weight[1:].float().eq(0).all()
    assert embedding.scale[:1].float().eq(2).all()
    assert embedding.scale[1:].float().eq(1).all()
    assert report == {"tensors": 2, "bytes": 132}
