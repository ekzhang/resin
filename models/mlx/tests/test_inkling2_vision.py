"""The MLX vision encoder and the Torch CPU reference load the same BF16 weights."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from resin_models_mlx.inkling2.checkpoint import load_vision_model

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")


def tiny_vision_config() -> dict:
    return {
        "temporal_patch_size": 1,
        "native_patch_size": 2,
        "adapter_merge_size": 2,
        "posemb_grid_size": 4,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "mlp_ratio": 2,
        "layer_norm_eps": 1e-5,
        "projection_hidden_dim": 24,
        "projector_bias": True,
    }


def write_vision_checkpoint(root: Path, text_hidden_size: int, *, seed: int = 53):
    """Append seeded, Torch-native visual weights to a new or existing checkpoint."""
    from resin_models_cuda.inkling2.vision import Inkling2VisionEncoder

    model = Inkling2VisionEncoder(tiny_vision_config(), text_hidden_size).eval()
    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            values = torch.randn(param.shape, generator=rng)
            if (".ln_" in name and name.endswith(".weight")) or name.endswith(".gamma"):
                values = 1 + values * 0.1
            elif param.ndim == 2:
                values *= param.shape[1] ** -0.5
            else:
                values *= 0.15
            param.copy_(values)
    tensors = {"model.visual." + k: v for k, v in model.state_dict().items()}
    shard = "vision.safetensors"
    safetensors.save_file(tensors, root / shard)
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text()) if path.exists() else {"weight_map": {}}
    index["weight_map"].update(dict.fromkeys(tensors, shard))
    path.write_text(json.dumps(index))
    return model


def test_batched_vision_matches_torch_cpu_and_independent_images(tmp_path):
    reference = write_vision_checkpoint(tmp_path, 13)
    model = load_vision_model(tmp_path, tiny_vision_config(), 13)
    grid = np.array([[1, 2, 3], [1, 1, 2]], dtype=np.int32)
    pixels = torch.randn(8, 4, 4, 3, generator=torch.Generator().manual_seed(97)).to(
        torch.bfloat16
    )
    with torch.no_grad():
        expected = reference(pixels, torch.from_numpy(grid)).float().numpy()
    patches = mx.array(pixels.float().numpy(), dtype=mx.bfloat16)
    embeddings = model(patches, grid)
    assert embeddings.shape == (8, 13) and embeddings.dtype == mx.bfloat16
    actual = np.asarray(embeddings.astype(mx.float32))
    np.testing.assert_allclose(actual, expected, atol=0.025, rtol=0.015)
    for i, part in enumerate((slice(0, 6), slice(6, 8))):
        separate = model(patches[part], grid[i : i + 1])
        np.testing.assert_array_equal(
            actual[part], np.asarray(separate.astype(mx.float32))
        )


def test_vision_loads_only_indexed_visual_weights(tmp_path):
    write_vision_checkpoint(tmp_path, 13)
    shard = tmp_path / "vision.safetensors"
    tensors = safetensors.load_file(shard)
    tensors["model.visual.unindexed"] = torch.zeros(2)
    safetensors.save_file(tensors, shard)
    index = tmp_path / "model.safetensors.index.json"
    files = json.loads(index.read_text())
    # Opening the unrelated shard would fail, even though the index is valid.
    text_shard = tmp_path / "text.safetensors"
    text_shard.write_bytes(b"not a safetensor")
    files["weight_map"]["model.llm.embed.weight"] = text_shard.name
    index.write_text(json.dumps(files))
    model = load_vision_model(tmp_path, tiny_vision_config(), 13)
    assert model.proj_out.weight.shape == (13, 24)


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("missing", "Missing.*parameters"),
        ("extra", "parameters not in model"),
        ("shape", "Expected shape"),
        ("dtype", "must use bfloat16"),
        ("missing_from_shard", "Missing indexed weight"),
    ],
)
def test_vision_loader_validates_checkpoint(tmp_path, problem, message):
    write_vision_checkpoint(tmp_path, 13)
    path = tmp_path / "model.safetensors.index.json"
    shard = tmp_path / "vision.safetensors"
    index = json.loads(path.read_text())
    tensors = safetensors.load_file(shard)
    name = "model.visual.proj_out.weight"
    if problem == "missing":
        del index["weight_map"][name]
    elif problem == "extra":
        tensors["model.visual.other"] = tensors[name].clone()
        index["weight_map"]["model.visual.other"] = shard.name
    elif problem == "shape":
        tensors[name] = tensors[name][:1].clone()
    elif problem == "dtype":
        tensors[name] = tensors[name].float()
    else:
        del tensors[name]
    safetensors.save_file(tensors, shard)
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match=message):
        load_vision_model(tmp_path, tiny_vision_config(), 13)
