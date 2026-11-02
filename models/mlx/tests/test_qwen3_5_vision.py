from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin_models_mlx.qwen3_5.vision import (  # noqa: E402
    Qwen35VisionModel,
    load_vision_model,
    sanitize_vision_weights,
)


def _config() -> dict[str, Any]:
    return dict(
        model_type="qwen3_5_vision",
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=24,
        num_heads=4,
        depth=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        num_position_embeddings=16,
        hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=[],
    )


def _reference(values=None):
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    torch.manual_seed(31)
    config = Qwen3_5VisionConfig(**(_config() if values is None else values))
    config._attn_implementation = "eager"
    return Qwen3_5VisionModel(config).float().eval()


def _weights(reference, *, mlx_format: bool = False):
    result = {}
    for name, value in reference.state_dict().items():
        array = value.float().numpy()
        if mlx_format and name == "patch_embed.proj.weight":
            array = array.transpose(0, 2, 3, 4, 1)
        prefix = "vision_tower." if mlx_format else "model.visual."
        result[prefix + name] = mx.array(np.ascontiguousarray(array))
    return result


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("grid", [[[1, 4, 6]], [[2, 2, 4], [1, 4, 2]]])
def test_vision_features_match_transformers(dtype, grid):
    import torch

    reference = _reference()
    model = Qwen35VisionModel(_config())
    model.load_weights(list(sanitize_vision_weights(_weights(reference)).items()))
    model.set_dtype(getattr(mx, dtype))
    reference = reference.to(getattr(torch, dtype))
    pixels = (
        np.random.default_rng(12)
        .normal(size=(sum(t * h * w for t, h, w in grid), 24))
        .astype(np.float32)
    )
    with torch.inference_mode():
        expected = (
            reference(torch.from_numpy(pixels), torch.tensor(grid))
            .pooler_output.float()
            .numpy()
        )
    actual = np.array(model(pixels, np.array(grid)).astype(mx.float32))
    tolerance = 2e-2 if dtype == "bfloat16" else 2e-5
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


def test_omitted_patch_size_matches_qwen35_reference_default():
    import torch

    values = _config()
    values.pop("patch_size")
    reference = _reference(values)
    model = Qwen35VisionModel(values)
    model.load_weights(list(sanitize_vision_weights(_weights(reference)).items()))
    pixels = np.random.default_rng(57).normal(size=(4, 1536)).astype(np.float32)
    grid = np.array([[1, 2, 2]])
    with torch.inference_mode():
        expected = reference(
            torch.from_numpy(pixels), torch.from_numpy(grid)
        ).pooler_output.numpy()
    np.testing.assert_allclose(model(pixels, grid), expected, atol=2e-5, rtol=2e-5)


def test_native_no_resize_image_features_match_transformers():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from resin.recipe import Image, Message, Qwen35Media, Role, get_native_recipe

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.add_special_tokens(
        [
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
            "<think>",
            "</think>",
        ]
    )
    recipe = get_native_recipe(
        "qwen3_5",
        tokenizer,
        media_config={
            "do_resize": False,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 2,
        },
    )
    pixels = np.random.default_rng(8).uniform(size=(8, 12, 3)).astype(np.float32)
    media = recipe.render([Message(Role.USER, [Image.from_rgb(pixels)])]).media
    assert isinstance(media, Qwen35Media)
    assert media.image_grid_thw.tolist() == [[1, 4, 6]]
    assert media.pixel_values.shape == (24, 24)
    reference = _reference()
    model = Qwen35VisionModel(_config())
    model.load_weights(list(sanitize_vision_weights(_weights(reference)).items()))
    with torch.inference_mode():
        expected = reference(
            torch.from_numpy(media.pixel_values.copy()),
            torch.from_numpy(media.image_grid_thw.copy()),
        ).pooler_output.numpy()
    actual = np.array(model(media.pixel_values, media.image_grid_thw))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_packed_images_and_frames_keep_attention_separate():
    reference = _reference()
    model = Qwen35VisionModel(_config())
    model.load_weights(list(sanitize_vision_weights(_weights(reference)).items()))
    pixels = np.random.default_rng(4).normal(size=(24, 24)).astype(np.float32)
    together = np.array(model(pixels, np.array([[2, 2, 4], [1, 4, 2]])))
    separate = np.concatenate(
        [
            np.array(model(pixels[:8], np.array([[1, 2, 4]]))),
            np.array(model(pixels[8:16], np.array([[1, 2, 4]]))),
            np.array(model(pixels[16:], np.array([[1, 4, 2]]))),
        ]
    )
    np.testing.assert_allclose(together, separate, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "dtype,width,heads", [("float32", 32, 4), ("bfloat16", 128, 2)]
)
def test_compiled_vision_matches_eager_across_frame_shapes(dtype, width, heads):
    # The actual tower has head_dim=64: its nontrivial rotary frequencies must
    # stay identical when the whole BF16 encoder is compiled.
    values = _config() | {
        "hidden_size": width,
        "num_heads": heads,
        "intermediate_size": width * 2,
    }
    reference = _reference(values)
    model = Qwen35VisionModel(values)
    model.load_weights(list(sanitize_vision_weights(_weights(reference)).items()))
    model.set_dtype(getattr(mx, dtype))
    rng = np.random.default_rng(93)
    inputs = []
    for grid in ([[1, 2, 4]], [[1, 4, 6]], [[2, 4, 4]]):
        pixels = rng.normal(size=(sum(t * h * w for t, h, w in grid), 24)).astype(
            np.float32
        )
        inputs.append((pixels, np.array(grid)))
    expected = [
        np.array(model(pixels, grid).astype(mx.float32)) for pixels, grid in inputs
    ]
    model.enable_compile()
    for _ in range(2):
        for (pixels, grid), output in zip(inputs, expected, strict=True):
            actual = np.array(model(pixels, grid).astype(mx.float32))
            if dtype == "bfloat16":
                np.testing.assert_array_equal(actual, output)
            else:
                np.testing.assert_allclose(actual, output, atol=2e-5, rtol=2e-5)
    model.enable_compile(False)
    np.testing.assert_array_equal(
        np.array(model(*inputs[0]).astype(mx.float32)), expected[0]
    )


@pytest.mark.parametrize("mlx_format,sharded", [(False, True), (True, False)])
def test_vision_checkpoint_loading_supports_hf_and_mlx_layouts(
    tmp_path, mlx_format, sharded
):
    from safetensors.numpy import save_file

    reference = _reference()
    weights = {
        name: np.array(value)
        for name, value in _weights(reference, mlx_format=mlx_format).items()
    }
    weights["language_model.model.embed_tokens.weight"] = np.ones((4, 8), np.float32)
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": _config()}))
    if sharded:
        members = list(weights)
        files = {
            "model-1.safetensors": members[::2],
            "model-2.safetensors": members[1::2],
        }
        for file, names in files.items():
            save_file({name: weights[name] for name in names}, str(tmp_path / file))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        name: file for file, names in files.items() for name in names
                    }
                }
            )
        )
    else:
        save_file(weights, str(tmp_path / "model.safetensors"))
    model = load_vision_model(tmp_path)
    pixels = np.random.default_rng(2).normal(size=(8, 24)).astype(np.float32)
    expected = Qwen35VisionModel(_config())
    expected.load_weights(list(sanitize_vision_weights(_weights(reference)).items()))
    np.testing.assert_allclose(
        np.array(model(pixels, np.array([[1, 2, 4]]))),
        np.array(expected(pixels, np.array([[1, 2, 4]]))),
        atol=1e-6,
        rtol=1e-6,
    )


def test_deepstack_and_packed_vision_weights_are_explicit_errors(tmp_path):
    from safetensors.numpy import save_file

    with pytest.raises(ValueError, match="DeepStack"):
        Qwen35VisionModel(_config() | {"deepstack_visual_indexes": [0]})
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": _config()}))
    save_file(
        {"vision_tower.blocks.0.attn.qkv.scales": np.zeros((1, 1), np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(ValueError, match="packed quantized"):
        load_vision_model(tmp_path)


@pytest.mark.parametrize("grid", [[[1, 3, 4]], [[0, 2, 4]], [[1.0, 2.0, 4.0]]])
def test_vision_grid_validation(grid):
    model = Qwen35VisionModel(_config())
    with pytest.raises(ValueError, match="grid"):
        model(np.zeros((8, 24), np.float32), np.array(grid))
