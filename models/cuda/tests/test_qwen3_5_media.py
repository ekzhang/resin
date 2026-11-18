from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image
from recipe_fuzz.qwen import QwenAdapter

import resin
from resin.models.vision import (
    image_patches,
    mrope_delta,
    mrope_positions,
    packed_images,
)
from resin.recipe import Media, get_native_recipe


def _media():
    return (Media(span=(1, 4), image=np.ones((8, 8, 3), dtype=np.float32)),)


def test_qwen_media_keeps_adjacent_images_distinct_for_mrope():
    media = (*_media(), Media(span=(5, 2), image=np.ones((4, 8, 3), dtype=np.float32)))
    positions = mrope_positions(media, np.arange(8), 2, 2)
    assert positions.tolist() == [
        [0, 1, 1, 1, 1, 3, 3, 5],
        [0, 1, 1, 2, 2, 3, 3, 5],
        [0, 1, 2, 1, 2, 3, 4, 5],
    ]
    assert mrope_delta(media, 2, 2) == -2
    np.testing.assert_array_equal(
        mrope_positions(media, np.arange(3, 7), 2, 2), positions[:, 3:7]
    )


def test_patch_packing_preserves_spatial_channel_merge_and_temporal_order():
    image = np.arange(4 * 8 * 3, dtype=np.float32).reshape(4, 8, 3)
    origins = [(0, 0), (0, 2), (2, 0), (2, 2), (0, 4), (0, 6), (2, 4), (2, 6)]
    expected = np.stack(
        [image[y : y + 2, x : x + 2].transpose(2, 0, 1) for y, x in origins]
    )
    actual = image_patches(image, 2, merge=2, temporal=2).reshape(8, 3, 2, 2, 2)
    np.testing.assert_array_equal(actual[:, :, 0], expected)
    np.testing.assert_array_equal(actual[:, :, 1], expected)
    spatial = np.stack(
        [image[y : y + 2, x : x + 2] for y in (0, 2) for x in (0, 2, 4, 6)]
    )
    np.testing.assert_array_equal(image_patches(image, 2, channels_last=True), spatial)
    np.testing.assert_array_equal(
        image_patches(image, 2).reshape(8, 3, 2, 2), spatial.transpose(0, 3, 1, 2)
    )


def _data_url() -> str:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def test_qwen_recipe_normalizes_images_and_attaches_processor_payload() -> None:
    tokenizer = QwenAdapter().recipe.tokenizer
    image_id = tokenizer.token_to_id("<|image_pad|>")

    model_recipe = get_native_recipe(
        "qwen3_5",
        tokenizer,
        media_config={
            "patch_size": 2,
            "temporal_patch_size": 1,
            "merge_size": 1,
            "min_pixels": 16,
            "max_pixels": 16,
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
        },
    )
    output = model_recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [resin.recipe.Image(source=_data_url()), resin.recipe.Text("describe")],
            )
        ]
    )
    assert model_recipe.stop_token_ids
    assert output.media is not None
    assert isinstance(output.media[0], Media)
    assert output.token_ids.tolist().count(image_id) == 4
    pixels, grid = packed_images(output.media, 2)
    assert pixels.shape == (4, 12)
    assert pixels.dtype == np.float32
    assert grid.tolist() == [[1, 2, 2]]
    np.testing.assert_allclose(pixels, [[1.0] * 4 + [-1.0] * 8] * 4, atol=1e-6)


def test_native_qwen_rotary_positions_cover_prompt_and_decode_continuation():
    from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner

    runner = object.__new__(Qwen3_5CudaRunner)
    runner.device = torch.device("cpu")
    runner.vision_patch_size, runner.spatial_merge_size = 2, 2
    media = _media()
    positions = torch.arange(8)[None]
    result = runner._prefill_rotary_positions(media, positions)
    expected = np.array(
        [[0, 1, 1, 1, 1, 3, 4, 5], [0, 1, 1, 2, 2, 3, 4, 5], [0, 1, 2, 1, 2, 3, 4, 5]]
    )
    np.testing.assert_array_equal(result[:, 0].numpy(), expected)


def test_qwen_recipe_rejects_video_instead_of_dropping_its_embeddings() -> None:
    model_recipe = get_native_recipe("qwen3_5", QwenAdapter().recipe.tokenizer)
    with pytest.raises(ValueError, match="does not support video"):
        model_recipe.render(
            [
                resin.recipe.Message(
                    resin.recipe.Role.USER,
                    [
                        resin.recipe.Video(
                            source=np.zeros((1, 2, 2, 3), dtype=np.float32)
                        )
                    ],
                )
            ]
        )
