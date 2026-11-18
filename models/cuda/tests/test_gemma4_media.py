from __future__ import annotations

import io

import numpy as np
import pytest
from output_helpers import forward_output
from PIL import Image
from recipe_fuzz.gemma import GemmaAdapter

import resin
from resin import Engine, SamplingParams
from resin.models.vision import padded_patches
from resin.recipe import Media, get_native_recipe


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def test_encoded_image_boundary_is_bytes_plus_explicit_mime() -> None:
    recipe = get_native_recipe(
        "gemma4",
        GemmaAdapter().recipe.tokenizer,
        media_config={
            "image_processor": {
                "patch_size": 1,
                "pooling_kernel_size": 1,
                "max_soft_tokens": 70,
                "do_resize": False,
            }
        },
    )
    prompt = recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [resin.recipe.Image.from_bytes(_png_bytes(), media_type="image/png")],
            )
        ]
    )
    assert prompt.media is not None
    assert isinstance(prompt.media[0], Media)
    pixels = prompt.media[0].image
    assert pixels is not None
    assert pixels.shape == (3, 4, 3)
    np.testing.assert_allclose(
        pixels.reshape(-1, 3), np.tile(np.array([10, 20, 30]) / 255, (12, 1))
    )
    with pytest.raises(ValueError, match="media type"):
        recipe.render(
            [
                resin.recipe.Message(
                    resin.recipe.Role.USER,
                    [
                        resin.recipe.Image.from_bytes(
                            _png_bytes(), media_type="application/octet-stream"
                        )
                    ],
                )
            ]
        )


@pytest.mark.parametrize("is_video", [False, True])
def test_native_patches_match_reference_without_resize_and_own_input(is_video) -> None:
    from transformers.models.gemma4.image_processing_gemma4 import (
        Gemma4ImageProcessor,
        Gemma4ImageProcessorKwargs,
    )
    from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor

    config: Gemma4ImageProcessorKwargs = {
        "patch_size": 2,
        "pooling_kernel_size": 1,
        "max_soft_tokens": 70,
        "do_resize": False,
        "do_normalize": True,
        "image_mean": [0.5, 0.25, 0.0],
        "image_std": [0.5, 0.25, 1.0],
    }
    image = (np.arange(24).reshape(2, 4, 3) / 24).astype(np.float32)
    if is_video:
        image = np.stack([image, image / 2], dtype=np.float32)
        expected = Gemma4VideoProcessor(**config, do_sample_frames=False)(
            videos=[image],
            do_rescale=False,
            return_tensors="np",
            input_data_format="channels_last",
        )
        block = resin.recipe.Video(source=image)
        field = "video_processor"
    else:
        expected = Gemma4ImageProcessor(**config)(
            images=[image],
            do_rescale=False,
            return_tensors="np",
            input_data_format="channels_last",
        )
        block = resin.recipe.Image.from_rgb(image)
        field = "image_processor"
    native_config = {**config, "do_sample_frames": False} if is_video else config
    recipe = get_native_recipe(
        "gemma4", GemmaAdapter().recipe.tokenizer, media_config={field: native_config}
    )
    messages = [
        resin.recipe.Message(
            resin.recipe.Role.USER, [block, resin.recipe.Text("describe")]
        )
    ]
    prompt = recipe.render(messages)
    marker = "<|video|>" if is_video else "<|image|>"
    frame = "<|image>" + marker * 2 + "<image|>"
    expansion = " ".join(["00:00 " + frame] * 2) if is_video else frame
    expected_text = recipe.debug_render_text(messages).replace(marker, expansion)
    assert prompt.token_ids.tolist() == recipe.tokenizer.encode(
        expected_text, add_special_tokens=False
    )
    image.fill(0)
    assert prompt.media is not None
    assert isinstance(prompt.media[0], Media)
    item = prompt.media[0]
    if is_video:
        assert item.video is not None
        bitmaps = item.video
    else:
        assert item.image is not None
        bitmaps = (item.image,)
    pixels, positions = padded_patches(bitmaps, 2)
    reference_pixels = (
        expected["pixel_values_videos"][0] if is_video else expected["pixel_values"]
    )
    reference_positions = (
        expected["video_position_ids"][0]
        if is_video
        else expected["image_position_ids"]
    )
    np.testing.assert_array_equal(pixels, reference_pixels[:, : pixels.shape[1]])
    np.testing.assert_array_equal(
        positions, reference_positions[:, : positions.shape[1]]
    )
    assert sum(
        length for _, length in (item.video_frame_spans if is_video else (item.span,))
    ) == (4 if is_video else 2)


def test_submit_passes_tokens_and_media_to_runner() -> None:
    class Runner:
        plans = []

        @classmethod
        def forward(cls, plan):
            cls.plans.append(plan)
            return forward_output([7])

    media = (Media(span=(1, 1), image=np.ones((2, 2, 3), dtype=np.float32)),)
    engine = Engine(Runner())

    engine.submit(
        [2, 8], media=media, sampling_params=SamplingParams(max_output_tokens=1)
    )
    engine.step()

    assert Runner.plans[0].token_ids.tolist() == [2, 8]
    assert Runner.plans[0].media == (media,)
