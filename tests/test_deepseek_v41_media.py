"""Image recipe equivalence, runner packing, and scheduler-neutral ownership."""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from deepseek_recipe import ImageInfo, MultiModalData
from output_helpers import forward_output
from PIL import Image
from recipe_fuzz.deepseek import DeepSeekV41Adapter
from tokenizers import Tokenizer

import resin
import resin._resin as native
from resin import Engine, RunnerConfig, SamplingParams, SchedulerConfig
from resin.models.vision import image_patches
from resin.recipe import get_native_recipe

IMAGE_TOKEN = "<｜deepseek_image｜>"
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/deepseek_v41_images.json").read_text()
)


def _pattern(width, height):
    y, x, channel = np.indices((height, width, 3), dtype=np.int32)
    return ((17 * x + 29 * y + 71 * channel) % 256).astype(np.uint8)


def _data_url(rgb):
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


@pytest.fixture(scope="module")
def recipe():
    tokenizer = Tokenizer.from_str(DeepSeekV41Adapter().recipe.tokenizer.to_str())
    tokenizer.add_special_tokens([IMAGE_TOKEN])
    return get_native_recipe("deepseek_v41", tokenizer)


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_pixels_preserve_published_geometry(case, recipe):
    recipe = get_native_recipe(
        "deepseek_v41",
        recipe.tokenizer,
        vision_config={"max_wh_ratio": case["max_wh_ratio"]},
    )
    media = recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [
                    resin.recipe.Image(
                        source=_pattern(case["width"], case["height"]).astype(
                            np.float32
                        )
                        / 255
                    )
                ],
            )
        ]
    ).media
    assert media and len(media) == 1 and isinstance(media[0], native.Media)
    image = media[0].image
    assert image is not None
    assert tuple(size // 14 for size in image.shape[:2]) == tuple(case["grid"])
    height, width = case["llm_grid"]
    assert media[0].span.length == height * (width + 1) + 2
    if case["max_wh_ratio"] is None:
        expected = (
            MultiModalData(
                [ImageInfo(b"", case["width"], case["height"])]
            ).image_token_adjustment()
            + 1
        )
        assert media[0].span.length == expected
    assert media[0].span.length <= 1024
    patches = image_patches(image, 14).reshape(-1, 3, 14, 14)
    assert patches.dtype == np.float32 and patches.flags.c_contiguous
    assert np.isfinite(patches).all()
    assert np.all(patches >= -1) and np.all(patches <= 1)
    # Resizing can differ from Pillow; packed values remain exactly BF16-rounded.
    expected = torch.from_numpy(patches).to(torch.bfloat16).float().numpy()
    np.testing.assert_array_equal(patches, expected)


def test_native_patch_order_and_normalization_are_exact_without_resizing(recipe):
    source = _pattern(28, 14).astype(np.float32) / 255
    recipe = get_native_recipe(
        "deepseek_v41", recipe.tokenizer, vision_config={"min_pixels": 0}
    )
    media = recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER, [resin.recipe.Image(source=source)]
            )
        ]
    ).media
    assert media is not None
    image = media[0].image
    assert image is not None
    patches = image_patches(image, 14).reshape(-1, 3, 14, 14)
    expected = torch.from_numpy(source).permute(2, 0, 1)
    expected = ((expected - 0.5) / 0.5).to(torch.bfloat16).float()
    expected = expected.reshape(3, 1, 14, 2, 14).permute(1, 3, 0, 2, 4)
    np.testing.assert_array_equal(patches, expected.reshape(2, 3, 14, 14).numpy())
    before = image.copy()
    source.fill(0)
    np.testing.assert_array_equal(media[0].image, before)


def test_media_recipe_expands_only_explicit_images_and_owns_inputs(recipe):
    second = np.zeros((2, 4, 3), dtype=np.float32)
    rendered = recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [
                    resin.recipe.Text("literal " + IMAGE_TOKEN),
                    resin.recipe.Image(source=_data_url(_pattern(17, 31))),
                    resin.recipe.Text("compare"),
                    resin.recipe.Image(source=second),
                ],
            )
        ],
        options=resin.recipe.RenderOptions(
            thinking_mode="thinking", reasoning_effort=37
        ),
    )
    media = rendered.media
    assert media and len(media) == 2
    assert recipe.stop_token_ids and not any(item.atomic for item in media)
    positions = [p for item in media for p in range(item.span.position, sum(item.span))]
    image_token_id = rendered.token_ids[positions[0]]
    assert rendered.token_ids.tolist().count(image_token_id) == len(positions)
    assert IMAGE_TOKEN in recipe.tokenizer.decode(
        rendered.token_ids[: media[0].span.position], skip_special_tokens=False
    )
    second.fill(1)
    image = media[1].image
    assert image is not None and np.all(image < 0)


def test_image_identity_changes_only_its_media_cache_identity(recipe):
    def render(color):
        return recipe.render(
            [
                resin.recipe.Message(
                    resin.recipe.Role.USER,
                    [
                        resin.recipe.Image(
                            source=np.full((2, 4, 3), color, dtype=np.float32)
                        ),
                        resin.recipe.Text("describe"),
                    ],
                )
            ]
        )

    black, white, black_again = render(0), render(1), render(0)
    np.testing.assert_array_equal(black.token_ids, white.token_ids)
    assert black.media and white.media and black_again.media
    assert black.media[0].span == white.media[0].span
    assert black.media[0].digest != white.media[0].digest
    assert black.media[0].digest == black_again.media[0].digest


def test_prepared_media_rejects_bad_layout_and_owns_bitmap_storage():
    pixels = np.ones((14, 14, 3), dtype=np.float32)
    media = native.Media((1, 4), image=pixels)
    pixels.fill(0)
    image = media.image
    assert image is not None
    assert np.all(image == 1) and not image.flags.writeable
    assert image_patches(image, 14).shape == (1, 588)
    with pytest.raises(ValueError, match="HWC"):
        native.Media((1, 4), image=np.ones((1, 3, 14, 14), np.float32))
    with pytest.raises(ValueError, match="align"):
        image_patches(np.ones((14, 15, 3), np.float32), 14)
    with pytest.raises(ValueError, match="non-overlapping"):
        native._media_to_buffers((media, media))


def test_engine_slices_causal_image_spans_across_prefill_chunks(recipe):
    class Runner:
        runner_config = RunnerConfig(continuation_prefill=True)
        contexts = []

        @classmethod
        def forward(cls, plan):
            cls.contexts.append(plan)
            return forward_output([7])

    rendered = recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [
                    resin.recipe.Image(source=np.zeros((2, 4, 3), dtype=np.float32)),
                    resin.recipe.Text("describe"),
                ],
            )
        ]
    )
    engine = Engine(Runner(), scheduler_config=SchedulerConfig(prefill_chunk_size=32))
    engine.submit(
        token_ids=rendered.token_ids,
        media=rendered.media,
        sampling_params=SamplingParams(max_output_tokens=1),
    )
    while engine.has_work():
        engine.step()
    assert len(Runner.contexts) > 1
    assert all(plan.media == (rendered.media,) for plan in Runner.contexts)
    assert sum(plan.num_tokens for plan in Runner.contexts) == len(rendered.token_ids)
