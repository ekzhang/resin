"""Image recipe equivalence, cache identity and scheduler-neutral ownership."""

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
from resin import Engine, MediaInput, RunnerConfig, SamplingParams, SchedulerConfig
from resin.recipe import (
    DeepSeekV41Image,
    DeepSeekV41Media,
    get_native_recipe,
)

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
    rgb = _pattern(case["width"], case["height"])
    media = recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [resin.recipe.Image(source=rgb.astype(np.float32) / 255)],
            )
        ]
    ).media
    assert isinstance(media, DeepSeekV41Media)
    image = media.images[0]
    assert (image.n_vit_h, image.n_vit_w) == tuple(case["grid"])
    height, width = case["llm_grid"]
    assert image.types == (0, *((1,) * width + (2,)) * height, 3)
    if case["max_wh_ratio"] is None:
        expected_tokens = (
            MultiModalData(
                [ImageInfo(b"", case["width"], case["height"])]
            ).image_token_adjustment()
            + 1
        )
        assert len(image.types) == expected_tokens
    assert len(image.types) <= 1024
    assert image.patches.dtype == np.float32
    assert image.patches.flags.c_contiguous
    assert np.isfinite(image.patches).all()
    assert np.all(image.patches >= -1) and np.all(image.patches <= 1)
    # Native resize need not reproduce Pillow pixels, but the tower still sees
    # exactly BF16-rounded values with the checkpoint's geometry and packing.
    expected = torch.from_numpy(image.patches).to(torch.bfloat16).float().numpy()
    np.testing.assert_array_equal(image.patches, expected)


def test_native_patch_order_and_normalization_are_exact_without_resizing(recipe):
    rgb = _pattern(28, 14)
    source = rgb.astype(np.float32) / 255
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
    assert isinstance(media, DeepSeekV41Media)
    image = media.images[0]
    expected = torch.from_numpy(source).permute(2, 0, 1)
    expected = ((expected - 0.5) / 0.5).to(torch.bfloat16).float()
    expected = expected.reshape(3, 1, 14, 2, 14).permute(1, 3, 0, 2, 4)
    np.testing.assert_array_equal(image.patches, expected.reshape(2, 3, 14, 14).numpy())
    before = image.patches.copy()
    source.fill(0)
    np.testing.assert_array_equal(image.patches, before)


def test_media_recipe_expands_only_explicit_images_and_owns_inputs(recipe):
    rgb = _pattern(17, 31)
    second = np.zeros((2, 4, 3), dtype=np.float32)
    messages = [
        resin.recipe.Message(
            resin.recipe.Role.USER,
            [
                resin.recipe.Text("literal " + IMAGE_TOKEN),
                resin.recipe.Image(source=_data_url(rgb)),
                resin.recipe.Text("compare"),
                resin.recipe.Image(source=second),
            ],
        )
    ]
    rendered = recipe.render(
        messages,
        options=resin.recipe.RenderOptions(
            thinking_mode="thinking", reasoning_effort=37
        ),
    )
    media = rendered.media
    assert isinstance(media, MediaInput)
    assert isinstance(media, DeepSeekV41Media)
    assert len(media.images) == 2
    assert media.token_ids == tuple(rendered.token_ids)
    assert recipe.stop_token_ids
    assert media.token_spans == ()
    positions = {
        position
        for image in media.images
        for position in range(image.start, image.start + len(image.types))
    }
    assert rendered.token_ids.tolist().count(media.image_token_id) == len(positions)
    assert all(
        (token != key) == (index in positions)
        for index, (token, key) in enumerate(
            zip(media.token_ids, media.cache_key, strict=True)
        )
    )
    assert IMAGE_TOKEN in recipe.tokenizer.decode(
        rendered.token_ids[: media.images[0].start], skip_special_tokens=False
    )
    second.fill(1)
    assert np.all(media.images[1].patches < 0)


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
        ).media

    black, white, black_again = render(0), render(1), render(0)
    assert black.token_ids == white.token_ids
    assert black.cache_key != white.cache_key
    assert black.digest == black_again.digest
    assert black.cache_key == black_again.cache_key


def test_prepared_media_rejects_bad_layout_and_owns_patch_storage():
    patches = np.ones((1, 3, 14, 14), dtype=np.float32)
    image = DeepSeekV41Image(
        start=1, patches=patches, n_vit_h=1, n_vit_w=1, types=(0, 1, 2, 3)
    )
    patches.fill(0)
    assert np.all(image.patches == 1)
    media = DeepSeekV41Media(
        token_ids=(7, 129264, 129264, 129264, 129264), images=(image,)
    )
    with pytest.raises(ValueError, match="different placeholder"):
        DeepSeekV41Media(token_ids=(7, 129264, 129264, 129264, 8), images=(image,))
    with pytest.raises(ValueError, match="outside an image span"):
        DeepSeekV41Media(token_ids=(*media.token_ids, 129264), images=(image,))
    with pytest.raises(ValueError, match="3x3 aligner"):
        DeepSeekV41Image(
            start=1, patches=image.patches, n_vit_h=1, n_vit_w=1, types=(0, 1, 3)
        )
    with pytest.raises(ValueError, match="ordered, disjoint"):
        DeepSeekV41Media(token_ids=media.token_ids, images=(image, image))


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
        media=rendered.media, sampling_params=SamplingParams(max_output_tokens=1)
    )
    while engine.has_work():
        engine.step()
    assert len(Runner.contexts) > 1
    assert all(plan.media == (rendered.media,) for plan in Runner.contexts)
    assert sum(plan.num_tokens for plan in Runner.contexts) == len(rendered.token_ids)
