"""Optional parity checks with real pinned processors already in the HF cache.

No model weights or network access are needed. Missing cached processors skip;
fetch their tokenizer/processor files to run the same checks on another machine.
"""

from __future__ import annotations

import numpy as np
import pytest
from recipe_fuzz.inputs import native_inputs

import resin
from resin.models.vision import packed_images, padded_patches
from resin.recipe import get_native_recipe

GEMMA = (
    "gemma4",
    "nvidia/Gemma-4-31B-IT-NVFP4",
    "4135a98a9b728a548947683219633b25682223ac",
)
GEMMA_ONLY = pytest.mark.parametrize(
    "model_processor", [GEMMA], indirect=True, ids=["gemma4"]
)
MODELS = (
    (
        "inkling2",
        "thinkingmachines/Inkling-2-Small",
        "2ad1157016e1aac77b6ff1b770e7c05229403134",
    ),
    (
        "qwen3_5",
        "RadixArk/Qwen3.8-27B-NVFP4",
        "554ebba9b5f1b79dc11246341960360e6ef05ef4",
    ),
    GEMMA,
)


@pytest.fixture(scope="module", params=MODELS, ids=lambda model: model[0])
def model_processor(request):
    from transformers import AutoProcessor, PreTrainedTokenizerBase

    model, repo, revision = request.param
    try:
        processor = AutoProcessor.from_pretrained(
            repo, revision=revision, local_files_only=True, trust_remote_code=False
        )
    except OSError:
        pytest.skip(f"pinned processor {repo}@{revision} is not cached")  # ty: ignore[too-many-positional-arguments]
    if model == "inkling2" and isinstance(processor, PreTrainedTokenizerBase):
        pytest.skip("installed Transformers has no Inkling 2 image processor")  # ty: ignore[too-many-positional-arguments]
    recipe_name = "qwen3_8" if model == "qwen3_5" else model
    image_config = processor.image_processor.to_dict()
    config = (
        {
            "image_processor": image_config,
            "video_processor": processor.video_processor.to_dict(),
        }
        if model == "gemma4"
        else image_config
    )
    return (
        model,
        processor,
        get_native_recipe(recipe_name, processor.tokenizer, media_config=config),
    )


def image():
    return np.linspace(0, 1, 64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)


def video():
    return np.linspace(0, 1, 32 * 64 * 64 * 3, dtype=np.float32).reshape(32, 64, 64, 3)


def assert_pixels(model, processor, media, expected, *, atol):
    assert media
    config = processor.image_processor
    if model == "gemma4":
        # HF pads to a fixed budget. The runner pads to the longest item.
        for field, position_field, items in (
            (
                "pixel_values",
                "image_position_ids",
                [m.image for m in media if m.image is not None],
            ),
            (
                "pixel_values_videos",
                "video_position_ids",
                [f for m in media for f in m.video or ()],
            ),
        ):
            if field not in expected:
                assert not items
                continue
            values, positions = padded_patches(
                items,
                config.patch_size
                if field == "pixel_values"
                else processor.video_processor.patch_size,
            )
            ref_values, ref_positions = expected[field], expected[position_field]
            if field == "pixel_values_videos":
                ref_values, ref_positions = (
                    ref_values.flatten(0, 1),
                    ref_positions.flatten(0, 1),
                )
            count = values.shape[1]
            np.testing.assert_array_equal(positions, ref_positions[:, :count])
            assert (ref_positions[:, count:] == -1).all()
            np.testing.assert_allclose(values, ref_values[:, :count], atol=atol, rtol=0)
    else:
        if model == "inkling2":
            values, grid = packed_images(
                media, config.size["height"], channels_last=True
            )
        else:
            values, grid = packed_images(
                media,
                config.patch_size,
                merge=config.merge_size,
                temporal=config.temporal_patch_size,
            )
        np.testing.assert_array_equal(grid, expected["image_grid_thw"])
        np.testing.assert_allclose(values, expected["pixel_values"], atol=atol, rtol=0)


def assert_reference_tensors(model_processor, messages, *, tools=None):
    model, processor, recipe = model_processor
    request = {"messages": messages, "tools": tools}
    expected = processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"do_rescale": False},
    )
    actual = recipe.render(**native_inputs(request))
    assert actual.token_ids.tolist() == expected["input_ids"][0].tolist()
    # Resize filters can differ slightly. Geometry and unresized pixels match exactly.
    assert_pixels(
        model,
        processor,
        actual.media,
        expected,
        atol=0 if model == "inkling2" else 0.01,
    )


def test_native_recipe_preserves_real_image_geometry_and_tools(model_processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe "},
                {"type": "image", "image": image()},
                {"type": "text", "text": " then "},
                {"type": "image", "image": image()[:, ::-1].copy()},
            ],
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a label.",
                "parameters": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                    "required": ["label"],
                },
            },
        }
    ]
    assert_reference_tensors(model_processor, messages, tools=tools)


def test_native_recipe_matches_unresized_reference_pixels(model_processor):
    model, processor, _recipe = model_processor
    config = processor.image_processor.to_dict()
    if model == "inkling2":
        factor = config["size"]["height"]
    else:
        config["do_resize"] = False
        factor = config["patch_size"] * config.get(
            "pooling_kernel_size", config.get("merge_size", 1)
        )
    pixels = np.linspace(0, 1, factor * factor * 2 * 3, dtype=np.float32).reshape(
        factor, factor * 2, 3
    )
    reference_processor = type(processor.image_processor)(**config)
    expected = reference_processor(
        images=[pixels], do_rescale=False, return_tensors="pt"
    )
    native_config = {"image_processor": config} if model == "gemma4" else config
    configured = get_native_recipe(
        "qwen3_8" if model == "qwen3_5" else model,
        processor.tokenizer,
        media_config=native_config,
    )
    actual = configured.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER, [resin.recipe.Image.from_rgb(pixels)]
            )
        ]
    ).media
    assert_pixels(model, processor, actual, expected, atol=0)


def test_literal_media_marker_does_not_acquire_a_pixel_slot(model_processor):
    _, processor, recipe = model_processor
    text = f"The literal spelling {processor.image_token} is ordinary text."
    plain = recipe.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text(text)])]
    )
    assert processor.image_token_id not in plain.token_ids
    pixels = processor.image_processor(
        images=[image()], return_tensors="pt", do_rescale=False
    )
    expansion = processor.replace_image_token(pixels, image_idx=0)
    expected_slots = processor.tokenizer.encode(
        expansion, add_special_tokens=False
    ).count(processor.image_token_id)
    actual = recipe.render(
        [
            resin.recipe.Message(
                resin.recipe.Role.USER,
                [resin.recipe.Text(text), resin.recipe.Image(source=image())],
            ),
        ]
    )

    assert actual.token_ids.tolist().count(processor.image_token_id) == expected_slots


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [
        ("Compare frame 12", "34 and describe."),
        ("Compare video:", ": describe."),
        ("Compare video ", " then describe."),
    ],
    ids=["digits", "colon", "space"],
)
@GEMMA_ONLY
def test_native_recipe_preserves_gemma_video_timestamps_and_geometry(
    model_processor, prefix, suffix
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prefix},
                {"type": "video", "video": video()},
                {"type": "text", "text": suffix},
            ],
        }
    ]
    assert_reference_tensors(model_processor, messages)


@GEMMA_ONLY
def test_native_recipe_preserves_gemma_mixed_media_order(model_processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First image: "},
                {"type": "image", "image": image()},
                {"type": "text", "text": " Compare with video 12:"},
                {"type": "video", "video": video()},
                {"type": "text", "text": " Then this differently shaped image: "},
                {"type": "image", "image": image()[:, ::-2].copy()},
                {"type": "text", "text": " Explain the differences."},
            ],
        }
    ]
    assert_reference_tensors(model_processor, messages)
