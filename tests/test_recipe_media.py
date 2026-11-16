"""Optional parity checks with real pinned processors already in the HF cache.

No model weights or network access are needed. Missing cached processors skip;
fetch their tokenizer/processor files to run the same checks on another machine.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from recipe_fuzz.inputs import native_inputs

import resin
from resin.recipe import get_native_recipe

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
    (
        "gemma4",
        "nvidia/Gemma-4-31B-IT-NVFP4",
        "4135a98a9b728a548947683219633b25682223ac",
    ),
)


@pytest.fixture(scope="module", params=MODELS, ids=lambda model: model[0])
def model_processor(request):
    from transformers import AutoProcessor

    model, repo, revision = request.param
    try:
        processor = AutoProcessor.from_pretrained(
            repo, revision=revision, local_files_only=True, trust_remote_code=False
        )
    except OSError:
        pytest.skip(f"pinned processor {repo}@{revision} is not cached")  # ty: ignore[too-many-positional-arguments]
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


def assert_reference_tensors(
    processor, recipe, messages, *, tools=None, resized_pixels=False
):
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
    assert actual.media is not None
    for key, value in expected.items():
        # The engine stores the prepared payload, without HF's batch/mask shell.
        if isinstance(value, torch.Tensor) and hasattr(actual.media, key):
            prepared = getattr(actual.media, key)
            if key == "mm_token_type_ids":
                assert tuple(value[0].tolist()) == prepared
            else:
                reference = value.numpy()
                assert reference.shape == prepared.shape, key
                assert reference.dtype == prepared.dtype, key
                if resized_pixels and key.startswith("pixel_values"):
                    # The Rust resize filter need not match Torchvision's pixel
                    # arithmetic. Placement and padding geometry remain exact;
                    # no-resize pixel parity is checked independently below.
                    np.testing.assert_allclose(
                        reference, prepared, atol=0.01, rtol=0, err_msg=key
                    )
                else:
                    assert np.array_equal(reference, prepared), key
    return actual


def test_native_recipe_preserves_real_image_geometry_and_tools(model_processor):
    model, processor, recipe = model_processor
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
    assert_reference_tensors(
        processor, recipe, messages, tools=tools, resized_pixels=model != "inkling2"
    )


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
    assert actual is not None
    for key, value in expected.items():
        if isinstance(value, torch.Tensor) and hasattr(actual, key):
            assert np.array_equal(value.numpy(), getattr(actual, key)), key


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
def test_native_recipe_preserves_gemma_video_timestamps_and_geometry(
    model_processor, prefix, suffix
):
    model, processor, recipe = model_processor
    if model != "gemma4":
        pytest.skip("only the Gemma runner implements video inputs")  # ty: ignore[too-many-positional-arguments]
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
    assert_reference_tensors(processor, recipe, messages, resized_pixels=True)


def test_native_recipe_preserves_gemma_mixed_media_order(model_processor):
    model, processor, recipe = model_processor
    if model != "gemma4":
        pytest.skip("only the Gemma runner implements mixed image/video inputs")  # ty: ignore[too-many-positional-arguments]
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
    assert_reference_tensors(processor, recipe, messages, resized_pixels=True)
