"""GLM bitmap preprocessing and runner packing against pinned HF fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_glm_recipe import backend as backend

import resin._resin as native
from resin.models.vision import image_grid, packed_images
from resin.recipe import Image, Message, Role, Text, Video, get_native_recipe

FIXTURE = Path(__file__).parent / "recipe_fuzz/templates/glm53_images_hf.npz"


def render(backend, images, **config):
    recipe = get_native_recipe("glm5_3_flash", backend, media_config=config)
    return recipe.render(
        [Message(Role.USER, [Text("Inspect"), *[Image(source=i) for i in images]])]
    )


def test_native_processor_matches_pinned_hf_geometry_pixels_and_padding(backend):
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        metadata = json.loads(str(fixture["metadata"]))
        assert (
            metadata["transformers_revision"]
            == "50e35cc0ca18ec19b80653690de553e6decf74f5"
        )
        for case in metadata["cases"]:
            name, config = case["name"], case["config"]
            media = render(backend, [fixture[name + "_input"]], **config).media
            assert media and len(media) == 1 and isinstance(media[0], native.Media)
            pixels, grid = packed_images(
                media,
                config.get("patch_size", 14),
                merge=config.get("merge_size", 2),
                temporal=config.get("temporal_patch_size", 2),
            )
            np.testing.assert_array_equal(grid, fixture[name + "_grid"])
            np.testing.assert_allclose(
                pixels, fixture[name + "_pixels"], atol=1e-5, rtol=1e-5, err_msg=name
            )
            assert (
                media[0].span.length
                == np.prod(grid) // config.get("merge_size", 2) ** 2
            )


def test_image_order_cache_identity_and_distributed_transport(backend):
    first = np.zeros((28, 56, 3), dtype=np.float32)
    second = np.ones((56, 28, 3), dtype=np.float32)
    prompt = render(backend, [first, second], min_image_tokens=1)
    media = prompt.media
    assert media is not None
    pixels, grid = packed_images(media, 14, merge=2, temporal=2)
    assert grid.tolist() == [[1, 2, 4], [1, 4, 2]]
    before = []
    for item, source in zip(media, [first, second], strict=True):
        image = item.image
        single = render(backend, [source], min_image_tokens=1).media
        assert image is not None and single is not None
        np.testing.assert_array_equal(image, single[0].image)
        before.append(image.copy())
    positions = [p for item in media for p in range(item.span.position, sum(item.span))]
    assert len(positions) == 4
    assert all(prompt.token_ids[p] == prompt.token_ids[positions[0]] for p in positions)
    assert positions[2] - positions[1] == 3
    assert not any(item.atomic for item in media)
    first.fill(0.5)
    second.fill(0.25)
    changed = render(backend, [first, second], min_image_tokens=1)
    assert changed.media is not None
    np.testing.assert_array_equal(changed.token_ids, prompt.token_ids)
    for item, different, stored in zip(media, changed.media, before, strict=True):
        np.testing.assert_array_equal(item.image, stored)
        assert item.digest != different.digest
    restored = native._media_from_buffers(*native._media_to_buffers(media))
    for item, copy in zip(media, restored, strict=True):
        assert copy.span == item.span and copy.digest == item.digest
        assert not np.shares_memory(item.image, copy.image)
        np.testing.assert_array_equal(copy.image, item.image)
    restored_pixels, restored_grid = packed_images(restored, 14, merge=2, temporal=2)
    np.testing.assert_array_equal(restored_pixels, pixels)
    np.testing.assert_array_equal(restored_grid, grid)


def test_payload_rejects_mismatched_layout_and_video(backend):
    with pytest.raises(ValueError, match="HWC"):
        native.Media((0, 1), image=np.zeros((1, 3, 14, 14), np.float32))
    unaligned = native.Media((0, 1), image=np.zeros((15, 28, 3), np.float32))
    with pytest.raises(ValueError, match="geometry"):
        image_grid((unaligned,), 14)
    with pytest.raises(ValueError, match="geometry"):
        packed_images((unaligned,), 14, merge=2, temporal=2)
    recipe = get_native_recipe("glm5_3_flash", backend)
    with pytest.raises(ValueError, match="video"):
        recipe.render(
            [Message(Role.USER, [Video(source=np.zeros((2, 28, 28, 3), np.float32))])]
        )


@pytest.mark.parametrize(
    "config",
    [
        {"patch_size": 0},
        {"max_image_tokens": 0},
        {"min_image_tokens": 20, "max_image_tokens": 10},
        {"resample": 1},
        {"image_std": [1, 0, 1]},
        {"do_resize": False},
    ],
)
def test_invalid_image_geometry_is_rejected_before_packing(backend, config):
    with pytest.raises(ValueError, match="GLM"):
        render(backend, [np.zeros((7, 11, 3), np.float32)], **config)
