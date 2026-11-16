"""Image reuse preserves per-request placement and immutable buffer ownership."""

from __future__ import annotations

import base64
import gc
import io
import weakref

import numpy as np
import pytest
from PIL import Image as PilImage
from test_recipe import _byte_tokenizer
from test_recipe_media_preparation import GEMMA_CONFIG, Tokenizer

import resin
from resin.recipe import (
    Gemma4Media,
    Image,
    Inkling2Media,
    Message,
    RenderedPrompt,
    Role,
    Text,
    Video,
    get_native_recipe,
)


def tokenizer():
    backend = _byte_tokenizer()
    backend.add_special_tokens(
        [
            "<|message_user|>",
            "<|message_system|>",
            "<|message_tool|>",
            "<|content_image|>",
            "<|unused_200054|>",
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜deepseek_image｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<｜System｜>",
            "<｜latest_reminder｜>",
            "<think>",
            "</think>",
            "<bos>",
            "<eos>",
            "<|image|>",
            "<|video|>",
            "<|image>",
            "<image|>",
            "<|turn>",
            "<turn|>",
            "<|channel>",
            "<channel|>",
            "<|think|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "[gMASK]",
            "<sop>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|begin_of_image|>",
            "<|end_of_image|>",
        ]
    )
    return backend


CASES = [
    ("deepseek_v41", {"vision_config": {"min_pixels": 0}}, (14, 28)),
    ("gemma4", {"media_config": GEMMA_CONFIG}, (4, 8)),
    ("inkling2", {"media_config": {"size": {"height": 2, "width": 2}}}, (4, 8)),
    (
        "qwen3_5",
        {"media_config": {"patch_size": 2, "merge_size": 2, "do_resize": False}},
        (4, 8),
    ),
    (
        "qwen3_8",
        {"media_config": {"patch_size": 2, "merge_size": 2, "do_resize": False}},
        (4, 8),
    ),
    (
        "glm5_3_flash",
        {"media_config": {"patch_size": 2, "merge_size": 2, "do_resize": False}},
        (4, 8),
    ),
]


def encoded_image(shape=(4, 8), color=(220, 30, 10), *, data_url=False):
    pixels = np.empty((*shape, 3), dtype=np.uint8)
    pixels[:] = color
    output = io.BytesIO()
    PilImage.fromarray(pixels).save(output, format="PNG")
    raw = output.getvalue()
    if data_url:
        return Image(source="data:image/png;base64," + base64.b64encode(raw).decode())
    return Image.from_bytes(raw, media_type="image/png")


def messages(*images, lead="Describe: "):
    return [Message(Role.USER, [Text(lead), *images])]


def inkling2_pixels(prompt: RenderedPrompt):
    media = prompt.media
    assert isinstance(media, Inkling2Media)
    pixels = media.pixel_values
    assert pixels is not None
    return pixels


def gemma_video_pixels(prompt: RenderedPrompt):
    media = prompt.media
    assert isinstance(media, Gemma4Media)
    pixels = media.pixel_values_videos
    assert pixels is not None
    return pixels


def assert_readonly_buffers(buffers):
    for buffer in buffers:
        assert not buffer.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            buffer.flat[0] = 123
        with pytest.raises(ValueError, match="WRITEABLE"):
            buffer.setflags(write=True)


def assert_same_preparation(actual, expected):
    np.testing.assert_array_equal(actual.token_ids, expected.token_ids)
    assert actual.media is not None and expected.media is not None
    actual_meta, actual_buffers = resin._resin._media_to_buffers(actual.media)
    expected_meta, expected_buffers = resin._resin._media_to_buffers(expected.media)
    assert actual_meta == expected_meta
    assert_readonly_buffers(actual_buffers)
    assert_readonly_buffers(expected_buffers)
    for left, right in zip(actual_buffers, expected_buffers, strict=True):
        np.testing.assert_array_equal(left, right)
        assert not np.shares_memory(left, right)


@pytest.mark.parametrize("name,config,shape", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("data_url", [False, True], ids=["bytes", "data-url"])
def test_cached_images_rebind_changed_history_without_sharing_arrays(
    name, config, shape, data_url
):
    backend = tokenizer()
    recipe = get_native_recipe(name, backend, **config)
    first = encoded_image(shape, data_url=data_url)
    second = encoded_image(shape, (10, 30, 220), data_url=data_url)
    original = messages(first, second)
    initial = recipe.render(original)
    info = recipe._media_preparation_cache_info()
    assert info["entries"] == info["misses"] == 1
    assert info["hits"] == 0 and 0 < info["retained_bytes"] < 128 * 1024**2

    # Native payload digests remain valid because output arrays are immutable.
    assert initial.media is not None
    _, buffers = resin._resin._media_to_buffers(initial.media)
    assert_readonly_buffers(buffers)

    continuation = original + [
        Message(Role.ASSISTANT, [Text("Red and blue.")]),
        Message(Role.USER, [Text("Which is first?")]),
    ]
    cached = recipe.render(continuation)
    fresh = get_native_recipe(name, backend, **config).render(continuation)
    assert_same_preparation(cached, fresh)
    assert recipe._media_preparation_cache_info()["hits"] == 1

    # A longer prefix changes absolute positions, but not source preprocessing.
    moved = messages(first, second, lead="A longer instruction before both images: ")
    assert_same_preparation(
        recipe.render(moved), get_native_recipe(name, backend, **config).render(moved)
    )
    assert recipe._media_preparation_cache_info()["hits"] == 2

    # Order, changed pixels at the same geometry, and changed geometry are misses.
    variants = [
        messages(second, first),
        messages(first, encoded_image(shape, (10, 220, 30), data_url=data_url)),
        messages(first, encoded_image((shape[0], shape[1] // 2), data_url=data_url)),
    ]
    for count, variant in enumerate(variants, start=2):
        assert_same_preparation(
            recipe.render(variant),
            get_native_recipe(name, backend, **config).render(variant),
        )
        assert recipe._media_preparation_cache_info()["misses"] == count


@pytest.mark.parametrize("name,config,shape", CASES, ids=[case[0] for case in CASES])
def test_image_transport_roundtrip_preserves_immutable_digest(name, config, shape):
    recipe = get_native_recipe(name, tokenizer(), **config)
    prompt = recipe.render(messages(encoded_image(shape)))
    assert prompt.media is not None
    metadata, original_buffers = resin._resin._media_to_buffers(prompt.media)
    mutable_buffers = tuple(buffer.copy() for buffer in original_buffers)
    restored = resin._resin._media_from_buffers(metadata, mutable_buffers)
    assert type(restored) is type(prompt.media)
    assert restored.cache_key == prompt.media.cache_key
    assert getattr(restored, "digest") == getattr(prompt.media, "digest")
    restored_meta, restored_buffers = resin._resin._media_to_buffers(restored)
    assert restored_meta == metadata
    assert_readonly_buffers(restored_buffers)
    for source, original, received in zip(
        mutable_buffers, original_buffers, restored_buffers, strict=True
    ):
        assert not np.shares_memory(source, received)
        source.fill(0)
        np.testing.assert_array_equal(received, original)
    assert getattr(restored, "digest") == getattr(prompt.media, "digest")


def test_recipe_image_cache_is_bounded_and_recipe_configuration_is_local():
    backend = tokenizer()
    config = {"size": {"height": 2, "width": 2}}
    recipe = get_native_recipe("inkling2", backend, media_config=config)
    images = [encoded_image(color=(value, 30, 90)) for value in range(10)]
    for image in images[:8]:
        recipe.render(messages(image))
    recipe.render(messages(images[0]))  # Protect the oldest source from eviction.
    recipe.render(messages(images[8]))
    info = recipe._media_preparation_cache_info()
    assert info["entries"] == 8 and info["misses"] == 9 and info["hits"] == 1
    recipe.render(messages(images[0]))
    recipe.render(messages(images[1]))  # The least recently used source was evicted.
    info = recipe._media_preparation_cache_info()
    assert info["entries"] == 8 and info["misses"] == 10 and info["hits"] == 2
    assert info["retained_bytes"] <= 128 * 1024**2

    other = get_native_recipe(
        "inkling2", backend, media_config={**config, "do_normalize": False}
    )
    normalized = inkling2_pixels(recipe.render(messages(images[0])))
    unnormalized = inkling2_pixels(other.render(messages(images[0])))
    assert not np.array_equal(normalized, unnormalized)
    assert other._media_preparation_cache_info()["misses"] == 1


def test_mutable_arrays_and_python_tokenizers_bypass_image_cache():
    config = {"size": {"height": 2, "width": 2}}
    native = get_native_recipe("inkling2", tokenizer(), media_config=config)
    pixels = np.full((4, 8, 3), 0.25, dtype=np.float32)
    request = messages(Image.from_rgb(pixels))
    first = inkling2_pixels(native.render(request)).copy()
    pixels.fill(0.75)
    second = inkling2_pixels(native.render(request))
    assert not np.array_equal(first, second)
    assert native._media_preparation_cache_info() == {
        "entries": 0,
        "retained_bytes": 0,
        "hits": 0,
        "misses": 0,
    }
    encoded = bytearray(encoded_image().source[0])
    request = messages(Image(source=(encoded, "image/png")))
    first = inkling2_pixels(native.render(request)).copy()
    encoded[:] = encoded_image(color=(10, 30, 220)).source[0]
    second = inkling2_pixels(native.render(request))
    assert not np.array_equal(first, second)
    assert native._media_preparation_cache_info()["entries"] == 0
    custom = get_native_recipe("inkling2", Tokenizer(), media_config=config)
    custom.render(messages(encoded_image()))
    custom.render(messages(encoded_image()))
    assert custom._media_preparation_cache_info()["entries"] == 0


def test_mixed_video_requests_bypass_image_cache_and_keep_frame_updates():
    native = get_native_recipe("gemma4", tokenizer(), media_config=GEMMA_CONFIG)
    frames = np.zeros((2, 4, 8, 3), dtype=np.float32)
    request = [Message(Role.USER, [encoded_image(), Video(source=frames)])]
    first = gemma_video_pixels(native.render(request)).copy()
    frames.fill(0.75)
    second = gemma_video_pixels(native.render(request))
    assert not np.array_equal(first, second)
    assert native._media_preparation_cache_info()["entries"] == 0


def test_image_cache_retains_neither_python_sources_nor_bound_buffers():
    class DataUrl(str):
        pass

    native = get_native_recipe(
        "inkling2", tokenizer(), media_config={"size": {"height": 2, "width": 2}}
    )
    source = DataUrl(encoded_image(data_url=True).source)
    source_ref = weakref.ref(source)
    request = messages(Image(source=source))
    prompt = native.render(request)
    pixels_ref = weakref.ref(inkling2_pixels(prompt))
    del source, request, prompt
    gc.collect()
    assert native._media_preparation_cache_info()["entries"] == 1
    assert source_ref() is None
    assert pixels_ref() is None


def test_image_cache_does_not_hide_invalid_mime_or_changed_payload():
    native = get_native_recipe(
        "inkling2", tokenizer(), media_config={"size": {"height": 2, "width": 2}}
    )
    image = encoded_image()
    native.render(messages(image))
    raw, _ = image.source
    with pytest.raises(ValueError, match="MIME|mime|unsupported"):
        native.render(messages(Image.from_bytes(raw, media_type="image/not-supported")))
    with pytest.raises(ValueError):
        native.render(
            messages(Image.from_bytes(b"invalid PNG", media_type="image/png"))
        )
    assert native._media_preparation_cache_info()["entries"] == 1
