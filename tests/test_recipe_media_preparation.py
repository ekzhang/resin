"""Native media geometry, ownership, and dependency-free recipe processing."""

from __future__ import annotations

import base64
import io
import json
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

import resin
from resin.recipe import get_native_recipe


class Tokenizer:
    """Minimal tokenizer for testing media without checkpoint files."""

    def __init__(self):
        self.tokens = {}

    def encode_ordinary(self, text):
        return list(text.encode())

    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode())

    def convert_tokens_to_ids(self, text):
        return self.tokens.setdefault(text, 1000 + len(self.tokens))

    def convert_ids_to_tokens(self, token):
        return next(text for text, value in self.tokens.items() if token == value)

    def decode(self, ids, *, skip_special_tokens=False):
        return bytes(ids).decode()


GEMMA_CONFIG = {
    "image_processor": {
        "patch_size": 2,
        "pooling_kernel_size": 1,
        "max_soft_tokens": 70,
        "do_resize": False,
    },
    "video_processor": {
        "patch_size": 2,
        "pooling_kernel_size": 1,
        "max_soft_tokens": 70,
        "do_resize": False,
        "do_sample_frames": False,
    },
}


def render_media(recipe, blocks):
    prompt = recipe.render([resin.recipe.Message(resin.recipe.Role.USER, blocks)])
    assert prompt.media is not None
    assert tuple(prompt.token_ids) == prompt.media.token_ids
    return prompt.media


def assert_buffer_roundtrip(media, *array_fields) -> Any:
    metadata, buffers = resin._resin._media_to_buffers(media)
    assert json.loads(metadata)["version"] == 1
    assert isinstance(buffers, tuple)
    assert buffers[0].dtype == np.uint32
    assert not buffers[0].flags.writeable
    assert tuple(buffers[0]) == media.token_ids
    assert all(value.flags.c_contiguous for value in buffers)
    assert "__reduce__" not in type(media).__dict__
    with pytest.raises(TypeError):
        pickle.dumps(media)
    restored: Any = resin._resin._media_from_buffers(metadata, buffers)
    assert type(restored) is type(media)
    assert restored.token_ids == media.token_ids
    assert restored.token_spans == media.token_spans
    assert restored.digest == media.digest
    assert restored.cache_key == media.cache_key
    for name in array_fields:
        original, copied = getattr(media, name), getattr(restored, name)
        if name != "position_ids":  # Qwen derives mRoPE on registration.
            assert any(value is original for value in buffers), name
        assert np.array_equal(original, copied), name
        assert not np.shares_memory(original, copied), name
    return restored


def test_media_transfer_rejects_unknown_types_versions_and_malformed_buffers():
    custom: Any = SimpleNamespace(token_ids=(1,))
    with pytest.raises(TypeError, match="custom media.*local"):
        resin._resin._media_to_buffers(custom)
    media = resin.recipe.Inkling2Media(
        token_ids=(2, 8),
        image_token_id=8,
        pixel_values=np.ones((1, 2, 2, 3), dtype=np.float32),
        image_grid_thw=np.array([[1, 1, 1]], dtype=np.int64),
    )
    metadata, buffers = resin._resin._media_to_buffers(media)
    descriptor = json.loads(metadata)
    descriptor["version"] = 99
    with pytest.raises(ValueError, match="version"):
        resin._resin._media_from_buffers(json.dumps(descriptor).encode(), buffers)
    descriptor = json.loads(metadata)
    descriptor["media"]["type"] = "builtins.eval"
    with pytest.raises(ValueError, match="unknown variant"):
        resin._resin._media_from_buffers(json.dumps(descriptor).encode(), buffers)
    with pytest.raises(ValueError, match="missing.*buffer"):
        resin._resin._media_from_buffers(metadata, buffers[:-1])
    with pytest.raises(ValueError, match="unexpected.*buffer"):
        resin._resin._media_from_buffers(metadata, (*buffers, buffers[-1]))
    with pytest.raises(TypeError):
        resin._resin._media_from_buffers(
            metadata, (buffers[0].astype(np.int64), *buffers[1:])
        )
    with pytest.raises(ValueError, match="rank 1"):
        resin._resin._media_from_buffers(metadata, (buffers[0][None], *buffers[1:]))
    with pytest.raises(ValueError, match="placeholder count"):
        resin._resin._media_from_buffers(
            metadata, (buffers[0], np.ones((2, 2, 2, 3), dtype=np.float32), buffers[2])
        )


def test_media_transfer_snapshots_receiver_and_keeps_sender_buffers_alive():
    media = resin.recipe.Inkling2Media(
        token_ids=(2, 8),
        image_token_id=8,
        pixel_values=np.ones((1, 2, 2, 3), dtype=np.float32),
        image_grid_thw=np.array([[1, 1, 1]], dtype=np.int64),
    )
    metadata, buffers = resin._resin._media_to_buffers(media)
    assert buffers[1] is media.pixel_values
    del media
    # The token view owns its frozen native payload, including its other arrays.
    assert buffers[0].tolist() == [2, 8]
    for buffer in buffers:
        assert not buffer.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            buffer.setflags(write=True)
    # Transport may deliver mutable buffers; the receiver snapshots those too.
    incoming = tuple(buffer.copy() for buffer in buffers)
    restored: Any = resin._resin._media_from_buffers(metadata, incoming)
    before = restored.pixel_values.copy()
    incoming[1].fill(0)
    assert not restored.pixel_values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        restored.pixel_values.fill(0)
    np.testing.assert_array_equal(restored.pixel_values, before)


def test_image_sources_share_one_owned_rgb_boundary() -> None:
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG")
    raw = encoded.getvalue()
    rgb = pixels.astype(np.float32) / np.float32(255)
    recipe = get_native_recipe(
        "inkling2", Tokenizer(), media_config={"size": {"height": 2, "width": 2}}
    )
    sources = (
        resin.recipe.Image.from_rgb(rgb),
        resin.recipe.Image.from_bytes(raw, media_type="image/png"),
        resin.recipe.Image(
            source="data:image/png;base64," + base64.b64encode(raw).decode()
        ),
    )
    payloads = [render_media(recipe, [source]) for source in sources]
    assert_buffer_roundtrip(payloads[0], "pixel_values")
    for payload in payloads:
        assert np.array_equal(payload.pixel_values, payloads[0].pixel_values)
        assert payload.pixel_values.flags.c_contiguous
        assert not np.shares_memory(payload.pixel_values, rgb)
        assert payload.digest == payloads[0].digest
        assert payload.cache_key == payloads[0].cache_key
    before = payloads[0].pixel_values.copy()
    rgb.fill(0)
    assert np.array_equal(payloads[0].pixel_values, before)


@pytest.mark.parametrize("recipe_name", ["inkling2", "qwen3_5"])
def test_image_only_recipes_reject_video(recipe_name) -> None:
    recipe = get_native_recipe(recipe_name, Tokenizer())
    with pytest.raises(ValueError, match="image|video"):
        render_media(
            recipe,
            [resin.recipe.Video(source=np.zeros((2, 2, 2, 3), dtype=np.float32))],
        )


def test_gemma_interleaved_geometry_and_atomic_spans() -> None:
    recipe = get_native_recipe("gemma4", Tokenizer(), media_config=GEMMA_CONFIG)
    first = np.full((2, 2, 3), 0.25, dtype=np.float32)
    second = np.full((2, 2, 3), 0.75, dtype=np.float32)
    frames = np.stack([np.full_like(first, 0.5), np.ones_like(first)])
    media = render_media(
        recipe,
        [
            resin.recipe.Text("first"),
            resin.recipe.Image.from_rgb(first),
            resin.recipe.Video(source=frames),
            resin.recipe.Image.from_rgb(second),
            resin.recipe.Text("last"),
        ],
    )
    assert media.pixel_values.shape == (2, 70, 12)
    assert media.pixel_values_videos.shape == (1, 2, 70, 12)
    assert np.array_equal(
        media.pixel_values[:, 0], np.stack([first, second]).reshape(2, 12)
    )
    assert np.array_equal(media.pixel_values_videos[0, :, 0], frames.reshape(2, 12))
    assert np.count_nonzero(media.pixel_values[:, 1:]) == 0
    assert np.array_equal(media.image_position_ids[:, 0], np.zeros((2, 2)))
    assert np.all(media.image_position_ids[:, 1:] == -1)
    first_position, second_position = media.image_positions
    assert (
        first_position
        < media.video_positions[0]
        < media.video_positions[1]
        < second_position
    )
    assert [(span.position, span.length) for span in media.token_spans] == [
        (position, 1)
        for position in sorted((*media.image_positions, *media.video_positions))
    ]
    assert media.cache_key[0] == media.token_ids[0]
    assert media.cache_key[-1] == media.token_ids[-1]
    first.fill(0)
    frames.fill(0)
    assert media.pixel_values[0, 0].sum() == 3
    assert media.pixel_values_videos[0, :, 0].sum() == 18
    restored = assert_buffer_roundtrip(
        media,
        "pixel_values",
        "image_position_ids",
        "pixel_values_videos",
        "video_position_ids",
    )
    assert restored.image_positions == media.image_positions
    assert restored.video_positions == media.video_positions


def test_qwen_and_inkling2_native_patch_counts_and_layout() -> None:
    pixels = np.full((4, 8, 3), 0.5, dtype=np.float32)
    qwen = get_native_recipe(
        "qwen3_5",
        Tokenizer(),
        media_config={
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "do_resize": False,
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
        },
    )
    media = render_media(qwen, [resin.recipe.Image.from_rgb(pixels)])
    assert len(media.image_positions) == 2
    assert media.image_grid_thw.tolist() == [[1, 2, 4]]
    assert media.pixel_values.shape == (8, 24)
    assert np.count_nonzero(media.pixel_values) == 0
    assert media.token_spans == ()
    restored = assert_buffer_roundtrip(
        media, "pixel_values", "image_grid_thw", "position_ids"
    )
    assert restored.rope_delta == media.rope_delta
    inkling2 = get_native_recipe(
        "inkling2",
        Tokenizer(),
        media_config={"size": {"height": 2, "width": 2}, "do_normalize": False},
    )
    media = render_media(inkling2, [resin.recipe.Image.from_rgb(pixels)])
    assert len(media.image_positions) == 8
    assert media.pixel_values.shape == (8, 2, 2, 3)
    assert media.image_grid_thw.tolist() == [[1, 2, 4]]
    assert np.all(media.pixel_values == 0.5)
    assert media.token_spans == ()


def test_deepseek_geometry_positions_and_cache_identity() -> None:
    recipe = get_native_recipe(
        "deepseek_v41", Tokenizer(), vision_config={"min_pixels": 0}
    )
    source = np.ones((14, 28, 3), dtype=np.float32)
    blocks = [
        resin.recipe.Image.from_rgb(source),
        resin.recipe.Text("between"),
        resin.recipe.Image.from_rgb(source),
    ]
    media = render_media(recipe, blocks)
    assert len(media.images) == 2
    assert [image.patches.shape for image in media.images] == [(2, 3, 14, 14)] * 2
    assert media.images[0].start + 4 < media.images[1].start
    assert media.token_spans == ()
    repeated = render_media(recipe, blocks)
    assert media.digest == repeated.digest
    assert media.cache_key == repeated.cache_key
    source.fill(0)
    assert np.all(media.images[0].patches == 1)
    restored = assert_buffer_roundtrip(media)
    for actual, expected in zip(restored.images, media.images, strict=True):
        assert "__reduce__" not in type(actual).__dict__
        with pytest.raises(TypeError):
            pickle.dumps(actual)
        assert actual.start == expected.start
        assert actual.n_vit_h == expected.n_vit_h
        assert actual.n_vit_w == expected.n_vit_w
        assert actual.types == expected.types
        assert np.array_equal(actual.patches, expected.patches)
        assert not np.shares_memory(actual.patches, expected.patches)


def test_gemma_text_only_skips_media_embeddings_and_custom_attention_mask() -> None:
    from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner

    runner = object.__new__(Gemma4CudaRunner)
    native: Any = SimpleNamespace(sequence_ids=[7, 8], media=(None, None))
    batch: Any = SimpleNamespace(phase=resin.BatchPhase.PREFILL)
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = hidden.clone()
    features = runner._merge_media_embeddings(native, batch, hidden)
    assert features == {}
    assert torch.equal(hidden, expected)
    assert runner._local_custom_mask(batch, features) is None


def test_native_media_does_not_import_python_image_or_tensor_libraries() -> None:
    import inspect
    import subprocess
    import sys
    import textwrap

    encoded = io.BytesIO()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(encoded, format="PNG")
    script = "import resin\nimport numpy as np\n" + inspect.getsource(Tokenizer)
    script += textwrap.dedent(f"""
        import importlib.abc
        import sys

        class BlockPythonMedia(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {{'PIL', 'torch', 'transformers'}}:
                    raise AssertionError('native media imported ' + fullname)

        sys.meta_path.insert(0, BlockPythonMedia())
        cases = [
            ('gemma4', {{'media_config': {GEMMA_CONFIG!r}}}, (2, 2)),
            ('qwen3_5', {{'media_config': {{'patch_size': 2, 'merge_size': 2, 'do_resize': False}}}}, (4, 4)),
            ('qwen3_8', {{'media_config': {{'patch_size': 2, 'merge_size': 2, 'do_resize': False}}}}, (4, 4)),
            ('glm5_3_flash', {{'media_config': {{'patch_size': 2, 'merge_size': 2, 'do_resize': False}}}}, (4, 4)),
            ('inkling2', {{'media_config': {{'size': {{'height': 2, 'width': 2}}}}}}, (2, 2)),
            ('deepseek_v41', {{'vision_config': {{'min_pixels': 0}}}}, (14, 14)),
        ]
        for name, config, size in cases:
            recipe = resin.recipe.get_native_recipe(name, Tokenizer(), **config)
            plain = recipe.render([resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text('hello')])])
            assert plain.media is None
            image = resin.recipe.Image.from_rgb(np.zeros((*size, 3), dtype=np.float32))
            prompt = recipe.render([resin.recipe.Message(resin.recipe.Role.USER, [image])])
            assert prompt.media is not None
            assert isinstance(prompt.media.token_ids, tuple)
            assert resin._resin._media_from_buffers(*resin._resin._media_to_buffers(prompt.media)).cache_key == prompt.media.cache_key
            if name == 'inkling2':
                encoded = resin.recipe.Image.from_bytes({encoded.getvalue()!r}, media_type='image/png')
                assert recipe.render([resin.recipe.Message(resin.recipe.Role.USER, [encoded])]).media is not None
            if name == 'gemma4':
                video = resin.recipe.Video(source=np.zeros((2, *size, 3), dtype=np.float32))
                assert recipe.render([resin.recipe.Message(resin.recipe.Role.USER, [video])]).media is not None
        assert not {{'PIL', 'torch', 'transformers'}} & sys.modules.keys()
    """)
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
