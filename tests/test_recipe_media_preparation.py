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
from resin.recipe import Media, get_native_recipe


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
    assert isinstance(prompt.media, tuple) and prompt.media
    assert all(isinstance(item, Media) for item in prompt.media)
    assert all(
        item.span.position + item.span.length <= len(prompt.token_ids)
        for item in prompt.media
    )
    return prompt.media


def assert_buffer_roundtrip(media):
    metadata, buffers = resin._resin._media_to_buffers(media)
    assert json.loads(metadata)["version"] == 2
    assert isinstance(buffers, tuple)
    assert all(
        value.dtype == np.float32 and value.flags.c_contiguous for value in buffers
    )
    assert all(not value.flags.writeable for value in buffers)
    restored = resin._resin._media_from_buffers(metadata, buffers)
    assert isinstance(restored, tuple)
    copied_metadata, copied_buffers = resin._resin._media_to_buffers(restored)
    assert copied_metadata == metadata
    for original, copied in zip(media, restored, strict=True):
        assert isinstance(copied, Media)
        assert copied.digest == original.digest
        assert copied.span == original.span
        assert copied.atomic == original.atomic
        assert copied.video_frame_spans == original.video_frame_spans
        assert copied.video_frame_indices == original.video_frame_indices
        with pytest.raises(TypeError):
            pickle.dumps(original)
    for original, copied in zip(buffers, copied_buffers, strict=True):
        np.testing.assert_array_equal(original, copied)
        assert not np.shares_memory(original, copied)
        assert not copied.flags.writeable
    return restored


def test_media_transfer_rejects_unknown_types_versions_and_malformed_buffers():
    with pytest.raises(TypeError):
        resin._resin._media_to_buffers((SimpleNamespace(token_ids=(1,)),))  # ty: ignore[invalid-argument-type]
    image = np.ones((2, 2, 3), dtype=np.float32)
    media = (Media((1, 1), image=image),)
    metadata, buffers = resin._resin._media_to_buffers(media)
    descriptor = json.loads(metadata)
    descriptor["version"] = 99
    with pytest.raises(ValueError, match="version"):
        resin._resin._media_from_buffers(json.dumps(descriptor).encode(), buffers)
    descriptor = json.loads(metadata)
    descriptor["items"][0]["content"]["kind"] = "builtins.eval"
    with pytest.raises(ValueError, match="unknown variant"):
        resin._resin._media_from_buffers(json.dumps(descriptor).encode(), buffers)
    with pytest.raises(ValueError, match="missing.*buffer"):
        resin._resin._media_from_buffers(metadata, ())
    with pytest.raises(ValueError, match="unexpected.*buffer"):
        resin._resin._media_from_buffers(metadata, (*buffers, buffers[0]))
    with pytest.raises(TypeError):
        resin._resin._media_from_buffers(metadata, (image.astype(np.int64),))
    for invalid in (image[None], np.full_like(image, np.nan)):
        with pytest.raises(ValueError, match="finite.*HWC"):
            resin._resin._media_from_buffers(metadata, (invalid,))
    for span in ((0, 0), (2**32 - 1, 1)):
        with pytest.raises(ValueError, match="spans"):
            Media(span, image=image)
    with pytest.raises(ValueError, match="ordered"):
        resin._resin._media_to_buffers((media[0], media[0]))

    # The same transport supports PCM without introducing a model payload type.
    pcm = np.linspace(-1, 1, 16, dtype=np.float32)
    audio = Media((2, 4), audio=pcm, audio_sample_rate=24000)
    restored = assert_buffer_roundtrip((audio,))[0]
    assert (
        restored.audio_sample_rate == 24000 and restored.image is restored.video is None
    )
    np.testing.assert_array_equal(restored.audio, pcm)
    assert Media((2, 4), audio=pcm, audio_sample_rate=16000).digest != audio.digest
    for sample_rate in (None, 0):
        with pytest.raises(ValueError, match="sample rate|sample_rate"):
            Media((2, 4), audio=pcm, audio_sample_rate=sample_rate)
    with pytest.raises(ValueError, match="exactly one"):
        Media((2, 4), audio=pcm, image=image, audio_sample_rate=24000)


def test_media_transfer_snapshots_receiver_and_keeps_sender_buffers_alive():
    source = np.ones((2, 2, 3), dtype=np.float32)
    media = (Media((1, 1), image=source),)
    metadata, buffers = resin._resin._media_to_buffers(media)
    assert buffers[0] is media[0].image
    source.fill(0)
    del media
    assert np.all(buffers[0] == 1)
    for buffer in buffers:
        assert not buffer.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            buffer.setflags(write=True)
    # Transport may deliver mutable buffers; the receiver snapshots those too.
    incoming = tuple(buffer.copy() for buffer in buffers)
    restored = resin._resin._media_from_buffers(metadata, incoming)
    incoming[0].fill(0)
    assert restored[0].image is not None
    assert not restored[0].image.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        restored[0].image.fill(0)
    assert np.all(restored[0].image == 1)


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
    assert_buffer_roundtrip(payloads[0])
    for (payload,) in payloads:
        np.testing.assert_array_equal(payload.image, payloads[0][0].image)
        assert payload.image.flags.c_contiguous
        assert not np.shares_memory(payload.image, rgb)
        assert payload.digest == payloads[0][0].digest
    before = payloads[0][0].image.copy()
    rgb.fill(0)
    np.testing.assert_array_equal(payloads[0][0].image, before)


@pytest.mark.parametrize("data_url", [False, True], ids=["bytes", "data-url"])
@pytest.mark.parametrize(
    "alpha,light,dark",
    [
        (0, (204, 204, 204), (153, 153, 153)),
        (128, (112, 152, 222), (86, 126, 197)),
        (255, (20, 100, 240), (20, 100, 240)),
    ],
    ids=["transparent", "half-transparent", "opaque"],
)
def test_inkling_png_alpha_uses_eight_pixel_checkerboard(data_url, alpha, light, dark):
    rgba = np.empty((16, 16, 4), dtype=np.uint8)
    rgba[:] = (20, 100, 240, alpha)
    encoded = io.BytesIO()
    Image.fromarray(rgba).save(encoded, format="PNG")
    source = (
        resin.recipe.Image(
            source="data:image/png;base64,"
            + base64.b64encode(encoded.getvalue()).decode()
        )
        if data_url
        else resin.recipe.Image.from_bytes(encoded.getvalue(), media_type="image/png")
    )
    checkerboard = np.empty((16, 16, 3), dtype=np.uint8)
    checkerboard[:8, :8] = checkerboard[8:, 8:] = light
    checkerboard[:8, 8:] = checkerboard[8:, :8] = dark
    for name, config, expected in (
        (
            "inkling2",
            {"size": {"height": 2, "width": 2}, "do_normalize": False},
            checkerboard,
        ),
        ("gemma4", GEMMA_CONFIG, rgba[:, :, :3]),
    ):
        recipe = get_native_recipe(name, Tokenizer(), media_config=config)
        (media,) = render_media(recipe, [source])
        np.testing.assert_allclose(
            media.image, expected.astype(np.float32) / 255, rtol=0, atol=1e-7
        )


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
            resin.recipe.Video(
                source={"frames": frames, "fps": 2.0, "frame_indices": [3, 9]}
            ),
            resin.recipe.Image.from_rgb(second),
            resin.recipe.Text("last"),
        ],
    )
    left, video, right = media
    np.testing.assert_array_equal(left.image, first)
    np.testing.assert_array_equal(right.image, second)
    assert isinstance(video.video, tuple)
    np.testing.assert_array_equal(np.stack(video.video), frames)
    assert left.span.length == right.span.length == 1
    assert all(item.atomic for item in media)
    assert left.span.position < video.span.position < right.span.position
    assert video.video_fps == 2.0 and video.video_frame_indices == (3, 9)
    assert len(video.video_frame_spans) == 2
    assert all(span.length == 1 for span in video.video_frame_spans)
    assert video.span.position <= video.video_frame_spans[0].position
    assert (
        video.video_frame_spans[-1].position < video.span.position + video.span.length
    )
    assert video.video_frame_spans[0].position + 1 < video.video_frame_spans[1].position
    first.fill(0)
    frames.fill(0)
    assert left.image.sum() == 3
    assert sum(frame.sum() for frame in video.video) == 18
    assert_buffer_roundtrip(media)

    # Content identity follows frame timing and relative placement, not prompt position.
    kwargs: dict[str, Any] = dict(
        video=video.video,
        video_fps=video.video_fps,
        video_frame_indices=video.video_frame_indices,
        atomic=True,
    )
    moved = Media(
        (video.span.position + 10, video.span.length),
        video_frame_spans=[(p + 10, n) for p, n in video.video_frame_spans],
        **kwargs,
    )
    assert moved.digest == video.digest
    kwargs["video_fps"] = 3.0
    changed = Media(video.span, video_frame_spans=video.video_frame_spans, **kwargs)
    assert changed.digest != video.digest
    with pytest.raises(ValueError, match="frame spans"):
        Media(
            video.span,
            video_frame_spans=[(0, 1), video.video_frame_spans[-1]],
            **kwargs,
        )
    for invalid_video in (list(video.video), np.stack(video.video)):
        kwargs["video"] = invalid_video
        with pytest.raises(TypeError, match="tuple"):
            Media(video.span, video_frame_spans=video.video_frame_spans, **kwargs)


def test_qwen_and_inkling2_native_bitmap_geometry_and_token_counts() -> None:
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
    (media,) = render_media(qwen, [resin.recipe.Image.from_rgb(pixels)])
    assert media.span.length == 2 and not media.atomic
    assert media.image.shape == (4, 8, 3)
    assert np.count_nonzero(media.image) == 0
    assert_buffer_roundtrip((media,))
    inkling2 = get_native_recipe(
        "inkling2",
        Tokenizer(),
        media_config={"size": {"height": 2, "width": 2}, "do_normalize": False},
    )
    (media,) = render_media(inkling2, [resin.recipe.Image.from_rgb(pixels)])
    assert media.span.length == 8 and not media.atomic
    np.testing.assert_array_equal(media.image, pixels)


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
    assert len(media) == 2
    assert all(
        item.image.shape == (14, 28, 3) and item.span.length == 4 for item in media
    )
    assert media[0].span.position + 4 < media[1].span.position
    assert all(not item.atomic for item in media)
    repeated = render_media(recipe, blocks)
    assert [item.digest for item in media] == [item.digest for item in repeated]
    source.fill(0)
    assert np.all(media[0].image == 1)
    assert_buffer_roundtrip(media)


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
            assert isinstance(prompt.media, tuple) and isinstance(prompt.media[0], resin.recipe.Media)
            restored = resin._resin._media_from_buffers(*resin._resin._media_to_buffers(prompt.media))
            assert [item.digest for item in restored] == [item.digest for item in prompt.media]
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
