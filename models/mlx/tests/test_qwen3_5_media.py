from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin import BatchPlan  # noqa: E402
from resin.recipe import Qwen35Media  # noqa: E402
from resin_models_mlx.qwen3_5.media import MediaEncoder, apply_mrope  # noqa: E402


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_multimodal_rotary_matches_transformers(dtype):
    import torch
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    config = Qwen3_5TextConfig(
        hidden_size=64,
        num_attention_heads=2,
        head_dim=32,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [6, 5, 5],
            "mrope_interleaved": True,
        },
    )
    values = np.random.default_rng(12).normal(size=(2, 2, 7, 32)).astype(np.float32)
    positions = np.random.default_rng(13).integers(0, 100, size=(3, 2, 7))
    tensor = torch.from_numpy(values).to(getattr(torch, dtype))
    rotary = Qwen3_5TextRotaryEmbedding(config)
    cosine, sine = rotary(tensor, torch.from_numpy(positions))
    expected, _ = apply_rotary_pos_emb(tensor, tensor, cosine, sine)
    actual = apply_mrope(
        mx.array(values).astype(getattr(mx, dtype)),
        mx.array(positions),
        32,
        10000.0,
        (6, 5, 5),
    )
    tolerance = 2e-2 if dtype == "bfloat16" else 2e-6
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        expected.float().numpy(),
        atol=tolerance,
        rtol=tolerance,
    )


def _media(value=1.0):
    return Qwen35Media(
        token_ids=(2, 8, 8, 8, 8, 3),
        image_positions=(1, 2, 3, 4),
        pixel_values=np.full((16, 12), value, dtype=np.float32),
        image_grid_thw=np.array([[1, 4, 4]], dtype=np.int64),
        spatial_merge_size=2,
        image_token_id=8,
    )


def test_media_chunks_merge_features_and_keep_decode_rotary_delta():
    calls = []

    def visual(pixels, grid):
        calls.append(grid.copy())
        return mx.arange(4 * 8).reshape(4, 8).astype(mx.bfloat16)

    encoder = MediaEncoder(visual, 8)
    media = _media()
    plan = cast(BatchPlan, SimpleNamespace(media=(media, None)))
    rows = np.array([0])
    first, positions = encoder.prepare(
        plan, rows, np.array([[2, 8, 8]]), np.array([0]), mx.zeros((1, 3, 8))
    )
    np.testing.assert_array_equal(np.array(first[0, 0]), np.zeros(8))
    np.testing.assert_array_equal(np.array(first[0, 1:]), np.arange(16).reshape(2, 8))
    np.testing.assert_array_equal(np.array(positions)[:, 0], media.position_ids[:, :3])
    second, positions = encoder.prepare(
        plan, rows, np.array([[8, 8, 3]]), np.array([3]), mx.zeros((1, 3, 8))
    )
    np.testing.assert_array_equal(
        np.array(second[0, :2]), np.arange(16, 32).reshape(2, 8)
    )
    np.testing.assert_array_equal(np.array(positions)[:, 0], media.position_ids[:, 3:])
    assert len(calls) == 1

    _, positions = encoder.prepare(
        plan,
        np.array([0, 1]),
        np.array([[9], [9]]),
        np.array([7, 7]),
        mx.zeros((2, 1, 8)),
    )
    np.testing.assert_array_equal(np.array(positions)[:, :, 0], [[5, 7]] * 3)
    with pytest.raises(ValueError, match="scheduled token"):
        encoder.prepare(
            plan, rows, np.array([[99]]), np.array([1]), mx.zeros((1, 1, 8))
        )


def test_vision_cache_uses_pixels_and_obeys_entry_limit():
    calls = []

    def visual(pixels, grid):
        calls.append(1)
        return mx.ones((4, 8)) * pixels[0, 0]

    encoder = MediaEncoder(visual, 8, max_entries=1)
    first = encoder.features(_media())
    assert encoder.features(_media()) is first
    second = encoder.features(_media(2.0))
    assert first is not second
    np.testing.assert_array_equal(np.array(second), np.full((4, 8), 2.0))
    encoder.features(_media())
    assert len(calls) == 3
    assert encoder.size_bytes == first.nbytes
