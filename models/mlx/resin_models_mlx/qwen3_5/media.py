"""Qwen image embeddings and multimodal rotary positions for MLX execution."""

from __future__ import annotations

from functools import cache

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from resin import BatchPlan
from resin.models.vision import mrope_positions
from resin_models_mlx.media import ImageEncoder


@cache
def _inverse_frequencies(rotary_dim: int, rope_theta: float) -> mx.array:
    # Frequencies depend only on the checkpoint. Compute them once on the CPU:
    # Metal's approximate power introduces avoidable phase error at long positions.
    exponents = np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim
    return mx.array(1.0 / (np.float32(rope_theta) ** exponents))


def apply_mrope(
    value: mx.array,
    positions: mx.array,
    rotary_dim: int,
    rope_theta: float,
    sections: tuple[int, int, int] | None,
) -> mx.array:
    """Apply Qwen's interleaved temporal, height and width rotary frequencies.

    ``value`` is [batch, heads, tokens, head_dim]; ``positions`` is [3, batch,
    tokens]. Text and generated positions have equal values on all three axes.
    """
    if sections is None:
        raise ValueError("3-D Qwen positions require mrope_section")
    if positions.shape != (3, value.shape[0], value.shape[2]):
        raise ValueError("Qwen rotary positions must have shape [3, batch, tokens]")
    inverse = _inverse_frequencies(rotary_dim, rope_theta)
    frequencies = positions.astype(mx.float32)[..., None] * inverse
    indices = mx.arange(rotary_dim // 2)
    height = (indices % 3 == 1) & (indices < 3 * sections[1])
    width = (indices % 3 == 2) & (indices < 3 * sections[2])
    angles = mx.where(
        height, frequencies[1], mx.where(width, frequencies[2], frequencies[0])
    )
    cosine = mx.cos(angles).astype(value.dtype)[:, None]
    sine = mx.sin(angles).astype(value.dtype)[:, None]
    left, right = mx.split(value[..., :rotary_dim], 2, axis=-1)
    rotated = mx.concatenate(
        [left * cosine - right * sine, right * cosine + left * sine], axis=-1
    )
    return mx.concatenate([rotated, value[..., rotary_dim:]], axis=-1)


class MediaEncoder(ImageEncoder):
    """Place image features and construct Qwen's three-axis rotary positions."""

    def prepare(
        self,
        plan: BatchPlan,
        rows: NDArray[np.int64],
        token_ids: NDArray[np.int64],
        contexts: NDArray[np.int64],
        embeddings: mx.array,
    ) -> tuple[mx.array, mx.array | None]:
        """Merge a scheduled prompt slice and construct positions for its tokens."""
        width = token_ids.shape[1]
        absolute = contexts[:, None] + np.arange(width, dtype=np.int64)
        positions = np.broadcast_to(absolute, (3, *absolute.shape)).copy()
        has_media = False
        for selected, row in enumerate(rows):
            media = plan.media[int(row)]
            if media is None:
                continue
            has_media = True
            columns = absolute[selected]
            positions[:, selected] = mrope_positions(
                media, columns, self.patch_size, self.merge_size
            )
            embeddings = self.insert(embeddings, media, int(columns[0]), row=selected)
        return embeddings, mx.array(positions) if has_media else None
