"""Qwen image embeddings and multimodal rotary positions for MLX execution."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from functools import cache

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from resin import BatchPlan
from resin.models.vision import (
    image_grid,
    media_key,
    mrope_positions,
    packed_images,
    span_slices,
)
from resin.recipe import Media


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


class MediaEncoder:
    """Bounded immutable vision features; the Engine owns request lifetimes.

    Entries contain only features and content digests, never media objects or
    request IDs. A new chunk can reuse an image independently of prefix reuse.
    """

    def __init__(
        self,
        visual: Callable[[mx.array, NDArray[np.int64]], mx.array],
        hidden_size: int,
        *,
        patch_size: int = 16,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        max_entries: int = 64,
        max_bytes: int = 256 << 20,
    ) -> None:
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("vision cache limits must be nonnegative")
        self.visual = visual
        self.hidden_size = hidden_size
        self.patch_size, self.merge_size, self.temporal_patch_size = (
            patch_size,
            merge_size,
            temporal_patch_size,
        )
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._features: OrderedDict[str, mx.array] = OrderedDict()
        self.size_bytes = 0

    def features(self, media: tuple[Media, ...]) -> mx.array:
        image_grid(media, self.patch_size, self.merge_size)
        cached = self._features.get(media_key(media))
        if cached is not None:
            self._features.move_to_end(media_key(media))
            return cached
        pixels, grid = packed_images(
            media,
            self.patch_size,
            merge=self.merge_size,
            temporal=self.temporal_patch_size,
        )
        features = self.visual(mx.array(pixels).astype(mx.bfloat16), grid)
        if features.shape != (sum(item.span[1] for item in media), self.hidden_size):
            raise ValueError("Qwen vision features do not match image placeholders")
        mx.eval(features)
        size = features.nbytes
        if self.max_entries > 0 and size <= self.max_bytes:
            while self._features and (
                len(self._features) >= self.max_entries
                or self.size_bytes + size > self.max_bytes
            ):
                _, previous = self._features.popitem(last=False)
                self.size_bytes -= previous.nbytes
            self._features[media_key(media)] = features
            self.size_bytes += size
        return features

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
            slices = tuple(
                span_slices(
                    (item.span for item in media), int(columns[0]), int(columns[-1]) + 1
                )
            )
            if slices:
                features = self.features(media).astype(embeddings.dtype)
                for destination, source in slices:
                    embeddings[selected, destination] = features[source]
        return embeddings, mx.array(positions) if has_media else None
