"""Image features shared by MLX runners across requests and prefill chunks."""

from collections import OrderedDict
from collections.abc import Callable

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from resin.models.vision import image_grid, media_key, packed_images, span_slices
from resin.recipe import Media


class ImageEncoder:
    """Cache evaluated vision features by ordered image digests across chunks and requests."""

    def __init__(
        self,
        visual: Callable[[mx.array, NDArray[np.int64]], mx.array],
        hidden_size: int,
        *,
        patch_size: int = 16,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        channels_last: bool = False,
        max_entries: int = 64,
        max_bytes: int = 256 << 20,
    ) -> None:
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("vision cache limits must be nonnegative")
        self.visual = visual
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.channels_last = channels_last
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._features: OrderedDict[str, mx.array] = OrderedDict()
        self.size_bytes = 0

    def features(self, media: tuple[Media, ...]) -> mx.array:
        key = media_key(media)
        cached = self._features.get(key)
        if cached is not None:
            # The content digest excludes placement, so validate this request's spans.
            image_grid(media, self.patch_size, self.merge_size)
            self._features.move_to_end(key)
            return cached
        pixels, grid = packed_images(
            media,
            self.patch_size,
            merge=self.merge_size,
            temporal=self.temporal_patch_size,
            channels_last=self.channels_last,
        )
        features = self.visual(mx.array(pixels).astype(mx.bfloat16), grid)
        if features.shape != (sum(item.span[1] for item in media), self.hidden_size):
            raise ValueError("vision features do not match image placeholders")
        mx.eval(features)
        size = features.nbytes
        if self.max_entries > 0 and size <= self.max_bytes:
            while self._features and (
                len(self._features) >= self.max_entries
                or self.size_bytes + size > self.max_bytes
            ):
                _, previous = self._features.popitem(last=False)
                self.size_bytes -= previous.nbytes
            self._features[key] = features
            self.size_bytes += size
        return features

    def insert(
        self,
        embeddings: mx.array,
        media: tuple[Media, ...],
        start: int,
        *,
        row: int = 0,
    ) -> mx.array:
        """Replace the image positions in one row of a [batch, tokens, hidden] chunk."""
        slices = tuple(
            span_slices(
                (item.span for item in media), start, start + embeddings.shape[1]
            )
        )
        if slices:
            features = self.features(media).astype(embeddings.dtype)
            for destination, source in slices:
                embeddings[row, destination] = features[source]
        return embeddings
