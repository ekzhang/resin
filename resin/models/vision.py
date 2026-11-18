"""Array layouts shared by vision runners; recipes supply normalized RGB bitmaps."""

from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from resin.protocols import MediaTokenSpan
from resin.recipe import Media


def media_key(media: Sequence[Media]) -> str:
    return ":".join(item.digest for item in media)


def media_identity(
    media: Sequence[Media],
) -> tuple[tuple[str, MediaTokenSpan, tuple[MediaTokenSpan, ...]], ...]:
    return tuple((item.digest, item.span, item.video_frame_spans) for item in media)


def span_slices(
    spans: Iterable[MediaTokenSpan], start: int, end: int
) -> Iterator[tuple[slice, slice]]:
    """Map only intersecting prompt ranges to their concatenated feature rows."""
    if start < 0 or end < start:
        raise ValueError("prefill bounds must be nonnegative and ordered")
    offset = 0
    for position, length in spans:
        lower, upper = max(start, position), min(end, position + length)
        if lower < upper:
            yield (
                slice(lower - start, upper - start),
                slice(offset + lower - position, offset + upper - position),
            )
        offset += length


def image_grid(
    media: Sequence[Media], patch_size: int, merge: int = 1
) -> NDArray[np.int64]:
    grids = []
    for item in media:
        if item.image is None:
            raise ValueError("this vision encoder requires image media")
        height, width, _ = item.image.shape
        side = patch_size * merge
        if (
            height % side
            or width % side
            or (height // side) * (width // side) != item.span[1]
        ):
            raise ValueError("image geometry does not match its placeholder span")
        grids.append((1, height // patch_size, width // patch_size))
    return np.asarray(grids, dtype=np.int64)


def image_patches(
    image: NDArray[np.float32],
    patch_size: int,
    *,
    merge: int = 1,
    temporal: int = 1,
    channels_last: bool = False,
) -> NDArray[np.float32]:
    """Pack HWC once before encoding, in merge-block-major patch order."""
    height, width, channels = image.shape
    p, m = patch_size, merge
    if height % (p * m) or width % (p * m):
        raise ValueError(
            "image dimensions must align to the vision patch and merge sizes"
        )
    patches = image.reshape(height // (p * m), m, p, width // (p * m), m, p, channels)
    if channels_last:
        return np.require(
            patches.transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, p, p, channels),
            requirements=("C", "W"),
        )
    patches = patches.transpose(0, 3, 1, 4, 6, 2, 5).reshape(-1, channels, 1, p, p)
    if temporal != 1:
        patches = np.repeat(patches, temporal, axis=2)
    return np.require(
        patches.reshape(-1, channels * temporal * p * p), requirements=("C", "W")
    )


def packed_images(
    media: Sequence[Media],
    patch_size: int,
    *,
    merge: int = 1,
    temporal: int = 1,
    channels_last: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    grid = image_grid(media, patch_size, merge)
    patches = []
    for item in media:
        image = item.image
        assert image is not None  # image_grid checked the input modality.
        patches.append(
            image_patches(
                image,
                patch_size,
                merge=merge,
                temporal=temporal,
                channels_last=channels_last,
            )
        )
    return (patches[0] if len(patches) == 1 else np.concatenate(patches)), grid


def mrope_delta(media: Sequence[Media], patch_size: int, merge: int) -> int:
    return int(
        sum(
            max(grid[1:]) // merge - item.span[1]
            for item, grid in zip(
                media, image_grid(media, patch_size, merge), strict=True
            )
        )
    )


def mrope_positions(
    media: Sequence[Media], absolute: NDArray[np.int64], patch_size: int, merge: int
) -> NDArray[np.int64]:
    """Construct only the scheduled Qwen rotary positions, including decode."""
    positions = np.broadcast_to(absolute, (3, *absolute.shape)).copy()
    delta = 0
    for item, (_, height, width) in zip(
        media, image_grid(media, patch_size, merge), strict=True
    ):
        start, length = item.span
        height, width = height // merge, width // merge
        inside = (absolute >= start) & (absolute < start + length)
        offset = absolute[inside] - start
        positions[:, inside] = (
            np.stack((np.zeros_like(offset), offset // width, offset % width))
            + start
            + delta
        )
        delta += max(height, width) - length
        positions[:, absolute >= start + length] = (
            absolute[absolute >= start + length] + delta
        )
    return positions


def padded_patches(
    images: Sequence[NDArray[np.float32]], patch_size: int
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Batch spatial RGB patches and XY coordinates, padding only to this batch."""
    patches, positions = [], []
    for image in images:
        height, width = (size // patch_size for size in image.shape[:2])
        patches.append(
            image_patches(image, patch_size, channels_last=True).reshape(
                height * width, -1
            )
        )
        y, x = np.indices((height, width))
        positions.append(np.stack((x, y), -1).reshape(-1, 2))
    count = max(map(len, patches))
    return (
        np.stack(
            [np.pad(value, ((0, count - len(value)), (0, 0))) for value in patches]
        ),
        np.stack(
            [
                np.pad(value, ((0, count - len(value)), (0, 0)), constant_values=-1)
                for value in positions
            ]
        ),
    )
