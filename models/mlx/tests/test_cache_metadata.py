"""CPU plan metadata preserves logical pages and has a stable Metal ABI."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin.backend.plans import PagedCachePlan  # noqa: E402
from resin_models_mlx.qwen3_5.cache import CacheArena, CacheGeometry  # noqa: E402


@pytest.mark.parametrize(
    "history",
    [
        [0],
        [129],
        [511, 513],
        [8192, 17, 0, 7937],
        [1] * 8,
        [1] * 9,
    ],
)
def test_paged_metadata_preserves_logical_columns_and_live_row_order(history):
    # The 0.8B slab holds thirteen consecutive pages for each of six layers.
    geometry = CacheGeometry(
        full_layers=6,
        kv_heads=2,
        head_dim=256,
        recurrent_shape=(18, 16, 128, 128),
        convolution_shape=(18, 3, 6144),
        slab_units=78,
        state_units=75,
    )
    arena = CacheArena(geometry.layout(1), geometry)
    selected = np.arange(len(history), dtype=np.int64)[::-1] * 3 + 2
    lengths = np.asarray(history, dtype=np.int64)
    width = max(1, math.ceil(max(history) / 128))

    def slots(row):
        # Reusing physical pages in this metadata-only fixture keeps the arena
        # small; the independent SDPA tests verify actual unique-page contents.
        return (np.arange(width, dtype=np.int64) + row) % 13

    group = SimpleNamespace(
        block_slabs=lambda _row: np.ones(width, dtype=np.int64),
        block_slots=slots,
    )
    plan = cast(PagedCachePlan, SimpleNamespace(groups=[group]))
    paged = arena.paged_history(plan, selected, lengths)
    original_lengths = lengths.copy()
    lengths[:] = -1
    pages, stored_lengths = np.array(paged.pages), np.array(paged.lengths)
    assert pages.dtype == stored_lengths.dtype == np.int32
    assert pages.shape[1] == width
    # Metadata stays in the device-address-space regime, including B1 at zero
    # history; row padding must never increase logical attention partitions.
    assert pages.size >= 8 and stored_lengths.size >= 8
    expected = np.zeros((len(history), width), dtype=np.int32)
    for index, (row, length) in enumerate(zip(selected, original_lengths, strict=True)):
        blocks = math.ceil(int(length) / 128)
        expected[index, :blocks] = 78 + slots(row)[:blocks]
        for layer in range(6):
            np.testing.assert_array_equal(
                pages[index, :blocks] + layer * paged.layer_stride,
                78 + layer * 13 + slots(row)[:blocks],
            )
    assert paged.layer_stride == 13
    np.testing.assert_array_equal(pages[: len(history)], expected)
    np.testing.assert_array_equal(stored_lengths[: len(history)], original_lengths)
    assert not pages[len(history) :].any()
    assert not stored_lengths[len(history) :].any()
    # A subsequent launch owns different metadata and cannot overwrite a graph
    # that still holds the first launch's arrays.
    arena.paged_history(plan, selected, np.zeros(len(history), dtype=np.int64))
    np.testing.assert_array_equal(np.array(paged.pages), pages)
    np.testing.assert_array_equal(np.array(paged.lengths), stored_lengths)
