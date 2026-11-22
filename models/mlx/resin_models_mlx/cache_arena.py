"""Engine-owned MLX byte storage and the page layout used by paged attention."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import mlx.core as mx
import numpy as np

from resin import CacheLayout
from resin.backend.plans import CacheGroupView
from resin.backend.resources import CachePoolBinding
from resin_models_mlx.cache_packing import pack_key_values


class SlabGeometry:
    """Physical byte layout shared by the model's KV and STATE cache groups."""

    __slots__ = ()
    kv_page: KVPage
    slab_units: int
    state_units: int

    @property
    def slab_bytes(self) -> int:
        return self.slab_units * self.kv_page.nbytes

    @property
    def state_bytes(self) -> int:
        return self.state_units * self.kv_page.nbytes

    @property
    def kv_width(self) -> int:
        return self.kv_page.width


_Geometry = TypeVar("_Geometry", bound=SlabGeometry)


class ByteArena(Generic[_Geometry]):
    """One mutable allocation. A reader's output must precede its next mutation."""

    resource: str

    def __init__(self, layout: CacheLayout, geometry: _Geometry) -> None:
        from resin_models_mlx._native import CacheStorage

        self.layout = layout
        self.geometry = geometry
        self.page = geometry.kv_page
        self.binding = CachePoolBinding(layout, resource=self.resource)
        self.slabs = layout.parent_capacity + 1  # Includes the reserved zero slab.
        shape = (self.slabs, geometry.slab_units, self.page.nbytes)
        self._storage = CacheStorage(shape)
        self._epoch = self._storage.initial_epoch()
        self._readers: list[mx.array] = []

    @property
    def nbytes(self) -> int:
        return self._storage.nbytes

    @property
    def buffer_identity(self) -> int:
        return self._storage.buffer_identity

    @property
    def epoch(self) -> mx.array:
        return self._epoch

    def state_start(self, slab: int, slot: int) -> int:
        return slab * self.geometry.slab_bytes + slot * self.geometry.state_bytes

    def order_after(self, reads: Sequence[mx.array]) -> None:
        self._readers.extend(reads)

    def view(self, dtype: mx.Dtype, shape: tuple[int, ...]) -> mx.array:
        """Alias the contiguous allocation after its last write, without conversion."""
        return self._storage.read_epoch(self._epoch).view(dtype).reshape(shape)

    def gather_units(
        self, starts: np.ndarray, count: int, unit_bytes: int, dtype: mx.Dtype
    ) -> mx.array:
        """Gather `count` whole contiguous units at each unit-aligned byte start."""
        units = starts[:, None] // unit_bytes + np.arange(count)
        storage = self.view(dtype, (-1, unit_bytes // dtype.size))
        return mx.take(storage, mx.array(units.astype(np.int32)), axis=0)

    def write_slice(self, byte_start: int, value: mx.array) -> None:
        self._epoch = self._storage.write_slice(
            self._epoch, byte_start, value, after=self._readers
        )
        self._readers.clear()

    def scatter_rows(self, rows: mx.array, values: mx.array, *, row_bytes: int) -> None:
        self._epoch = self._storage.scatter_rows(
            self._epoch, rows, values, row_bytes=row_bytes, after=self._readers
        )
        self._readers.clear()


@dataclass(frozen=True, slots=True)
class KVPage:
    """One layer's keys followed by its values, each stored in token order."""

    tokens: int
    width: int
    dtype: mx.Dtype

    @property
    def row_bytes(self) -> int:
        return self.width * self.dtype.size

    @property
    def nbytes(self) -> int:
        return 2 * self.tokens * self.row_bytes

    def view(self, arena: ByteArena) -> mx.array:
        return arena.view(self.dtype, (-1, 2, self.tokens, self.width))

    def retained_table(
        self,
        group: CacheGroupView,
        rows: np.ndarray,
        lengths: np.ndarray,
        slab_units: int,
        *,
        earliest: np.ndarray | None = None,
        min_width: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Layer-zero page IDs and absolute first block per row, omitting new tokens."""
        first = group.block_table_start_indices[rows].astype(np.int64)
        offsets = group.block_indptr[rows].astype(np.int64)
        available = group.block_indptr[rows + 1] - offsets
        ends = (lengths.astype(np.int64) + self.tokens - 1) // self.tokens
        counts = np.clip(ends - first, 0, available)
        starts = first
        if earliest is not None:
            skips = np.clip(earliest // self.tokens - first, 0, counts)
            starts = first + skips
            offsets += skips
            counts -= skips

        width = int(counts.max(initial=min_width))
        bases = np.zeros((len(rows), width), np.int64)
        for index, (offset, count) in enumerate(zip(offsets, counts, strict=True)):
            selected = slice(offset, offset + count)
            slabs = group.block_slab_ids[selected].astype(np.int64)
            bases[index, :count] = slabs * slab_units + group.block_slotnums[selected]
        return bases, starts

    def write(
        self,
        arena: ByteArena,
        group: CacheGroupView,
        keys: Sequence[mx.array],
        values: Sequence[mx.array],
        tokens: np.ndarray,
        *,
        slab_units: int,
        layer_stride: int,
    ) -> None:
        slabs = group.append_slab_ids[tokens].astype(np.int64)
        slots = group.append_slotnums[tokens].astype(np.int64)
        offsets = group.append_block_offsets[tokens].astype(np.int64)
        bases = slabs * slab_units + slots
        units = bases[:, None] + np.arange(len(keys)) * layer_stride
        rows = units[:, :, None] * (2 * self.tokens)
        rows = rows + np.arange(2) * self.tokens + offsets[:, None, None]
        # Row IDs can outgrow int32 when a large allocation holds narrow rows.
        index_dtype = np.int64 if arena.nbytes // self.row_bytes > 2**31 else np.int32
        packed = pack_key_values(keys, values, len(tokens), self.width)
        arena.scatter_rows(
            mx.array(rows.reshape(-1).astype(index_dtype, copy=False)),
            packed.astype(self.dtype),
            row_bytes=self.row_bytes,
        )
