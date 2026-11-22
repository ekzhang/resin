"""Engine-owned MLX byte storage with layer-major KV pages."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import mlx.core as mx
import numpy as np

from resin import CacheLayout
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin_models_mlx.cache_packing import pack_key_values

PAGE_TOKENS = 128


class SlabGeometry:
    """One unit holds a single layer's keys followed by its values for a page."""

    __slots__ = ()
    kv_heads: int
    head_dim: int
    slab_units: int
    state_units: int
    page_tokens = PAGE_TOKENS
    kv_dtype = mx.bfloat16

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def unit_bytes(self) -> int:
        return 2 * self.page_tokens * self.kv_width * self.kv_dtype.size

    @property
    def slab_bytes(self) -> int:
        return self.slab_units * self.unit_bytes

    @property
    def state_bytes(self) -> int:
        return self.state_units * self.unit_bytes


@dataclass(frozen=True, slots=True)
class KVHistory:
    """Layer-zero page IDs and absolute first block of each retained history."""

    units: mx.array
    bases: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    layer_stride: int
    kv_heads: int

    def gather(self, layer: int, row: int = 0) -> tuple[mx.array, mx.array]:
        """One row's unpadded keys and values, both [1, tokens, KV heads, dim]."""
        page, width = self.units.shape[2:]
        length = max(0, int(self.lengths[row] - self.starts[row] * page))
        bases = self.bases[row, : -(-length // page)]
        pages = mx.take(self.units, mx.array(bases + layer * self.layer_stride), axis=0)
        shape = (1, -1, self.kv_heads, width // self.kv_heads)
        keys = pages[:, 0].reshape(shape)[:, :length]
        values = pages[:, 1].reshape(shape)[:, :length]
        return keys, values


_Geometry = TypeVar("_Geometry", bound=SlabGeometry)


class ByteArena(Generic[_Geometry]):
    """One mutable allocation. A reader's output must precede its next mutation."""

    resource: str

    def __init__(self, layout: CacheLayout, geometry: _Geometry) -> None:
        from resin_models_mlx._native import CacheStorage

        self.layout = layout
        self.geometry = geometry
        self.binding = CachePoolBinding(layout, resource=self.resource)
        self.slabs = layout.parent_capacity + 1  # Includes the reserved zero slab.
        self._storage = CacheStorage(
            (self.slabs, geometry.slab_units, geometry.unit_bytes)
        )
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

    def read_state_part(
        self, starts: np.ndarray, shape: tuple[int, ...], dtype: mx.Dtype
    ) -> mx.array:
        """Gather a unit-aligned payload per row, trimming its padding."""
        unit_bytes = self.geometry.unit_bytes
        elements = math.prod(shape)
        count = math.ceil(elements * dtype.size / unit_bytes)
        units = starts[:, None] // unit_bytes + np.arange(count)
        storage = self.view(dtype, (-1, unit_bytes // dtype.size))
        values = mx.take(storage, mx.array(units.astype(np.int32)), axis=0)
        return values.reshape(len(starts), -1)[:, :elements].reshape(
            len(starts), *shape
        )

    def write_state_part(
        self, starts: Sequence[int], values: mx.array, *, offset: int = 0
    ) -> None:
        for index, start in enumerate(starts):
            self._epoch = self._storage.write_slice(
                self._epoch, start + offset, values[index], after=self._readers
            )
            self._readers.clear()

    def retained_pages(
        self,
        plan: PagedCachePlan,
        group: int,
        rows: np.ndarray,
        lengths: np.ndarray,
        *,
        earliest: np.ndarray | None = None,
    ) -> KVHistory:
        """Cover the retained history with pages. Zero pads shorter and empty rows."""
        table = plan.groups[group]
        first = table.block_table_start_indices[rows].astype(np.int64)
        offsets = table.block_indptr[rows].astype(np.int64)
        available = table.block_indptr[rows + 1] - offsets
        geometry = self.geometry
        lengths = lengths.astype(np.int64, copy=False)
        ends = -(-lengths // geometry.page_tokens)
        counts = np.clip(ends - first, 0, available)
        if earliest is not None:
            skips = np.clip(earliest // geometry.page_tokens - first, 0, counts)
            first += skips
            offsets += skips
            counts -= skips

        bases = np.zeros((len(rows), int(counts.max(initial=1))), np.int64)
        for index, (offset, count) in enumerate(zip(offsets, counts, strict=True)):
            selected = slice(offset, offset + count)
            slabs = table.block_slab_ids[selected].astype(np.int64)
            bases[index, :count] = (
                slabs * geometry.slab_units + table.block_slotnums[selected]
            )
        units = self.view(
            geometry.kv_dtype, (-1, 2, geometry.page_tokens, geometry.kv_width)
        )
        return KVHistory(
            units, bases, first, lengths, table.blocks_per_slab, geometry.kv_heads
        )

    def write_tokens(
        self,
        plan: PagedCachePlan,
        keys: Sequence[mx.array],
        values: Sequence[mx.array],
        tokens: np.ndarray,
        *,
        group: int = 0,
    ) -> None:
        """Write each step's K/V at its validated append addresses, across layers."""
        tokens = tokens.reshape(-1)
        table = plan.groups[group]
        geometry = self.geometry
        slabs = table.append_slab_ids[tokens].astype(np.int64)
        bases = slabs * geometry.slab_units + table.append_slotnums[tokens]
        units = bases[:, None] + np.arange(len(keys)) * table.blocks_per_slab
        rows = (
            units[:, :, None] * (2 * geometry.page_tokens)
            + np.arange(2) * geometry.page_tokens
        )
        rows += table.append_block_offsets[tokens, None, None]
        index_dtype = np.int64 if self.kv_rows_need_i64 else np.int32
        self.scatter_kv(
            mx.array(rows.reshape(-1).astype(index_dtype)), keys, values, len(tokens)
        )

    @property
    def kv_rows_need_i64(self) -> bool:
        return (
            self.nbytes // (self.geometry.kv_width * self.geometry.kv_dtype.size)
            > 2**31
        )

    def scatter_kv(
        self,
        rows: mx.array,
        keys: Sequence[mx.array],
        values: Sequence[mx.array],
        count: int,
    ) -> None:
        """Write K/V to precomputed, leased physical row indices."""
        geometry = self.geometry
        row_bytes = geometry.kv_width * geometry.kv_dtype.size
        index_dtype = mx.int64 if self.kv_rows_need_i64 else mx.int32
        packed = pack_key_values(keys, values, count, geometry.kv_width)
        self._epoch = self._storage.scatter_rows(
            self._epoch,
            rows.reshape(-1).astype(index_dtype),
            packed.astype(geometry.kv_dtype),
            row_bytes=row_bytes,
            after=self._readers,
        )
        self._readers.clear()
