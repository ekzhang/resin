"""Inkling 2 global KV, sliding-window KV and convolution history in one MLX arena.

The arena is one BF16 array of *units* shaped ``[units, 2 * page, kv_width]``.
A unit holds the keys, then the values, of one 128-token page of one layer
visit. A slab belongs to one cache group at a time. Within a KV slab, units are
arranged as ``[layers, blocks_per_slab]``: a visit's pages are consecutive, and
the visits of one logical block are ``blocks_per_slab`` units apart. A STATE
block holds every visit's raw convolution history, ``[visits, 3, width]``,
padded to whole units.

The layout mirrors the CUDA runner without its DSpark groups: GLOBAL keeps full
history, LOCAL keeps a window large enough for its 1,024 readable tokens plus
one prefill chunk of writes, and STATE publishes prefix snapshots.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout, CacheRetention
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin.models.inkling2.config import Inkling2Config
from resin_models_mlx.cache_packing import pack_key_values

PAGE_TOKENS = 128
GLOBAL_GROUP, LOCAL_GROUP, STATE_GROUP = 0, 1, 2
BF16_BYTES = 2


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    global_layers: int
    local_layers: int
    kv_heads: int
    head_dim: int
    state_shape: tuple[int, int, int]
    window_tokens: int
    prefill_chunk_size: int

    @classmethod
    def from_config(
        cls, config: Inkling2Config, prefill_chunk_size: int
    ) -> CacheGeometry:
        if prefill_chunk_size <= 0 or prefill_chunk_size % PAGE_TOKENS:
            raise ValueError(
                "Inkling 2 prefill chunks must be positive multiples of 128"
            )
        return cls(
            global_layers=config.execution_global_layer_count,
            local_layers=config.execution_local_layer_count,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            state_shape=(
                config.execution_layer_count,
                config.sconv_kernel_size - 1,
                config.state_width(),
            ),
            window_tokens=config.sliding_window_size,
            prefill_chunk_size=prefill_chunk_size,
        )

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def page_tokens(self) -> int:
        return PAGE_TOKENS

    @property
    def unit_elements(self) -> int:
        return 2 * PAGE_TOKENS * self.kv_width

    @property
    def state_elements(self) -> int:
        return math.prod(self.state_shape)

    @property
    def state_units(self) -> int:
        return math.ceil(self.state_elements / self.unit_elements)

    @property
    def slab_units(self) -> int:
        # The smallest slab holding whole blocks of every group.
        return math.lcm(self.global_layers, self.local_layers, self.state_units)

    @property
    def slab_bytes(self) -> int:
        return self.slab_units * self.unit_elements * BF16_BYTES

    @property
    def state_bytes(self) -> int:
        return self.state_units * self.unit_elements * BF16_BYTES

    def layers(self, group: int) -> int:
        return self.global_layers if group == GLOBAL_GROUP else self.local_layers

    def blocks_per_slab(self, group: int) -> int:
        if group == STATE_GROUP:
            return self.slab_units // self.state_units
        return self.slab_units // self.layers(group)

    @property
    def local_retained_tokens(self) -> int:
        return self.window_tokens + self.prefill_chunk_size - 1

    def sequence_slabs(self, context: int) -> int:
        """Slabs one request needs at ``context`` tokens, excluding snapshots."""
        global_blocks = math.ceil(context / PAGE_TOKENS)
        local_blocks = min(
            global_blocks, math.ceil(self.local_retained_tokens / PAGE_TOKENS) + 1
        )
        return (
            math.ceil(global_blocks / self.blocks_per_slab(GLOBAL_GROUP))
            + math.ceil(local_blocks / self.blocks_per_slab(LOCAL_GROUP))
            + 1
        )

    def layout(self, slab_capacity: int) -> CacheLayout:
        return CacheLayout(
            parent_capacity=slab_capacity,
            groups=[
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.blocks_per_slab(GLOBAL_GROUP)
                ),
                CacheGroupLayout(
                    PAGE_TOKENS,
                    blocks_per_slab=self.blocks_per_slab(LOCAL_GROUP),
                    retention=CacheRetention.WINDOW,
                    window_tokens=self.local_retained_tokens,
                    read_window_tokens=self.window_tokens,
                ),
                CacheGroupLayout(
                    PAGE_TOKENS,
                    blocks_per_slab=self.blocks_per_slab(STATE_GROUP),
                    family=CacheGroupFamily.STATE,
                ),
            ],
        )


@dataclass(frozen=True, slots=True)
class History:
    """One group's retained earlier tokens for a set of rows.

    ``bases[r, j]`` is the visit-zero unit of row ``r``'s ``j``th retained block,
    whose first token is at ``(starts[r] + j) * 128``. ``lengths[r]`` is the
    row's context; entries at or beyond it are not history.
    """

    units: mx.array
    bases: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    layer_stride: int
    kv_heads: int

    def gather(self, layer: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Keys, values ``[R, L, KV, D]``, absolute positions and validity ``[R, L]``."""
        rows, width = self.bases.shape
        pages = mx.take(
            self.units,
            mx.array((self.bases + layer * self.layer_stride).reshape(-1)),
            axis=0,
        )
        pages = pages.reshape(rows, width, 2, PAGE_TOKENS, self.kv_heads, -1)
        keys = pages[:, :, 0].reshape(rows, width * PAGE_TOKENS, self.kv_heads, -1)
        values = pages[:, :, 1].reshape(rows, width * PAGE_TOKENS, self.kv_heads, -1)
        positions = (self.starts[:, None] * PAGE_TOKENS) + np.arange(
            width * PAGE_TOKENS
        )[None]
        counts = self.lengths - self.starts * PAGE_TOKENS
        offsets = mx.arange(width * PAGE_TOKENS)[None]
        valid = offsets < mx.array(counts.astype(np.int32))[:, None]
        return keys, values, mx.array(positions.astype(np.int32)), valid


class CacheArena:
    """Engine-addressed mutable storage, ordered by an MLX dependency epoch."""

    def __init__(self, layout: CacheLayout, geometry: CacheGeometry) -> None:
        self.layout = layout
        self.geometry = geometry
        self.binding = CachePoolBinding(layout, resource="Inkling 2 MLX cache")
        self.slabs = layout.parent_capacity + 1
        from resin_models_mlx._native import CacheStorage

        self._storage = CacheStorage(
            [self.slabs * geometry.slab_units, 2 * PAGE_TOKENS, geometry.kv_width],
            mx.bfloat16,
        )
        self._epoch = self._storage.initial_epoch()
        self._readers: list[mx.array] = []
        mx.eval(self._epoch)

    @property
    def nbytes(self) -> int:
        return self._storage.nbytes

    @property
    def epoch(self) -> mx.array:
        return self._epoch

    def order_after(self, reads: Sequence[mx.array]) -> None:
        """Queue later mutation after every consumer of the preceding view."""
        self._readers.extend(reads)

    def _read(self) -> mx.array:
        return self._storage.read_epoch(self._epoch)

    def _units(self) -> mx.array:
        return self._read().reshape(-1, 2, PAGE_TOKENS, self.geometry.kv_width)

    def units(self) -> mx.array:
        """The arena as ``[units, 2, page, kv_width]``, ordered after prior writes."""
        return self._units()

    # Reads ---------------------------------------------------------------

    def history(
        self,
        plan: PagedCachePlan,
        group: int,
        rows: np.ndarray,
        contexts: np.ndarray,
        earliest: np.ndarray | None = None,
    ) -> History | None:
        """Retained blocks holding tokens before ``contexts``; ``None`` if empty.

        ``earliest`` optionally skips whole blocks before that position.
        """
        cache_group = plan.groups[group]
        geometry = self.geometry
        stride = geometry.blocks_per_slab(group)
        starts = np.zeros(len(rows), dtype=np.int64)
        tables = []
        for index, row in enumerate(rows.tolist()):
            first = int(cache_group.block_table_start_indices[row])
            end = math.ceil(int(contexts[index]) / PAGE_TOKENS)
            skip = 0
            if earliest is not None:
                skip = max(0, int(earliest[index]) // PAGE_TOKENS - first)
            slabs = cache_group.block_slabs(row).astype(np.int64)
            slots = cache_group.block_slots(row).astype(np.int64)
            count = max(0, end - first)
            selected = slice(min(skip, count), count)
            tables.append(slabs[selected] * geometry.slab_units + slots[selected])
            starts[index] = first + min(skip, count)
        width = max((len(table) for table in tables), default=0)
        if width == 0:
            return None
        bases = np.zeros((len(rows), width), dtype=np.int64)
        for index, table in enumerate(tables):
            bases[index, : len(table)] = table
        return History(
            self._units(),
            bases,
            starts,
            contexts.astype(np.int64),
            stride,
            geometry.kv_heads,
        )

    def state_start(self, slab: int, slot: int) -> int:
        return slab * self.geometry.slab_units + slot * self.geometry.state_units

    def read_states(self, starts: np.ndarray) -> mx.array:
        geometry = self.geometry
        units = starts[:, None] + np.arange(geometry.state_units)[None]
        storage = self._read()
        values = mx.take(storage, mx.array(units.reshape(-1).astype(np.int32)), 0)
        values = values.reshape(len(starts), -1)[:, : geometry.state_elements]
        return values.reshape(len(starts), *geometry.state_shape)

    # Writes --------------------------------------------------------------

    def write_tokens(
        self,
        plan: PagedCachePlan,
        group: int,
        keys: Sequence[mx.array],
        values: Sequence[mx.array],
        tokens: np.ndarray,
    ) -> None:
        """Scatter every visit's K/V rows for flattened plan ``tokens`` into ``group``."""
        cache_group = plan.groups[group]
        geometry = self.geometry
        slab = cache_group.append_slab_ids[tokens].astype(np.int64)
        slot = cache_group.append_slotnums[tokens].astype(np.int64)
        offset = cache_group.append_block_offsets[tokens].astype(np.int64)
        layers = len(keys)
        units = slab[:, None] * geometry.slab_units + (
            slot[:, None] + np.arange(layers) * geometry.blocks_per_slab(group)
        )
        rows = np.arange(2)[None, None, :] * PAGE_TOKENS + offset[:, None, None]
        update = pack_key_values(
            tuple(keys), tuple(values), len(tokens), geometry.kv_width
        )
        indices = units[:, :, None] * (2 * PAGE_TOKENS) + rows
        self._epoch = self._storage.scatter_rows(
            self._epoch,
            mx.array(indices.reshape(-1).astype(np.int32)),
            update.reshape(-1, geometry.kv_width).astype(mx.bfloat16),
            after=self._readers,
        )
        self._readers.clear()

    def write_states(self, starts: Sequence[int], state: mx.array) -> None:
        geometry = self.geometry
        count = len(starts)
        padding = (
            geometry.state_units * geometry.unit_elements - geometry.state_elements
        )
        flat = state.astype(mx.bfloat16).reshape(count, -1)
        if padding:
            flat = mx.concatenate(
                [flat, mx.zeros((count, padding), dtype=mx.bfloat16)], axis=1
            )
        units = flat.reshape(
            count, geometry.state_units, 2 * PAGE_TOKENS, geometry.kv_width
        )
        for index, start in enumerate(starts):
            self._epoch = self._storage.write_slice(
                self._epoch,
                start * geometry.unit_elements,
                units[index],
                after=self._readers,
            )
            self._readers.clear()


__all__ = [
    "GLOBAL_GROUP",
    "LOCAL_GROUP",
    "PAGE_TOKENS",
    "STATE_GROUP",
    "CacheArena",
    "CacheGeometry",
    "History",
]
