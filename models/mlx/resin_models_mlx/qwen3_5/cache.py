"""Fungible full-attention and DeltaNet-state slabs in one MLX arena.

The arena is one BF16 array of *units* shaped ``[units, 2 * page, kv_width]``.
A unit holds the keys, then the values, of one 128-token page of one
full-attention layer. Within a KV slab, units are arranged as
``[full_layers, kv_blocks_per_slab]``: each layer's pages are consecutive,
and the layers of a logical KV block are separated by ``kv_blocks_per_slab``
units. A DeltaNet checkpoint is packed into whole units: each linear
layer's FP32 recurrent state fills whole units, followed by the BF16
convolution tails of every layer.

Reads gather with ``take`` or use a private view for paged attention. Keys and
values scatter single rows; DeltaNet snapshots use contiguous slice updates.
Native resource operations order every write after prior readers on the GPU,
without rebinding an immutable MLX arena value or requiring buffer donation.
Live DeltaNet state stays outside the arena between steps (see ``state.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.cache_packing import pack_key_values
from resin_models_mlx.qwen3_5.model import (
    AttentionHistory,
    LinearState,
    PagedHistory,
)

PAGE_TOKENS = 128
PREFILL_CHUNK_SIZE = 4096
KV_GROUP = 0
STATE_GROUP = 1

BF16_BYTES = 2
FP32_BYTES = 4
# MLX 0.32.4 uses constant-address-space kernel arguments below this size.
# Keep metadata in device buffers so new lengths/batches do not change the ABI.
_MIN_DEVICE_ELEMENTS = 8


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """Shared slab capacity with layer-major KV and contiguous STATE units.

    KV unit ``(slab, layer, slot)`` is ``slab * slab_units +
    layer * kv_blocks_per_slab + slot``. STATE blocks retain their
    contiguous ``slot * state_units`` placement within a slab.
    """

    full_layers: int
    kv_heads: int
    head_dim: int
    recurrent_shape: tuple[int, int, int, int]
    convolution_shape: tuple[int, int, int]
    slab_units: int
    state_units: int

    @classmethod
    def from_config(cls, config: Qwen35Config) -> CacheGeometry:
        unit_elements = 2 * PAGE_TOKENS * config.num_key_value_heads * config.head_dim
        recurrent_shape = (
            config.linear_layer_count,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            config.linear_key_head_dim,
        )
        convolution_shape = (
            config.linear_layer_count,
            config.linear_conv_kernel_dim - 1,
            config.linear_conv_dim,
        )
        kv_width = config.num_key_value_heads * config.head_dim
        if (
            math.prod(recurrent_shape[1:]) * FP32_BYTES // BF16_BYTES
        ) % unit_elements or (math.prod(convolution_shape[1:]) % kv_width):
            raise ValueError("each layer's DeltaNet state must align to cache units")
        state_elements = math.prod(
            recurrent_shape
        ) * FP32_BYTES // BF16_BYTES + math.prod(convolution_shape)
        state_units = math.ceil(state_elements / unit_elements)
        layers = config.full_layer_count
        return cls(
            full_layers=layers,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            recurrent_shape=recurrent_shape,
            convolution_shape=convolution_shape,
            slab_units=math.ceil(state_units / layers) * layers,
            state_units=state_units,
        )

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def unit_elements(self) -> int:
        return 2 * PAGE_TOKENS * self.kv_width

    @property
    def slab_bytes(self) -> int:
        return self.slab_units * self.unit_elements * BF16_BYTES

    @property
    def state_bytes(self) -> int:
        """Bytes of one DeltaNet STATE block."""

        return self.state_units * self.unit_elements * BF16_BYTES

    @property
    def kv_blocks_per_slab(self) -> int:
        return self.slab_units // self.full_layers

    @property
    def state_blocks_per_slab(self) -> int:
        return self.slab_units // self.state_units

    @property
    def convolution_elements(self) -> int:
        return math.prod(self.convolution_shape)

    def layout(self, slab_capacity: int) -> CacheLayout:
        return CacheLayout(
            parent_capacity=slab_capacity,
            groups=[
                CacheGroupLayout(PAGE_TOKENS, blocks_per_slab=self.kv_blocks_per_slab),
                CacheGroupLayout(
                    PREFILL_CHUNK_SIZE,
                    blocks_per_slab=self.state_blocks_per_slab,
                    family=CacheGroupFamily.STATE,
                ),
            ],
        )


class CacheArena:
    """Engine-addressed mutable storage, ordered by an MLX dependency epoch.

    Private read views are consumed by model graphs. Retained, immutable values
    must be obtained through ``snapshot`` instead of exposing the allocation.
    """

    def __init__(self, layout: CacheLayout, geometry: CacheGeometry) -> None:
        self.layout = layout
        self.geometry = geometry
        self.binding = CachePoolBinding(layout, resource="Qwen3.5 MLX cache")
        self.slabs = layout.parent_capacity + 1
        # Import only when constructing an Apple GPU runner. The package can
        # still be imported on platforms which do not build the Metal extension.
        from resin_models_mlx.native import CacheStorage

        self._storage = CacheStorage(
            [self.slabs * geometry.slab_units, 2 * PAGE_TOKENS, geometry.kv_width],
            mx.bfloat16,
        )
        self._epoch = self._storage.initial_epoch()
        self._readers: list[mx.array] = []
        # Finish one-time allocation/zeroing before a native serving actor can
        # use the runner from a different thread.
        mx.eval(self._epoch)

    @property
    def nbytes(self) -> int:
        return self._storage.nbytes

    @property
    def epoch(self) -> mx.array:
        """Completion dependency for all cache writes issued so far."""
        return self._epoch

    @property
    def buffer_identity(self) -> int:
        return self._storage.buffer_identity

    @property
    def allocation_count(self) -> int:
        return self._storage.allocation_count

    def order_after(self, reads: Sequence[mx.array]) -> None:
        """Queue later mutation after every consumer of the preceding view."""
        if reads:
            self._readers.extend(reads)

    def snapshot(self) -> mx.array:
        """Return an independent, immutable copy, ordered before later writes."""
        return self._storage.snapshot(self._epoch)

    def _read(self) -> mx.array:
        return self._storage.read_epoch(self._epoch)

    def _units(self) -> mx.array:
        return self._read().reshape(-1, 2, PAGE_TOKENS, self.geometry.kv_width)

    def _unit(self, slabs: np.ndarray, units: np.ndarray) -> np.ndarray:
        return slabs.astype(np.int64) * self.geometry.slab_units + units

    # Reads ---------------------------------------------------------------

    def _kv_bases(
        self, plan: PagedCachePlan, rows: np.ndarray, lengths: np.ndarray
    ) -> np.ndarray:
        """Layer-zero unit per block covering ``lengths[i]`` tokens; 0 pads."""

        group = plan.groups[KV_GROUP]
        width = max(1, math.ceil(int(lengths.max(initial=0)) / PAGE_TOKENS))
        bases = np.zeros((len(rows), width), dtype=np.int64)
        for index, row in enumerate(rows.tolist()):
            blocks = math.ceil(int(lengths[index]) / PAGE_TOKENS)
            slabs = group.block_slabs(row)[:blocks]
            slots = group.block_slots(row)[:blocks]
            bases[index, :blocks] = self._unit(slabs, slots.astype(np.int64))
        return bases

    def history(
        self, plan: PagedCachePlan, row: int, length: int
    ) -> Sequence[AttentionHistory | None]:
        """One row's earlier tokens, gathered per layer only when it is read."""

        layers = self.geometry.full_layers
        if length == 0:
            return (None,) * layers
        bases = self._kv_bases(plan, np.array([row]), np.array([length]))[0]
        return _GatheredHistory(self._units(), bases, length, self.geometry)

    def paged_history(
        self, plan: PagedCachePlan, rows: np.ndarray, lengths: np.ndarray
    ) -> PagedHistory:
        """Page tables for attending to history in place during decode."""

        bases = self._kv_bases(plan, rows, lengths)
        if bases.size < _MIN_DEVICE_ELEMENTS:
            # Pad rows, preserving the logical page-table width and therefore
            # the attention partition count. Only queries select live rows.
            capacity = max(len(rows), math.ceil(_MIN_DEVICE_ELEMENTS / bases.shape[1]))
            pages = np.zeros((capacity, bases.shape[1]), dtype=np.int32)
            pages[: len(rows)] = bases
        else:
            pages = bases.astype(np.int32)
        history_lengths = np.zeros(max(_MIN_DEVICE_ELEMENTS, len(rows)), dtype=np.int32)
        history_lengths[: len(rows)] = lengths
        return PagedHistory(
            self._units(),
            mx.array(pages),
            mx.array(history_lengths),
            layer_stride=self.geometry.kv_blocks_per_slab,
        )

    def state_start(self, slab: int, slot: int) -> int:
        """First unit of the STATE block at ``(slab, slot)``."""

        return slab * self.geometry.slab_units + slot * self.geometry.state_units

    @property
    def _recurrent_units(self) -> int:
        """Units holding one linear layer's recurrent state."""

        geometry = self.geometry
        return (
            math.prod(geometry.recurrent_shape[1:])
            * FP32_BYTES
            // (BF16_BYTES * geometry.unit_elements)
        )

    def read_states(self, starts: np.ndarray) -> LinearState:
        """Copy the STATE blocks starting at units ``starts``, one row each."""

        geometry = self.geometry
        count = len(starts)
        layers = geometry.recurrent_shape[0]
        per_layer = self._recurrent_units
        recurrent_shape = (count, *geometry.recurrent_shape[1:])
        storage = self._read()
        recurrent = []
        for layer in range(layers):
            units = starts[:, None] + layer * per_layer + np.arange(per_layer)
            recurrent.append(
                mx.take(storage, mx.array(units.reshape(-1).astype(np.int32)), 0)
                .view(mx.float32)
                .reshape(recurrent_shape)
            )
        # Every layer's convolution tail follows the recurrent states.
        tail_units = geometry.state_units - layers * per_layer
        units = starts[:, None] + layers * per_layer + np.arange(tail_units)
        tails = mx.take(storage, mx.array(units.reshape(-1).astype(np.int32)), 0)
        tails = tails.reshape(count, -1)[:, : geometry.convolution_elements]
        tails = tails.reshape(count, layers, *geometry.convolution_shape[1:])
        convolution = [tails[:, layer] for layer in range(layers)]
        return LinearState(tuple(recurrent), tuple(convolution))

    # Writes --------------------------------------------------------------

    def write_tokens(
        self,
        plan: PagedCachePlan,
        keys: tuple[mx.array, ...],
        values: tuple[mx.array, ...],
        tokens: np.ndarray,
    ) -> None:
        """Scatter K/V rows of every layer for flattened plan ``tokens``.

        Targets are the plan's validated append addresses. Each layer writes
        its slot in the layer's contiguous within-slab page range.
        """

        group = plan.groups[KV_GROUP]
        slab = group.append_slab_ids[tokens]
        slot = group.append_slotnums[tokens].astype(np.int64)
        offset = group.append_block_offsets[tokens].astype(np.int64)
        layers = self.geometry.full_layers
        count = len(tokens)
        units = self._unit(
            slab[:, None],
            slot[:, None] + np.arange(layers) * self.geometry.kv_blocks_per_slab,
        )
        # Each (token, layer) writes its key row and value row in one unit.
        rows = np.arange(2)[None, None, :] * PAGE_TOKENS + offset[:, None, None]
        update = pack_key_values(keys, values, count, self.geometry.kv_width)
        indices = units[:, :, None] * (2 * PAGE_TOKENS) + rows
        self._epoch = self._storage.scatter_rows(
            self._epoch,
            mx.array(indices.reshape(-1).astype(np.int32)),
            update.reshape(-1, self.geometry.kv_width).astype(mx.bfloat16),
            after=self._readers,
        )
        self._readers.clear()

    def write_states(self, starts: Sequence[int], state: LinearState) -> None:
        """Store row ``i`` of ``state`` in the STATE block starting at ``starts[i]``.

        Writes are contiguous slice updates per row and layer: on an M4 Max at
        batch 16 this moves 1 GB in 12 ms, against 13 ms when rows are first
        concatenated into one update each and 57 ms as one indexed scatter per
        layer.
        """

        geometry = self.geometry
        count = len(starts)
        unit_shape = (2 * PAGE_TOKENS, geometry.kv_width)
        per_layer = self._recurrent_units
        for layer, value in enumerate(state.recurrent):
            units = value.astype(mx.float32).view(mx.bfloat16)
            units = units.reshape(count, per_layer, *unit_shape)
            first = layer * per_layer
            for index, start in enumerate(starts):
                self._epoch = self._storage.write_slice(
                    self._epoch,
                    (start + first) * geometry.unit_elements,
                    units[index],
                    after=self._readers,
                )
                self._readers.clear()
        layers = geometry.recurrent_shape[0]
        tail_units = geometry.state_units - layers * per_layer
        tails = [
            value.astype(mx.bfloat16).reshape(count, -1) for value in state.convolution
        ]
        padding = tail_units * geometry.unit_elements - geometry.convolution_elements
        tails.append(mx.zeros((count, padding), dtype=mx.bfloat16))
        tails = mx.concatenate(tails, axis=1).reshape(count, tail_units, *unit_shape)
        first = layers * per_layer
        for index, start in enumerate(starts):
            self._epoch = self._storage.write_slice(
                self._epoch,
                (start + first) * geometry.unit_elements,
                tails[index],
                after=self._readers,
            )
            self._readers.clear()


@dataclass(frozen=True, slots=True)
class _GatheredHistory(Sequence[AttentionHistory | None]):
    """Gather a layer's pages using their within-slab layer stride.

    Nothing holds a layer's gathered keys and values beyond the attention that
    consumes them, so MLX can free each layer's copy before the next.
    """

    units: mx.array
    bases: np.ndarray
    length: int
    geometry: CacheGeometry

    def __len__(self) -> int:
        return self.geometry.full_layers

    def __getitem__(self, layer):  # type: ignore[override]
        if not isinstance(layer, int) or not 0 <= layer < len(self):
            raise IndexError(layer)
        geometry = self.geometry
        pages = mx.take(
            self.units,
            mx.array(self.bases + layer * geometry.kv_blocks_per_slab),
            axis=0,
        )
        shape = (1, len(self.bases) * PAGE_TOKENS, geometry.kv_heads, -1)
        keys = pages[:, 0].reshape(shape)[:, : self.length]
        values = pages[:, 1].reshape(shape)[:, : self.length]
        return AttentionHistory(keys, values)


__all__ = [
    "PAGE_TOKENS",
    "PREFILL_CHUNK_SIZE",
    "CacheArena",
    "CacheGeometry",
]
