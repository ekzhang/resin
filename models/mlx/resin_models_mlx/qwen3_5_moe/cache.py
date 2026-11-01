"""Fungible full-attention and DeltaNet-state slabs in one MLX arena.

The arena is BF16 *units* shaped ``[units, 2 * page, kv_width]``.
A unit holds the keys, then the values, of one 128-token page of one
full-attention layer; a KV block of ``L`` layers occupies ``L`` consecutive
units of a slab. A DeltaNet checkpoint is packed into whole units: each linear
layer's FP32 recurrent state fills whole units, followed by the BF16
convolution tails of every layer.

The units are split across parts, separate arrays of whole slabs. The first
use of a Metal buffer left unused for a couple of seconds costs about 11 ms per
GiB on an M4 Max; a step after an idle pause pays that only for the parts it
touches, not the whole arena.

Reads gather with ``take``. Keys and values scatter single rows; DeltaNet
snapshots are written as contiguous slice updates, which MLX performs far
faster than large-payload scatters. Every write happens after the step's reads
are evaluated, so MLX updates each part in place instead of copying it. Live
DeltaNet state stays outside the arena between steps (see ``state.py``).
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
from resin_models_mlx.qwen3_5_moe.model import (
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

# Target part size in whole slabs; MAX_PARTS parts share a larger arena.
PART_BYTES = 1 << 30
# Decode attention binds the parts it reads, within Metal's 31 kernel buffers.
MAX_PARTS = 16


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """Slab geometry shared by the KV and DeltaNet-state groups."""

    full_layers: int
    linear_layers: int
    kv_heads: int
    head_dim: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_dim: int
    conv_state_length: int
    slab_units: int
    state_units: int

    @classmethod
    def from_config(cls, config: Qwen35Config) -> CacheGeometry:
        unit_elements = 2 * PAGE_TOKENS * config.num_key_value_heads * config.head_dim
        recurrent = (
            config.linear_layer_count
            * config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim
        )
        convolution = (
            config.linear_layer_count
            * config.linear_conv_dim
            * (config.linear_conv_kernel_dim - 1)
        )
        kv_width = config.num_key_value_heads * config.head_dim
        layers = config.linear_layer_count
        if (recurrent * FP32_BYTES // BF16_BYTES // layers) % unit_elements or (
            convolution // layers
        ) % kv_width:
            raise ValueError("each layer's DeltaNet state must align to cache units")
        state_elements = recurrent * FP32_BYTES // BF16_BYTES + convolution
        state_units = math.ceil(state_elements / unit_elements)
        layers = config.full_layer_count
        return cls(
            full_layers=layers,
            linear_layers=config.linear_layer_count,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_dim=config.linear_conv_dim,
            conv_state_length=config.linear_conv_kernel_dim - 1,
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
    def recurrent_shape(self) -> tuple[int, ...]:
        return (
            self.linear_layers,
            self.value_heads,
            self.value_head_dim,
            self.key_head_dim,
        )

    @property
    def convolution_shape(self) -> tuple[int, ...]:
        return (self.linear_layers, self.conv_state_length, self.conv_dim)

    @property
    def recurrent_elements(self) -> int:
        """Recurrent FP32 values, counted in BF16 storage elements."""

        return math.prod(self.recurrent_shape) * FP32_BYTES // BF16_BYTES

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
    """Engine-addressed MLX storage for one Qwen3.5-MoE runner."""

    def __init__(self, layout: CacheLayout, geometry: CacheGeometry) -> None:
        self.layout = layout
        self.geometry = geometry
        self.binding = CachePoolBinding(layout, resource="Qwen3.5-MoE MLX cache")
        self.slabs = layout.parent_capacity + 1
        part_slabs = max(
            1,
            PART_BYTES // geometry.slab_bytes,
            math.ceil(self.slabs / MAX_PARTS),
        )
        self.part_units = part_slabs * geometry.slab_units
        self.parts = [
            mx.zeros(
                (
                    min(part_slabs, self.slabs - first) * geometry.slab_units,
                    2 * PAGE_TOKENS,
                    geometry.kv_width,
                ),
                dtype=mx.bfloat16,
            )
            for first in range(0, self.slabs, part_slabs)
        ]
        mx.eval(*self.parts)

    def _units(self, part: int) -> mx.array:
        return self.parts[part].reshape(-1, 2, PAGE_TOKENS, self.geometry.kv_width)

    def _unit(self, slabs: np.ndarray, units: np.ndarray) -> np.ndarray:
        return slabs.astype(np.int64) * self.geometry.slab_units + units

    def _take(self, units: np.ndarray) -> mx.array:
        """Gather arena ``units`` in order."""

        parts, units = np.divmod(units, self.part_units)
        # Each run within one part gathers at once.
        cuts = np.flatnonzero(np.diff(parts)) + 1
        gathered = [
            mx.take(self.parts[int(part[0])], mx.array(run.astype(np.int32)), 0)
            for part, run in zip(np.split(parts, cuts), np.split(units, cuts))
        ]
        return gathered[0] if len(gathered) == 1 else mx.concatenate(gathered)

    # Reads ---------------------------------------------------------------

    def _kv_bases(
        self, plan: PagedCachePlan, rows: np.ndarray, lengths: np.ndarray
    ) -> np.ndarray:
        """First unit of each block covering ``lengths[i]`` tokens; 0 pads."""

        group = plan.groups[KV_GROUP]
        layers = self.geometry.full_layers
        width = max(1, math.ceil(int(lengths.max(initial=0)) / PAGE_TOKENS))
        bases = np.zeros((len(rows), width), dtype=np.int64)
        for index, row in enumerate(rows.tolist()):
            blocks = math.ceil(int(lengths[index]) / PAGE_TOKENS)
            slabs = group.block_slabs(row)[:blocks]
            slots = group.block_slots(row)[:blocks]
            bases[index, :blocks] = self._unit(slabs, slots.astype(np.int64) * layers)
        return bases

    def history(
        self, plan: PagedCachePlan, row: int, length: int
    ) -> Sequence[AttentionHistory | None]:
        """One row's earlier tokens, gathered per layer only when it is read."""

        layers = self.geometry.full_layers
        if length == 0:
            return (None,) * layers
        bases = self._kv_bases(plan, np.array([row]), np.array([length]))[0]
        return _GatheredHistory(self, bases, length, layers, self.geometry)

    def paged_history(
        self, plan: PagedCachePlan, rows: np.ndarray, lengths: np.ndarray
    ) -> PagedHistory:
        """Page tables for attending to history in place during decode.

        Only the parts holding the rows' pages are bound, renumbered in order
        and repeated to a fixed count, so the kernel doesn't depend on how many
        parts a batch reads.
        """

        bases = self._kv_bases(plan, rows, lengths)
        parts, units = np.divmod(bases, self.part_units)
        blocks = np.arange(bases.shape[1])
        read = blocks[None, :] < np.ceil(lengths / PAGE_TOKENS)[:, None]
        used = np.unique(parts[read]) if read.any() else np.zeros(1, np.int64)
        # Padding reads nothing; its base 0 resolves to the first bound part.
        slots = np.searchsorted(used, parts)
        bound = [self._units(int(part)) for part in used]
        return PagedHistory(
            tuple(bound + bound[:1] * (len(self.parts) - len(bound))),
            self.part_units,
            mx.array((slots * self.part_units + units).astype(np.int32)),
            mx.array(lengths.astype(np.int32)),
        )

    def state_start(self, slab: int, slot: int) -> int:
        """First unit of the STATE block at ``(slab, slot)``."""

        return slab * self.geometry.slab_units + slot * self.geometry.state_units

    @property
    def _recurrent_units(self) -> int:
        """Units holding one linear layer's recurrent state."""

        geometry = self.geometry
        return (
            geometry.recurrent_elements
            // geometry.linear_layers
            // geometry.unit_elements
        )

    def read_states(self, starts: np.ndarray) -> LinearState:
        """Copy the STATE blocks starting at units ``starts``, one row each."""

        geometry = self.geometry
        count = len(starts)
        layers = geometry.linear_layers
        per_layer = self._recurrent_units
        recurrent_shape = (count, *geometry.recurrent_shape[1:])
        recurrent = []
        for layer in range(layers):
            units = starts[:, None] + layer * per_layer + np.arange(per_layer)
            recurrent.append(
                self._take(units.reshape(-1)).view(mx.float32).reshape(recurrent_shape)
            )
        # Every layer's convolution tail follows the recurrent states.
        tail_units = geometry.state_units - layers * per_layer
        units = starts[:, None] + layers * per_layer + np.arange(tail_units)
        tails = self._take(units.reshape(-1))
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

        Targets are the plan's validated append addresses.
        """

        group = plan.groups[KV_GROUP]
        slab = group.append_slab_ids[tokens]
        slot = group.append_slotnums[tokens].astype(np.int64)
        offset = group.append_block_offsets[tokens].astype(np.int64)
        layers = self.geometry.full_layers
        count = len(tokens)
        units = self._unit(slab[:, None], slot[:, None] * layers + np.arange(layers))
        # Each (token, layer) writes its key row and value row in one unit.
        rows = np.arange(2)[None, None, :] * PAGE_TOKENS + offset[:, None, None]
        update = mx.stack(
            [
                mx.stack([key.reshape(count, -1), value.reshape(count, -1)], axis=1)
                for key, value in zip(keys, values, strict=True)
            ],
            axis=1,
        )
        shape = (count, layers, 2)
        parts, units = np.divmod(
            np.broadcast_to(units[:, :, None], shape), self.part_units
        )
        parts, units = parts.reshape(-1), units.reshape(-1)
        rows = np.broadcast_to(rows, shape).reshape(-1)
        update = update.reshape(-1, self.geometry.kv_width)
        for part in np.unique(parts).tolist():
            selected = np.flatnonzero(parts == part)
            self.parts[part][
                mx.array(units[selected].astype(np.int32)),
                mx.array(rows[selected].astype(np.int32)),
            ] = (
                update
                if len(selected) == len(parts)
                else update[mx.array(selected.astype(np.int32))]
            )

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
                part, start = divmod(start, self.part_units)
                self.parts[part][start + first : start + first + per_layer] = units[
                    index
                ]
        layers = geometry.linear_layers
        tail_units = geometry.state_units - layers * per_layer
        tails = mx.concatenate(
            [
                value.astype(mx.bfloat16).reshape(count, -1)
                for value in state.convolution
            ]
            # The checkpoint's final unit has room to pad the tails to whole units.
            + [
                mx.zeros(
                    (
                        count,
                        tail_units * geometry.unit_elements
                        - geometry.convolution_elements,
                    ),
                    dtype=mx.bfloat16,
                )
            ],
            axis=1,
        ).reshape(count, tail_units, *unit_shape)
        first = layers * per_layer
        for index, start in enumerate(starts):
            part, start = divmod(start, self.part_units)
            self.parts[part][start + first : start + first + tail_units] = tails[index]


class _GatheredHistory(Sequence[AttentionHistory | None]):
    """Per-layer history views that gather a layer's pages only on access.

    Nothing holds a layer's gathered keys and values beyond the attention that
    consumes them, so MLX can free each layer's copy before the next.
    """

    def __init__(
        self,
        arena: CacheArena,
        bases: np.ndarray,
        length: int,
        layers: int,
        geometry: CacheGeometry,
    ) -> None:
        self._arena = arena
        self._bases = bases
        self._length = length
        self._layers = layers
        self._shape = (1, len(bases) * PAGE_TOKENS, geometry.kv_heads, -1)
        self._unit_shape = (-1, 2, PAGE_TOKENS, geometry.kv_width)

    def __len__(self) -> int:
        return self._layers

    def __getitem__(self, layer):  # type: ignore[override]
        if not isinstance(layer, int) or not 0 <= layer < self._layers:
            raise IndexError(layer)
        pages = self._arena._take(self._bases + layer).reshape(self._unit_shape)
        keys = pages[:, 0].reshape(self._shape)[:, : self._length]
        values = pages[:, 1].reshape(self._shape)[:, : self._length]
        return AttentionHistory(keys, values)


__all__ = [
    "PAGE_TOKENS",
    "PREFILL_CHUNK_SIZE",
    "CacheArena",
    "CacheGeometry",
]
