"""Fungible full-attention and DeltaNet-state slabs in one MLX arena.

The byte arena has a BF16 KV view shaped ``[units, 2 * page, kv_width]``.
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
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.cache_arena import ByteArena, KVPage
from resin_models_mlx.qwen3_5.model import (
    AttentionHistory,
    LinearState,
    PagedHistory,
)

PAGE_TOKENS = 128
STATE_BLOCK_TOKENS = 4096
KV_GROUP = 0
STATE_GROUP = 1

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
        unit_bytes = KVPage(PAGE_TOKENS, kv_width, mx.bfloat16).nbytes
        recurrent_layer_bytes = math.prod(recurrent_shape[1:]) * mx.float32.size
        if (
            recurrent_layer_bytes % unit_bytes
            or math.prod(convolution_shape[1:]) % kv_width
        ):
            raise ValueError("each layer's DeltaNet state must align to cache units")
        state_bytes = (
            math.prod(recurrent_shape) * mx.float32.size
            + math.prod(convolution_shape) * mx.bfloat16.size
        )
        state_units = math.ceil(state_bytes / unit_bytes)
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
    def kv_page(self) -> KVPage:
        return KVPage(PAGE_TOKENS, self.kv_width, mx.bfloat16)

    @property
    def slab_bytes(self) -> int:
        return self.slab_units * self.kv_page.nbytes

    @property
    def state_bytes(self) -> int:
        """Bytes of one DeltaNet STATE block."""

        return self.state_units * self.kv_page.nbytes

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
                    STATE_BLOCK_TOKENS,
                    blocks_per_slab=self.state_blocks_per_slab,
                    family=CacheGroupFamily.STATE,
                ),
            ],
        )


class CacheArena(ByteArena):
    """Engine-addressed mutable storage, ordered by an MLX dependency epoch.

    Private read views are consumed by model graphs. Retained, immutable values
    must be obtained through ``snapshot`` instead of exposing the allocation.
    """

    def __init__(self, layout: CacheLayout, geometry: CacheGeometry) -> None:
        self.geometry = geometry
        self.page = geometry.kv_page
        super().__init__(
            layout,
            slab_shape=(geometry.slab_units, self.page.nbytes),
            resource="Qwen3.5 MLX cache",
        )

    def snapshot(self) -> mx.array:
        """Return an independent, immutable copy, ordered before later writes."""
        # Copy raw words so NaNs retain their payloads. The copy is a reader
        # of the arena, just like attention, until the next ordered write.
        words = self.view(mx.uint16, (-1, 2 * PAGE_TOKENS, self.page.width))
        result = mx.full(words.shape, words, dtype=mx.uint16).view(self.page.dtype)
        self.order_after([result])
        return result

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
            bases[index, :blocks] = (
                slabs.astype(np.int64) * self.geometry.slab_units + slots
            )
        return bases

    def history(
        self, plan: PagedCachePlan, row: int, length: int
    ) -> Sequence[AttentionHistory | None]:
        """One row's earlier tokens, gathered per layer only when it is read."""

        layers = self.geometry.full_layers
        if length == 0:
            return (None,) * layers
        bases = self._kv_bases(plan, np.array([row]), np.array([length]))[0]
        return _GatheredHistory(self.page.view(self), bases, length, self.geometry)

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
            self.page.view(self),
            mx.array(pages),
            mx.array(history_lengths),
            layer_stride=self.geometry.kv_blocks_per_slab,
        )

    def state_start(self, slab: int, slot: int) -> int:
        """Byte offset of the STATE block at ``(slab, slot)``."""
        return slab * self.geometry.slab_bytes + slot * self.geometry.state_bytes

    def read_states(self, starts: np.ndarray) -> LinearState:
        """Restore each layer's FP32 recurrent state and BF16 convolution tail."""
        geometry = self.geometry
        count = len(starts)
        layers = geometry.recurrent_shape[0]
        layer_bytes = math.prod(geometry.recurrent_shape[1:]) * mx.float32.size
        unit_bytes = self.page.nbytes
        recurrent = []
        for layer in range(layers):
            values = self.gather_units(
                starts + layer * layer_bytes,
                layer_bytes // unit_bytes,
                unit_bytes,
                mx.float32,
            )
            recurrent.append(values.reshape(count, *geometry.recurrent_shape[1:]))
        tail_start = layers * layer_bytes
        tails = self.gather_units(
            starts + tail_start,
            (geometry.state_bytes - tail_start) // unit_bytes,
            unit_bytes,
            mx.bfloat16,
        )
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

        self.page.write(
            self,
            plan.groups[KV_GROUP],
            keys,
            values,
            tokens,
            slab_units=self.geometry.slab_units,
            layer_stride=self.geometry.kv_blocks_per_slab,
        )

    def write_states(self, starts: Sequence[int], state: LinearState) -> None:
        """Store row ``i`` of ``state`` in the STATE block at byte offset ``starts[i]``.

        Writes are contiguous slice updates per row and layer: on an M4 Max at
        batch 16 this moves 1 GB in 12 ms, against 13 ms when rows are first
        concatenated into one update each and 57 ms as one indexed scatter per
        layer.
        """

        geometry = self.geometry
        count = len(starts)
        layer_bytes = math.prod(geometry.recurrent_shape[1:]) * mx.float32.size
        for layer, value in enumerate(state.recurrent):
            values = value.astype(mx.float32)
            for index, start in enumerate(starts):
                self.write_slice(start + layer * layer_bytes, values[index])
        tail_start = geometry.recurrent_shape[0] * layer_bytes
        tails = [
            value.astype(mx.bfloat16).reshape(count, -1) for value in state.convolution
        ]
        tail_elements = (geometry.state_bytes - tail_start) // mx.bfloat16.size
        padding = tail_elements - geometry.convolution_elements
        tails.append(mx.zeros((count, padding), dtype=mx.bfloat16))
        packed_tails = mx.concatenate(tails, axis=1)
        for index, start in enumerate(starts):
            self.write_slice(start + tail_start, packed_tails[index])


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
    "STATE_BLOCK_TOKENS",
    "CacheArena",
    "CacheGeometry",
]
