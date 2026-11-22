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
Live DeltaNet state stays outside the arena between steps (see ``live_state.py``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout
from resin.backend.plans import PagedCachePlan
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.cache_arena import PAGE_TOKENS, ByteArena, SlabGeometry
from resin_models_mlx.metal import device_array
from resin_models_mlx.qwen3_5.model import (
    AttentionHistory,
    LinearState,
    PagedHistory,
)

STATE_BLOCK_TOKENS = 4096
KV_GROUP = 0
STATE_GROUP = 1


@dataclass(frozen=True, slots=True)
class CacheGeometry(SlabGeometry):
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
        unit_bytes = 2 * PAGE_TOKENS * kv_width * cls.kv_dtype.size
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
    def kv_blocks_per_slab(self) -> int:
        return self.slab_units // self.full_layers

    @property
    def state_blocks_per_slab(self) -> int:
        return self.slab_units // self.state_units

    @property
    def recurrent_layer_bytes(self) -> int:
        return math.prod(self.recurrent_shape[1:]) * mx.float32.size

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


class CacheArena(ByteArena[CacheGeometry]):
    """Engine-addressed mutable storage, ordered by an MLX dependency epoch.

    Private read views are consumed by model graphs. Retained, immutable values
    must be obtained through ``snapshot`` instead of exposing the allocation.
    """

    resource = "Qwen3.5 MLX cache"

    def snapshot(self) -> mx.array:
        """Return an independent, immutable copy, ordered before later writes."""
        # Copy raw words so NaNs retain their payloads. The copy is a reader
        # of the arena, just like attention, until the next ordered write.
        words = self.view(mx.uint16, (-1, 2 * PAGE_TOKENS, self.geometry.kv_width))
        result = mx.full(words.shape, words, dtype=mx.uint16).view(
            self.geometry.kv_dtype
        )
        self.order_after([result])
        return result

    def history(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        lengths: np.ndarray,
        *,
        decode: bool,
    ) -> Iterable[AttentionHistory | None] | PagedHistory:
        """Decode reads in place. Prefill gathers one row, one layer at a time."""
        layers = self.geometry.full_layers
        if not decode and lengths[0] == 0:
            return (None,) * layers
        history = self.retained_pages(plan, KV_GROUP, rows, lengths)
        if decode:
            return PagedHistory(
                history.units,
                device_array(history.bases),
                device_array(lengths),
                history.layer_stride,
            )
        return (AttentionHistory(*history.gather(layer)) for layer in range(layers))

    def read_states(self, starts: np.ndarray) -> LinearState:
        """Restore each layer's FP32 recurrent state and BF16 convolution tail."""
        geometry = self.geometry
        layers = geometry.recurrent_shape[0]
        recurrent = tuple(
            self.read_state_part(
                starts + layer * geometry.recurrent_layer_bytes,
                geometry.recurrent_shape[1:],
                mx.float32,
            )
            for layer in range(layers)
        )
        tails = self.read_state_part(
            starts + layers * geometry.recurrent_layer_bytes,
            geometry.convolution_shape,
            mx.bfloat16,
        )
        return LinearState(recurrent, tuple(tails[:, layer] for layer in range(layers)))

    def write_states(self, starts: Sequence[int], state: LinearState) -> None:
        """Store row ``i`` of ``state`` in the STATE block at byte offset ``starts[i]``.

        Writes are contiguous slice updates per row and layer: on an M4 Max at
        batch 16 this moves 1 GB in 12 ms, against 13 ms when rows are first
        concatenated into one update each and 57 ms as one indexed scatter per
        layer.
        """

        geometry = self.geometry
        for layer, value in enumerate(state.recurrent):
            self.write_state_part(
                starts,
                value.astype(mx.float32),
                offset=layer * geometry.recurrent_layer_bytes,
            )
        tails = mx.stack(
            [value.astype(mx.bfloat16) for value in state.convolution], axis=1
        )
        self.write_state_part(
            starts, tails, offset=len(state.recurrent) * geometry.recurrent_layer_bytes
        )


__all__ = [
    "PAGE_TOKENS",
    "STATE_BLOCK_TOKENS",
    "CacheArena",
    "CacheGeometry",
]
