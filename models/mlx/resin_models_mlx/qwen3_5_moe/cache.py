"""Fungible full-attention and DeltaNet-state slabs in one MLX arena.

The arena is one BF16 array of *units* shaped ``[units, 2 * page, kv_width]``.
A unit holds the keys, then the values, of one 128-token page of one
full-attention layer; a KV block of ``L`` layers occupies ``L`` consecutive
units of a slab. A DeltaNet checkpoint is packed into whole units: each linear
layer's FP32 recurrent state fills whole units, followed by the BF16
convolution tails of every layer.

Reads gather with ``take``. Keys and values scatter single rows; checkpoints
are written as contiguous slice updates, which MLX performs far faster than
large-payload scatters. Every write happens after the step's reads are
evaluated, so MLX updates the arena in place instead of copying it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout
from resin.backend.plans import CacheGroupView, PagedCachePlan
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


def _mask(values: np.ndarray) -> mx.array:
    # MLX accepts NumPy arrays through the buffer protocol at runtime.
    return mx.array(cast(Any, values))


def _locate(
    group: CacheGroupView, rows: np.ndarray, positions: np.ndarray, block_tokens: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map row-relative absolute positions to ``(slab, slot, offset)``."""

    columns = positions // block_tokens - group.block_table_start_indices[rows]
    addresses = group.block_indptr[rows] + columns
    return (
        group.block_slab_ids[addresses].astype(np.int32),
        group.block_slotnums[addresses].astype(np.int32),
        (positions % block_tokens).astype(np.int32),
    )


class CacheArena:
    """Engine-addressed MLX storage for one Qwen3.5-MoE runner."""

    def __init__(self, layout: CacheLayout, geometry: CacheGeometry) -> None:
        self.layout = layout
        self.geometry = geometry
        self.binding = CachePoolBinding(layout, resource="Qwen3.5-MoE MLX cache")
        self.slabs = layout.parent_capacity + 1
        self.storage = mx.zeros(
            (self.slabs * geometry.slab_units, 2 * PAGE_TOKENS, geometry.kv_width),
            dtype=mx.bfloat16,
        )
        mx.eval(self.storage)

    def _units(self) -> mx.array:
        return self.storage.reshape(-1, 2, PAGE_TOKENS, self.geometry.kv_width)

    def _unit(self, slabs: np.ndarray, units: np.ndarray) -> np.ndarray:
        return slabs.astype(np.int64) * self.geometry.slab_units + units

    # Reads ---------------------------------------------------------------

    def history(
        self, plan: PagedCachePlan, rows: np.ndarray, lengths: np.ndarray
    ) -> tuple[AttentionHistory | None, ...]:
        """Gather each full-attention layer's first ``lengths[i]`` tokens."""

        layers = self.geometry.full_layers
        longest = int(lengths.max()) if len(lengths) else 0
        if longest == 0:
            return (None,) * layers
        count = len(rows)
        width = math.ceil(longest / PAGE_TOKENS)
        group = plan.groups[KV_GROUP]
        slabs = np.zeros((count, width), dtype=np.int64)
        slots = np.zeros((count, width), dtype=np.int64)
        for index, row in enumerate(rows):
            blocks = math.ceil(int(lengths[index]) / PAGE_TOKENS)
            selected = group.block_slice(int(row))
            slabs[index, :blocks] = group.block_slab_ids[selected][:blocks]
            slots[index, :blocks] = group.block_slotnums[selected][:blocks]
        valid = _mask(np.arange(width * PAGE_TOKENS)[None, :] < lengths[:, None])
        base = self._unit(slabs, slots * layers).reshape(-1)
        units = self._units()
        shape = (count, width, 2, PAGE_TOKENS, self.geometry.kv_heads, -1)
        result = []
        for layer in range(layers):
            pages = mx.take(units, mx.array(base + layer), axis=0).reshape(shape)
            history = (count, width * PAGE_TOKENS, self.geometry.kv_heads, -1)
            result.append(
                AttentionHistory(
                    pages[:, :, 0].reshape(history),
                    pages[:, :, 1].reshape(history),
                    valid,
                )
            )
        return tuple(result)

    def paged_history(
        self, plan: PagedCachePlan, rows: np.ndarray, lengths: np.ndarray
    ) -> PagedHistory:
        """Page tables for attending to history in place during decode."""

        width = max(1, math.ceil(int(lengths.max()) / PAGE_TOKENS))
        group = plan.groups[KV_GROUP]
        slabs = np.zeros((len(rows), width), dtype=np.int64)
        slots = np.zeros((len(rows), width), dtype=np.int64)
        for index, row in enumerate(rows):
            blocks = math.ceil(int(lengths[index]) / PAGE_TOKENS)
            selected = group.block_slice(int(row))
            slabs[index, :blocks] = group.block_slab_ids[selected][:blocks]
            slots[index, :blocks] = group.block_slotnums[selected][:blocks]
        pages = self._unit(slabs, slots * self.geometry.full_layers)
        return PagedHistory(
            self._units(),
            mx.array(pages.astype(np.int32)),
            mx.array(lengths.astype(np.int32)),
        )

    def _state_units(
        self, plan: PagedCachePlan, rows: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """First unit of each row's checkpoint containing ``positions``."""

        slab, slot, _ = _locate(
            plan.groups[STATE_GROUP], rows, positions, PREFILL_CHUNK_SIZE
        )
        return self._unit(slab, slot * self.geometry.state_units)

    @property
    def _recurrent_units(self) -> int:
        """Units holding one linear layer's recurrent state."""

        geometry = self.geometry
        return (
            geometry.recurrent_elements
            // geometry.linear_layers
            // geometry.unit_elements
        )

    def state(
        self, plan: PagedCachePlan, rows: np.ndarray, contexts: np.ndarray
    ) -> LinearState:
        """Read each row's checkpoint after ``contexts[i] - 1``; zero if new."""

        geometry = self.geometry
        count = len(rows)
        live = contexts > 0
        starts = np.zeros(count, dtype=np.int64)
        if live.any():
            indices = np.flatnonzero(live)
            starts[indices] = self._state_units(
                plan, rows[indices], contexts[indices].astype(np.int64) - 1
            )
        layers = geometry.linear_layers
        per_layer = self._recurrent_units
        recurrent_shape = (count, *geometry.recurrent_shape[1:])
        recurrent = []
        for layer in range(layers):
            units = starts[:, None] + layer * per_layer + np.arange(per_layer)
            recurrent.append(
                mx.take(self.storage, mx.array(units.reshape(-1).astype(np.int32)), 0)
                .view(mx.float32)
                .reshape(recurrent_shape)
            )
        # Every layer's convolution tail follows the recurrent states.
        tail_units = geometry.state_units - layers * per_layer
        units = starts[:, None] + layers * per_layer + np.arange(tail_units)
        tails = mx.take(self.storage, mx.array(units.reshape(-1).astype(np.int32)), 0)
        tails = tails.reshape(count, -1)[:, : geometry.convolution_elements]
        tails = tails.reshape(count, layers, *geometry.convolution_shape[1:])
        convolution = [tails[:, layer] for layer in range(layers)]
        if not live.all():
            keep = _mask(live)
            recurrent = [mx.where(keep[:, None, None, None], r, 0) for r in recurrent]
            convolution = [mx.where(keep[:, None, None], c, 0) for c in convolution]
        return LinearState(tuple(recurrent), tuple(convolution))

    # Writes --------------------------------------------------------------

    def write_tokens(
        self,
        plan: PagedCachePlan,
        keys: tuple[mx.array, ...],
        values: tuple[mx.array, ...],
        token_rows: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """Scatter ``[tokens, kv_heads, head_dim]`` K/V rows for every layer."""

        slab, slot, offset = _locate(
            plan.groups[KV_GROUP], token_rows, positions, PAGE_TOKENS
        )
        layers = self.geometry.full_layers
        tokens = len(positions)
        units = self._unit(slab[:, None], slot[:, None] * layers + np.arange(layers))
        # Each (token, layer) writes its key row and value row in one unit.
        rows = np.arange(2)[None, None, :] * PAGE_TOKENS + offset[:, None, None]
        update = mx.stack(
            [
                mx.stack([key.reshape(tokens, -1), value.reshape(tokens, -1)], axis=1)
                for key, value in zip(keys, values, strict=True)
            ],
            axis=1,
        )
        self.storage[
            mx.array(
                np.broadcast_to(units[:, :, None], rows.shape[:1] + (layers, 2))
                .reshape(-1)
                .astype(np.int32)
            ),
            mx.array(
                np.broadcast_to(rows, (tokens, layers, 2)).reshape(-1).astype(np.int32)
            ),
        ] = update.reshape(-1, self.geometry.kv_width)

    def write_state(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        positions: np.ndarray,
        state: LinearState,
    ) -> None:
        """Store each row's checkpoint after processing ``positions[i]``."""

        geometry = self.geometry
        count = len(rows)
        unit_shape = (2 * PAGE_TOKENS, geometry.kv_width)
        starts = self._state_units(plan, rows, positions.astype(np.int64)).tolist()
        per_layer = self._recurrent_units
        for layer, value in enumerate(state.recurrent):
            units = value.astype(mx.float32).view(mx.bfloat16)
            units = units.reshape(count, per_layer, *unit_shape)
            first = layer * per_layer
            for index, start in enumerate(starts):
                self.storage[start + first : start + first + per_layer] = units[index]
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
            self.storage[start + first : start + first + tail_units] = tails[index]


__all__ = [
    "PAGE_TOKENS",
    "PREFILL_CHUNK_SIZE",
    "CacheArena",
    "CacheGeometry",
]
