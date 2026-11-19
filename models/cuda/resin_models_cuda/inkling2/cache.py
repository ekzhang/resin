"""Shared slab geometry for Inkling 2 target, DSpark, and convolution history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout, CacheRetention
from resin.backend.cache_storage import byte_regions, live_rows
from resin.backend.cuda.slab import (
    DevicePageGroup,
    GraphPageTableArena,
    SlabDeviceCache,
    graph_scratch_address,
    graph_scratch_parent_count,
    stage_page_groups,
    state_view_factory,
)
from resin.backend.resources import CachePoolBinding

PAGE_TOKENS = 128
PREFILL_CHUNK_SIZE = 8192
GLOBAL_GROUP, LOCAL_GROUP, DRAFT_GLOBAL_GROUP, DRAFT_LOCAL_GROUP, STATE_GROUP = range(5)


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    global_layers: int
    local_layers: int
    draft_global_layers: int
    draft_local_layers: int
    kv_heads: int
    head_dim: int
    hidden_size: int
    state_length: int
    window_tokens: int
    draft_window_tokens: int
    prefill_chunk_size: int = PREFILL_CHUNK_SIZE
    draft_width: int = 8
    draft_enabled: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            *self.layer_counts,
            self.kv_heads,
            self.head_dim,
            self.hidden_size,
            self.state_length,
            self.window_tokens,
            self.draft_window_tokens,
            self.prefill_chunk_size,
            self.draft_width,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Inkling 2 cache dimensions must be positive integers")

    @classmethod
    def from_config(
        cls,
        target,
        draft,
        *,
        attn_tp: int = 1,
        prefill_chunk_size=PREFILL_CHUNK_SIZE,
        draft_enabled: bool = True,
    ):
        if attn_tp <= 0 or target.num_key_value_heads % attn_tp:
            raise ValueError("Inkling 2 attention TP must divide its KV heads")
        if (target.num_key_value_heads, target.head_dim) != (
            draft.num_key_value_heads,
            draft.head_dim,
        ):
            raise ValueError("Inkling 2 target and draft KV geometry must match")
        return cls(
            global_layers=target.num_hidden_layers - len(target.local_layer_ids),
            local_layers=len(target.local_layer_ids),
            draft_global_layers=draft.num_hidden_layers - len(draft.local_layer_ids),
            draft_local_layers=len(draft.local_layer_ids),
            kv_heads=target.num_key_value_heads // attn_tp,
            head_dim=target.head_dim,
            hidden_size=target.hidden_size,
            state_length=target.sconv_kernel_size - 1,
            window_tokens=target.sliding_window_size,
            draft_window_tokens=draft.sliding_window_size,
            prefill_chunk_size=prefill_chunk_size,
            draft_width=draft.block_size,
            draft_enabled=draft_enabled,
        )

    @property
    def state_group(self) -> int:
        return STATE_GROUP if self.draft_enabled else 2

    @property
    def layer_counts(self) -> tuple[int, int, int, int]:
        return (
            self.global_layers,
            self.local_layers,
            self.draft_global_layers,
            self.draft_local_layers,
        )

    @property
    def target_layers(self) -> int:
        return self.global_layers + self.local_layers

    @property
    def state_width(self) -> int:
        return 2 * self.kv_heads * self.head_dim + 2 * self.hidden_size

    @property
    def state_elements(self) -> int:
        return self.convolution_elements + (
            self.hidden_size if self.draft_enabled else 0
        )

    @property
    def convolution_elements(self) -> int:
        return self.target_layers * self.state_length * self.state_width

    @property
    def unit_elements(self) -> int:
        return 2 * PAGE_TOKENS * self.kv_heads * self.head_dim

    @property
    def slab_units(self) -> int:
        # Two pages of the largest layer group, and room for one complete state.
        return max(
            2 * max(self.layer_counts),
            (self.state_elements + self.unit_elements - 1) // self.unit_elements,
        )

    @property
    def slab_elements(self) -> int:
        return self.slab_units * self.unit_elements

    @property
    def bytes_per_slab(self) -> int:
        return self.slab_elements * 2

    @property
    def state_blocks_per_slab(self) -> int:
        return self.slab_elements // self.state_elements

    @property
    def packings(self) -> tuple[int, ...]:
        return (
            *(self.slab_units // n for n in self.layer_counts[: self.state_group]),
            self.state_blocks_per_slab,
        )

    def layout(self, slab_capacity: int) -> CacheLayout:
        groups = []
        for index, layers in enumerate(self.layer_counts[: self.state_group]):
            window = (
                self.window_tokens
                if index == LOCAL_GROUP
                else self.draft_window_tokens
                if index == DRAFT_LOCAL_GROUP
                else None
            )
            retention: dict[str, Any] = {}
            if window is not None:
                retention = dict(
                    retention=CacheRetention.WINDOW,
                    window_tokens=window
                    + max(self.prefill_chunk_size, self.draft_width)
                    - 1,
                    read_window_tokens=window,
                )
            groups.append(
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.slab_units // layers, **retention
                )
            )
        groups.append(
            CacheGroupLayout(
                PAGE_TOKENS,
                blocks_per_slab=self.state_blocks_per_slab,
                family=CacheGroupFamily.STATE,
            )
        )
        return CacheLayout(parent_capacity=slab_capacity, groups=groups)


@dataclass(frozen=True, slots=True)
class DeviceCache(SlabDeviceCache):
    geometry: CacheGeometry
    groups: tuple[DevicePageGroup, ...]
    keys: Tensor
    values: Tensor
    states: Tensor
    draft_features: Tensor | None

    def store_kv(
        self,
        keys: Tensor,
        values: Tensor,
        positions: Tensor,
        *,
        group_index: int,
        layer: int,
        layer_count: int,
    ) -> None:
        """Write both KV fields directly from logical positions into their pages."""
        if not 0 <= layer < layer_count:
            raise ValueError("cache layer lies outside its cache group")
        if self.keys.is_cuda:
            from resin_models_cuda.inkling2.triton import store_kv

            store_kv(self, keys, values, positions, group_index, layer, layer_count)
        else:
            pages, offsets = self.page_locations(
                group_index, layer, layer_count, positions
            )
            self.keys[pages, offsets] = keys.to(self.keys.dtype)
            self.values[pages, offsets] = values.to(self.values.dtype)

    def _state_indices(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        slabs, slots, _ = self.locations(
            self.geometry.state_group,
            positions,
            block_tokens=PAGE_TOKENS,
        )
        return slabs.long(), slots.long()

    def state_at(self, positions: Tensor) -> Tensor:
        return self.states[self._state_indices(positions)]

    def store_state(self, positions: Tensor, state: Tensor) -> None:
        self.states[self._state_indices(positions)] = state

    def draft_features_at(self, positions: Tensor) -> Tensor:
        assert self.draft_features is not None
        return self.draft_features[self._state_indices(positions)]

    def store_draft_features(self, positions: Tensor, features: Tensor) -> None:
        assert self.draft_features is not None
        self.draft_features[self._state_indices(positions)] = features


class CacheArena(GraphPageTableArena):
    graph_label = "Inkling 2"

    def __init__(
        self,
        device: Any,
        layout: CacheLayout,
        geometry: CacheGeometry,
        *,
        max_batch_size: int = 1,
    ):
        self.device, self.layout, self.geometry = device, layout, geometry
        self.max_batch_size = max_batch_size
        self.binding = CachePoolBinding(layout, resource="Inkling 2 cache")
        physical = (
            layout.parent_capacity
            + 1
            + graph_scratch_parent_count(geometry.packings, max_batch_size)
        )
        self.storage = torch.zeros(
            (physical, geometry.slab_elements), dtype=torch.bfloat16, device=device
        )
        page_elements = PAGE_TOKENS * geometry.kv_heads * geometry.head_dim
        page_shape = (
            physical * geometry.slab_units,
            PAGE_TOKENS,
            geometry.kv_heads,
            geometry.head_dim,
        )
        page_stride = (
            geometry.unit_elements,
            geometry.kv_heads * geometry.head_dim,
            geometry.head_dim,
            1,
        )
        self.keys = self.storage.as_strided(page_shape, page_stride)
        self.values = self.storage.as_strided(page_shape, page_stride, page_elements)
        state_view = state_view_factory(
            self.storage, geometry.state_blocks_per_slab, geometry.state_elements * 2
        )
        self.states = state_view(
            torch.bfloat16,
            geometry.target_layers,
            geometry.state_length,
            geometry.state_width,
        )
        self.draft_features = (
            state_view(
                torch.bfloat16,
                geometry.hidden_size,
                offset=geometry.convolution_elements * 2,
            )
            if geometry.draft_enabled
            else None
        )
        self.state_blocks = {
            geometry.state_group: state_view(torch.bfloat16, geometry.state_elements)
        }

    def cache_storage_buffers(self, groups, *, prefix_length):
        buffers = []
        for group, (first, addresses) in enumerate(groups):
            for block, (slab, slot) in enumerate(addresses, first):
                if group == self.geometry.state_group:
                    tensors = {"convolution": self.states[slab, slot]}
                    if self.draft_features is not None:
                        tensors["draft_features"] = self.draft_features[slab, slot]
                else:
                    layers = self.geometry.layer_counts[group]
                    rows = live_rows(
                        prefix_length,
                        block,
                        PAGE_TOKENS,
                        self.layout.groups[group].read_window_tokens,
                    )
                    base = slab * self.geometry.slab_units + slot * layers
                    tensors = {
                        f"{name}/{layer}": values[base + layer, rows]
                        for name, values in (
                            ("keys", self.keys),
                            ("values", self.values),
                        )
                        for layer in range(layers)
                    }
                buffers.append(byte_regions(tensors))
        return tuple(buffers)

    def reset_null(self) -> None:
        self.storage[0].zero_()
        self.storage[self.layout.parent_capacity + 1 :].zero_()

    def _graph_padding_address(self, index: int, row: int) -> tuple[int, int]:
        return graph_scratch_address(
            self.layout.parent_capacity,
            self.geometry.packings,
            self.max_batch_size,
            index,
            row,
        )

    def tensorize(self, plan, staging) -> DeviceCache:
        return self._make_graph_cache(stage_page_groups(plan, staging))

    def _make_graph_cache(self, groups) -> DeviceCache:
        return DeviceCache(
            self.geometry,
            groups,
            self.keys,
            self.values,
            self.states,
            self.draft_features,
        )
