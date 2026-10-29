"""Fungible Inkling target, draft, and convolution-state cache geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from resin import (
    CacheGroupFamily,
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
)
from resin.backend.cuda.slab import (
    DevicePageGroup as DeviceGroup,
)
from resin.backend.cuda.slab import (
    GraphPageTableArena,
    SlabDeviceCache,
    stage_page_groups,
)
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding

PAGE_TOKENS = 128
PREFILL_CHUNK_SIZE = 4096
GLOBAL_GROUP = 0
LOCAL_GROUP = 1
DRAFT_GROUP = 2
STATE_GROUP = 3
GROUP_COUNT = 4

GLOBAL_LAYERS = 7
LOCAL_LAYERS = 35
DRAFT_LAYERS = 6
KV_HEADS = 8
HEAD_DIM = 128
SLAB_UNITS = 70
GLOBAL_BLOCKS_PER_SLAB = 10
LOCAL_BLOCKS_PER_SLAB = 2
DRAFT_BLOCKS_PER_SLAB = 11
STATE_BLOCKS_PER_SLAB = 14
STATE_WIDTH_PER_LAYER = 1024 + 1024 + 4096 + 4096
STATE_LENGTH = 3
STATE_ELEMENTS = 42 * STATE_WIDTH_PER_LAYER * STATE_LENGTH
UNIT_ELEMENTS = 2 * PAGE_TOKENS * KV_HEADS * HEAD_DIM
SLAB_ELEMENTS = SLAB_UNITS * UNIT_ELEMENTS
LOCAL_RETENTION_TOKENS = PREFILL_CHUNK_SIZE + 512 - 1


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """A 35 MiB slab split into small single-layer KV address units."""

    slab_units: int = SLAB_UNITS
    global_blocks_per_slab: int = GLOBAL_BLOCKS_PER_SLAB
    local_blocks_per_slab: int = LOCAL_BLOCKS_PER_SLAB
    draft_blocks_per_slab: int = DRAFT_BLOCKS_PER_SLAB
    state_blocks_per_slab: int = STATE_BLOCKS_PER_SLAB

    def __post_init__(self) -> None:
        usages = {
            "global": self.global_blocks_per_slab * GLOBAL_LAYERS * UNIT_ELEMENTS,
            "local": self.local_blocks_per_slab * LOCAL_LAYERS * UNIT_ELEMENTS,
            "draft": self.draft_blocks_per_slab * DRAFT_LAYERS * UNIT_ELEMENTS,
            "state": self.state_blocks_per_slab * STATE_ELEMENTS,
        }
        if any(elements > self.slab_elements for elements in usages.values()):
            raise ValueError(f"Inkling cache block packing exceeds a slab: {usages}")

    @property
    def slab_elements(self) -> int:
        return self.slab_units * UNIT_ELEMENTS

    @property
    def bytes_per_slab(self) -> int:
        return self.slab_elements * 2

    def layout(self, slab_capacity: int) -> CacheLayout:
        return CacheLayout(
            parent_capacity=slab_capacity,
            groups=(
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.global_blocks_per_slab
                ),
                CacheGroupLayout(
                    PAGE_TOKENS,
                    blocks_per_slab=self.local_blocks_per_slab,
                    retention=CacheRetention.WINDOW,
                    window_tokens=LOCAL_RETENTION_TOKENS,
                    read_window_tokens=512,
                ),
                CacheGroupLayout(
                    PAGE_TOKENS, blocks_per_slab=self.draft_blocks_per_slab
                ),
                CacheGroupLayout(
                    PREFILL_CHUNK_SIZE,
                    blocks_per_slab=self.state_blocks_per_slab,
                    family=CacheGroupFamily.STATE,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class DeviceCache(SlabDeviceCache):
    """Graph-stable cache tables over one slab arena."""

    geometry: CacheGeometry
    groups: tuple[DeviceGroup, DeviceGroup, DeviceGroup, DeviceGroup]
    keys: Tensor
    values: Tensor
    states: Tensor

    def state_at(self, positions: Tensor) -> Tensor:
        slabs, slots, _ = self.locations(
            STATE_GROUP, positions, block_tokens=PREFILL_CHUNK_SIZE
        )
        return self.states[slabs.long(), slots.long()]

    def store_state(self, positions: Tensor, state: Tensor) -> None:
        slabs, slots, _ = self.locations(
            STATE_GROUP, positions, block_tokens=PREFILL_CHUNK_SIZE
        )
        self.states[slabs.long(), slots.long()] = state


class CacheArena(GraphPageTableArena):
    """Layer-addressable views over one Engine-owned fungible slab allocation."""

    graph_label = "Inkling"

    def __init__(self, device: Any, layout: CacheLayout) -> None:
        self.device = device
        self.layout = layout
        self.geometry = CacheGeometry()
        self.binding = CachePoolBinding(layout, resource="Inkling cache")
        physical = layout.parent_capacity + 1
        self.storage = torch.zeros(
            (physical, SLAB_ELEMENTS), dtype=torch.bfloat16, device=device
        )
        flat = self.storage.flatten()
        page_elements = PAGE_TOKENS * KV_HEADS * HEAD_DIM
        page_shape = (physical * SLAB_UNITS, PAGE_TOKENS, KV_HEADS, HEAD_DIM)
        page_stride = (UNIT_ELEMENTS, KV_HEADS * HEAD_DIM, HEAD_DIM, 1)
        self.keys = torch.as_strided(flat, page_shape, page_stride)
        self.values = torch.as_strided(
            flat, page_shape, page_stride, storage_offset=page_elements
        )
        self.states = torch.as_strided(
            flat,
            (
                physical,
                STATE_BLOCKS_PER_SLAB,
                42,
                STATE_LENGTH,
                STATE_WIDTH_PER_LAYER,
            ),
            (
                SLAB_ELEMENTS,
                STATE_ELEMENTS,
                STATE_LENGTH * STATE_WIDTH_PER_LAYER,
                STATE_WIDTH_PER_LAYER,
                1,
            ),
        )
        self.state_blocks = {STATE_GROUP: self.states.flatten(2)}

    @staticmethod
    def unit_page(slab_id: int, slotnum: int, layers: int, layer: int) -> int:
        if not 0 <= layer < layers:
            raise ValueError("cache layer offset is outside its block")
        return slab_id * SLAB_UNITS + slotnum * layers + layer

    def state(self, slab_id: int, slotnum: int) -> Tensor:
        if (
            slab_id < 0
            or slab_id > self.layout.parent_capacity
            or slotnum < 0
            or slotnum >= STATE_BLOCKS_PER_SLAB
        ):
            raise ValueError("Inkling state address lies outside the cache arena")
        return self.states[slab_id, slotnum]

    def reset_null(self) -> None:
        self.storage[0].zero_()

    def tensorize(self, plan: PagedCachePlan, staging: Any) -> DeviceCache:
        return DeviceCache(
            self.geometry,
            cast(
                tuple[DeviceGroup, DeviceGroup, DeviceGroup, DeviceGroup],
                stage_page_groups(plan, staging),
            ),
            self.keys,
            self.values,
            self.states,
        )

    def _make_graph_cache(self, groups) -> DeviceCache:
        return DeviceCache(
            self.geometry,
            groups,  # type: ignore[arg-type]
            self.keys,
            self.values,
            self.states,
        )


__all__ = [
    "CacheArena",
    "CacheGeometry",
    "DeviceCache",
    "DeviceGroup",
    "DRAFT_GROUP",
    "GLOBAL_GROUP",
    "GROUP_COUNT",
    "LOCAL_GROUP",
    "LOCAL_RETENTION_TOKENS",
    "PAGE_TOKENS",
    "PREFILL_CHUNK_SIZE",
    "SLAB_ELEMENTS",
    "SLAB_UNITS",
    "STATE_ELEMENTS",
    "STATE_GROUP",
    "STATE_LENGTH",
    "STATE_WIDTH_PER_LAYER",
    "UNIT_ELEMENTS",
]
