"""Inkling 2 global KV, sliding-window KV and convolution history in one MLX arena.

The byte arena has a BF16 KV view shaped ``[units, 2 * page, kv_width]``.
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
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout, CacheRetention
from resin.models.inkling2.config import Inkling2Config
from resin_models_mlx.cache_arena import PAGE_TOKENS, ByteArena, SlabGeometry

GLOBAL_GROUP, LOCAL_GROUP, STATE_GROUP = 0, 1, 2


@dataclass(frozen=True, slots=True)
class CacheGeometry(SlabGeometry):
    global_layers: int
    local_layers: int
    kv_heads: int
    head_dim: int
    state_shape: tuple[int, int, int]
    window_tokens: int
    prefill_chunk_size: int
    slab_units: int = field(init=False)
    state_units: int = field(init=False)

    def __post_init__(self) -> None:
        state_units = math.ceil(
            math.prod(self.state_shape) * mx.bfloat16.size / self.unit_bytes
        )
        slab_units = math.lcm(self.global_layers, self.local_layers, state_units)
        object.__setattr__(self, "state_units", state_units)
        object.__setattr__(self, "slab_units", slab_units)

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


class CacheArena(ByteArena[CacheGeometry]):
    resource = "Inkling 2 MLX cache"

    def read_states(self, starts: np.ndarray) -> mx.array:
        return self.read_state_part(starts, self.geometry.state_shape, mx.bfloat16)

    def write_states(self, starts: Sequence[int], state: mx.array) -> None:
        self.write_state_part(starts, state.astype(mx.bfloat16))


__all__ = [
    "GLOBAL_GROUP",
    "LOCAL_GROUP",
    "PAGE_TOKENS",
    "STATE_GROUP",
    "CacheArena",
    "CacheGeometry",
]
