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
from resin.backend.plans import PagedCachePlan
from resin.models.inkling2.config import Inkling2Config
from resin_models_mlx.cache_arena import ByteArena, KVPage, SlabGeometry

PAGE_TOKENS = 128
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
    kv_page: KVPage = field(init=False)
    slab_units: int = field(init=False)
    state_units: int = field(init=False)
    page_tokens = PAGE_TOKENS

    def __post_init__(self) -> None:
        page = KVPage(PAGE_TOKENS, self.kv_heads * self.head_dim, mx.bfloat16)
        state_units = math.ceil(self.state_elements * mx.bfloat16.size / page.nbytes)
        slab_units = math.lcm(self.global_layers, self.local_layers, state_units)
        object.__setattr__(self, "state_units", state_units)
        object.__setattr__(self, "slab_units", slab_units)
        object.__setattr__(self, "kv_page", page)

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
    def state_elements(self) -> int:
        return math.prod(self.state_shape)

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


class CacheArena(ByteArena[CacheGeometry]):
    """Engine-addressed mutable storage, ordered by an MLX dependency epoch."""

    resource = "Inkling 2 MLX cache"

    def units(self) -> mx.array:
        return self.page.view(self)

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
        geometry = self.geometry
        bases, starts = self.page.retained_table(
            plan.groups[group], rows, contexts, geometry.slab_units, earliest=earliest
        )
        if bases.shape[1] == 0:
            return None
        return History(
            self.units(),
            bases,
            starts,
            contexts.astype(np.int64),
            geometry.blocks_per_slab(group),
            geometry.kv_heads,
        )

    def read_states(self, starts: np.ndarray) -> mx.array:
        geometry = self.geometry
        values = self.gather_units(
            starts, geometry.state_units, self.page.nbytes, mx.bfloat16
        )
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
        self.page.write(
            self,
            plan.groups[group],
            keys,
            values,
            tokens,
            slab_units=self.geometry.slab_units,
            layer_stride=self.geometry.blocks_per_slab(group),
        )

    def write_states(self, starts: Sequence[int], state: mx.array) -> None:
        geometry = self.geometry
        count = len(starts)
        padding = geometry.state_bytes // mx.bfloat16.size - geometry.state_elements
        flat = state.astype(mx.bfloat16).reshape(count, -1)
        if padding:
            flat = mx.concatenate(
                [flat, mx.zeros((count, padding), dtype=mx.bfloat16)], axis=1
            )
        for index, start in enumerate(starts):
            self.write_slice(start, flat[index])


__all__ = [
    "GLOBAL_GROUP",
    "LOCAL_GROUP",
    "PAGE_TOKENS",
    "STATE_GROUP",
    "CacheArena",
    "CacheGeometry",
    "History",
]
