"""Engine-owned MLA/index pages and KDA checkpoints for GLM-5.3-Flash."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout
from resin.backend.cuda.slab import (
    GraphPageTableArena,
    SlabDeviceCache,
    graph_scratch_address,
    graph_scratch_parent_count,
    stage_page_groups,
)
from resin.backend.resources import CachePoolBinding

PAGE_TOKENS = 64
STATE_GROUP = 1


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    layer_types: tuple[str, ...]
    hidden_size: int
    latent_dim: int
    index_dim: int
    heads: int
    head_dim: int
    conv_history: int
    state_tokens: int
    include_mtp: bool = False
    latent_dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_config(
        cls,
        config,
        *,
        state_tokens: int,
        include_mtp: bool = False,
        latent_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        if state_tokens <= 0 or state_tokens % PAGE_TOKENS:
            raise ValueError("GLM state checkpoint width must be a multiple of 64")
        return cls(
            tuple(config.layer_types),
            config.hidden_size,
            config.kv_lora_rank,
            config.index_head_dim,
            config.linear_num_heads,
            config.linear_head_dim,
            config.linear_conv_kernel_dim - 1,
            state_tokens,
            include_mtp,
            latent_dtype,
        )

    @property
    def full_layers(self):
        layers = tuple(
            i
            for i, kind in enumerate(self.layer_types)
            if kind == "deepseek_sparse_attention"
        )
        return layers + ((len(self.layer_types),) if self.include_mtp else ())

    @property
    def linear_layers(self):
        return tuple(
            i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"
        )

    @property
    def latent_bytes(self):
        return PAGE_TOKENS * self.latent_dim * self.latent_dtype.itemsize

    @property
    def unit_bytes(self):
        raw = self.latent_bytes + PAGE_TOKENS // 4 * self.index_dim * 2
        # Both cache consumers address an envelope of latent-width rows.
        alignment = math.lcm(self.latent_dim * self.latent_dtype.itemsize, 8)
        return math.ceil(raw / alignment) * alignment

    @property
    def tail_elements(self):
        return len(self.full_layers) * 4 * self.index_dim

    @property
    def base_state_bytes(self):
        return (
            self.recurrent_elements * 4
            + (self.convolution_elements + self.hidden_size) * 2
        )

    @property
    def recurrent_elements(self):
        return len(self.linear_layers) * self.heads * self.head_dim**2

    @property
    def convolution_elements(self):
        return (
            len(self.linear_layers) * 3 * self.heads * self.head_dim * self.conv_history
        )

    @property
    def state_bytes(self):
        # Four raw rows per sparse layer plus their last logical position.
        # The two Engine STATE checkpoints own these tails, including prefixes.
        raw = self.base_state_bytes + self.tail_elements * 6 + len(self.full_layers) * 8
        return math.ceil(raw / 8) * 8

    @property
    def slab_bytes(self):
        block = len(self.full_layers) * self.unit_bytes
        return math.ceil(self.state_bytes / block) * block

    @property
    def slab_units(self):
        return self.slab_bytes // self.unit_bytes

    @property
    def packings(self):
        return self.slab_units // len(
            self.full_layers
        ), self.slab_bytes // self.state_bytes

    def layout(self, parent_capacity):
        return CacheLayout(
            parent_capacity=parent_capacity,
            decode_write_extra_tokens=int(self.include_mtp),
            groups=[
                CacheGroupLayout(PAGE_TOKENS, blocks_per_slab=self.packings[0]),
                CacheGroupLayout(
                    self.state_tokens,
                    blocks_per_slab=self.packings[1],
                    family=CacheGroupFamily.STATE,
                ),
            ],
        )


@dataclass
class LayerCache:
    positions: Tensor
    sequence_indices: Tensor
    cu_seqlens: Tensor
    context_lengths: Tensor
    lengths: Tensor
    cu_seqlens_cpu: Tensor | None = None
    page_table: Tensor | None = None
    latent: Tensor | None = None
    index_keys: Tensor | None = None
    index_gates: Tensor | None = None
    pooled_keys: Tensor | None = None
    recurrent: Tensor | None = None
    convolution: Tensor | None = None
    page_tokens: int = PAGE_TOKENS
    # MTP (h_i, x_{i+1}) keeps logical position i, but writes token row i+1
    # and completed pool g at pool slot g+1. Its first continuation write then
    # belongs to the Engine's append page, never to an immutable prefix page.
    storage_origin: int = 0
    index_tail: Any = None
    index_update: Any = None
    dsa_workspace: Any = None
    # Set only for the CUDA gather's owned working copy. Ordinary decode may
    # overwrite it; the Engine slab allocation remains an explicit alias guard.
    initial_state_owned: bool = False
    initial_state_arena: Tensor | None = None


class BatchCache:
    """Per-forward views; the Engine remains the only page allocator."""

    def __init__(
        self,
        device_cache,
        positions,
        sequence_indices,
        cu_seqlens,
        contexts,
        cu_seqlens_cpu=None,
    ):
        self.device_cache = device_cache
        # Explicit execution metadata; a coincident row count cannot identify
        # prefill, ordinary decode, or a sequence-major verification block.
        self.is_decode = False
        self.projection_rows = 0
        self.positions, self.sequence_indices = positions, sequence_indices
        self.cu_seqlens, self.context_lengths = cu_seqlens, contexts
        self.cu_seqlens_cpu = cu_seqlens_cpu
        self.lengths = contexts + cu_seqlens[1:] - cu_seqlens[:-1]
        self.layers = {}
        if contexts.is_cuda:
            from resin_models_cuda.glm5_3_flash.cache_cuda import (
                gather_state_layer_major,
            )

            recurrent, convolution, hidden = gather_state_layer_major(
                device_cache, contexts
            )
        else:
            recurrent, convolution, hidden = device_cache.state_at(
                (contexts - 1).clamp_min(0)
            )
            new = contexts == 0
            recurrent.masked_fill_(new[:, None, None, None, None], 0)
            convolution.masked_fill_(new[:, None, None, None, None], 0)
            hidden.masked_fill_(new[:, None], 0)
            recurrent = recurrent.transpose(0, 1)
            convolution = convolution.transpose(0, 1)
        self.previous_hidden = hidden
        geometry = device_cache.geometry
        for index, layer in enumerate(geometry.linear_layers):
            self.layers[layer] = LayerCache(
                positions,
                sequence_indices,
                cu_seqlens,
                contexts,
                self.lengths,
                cu_seqlens_cpu=cu_seqlens_cpu,
                recurrent=recurrent[index],
                convolution=convolution[index],
                initial_state_owned=contexts.is_cuda,
                initial_state_arena=device_cache.recurrent_states
                if contexts.is_cuda
                else None,
            )
        tails = device_cache.gather_index_tails(contexts)
        for index, layer in enumerate(geometry.full_layers):
            self.layers[layer] = LayerCache(
                positions,
                sequence_indices,
                cu_seqlens,
                contexts,
                self.lengths,
                cu_seqlens_cpu=cu_seqlens_cpu,
                page_table=device_cache.pages(0, index, len(geometry.full_layers)),
                latent=device_cache.latent,
                pooled_keys=device_cache.pooled_keys,
                storage_origin=int(layer == len(geometry.layer_types)),
                index_tail=tails[index],
                dsa_workspace=device_cache.dsa_workspace,
            )

    def layer(self, index: int) -> LayerCache:
        return self.layers[index]

    def store(self, hidden: Tensor) -> None:
        geometry = self.device_cache.geometry
        self.device_cache.store_index_tails(
            self.lengths - 1, [self.layers[i].index_tail for i in geometry.full_layers]
        )
        states, convolutions = [], []
        for index in geometry.linear_layers:
            layer = self.layers[index]
            assert layer.recurrent is not None and layer.convolution is not None
            states.append(layer.recurrent)
            convolutions.append(layer.convolution)
        if self.context_lengths.is_cuda:
            from resin_models_cuda.glm5_3_flash.cache_cuda import scatter_state_layers

            scatter_state_layers(
                self.device_cache, self.lengths - 1, states, convolutions, hidden
            )
            return
        recurrent = torch.stack(states, dim=1)
        convolution = torch.stack(convolutions, dim=1)
        self.device_cache.store_state(self.lengths - 1, recurrent, convolution, hidden)


@dataclass(frozen=True)
class DeviceCache(SlabDeviceCache):
    geometry: CacheGeometry
    latent: Tensor
    pooled_keys: Tensor
    recurrent_states: Tensor
    convolution_states: Tensor
    target_hidden_states: Tensor
    index_tail_keys: Tensor
    index_tail_gates: Tensor
    index_tail_ends: Tensor
    dsa_workspace: Any = None
    page_tokens = PAGE_TOKENS

    def gather_index_tails(self, contexts):
        from resin_models_cuda.glm5_3_flash.index_tail import IndexTail

        if contexts.is_cuda:
            from resin_models_cuda.glm5_3_flash.cache_cuda import gather_index_tails

            return gather_index_tails(self, contexts)
        slabs, slots, _ = self.locations(
            STATE_GROUP,
            (contexts - 1).clamp_min(0)[:, None],
            block_tokens=self.geometry.state_tokens,
        )
        indices = slabs[:, 0].long(), slots[:, 0].long()
        keys = self.index_tail_keys[indices]
        gates = self.index_tail_gates[indices]
        ends = self.index_tail_ends[indices]
        new = (contexts == 0) | (slabs[:, 0] == 0)
        keys[new] = 0
        gates[new] = 0
        ends[new] = -1
        return [
            IndexTail(
                keys[:, i].contiguous(),
                gates[:, i].contiguous(),
                ends[:, i].contiguous(),
            )
            for i in range(len(self.geometry.full_layers))
        ]

    def store_index_tails(self, positions, tails):
        if positions.is_cuda:
            from resin_models_cuda.glm5_3_flash.cache_cuda import scatter_index_tails

            scatter_index_tails(self, positions, tails)
            return
        slabs, slots, _ = self.locations(
            STATE_GROUP,
            positions[:, None],
            block_tokens=self.geometry.state_tokens,
        )
        indices = slabs[:, 0].long(), slots[:, 0].long()
        valid = slabs[:, 0] > 0
        indices = indices[0][valid], indices[1][valid]
        self.index_tail_keys[indices] = torch.stack([t.keys for t in tails], dim=1)[
            valid
        ]
        self.index_tail_gates[indices] = torch.stack([t.gates for t in tails], dim=1)[
            valid
        ]
        self.index_tail_ends[indices] = torch.stack([t.end for t in tails], dim=1)[
            valid
        ]

    def state_at(self, positions):
        slabs, slots, _ = self.locations(
            STATE_GROUP, positions[:, None], block_tokens=self.geometry.state_tokens
        )
        indices = slabs[:, 0].long(), slots[:, 0].long()
        return (
            self.recurrent_states[indices],
            self.convolution_states[indices],
            self.target_hidden_states[indices],
        )

    def store_state(self, positions, recurrent, convolution, hidden):
        slabs, slots, _ = self.locations(
            STATE_GROUP, positions[:, None], block_tokens=self.geometry.state_tokens
        )
        indices = slabs[:, 0].long(), slots[:, 0].long()
        self.recurrent_states[indices] = recurrent
        self.convolution_states[indices] = convolution
        self.target_hidden_states[indices] = hidden

    def batch(
        self, positions, sequence_indices, cu_seqlens, contexts, cu_seqlens_cpu=None
    ):
        return BatchCache(
            self, positions, sequence_indices, cu_seqlens, contexts, cu_seqlens_cpu
        )


class CacheArena(GraphPageTableArena):
    graph_label = "GLM-5.3-Flash"

    @property
    def cache_storage_block_bytes(self):
        return (
            len(self.geometry.full_layers) * self.geometry.unit_bytes,
            self.geometry.state_bytes,
        )

    def __init__(
        self,
        device: Any,
        layout: CacheLayout,
        geometry: CacheGeometry,
        *,
        max_batch_size: int,
    ):
        self.device, self.layout, self.geometry = device, layout, geometry
        self.max_batch_size = max_batch_size
        self.binding = CachePoolBinding(layout, resource=self.graph_label)
        reserved = 1 + graph_scratch_parent_count(geometry.packings, max_batch_size)
        physical = layout.parent_capacity + reserved
        self.storage = torch.zeros(
            (physical, geometry.slab_bytes), dtype=torch.uint8, device=device
        )
        bf16, fp32 = self.storage.view(torch.bfloat16), self.storage.view(torch.float32)
        pages = physical * geometry.slab_units
        latent_storage = self.storage.view(geometry.latent_dtype)
        self.latent = latent_storage.as_strided(
            (pages, PAGE_TOKENS, geometry.latent_dim),
            (
                geometry.unit_bytes // geometry.latent_dtype.itemsize,
                geometry.latent_dim,
                1,
            ),
        )
        self.pooled_keys = bf16.as_strided(
            (pages, PAGE_TOKENS // 4, geometry.index_dim),
            (geometry.unit_bytes // 2, geometry.index_dim, 1),
            geometry.latent_bytes // 2,
        )
        n, h, d, c = (
            len(geometry.linear_layers),
            geometry.heads,
            geometry.head_dim,
            geometry.conv_history,
        )
        state_slots = geometry.packings[1]
        self.state_blocks = {
            STATE_GROUP: self.storage.as_strided(
                (physical, state_slots, geometry.state_bytes),
                (geometry.slab_bytes, geometry.state_bytes, 1),
            )
        }
        self.recurrent_states = fp32.as_strided(
            (physical, state_slots, n, h, d, d),
            (
                geometry.slab_bytes // 4,
                geometry.state_bytes // 4,
                h * d * d,
                d * d,
                d,
                1,
            ),
        )
        self.convolution_states = bf16.as_strided(
            (physical, state_slots, n, 3, h * d, c),
            (
                geometry.slab_bytes // 2,
                geometry.state_bytes // 2,
                3 * h * d * c,
                h * d * c,
                c,
                1,
            ),
            geometry.recurrent_elements * 2,
        )
        self.target_hidden_states = bf16.as_strided(
            (physical, state_slots, geometry.hidden_size),
            (geometry.slab_bytes // 2, geometry.state_bytes // 2, 1),
            geometry.recurrent_elements * 2 + geometry.convolution_elements,
        )

        f, d = len(geometry.full_layers), geometry.index_dim
        self.index_tail_keys = bf16.as_strided(
            (physical, state_slots, f, 4, d),
            (geometry.slab_bytes // 2, geometry.state_bytes // 2, 4 * d, d, 1),
            geometry.base_state_bytes // 2,
        )
        self.index_tail_gates = fp32.as_strided(
            (physical, state_slots, f, 4, d),
            (geometry.slab_bytes // 4, geometry.state_bytes // 4, 4 * d, d, 1),
            (geometry.base_state_bytes + geometry.tail_elements * 2) // 4,
        )
        self.index_tail_ends = self.storage.view(torch.int64).as_strided(
            (physical, state_slots, f),
            (geometry.slab_bytes // 8, geometry.state_bytes // 8, 1),
            (geometry.base_state_bytes + geometry.tail_elements * 6) // 8,
        )

        self.dsa_workspace = None
        if self.storage.is_cuda and geometry.latent_dtype == torch.float8_e4m3fn:
            from resin_models_cuda.glm5_3_flash.dsa import DsaWorkspace

            self.dsa_workspace = DsaWorkspace(device)

    def _graph_padding_address(self, index, row):
        return graph_scratch_address(
            self.layout.parent_capacity,
            self.geometry.packings,
            self.max_batch_size,
            index,
            row,
        )

    def _graph_widths(self, sequence_capacity: int) -> tuple[int, ...]:
        widths = super()._graph_widths(sequence_capacity)
        # MTP repair's final physical row is one past its logical position.
        # The cache layout declares this shifted physical write to the Engine.
        return (
            math.ceil(
                (sequence_capacity + int(self.geometry.include_mtp)) / PAGE_TOKENS
            ),
            *widths[1:],
        )

    def reset_null(self):
        self.storage[0].zero_()
        self.storage[self.layout.parent_capacity + 1 :].zero_()

    def tensorize(self, plan, staging):
        return self._make_graph_cache(stage_page_groups(plan, staging))

    def _make_graph_cache(self, groups):
        return DeviceCache(
            self.geometry,
            groups,
            self.latent,
            self.pooled_keys,
            self.recurrent_states,
            self.convolution_states,
            self.target_hidden_states,
            self.index_tail_keys,
            self.index_tail_gates,
            self.index_tail_ends,
            self.dsa_workspace,
        )
