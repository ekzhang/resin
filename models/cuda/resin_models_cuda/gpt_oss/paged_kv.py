"""GPT-OSS per-layer K/V storage and dense FA4 metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from resin import BatchPhase
from resin.backend.cache_storage import byte_regions, live_rows
from resin.backend.cuda.context_parallel import ContextParallelTokens
from resin.backend.resources import CachePoolBinding
from resin.models.gpt_oss.plan import ATTENTION_GROUPS, dense_page_table

if TYPE_CHECKING:
    from resin import CacheLayout
    from resin.backend.cuda.graphs import GraphShape
    from resin.backend.cuda.staging import TensorStager
    from resin.models.gpt_oss.plan import AttentionBatch


class PagedKvArena:
    """Per-layer K/V storage with one block per slab (slot zero only)."""

    def __init__(
        self,
        device: str | torch.device,
        layout: CacheLayout,
        *,
        layer_count: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        resource: str = "paged KV",
    ) -> None:
        self.capacity = layout.parent_capacity
        self.layout = layout
        self.page_size = page_size
        self.binding = CachePoolBinding(layout, resource=resource)
        shape = (2, layer_count, self.capacity + 1, page_size, num_kv_heads, head_dim)
        self.storage = torch.zeros(shape, dtype=dtype, device=device)
        self.keys = self.storage[0]
        self.values = self.storage[1]
        self._cache_storage_views: dict[int, dict[str, torch.Tensor]] = {}

    def cache_storage_buffers(self, groups, *, prefix_length):
        buffers = []
        for group, (first, addresses) in zip(self.layout.groups, groups):
            for block, (slab, slot) in enumerate(addresses, first):
                if not 0 < slab <= self.capacity or slot:
                    raise ValueError(
                        "cache storage address is outside the managed arena"
                    )
                rows = live_rows(
                    prefix_length, block, self.page_size, group.read_window_tokens
                )
                # FA4 can load the complete page even when attention masks rows.
                if rows.stop <= rows.start:
                    buffers.append({})
                    continue
                regions = self._cache_storage_views.get(slab)
                if regions is None:
                    regions = byte_regions(
                        {
                            f"{('keys', 'values')[kv]}/{layer}": self.storage[
                                kv, layer, slab
                            ]
                            for kv in range(2)
                            for layer in range(self.storage.shape[1])
                        }
                    )
                    self._cache_storage_views[slab] = regions
                buffers.append(regions)
        return tuple(buffers)


@dataclass(frozen=True, slots=True)
class PagedKvTable:
    """Dense page table, logical lengths, and append slots for one KV group."""

    page_table: torch.Tensor
    lengths: torch.Tensor
    append_slots: torch.Tensor
    min_length: int
    max_length: int


@dataclass(frozen=True, slots=True)
class DeviceAttentionBatch:
    """Dense named attention tables and token metadata staged for CUDA."""

    phase: BatchPhase
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    groups: tuple[tuple[str, PagedKvTable], ...]

    def group(self, name: str) -> PagedKvTable:
        try:
            return next(
                group for group_name, group in self.groups if group_name == name
            )
        except StopIteration as error:
            raise KeyError(f"unknown device paged attention group: {name!r}") from error


def stage_paged_attention_batch(
    batch: AttentionBatch, staging: TensorStager
) -> DeviceAttentionBatch:
    """Copy one semantic paged-attention batch to CUDA tensors."""

    cu_seqlens_q = [0]
    for length in batch.query_lengths:
        cu_seqlens_q.append(cu_seqlens_q[-1] + length)
    row_count = len(batch.query_lengths)
    groups = []
    for group in batch.groups:
        groups.append(
            (
                group.name,
                PagedKvTable(
                    page_table=staging.device(
                        f"{group.name}.pages",
                        dense_page_table(group, row_count, group.width),
                        torch.int32,
                    ),
                    lengths=staging.device(
                        f"{group.name}.lengths", group.lengths, torch.int32
                    ),
                    append_slots=staging.device(
                        f"{group.name}.append", group.append_slots, torch.long
                    ),
                    min_length=min(group.lengths),
                    max_length=max(group.lengths),
                ),
            )
        )
    return DeviceAttentionBatch(
        phase=batch.phase,
        cu_seqlens_q=staging.device("cu_seqlens_q", cu_seqlens_q, torch.int32),
        max_seqlen_q=max(batch.query_lengths),
        groups=tuple(groups),
    )


def _decode_group_widths(shape: GraphShape, window_width: int):
    return zip(ATTENTION_GROUPS, (shape[1], window_width), strict=True)


def empty_decode_paged_attention_batch(
    device: str | torch.device, shape: GraphShape, window_width: int, page_size: int
) -> DeviceAttentionBatch:
    """Allocate the fixed addresses that replay will overwrite."""
    rows = shape[0]

    def empty(dtype: torch.dtype, *shape: int) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)

    return DeviceAttentionBatch(
        phase=BatchPhase.DECODE,
        cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device=device),
        max_seqlen_q=1,
        groups=tuple(
            (
                name,
                PagedKvTable(
                    page_table=empty(torch.int32, rows, width),
                    lengths=empty(torch.int32, rows),
                    append_slots=empty(torch.long, rows),
                    min_length=1,
                    max_length=width * page_size,
                ),
            )
            for name, width in _decode_group_widths(shape, window_width)
        ),
    )


def _graph_fields(name, pages, lengths, slots):
    return {
        f"{name}_page_table": pages,
        f"{name}_seqused_k": lengths,
        f"{name}_append_slots": slots,
    }


def paged_attention_graph_inputs(
    batch: DeviceAttentionBatch,
) -> dict[str, torch.Tensor]:
    """Expose stable graph input names for one device batch."""

    inputs = {}
    for name, group in batch.groups:
        inputs.update(
            _graph_fields(name, group.page_table, group.lengths, group.append_slots)
        )
    return inputs


def decode_paged_attention_inputs(
    shape: GraphShape, window_width: int, batch: AttentionBatch | None
) -> dict[str, Any]:
    """Pack sparse decode rows directly into capture or replay shapes."""
    rows = shape[0]
    inputs = {}
    groups = (None, None) if batch is None else batch.groups
    for (name, width), group in zip(
        _decode_group_widths(shape, window_width), groups, strict=True
    ):
        lengths = np.ones(rows, dtype=np.int32)
        # Each inactive row writes a separate slot on the shared null page.
        slots = np.arange(rows, dtype=np.int64)
        if group is not None:
            live = len(group.lengths)
            lengths[:live] = group.lengths
            slots[:live] = group.append_slots
        inputs.update(
            _graph_fields(name, dense_page_table(group, rows, width), lengths, slots)
        )
    return inputs


def context_parallel_query_batch(
    batch: DeviceAttentionBatch, tokens: ContextParallelTokens
) -> DeviceAttentionBatch:
    """Lower local Q metadata against globally reconstructed paged K/V.

    FlashAttention aligns causal queries to the end of each K sequence. Trim
    K lengths by the global query suffix following this rank's segment, so
    that local queries retain their original causal positions. Exclude rows
    with no local query. Page tables and append slots still refer to global
    cache storage; callers write gathered K/V with the original batch first.
    """

    if batch.phase != BatchPhase.PREFILL:
        raise ValueError("context-parallel query sharding is only used for prefill")
    if batch.groups[0][1].append_slots.numel() != tokens.plan.total_tokens:
        raise ValueError("attention batch does not match its context token plan")
    groups = tuple(
        (
            name,
            replace(
                group,
                page_table=group.page_table.index_select(0, tokens.active_rows),
                lengths=(group.lengths - tokens.suffix_tokens).index_select(
                    0, tokens.active_rows
                ),
                min_length=0,
            ),
        )
        for name, group in batch.groups
    )
    return replace(
        batch,
        cu_seqlens_q=tokens.cu_seqlens_q,
        max_seqlen_q=tokens.max_seqlen_q,
        groups=groups,
    )


__all__ = [
    "DeviceAttentionBatch",
    "context_parallel_query_batch",
    "PagedKvArena",
    "PagedKvTable",
    "decode_paged_attention_inputs",
    "empty_decode_paged_attention_batch",
    "paged_attention_graph_inputs",
    "stage_paged_attention_batch",
]
