"""Physical paged-KV storage and common device metadata views."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import torch

from resin.backend.resources import CachePoolBinding

if TYPE_CHECKING:
    from resin import BatchPhase, CacheLayout
    from resin.backend.cuda.staging import TensorStager
    from resin.backend.paged_attention import PagedAttentionBatch, PagedAttentionGroup


class SlabPageGroup(Protocol):
    """The page-table fields shared by slab-packed model caches."""

    @property
    def slab_ids(self) -> torch.Tensor: ...

    @property
    def slotnums(self) -> torch.Tensor: ...

    @property
    def first_blocks(self) -> torch.Tensor: ...


class SlabPagedKvCache(Protocol):
    """Narrow cache interface consumed by importable attention kernels."""

    @property
    def page_tokens(self) -> int: ...

    @property
    def slab_units(self) -> int: ...

    @property
    def groups(self) -> tuple[SlabPageGroup, ...]: ...

    @property
    def keys(self) -> torch.Tensor: ...

    @property
    def values(self) -> torch.Tensor: ...

    def rows(
        self, group_index: int, layer: int, layer_count: int, positions: torch.Tensor
    ) -> torch.Tensor: ...

    def pages(self, group_index: int, layer: int, layer_count: int) -> torch.Tensor: ...


class PagedCacheMapping(Protocol):
    """Physical address mapping accepted by fused cache-aware kernels."""

    @property
    def parent_ids(self) -> torch.Tensor: ...

    @property
    def child_slots(self) -> torch.Tensor: ...

    @property
    def first_blocks(self) -> torch.Tensor: ...

    @property
    def block_tokens(self) -> int: ...

    @property
    def divisor(self) -> int: ...

    @property
    def parent_rows(self) -> int: ...

    @property
    def packing(self) -> int: ...

    @property
    def row_scale(self) -> int: ...

    @property
    def layer_offset(self) -> int: ...


class PagedKvArena:
    """One zero-initialized K/V allocation backed by a native cache pool."""

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
        self.page_size = page_size
        self.binding = CachePoolBinding(layout, resource=resource)
        shape = (2, layer_count, self.capacity + 1, page_size, num_kv_heads, head_dim)
        self.storage = torch.zeros(shape, dtype=dtype, device=device)
        self.keys = self.storage[0]
        self.values = self.storage[1]

    def check_pages(self, page_ids: Iterable[int | np.integer]) -> None:
        if any(page < 0 or page > self.capacity for page in page_ids):
            raise ValueError("BatchPlan cache block ID is outside the physical slab")

    def cache_storage_buffers(self, groups):
        buffers = []
        for _, addresses in groups:
            for slab, slot in addresses:
                if not 0 < slab <= self.capacity or slot:
                    raise ValueError(
                        "cache storage address is outside the managed arena"
                    )
                buffers.append(
                    tuple(
                        self.storage[kv, layer, slab].view(torch.uint8).reshape(-1)
                        for kv in range(2)
                        for layer in range(self.storage.shape[1])
                    )
                )
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
class PagedKvCsr:
    """CSR page metadata shared by paged append and decode kernels."""

    indptr: torch.Tensor
    page_ids: torch.Tensor
    last_page_lengths: torch.Tensor
    append_batch_indices: torch.Tensor
    append_positions: torch.Tensor


@dataclass(frozen=True, slots=True)
class DevicePagedAttentionBatch:
    """CUDA tensors for one backend-neutral named paged-attention batch."""

    phase: BatchPhase
    round_index: int
    token_indices: torch.Tensor
    sequence_indices: torch.Tensor
    last_token_indices: torch.Tensor
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
    batch: PagedAttentionBatch, staging: TensorStager, prefix: str
) -> DevicePagedAttentionBatch:
    """Copy one semantic paged-attention batch to CUDA tensors."""

    def device(name: str, values: object, dtype: torch.dtype) -> torch.Tensor:
        return staging.device(f"{prefix}.{name}", values, dtype)

    cu_seqlens_q = [0]
    for length in batch.query_lengths:
        cu_seqlens_q.append(cu_seqlens_q[-1] + length)
    row_count = len(batch.sequence_indices)
    groups = []
    for group in batch.groups:
        if (
            group.page_table.size
            and int(cast(np.ndarray, group.page_table).max()) > np.iinfo(np.int32).max
        ):
            raise ValueError(f"paged attention group {group.name!r} exceeds CUDA int32")
        groups.append(
            (
                group.name,
                PagedKvTable(
                    page_table=device(
                        f"{group.name}.pages", group.page_table, torch.int32
                    ).view(row_count, group.width),
                    lengths=device(f"{group.name}.lengths", group.lengths, torch.int32),
                    append_slots=device(
                        f"{group.name}.append", group.append_slots, torch.long
                    ),
                    min_length=min(group.lengths),
                    max_length=max(group.lengths),
                ),
            )
        )
    return DevicePagedAttentionBatch(
        phase=batch.phase,
        round_index=0,
        token_indices=device("token_indices", batch.token_indices, torch.long),
        sequence_indices=device("sequence_indices", batch.sequence_indices, torch.long),
        last_token_indices=device(
            "last_token_indices",
            tuple(end - 1 for end in cu_seqlens_q[1:]),
            torch.long,
        ),
        cu_seqlens_q=device("cu_seqlens_q", cu_seqlens_q, torch.int32),
        max_seqlen_q=max(batch.query_lengths),
        groups=tuple(groups),
    )


def empty_decode_paged_attention_batch(
    device: str | torch.device,
    batch_size: int,
    group_shapes: Mapping[str, tuple[int, int]],
) -> DevicePagedAttentionBatch:
    """Allocate one fixed-address decode batch for CUDA graph capture."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if not group_shapes:
        raise ValueError("a decode attention batch needs at least one cache group")
    for name, dimensions in group_shapes.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("paged attention group name must be nonempty")
        if (
            not isinstance(dimensions, tuple)
            or len(dimensions) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in dimensions
            )
        ):
            raise ValueError("paged attention group shapes must be positive pairs")

    def empty(dtype: torch.dtype, *shape: int) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)

    groups = tuple(
        (
            name,
            PagedKvTable(
                page_table=empty(torch.int32, batch_size, width),
                lengths=empty(torch.int32, batch_size),
                append_slots=empty(torch.long, batch_size),
                min_length=1,
                max_length=width * page_size,
            ),
        )
        for name, (width, page_size) in group_shapes.items()
    )
    from resin import BatchPhase

    return DevicePagedAttentionBatch(
        phase=BatchPhase.DECODE,
        round_index=0,
        token_indices=empty(torch.long, batch_size),
        sequence_indices=empty(torch.long, batch_size),
        last_token_indices=torch.arange(batch_size, dtype=torch.long, device=device),
        cu_seqlens_q=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        max_seqlen_q=1,
        groups=groups,
    )


def paged_attention_graph_inputs(
    batch: DevicePagedAttentionBatch,
) -> dict[str, torch.Tensor]:
    """Expose stable graph input names for one device batch."""

    inputs = {
        "token_indices": batch.token_indices,
        "sequence_indices": batch.sequence_indices,
    }
    for name, group in batch.groups:
        inputs[f"{name}_page_table"] = group.page_table
        inputs[f"{name}_seqused_k"] = group.lengths
        inputs[f"{name}_append_slots"] = group.append_slots
    return inputs


def pad_paged_attention_table(
    group: PagedAttentionGroup, width: int
) -> npt.NDArray[np.uint64]:
    """Pad each host row by repeating one page already owned by that row."""

    if group.width > width:
        raise ValueError("paged attention table exceeds fixed CUDA graph shape")
    rows = len(group.lengths)
    source = group.page_table.reshape(rows, group.width)
    padded = np.empty((rows, width), dtype=group.page_table.dtype)
    padded[:, : group.width] = source
    padded[:, group.width :] = source[:, :1]
    return padded.reshape(-1)


def paged_attention_host_inputs(
    batch: PagedAttentionBatch, group_widths: Mapping[str, int]
) -> dict[str, Any]:
    """Build named host values matching :func:`paged_attention_graph_inputs`."""

    inputs = {
        "token_indices": batch.token_indices,
        "sequence_indices": batch.sequence_indices,
    }
    if set(group_widths) != {group.name for group in batch.groups}:
        raise ValueError("CUDA graph group widths do not match attention groups")
    for group in batch.groups:
        inputs[f"{group.name}_page_table"] = pad_paged_attention_table(
            group, group_widths[group.name]
        )
        inputs[f"{group.name}_seqused_k"] = group.lengths
        inputs[f"{group.name}_append_slots"] = group.append_slots
    return inputs


__all__ = [
    "DevicePagedAttentionBatch",
    "PagedKvArena",
    "PagedCacheMapping",
    "PagedKvCsr",
    "PagedKvTable",
    "SlabPageGroup",
    "SlabPagedKvCache",
    "empty_decode_paged_attention_batch",
    "pad_paged_attention_table",
    "paged_attention_graph_inputs",
    "paged_attention_host_inputs",
    "stage_paged_attention_batch",
]
