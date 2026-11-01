"""GPT-OSS full-history and sliding-window attention metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from resin import BatchPhase
from resin.backend.plans import PagedCachePlan

PAGE_SIZE = 128
SLIDING_WINDOW = 128
PREFILL_CHUNK_SIZE = 8192
# Paged prefill retains the whole query plus the preceding attention window.
WINDOW_RETENTION_TOKENS = PREFILL_CHUNK_SIZE + SLIDING_WINDOW - 1
WINDOW_TABLE_WIDTH = 1 + (WINDOW_RETENTION_TOKENS - 1 + PAGE_SIZE - 1) // PAGE_SIZE


@dataclass(frozen=True, slots=True)
class AttentionGroup:
    """One dense page table and its cache append locations."""

    name: str
    width: int
    page_table: npt.NDArray[np.uint64]
    lengths: tuple[int, ...]
    append_slots: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class AttentionBatch:
    phase: BatchPhase
    token_indices: npt.NDArray[np.int64]
    sequence_indices: tuple[int, ...]
    query_lengths: tuple[int, ...]
    groups: tuple[AttentionGroup, ...]

    def group(self, name: str) -> AttentionGroup:
        try:
            return next(group for group in self.groups if group.name == name)
        except StopIteration as error:
            raise KeyError(f"unknown attention group: {name!r}") from error


def build_attention_batches(
    plan, cache_layout
) -> tuple[tuple[AttentionBatch, ...], AttentionBatch | None]:
    """Lower a homogeneous native plan to GPT-OSS's two dense page tables."""

    cache = PagedCachePlan.from_plan(plan, cache_layout)
    if len(cache.groups) != 2:
        raise ValueError("GPT-OSS attention requires full and sliding cache groups")
    native = cache.batch
    query_lengths = tuple(map(int, native.query_lengths))
    if any(length > PREFILL_CHUNK_SIZE for length in query_lengths):
        raise ValueError(f"attention query length exceeds {PREFILL_CHUNK_SIZE} tokens")
    if native.phase == BatchPhase.DECODE and any(
        length != 1 for length in query_lengths
    ):
        raise ValueError("decode rows must contain one token")
    if np.intersect1d(
        cache.groups[0].block_slab_ids, cache.groups[1].block_slab_ids
    ).size:
        raise ValueError("attention cache groups cannot address one parent")

    rows = tuple(range(native.num_sequences))
    post_lengths = native.post_lengths
    groups = []
    for name, group, minimum_width in zip(
        ("full", "sliding"), cache.groups, (1, WINDOW_TABLE_WIDTH), strict=True
    ):
        packed_blocks = group.packed_block_ids
        page_rows = tuple(packed_blocks[group.block_slice(row)] for row in rows)
        width = max(minimum_width, *(len(row) for row in page_rows))
        table = np.empty((len(rows), width), dtype=np.uint64)
        for index, row in enumerate(page_rows):
            table[index, : len(row)] = row
            table[index, len(row) :] = row[0]
        page_size = int(cache_layout.groups[group.group_index].block_tokens)
        lengths = tuple(
            int(post_lengths[row])
            - int(group.block_table_start_indices[row]) * page_size
            for row in rows
        )
        append_slots = group.packed_append_ids.astype(
            np.int64, copy=False
        ) * page_size + group.append_block_offsets.astype(np.int64, copy=False)
        groups.append(
            AttentionGroup(
                name=name,
                width=width,
                page_table=table.reshape(-1),
                lengths=lengths,
                append_slots=append_slots,
            )
        )
    batch = AttentionBatch(
        phase=native.phase,
        token_indices=np.arange(native.num_tokens, dtype=np.int64),
        sequence_indices=rows,
        query_lengths=query_lengths,
        groups=tuple(groups),
    )
    if batch.phase == BatchPhase.PREFILL:
        return (batch,), None
    return (), batch


__all__ = [
    "AttentionBatch",
    "AttentionGroup",
    "PAGE_SIZE",
    "PREFILL_CHUNK_SIZE",
    "SLIDING_WINDOW",
    "WINDOW_RETENTION_TOKENS",
    "WINDOW_TABLE_WIDTH",
    "build_attention_batches",
]
