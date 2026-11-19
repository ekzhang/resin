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
ATTENTION_GROUPS = ("full", "sliding")


@dataclass(frozen=True, slots=True)
class AttentionGroup:
    """Sparse page rows and append slots before the runner chooses a shape."""

    name: str
    width: int
    pages: tuple[npt.NDArray[np.uint64], ...]
    lengths: tuple[int, ...]
    append_slots: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class AttentionBatch:
    phase: BatchPhase
    query_lengths: tuple[int, ...]
    groups: tuple[AttentionGroup, ...]


def build_attention_batch(plan, cache_layout) -> AttentionBatch:
    """Validate and retain sparse attention rows for eager or graph execution."""

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

    rows = range(native.num_sequences)
    post_lengths = native.post_lengths
    groups = []
    for name, group, minimum_width in zip(
        ATTENTION_GROUPS, cache.groups, (1, WINDOW_TABLE_WIDTH), strict=True
    ):
        packed_blocks = group.packed_block_ids
        if packed_blocks.size and np.max(packed_blocks) > np.iinfo(np.int32).max:
            raise ValueError(f"paged attention group {name!r} exceeds CUDA int32")
        page_rows = tuple(packed_blocks[group.block_slice(row)] for row in rows)
        width = max(minimum_width, *(len(row) for row in page_rows))
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
                pages=page_rows,
                lengths=lengths,
                append_slots=append_slots,
            )
        )
    return AttentionBatch(
        phase=native.phase,
        query_lengths=query_lengths,
        groups=tuple(groups),
    )


def dense_page_table(
    group: AttentionGroup | None, rows: int, width: int
) -> npt.NDArray[np.int32]:
    """Write directly into the final table. Keep inactive rows on null page zero."""
    pages = () if group is None else group.pages
    if group is not None and (group.width > width or len(pages) > rows):
        raise ValueError("paged attention table exceeds target shape")
    table = np.empty((rows, width), dtype=np.int32)
    for index, row in enumerate(pages):
        table[index, : len(row)] = row
        # A live row may only address its own pages, including masked columns.
        table[index, len(row) :] = row[0]
    table[len(pages) :] = 0
    return table


__all__ = [
    "ATTENTION_GROUPS",
    "AttentionBatch",
    "AttentionGroup",
    "PAGE_SIZE",
    "PREFILL_CHUNK_SIZE",
    "SLIDING_WINDOW",
    "WINDOW_RETENTION_TOKENS",
    "WINDOW_TABLE_WIDTH",
    "build_attention_batch",
    "dense_page_table",
]
