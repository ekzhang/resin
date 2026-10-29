"""Translate Resin cache metadata to Gemma 4 paged-attention inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from resin import BatchPhase
from resin.backend.plans import CacheGroupView, PagedCachePlan

PAGE_SIZE = 64
PREFILL_CHUNK_SIZE = 8192
SLIDING_WINDOW = 1024
WINDOW_RETENTION_TOKENS = PREFILL_CHUNK_SIZE + SLIDING_WINDOW - 1


@dataclass(frozen=True, slots=True)
class PagedAttentionPlan:
    qo_indptr: npt.NDArray[np.int32]
    kv_indptr: npt.NDArray[np.int32]
    page_indices: npt.NDArray[np.int32]
    last_page_lengths: npt.NDArray[np.int32]
    append_slots: npt.NDArray[np.int64]
    first_token_positions: tuple[int, ...]
    kv_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Gemma4BatchPlan:
    phase: BatchPhase
    token_ids: npt.NDArray[np.int64]
    positions: npt.NDArray[np.int64]
    token_indptr: npt.NDArray[np.int32]
    local: PagedAttentionPlan
    global_: PagedAttentionPlan
    last_token_indices: npt.NDArray[np.int64]
    local_envelope: PagedAttentionPlan | None = None
    global_envelope: PagedAttentionPlan | None = None


def _paged_plan(
    group: CacheGroupView,
    token_indptr: npt.NDArray[np.int32],
    positions: npt.NDArray[np.int64],
    *,
    decode_lookahead: int,
) -> PagedAttentionPlan:
    packed_blocks = group.packed_block_ids.astype(np.int64, copy=False)
    packed_appends = group.packed_append_ids.astype(np.int64, copy=False)
    page_rows: list[npt.NDArray[np.int64]] = []
    last_page_lengths = []
    first_positions = []
    kv_lengths = []
    for row in range(len(token_indptr) - 1):
        start, end = map(int, token_indptr[row : row + 2])
        post_length = int(positions[end - 1]) + 1
        leased_length = post_length + decode_lookahead
        first_block = int(group.block_table_start_indices[row])
        pages = packed_blocks[group.block_slice(row)]
        first_position = first_block * PAGE_SIZE
        kv_length = leased_length - first_position
        page_rows.append(pages)
        last_page_lengths.append((leased_length - 1) % PAGE_SIZE + 1)
        first_positions.append(first_position)
        kv_lengths.append(kv_length)

    kv_indptr = np.zeros(len(page_rows) + 1, dtype=np.int32)
    kv_indptr[1:] = np.cumsum([len(row) for row in page_rows], dtype=np.int32)
    return PagedAttentionPlan(
        qo_indptr=token_indptr,
        kv_indptr=kv_indptr,
        # Physical parent zero is reserved for inactive CUDA-graph rows, so
        # positive Engine page IDs map directly into the device slabs.
        page_indices=(np.concatenate(page_rows)).astype(np.int32, copy=False),
        last_page_lengths=np.asarray(last_page_lengths, dtype=np.int32),
        append_slots=(packed_appends * PAGE_SIZE + group.append_block_offsets).astype(
            np.int64, copy=False
        ),
        first_token_positions=tuple(first_positions),
        kv_lengths=tuple(kv_lengths),
    )


def _reservation_envelope(group: CacheGroupView, token_indptr):
    packed = group.packed_block_ids.astype(np.int32)
    first = tuple(int(value) * PAGE_SIZE for value in group.block_table_start_indices)
    counts = np.diff(group.block_indptr).astype(np.int64)
    return PagedAttentionPlan(
        qo_indptr=token_indptr,
        kv_indptr=group.block_indptr.astype(np.int32),
        page_indices=packed,
        last_page_lengths=np.full(len(counts), PAGE_SIZE, dtype=np.int32),
        append_slots=(
            group.packed_append_ids * PAGE_SIZE + group.append_block_offsets
        ).astype(np.int64),
        first_token_positions=first,
        kv_lengths=tuple(int(value) * PAGE_SIZE for value in counts),
    )


def build_gemma4_batch_plan(
    plan,
    cache_layout,
    *,
    decode_lookahead: int = 0,
    prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
) -> Gemma4BatchPlan:
    cache = PagedCachePlan.from_plan(
        plan, cache_layout, decode_lookahead=decode_lookahead
    )
    batch = cache.batch
    token_ids = batch.token_ids.astype(np.int64, copy=False)
    positions = batch.positions.astype(np.int64, copy=False)
    token_indptr = batch.token_indptr.astype(np.int32, copy=False)
    if batch.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
        raise ValueError("plan phase must be prefill or decode")
    if decode_lookahead < 0:
        raise ValueError("decode_lookahead cannot be negative")
    if batch.phase != BatchPhase.DECODE and decode_lookahead:
        raise ValueError("decode_lookahead is only valid during decode")
    query_lengths = batch.query_lengths
    if type(prefill_chunk_size) is not int or prefill_chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be a positive integer")
    if batch.phase == BatchPhase.PREFILL and np.any(query_lengths > prefill_chunk_size):
        raise ValueError(f"prefill rows cannot exceed {prefill_chunk_size} tokens")
    if batch.phase == BatchPhase.DECODE and np.any(query_lengths != 1):
        raise ValueError("decode rows must contain one token")

    history, local = cache.groups
    if batch.phase == BatchPhase.DECODE:
        local_envelope = _reservation_envelope(local, token_indptr)
        global_envelope = _reservation_envelope(history, token_indptr)

        def readable(source):
            return _reshape_paged_plan(
                source,
                token_indptr,
                positions,
                positions + decode_lookahead + 1,
                truncate=True,
                include_append=True,
            )

        return Gemma4BatchPlan(
            phase=batch.phase,
            token_ids=token_ids,
            positions=positions,
            token_indptr=token_indptr,
            local=readable(local_envelope),
            global_=readable(global_envelope),
            last_token_indices=(token_indptr[1:] - 1).astype(np.int64),
            local_envelope=local_envelope,
            global_envelope=global_envelope,
        )

    return Gemma4BatchPlan(
        phase=batch.phase,
        token_ids=token_ids,
        positions=positions,
        token_indptr=token_indptr,
        local=_paged_plan(
            local,
            token_indptr,
            positions,
            decode_lookahead=decode_lookahead,
        ),
        global_=_paged_plan(
            history,
            token_indptr,
            positions,
            decode_lookahead=decode_lookahead,
        ),
        last_token_indices=(token_indptr[1:] - 1).astype(np.int64),
    )


def _reshape_paged_plan(
    source: PagedAttentionPlan,
    token_indptr: npt.NDArray[np.int32],
    positions: npt.NDArray[np.int64],
    post_lengths: npt.NDArray[np.int64],
    *,
    truncate: bool,
    include_append: bool,
) -> PagedAttentionPlan:
    page_rows = []
    append_slots = []
    last_page_lengths = []
    kv_lengths = []
    for row in range(len(token_indptr) - 1):
        page_start, page_end = map(int, source.kv_indptr[row : row + 2])
        pages = source.page_indices[page_start:page_end]
        first_position = source.first_token_positions[row]
        post_length = int(post_lengths[row])
        if post_length <= first_position:
            raise ValueError("Gemma 4 paged attention row has no retained context")
        required_pages = (
            (post_length - 1) // PAGE_SIZE - first_position // PAGE_SIZE + 1
        )
        if required_pages > len(pages):
            raise ValueError("Gemma 4 paged attention row exceeds its lease")
        if truncate:
            pages = pages[:required_pages]
        start, end = map(int, token_indptr[row : row + 2])
        row_positions = positions[start:end]
        if include_append:
            relative = row_positions // PAGE_SIZE - first_position // PAGE_SIZE
            if np.any(relative < 0) or np.any(relative >= len(pages)):
                raise ValueError("Gemma 4 query positions fall outside retained pages")
            append_slots.extend(
                (pages[relative] * PAGE_SIZE + row_positions % PAGE_SIZE).tolist()
            )
        page_rows.append(pages)
        last_page_lengths.append((post_length - 1) % PAGE_SIZE + 1)
        kv_lengths.append(post_length - first_position)

    kv_indptr = np.zeros(len(page_rows) + 1, dtype=np.int32)
    kv_indptr[1:] = np.cumsum([len(row) for row in page_rows], dtype=np.int32)
    return PagedAttentionPlan(
        qo_indptr=token_indptr,
        kv_indptr=kv_indptr,
        page_indices=np.concatenate(page_rows).astype(np.int32, copy=False),
        last_page_lengths=np.asarray(last_page_lengths, dtype=np.int32),
        append_slots=np.asarray(append_slots, dtype=np.int64),
        first_token_positions=source.first_token_positions,
        kv_lengths=tuple(kv_lengths),
    )


def expand_gemma4_decode_batch(
    batch: Gemma4BatchPlan,
    draft_ids: npt.NDArray[np.int64],
) -> Gemma4BatchPlan:
    """Expand one-token decode rows into anchor-plus-draft verification rows."""

    if batch.phase != BatchPhase.DECODE:
        raise ValueError("only decode batches can be expanded for verification")
    rows = len(batch.token_indptr) - 1
    drafts = np.asarray(draft_ids, dtype=np.int64)
    if drafts.ndim != 2 or drafts.shape[0] != rows:
        raise ValueError("draft_ids must have shape [decode rows, tokens]")
    if len(batch.token_ids) != rows:
        raise ValueError("decode batches must contain one anchor per row")
    width = drafts.shape[1] + 1
    token_ids = np.concatenate((batch.token_ids[:, None], drafts), axis=1).reshape(-1)
    positions = np.concatenate(
        [
            np.arange(int(position), int(position) + width, dtype=np.int64)
            for position in batch.positions
        ]
    )
    token_indptr = np.arange(0, (rows + 1) * width, width, dtype=np.int32)
    post_lengths = batch.positions.astype(np.int64, copy=False) + width
    return Gemma4BatchPlan(
        phase=BatchPhase.PREFILL,
        token_ids=token_ids,
        positions=positions,
        token_indptr=token_indptr,
        local=_reshape_paged_plan(
            batch.local,
            token_indptr,
            positions,
            post_lengths,
            truncate=True,
            include_append=True,
        ),
        global_=_reshape_paged_plan(
            batch.global_,
            token_indptr,
            positions,
            post_lengths,
            truncate=True,
            include_append=True,
        ),
        last_token_indices=(token_indptr[1:] - 1).astype(np.int64),
    )


def _trim_assistant_local_window(
    source: PagedAttentionPlan,
    window_tokens: int,
) -> PagedAttentionPlan:
    """Drop whole old pages from a read-only assistant view.

    The unchanged fractional last page and relative sliding-window mask select
    the same final keys. Absolute query RoPE positions and cache ownership do
    not depend on this view's first page.
    """
    page_rows = []
    first_positions = []
    kv_lengths = []
    for row, length in enumerate(source.kv_lengths):
        start, end = map(int, source.kv_indptr[row : row + 2])
        skip_pages = max(0, (length - window_tokens) // PAGE_SIZE)
        page_rows.append(source.page_indices[start + skip_pages : end])
        first_positions.append(
            source.first_token_positions[row] + skip_pages * PAGE_SIZE
        )
        kv_lengths.append(length - skip_pages * PAGE_SIZE)
    indptr = np.zeros(len(page_rows) + 1, dtype=np.int32)
    indptr[1:] = np.cumsum([len(row) for row in page_rows], dtype=np.int32)
    return replace(
        source,
        kv_indptr=indptr,
        page_indices=np.concatenate(page_rows).astype(np.int32, copy=False),
        first_token_positions=tuple(first_positions),
        kv_lengths=tuple(kv_lengths),
    )


def gemma4_assistant_context(
    batch: Gemma4BatchPlan,
    *,
    local_window_tokens: int | None = None,
) -> Gemma4BatchPlan:
    """Expose only target K/V preceding each unprocessed decode anchor."""

    if batch.phase != BatchPhase.DECODE or np.any(np.diff(batch.token_indptr) != 1):
        raise ValueError("assistant context requires one-token decode rows")
    if local_window_tokens is not None and (
        type(local_window_tokens) is not int or local_window_tokens <= 0
    ):
        raise ValueError("assistant local window must be a positive integer")
    contexts = batch.positions.astype(np.int64, copy=False)
    context = Gemma4BatchPlan(
        phase=BatchPhase.DECODE,
        token_ids=batch.token_ids,
        positions=batch.positions,
        token_indptr=batch.token_indptr,
        local=_reshape_paged_plan(
            batch.local,
            batch.token_indptr,
            batch.positions,
            contexts,
            truncate=True,
            include_append=False,
        ),
        global_=_reshape_paged_plan(
            batch.global_,
            batch.token_indptr,
            batch.positions,
            contexts,
            truncate=True,
            include_append=False,
        ),
        last_token_indices=batch.last_token_indices,
    )

    if local_window_tokens is None:
        return context
    return replace(
        context, local=_trim_assistant_local_window(context.local, local_window_tokens)
    )


__all__ = [
    "Gemma4BatchPlan",
    "PAGE_SIZE",
    "PREFILL_CHUNK_SIZE",
    "PagedAttentionPlan",
    "SLIDING_WINDOW",
    "WINDOW_RETENTION_TOKENS",
    "build_gemma4_batch_plan",
    "expand_gemma4_decode_batch",
    "gemma4_assistant_context",
]
