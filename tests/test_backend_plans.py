from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupFamily,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    CacheRetention,
)
from resin.backend.paged_attention import (
    PagedAttentionGroupSpec,
    build_paged_attention_batch,
)
from resin.backend.plans import BatchView, CacheGroupView, PagedCachePlan, plan_rows


def _plan() -> BatchPlan:
    return BatchPlan(
        sequence_ids=[10, 11],
        context_lengths=[0, 2],
        token_ids=[1, 2, 3],
        token_indptr=[0, 2, 3],
        sampling_mask=[True, True],
        cache_groups=[
            CacheGroupPlan(
                group_index=0,
                block_indptr=[0, 1, 2],
                block_slab_ids=[1, 2],
                block_slotnums=[0, 1],
                block_table_start_indices=[0, 0],
                append_slab_ids=[1, 1, 2],
                append_slotnums=[0, 0, 1],
                append_block_offsets=[0, 1, 0],
                blocks_per_slab=2,
            )
        ],
    )


def test_batch_view_derives_canonical_row_geometry() -> None:
    native = _plan()
    batch = BatchView.from_plan(native)
    assert batch.num_sequences == 2
    assert batch.num_tokens == 3
    assert batch.query_lengths.tolist() == [2, 1]
    assert batch.post_lengths.tolist() == [2, 3]
    assert batch.last_token_indices.tolist() == [1, 2]
    assert batch.token_ids[batch.token_slice(1)].tolist() == [3]
    assert np.shares_memory(batch.token_ids, native.token_ids)
    assert np.shares_memory(batch.positions, native.positions)
    assert not batch.token_ids.flags.writeable
    with pytest.raises(TypeError, match="init=False"):
        replace(batch, positions=np.zeros(3, dtype=np.uint32))
    with pytest.raises(FrozenInstanceError):
        setattr(batch, "positions", np.zeros(3, dtype=np.uint32))
    del native
    assert batch.positions.tolist() == [0, 1, 2]


@pytest.mark.parametrize("construct", [BatchView, BatchView.from_plan])
def test_batch_view_requires_a_native_plan_owner(construct) -> None:
    native = _plan()
    plan = SimpleNamespace(
        **{
            name: getattr(native, name)
            for name in BatchView.__dataclass_fields__
            if name != "_plan"
        }
    )
    with pytest.raises(TypeError, match="native BatchPlan"):
        construct(cast(BatchPlan, plan))


def test_cache_group_view_preserves_structured_and_packed_addresses() -> None:
    plan = _plan()
    batch = BatchView.from_plan(plan)
    group = CacheGroupView.from_plan(
        batch,
        group_index=0,
        blocks_per_slab=2,
        parent_capacity=2,
        block_tokens=2,
    )
    assert group.block_slabs(0).tolist() == [1]
    assert group.block_slots(1).tolist() == [1]
    assert group.packed_block_ids.tolist() == [2, 5]
    assert group.packed_append_ids.tolist() == [2, 2, 5]
    assert group.append_block_offsets.dtype == np.uint32


@pytest.mark.parametrize(
    "options,message",
    [
        ({"group_index": 1}, "outside the native plan"),
        ({"group_index": 0, "blocks_per_slab": 1}, "wrong slab packing"),
        ({"group_index": 0, "parent_capacity": 1}, "outside its layout"),
        ({"group_index": 0, "block_tokens": 1}, "outside its layout"),
    ],
)
def test_cache_group_view_keeps_layout_specific_bounds(options, message):
    with pytest.raises(ValueError, match=message):
        CacheGroupView.from_plan(BatchView.from_plan(_plan()), **options)


def test_cache_group_view_cannot_select_from_forged_batch_arrays():
    fake = SimpleNamespace(cache_groups=_plan().cache_groups)
    with pytest.raises(TypeError, match="native-backed BatchView"):
        CacheGroupView.from_plan(cast(BatchView, fake), group_index=0)


def test_request_local_rows_check_cache_progress_and_vocabulary():
    plan = _plan()
    caches = {11: SimpleNamespace(length=2)}
    rows = plan_rows(plan, caches, vocab_size=4)
    assert rows[1].cache is caches[11]
    assert rows[0].token_ids.tolist() == [1, 2]
    assert rows[1].positions.tolist() == [2]
    with pytest.raises(ValueError, match="non-contiguous positions for request 11"):
        plan_rows(plan, {}, vocab_size=4)
    with pytest.raises(ValueError, match="token id is out of range for request 11"):
        plan_rows(plan, caches, vocab_size=3)


def test_paged_attention_names_semantics_and_preserves_packed_addresses() -> None:
    plan = BatchPlan(
        sequence_ids=[10],
        context_lengths=[0],
        token_ids=[1, 2, 3],
        token_indptr=[0, 3],
        cache_groups=[
            CacheGroupPlan(
                group_index=0,
                block_indptr=[0, 2],
                block_slab_ids=[1, 2],
                block_slotnums=[0, 1],
                block_table_start_indices=[0],
                append_slab_ids=[1, 1, 2],
                append_slotnums=[0, 0, 1],
                append_block_offsets=[0, 1, 0],
                blocks_per_slab=2,
            )
        ],
    )
    batch = build_paged_attention_batch(
        plan,
        CacheLayout(parent_capacity=2, groups=[CacheGroupLayout(2, blocks_per_slab=2)]),
        (PagedAttentionGroupSpec("history", 0, minimum_table_width=3),),
        max_query_tokens=3,
    )
    history = batch.group("history")

    assert batch.query_lengths == (3,)
    assert history.width == 3
    assert history.page_table.tolist() == [2, 5, 2]
    assert history.lengths == (3,)
    assert history.append_slots.tolist() == [4, 5, 10]


def test_paged_attention_lowers_unequal_rows_and_window_origins():
    offsets = [0, 1, 2, 1, 1, 2]
    plan = BatchPlan(
        sequence_ids=[10, 11, 12],
        context_lengths=[0, 5, 17],
        token_ids=[1, 2, 3, 4, 5, 6],
        token_indptr=[0, 3, 4, 6],
        cache_groups=[
            CacheGroupPlan(
                group_index=0,
                block_indptr=[0, 1, 3, 8],
                block_slab_ids=list(range(1, 9)),
                block_slotnums=[0, 1, 1, 0, 0, 0, 0, 0],
                block_table_start_indices=[0, 0, 0],
                append_slab_ids=[1, 1, 1, 3, 8, 8],
                append_slotnums=[0, 0, 0, 1, 0, 0],
                append_block_offsets=offsets,
                blocks_per_slab=2,
            ),
            CacheGroupPlan(
                group_index=1,
                block_indptr=[0, 1, 3, 6],
                block_slab_ids=list(range(9, 15)),
                block_slotnums=[0] * 6,
                block_table_start_indices=[0, 0, 2],
                append_slab_ids=[9, 9, 9, 11, 14, 14],
                append_slotnums=[0] * 6,
                append_block_offsets=offsets,
            ),
        ],
    )
    layout = CacheLayout(
        parent_capacity=14,
        groups=[
            CacheGroupLayout(4, blocks_per_slab=2),
            CacheGroupLayout(
                4,
                retention=CacheRetention.WINDOW,
                window_tokens=8,
                read_window_tokens=4,
            ),
        ],
    )
    batch = build_paged_attention_batch(
        plan,
        layout,
        (PagedAttentionGroupSpec("history", 0), PagedAttentionGroupSpec("window", 1)),
        max_query_tokens=3,
    )
    assert batch.query_lengths == (3, 1, 2)
    history, window = batch.group("history"), batch.group("window")
    assert history.page_table.reshape(3, -1).tolist() == [
        [2, 2, 2, 2, 2],
        [5, 7, 5, 5, 5],
        [8, 10, 12, 14, 16],
    ]
    assert window.page_table.reshape(3, -1).tolist() == [
        [9, 9, 9],
        [10, 11, 10],
        [12, 13, 14],
    ]
    assert history.lengths == (3, 6, 19)
    assert window.lengths == (3, 6, 11)
    assert history.append_slots.tolist() == [8, 9, 10, 29, 65, 66]
    assert window.append_slots.tolist() == [36, 37, 38, 45, 57, 58]


def test_paged_cache_plan_validates_exact_leased_tables() -> None:
    layout = CacheLayout(parent_capacity=2, groups=[CacheGroupLayout(2)])
    plan = BatchPlan(
        sequence_ids=[10],
        context_lengths=[1],
        token_ids=[3],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
        cache_groups=[
            CacheGroupPlan(
                group_index=0,
                block_indptr=[0, 2],
                block_slab_ids=[1, 2],
                block_slotnums=[0, 0],
                block_table_start_indices=[0],
                append_slab_ids=[1],
                append_slotnums=[0],
                append_block_offsets=[1],
            )
        ],
    )
    cache = PagedCachePlan.from_plan(plan, layout, decode_lookahead=1)
    assert cache.rows == 1
    assert cache.groups[0].address(0, 1) == (2, 0)


def test_paged_cache_plan_accepts_the_two_checkpoint_state_ring() -> None:
    layout = CacheLayout(
        parent_capacity=2, groups=[CacheGroupLayout(4, family=CacheGroupFamily.STATE)]
    )
    plan = BatchPlan(
        sequence_ids=[10],
        context_lengths=[11],
        token_ids=[3],
        token_indptr=[0, 1],
        cache_groups=[
            CacheGroupPlan(
                group_index=0,
                block_indptr=[0, 2],
                block_slab_ids=[1, 2],
                block_slotnums=[0, 0],
                block_table_start_indices=[1],
                append_slab_ids=[2],
                append_slotnums=[0],
                append_block_offsets=[3],
            )
        ],
    )
    cache = PagedCachePlan.from_plan(plan, layout)
    assert cache.groups[0].address(0, 2) == (2, 0)


@pytest.mark.parametrize(
    "context,history_pages", [(125, 3), (128, 3), (381, 3), (4094, 33)]
)
def test_reserved_history_keeps_exact_recurrent_checkpoints(context, history_pages):
    layout = CacheLayout(
        parent_capacity=40,
        groups=[
            CacheGroupLayout(128),
            CacheGroupLayout(4096, family=CacheGroupFamily.STATE),
        ],
    )
    state_last = (context + 2) // 4096
    state_first = max(0, state_last - 1)
    state_pages = state_last - state_first + 1
    plan = BatchPlan(
        sequence_ids=[10],
        context_lengths=[context],
        token_ids=[7],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
        cache_groups=[
            CacheGroupPlan(
                group_index=0,
                block_indptr=[0, history_pages],
                block_slab_ids=list(range(1, history_pages + 1)),
                block_slotnums=[0] * history_pages,
                block_table_start_indices=[0],
                append_slab_ids=[context // 128 + 1],
                append_slotnums=[0],
                append_block_offsets=[context % 128],
            ),
            CacheGroupPlan(
                group_index=1,
                block_indptr=[0, state_pages],
                block_slab_ids=list(range(39, 39 + state_pages)),
                block_slotnums=[0] * state_pages,
                block_table_start_indices=[state_first],
                append_slab_ids=[39 + context // 4096 - state_first],
                append_slotnums=[0],
                append_block_offsets=[context % 4096],
            ),
        ],
    )
    options = dict(decode_lookahead=2, decode_reservation_tokens=136)
    cache = PagedCachePlan.from_plan(plan, layout, **options)
    assert len(cache.groups[0].block_slabs(0)) == history_pages
    assert len(cache.groups[1].block_slabs(0)) == state_pages
    if context == 125:
        with pytest.raises(ValueError, match="exact active range"):
            PagedCachePlan.from_plan(plan, layout, decode_lookahead=2)
        for group, pages, slabs in ((0, 4, [1, 2, 3, 4]), (1, 2, [39, 40])):
            invalid = _replace_plan_group(
                plan,
                group,
                block_indptr=[0, pages],
                block_slab_ids=slabs,
                block_slotnums=[0] * pages,
            )
            with pytest.raises(ValueError, match="exact active range"):
                PagedCachePlan.from_plan(invalid, layout, **options)
    if context == 128:
        short = _replace_plan_group(
            plan,
            0,
            block_indptr=[0, 1],
            block_slab_ids=[2],
            block_slotnums=[0],
            block_table_start_indices=[1],
        )
        with pytest.raises(ValueError, match="exact active range"):
            PagedCachePlan.from_plan(short, layout, **options)
        wrong_append = _replace_plan_group(plan, 0, append_slab_ids=[1])
        with pytest.raises(ValueError, match="append target"):
            PagedCachePlan.from_plan(wrong_append, layout, **options)


@pytest.mark.parametrize("decode", [False, True])
def test_source_free_window_plan_rejects_writes_that_discard_read_history(
    decode: bool,
) -> None:
    layout = CacheLayout(
        parent_capacity=8,
        groups=[CacheGroupLayout(2, retention=CacheRetention.WINDOW, window_tokens=4)],
    )
    token_count = 1 if decode else 6
    plan = BatchPlan(
        sequence_ids=[10],
        context_lengths=[6],
        token_ids=list(range(6, 6 + token_count)),
        token_indptr=[0, token_count],
        phase=BatchPhase.DECODE if decode else BatchPhase.PREFILL,
        cache_groups=[
            CacheGroupPlan(
                group_index=0,
                block_indptr=[0, 3],
                block_slab_ids=[1, 2, 3],
                block_slotnums=[0, 0, 0],
                block_table_start_indices=[3],
                append_slab_ids=[1 + index // 2 for index in range(token_count)],
                append_slotnums=[0] * token_count,
                append_block_offsets=[index % 2 for index in range(token_count)],
            )
        ],
    )
    # The final suffix and every append address look valid, but the first
    # query still needs positions3..5, which this table has already discarded.
    with pytest.raises(ValueError, match="Window read/write capacity"):
        PagedCachePlan.from_plan(plan, layout, decode_lookahead=5 if decode else 0)


def _mixed_cache_plan(*, decode: bool) -> tuple[BatchPlan, CacheLayout]:
    layout = CacheLayout(
        parent_capacity=64,
        groups=[
            CacheGroupLayout(4, blocks_per_slab=2),
            CacheGroupLayout(
                4,
                blocks_per_slab=3,
                retention=CacheRetention.WINDOW,
                window_tokens=12,
                read_window_tokens=5,
            ),
            CacheGroupLayout(4, blocks_per_slab=4, family=CacheGroupFamily.STATE),
        ],
    )
    contexts, lengths = [0, 7, 14], [3, 1, 5]
    groups = []
    for index, descriptor in enumerate(layout.groups):
        indptr, slabs, slots, firsts = [0], [], [], []
        append_slabs, append_slots, offsets = [], [], []
        for row, (context, length) in enumerate(zip(contexts, lengths, strict=True)):
            last = (context + length + (3 if decode else 0) - 1) // 4
            if descriptor.family == CacheGroupFamily.STATE:
                first = max(0, last - 1)
            elif descriptor.window_resident_block_count is not None:
                first = max(0, last + 1 - descriptor.window_resident_block_count)
            else:
                first = 0
            count = last - first + 1
            row_slabs = list(range(1 + len(slabs), 1 + len(slabs) + count))
            slot = row % descriptor.blocks_per_slab
            slabs.extend(row_slabs)
            slots.extend([slot] * count)
            firsts.append(first)
            indptr.append(len(slabs))
            for position in range(context, context + length):
                # Older STATE writes can fall outside the retained checkpoint ring.
                append_slabs.append(row_slabs[max(0, position // 4 - first)])
                append_slots.append(slot)
                offsets.append(position % 4)
        groups.append(
            CacheGroupPlan(
                group_index=index,
                block_indptr=indptr,
                block_slab_ids=slabs,
                block_slotnums=slots,
                block_table_start_indices=firsts,
                append_slab_ids=append_slabs,
                append_slotnums=append_slots,
                append_block_offsets=offsets,
                blocks_per_slab=descriptor.blocks_per_slab,
            )
        )
    plan = BatchPlan(
        sequence_ids=[10, 11, 12],
        context_lengths=contexts,
        token_ids=list(range(sum(lengths))),
        token_indptr=[0, 3, 4, 9],
        phase=BatchPhase.DECODE if decode else BatchPhase.PREFILL,
        cache_groups=groups,
    )
    return plan, layout


def _replace_plan_group(plan: BatchPlan, index: int, **fields: object) -> BatchPlan:
    """Build native metadata changes; layout-specific validity is checked later."""

    group: dict[str, Any] = {
        name: getattr(plan.cache_groups[index], name)
        for name in CacheGroupView.__dataclass_fields__
    }
    group.update(fields)
    groups = list(plan.cache_groups)
    groups[index] = CacheGroupPlan(**group)
    return BatchPlan(
        sequence_ids=plan.sequence_ids,
        context_lengths=plan.context_lengths,
        token_ids=plan.token_ids,
        token_indptr=plan.token_indptr,
        prefill_lengths=plan.prefill_lengths,
        phase=plan.phase,
        cache_groups=groups,
        sampling_mask=plan.sampling_mask,
    )


@pytest.mark.parametrize("decode", [False, True])
def test_paged_cache_plan_validates_heterogeneous_rows_across_retention_groups(
    decode: bool,
) -> None:
    plan, layout = _mixed_cache_plan(decode=decode)
    cache = PagedCachePlan.from_plan(plan, layout, decode_lookahead=3)
    assert cache.batch.query_lengths.tolist() == [3, 1, 5]
    assert cache.batch.positions.tolist() == [0, 1, 2, 7, 14, 15, 16, 17, 18]
    expected = (
        [[0, 0, 0], [0, 0, 2], [0, 1, 4]]
        if decode
        else [[0, 0, 0], [0, 0, 1], [0, 0, 3]]
    )
    assert [
        group.block_table_start_indices.tolist() for group in cache.groups
    ] == expected


@pytest.mark.parametrize("field", ["append_slab_ids", "append_slotnums"])
def test_native_plan_rejects_cross_row_append_targets(field: str) -> None:
    plan, _ = _mixed_cache_plan(decode=True)
    group = plan.cache_groups[0]
    values = getattr(group, field).copy()
    values[3] = values[4]  # Row 1 writes into row 2's valid physical address.
    with pytest.raises(ValueError, match="append targets must belong"):
        _replace_plan_group(plan, 0, **{field: values})


def test_paged_cache_plan_rejects_a_row_owned_append_to_the_wrong_logical_block():
    plan, layout = _mixed_cache_plan(decode=True)
    group = plan.cache_groups[0]
    slabs = group.append_slab_ids.copy()
    # Row 2 owns both physical blocks, but position 14 must use logical block 3.
    slabs[4] = group.block_slab_ids[group.block_indptr[2]]
    mutated = _replace_plan_group(plan, 0, append_slab_ids=slabs)
    with pytest.raises(ValueError, match="cache group 0 row 2 append target"):
        PagedCachePlan.from_plan(mutated, layout, decode_lookahead=3)


def test_paged_cache_plan_reports_the_first_invalid_row_before_later_ranges() -> None:
    plan, layout = _mixed_cache_plan(decode=True)
    group = plan.cache_groups[0]
    offsets = group.append_block_offsets.copy()
    offsets[0] = 1  # Earlier row has an invalid append, later row an invalid extent.
    firsts = group.block_table_start_indices.copy()
    firsts[1] = 1
    mutated = _replace_plan_group(
        plan, 0, append_block_offsets=offsets, block_table_start_indices=firsts
    )
    with pytest.raises(ValueError, match="cache group 0 row 0 append target"):
        PagedCachePlan.from_plan(mutated, layout, decode_lookahead=3)


@pytest.mark.parametrize("earlier_bad_append", [False, True])
def test_paged_cache_plan_preserves_row_order_for_window_write_failures(
    earlier_bad_append: bool,
) -> None:
    plan, layout = _mixed_cache_plan(decode=True)
    layout = CacheLayout(
        parent_capacity=layout.parent_capacity,
        groups=[
            layout.groups[0],
            CacheGroupLayout(
                4,
                blocks_per_slab=3,
                retention=CacheRetention.WINDOW,
                window_tokens=12,
                read_window_tokens=7,
            ),
            layout.groups[2],
        ],
    )
    if earlier_bad_append:
        offsets = plan.cache_groups[1].append_block_offsets.copy()
        offsets[3] = 0
        plan = _replace_plan_group(plan, 1, append_block_offsets=offsets)
    message = (
        "cache group 1 row 1 append target"
        if earlier_bad_append
        else "cache group 1 row 2 write span 8 exceeds 6-token"
    )
    with pytest.raises(ValueError, match=message):
        PagedCachePlan.from_plan(plan, layout, decode_lookahead=3)


def test_paged_cache_plan_skips_discarded_checkpoint_append_targets() -> None:
    plan, layout = _mixed_cache_plan(decode=True)
    group = plan.cache_groups[2]
    offsets = group.append_block_offsets.copy()
    offsets[4] = 0  # Position 14 is older than the retained ring beginning at 16.
    mutated = _replace_plan_group(plan, 2, append_block_offsets=offsets)
    PagedCachePlan.from_plan(mutated, layout, decode_lookahead=3)


def test_native_plan_rejects_empty_cache_rows_before_adapter_indexing() -> None:
    plan, _ = _mixed_cache_plan(decode=True)
    indptr = plan.cache_groups[0].block_indptr.copy()
    indptr[1] = 0
    with pytest.raises(ValueError, match="append targets must belong"):
        _replace_plan_group(plan, 0, block_indptr=indptr)


def test_paged_cache_plan_keeps_decode_leases_wider_than_u32() -> None:
    context = 2**32 - 5
    block_tokens = 2**31 - 1
    plan = BatchPlan(
        sequence_ids=[10],
        context_lengths=[context],
        token_ids=[1, 2, 3],
        token_indptr=[0, 3],
        phase=BatchPhase.DECODE,
        cache_groups=[
            CacheGroupPlan(
                group_index=0,
                block_indptr=[0, 3],
                block_slab_ids=[1, 2, 3],
                block_slotnums=[0, 0, 0],
                block_table_start_indices=[0],
                append_slab_ids=[2, 2, 2],
                append_slotnums=[0, 0, 0],
                append_block_offsets=[
                    block_tokens - 3,
                    block_tokens - 2,
                    block_tokens - 1,
                ],
            )
        ],
    )
    layout = CacheLayout(parent_capacity=3, groups=[CacheGroupLayout(block_tokens)])
    cache = PagedCachePlan.from_plan(plan, layout, decode_lookahead=3)
    assert cache.groups[0].address(0, 2) == (3, 0)
    with pytest.raises(ValueError, match="row 0 is not its exact active range"):
        PagedCachePlan.from_plan(plan, layout)


@pytest.mark.parametrize("lookahead", [2**63, 2**64, 10**100])
def test_paged_cache_plan_does_not_wrap_oversized_decode_lookahead(
    lookahead: int,
) -> None:
    plan, layout = _mixed_cache_plan(decode=True)
    with pytest.raises(ValueError, match="row 0 is not its exact active range"):
        PagedCachePlan.from_plan(plan, layout, decode_lookahead=lookahead)
