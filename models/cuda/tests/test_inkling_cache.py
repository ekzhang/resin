from __future__ import annotations

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupFamily,
    CacheGroupPlan,
    CacheRetention,
    CacheStatePlan,
)
from resin.backend.plans import PagedCachePlan
from resin_models_cuda.inkling.cache import (
    DRAFT_BLOCKS_PER_SLAB,
    GLOBAL_BLOCKS_PER_SLAB,
    LOCAL_BLOCKS_PER_SLAB,
    LOCAL_RETENTION_TOKENS,
    SLAB_UNITS,
    STATE_BLOCKS_PER_SLAB,
    CacheArena,
    CacheGeometry,
)


def _group(
    index: int,
    pages: list[tuple[int, int]],
    append: list[tuple[int, int]],
    offsets: list[int],
    *,
    packing: int,
) -> CacheGroupPlan:
    return CacheGroupPlan(
        group_index=index,
        block_indptr=[0, len(pages)],
        block_slab_ids=[parent for parent, _ in pages],
        block_slotnums=[slot for _, slot in pages],
        block_table_start_indices=[0],
        append_slab_ids=[parent for parent, _ in append],
        append_slotnums=[slot for _, slot in append],
        append_block_offsets=[0] * len(offsets) if index == 3 else offsets,
        blocks_per_slab=packing,
        state=(CacheStatePlan() if index == 3 else None),
    )


def test_small_slab_packings_bound_fragmentation() -> None:
    geometry = CacheGeometry()
    assert geometry.bytes_per_slab == 35 * 1024 * 1024
    assert (
        geometry.local_blocks_per_slab,
        geometry.global_blocks_per_slab,
        geometry.draft_blocks_per_slab,
        geometry.state_blocks_per_slab,
    ) == (2, 10, 11, 14)
    layout = geometry.layout(100)
    assert layout.groups[1].retention is CacheRetention.WINDOW
    assert layout.groups[1].window_tokens == LOCAL_RETENTION_TOKENS
    assert layout.groups[3].family is CacheGroupFamily.STATE


def test_layer_page_ids_use_slab_units_not_combined_block_ids() -> None:
    assert CacheArena.unit_page(3, 1, 35, 7) == 3 * SLAB_UNITS + 35 + 7
    assert CacheArena.unit_page(3, 9, 7, 6) == 3 * SLAB_UNITS + 9 * 7 + 6
    assert CacheArena.unit_page(3, 10, 6, 5) == 3 * SLAB_UNITS + 10 * 6 + 5


def test_decode_lookahead_leases_the_second_attention_block() -> None:
    layout = CacheGeometry().layout(20)
    plan = BatchPlan(
        sequence_ids=[9],
        context_lengths=[127],
        token_ids=[42],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
        cache_groups=[
            _group(
                0,
                [(1, 0), (1, 1)],
                [(1, 0)],
                [127],
                packing=GLOBAL_BLOCKS_PER_SLAB,
            ),
            _group(
                1,
                [(2, 0), (2, 1)],
                [(2, 0)],
                [127],
                packing=LOCAL_BLOCKS_PER_SLAB,
            ),
            _group(
                2,
                [(3, 0), (3, 1)],
                [(3, 0)],
                [127],
                packing=DRAFT_BLOCKS_PER_SLAB,
            ),
            _group(
                3,
                [(4, 0)],
                [(4, 0)],
                [127],
                packing=STATE_BLOCKS_PER_SLAB,
            ),
        ],
    )
    cache = PagedCachePlan.from_plan(plan, layout, decode_lookahead=5)
    for index, group in enumerate(cache.groups):
        assert group.block_slabs(0).tolist() == [index + 1] * (2 if index < 3 else 1)
        assert group.block_slots(0).tolist() == ([0, 1] if index < 3 else [0])
