from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    CacheRetention,
)
from resin.models.gemma4 import plan as edge_plan
from resin.models.gemma4.plan import (
    PAGE_SIZE,
    SLIDING_WINDOW,
    WINDOW_RETENTION_TOKENS,
    expand_gemma4_decode_batch,
    gemma4_assistant_context,
)
from resin.models.gemma4.plan import (
    build_gemma4_batch_plan as _build_gemma4_batch_plan,
)


def build_gemma4_batch_plan(
    plan,
    *,
    history_blocks_per_parent: int,
    parent_capacity: int,
    decode_lookahead: int = 0,
):
    layout = CacheLayout(
        parent_capacity=parent_capacity,
        groups=[
            CacheGroupLayout(PAGE_SIZE, blocks_per_slab=history_blocks_per_parent),
            CacheGroupLayout(
                PAGE_SIZE,
                retention=CacheRetention.WINDOW,
                window_tokens=WINDOW_RETENTION_TOKENS,
                read_window_tokens=SLIDING_WINDOW,
            ),
        ],
    )
    return _build_gemma4_batch_plan(plan, layout, decode_lookahead=decode_lookahead)


def _append_metadata(pages: list[int], token_count: int) -> tuple[list[int], list[int]]:
    return (
        [pages[position // PAGE_SIZE] for position in range(token_count)],
        [position % PAGE_SIZE for position in range(token_count)],
    )


def _cache_group(
    index: int,
    indptr: list[int],
    compact_pages: list[int],
    first_blocks: list[int],
    compact_append_pages: list[int],
    append_offsets: list[int],
    *,
    packing: int,
) -> CacheGroupPlan:
    def addresses(pages: list[int]) -> tuple[list[int], list[int]]:
        return (
            [(page - 1) // packing + 1 for page in pages],
            [(page - 1) % packing for page in pages],
        )

    parents, child_slots = addresses(compact_pages)
    append_parents, append_slotnums = addresses(compact_append_pages)
    return CacheGroupPlan(
        group_index=index,
        block_indptr=indptr,
        block_slab_ids=parents,
        block_slotnums=child_slots,
        block_table_start_indices=first_blocks,
        append_slab_ids=append_parents,
        append_slotnums=append_slotnums,
        append_block_offsets=append_offsets,
        blocks_per_slab=packing,
    )


def test_gemma4_plan_preserves_positive_ids_after_graph_null_page() -> None:
    tokens = list(range(70))
    history_pages = [11, 12]
    local_pages = [1, 2]
    history_append, offsets = _append_metadata(history_pages, len(tokens))
    local_append, _ = _append_metadata(local_pages, len(tokens))
    plan = BatchPlan(
        sequence_ids=[7],
        context_lengths=[0],
        token_ids=tokens,
        token_indptr=[0, len(tokens)],
        cache_groups=[
            _cache_group(
                0,
                [0, 2],
                history_pages,
                [0],
                history_append,
                offsets,
                packing=10,
            ),
            _cache_group(
                1,
                [0, 2],
                local_pages,
                [0],
                local_append,
                offsets,
                packing=1,
            ),
        ],
    )

    batch = build_gemma4_batch_plan(
        plan, history_blocks_per_parent=10, parent_capacity=256
    )

    assert batch.global_.page_indices.tolist() == [20, 21]
    assert batch.local.page_indices.tolist() == [1, 2]
    assert batch.global_.append_slots[[0, 64, 69]].tolist() == [1280, 1344, 1349]
    assert batch.local.append_slots[[0, 64, 69]].tolist() == [64, 128, 133]
    assert batch.global_.last_page_lengths.tolist() == [6]
    assert batch.last_token_indices.tolist() == [69]


def test_decode_lookahead_expands_verify_but_not_assistant_context() -> None:
    plan = BatchPlan(
        sequence_ids=[9],
        context_lengths=[64],
        token_ids=[123],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
        cache_groups=[
            _cache_group(0, [0, 2], [11, 12], [0], [12], [0], packing=10),
            _cache_group(1, [0, 2], [21, 22], [0], [22], [0], packing=1),
        ],
    )

    batch = build_gemma4_batch_plan(
        plan,
        history_blocks_per_parent=10,
        parent_capacity=256,
        decode_lookahead=3,
    )
    assistant = gemma4_assistant_context(batch)
    verify = expand_gemma4_decode_batch(
        batch,
        np.array([[201, 202, 203]], dtype=np.int64),
    )

    assert assistant.global_.page_indices.tolist() == [20]
    assert assistant.local.page_indices.tolist() == [21]
    assert assistant.global_.last_page_lengths.tolist() == [64]
    assert assistant.global_.append_slots.tolist() == []
    assert verify.token_ids.tolist() == [123, 201, 202, 203]
    assert verify.positions.tolist() == [64, 65, 66, 67]
    assert verify.global_.page_indices.tolist() == [20, 21]
    assert verify.global_.append_slots.tolist() == [1344, 1345, 1346, 1347]
    assert verify.local.append_slots.tolist() == [1408, 1409, 1410, 1411]
    assert verify.global_.last_page_lengths.tolist() == [4]


def test_five_drafts_reserve_the_local_page_beyond_a_full_window() -> None:
    pages = list(range(1, 18))
    plan = BatchPlan(
        sequence_ids=[9],
        context_lengths=[1024],
        token_ids=[123],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
        cache_groups=[
            _cache_group(0, [0, 17], pages, [0], [17], [0], packing=10),
            _cache_group(1, [0, 17], pages, [0], [17], [0], packing=1),
        ],
    )

    batch = build_gemma4_batch_plan(
        plan,
        history_blocks_per_parent=10,
        parent_capacity=256,
        decode_lookahead=5,
    )
    assistant = gemma4_assistant_context(batch)
    verify = expand_gemma4_decode_batch(
        batch, np.array([[201, 202, 203, 204, 205]], dtype=np.int64)
    )

    assert len(assistant.local.page_indices) == 16
    assert assistant.local.last_page_lengths.tolist() == [64]
    assert len(verify.local.page_indices) == 17
    assert verify.local.kv_lengths == (1030,)
    assert verify.local.append_slots.tolist() == [1088, 1089, 1090, 1091, 1092, 1093]


def test_explicit_none_feedback_drains_without_a_continuation_or_batch_id_gap():
    """A reserved runner can decline one continuation without losing its lease."""
    from resin import (
        CacheGroupLayout,
        CacheLayout,
        Engine,
        ForwardOutput,
        RunnerConfig,
        SamplingParams,
    )

    def output(tokens):
        return ForwardOutput(
            token_ids=np.asarray(tokens, dtype=np.uint32),
            token_indptr=np.asarray([0, len(tokens)], dtype=np.uint32),
        )

    class Runner:
        runner_config = RunnerConfig(
            cache_layout=CacheLayout(parent_capacity=16, groups=[CacheGroupLayout(64)]),
            max_accepted_per_step=4,
        )

        def __init__(self):
            self.batch_ids = []
            self.enqueues = 0
            self.harvests = 0

        def bind_resources(self, binding):
            self.binding = binding

        def forward(self, plan):
            self.batch_ids.append(plan.batch_id)
            return output([int(plan.token_ids[-1]) + 1])

        def enqueue(self, plan, *, feedback=None):
            assert feedback is None
            self.batch_ids.append(plan.batch_id)
            self.enqueues += 1
            tokens = [int(plan.token_ids[-1]) + 1, int(plan.token_ids[-1]) + 2]

            def result():
                self.harvests += 1
                return output(tokens)

            return SimpleNamespace(feedback=None, result=result)

    runner = Runner()
    engine = Engine(runner)
    request = engine.submit(
        [1] * 63, sampling_params=SamplingParams(max_output_tokens=14)
    )
    try:
        assert [event.token_id for event in engine.step()] == [2]
        assert engine.step() == []
        assert engine.overlap_pending
        assert not engine.overlap_ready  # Every distributed replica must drain.
        assert [event.token_id for event in engine.step()] == [3, 4]
        assert not engine.overlap_pending
        assert engine.overlap_ready
        assert runner.enqueues == runner.harvests == 1
        assert engine.overlap_stats == (1, 0)
        assert engine.step() == []
        assert runner.batch_ids == [1, 2, 3]
        assert [event.token_id for event in engine.step()] == [5, 6]
        while not request.done:
            engine.step()
        assert request.take_result() == list(range(2, 16))
        assert engine.overlap_stats[1] == 0
    finally:
        engine.close()


def _reserved_plan(initial_positions, *, advance=0, reservation=8):
    from resin import (
        BatchPhase,
        BatchPlan,
        CacheGroupLayout,
        CacheGroupPlan,
        CacheLayout,
        CacheRetention,
    )

    layout = CacheLayout(
        parent_capacity=4096,
        groups=[
            CacheGroupLayout(64),
            CacheGroupLayout(
                64,
                retention=CacheRetention.WINDOW,
                window_tokens=8703,
                read_window_tokens=512,
            ),
        ],
    )
    groups = []
    for group_index, descriptor in enumerate(layout.groups):
        indptr, pages, firsts, appends = [0], [], [], []
        for row, initial in enumerate(initial_positions):
            last = (initial + reservation - 1) // 64
            resident = descriptor.window_resident_block_count
            first = max(0, last + 1 - resident) if resident is not None else 0
            row_pages = list(
                range(
                    1 + row * 256 + group_index * 1024,
                    1 + row * 256 + group_index * 1024 + last - first + 1,
                )
            )
            pages.extend(row_pages)
            indptr.append(len(pages))
            firsts.append(first)
            appends.append(row_pages[(initial + advance) // 64 - first])
        groups.append(
            CacheGroupPlan(
                group_index=group_index,
                blocks_per_slab=1,
                block_indptr=indptr,
                block_slab_ids=pages,
                block_slotnums=[0] * len(pages),
                block_table_start_indices=firsts,
                append_slab_ids=appends,
                append_slotnums=[0] * len(appends),
                append_block_offsets=[(p + advance) % 64 for p in initial_positions],
            )
        )
    return BatchPlan(
        sequence_ids=list(range(1, len(initial_positions) + 1)),
        context_lengths=[p + advance for p in initial_positions],
        token_ids=[2] * len(initial_positions),
        token_indptr=list(range(len(initial_positions) + 1)),
        phase=BatchPhase.DECODE,
        cache_groups=groups,
    ), layout


def test_reserved_envelopes_crop_unwritten_future_pages_and_keep_owner_tables():
    for initial in (58, 63, 64, 65, 511, 512, 513, 8191, 8192, 8193, 8703, 8704, 8705):
        for advance in (0, 1, 4):
            native, layout = _reserved_plan([initial], advance=advance)
            batch = edge_plan.build_gemma4_batch_plan(
                native, layout, decode_lookahead=3
            )
            assert (
                batch.global_envelope is not None and batch.local_envelope is not None
            )
            anchor = initial + advance
            for readable, envelope in (
                (batch.local, batch.local_envelope),
                (batch.global_, batch.global_envelope),
            ):
                assert (
                    readable.kv_lengths[0]
                    == anchor + 4 - readable.first_token_positions[0]
                )
                expected = (
                    (anchor + 3) // 64 - readable.first_token_positions[0] // 64 + 1
                )
                assert readable.kv_indptr[-1] == expected
                assert (
                    readable.page_indices.tolist()
                    == envelope.page_indices[:expected].tolist()
                )
                assert readable.last_page_lengths[0] == (anchor + 3) % 64 + 1
            assistant = gemma4_assistant_context(batch, local_window_tokens=512)
            verify = expand_gemma4_decode_batch(batch, np.zeros((1, 3), dtype=np.int64))
            assert assistant.global_.kv_lengths == (anchor,)
            assert verify.global_.kv_lengths == (anchor + 4,)
            pages = batch.global_envelope.page_indices
            assert verify.global_.append_slots.tolist() == [
                int(pages[p // 64]) * 64 + p % 64 for p in range(anchor, anchor + 4)
            ]


def test_reservation_validator_rejects_insufficient_span():
    import pytest

    native, layout = _reserved_plan([58], advance=69)
    with pytest.raises(ValueError, match="invalid active range"):
        edge_plan.build_gemma4_batch_plan(native, layout, decode_lookahead=3)


def test_native_reservation_preserves_exact_capacity_final_short_output():
    """Optional pipeline room cannot make an ordinary final step fail to fit."""
    from resin import (
        BatchPhase,
        CacheGroupLayout,
        CacheLayout,
        Engine,
        ForwardOutput,
        RunnerConfig,
        SamplingParams,
    )

    class Runner:
        runner_config = RunnerConfig(
            cache_layout=CacheLayout(parent_capacity=16, groups=[CacheGroupLayout(4)]),
            max_accepted_per_step=4,
        )

        def bind_resources(self, binding):
            self.binding = binding

        def forward(self, plan):
            tokens = [2] if plan.phase == BatchPhase.PREFILL else [3, 4, 5]
            return ForwardOutput(
                token_ids=np.asarray(tokens, dtype=np.uint32),
                token_indptr=np.asarray([0, len(tokens)], dtype=np.uint32),
            )

    engine = Engine(Runner(), context_length=64, max_running_requests=1)
    request = engine.submit(
        [1] * 60, sampling_params=SamplingParams(max_output_tokens=4)
    )
    try:
        while engine.has_work():
            engine.step()
        assert request.take_result() == [2, 3, 4, 5]
        assert engine.cache_free_parent_count == 16
    finally:
        engine.close()


def test_large_reservation_keeps_unwritten_tail_out_of_readable_attention():

    native, layout = _reserved_plan([58], reservation=72)
    batch = edge_plan.build_gemma4_batch_plan(native, layout, decode_lookahead=3)
    assert batch.global_envelope is not None
    assert batch.global_envelope.kv_indptr.tolist() == [0, 3]
    assert batch.global_.kv_indptr.tolist() == [0, 1]
    assert batch.global_.kv_lengths == (62,)
    assert batch.global_.last_page_lengths.tolist() == [62]


def test_small_write_guard_allows_unused_page_padding():
    from resin import (
        BatchPhase,
        CacheGroupLayout,
        CacheLayout,
        CacheRetention,
        Engine,
        ForwardOutput,
        RunnerConfig,
        SamplingParams,
    )

    class Runner:
        runner_config = RunnerConfig(
            prefill_chunk_size=72,
            continuation_prefill=True,
            max_accepted_per_step=4,
            cache_layout=CacheLayout(
                parent_capacity=256,
                groups=[
                    CacheGroupLayout(64),
                    CacheGroupLayout(
                        64,
                        retention=CacheRetention.WINDOW,
                        window_tokens=583,
                        read_window_tokens=512,
                    ),
                ],
            ),
        )

        def bind_resources(self, binding):
            pass

        def forward(self, plan):
            edge_plan.build_gemma4_batch_plan(
                plan,
                self.runner_config.cache_layout,
                decode_lookahead=3 if plan.phase == BatchPhase.DECODE else 0,
                prefill_chunk_size=72,
            )
            return ForwardOutput(
                token_ids=np.asarray([2], dtype=np.uint32),
                token_indptr=np.asarray([0, 1], dtype=np.uint32),
            )

    engine = Engine(Runner(), context_length=4096, max_running_requests=1)
    request = engine.submit(
        [1] * 63, sampling_params=SamplingParams(max_output_tokens=40)
    )
    try:
        while engine.has_work():
            engine.step()
        assert request.take_result() == [2] * 40
    finally:
        engine.close()


def test_small_write_guard_rejects_actual_verification_overrun():
    from resin import CacheGroupLayout, CacheLayout, CacheRetention

    native, original = _reserved_plan([58])
    layout = CacheLayout(
        parent_capacity=original.parent_capacity,
        groups=[
            CacheGroupLayout(64),
            CacheGroupLayout(
                64,
                retention=CacheRetention.WINDOW,
                window_tokens=514,
                read_window_tokens=512,
            ),
        ],
    )
    with pytest.raises(ValueError, match="Window read/write capacity"):
        edge_plan.build_gemma4_batch_plan(native, layout, decode_lookahead=3)


def _assistant_window_batch(module, positions):
    from resin import BatchPhase

    rows = len(positions)
    query_indptr = np.arange(rows + 1, dtype=np.int32)

    def pages(local):
        indices, counts, firsts, lengths, lasts = [], [], [], [], []
        for row, position in enumerate(positions):
            end = position + 4
            last_block = (end - 1) // 64
            first_block = max(0, last_block + 1 - 137) if local else 0
            count = last_block - first_block + 1
            # Distinct reordered/recycled physical IDs; logical order is only
            # represented by the CSR view, never by arithmetic on page IDs.
            indices.extend(
                (10000 * (row + 1) + (np.arange(count) * 37) % 1009).tolist()
            )
            counts.append(count)
            firsts.append(first_block * 64)
            lengths.append(end - first_block * 64)
            lasts.append((end - 1) % 64 + 1)
        indptr = np.zeros(rows + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(counts)
        return module.PagedAttentionPlan(
            qo_indptr=query_indptr,
            kv_indptr=indptr,
            page_indices=np.asarray(indices, dtype=np.int32),
            last_page_lengths=np.asarray(lasts, dtype=np.int32),
            append_slots=np.arange(rows, dtype=np.int64),
            first_token_positions=tuple(firsts),
            kv_lengths=tuple(lengths),
        )

    return module.Gemma4BatchPlan(
        phase=BatchPhase.DECODE,
        token_ids=np.arange(rows, dtype=np.int64) + 100,
        positions=np.asarray(positions, dtype=np.int64),
        token_indptr=query_indptr,
        local=pages(True),
        global_=pages(False),
        last_token_indices=np.arange(rows, dtype=np.int64),
    )


@pytest.mark.parametrize(
    "position",
    [1, 63, 64, 65, 511, 512, 513, 8191, 8192, 8193, 8703, 8704, 8705, 65536],
)
def test_assistant_window_has_identical_contributing_keys_and_unchanged_anchor(
    position,
):
    batch = _assistant_window_batch(edge_plan, [position])
    original_indices = batch.local.page_indices.copy()
    full = edge_plan.gemma4_assistant_context(batch)
    trimmed = edge_plan.gemma4_assistant_context(batch, local_window_tokens=512)
    assert trimmed.positions is batch.positions
    assert trimmed.token_ids is batch.token_ids
    assert trimmed.token_indptr is batch.token_indptr
    np.testing.assert_array_equal(batch.local.page_indices, original_indices)
    np.testing.assert_array_equal(
        trimmed.local.last_page_lengths, full.local.last_page_lengths
    )
    np.testing.assert_array_equal(
        trimmed.global_.page_indices, full.global_.page_indices
    )
    np.testing.assert_array_equal(trimmed.global_.kv_indptr, full.global_.kv_indptr)
    assert trimmed.global_.kv_lengths == full.global_.kv_lengths
    assert trimmed.global_.first_token_positions == full.global_.first_token_positions
    assert trimmed.local.append_slots.size == 0
    first = trimmed.local.first_token_positions[0]
    assert first == max(
        full.local.first_token_positions[0], max(0, position - 512) // 64 * 64
    )
    assert trimmed.local.kv_lengths == (position - first,)
    assert min(position, 512) <= position - first <= 575
    actual_allowed = [p for p in range(first, position) if p + 512 >= position]
    assert actual_allowed == list(range(max(0, position - 512), position))
    skip = (first - full.local.first_token_positions[0]) // 64
    np.testing.assert_array_equal(
        trimmed.local.page_indices, full.local.page_indices[skip:]
    )


def test_unequal_rows_have_independent_suffix_views():
    positions = [511, 512, 513, 8191, 8192, 8193, 8703, 8704, 8705]
    batch = _assistant_window_batch(edge_plan, positions)
    all_rows = edge_plan.gemma4_assistant_context(batch, local_window_tokens=512).local
    for row, position in enumerate(positions):
        first = max(0, position - 512) // 64 * 64
        assert all_rows.first_token_positions[row] == first
        assert all_rows.kv_lengths[row] == position - first
        start, end = all_rows.kv_indptr[row : row + 2]
        assert end - start == (position - first + 63) // 64
        assert np.all((all_rows.page_indices[start:end] // 10000) == row + 1)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_invalid_windows_fail_closed(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        edge_plan.gemma4_assistant_context(
            _assistant_window_batch(edge_plan, [8192]), local_window_tokens=invalid
        )
