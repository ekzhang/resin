from __future__ import annotations

import numpy as np
import pytest
from output_helpers import forward_output

import resin
from resin import (
    BatchPlan,
    CacheGroupFamily,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    CacheRetention,
    Engine,
    ForwardOutput,
    MediaTokenSpan,
    ResourceBinding,
    RunnerConfig,
    SamplingParams,
    SchedulerConfig,
)


class RecordingCachedRunner:
    def __init__(
        self,
        layout: CacheLayout,
        *,
        forward_failures: int = 0,
    ) -> None:
        self.cache_layout = layout
        self.runner_config = RunnerConfig(
            continuation_prefill=True,
            cache_layout=layout,
        )
        self.forward_failures = forward_failures
        self.bindings: list[tuple[int, CacheLayout]] = []
        self.plans: list[BatchPlan] = []
        self.released: list[int] = []

    def bind_resources(self, binding: ResourceBinding) -> None:
        assert binding.cache_pool_id is not None
        assert binding.cache_layout is not None
        self.bindings.append((binding.cache_pool_id, binding.cache_layout))

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self.plans.append(plan)
        if self.forward_failures:
            self.forward_failures -= 1
            raise RuntimeError("runner rejected the batch")
        return forward_output(
            [
                plan.token_ids[end - 1] + 1
                for _, end in map(plan.token_range, range(plan.num_sequences))
            ]
        )

    def release(self, request_ids: list[int]) -> None:
        self.released.extend(request_ids)


def _block_addresses(group: CacheGroupPlan) -> list[tuple[int, int]]:
    return list(zip(group.block_slab_ids, group.block_slotnums, strict=True))


def _append_addresses(group: CacheGroupPlan) -> list[tuple[int, int]]:
    return list(zip(group.append_slab_ids, group.append_slotnums, strict=True))


class TestFlatCacheApi:
    def test_flat_cache_descriptors_are_exported(self) -> None:
        for name in ("CacheGroupLayout", "CacheLayout", "CacheGroupPlan"):
            assert name in resin.__all__
            assert hasattr(resin, name)

    def test_layout_is_frozen_ordered_and_derives_kernel_geometry(self) -> None:
        coarse = CacheGroupLayout(4)
        fine = CacheGroupLayout(2, blocks_per_slab=2)
        source = [coarse, fine]
        layout = CacheLayout(parent_capacity=3, groups=source)
        source.reverse()

        assert [
            (group.block_tokens, group.blocks_per_slab) for group in layout.groups
        ] == [(4, 1), (2, 2)]
        assert layout.group_count == 2
        assert layout.group_kernel_page_capacity(0) == 3
        assert layout.group_kernel_page_count(0) == 4
        assert layout.group_kernel_page_capacity(1) == 6
        assert layout.group_kernel_page_count(1) == 7
        with pytest.raises(AttributeError):
            setattr(layout, "parent_capacity", 4)

    def test_cache_retention_is_only_history_or_window(self) -> None:
        history = CacheGroupLayout(4)
        window = CacheGroupLayout(4, retention=CacheRetention.WINDOW, window_tokens=4)

        assert history.retention is CacheRetention.HISTORY
        assert history.window_tokens is None
        assert window.retention is CacheRetention.WINDOW
        assert window.window_tokens == 4
        assert window.read_window_tokens == 4
        assert history.read_window_tokens is None
        with pytest.raises(ValueError, match="read_window_tokens"):
            CacheGroupLayout(4, read_window_tokens=1)
        for read_window in (0, 5):
            with pytest.raises(ValueError, match="read_window_tokens"):
                CacheGroupLayout(
                    4,
                    retention=CacheRetention.WINDOW,
                    window_tokens=4,
                    read_window_tokens=read_window,
                )
        with pytest.raises(AttributeError):
            # Removed API must fail at runtime.
            _ = history.resident_blocks  # ty: ignore[unresolved-attribute]
        with pytest.raises(TypeError, match="resident_blocks"):
            # Removed constructor argument must fail at runtime.
            CacheGroupLayout(4, resident_blocks=1)  # ty: ignore[unknown-argument]
        with pytest.raises(TypeError):
            # Invalid retention type must fail at runtime.
            CacheGroupLayout(4, retention="state")  # ty: ignore[invalid-argument-type]

    def test_layout_estimates_packed_retained_cohort_without_allocating(self) -> None:
        layout = CacheLayout(
            parent_capacity=1,
            groups=[
                CacheGroupLayout(4, blocks_per_slab=2),
                CacheGroupLayout(
                    4,
                    blocks_per_slab=4,
                    retention=CacheRetention.WINDOW,
                    window_tokens=5,
                    read_window_tokens=2,
                ),
                CacheGroupLayout(8, blocks_per_slab=2, family=CacheGroupFamily.STATE),
            ],
        )
        assert layout.estimated_parent_count([]) == 0
        assert layout.estimated_parent_count([0, 0]) == 0
        assert layout.estimated_parent_count([0, 1, 4, 5, 17]) == 9
        assert layout.estimated_parent_count((0, 1, 4, 5, 17)) == 9
        assert (
            layout.estimated_parent_count([24, 24, 24], shared_prefix_tokens=16) == 9
        )
        with pytest.raises(ValueError, match="align"):
            layout.estimated_parent_count([24, 24], shared_prefix_tokens=15)
        with pytest.raises(ValueError, match="exceeds"):
            layout.estimated_parent_count([12, 24], shared_prefix_tokens=16)
        assert (
            layout.parent_capacity == 1
        )  # Demand is reported even when it cannot fit.
        for invalid in (-1, 2**32):
            with pytest.raises(OverflowError):
                layout.estimated_parent_count([invalid])

    def test_manual_sparse_group_plan_is_the_only_batch_cache_metadata(self) -> None:
        group = CacheGroupPlan(
            group_index=0,
            block_indptr=[0, 2],
            block_slab_ids=[3, 4],
            block_slotnums=[0, 0],
            block_table_start_indices=[0],
            append_slab_ids=[3, 4],
            append_slotnums=[0, 0],
            append_block_offsets=[1, 0],
        )
        plan = BatchPlan(
            sequence_ids=[11],
            context_lengths=[1],
            token_ids=[8, 9],
            token_indptr=[0, 2],
            cache_groups=[group],
        )

        assert len(plan.cache_groups) == 1
        frozen = plan.cache_groups[0]
        assert frozen.group_index == group.group_index
        assert frozen.block_indptr.tolist() == [0, 2]
        assert frozen.block_slab_ids.tolist() == [3, 4]
        assert frozen.block_slotnums.tolist() == [0, 0]
        assert frozen.block_table_start_indices.tolist() == [0]
        assert frozen.append_slab_ids.tolist() == [3, 4]
        assert frozen.append_slotnums.tolist() == [0, 0]
        assert frozen.append_block_offsets.tolist() == [1, 0]

    def test_cache_group_numeric_metadata_is_read_only_zero_copy_numpy(self) -> None:
        group = CacheGroupPlan(
            group_index=0,
            block_indptr=np.asarray([0, 2], dtype=np.uint32),
            block_slab_ids=np.asarray([3, 4], dtype=np.uint32),
            block_slotnums=np.asarray([0, 0], dtype=np.uint32),
            block_table_start_indices=np.asarray([0], dtype=np.uint32),
            append_slab_ids=np.asarray([3, 4], dtype=np.uint32),
            append_slotnums=np.asarray([0, 0], dtype=np.uint32),
            append_block_offsets=np.asarray([1, 0], dtype=np.uint32),
        )
        for name in (
            "block_indptr",
            "block_slab_ids",
            "block_slotnums",
            "block_table_start_indices",
            "append_slab_ids",
            "append_slotnums",
            "append_block_offsets",
        ):
            array = getattr(group, name)
            assert array.dtype == np.uint32
            assert array.base is group
            assert not array.flags.writeable
            assert np.shares_memory(array, getattr(group, name))
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0

    def test_binding_requires_an_exact_layout_callable_and_pool_ack(self) -> None:
        layout = CacheLayout(parent_capacity=2, groups=[CacheGroupLayout(2)])

        class MissingBinder:
            runner_config = RunnerConfig(cache_layout=layout)

            @staticmethod
            def forward(plan: BatchPlan) -> ForwardOutput:
                return forward_output([0] * plan.num_sequences)

        with pytest.raises(TypeError, match="bind_resources"):
            Engine(MissingBinder())

        class BadLayout(MissingBinder):
            runner_config = {"cache_layout": {"parent_capacity": 2}}

            @staticmethod
            def bind_resources(binding: object) -> None:
                del binding

        with pytest.raises(TypeError, match="exactly a RunnerConfig"):
            Engine(BadLayout())

        class BadAck(MissingBinder):
            def __init__(self, acknowledgement: object) -> None:
                self.acknowledgement = acknowledgement

            def bind_resources(self, binding: ResourceBinding):
                del binding
                return self.acknowledgement

        with pytest.raises(TypeError, match="must return None"):
            Engine(BadAck(True))
        with pytest.raises(TypeError, match="must return None"):
            Engine(BadAck(999_999))

        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        assert engine.cache_pool_id is not None
        assert engine.cache_pool_id > 0
        assert runner.bindings == [(engine.cache_pool_id, layout)]
        assert runner.bindings[0][1] is layout

    def test_explicit_none_layout_is_unmanaged_and_ignores_stray_binder(self) -> None:
        class UnmanagedRunner:
            def __init__(self) -> None:
                self.bind_calls = 0

            def bind_resources(self, binding: object) -> None:
                del binding
                self.bind_calls += 1
                raise AssertionError("an unmanaged runner must not bind a cache pool")

            @staticmethod
            def forward(plan: BatchPlan) -> ForwardOutput:
                return forward_output([0] * plan.num_sequences)

        runner = UnmanagedRunner()
        engine = Engine(runner)
        assert engine.cache_pool_id is None
        assert runner.bind_calls == 0
        engine.close()


class TestFlatCacheEngine:
    def test_opt_in_lru_prefix_cache_skips_the_shared_prefill(self) -> None:
        layout = CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(2)])
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner, prefix_cache=True)

        first = engine.submit(
            [1, 2, 3, 4], sampling_params=SamplingParams(max_output_tokens=1)
        )
        while engine.has_work():
            engine.step()
        assert first.done
        assert first.cached_tokens == 0
        assert engine.prefix_cache_stats == (2, 0, 1, 0)
        assert engine.cache_occupied_block_count == 2

        second = engine.submit(
            [1, 2, 3, 4, 5], sampling_params=SamplingParams(max_output_tokens=1)
        )
        assert second.cached_tokens == 4
        assert engine.prefix_cache_stats == (2, 1, 1, 0)
        engine.step()
        reused = runner.plans[-1]
        assert reused.token_ids.tolist() == [5]
        assert reused.context_lengths.tolist() == [4]
        assert second.done

        engine.clear_prefix_cache()
        assert engine.prefix_cache_stats == (0, 1, 1, 2)
        assert engine.cache_occupied_block_count == 0

    def test_media_digest_identity_controls_prefix_reuse(self) -> None:
        class Media:
            token_ids = [1, 2, 3, 4]
            token_spans = ()

            def __init__(self, identity: list[int]) -> None:
                self.cache_key = tuple(identity)

        layout = CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(2)])
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner, prefix_cache=True)

        first = Media([1, 101, 102, 4])
        engine.submit(media=first, sampling_params=SamplingParams(max_output_tokens=1))
        while engine.has_work():
            engine.step()

        same_media = Media([1, 101, 102, 4])
        engine.submit(
            media=same_media, sampling_params=SamplingParams(max_output_tokens=1)
        )
        while engine.has_work():
            engine.step()
        assert runner.plans[-1].context_lengths.tolist() == [2]
        assert runner.plans[-1].media == (same_media,)

        changed_media = Media([1, 201, 202, 4])
        engine.submit(
            media=changed_media, sampling_params=SamplingParams(max_output_tokens=1)
        )
        engine.step()
        assert runner.plans[-1].context_lengths.tolist() == [0]

    def test_media_atomic_prefill_span_is_not_cut_at_cache_alignment(self) -> None:
        class Media:
            token_ids = [1, 2, 3, 4, 5, 6]
            cache_key = (1, 20, 30, 40, 50, 6)
            token_spans = (MediaTokenSpan(1, 4),)

        runner = RecordingCachedRunner(
            CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(4)])
        )
        engine = Engine(runner, prefix_cache=True)

        engine.submit(
            media=Media(), sampling_params=SamplingParams(max_output_tokens=1)
        )
        engine.step()

        assert runner.plans[0].token_ids.tolist() == [1, 2, 3, 4, 5]
        assert runner.plans[0].context_lengths.tolist() == [0]

    def test_prefix_cache_is_disabled_by_default(self) -> None:
        runner = RecordingCachedRunner(
            CacheLayout(parent_capacity=2, groups=[CacheGroupLayout(2)])
        )
        engine = Engine(runner)
        engine.submit([1, 2], sampling_params=SamplingParams(max_output_tokens=1))
        engine.step()
        assert engine.prefix_cache_stats is None
        assert engine.cache_occupied_block_count == 0

    def test_one_history_group_uses_positive_sparse_physical_addresses(self) -> None:
        layout = CacheLayout(parent_capacity=3, groups=[CacheGroupLayout(2)])
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        request = engine.submit(
            [4, 5, 6], sampling_params=SamplingParams(max_output_tokens=2)
        )

        assert engine.cache_free_parent_count == 3
        assert engine.cache_occupied_block_count == 0
        first_event = engine.step()[0]
        assert engine.cache_free_parent_count == 1
        assert engine.cache_occupied_block_count == 2
        assert not first_event.finished
        prefill = runner.plans[0]
        assert len(prefill.cache_groups) == 1
        group = prefill.cache_groups[0]
        assert group.group_index == 0
        assert group.block_indptr.tolist() == [0, 2]
        assert group.block_table_start_indices.tolist() == [0]
        assert len(group.block_slab_ids) == 2
        assert all(parent > 0 for parent in group.block_slab_ids)
        assert group.block_slotnums.tolist() == [0, 0]
        assert _append_addresses(group) == [
            _block_addresses(group)[0],
            _block_addresses(group)[0],
            _block_addresses(group)[1],
        ]
        assert group.append_block_offsets.tolist() == [0, 1, 0]

        final_event = engine.step()[0]
        assert final_event.finished
        decode_group = runner.plans[1].cache_groups[0]
        assert _block_addresses(decode_group) == _block_addresses(group)
        assert _append_addresses(decode_group) == [_block_addresses(group)[1]]
        assert decode_group.append_block_offsets.tolist() == [1]
        assert runner.released == [request]
        assert engine.cache_free_parent_count == 3
        assert engine.cache_occupied_block_count == 0

    def test_ordered_groups_share_one_parent_pool_with_group_specific_ids(self) -> None:
        layout = CacheLayout(
            parent_capacity=3,
            groups=[
                CacheGroupLayout(4),
                CacheGroupLayout(2, blocks_per_slab=2),
            ],
        )
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        engine.submit([1, 2, 3], sampling_params=SamplingParams(max_output_tokens=2))

        assert engine.cache_free_parent_count == 3
        assert engine.cache_occupied_block_count == 0
        engine.step()
        plan = runner.plans[0]
        assert [group.group_index for group in plan.cache_groups] == [0, 1]

        coarse, fine = plan.cache_groups
        assert coarse.block_indptr.tolist() == [0, 1]
        assert coarse.block_table_start_indices.tolist() == [0]
        assert coarse.append_block_offsets.tolist() == [0, 1, 2]
        assert _append_addresses(coarse) == [_block_addresses(coarse)[0]] * 3

        assert fine.block_indptr.tolist() == [0, 2]
        assert fine.block_table_start_indices.tolist() == [0]
        assert fine.append_block_offsets.tolist() == [0, 1, 0]
        assert _append_addresses(fine) == [
            _block_addresses(fine)[0],
            _block_addresses(fine)[0],
            _block_addresses(fine)[1],
        ]

        engine.step()
        assert engine.cache_free_parent_count == 3
        assert engine.cache_occupied_block_count == 0

    def test_preaccept_rejection_keeps_pages_owned_and_retry_uses_same_ids(
        self,
    ) -> None:
        layout = CacheLayout(parent_capacity=1, groups=[CacheGroupLayout(2)])
        runner = RecordingCachedRunner(layout, forward_failures=1)
        engine = Engine(runner)
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=1)
        )

        with pytest.raises(RuntimeError, match="rejected the batch"):
            engine.step()
        assert not engine.poisoned
        assert engine.ready_request_count == 1
        assert engine.cache_free_parent_count == 0
        assert engine.cache_occupied_block_count == 1

        events = engine.step()
        assert [event.request_id for event in events] == [request]
        assert _block_addresses(runner.plans[0].cache_groups[0]) == _block_addresses(
            runner.plans[1].cache_groups[0]
        )
        assert engine.cache_free_parent_count == 1
        assert engine.cache_occupied_block_count == 0

    def test_window_group_exposes_a_chronological_suffix_without_copy_metadata(
        self,
    ) -> None:
        layout = CacheLayout(
            parent_capacity=4,
            groups=[
                CacheGroupLayout(
                    4,
                    retention=CacheRetention.WINDOW,
                    window_tokens=12,
                    read_window_tokens=1,
                )
            ],
        )
        runner = RecordingCachedRunner(layout)
        engine = Engine(runner)
        engine.submit(
            list(range(12)), sampling_params=SamplingParams(max_output_tokens=1)
        )

        events = engine.step()
        assert len(events) == 1
        prefill = runner.plans[-1].cache_groups[0]
        assert prefill.block_table_start_indices.tolist() == [0]
        assert len(prefill.block_slab_ids) == 3
        assert _append_addresses(prefill) == [
            address for address in _block_addresses(prefill) for _ in range(4)
        ]
        assert engine.cache_occupied_block_count == 0
        assert engine.cache_free_parent_count == 4


@pytest.mark.parametrize("prefix_cache", [False, True])
@pytest.mark.parametrize("write_tokens", [1, 6])
def test_window_attention_reads_survive_prefill_writes_and_prefix_reuse(
    prefix_cache: bool, write_tokens: int
) -> None:
    window = 4
    block_tokens = 2
    layout = CacheLayout(
        parent_capacity=40,
        groups=[
            CacheGroupLayout(
                block_tokens,
                retention=CacheRetention.WINDOW,
                window_tokens=window + write_tokens - 1,
                read_window_tokens=window,
            )
        ],
    )

    class WindowRunner(RecordingCachedRunner):
        def __init__(self) -> None:
            super().__init__(layout)
            self.storage: dict[tuple[int, int, int], int] = {}
            self.query_count = 0

        def forward(self, plan: BatchPlan) -> ForwardOutput:
            self.plans.append(plan)
            group = plan.cache_groups[0]
            # Device kernels commonly append the entire batch before reading
            # any attention row. This catches both aliasing and missing COW
            # history, which metadata-only shape assertions cannot detect.
            for slab, slot, offset, token in zip(
                group.append_slab_ids,
                group.append_slotnums,
                group.append_block_offsets,
                plan.token_ids,
                strict=True,
            ):
                self.storage[int(slab), int(slot), int(offset)] = int(token)
            outputs = []
            for row in range(plan.num_sequences):
                start, end = plan.token_range(row)
                assert end - start <= write_tokens
                first = int(group.block_table_start_indices[row])
                table_start = int(group.block_indptr[row])
                table_end = int(group.block_indptr[row + 1])
                for query in plan.positions[start:end]:
                    for position in range(
                        max(0, int(query) - window + 1), int(query) + 1
                    ):
                        column = table_start + position // block_tokens - first
                        assert table_start <= column < table_end
                        address = (
                            int(group.block_slab_ids[column]),
                            int(group.block_slotnums[column]),
                            position % block_tokens,
                        )
                        assert self.storage[address] == position
                    self.query_count += 1
                outputs.append(int(plan.token_ids[end - 1]) + 1)
            return forward_output(outputs)

    runner = WindowRunner()
    with Engine(
        runner,
        context_length=64,
        prefix_cache=prefix_cache,
        scheduler_config=SchedulerConfig(prefill_chunk_size=6),
    ) as engine:
        first = engine.submit(
            list(range(13)), sampling_params=SamplingParams(max_output_tokens=1)
        )
        while engine.has_work():
            engine.step()
        assert first.take_result() == [13]
        if write_tokens == 6:
            assert runner.plans[0].num_tokens == 6
            assert runner.plans[1].num_tokens == 6
        second = engine.submit(
            list(range(19)), sampling_params=SamplingParams(max_output_tokens=1)
        )
        assert second.cached_tokens == (12 if prefix_cache else 0)
        while engine.has_work():
            engine.step()
        assert second.take_result() == [19]
        assert runner.query_count == 13 + 19 - second.cached_tokens
