from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from output_helpers import forward_output

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
    Engine,
    ForwardOutput,
    ResourceBinding,
    RunnerConfig,
    SamplingParams,
    SchedulerConfig,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.plans: list[BatchPlan] = []
        self.released: list[int] = []

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self.plans.append(plan)
        return forward_output(
            [
                plan.token_ids[end - 1] + 1
                for _, end in map(plan.token_range, range(plan.num_sequences))
            ]
        )

    def release(self, request_ids: list[int]) -> None:
        self.released.extend(request_ids)


class ContinuationRecordingRunner(RecordingRunner):
    runner_config = RunnerConfig(continuation_prefill=True)


class ControlledRunner:
    def __init__(
        self,
        results: list[list[int]] | None = None,
        *,
        forward_failures: int = 0,
    ) -> None:
        self.results = list(results or [])
        self.forward_failures = forward_failures
        self.plans: list[BatchPlan] = []
        self.released: list[int] = []

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self.plans.append(plan)
        if self.forward_failures:
            self.forward_failures -= 1
            raise RuntimeError("forward rejected")
        return forward_output(self.results.pop(0))

    def release(self, request_ids: list[int]) -> None:
        self.released.extend(request_ids)


class ContinuationControlledRunner(ControlledRunner):
    runner_config = RunnerConfig(continuation_prefill=True)


class TestSchedulerConfigContract:
    def test_runner_prefill_limit_is_an_execution_bound(self) -> None:
        class LimitedRunner(ContinuationRecordingRunner):
            runner_config = RunnerConfig(
                continuation_prefill=True,
                prefill_chunk_size=2,
            )

        runner = LimitedRunner()
        engine = Engine(runner)
        request = engine.submit(
            [1, 2, 3, 4, 5], sampling_params=SamplingParams(max_output_tokens=1)
        )

        engine.step()

        assert runner.plans[0].token_indptr.tolist() == [0, 2]
        assert not request.done
        engine.close()

    def test_runner_batch_limit_is_consumed_in_request_order(self) -> None:
        class LimitedRunner(ContinuationRecordingRunner):
            runner_config = RunnerConfig(
                continuation_prefill=True,
                prefill_chunk_size=8192,
                max_batch_tokens=4,
            )

        runner = LimitedRunner()
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit(
            [1, 2, 3, 4], sampling_params=SamplingParams(max_output_tokens=1)
        )
        second = engine.submit(
            [5, 6, 7, 8], sampling_params=SamplingParams(max_output_tokens=1)
        )

        engine.step()

        assert runner.plans[0].sequence_ids.tolist() == [first]
        assert runner.plans[0].token_indptr.tolist() == [0, 4]
        assert first.done
        assert not second.done
        engine.close()

    def test_window_guard_band_permits_multi_page_prefill(self) -> None:
        layout = CacheLayout(
            parent_capacity=8,
            groups=[
                CacheGroupLayout(
                    2,
                    retention=CacheRetention.WINDOW,
                    window_tokens=8,
                    read_window_tokens=3,
                )
            ],
        )

        class MultiPageRunner(ContinuationRecordingRunner):
            runner_config = RunnerConfig(
                continuation_prefill=True,
                prefill_chunk_size=8,
                max_batch_tokens=8,
                cache_layout=layout,
            )

            def bind_resources(self, binding: ResourceBinding) -> None:
                self.bound_layout = binding.cache_layout

        runner = MultiPageRunner()
        engine = Engine(runner)
        request = engine.submit(
            [1, 2, 3, 4, 5, 6], sampling_params=SamplingParams(max_output_tokens=1)
        )

        events = engine.step()

        assert runner.plans[0].token_ids.tolist() == [1, 2, 3, 4, 5, 6]
        assert [event.request_id for event in events] == [request]
        assert request.done
        engine.close()

    def test_config_limits_and_conservativeness_are_validated(self) -> None:
        default = SchedulerConfig()
        assert default.max_batch_tokens is None and default.prefill_chunk_size == 8192
        limited = SchedulerConfig(max_batch_tokens=17, prefill_chunk_size=5)
        assert (limited.max_batch_tokens, limited.prefill_chunk_size) == (17, 5)
        for field in ("max_batch_tokens", "prefill_chunk_size"):
            with (
                pytest.raises(ValueError, match=f"{field} must be positive"),
            ):
                SchedulerConfig(**{field: 0})
        for value in (-0.1, float("inf"), float("nan")):
            with (
                pytest.raises(
                    ValueError,
                    match="schedule_conservativeness must be finite and non-negative",
                ),
            ):
                SchedulerConfig(schedule_conservativeness=value)
        aggressive = SchedulerConfig(schedule_conservativeness=0.0)
        assert aggressive.schedule_conservativeness == 0.0
        default_engine = Engine(RecordingRunner())
        assert default_engine.new_token_ratio == 0.7
        default_engine.close()
        conservative = Engine(
            RecordingRunner(),
            scheduler_config=SchedulerConfig(schedule_conservativeness=2.0),
        )
        assert conservative.new_token_ratio == 1.0
        conservative.close()

    @pytest.mark.parametrize("enabled", [False, True])
    def test_continuation_capability_controls_chunked_prefill(self, enabled) -> None:
        class Runner(RecordingRunner):
            runner_config = RunnerConfig(continuation_prefill=enabled)

        runner = Runner()
        engine = Engine(runner, scheduler_config=SchedulerConfig(prefill_chunk_size=2))
        request = engine.submit(
            [1, 2, 3], sampling_params=SamplingParams(max_output_tokens=1)
        )
        if enabled:
            while engine.has_work():
                engine.step()
            assert [len(plan.token_ids) for plan in runner.plans] == [2, 1]
            assert request.done
        else:
            with pytest.raises(ValueError, match="continuation_prefill is false"):
                engine.step()
            assert runner.plans == [] and not request.done
        engine.close()

    def test_batch_plan_batch_key_defaults_and_range_validation(self) -> None:
        default = BatchPlan(
            sequence_ids=[7], context_lengths=[0], token_ids=[1], token_indptr=[0, 1]
        )
        keyed = BatchPlan(
            sequence_ids=[7],
            context_lengths=[0],
            token_ids=[1],
            token_indptr=[0, 1],
            batch_key=9,
        )
        assert default.batch_key is None
        assert keyed.batch_key == 9
        with pytest.raises(OverflowError):
            BatchPlan(
                sequence_ids=[7],
                context_lengths=[0],
                token_ids=[1],
                token_indptr=[0, 1],
                batch_key=2**64,
            )

        engine = Engine(RecordingRunner())
        with pytest.raises(OverflowError):
            engine.submit(
                [1], priority=2**31, sampling_params=SamplingParams(max_output_tokens=1)
            )
        with pytest.raises(OverflowError):
            engine.submit(
                [1],
                batch_key=2**64,
                sampling_params=SamplingParams(max_output_tokens=1),
            )

    def test_batch_plan_freezes_phase_and_sampling_mask(self) -> None:
        plan = BatchPlan(
            sequence_ids=[7, 8, 9],
            context_lengths=[0, 4, 0],
            token_ids=[10, 20, 30],
            token_indptr=[0, 1, 2, 3],
            sampling_mask=[False, True, True],
        )
        assert plan.phase == BatchPhase.PREFILL
        assert plan.sampling_mask.tolist() == [False, True, True]
        assert plan.num_sequences == 3
        assert len(RecordingRunner().forward(plan).token_ids) == 3
        with pytest.raises(AttributeError):
            setattr(plan, "sampling_mask", [True, False, False])

    def test_opaque_media_preserves_identity(self) -> None:
        payload = object()
        plan = BatchPlan(
            sequence_ids=[7, 8],
            context_lengths=[0, 0],
            token_ids=[10, 20],
            token_indptr=[0, 1, 2],
            # BatchPlan deliberately preserves opaque model-owned payloads.
            media=[payload, None],  # ty: ignore[invalid-argument-type]
        )

        assert plan.media[0] is payload
        assert plan.media[1] is None
        with pytest.raises(ValueError, match="one item per sequence"):
            BatchPlan(
                sequence_ids=[7, 8],
                context_lengths=[0, 0],
                token_ids=[10, 20],
                token_indptr=[0, 1, 2],
                media=[payload],
            )

    def test_engine_carries_media_through_prefill_and_decode(self) -> None:
        class ChunkedRunner(ContinuationRecordingRunner):
            runner_config = RunnerConfig(
                continuation_prefill=True,
                prefill_chunk_size=2,
            )

        runner = ChunkedRunner()
        engine = Engine(runner)
        payload = SimpleNamespace(
            token_ids=(1, 2, 3), cache_key=(1, 2, 3), token_spans=(), image=object()
        )
        engine.submit(
            [1, 2, 3],
            sampling_params=SamplingParams(max_output_tokens=2),
            media=payload,
        )

        while engine.has_work():
            engine.step()

        assert [plan.phase for plan in runner.plans] == [
            BatchPhase.PREFILL,
            BatchPhase.PREFILL,
            BatchPhase.DECODE,
        ]
        assert all(plan.media == (payload,) for plan in runner.plans)

    def test_media_can_supply_its_own_token_ids(self) -> None:
        class Media:
            token_ids = [4, 5]
            cache_key = (4, 5)
            token_spans = ()

        runner = RecordingRunner()
        engine = Engine(runner)
        media = Media()

        engine.submit(sampling_params=SamplingParams(max_output_tokens=1), media=media)
        engine.step()

        assert runner.plans[0].token_ids.tolist() == [4, 5]
        assert runner.plans[0].media == (media,)

        with pytest.raises(ValueError, match="only when media supplies"):
            engine.submit(sampling_params=SamplingParams(max_output_tokens=1))

    @pytest.mark.parametrize("missing", ["cache_key", "token_spans"])
    @pytest.mark.parametrize("tokens", [None, [1, 2]])
    def test_incomplete_media_is_rejected_before_admission(self, missing, tokens):
        runner = RecordingRunner()
        engine = Engine(runner)
        media = SimpleNamespace(token_ids=(1, 2), cache_key=(1, 2), token_spans=())
        delattr(media, missing)
        with pytest.raises(AttributeError, match=missing):
            engine.submit(
                tokens, sampling_params=SamplingParams(max_output_tokens=1), media=media
            )
        assert not engine.has_work() and not engine.poisoned
        assert runner.plans == []

    def test_batch_plan_phase_is_batch_homogeneous(self) -> None:
        prefill = BatchPlan(
            sequence_ids=[7, 8],
            context_lengths=[0, 0],
            token_ids=[1, 2],
            token_indptr=[0, 1, 2],
        )
        assert prefill.phase == BatchPhase.PREFILL
        assert prefill.sampling_mask.tolist() == [True, True]

        decode = BatchPlan(
            sequence_ids=[7, 8],
            context_lengths=[3, 5],
            token_ids=[1, 2],
            token_indptr=[0, 1, 2],
            phase=BatchPhase.DECODE,
        )
        assert decode.phase == BatchPhase.DECODE

    def test_batch_plan_numeric_metadata_is_read_only_zero_copy_numpy(self) -> None:
        plan = BatchPlan(
            sequence_ids=np.asarray([7, 8], dtype=np.uint64),
            context_lengths=np.asarray([2, 4], dtype=np.uint32),
            token_ids=np.asarray([11, 12, 13], dtype=np.uint32),
            token_indptr=np.asarray([0, 2, 3], dtype=np.uint32),
            sampling_mask=np.asarray([False, True], dtype=np.bool_),
        )
        expected = {
            "sequence_ids": (np.uint64, [7, 8]),
            "context_lengths": (np.uint32, [2, 4]),
            "token_ids": (np.uint32, [11, 12, 13]),
            "token_indptr": (np.uint32, [0, 2, 3]),
            "positions": (np.uint32, [2, 3, 4]),
            "sampling_mask": (np.bool_, [False, True]),
        }
        for name, (dtype, values) in expected.items():
            array = getattr(plan, name)
            assert array.dtype == dtype
            assert array.tolist() == values
            assert array.base is plan
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0
            with pytest.raises(ValueError, match="WRITEABLE"):
                array.flags.writeable = True
        assert np.shares_memory(plan.positions, plan.positions)

    def test_batch_plan_rejects_invalid_sampling_mask(self) -> None:
        for value in ([True], [True, False, True, False]):
            with pytest.raises(ValueError, match="one item per sequence"):
                BatchPlan(
                    sequence_ids=[7, 8, 9],
                    context_lengths=[0, 4, 0],
                    token_ids=[10, 20, 30],
                    token_indptr=[0, 1, 2, 3],
                    sampling_mask=value,
                )


class TestNativePolicyIntegration:
    def test_default_config_keeps_survivors_ahead_of_waiting_requests(self) -> None:
        def trace(config: SchedulerConfig | None) -> list[list[int]]:
            runner = RecordingRunner()
            kwargs: dict[str, Any] = {"max_running_requests": 2}
            if config is not None:
                kwargs["scheduler_config"] = config
            engine = Engine(runner, **kwargs)
            first = engine.submit(
                [1], sampling_params=SamplingParams(max_output_tokens=3)
            )
            second = engine.submit(
                [10], sampling_params=SamplingParams(max_output_tokens=1)
            )
            third = engine.submit(
                [20], sampling_params=SamplingParams(max_output_tokens=1)
            )
            engine.step()
            engine.step()
            engine.close()
            assert [first, second, third] == [1, 2, 3]
            return [plan.sequence_ids.tolist() for plan in runner.plans]

        expected = [[1, 2], [3]]
        assert trace(None) == expected
        assert trace(SchedulerConfig()) == expected

    def test_priority_is_strict_stable_and_never_backfills(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=4)
        low = engine.submit(
            [10],
            priority=0,
            batch_key=7,
            sampling_params=SamplingParams(max_output_tokens=1),
        )
        high_keyed_first = engine.submit(
            [20],
            priority=5,
            batch_key=7,
            sampling_params=SamplingParams(max_output_tokens=1),
        )
        high_none = engine.submit(
            [30], priority=5, sampling_params=SamplingParams(max_output_tokens=1)
        )
        high_keyed_second = engine.submit(
            [40],
            priority=5,
            batch_key=7,
            sampling_params=SamplingParams(max_output_tokens=1),
        )
        middle = engine.submit(
            [50],
            priority=4,
            batch_key=7,
            sampling_params=SamplingParams(max_output_tokens=1),
        )

        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [
            high_keyed_first,
            high_keyed_second,
        ]
        assert runner.plans[-1].batch_key == 7
        assert runner.plans[-1].num_sequences == 2

        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [high_none]
        assert runner.plans[-1].batch_key is None
        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [middle]
        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [low]

    def test_none_and_zero_are_distinct_homogeneous_keys(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=3)
        none_first = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=1)
        )
        zero = engine.submit(
            [2], batch_key=0, sampling_params=SamplingParams(max_output_tokens=1)
        )
        none_second = engine.submit(
            [3], sampling_params=SamplingParams(max_output_tokens=1)
        )

        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [none_first, none_second]
        assert runner.plans[-1].batch_key is None
        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [zero]
        assert runner.plans[-1].batch_key == 0

    def test_fresh_prefill_runs_before_the_decode_batch_resumes(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        decode = engine.submit(
            [10], sampling_params=SamplingParams(max_output_tokens=2)
        )
        engine.step()
        fresh = engine.submit([20], sampling_params=SamplingParams(max_output_tokens=1))
        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [fresh]
        assert runner.plans[-1].phase == BatchPhase.PREFILL
        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [decode]
        assert runner.plans[-1].phase == BatchPhase.DECODE

    def test_survivors_stay_ahead_of_later_submissions(self) -> None:
        runner = ControlledRunner([[11, 21], [12, 22]])
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit([10], sampling_params=SamplingParams(max_output_tokens=3))
        second = engine.submit(
            [20], sampling_params=SamplingParams(max_output_tokens=3)
        )
        engine.step()
        engine.submit([30], sampling_params=SamplingParams(max_output_tokens=1))
        engine.submit([40], sampling_params=SamplingParams(max_output_tokens=1))

        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [first, second]
        engine.close()

    def test_noncontiguous_rejection_restores_exact_cohort_and_age(self) -> None:
        runner = ControlledRunner([[11, 31], [21]], forward_failures=1)
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit(
            [10],
            priority=3,
            batch_key=7,
            sampling_params=SamplingParams(max_output_tokens=1),
        )
        other_key = engine.submit(
            [20], priority=3, sampling_params=SamplingParams(max_output_tokens=1)
        )
        third = engine.submit(
            [30],
            priority=3,
            batch_key=7,
            sampling_params=SamplingParams(max_output_tokens=1),
        )
        lower = engine.submit(
            [40],
            priority=2,
            batch_key=7,
            sampling_params=SamplingParams(max_output_tokens=1),
        )

        with pytest.raises(RuntimeError, match="forward rejected"):
            engine.step()
        assert not engine.poisoned
        assert engine.ready_request_count == 4

        engine.step()
        failed, retry = runner.plans[:2]
        assert (failed.batch_id, retry.batch_id) == (1, 2)
        assert failed.sequence_ids.tolist() == [first, third]
        assert retry.sequence_ids.tolist() == [first, third]
        for field in (
            "batch_key",
            "phase",
            "token_ids",
            "token_indptr",
            "context_lengths",
            "sampling_mask",
        ):
            left = getattr(failed, field)
            right = getattr(retry, field)
            assert (
                left.tolist() == right.tolist()
                if hasattr(left, "tolist")
                else left == right
            ), field
        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [other_key]
        engine.close()
        assert lower.done
        assert lower.take_result() == []

    def test_ready_cancellation_preserves_selected_row_order(self) -> None:
        ready_runner = RecordingRunner()
        ready_engine = Engine(ready_runner, max_running_requests=3)
        first = ready_engine.submit(
            [10], sampling_params=SamplingParams(max_output_tokens=1)
        )
        middle = ready_engine.submit(
            [20], sampling_params=SamplingParams(max_output_tokens=1)
        )
        third = ready_engine.submit(
            [30], sampling_params=SamplingParams(max_output_tokens=1)
        )
        assert ready_engine.cancel(middle)
        ready_engine.step()
        assert ready_runner.plans[-1].sequence_ids.tolist() == [first, third]

    def test_decode_and_partial_prefill_are_dispatched_separately(
        self,
    ) -> None:
        runner = ContinuationRecordingRunner()
        engine = Engine(
            runner,
            max_running_requests=2,
            scheduler_config=SchedulerConfig(
                max_batch_tokens=3,
                prefill_chunk_size=2,
            ),
        )
        decoding = engine.submit(
            [10],
            sampling_params=SamplingParams(max_output_tokens=2, stop_token_ids=[22]),
        )
        assert [event.token_id for event in engine.step()] == [11]
        partial = engine.submit(
            [20, 21, 22, 23],
            sampling_params=SamplingParams(max_output_tokens=1, stop_token_ids=[22]),
        )

        events = engine.step()
        plan = runner.plans[-1]
        assert plan.sequence_ids.tolist() == [partial]
        assert plan.phase == BatchPhase.PREFILL
        assert plan.token_ids.tolist() == [20, 21]
        assert plan.context_lengths.tolist() == [0]
        assert plan.sampling_mask.tolist() == [False]
        assert events == []
        assert not partial.done
        assert runner.released == []

        events = engine.step()
        assert runner.plans[-1].phase == BatchPhase.PREFILL
        assert runner.plans[-1].context_lengths.tolist() == [2]
        assert runner.plans[-1].sampling_mask.tolist() == [True]
        assert [event.request_id for event in events] == [partial]
        assert partial.done
        assert partial.take_result() == [24]
        assert runner.released == [partial]

        events = engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [decoding]
        assert runner.plans[-1].phase == BatchPhase.DECODE
        assert runner.plans[-1].token_ids.tolist() == [11]
        assert [event.request_id for event in events] == [decoding]
        assert runner.released == [partial, decoding]

    def test_partial_prefill_survivor_finishes_before_waiting_request(self) -> None:
        runner = ContinuationRecordingRunner()
        engine = Engine(
            runner,
            max_running_requests=2,
            scheduler_config=SchedulerConfig(
                max_batch_tokens=1,
                prefill_chunk_size=1,
            ),
        )
        first = engine.submit(
            [1, 2], sampling_params=SamplingParams(max_output_tokens=1)
        )
        second = engine.submit(
            [10, 11], sampling_params=SamplingParams(max_output_tokens=1)
        )
        while engine.has_work():
            engine.step()
        assert (first.take_result(), second.take_result()) == ([3], [12])
        assert [plan.sequence_ids[0] for plan in runner.plans] == [1, 1, 2, 2]

    def test_compact_partial_prefill_output_is_quarantined(self) -> None:
        compact = ContinuationControlledRunner([[]])
        poisoned = Engine(
            compact,
            scheduler_config=SchedulerConfig(prefill_chunk_size=1),
        )
        poisoned.submit([1, 2], sampling_params=SamplingParams(max_output_tokens=1))
        with pytest.raises(
            RuntimeError, match="runner returned 0 token rows for 1 sequences"
        ):
            poisoned.step()
        assert poisoned.poisoned

    def test_cleanup_failure_does_not_publish_a_sampled_token(self) -> None:
        class FailingRelease(ContinuationRecordingRunner):
            def release(self, request_ids: list[int]) -> None:
                raise RuntimeError("cleanup failed")

        engine = Engine(
            FailingRelease(),
            scheduler_config=SchedulerConfig(prefill_chunk_size=1),
        )
        request = engine.submit(
            [4], sampling_params=SamplingParams(max_output_tokens=1)
        )
        with pytest.raises(RuntimeError, match="cleanup failed"):
            engine.step()
        assert engine.poisoned
        assert not request.done
