from __future__ import annotations

import numpy as np
import pytest
from output_helpers import forward_output

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    ForwardOutput,
    RequestLease,
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


class TestEngine:
    def test_engine_context_length_and_request_limit_are_top_level(self) -> None:
        engine = Engine(RecordingRunner(), max_running_requests=3, context_length=4)
        assert engine.max_running_requests == 3
        assert engine.context_length == 4
        with pytest.raises(ValueError, match="exceeds Engine context_length 4"):
            engine.submit(
                [1, 2, 3], sampling_params=SamplingParams(max_output_tokens=3)
            )
        with pytest.raises(ValueError, match="context_length must be positive"):
            Engine(RecordingRunner(), context_length=0)

    @pytest.mark.parametrize("params", [None, SamplingParams(max_output_tokens=None)])
    def test_missing_generation_limit_uses_remaining_model_context(
        self, params
    ) -> None:
        engine = Engine(RecordingRunner(), context_length=3)
        request = engine.submit([1, 2], sampling_params=params)
        while engine.has_work():
            engine.step()
        assert request.take_result() == [3, 4]

    @pytest.mark.parametrize("params", [None, SamplingParams(max_output_tokens=3)])
    def test_runner_eos_does_not_supply_an_implicit_stop_policy(self, params) -> None:
        class EosRunner(RecordingRunner):
            eos_token_id = 2

        engine = Engine(EosRunner(), context_length=3)
        request = engine.submit([1], sampling_params=params)
        events = []
        while engine.has_work():
            events.extend(engine.step())

        assert request.take_result() == [2, 3, 4]
        assert [event.finish_reason for event in events] == [None, None, "length"]

    def test_generation_policies_are_isolated_between_requests(self) -> None:
        engine = Engine(RecordingRunner(), max_running_requests=3)
        first = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=6, stop_token_ids=[3])
        )
        second = engine.submit(
            [1],
            sampling_params=SamplingParams(max_output_tokens=6, stop_token_ids=(4,)),
        )
        third = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=4, stop_token_ids=[])
        )
        finished = {}
        while engine.has_work():
            for event in engine.step():
                if event.finished:
                    finished[event.request_id] = event.finish_reason

        assert first.take_result() == [2, 3]
        assert second.take_result() == [2, 3, 4]
        assert third.take_result() == [2, 3, 4, 5]
        assert finished == {first.id: "stop", second.id: "stop", third.id: "length"}

    def test_zero_generation_budget_requires_and_supports_prompt_scoring(self) -> None:
        engine = Engine(RecordingRunner())
        with pytest.raises(ValueError, match="unless prompt_logprobs is enabled"):
            engine.submit([1], sampling_params=SamplingParams(max_output_tokens=0))

        request = engine.submit(
            [1],
            sampling_params=SamplingParams(max_output_tokens=0, prompt_logprobs=True),
        )
        assert engine.step() == []
        assert request.done and request.take_result() == []
        assert request.prompt_logprobs == [None]

    @pytest.mark.parametrize(
        "retired",
        [{"max_output_tokens": 1}, {"max_new_tokens": 1}, {"stop_token_ids": [2]}],
    )
    def test_submit_rejects_retired_generation_keywords(self, retired) -> None:
        engine = Engine(RecordingRunner())
        with pytest.raises(TypeError, match=next(iter(retired))):
            engine.submit([1], **retired)
        assert not engine.has_work()

    def test_submit_rejects_retired_positional_generation_limit(self) -> None:
        engine = Engine(RecordingRunner())
        with pytest.raises(TypeError):
            engine.submit([1], 1)  # ty: ignore[too-many-positional-arguments]
        assert not engine.has_work()

    def test_engine_rejects_retired_eos_keyword(self) -> None:
        with pytest.raises(TypeError, match="eos_token_id"):
            Engine(RecordingRunner(), eos_token_id=2)  # ty: ignore[unknown-argument]

    def test_request_lease_owns_result_without_engine_retention(self) -> None:
        engine = Engine(RecordingRunner())
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=1)
        )

        assert isinstance(request, RequestLease)
        assert request.id == request.request_id
        assert not request.done
        with pytest.raises(RuntimeError, match="not completed"):
            request.take_result()

        engine.step()

        assert request.done
        assert not request.cancelled
        assert not hasattr(engine, "output")
        assert request.take_result() == [2]
        with pytest.raises(RuntimeError, match="already been consumed"):
            request.take_result()

    def test_request_lease_is_bound_to_its_engine(self) -> None:
        first = Engine(RecordingRunner())
        second = Engine(RecordingRunner())
        request = first.submit([1], sampling_params=SamplingParams(max_output_tokens=1))

        with pytest.raises(ValueError, match="different Engine"):
            second.cancel(request)
        assert first.ready_request_count == 1
        assert second.ready_request_count == 0

    def test_rust_owns_prefill_decode_and_completion_transitions(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit(
            [1, 2], sampling_params=SamplingParams(max_output_tokens=2)
        )
        second = engine.submit([8], sampling_params=SamplingParams(max_output_tokens=1))

        initial = engine.step()
        assert [event.token_id for event in initial] == [3, 9]
        assert runner.plans[0].phase == BatchPhase.PREFILL
        assert runner.plans[0].token_indptr.tolist() == [0, 2, 3]
        assert runner.plans[0].sampling_mask.tolist() == [True, True]
        assert runner.released == [second]

        final = engine.step()
        assert [event.token_id for event in final] == [4]
        assert runner.plans[1].phase == BatchPhase.DECODE
        assert runner.plans[1].context_lengths.tolist() == [2]
        assert runner.plans[1].sampling_mask.tolist() == [True]
        assert not engine.has_work()
        assert first.done
        assert first.take_result() == [3, 4]
        assert second.done
        assert second.take_result() == [9]

    @pytest.mark.parametrize(
        "rows,error", [([], "runner returned 0 token rows"), ([[]], "empty token row")]
    )
    def test_sampled_sequence_requires_an_output_row_and_token(self, rows, error):
        class BadRunner:
            def forward(self, plan):
                return forward_output(rows)

        engine = Engine(BadRunner())
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=1)
        )
        with pytest.raises(RuntimeError, match=error):
            engine.step()
        assert engine.poisoned and not request.done

    @pytest.mark.parametrize("invalid", [[3], [[3]], (3,), ((3,),)])
    def test_sequence_results_quarantine_after_a_committed_step(self, invalid):
        class InvalidRunner(RecordingRunner):
            runner_config = RunnerConfig(
                cache_layout=CacheLayout(
                    parent_capacity=4, groups=[CacheGroupLayout(2)]
                )
            )

            def bind_resources(self, binding):
                pass

            def forward(self, plan):
                if plan.phase == BatchPhase.PREFILL:
                    return super().forward(plan)
                return invalid

        runner = InvalidRunner()
        engine = Engine(runner)
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=3)
        )
        assert [event.token_id for event in engine.step()] == [2]
        with pytest.raises(TypeError, match="runner must return ForwardOutput"):
            engine.step()
        assert engine.poisoned and not request.done
        assert engine.cache_occupied_block_count == 1
        assert runner.released == []
        with pytest.raises(RuntimeError, match="quarantined"):
            engine.close()

    @pytest.mark.parametrize("scored", [False, True])
    def test_native_token_only_and_scored_results_complete_normally(self, scored):
        class NativeRunner:
            def forward(self, plan):
                return ForwardOutput(
                    token_ids=np.asarray([plan.token_ids[-1] + 1], np.uint32),
                    token_indptr=np.asarray([0, 1], np.uint32),
                    logprobs=np.asarray([-0.25], np.float32) if scored else None,
                )

        engine = Engine(NativeRunner())
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=2, logprobs=scored)
        )
        events = [*engine.step(), *engine.step()]
        assert request.take_result() == [2, 3]
        assert not engine.poisoned
        for event in events:
            if scored:
                assert event.logprobs is not None
                assert event.logprobs.logprob == -0.25
            else:
                assert event.logprobs is None

    def test_decode_runner_can_commit_a_verified_token_block(self) -> None:
        class BlockRunner:
            runner_config = RunnerConfig(max_accepted_per_step=3)

            def __init__(self) -> None:
                self.plans: list[BatchPlan] = []

            def forward(self, plan: BatchPlan):
                self.plans.append(plan)
                if plan.phase == BatchPhase.PREFILL:
                    return forward_output([2])
                if len(self.plans) == 2:
                    return forward_output([[3, 4, 5]])
                return forward_output([[6, 7]])

        runner = BlockRunner()
        engine = Engine(runner)
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=6)
        )

        assert [event.token_id for event in engine.step()] == [2]
        block = engine.step()
        assert [event.token_id for event in block] == [3, 4, 5]
        assert [event.index for event in block] == [1, 2, 3]
        assert not any(event.finished for event in block)

        final = engine.step()
        assert [event.token_id for event in final] == [6, 7]
        assert [event.index for event in final] == [4, 5]
        assert [event.finished for event in final] == [False, True]
        assert runner.plans[2].token_ids.tolist() == [5]
        assert runner.plans[2].context_lengths.tolist() == [4]
        assert request.take_result() == [2, 3, 4, 5, 6, 7]

    def test_managed_cache_leases_declared_speculative_write_lookahead(self) -> None:
        layout = CacheLayout(parent_capacity=2, groups=[CacheGroupLayout(2)])

        class BlockRunner:
            runner_config = RunnerConfig(
                cache_layout=layout,
                max_accepted_per_step=3,
            )

            def __init__(self) -> None:
                self.plans: list[BatchPlan] = []

            @staticmethod
            def bind_resources(_binding: ResourceBinding) -> None:
                return None

            def forward(self, plan: BatchPlan):
                self.plans.append(plan)
                return forward_output(
                    [2] if plan.phase == BatchPhase.PREFILL else [[3, 4, 5]]
                )

        runner = BlockRunner()
        assert runner.runner_config.max_accepted_per_step == 3
        engine = Engine(runner)
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=4)
        )

        engine.step()
        events = engine.step()

        assert [event.token_id for event in events] == [3, 4, 5]
        decode = runner.plans[1]
        assert len(decode.cache_groups[0].block_slab_ids) == 2
        assert len(decode.cache_groups[0].append_slab_ids) == 1
        assert request.take_result() == [2, 3, 4, 5]

    def test_verified_block_stops_at_request_token_and_discards_its_suffix(
        self,
    ) -> None:
        class BlockRunner:
            runner_config = RunnerConfig(max_accepted_per_step=3)

            @staticmethod
            def forward(plan: BatchPlan):
                if plan.phase == BatchPhase.PREFILL:
                    return forward_output([2])
                return forward_output([[3, 7, 8]])

        engine = Engine(BlockRunner())
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=8, stop_token_ids=[7])
        )
        engine.step()

        events = engine.step()

        assert [event.token_id for event in events] == [3, 7]
        assert [event.finished for event in events] == [False, True]
        assert [event.finish_reason for event in events] == [None, "stop"]
        assert request.take_result() == [2, 3, 7]

    def test_request_stops_at_any_listed_token(self) -> None:
        engine = Engine(RecordingRunner())
        request = engine.submit(
            [1],
            sampling_params=SamplingParams(max_output_tokens=5, stop_token_ids=[99, 3]),
        )

        first = engine.step()[0]
        second = engine.step()[0]

        assert not first.finished
        assert first.finish_reason is None
        assert second.finished
        assert second.finish_reason == "stop"
        assert request.take_result() == [2, 3]

    def test_length_finish_reason_is_explicit(self) -> None:
        engine = Engine(RecordingRunner())
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=1)
        )

        event = engine.step()[0]

        assert event.finished
        assert event.finish_reason == "length"
        assert request.take_result() == [2]

    @pytest.mark.parametrize("max_output_tokens", [1, 2])
    def test_speculative_rows_are_rejected_during_prefill(
        self, max_output_tokens
    ) -> None:
        class BadRunner:
            @staticmethod
            def forward(plan: BatchPlan):
                return forward_output([[2, 3] for _ in range(plan.num_sequences)])

        engine = Engine(BadRunner())
        engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=max_output_tokens)
        )

        with pytest.raises(RuntimeError, match="only supported during decode"):
            engine.step()

    @pytest.mark.parametrize("stop_token_ids", [[], [3]])
    def test_terminal_truncation_cannot_hide_unleased_decode_tokens(
        self, stop_token_ids
    ):
        class BadRunner:
            runner_config = RunnerConfig(
                cache_layout=CacheLayout(
                    parent_capacity=4, groups=[CacheGroupLayout(2)]
                )
            )

            def bind_resources(self, binding):
                pass

            def forward(self, plan):
                return forward_output(
                    [2] if plan.phase == BatchPhase.PREFILL else [[3, 4, 5]]
                )

        engine = Engine(BadRunner())
        request = engine.submit(
            [1],
            sampling_params=SamplingParams(
                max_output_tokens=2, stop_token_ids=stop_token_ids
            ),
        )
        engine.step()
        with pytest.raises(RuntimeError, match="declared output maximum is 1"):
            engine.step()
        assert engine.poisoned
        assert not request.done

    def test_batch_limit_keeps_running_request_ahead_of_waiter(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=1)
        first = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=3))
        second = engine.submit(
            [10], sampling_params=SamplingParams(max_output_tokens=1)
        )
        engine.step()
        event = engine.step()[0]
        assert event.request_id == first
        assert not first.done
        assert not second.done
        engine.step()
        event = engine.step()[0]
        assert event.request_id == second
        assert second.done
        assert second.take_result() == [11]

    def test_ready_prefill_fills_a_vacancy_before_decode_resumes(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        surviving = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=3)
        )
        terminal = engine.submit(
            [10], sampling_params=SamplingParams(max_output_tokens=1)
        )
        untouched = engine.submit(
            [20], sampling_params=SamplingParams(max_output_tokens=1)
        )

        first_events = engine.step()
        assert [event.request_id for event in first_events] == [surviving, terminal]
        assert runner.plans[0].sequence_ids.tolist() == [surviving, terminal]

        second_events = engine.step()
        assert [event.request_id for event in second_events] == [untouched]
        assert runner.plans[1].sequence_ids.tolist() == [untouched]
        assert untouched.done
        assert not surviving.done

        third_events = engine.step()
        assert [event.request_id for event in third_events] == [surviving]
        assert runner.plans[2].sequence_ids.tolist() == [surviving]

        fourth_events = engine.step()
        assert [event.request_id for event in fourth_events] == [surviving]
        assert runner.plans[3].sequence_ids.tolist() == [surviving]
        assert untouched.take_result() == [21]

    def test_public_batch_plan_validates_indptr(self) -> None:
        with pytest.raises(ValueError, match="end at the token count"):
            BatchPlan(
                sequence_ids=[7],
                context_lengths=[0],
                token_ids=[1, 2],
                token_indptr=[0, 1],
            )

    def test_rust_owns_flat_cache_allocation_and_sparse_group_metadata(self) -> None:
        layout = CacheLayout(parent_capacity=3, groups=[CacheGroupLayout(2)])

        class CachedRunner(RecordingRunner):
            runner_config = RunnerConfig(cache_layout=layout)

            def __init__(self) -> None:
                super().__init__()
                self.binding = None

            def bind_resources(self, binding: ResourceBinding) -> None:
                self.binding = (binding.cache_pool_id, binding.cache_layout)

        runner = CachedRunner()
        engine = Engine(runner)
        request = engine.submit(
            [4, 5, 6], sampling_params=SamplingParams(max_output_tokens=2)
        )

        assert runner.binding == (engine.cache_pool_id, layout)
        assert runner.binding[1] is layout
        assert engine.cache_occupied_block_count == 0
        assert engine.cache_free_parent_count == 3

        engine.step()
        assert engine.cache_occupied_block_count == 2
        assert engine.cache_free_parent_count == 1
        prefill = runner.plans[0]
        assert len(prefill.cache_groups) == 1
        group = prefill.cache_groups[0]
        assert group.group_index == 0
        assert group.block_indptr.tolist() == [0, 2]
        assert group.block_table_start_indices.tolist() == [0]
        blocks = list(zip(group.block_slab_ids, group.block_slotnums, strict=True))
        appends = list(zip(group.append_slab_ids, group.append_slotnums, strict=True))
        assert appends == [
            blocks[0],
            blocks[0],
            blocks[1],
        ]
        assert group.append_block_offsets.tolist() == [0, 1, 0]

        engine.step()
        decode = runner.plans[1]
        assert decode.context_lengths.tolist() == [3]
        decode_group = decode.cache_groups[0]
        decode_blocks = list(
            zip(
                decode_group.block_slab_ids,
                decode_group.block_slotnums,
                strict=True,
            )
        )
        decode_appends = list(
            zip(
                decode_group.append_slab_ids,
                decode_group.append_slotnums,
                strict=True,
            )
        )
        assert decode_blocks == blocks
        assert decode_appends == [blocks[1]]
        assert decode_group.append_block_offsets.tolist() == [1]

        assert runner.released == [request]
        assert engine.cache_occupied_block_count == 0
        assert engine.cache_free_parent_count == 3

    def test_kv_grows_on_demand_and_retracts_for_replay(self) -> None:
        layout = CacheLayout(parent_capacity=3, groups=[CacheGroupLayout(1)])

        class TinyCachedRunner(RecordingRunner):
            runner_config = RunnerConfig(
                cache_layout=layout,
                continuation_prefill=True,
            )

            def bind_resources(self, binding: ResourceBinding) -> None:
                self.binding = (binding.cache_pool_id, binding.cache_layout)

        runner = TinyCachedRunner()
        engine = Engine(
            runner,
            max_running_requests=2,
            context_length=3,
            scheduler_config=SchedulerConfig(schedule_conservativeness=0.0),
        )
        first = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=3))
        second = engine.submit(
            [10], sampling_params=SamplingParams(max_output_tokens=3)
        )

        engine.step()
        assert engine.cache_occupied_block_count == 2
        assert runner.plans[-1].sequence_ids.tolist() == [first, second]

        while engine.has_work():
            engine.step()

        assert second.take_result() == [11, 12, 13]
        assert first.take_result() == [2, 3, 4]
        assert runner.released == [first, second, first]
        first_replay = next(
            plan
            for plan in runner.plans[1:]
            if plan.sequence_ids.tolist() == [first]
            and plan.phase == BatchPhase.PREFILL
        )
        assert first_replay.sampling_mask.tolist() == [False]
        assert engine.cache_occupied_block_count == 0

    def test_conservativeness_limits_initial_kv_admission(self) -> None:
        layout = CacheLayout(parent_capacity=3, groups=[CacheGroupLayout(1)])

        class TinyCachedRunner(RecordingRunner):
            runner_config = RunnerConfig(cache_layout=layout)

            @staticmethod
            def bind_resources(binding: ResourceBinding) -> None:
                del binding

        runner = TinyCachedRunner()
        engine = Engine(runner, max_running_requests=2, context_length=3)
        first = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=3))
        second = engine.submit(
            [10], sampling_params=SamplingParams(max_output_tokens=3)
        )
        engine.step()
        assert runner.plans[-1].sequence_ids.tolist() == [first]
        assert not second.done
        engine.close()
