from __future__ import annotations

from types import SimpleNamespace

import pytest
from output_helpers import forward_output

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    ResourceBinding,
    RunnerConfig,
    SamplingParams,
)


def history_layout(parent_capacity: int, block_tokens: int = 2) -> CacheLayout:
    return CacheLayout(
        parent_capacity=parent_capacity, groups=[CacheGroupLayout(block_tokens)]
    )


class RecordingRunner:
    def __init__(
        self,
        results: list[object] | None = None,
        *,
        failures: int = 0,
        release_error: BaseException | None = None,
        cache_layout: CacheLayout | None = None,
    ) -> None:
        self.results = list(results or [])
        self.failures = failures
        self.release_error = release_error
        self.plans: list[BatchPlan] = []
        self.released: list[int] = []
        self.cache_binding: tuple[int, CacheLayout] | None = None
        if cache_layout is not None:
            self.cache_layout = cache_layout
            self.runner_config = RunnerConfig(cache_layout=cache_layout)

    def bind_resources(self, binding: ResourceBinding) -> None:
        assert binding.cache_pool_id is not None
        assert binding.cache_layout is not None
        self.cache_binding = (binding.cache_pool_id, binding.cache_layout)

    def forward(self, plan: BatchPlan):
        self.plans.append(plan)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("forward rejected")
        if self.results:
            return self.results.pop(0)
        return forward_output(
            [plan.token_ids[end - 1] + 1 for end in plan.token_indptr[1:]]
        )

    def release(self, request_ids: list[int]) -> None:
        if self.release_error is not None:
            raise self.release_error
        self.released.extend(request_ids)


class TestEngineExecution:
    def test_step_executes_and_commits_one_batch(self) -> None:
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=1)
        first = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=2))
        second = engine.submit(
            [10], sampling_params=SamplingParams(max_output_tokens=1)
        )

        first_events = engine.step()
        assert [event.request_id for event in first_events] == [first]
        assert not first.done

        # Sticky admission keeps the surviving request running.
        second_events = engine.step()
        assert [event.request_id for event in second_events] == [first]
        assert first.done
        assert not second.done
        final_events = engine.step()
        assert [event.request_id for event in final_events] == [second]
        assert second.take_result() == [11]
        assert runner.released == [first, second]

    def test_marked_failure_quarantines_accepted_native_resources(self) -> None:
        class PostAcceptError(RuntimeError):
            _resin_post_accept = True

        class PostAcceptRunner(RecordingRunner):
            def forward(self, plan: BatchPlan):
                self.plans.append(plan)
                raise PostAcceptError("CUDA quiescence failed")

        runner = PostAcceptRunner(cache_layout=history_layout(2))
        engine = Engine(runner)
        request = engine.submit(
            [1, 2], sampling_params=SamplingParams(max_output_tokens=1)
        )

        with pytest.raises(PostAcceptError, match="quiescence failed"):
            engine.step()

        assert engine.poisoned
        assert len(runner.plans) == 1
        assert runner.released == []
        assert engine.ready_request_count == 0
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            1,
            1,
        )
        with pytest.raises(RuntimeError, match="resources are quarantined"):
            engine.cancel(request)
        with pytest.raises(RuntimeError, match="resources are quarantined"):
            engine.close()

    def test_unmarked_failure_is_retryable_and_preserves_the_cohort(self) -> None:
        runner = RecordingRunner(failures=1, cache_layout=history_layout(2))
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit([10], sampling_params=SamplingParams(max_output_tokens=1))
        second = engine.submit(
            [20], sampling_params=SamplingParams(max_output_tokens=1)
        )

        with pytest.raises(RuntimeError, match="forward rejected"):
            engine.step()
        assert not engine.poisoned
        assert engine.ready_request_count == 2

        events = engine.step()
        assert [event.request_id for event in events] == [first, second]
        failed, retry = runner.plans
        assert (failed.batch_id, retry.batch_id) == (1, 2)
        assert (
            failed.sequence_ids.tolist()
            == retry.sequence_ids.tolist()
            == [first, second]
        )
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            2,
            0,
        )

    @pytest.mark.parametrize(
        ("malformed", "error_type"),
        ((object(), TypeError), ([], TypeError), (forward_output([]), RuntimeError)),
    )
    def test_malformed_result_quarantines_after_forward_returns(
        self, malformed: object, error_type: type[BaseException]
    ) -> None:
        runner = RecordingRunner([malformed], cache_layout=history_layout(2))
        engine = Engine(runner)
        request = engine.submit(
            [1, 2], sampling_params=SamplingParams(max_output_tokens=1)
        )

        with pytest.raises(error_type):
            engine.step()

        assert engine.poisoned
        assert len(runner.plans) == 1
        assert runner.released == []
        assert engine.ready_request_count == 0
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            1,
            1,
        )
        with pytest.raises(RuntimeError, match="resources are quarantined"):
            engine.cancel(request)

    def test_terminal_release_failure_quarantines_before_native_reuse(self) -> None:
        runner = RecordingRunner(
            [forward_output([7])],
            release_error=RuntimeError("physical cleanup failed"),
            cache_layout=history_layout(1),
        )
        engine = Engine(runner)
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=1)
        )

        with pytest.raises(RuntimeError, match="physical cleanup failed"):
            engine.step()
        assert engine.poisoned
        assert engine.poison_error is not None
        assert "release" in engine.poison_error
        assert not request.done
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            0,
            1,
        )

    def test_close_retires_ready_requests_and_is_idempotent(self) -> None:
        runner = RecordingRunner(cache_layout=history_layout(2))
        engine = Engine(runner)
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=2)
        )

        engine.close()

        assert engine.closed
        assert not engine.poisoned
        assert not engine.has_work()
        assert request.done
        assert request.take_result() == []
        assert runner.released == [request]
        assert (engine.cache_free_parent_count, engine.cache_occupied_block_count) == (
            2,
            0,
        )

        engine.close()
        assert runner.released == [request]
        for operation in (
            lambda: engine.submit(
                [3], sampling_params=SamplingParams(max_output_tokens=1)
            ),
            engine.step,
            lambda: engine.cancel(request),
        ):
            with pytest.raises(RuntimeError, match="closed"):
                operation()

    def test_context_manager_closes_without_suppressing_exceptions(self) -> None:
        runner = RecordingRunner()
        with pytest.raises(ValueError, match="body failed"):
            with Engine(runner) as engine:
                request = engine.submit(
                    [1], sampling_params=SamplingParams(max_output_tokens=1)
                )
                raise ValueError("body failed")

        assert engine.closed
        assert request.done
        assert request.take_result() == []
        assert runner.released == [request]


class TestRunnerContractConstruction:
    @pytest.mark.parametrize(
        "callbacks",
        [
            {},
            {"forward": 7},
            {"forward": lambda plan: forward_output([1]), "release": 7},
        ],
        ids=["missing-forward", "noncallable-forward", "noncallable-release"],
    )
    def test_constructor_requires_callable_forward_and_release(self, callbacks) -> None:
        with pytest.raises(TypeError, match="callable"):
            Engine(SimpleNamespace(**callbacks))  # ty: ignore[invalid-argument-type]


class TestPhaseCoordination:
    def test_proposal_and_mismatched_gate_do_not_reserve_or_execute(self):
        from resin import BatchPhase

        runner = RecordingRunner(cache_layout=history_layout(4))
        engine = Engine(runner)
        assert engine.next_phase() is None
        request = engine.submit(
            [1, 2], sampling_params=SamplingParams(max_output_tokens=2)
        )
        before = engine.cache_occupied_block_count
        assert engine.next_phase() == BatchPhase.PREFILL
        assert engine.next_phase() == BatchPhase.PREFILL
        assert engine.step(phase=BatchPhase.DECODE) == []
        assert engine.cache_occupied_block_count == before
        assert runner.plans == [] and not request.done
        engine.step(phase=BatchPhase.PREFILL)
        assert runner.plans[0].batch_id == 1
        assert engine.next_phase() == BatchPhase.DECODE
        assert engine.step(phase=BatchPhase.PREFILL) == []
        engine.step(phase=BatchPhase.DECODE)
        assert request.done and engine.next_phase() is None

    def test_proposal_uses_sticky_admission_instead_of_waiting_prefill(self):
        from resin import BatchPhase

        engine = Engine(RecordingRunner(), max_running_requests=1)
        first = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=2))
        engine.step()
        second = engine.submit(
            [10], sampling_params=SamplingParams(max_output_tokens=1)
        )
        assert engine.next_phase() == BatchPhase.DECODE
        engine.step(phase=BatchPhase.DECODE)
        assert first.done and not second.done
        assert engine.next_phase() == BatchPhase.PREFILL
        engine.step(phase=BatchPhase.PREFILL)
        assert second.done
