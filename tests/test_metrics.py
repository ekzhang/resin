"""Metrics observe committed engine state without changing the runner contract."""

from __future__ import annotations

import pytest
from output_helpers import forward_output

from resin import (
    BatchPhase,
    BatchPlan,
    Engine,
    EngineMetrics,
    ForwardOutput,
    SamplingParams,
)


class Runner:
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        return forward_output(
            [
                int(plan.token_ids[end - 1]) + 1
                for _, end in map(plan.token_range, range(plan.num_sequences))
            ]
        )


def test_metrics_default_on_with_an_explicit_opt_out_and_no_new_runner_methods():
    class UnobservedRunner(Runner):
        enabled = True

        def configure_metrics(self, *, enabled: bool):
            self.enabled = enabled

        def metrics_snapshot(self):
            raise AssertionError("disabled metrics must not poll the runner")

    runner = UnobservedRunner()
    with Engine(runner, enable_metrics=False) as engine:
        assert not runner.enabled
        assert engine.metrics_snapshot() is None
        engine.submit([1], sampling_params=SamplingParams(max_output_tokens=1))
        engine.step()
        assert engine.metrics_snapshot() is None

    with Engine(Runner()) as engine:
        snapshot = engine.metrics_snapshot()
        assert isinstance(snapshot, EngineMetrics)
        assert snapshot.runner is None
        assert snapshot.cache is None
        snapshot.requests["accepted_requests"] = 100
        assert snapshot.requests["accepted_requests"] == 0
        with pytest.raises(AttributeError):
            setattr(snapshot, "runner", None)


def test_stop_clipping_and_cancellation_keep_live_and_completed_totals_distinct():
    class SpeculativeRunner(Runner):
        def forward(self, plan: BatchPlan) -> ForwardOutput:
            return forward_output(
                [[2] if plan.phase == BatchPhase.PREFILL else [3, 4, 5]]
            )

    with Engine(SpeculativeRunner(), enable_metrics=True) as engine:
        stopped = engine.submit(
            [1],
            sampling_params=SamplingParams(max_output_tokens=10, stop_token_ids=[4]),
        )
        engine.step()
        engine.step()
        assert stopped.take_result() == [2, 3, 4]
        cancelled = engine.submit(
            [9], sampling_params=SamplingParams(max_output_tokens=10)
        )
        engine.step()
        engine.cancel(cancelled)
        snapshot = engine.metrics_snapshot()
        assert snapshot is not None
        values = snapshot.requests
        assert values["accepted_requests"] == 2
        assert values["completed_requests"] == 1
        assert values["aborted_requests"] == 1
        assert values["decode_tokens"] == 4
        assert values["generation_tokens"] == 3
        assert values["prefill_compute_tokens"] == 2
        assert values["prompt_tokens"] == 1
        assert values["generation_tokens_histogram"]["sum"] == 3
        assert values["inter_token_latency_seconds"]["count"] == 2
        assert values["time_to_first_token_seconds"]["count"] == 2
        assert values["request_time_per_output_token_seconds"]["count"] == 1
        assert values["running_requests"] == values["waiting_requests"] == 0
