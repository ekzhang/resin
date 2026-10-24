"""Metrics observe committed engine state without changing the runner contract."""

from __future__ import annotations

import json
import re
import time

import pytest
from output_helpers import forward_output

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    ForwardOutput,
    ResourceBinding,
    RunnerConfig,
    SamplingParams,
)
from resin.backend.metrics import RunnerMetrics


class Runner:
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        return forward_output(
            [
                int(plan.token_ids[end - 1]) + 1
                for _, end in map(plan.token_range, range(plan.num_sequences))
            ]
        )


def samples(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Parse exported samples while respecting quoted newlines and backslashes."""
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)", line)
        assert match is not None, line
        name, raw_labels, value = match.groups()
        labels = tuple(
            sorted(
                (key, json.loads(quoted))
                for key, quoted in re.findall(
                    r'([a-zA-Z_]\w*)=("(?:[^"\\]|\\.)*")', raw_labels or ""
                )
            )
        )
        key = (name, labels)
        assert key not in result, f"duplicate sample: {line}"
        result[key] = float(value)
    return result


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
        assert snapshot is not None
        assert snapshot.runner is None
        assert snapshot.cache is None
        assert snapshot.requests.accepted_requests == 0


def test_snapshots_are_independent_and_native_fields_are_read_only():
    with Engine(Runner(), enable_metrics=True) as engine:
        request = engine.submit(
            [1, 2], sampling_params=SamplingParams(max_output_tokens=2)
        )
        queued = engine.metrics_snapshot()
        assert queued is not None
        assert queued.requests.waiting_requests == 1
        engine.step()
        live = engine.metrics_snapshot()
        assert live is not None
        assert live.requests.running_requests == 1
        assert live.requests.waiting_requests == 0
        assert live.requests.decode_tokens == 1
        assert live.requests.generation_tokens == 0
        assert live.requests.completed_requests == 0
        assert queued.requests.decode_tokens == 0
        assert queued.requests.waiting_requests == 1
        with pytest.raises(AttributeError):
            setattr(live.requests, "decode_tokens", 100)
        live.requests.queue_time_seconds.counts.clear()
        assert live.requests.queue_time_seconds.counts
        engine.step()
        assert request.done
        completed = engine.metrics_snapshot()
        assert completed is not None
        assert completed.requests.generation_tokens == 2
        assert completed.requests.completed_requests == 1
        assert completed.requests.prompt_tokens_histogram.sum == 2
        assert completed.requests.queue_time_seconds.count == 1
        assert completed.requests.request_time_per_output_token_seconds.count == 1


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
        assert values.accepted_requests == 2
        assert values.completed_requests == 1
        assert values.aborted_requests == 1
        assert values.decode_tokens == 4
        assert values.generation_tokens == 3
        assert values.prefill_compute_tokens == 2
        assert values.prompt_tokens == 1
        assert values.generation_tokens_histogram.sum == 3
        assert values.inter_token_latency_seconds.count == 2
        assert values.time_to_first_token_seconds.count == 2
        assert values.request_time_per_output_token_seconds.count == 1
        assert values.running_requests == values.waiting_requests == 0


def test_prompt_scoring_without_output_has_no_ttft_or_tpot_observations():
    with Engine(Runner(), enable_metrics=True) as engine:
        request = engine.submit(
            [1],
            sampling_params=SamplingParams(max_output_tokens=0, prompt_logprobs=True),
        )
        assert engine.step() == []
        assert request.take_result() == []
        snapshot = engine.metrics_snapshot()
        assert snapshot is not None
        values = snapshot.requests
        assert values.completed_requests == 1
        assert values.prefill_compute_tokens == values.prompt_tokens == 1
        assert values.decode_tokens == values.generation_tokens == 0
        assert values.time_to_first_token_seconds.count == 0
        assert values.inter_token_latency_seconds.count == 0
        assert values.request_time_per_output_token_seconds.count == 0


def test_runner_observation_hook_precedes_resource_binding_and_is_read_once():
    class ObservableRunner(Runner):
        runner_config = RunnerConfig(cache_layout=CacheLayout(4, [CacheGroupLayout(2)]))

        def __init__(self):
            self.enabled = False
            self.bound = False
            self.reads = 0

        def configure_metrics(self, *, enabled: bool = True):
            assert not self.bound
            self.enabled = enabled

        def bind_resources(self, binding: ResourceBinding):
            assert self.enabled
            self.bound = True

        def metrics_snapshot(self):
            self.reads += 1
            return RunnerMetrics(
                forward_seconds={"decode": 0.25},
                cuda_graph_passes={"decode_cuda_graph": 2},
                speculative_drafts=12,
                speculative_accepted=9,
                speculative_rows=4,
            )

    runner = ObservableRunner()
    with Engine(runner, enable_metrics=True) as engine:
        assert runner.enabled and runner.bound
        snapshot = engine.metrics_snapshot()
        assert snapshot is not None
        assert runner.reads == 1
        assert snapshot.cache is not None
        assert snapshot.cache.capacity_slabs == 4
        assert snapshot.cache.free_slabs == 4
        assert snapshot.runner is not None
        assert snapshot.runner.forward_seconds == {"decode": 0.25}
        exported = samples(snapshot.prometheus(model_name="test"))
        by_name = {name: value for (name, _), value in exported.items()}
        assert by_name["resin:forward_execution_seconds_total"] == 0.25
        assert by_name["resin:cuda_graph_passes_total"] == 2
        assert by_name["resin:spec_accept_rate"] == 0.75
        assert by_name["resin:spec_accept_length"] == 3.25
        assert runner.reads == 1


def test_prometheus_labels_histograms_and_live_counters_are_well_formed():
    with Engine(Runner(), enable_metrics=True) as engine:
        engine.submit([1, 2], sampling_params=SamplingParams(max_output_tokens=2))
        while engine.has_work():
            engine.step()
        snapshot = engine.metrics_snapshot()
        assert snapshot is not None
        name = 'model"\\line\nnext'
        text = snapshot.prometheus(model_name=name, rank=3, attn_dp_rank=1)
        parsed = samples(text)

        def values(metric: str, **labels: str) -> list[float]:
            return [
                value
                for (sample_name, sample_labels), value in parsed.items()
                if sample_name == metric
                and all(
                    dict(sample_labels).get(key) == val for key, val in labels.items()
                )
            ]

        assert values("resin:num_requests_total", model_name=name) == [1]
        assert values("resin:prompt_tokens_total", model_name=name) == [2]
        assert values("resin:generation_tokens_total", model_name=name) == [2]
        assert values("resin:realtime_tokens_total", mode="prefill_compute") == [2]
        assert values("resin:realtime_tokens_total", mode="decode") == [2]
        assert values("resin:prompt_tokens_histogram_count") == [1]
        assert values("resin:prompt_tokens_histogram_sum") == [2]
        assert values("resin:prompt_tokens_histogram_bucket", le="+Inf") == [1]
        finite = sorted(
            (float(dict(labels)["le"]), value)
            for (metric, labels), value in parsed.items()
            if metric == "resin:prompt_tokens_histogram_bucket"
            and dict(labels)["le"] != "+Inf"
        )
        # Text is a diagnostic view; full distributions travel in native protobuf.
        assert finite == []
        encoded = snapshot.prometheus_protobuf(model_name=name, rank=3, attn_dp_rank=1)
        assert isinstance(encoded, bytes) and encoded


def test_throughput_uses_live_output_and_retains_the_last_collection_window():
    with Engine(Runner(), enable_metrics=True) as engine:
        engine.submit([1], sampling_params=SamplingParams(max_output_tokens=8))
        engine.step()
        time.sleep(1.01)
        snapshot = engine.metrics_snapshot()
        assert snapshot is not None
        assert snapshot.requests.generation_tokens == 0
        assert snapshot.requests.decode_tokens == 1
        assert 0 < snapshot.generation_throughput <= 1
        following = engine.metrics_snapshot()
        assert following is not None
        assert following.generation_throughput == snapshot.generation_throughput
