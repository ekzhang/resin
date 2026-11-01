from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from output_helpers import forward_output

from resin import (
    BatchPhase,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    Parallelism,
    RunnerConfig,
    SamplingParams,
    SchedulerConfig,
)
from resin.backend.speculative import SpeculativeStats
from resin.backend.torch import benchmark_expert_indices
from resin.bench import (
    BenchCase,
    BenchmarkRunner,
    distributed_reducer,
    run_sweep,
)


class Runner:
    vocab_size = 1024

    def __init__(self, *, chunked=False, drafts=0):
        self.runner_config = RunnerConfig(
            max_batch_tokens=5 if chunked else None,
            prefill_chunk_size=3 if chunked else None,
            continuation_prefill=chunked,
            max_accepted_per_step=drafts + 1,
            cache_layout=CacheLayout(parent_capacity=256, groups=[CacheGroupLayout(2)])
            if drafts
            else None,
        )
        self.speculative_draft_tokens = drafts
        self._speculation = SpeculativeStats(drafts)
        self.plans = []
        self.processed_counts = []

    def bind_resources(self, resources):
        pass

    def forward(self, plan):
        self.plans.append(plan)
        if plan.phase == BatchPhase.DECODE and self.speculative_draft_tokens:
            drafts = [[2] * self.speculative_draft_tokens for _ in plan.sequence_ids]
            targets = [
                [9] * (self.speculative_draft_tokens + 1) for _ in plan.sequence_ids
            ]
            acceptance = self._speculation.accept(drafts, targets)
            self.processed_counts.append(acceptance.processed_counts)
            return forward_output(acceptance.token_rows)
        return forward_output([0] * plan.num_sequences)


def test_distributed_benchmark_forwards_preparation_without_rebuilding_it():
    prepared, context = object(), object()
    output = forward_output([3])

    class PreparedRunner(Runner):
        def forward_distributed(self, plan, step_context, *, prepared):
            assert step_context is context
            assert prepared is preparation
            return output

    preparation = prepared
    runner = BenchmarkRunner(PreparedRunner())
    assert runner.forward_distributed(None, context, prepared=prepared) is output
    assert runner._plan is None


def test_benchmark_does_not_wrap_noncallable_result_as_pending():
    class InvalidRunner(Runner):
        def enqueue(self, plan, *, feedback=None):
            return SimpleNamespace(result=[])

    runner = InvalidRunner()
    runner.runner_config = RunnerConfig(
        cache_layout=CacheLayout(parent_capacity=16, groups=[CacheGroupLayout(128)]),
        max_accepted_per_step=3,
    )
    engine = Engine(BenchmarkRunner(runner))
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=8))
    assert [event.token_id for event in engine.step()] == [0]
    with pytest.raises(
        TypeError, match=r"runner.enqueue\(\) must return ForwardOutput"
    ):
        engine.step()
    assert engine.poisoned and not engine.overlap_pending and not request.done


def test_prefill_measures_actual_chunked_inputs_and_keeps_engine_resident():
    raw = Runner(chunked=True)
    runner = BenchmarkRunner(raw)
    engine = Engine(
        runner, max_running_requests=2, context_length=64, prefix_cache=True
    )
    records = run_sweep(
        engine,
        [BenchCase(phase="prefill", context_length=7, batch_size=2)],
        runner=runner,
        prompt_tokens=[1, 2, 3],
        warmup_steps=1,
        measure_steps=1,
        metadata={"model": "test"},
    )
    record = records[0]
    assert record["input_tokens"] == 14
    assert record["prefill_tokens"] == 14 / record["measure_steps"]
    assert record["measure_steps"] > 1  # A whole cohort spans several steps.
    assert record["warmup_steps"] == record["measure_steps"]
    assert record["committed_tokens"] == 2
    assert record["measured_global_batch_size_min"] == 2
    assert record["measured_global_batch_size_max"] == 2
    assert record["window_seconds"] == sum(record["step_seconds"])
    assert record["prefill_seconds"] == [record["window_seconds"]]
    assert record["metadata"] == {"model": "test"}
    assert not engine.has_work()
    # run_sweep never closes a caller-owned resident engine.
    assert (
        engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=1)
        ).cached_tokens
        == 0
    )


@pytest.mark.parametrize("budget,actual_batch", [(4, 1), (16, 4)])
def test_prefill_cohort_budget_and_gpt_capacity_cover_the_measured_batch(
    monkeypatch, budget, actual_batch
):
    from resin_models_cuda.gpt_oss import cuda
    from scripts.bench import _runner

    class GptRunner(Runner):
        prefill_chunk_size = 1

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.runner_config = RunnerConfig(
                max_batch_tokens=self.prefill_chunk_size,
                prefill_chunk_size=self.prefill_chunk_size,
                continuation_prefill=True,
            )

    monkeypatch.setattr(cuda, "GptOssCudaRunner", GptRunner)
    cases = [BenchCase(phase="prefill", context_length=4, batch_size=4)]
    raw = _runner(
        "gpt-oss-20b",
        "unused",
        device="cpu",
        groups=None,
        cases=cases,
        options=dict(
            prefill_chunk_size=4,
            prefill_graph="breakable",
            decode_graph="disabled",
            gpu_memory_utilization=0.9,
            draft_tokens=0,
        ),
        context_capacity=64,
    )
    assert raw.runner_config.max_batch_tokens == 16
    runner = BenchmarkRunner(raw)
    engine = Engine(
        runner,
        max_running_requests=4,
        context_length=64,
        scheduler_config=SchedulerConfig(max_batch_tokens=budget, prefill_chunk_size=4),
    )
    record = run_sweep(
        engine,
        cases,
        runner=runner,
        prompt_tokens=[1],
        warmup_steps=0,
        measure_steps=1,
    )[0]
    assert record["input_tokens"] == 16
    assert record["prefill_tokens"] == budget
    assert record["measure_steps"] == 4 // actual_batch
    assert record["measured_global_batch_size_min"] == actual_batch
    assert record["measured_global_batch_size_max"] == actual_batch


@pytest.mark.parametrize(
    "world,batch,error",
    [
        (1, 3, "single-GPU GPT-OSS prefill budget 12288.*8192-token limit"),
        (1, 2, "requires 1 visible GPUs"),
        (2, 3, "requires 2 visible GPUs"),
    ],
)
def test_gpt_single_gpu_prefill_budget_respects_normal_kernel_capacity(
    monkeypatch, world, batch, error
):
    from scripts.bench import run_benchmark

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match=error):
        run_benchmark(
            "gpt-oss-20b",
            "unused",
            [dict(phase="prefill", context_length=128, batch_size=batch)],
            parallelism=Parallelism(world_size=world),
            draft_tokens=0,
            accepted_drafts=0,
            prefill_chunk_size=4096,
        )


def test_decode_counts_real_steps_and_keeps_cohort_until_window_completes():
    runner = BenchmarkRunner(Runner())
    engine = Engine(runner, max_running_requests=2, context_length=64)
    record = run_sweep(
        engine,
        [BenchCase(phase="decode", context_length=4, batch_size=2)],
        runner=runner,
        prompt_tokens=[1],
        warmup_steps=2,
        measure_steps=3,
    )[0]
    assert record["measure_steps"] == 3
    assert record["committed_tokens"] == 6
    assert (
        record["measured_context_start_min"]
        == record["measured_context_start_max"]
        == 6
    )
    assert record["measured_context_end_min"] == record["measured_context_end_max"] == 9
    assert record["input_tokens"] == 0
    assert record["measured_accept_rate"] is None
    assert not engine.has_work()


def test_overlap_benchmark_defers_reduction_and_observes_harvested_positions():
    from test_engine_overlap import AsyncRunner

    class PipelinedRunner(AsyncRunner):
        vocab_size = 1024
        speculative_draft_tokens = 2

        def __init__(self):
            super().__init__()
            self.harvested = 0

        def enqueue(self, plan, *, feedback=None):
            completion = super().enqueue(plan, feedback=feedback)
            completion.positions = torch.tensor(
                plan.positions.tolist() if feedback is None else feedback.positions
            )
            result = completion.result

            def harvest():
                rows = result()
                self.harvested += 1
                return rows

            completion.result = harvest
            return completion

    raw = PipelinedRunner()
    runner = BenchmarkRunner(raw)
    engine = Engine(runner, max_running_requests=2, context_length=512)
    reductions, synchronizations = [], []

    def reduce_step(value):
        reductions.append(raw.harvested)
        return value

    try:
        record = run_sweep(
            engine,
            [BenchCase(phase="decode", context_length=4, batch_size=2)],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=1,
            measure_steps=3,
            synchronize=lambda: synchronizations.append(raw.harvested),
            reduce_step=reduce_step,
        )[0]
        assert record["committed_tokens"] == 12
        assert record["measured_context_start_min"] == 6
        assert record["measured_context_end_max"] == 12
        # GPU work must not progress in untimed per-step synchronization or
        # reduction calls. All measured reductions happen after all harvests.
        assert synchronizations == []
        assert reductions == [0, 4, 4, 4]  # Prefill, then the measured window.
        assert engine.overlap_stats[1] >= 3
        assert record["overlap_counts"] == {"enqueues": 3, "continuations": 3}
        assert not engine.has_work()
    finally:
        engine.close()
        raw.shutdown()


@pytest.mark.parametrize("batch_size", [1, 3])
def test_decode_trace_uses_live_full_cohort_without_changing_measurements(
    monkeypatch, batch_size
):
    clock = iter(range(1000))
    monkeypatch.setattr("resin.bench.time.perf_counter", lambda: next(clock))
    case = BenchCase(phase="decode", context_length=4, batch_size=batch_size)
    traces = []

    class GraphRunner(Runner):
        @property
        def cuda_graph_stats(self):
            count = sum(plan.phase == BatchPhase.DECODE for plan in self.plans)
            return {
                "decode": {
                    "replay_count": count,
                    "eager_fallback_count": 0,
                    "fallback_counts": {},
                }
            }

    def run(traced):
        raw = GraphRunner(drafts=4)
        runner = BenchmarkRunner(raw, accepted_drafts=4)
        # Reserve the prompt, prefill sample and five maximum-acceptance
        # steps: warmup, two measured, then two traced completions.
        engine = Engine(runner, max_running_requests=batch_size, context_length=30)

        def trace(actual_case, step):
            assert actual_case is case
            assert engine.has_work() and engine.next_phase() == BatchPhase.DECODE
            assert engine.active_request_count == batch_size
            assert len(raw.plans) == 4  # Prefill, warmup, then two measured steps.
            traced_step = step()
            assert list(raw.plans[-1].context_lengths) == [19] * batch_size
            assert traced_step["sequences"] == batch_size
            assert traced_step["committed_tokens"] == 5 * batch_size
            traces.append(traced_step)
            second = step()
            assert list(raw.plans[-1].context_lengths) == [24] * batch_size
            assert second["committed_tokens"] == 5 * batch_size

        completed = []

        def on_result(record):
            assert not engine.has_work()
            completed.append(record)

        records = run_sweep(
            engine,
            [case],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=1,
            measure_steps=2,
            trace_step=trace if traced else None,
            on_result=on_result,
        )
        assert not engine.has_work()
        assert completed == records
        assert len(raw.plans) == (6 if traced else 4)
        return records

    baseline, profiled = run(False), run(True)
    assert baseline == profiled
    record = profiled[0]
    assert record["step_seconds"] == [1, 1]
    assert record["graph_counts"]["decode.replay_count"] == 2
    assert record["committed_tokens"] == 10 * batch_size
    assert record["measured_accept_rate"] == 1.0
    assert record["measured_context_start_min"] == 9
    assert record["measured_context_end_max"] == 19
    assert len(traces) == 1


def test_prefill_trace_runs_two_fresh_complete_cohorts_outside_measurements(
    monkeypatch,
):
    monkeypatch.setattr("resin.bench.time.perf_counter", iter(range(1000)).__next__)
    traces = []

    def run(traced):
        raw = Runner(chunked=True)
        runner = BenchmarkRunner(raw)
        engine = Engine(
            runner, max_running_requests=2, context_length=64, prefix_cache=True
        )
        case = BenchCase(phase="prefill", context_length=7, batch_size=2)

        def trace(actual_case, run_pass):
            assert actual_case is case and not engine.has_work()
            for _ in range(2):
                first_plan = len(raw.plans)
                result = run_pass()
                assert not engine.has_work()
                assert result["input_tokens"] == 14
                assert result["steps"] > 1
                assert raw.plans[first_plan].context_lengths.tolist() == [0, 0]
                assert all(
                    plan.phase == BatchPhase.PREFILL for plan in raw.plans[first_plan:]
                )
                traces.append(result)

        records = run_sweep(
            engine,
            [case],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=1,
            measure_steps=1,
            trace_step=trace if traced else None,
        )
        assert not engine.has_work()
        return records

    assert run(False) == run(True)
    assert len(traces) == 2


@pytest.mark.parametrize("traced", [False, True])
def test_zero_measure_steps_fail_before_any_work_or_tracing(traced):
    raw = Runner()
    runner = BenchmarkRunner(raw)
    engine = Engine(runner)
    traces = []
    try:
        with pytest.raises(ValueError, match="measure_steps must be an integer >= 1"):
            run_sweep(
                engine,
                [BenchCase(phase="decode", context_length=4, batch_size=1)],
                runner=runner,
                prompt_tokens=[1],
                warmup_steps=0,
                measure_steps=0,
                trace_step=(lambda *args: traces.append(args)) if traced else None,
            )
        assert not raw.plans and not traces and not engine.has_work()
    finally:
        engine.close()


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_trace_failure_cleans_up_requests(phase):
    runner = BenchmarkRunner(Runner(drafts=4), accepted_drafts=0)
    engine = Engine(runner, max_running_requests=2, context_length=64)
    completed = []

    def failing_trace(case, run_pass):
        assert case.phase == phase
        run_pass()
        if phase == "decode":
            assert engine.has_work()
        raise RuntimeError("trace export failed")

    with pytest.raises(RuntimeError, match="trace export failed"):
        run_sweep(
            engine,
            [BenchCase(phase=phase, context_length=4, batch_size=2)],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=0,
            measure_steps=1,
            trace_step=failing_trace,
            on_result=completed.append,
        )
    assert not completed
    assert not engine.has_work()
    assert engine.active_request_count == 0


@pytest.mark.parametrize("accepted", [0, 3, 4])
def test_fixed_acceptance_changes_engine_commit_and_model_rollback_depth(accepted):
    raw = Runner(drafts=4)
    runner = BenchmarkRunner(raw, accepted_drafts=accepted)
    engine = Engine(runner, max_running_requests=2, context_length=128)
    record = run_sweep(
        engine,
        [BenchCase(phase="decode", context_length=4, batch_size=2)],
        runner=runner,
        prompt_tokens=[1],
        warmup_steps=1,
        measure_steps=2,
    )[0]
    assert record["committed_tokens"] == 2 * 2 * (accepted + 1)
    assert record["measured_accept_rate"] == accepted / 4
    assert record["measured_context_start_min"] == 4 + accepted + 1
    assert record["measured_context_end_max"] == 4 + 3 * (accepted + 1)
    assert raw.processed_counts == [[accepted + 1] * 2] * 3
    decode = [p for p in raw.plans if p.phase == BatchPhase.DECODE]
    assert [list(p.context_lengths) for p in decode] == [
        [4 + step * (accepted + 1)] * 2 for step in range(3)
    ]


def test_balanced_routes_are_distinct_per_token_and_globally_even():
    scores = torch.randn(13, 8)
    indices = benchmark_expert_indices(scores, 3)
    assert indices.shape == (13, 3)
    assert all(len(set(row)) == 3 for row in indices.tolist())
    counts = indices.flatten().bincount(minlength=8)
    assert int(counts.max() - counts.min()) <= 1
    assert benchmark_expert_indices(scores[:0], 3).shape == (0, 3)
    routes = torch.cat(
        [benchmark_expert_indices(scores[:1], 3, rank) for rank in range(3)]
    )
    counts = routes.flatten().bincount(minlength=8)
    assert int(counts.max() - counts.min()) <= 1


@pytest.mark.parametrize(
    "experts,ep,top_k,batch,replicas",
    [
        (384, 4, 6, 1, 1),
        (32, 2, 4, 1, 1),
        (384, 4, 6, 1, 4),
        (32, 2, 4, 5, 3),
        (32, 2, 32, 1, 2),
    ],
)
def test_balanced_routes_cover_contiguous_owners_before_their_experts(
    experts, ep, top_k, batch, replicas
):
    scores = torch.zeros(batch, experts)
    routes = torch.cat(
        [
            benchmark_expert_indices(scores, top_k, replica, ep)
            for replica in range(replicas)
        ]
    )
    assert all(len(set(row)) == top_k for row in routes.tolist())
    for indices, count in ((routes, experts), (routes // (experts // ep), ep)):
        loads = indices.flatten().bincount(minlength=count)
        assert int(loads.max() - loads.min()) <= 1
    assert routes[0, : min(ep, top_k)].tolist() == [
        owner * (experts // ep) for owner in range(min(ep, top_k))
    ]


def test_graph_counters_exclude_prefill_and_decode_warmup():
    class GraphRunner(Runner):
        def __init__(self):
            super().__init__()
            self.replays = {"decode": 0, "prefill": 0}

        @property
        def cuda_graph_stats(self):
            return {
                mode: {
                    "replay_count": count,
                    "eager_fallback_count": count,
                    "fallback_counts": {"test": count},
                }
                for mode, count in self.replays.items()
            }

        def forward(self, plan):
            self.replays[
                "prefill" if plan.phase == BatchPhase.PREFILL else "decode"
            ] += 1
            return super().forward(plan)

    runner = BenchmarkRunner(GraphRunner())
    record = run_sweep(
        Engine(runner, context_length=64),
        [BenchCase(phase="decode", context_length=4, batch_size=1)],
        runner=runner,
        prompt_tokens=[1],
        warmup_steps=2,
        measure_steps=3,
    )[0]
    assert record["graph_counts"] == {
        f"{mode}.{name}": 3 if mode == "decode" else 0
        for mode in ("decode", "prefill")
        for name in ("replay_count", "eager_fallback_count", "fallback.test")
    }


def test_benchmark_graph_counts_avoid_diagnostic_catalogs():
    class GraphRunner(Runner):
        def __init__(self):
            super().__init__()
            self.replays = 0

        @property
        def cuda_graph_counts(self):
            return {
                "decode": {
                    "replay_count": self.replays,
                    "eager_fallback_count": 0,
                    "fallback_counts": {},
                }
            }

        @property
        def cuda_graph_stats(self):
            raise AssertionError("benchmark rebuilt per-shape graph diagnostics")

        def forward(self, plan):
            if plan.phase == BatchPhase.DECODE:
                self.replays += 1
            return super().forward(plan)

    runner = BenchmarkRunner(GraphRunner())
    record = run_sweep(
        Engine(runner, context_length=64),
        [BenchCase(phase="decode", context_length=4, batch_size=1)],
        runner=runner,
        prompt_tokens=[1],
        warmup_steps=2,
        measure_steps=3,
    )[0]
    assert record["graph_counts"] == {
        "decode.replay_count": 3,
        "decode.eager_fallback_count": 0,
    }


def test_partial_admission_is_canceled_on_failure():
    class EngineAdapter:
        def __init__(self, runner):
            self.local = Engine(runner, max_running_requests=2, context_length=64)
            self.calls = 0

        def submit(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("admission failed")
            return self.local.submit(*args, **kwargs)

        def cancel(self, lease):
            return self.local.cancel(lease)

    runner = BenchmarkRunner(Runner())
    engine = EngineAdapter(runner)
    with pytest.raises(ValueError, match="admission failed"):
        run_sweep(
            engine,
            [BenchCase(phase="prefill", context_length=4, batch_size=2)],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=0,
            measure_steps=1,
        )
    assert not engine.local.has_work()


@pytest.mark.parametrize(
    "drafts,targets", [([[1, 2]], [[3, 4, 5]]), ([[1, 2, 3]], [[4]])]
)
def test_fixed_acceptance_retains_shape_validation_without_updating_stats(
    drafts, targets
):
    stats = SpeculativeStats(4, benchmark_accepted_drafts=3)
    with pytest.raises(ValueError):
        stats.accept(drafts, targets)
    assert stats.blocks == stats.drafts == stats.accepted == stats.emitted == 0


def test_rejects_unsupported_or_incomplete_measurements():
    assert BenchmarkRunner(Runner(), accepted_drafts=0).accepted_drafts == 0
    with pytest.raises(ValueError, match="balanced experts"):
        BenchmarkRunner(Runner(), balanced_experts=True)
    for kwargs in ({"accepted_drafts": -1}, {"accepted_drafts": 5}):
        with pytest.raises(ValueError, match="accepted_drafts"):
            BenchmarkRunner(Runner(drafts=4), **kwargs)
    runner = BenchmarkRunner(Runner())
    engine = Engine(runner, max_running_requests=1, context_length=16)
    case = BenchCase(phase="decode", context_length=10, batch_size=1)
    with pytest.raises(ValueError, match="full decode window"):
        run_sweep(engine, [case], runner=runner, prompt_tokens=[1])
    assert not engine.has_work()
    with pytest.raises(ValueError, match="reducer"):
        run_sweep(
            engine, [case], runner=runner, prompt_tokens=[1], gpu_count=2, attn_dp=2
        )
    with pytest.raises(RuntimeError, match="BenchmarkRunner"):
        run_sweep(
            Engine(Runner(), context_length=64),
            [case],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=0,
            measure_steps=1,
        )


def test_driver_reduction_uses_slowest_time_and_unique_replica_totals():
    from types import SimpleNamespace

    class Distributed:
        @staticmethod
        def all_gather_object(values, local, *, group):
            values[:] = [
                local,
                {
                    **local,
                    "seconds": 2 * local["seconds"],
                    "device_phase_ms": {
                        key: 2 * value
                        for key, value in local["device_phase_ms"].items()
                    },
                },
            ]

    groups = SimpleNamespace(
        drivers=SimpleNamespace(size=2, handle=object()), distributed=Distributed()
    )
    reduced = distributed_reducer(groups)(
        {
            "seconds": 0.1,
            "device_phase_ms": {
                "draft": 5.0,
                "target_verify": 15.0,
                "device_total": 20.0,
            },
            "input_tokens": 4,
            "committed_tokens": 2,
            "overlap_counts": {"enqueues": 3, "continuations": 2},
            "drafts": 4,
            "accepted": 3,
            "sequences": 2,
            "context_start_min": 3,
            "context_start_max": 4,
            "context_end_min": 4,
            "context_end_max": 5,
        }
    )
    assert reduced == {
        "seconds": 0.2,
        "device_phase_ms": {"draft": 10.0, "target_verify": 30.0, "device_total": 40.0},
        "replica_sequences": [2, 2],
        "input_tokens": 8,
        "committed_tokens": 4,
        "drafts": 8,
        "accepted": 6,
        "sequences": 4,
        "graph_counts": {},
        "overlap_counts": {"enqueues": 6, "continuations": 4},
        "context_start_min": 3,
        "context_start_max": 4,
        "context_end_min": 4,
        "context_end_max": 5,
    }


def test_completed_case_is_saved_before_later_failure_and_requests_are_cleaned(
    tmp_path, capsys
):
    import json

    from scripts.bench import _result_callback

    runner = BenchmarkRunner(Runner())

    class FailingAdmission:
        def __init__(self):
            self.local = Engine(runner, max_running_requests=2, context_length=64)
            self.calls = 0

        def __getattr__(self, name):
            return getattr(self.local, name)

        def submit(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 3:
                raise ValueError("second case admission failed")
            return self.local.submit(*args, **kwargs)

    engine = FailingAdmission()
    output = tmp_path / "results.json"
    save = _result_callback(output)
    completed = []

    def on_result(record):
        # Decode leases have been canceled before user I/O begins.
        assert not engine.local.has_work()
        save(record)
        completed.append(record)

    with pytest.raises(ValueError, match="second case admission failed"):
        run_sweep(
            engine,
            [
                BenchCase(phase="decode", context_length=4, batch_size=1),
                BenchCase(phase="prefill", context_length=8, batch_size=2),
            ],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=0,
            measure_steps=2,
            on_result=on_result,
        )
    assert len(completed) == 1
    assert json.loads(output.read_text()) == completed
    assert not output.with_suffix(".json.tmp").exists()
    assert not engine.local.has_work()
    point, error = capsys.readouterr().out.strip().splitlines()
    assert point.startswith("[bench-result] ")
    assert json.loads(point.split(" ", 1)[1]) == completed[0]
    assert error.startswith("[bench-error] ")
    assert "second case admission failed" in json.loads(error.split(" ", 1)[1])["error"]


def test_decode_reports_when_cache_admission_only_fits_part_of_the_cohort(capsys):
    import json

    raw = Runner()
    raw.runner_config = RunnerConfig(
        cache_layout=CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(4)])
    )
    runner = BenchmarkRunner(raw)
    engine = Engine(runner, max_running_requests=4, context_length=64)
    with pytest.raises(RuntimeError, match="scheduled_global=2, expected_global=4"):
        run_sweep(
            engine,
            [BenchCase(phase="decode", context_length=4, batch_size=4)],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=0,
            measure_steps=1,
        )
    line = capsys.readouterr().out.strip()
    assert line.startswith("[bench-error] ")
    diagnostic = json.loads(line.split(" ", 1)[1])
    assert diagnostic["observation"]["sequences"] == 2
    assert diagnostic["cache"]["cache_parent_capacity"] == 4
    assert diagnostic["cache"]["ready_request_count"] == 4
    assert diagnostic["cache"]["active_request_count"] == 4
    assert not engine.has_work()


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_shared_prefix_is_seeded_once_and_forks_real_pages_for_private_work(phase):
    raw = Runner()
    raw.runner_config = RunnerConfig(
        cache_layout=CacheLayout(
            parent_capacity=4 if phase == "prefill" else 64,
            groups=[CacheGroupLayout(4)],
        ),
        continuation_prefill=True,
        prefill_chunk_size=4,
        max_batch_tokens=8,
    )
    runner = BenchmarkRunner(raw)
    engine = Engine(
        runner, max_running_requests=2, context_length=64, prefix_cache=True
    )
    record = run_sweep(
        engine,
        [
            BenchCase(
                phase=phase, context_length=12, batch_size=2, shared_prefix_tokens=8
            )
        ],
        runner=runner,
        prompt_tokens=[10],
        warmup_steps=1,
        measure_steps=1,
    )[0]
    assert (
        record["shared_prefix_tokens"]
        == record["cached_tokens_min"]
        == record["cached_tokens_max"]
        == 8
    )
    assert record["private_prompt_tokens"] == 4
    if phase == "prefill":
        assert record["input_tokens"] == record["prefill_tokens"] == 8
        assert record["measured_context_start_min"] == 8
        assert record["measured_context_end_max"] == 12
    else:
        assert record["measured_context_start_min"] == 13
        assert record["measured_context_end_max"] == 14
    # Two seed chunks, followed only by private suffixes on all later prefill rows.
    prefills = [plan for plan in raw.plans if plan.phase == BatchPhase.PREFILL]
    assert [plan.context_lengths.tolist() for plan in prefills[:2]] == [[0], [4]]
    assert all(plan.context_lengths.tolist() == [8, 8] for plan in prefills[2:])
    group = prefills[2].cache_groups[0]
    first, second = group.block_slab_ids[:3], group.block_slab_ids[3:]
    assert first[:2].tolist() == second[:2].tolist()  # Shared physical parents.
    assert first[2] != second[2]  # Private suffixes own distinct parents.
    assert not engine.has_work() and engine.cache_occupied_block_count == 0


def test_shared_prefix_requires_real_enabled_cache_and_exact_checkpoint_reuse():
    raw = Runner()
    raw.runner_config = RunnerConfig(
        cache_layout=CacheLayout(parent_capacity=64, groups=[CacheGroupLayout(4)]),
        continuation_prefill=True,
    )
    runner = BenchmarkRunner(raw)
    case = BenchCase(
        phase="prefill", context_length=12, batch_size=2, shared_prefix_tokens=7
    )
    for enabled, error in ((False, "require Engine"), (True, "requested=7, cached=4")):
        engine = Engine(
            runner, max_running_requests=2, context_length=64, prefix_cache=enabled
        )
        with pytest.raises(ValueError, match=error):
            run_sweep(
                engine,
                [case],
                runner=runner,
                prompt_tokens=[1],
                warmup_steps=0,
                measure_steps=1,
            )
        assert not engine.has_work()
