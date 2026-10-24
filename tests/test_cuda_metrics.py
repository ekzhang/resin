"""Real-device validation of asynchronous and captured phase observations."""

from types import SimpleNamespace

import pytest
import torch

from resin import BatchPhase
from resin.backend.cuda.graphs import CudaGraphMode, CudaGraphStep
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.runner import CudaRunner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class TimingRunner(CudaRunner):
    def __init__(self, *, captured_metrics=True):
        # Exercise the shared runtime without checkpoint or attention dependencies.
        self.device = torch.device("cuda", 0)
        self.cuda_graph_decode = "full"
        self.cuda_graph_prefill = "disabled"
        self._cuda_graph_executor = None
        self._cuda_failure = lambda: torch.cuda.synchronize(self.device)
        self._parallel_graph_model = None
        self._parallel_graph_groups = None
        self._metrics = None
        if captured_metrics:
            self.configure_metrics()
        self._configure_cuda_graphs(
            {"decode": CudaGraphMode(((1,),), self.build, lambda _: {})}
        )
        self.capture_cuda_graphs()
        self.configure_metrics()

    def forward(self, plan):
        raise NotImplementedError("the timing probe uses enqueue")

    def build(self, _):
        phases = self._new_phase_timing("draft", "target_verify")
        value = torch.ones((256, 256), device=self.device)
        output = torch.empty_like(value)

        def launch():
            if phases is not None:
                phases.mark(0)
            torch.mm(value, value, out=output)
            if phases is not None:
                phases.mark(1)
            torch.mm(value, value, out=output)
            if phases is not None:
                phases.mark(2)

        return CudaGraphStep(
            inputs={}, output=output, launch=launch, phase_timing=phases
        )

    @observe_execution
    def enqueue(self, plan):
        return self._run_cuda_graph("decode", (1,), {})

    @observe_execution
    def eager(self, plan):
        phases = self._eager_phase_timing("draft", "target_verify")
        assert phases is not None
        value = torch.ones((256, 256), device=self.device)
        output = torch.empty_like(value)
        phases.mark(0)
        torch.mm(value, value, out=output)
        phases.mark(1)
        torch.mm(value, value, out=output)
        phases.mark(2)
        return output


def test_captured_phase_events_belong_to_each_queued_replay(monkeypatch):
    runner = TimingRunner()
    assert runner._metrics is not None
    plan = SimpleNamespace(phase=BatchPhase.DECODE)
    try:
        # Keep the first replay outstanding while its successor is submitted.
        torch.cuda._sleep(50_000_000)
        runner.enqueue(plan)
        first = runner._metrics.pending[0]
        # An unobserved direct replay cannot overwrite that pending generation.
        runner._run_cuda_graph("decode", (1,), {})
        runner.enqueue(plan)
        second = runner._metrics.pending[1]
        assert first.phases and second.phases
        assert all(
            left is not right
            for left, right in zip(first.phases[0].events, second.phases[0].events)
        )
        assert not any(runner.metrics_snapshot().forward_seconds.values())
        torch.cuda.synchronize()

        def no_synchronize(_):
            raise AssertionError("snapshot synchronized an event")

        monkeypatch.setattr(torch.cuda.Event, "synchronize", no_synchronize)
        snapshot = runner.metrics_snapshot()
        assert snapshot.cuda_graph_passes["decode_cuda_graph"] == 2
        assert snapshot.forward_seconds["draft"] > 0
        assert snapshot.forward_seconds["target_verify"] > 0
        assert snapshot.forward_seconds["decode"] >= 0
        assert runner.metrics_snapshot() == snapshot
        assert not runner._metrics.pending
        # Completed event generations are reusable without changing old totals.
        runner.enqueue(plan)
        torch.cuda.synchronize()
        assert runner.metrics_snapshot().cuda_graph_passes["decode_cuda_graph"] == 3
    finally:
        runner.close_cuda_graphs()


def test_enabling_after_capture_keeps_accurate_whole_pass_timing():
    runner = TimingRunner(captured_metrics=False)
    try:
        runner.enqueue(SimpleNamespace(phase=BatchPhase.DECODE))
        torch.cuda.synchronize()
        snapshot = runner.metrics_snapshot()
        assert snapshot.forward_seconds["draft"] == 0
        assert snapshot.forward_seconds["target_verify"] == 0
        assert snapshot.forward_seconds["decode"] > 0
        assert snapshot.cuda_graph_passes["decode_cuda_graph"] == 1
    finally:
        runner.close_cuda_graphs()


def test_eager_phase_observations_use_completed_owned_events():
    runner = TimingRunner()
    assert runner._metrics is not None
    try:
        plan = SimpleNamespace(phase=BatchPhase.DECODE)
        torch.cuda._sleep(50_000_000)
        runner.eager(plan)
        runner.eager(plan)
        pending = tuple(runner._metrics.pending)
        assert len(pending) == 2
        assert all(
            event.device == runner.device for event in pending[0].phases[0].events
        )
        assert not any(runner.metrics_snapshot().forward_seconds.values())
        torch.cuda.synchronize()
        total = sum(item.start.elapsed_time(item.end) / 1000 for item in pending)
        snapshot = runner.metrics_snapshot()
        assert snapshot.forward_seconds["draft"] > 0
        assert snapshot.forward_seconds["target_verify"] > 0
        assert sum(snapshot.forward_seconds.values()) == pytest.approx(total, abs=1e-6)
        assert snapshot.cuda_graph_passes["decode_none"] == 2
    finally:
        runner.close_cuda_graphs()


def benchmark_observation_overhead(iterations=200):
    """Report warmed host+device pass cost for a tiny two-matmul graph."""
    from statistics import median
    from time import perf_counter

    result = {}
    plan = SimpleNamespace(phase=BatchPhase.DECODE)
    for name in ("disabled", "whole_pass", "captured_phases"):
        runner = TimingRunner(captured_metrics=name == "captured_phases")
        if name == "disabled":
            runner.configure_metrics(enabled=False)
        try:
            for _ in range(10):
                runner.enqueue(plan)
            torch.cuda.synchronize()
            rounds = []
            for _ in range(5):
                started = perf_counter()
                for _ in range(iterations):
                    runner.enqueue(plan)
                torch.cuda.synchronize()
                rounds.append((perf_counter() - started) * 1e6 / iterations)
            result[f"{name}_us_per_pass"] = median(rounds)
        finally:
            runner.close_cuda_graphs()
    return result
