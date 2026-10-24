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
    def __init__(self):
        # Exercise the shared runtime without checkpoint or attention dependencies.
        self.device = torch.device("cuda", 0)
        self.cuda_graph_decode = "full"
        self.cuda_graph_prefill = "disabled"
        self._cuda_graph_executor = None
        self._cuda_failure = lambda: torch.cuda.synchronize(self.device)
        self._parallel_graph_model = None
        self._parallel_graph_groups = None
        self._metrics = None
        self.configure_metrics()
        self._configure_cuda_graphs(
            {"decode": CudaGraphMode(((1,),), self.build, lambda _: {})}
        )
        self.capture_cuda_graphs()

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
        assert not any(runner.metrics_snapshot().forward_execution_seconds.values())
        torch.cuda.synchronize()

        def no_synchronize(_):
            raise AssertionError("snapshot synchronized an event")

        monkeypatch.setattr(torch.cuda.Event, "synchronize", no_synchronize)
        snapshot = runner.metrics_snapshot()
        assert snapshot.cuda_graph_passes["decode_cuda_graph"] == 2
        assert snapshot.forward_execution_seconds["draft"] > 0
        assert snapshot.forward_execution_seconds["target_verify"] > 0
        assert snapshot.forward_execution_seconds["decode"] >= 0
        assert runner.metrics_snapshot() == snapshot
        assert not runner._metrics.pending
        # Completed event generations are reusable without changing old totals.
        runner.enqueue(plan)
        torch.cuda.synchronize()
        assert runner.metrics_snapshot().cuda_graph_passes["decode_cuda_graph"] == 3
    finally:
        runner.close_cuda_graphs()
