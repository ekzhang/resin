"""Execution metrics count passes once and never synchronize the device."""

from types import SimpleNamespace

import pytest

from resin import BatchPhase, RunnerMetrics
from resin._resin import _HistogramAccumulator
from resin.backend.cuda.metrics import CudaMetrics, observe_execution


class Event:
    def __init__(self, **_):
        self.ready = False
        self.recorded = False

    def record(self, _stream=None):
        self.recorded = True
        self.stream = _stream

    def query(self):
        return self.ready

    def elapsed_time(self, end):
        assert self.recorded and end.recorded and end.ready
        return 5.0

    def synchronize(self):
        raise AssertionError("metrics must never synchronize a CUDA event")


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.setattr("torch.cuda.Event", Event)
    monkeypatch.setattr("torch.cuda.current_stream", lambda device: f"stream:{device}")
    return CudaMetrics("cuda")


def test_pending_execution_events_are_owned_and_published_once(metrics):
    first = metrics.begin(BatchPhase.DECODE, idle=False)
    metrics.finish(first)
    second = metrics.begin(BatchPhase.DECODE, idle=False)
    metrics.finish(second)
    assert first.start is not second.start
    assert first.end is not second.end
    assert not any(metrics.snapshot().forward_execution_seconds.values())
    first.end.ready = True
    assert metrics.snapshot().forward_execution_seconds["decode"] == 0.005
    assert metrics.snapshot().forward_execution_seconds["decode"] == 0.005
    second.end.ready = True
    assert metrics.snapshot().forward_execution_seconds["decode"] == 0.010


def test_prefill_regions_and_nested_forward_count_as_one_pass(metrics):
    class Runner:
        _metrics = metrics

        @observe_execution
        def forward(self, plan):
            metrics.graph("prefill")
            metrics.graph("prefill")

        @observe_execution
        def enqueue(self, plan):
            return self.forward(plan)

    Runner().enqueue(SimpleNamespace(phase=BatchPhase.PREFILL))
    assert len(metrics.pending) == 1
    assert metrics.snapshot().cuda_graph_passes["prefill_cuda_graph"] == 1


def test_placeholder_participation_has_its_own_time_category(metrics):
    class Runner:
        _metrics = metrics

        @observe_execution
        def forward_distributed(self, plan, context):
            return None

    Runner().forward_distributed(None, SimpleNamespace(phase=BatchPhase.DECODE))
    metrics.pending[0].end.ready = True
    snapshot = metrics.snapshot()
    assert snapshot.forward_execution_seconds["idle"] == 0.005
    assert snapshot.cuda_graph_passes["decode_none"] == 1


def test_histogram_snapshot_merge_and_weighted_observations():
    first, second = _HistogramAccumulator(), _HistogramAccumulator()
    first.observe(0, weight=2)
    first.observe(0.125, weight=3)
    second.observe(2e12)
    left, right = first.snapshot(), second.snapshot()
    runner = RunnerMetrics(dp_attention_sync_wait_seconds=left)
    merged = second.attach(runner).as_dict()["dp_attention_sync_wait_seconds"]
    assert merged["count"] == 6
    assert merged["zero_count"] == 2
    assert merged["sum"] == 2e12 + 0.375
    assert sum(merged["counts"]) == 4
    assert left["count"] == 5 and right["count"] == 1
    assert first.snapshot() == left
    assert runner == RunnerMetrics(**runner.as_dict())


def test_eager_phases_are_exclusive_and_recycle_completed_events(metrics, monkeypatch):
    execution = metrics.begin(BatchPhase.DECODE, idle=False)
    phases = metrics.eager_phases(("draft", "target_verify"))
    assert phases is not None
    for index, event in enumerate(phases.events):
        phases.mark(index)
        event.ready = True
    metrics.finish(execution)
    execution.end.ready = True
    monkeypatch.setattr(execution.start, "elapsed_time", lambda _: 20.0)
    snapshot = metrics.snapshot()
    assert snapshot.forward_execution_seconds["draft"] == 0.005
    assert snapshot.forward_execution_seconds["target_verify"] == 0.005
    assert snapshot.forward_execution_seconds["decode"] == 0.010
    metrics.begin(BatchPhase.DECODE, idle=False)
    reused = metrics.eager_phases(("draft", "target_verify"))
    assert reused is not None
    assert reused.events == phases.events
