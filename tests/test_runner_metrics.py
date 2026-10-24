"""Execution metrics count passes once and never synchronize the device."""

from types import SimpleNamespace

import pytest

from resin import BatchPhase
from resin.backend.cuda.metrics import CudaMetrics, observe_execution
from resin.backend.metrics import DP_SYNC_BUCKETS, DpSyncMetrics


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
    assert not any(metrics.snapshot().forward_seconds.values())
    first.end.ready = True
    assert metrics.snapshot().forward_seconds["decode"] == 0.005
    assert metrics.snapshot().forward_seconds["decode"] == 0.005
    second.end.ready = True
    assert metrics.snapshot().forward_seconds["decode"] == 0.010


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
    assert snapshot.forward_seconds["idle"] == 0.005
    assert snapshot.cuda_graph_passes["decode_none"] == 1


def test_failed_execution_is_not_queried(metrics):
    class Runner:
        _metrics = metrics

        @observe_execution
        def forward(self, plan):
            raise RuntimeError("CUDA failure")

    with pytest.raises(RuntimeError, match="CUDA failure"):
        Runner().forward(plan=SimpleNamespace(phase=BatchPhase.DECODE))
    assert metrics.active is None
    assert not metrics.pending
    assert not any(metrics.snapshot().forward_seconds.values())


def test_speculation_snapshot_uses_raw_acceptance_counters(metrics):
    speculation = SimpleNamespace(drafts=12, accepted=9, blocks=4)
    snapshot = metrics.snapshot(speculation)
    assert (snapshot.speculative_drafts, snapshot.speculative_accepted) == (12, 9)
    assert snapshot.speculative_rows == 4


def test_dp_wait_histogram_retains_cumulative_buckets():
    metrics = DpSyncMetrics()
    metrics.observe(0.0002)
    metrics.observe(0.02)
    metrics.observe(2.0)
    assert metrics.seconds == pytest.approx(2.0202)
    assert metrics.count == 3
    assert metrics.buckets == [
        sum(value <= bound for value in (0.0002, 0.02, 2.0))
        for bound in DP_SYNC_BUCKETS
    ]


def test_completed_outer_events_are_recycled(metrics):
    first = metrics.begin(BatchPhase.DECODE, idle=False)
    metrics.finish(first)
    first.end.ready = True
    metrics.snapshot()
    second = metrics.begin(BatchPhase.DECODE, idle=False)
    assert (second.start, second.end) == (first.start, first.end)


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
    assert snapshot.forward_seconds["draft"] == 0.005
    assert snapshot.forward_seconds["target_verify"] == 0.005
    assert snapshot.forward_seconds["decode"] == 0.010
    metrics.begin(BatchPhase.DECODE, idle=False)
    reused = metrics.eager_phases(("draft", "target_verify"))
    assert reused is not None
    assert reused.events == phases.events


def test_eager_markers_use_runner_device_not_thread_current_device(metrics):
    metrics.device = "cuda:1"
    metrics.begin(BatchPhase.DECODE, idle=False)
    phases = metrics.eager_phases(("draft", "target_verify"))
    assert phases is not None
    phases.mark(0)
    assert phases.events[0].stream == "stream:cuda:1"


def test_speculation_starts_at_metrics_enable_boundary():
    speculation = SimpleNamespace(drafts=20, accepted=12, blocks=5)
    metrics = CudaMetrics("cuda", speculation=speculation)
    assert metrics.snapshot(speculation).speculative_drafts == 0
    speculation.drafts += 4
    speculation.accepted += 3
    speculation.blocks += 1
    snapshot = metrics.snapshot(speculation)
    assert (
        snapshot.speculative_drafts,
        snapshot.speculative_accepted,
        snapshot.speculative_rows,
    ) == (4, 3, 1)
