"""Optional GPU observations with one event generation per execution."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import ParamSpec, TypeVar

import torch

from resin import BatchPhase
from resin._resin import _HistogramAccumulator
from resin.backend.metrics import (
    CUDA_GRAPH_MODES,
    FORWARD_CATEGORIES,
    RunnerMetrics,
)


def _cuda_result(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f"CUDA graph timing operation failed: {error}")
    return values


@dataclass(slots=True)
class _PhaseMeasurement:
    owner: GraphPhaseTiming
    events: tuple[torch.cuda.Event, ...]

    def mark(self, index: int) -> None:
        self.events[index].record(torch.cuda.current_stream(self.owner.device))

    def collect(self) -> dict[str, float]:
        seconds: dict[str, float] = defaultdict(float)
        for category, start, end in zip(
            self.owner.categories, self.events, self.events[1:]
        ):
            seconds[category] += start.elapsed_time(end) / 1000
        self.owner.available.append(self.events)
        return dict(seconds)


class GraphPhaseTiming:
    """Captured phase boundaries with independent event generations per replay.

    Updating a graph executable's event-record node affects future launches
    only. An already queued replay keeps its event generation until collection,
    so overlapped execution needs neither a synchronization nor extra kernels.
    """

    def __init__(self, categories: tuple[str, ...], *, device) -> None:
        self.device = device
        self.categories = categories
        self.events = tuple(
            torch.cuda.Event(enable_timing=True, external=True)
            for _ in range(len(categories) + 1)
        )
        self.available: list[tuple[torch.cuda.Event, ...]] = []
        self._nodes = None

    def mark(self, index: int) -> None:
        self.events[index].record(torch.cuda.current_stream(self.device))

    def _bind(self, graphs, events) -> None:
        from cuda.bindings import runtime

        if self._nodes is None:
            indices = {
                event.cuda_event: index for index, event in enumerate(self.events)
            }
            self._nodes = []
            for graph in graphs:
                if graph is None:
                    continue
                raw = graph.raw_cuda_graph()
                _, count = _cuda_result(runtime.cudaGraphGetNodes(raw, numNodes=0))
                nodes, _ = _cuda_result(runtime.cudaGraphGetNodes(raw, numNodes=count))
                for node in nodes:
                    (kind,) = _cuda_result(runtime.cudaGraphNodeGetType(node))
                    if kind != runtime.cudaGraphNodeType.cudaGraphNodeTypeEventRecord:
                        continue
                    (event,) = _cuda_result(
                        runtime.cudaGraphEventRecordNodeGetEvent(node)
                    )
                    if int(event) in indices:
                        self._nodes.append(
                            (graph.raw_cuda_graph_exec(), node, indices[int(event)])
                        )
            if len(self._nodes) != len(self.events):
                raise RuntimeError(
                    "every timing boundary must be captured exactly once"
                )
        for graph, node, index in self._nodes:
            _cuda_result(
                runtime.cudaGraphExecEventRecordNodeSetEvent(
                    graph, node, events[index].cuda_event
                )
            )

    def replay(self, graphs) -> _PhaseMeasurement:
        if self.available:
            events = self.available.pop()
        else:
            events = tuple(torch.cuda.Event(enable_timing=True) for _ in self.events)
            # Torch lazily allocates events on first record. Initialize each new
            # generation once; subsequent replays recycle completed generations.
            for event in events:
                event.record(torch.cuda.current_stream(self.device))
        self._bind(graphs, events)
        return _PhaseMeasurement(self, events)

    def prepare(self, graphs) -> None:
        """Discover captured marker nodes during startup, outside inference."""
        self._bind(graphs, self.events)

    def unobserved(self, graphs) -> None:
        # An unobserved launch must not overwrite the last observed generation,
        # which can still be pending when a benchmark calls a graph directly.
        if self._nodes is not None:
            self._bind(graphs, self.events)


@dataclass(slots=True)
class _Execution:
    category: str
    mode: str
    start: torch.cuda.Event
    end: torch.cuda.Event
    graphs: set[str] = field(default_factory=set)
    phases: list[_PhaseMeasurement] = field(default_factory=list)


class CudaMetrics:
    """Poll completed owned events; never wait for device work during a scrape.

    Events are recorded outside captured graphs. Every pending execution owns
    its pair, so an overlapping replay cannot overwrite an earlier timestamp.
    The active scope also collapses many prefill region replays into one pass.
    """

    def __init__(self, device, *, speculation=None) -> None:
        self.device = device
        self.active: _Execution | None = None
        self.pending: deque[_Execution] = deque()
        self.available: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.seconds = dict.fromkeys(FORWARD_CATEGORIES, 0.0)
        self.graphs: Counter[str] = Counter(dict.fromkeys(CUDA_GRAPH_MODES, 0))
        self.dp_attention_sync_wait_seconds = _HistogramAccumulator()
        self._eager_phases: dict[tuple[str, ...], GraphPhaseTiming] = {}
        self._speculation_baseline = (
            (0, 0, 0)
            if speculation is None
            else (speculation.drafts, speculation.accepted, speculation.blocks)
        )

    def eager_phases(self, categories: tuple[str, ...]) -> _PhaseMeasurement | None:
        if self.active is None:
            return None
        owner = self._eager_phases.get(categories)
        if owner is None:
            owner = self._eager_phases[categories] = GraphPhaseTiming(
                categories, device=self.device
            )
        events = (
            owner.available.pop()
            if owner.available
            else tuple(torch.cuda.Event(enable_timing=True) for _ in owner.events)
        )
        measurement = _PhaseMeasurement(owner, events)
        self.active.phases.append(measurement)
        return measurement

    def begin(self, phase: BatchPhase, *, idle: bool) -> _Execution:
        self.collect()
        mode = "prefill" if phase == BatchPhase.PREFILL else "decode"
        start, end = (
            self.available.pop()
            if self.available
            else (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        )
        execution = _Execution("idle" if idle else mode, mode, start, end)
        execution.start.record(torch.cuda.current_stream(self.device))
        self.active = execution
        return execution

    def finish(self, execution: _Execution) -> None:
        execution.end.record(torch.cuda.current_stream(self.device))
        self.active = None
        self.pending.append(execution)
        suffix = "cuda_graph" if execution.graphs else "none"
        self.graphs[f"{execution.mode}_{suffix}"] += 1

    def graph(self, mode: str) -> None:
        if self.active is not None:
            self.active.graphs.add(mode)

    def collect(self) -> None:
        while self.pending and self.pending[0].end.query():
            execution = self.pending.popleft()
            total = execution.start.elapsed_time(execution.end) / 1000
            detailed = 0.0
            for measurement in execution.phases:
                phases = measurement.collect()
                if execution.category != "idle":
                    for category, seconds in phases.items():
                        self.seconds[category] += seconds
                    detailed += sum(phases.values())
            self.seconds[execution.category] += max(0.0, total - detailed)
            self.available.append((execution.start, execution.end))

    def snapshot(self, speculation=None) -> RunnerMetrics:
        self.collect()
        drafts, accepted, rows = self._speculation_baseline
        return self.dp_attention_sync_wait_seconds.attach(
            RunnerMetrics(
                forward_execution_seconds=self.seconds,
                cuda_graph_passes=self.graphs,
                spec_draft_tokens=0
                if speculation is None
                else speculation.drafts - drafts,
                spec_accepted_tokens=0
                if speculation is None
                else speculation.accepted - accepted,
                spec_verify_rows=0
                if speculation is None
                else speculation.blocks - rows,
            )
        )


P = ParamSpec("P")
T = TypeVar("T")


def observe_execution(method: Callable[P, T]) -> Callable[P, T]:
    """Observe a runner entry point, including enqueue-to-forward fallbacks."""

    @wraps(method)
    def observed(*args: P.args, **kwargs: P.kwargs) -> T:
        runner = args[0]
        plan = args[1] if len(args) > 1 else kwargs["plan"]
        metrics = getattr(runner, "_metrics", None)
        if metrics is None or metrics.active is not None:
            return method(*args, **kwargs)
        context = args[2] if len(args) > 2 else kwargs.get("context")
        phase = getattr(context if context is not None else plan, "phase")
        execution = metrics.begin(phase, idle=plan is None)
        try:
            result = method(*args, **kwargs)
        except BaseException:
            # Failed kernels may poison CUDA; a monitoring query must not
            # become a second synchronization/error path for that execution.
            metrics.active = None
            raise
        metrics.finish(execution)
        return result

    return observed
