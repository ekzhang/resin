"""Reusable CUDA graph execution for prefill, decode, and other modes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, NoReturn, NotRequired, TypedDict

import torch

from resin.backend.cuda.failures import GraphInputError, PostAcceptCudaError
from resin.backend.cuda.metrics import CudaMetrics, GraphPhaseTiming
from resin.backend.cuda.staging import StagingError, TensorStagerPool

GraphShape = tuple[int, ...]
DEFAULT_MAX_NUM_SEQS = 128
DEFAULT_DECODE_GRAPH_MAX_BATCH_SIZE = 1024

_SPECULATIVE_DECODE_BATCH_SIZES = (
    *range(1, 9),
    *range(12, 33, 4),
    *range(40, 65, 8),
    *range(96, 257, 32),
    *range(384, 1025, 128),
)
_DECODE_BATCH_SIZES = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    *range(256, 1025, 128),
)


class CudaGraphShapeStats(TypedDict):
    shape: GraphShape
    captured: bool
    capture_count: int
    replay_count: int
    warmup_count: int
    capture_ms: float | None
    capture_memory_bytes: int | None
    graph_region_count: int
    graph_break_count: int


class CudaGraphCounts(TypedDict):
    replay_count: int
    eager_fallback_count: int
    fallback_counts: dict[str, int]


class CudaGraphStats(TypedDict):
    capture_scope: NotRequired[str]
    captured_operations: NotRequired[tuple[str, ...]]
    enabled: bool
    configured_shapes: tuple[GraphShape, ...]
    captured_shapes: tuple[GraphShape, ...]
    capture_count: int
    replay_count: int
    graph_region_count: int
    graph_break_count: int
    eager_fallback_count: int
    fallback_counts: dict[str, int]
    shape_stats: tuple[CudaGraphShapeStats, ...]
    poisoned: bool
    poison_error: str | None


def _graph_shapes(value: object, *, dimensions: int) -> tuple[GraphShape, ...]:
    """Freeze positive integer CUDA graph shapes in deterministic order."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        raw = tuple(value.items())
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        raw = tuple(value)
    else:
        raise TypeError("CUDA graph shapes must be an iterable of tuples")
    shapes: set[GraphShape] = set()
    for item in raw:
        if not isinstance(item, tuple) or len(item) != dimensions:
            raise TypeError(
                f"each CUDA graph shape must be a {dimensions}-integer tuple"
            )
        if any(
            isinstance(part, bool) or not isinstance(part, int) or part <= 0
            for part in item
        ):
            raise ValueError("CUDA graph shape dimensions must be positive integers")
        if item in shapes:
            raise ValueError(f"duplicate CUDA graph shape: {item}")
        shapes.add(item)
    return tuple(sorted(shapes))


def default_decode_graph_shapes(
    width: int,
    *,
    speculative: bool,
    max_batch_size: int = DEFAULT_DECODE_GRAPH_MAX_BATCH_SIZE,
) -> tuple[GraphShape, ...]:
    """Return shared decode graph buckets with one dynamic-history capacity.

    Ordinary decode includes powers of two and their 1.5x intermediates.
    Speculative decode uses denser small-batch buckets because padding repeats
    the whole draft-and-verify path.
    The shared catalog extends through 1024 rows. Deployments should cap it at
    their scheduler's maximum number of sequences so every admitted decode
    batch has a compatible graph bucket.
    """

    for name, value in (("width", width), ("max_batch_size", max_batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    batch_sizes = (
        _SPECULATIVE_DECODE_BATCH_SIZES if speculative else _DECODE_BATCH_SIZES
    )
    selected = tuple(
        batch_size for batch_size in batch_sizes if batch_size <= max_batch_size
    )
    if selected[-1] != max_batch_size:
        selected = (*selected, max_batch_size)
    return tuple((batch_size, width) for batch_size in selected)


def resolve_decode_graph_shapes(
    value: object,
    *,
    policy: str,
    width: int,
    speculative: bool,
    max_batch_size: int,
) -> tuple[GraphShape, ...]:
    """Resolve automatic buckets while preserving an explicit empty override."""

    if policy not in ("disabled", "full"):
        raise ValueError("CUDA decode graph policy must be 'disabled' or 'full'")
    if value is None and policy != "disabled":
        return default_decode_graph_shapes(
            width, speculative=speculative, max_batch_size=max_batch_size
        )
    shapes = _graph_shapes(value, dimensions=2)
    if policy == "disabled" and shapes:
        raise ValueError("disabled CUDA graphs cannot have nonempty graph shapes")
    return shapes


@dataclass(frozen=True, slots=True)
class CudaGraphStep:
    """One fixed-address operation whose output is reused across replays.

    Consume or stage the output before the next replay can overwrite it.
    """

    inputs: Mapping[str, Any]
    output: Any
    launch: Callable[[], None]
    prepare: Callable[[], None] | None = None
    validate_eager_output: bool = True
    phase_timing: GraphPhaseTiming | None = None


def _map_output(output, operation):
    if isinstance(output, tuple):
        return tuple(_map_output(value, operation) for value in output)
    return operation(output)


def _outputs_equal(left, right) -> bool:
    if isinstance(left, tuple):
        return (
            isinstance(right, tuple)
            and len(left) == len(right)
            and all(_outputs_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return not isinstance(right, tuple) and torch.equal(left, right)


def _output_summary(output: Any) -> str:
    """Keep capture failures bounded even for full prefill hidden tensors."""

    if isinstance(output, tuple):
        return "(" + ", ".join(_output_summary(value) for value in output) + ")"
    sample = output.reshape(-1)[:8].tolist()
    return (
        f"shape={tuple(output.shape)}, dtype={output.dtype}, "
        f"elements={output.numel()}, sample={sample}"
    )


def _output_difference(expected: Any, actual: Any) -> str:
    """Compute bounded diagnostics only after strict replay validation fails."""
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        return (
            "("
            + ", ".join(_output_difference(a, b) for a, b in zip(expected, actual))
            + ")"
        )
    if (
        isinstance(expected, tuple)
        or isinstance(actual, tuple)
        or expected.shape != actual.shape
    ):
        return "output structure differs"
    mismatched = (expected != actual).sum().item()
    maximum = (
        (expected.float() - actual.float())
        .abs()
        .nan_to_num(nan=float("inf"), posinf=float("inf"))
        .max()
        .item()
        if expected.numel()
        else 0
    )
    return f"mismatched={mismatched}/{expected.numel()}, max_abs={maximum}"


@dataclass(frozen=True, slots=True)
class CudaGraphMode:
    """One independently bucketed execution mode in a CUDA graph runner.

    Modes are intentionally semantic rather than CUDA-specific algorithms:
    callers commonly register ``prefill`` and ``decode``. Each shape builds
    one graph with :class:`CudaGraphStep`.
    All modes and shapes in an executor share one memory pool and capture stream.
    """

    shapes: tuple[GraphShape, ...]
    build: Callable[[GraphShape], CudaGraphStep]
    capture_inputs: Callable[[GraphShape], Mapping[str, object]]

    def __post_init__(self) -> None:
        if not callable(self.build) or not callable(self.capture_inputs):
            raise TypeError("CUDA graph mode hooks must be callable")
        object.__setattr__(self, "shapes", tuple(self.shapes))


@dataclass(slots=True)
class _CudaGraphWorkspace:
    """Lazily allocated resources shared by one executor's serialized graphs."""

    pool: Any | None = None
    stream: torch.cuda.Stream | None = None


@dataclass(slots=True)
class _GraphEntry:
    shape: GraphShape
    step: CudaGraphStep | None = None
    staging: TensorStagerPool | None = None
    graph: torch.cuda.CUDAGraph | None = None
    captured: bool = False
    captures: int = 0
    replays: int = 0
    warmups: int = 0
    capture_ms: float | None = None
    memory_bytes: int | None = None


class _CudaGraphCache:
    """Capture and replay reusable static CUDA steps.

    A model describes each physical shape with a :class:`CudaGraphStep`. This
    cache owns persistent input staging, capture and replay, parity validation,
    failure poisoning, and metrics. It executes only the current batch.

    Captures share a memory pool and must replay serially. Persistent state
    and outputs needed across replays must use storage allocated outside capture.
    An executor supplies the same workspace to every mode; standalone caches
    own their workspace.
    """

    def __init__(
        self,
        device: str | torch.device,
        shapes: tuple[GraphShape, ...],
        *,
        build: Callable[[GraphShape], CudaGraphStep],
        synchronize_failure: Callable[[], None],
        _workspace: _CudaGraphWorkspace | None = None,
    ) -> None:
        self.device = device
        self.shapes = shapes
        self._build = build
        self._synchronize_failure = synchronize_failure
        self._entries = {shape: _GraphEntry(shape) for shape in shapes}
        self._pending_captures = len(self._entries)
        self._fallbacks: Counter[str] = Counter()
        self._replay_count = 0
        self._poison: str | None = None
        self._workspace: _CudaGraphWorkspace | None = (
            _workspace if _workspace is not None else _CudaGraphWorkspace()
        )
        self._closed = False
        self._closed_stats: CudaGraphStats | None = None

    def match(self, predicate: Callable[[GraphShape], bool]) -> GraphShape | None:
        """Return the first configured shape accepted by ``predicate``."""

        self.check()
        return next((shape for shape in self.shapes if predicate(shape)), None)

    def fallback(self, reason: str) -> None:
        self.check()
        if self.shapes:
            self._fallbacks[reason] += 1

    def check(self) -> None:
        if self._closed:
            raise RuntimeError("CUDA graph cache is closed")
        if self._poison is not None:
            raise RuntimeError(
                "this runner is poisoned after a CUDA graph failure; "
                f"construct a fresh runner ({self._poison})"
            )

    def check_ready(self) -> None:
        """Reject missing startup capture before model state can be mutated."""

        self.check()
        if self._pending_captures:
            raise RuntimeError(
                f"{self._pending_captures} configured CUDA graph shapes have not "
                "been captured; call runner.capture_cuda_graphs() before inference"
            )

    def close(self) -> None:
        """Release captured CUDA/NCCL resources before process-group teardown."""

        if self._closed:
            return
        if any(entry.graph is not None for entry in self._entries.values()):
            torch.cuda.synchronize(self.device)
        self._release()

    def _release(self) -> None:
        """Release an already synchronized cache; shared by executor shutdown."""

        if self._closed:
            return
        stats = self.stats
        for entry in self._entries.values():
            if entry.graph is not None:
                entry.graph.reset()
            entry.graph = None
            entry.step = None
            entry.staging = None
        # Drop this cache's reference without invalidating other modes' graphs.
        self._workspace = None
        self._closed_stats = stats
        self._closed = True

    def capture_all(
        self,
        inputs_for_shape: Callable[[GraphShape], Mapping[str, object]],
        *,
        before_capture: Callable[[], None] | None = None,
        after_capture: Callable[[], None] | None = None,
    ) -> CudaGraphStats:
        """Capture every pending shape with optional model-state hooks.

        The hooks run whenever capture is configured, including idempotent
        calls after every shape is already captured. This keeps restoration
        behavior owned by the model while centralizing capture bookkeeping.
        """

        self.check()
        if not self.shapes:
            return self.stats
        if before_capture is not None:
            before_capture()
        captured = set(self.stats["captured_shapes"])
        # Seed the shared pool with the largest workspace before smaller shapes.
        for shape in sorted(self.shapes, reverse=True):
            if shape not in captured:
                self.capture(shape, inputs_for_shape(shape))
        if after_capture is not None:
            after_capture()
        return self.stats

    def _fail(self, stage: str, error: BaseException) -> NoReturn:
        if self._poison is None:
            self._poison = f"{stage}: {type(error).__name__}: {error}"
        try:
            self._synchronize_failure()
        except BaseException as synchronization_error:
            raise PostAcceptCudaError(
                f"{stage} failed and CUDA quiescence could not be established; "
                f"resources must remain quarantined: {error}"
            ) from synchronization_error
        raise PostAcceptCudaError(
            f"{stage} failed; persistent state may have changed and cannot be "
            f"retried: {error}"
        ) from error

    def capture(self, shape: GraphShape, inputs: Mapping[str, object]) -> Any:
        """Build and capture one configured shape before request execution."""

        self.check()
        try:
            entry = self._entries[shape]
        except KeyError as error:
            raise ValueError(f"unconfigured CUDA graph shape: {shape}") from error
        if entry.captured:
            raise RuntimeError(f"CUDA graph shape is already captured: {shape}")
        before = torch.cuda.memory_allocated(self.device)
        step = self._build(shape)
        if not isinstance(step, CudaGraphStep):
            raise TypeError("CUDA graph builder must return a CudaGraphStep")
        entry.step = step
        try:
            entry.staging = TensorStagerPool(self.device)
            entry.staging.copy_into(step.inputs, inputs)
        except StagingError as error:
            entry.step = None
            entry.staging = None
            raise GraphInputError(str(error)) from error
        except BaseException as error:
            self._fail("graph input copy", error)
        try:
            self._capture(entry, before)
            self._pending_captures -= 1
        except BaseException as error:
            self._fail("graph capture", error)
        return step.output

    def run(
        self,
        shape: GraphShape,
        inputs: Mapping[str, object],
        *,
        metrics: CudaMetrics | None = None,
    ) -> Any:
        """Stage and replay one current batch using a pre-captured graph."""

        self.check()
        try:
            entry = self._entries[shape]
        except KeyError as error:
            raise ValueError(f"unconfigured CUDA graph shape: {shape}") from error
        step, graph = entry.step, entry.graph
        if not entry.captured or step is None or graph is None or entry.staging is None:
            raise RuntimeError(
                f"CUDA graph shape {shape} has not been captured; "
                "capture configured graphs during runner initialization"
            )
        try:
            entry.staging.copy_into(step.inputs, inputs)
        except StagingError as error:
            raise GraphInputError(str(error)) from error
        except BaseException as error:
            self._fail("graph input copy", error)

        try:
            with torch.cuda.device(self.device), torch.inference_mode():
                timing = step.phase_timing
                if (
                    metrics is not None
                    and metrics.active is not None
                    and timing is not None
                ):
                    metrics.active.phases.append(timing.replay(graph))
                elif timing is not None:
                    timing.unobserved(graph)
                graph.replay()
        except BaseException as error:
            self._fail("graph replay", error)
        entry.replays += 1
        self._replay_count += 1
        return step.output

    def _capture(self, entry: _GraphEntry, allocation_before: int) -> None:
        step = entry.step
        assert step is not None
        current = torch.cuda.current_stream(self.device)
        workspace = self._workspace
        assert workspace is not None
        if workspace.stream is None:
            workspace.stream = torch.cuda.Stream(device=self.device)
        warmup = workspace.stream
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.inference_mode():
            for _ in range(3):
                if step.prepare is not None:
                    step.prepare()
                step.launch()
            expected = _map_output(step.output, lambda tensor: tensor.clone())
        current.wait_stream(warmup)
        entry.warmups = 3
        torch.cuda.synchronize(self.device)

        started = perf_counter()
        # Stream-local kernel state must be initialized by the warmup, and
        # captures sharing a memory pool must use the same explicit stream.
        with torch.cuda.device(self.device), torch.cuda.stream(warmup):
            if step.prepare is not None:
                step.prepare()
            if workspace.pool is None:
                workspace.pool = torch.cuda.graph_pool_handle()
            graph = (
                torch.cuda.CUDAGraph(keep_graph=True)
                if step.phase_timing is not None
                else torch.cuda.CUDAGraph()
            )
            # Keep ownership on capture failure so close() can release NCCL
            # graph references before communicator teardown.
            entry.graph = graph
            with (
                torch.cuda.graph(graph, pool=workspace.pool, stream=warmup),
                torch.inference_mode(),
            ):
                step.launch()
            if step.phase_timing is not None:
                graph.instantiate()
            if step.prepare is not None:
                step.prepare()
            graph.replay()
        current.wait_stream(warmup)
        torch.cuda.synchronize(self.device)
        if step.validate_eager_output:
            if not _outputs_equal(expected, step.output):
                raise RuntimeError(
                    "CUDA graph replay differs from eager warmup "
                    f"({_output_difference(expected, step.output)}; "
                    f"expected: {_output_summary(expected)}; "
                    f"actual: {_output_summary(step.output)})"
                )
        else:
            expected_replay = _map_output(step.output, lambda tensor: tensor.clone())
            graph.replay()
            torch.cuda.synchronize(self.device)
            if not _outputs_equal(expected_replay, step.output):
                raise RuntimeError(
                    "CUDA graph replays are not deterministic "
                    f"({_output_difference(expected_replay, step.output)}; "
                    f"expected: {_output_summary(expected_replay)}; "
                    f"actual: {_output_summary(step.output)})"
                )
        if step.phase_timing is not None:
            step.phase_timing.prepare(graph)
        entry.captured = True
        entry.captures = 1
        entry.capture_ms = (perf_counter() - started) * 1000
        entry.memory_bytes = max(
            0, torch.cuda.memory_allocated(self.device) - allocation_before
        )

    @property
    def runtime_counts(self) -> CudaGraphCounts:
        """Snapshot runtime counters without visiting the configured graph catalog."""
        return {
            "replay_count": self._replay_count,
            "eager_fallback_count": sum(self._fallbacks.values()),
            "fallback_counts": dict(self._fallbacks),
        }

    @property
    def stats(self) -> CudaGraphStats:
        if self._closed_stats is not None:
            return self._closed_stats
        rows: tuple[CudaGraphShapeStats, ...] = tuple(
            {
                "shape": entry.shape,
                "captured": entry.captured,
                "capture_count": entry.captures,
                "replay_count": entry.replays,
                "warmup_count": entry.warmups,
                "capture_ms": entry.capture_ms,
                "capture_memory_bytes": entry.memory_bytes,
                "graph_region_count": int(entry.step is not None),
                "graph_break_count": 0,
            }
            for entry in self._entries.values()
        )
        return {
            "enabled": bool(self.shapes),
            "configured_shapes": self.shapes,
            "captured_shapes": tuple(row["shape"] for row in rows if row["captured"]),
            "capture_count": sum(entry.captures for entry in self._entries.values()),
            "replay_count": self._replay_count,
            "graph_region_count": sum(row["graph_region_count"] for row in rows),
            "graph_break_count": 0,
            "eager_fallback_count": sum(self._fallbacks.values()),
            "fallback_counts": dict(self._fallbacks),
            "shape_stats": rows,
            "poisoned": self._poison is not None,
            "poison_error": self._poison,
        }


class CudaGraphExecutor:
    """Own mode-specific graph buckets sharing one pool and capture stream.

    Prefill, decode, and any additional modes must execute serially, with no
    capture-allocated state needed across modes. Their shared workspace stays
    alive until every cache has released its graphs.
    """

    def __init__(
        self,
        device: str | torch.device,
        modes: Mapping[str, CudaGraphMode],
        *,
        synchronize_failure: Callable[[], None],
    ) -> None:
        if any(not isinstance(name, str) or not name.strip() for name in modes):
            raise ValueError("CUDA graph mode names must be nonempty")
        self.device = device
        self._synchronize_failure = synchronize_failure
        self._workspace: _CudaGraphWorkspace | None = _CudaGraphWorkspace()
        self._modes: dict[str, CudaGraphMode] = {}
        self._caches: dict[str, _CudaGraphCache] = {}
        self._closed = False
        for name, mode in modes.items():
            self.add_mode(name, mode)

    def add_mode(self, name: str, mode: CudaGraphMode) -> None:
        """Register a model phase after its modules and buffers are available."""

        if self._closed:
            raise RuntimeError("CUDA graph executor is closed")
        if not isinstance(name, str) or not name.strip() or name in self._modes:
            raise ValueError(f"invalid or duplicate CUDA graph mode: {name!r}")
        self._modes[name] = mode
        self._caches[name] = _CudaGraphCache(
            self.device,
            mode.shapes,
            build=mode.build,
            synchronize_failure=self._synchronize_failure,
            _workspace=self._workspace,
        )

    def mode(self, name: str) -> _CudaGraphCache:
        if self._closed:
            raise RuntimeError("CUDA graph executor is closed")
        try:
            return self._caches[name]
        except KeyError as error:
            raise KeyError(f"unknown CUDA graph mode: {name}") from error

    def check(self) -> None:
        if self._closed:
            raise RuntimeError("CUDA graph executor is closed")
        for cache in self._caches.values():
            cache.check()

    def close(self) -> None:
        """Synchronize once and destroy graphs before their communicators."""

        if self._closed:
            return
        if any(
            entry.graph is not None
            for cache in self._caches.values()
            for entry in cache._entries.values()
        ):
            torch.cuda.synchronize(self.device)
        for cache in self._caches.values():
            cache._release()
        self._workspace = None
        self._modes.clear()
        self._closed = True

    def capture_all(
        self,
        *,
        before_capture: Callable[[], None] | None = None,
        after_capture: Callable[[], None] | None = None,
    ) -> dict[str, CudaGraphStats]:
        """Capture all configured modes under one pair of state hooks."""

        self.check()
        if not any(cache.shapes for cache in self._caches.values()):
            return self.stats
        if before_capture is not None:
            before_capture()
        try:
            for name, mode in self._modes.items():
                self._caches[name].capture_all(mode.capture_inputs)
        finally:
            if after_capture is not None:
                after_capture()
        return self.stats

    @property
    def runtime_counts(self) -> dict[str, CudaGraphCounts]:
        """Read mode counters without constructing per-shape diagnostics."""
        return {name: cache.runtime_counts for name, cache in self._caches.items()}

    @property
    def stats(self) -> dict[str, CudaGraphStats]:
        return {name: cache.stats for name, cache in self._caches.items()}
