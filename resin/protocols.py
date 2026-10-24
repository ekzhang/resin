"""Structural protocols accepted by Resin's native control plane."""

from __future__ import annotations

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from resin._resin import (
        BatchPlan,
        ForwardOutput,
        ParserEvent,
        ResourceBinding,
        RunnerConfig,
        RunnerMetrics,
    )


class MediaTokenSpan(NamedTuple):
    """A half-open media range described by its prompt position and length."""

    position: int
    length: int


@runtime_checkable
class MediaInput(Protocol):
    """Model-owned prompt payload understood by the generic Engine.

    ``cache_key`` is a token-aligned, hashable identity. It normally equals
    ``token_ids`` outside media spans and is digest-salted inside them, allowing
    the Engine's ordinary prefix tree to distinguish equal placeholders backed
    by different media. ``token_spans`` marks ranges that must remain intact in
    one prefill row when a model uses bidirectional attention within media.
    Both properties are required at admission; use an empty tuple for spans
    when all prompt positions may be split across prefill chunks.
    """

    @property
    def token_ids(self) -> Sequence[int]: ...

    @property
    def cache_key(self) -> tuple[int, ...]: ...

    @property
    def token_spans(self) -> tuple[MediaTokenSpan, ...]: ...


# (generated token index, source byte start/end, event text byte start/end).
# Ranges are half-open u32 offsets; several tokens may share one UTF-8 character.
TokenSpan = tuple[int, int, int, int, int]


class ResponseParser(Protocol):
    """Turn generated IDs into semantic deltas carrying source-byte spans.

    Preserve ``token_spans`` through buffering. Empty spans identify synthetic
    text or unknown provenance; they never imply a guessed token map.
    """

    def feed(self, token_ids: Sequence[int]) -> Sequence[ParserEvent]: ...


class FinalizingResponseParser(ResponseParser, Protocol):
    """Flush buffered output before a response ends normally or at its limit."""

    def finish(self, reason: Literal["stop", "length"]) -> Sequence[ParserEvent]: ...


class Runner(Protocol):
    """Execute one Engine-owned immutable plan synchronously."""

    def forward(self, plan: BatchPlan, /) -> ForwardOutput: ...


class RunnerCompletion(Protocol):
    """Owned accepted output with device feedback for an Engine continuation.

    ``result`` waits only for this step and may be called repeatedly. Feedback
    owns its tensors independently of graph buffers reused by the next launch.
    """

    @property
    def feedback(self) -> Any: ...

    def result(self) -> ForwardOutput: ...


class EnqueuingRunner(Runner, Protocol):
    """Optional device submission boundary; overlap belongs to the Engine.

    A synchronous result is a supported fallback. Engine(enable_overlap=False) uses
    forward directly. Models themselves retain their ordinary tensor forward.
    """

    def enqueue(
        self, plan: BatchPlan, *, feedback: Any = None
    ) -> RunnerCompletion | ForwardOutput: ...


class ManagedRunner(Runner, Protocol):
    """Declare cache resources and bind their arena once during construction.

    An optional ``release(sequence_ids)`` callback cleans up model-local state
    on completion, cancellation, or retraction. Physical page ownership remains
    with the Engine; release must never independently return cache pages.
    """

    runner_config: RunnerConfig

    def bind_resources(self, resources: ResourceBinding) -> None: ...


class MetricsRunner(Protocol):
    """Optional physical execution observations, independent of forward output.

    The Engine enables collection before resource binding, so captured runtimes
    can include timing boundaries. Snapshots contain cumulative observations and
    must not synchronize the device or enter distributed collectives. Each read
    returns independent data; the serving layer publishes it periodically.
    """

    def configure_metrics(self, *, enabled: bool = True) -> None: ...

    def metrics_snapshot(self) -> RunnerMetrics: ...


__all__ = [
    "FinalizingResponseParser",
    "ManagedRunner",
    "MediaInput",
    "MediaTokenSpan",
    "MetricsRunner",
    "ResponseParser",
    "Runner",
    "RunnerCompletion",
    "EnqueuingRunner",
    "TokenSpan",
]
