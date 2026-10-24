"""HTTP serving for a token-native Resin engine.

The networking, scheduling bridge, wire protocols, and streaming adapters run
in Rust. Stock model formats run in resin-recipe; custom recipes implement the same
structural Python protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from resin._resin import Engine, _HttpServer
from resin.protocols import ResponseParser
from resin.recipe import ParserEvent, Recipe, RenderedPrompt

if TYPE_CHECKING:
    from resin.backend.distributed import DistributedEngine


class HttpServer:
    """Serve a local or distributed engine through the shared HTTP frontend.

    With DistributedEngine, every attention-DP driver constructs a server and
    calls serve. The first driver listens; the others participate in coordinated
    inference. Requests select a replica with routed_dp_rank or the
    resin-routed-dp-rank header; unassigned requests use round robin.

    stream_interval batches later output until N generated model tokens have
    accumulated, checked at step boundaries. The first semantic output is sent
    immediately, and completion flushes the remainder. Stop matching and
    cancellation still run on every step.

    request_queue_capacity bounds requests waiting for engine admission across
    all replicas. It does not change the engine's active batch size. A full
    queue returns HTTP 429 so clients can retry with backoff.
    """

    def __init__(
        self,
        engine: Engine | DistributedEngine,
        *,
        recipe: Recipe,
        model: str | None = None,
        stream_interval: int = 1,
        request_queue_capacity: int = 1024,
        enable_openai: bool = True,
        enable_anthropic: bool = True,
        enable_demo: bool = True,
    ) -> None:
        from resin.backend.distributed import (
            DistributedEngine,
            DistributedExecutionError,
        )

        error = None
        try:
            if (
                type(stream_interval) is not int
                or not 1 <= stream_interval <= 2**32 - 1
            ):
                raise ValueError(
                    "stream_interval must be a positive integer fitting u32"
                )
            if (
                type(request_queue_capacity) is not int
                or not 1 <= request_queue_capacity <= 2**32 - 1
            ):
                raise ValueError(
                    "request_queue_capacity must be a positive integer fitting u32"
                )
            if model is None:
                model = getattr(
                    getattr(recipe, "tokenizer", None), "name_or_path", None
                )
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    "model is required when it cannot be inferred from the recipe"
                )
            if any(
                type(flag) is not bool
                for flag in (enable_openai, enable_anthropic, enable_demo)
            ):
                raise ValueError("HTTP route flags must be booleans")
            if enable_demo and not enable_openai:
                raise ValueError("enable_demo=True requires enable_openai=True")
        except BaseException as exc:
            error = exc
        if isinstance(engine, DistributedEngine):
            # No driver may return before its peers have validated startup.
            # Inference of a recipe's model name can itself run user code.
            config = (
                None
                if error is not None
                else (
                    model,
                    enable_openai,
                    enable_anthropic,
                    enable_demo,
                    stream_interval,
                    request_queue_capacity,
                    engine.local.metrics_enabled,
                )
            )
            states = engine.worker._drivers(
                "http configuration", (config, None if error is None else str(error))
            )
            group = engine.worker._driver_group
            failure = next(
                (
                    f"HTTP configuration failed on rank {rank}: {error}"
                    for rank, (_, error) in zip(group.ranks, states, strict=True)
                    if error
                ),
                None,
            )
            if failure is None and any(config != states[0][0] for config, _ in states):
                failure = "HTTP server configuration must agree across replica drivers"
            if failure:
                try:
                    engine.close()
                finally:
                    raise DistributedExecutionError(failure)
        elif error is not None:
            raise error
        assert isinstance(model, str)
        self._model = model
        self._coordinator = None
        if isinstance(engine, DistributedEngine):
            from resin.server.distributed import DistributedHttpCoordinator

            self._coordinator = DistributedHttpCoordinator(
                engine, recipe, model=model, stream_interval=stream_interval
            )
        self._native = None
        error = None
        try:
            if self._coordinator is None or self._coordinator.is_frontend:
                self._native = _HttpServer(
                    engine.local if isinstance(engine, DistributedEngine) else engine,
                    recipe,
                    model=model,
                    stream_interval=stream_interval,
                    request_queue_capacity=request_queue_capacity,
                    enable_openai=enable_openai,
                    enable_anthropic=enable_anthropic,
                    enable_demo=enable_demo,
                    distributed=self._coordinator,
                )
        except BaseException as exc:
            error = exc
        if self._coordinator is not None:
            failure = self._coordinator._vote(
                "frontend initialize", None if error is None else str(error)
            )
            if failure:
                self._coordinator._close(failure)
        elif error is not None:
            raise error

    @property
    def model(self) -> str:
        return self._model

    @property
    def running(self) -> bool:
        if self._native is not None:
            return self._native.running
        return self._coordinator is not None and self._coordinator.running

    def serve(self, *, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Block while serving, until Ctrl-C or another thread calls shutdown."""

        try:
            if self._native is not None:
                self._native.serve(host=host, port=port)
            elif self._coordinator is not None:
                self._coordinator.serve()
        finally:
            if self._coordinator is not None:
                self._coordinator.close()

    def shutdown(self) -> None:
        """Request shutdown after the current inference step finishes.

        This does not block the caller. Join the thread running ``serve()`` to
        wait for shutdown; active requests are cancelled and the Engine closes.
        """

        if self._native is not None:
            self._native.shutdown()
        elif self._coordinator is not None:
            self._coordinator.shutdown()


__all__ = [
    "Recipe",
    "HttpServer",
    "ParserEvent",
    "RenderedPrompt",
    "ResponseParser",
]
