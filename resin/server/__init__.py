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

    - `enable_priority_scheduling` accepts a signed 32-bit `priority` on HTTP
      requests. Higher values are admitted first among waiting requests within
      each engine; running requests are not preempted. Requires opt-in for integer
      priorities; omitted or null priorities use the default of zero.

    - `stream_interval` batches later output until N generated model tokens have
      accumulated, checked at step boundaries. The first semantic output is sent
      immediately, and completion flushes the remainder.

    - `request_queue_capacity` bounds requests waiting for engine admission on
      all replicas. A full queue returns HTTP 429.

    - `max_active_requests` bounds admitted requests, including prompts being
      rendered. Defaults to the engine's `max_running_requests` plus
      `render_workers` For distributed engines, the total cap scales with the
      number of attention-DP replicas.

    - `render_workers` sets the number of threads rendering prompts and media.
      Defaults to available CPU parallelism, capped at 16.

    - `admissions_per_step` limits how many requests are admitted between
      inference steps.
    """

    def __init__(
        self,
        engine: Engine | DistributedEngine,
        *,
        recipe: Recipe,
        model: str | None = None,
        stream_interval: int = 1,
        request_queue_capacity: int = 1024,
        max_active_requests: int | None = None,
        render_workers: int | None = None,
        admissions_per_step: int = 32,
        enable_openai: bool = True,
        enable_anthropic: bool = True,
        enable_demo: bool = True,
        enable_priority_scheduling: bool = False,
    ) -> None:
        from resin.backend.distributed import (
            DistributedEngine,
            DistributedExecutionError,
        )

        error = None
        try:
            for name, value in (
                ("stream_interval", stream_interval),
                ("request_queue_capacity", request_queue_capacity),
                ("max_active_requests", max_active_requests),
                ("render_workers", render_workers),
                ("admissions_per_step", admissions_per_step),
            ):
                if name in ("max_active_requests", "render_workers") and value is None:
                    continue
                if type(value) is not int or not 1 <= value <= 2**32 - 1:
                    raise ValueError(f"{name} must be a positive integer fitting u32")
            if model is None:
                tokenizer = recipe.tokenizer
                model = None if tokenizer is None else tokenizer.name_or_path
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    "model is required when it cannot be inferred from the recipe"
                )
            if any(
                type(flag) is not bool
                for flag in (
                    enable_openai,
                    enable_anthropic,
                    enable_demo,
                    enable_priority_scheduling,
                )
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
                    enable_priority_scheduling,
                    stream_interval,
                    request_queue_capacity,
                    max_active_requests,
                    render_workers,
                    admissions_per_step,
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
                    max_active_requests=max_active_requests,
                    render_workers=render_workers,
                    admissions_per_step=admissions_per_step,
                    enable_openai=enable_openai,
                    enable_anthropic=enable_anthropic,
                    enable_demo=enable_demo,
                    enable_priority_scheduling=enable_priority_scheduling,
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
