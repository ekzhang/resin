"""Route HTTP ingress while every attention replica joins coordinated steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from resin._resin import (
    _HttpEvents,
    _HttpIngress,
    _HttpSession,
    _runner_metrics_prometheus_protobuf,
)
from resin.backend.distributed import DistributedExecutionError
from resin.server.transport import HttpTransport

if TYPE_CHECKING:
    from resin.backend.distributed import DistributedEngine
    from resin.recipe import Recipe


class DistributedHttpCoordinator:
    """One native HTTP session per driver; only the first driver owns clients.

    Construct collectively on replica drivers. The first driver calls round()
    from its native HTTP actor, while the remaining drivers call serve().
    Request IDs and events cross the control group; leases, KV, recipes, media,
    and parsers remain local to the selected replica.
    """

    def __init__(
        self,
        engine: DistributedEngine,
        recipe: Recipe,
        *,
        stream_interval: int = 1,
        model: str = "resin",
    ):
        self.engine = engine
        self.model = model
        self.group = engine.worker._driver_group
        self._wire = HttpTransport(engine.worker)
        self.rank = self.group.local_rank
        self.routed_dp_size = self.group.size
        self.is_root = self.rank == 0
        self.closed = False
        self.running = False
        self._shutdown = False
        self.session = None
        error = None
        try:
            self.session = _HttpSession(
                engine.local,
                recipe,
                stream_interval=stream_interval,
                emit_first_token=engine.local.metrics_enabled,
            )
        except BaseException as exc:
            error = str(exc)
        failure = self._vote("initialize", error)
        if failure:
            self._close(failure)

    @property
    def is_frontend(self):
        return self.is_root

    @property
    def size(self):
        return self.routed_dp_size

    def shutdown(self):
        """Request collective shutdown at the next round boundary."""
        self._shutdown = True

    def close(self):
        """Root startup/shutdown cleanup; peers must be in their serve loop."""
        if self.closed:
            return
        if not self.is_root:
            self.shutdown()
            return
        self.round(_HttpIngress(shutdown=True))

    def _vote(self, stage, error):
        try:
            self._wire.vote(stage, error=error)
        except DistributedExecutionError as exc:
            return str(exc)
        return None

    def round(self, ingress: _HttpIngress) -> _HttpEvents:
        """Root actor boundary: dispatch ingress and collect one step's events."""
        if not self.is_root:
            raise ValueError("only the first replica driver owns HTTP ingress")
        if self.closed:
            if ingress.shutdown:
                return _HttpEvents()
            raise RuntimeError("distributed HTTP coordinator is closed")
        self.running = True
        return self._round(ingress)

    def serve(self):
        """Block on a non-root driver until the frontend closes all replicas."""
        if self.is_root:
            raise ValueError("the first replica driver serves the HTTP frontend")
        self.running = True
        while not self.closed:
            self._round(None)

    def _round(self, ingress: _HttpIngress | None) -> _HttpEvents:
        payload, error = b"", None
        try:
            if ingress is not None and self.group.size > 1:
                payload = ingress.to_bytes()
        except BaseException as exc:
            error = str(exc)
        try:
            requested = self._wire.vote(
                "shutdown requested",
                (int(self._shutdown or (ingress is not None and ingress.shutdown)),),
                error=error
                if error is not None
                else (None if ingress is None else ingress.error),
            )
            if any(row[0] for row in requested):
                self._close()
                return _HttpEvents()
        except BaseException as exc:
            self._close(str(exc))
            raise

        error = None
        try:
            payload = self._wire.broadcast(payload)
            if ingress is None:
                ingress = _HttpIngress.from_bytes(payload)
            assert self.session is not None
            self.session.admit(ingress, rank=self.rank)
        except BaseException as exc:
            error = str(exc)
        failure = self._vote("ingress", error)
        if failure:
            self._close(failure)

        events = []
        error = None
        try:
            if self.engine.has_work():
                events = self.engine.step()
        except BaseException as exc:
            error = str(exc)
        failure = self._vote("step", error)
        if failure:
            self._close(failure)

        packet, payload, error = None, b"", None
        try:
            assert self.session is not None
            packet = self.session.process(events)
            assert ingress is not None
            if ingress.collect_metrics:
                snapshot = self.engine.local._metrics_prometheus_protobuf(
                    model_name=self.model,
                    rank=self.engine.worker.groups.rank,
                    attn_dp_rank=self.rank,
                    include_runner=False,
                )
                assert snapshot is not None
                physical = self.engine.worker.collect_metrics()
                packet = packet.with_metrics(
                    [
                        snapshot,
                        *(
                            _runner_metrics_prometheus_protobuf(
                                observation,
                                model_name=self.model,
                                rank=rank,
                                attn_dp_rank=self.rank,
                            )
                            for rank, observation in physical
                        ),
                    ]
                )
            if not self.is_root:
                payload = packet.to_bytes()
        except BaseException as exc:
            error = str(exc)
        output = None
        try:
            gathered = self._wire.gather(payload, error=error)
            if self.is_root:
                assert packet is not None
                output = (
                    _HttpEvents.combine(
                        [packet, *(_HttpEvents.from_bytes(data) for data in gathered)]
                    )
                    if gathered
                    else packet
                )
            error = None
        except BaseException as exc:
            error = str(exc)
        failure = self._vote("output decode", error)
        if failure:
            self._close(failure)
        return _HttpEvents() if output is None else output

    def _close(self, failure=None):
        """Close collectively even when native admission or parsing has failed."""
        if self.closed:
            if failure:
                raise DistributedExecutionError(failure)
            return
        error = None
        try:
            if self.session is not None:
                self.session.close()
        except BaseException as exc:
            error = str(exc)
        failure = failure or self._vote("shutdown", error)
        try:
            self.engine.close()
        except BaseException as exc:
            failure = failure or str(exc)
        finally:
            self.closed = True
            self.running = False
        if failure:
            raise DistributedExecutionError(failure)
