"""Native packet encoding and real Gloo transport, independent of inference."""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from resin._resin import _HttpEvents, _HttpIngress
from resin.backend.control_plane import ControlPlane
from resin.backend.distributed import DistributedExecutionError
from resin.backend.topology import RankGroup
from resin.server.transport import HttpTransport


def test_native_http_packets_encode_only_at_the_wire_boundary():
    ingress = _HttpIngress.from_bytes(
        b'{"canceled":["request-1"],"shutdown":true,"error":"failure"}'
    )
    assert ingress.shutdown and ingress.error == "failure"
    assert json.loads(ingress.to_bytes()) == {
        "requests": [],
        "canceled": ["request-1"],
        "shutdown": True,
        "error": "failure",
        "collect_metrics": False,
    }
    assert _HttpIngress().to_bytes() == b""
    assert not _HttpIngress.from_bytes(b"").shutdown
    with pytest.raises(AttributeError):
        setattr(ingress, "shutdown", False)

    payload = [["request-1", {"type": "text_delta", "data": {"delta": "hello 🌎"}}]]
    events = _HttpEvents.from_bytes(json.dumps({"events": payload}).encode())
    assert isinstance(events.to_bytes(), bytes)
    assert json.loads(events.to_bytes()) == {"events": payload, "metrics": []}
    assert _HttpEvents().to_bytes() == _HttpEvents.from_bytes(b"").to_bytes() == b""
    assert json.loads(
        _HttpEvents.combine([events, _HttpEvents(), events]).to_bytes()
    ) == {"events": payload * 2, "metrics": []}


@pytest.mark.parametrize(
    "payload", [b"invalid", b"[42]", b'{"requests":42}', b'{"unknown":1}']
)
def test_native_ingress_rejects_malformed_wire_packets(payload):
    with pytest.raises(ValueError):
        _HttpIngress.from_bytes(payload)


@pytest.mark.parametrize(
    "payload", [b"invalid", b'{"events":42}', b'{"events":[["x",{"type":"unknown"}]]}']
)
def test_native_events_reject_malformed_wire_packets(payload):
    with pytest.raises(ValueError):
        _HttpEvents.from_bytes(payload)


def test_single_driver_packet_transport_needs_no_collectives():
    group = RankGroup((3,), None, 0)
    control = ControlPlane(object())
    worker: Any = SimpleNamespace(
        groups=SimpleNamespace(drivers=group, distributed=object()),
        driver_vote=lambda operation, values=(), *, error=None: control.gather(
            group, operation, values, error=error
        ),
    )
    wire = HttpTransport(worker)
    assert wire.broadcast(b"local") == b"local"
    assert wire.gather(b"") == []


def _transport_rank(rank, directory):
    patch = pytest.MonkeyPatch()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=20),
    )
    handle = dist.new_group([1, 3], backend="gloo")
    try:
        if rank not in (1, 3):
            return
        group = RankGroup((1, 3), handle, (1, 3).index(rank))
        control = ControlPlane(dist)
        worker: Any = SimpleNamespace(
            groups=SimpleNamespace(drivers=group, distributed=dist),
            driver_vote=lambda operation, values=(), *, error=None: control.gather(
                group, operation, values, error=error
            ),
        )
        wire = HttpTransport(worker)
        original_objects = dist.all_gather_object

        def unexpected_objects(*args, **kwargs):
            raise AssertionError("successful HTTP exchange serialized Python objects")

        patch.setattr(dist, "all_gather_object", unexpected_objects)
        patch.setattr(dist, "broadcast_object_list", unexpected_objects)
        for payload in (b"", b"small", bytes(range(256)) * 1000, b"last"):
            with torch.inference_mode():
                assert wire.broadcast(payload if rank == 1 else b"") == payload
                gathered = wire.gather(payload if rank == 3 else b"")
            assert gathered == ([payload] if rank == 1 and payload else [])

        # A receiver allocation failure is voted before root enters broadcast.
        patch.setattr(dist, "all_gather_object", original_objects)
        for message in ("injected HTTP allocation failure", ""):
            with pytest.MonkeyPatch.context() as allocation_patch:
                if rank == 3:
                    zeros = torch.zeros

                    def fail_allocation(*args, **kwargs):
                        if kwargs.get("dtype") == torch.uint8:
                            raise MemoryError(message)
                        return zeros(*args, **kwargs)

                    allocation_patch.setattr(torch, "zeros", fail_allocation)
                with pytest.raises(
                    DistributedExecutionError,
                    match="HTTP ingress buffers failed on rank 3",
                ):
                    wire.broadcast(b"next" if rank == 1 else b"")
        # The failed exchange did not leave peers inside a payload collective.
        assert wire.broadcast(b"recovered" if rank == 1 else b"") == b"recovered"
        Path(directory, f"rank-{rank}").touch()
    finally:
        patch.undo()
        if rank in (1, 3):
            dist.destroy_process_group(handle)
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
def test_http_bytes_use_nonzero_subgroup_source_and_vote_before_payload_io(tmp_path):
    context = mp.spawn(_transport_rank, args=(str(tmp_path),), nprocs=4, join=False)
    deadline = time.monotonic() + 35
    try:
        while time.monotonic() < deadline:
            if context.join(timeout=0.2):
                break
        else:
            raise AssertionError("HTTP packet collective did not finish")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join()
    assert (tmp_path / "rank-1").exists() and (tmp_path / "rank-3").exists()
