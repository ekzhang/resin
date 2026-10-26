"""Opaque cache objects and asynchronous storage, independent of model execution.

Providers borrow buffers until an operation returns a terminal result. A result,
including an error result, proves all accesses have retired; an exception does
not. Provider lifetime and stored objects are independent of an Engine.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock, RLock
from typing import Any, Protocol, cast


def _torch_tensor(region):
    torch = sys.modules.get("torch")
    return torch if torch is not None and isinstance(region, torch.Tensor) else None


def _byte_view(region, *, writable=False):
    torch = _torch_tensor(region)
    if torch is not None:
        if region.dtype != torch.uint8 or not region.is_contiguous():
            raise ValueError("cache tensor regions must be contiguous uint8 tensors")
        if region.device.type != "cpu":
            raise ValueError("a host byte view requires a CPU tensor")
        region = region.detach().numpy()
    view = memoryview(region)
    if not view.c_contiguous or view.itemsize != 1:
        raise ValueError("cache regions must be contiguous byte buffers")
    if writable and view.readonly:
        raise ValueError("cache load regions must be writable")
    return view.cast("B")


@dataclass(frozen=True)
class CacheBuffer:
    """Borrowed byte regions of one logical cache block, in payload order.

    Regions may be contiguous host buffer views or uint8 Torch tensors. Multiple
    regions represent scatter/gather storage; their combined size is ``nbytes``.
    The owner must retain and protect the allocation through transfer retirement.
    CUDA producers must be ordered on the calling thread's current device stream;
    a provider may establish dependencies and copy on its own stream.
    """

    regions: tuple[object, ...]
    nbytes: int
    device: str

    def __post_init__(self):
        object.__setattr__(self, "regions", tuple(self.regions))
        if (
            isinstance(self.nbytes, bool)
            or not isinstance(self.nbytes, int)
            or self.nbytes < 0
        ):
            raise ValueError("cache buffer nbytes must be a nonnegative integer")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("cache buffer device must be nonempty")
        size = 0
        for region in self.regions:
            region = cast(Any, region)
            torch = _torch_tensor(region)
            if torch is not None:
                if region.dtype != torch.uint8 or not region.is_contiguous():
                    raise ValueError(
                        "cache tensor regions must be contiguous uint8 tensors"
                    )
                size += region.numel()
                device = str(region.device)
            else:
                size += _byte_view(region).nbytes
                device = "cpu"
            if device != self.device and not (
                self.device == "cuda" and device.startswith("cuda:")
            ):
                raise ValueError("cache region device differs from its buffer device")
        if size != self.nbytes:
            raise ValueError("cache region sizes must sum to nbytes")


@dataclass(frozen=True)
class CacheResult:
    """Per-object outcomes; every access has retired, even when error is set."""

    success: tuple[bool, ...]
    error: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "success", tuple(self.success))
        if any(not isinstance(value, bool) for value in self.success):
            raise TypeError("cache outcomes must be bool values")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("cache error must be a string or None")


class CacheOperation(Protocol):
    """Own supplied buffers until terminal; polling never waits for I/O."""

    def poll(self) -> CacheResult | None: ...

    def wait(self) -> CacheResult: ...


class CacheStorage(Protocol):
    """Batch operations over immutable opaque keys and caller-owned buffers.

    Lookup is a hint, not a reservation: objects can disappear before load.
    Implementations may support host, GPU, or other memory kinds explicitly.
    """

    def lookup(self, keys: tuple[bytes, ...]) -> CacheOperation: ...

    def load(
        self, keys: tuple[bytes, ...], buffers: tuple[CacheBuffer, ...]
    ) -> CacheOperation: ...

    def store(
        self, keys: tuple[bytes, ...], buffers: tuple[CacheBuffer, ...]
    ) -> CacheOperation: ...


class _Operation:
    def __init__(self, buffers=(), finish: Callable[[], CacheResult] | None = None):
        self.buffers = buffers
        self.keep: list[Any] = []
        self.events: list[Any] = []
        self.devices: dict[str, tuple[Any, Any]] = {}
        self.error: Exception | None = None
        self.unretired: Exception | None = None
        self._finish = finish
        self._result: CacheResult | None = None
        self._lock = Lock()

    def fence(self):
        try:
            for torch, stream in self.devices.values():
                event = torch.cuda.Event()
                event.record(stream)
                self.events.append(event)
        except Exception as error:
            self.unretired = error

    def _complete(self):
        if self._result is None:
            assert self._finish is not None
            self._result = self._finish()
            self._finish = None
            self.buffers = ()
            self.keep.clear()
        return self._result

    def poll(self):
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self.unretired is not None:
                raise RuntimeError(
                    "cache transfer retirement is unproven"
                ) from self.unretired
            if self._result is None and not all(event.query() for event in self.events):
                return None
            return self._complete()
        finally:
            self._lock.release()

    def wait(self):
        with self._lock:
            if self.unretired is not None:
                raise RuntimeError(
                    "cache transfer retirement is unproven"
                ) from self.unretired
            if self._result is None:
                for event in self.events:
                    event.synchronize()
            return self._complete()


@dataclass(frozen=True)
class _Payload:
    regions: tuple[object, ...]
    nbytes: int


def _validate(keys, buffers=None, *, writable=False):
    if not isinstance(keys, tuple) or any(not isinstance(key, bytes) for key in keys):
        raise TypeError("cache keys must be a tuple of bytes")
    if buffers is None:
        return
    if len(keys) != len(buffers):
        raise ValueError("cache keys and buffers must have equal lengths")
    for buffer in buffers:
        if not isinstance(buffer, CacheBuffer):
            raise TypeError("cache transfers require CacheBuffer objects")
        if writable and buffer.device == "cpu":
            for region in buffer.regions:
                _byte_view(region, writable=True)


def _copy_host_payload(payload, targets):
    """Repartition scatter/gather bytes with one pass over the source regions."""
    sources = iter(payload.regions)
    source = memoryview(b"")
    for target in targets:
        written = 0
        while written < target.nbytes:
            if not source:
                source = _byte_view(next(sources))
                continue
            count = min(source.nbytes, target.nbytes - written)
            target[written : written + count] = source[:count]
            source = source[count:]
            written += count


class MemoryCacheStorage:
    """Bounded process-local LRU storage with immutable, idempotent puts.

    Host buffers are copied into owned immutable bytes. CUDA buffers use owned
    pinned host snapshots, per-device copy streams and per-operation events;
    Torch is never imported for host-only use. Eviction affects future lookups,
    not already submitted loads.
    """

    def __init__(self, max_bytes: int = 1 << 30):
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 0
        ):
            raise ValueError("max_bytes must be a nonnegative integer")
        self.max_bytes = max_bytes
        self._objects: OrderedDict[bytes, _Payload] = OrderedDict()
        self._nbytes = 0
        self._lock = RLock()
        self._copy_streams: dict[str, Any] = {}

    @property
    def nbytes(self):
        with self._lock:
            return self._nbytes

    def _put(self, key, payload):
        with self._lock:
            if key in self._objects:
                self._objects.move_to_end(key)
                return True
            if payload.nbytes > self.max_bytes:
                return False
            while self._nbytes + payload.nbytes > self.max_bytes:
                _, evicted = self._objects.popitem(last=False)
                self._nbytes -= evicted.nbytes
            self._objects[key] = payload
            self._nbytes += payload.nbytes
            return True

    def _copy_stream(self, operation, device, torch):
        name = str(device)
        if name not in operation.devices:
            with self._lock:
                stream = self._copy_streams.get(name)
                if stream is None:
                    stream = torch.cuda.Stream(device=device)
                    self._copy_streams[name] = stream
            operation.devices[name] = (torch, stream)
            stream.wait_stream(torch.cuda.current_stream(device))
        return operation.devices[name][1]

    def lookup(self, keys: tuple[bytes, ...]) -> CacheOperation:
        _validate(keys)
        with self._lock:
            found = tuple(key in self._objects for key in keys)
            for key, hit in zip(keys, found, strict=True):
                if hit:
                    self._objects.move_to_end(key)
        return _Operation(finish=lambda: CacheResult(found))

    def store(self, keys, buffers) -> CacheOperation:
        _validate(keys, buffers)
        operation = _Operation(buffers)
        payloads = []
        with self._lock:
            existing = tuple(self._objects.get(key) for key in keys)
        try:
            for buffer, previous in zip(buffers, existing, strict=True):
                if previous is not None or buffer.nbytes > self.max_bytes:
                    payloads.append(previous)
                    continue
                regions = []
                for source in buffer.regions:
                    torch = _torch_tensor(source)
                    if torch is not None and source.device.type == "cuda":
                        host = torch.empty_like(source, device="cpu", pin_memory=True)
                        operation.keep.append(host)
                        stream = self._copy_stream(operation, source.device, torch)
                        with torch.cuda.stream(stream):
                            host.copy_(source, non_blocking=True)
                        regions.append(host)
                    else:
                        regions.append(bytes(_byte_view(source)))
                payloads.append(_Payload(tuple(regions), buffer.nbytes))
        except Exception as error:
            operation.error = error
        operation.fence()

        def finish():
            if operation.error is not None:
                return CacheResult((False,) * len(keys), str(operation.error))
            return CacheResult(
                tuple(
                    data is not None and self._put(key, data)
                    for key, data in zip(keys, payloads, strict=True)
                )
            )

        operation._finish = finish
        return operation

    def load(self, keys, buffers) -> CacheOperation:
        _validate(keys, buffers, writable=True)
        with self._lock:
            payloads = tuple(self._objects.get(key) for key in keys)
            for key, payload in zip(keys, payloads, strict=True):
                if payload is not None:
                    self._objects.move_to_end(key)
        operation = _Operation(buffers)
        operation.keep.extend(payloads)
        success = []
        try:
            for buffer, payload in zip(buffers, payloads, strict=True):
                hit = payload is not None and payload.nbytes == buffer.nbytes
                success.append(hit)
                if not hit:
                    continue
                assert payload is not None
                if len(buffer.regions) == len(payload.regions) == 1:
                    target = buffer.regions[0]
                    source = cast(Any, payload.regions[0])
                    torch = _torch_tensor(target)
                    if (
                        torch is not None
                        and target.device.type == "cuda"
                        and _torch_tensor(source) is not None
                        and source.is_pinned()
                    ):
                        stream = self._copy_stream(operation, target.device, torch)
                        with torch.cuda.stream(stream):
                            target.copy_(source.view_as(target), non_blocking=True)
                        continue
                targets, copies = [], []
                for target in buffer.regions:
                    torch = _torch_tensor(target)
                    if torch is not None and target.device.type == "cuda":
                        host = torch.empty_like(target, device="cpu", pin_memory=True)
                        operation.keep.append(host)
                        targets.append(_byte_view(host, writable=True))
                        copies.append((target, host, torch))
                    else:
                        targets.append(_byte_view(target, writable=True))
                _copy_host_payload(payload, targets)
                for target, host, torch in copies:
                    stream = self._copy_stream(operation, target.device, torch)
                    with torch.cuda.stream(stream):
                        target.copy_(host, non_blocking=True)
        except Exception as error:
            operation.error = error
        operation.fence()
        operation._finish = lambda: CacheResult(
            tuple(success) if operation.error is None else (False,) * len(keys),
            None if operation.error is None else str(operation.error),
        )
        return operation


__all__ = [
    "CacheBuffer",
    "CacheOperation",
    "CacheResult",
    "CacheStorage",
    "MemoryCacheStorage",
]
