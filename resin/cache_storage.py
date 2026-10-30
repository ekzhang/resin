"""A DRAM LRU for immutable objects, budgeting keys, region names and payloads."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

Regions = Mapping[str, "Tensor"]


class CacheStorage(Protocol):
    """Nonblocking submission; terminal transfer results prove buffer retirement."""

    def lookup(self, keys: Sequence[bytes]) -> Future[tuple[bool, ...]]: ...
    def load(
        self, keys: Sequence[bytes], buffers: Sequence[Regions]
    ) -> Future[bool | None]: ...
    def store(
        self, keys: Sequence[bytes], buffers: Sequence[Regions]
    ) -> Future[bool | None]: ...


# A failed device fence cannot authorize reclaiming transfer memory.
_UNRETIRED = []


class MemoryCacheStorage:
    def __init__(self, max_bytes):
        import torch

        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes, self.nbytes = max_bytes, 0
        self.objects, self.executor = OrderedDict(), ThreadPoolExecutor(1)
        self.pool = torch.empty(
            max_bytes, dtype=torch.uint8, pin_memory=torch.cuda.is_available()
        )
        self.free = [(0, max_bytes)] if max_bytes else []
        self.streams, self.runtime = {}, None
        self.halted = self.closed = False
        if torch.cuda.is_available():
            try:
                from cuda.bindings import runtime

                if (
                    min(
                        runtime.cudaDriverGetVersion()[1],
                        runtime.cudaRuntimeGetVersion()[1],
                    )
                    >= 13000
                ):
                    self.runtime = runtime
            except (ImportError, AttributeError, RuntimeError, TypeError):
                pass

    def lookup(self, keys):
        if self.closed:
            raise RuntimeError("cache storage is closed")
        future = Future()
        # Presence is advisory; load rechecks after any intervening eviction.
        future.set_result(tuple(not self.halted and k in self.objects for k in keys))
        return future

    def _release(self, start, size):
        if not size:
            return
        merged = []
        for start, size in sorted([*self.free, (start, size)]):
            if merged and merged[-1][0] + merged[-1][1] == start:
                old, count = merged.pop()
                merged.append((old, count + size))
            else:
                merged.append((start, size))
        self.free = merged

    def _reserve(self, size, protected):
        while True:
            for i, (start, available) in enumerate(self.free):
                if available >= size:
                    self.free[i : i + 1] = (
                        [(start + size, available - size)] if available > size else []
                    )
                    return start
            victim = next((k for k in self.objects if k not in protected), None)
            if victim is None:
                return None
            _, count, start = self.objects.pop(victim)
            self.nbytes -= count
            self._release(start, count)

    def _copy(self, pairs, load):
        import torch

        devices = defaultdict(list)
        for region, offset in pairs:
            if region.nbytes:
                devices[region.device].append((region, offset))
        for device, rows in devices.items():
            sizes = [r.nbytes for r, _ in rows]
            pointers = [r.data_ptr() for r, _ in rows]
            # Batch copies have no defined order; preserve ordered scatter for aliases.
            intervals = sorted(zip(pointers, sizes)) if load else []
            aliases = any(a + n > b for (a, n), (b, _) in zip(intervals, intervals[1:]))
            if device.type != "cuda" or self.runtime is None or aliases:
                for region, offset in rows:
                    host = self.pool[offset : offset + region.nbytes]
                    target, source = (
                        (region.view(-1), host) if load else (host, region.view(-1))
                    )
                    target.copy_(source, non_blocking=True)
                continue
            with torch.cuda.device(device):
                hosts = [self.pool.data_ptr() + offset for _, offset in rows]
                destination, source = (pointers, hosts) if load else (hosts, pointers)
                attrs = self.runtime.cudaMemcpyAttributes()
                attrs.srcAccessOrder = (
                    self.runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
                )
                if max(sizes) <= 128 << 10:
                    attrs.flags = self.runtime.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute
                (status,) = self.runtime.cudaMemcpyBatchAsync(
                    destination,
                    source,
                    sizes,
                    len(rows),
                    [attrs],
                    [0],
                    1,
                    self.streams[device].cuda_stream,
                )
                if status != self.runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaMemcpyBatchAsync: {status}")

    def load(self, keys, buffers):
        def read():
            pairs = []
            for key, regions in zip(keys, buffers, strict=True):
                data, _, _ = self.objects[key]
                if data.keys() != regions.keys() or any(
                    data[name][1] != r.nbytes for name, r in regions.items()
                ):
                    return False
                self.objects.move_to_end(key)
                pairs.extend((r, data[name][0]) for name, r in regions.items())
            self._copy(pairs, True)
            return True

        return self._submit(read, buffers)

    def store(self, keys, buffers):
        reservation, pending = [], []

        def write():
            planned = {}
            for key, regions in zip(keys, buffers, strict=True):
                if key not in self.objects and key not in planned:
                    size = len(key) + sum(
                        len(n.encode()) + r.nbytes for n, r in regions.items()
                    )
                    planned[key] = regions, size
            charge = sum(size for _, size in planned.values())
            protected = set(keys) & self.objects.keys()
            if charge + sum(self.objects[k][1] for k in protected) > self.max_bytes:
                return False
            start = self._reserve(charge, protected) if charge else 0
            if start is None:
                return False
            reservation.append((start, charge))
            pairs = []
            for key, (regions, size) in planned.items():
                cursor, data = start, {}
                for name, region in regions.items():
                    data[name] = cursor, region.nbytes
                    pairs.append((region, cursor))
                    cursor += region.nbytes
                pending.append((key, (data, size, start)))
                start += size
            self._copy(pairs, False)
            return True

        def publish(success):
            if success:
                for key, value in pending:
                    self.objects[key] = value
                    self.nbytes += value[1]
            else:
                for span in reservation:
                    self._release(*span)
            return success

        return self._submit(write, buffers, publish)

    def _submit(self, action, buffers, finish=lambda value: value):
        import torch

        if self.closed:
            raise RuntimeError("cache storage is closed")

        devices = {r.device for b in buffers for r in b.values() if r.is_cuda}
        ready = {d: torch.cuda.current_stream(d).record_event() for d in devices}

        def run():
            if self.halted:
                return False
            try:
                for device, event in ready.items():
                    if device not in self.streams:
                        self.streams[device] = torch.cuda.Stream(device=device)
                    self.streams[device].wait_event(event)
                    torch.cuda.set_stream(self.streams[device])
                with torch.inference_mode():
                    result = action()
            except Exception:
                result = False
            try:
                for device in devices:
                    if device in self.streams:
                        self.streams[device].synchronize()
            except Exception:
                self.halted = True
                _UNRETIRED.append((self, action, finish, buffers, ready))
                return None
            return finish(result)

        return self.executor.submit(run)

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True)
