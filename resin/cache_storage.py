"""Immutable opaque objects; every call retires all buffer access, even on error.

Buffers are tuples of contiguous byte regions (memoryviews or uint8 tensors).
Providers own eviction and object lifetimes independently of any Engine.
"""

from collections import OrderedDict
from threading import RLock
from typing import Protocol


class CacheStorage(Protocol):
    def lookup(self, keys: tuple[bytes, ...]) -> tuple[bool, ...]: ...
    def load(self, keys: tuple[bytes, ...], buffers: tuple) -> tuple[bool, ...]: ...
    def store(self, keys: tuple[bytes, ...], buffers: tuple) -> tuple[bool, ...]: ...


class MemoryCacheStorage:
    """Bounded DRAM LRU with immutable, idempotent writes and blocking CUDA copies."""

    def __init__(self, max_bytes=1 << 30):
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes, self.nbytes = max_bytes, 0
        self.objects, self.lock = OrderedDict(), RLock()

    def lookup(self, keys):
        with self.lock:
            return tuple(key in self.objects for key in keys)

    def store(self, keys, buffers):
        result = []
        with self.lock:
            for key, regions in zip(keys, buffers, strict=True):
                if (
                    key not in self.objects
                    and sum(r.nbytes for r in regions) <= self.max_bytes
                ):
                    data = b"".join(
                        region.detach().cpu().numpy().tobytes()
                        if hasattr(region, "device")
                        else bytes(memoryview(region))
                        for region in regions
                    )
                    while self.nbytes + len(data) > self.max_bytes:
                        self.nbytes -= len(self.objects.popitem(last=False)[1])
                    self.objects[key] = data
                    self.nbytes += len(data)
                result.append(key in self.objects)
        return tuple(result)

    def load(self, keys, buffers):
        result = []
        with self.lock:
            for key, regions in zip(keys, buffers, strict=True):
                data = self.objects.get(key)
                found = data is not None and len(data) == sum(r.nbytes for r in regions)
                result.append(found)
                if not found:
                    continue
                self.objects.move_to_end(key)
                offset = 0
                for region in regions:
                    chunk = data[offset : offset + region.nbytes]
                    if hasattr(region, "device"):
                        import torch

                        try:
                            region.reshape(-1).copy_(
                                torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
                            )
                        finally:
                            if region.is_cuda:
                                torch.cuda.current_stream(region.device).synchronize()
                    else:
                        memoryview(region).cast("B")[:] = chunk
                    offset += region.nbytes
        return tuple(result)
