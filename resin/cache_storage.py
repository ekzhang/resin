"""A DRAM LRU for immutable objects, budgeting keys, region names and payloads."""

from __future__ import annotations

from collections import OrderedDict
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


class MemoryCacheStorage:
    def __init__(self, max_bytes):
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes, self.nbytes = max_bytes, 0
        self.objects, self.executor = OrderedDict(), ThreadPoolExecutor(1)

    def lookup(self, keys):
        return self.executor.submit(lambda: tuple(key in self.objects for key in keys))

    def load(self, keys, buffers):
        def read():
            for key, regions in zip(keys, buffers, strict=True):
                data, _ = self.objects[key]
                if data.keys() != regions.keys() or any(
                    len(data[name]) != region.nbytes for name, region in regions.items()
                ):
                    return False
                self.objects.move_to_end(key)
                import torch

                for name, region in regions.items():
                    chunk = data[name]
                    if not chunk:
                        continue
                    with torch.inference_mode():
                        region.view(-1).copy_(
                            torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
                        )
            return True

        return self._submit(read, buffers)

    def store(self, keys, buffers):
        def write():
            success = True
            for key, regions in zip(keys, buffers, strict=True):
                if key in self.objects:
                    continue
                size = len(key) + sum(
                    len(name.encode()) + region.nbytes
                    for name, region in regions.items()
                )
                if size > self.max_bytes:
                    success = False
                    continue
                data = {
                    name: region.detach().cpu().numpy().tobytes()
                    for name, region in regions.items()
                }
                while self.nbytes + size > self.max_bytes:
                    _, (_, old_size) = self.objects.popitem(last=False)
                    self.nbytes -= old_size
                self.objects[key] = data, size
                self.nbytes += size
            return success

        return self._submit(write, buffers)

    def _submit(self, action, buffers):
        devices = {
            r.device for regions in buffers for r in regions.values() if r.is_cuda
        }
        ready = {}
        if devices:
            import torch

            ready = {d: torch.cuda.current_stream(d).record_event() for d in devices}

        def run():
            streams = []
            try:
                for device, event in ready.items():
                    stream = torch.cuda.Stream(device=device)
                    streams.append(stream)
                    stream.wait_event(event)
                    torch.cuda.set_stream(stream)
                result = action()
            except Exception:
                result = False
            try:
                for stream in streams:
                    stream.synchronize()
            except Exception:
                return None
            return result

        return self.executor.submit(run)

    def close(self):
        self.executor.shutdown(wait=True)
