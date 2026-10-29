"""Bounded, content-addressed vision tensors owned by one model runner.

Only immutable tensor results and content digests survive request retirement.
Request-local media objects, prompts and placement maps stay with the runner's
existing request bookkeeping. Callers must treat returned tensors as read-only.
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch

VisionKey = str | tuple[str, int]
VisionTensors = tuple[torch.Tensor | None, ...]


@dataclass(frozen=True, slots=True)
class _Entry:
    tensors: VisionTensors
    size_bytes: int
    ready: tuple[tuple[torch.device, torch.cuda.Event], ...]


class VisionFeatureCache:
    """An LRU bounded independently by tensor bytes and entry count.

    The bounds cover reusable entries; active requests can still own evicted
    tensors until they finish. Oversized results are not retained. All cache
    insertion happens outside CUDA capture, so cloned results cannot alias a
    graph pool or an encoder's reusable workspace.
    """

    def __init__(
        self, *, max_entries: int = 64, max_bytes: int = 256 * 1024**2
    ) -> None:
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("vision cache limits must be nonnegative")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[VisionKey, _Entry] = OrderedDict()
        self.size_bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: VisionKey) -> VisionTensors | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        for device, event in entry.ready:
            stream = torch.cuda.current_stream(device)
            stream.wait_event(event)
            for tensor in entry.tensors:
                if tensor is not None and tensor.device == device:
                    tensor.record_stream(stream)
        return entry.tensors

    def put(self, key: VisionKey, tensors: VisionTensors) -> VisionTensors:
        devices = {t.device for t in tensors if t is not None and t.is_cuda}
        for device in devices:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "vision cache insertion must be outside CUDA capture"
                    )
        size = sum(t.numel() * t.element_size() for t in tensors if t is not None)
        old = self._entries.pop(key, None)
        if old is not None:
            self.size_bytes -= old.size_bytes
        retain = bool(self.max_entries and self.max_bytes and size <= self.max_bytes)
        while (
            retain
            and self._entries
            and (
                len(self._entries) >= self.max_entries
                or self.size_bytes + size > self.max_bytes
            )
        ):
            _, evicted = self._entries.popitem(last=False)
            self.size_bytes -= evicted.size_bytes
        with torch.inference_mode():
            owned = tuple(
                None
                if t is None
                else t.detach().clone(memory_format=torch.contiguous_format)
                for t in tensors
            )
        if not retain:
            return owned
        ready = []
        for device in devices:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            ready.append((device, event))
        self._entries[key] = _Entry(owned, size, tuple(ready))
        self.size_bytes += size
        return owned

    def clear(self) -> None:
        self._entries.clear()
        self.size_bytes = 0
