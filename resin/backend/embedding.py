"""Bounded owner lookups across a device group, independent of model format."""

from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from resin.backend.collectives import LayerCollective
from resin.backend.topology import RankGroup


class ShardedEmbeddingLookup(LayerCollective):
    """NCCL prototype: gather IDs, look up owners, scatter BF16 responses.

    Buffers scale with group size and the common token capacity. This is a
    replaceable GPU transport, rather than DeepEP's sparse RDMA table fetch.
    Results alias reusable buffers until the next lookup on this collective.
    """

    def __init__(self, group: RankGroup, dimension: int) -> None:
        super().__init__()
        self.group, self.dimension = group, dimension
        self._buffers = {}
        self._stream: torch.cuda.Stream | None = None
        self._events = {}

    @property
    def allocated(self) -> bool:
        return bool(self._buffers)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self._stream = None
        self._events.clear()
        self._buffers.clear()

    def __call__(
        self, indices: Tensor, local: Callable[[Tensor, Tensor | None], Tensor]
    ) -> Tensor:
        return self._lookup(indices, local)

    def prefetch(
        self,
        requests: Sequence[tuple[Tensor, Callable[[Tensor, Tensor | None], Tensor]]],
    ) -> list[Callable[[], Tensor]]:
        """Issue all table fetches now; each hook waits only for its own result.

        Like DeepInfer's EngramBuffer.fetch boundary, but using the existing
        NCCL owner lookup. Every rank supplies the same ordered table list,
        including ranks with no local tokens. Captured calls use fixed capacity.
        """
        if not requests:
            return []
        device = requests[0][0].device
        if any(indices.device != device for indices, _ in requests):
            raise ValueError("prefetched embeddings must share one device")
        if device.type != "cuda":
            return [
                lambda value=self._lookup(indices, local, slot=slot): value
                for slot, (indices, local) in enumerate(requests)
            ]
        if self._stream is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up embedding prefetch before graph capture")
            self._stream = torch.cuda.Stream(device=device)
        self._stream.wait_stream(torch.cuda.current_stream(device))
        hooks = []
        with torch.cuda.stream(self._stream):
            for slot, (indices, local) in enumerate(requests):
                key = slot, device
                if key not in self._events:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "warm up every embedding table before capture"
                        )
                    self._events[key] = torch.cuda.Event()
                value = self._lookup(indices, local, slot=slot)
                event = self._events[key]
                event.record(self._stream)

                def wait(
                    value: Tensor = value, event: torch.cuda.Event = event
                ) -> Tensor:
                    torch.cuda.current_stream(value.device).wait_event(event)
                    return value

                hooks.append(wait)
        return hooks

    def _lookup(
        self,
        indices: Tensor,
        local: Callable[[Tensor, Tensor | None], Tensor],
        *,
        slot: int | None = None,
    ) -> Tensor:
        shape, columns = indices.shape, indices.shape[-1]
        flat = indices.reshape(-1, columns)
        rows, group = len(flat), self.group
        capacity = self._capacity
        if capacity is None:
            counts = torch.tensor([rows], device=indices.device, dtype=torch.int64)
            received = torch.empty(group.size, device=indices.device, dtype=torch.int64)
            dist.all_gather_into_tensor(received, counts, group=group.handle)
            capacity = max(1, int(received.max().item()))
        if rows > capacity:
            raise ValueError("embedding rows exceed the collective capacity")
        key = (capacity, columns, indices.device) + (() if slot is None else (slot,))
        if key not in self._buffers:
            self._buffers[key] = (
                indices.new_empty((capacity, columns)),
                indices.new_empty((group.size * capacity, columns)),
                torch.empty(
                    (group.size * capacity, columns, self.dimension),
                    device=indices.device,
                    dtype=torch.bfloat16,
                ),
                torch.empty(
                    (capacity, columns, self.dimension),
                    device=indices.device,
                    dtype=torch.bfloat16,
                ),
            )
        source, gathered, partial, output = self._buffers[key]
        source.fill_(-1)
        source[:rows].copy_(flat)
        if self._valid_rows is not None:
            if self._valid_rows.device != indices.device:
                raise ValueError("valid embedding rows must be on the input device")
            source.masked_fill_(
                (torch.arange(capacity, device=indices.device) >= self._valid_rows)[
                    :, None
                ],
                -1,
            )
        dist.all_gather_into_tensor(gathered, source, group=group.handle)
        local(gathered, partial)
        # Each ID has exactly one owner, so the BF16 sum introduces no rounding.
        dist.reduce_scatter_tensor(output, partial, group=group.handle)
        return output[:rows].reshape(*shape, self.dimension)
