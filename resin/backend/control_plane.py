"""Small synchronous rank votes without object serialization on successful steps."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import torch

from resin.backend.topology import RankGroup


class ControlPlaneError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _VoteBuffers:
    send: torch.Tensor
    receive: torch.Tensor
    rows: list[torch.Tensor]


class ControlPlane:
    """Gather fixed-size integer records, then broadcast the complete table.

    Every operation uses the same shapes and collectives, including the operation
    tag check; ranks calling different step operations therefore fail together.
    Error text takes a separate object gather only when a rank reports failure.
    """

    _fields = 6
    _width = _fields + 2  # Operation tag, error flag, then integer payload.

    def __init__(self, distributed):
        self.distributed = distributed
        self._buffers: dict[tuple[int, tuple[int, ...]], _VoteBuffers] = {}
        self._tags: dict[str, int] = {}

    def gather(
        self,
        group: RankGroup,
        operation: str,
        values=(),
        *,
        error: str | None = None,
    ) -> tuple[list[tuple[int, ...]], list[str | None]]:
        values = tuple(map(int, values))
        if len(values) > self._fields:
            raise ValueError(f"control records accept at most {self._fields} integers")
        if operation not in self._tags:
            self._tags[operation] = (
                int.from_bytes(blake2b(operation.encode(), digest_size=8).digest())
                & ((1 << 62) - 1)
            ) | (1 << 62)
        tag = self._tags[operation]
        if group.size == 1:
            return [values], [error]

        key = id(group.handle), group.ranks
        buffers = self._buffers.get(key)
        if buffers is None:
            # Votes can first run inside a model's inference scope, then be
            # reused outside it. Gloo also writes receive on its worker thread.
            # Both buffers must therefore remain ordinary mutable tensors.
            with torch.inference_mode(False):
                send = torch.empty(self._width, dtype=torch.int64, device="cpu")
                receive = torch.empty(
                    (group.size, self._width), dtype=torch.int64, device="cpu"
                )
                buffers = _VoteBuffers(send, receive, list(receive.unbind(0)))
            self._buffers[key] = buffers
        send, receive = buffers.send, buffers.receive
        send.zero_()
        send[0], send[1] = tag, int(error is not None)
        for column, value in enumerate(values, 2):
            send[column] = value
        # Gathering to the source and broadcasting the table preserves every
        # rank's exact record with lower Gloo latency for small, wide votes.
        self.distributed.gather(
            send,
            gather_list=buffers.rows if group.local_rank == 0 else None,
            dst=group.source,
            group=group.handle,
        )
        self.distributed.broadcast(receive, src=group.source, group=group.handle)
        records = receive.tolist()
        if any(record[0] != tag for record in records):
            raise ControlPlaneError(
                "ranks called collective engine operations in different orders"
            )
        errors = [None] * group.size
        if any(record[1] for record in records):
            self.distributed.all_gather_object(errors, error, group=group.handle)
        return [tuple(record[2 : 2 + len(values)]) for record in records], errors
