"""Contiguous query ownership and fixed-shape context-parallel token exchange."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import accumulate

import torch
import torch.distributed as dist

from resin import BatchPhase
from resin.backend.cuda.paged_kv import DevicePagedAttentionBatch
from resin.backend.topology import RankGroup


@dataclass(frozen=True, slots=True)
class ContextParallelPlan:
    """Partition one packed token axis; each sequence has one segment per rank."""

    rank: int
    size: int
    start: int
    stop: int
    global_lengths: tuple[int, ...]
    query_lengths: tuple[int, ...]
    token_indices: tuple[int, ...]
    gather_indices: tuple[int, ...]
    suffix_tokens: tuple[int, ...]
    capacity: int

    @classmethod
    def build(
        cls, query_lengths: Iterable[int], *, rank: int, size: int
    ) -> ContextParallelPlan:
        lengths = tuple(query_lengths)
        if type(size) is not int or size <= 0:
            raise ValueError("context-parallel size must be a positive integer")
        if type(rank) is not int or not 0 <= rank < size:
            raise ValueError("context-parallel rank is outside its group")
        if any(type(length) is not int or length < 0 for length in lengths):
            raise ValueError("query lengths must be nonnegative integers")
        offsets = tuple(accumulate(lengths, initial=0))
        total = offsets[-1]
        start, stop = total * rank // size, total * (rank + 1) // size
        local_lengths, suffix = [], []
        for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
            local_start = min(max(start, begin), end)
            local_stop = min(max(stop, begin), end)
            local_lengths.append(local_stop - local_start)
            suffix.append(end - local_stop)
        capacity = (total + size - 1) // size
        gather = tuple(
            source * capacity + index
            for source in range(size)
            for index in range(total * (source + 1) // size - total * source // size)
        )
        return cls(
            rank,
            size,
            start,
            stop,
            lengths,
            tuple(local_lengths),
            tuple(range(start, stop)),
            gather,
            tuple(suffix),
            capacity,
        )

    @property
    def local_tokens(self) -> int:
        return self.stop - self.start

    @property
    def total_tokens(self) -> int:
        return sum(self.global_lengths)


class ContextParallelTokens:
    """Prepare maps once, then select/gather token tensors without host reads.

    Local tensors contain actual owned rows. Gather pads each rank to a common
    capacity and removes that padding using a prepared device map. Collective
    shapes depend only on the host plan, making gather CUDA-graph capturable.
    """

    def __init__(
        self,
        query_lengths: Iterable[int],
        group: RankGroup,
        *,
        device: str | torch.device,
        distributed=dist,
    ) -> None:
        self.plan = ContextParallelPlan.build(
            query_lengths, rank=group.local_rank, size=group.size
        )
        self.group, self.distributed = group, distributed
        self.device = torch.device(device)
        self.token_indices = torch.tensor(
            self.plan.token_indices, device=self.device, dtype=torch.long
        )
        self.gather_indices = torch.tensor(
            self.plan.gather_indices, device=self.device, dtype=torch.long
        )
        self.suffix_tokens = torch.tensor(
            self.plan.suffix_tokens, device=self.device, dtype=torch.int32
        )
        self.active_rows = torch.tensor(
            [row for row, length in enumerate(self.plan.query_lengths) if length],
            device=self.device,
            dtype=torch.long,
        )
        self.cu_seqlens_q = torch.tensor(
            tuple(accumulate((n for n in self.plan.query_lengths if n), initial=0)),
            device=self.device,
            dtype=torch.int32,
        )
        self.max_seqlen_q = max(self.plan.query_lengths, default=0)

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        """Select this rank's contiguous segment from global packed rows."""

        if tensor.shape[0] != self.plan.total_tokens:
            raise ValueError("global token tensor does not match its context plan")
        if tensor.device != self.device:
            raise ValueError("context-parallel tensors must share the prepared device")
        return tensor.index_select(0, self.token_indices)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Restore every rank's actual rows to their original packed order."""

        if tensor.shape[0] != self.plan.local_tokens:
            raise ValueError("local token tensor does not match its context plan")
        if tensor.device != self.device:
            raise ValueError("context-parallel tensors must share the prepared device")
        if self.group.size == 1 or self.plan.capacity == 0:
            return tensor
        padded = tensor.new_zeros((self.plan.capacity, *tensor.shape[1:]))
        padded[: self.plan.local_tokens].copy_(tensor)
        gathered = tensor.new_empty(
            (self.group.size * self.plan.capacity, *tensor.shape[1:])
        )
        self.distributed.all_gather_into_tensor(
            gathered, padded, group=self.group.handle
        )
        return gathered.index_select(0, self.gather_indices)


def context_parallel_query_batch(
    batch: DevicePagedAttentionBatch, tokens: ContextParallelTokens
) -> DevicePagedAttentionBatch:
    """Lower local Q metadata against globally reconstructed paged K/V.

    FlashAttention aligns causal queries to the end of each K sequence. Trim
    K lengths by the global query suffix following this rank's segment, so
    that local queries retain their original causal positions. Exclude rows
    with no local query. Page tables and append slots still refer to global
    cache storage; callers write gathered K/V with the original batch first.
    """

    if batch.phase != BatchPhase.PREFILL:
        raise ValueError("context-parallel query sharding is only used for prefill")
    if batch.token_indices.shape[0] != tokens.plan.total_tokens:
        raise ValueError("attention batch does not match its context token plan")
    groups = tuple(
        (
            name,
            replace(
                group,
                page_table=group.page_table.index_select(0, tokens.active_rows),
                lengths=(group.lengths - tokens.suffix_tokens).index_select(
                    0, tokens.active_rows
                ),
                min_length=0,
            ),
        )
        for name, group in batch.groups
    )
    return replace(
        batch,
        token_indices=tokens.select(batch.token_indices),
        sequence_indices=batch.sequence_indices.index_select(0, tokens.active_rows),
        cu_seqlens_q=tokens.cu_seqlens_q,
        max_seqlen_q=tokens.max_seqlen_q,
        groups=groups,
    )


__all__ = [
    "ContextParallelPlan",
    "ContextParallelTokens",
    "context_parallel_query_batch",
]
