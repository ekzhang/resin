"""Small CUDA process-group primitives shared by model parallel backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from resin.backend.topology import RankGroup


@dataclass(frozen=True, slots=True)
class TensorShard:
    """One rank's contiguous interval in a globally ordered tensor axis."""

    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class TensorParallelLayout:
    """A rank coordinate plus deterministic contiguous sharding rules."""

    rank: int = 0
    size: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size <= 0
        ):
            raise ValueError("tensor parallel size must be a positive integer")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 0 <= self.rank < self.size
        ):
            raise ValueError("tensor parallel rank must be in [0, size)")

    @property
    def enabled(self) -> bool:
        return self.size > 1

    def even_shard(self, total: int, *, name: str) -> TensorShard:
        """Split an axis evenly, rejecting geometries that cannot be sharded."""

        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if total % self.size:
            raise ValueError(f"{name}={total} must be divisible by TP={self.size}")
        width = total // self.size
        start = self.rank * width
        return TensorShard(start, start + width)

    def aligned_shard(self, total: int, *, alignment: int, name: str) -> TensorShard:
        """Balance aligned units across ranks, allowing unequal final widths."""

        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(alignment, bool)
            or not isinstance(alignment, int)
            or alignment <= 0
        ):
            raise ValueError("shard alignment must be a positive integer")
        if total % alignment:
            raise ValueError(f"{name}={total} must be divisible by {alignment}")
        units = total // alignment
        if units < self.size:
            raise ValueError(
                f"{name} has only {units} aligned units for TP={self.size}"
            )
        base, extra = divmod(units, self.size)
        start_units = self.rank * base + min(self.rank, extra)
        local_units = base + (self.rank < extra)
        start = start_units * alignment
        return TensorShard(start, start + local_units * alignment)


@dataclass(frozen=True, slots=True)
class TensorParallelContext:
    """One tensor-parallel rank and its device collective group."""

    layout: TensorParallelLayout = TensorParallelLayout()
    process_group: ProcessGroup | None = None

    def __post_init__(self) -> None:
        if self.layout.enabled and self.process_group is None:
            raise ValueError("multi-rank tensor parallelism requires a process group")
        if not self.layout.enabled and self.process_group is not None:
            raise ValueError(
                "single-rank tensor parallelism cannot use a process group"
            )

    @classmethod
    def from_group(cls, group: RankGroup | None) -> TensorParallelContext:
        """Bind one mesh axis, treating singleton groups as local execution."""

        if group is None or group.size == 1:
            return cls()
        return cls(TensorParallelLayout(group.local_rank, group.size), group.handle)

    @property
    def rank(self) -> int:
        return self.layout.rank

    @property
    def size(self) -> int:
        return self.layout.size

    @property
    def enabled(self) -> bool:
        return self.layout.enabled

    def sum(self, tensor: Tensor) -> Tensor:
        """Sum a row-parallel partial result in place on every rank."""

        if self.process_group is not None:
            dist.all_reduce(tensor, group=self.process_group)
        return tensor


def replicated_cache_capacity(capacity: int, group: RankGroup | None) -> int:
    """Agree on cache capacity before ranks allocate the same logical slots.

    ``group`` is the host-side attention replica group, including every KV head
    shard and replicated attention copy driven by the same scheduler.
    """

    if group is None or group.size == 1:
        return capacity
    # Gloo writes on worker threads outside the caller's inference-mode scope.
    with torch.inference_mode(False):
        value = torch.tensor(capacity, dtype=torch.int64, device="cpu")
        dist.all_reduce(value, op=dist.ReduceOp.MIN, group=group.handle)
    return int(value.item())


class FlashInferAllReduce:
    """Use FlashInfer's low-latency all-reduce for one small tensor geometry.

    Tensors outside the configured dtype, hidden width, or token limit fall
    back to the process group's ordinary collective. This lets decode use the
    custom kernel without reserving a context-sized communication workspace
    for prefill.
    """

    def __init__(
        self,
        context: TensorParallelContext,
        *,
        max_token_num: int,
        hidden_dim: int,
        dtype: Any,
        enabled: bool = True,
    ) -> None:
        if max_token_num <= 0 or hidden_dim <= 0:
            raise ValueError("FlashInfer all-reduce geometry must be positive")
        self.context = context
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.workspace: Any | None = None
        self.backend = "local" if not context.enabled else "nccl"
        if context.enabled and enabled:
            from flashinfer.comm import create_allreduce_fusion_workspace
            from flashinfer.comm.mnnvl import TorchDistBackend

            self.workspace = create_allreduce_fusion_workspace(
                backend="auto",
                world_size=context.size,
                rank=context.rank,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
                comm_backend=TorchDistBackend(group=context.process_group),
                group=context.process_group,
            )
            self.backend = self.workspace.backend

    def sum(self, tensor: Tensor) -> Tensor:
        """Sum a matching tensor out of place, otherwise use NCCL in place."""

        workspace = self.workspace
        if (
            workspace is None
            or tensor.dtype != self.dtype
            or tensor.shape[-1] != self.hidden_dim
            or tensor.numel() // self.hidden_dim > self.max_token_num
            or not tensor.is_contiguous()
        ):
            return self.context.sum(tensor)

        from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion

        shape = tensor.shape
        output = allreduce_fusion(
            input=tensor.reshape(-1, self.hidden_dim),
            workspace=workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=False,
            trigger_completion_at_end=True,
        )
        return output.view(shape)


__all__ = [
    "FlashInferAllReduce",
    "TensorParallelContext",
    "TensorParallelLayout",
    "TensorShard",
    "replicated_cache_capacity",
]
