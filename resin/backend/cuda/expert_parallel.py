"""GPU-driven owner-sparse MoE transport using FlashInfer's NVLink workspace."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import torch
from torch import Tensor

if TYPE_CHECKING:
    from resin.backend.topology import RankGroup


_MAX_DESTINATIONS_PER_PASS = 22


class SparseExpertTransport:
    """Fixed receive storage with device-resident, sparse dispatch/combine counts.

    FlashInfer writes each token only to its selected destination ranks. Static
    receive slots are masked before the local expert operation; their presence
    does not cause padded network transfers. Initialization belongs to warmup,
    and all subsequent routing, synchronization and combination is capturable.
    """

    def __init__(
        self,
        group: RankGroup,
        capacity: int,
        width: int,
        dtype: torch.dtype,
        top_k: int,
        device: torch.device,
    ) -> None:
        from flashinfer.comm.trtllm_moe_alltoall import (
            moe_a2a_get_workspace_size_per_rank,
            moe_a2a_initialize,
        )

        from resin.backend.cuda.shared_workspace import SharedWorkspace

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up expert dispatch before CUDA graph capture")
        if group.size > 64:
            raise ValueError(
                "FlashInfer sparse expert transport supports up to 64 ranks"
            )
        if group.ranks != tuple(range(group.size)):
            raise ValueError(
                "CUDA expert transport requires the full rank domain; pass groups=ParallelGroups"
            )
        self.group, self.capacity = group, capacity
        # Physical GPUs are transport destinations, independent of the model's
        # expert numbering and of its attention/expert tensor-parallel meshes.
        workspace_bytes = moe_a2a_get_workspace_size_per_rank(
            group.size,
            capacity,
            width * torch.empty((), dtype=dtype).element_size() + top_k * 8,
            width * 4,  # Preserve FP32 expert partials through combination.
        )
        self.allocation = SharedWorkspace(group, workspace_bytes, device=device)
        self.workspace = self.allocation.tensor
        self.metainfo = moe_a2a_initialize(
            self.workspace, group.local_rank, group.size, capacity
        )
        # No peer may publish routing flags into an allocation another rank is
        # still zeroing during its first warmup call.
        torch.distributed.barrier(group=group.handle)

    def __call__(
        self,
        local: Callable[[Tensor, Tensor, Tensor], Tensor],
        hidden: Tensor,
        indices: Tensor,
        weights: Tensor,
        owners: Tensor,
        active: Tensor,
        num_experts: int,
    ) -> Tensor:
        from flashinfer.comm.trtllm_moe_alltoall import (
            moe_a2a_combine,
            moe_a2a_dispatch,
            moe_a2a_sanitize_expert_ids,
        )

        from resin.backend.cuda.expert_parallel_kernels import (
            prepare_receive,
            prepare_routes,
        )

        capacity, world_size = self.capacity, self.group.size
        # The pinned primitive specializes K in {1,2,4,6,8,10,16,22}. Usual
        # top-k routing fits one pass. Large expert-TP fanout is partitioned by
        # physical rank, retaining owner-sparse writes in every pass.
        routes = prepare_routes(
            owners,
            active,
            indices,
            weights,
            world_size=world_size,
            rank=self.group.local_rank,
            num_experts=num_experts,
            max_destinations_per_pass=_MAX_DESTINATIONS_PER_PASS,
        )

        result = None
        for route, payload_indices, payload_weights in routes:
            k = route.shape[1]
            dispatched = moe_a2a_dispatch(
                route,
                [payload_indices, payload_weights, hidden],
                self.workspace,
                self.metainfo,
                capacity,
                self.group.local_rank,
                world_size,
                k,
                world_size,
            )
            if len(dispatched) == 3:
                # FlashInfer 0.7 adds optional EPLB gathered statistics. This
                # transport does not enable EPLB at initialization or dispatch;
                # the payload and combine-offset contracts remain unchanged.
                payloads, combine_offset, eplb_stats = dispatched
                if eplb_stats is not None:
                    raise RuntimeError("unexpected FlashInfer EPLB statistics")
            else:
                # FlashInfer 0.6 returns just the payloads and combine offset.
                payloads, combine_offset = dispatched
            received_indices, received_weights, received_hidden = payloads
            moe_a2a_sanitize_expert_ids(
                received_indices,
                self.workspace,
                self.metainfo,
                self.group.local_rank,
                num_experts,
            )
            # The transport ABI stores routing weights in FP32. Local expert
            # kernels retain the caller's dtype contract (BF16 for DeepSeek).
            received_hidden, received_weights = prepare_receive(
                received_hidden,
                received_indices,
                received_weights,
                num_experts=num_experts,
                weight_dtype=weights.dtype,
            )
            partial = local(
                received_hidden,
                received_indices.flatten(0, 1),
                received_weights,
            )
            combined = moe_a2a_combine(
                partial.reshape(world_size, capacity, -1).contiguous(),
                capacity,
                self.workspace,
                self.metainfo,
                capacity,
                self.group.local_rank,
                world_size,
                k,
                combine_offset,
            )
            result = combined if result is None else result + combined
        assert result is not None
        return result


_TRANSPORTS: WeakValueDictionary[tuple, SparseExpertTransport] = WeakValueDictionary()


def sparse_expert_transport(
    group: RankGroup,
    capacity: int,
    width: int,
    dtype: torch.dtype,
    top_k: int,
    device: torch.device,
) -> SparseExpertTransport:
    """Share layer workspaces without retaining closed models or process groups."""
    key = group, capacity, width, dtype, top_k, device
    transport = _TRANSPORTS.get(key)
    if transport is None:
        with torch.cuda.device(device):
            transport = SparseExpertTransport(
                group, capacity, width, dtype, top_k, device
            )
        _TRANSPORTS[key] = transport
    return transport
