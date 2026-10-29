"""Shared CUDA storage for sparse graph collectives."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from resin.backend.topology import RankGroup


class SharedWorkspace:
    """Keep FlashInfer's cooperative peer allocation alive beside its tensor view."""

    def __init__(self, group: RankGroup, size: int, *, device: torch.device) -> None:
        from flashinfer.comm.dlpack_utils import pack_strided_memory
        from flashinfer.comm.mnnvl import (
            SymmDeviceMemory,
            TorchDistBackend,
            is_mnnvl_fabric_supported,
        )

        assert device.index is not None
        device_index = device.index
        comm = TorchDistBackend(group.handle)
        if len(
            set(comm.allgather(socket.gethostname()))
        ) != 1 and not is_mnnvl_fabric_supported(device_index):
            raise ValueError(
                "sparse expert graphs require one NVLink host or a shared MNNVL fabric; use eager NCCL for RDMA"
            )
        self.allocation = SymmDeviceMemory(
            size,
            group.size,
            group.local_rank,
            device_index,
            comm_backend_for_handle_transfer=comm,
            enable_multicast=False,
            allocate_signal_pads=False,
        )
        self.tensor = pack_strided_memory(
            int(self.allocation.uc_base_ptr),
            size,
            self.allocation.allocation_size,
            group.size,
            torch.uint8,
            device_index,
        )
