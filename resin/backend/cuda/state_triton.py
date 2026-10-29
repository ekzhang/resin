"""Batched direct copies of complete native STATE slots."""

from typing import Any, cast

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_state(
    Storage,
    Addresses,
    Candidates,
    Ends,
    Flags,
    SLAB_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    CHECKPOINT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    item, tile = tl.program_id(0), tl.program_id(1)
    source_slab = tl.load(Addresses + item * 4)
    source_slot = tl.load(Addresses + item * 4 + 1)
    destination_slab = tl.load(Addresses + item * 4 + 2)
    destination_slot = tl.load(Addresses + item * 4 + 3)
    if CHECKPOINT:
        row = tl.load(Candidates + item * 2)
        prefix = tl.load(Candidates + item * 2 + 1)
        materialized = tl.load(Ends + row) == prefix
        if tile == 0:
            tl.store(Flags + item, materialized)
    else:
        materialized = True
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    source = source_slab * SLAB_STRIDE + source_slot * SLOT_STRIDE + offset
    destination = (
        destination_slab * SLAB_STRIDE + destination_slot * SLOT_STRIDE + offset
    )
    values = tl.load(
        Storage + source, (offset < WIDTH) & materialized & (source_slab != 0), other=0
    )
    tl.store(
        Storage + destination,
        values,
        (offset < WIDTH) & materialized & (destination_slab != 0),
    )


def copy_state(
    storage: torch.Tensor,
    addresses: torch.Tensor,
    *,
    candidates: torch.Tensor | None = None,
    ends: torch.Tensor | None = None,
) -> torch.Tensor:
    if storage.ndim != 3 or storage.stride(2) != 1:
        raise ValueError("STATE storage must be [slab, slot, contiguous elements]")
    count, width = len(addresses), storage.shape[2]
    flags = torch.empty(count, dtype=torch.bool, device=storage.device)
    cast(Any, _copy_state)[(count, triton.cdiv(width, 4096))](
        storage,
        addresses,
        candidates,
        ends,
        flags,
        storage.stride(0),
        storage.stride(1),
        width,
        candidates is not None,
        4096,
    )
    return flags
