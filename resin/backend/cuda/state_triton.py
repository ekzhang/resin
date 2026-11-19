"""Batched direct copies of complete native STATE slots."""

from typing import Any, cast

import torch
import triton
import triton.language as tl

_COPY_BLOCK = 4096


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


def _copy_layout(storage: torch.Tensor) -> tuple[int, int, int]:
    if storage.ndim != 3 or storage.stride(2) != 1:
        raise ValueError("STATE storage must be [slab, slot, contiguous elements]")
    return storage.stride(0), storage.stride(1), storage.shape[2]


def compile_state_checkpoint(storage: torch.Tensor) -> None:
    """Compile the checkpoint variant for this cache layout without launching it."""
    layout = _copy_layout(storage)
    with torch.cuda.device(storage.device):
        cast(Any, _copy_state).warmup(
            storage,
            torch.int64,
            torch.int64,
            torch.int64,
            torch.bool,
            *layout,
            True,
            _COPY_BLOCK,
            grid=(1, triton.cdiv(layout[2], _COPY_BLOCK)),
        )


def copy_state(
    storage: torch.Tensor,
    addresses: torch.Tensor,
    *,
    candidates: torch.Tensor | None = None,
    ends: torch.Tensor | None = None,
) -> torch.Tensor:
    layout = _copy_layout(storage)
    count = len(addresses)
    flags = torch.empty(count, dtype=torch.bool, device=storage.device)
    cast(Any, _copy_state)[(count, triton.cdiv(layout[2], _COPY_BLOCK))](
        storage,
        addresses,
        candidates,
        ends,
        flags,
        *layout,
        candidates is not None,
        _COPY_BLOCK,
    )
    return flags
