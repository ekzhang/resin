"""Copy Engine-owned states into layer-major working storage.

The Engine's physical layout and ownership are unchanged. Initial states remain
independent copies for speculative rollback. Only scatter writes final states to
leased destinations, preserving the existing address and null semantics.
Request positions, table widths, parent IDs and child slots are runtime data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from resin_models_cuda.glm5_3_flash.cache import DeviceCache


@triton.jit(
    do_not_specialize=[
        "batch",
        "table_width",
        "table_stride",
        "recurrent_parent_stride",
        "recurrent_slot_stride",
        "convolution_parent_stride",
        "convolution_slot_stride",
        "hidden_parent_stride",
        "hidden_slot_stride",
    ]
)
def _gather_state(
    Recurrent,
    Convolution,
    Hidden,
    Parents,
    Slots,
    FirstBlocks,
    Contexts,
    RecurrentOut,
    ConvolutionOut,
    HiddenOut,
    batch,
    table_width,
    table_stride,
    recurrent_parent_stride,
    recurrent_slot_stride,
    convolution_parent_stride,
    convolution_slot_stride,
    hidden_parent_stride,
    hidden_slot_stride,
    STATE_TOKENS: tl.constexpr,
    RECURRENT_WIDTH: tl.constexpr,
    CONVOLUTION_WIDTH: tl.constexpr,
    HIDDEN_WIDTH: tl.constexpr,
    RECURRENT_BLOCKS: tl.constexpr,
    CONVOLUTION_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    LIVE_STATE: tl.constexpr,
):
    tile, sequence, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    context = tl.load(Contexts + sequence).to(tl.int64)
    first = tl.load(FirstBlocks + sequence).to(tl.int64)
    column = 0 if LIVE_STATE else tl.maximum(context - 1, 0) // STATE_TOKENS - first
    valid = (column >= 0) & (column < table_width)
    parent = tl.load(
        Parents + sequence.to(tl.int64) * table_stride + column, valid, other=0
    ).to(tl.int64)
    slot = tl.load(
        Slots + sequence.to(tl.int64) * table_stride + column, valid, other=0
    ).to(tl.int64)
    # Invalid table columns map to the reserved null state, as slab_locations
    # does. Fresh requests instead produce zero without reading stale lease data.
    live = context != 0
    offsets = tl.arange(0, BLOCK)
    output_row = layer.to(tl.int64) * batch + sequence
    if tile < RECURRENT_BLOCKS:
        offsets += tile * BLOCK
        source = (
            parent * recurrent_parent_stride
            + slot * recurrent_slot_stride
            + layer * RECURRENT_WIDTH
            + offsets
        )
        values = tl.load(
            Recurrent + source, live & (offsets < RECURRENT_WIDTH), other=0
        )
        tl.store(
            RecurrentOut + output_row * RECURRENT_WIDTH + offsets,
            values,
            offsets < RECURRENT_WIDTH,
        )
    elif tile < RECURRENT_BLOCKS + CONVOLUTION_BLOCKS:
        offsets += (tile - RECURRENT_BLOCKS) * BLOCK
        source = (
            parent * convolution_parent_stride
            + slot * convolution_slot_stride
            + layer * CONVOLUTION_WIDTH
            + offsets
        )
        values = tl.load(
            Convolution + source, live & (offsets < CONVOLUTION_WIDTH), other=0
        )
        tl.store(
            ConvolutionOut + output_row * CONVOLUTION_WIDTH + offsets,
            values,
            offsets < CONVOLUTION_WIDTH,
        )
    elif layer == 0:
        offsets += (tile - RECURRENT_BLOCKS - CONVOLUTION_BLOCKS) * BLOCK
        source = parent * hidden_parent_stride + slot * hidden_slot_stride + offsets
        values = tl.load(Hidden + source, live & (offsets < HIDDEN_WIDTH), other=0)
        tl.store(
            HiddenOut + sequence * HIDDEN_WIDTH + offsets,
            values,
            offsets < HIDDEN_WIDTH,
        )


def gather_state_layer_major(
    cache: DeviceCache,
    contexts: torch.Tensor,
    *,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return owned ``[layer,batch,...]`` states and ``[batch,hidden]`` values.

    ``out`` permits preallocated graph-owned storage. Initial state must not
    alias an Engine-owned slab: speculative rollback retains it after final
    states have been computed and committed.
    """
    geometry = cache.geometry
    if contexts.ndim != 1 or contexts.dtype not in (torch.int32, torch.int64):
        raise ValueError("contexts must be a vector of integer token counts")
    if not contexts.is_cuda or not contexts.is_contiguous():
        raise ValueError("contexts must be contiguous CUDA storage")
    recurrent, convolution, hidden = (
        cache.recurrent_states,
        cache.convolution_states,
        cache.target_hidden_states,
    )
    group = cache.groups[1]
    batch = contexts.numel()
    layers = len(geometry.linear_layers)
    if (
        group.slab_ids.shape != group.slotnums.shape
        or group.slab_ids.shape[0] != batch
        or group.first_blocks.shape != (batch,)
        or not group.slab_ids.is_contiguous()
        or not group.slotnums.is_contiguous()
        or not group.first_blocks.is_contiguous()
    ):
        raise ValueError("state page tables must have one contiguous row per sequence")
    source_tensors = (recurrent, convolution, hidden)
    table_tensors = (group.slab_ids, group.slotnums, group.first_blocks)
    if any(
        value.device != contexts.device for value in (*source_tensors, *table_tensors)
    ):
        raise ValueError("state storage, page tables and contexts must share a device")
    specs = (
        (
            (layers, batch, geometry.heads, geometry.head_dim, geometry.head_dim),
            torch.float32,
        ),
        (
            (
                layers,
                batch,
                3,
                geometry.heads * geometry.head_dim,
                geometry.conv_history,
            ),
            torch.bfloat16,
        ),
        ((batch, geometry.hidden_size), torch.bfloat16),
    )
    if out is None:
        buffers = [
            torch.empty(shape, dtype=dtype, device=contexts.device)
            for shape, dtype in specs
        ]
        out = buffers[0], buffers[1], buffers[2]
    elif len(out) != 3 or any(
        tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != contexts.device
        or not value.is_contiguous()
        for value, (shape, dtype) in zip(out, specs, strict=True)
    ):
        raise ValueError("gather outputs must be contiguous layer-major CUDA tensors")
    source_allocations = {
        value.untyped_storage().data_ptr() for value in source_tensors
    }
    if any(value.untyped_storage().data_ptr() in source_allocations for value in out):
        raise ValueError(
            "initial-state working copies must not alias Engine-owned storage"
        )
    output_allocations = [
        value.untyped_storage().data_ptr() for value in out if value.numel()
    ]
    if len(set(output_allocations)) != len(output_allocations):
        raise ValueError("gather output buffers must have independent storage")
    if not batch or not layers:
        if batch:
            raise ValueError("GLM state gather requires at least one recurrent layer")
        return out
    recurrent_width = geometry.heads * geometry.head_dim**2
    convolution_width = 3 * geometry.heads * geometry.head_dim * geometry.conv_history
    block = 4096
    recurrent_blocks = triton.cdiv(recurrent_width, block)
    convolution_blocks = triton.cdiv(convolution_width, block)
    blocks = (
        recurrent_blocks + convolution_blocks + triton.cdiv(geometry.hidden_size, block)
    )
    cast(Any, _gather_state)[(blocks, batch, layers)](
        recurrent,
        convolution,
        hidden,
        group.slab_ids,
        group.slotnums,
        group.first_blocks,
        contexts,
        *out,
        batch,
        group.slab_ids.shape[1],
        group.slab_ids.stride(0),
        recurrent.stride(0),
        recurrent.stride(1),
        convolution.stride(0),
        convolution.stride(1),
        hidden.stride(0),
        hidden.stride(1),
        geometry.state_tokens,
        recurrent_width,
        convolution_width,
        geometry.hidden_size,
        recurrent_blocks,
        convolution_blocks,
        block,
        LIVE_STATE=group.live_state,
        num_warps=4,
    )
    return out


@triton.jit(
    do_not_specialize=[
        "batch",
        "table_width",
        "table_stride",
        "r_parent",
        "r_slot",
        "c_parent",
        "c_slot",
        "h_parent",
        "h_slot",
    ]
)
def _scatter_state(
    Recurrent,
    Convolution,
    Hidden,
    Parents,
    Slots,
    FirstBlocks,
    Positions,
    RInputs,
    CInputs,
    HInput,
    batch,
    table_width,
    table_stride,
    r_parent,
    r_slot,
    c_parent,
    c_slot,
    h_parent,
    h_slot,
    STATE_TOKENS: tl.constexpr,
    RWIDTH: tl.constexpr,
    CWIDTH: tl.constexpr,
    HWIDTH: tl.constexpr,
    RBLOCKS: tl.constexpr,
    CBLOCKS: tl.constexpr,
    LAYERS: tl.constexpr,
    BLOCK: tl.constexpr,
    LIVE_STATE: tl.constexpr,
):
    tile, sequence, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    position = tl.load(Positions + sequence).to(tl.int64)
    first = tl.load(FirstBlocks + sequence).to(tl.int64)
    # torch.div(..., rounding_mode='floor') also handles -1 for idle rows.
    logical = tl.where(
        position < 0,
        -((-position + STATE_TOKENS - 1) // STATE_TOKENS),
        position // STATE_TOKENS,
    )
    column = 0 if LIVE_STATE else logical - first
    valid = (column >= 0) & (column < table_width)
    if LIVE_STATE:
        valid = valid & (position >= 0)
    parent = tl.load(
        Parents + sequence.to(tl.int64) * table_stride + column, valid, other=0
    ).to(tl.int64)
    slot = tl.load(
        Slots + sequence.to(tl.int64) * table_stride + column, valid, other=0
    ).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    # Separate final layer allocations are consumed directly. Tensor arguments
    # keep their lifetimes visible to PyTorch/CUDA graph capture; no raw pointer
    # table, device-to-host transfer, or extra stacked state allocation exists.
    if tile < RBLOCKS:
        offset += tile * BLOCK
        for index in tl.static_range(LAYERS):
            if layer == index:
                value = tl.load(
                    RInputs[index] + sequence.to(tl.int64) * RWIDTH + offset,
                    offset < RWIDTH,
                    other=0,
                )
                address = (
                    parent * r_parent
                    + slot * r_slot
                    + layer.to(tl.int64) * RWIDTH
                    + offset
                )
                tl.store(Recurrent + address, value, offset < RWIDTH)
    elif tile < RBLOCKS + CBLOCKS:
        offset += (tile - RBLOCKS) * BLOCK
        for index in tl.static_range(LAYERS):
            if layer == index:
                value = tl.load(
                    CInputs[index] + sequence.to(tl.int64) * CWIDTH + offset,
                    offset < CWIDTH,
                    other=0,
                )
                address = (
                    parent * c_parent
                    + slot * c_slot
                    + layer.to(tl.int64) * CWIDTH
                    + offset
                )
                tl.store(Convolution + address, value, offset < CWIDTH)
    elif layer == 0:
        offset += (tile - RBLOCKS - CBLOCKS) * BLOCK
        value = tl.load(
            HInput + sequence.to(tl.int64) * HWIDTH + offset, offset < HWIDTH, other=0
        )
        tl.store(
            Hidden + parent * h_parent + slot * h_slot + offset, value, offset < HWIDTH
        )


def scatter_state_layers(
    cache: DeviceCache,
    positions: torch.Tensor,
    recurrent_layers: Sequence[torch.Tensor],
    convolution_layers: Sequence[torch.Tensor],
    hidden: torch.Tensor,
) -> None:
    """Store owned final states without stacking, with original null semantics.

    Engine leases must supply unique destinations for active/padded rows, as for
    the existing advanced-index scatter. This helper does not allocate or choose
    pages, retire requests, or resolve conflicting writes to the reserved null.
    """
    g = cache.geometry
    recurrent_layers, convolution_layers = (
        tuple(recurrent_layers),
        tuple(convolution_layers),
    )
    n, b = len(g.linear_layers), positions.numel()
    if (
        positions.ndim != 1
        or positions.dtype not in (torch.int32, torch.int64)
        or not positions.is_contiguous()
    ):
        raise ValueError("positions must be a contiguous integer vector")
    if len(recurrent_layers) != n or len(convolution_layers) != n:
        raise ValueError("one final state per recurrent layer is required")
    specs: list[tuple[torch.Tensor, tuple[int, ...], torch.dtype]] = [
        (x, (b, g.heads, g.head_dim, g.head_dim), torch.float32)
        for x in recurrent_layers
    ]
    specs += [
        (x, (b, 3, g.heads * g.head_dim, g.conv_history), torch.bfloat16)
        for x in convolution_layers
    ]
    specs += [(hidden, (b, g.hidden_size), torch.bfloat16)]
    arena = {
        x.untyped_storage().data_ptr()
        for x in (
            cache.recurrent_states,
            cache.convolution_states,
            cache.target_hidden_states,
        )
    }
    for value, shape, dtype in specs:
        if (
            tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != positions.device
            or not value.is_cuda
            or not value.is_contiguous()
        ):
            raise ValueError("final states must be contiguous CUDA layer tensors")
        if value.untyped_storage().data_ptr() in arena:
            raise ValueError("final state inputs must not alias Engine slabs")
    if not b:
        return
    group = cache.groups[1]
    if (
        group.slab_ids.shape != group.slotnums.shape
        or group.slab_ids.shape[0] != b
        or group.first_blocks.shape != (b,)
    ):
        raise ValueError("state tables must match the batch")
    if any(
        x.device != positions.device or not x.is_contiguous()
        for x in (group.slab_ids, group.slotnums, group.first_blocks)
    ):
        raise ValueError("state tables must be contiguous on the same device")
    r, c, h = (
        cache.recurrent_states,
        cache.convolution_states,
        cache.target_hidden_states,
    )
    rw, cw = g.heads * g.head_dim**2, 3 * g.heads * g.head_dim * g.conv_history
    block = 4096
    rb, cb = triton.cdiv(rw, block), triton.cdiv(cw, block)
    cast(Any, _scatter_state)[(rb + cb + triton.cdiv(g.hidden_size, block), b, n)](
        r,
        c,
        h,
        group.slab_ids,
        group.slotnums,
        group.first_blocks,
        positions,
        recurrent_layers,
        convolution_layers,
        hidden,
        b,
        group.slab_ids.shape[1],
        group.slab_ids.stride(0),
        r.stride(0),
        r.stride(1),
        c.stride(0),
        c.stride(1),
        h.stride(0),
        h.stride(1),
        g.state_tokens,
        rw,
        cw,
        g.hidden_size,
        rb,
        cb,
        n,
        block,
        LIVE_STATE=group.live_state,
        num_warps=4,
    )


@triton.jit(
    do_not_specialize=[
        "batch",
        "table_width",
        "table_stride",
        "key_parent",
        "key_slot",
        "gate_parent",
        "gate_slot",
        "end_parent",
        "end_slot",
        "output_end_stride",
    ]
)
def _gather_index_tails(
    Keys,
    Gates,
    Ends,
    Parents,
    Slots,
    First,
    Contexts,
    OutKeys,
    OutGates,
    OutEnds,
    batch,
    table_width,
    table_stride,
    key_parent,
    key_slot,
    gate_parent,
    gate_slot,
    end_parent,
    end_slot,
    output_end_stride,
    STATE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    LIVE_STATE: tl.constexpr,
):
    seq, layer = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    context = tl.load(Contexts + seq)
    column = (
        0 if LIVE_STATE else tl.maximum(context - 1, 0) // STATE - tl.load(First + seq)
    )
    valid = (column >= 0) & (column < table_width)
    parent = tl.load(Parents + seq * table_stride + column, valid, other=0).to(tl.int64)
    slot = tl.load(Slots + seq * table_stride + column, valid, other=0).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    live = (context > 0) & (parent > 0)
    key = tl.load(
        Keys + parent * key_parent + slot * key_slot + layer * WIDTH + offset,
        live & (offset < WIDTH),
        other=0,
    )
    gate = tl.load(
        Gates + parent * gate_parent + slot * gate_slot + layer * WIDTH + offset,
        live & (offset < WIDTH),
        other=0,
    )
    end = tl.load(Ends + parent * end_parent + slot * end_slot + layer, live, other=-1)
    output = (layer * batch + seq) * WIDTH + offset
    tl.store(OutKeys + output, key, offset < WIDTH)
    tl.store(OutGates + output, gate, offset < WIDTH)
    tl.store(OutEnds + layer * output_end_stride + seq, end)


def gather_index_tails(cache, contexts):
    from resin_models_cuda.glm5_3_flash.index_tail import IndexTail

    g = cache.geometry
    layers, batch = len(g.full_layers), contexts.numel()
    keys = torch.empty(
        (layers, batch, 4, g.index_dim), device=contexts.device, dtype=torch.bfloat16
    )
    gates = torch.empty_like(keys, dtype=torch.float32)
    # The scatter consumes a tuple of layer views. Triton 3.7.1 specializes
    # tuple element pointer alignment even with the argument's alignment flag
    # disabled. Keep every layer aligned at B1/odd B without copies or kernels.
    ends = torch.empty_strided(
        (layers, batch),
        (triton.cdiv(batch, 2) * 2, 1),
        device=contexts.device,
        dtype=torch.int64,
    )
    group = cache.groups[1]
    if batch:
        cast(Any, _gather_index_tails)[(batch, layers)](
            cache.index_tail_keys,
            cache.index_tail_gates,
            cache.index_tail_ends,
            group.slab_ids,
            group.slotnums,
            group.first_blocks,
            contexts,
            keys,
            gates,
            ends,
            batch,
            group.slab_ids.shape[1],
            group.slab_ids.stride(0),
            *cache.index_tail_keys.stride()[:2],
            *cache.index_tail_gates.stride()[:2],
            *cache.index_tail_ends.stride()[:2],
            ends.stride(0),
            g.state_tokens,
            4 * g.index_dim,
            triton.next_power_of_2(4 * g.index_dim),
            LIVE_STATE=group.live_state,
        )
    return [IndexTail(keys[i], gates[i], ends[i]) for i in range(layers)]


@triton.jit(
    do_not_specialize=[
        "table_width",
        "table_stride",
        "key_parent",
        "key_slot",
        "gate_parent",
        "gate_slot",
        "end_parent",
        "end_slot",
    ]
)
def _scatter_index_tails(
    Keys,
    Gates,
    Ends,
    Parents,
    Slots,
    First,
    Positions,
    InKeys,
    InGates,
    InEnds,
    table_width,
    table_stride,
    key_parent,
    key_slot,
    gate_parent,
    gate_slot,
    end_parent,
    end_slot,
    STATE: tl.constexpr,
    WIDTH: tl.constexpr,
    LAYERS: tl.constexpr,
    BLOCK: tl.constexpr,
    LIVE_STATE: tl.constexpr,
):
    seq, layer = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    position = tl.load(Positions + seq)
    logical = tl.where(
        position < 0, -((-position + STATE - 1) // STATE), position // STATE
    )
    column = 0 if LIVE_STATE else logical - tl.load(First + seq)
    valid = (column >= 0) & (column < table_width)
    if LIVE_STATE:
        valid = valid & (position >= 0)
    parent = tl.load(Parents + seq * table_stride + column, valid, other=0).to(tl.int64)
    slot = tl.load(Slots + seq * table_stride + column, valid, other=0).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    for i in tl.static_range(LAYERS):
        if layer == i:
            key = tl.load(InKeys[i] + seq * WIDTH + offset, offset < WIDTH, other=0)
            gate = tl.load(InGates[i] + seq * WIDTH + offset, offset < WIDTH, other=0)
            end = tl.load(InEnds[i] + seq)
            tl.store(
                Keys + parent * key_parent + slot * key_slot + layer * WIDTH + offset,
                key,
                (offset < WIDTH) & (parent > 0),
            )
            tl.store(
                Gates
                + parent * gate_parent
                + slot * gate_slot
                + layer * WIDTH
                + offset,
                gate,
                (offset < WIDTH) & (parent > 0),
            )
            tl.store(
                Ends + parent * end_parent + slot * end_slot + layer, end, parent > 0
            )


def scatter_index_tails(cache, positions, tails):
    g, group = cache.geometry, cache.groups[1]
    if positions.numel():
        cast(Any, _scatter_index_tails)[(positions.numel(), len(tails))](
            cache.index_tail_keys,
            cache.index_tail_gates,
            cache.index_tail_ends,
            group.slab_ids,
            group.slotnums,
            group.first_blocks,
            positions,
            tuple(t.keys for t in tails),
            tuple(t.gates for t in tails),
            tuple(t.end for t in tails),
            group.slab_ids.shape[1],
            group.slab_ids.stride(0),
            *cache.index_tail_keys.stride()[:2],
            *cache.index_tail_gates.stride()[:2],
            *cache.index_tail_ends.stride()[:2],
            g.state_tokens,
            4 * g.index_dim,
            len(tails),
            triton.next_power_of_2(4 * g.index_dim),
            LIVE_STATE=group.live_state,
        )
