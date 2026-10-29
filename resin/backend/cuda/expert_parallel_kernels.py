"""Capturable tensor preparation around the shared expert transport.

Collectives, expert execution, and their lifetime remain in expert_parallel.py.
Rows and source strides are runtime values; topology and feature widths are fixed.
"""

from collections.abc import Iterator
from typing import Any, cast

import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["rows", "h0", "h1", "i0", "i1", "w0", "w1"])
def _prepare(
    H,
    Indices,
    W,
    Valid,
    PH,
    PI,
    PW,
    Owners,
    Active,
    rows,
    h0,
    h1,
    i0,
    i1,
    w0,
    w1,
    HIDDEN: tl.constexpr,
    TOPK: tl.constexpr,
    TP: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    SOURCE: tl.constexpr,
    HAS_VALID: tl.constexpr,
    META: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    column = block * BLOCK + tl.arange(0, BLOCK)
    live = (row < rows) & SOURCE
    value = tl.load(
        H + row * h0 + column.to(tl.int64) * h1, live & (column < HIDDEN), other=0
    )
    tl.store(PH + row * HIDDEN + column, value, column < HIDDEN)
    if block == 0:
        slot = tl.arange(0, META)
        index = tl.load(
            Indices + row * i0 + slot.to(tl.int64) * i1, live & (slot < TOPK), other=0
        )
        weight = tl.load(
            W + row * w0 + slot.to(tl.int64) * w1, live & (slot < TOPK), other=0
        )
        tl.store(PI + row * TOPK + slot, index, slot < TOPK)
        tl.store(PW + row * TOPK + slot, weight, slot < TOPK)
        expert_slot = slot // TP
        expert = tl.load(
            Indices + row * i0 + expert_slot.to(tl.int64) * i1,
            live & (slot < TOPK * TP),
            other=0,
        ).to(tl.int64)
        owner = expert // EXPERTS_PER_RANK * TP + slot % TP
        tl.store(Owners + row * TOPK * TP + slot, owner, slot < TOPK * TP)
        count = tl.load(Valid) if HAS_VALID else rows
        tl.store(Active + row, (row < count) & SOURCE)


def prepare_dispatch(
    hidden: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    capacity: int,
    *,
    source: bool,
    valid_rows: torch.Tensor | None,
    expert_tp: int,
    experts_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, width = hidden.shape
    topk = indices.shape[1]
    padded_hidden = hidden.new_empty((capacity, width))
    padded_indices = indices.new_empty((capacity, topk))
    padded_weights = weights.new_empty((capacity, topk))
    # Torch's old owner expression promotes integer IDs with arange(int64).
    owners = torch.empty(
        (capacity, topk * expert_tp), device=hidden.device, dtype=torch.int64
    )
    active = torch.empty(capacity, device=hidden.device, dtype=torch.bool)
    with torch.cuda.device(hidden.device):
        cast(Any, _prepare)[(capacity, tr.cdiv(width, 256))](
            hidden,
            indices,
            weights,
            hidden if valid_rows is None else valid_rows,
            padded_hidden,
            padded_indices,
            padded_weights,
            owners,
            active,
            rows,
            *hidden.stride(),
            *indices.stride(),
            *weights.stride(),
            HIDDEN=width,
            TOPK=topk,
            TP=expert_tp,
            EXPERTS_PER_RANK=experts_per_rank,
            SOURCE=source,
            HAS_VALID=valid_rows is not None,
            META=tr.next_power_of_2(topk * expert_tp),
            BLOCK=256,
        )
    return padded_hidden, padded_indices, padded_weights, owners, active


@tr.jit
def _or(left, right):
    return left | right


@tr.jit(do_not_specialize=["o0", "o1", "a0", "i0", "i1", "w0", "w1", "START", "RANK"])
def _routes(
    Owners,
    Active,
    Indices,
    Weights,
    Route,
    PayloadIds,
    PayloadWeights,
    o0,
    o1,
    a0,
    i0,
    i1,
    w0,
    w1,
    OWNER_COUNT: tl.constexpr,
    OWNER_BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    META: tl.constexpr,
    WORLD: tl.constexpr,
    RANK,
    EXPERTS: tl.constexpr,
    DEDUP: tl.constexpr,
    START,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, OWNER_BLOCK)
    owners = tl.load(
        Owners + row * o0 + lane.to(tl.int64) * o1, lane < OWNER_COUNT, other=2147483647
    ).to(tl.int64)
    slots = tl.arange(0, ROUTE_BLOCK)
    if DEDUP:
        # OR preserves duplicate destinations, unlike a sum of bit flags.
        bit = tl.where(
            (lane < OWNER_COUNT) & (owners >= 0) & (owners < WORLD),
            tl.full((), 1, tl.uint64) << owners.to(tl.uint64),
            0,
        )
        destinations = tl.reduce(bit, 0, _or)
        ranks = START + slots
        present = ((destinations >> ranks.to(tl.uint64)) & 1) != 0
        route = tl.where((slots < WIDTH) & present, ranks, WORLD)
    else:
        ordered = tl.sort(owners, descending=False)
        route = tl.gather(ordered, tl.minimum(slots, OWNER_BLOCK - 1), 0)
        route = tl.where(slots < WIDTH, route, 2147483647)
    first = tl.min(route, 0)
    selected = tl.load(Active + row * a0) & (first < WORLD)
    route = tl.where((slots < WIDTH) & (route < WORLD), route, first)
    route = tl.where(selected, route, RANK)
    tl.store(Route + row * K + slots, route, slots < K)
    expert_slot = tl.arange(0, META)
    index = tl.load(
        Indices + row * i0 + expert_slot.to(tl.int64) * i1, expert_slot < TOPK, other=0
    )
    weight = tl.load(
        Weights + row * w0 + expert_slot.to(tl.int64) * w1, expert_slot < TOPK, other=0
    )
    tl.store(
        PayloadIds + row * TOPK + expert_slot,
        tl.where(selected, index, EXPERTS),
        expert_slot < TOPK,
    )
    tl.store(
        PayloadWeights + row * TOPK + expert_slot,
        tl.where(selected, weight, 0).to(tl.float32),
        expert_slot < TOPK,
    )


def prepare_routes(
    owners: torch.Tensor,
    active: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    world_size: int,
    rank: int,
    num_experts: int,
    max_destinations_per_pass: int = 22,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    capacity, owner_count = owners.shape
    topk = indices.shape[1]
    dedup = owner_count > max_destinations_per_pass
    widths = (
        [
            (start, min(max_destinations_per_pass, world_size - start))
            for start in range(0, world_size, max_destinations_per_pass)
        ]
        if dedup
        else [(0, owner_count)]
    )
    for start, width in widths:
        k = next(k for k in (1, 2, 4, 6, 8, 10, 16, 22) if k >= width)
        route = torch.empty((capacity, k), dtype=torch.int32, device=indices.device)
        payload_ids = torch.empty(
            (capacity, topk), dtype=torch.int32, device=indices.device
        )
        payload_weights = torch.empty(
            (capacity, topk), dtype=torch.float32, device=weights.device
        )
        with torch.cuda.device(indices.device):
            cast(Any, _routes)[(capacity,)](
                owners,
                active,
                indices,
                weights,
                route,
                payload_ids,
                payload_weights,
                *owners.stride(),
                active.stride(0),
                *indices.stride(),
                *weights.stride(),
                OWNER_COUNT=owner_count,
                OWNER_BLOCK=tr.next_power_of_2(owner_count),
                TOPK=topk,
                META=tr.next_power_of_2(topk),
                WORLD=world_size,
                RANK=rank,
                EXPERTS=num_experts,
                DEDUP=dedup,
                START=start,
                WIDTH=width,
                K=k,
                ROUTE_BLOCK=tr.next_power_of_2(k),
            )
        yield route, payload_ids, payload_weights


@tr.jit(do_not_specialize=["capacity", "h0", "h1", "h2", "i0", "i1", "w0", "w1", "w2"])
def _receive(
    Hidden,
    Indices,
    Weights,
    CleanHidden,
    CleanWeights,
    capacity,
    h0,
    h1,
    h2,
    i0,
    i1,
    w0,
    w1,
    w2,
    HIDDEN: tl.constexpr,
    TOPK: tl.constexpr,
    EXPERTS: tl.constexpr,
    META: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    source, token = row // capacity, row % capacity
    valid = tl.load(Indices + source * i0 + token * i1) < EXPERTS
    block = tl.program_id(1)
    column = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        Hidden + source * h0 + token * h1 + column.to(tl.int64) * h2,
        valid & (column < HIDDEN),
        other=0,
    )
    tl.store(CleanHidden + row * HIDDEN + column, value, column < HIDDEN)
    if block == 0:
        slot = tl.arange(0, META)
        weight = tl.load(
            Weights + source * w0 + token * w1 + slot.to(tl.int64) * w2,
            valid & (slot < TOPK),
            other=0,
        )
        tl.store(CleanWeights + row * TOPK + slot, weight, slot < TOPK)


def prepare_receive(
    hidden: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_experts: int,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    sources, capacity, width = hidden.shape
    topk = indices.shape[-1]
    clean_hidden = hidden.new_empty((sources * capacity, width))
    clean_weights = torch.empty(
        (sources * capacity, topk), device=weights.device, dtype=weight_dtype
    )
    with torch.cuda.device(hidden.device):
        cast(Any, _receive)[(sources * capacity, tr.cdiv(width, 256))](
            hidden,
            indices,
            weights,
            clean_hidden,
            clean_weights,
            capacity,
            *hidden.stride(),
            *indices.stride()[:2],
            *weights.stride(),
            HIDDEN=width,
            TOPK=topk,
            EXPERTS=num_experts,
            META=tr.next_power_of_2(topk),
            BLOCK=256,
        )
    return clean_hidden, clean_weights
