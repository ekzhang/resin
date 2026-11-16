# Copyright 2026 SGLang contributors. Licensed under Apache-2.0.
# Adapted from sgl-project/sglang e62ac6279646c686d3b328f58d9b5c2c2d5927dc,
# python/sglang/kernels/ops/inkling2/{mxfp8,moe}.py.
"""SGLang CuTe NVFP4 and Triton MXFP8/BF16 packed expert GEMMs."""

import functools
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from resin_models_cuda.inkling2.expert_quantization import _capped_swiglu, quantize

# GPU-only Triton is absent from the development type-checking environment.
jit: Any = triton.jit


@jit
def _matmul(
    X,
    XS,
    XA,
    W,
    WS,
    WA,
    WD,
    Y,
    Permutation,
    Offsets,
    Blocks,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    PRECISION: tl.constexpr,
    TOP_K: tl.constexpr,
    GATHER: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TMA: tl.constexpr,
):
    tile = tl.program_id(0)
    if tile >= tl.load(Blocks + E):
        return
    # Block offsets compact variable expert counts into a fixed launch bound.
    lo, hi = tl.full((), 0, tl.int32), E
    while lo < hi:
        mid = (lo + hi) // 2
        end = tl.load(Blocks + mid + 1)
        lo = tl.where(end <= tile, mid + 1, lo)
        hi = tl.where(end <= tile, hi, mid)
    expert = lo.to(tl.int64)
    rows = (tile - tl.load(Blocks + expert)) * BM + tl.arange(0, BM)
    row_ids = tl.load(Offsets + expert).to(tl.int64) + rows
    valid = row_ids < tl.load(Offsets + expert + 1)
    source = row_ids
    if GATHER:
        source = tl.load(Permutation + row_ids, valid, 0).to(tl.int64) // TOP_K
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    # Weight rows occupy the M axis: a sparsely selected expert needs only
    # 16/32 padded token columns, rather than 128 padded token rows.
    acc = tl.zeros((BN, BM), tl.float32)
    if PRECISION == "bf16":
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            w = tl.load(
                W + (expert * N + cols[:, None]) * K + ks[None, :],
                (cols[:, None] < N) & (ks[None, :] < K),
                0.0,
            )
            x = tl.load(
                X + source[:, None] * K + ks[None, :],
                valid[:, None] & (ks[None, :] < K),
                0.0,
            )
            acc = tl.dot(w, tl.trans(x), acc)
    else:
        PACK: tl.constexpr = 2 if PRECISION == "nvfp4" else 1  # ty: ignore[invalid-assignment]
        GROUP: tl.constexpr = 16 if PRECISION == "nvfp4" else 32  # ty: ignore[invalid-assignment]
        FORMAT: tl.constexpr = "e2m1" if PRECISION == "nvfp4" else "e4m3"  # ty: ignore[invalid-assignment]
        for k0 in range(0, K, BK):
            ks = k0 // PACK + tl.arange(0, BK // PACK)
            ss = k0 // GROUP + tl.arange(0, BK // GROUP)
            if TMA:
                w = W.load([(expert * N + tl.program_id(1) * BN).to(tl.int32), k0])
            else:
                w = tl.load(
                    W + (expert * N + cols[:, None]) * (K // PACK) + ks[None, :],
                    (cols[:, None] < N) & (ks[None, :] < K // PACK),
                    0.0,
                )
            x = tl.load(
                X + source[:, None] * (K // PACK) + ks[None, :],
                valid[:, None] & (ks[None, :] < K // PACK),
                0.0,
            )
            ws = tl.load(
                WS + (expert * N + cols[:, None]) * (K // GROUP) + ss[None, :],
                (cols[:, None] < N) & (ss[None, :] < K // GROUP),
                0.0,
            )
            xs = tl.load(
                XS + source[:, None] * (K // GROUP) + ss[None, :],
                valid[:, None] & (ss[None, :] < K // GROUP),
                0.0,
            )
            acc = tl.dot_scaled(
                w, ws, FORMAT, tl.trans(x), xs, FORMAT, acc, fast_math=True
            )
        if PRECISION == "nvfp4":
            scale = (
                tl.load(XA + source, valid, 0) * tl.load(WA + expert)
            ) * libdevice.rcp_rn(1536.0 * tl.load(WD))
            acc *= scale[None, :]
    tl.store(
        Y + row_ids[None, :] * N + cols[:, None],
        acc,
        valid[None, :] & (cols[:, None] < N),
    )


@jit
def _activate(
    X,
    Weights,
    Permutation,
    Offsets,
    Out,
    INTERMEDIATE: tl.constexpr,
    E: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = (row >= tl.load(Offsets)) & (row < tl.load(Offsets + E))
    gate = tl.load(
        X + row * (2 * INTERMEDIATE) + 2 * col, inside & (col < INTERMEDIATE), 0
    ).to(tl.float32)
    up = tl.load(
        X + row * (2 * INTERMEDIATE) + 2 * col + 1, inside & (col < INTERMEDIATE), 0
    ).to(tl.float32)
    source = tl.load(Permutation + row).to(tl.int64)
    gamma = tl.load(Weights + source)
    value = _capped_swiglu(gate, up, gamma, GATE_CAP, UP_CAP)
    tl.store(Out + row * INTERMEDIATE + col, value, col < INTERMEDIATE)


@jit(do_not_specialize=["rows"])
def _inverse(Permutation, Inverse, rows, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(Permutation + row, row < rows, 0).to(tl.int64)
    tl.store(Inverse + source, row, row < rows)


@jit(do_not_specialize=["routes"], do_not_specialize_on_alignment=["Ids"])
def _small_routing(
    Ids,
    Permutation,
    Inverse,
    Offsets,
    Counts,
    Blocks,
    CuteBlocks,
    routes,
    START: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    CUTE: tl.constexpr,
    BINS: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    # Fixed capacity avoids specializing on the live verification row count.
    # Including the original route index makes the expert sort stable.
    r = tl.arange(0, CAPACITY)
    ids = tl.load(Ids + r, r < routes, 0).to(tl.int32)
    keys = tl.where(r < routes, ids * CAPACITY + r, 0x7FFFFFFF)
    permutation = tl.sort(keys, descending=False) & (CAPACITY - 1)
    tl.store(Permutation + r, permutation, r < routes)
    tl.store(Inverse + permutation, r, r < routes)
    local = ids - START
    counts = tl.histogram(local, BINS, mask=(r < routes) & (local >= 0) & (local < E))
    below = tl.sum(((r < routes) & (ids < START)).to(tl.int32), 0)
    e = tl.arange(0, BINS)
    tl.store(Counts + e, counts, e < E)
    tl.store(Offsets, below)
    tl.store(Offsets + e + 1, below + tl.cumsum(counts), e < E)
    tl.store(Blocks, 0)
    tl.store(Blocks + e + 1, tl.cumsum(tl.cdiv(counts, BM)), e < E)
    if CUTE:
        tl.store(CuteBlocks, 0)
        tl.store(CuteBlocks + e + 1, tl.cumsum(tl.cdiv(counts, 128)), e < E)


def _small_routing_metadata(ids, start, experts, block_m, cute):
    """Stable routing layout in fixed capacities through 2,048 routes."""
    routes = ids.numel()
    if routes > 2048:
        raise ValueError("fused routing supports at most 2048 routes")
    capacity = next(bound for bound in (128, 512, 1024, 2048) if routes <= bound)
    ids = ids.contiguous()
    permutation = torch.empty(routes, device=ids.device, dtype=torch.int64)
    inverse = torch.empty(routes, device=ids.device, dtype=torch.int32)
    offsets = torch.empty(experts + 1, device=ids.device, dtype=torch.int32)
    counts = torch.empty(experts, device=ids.device, dtype=torch.int32)
    blocks = torch.empty_like(offsets)
    cute_blocks = torch.empty_like(offsets) if cute else None
    _small_routing[(1,)](
        ids,
        permutation,
        inverse,
        offsets,
        counts,
        blocks,
        cute_blocks,
        routes,
        start,
        experts,
        block_m,
        cute,
        triton.next_power_of_2(experts),
        capacity,
        num_warps=4 if capacity <= 512 else 16,
    )
    return permutation, inverse, offsets, counts, blocks, cute_blocks


@functools.cache
def warm_routing(start, experts, device):
    """Prepare fixed decode capacities, independent of live request lengths."""
    # Direct router IDs are int64; sparse expert transport uses int32.
    for dtype in (torch.int32, torch.int64):
        for routes in (128, 512, 1024, 2048):
            ids = torch.zeros(routes, dtype=dtype, device=device)
            for block_m in (16, 32):
                for cute in (False, True):
                    _small_routing_metadata(ids, start, experts, block_m, cute)
    # The larger-prefill path still needs its independent inverse kernel.
    _routing_metadata(ids, start, experts, 32, True)


def _routing_metadata(ids, start, experts, block_m, cute):
    sorted_ids, permutation = torch.sort(ids.flatten(), stable=True)
    boundaries = torch.arange(
        start,
        start + experts + 1,
        device=ids.device,
        dtype=sorted_ids.dtype,
    )
    offsets = torch.searchsorted(sorted_ids, boundaries).to(torch.int32)
    counts = offsets[1:] - offsets[:-1]
    blocks = torch.cat(
        (
            offsets.new_zeros(1),
            ((counts + block_m - 1) // block_m).cumsum(0, dtype=torch.int32),
        )
    )
    cute_blocks = (
        torch.cat(
            (offsets.new_zeros(1), ((counts + 127) // 128).cumsum(0, dtype=torch.int32))
        )
        if cute
        else None
    )
    inverse = torch.empty_like(permutation, dtype=torch.int32)
    _inverse[(triton.cdiv(ids.numel(), 256),)](permutation, inverse, ids.numel(), 256)
    return permutation, inverse, offsets, counts, blocks, cute_blocks


@jit
def _combine(
    X,
    Inverse,
    Offsets,
    Y,
    H: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    lo, hi = tl.load(Offsets), tl.load(Offsets + E)
    total = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(TOP_K):
        row = tl.load(Inverse + token * TOP_K + k).to(tl.int64)
        value = tl.load(X + row * H + cols, (row >= lo) & (row < hi) & (cols < H), 0)
        total += value.to(tl.float32)
    tl.store(Y + token * H + cols, total, cols < H)


def local_experts(module, hidden, ids, weights, *, max_rows_per_expert=None):
    """A rank's FP32 partial sum, preserving each route's BF16 rounding."""
    rows, top_k = ids.shape
    expert_tp = module.expert_tp_size > 1
    if not rows:
        return hidden.new_empty(hidden.shape, dtype=torch.float32)
    if module.expert_backend == "cute-dsl":
        from resin_models_cuda.inkling2 import expert_native

        if expert_native.supports(module, hidden, ids, weights):
            return expert_native.local_experts(
                module, hidden, ids, weights, max_rows_per_expert=max_rows_per_expert
            )
    block_m = 16 if rows <= 16 else 32
    prepare_routes = (
        _small_routing_metadata if ids.numel() <= 2048 else _routing_metadata
    )
    permutation, inverse, offsets, counts, blocks, cute_blocks = prepare_routes(
        ids,
        module.expert_start,
        module.expert_count,
        block_m,
        module.expert_backend == "cute-dsl" and "nvfp4" in module.precision.values(),
    )
    max_tiles = min(
        ids.numel(), triton.cdiv(ids.numel(), block_m) + module.expert_count - 1
    )

    def matmul(name, source, gather):
        precision = module.precision[name]
        values, scales, amax = quantize(
            source,
            precision,
            reduce_amax=module.reduce_activation_amax
            if name == "w2" and expert_tp
            else None,
        )
        partial = name == "w2" and expert_tp
        dtype = torch.float32 if partial else hidden.dtype
        if precision == "nvfp4" and module.expert_backend == "cute-dsl":
            from resin_models_cuda.inkling2.expert_cute import matmul as cute_matmul

            return cute_matmul(
                module,
                name,
                values,
                scales,
                amax,
                permutation,
                offsets,
                counts,
                cute_blocks,
                top_k,
                gather,
                dtype,
            )
        weight = getattr(module, name + "_weight")
        output_width, width = weight.shape[1], source.shape[1]
        # Only ETP reads complete rows in a collective; EP masks invalid rows.
        allocate = torch.zeros if partial else torch.empty
        out = allocate((ids.numel(), output_width), device=hidden.device, dtype=dtype)
        # Width is model geometry. Ordinary row lengths affect launch bounds only.
        block_k = next(b for b in (256, 128, 64, 32) if width % b == 0)
        tma = (
            precision == "mxfp8" and output_width % 128 == 0 and weight.is_contiguous()
        )
        if tma:
            from triton.tools.tensor_descriptor import TensorDescriptor

            weight = TensorDescriptor.from_tensor(
                weight.view(module.expert_count * output_width, width), [128, block_k]
            )
        _matmul[(max_tiles, triton.cdiv(output_width, 128))](
            values,
            scales,
            amax,
            weight,
            getattr(module, name + "_weight_scale", None),
            getattr(module, name + "_weight_amax", None),
            getattr(module, name + "_weight_outer_scale_denominator", None),
            out,
            permutation,
            offsets,
            blocks,
            output_width,
            width,
            module.expert_count,
            precision,
            top_k,
            gather,
            block_m,
            128,
            block_k,
            tma,
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
        return out

    gate_up = matmul("w13", hidden, True)
    activated = hidden.new_empty((ids.numel(), module.intermediate))
    _activate[(ids.numel(), triton.cdiv(module.intermediate, 256))](
        gate_up,
        weights,
        permutation,
        offsets,
        activated,
        module.intermediate,
        module.expert_count,
        module.gate_cap,
        module.up_cap,
        256,
        enable_fp_fusion=False,
    )
    down = matmul("w2", activated, False)
    if expert_tp:
        down = module.reduce_down_partials(down)
    out = hidden.new_empty(hidden.shape, dtype=torch.float32)
    _combine[(rows, triton.cdiv(hidden.shape[1], 512))](
        down,
        inverse,
        offsets,
        out,
        hidden.shape[1],
        module.expert_count,
        top_k,
        512,
    )
    return out
