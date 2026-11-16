"""Compact routed NVFP4 GEMM adapter for SGLang's pinned CuTe kernel."""

from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from resin_models_cuda.inkling2.cute_grouped_nvfp4 import routed_swap_matmul

jit: Any = triton.jit


def prepare_backend(module):
    """Replace native block scales with the equally sized MMA layout once."""
    if getattr(module, "_cute_prepared", False):
        return
    for name, precision in module.precision.items():
        if precision != "nvfp4":
            continue
        scale = getattr(module, name + "_weight_scale")
        experts, n, groups = scale.shape
        if n % 128 or groups % 4:
            raise ValueError(
                "Inkling CuTe NVFP4 requires N divisible by 128 and K by 64"
            )
        swizzled = (
            scale.view(experts, n // 128, 4, 32, groups // 4, 4)
            .permute(0, 1, 4, 3, 2, 5)
            .contiguous()
            .view(experts, -1)
        )
        setattr(
            module,
            name + "_weight_scale",
            torch.nn.Parameter(swizzled, requires_grad=False),
        )
    module._cute_prepared = True


@jit
def _expert(row, Offsets, E: tl.constexpr):
    lo, hi = tl.full((), 0, tl.int32), E
    while lo < hi:
        mid = (lo + hi) // 2
        right = row >= tl.load(Offsets + mid + 1)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    return tl.minimum(lo, E - 1)


@jit(do_not_specialize=["slots"])
def _schedule(Blocks, Schedule, slots, E: tl.constexpr, BLOCK: tl.constexpr):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo, hi = tl.full((BLOCK,), 0, tl.int32), tl.full((BLOCK,), E, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(Blocks + mid + 1, mid < E, other=2147483647)
        right = end <= slot
        lo = tl.where((lo < hi) & right, mid + 1, lo)
        hi = tl.where((lo < hi) & ~right, mid, hi)
    expert = tl.minimum(lo, E - 1)
    entry = expert | ((slot - tl.load(Blocks + expert)) << 16)
    tl.store(
        Schedule + slot, tl.where(slot < tl.load(Blocks + E), entry, -1), slot < slots
    )


@jit
def _pack_routed(
    X,
    XS,
    Permutation,
    Offsets,
    Values,
    Scales,
    K: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start, end = tl.load(Offsets), tl.load(Offsets + E)
    valid = (row >= start) & (row < end)
    expert = _expert(row, Offsets, E)
    base = tl.load(Offsets + expert)
    local = row - base
    source = row
    if GATHER:
        source = tl.load(Permutation + row).to(tl.int64) // TOP_K
    packed = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + source * (K // 2) + packed, valid & (packed < K // 2), 0)
    tl.store(Values + row * (K // 2) + packed, value, packed < K // 2)
    groups = tl.program_id(1) * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
    scale = tl.load(XS + source * (K // 16) + groups, valid & (groups < K // 16), 0.0)
    scale_base = ((base - start + expert * 127) // 128 * 128) * (K // 16)
    address = (
        scale_base
        + ((local // 128) * (K // 64) + groups // 4) * 512
        + (local % 32) * 16
        + ((local // 32) % 4) * 4
        + groups % 4
    )
    tl.store(Scales + address, scale, valid & (groups < K // 16))


@jit
def _scale_result(
    Acc,
    XA,
    WA,
    WD,
    Permutation,
    Offsets,
    Out,
    N: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row >= tl.load(Offsets)) & (row < tl.load(Offsets + E))
    expert = _expert(row, Offsets, E)
    source = row
    if GATHER:
        source = tl.load(Permutation + row).to(tl.int64) // TOP_K
    scale = (tl.load(XA + source) * tl.load(WA + expert)) * libdevice.rcp_rn(
        1536.0 * tl.load(WD)
    )
    acc = tl.load(Acc + row * N + cols, valid & (cols < N), 0)
    tl.store(Out + row * N + cols, tl.where(valid, acc * scale, 0), cols < N)


def matmul(
    module,
    name,
    values,
    scales,
    amax,
    permutation,
    offsets,
    counts,
    blocks,
    top_k,
    gather,
    dtype,
):
    experts, rows = module.expert_count, permutation.numel()
    weight = getattr(module, name + "_weight")[:experts]
    width, output_width = weight.shape[-1] * 2, weight.shape[1]
    routed = values.new_empty((rows, width // 2))
    scale_rows = triton.cdiv(rows + 127 * experts, 128) * 128
    routed_scales = scales.new_empty((scale_rows, width // 16))
    _pack_routed[(rows, triton.cdiv(width // 2, 256))](
        values,
        scales,
        permutation,
        offsets,
        routed,
        routed_scales,
        width,
        experts,
        top_k,
        gather,
        256,
    )
    slots = triton.cdiv(rows, 128) + experts - 1
    schedule = offsets.new_empty(slots)
    _schedule[(triton.cdiv(slots, 128),)](blocks, schedule, slots, experts, 128)
    accumulator = torch.empty(
        (rows, output_width), dtype=torch.float32, device=values.device
    )
    routed_swap_matmul(
        routed,
        routed_scales,
        weight,
        getattr(module, name + "_weight_scale")[:experts],
        counts,
        offsets,
        schedule,
        accumulator,
    )
    out = torch.empty_like(accumulator, dtype=dtype)
    _scale_result[(rows, triton.cdiv(output_width, 256))](
        accumulator,
        amax,
        getattr(module, name + "_weight_amax"),
        getattr(module, name + "_weight_outer_scale_denominator"),
        permutation,
        offsets,
        out,
        output_width,
        experts,
        top_k,
        gather,
        256,
        enable_fp_fusion=False,
    )
    return out
