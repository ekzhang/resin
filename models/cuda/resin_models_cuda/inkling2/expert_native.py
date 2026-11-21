# Copyright 2026 SGLang contributors. Licensed under Apache-2.0.
# Adapted from sgl-project/sglang e62ac6279646c686d3b328f58d9b5c2c2d5927dc,
# python/sglang/kernels/ops/inkling2/nvfp4.py.
"""Fused routed NVFP4 quantization for the optional CuTe EP backend.

Only fixed model widths enter compilation. Each captured call owns its buffers;
request row counts, routes, and positions remain runtime data.
"""

import functools
import hashlib
from pathlib import Path
from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda, libdevice


@functools.cache
def native():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm native Inkling2 experts before CUDA graph capture")
    import flashinfer
    from flashinfer.jit import gen_jit_spec

    source = Path(__file__).with_name("expert_native.cu")
    digest = hashlib.sha256(
        source.read_bytes() + flashinfer.__version__.encode()
    ).hexdigest()[:16]
    return gen_jit_spec(
        "resin_inkling2_native_quant_" + digest,
        [source],
        extra_cuda_cflags=[
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "--ftz=false",
        ],
    ).build_and_load()


def supports(module, hidden, ids, weights):
    return (
        module.expert_tp_size == 1
        and module.expert_count <= 1024  # kMaxExperts in expert_native.cu
        and set(module.precision.values()) == {"nvfp4"}
        and hidden.shape[1] in (128, 256, 2304, 4608)
        and module.intermediate in (128, 256, 1152, 2304, 4608)
        and hidden.dtype == torch.bfloat16
        and weights.dtype == torch.float32
        and hidden.is_contiguous()
        and ids.is_contiguous()
        and hidden.data_ptr() % 16 == 0
        and weights.is_contiguous()
    )


@triton.jit
def _combine_scaled(
    Acc,
    Amax,
    WeightAmax,
    Denominator,
    Ids,
    Inverse,
    Out,
    START: tl.constexpr,
    E: tl.constexpr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    if PDL:
        cuda.gdc_wait()
        cuda.gdc_launch_dependents()
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    denominator = libdevice.rcp_rn(1536.0 * tl.load(Denominator))
    total = tl.full((BLOCK,), 0, tl.float32)
    for route in tl.static_range(TOP_K):
        expert = tl.load(Ids + token * TOP_K + route).to(tl.int32) - START
        valid = (expert >= 0) & (expert < E)
        row = tl.load(Inverse + token * TOP_K + route).to(tl.int64)
        amax = tl.load(Amax + row, valid, 0)
        weight_amax = tl.load(WeightAmax + expert, valid, 0)
        scale = (amax * weight_amax) * denominator
        accumulator = tl.load(Acc + row * H + col, valid & (col < H), 0)
        # Preserve each route's BF16 boundary and the original top-k sum order.
        rounded = (accumulator * scale).to(tl.bfloat16).to(tl.float32)
        total = total + rounded
    tl.store(Out + token * H + col, total, col < H)


def _token_tiles(route_count, max_rows_per_expert):
    # This is a caller-proven bound, never a value inferred from GPU IDs.
    # SparseMlp supplies it after unique top-k; arbitrary/dispatch callers omit it.
    if max_rows_per_expert is not None and max_rows_per_expert < 0:
        raise ValueError("expert row bound must be nonnegative")
    if max_rows_per_expert is not None and max_rows_per_expert <= 32:
        return 32, 32
    return 128, 64 if route_count <= 64 else 128


def local_experts(module, hidden, ids, weights, *, pdl=True, max_rows_per_expert=None):
    from resin_models_cuda.inkling2 import expert_cute, expert_kernels
    from resin_models_cuda.inkling2.cute_grouped_nvfp4 import routed_swap_matmul

    rows, top_k = ids.shape
    if not rows:
        return hidden.new_empty(hidden.shape, dtype=torch.float32)
    experts = module.expert_count
    w13_tile, w2_tile = _token_tiles(ids.numel(), max_rows_per_expert)
    sms = torch.cuda.get_device_properties(hidden.device).multi_processor_count
    prepare = (
        expert_kernels._small_routing_metadata
        if ids.numel() <= 2048
        else expert_kernels._routing_metadata
    )
    permutation, inverse, offsets, counts, _, blocks = prepare(
        ids, module.expert_start, experts, 16 if rows <= 16 else 32, True
    )
    slots = triton.cdiv(ids.numel(), 128) + experts - 1
    schedule = offsets.new_empty(slots)
    expert_cute._schedule[(triton.cdiv(slots, 128),)](
        blocks, schedule, slots, experts, 128
    )
    scale_rows = triton.cdiv(ids.numel() + 127 * experts, 128) * 128

    def buffers(width):
        return (
            torch.empty(
                (ids.numel(), width // 2), device=hidden.device, dtype=torch.uint8
            ),
            torch.empty(
                (scale_rows, width // 16), device=hidden.device, dtype=torch.uint8
            ),
            torch.empty(ids.numel(), device=hidden.device, dtype=torch.float32),
        )

    values, scales, amax = buffers(hidden.shape[1])
    quant_grid = 4 * sms if ids.numel() >= 4 * sms else sms
    native().quantize(
        hidden, permutation, values, scales, amax, offsets, top_k, quant_grid, pdl
    )
    accumulator = torch.empty(
        (ids.numel(), 2 * module.intermediate),
        device=hidden.device,
        dtype=torch.float32,
    )
    routed_swap_matmul(
        values,
        scales,
        module.w13_weight[:experts],
        module.w13_weight_scale[:experts],
        counts,
        offsets,
        schedule,
        accumulator,
        prefetch_weights=True,
        token_tile=w13_tile,
        max_rows_per_expert=max_rows_per_expert,
    )
    values2, scales2, amax2 = buffers(module.intermediate)
    native().bridge(
        accumulator,
        amax,
        module.w13_weight_amax,
        module.w13_weight_outer_scale_denominator,
        weights,
        permutation,
        values2,
        scales2,
        amax2,
        offsets,
        module.gate_cap,
        module.up_cap,
        quant_grid,
        pdl,
    )
    accumulator2 = torch.empty(
        (ids.numel(), hidden.shape[1]), device=hidden.device, dtype=torch.float32
    )
    routed_swap_matmul(
        values2,
        scales2,
        module.w2_weight[:experts],
        module.w2_weight_scale[:experts],
        counts,
        offsets,
        schedule,
        accumulator2,
        prefetch_weights=True,
        token_tile=w2_tile,
        max_rows_per_expert=max_rows_per_expert,
    )
    out = torch.empty_like(hidden, dtype=torch.float32)
    cast(Any, _combine_scaled)[(rows, triton.cdiv(hidden.shape[1], 256))](
        accumulator2,
        amax2,
        module.w2_weight_amax,
        module.w2_weight_outer_scale_denominator,
        ids,
        inverse,
        out,
        module.expert_start,
        experts,
        hidden.shape[1],
        top_k,
        256,
        pdl,
        num_warps=4,
        enable_fp_fusion=False,
        launch_pdl=pdl,
    )
    return out
