# Copyright 2026 SGLang contributors. Licensed under Apache-2.0.
# Adapted from sgl-project/sglang e62ac6279646c686d3b328f58d9b5c2c2d5927dc,
# python/sglang/kernels/ops/inkling2/{nvfp4,moe}.py.
"""Inkling 2 activation quantization and interleaved, capped SwiGLU.

The NVFP4 scale is selected by four-versus-six mean absolute reconstruction
error, with per-token outer scales. Routing weights precede the second GEMM.
"""

from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# GPU-only Triton is absent from the development type-checking environment.
jit: Any = triton.jit


@jit
def _pack_roundtrip(x):
    low, high = tl.split(x)
    packed, halves = tl.inline_asm_elementwise(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $3, $2; "
        "cvt.u32.u8 $0, b; cvt.rn.f16x2.e2m1x2 $1, b; }",
        "=r,=r,f,f",
        [low, high],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    low = (halves & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    high = (halves >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return packed.to(tl.uint8), tl.join(low, high).to(tl.float32)


@jit
def _block_error(x, reconstructed):
    groups: tl.constexpr = x.shape[0]
    error = tl.abs(reconstructed - x)
    for width in tl.static_range(3, -1, -1):
        pairs = tl.reshape(error, (groups, 1 << width, 2))
        low, high = tl.split(pairs)
        error = low + high
    return tl.reshape(error, (groups,))


@jit
def _row_amax(X, Amax, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
    tl.store(Amax + row, tl.max(tl.abs(x), 0))


@jit
def _quantize(X, Values, Scales, Amax, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    _quantize_chunk(
        X, Values, Scales, tl.load(Amax + row), row, tl.program_id(1), K, BLOCK
    )


@jit
def _quantize_chunk(
    X, Values, Scales, amax, row, chunk, K: tl.constexpr, BLOCK: tl.constexpr
):
    packed, scales = _quantize_values(X, amax, row, chunk, K, BLOCK)
    packed_cols = chunk * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    scale_cols = chunk * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
    tl.store(Values + row * (K // 2) + packed_cols, packed, packed_cols < K // 2)
    tl.store(Scales + row * (K // 16) + scale_cols, scales, scale_cols < K // 16)


@jit
def _quantize_values(X, amax, row, chunk, K: tl.constexpr, BLOCK: tl.constexpr):
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
    x = tl.reshape(x, (BLOCK // 16, 16))
    encoder_scale = tl.minimum(tl.div_rn(1536.0, amax), 3.4028234663852886e38)
    decoder_scale = tl.div_rn(1.0, encoder_scale)
    outer_scale = amax * (1.0 / 1536.0)
    block_max = tl.max(tl.abs(x), 1)
    scale6 = tl.minimum(block_max * (encoder_scale * (1.0 / 6.0)), 448.0)
    scale4 = tl.minimum(block_max * (encoder_scale * 0.25), 448.0)
    scale6 = scale6.to(tl.float8e4nv).to(tl.float32)
    scale4 = scale4.to(tl.float8e4nv).to(tl.float32)
    inverse6 = tl.minimum(tl.div_rn(1.0, scale6 * decoder_scale), 3.4028234663852886e38)
    inverse4 = tl.minimum(tl.div_rn(1.0, scale4 * decoder_scale), 3.4028234663852886e38)
    packed6, rounded6 = _pack_roundtrip(
        tl.reshape(x * inverse6[:, None], (BLOCK // 16, 8, 2))
    )
    packed4, rounded4 = _pack_roundtrip(
        tl.reshape(x * inverse4[:, None], (BLOCK // 16, 8, 2))
    )
    reconstructed6 = tl.reshape(rounded6, (BLOCK // 16, 16)) * scale6[:, None]
    reconstructed4 = tl.reshape(rounded4, (BLOCK // 16, 16)) * scale4[:, None]
    error6 = _block_error(x, reconstructed6 * outer_scale)
    error4 = _block_error(x, reconstructed4 * outer_scale)
    choose4 = error4 < error6
    packed = tl.reshape(tl.where(choose4[:, None], packed4, packed6), (BLOCK // 2,))
    scales = tl.where(choose4, scale4, scale6).to(tl.float8e4nv)
    return packed, scales


@jit
def _mxfp8_block(groups):
    amax = tl.max(tl.abs(groups), 1)
    raw_scale = amax * (1.0 / 448.0)
    min_scale = tl.full((), 2.0**-127, tl.float32)
    exponent = ((raw_scale.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = tl.where(raw_scale <= min_scale, 0, tl.minimum(exponent, 254))
    exponent = tl.where(raw_scale != raw_scale, 255, exponent)
    scale = tl.maximum((exponent << 23).to(tl.float32, bitcast=True), min_scale)
    scale = tl.where(exponent == 255, float("nan"), scale)
    inverse = tl.where(exponent == 0, 2.0**127, libdevice.rcp_rn(scale))
    inverse = tl.where(exponent == 255, 0.0, inverse)
    values = groups * inverse[:, None]
    values = tl.minimum(
        tl.maximum(values, -448.0), 448.0, propagate_nan=tl.PropagateNan.ALL
    )
    return values.to(tl.float8e4nv), exponent


@jit
def _mxfp8_quantize(X, Y, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * N + cols, cols < N, other=0).to(tl.float32)
    values, exponent = _mxfp8_block(tl.reshape(x, (BLOCK // 32, 32)))
    groups = tl.program_id(1) * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    tl.store(S + row * (N // 32) + groups, exponent, groups < N // 32)
    tl.store(Y + row * N + cols, tl.reshape(values, (BLOCK,)), cols < N)


@jit
def _capped_swiglu(gate, up, gamma, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    sigmoid = tl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 $0, t; }",
        "=f,f",
        [gate],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    if GATE_CAP > 0:
        gate = GATE_CAP * tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            [tl.div_rn(gate, GATE_CAP)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    if UP_CAP > 0:
        up = UP_CAP * tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            [tl.div_rn(up, UP_CAP)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    return ((gate * sigmoid) * up) * gamma


def quantize(x: torch.Tensor, precision: str, *, reduce_amax=None):
    """Packed native activation values, block scales, and NVFP4 row maxima."""
    rows, width = x.shape
    if precision == "bf16":
        return x, None, None
    nvfp4 = precision == "nvfp4"
    values = torch.empty(
        (rows, width // 2 if nvfp4 else width),
        device=x.device,
        dtype=torch.uint8 if nvfp4 else torch.float8_e4m3fn,
    )
    scales = torch.empty(
        (rows, width // (16 if nvfp4 else 32)),
        device=x.device,
        dtype=torch.float8_e4m3fn if nvfp4 else torch.uint8,
    )
    amax = torch.empty((rows,), device=x.device, dtype=torch.float32) if nvfp4 else None
    if rows:
        if nvfp4:
            _row_amax[(rows,)](x, amax, width, triton.next_power_of_2(width))
            if reduce_amax is not None:
                reduce_amax(amax)
            _quantize[(rows, triton.cdiv(width, 256))](
                x,
                values,
                scales,
                amax,
                width,
                256,
                enable_fp_fusion=False,
            )
        else:
            _mxfp8_quantize[(rows, triton.cdiv(width, 256))](
                x,
                values,
                scales,
                width,
                256,
                enable_fp_fusion=False,
            )
    return values, scales, amax
