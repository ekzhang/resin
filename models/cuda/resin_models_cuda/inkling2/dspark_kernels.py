# Copyright 2026 SGLang contributors. Licensed under the Apache License, Version 2.0.
# Adapted from https://github.com/sgl-project/sglang/blob/
# e62ac6279646c686d3b328f58d9b5c2c2d5927dc/python/sglang/kernels/ops/inkling2/mtp.py
"""Fixed-launch target-mixture normalization and BF16 recurrent DSpark kernels."""

from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit(
    do_not_specialize=["BASE_STRIDE", "BIAS_STRIDE"],
    do_not_specialize_on_alignment=["Base"],
)
def _corrected_logits(
    Base,
    Bias,
    Out,
    BASE_STRIDE,
    BIAS_STRIDE,
    VOCAB: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = tl.load(Base + row * BASE_STRIDE + cols, cols < VOCAB, 0).to(tl.float32)
    value = base
    if Bias is not None:
        bias = tl.load(Bias + row * BIAS_STRIDE + cols, cols < VOCAB, 0).to(tl.float32)
        # The checkpoint adds in BF16 before applying its FP32 logit soft cap.
        value = (
            tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;",
                "=f,f,f",
                [base, bias],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            .to(Base.dtype.element_ty)
            .to(tl.float32)
        )
    scaled = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [value, 1.0 / CAP],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # tanh rounds to its input below 2**-13 in FP32. Keep subnormals on this
    # path: the approximate CUDA tanh instruction flushes them to zero.
    activated = tl.where(tl.abs(scaled) < 2.0**-13, scaled, libdevice.tanh(scaled))
    value = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [activated, CAP],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(Out + row * VOCAB + cols, value, cols < VOCAB)


def corrected_logits(base, bias, cap):
    rows, vocab = base.shape
    output = torch.empty_like(base, memory_format=torch.contiguous_format)
    cast(Any, _corrected_logits)[(rows, triton.cdiv(vocab, 1024))](
        base,
        bias,
        output,
        base.stride(0),
        bias.stride(0) if bias is not None else 0,
        vocab,
        cap,
        1024,
        enable_fp_fusion=False,
    )
    return output


def softcap_logits(values, cap):
    logits = corrected_logits(values.reshape(-1, values.shape[-1]), None, cap)
    return logits.view(values.shape)


@triton.jit(
    do_not_specialize=[
        "tokens",
        "batch_stride",
        "token_stride",
        "head_stride",
        "position_batch_stride",
        "position_token_stride",
    ],
    do_not_specialize_on_alignment=["Positions"],
)
def _norm_neox_rope(
    X,
    Weight,
    Positions,
    Rope,
    Out,
    tokens,
    batch_stride,
    token_stride,
    head_stride,
    position_batch_stride,
    position_token_stride,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0).to(tl.int64), tl.program_id(1)
    batch, token = row // tokens, row % tokens
    cols = tl.arange(0, BLOCK)
    x = tl.load(
        X + batch * batch_stride + token * token_stride + head * head_stride + cols,
        cols < DIM,
        0,
    ).to(tl.float32)
    weight = tl.load(Weight + cols, cols < DIM, 0).to(tl.float32)
    squared = x * x
    if DIM == 128:
        # Quack 0.6.4 uses 16 lanes with eight contiguous values per lane
        # for this BF16 head width. Preserve its local accumulation order.
        lanes = tl.arange(0, 16)
        partial = tl.full((16,), 0, tl.float32)
        for item in tl.static_range(8):
            partial += tl.gather(squared, lanes * 8 + item, 0)
        sum_sq = tl.sum(partial, 0)
    else:
        sum_sq = tl.sum(squared, 0)
    inv_rms = tl.rsqrt(sum_sq / DIM + EPS)
    # RMSNorm materializes BF16 before RoPE in the checkpoint computation.
    normalized = ((x * inv_rms) * weight).to(X.dtype.element_ty).to(tl.float32)
    half: tl.constexpr = DIM // 2
    partner = tl.gather(normalized, (cols + half) % DIM, 0)
    position = tl.load(
        Positions + batch * position_batch_stride + token * position_token_stride
    ).to(tl.int64)
    cos = tl.load(Rope + position * DIM + cols % half, cols < DIM, 0)
    sin = tl.load(Rope + position * DIM + half + cols % half, cols < DIM, 0)
    # Keep the separate FP32 multiplies and add/subtract of the eager formula.
    direct, rotated = normalized * cos, partner * sin
    value = tl.where(cols < half, direct - rotated, direct + rotated)
    tl.store(Out + (row * HEADS + head) * DIM + cols, value, cols < DIM)


def norm_neox_rope(hidden, positions, rope, weight, eps):
    """Normalize packed projection views and apply NeoX RoPE in one launch."""
    assert hidden.ndim in (3, 4) and positions.shape == hidden.shape[:-2]
    assert hidden.stride(-1) == 1 and weight.is_contiguous() and rope.is_contiguous()
    heads, dim = hidden.shape[-2:]
    assert dim % 2 == 0 and weight.shape == (dim,) and rope.shape[-1] == dim
    tokens = positions.shape[-1]
    batch = positions.shape[0] if positions.ndim == 2 else 1
    out = hidden.new_empty(hidden.shape)
    if batch * tokens:
        cast(Any, _norm_neox_rope)[(batch * tokens, heads)](
            hidden,
            weight,
            positions,
            rope,
            out,
            tokens,
            hidden.stride(0) if hidden.ndim == 4 else 0,
            hidden.stride(-3),
            hidden.stride(-2),
            positions.stride(0) if positions.ndim == 2 else 0,
            positions.stride(-1),
            heads,
            dim,
            eps,
            triton.next_power_of_2(dim),
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _mix_rmsnorm(
    X,
    ALPHA,
    WEIGHT,
    OUT,
    ROW_STRIDE: tl.constexpr,
    TAP_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    mixed = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(TAPS):
        x = tl.load(
            X + row * ROW_STRIDE + tap * TAP_STRIDE + h * HIDDEN_STRIDE,
            h < HIDDEN,
            0,
        ).to(tl.float32)
        alpha = tl.load(ALPHA + tap).to(tl.float32)
        mixed += x * alpha
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    weight = tl.load(WEIGHT + h, h < HIDDEN, 0).to(tl.float32)
    tl.store(OUT + row * HIDDEN + h, mixed * inv_rms * weight, h < HIDDEN)


def mix_rmsnorm(
    targets: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    rows, taps, hidden = targets.shape
    assert alpha.shape == (taps,) and alpha.is_contiguous()
    assert weight.shape == (hidden,) and weight.is_contiguous()
    out = targets.new_empty((rows, hidden))
    if rows:
        cast(Any, _mix_rmsnorm)[(rows,)](
            targets,
            alpha,
            weight,
            out,
            *targets.stride(),
            taps,
            hidden,
            eps,
            triton.next_power_of_2(hidden),
            num_warps=8,
        )
    return out


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _rnn_gates(
    Proj, State, NewState, Latent, SP, SS, R: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < R
    gate_raw = tl.load(Proj + row * SP + cols, mask, other=0).to(tl.float32)
    cand_raw = tl.load(Proj + row * SP + R + cols, mask, other=0).to(tl.float32)
    out_raw = tl.load(Proj + row * SP + 2 * R + cols, mask, other=0).to(tl.float32)
    state = tl.load(State + row * SS + cols, mask, other=0).to(tl.float32)
    # Each op rounds to BF16 like the eager torch ops it replaces.
    gate = _bf16(tl.div_rn(1.0, 1.0 + libdevice.exp(-gate_raw)))
    kept = _bf16(gate * state)
    blend = _bf16(_bf16(1.0 - gate) * _bf16(libdevice.tanh(cand_raw)))
    tl.store(NewState + row * R + cols, kept + blend, mask)
    tl.store(Latent + row * R + cols, libdevice.tanh(out_raw), mask)


def rnn_gates(proj: torch.Tensor, state: torch.Tensor):
    """(state', latent) of Inkling2MTPHead._rnn_step from the rnn_proj output."""
    rows, rank = state.shape
    assert proj.shape == (rows, 3 * rank) and proj.stride(-1) == state.stride(-1) == 1
    new_state, latent = state.new_empty((rows, rank)), state.new_empty((rows, rank))
    if rows:
        cast(Any, _rnn_gates)[(rows,)](
            proj,
            state,
            new_state,
            latent,
            proj.stride(0),
            state.stride(0),
            rank,
            triton.next_power_of_2(rank),
            num_warps=1,
            enable_fp_fusion=False,
        )
    return new_state, latent


@triton.jit
def _rnn_inputs(
    State,
    Tokens,
    Embedding,
    Hidden,
    FRnn,
    Out,
    token_stride,
    hidden_stride,
    RANK: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    width: tl.constexpr = 2 * RANK + 2 * HIDDEN
    r = tl.arange(0, BLOCK_R)
    h = tl.arange(0, BLOCK_H)
    token = tl.load(Tokens + row * token_stride).to(tl.int64)
    out = Out + row * width
    tl.store(out + r, tl.load(State + row * RANK + r, r < RANK), r < RANK)
    embedding = tl.load(Embedding + token * RANK + r, r < RANK)
    tl.store(out + RANK + r, embedding.to(Out.dtype.element_ty), r < RANK)
    hidden = tl.load(Hidden + row * hidden_stride + h, h < HIDDEN)
    tl.store(out + 2 * RANK + h, hidden, h < HIDDEN)
    tl.store(
        out + 2 * RANK + HIDDEN + h,
        tl.load(FRnn + row * HIDDEN + h, h < HIDDEN),
        h < HIDDEN,
    )


def rnn_inputs(state, tokens, embedding, hidden, f_rnn):
    """torch.cat([state, embedding[tokens], hidden, f_rnn], -1) in one kernel."""
    rows, rank = state.shape
    width = hidden.shape[1]
    assert state.is_contiguous() and f_rnn.is_contiguous() and hidden.stride(1) == 1
    assert state.dtype == hidden.dtype == f_rnn.dtype and embedding.is_contiguous()
    out = state.new_empty((rows, 2 * rank + 2 * width))
    if rows:
        cast(Any, _rnn_inputs)[(rows,)](
            state,
            tokens,
            embedding,
            hidden,
            f_rnn,
            out,
            tokens.stride(0),
            hidden.stride(0),
            rank,
            width,
            triton.next_power_of_2(rank),
            triton.next_power_of_2(width),
        )
    return out
