"""Packed/split B1 decode attention, qualified against FP32 unsplit attention.

The released model has six query heads per KV head. Padding each group to eight
lets FA4 pack its tensor-core tiles; KV splits restore parallel work at batch one.
The final kernel combines splits, drops padded heads, and applies the sink/gate.
"""

from typing import Any, cast

import triton
import triton.language as tl

from resin_models_cuda.inkling2.attention import relative_attention_score_mod
from resin_models_cuda.inkling2.fa4 import paged_attention_fp32


@triton.jit
def _pad(
    Q,
    Bias,
    OQ,
    OB,
    H: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, padded_head = tl.program_id(0), tl.program_id(1)
    head = padded_head // 8 * 6 + padded_head % 8
    valid = padded_head % 8 < 6
    col = tl.arange(0, BLOCK)
    q = tl.load(Q + (token * H + head) * D + col, valid & (col < D), 0)
    bias = tl.load(Bias + (token * H + head) * R + col, valid & (col < R), 0)
    padded = H // 6 * 8
    tl.store(OQ + (token * padded + padded_head) * D + col, q, col < D)
    tl.store(OB + (token * padded + padded_head) * R + col, bias, col < R)


def pad_query_bias(q, bias):
    rows, heads, dim = q.shape
    if heads % 6 or dim != 128 or bias.shape[:2] != (rows, heads):
        raise ValueError("Inkling 2 padded decode requires six query heads per KV head")
    padded = heads // 6 * 8
    out = q.new_empty((rows, padded, dim))
    relative = bias.new_empty((rows, padded, bias.shape[-1]))
    cast(Any, _pad)[(rows, padded)](
        q.contiguous(),
        bias.contiguous(),
        out,
        relative,
        heads,
        bias.shape[-1],
        dim,
        triton.next_power_of_2(bias.shape[-1]),
    )
    return out, relative


@triton.jit(
    do_not_specialize=["Rows", "Splits", "GR", "GH", "GD", "SR", "SH"],
    do_not_specialize_on_alignment=["Gate", "Sinks"],
)
def _combine(
    Partials,
    LSE,
    Gate,
    Sinks,
    Y,
    Rows,
    Splits,
    GR,
    GH,
    GD,
    SR,
    SH,
    H: tl.constexpr,
    PH: tl.constexpr,
    D: tl.constexpr,
    PADDED: tl.constexpr,
    BS: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    source = head // 6 * 8 + head % 6 if PADDED else head
    split, col = tl.arange(0, BS), tl.arange(0, D)
    lse = tl.load(
        LSE + (split * PH + source) * Rows + row, split < Splits, float("-inf")
    )
    active = lse > float("-inf")
    maximum = tl.max(lse, 0)
    weight = tl.exp(lse - maximum)
    denominator = tl.sum(weight, 0)
    partial = tl.load(
        Partials + ((split[:, None] * Rows + row) * PH + source) * D + col[None, :],
        active[:, None],
        0,
    )
    combined = tl.sum(partial * weight[:, None], 0) / denominator
    gate = tl.load(Gate + row * GR + head * GH + col * GD).to(tl.float32)
    sink = tl.load(Sinks + row * SR + head * SH).to(tl.float32)
    value = (
        combined * tl.sigmoid(gate) * tl.sigmoid(maximum + tl.log(denominator) - sink)
    )
    tl.store(Y + (row * H + head) * D + col, value)


def combine_gate_sink(partials, lse, gate, sinks, *, padded=False):
    if partials.ndim == 3:
        partials, lse = partials[None], lse[None]
    splits, rows, padded_heads, dim = partials.shape
    heads = sinks.shape[-1]
    expected_heads = heads // 6 * 8 if padded else heads
    if padded_heads != expected_heads:
        raise ValueError("split attention head geometry differs from its gate")
    output = gate.new_empty((rows, heads, dim))
    cast(Any, _combine)[(rows, heads)](
        partials,
        lse,
        gate,
        sinks,
        output,
        rows,
        splits,
        *gate.stride(),
        *sinks.stride(),
        heads,
        padded_heads,
        dim,
        padded,
        triton.next_power_of_2(splits),
        num_warps=4,
    )
    return output


def gated_decode(
    q,
    k,
    v,
    relative,
    gate,
    sinks,
    *,
    page_table,
    seqused_k,
    cu_seqlens_q,
    tokens,
    num_splits,
    padded=True,
    inputs_padded=False,
    window_left=None,
):
    if padded and not inputs_padded:
        q, relative = pad_query_bias(q, relative)
    partials, lse = paged_attention_fp32(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        page_table=page_table,
        max_seqlen_q=tokens,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=True,
        window_size=(window_left, None),
        score_mod=relative_attention_score_mod(relative.shape[-1]),
        aux_tensors=[relative],
        num_splits=num_splits,
    )
    return combine_gate_sink(partials, lse, gate, sinks, padded=padded)
