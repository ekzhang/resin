"""Full-attention prefill with a page-aligned unbiased prefix."""

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor

from resin_models_cuda.inkling2.attention import (
    MIN_SPLIT_QUERY_TOKENS,
    _unbiased_prefix_pages,
    relative_attention_score_mod,
)
from resin_models_cuda.inkling2.cache import PAGE_TOKENS
from resin_models_cuda.inkling2.fa4 import paged_attention_fp32


@triton.jit(
    do_not_specialize=["Rows", "GR", "GH", "GD", "SR", "SH"],
    do_not_specialize_on_alignment=["Gate", "Sinks"],
)
def _merge_gate(
    Far,
    Near,
    FarLSE,
    NearLSE,
    Gate,
    Sinks,
    Y,
    Rows,
    GR,
    GH,
    GD,
    SR,
    SH,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, head = index // H, index % H
    valid = row < Rows
    lse_index = head * Rows + row
    far_lse = tl.load(FarLSE + lse_index, valid, 0)
    near_lse = tl.load(NearLSE + lse_index, valid, 0)
    maximum = tl.maximum(far_lse, near_lse)
    far_weight = tl.exp(far_lse - maximum)
    near_weight = tl.exp(near_lse - maximum)
    total = far_weight + near_weight
    channel = tl.arange(0, D)[None, :]
    col = index[:, None] * D + channel
    far = tl.load(Far + col, valid[:, None], 0)
    near = tl.load(Near + col, valid[:, None], 0)
    gate = tl.load(
        Gate + (row * GR + head * GH)[:, None] + channel * GD, valid[:, None], 0
    ).to(tl.float32)
    sink = tl.load(Sinks + row * SR + head * SH, valid, 0).to(tl.float32)
    combined = (far * far_weight[:, None] + near * near_weight[:, None]) / total[
        :, None
    ]
    lse = maximum + tl.log(total)
    output = combined * tl.sigmoid(gate) * tl.sigmoid(lse[:, None] - sink[:, None])
    tl.store(Y + col, output, valid[:, None])


def gated_prefill_split(
    q: Tensor,
    keys: Tensor,
    values: Tensor,
    relative_logits: Tensor,
    gate: Tensor,
    sinks: Tensor,
    *,
    page_table: Tensor,
    far_pages: int,
    far_lengths: Tensor,
    near_lengths: Tensor,
    cu_query: Tensor,
    softmax_scale: float,
) -> Tensor:
    rows, heads, dim = q.shape
    if relative_logits.shape[:2] != (rows, heads):
        raise ValueError("Inkling relative logits do not match attention queries")
    kwargs = dict(
        cu_seqlens_q=cu_query,
        max_seqlen_q=rows,
        softmax_scale=softmax_scale,
        window_size=(None, None),
    )
    far, far_lse = paged_attention_fp32(
        q,
        keys,
        values,
        seqused_k=far_lengths,
        page_table=page_table[:, :far_pages],
        causal=False,
        **kwargs,
    )
    # Dropping the prefix from K also shifts FA4's bottom-right query anchor,
    # which keeps the original causal mask and relative distances in the suffix.
    near, near_lse = paged_attention_fp32(
        q,
        keys,
        values,
        seqused_k=near_lengths,
        page_table=page_table[:, far_pages:],
        causal=True,
        score_mod=relative_attention_score_mod(
            relative_logits.shape[-1], skip_distant_bias=True
        ),
        aux_tensors=[relative_logits],
        **kwargs,
    )
    result = gate.new_empty(q.shape)
    cast(Any, _merge_gate)[(triton.cdiv(rows * heads, 4),)](
        far,
        near,
        far_lse,
        near_lse,
        gate,
        sinks,
        result,
        rows,
        *gate.stride(),
        *sinks.stride(),
        heads,
        dim,
        4,
    )
    return result


@torch.inference_mode()
def warm_prefill_split(
    kv_heads: int,
    relative_extent: int,
    device: torch.device,
    *,
    prefill_chunk_size: int,
    max_sequence_length: int,
) -> None:
    """Compile the two attention launches and the merge with a small KV table."""
    if (
        prefill_chunk_size < MIN_SPLIT_QUERY_TOKENS
        or not _unbiased_prefix_pages(
            max_sequence_length, MIN_SPLIT_QUERY_TOKENS, relative_extent
        )
        or torch.cuda.get_device_capability(device) != (10, 3)
    ):
        return
    storage = torch.zeros(
        4, 2, PAGE_TOKENS, kv_heads, 128, device=device, dtype=torch.bfloat16
    )
    keys, values = storage.unbind(1)
    tokens, heads = PAGE_TOKENS, kv_heads * 6
    q = keys.new_zeros((tokens, heads, 128))
    relative = keys.new_zeros((tokens, heads, relative_extent))
    pages = torch.arange(4, device=device, dtype=torch.int32)[None]
    lengths = torch.full((1,), 2 * PAGE_TOKENS, device=device, dtype=torch.int32)
    gated_prefill_split(
        q,
        keys,
        values,
        relative,
        q,
        q[..., 0],
        page_table=pages,
        far_pages=2,
        far_lengths=lengths,
        near_lengths=lengths,
        cu_query=torch.tensor([0, tokens], device=device, dtype=torch.int32),
        softmax_scale=128**-0.5,
    )
