"""Paged Inkling 2 attention, relative bias, and the DSpark block mask."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from torch import Tensor

from resin_models_cuda.inkling2.cache import DeviceCache


@dataclass(frozen=True, slots=True)
class PagedAttentionBatch:
    """Per-forward attention metadata shared by every layer in each cache group."""

    page_tables: dict[int, Tensor]
    used_lengths: dict[int, Tensor]
    max_lengths: dict[int, int]
    cu_query_lengths: Tensor
    log_tau: Tensor | None = None


def prepare_paged_attention_batch(
    cache: DeviceCache,
    sequence_lengths: Tensor,
    max_sequence_length: int,
    *,
    batch: int,
    tokens: int,
    layer_counts: Mapping[int, int],
    log_tau: Tensor | None = None,
) -> PagedAttentionBatch:
    """Build graph-stable page tables once for all attention layers."""

    if sequence_lengths.shape != (batch,):
        raise ValueError("Inkling sequence lengths do not match the attention batch")
    if not layer_counts or any(
        not 0 <= index < len(cache.groups) or count <= 0
        for index, count in layer_counts.items()
    ):
        raise ValueError("Inkling attention cache-group geometry is invalid")

    page_tables = {}
    used_lengths = {}
    max_lengths = {}
    for index, layer_count in layer_counts.items():
        group = cache.groups[index]
        base_pages = cache.pages(index, 0, layer_count)
        layer_offsets = torch.arange(
            layer_count,
            dtype=base_pages.dtype,
            device=base_pages.device,
        ).view(layer_count, 1, 1)
        tables = (base_pages.unsqueeze(0) + layer_offsets).to(torch.int32)
        page_tables[index] = tables
        retained_starts = group.first_blocks * cache.page_tokens
        used_lengths[index] = (sequence_lengths - retained_starts).to(torch.int32)
        max_lengths[index] = min(
            max_sequence_length, tables.shape[-1] * cache.page_tokens
        )
    cu_query_lengths = (
        torch.arange(batch + 1, dtype=torch.int32, device=sequence_lengths.device)
        * tokens
    )
    return PagedAttentionBatch(
        page_tables,
        used_lengths,
        max_lengths,
        cu_query_lengths,
        log_tau,
    )


@cache
def relative_attention_score_mod(
    rel_extent: int, window_at_start: int = 0, *, skip_distant_bias: bool = False
) -> Callable[..., Any]:
    """Build FA4's hidden-conditioned relative-bias score modifier."""

    try:
        import cutlass.cute as cute
        from cutlass import Float32, const_expr
        from flash_attn.cute.seqlen_info import SeqlenInfoQK
    except ImportError as error:
        raise ImportError(
            "Inkling relative attention requires FlashAttention-4's CUTE interface"
        ) from error

    @cute.jit
    def score_mod(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        if const_expr(rel_extent > 0):
            rel_logits = aux_tensors[0]
            distance = (q_idx + local_offset) - kv_idx
            global_q = seqlen_info.offset_q + q_idx
            relative = distance[0]
            if const_expr(skip_distant_bias):
                bias = Float32(0.0)
                if (relative >= 0) & (relative < rel_extent):
                    bias = Float32(rel_logits[global_q[0], h_idx[0], relative])
            else:
                index = relative if relative >= 0 else 0
                index = index if index < rel_extent else rel_extent - 1
                bias = rel_logits[global_q[0], h_idx[0], index]
                bias = Float32(bias) if relative == index else Float32(0.0)
            scores = scores + bias
        if const_expr(window_at_start > 0):
            if kv_idx[0] < local_offset - window_at_start:
                scores = scores * Float32(0.0) + Float32(float("-inf"))
        return scores

    return score_mod


def log_scaling_tau(positions: Tensor, *, n_floor: int, alpha: float) -> Tensor:
    effective = (positions + 1).float()
    return 1.0 + alpha * torch.log(torch.clamp(effective / float(n_floor), min=1.0))


def paged_attention(
    queries: Tensor,
    *,
    cache: DeviceCache,
    group_index: int,
    layer: int,
    layer_count: int,
    positions: Tensor,
    sequence_lengths: Tensor,
    max_sequence_length: int,
    softmax_scale: float,
    window_left: int | None = None,
    relative_logits: Tensor | None = None,
    prepared_batch: PagedAttentionBatch | None = None,
    causal: bool = True,
    window_anchor_at_start: bool = False,
    return_lse: bool = False,
    output_dtype: torch.dtype | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run FA4 with graph-stable capacity bounds and optional softmax statistics."""

    from flash_attn.cute import flash_attn_varlen_func

    if queries.ndim != 4 or positions.shape != queries.shape[:2]:
        raise ValueError("Inkling attention expects [batch, tokens, heads, dim]")
    batch, tokens = positions.shape
    if prepared_batch is None:
        group = cache.groups[group_index]
        page_table = cache.pages(group_index, layer, layer_count)
        retained_starts = group.first_blocks * cache.page_tokens
        used_lengths = (sequence_lengths - retained_starts).to(torch.int32)
        max_k = min(max_sequence_length, page_table.shape[1] * cache.page_tokens)
        cu_q = (
            torch.arange(batch + 1, dtype=torch.int32, device=queries.device) * tokens
        )
    else:
        if (
            group_index not in prepared_batch.page_tables
            or prepared_batch.page_tables[group_index].shape[0] != layer_count
            or not 0 <= layer < layer_count
        ):
            raise ValueError("prepared Inkling attention metadata is malformed")
        page_table = prepared_batch.page_tables[group_index][layer]
        used_lengths = prepared_batch.used_lengths[group_index]
        max_k = prepared_batch.max_lengths[group_index]
        cu_q = prepared_batch.cu_query_lengths
    # Keep the fixed-anchor score modifier authoritative. The wider native
    # window only lets FA4 skip KV tiles that no query in this block can read.
    tile_window_left = window_left
    if window_anchor_at_start and window_left is not None:
        tile_window_left = window_left + tokens - 1
    kwargs: dict[str, Any] = {}
    if relative_logits is not None:
        if relative_logits.shape[:2] != (batch * tokens, queries.shape[2]):
            raise ValueError("Inkling relative logits do not match attention queries")
    if relative_logits is not None or window_anchor_at_start:
        kwargs.update(
            score_mod=relative_attention_score_mod(
                relative_logits.shape[-1] if relative_logits is not None else 0,
                window_left
                if window_anchor_at_start and window_left is not None
                else 0,
                # Full attention traverses tiles beyond the relative-bias table.
                # Sliding layers benefit from their branch-free bias load.
                skip_distant_bias=causal and window_left is None,
            ),
            aux_tensors=[relative_logits] if relative_logits is not None else [],
        )
    if output_dtype == torch.float32:
        from resin_models_cuda.inkling2.fa4 import paged_attention_fp32

        output, lse = paged_attention_fp32(
            queries.reshape(batch * tokens, *queries.shape[2:]),
            cache.keys,
            cache.values,
            cu_seqlens_q=cu_q,
            seqused_k=used_lengths,
            page_table=page_table,
            max_seqlen_q=tokens,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(tile_window_left, None),
            **kwargs,
        )
        output = output.view(batch, tokens, *output.shape[1:])
        return (output, lse) if return_lse else output
    if output_dtype is not None and output_dtype != queries.dtype:
        raise ValueError("unsupported Inkling 2 attention output precision")
    # Pack each draft block's grouped queries; batch size supplies more KV
    # parallelism as split counts shrink. History lengths stay out of JIT policy.
    packed_draft = (
        not causal
        and relative_logits is None
        and (tokens, queries.shape[2], cache.keys.shape[-2], queries.shape[-1])
        in ((8, 24, 4, 128), (8, 12, 2, 128))
    )
    output = flash_attn_varlen_func(
        queries.reshape(batch * tokens, *queries.shape[2:]),
        cache.keys,
        cache.values,
        cu_seqlens_q=cu_q,
        max_seqlen_q=tokens,
        max_seqlen_k=max_k,
        min_seqlen_k=1,
        seqused_k=used_lengths,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(tile_window_left, None),
        num_splits=max(1, 16 // batch) if packed_draft else 1,
        pack_gqa=packed_draft or 128 % (queries.shape[2] // cache.keys.shape[-2]) == 0,
        return_lse=return_lse,
        **kwargs,
    )
    if return_lse:
        output, lse = output[:2]
        return output.view(batch, tokens, *output.shape[1:]), lse
    if isinstance(output, tuple):
        output = output[0]
    return output.view(batch, tokens, *output.shape[1:])


def gated_attention(
    queries: Tensor,
    gate: Tensor,
    sinks: Tensor,
    *,
    cache: DeviceCache,
    group_index: int,
    layer: int,
    prepared_batch: PagedAttentionBatch,
    relative_logits: Tensor,
    window_left: int | None = None,
    inputs_padded: bool = False,
    **kwargs: Any,
) -> Tensor:
    """Target attention with the FP32 sink/gate boundary and packed decode."""

    from resin_models_cuda.inkling2.kernels import gate_sink

    batch, tokens, query_heads, dim = queries.shape
    heads = sinks.shape[-1]
    kv_heads = cache.keys.shape[-2]
    if (
        queries.is_cuda
        and (batch == 1 or inputs_padded)
        and tokens in (1, 8)
        and kv_heads in (2, 4, 8)
        and heads == 6 * kv_heads
        and query_heads == (8 * kv_heads if inputs_padded else heads)
        and dim == 128
    ):
        from resin_models_cuda.inkling2.fa4_bias import decode_splits

        splits = decode_splits(batch, kv_heads, window_left)
        if inputs_padded and relative_logits.shape[-1] == 1280:
            from resin_models_cuda.inkling2.fa4_bias import native_bias_attention

            return native_bias_attention(
                queries.flatten(0, 1),
                cache.keys,
                cache.values,
                relative_logits,
                gate.reshape(batch * tokens, heads, dim),
                sinks.reshape(batch * tokens, heads),
                page_table=prepared_batch.page_tables[group_index][layer],
                seqused_k=prepared_batch.used_lengths[group_index],
                cu_seqlens_q=prepared_batch.cu_query_lengths,
                tokens=tokens,
                num_splits=splits,
                window_left=window_left,
            ).view(batch, tokens, heads, dim)
        from resin_models_cuda.inkling2.decode_attention import gated_decode

        # Paired B300 measurements include padding and the fused combine/gate.
        # Only model geometry chooses the policy; request lengths stay dynamic.
        return gated_decode(
            queries.flatten(0, 1),
            cache.keys,
            cache.values,
            relative_logits,
            gate.reshape(batch * tokens, heads, dim),
            sinks.reshape(batch * tokens, heads),
            page_table=prepared_batch.page_tables[group_index][layer],
            seqused_k=prepared_batch.used_lengths[group_index],
            cu_seqlens_q=prepared_batch.cu_query_lengths,
            tokens=tokens,
            num_splits=splits,
            padded=True,
            inputs_padded=inputs_padded,
            window_left=window_left,
        ).view(batch, tokens, heads, dim)
    if inputs_padded:
        raise ValueError("padded Inkling inputs require the selected decode geometry")
    output, lse = paged_attention(
        queries,
        cache=cache,
        group_index=group_index,
        layer=layer,
        prepared_batch=prepared_batch,
        relative_logits=relative_logits,
        window_left=window_left,
        return_lse=True,
        output_dtype=torch.float32,
        **kwargs,
    )
    return gate_sink(
        output.reshape(batch * tokens, heads, dim),
        gate.reshape(batch * tokens, -1),
        lse,
        sinks.reshape(batch * tokens, heads),
    ).view_as(queries)


__all__ = [
    "PagedAttentionBatch",
    "gated_attention",
    "log_scaling_tau",
    "paged_attention",
    "prepare_paged_attention_batch",
    "relative_attention_score_mod",
]
