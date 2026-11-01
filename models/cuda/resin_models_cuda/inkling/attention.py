"""Inkling FlashAttention-4 relative and DSpark rotary attention kernels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from torch import Tensor

from resin_models_cuda.inkling.cache import DeviceCache


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
def relative_attention_score_mod(rel_extent: int) -> Callable[..., Any]:
    """Build FA4's hidden-conditioned relative-bias score modifier."""

    try:
        import cutlass.cute as cute
        from cutlass import Float32
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
        rel_logits = aux_tensors[0]
        local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        distance = (q_idx + local_offset) - kv_idx
        global_q = seqlen_info.offset_q + q_idx
        distance_0 = distance[0]
        relative = distance_0 if distance_0 >= 0 else 0
        relative = relative if relative < rel_extent else rel_extent - 1
        bias = rel_logits[global_q[0], h_idx[0], relative]
        bias = Float32(bias) if distance_0 == relative else Float32(0.0)
        return scores + bias

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
) -> Tensor:
    """Run causal paged FA4 for a uniform-width batch."""

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
    kwargs: dict[str, Any] = {}
    if relative_logits is not None:
        if relative_logits.shape[:2] != (batch * tokens, queries.shape[2]):
            raise ValueError("Inkling relative logits do not match attention queries")
        kwargs.update(
            score_mod=relative_attention_score_mod(relative_logits.shape[-1]),
            aux_tensors=[relative_logits],
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
        causal=True,
        window_size=(window_left, None),
        num_splits=1,
        pack_gqa=True,
        **kwargs,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.view(batch, tokens, *output.shape[1:])


__all__ = [
    "PagedAttentionBatch",
    "log_scaling_tau",
    "paged_attention",
    "prepare_paged_attention_batch",
    "relative_attention_score_mod",
]
