"""Qwen3.5 FlashInfer paged-attention kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from resin.backend.cuda.paged_kv import SlabPagedKvCache
from resin_models_cuda.paged_kv import append_slab_kv

append_kv = append_slab_kv


def _plan_wrapper(
    wrapper,
    tokens: int,
    kv_indptr,
    pages,
    last_page_lengths,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    page_tokens: int,
    common: dict[str, Any],
) -> None:
    args = (
        kv_indptr,
        pages,
        last_page_lengths,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        page_tokens,
    )
    if tokens == 1:
        wrapper.plan(*args, **common)
    else:
        qo_indptr = (
            torch.arange(len(kv_indptr), dtype=torch.int32, device=kv_indptr.device)
            * tokens
        )
        wrapper.plan(qo_indptr, *args, causal=True, **common)


@dataclass(slots=True)
class PreparedPagedAttention:
    """One FlashInfer plan shared by every physical layer in a cache group."""

    wrapper: Any
    cache: SlabPagedKvCache
    base_pages: Tensor
    physical_pages: Tensor
    layer_count: int
    tokens: int

    def run(self, queries: Tensor, layer: int) -> Tensor:
        if not 0 <= layer < self.layer_count:
            raise ValueError("Qwen3.5 attention layer lies outside its cache group")
        torch.add(self.base_pages, layer, out=self.physical_pages)
        query = queries[:, 0] if self.tokens == 1 else queries.flatten(0, 1)
        output = self.wrapper.run(query, (self.cache.keys, self.cache.values))
        return output.view_as(queries)


class PagedAttentionRuntime:
    """Reusable FlashInfer planning workspace for eager Qwen attention."""

    def __init__(
        self,
        device: torch.device,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        workspace_bytes: int,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.device = device
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        self.prefill = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, "NHD", backend="auto"
        )
        self.decode = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_tensor_cores=True,
            backend="auto",
        )

    def prepare(
        self,
        cache: SlabPagedKvCache,
        *,
        group_index: int,
        layer_count: int,
        sequence_lengths: Tensor,
        tokens: int,
        softmax_scale: float,
    ) -> PreparedPagedAttention:
        group = cache.groups[group_index]
        page_tokens = cache.page_tokens
        retained_starts = group.first_blocks * page_tokens
        used_lengths = (sequence_lengths - retained_starts).to(torch.int32)
        if bool(torch.any(used_lengths <= 0)):
            raise ValueError("Qwen3.5 paged attention requires retained context")
        required_pages = torch.div(
            used_lengths + page_tokens - 1,
            page_tokens,
            rounding_mode="floor",
        )
        width = group.slab_ids.shape[1]
        if bool(torch.any(required_pages > width)):
            raise ValueError("Qwen3.5 paged attention exceeds its cache lease")

        pages_per_slab = cache.slab_units // layer_count
        page_table = group.slab_ids * pages_per_slab + group.slotnums
        columns = torch.arange(width, device=self.device)[None]
        logical_pages = page_table[columns < required_pages[:, None]].to(torch.int32)
        base_pages = logical_pages * layer_count
        physical_pages = base_pages.clone()
        kv_indptr = torch.zeros(
            len(used_lengths) + 1, dtype=torch.int32, device=self.device
        )
        kv_indptr[1:] = required_pages.cumsum(0)
        last_page_lengths = (used_lengths - 1).remainder(page_tokens) + 1
        common = {
            "pos_encoding_mode": "NONE",
            "q_data_type": torch.bfloat16,
            "kv_data_type": cache.keys.dtype,
            "o_data_type": torch.bfloat16,
            "sm_scale": softmax_scale,
        }
        wrapper = self.decode if tokens == 1 else self.prefill
        _plan_wrapper(
            wrapper,
            tokens,
            kv_indptr,
            physical_pages,
            last_page_lengths,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            page_tokens=page_tokens,
            common=common,
        )
        return PreparedPagedAttention(
            wrapper,
            cache,
            base_pages,
            physical_pages,
            layer_count,
            tokens,
        )


@dataclass(frozen=True, slots=True)
class GraphAttentionSpec:
    """One statically ordered attention call inside a Qwen decode graph."""

    group_index: int
    layer_count: int
    tokens: int
    sequence_offset: int


@dataclass(slots=True)
class _GraphAttentionCall:
    spec: GraphAttentionSpec
    prepared: PreparedPagedAttention
    kv_indptr: Tensor
    last_page_lengths: Tensor
    qo_indptr: Tensor | None


class GraphPagedAttentionRuntime:
    """Fixed-buffer FlashInfer calls used by one captured decode bucket."""

    def __init__(
        self,
        device: torch.device,
        cache: SlabPagedKvCache,
        specs: tuple[GraphAttentionSpec, ...],
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        workspace: Tensor,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.device = device
        self.cache = cache
        self.specs = specs
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.softmax_scale = head_dim**-0.5
        self.workspace = workspace
        self._cursor = 0
        if not specs:
            raise ValueError("Qwen3.5 graph attention requires at least one call")
        calls = []
        batch_size = cache.groups[0].slab_ids.shape[0]
        for spec in specs:
            if (
                spec.tokens <= 0
                or spec.sequence_offset < spec.tokens
                or spec.layer_count <= 0
                or cache.slab_units % spec.layer_count
            ):
                raise ValueError("invalid Qwen3.5 graph attention geometry")
            group = cache.groups[spec.group_index]
            initial_pages = (spec.tokens + cache.page_tokens - 1) // cache.page_tokens
            if initial_pages > group.slab_ids.shape[1]:
                raise ValueError(
                    "Qwen3.5 graph attention queries exceed its page capacity"
                )
            page_capacity = batch_size * group.slab_ids.shape[1]
            base_pages = torch.empty(page_capacity, dtype=torch.int32, device=device)
            physical_pages = torch.empty_like(base_pages)
            kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
            last = torch.empty(batch_size, dtype=torch.int32, device=device)
            common = {
                "pos_encoding_mode": "NONE",
                "q_data_type": torch.bfloat16,
                "kv_data_type": cache.keys.dtype,
                "o_data_type": torch.bfloat16,
                "sm_scale": self.softmax_scale,
                "disable_split_kv": True,
                "non_blocking": False,
            }
            initial_indptr = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=device)
                * initial_pages
            )
            initial_indices = torch.zeros(
                batch_size * initial_pages, dtype=torch.int32, device=device
            )
            initial_last = torch.full(
                (batch_size,),
                (spec.tokens - 1) % cache.page_tokens + 1,
                dtype=torch.int32,
                device=device,
            )
            if spec.tokens == 1:
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=True,
                    paged_kv_indptr_buffer=kv_indptr,
                    paged_kv_indices_buffer=physical_pages,
                    paged_kv_last_page_len_buffer=last,
                    backend="fa2",
                )
            else:
                qo_indptr = torch.empty(
                    batch_size + 1, dtype=torch.int32, device=device
                )
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=qo_indptr,
                    paged_kv_indptr_buf=kv_indptr,
                    paged_kv_indices_buf=physical_pages,
                    paged_kv_last_page_len_buf=last,
                    backend="fa2",
                )
            _plan_wrapper(
                wrapper,
                spec.tokens,
                initial_indptr,
                initial_indices,
                initial_last,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                page_tokens=cache.page_tokens,
                common=common,
            )
            calls.append(
                _GraphAttentionCall(
                    spec,
                    PreparedPagedAttention(
                        wrapper,
                        cache,
                        base_pages,
                        physical_pages,
                        spec.layer_count,
                        spec.tokens,
                    ),
                    kv_indptr,
                    last,
                    None if spec.tokens == 1 else qo_indptr,
                )
            )
        self.calls = tuple(calls)

    def refresh(self, positions: Tensor) -> None:
        """Pack the exact readable prefix of each frozen lease on the GPU.

        FA2's no-split schedule depends on the fixed batch/query geometry, not
        KV lengths. The wrappers are planned once; replay updates all of their
        mutable CSR buffers before any layer reads them. Future reserved pages
        stay hidden, including when rows accept different numbers of drafts.
        """
        from resin_models_cuda.qwen3_5.attention_triton import refresh

        if positions.ndim != 1 or len(positions) != len(
            self.calls[0].last_page_lengths
        ):
            raise ValueError("Qwen3.5 graph positions do not match its physical rows")
        for call in self.calls:
            spec = call.spec
            refresh(
                positions,
                self.cache.groups[spec.group_index],
                call.kv_indptr,
                call.last_page_lengths,
                call.prepared.base_pages,
                page=self.cache.page_tokens,
                tokens=spec.tokens,
                offset=spec.sequence_offset,
                slab_units=self.cache.slab_units,
                layers=spec.layer_count,
            )

    def reset(self) -> None:
        self._cursor = 0

    def prepare(
        self,
        cache: SlabPagedKvCache,
        *,
        group_index: int,
        layer_count: int,
        sequence_lengths: Tensor,
        tokens: int,
        softmax_scale: float,
    ) -> PreparedPagedAttention:
        del sequence_lengths
        if cache is not self.cache or self._cursor >= len(self.calls):
            raise RuntimeError("Qwen3.5 graph attention call order changed")
        call = self.calls[self._cursor]
        self._cursor += 1
        expected = call.spec
        if (
            expected.group_index != group_index
            or expected.layer_count != layer_count
            or expected.tokens != tokens
            or softmax_scale != self.softmax_scale
        ):
            raise RuntimeError("Qwen3.5 graph attention signature changed")
        return call.prepared

    def graph_inputs(self) -> dict[str, Tensor]:
        # CSR and base pages are graph outputs of refresh(), not staged inputs.
        return {}

    def plan(self, cache_plan: Any, positions: Any) -> dict[str, object]:
        """Validate a newly acquired lease; retain the fixed no-split plans.

        Continuations use the already validated immutable tables and bypass
        this host method. Exact accepted positions never need a host readback.
        """

        import numpy as np

        positions = np.asarray(positions, dtype=np.int64)
        rows = cache_plan.rows
        graph_rows = self.cache.groups[0].slab_ids.shape[0]
        if positions.shape != (rows,) or rows > graph_rows:
            raise ValueError("Qwen3.5 graph attention rows do not match its cache plan")
        for call in self.calls:
            spec = call.spec
            group = cache_plan.groups[spec.group_index]
            graph_width = self.cache.groups[spec.group_index].slab_ids.shape[1]
            lengths = positions + spec.sequence_offset
            used = (
                lengths
                - group.block_table_start_indices.astype(np.int64)
                * self.cache.page_tokens
            )
            count = (used + self.cache.page_tokens - 1) // self.cache.page_tokens
            if np.any(used < spec.tokens):
                raise ValueError("Qwen3.5 graph attention requires retained queries")
            if np.any(count > np.diff(group.block_indptr)) or np.any(
                count > graph_width
            ):
                raise ValueError("Qwen3.5 graph attention exceeds its cache lease")
        self.reset()
        return {}


__all__ = [
    "GraphAttentionSpec",
    "GraphPagedAttentionRuntime",
    "PagedAttentionRuntime",
    "PreparedPagedAttention",
    "append_kv",
]
