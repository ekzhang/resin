"""DeepInfer-style per-forward metadata over Engine-owned parent/child pages."""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(kw_only=True)
class ForwardMetadata:
    """One batch's prepared kernel inputs, shared by every consuming layer."""

    positions: Tensor
    context_lens: Tensor
    valid_token_mask: Tensor
    slot_mapping_store_swa: Tensor
    kv_swa_index: Tensor
    positions_c: dict[int, Tensor]
    slot_mapping_store_ca: dict[int, Tensor]
    klen: dict[int, Tensor]
    state_block_idx: Tensor
    query_indices: Tensor
    block_tables: Tensor  # [batch * queries, pages], in flattened query order.
    draft_slots: Tensor
    draft_indices: Tensor
    _selected: dict[tuple[int, int], tuple[Tensor, Tensor]] = field(
        default_factory=dict, repr=False
    )

    def compressed_indices(self, source: int, ratio: int, selected: Tensor) -> Tensor:
        """Map each source's logical top-k once, retaining shared consumer identity."""
        key = source, ratio
        previous = self._selected.get(key)
        if previous is not None and previous[0] is selected:
            return previous[1]
        if selected.ndim != 3 or selected.shape[:2] != self.positions.shape:
            raise ValueError("selected indices must have shape [batch, queries, topk]")
        if selected.is_cuda:
            from resin_models_cuda.deepseek_v41.upstream.cuda.flash_mla_utils import (
                convert_kv_index_tl,
            )

            output = torch.empty_like(selected, dtype=torch.int32)
            batch, queries, topk = selected.shape
            convert_kv_index_tl(topk, 128 // ratio)(
                selected,
                self.klen[ratio],
                self.block_tables,
                output,
            )
        else:
            # The CPU branch is the address equation, independent of TileLang.
            batch, queries, topk = selected.shape
            table = self.block_tables
            block_size = 128 // ratio
            columns = selected.clamp_min(0).long() // block_size
            pages = table.gather(
                1, columns.clamp_max(table.shape[1] - 1).reshape(batch * queries, topk)
            )
            pages = pages.reshape(batch, queries, topk)
            output = torch.where(
                (selected >= 0)
                & (selected < self.klen[ratio][..., None])
                & (columns < table.shape[1])
                & (pages >= 0),
                pages * block_size + selected.remainder(block_size),
                -1,
            ).int()
        self._selected[key] = selected, output
        return output


def prepare_forward(cache, positions: Tensor, *, draft=False, valid_tokens=None):
    """Prepare mutable graph inputs once; no CPU position values enter the graph."""
    if positions.ndim != 2 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be an int32/int64 [batch, queries] tensor")
    batch, queries = positions.shape
    if not batch or not queries or not positions.is_contiguous():
        raise ValueError("positions must be nonempty and contiguous")
    if valid_tokens is not None and (
        valid_tokens.dtype != torch.bool
        or valid_tokens.shape != (batch * queries,)
        or valid_tokens.device != positions.device
        or not valid_tokens.is_contiguous()
    ):
        raise ValueError(
            "valid_tokens must be a contiguous bool mask of flattened rows"
        )
    ratios = tuple(sorted(set(cache.geometry.ratios) - {0}))
    if not positions.is_cuda:
        return _reference(cache, positions, ratios, draft, valid_tokens)

    from resin_models_cuda.deepseek_v41.upstream.cuda.batch_metadata import (
        copy_engine_tables_tl,
        prepare_engine_metadata_tl,
    )
    from resin_models_cuda.deepseek_v41.upstream.cuda.get_kv_index import (
        load_kv_index_tl,
    )

    def empty(*shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device=positions.device)

    swa, ca, state = cache.groups[:3]
    width_swa, width_ca = (group.parent_ids.shape[1] for group in (swa, ca))
    position_out, context, store_swa, state_idx, draft_slots = (
        empty(batch, queries) for _ in range(5)
    )
    valid, request_ids = (
        empty(batch * queries, dtype=torch.bool),
        empty(batch * queries),
    )
    ca_positions, ca_stores, lengths = (
        empty(max(1, len(ratios)), batch, queries) for _ in range(3)
    )
    swa_pivots = empty(batch * queries, 2)
    swa_tables = empty(batch * queries, width_swa)
    block_tables = empty(batch * queries, width_ca)
    draft_indices = empty(batch, queries, (queries + 7) // 8 * 8 if draft else 1)
    prepare_engine_metadata_tl(
        queries if draft else None,
        tuple(cache.geometry.packings[:3]),
        cache.geometry.window_size,
        cache.geometry.draft_tokens,
        ratios,
        str(positions.dtype).removeprefix("torch."),
        draft,
        valid_tokens is not None,
    )(
        positions,
        valid_tokens,
        swa.parent_ids,
        swa.child_slots,
        swa.first_blocks,
        ca.parent_ids,
        ca.child_slots,
        ca.first_blocks,
        state.parent_ids,
        state.child_slots,
        state.first_blocks,
        position_out,
        context,
        valid,
        store_swa,
        state_idx,
        request_ids,
        ca_positions,
        ca_stores,
        lengths,
        swa_pivots,
        swa_tables,
        block_tables,
        draft_slots,
        draft_indices,
    )
    if queries > 8:
        copy_engine_tables_tl(tuple(cache.geometry.packings[:2]))(
            swa.parent_ids,
            swa.child_slots,
            ca.parent_ids,
            ca.child_slots,
            swa_tables,
            block_tables,
        )
    swa_indices = empty(batch * queries, cache.geometry.window_size)
    load_kv_index_tl(cache.geometry.window_size, 128, True)(
        swa_pivots, swa_indices, swa_tables.flatten()
    )
    return ForwardMetadata(
        positions=position_out,
        context_lens=context,
        valid_token_mask=valid,
        slot_mapping_store_swa=store_swa,
        kv_swa_index=swa_indices.reshape(batch, queries, -1),
        positions_c=dict(zip(ratios, ca_positions, strict=False)),
        slot_mapping_store_ca=dict(zip(ratios, ca_stores, strict=False)),
        klen=dict(zip(ratios, lengths, strict=False)),
        state_block_idx=state_idx,
        query_indices=request_ids,
        block_tables=block_tables,
        draft_slots=draft_slots,
        draft_indices=draft_indices,
    )


def _reference(cache, positions, ratios, draft, valid_tokens):
    batch, queries = positions.shape
    valid = (
        torch.ones_like(positions, dtype=torch.bool)
        if valid_tokens is None
        else valid_tokens.reshape_as(positions)
    )
    end = positions[:, :1].expand_as(positions) if draft else positions + 1
    counts = end.clamp(0, cache.geometry.window_size)
    columns = torch.arange(cache.geometry.window_size, device=positions.device)
    history = end[..., None] - counts[..., None] + columns
    read = cache.window_rows(history)
    read = torch.where((columns < counts[..., None]) & valid[..., None], read, -1)
    ca_positions, ca_stores, lengths = {}, {}, {}
    for ratio in ratios:
        complete = (positions + 1).remainder(ratio) == 0
        ca_positions[ratio] = torch.where(
            complete & valid, positions - positions % ratio, 0
        ).int()
        ca_stores[ratio] = torch.where(
            complete & valid, cache._rows(1, positions, ratio), -1
        ).int()
        lengths[ratio] = torch.where(valid, (positions + 1) // ratio, 0).int()
    slots = torch.arange(queries, device=positions.device)
    draft_slots = (
        torch.arange(batch, device=positions.device)[:, None]
        * cache.geometry.draft_tokens
        + slots
    )
    draft_slots = torch.where(valid & draft, draft_slots, -1).int()
    draft_columns = torch.arange(
        (queries + 7) // 8 * 8 if draft else 1, device=positions.device
    )
    private = (
        torch.arange(batch, device=positions.device)[:, None, None]
        * cache.geometry.draft_tokens
        + draft_columns
    )
    private = private.expand(batch, queries, -1)
    valid_keys = valid[:, draft_columns.clamp_max(queries - 1)]
    private = torch.where(
        valid[..., None] & valid_keys[:, None, :] & draft & (draft_columns < queries),
        private,
        -1,
    ).int()
    pages = (
        cache.groups[1].parent_ids * cache.geometry.packings[1]
        + cache.groups[1].child_slots
    )
    pages = torch.where(cache.groups[1].parent_ids > 0, pages, -1).int()
    return ForwardMetadata(
        positions=positions.int(),
        context_lens=(positions + 1).int(),
        valid_token_mask=valid.flatten(),
        slot_mapping_store_swa=torch.where(
            valid, cache.window_rows(positions), -1
        ).int(),
        kv_swa_index=read,
        positions_c=ca_positions,
        slot_mapping_store_ca=ca_stores,
        klen=lengths,
        state_block_idx=cache.compressor_pages(positions).int(),
        query_indices=torch.arange(batch, device=positions.device)
        .repeat_interleave(queries)
        .int(),
        block_tables=pages.repeat_interleave(queries, dim=0),
        draft_slots=draft_slots,
        draft_indices=private,
    )
