"""Consumer compaction preserves full compressed history and bounded SWA writes."""

from types import SimpleNamespace

import pytest
import torch
from test_deepseek_v41_batch import cache_and_positions, physical

from resin_models_cuda.deepseek_v41.yoco import compact_metadata, compact_runtime


def test_compact_metadata_preserves_page_ownership_masks_and_placeholder_history():
    cache, positions = cache_and_positions()
    original = cache.prepare_metadata(positions)
    original._selected[(1, 2)] = (torch.empty(0), torch.empty(0))
    original_stores = original.slot_mapping_store_swa.clone()
    chosen = torch.tensor([2, 3, 4, -1])
    compact = compact_metadata(original, chosen, swa_start=128)
    assert compact.positions.tolist() == [
        row[[2, 3, 4, 0]].tolist() for row in positions
    ]
    assert compact.valid_token_mask.reshape(3, 4).tolist() == [[True] * 3 + [False]] * 3
    assert compact._selected == {}
    assert original._selected
    assert torch.equal(original.slot_mapping_store_swa, original_stores)
    for row in range(3):
        for query, source in enumerate((2, 3, 4, 0)):
            torch.testing.assert_close(
                compact.block_tables[row * 4 + query],
                original.block_tables[row * 5 + source],
            )
            pos = int(positions[row, source])
            valid = query != 3
            writable = valid and pos >= 128
            assert compact.slot_mapping_store_swa[row, query] == (
                physical(cache, 0, row, pos) if writable else -1
            )
            assert compact.state_block_idx[row, query] == (
                original.state_block_idx[row, source] if writable else -1
            )
            for ratio in (1, 2):
                assert compact.klen[ratio][row, query] == (
                    (pos + 1) // ratio if valid else 0
                )
                assert compact.slot_mapping_store_ca[ratio][row, query] == (
                    original.slot_mapping_store_ca[ratio][row, source]
                    if writable
                    else -1
                )
            start = max(0, pos + 1 - 128)
            expected = [
                physical(cache, 0, row, key) if valid and key >= 128 else -1
                for key in range(start, pos + 1)
            ]
            expected += [-1] * (128 - len(expected))
            assert compact.kv_swa_index[row, query].tolist() == expected
    # The early real placeholders retain causal compressed reads despite having
    # no consumer SWA rectangle yet. Only CP padding gets zero compressed length.
    assert compact.klen[1][0].tolist() == [3, 4, 5, 0]
    assert (compact.slot_mapping_store_swa[0] == -1).all()
    selected = torch.zeros(3, 4, 2, dtype=torch.int32)
    selected[..., 1] = compact.klen[2] - 1
    mapped = compact.compressed_indices(20, 2, selected)
    for row in range(3):
        for query in range(4):
            for slot in range(2):
                logical = int(selected[row, query, slot])
                assert mapped[row, query, slot] == (
                    physical(cache, 1, row, logical * 2, 2)
                    if 0 <= logical < compact.klen[2][row, query]
                    else -1
                )


def test_runtime_compaction_rebuilds_query_caches_and_slices_candidate_rows():
    cache, positions = cache_and_positions()
    frame = cache.prepare_metadata(positions)
    indices = torch.arange(3 * 5 * 2).reshape(3, 5, 2).int()
    candidates = indices.flatten(0, 1) + 100
    runtime = dict(
        metadata=frame,
        indices=indices,
        candidate_blocks=candidates,
        index_metadata=object(),
        prefill_layout=object(),
        prefill_compressed=object(),
        full_index_kv_stored={20},
    )
    chosen = torch.tensor([3, 4, -1])
    result = compact_runtime(runtime, chosen, swa_start=128)
    assert result["full_index_kv_stored"] == {20}
    assert result["swa_start"] == 128
    assert result["metadata"] is not frame
    assert torch.equal(result["indices"][:, :2], indices[:, 3:])
    assert (result["indices"][:, 2] == -1).all()
    candidate_rows = result["candidate_blocks"].reshape(3, 3, 2)
    assert torch.equal(candidate_rows[:, :2], candidates.reshape(3, 5, 2)[:, 3:])
    assert (candidate_rows[:, 2] == -1).all()
    for name in ("index_metadata", "prefill_layout", "prefill_compressed"):
        assert name not in result and name in runtime


@pytest.mark.parametrize("lower", [128, 140])
def test_prefill_gather_clamps_consumer_swa_but_keeps_complete_c1(monkeypatch, lower):
    from resin_models_cuda.deepseek_v41 import attention

    cache, positions = cache_and_positions()
    frame = cache.row(1).prepare_metadata(positions[1:2])
    runtime = compact_runtime(
        dict(metadata=frame, indices=torch.zeros(1, 5, 2, dtype=torch.int32)),
        torch.tensor([2, 3, 4, -1]),
        swa_start=lower,
    )
    reads = []

    def gather(_cache, indices):
        reads.append(indices.clone())
        return indices.flatten().float()[:, None].expand(-1, 512)

    monkeypatch.setattr(attention, "gather_kv", gather)
    fake_cache = SimpleNamespace(
        window_rows=lambda history: history + 1000,
        window_cache=lambda _layer: None,
        compressed_rows=lambda _layer, _ratio, history: history + 2000,
        compressed_cache=lambda _layer: None,
        geometry=SimpleNamespace(compressed_offsets={20: 0}),
    )
    values, indices = attention.prefill_inputs(
        SimpleNamespace(layer_id=20, window_size=128, ratio=1),
        fake_cache,
        runtime["metadata"],
        131,
        runtime,
    )
    start = min(lower, 131)
    assert reads[0].tolist() == [list(range(1000 + start, 1131))]
    assert reads[1].tolist() == [list(range(2000, 2131))]
    assert values.shape[0] == 131 + max(0, 131 - lower)
    assert values[max(0, 131 - lower), 0] == 2000  # Oldest C1 key is retained.
    assert (indices[0, -1, :128] == -1).all()  # CP pad has no SWA reads.
    for row, pos in enumerate(runtime["metadata"].positions[0, :3].tolist()):
        actual = indices[0, row, :128]
        actual = actual[actual >= 0].tolist()
        assert actual == [
            key - lower for key in range(max(lower, pos + 1 - 128), pos + 1)
        ]
    # Full-row masking and actual compaction expose identical keys to retained
    # queries, including the complete compressed history in either path.
    full = compact_runtime(
        dict(metadata=frame, indices=torch.zeros(1, 5, 2, dtype=torch.int32)),
        torch.arange(5),
        swa_start=lower,
    )
    full_values, full_indices = attention.prefill_inputs(
        SimpleNamespace(layer_id=20, window_size=128, ratio=1),
        fake_cache,
        full["metadata"],
        131,
        full,
    )
    assert torch.equal(full_values, values)
    assert torch.equal(full_indices[:, 2:5], indices[:, :3])


def test_full_index_source_store_uses_full_metadata_before_query_compaction(
    monkeypatch,
):
    import sys

    from resin_models_cuda.deepseek_v41.indexer import store_index_kv

    cache, positions = cache_and_positions()
    frame = cache.prepare_metadata(positions)
    observed = []
    monkeypatch.setitem(
        sys.modules,
        "resin_models_cuda.deepseek_v41.upstream.fused_transform",
        SimpleNamespace(
            CastArgs=lambda *args, **kwargs: (args, kwargs),
            NormArgs=lambda *args: args,
            RopeArgs=lambda *args, **kwargs: (args, kwargs),
            KVCacheOutArgs=lambda *args: args,
            fused_norm_rope_hadamard_cast_store_kvcache=lambda *args, **kwargs: (
                observed.append((args, kwargs))
            ),
        ),
    )
    indexer = SimpleNamespace(
        owns_k=True,
        layer=20,
        ratio=1,
        head_dim=4,
        rope_dim=2,
        wk=lambda value: value + 2,
        k_norm=SimpleNamespace(weight=torch.ones(4), eps=1e-6),
    )
    target = object()
    latent = torch.arange(60).reshape(3, 5, 4).float()
    store_index_kv(
        indexer,
        latent,
        SimpleNamespace(index_cache=lambda _: target),
        torch.ones(512, 1, dtype=torch.complex64),
        frame,
    )
    args, kwargs = observed[0]
    assert torch.equal(args[0], (latent + 2).flatten(0, 1))
    assert kwargs["out_args"][0] is target
    assert torch.equal(kwargs["out_args"][1], frame.slot_mapping_store_ca[1].flatten())


@pytest.mark.parametrize("prefill", [False, True])
def test_compact_index_query_does_not_overwrite_pre_stored_full_source(
    monkeypatch, prefill
):
    import sys

    from resin_models_cuda.deepseek_v41 import indexer as operations

    cache, _ = cache_and_positions()
    positions = torch.tensor([[0, 1]])
    metadata = cache.row(0).prepare_metadata(positions)
    writes = []
    monkeypatch.setattr(
        operations,
        "store_index_kv",
        lambda *args, prefill: writes.append((args, prefill)),
    )
    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(
            get_num_sms=lambda: 148,
            get_paged_mqa_logits_metadata=lambda *args, **kwargs: None,
            fp8_fp4_paged_mqa_logits=lambda q, kv, weights, lengths, table, schedule, width, **kwargs: (
                torch.ones(len(lengths), width)
            ),
        ),
    )
    selection = SimpleNamespace(
        topk=lambda scores, count, **kwargs: (
            None,
            torch.zeros(len(scores), count, dtype=torch.int32),
        )
    )
    monkeypatch.setitem(
        sys.modules, "deep_select", SimpleNamespace(interface=selection)
    )
    monkeypatch.setitem(sys.modules, "deep_select.interface", selection)
    monkeypatch.setitem(
        sys.modules,
        "resin_models_cuda.deepseek_v41.upstream.candidate_blocks",
        SimpleNamespace(
            candidate_block_max=None,
            get_candidate_compact_lens=None,
            restore_candidate_indices_inplace=None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "resin_models_cuda.deepseek_v41.upstream.fused_transform",
        SimpleNamespace(
            CastArgs=lambda *args, **kwargs: None,
            RopeArgs=lambda *args, **kwargs: None,
            fused_norm_rope_hadamard_cast_store_kvcache=lambda value, **kwargs: (
                value,
                torch.ones(len(value), 1),
            ),
        ),
    )
    indexer = SimpleNamespace(
        owns_k=True,
        layer=20,
        ratio=1,
        head_dim=4,
        local_heads=1,
        n_heads=1,
        rope_dim=2,
        topk=1,
        max_kv_len=8,
        candidate_source=-1,
        weights_proj=lambda hidden: torch.ones(*hidden.shape[:2], 1),
    )
    fake_cache = SimpleNamespace(index_cache=lambda _: torch.zeros(1, 128, 1, 68))
    hidden, query = torch.zeros(1, 2, 1), torch.zeros(1, 2, 1, 4)
    freqs = torch.ones(8, 1, dtype=torch.complex64)
    for stored in (False, True):
        runtime = dict(metadata=metadata)
        if stored:
            runtime["full_index_kv_stored"] = {20}
        output = operations.paged_index(
            indexer,
            hidden,
            query,
            None,
            positions,
            2,
            fake_cache,
            freqs,
            runtime,
            prefill=prefill,
        )
        assert output.shape == (1, 2, 1)
    assert len(writes) == 1
    assert writes[0][1] == prefill
