from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_inkling_config import _target

from resin.models.inkling.config import InklingConfig
from resin_models_cuda.inkling.attention import (
    PagedAttentionBatch,
    log_scaling_tau,
    paged_attention,
    prepare_paged_attention_batch,
)
from resin_models_cuda.inkling.cache import DRAFT_GROUP, DRAFT_LAYERS
from resin_models_cuda.inkling.dspark import DSparkDraft
from resin_models_cuda.inkling.model import InklingAttention


def _cache():
    groups = tuple(
        SimpleNamespace(
            slab_ids=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
            slotnums=torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int32),
            first_blocks=torch.tensor([index, 0], dtype=torch.int32),
        )
        for index in range(3)
    )
    return SimpleNamespace(
        groups=groups,
        slab_units=70,
        page_tokens=128,
        keys=object(),
        values=object(),
        pages=lambda group, layer, count: (
            groups[group].slab_ids * 70 + groups[group].slotnums * count + layer
        ),
    )


def test_prepared_metadata_matches_each_layer_with_retained_windows(monkeypatch):
    calls = []

    def attention(query, keys, values, **kwargs):
        calls.append(kwargs)
        return query

    monkeypatch.setitem(
        sys.modules,
        "flash_attn.cute",
        SimpleNamespace(flash_attn_varlen_func=attention),
    )
    cache = _cache()
    sequence_lengths = torch.tensor([330, 200], dtype=torch.int32)
    positions = sequence_lengths[:, None] - 3 + torch.arange(3)
    queries = torch.randn(2, 3, 4, 8)
    for counts in ({0: 7, 1: 35}, {DRAFT_GROUP: DRAFT_LAYERS}):
        prepared = prepare_paged_attention_batch(
            cache, sequence_lengths, 330, batch=2, tokens=3, layer_counts=counts
        )
        assert prepared.page_tables.keys() == counts.keys()
        for group, layer_count in counts.items():
            for layer in range(layer_count):

                def invoke(metadata=None):
                    return paged_attention(
                        queries,
                        cache=cache,
                        group_index=group,
                        layer=layer,
                        layer_count=layer_count,
                        positions=positions,
                        sequence_lengths=sequence_lengths,
                        max_sequence_length=330,
                        softmax_scale=0.25,
                        prepared_batch=metadata,
                    )

                reference = invoke()
                actual = invoke(prepared)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                before, after = calls[-2:]
                for key, expected in before.items():
                    if isinstance(expected, torch.Tensor):
                        torch.testing.assert_close(after[key], expected, rtol=0, atol=0)
                    else:
                        assert after[key] == expected


@pytest.mark.parametrize("counts", [{}, {-1: 6}, {3: 6}, {2: 0}])
def test_prepare_rejects_invalid_selected_groups(counts):
    with pytest.raises(ValueError, match="cache-group geometry"):
        prepare_paged_attention_batch(
            _cache(),
            torch.tensor([330, 200]),
            330,
            batch=2,
            tokens=3,
            layer_counts=counts,
        )


def test_dspark_layers_share_one_draft_metadata_batch(monkeypatch):
    seen = []

    class Layer(torch.nn.Module):
        def forward(self, hidden, _positions, _cache, _lengths, _maximum, metadata):
            seen.append(metadata)
            return hidden

    draft = object.__new__(DSparkDraft)
    torch.nn.Module.__init__(draft)
    monkeypatch.setattr(
        draft,
        "target",
        SimpleNamespace(
            embed_tokens=torch.nn.Embedding(16, 8), unembed=torch.nn.Linear(8, 16)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        draft,
        "target_config",
        SimpleNamespace(logits_mup_width_multiplier=1, unpadded_vocab_size=16),
        raising=False,
    )
    monkeypatch.setattr(draft, "norm", torch.nn.Identity(), raising=False)
    draft.layers = torch.nn.ModuleList(Layer() for _ in range(DRAFT_LAYERS))
    logits, hidden = draft(
        torch.ones(2, 5, dtype=torch.long),
        torch.arange(5).expand(2, -1),
        _cache(),
        torch.tensor([330, 200]),
        330,
    )
    assert logits.shape == (2, 5, 16)
    assert hidden.shape == (2, 5, 8)
    assert len(seen) == DRAFT_LAYERS
    assert all(metadata is seen[0] for metadata in seen)
    assert set(seen[0].page_tables) == {DRAFT_GROUP}


@pytest.mark.parametrize("maximum", [128000, 128001, 130002])
def test_global_attention_reuses_or_skips_the_target_log_scale(monkeypatch, maximum):
    from resin_models_cuda.inkling import model

    config = replace(
        InklingConfig.from_hugging_face(_target()),
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        d_rel=4,
        rel_extent=8,
    )
    attention = InklingAttention(config, config.global_layer_ids[0])
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.uniform_(-0.25, 0.25)
    hidden = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    positions = torch.tensor([[maximum - 3, maximum - 2, maximum - 1], [10, 11, 12]])
    state = torch.zeros(2, 3, 8, dtype=torch.bfloat16)
    metadata = PagedAttentionBatch({}, {}, {}, torch.arange(3))
    cache = _cache()
    cache.keys = torch.empty(1, positions.numel(), 2, 4, dtype=hidden.dtype)
    cache.values = torch.empty_like(cache.keys)
    cache.page_locations = lambda *, positions, **_kwargs: (
        torch.zeros_like(positions),
        torch.arange(positions.numel()).reshape_as(positions),
    )
    monkeypatch.setattr(model, "paged_attention", lambda queries, **_kwargs: queries)

    def invoke(batch, maximum_length=maximum):
        return attention(
            hidden,
            positions,
            positions[:, -1] + 1,
            maximum_length,
            cache,
            batch,
            state,
            state,
            prefill=True,
        )

    # Force the old scaling path even when every actual position is below the
    # floor, so the skip is checked against its original multiply-by-one math.
    expected = invoke(metadata, max(maximum, config.log_scaling_n_floor + 1))
    prepared = replace(
        metadata,
        log_tau=(
            log_scaling_tau(
                positions,
                n_floor=config.log_scaling_n_floor,
                alpha=config.log_scaling_alpha,
            )
            if maximum > config.log_scaling_n_floor
            else None
        ),
    )

    def redundant_scale(*_args, **_kwargs):
        raise AssertionError("global layers must reuse the prepared log scale")

    monkeypatch.setattr(model, "log_scaling_tau", redundant_scale)
    for actual, reference in zip(invoke(prepared), expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
