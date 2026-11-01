"""Auxiliary prefill projections preserve dtype, phase, and live state rows."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_cuda_prefill_graphs import _runner
from torch import nn

from resin_models_cuda.deepseek_v41.compressor import PreparedCompressor
from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner
from resin_models_cuda.deepseek_v41.indexer import _projection
from resin_models_cuda.deepseek_v41.model import Attention, Transformer


class Projection(nn.Module):
    def __init__(self, inputs, outputs, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(outputs, inputs).bfloat16())
        self.output_dtype = dtype

    def forward(self, values):
        return (values.float() @ self.weight.float().T).to(self.output_dtype)


def _compressor(ratio):
    result: Any = PreparedCompressor.__new__(PreparedCompressor)
    nn.Module.__init__(result)
    result.ratio, result.layer, result.norm = ratio, 0, nn.Identity()
    result.projection = Projection(
        16, 8 * ratio, torch.float32 if ratio == 2 else torch.bfloat16
    )
    result.prefill_projection = None
    return result


def _configured(monkeypatch):
    runner: Any = _runner()
    runner.prefill_chunk_size, runner.speculative_draft_tokens = 1024, 3
    runner._prefill_compute_context = nullcontext
    c1, c2 = _compressor(1), _compressor(2)
    indexer = SimpleNamespace(
        owns_k=True, weights_proj=Projection(16, 4), wk=Projection(8, 4)
    )
    draft = SimpleNamespace(
        layer_id=40,
        main_proj=Projection(48, 16),
        attn=SimpleNamespace(wkv=Projection(16, 8)),
    )
    runner._model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_id=2, attn=SimpleNamespace(compressor=c2, indexer=indexer)
            ),
            SimpleNamespace(layer_id=20, attn=SimpleNamespace(compressor=c1)),
        ],
        mtp=[draft],
    )
    DeepSeekV41CudaRunner._configure_prefill_auxiliary_graphs(runner)
    graphs = runner._prefill_tensor_graphs["prefill_auxiliary"]
    calls = []

    def replay(mode, shape, inputs):
        assert mode == "prefill_auxiliary"
        calls.append((shape, inputs["0"].clone()))
        step = graphs.build(shape)
        step.launch()
        return step.output

    monkeypatch.setattr(runner, "_run_cuda_graph", replay)
    return runner, graphs, calls, c1, c2, indexer, draft


def test_auxiliary_callbacks_bind_model_projections_and_preserve_output_dtypes(
    monkeypatch,
):
    runner, _, _, c1, c2, indexer, draft = _configured(monkeypatch)
    pairs = (
        (c2.prefill_projection, c2.projection),
        (indexer.prefill_weights_proj, indexer.weights_proj),
        (indexer.prefill_wk, indexer.wk),
        (c1.prefill_projection, c1.projection),
        (runner._model.prefill_main_projection, draft.main_proj),
        (draft.attn.prefill_store_projection, draft.attn.wkv),
    )
    with torch.inference_mode(), runner._prefill_graphs():
        for callback, projection in pairs:
            values = torch.randn(1, 3, projection.weight.shape[1]).bfloat16()
            output = callback(values)
            assert output.dtype == projection.output_dtype
            torch.testing.assert_close(output, projection(values), rtol=0, atol=0)


def test_compressor_pools_only_live_fp32_rows_after_projection(monkeypatch):
    import resin_models_cuda.deepseek_v41.compressor as implementation

    runner, _, calls, _, c2, _, _ = _configured(monkeypatch)
    hidden = torch.randn(1, 7, 16).bfloat16()
    positions = torch.arange(127, 134)[None]
    metadata, cache = object(), object()
    observed = []

    def compress(
        scores, values, actual_positions, actual_cache, layer, ratio, *, metadata
    ):
        assert scores.dtype == values.dtype == torch.float32
        assert scores.shape == values.shape == (1, 7, 8)
        assert actual_positions is positions and actual_cache is cache
        observed.append(metadata)
        return (scores + values).bfloat16()

    monkeypatch.setattr(implementation, "compress_projected", compress)
    with torch.inference_mode(), runner._prefill_graphs():
        reference = c2(hidden, positions, cache, metadata=metadata)
        assert not calls
        result = c2(hidden, positions, cache, metadata=metadata, prefill=True)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert observed == [metadata, metadata]


def test_shared_producer_projects_full_rows_before_compaction(monkeypatch):
    import resin_models_cuda.deepseek_v41.indexer as implementation

    runner, _, calls, c1, _, indexer, _ = _configured(monkeypatch)
    attention: Any = Attention.__new__(Attention)
    nn.Module.__init__(attention)
    attention.layer_id, attention.compressor, attention.indexer = 20, c1, indexer
    attention.freqs_cis = torch.empty(0)
    hidden, positions = torch.randn(1, 779, 16).bfloat16(), torch.arange(779)[None]
    metadata, stored = SimpleNamespace(positions=positions), []
    runtime = {"metadata": metadata}

    def store_keys(indexer, latent, cache, freqs, frame, *, prefill):
        assert prefill and frame is metadata and latent.shape == (1, 779, 8)
        stored.append(_projection(indexer, "wk", latent, prefill).clone())

    monkeypatch.setattr(implementation, "store_index_kv", store_keys)
    attention._store_compressed_kv = lambda latent, frame, cache: stored.append(
        latent.clone()
    )
    with torch.inference_mode(), runner._prefill_graphs():
        attention.prepare_shared_kv(hidden, positions, None, runtime)
        expected = c1.projection(hidden)
        torch.testing.assert_close(stored[0], indexer.wk(expected), rtol=0, atol=0)
        torch.testing.assert_close(stored[1], expected, rtol=0, atol=0)
    assert runtime["full_index_kv_stored"] == {20}


@pytest.mark.parametrize("prefill", [False, True])
def test_dspark_commit_keeps_absolute_positions_masks_and_decode_path(
    monkeypatch, prefill
):
    import resin_models_cuda.deepseek_v41.model as implementation

    runner, _, calls, _, _, _, draft = _configured(monkeypatch)
    model: Any = Transformer.__new__(Transformer)
    nn.Module.__init__(model)
    attention: Any = Attention.__new__(Attention)
    nn.Module.__init__(attention)
    attention.wkv, attention.prefill_store_projection = (
        draft.attn.wkv,
        draft.attn.prefill_store_projection,
    )
    attention.layer_id, attention.cos_sin = 40, torch.empty(0)
    model.mtp = [
        SimpleNamespace(
            main_proj=draft.main_proj, main_norm=nn.Identity(), attn=attention
        )
    ]
    model.prefill_main_projection = runner._model.prefill_main_projection
    hidden = torch.randn(1, 7, 48).bfloat16()
    start, valid, stored = (
        torch.tensor([131065]),
        torch.tensor([False] + [True] * 6),
        [],
    )

    def prepare(positions, *, valid_tokens):
        torch.testing.assert_close(positions, torch.arange(131065, 131072)[None])
        assert valid_tokens is valid
        return SimpleNamespace(
            positions=positions, slot_mapping_store_swa=torch.arange(7)
        )

    cache = SimpleNamespace(prepare_metadata=prepare, window_cache=lambda layer: layer)
    monkeypatch.setattr(
        implementation,
        "store_attention_kv",
        lambda attention, values, positions, cache, slots: stored.append(
            (values.clone(), positions.clone(), slots.clone())
        ),
    )
    with torch.inference_mode(), runner._prefill_graphs():
        model.commit_dspark_main_block(
            hidden, start, cache=cache, valid_tokens=valid, prefill=prefill
        )
        expected = draft.attn.wkv(draft.main_proj(hidden))
        torch.testing.assert_close(stored[0][0], expected, rtol=0, atol=0)
    assert stored[0][0].shape == (1, 7, 8)
