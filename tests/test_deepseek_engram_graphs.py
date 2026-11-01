"""Engram callbacks preserve layer weights, request masks, and retained outputs."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch
from test_cuda_prefill_graphs import _runner
from torch import nn

from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner
from resin_models_cuda.deepseek_v41.engram import Engram


def _engram() -> Any:
    module: Any = Engram.__new__(Engram)
    nn.Module.__init__(module)
    module.dim, module.hc_mult, module.eps = 4, 2, 1e-5
    module.wkv = nn.Linear(192, 12, bias=False, dtype=torch.bfloat16)
    module.q_weight = nn.Parameter(torch.ones(2, 4))
    module.k_weight = nn.Parameter(torch.ones(2, 4))
    module.prefill_projection = None
    return module


def test_engram_callbacks_keep_layer_gating_masks_and_retained_results(monkeypatch):
    runner: Any = _runner()
    runner.prefill_chunk_size = 8
    runner._prefill_compute_context = nullcontext
    modules = (_engram(), _engram())
    runner._model = SimpleNamespace(
        layers=[
            SimpleNamespace(layer_id=1, engram=modules[0]),
            SimpleNamespace(layer_id=2, engram=None),
            SimpleNamespace(layer_id=14, engram=modules[1]),
        ]
    )
    names = [tuple(module.state_dict()) for module in modules]
    DeepSeekV41CudaRunner._configure_prefill_engram_graphs(runner)
    assert [tuple(module.state_dict()) for module in modules] == names

    def replay(mode, shape, inputs):
        step = runner._prefill_tensor_graphs[mode].build(shape)
        step.launch()
        return step.output

    monkeypatch.setattr(runner, "_run_cuda_graph", replay)
    hidden = torch.randn(1, 3, 2, 4).bfloat16()
    embedding = torch.randn(1, 3, 6, 32).bfloat16()
    mask = torch.tensor([[True, False, True]])
    saved = []
    with torch.inference_mode(), runner._prefill_graphs():
        for module in modules:
            expected = module(hidden, token_mask=mask, embed_hook=lambda: embedding)
            output = module(
                hidden, token_mask=mask, embed_hook=lambda: embedding, prefill=True
            )
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(output[:, ~mask[0]], hidden[:, ~mask[0]])
            saved.append((output, output.clone()))
    for output, expected in saved:
        torch.testing.assert_close(output, expected)
