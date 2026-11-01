"""DeepSeek projection callbacks retain each layer's tuple outputs and dtypes."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_cuda_prefill_graphs import _runner

from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner


@pytest.mark.parametrize("with_index", [False, True])
def test_attention_callbacks_bind_each_layer_and_preserve_tuple_outputs(
    monkeypatch, with_index
):
    import resin_models_cuda.deepseek_v41.attention as implementation

    runner: Any = _runner()
    runner.prefill_chunk_size = 8
    runner._prefill_compute_context = nullcontext
    layers = []
    for layer_id in (2, 7):
        attention = SimpleNamespace(
            dim=4,
            q_lora_rank=8,
            n_local_heads=1,
            n_local_groups=1,
            head_dim=128,
            _project=lambda hidden, offset=layer_id: (
                hidden.repeat(1, 1, 2) + offset,
                hidden.repeat(1, 1, 32) - offset,
            ),
            project_query=lambda qr, offset=layer_id: (
                qr.repeat(1, 16) + offset,
                qr.repeat(1, 4) if with_index else None,
            ),
        )
        if with_index:
            attention.indexer = SimpleNamespace(local_heads=1, head_dim=32)
        layers.append(SimpleNamespace(layer_id=layer_id, attn=attention))
    runner._model = SimpleNamespace(layers=layers)

    def finish(_attention, output):
        values, scales = output
        return (values[:, 0, :4].float() + scales[:, 0, :1]).bfloat16()

    monkeypatch.setattr(implementation, "finish_output", finish)
    DeepSeekV41CudaRunner._configure_prefill_attention_graphs(runner)

    def replay(mode, shape, inputs):
        step = runner._prefill_tensor_graphs[mode].build(shape)
        step.launch()
        return step.output

    monkeypatch.setattr(runner, "_run_cuda_graph", replay)
    hidden = torch.ones(1, 3, 4, dtype=torch.bfloat16)
    values = torch.ones((3, 1, 128), dtype=torch.float8_e4m3fn)
    scales = torch.full((3, 1, 1), 2, dtype=torch.int32)
    with torch.inference_mode(), runner._prefill_graphs():
        for layer in layers:
            attention = layer.attn
            expected_qr, expected_kv = attention._project(hidden)
            qr, kv = attention.prefill_project(hidden)
            torch.testing.assert_close(qr, expected_qr)
            torch.testing.assert_close(kv, expected_kv)
            query, index = attention.prefill_query(qr)
            expected_query, expected_index = attention.project_query(expected_qr[0])
            torch.testing.assert_close(query, expected_query[None])
            if with_index:
                torch.testing.assert_close(index, expected_index[None])
            else:
                assert index is None
            output = attention.prefill_finish((values, scales), torch.arange(3))
            assert output.dtype == torch.bfloat16
            torch.testing.assert_close(
                output, finish(attention, (values, scales))[None]
            )
