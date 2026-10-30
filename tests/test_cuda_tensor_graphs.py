"""Projection graph buffers bridge eager attention without capturing metadata."""

from __future__ import annotations

import pytest
import torch
from test_cuda_prefill_graphs import _runner

from resin import BatchPhase
from resin.backend.cuda.tensor_graphs import CudaGraphTensorRegion, TensorSpec


def _regions(runner, *, pad_up_to=128):
    return runner._configure_prefill_tensor_graphs(
        (
            CudaGraphTensorRegion(
                "project",
                lambda hidden: (hidden * 2, hidden[:, :2] + 1),
                (TensorSpec((4,)),),
                (TensorSpec((4,)), TensorSpec((2,))),
            ),
            CudaGraphTensorRegion(
                "finish",
                lambda hidden, positions: hidden.float() + positions[:, None],
                (TensorSpec((4,)), TensorSpec((), torch.int64)),
                TensorSpec((4,), torch.float32),
            ),
        ),
        max_tokens=8,
        pad_up_to=pad_up_to,
    )


def test_projection_callbacks_stage_live_rows_and_changing_positions(monkeypatch):
    runner = _runner()
    project, finish = _regions(runner)
    graphs = runner._prefill_tensor_graphs["prefill_attention"]
    calls = []

    def replay(mode, shape, inputs):
        assert mode == "prefill_attention"
        step = graphs.build(shape)
        assert step.inputs is inputs
        calls.append(shape)
        step.launch()
        return step.output

    monkeypatch.setattr(runner, "_run_cuda_graph", replay)
    with runner._prefill_graphs():
        for count, start in ((3, 9), (4, 32), (1, 128)):
            hidden = torch.arange(count * 4, dtype=torch.bfloat16).reshape(count, 4)
            positions = start + torch.arange(count)
            qr, kv = project(hidden)
            torch.testing.assert_close(qr, hidden * 2)
            torch.testing.assert_close(kv, hidden[:, :2] + 1)
            output = finish(qr, positions)
            torch.testing.assert_close(output, hidden.float() * 2 + positions[:, None])
            assert output.dtype == torch.float32
    assert calls == [(4, 1), (4, 2), (4, 1), (4, 2), (1, 1), (1, 2)]
    # Buffers are shared by layout, not by layer, and padded positions are safe.
    inputs, output = graphs._buffers(4, 2)
    assert inputs["1"].tolist() == [32, 33, 34, 35]
    assert output.shape == (4, 4)
    stats = runner.cuda_graph_stats["prefill_attention"]
    assert stats["capture_scope"] == "attention_projections"
    assert stats["captured_operations"] == ("project", "finish")


def test_projection_local_buckets_do_not_inherit_expert_padding(monkeypatch):
    runner = _runner()
    project, _ = _regions(runner, pad_up_to=0)
    # Expert replicas agree on their exchange shape; pure projections need
    # only agree within their own TP group and can preserve exact row counts.
    runner._distributed_prefill_bucket = 8
    runner._distributed_prefill_scope = True
    # ty misbinds pytest's decorated helper and drops its reason parameter.
    monkeypatch.setattr(runner, "_run_cuda_graph", lambda *args: pytest.fail("replay"))  # ty: ignore[invalid-argument-type]
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    with runner._prefill_graphs():
        qr, _ = project(hidden)
    torch.testing.assert_close(qr, hidden * 2)
    assert runner.cuda_graph_stats["prefill_attention"]["fallback_counts"] == {
        "padding": 1
    }


def test_coordinated_ffns_and_local_projections_keep_distinct_row_policies(monkeypatch):
    runner = _runner()
    (feed_forward,) = runner._configure_prefill_graphs(
        (lambda hidden, ids: hidden + ids[:, None] + 5,),
        hidden_size=4,
        max_tokens=8,
        token_ids=True,
        pad_up_to=0,
    )
    project, _ = _regions(runner, pad_up_to=0)
    runner._distributed_prefill_scope = True
    runner._distributed_prefill_bucket = 8
    runner._distributed_prefill_region_buckets = {}
    calls = []

    def replay(mode, shape, inputs):
        step = runner._prefill_tensor_graphs[mode].build(shape)
        assert step.inputs is inputs
        calls.append((mode, shape))
        step.launch()
        return step.output

    monkeypatch.setattr(runner, "_run_cuda_graph", replay)
    with runner._prefill_graphs():
        # An idle replica still joins all FFN work. Local projections stay eager
        # for empty rows, and large peer buckets never affect their own rows.
        for count in (3, 4, 0):
            hidden = torch.arange(count * 4, dtype=torch.bfloat16).reshape(count, 4)
            ids = torch.arange(count)
            output = feed_forward(hidden, ids)
            expected = hidden + ids[:, None] + 5
            torch.testing.assert_close(output, expected)
            qr, kv = project(output)
            torch.testing.assert_close(qr, expected * 2)
            torch.testing.assert_close(kv, expected[:, :2] + 1)

    assert calls == [
        ("prefill", (8, 1)),
        ("prefill", (8, 1)),
        ("prefill_attention", (4, 1)),
        ("prefill", (8, 1)),
    ]
    inputs, _ = runner._prefill_tensor_graphs["prefill"]._buffers(8, 1)
    assert all(torch.count_nonzero(value) == 0 for value in inputs.values())


def test_all_projection_modes_must_be_captured_before_prefill():
    runner = _runner()
    runner._configure_prefill_graphs((), hidden_size=4, max_tokens=8)
    _regions(runner)
    with pytest.raises(RuntimeError, match="capture_cuda_graphs"):
        runner._check_cuda_graphs(BatchPhase.PREFILL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_projection_graphs_replay_tuple_outputs_and_new_positions():
    runner = _runner("cuda")
    project, finish = _regions(runner)
    runner.capture_cuda_graphs()
    with runner._prefill_graphs():
        for count, start in ((3, 9), (4, 32), (1, 128), (3, 47)):
            hidden = torch.randn((count, 4), dtype=torch.bfloat16, device="cuda")
            positions = start + torch.arange(count, device="cuda")
            expected = hidden.float() * 2 + positions[:, None]
            qr, kv = project(hidden)
            torch.testing.assert_close(kv, hidden[:, :2] + 1, rtol=0, atol=0)
            torch.testing.assert_close(finish(qr, positions), expected, rtol=0, atol=0)
    stats = runner.cuda_graph_stats["prefill_attention"]
    assert stats["replay_count"] == 8 and not stats["poisoned"]
    runner.close_cuda_graphs()
