"""Prefill graphs capture stateless regions; request-dependent work stays eager."""

from __future__ import annotations

from contextlib import contextmanager
from typing import NoReturn

import pytest
import torch
from torch.nn import functional as F

from resin import BatchPhase
from resin.backend.cuda.graphs import CudaGraphMode, CudaGraphProgram, CudaGraphRegion
from resin.backend.cuda.runner import CudaRunner


# Empty shape catalogs must never invoke their builder.
def _unreachable(_shape) -> NoReturn:
    raise AssertionError("empty graph catalog must not build")


def _runner(device="cpu", *, prefill="breakable", decode="full", decode_builder=None):
    runner = object.__new__(CudaRunner)
    runner.device = torch.device(device)
    runner.cuda_graph_prefill, runner.cuda_graph_decode = prefill, decode
    runner._prefill_graph_active = False
    runner._prefill_operations = ()
    runner._prefill_tensor_graphs = {}
    runner._cuda_graph_executor = None
    runner._parallel_graph_model = None
    runner._parallel_graph_groups = None
    setattr(runner, "_cuda_failure", lambda: torch.cuda.synchronize(runner.device))
    runner._configure_cuda_graphs(
        {"decode": CudaGraphMode((), decode_builder or (_unreachable), lambda _: {})}
    )
    return runner


def _launch_instead_of_replay(monkeypatch, runner, calls):
    def run(mode, shape, inputs):
        assert mode == "prefill"
        step = runner._prefill_tensor_graphs["prefill"].build(shape)
        assert step.inputs is inputs
        calls.append((shape, {key: value.clone() for key, value in inputs.items()}))
        step.launch()
        return step.output

    monkeypatch.setattr(runner, "_run_cuda_graph", run)


@pytest.mark.parametrize("with_ids", [False, True])
def test_prefill_pads_stateless_regions_and_restores_output_shape(
    monkeypatch, with_ids
):
    runner = _runner()

    def operation(hidden, ids):
        return hidden * 2 + (ids[:, None] if with_ids else 1)

    first, second = runner._configure_prefill_graphs(
        (operation, lambda hidden, ids: hidden + 3),
        hidden_size=4,
        max_tokens=6,
        token_ids=with_ids,
    )
    calls = []
    _launch_instead_of_replay(monkeypatch, runner, calls)
    hidden = torch.arange(12, dtype=torch.bfloat16).view(1, 3, 4)
    ids = torch.tensor([[2, 5, 1]]) if with_ids else None
    expected = operation(hidden.reshape(3, 4), None if ids is None else ids[0])
    with runner._prefill_graphs():
        output = first(hidden, ids)
        assert output.shape == hidden.shape and output.device == hidden.device
        assert (
            output.data_ptr()
            == runner._prefill_tensor_graphs["prefill"]._buffers(4, 1)[1].data_ptr()
        )
        torch.testing.assert_close(output, expected.view_as(hidden))
        consumed = output.float().square()
        torch.testing.assert_close(second(output, ids), (expected + 3).view_as(hidden))
    torch.testing.assert_close(consumed, expected.view_as(hidden).float().square())
    assert [shape for shape, _ in calls] == [(4, 1), (4, 2)]
    assert torch.count_nonzero(calls[0][1]["0"][3:]) == 0
    if with_ids:
        torch.testing.assert_close(calls[0][1]["1"], torch.tensor([2, 5, 1, 0]))
    first_step = runner._prefill_tensor_graphs["prefill"].build((4, 1))
    second_step = runner._prefill_tensor_graphs["prefill"].build((4, 2))
    smaller_step = runner._prefill_tensor_graphs["prefill"].build((2, 1))
    assert first_step.inputs is second_step.inputs
    assert first_step.output is second_step.output
    assert smaller_step.inputs["0"].shape == (2, 4)
    assert (
        runner._prefill_tensor_graphs["prefill"].build((2, 2)).output
        is smaller_step.output
    )
    assert runner._prefill_tensor_graphs["prefill"].buckets == (1, 2, 4, 6)
    assert runner.cuda_graph_stats["prefill"]["capture_scope"] == "feed_forward_regions"


@pytest.mark.parametrize("policy", ["breakable", "disabled"])
def test_decode_and_unsupported_prefill_inputs_keep_the_eager_operation(
    monkeypatch, policy
):
    runner = _runner(prefill=policy)
    observed = []

    def operation(hidden, ids):
        observed.append((hidden.shape, hidden.dtype))
        return hidden + 1

    (invoke,) = runner._configure_prefill_graphs(
        (operation,), hidden_size=4, max_tokens=4
    )
    monkeypatch.setattr(
        runner,
        "_run_cuda_graph",
        # ty misbinds pytest's decorated helper and drops its reason parameter.
        lambda *args: pytest.fail("unexpected replay"),  # ty: ignore[invalid-argument-type]
    )
    decode = torch.ones(1, 3, 4, dtype=torch.bfloat16)
    torch.testing.assert_close(invoke(decode), decode + 1)
    with runner._prefill_graphs():
        for hidden in (torch.ones(5, 4, dtype=torch.bfloat16), torch.ones(3, 4)):
            torch.testing.assert_close(invoke(hidden), hidden + 1)
    assert observed == [
        ((3, 4), torch.bfloat16),
        ((5, 4), torch.bfloat16),
        ((3, 4), torch.float32),
    ]
    assert runner.cuda_graph_stats["prefill"]["fallback_counts"] == (
        {"tokens_or_layout": 2} if policy == "breakable" else {}
    )
    assert not runner._prefill_graph_active


def test_prefill_and_capture_contexts_restore_state_after_operation_failure(
    monkeypatch,
):
    runner = _runner()
    active = []

    @contextmanager
    def capture_context():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def operation(hidden, ids):
        assert active == [True]
        raise RuntimeError("operation failed")

    (invoke,) = runner._configure_prefill_graphs(
        (operation,), hidden_size=4, max_tokens=4, capture_context=capture_context
    )
    _launch_instead_of_replay(monkeypatch, runner, [])
    with pytest.raises(RuntimeError, match="operation failed"):
        with runner._prefill_graphs():
            with runner._prefill_graphs():
                assert runner._prefill_graph_active
            assert runner._prefill_graph_active
            invoke(torch.ones(3, 4, dtype=torch.bfloat16))
    assert not runner._prefill_graph_active and active == []


def test_large_partial_prefill_keeps_actual_rows_while_exact_buckets_replay(
    monkeypatch,
):
    runner = _runner()
    observed_rows = []

    def operation(hidden, ids):
        observed_rows.append(len(hidden))
        return hidden + 1

    (invoke,) = runner._configure_prefill_graphs(
        (operation,), hidden_size=4, max_tokens=256
    )
    calls = []
    _launch_instead_of_replay(monkeypatch, runner, calls)
    with runner._prefill_graphs():
        for tokens in (128, 129, 256):
            hidden = torch.ones(tokens, 4, dtype=torch.bfloat16)
            torch.testing.assert_close(invoke(hidden), hidden + 1)
    assert observed_rows == [128, 129, 256]
    assert [shape for shape, _ in calls] == [(128, 1), (256, 1)]
    assert runner.cuda_graph_stats["prefill"]["fallback_counts"] == {"padding": 1}


def test_exact_bucket_policy_preserves_eager_tensor_identity_and_graph_buffer(
    monkeypatch,
):
    runner = _runner()
    received = []

    def operation(hidden, ids):
        output = hidden + ids[:, None]
        received.append((hidden, ids, output))
        return output

    (invoke,) = runner._configure_prefill_graphs(
        (operation,), hidden_size=4, max_tokens=8, token_ids=True, pad_up_to=0
    )
    calls = []
    _launch_instead_of_replay(monkeypatch, runner, calls)
    with runner._prefill_graphs():
        for tokens in (3, 4, 5, 8):
            hidden = torch.ones(tokens, 4, dtype=torch.bfloat16)
            ids = torch.arange(tokens)
            output = invoke(hidden, ids)
            torch.testing.assert_close(output, hidden + ids[:, None])
            if tokens in (3, 5):
                assert received[-1][0] is hidden and received[-1][1] is ids
                assert output is received[-1][2]
            else:
                assert (
                    output
                    is runner._prefill_tensor_graphs["prefill"]._buffers(tokens, 1)[1]
                )
    output = invoke(hidden, ids)
    assert received[-1][0] is hidden and received[-1][1] is ids
    assert output is received[-1][2]
    assert [shape for shape, _ in calls] == [(4, 1), (8, 1)]
    assert runner.cuda_graph_stats["prefill"]["fallback_counts"] == {"padding": 2}


def test_full_decode_policy_rejects_eager_regions_before_capture():
    program = CudaGraphProgram(
        {},
        torch.zeros(1),
        (
            CudaGraphRegion("compute", lambda: None),
            CudaGraphRegion("dynamic", lambda: None, capture=False),
        ),
    )
    for policy in ("full", "breakable"):
        runner = _runner(decode=policy, decode_builder=lambda _: program)
        build = runner._require_cuda_graph_executor().mode("decode")._build
        if policy == "full":
            with pytest.raises(ValueError, match="cannot contain eager regions"):
                build((1, 1))
        else:
            assert build((1, 1)) is program


@pytest.mark.parametrize("prefill", ["breakable", "disabled"])
def test_prefill_readiness_is_checked_before_model_work(prefill):
    runner = _runner(prefill=prefill)
    runner._configure_prefill_graphs(
        (lambda hidden, _ids: hidden,), hidden_size=4, max_tokens=4
    )
    runner._check_cuda_graphs(BatchPhase.DECODE)
    if prefill == "breakable":
        with pytest.raises(RuntimeError, match="call runner.capture_cuda_graphs"):
            runner._check_cuda_graphs(BatchPhase.PREFILL)
    else:
        runner._check_cuda_graphs(BatchPhase.PREFILL)
    assert not runner.cuda_graph_stats["prefill"]["poisoned"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("with_ids", [False, True])
def test_cuda_prefill_regions_replay_dynamic_lengths_with_shared_memory(with_ids):
    torch.manual_seed(29)
    runner = _runner("cuda")
    weights = [
        torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) / 8 for _ in range(2)
    ]
    embeddings = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)

    def operation(weight):
        def forward(hidden, ids):
            projected = F.layer_norm(hidden, (64,)) @ weight
            if with_ids:
                projected = projected + embeddings[ids]
            return F.gelu(projected)

        return forward

    operations = tuple(map(operation, weights))
    first, second = runner._configure_prefill_graphs(
        operations, hidden_size=64, max_tokens=8, token_ids=with_ids
    )
    capture_stats = runner.capture_cuda_graphs()
    assert capture_stats["prefill"]["capture_scope"] == "feed_forward_regions"
    for tokens in (3, 5, 1, 8, 3):
        hidden = torch.randn(1, tokens, 64, device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(0, 16, (1, tokens), device="cuda") if with_ids else None
        flat_ids = None if ids is None else ids.flatten()
        expected_first = operations[0](hidden[0], flat_ids)
        expected_second = operations[1](expected_first, flat_ids)
        with runner._prefill_graphs():
            intermediate = first(hidden, ids)
            consumed = intermediate.float().square()
            output = second(intermediate, ids)
        assert output.is_cuda and output.shape == hidden.shape
        torch.testing.assert_close(
            consumed[0], expected_first.float().square(), atol=0.03, rtol=0.03
        )
        torch.testing.assert_close(output[0], expected_second, atol=0.03125, rtol=0.02)
    stats = runner.cuda_graph_stats
    assert stats["prefill"]["capture_scope"] == "feed_forward_regions"
    assert (
        stats["prefill"]["capture_count"] == stats["prefill"]["graph_region_count"] == 8
    )
    assert stats["prefill"]["replay_count"] == 10
    assert (
        stats["prefill"]["graph_break_count"] == stats["decode"]["capture_count"] == 0
    )
    assert all(
        row["capture_memory_bytes"] >= 0 for row in stats["prefill"]["shape_stats"]
    )
    cache = runner._require_cuda_graph_executor().mode("prefill")
    assert cache._workspace is not None and cache._workspace.pool is not None
    assert {
        graph.pool() for entry in cache._entries.values() for graph in entry.graphs
    } == {cache._workspace.pool}
    assert (
        runner._require_cuda_graph_executor().mode("decode")._workspace
        is cache._workspace
    )
    runner.close_cuda_graphs()
