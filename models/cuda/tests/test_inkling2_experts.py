"""Representative expert arithmetic, routing, ownership and graph regressions."""

from types import SimpleNamespace

import pytest
import torch
from _inkling2_expert_reference import expert_reference, make_experts, nvfp4_reference

from resin import Parallelism
from resin.backend.topology import RankGroup
from resin_models_cuda.inkling2.experts import Inkling2Experts
from resin_models_cuda.inkling2.model import SparseMlp

BLACKWELL = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires Blackwell packed expert arithmetic",
)


def config():
    return SimpleNamespace(
        hidden_size=16,
        intermediate_size=16,
        num_routed_experts=8,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        experts_per_token=8,
        route_scale=1.0,
    )


def test_router_bound_is_local_to_distinct_top_k(monkeypatch):
    mlp = SparseMlp(config(), precision=None, parallel_groups=None)
    with torch.no_grad():
        mlp.gate.weight.zero_()
        mlp.gate_bias.zero_()
    observed = []

    def local(hidden, ids, weights, *, max_rows_per_expert=None):
        observed.append((ids.clone(), max_rows_per_expert))
        return hidden.float()

    monkeypatch.setattr(mlp.experts, "_local", local)
    for rows in (1, 33):
        mlp(torch.randn(1, rows, 16).bfloat16())
        ids, bound = observed[-1]
        assert bound == rows
        assert all(row.unique().numel() == 8 for row in ids)
        assert int(torch.bincount(ids.flatten()).max()) == rows

    hidden, ids, weights = (
        torch.zeros(8, 16),
        torch.zeros(8, 8).long(),
        torch.ones(8, 8),
    )
    mlp.experts(hidden, ids, weights)
    assert observed[-1][1] is None  # Generic repeated IDs have no top-k proof.
    monkeypatch.setattr(
        mlp.experts, "dispatch", lambda *args: mlp.experts._local(*args)
    )
    mlp.experts(hidden, ids, weights, max_rows_per_expert=8)
    assert observed[-1][1] is None  # Transport cannot retain a source-local bound.


def test_parallel_ownership_and_replicated_attention_bypass():
    for topology, bypass in (
        (Parallelism(world_size=2, attn_tp=2, ep=2), True),
        (Parallelism(world_size=2, attn_dp=2, ep=2), False),
    ):
        for rank in range(2):
            groups = {
                name: RankGroup(ranks, object(), ranks.index(rank))
                for name, partitions in topology.partitions().items()
                for ranks in partitions
                if rank in ranks
            }
            module = Inkling2Experts(
                config(),
                parallel_groups=SimpleNamespace(
                    **groups, device_domain=groups["domain"]
                ),
            )
            assert (module.expert_start, module.expert_count) == (rank * 4, 4)
            assert (module.replicated_ep is not None) == bypass
            assert (module.dispatch is None) == bypass


@BLACKWELL
@pytest.mark.parametrize("mode,width", [("nvfp4", 256), ("mxfp8", 2304)])
def test_native_quantization_matches_independent_reference(mode, width):
    from resin_models_cuda.inkling2.expert_quantization import quantize

    torch.manual_seed(9 if mode == "nvfp4" else 21)
    x = torch.randn(17, width, device="cuda", dtype=torch.bfloat16)
    x[0, : 16 if mode == "nvfp4" else 32] = 0
    if mode == "nvfp4":
        x[-1, -1] = 30
    actual = quantize(x, mode)
    if mode == "nvfp4":
        expected = nvfp4_reference(x)
    else:
        blocks = x.float().reshape(17, -1, 32)
        exponent = torch.ceil(torch.log2(blocks.abs().amax(-1) * (1.0 / 448.0))).clamp(
            -127, 127
        )
        scales = (exponent + 127).to(torch.uint8)
        values = (
            (blocks * torch.exp2(-exponent)[..., None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .reshape_as(x)
        )
        expected = values, scales, None
    for value, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert value is None
        else:
            torch.testing.assert_close(value.float(), reference.float(), atol=0, rtol=0)


@BLACKWELL
@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
def test_experts_match_reference_and_changed_input_graph(mode):
    torch.manual_seed(8)
    module = make_experts(mode)
    hidden = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 4, (17, 3), device="cuda")
    weights = torch.rand(17, 3, device="cuda") * 2
    actual = module(hidden, ids, weights)
    assert actual.dtype == torch.float32
    # Preserve the original bound for approximate tanh/sigmoid before BF16 rounding.
    torch.testing.assert_close(
        actual, expert_reference(module, hidden, ids, weights), atol=2**-8, rtol=0.015
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(hidden, ids, weights)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = module(hidden, ids, weights)
    hidden.normal_()
    ids.fill_(2)
    weights.uniform_(0.1, 2)
    graph.replay()
    torch.testing.assert_close(replayed, module(hidden, ids, weights), atol=0, rtol=0)
    assert module(hidden[:0], ids[:0], weights[:0]).shape == (0, 256)


@BLACKWELL
@torch.inference_mode()
def test_native_adapter_matches_generic_for_changed_routes(monkeypatch):
    from resin_models_cuda.inkling2 import expert_kernels, expert_native

    monkeypatch.setattr(expert_native, "supports", lambda *args: False)
    torch.manual_seed(75)
    module = make_experts("nvfp4")
    hidden = torch.randn(8, 256, device="cuda").bfloat16()
    ids = torch.randint(8, (8, 8), device="cuda")
    weights = torch.rand(8, 8, device="cuda")
    expected = expert_kernels.local_experts(module, hidden, ids, weights)
    actual = expert_native.local_experts(module, hidden, ids, weights)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for _ in range(3):
        expert_native.local_experts(module, hidden, ids, weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = expert_native.local_experts(module, hidden, ids, weights)
    for expert in (2, 7):  # Owned repeats, then an entirely unowned batch.
        hidden.normal_()
        weights.uniform_()
        ids.fill_(expert)
        graph.replay()
        expected = expert_kernels.local_experts(module, hidden, ids, weights)
        torch.testing.assert_close(replayed, expected, atol=0, rtol=0)
