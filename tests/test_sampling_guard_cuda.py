"""Conservative top-k mass guard checked against independent sampling policy."""

import pytest
import torch
from test_sampling_filters_cuda import _assert_reference, _inputs

from resin.backend.cuda.sampling import FusedSampler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _prepare(logits, inputs):
    from resin.backend.cuda.filtered_sampling import prepare_filters

    return prepare_filters(
        logits,
        inputs.temperatures,
        inputs.top_ks,
        inputs.top_ps,
        inputs.stochastic_mask,
    )


def test_guard_skips_only_distant_redundant_nuclei():
    logits = torch.full((7, 2051), -torch.inf, device="cuda")
    logits[:, :3] = torch.tensor([0.60, 0.25, 0.15], device="cuda").log()
    logits[4, :4] = torch.tensor([0.30, 0.30, 0.30, 0.10], device="cuda").log()
    inputs = _inputs(logits)
    inputs.temperatures.fill_(1)
    inputs.top_ks.copy_(torch.tensor([1, 2, 1, 1, 1, 2051, 1], device="cuda"))
    inputs.top_ps.copy_(
        torch.tensor([0.9, 0.65, 0.6005, 0.5995, 0.8, 0.65, 0.9], device="cuda")
    )
    inputs.stochastic_mask.fill_(True)
    inputs.stochastic_mask[-1] = False
    state = _prepare(logits, inputs)
    # Near the .001 margin, and when inclusive K ties extend beyond P, the
    # nucleus must still run. P-only and disabled rows cannot read K scratch.
    assert (state[:, 1] == 0).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    _assert_reference(logits, inputs, FusedSampler().sample_tensors(logits, inputs))


def test_guard_preserves_small_p_and_its_conservative_margin():
    logits = torch.linspace(1.0, 0.5, 4096, device="cuda").log().repeat(4, 1)
    inputs = _inputs(logits)
    inputs.temperatures.fill_(1)
    inputs.top_ks.fill_(1)
    inputs.top_ps.copy_(torch.tensor([0.0009, 0.001, 0.0011, 0.002], device="cuda"))
    inputs.stochastic_mask.fill_(True)
    state = _prepare(logits, inputs)
    assert (state[:, 1] == 0).tolist() == [False, False, False, True]
    _assert_reference(logits, inputs, FusedSampler().sample_tensors(logits, inputs))


def test_fp32_guard_partials_preserve_large_k_margin():
    logits = torch.full((4, 8193), 1 / 32, device="cuda")
    logits[:, :4096] = torch.linspace(1.0, 0.5, 4096, device="cuda")
    logits.log_()
    # Derive the boundary independently from the actual stored FP32 logits.
    weights = logits[0].cpu().double().exp()
    ratio = float(weights[:4096].sum() / weights.sum())
    inputs = _inputs(logits)
    inputs.temperatures.fill_(1)
    inputs.top_ks.fill_(4096)
    inputs.top_ps.copy_(
        torch.tensor(
            [ratio + delta for delta in (-0.0015, -0.0005, 0.0005, 0.0015)],
            device="cuda",
        )
    )
    inputs.stochastic_mask.fill_(True)
    state = _prepare(logits, inputs)
    assert (state[:, 1] == 0).tolist() == [False, False, False, True]
    _assert_reference(logits, inputs, FusedSampler().sample_tensors(logits, inputs))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_guarded_preparation_retains_independent_sampling_support(dtype):
    logits = torch.randn(8, 2051, generator=torch.Generator().manual_seed(875)).to(
        "cuda", dtype=dtype
    )
    inputs = _inputs(logits)
    inputs.temperatures.fill_(1)
    inputs.top_ks.fill_(40)
    inputs.top_ps.fill_(0.9)
    inputs.stochastic_mask.fill_(True)
    state = _prepare(logits, inputs)
    assert (state[:, 1] == 0).all()
    _assert_reference(logits, inputs, FusedSampler().sample_tensors(logits, inputs))


def test_rounded_weight_ties_preserve_raw_top_k():
    logits = torch.full((2, 2051), -torch.inf, device="cuda")
    logits[:, :3] = torch.tensor([-1.0, 0.0, 1.0], device="cuda")
    inputs = _inputs(logits)
    inputs.temperatures.fill_(1e20)
    inputs.top_ks.fill_(1)
    inputs.top_ps.copy_(torch.tensor([0.1, 0.5], device="cuda"))
    inputs.stochastic_mask.fill_(True)
    state = _prepare(logits, inputs)
    assert (state[:, 0] == 1).all()
    assert (state[:, 1] == 0).tolist() == [False, True]
    actual = FusedSampler().sample_tensors(logits, inputs)
    _assert_reference(logits, inputs, actual)
    assert actual.tolist() == [2, 2]


def test_one_graph_replays_guard_boundary_and_disabled_rows():
    logits = torch.full((4, 2051), -torch.inf, device="cuda")
    logits[:, :3] = torch.tensor([0.60, 0.25, 0.15], device="cuda").log()
    inputs = _inputs(logits)
    inputs.temperatures.fill_(1)
    inputs.top_ks.fill_(1)
    inputs.top_ps.fill_(0.9)
    inputs.stochastic_mask.fill_(True)
    sampler = FusedSampler()
    _prepare(logits, inputs)
    sampler.sample_tensors(logits, inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state = _prepare(logits, inputs)
        output = sampler.sample_tensors(logits, inputs)
    for k, p, temperature, stochastic, skipped in (
        (1, 0.9, 1, True, True),
        (2, 0.65, 1, True, False),
        (2, 0.8505, 1, True, False),  # Only .0005 beyond K mass.
        (2051, 0.9, 1, True, False),
        (1, 1.0, 1, True, True),
        (1, 0.9, 0, True, True),
        (1, 0.9, 1, False, True),
        (1, 0.9, 1, True, True),
    ):
        inputs.top_ks.fill_(k)
        inputs.top_ps.fill_(p)
        inputs.temperatures.fill_(temperature)
        inputs.stochastic_mask.fill_(stochastic)
        inputs.offsets.add_(32)
        graph.replay()
        assert bool((state[:, 1] == 0).all()) == skipped
        _assert_reference(logits, inputs, output)
