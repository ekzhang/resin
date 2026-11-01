"""Filtered fused sampling against independent sorting and Philox oracles."""

from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from test_sampling_fused_cuda import _philox

from resin.backend.cuda.sampling import FusedSampler, SamplingTensors

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _inputs(logits):
    rows, vocab = logits.shape
    row = torch.arange(rows, device="cuda")
    return SamplingTensors(
        temperatures=0.5 + (row % 4).float() * 0.5,
        top_ks=torch.where(row % 3 == 0, vocab, 7 + row % 5).int(),
        top_ps=torch.where(row % 3 == 1, 1.0, 0.71).float(),
        seeds=(1 << 62) + 17 + row * (1 << 32),
        offsets=(1 << 40) + 32 + row * 32,
        stochastic_mask=row % 5 != 4,
    )


def _reference(logits, inputs, *, return_details=False):
    values = logits.cpu().double().numpy()
    state = {field.name: getattr(inputs, field.name).cpu() for field in fields(inputs)}
    result = logits.argmax(-1).cpu()
    details = [None] * len(values)
    for row, raw in enumerate(values):
        temp = float(state["temperatures"][row])
        if (
            not state["stochastic_mask"][row]
            or not 0 < temp < np.inf
            or np.isnan(raw).any()
            or np.isposinf(raw).any()
            or not np.isfinite(raw).any()
        ):
            continue
        k = min(int(state["top_ks"][row]), len(raw))
        keep = np.isfinite(raw) & (raw >= np.partition(raw, -k)[-k])
        scaled = (raw - raw.max()) / temp
        probability = np.exp(scaled)
        probability /= probability.sum()
        keep &= probability > 0
        p = float(state["top_ps"][row])
        if p < 1:
            descending = np.sort(probability)[::-1]
            crossing = min(np.searchsorted(np.cumsum(descending), p), len(raw) - 1)
            keep &= probability >= descending[crossing]
        random = _philox(
            int(state["seeds"][row]) & ((1 << 64) - 1),
            int(state["offsets"][row]) & ((1 << 64) - 1),
            np.zeros(1, dtype=np.uint64),
        )[0][0]
        uniform = (float(random) + 0.5) * 2**-32
        cumulative = np.cumsum(np.where(keep, probability, 0.0))
        total = cumulative[-1]
        target = uniform * total
        selected = int(np.searchsorted(cumulative, target, side="right"))
        result[row] = selected
        before = cumulative[selected - 1] if selected else 0.0
        margin = min(target - before, cumulative[selected] - target) / total
        if return_details:
            details[row] = (keep, cumulative, target, total, margin)
    return (result, details) if return_details else result


def _assert_reference(logits, inputs, actual):
    expected, details = _reference(logits, inputs, return_details=True)
    tokens = actual.cpu().tolist()
    for row, (token, detail) in enumerate(zip(tokens, details, strict=True)):
        assert 0 <= token < logits.shape[-1], (row, token)
        if detail is None:
            assert token == int(expected[row]), (row, token, int(expected[row]))
            continue
        keep, cumulative, target, total, margin = detail
        # Support is checked independently against the full sorted nucleus,
        # including its crossing token. Do not compare FP64 exp values to a
        # rounded device cutoff: that can incorrectly exclude the cutoff token.
        assert keep[token], (row, token)
        # A measured FP32 tiled-CDF error of 1.54e-7 motivates this narrow
        # draw-boundary band. Policy boundaries and support get no tolerance.
        if margin > 2e-7:
            assert token == int(expected[row]), (row, token, int(expected[row]), margin)
        else:
            before = cumulative[token - 1] if token else 0.0
            assert before <= target + 2e-7 * total
            assert cumulative[token] > target - 2e-7 * total


@pytest.mark.parametrize("vocab", [3, 2051])
def test_nucleus_uses_full_distribution_before_top_k(vocab):
    logits = torch.full((64, vocab), -torch.inf, device="cuda")
    logits[:, :3] = torch.tensor([0.60, 0.25, 0.15], device="cuda").log()
    state = _inputs(logits)
    state.temperatures.fill_(1)
    state.top_ks.fill_(2)
    state.top_ps.fill_(0.65)
    state.stochastic_mask.fill_(True)
    actual = FusedSampler().sample_tensors(logits, state)
    _assert_reference(logits, state, actual)
    # Full-distribution nucleus selection retains 0.60 and 0.25. Normalizing
    # after top-k first would turn 0.60 into 0.60/0.85 and discard token one.
    assert sorted(actual.unique().tolist()) == [0, 1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab", [257, 2051, 154880, 201088])
def test_filtered_draw_matches_sorting_and_philox(dtype, vocab):
    generator = torch.Generator().manual_seed(892)
    logits = torch.randn(8, vocab, generator=generator).to("cuda", dtype=dtype)
    inputs = _inputs(logits)
    actual = FusedSampler().sample_tensors(logits, inputs)
    _assert_reference(logits, inputs, actual)


@pytest.mark.parametrize("vocab", [1, 2, 31, 511, 512, 513, 2047, 2048])
def test_small_vocabularies_share_filtered_sampling(vocab):
    logits = (torch.arange(vocab, device="cuda") % 11).float().mul_(0.25).repeat(4, 1)
    logits[:2, [0, vocab - 1]] = 4  # Inclusive ties at both vocabulary ends.
    logits[2, -1] = 5  # Native top-k must find a unique winner in the tail.
    inputs = _inputs(logits)
    inputs.temperatures.fill_(1)
    inputs.temperatures[3] = float("nan")
    inputs.top_ks.copy_(torch.tensor([vocab, 1, 1, 1], device="cuda"))
    inputs.top_ps.copy_(torch.tensor([0.63, 0.79, 1.0, 0.8], device="cuda"))
    inputs.stochastic_mask.fill_(True)
    sampler = FusedSampler()
    actual = sampler.sample_tensors(logits, inputs)
    _assert_reference(logits, inputs, actual)
    assert actual[2].item() == vocab - 1

    if vocab == 513:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sampler.sample_tensors(logits, inputs)
        inputs.top_ks.copy_(torch.tensor([1, vocab, vocab, vocab], device="cuda"))
        inputs.top_ps.copy_(torch.tensor([1.0, 0.41, 0.72, 1.0], device="cuda"))
        inputs.temperatures.fill_(1)
        inputs.stochastic_mask[0] = False
        inputs.offsets.add_(32)
        graph.replay()
        _assert_reference(logits, inputs, actual)


@pytest.mark.parametrize("top_k,top_p", [(3, 1.0), (2051, 0.25), (3, 0.25)])
def test_cutoff_ties_across_tiles_are_all_retained(top_k, top_p):
    logits = torch.zeros(64, 2051, device="cuda")
    logits[:, [0, 32, 1023, 1024, 2048, 2050]] = 4
    inputs = _inputs(logits)
    inputs.top_ks.fill_(top_k)
    inputs.top_ps.fill_(top_p)
    inputs.stochastic_mask.fill_(True)
    actual = FusedSampler().sample_tensors(logits, inputs)
    _assert_reference(logits, inputs, actual)
    # The cutoff can retain more than k tokens. Include ties in later tiles.
    assert (actual > 1024).any()


def test_one_graph_replays_unfiltered_filtered_and_greedy_policies(monkeypatch):
    from resin.backend.cuda import (
        _sampling_cutedsl,
        filtered_sampling,
        logits_sampling,
        sampling_native,
    )

    def entries():
        device = torch.cuda.current_device()
        triton_entries = {
            (module.__name__, name): set(value.device_caches[device][0])
            for module in (filtered_sampling, logits_sampling, _sampling_cutedsl)
            for name, value in vars(module).items()
            if hasattr(value, "device_caches")
        }
        native = sampling_native._load_native_module
        cache = native.cache_info()
        # Ignore hits, which grow normally. Warmup loads the sole native module.
        return triton_entries, (cache.misses, cache.currsize), id(native())

    logits = torch.randn(
        8, 2051, device="cuda", generator=torch.Generator("cuda").manual_seed(73)
    )
    inputs = _inputs(logits)
    inputs.top_ks.fill_(logits.shape[1])
    inputs.top_ps.fill_(1)
    inputs.temperatures.fill_(1)
    inputs.stochastic_mask.fill_(True)
    sampler = FusedSampler()
    output = torch.empty(8, dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sampler.sample_tensors(logits, inputs, out=output)
    stream.synchronize()

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("CuTe kernels must not compile after eager warmup")

    monkeypatch.setattr(_sampling_cutedsl.cute, "compile", unexpected_compile)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        sampler.sample_tensors(logits, inputs, out=output)
    compiled = entries()
    for k, p, temperature, stochastic in (
        (2051, 1.0, 1.0, True),
        (17, 0.8, 0.7, True),
        (2051, 0.9, 1.3, True),
        (1, 1.0, 1.0, True),
        (2051, 1.0, 1.0, False),
        (2051, 1.0, 1.0, True),
    ):
        inputs.top_ks.fill_(k)
        inputs.top_ps.fill_(p)
        inputs.temperatures.fill_(temperature)
        inputs.stochastic_mask.fill_(stochastic)
        inputs.offsets.add_(32)
        graph.replay()
        _assert_reference(logits, inputs, output)
    assert entries() == compiled

    # The same dtype/vocabulary kernel also accepts different row counts,
    # strides and pointer alignment, without policy-based specialization.
    for rows in (1, 4, 16):
        strided = torch.randn(rows * 2, 4103, device="cuda")[::2, 1::2]
        state = _inputs(strided)
        shifted = {}
        for field in fields(state):
            value = getattr(state, field.name)
            storage = torch.empty(rows * 2 + 1, dtype=value.dtype, device="cuda")
            shifted[field.name] = storage[1::2]
            shifted[field.name].copy_(value)
        state = replace(state, **shifted)
        output = torch.empty(rows + 1, dtype=torch.int64, device="cuda")[1:]
        actual = sampler.sample_tensors(strided, state, out=output)
        _assert_reference(strided, state, actual)
    assert entries() == compiled


def test_speculative_columns_retain_independent_scratch_and_offsets():
    sampler = FusedSampler()
    logits = torch.randn(
        8, 2051, device="cuda", generator=torch.Generator("cuda").manual_seed(74)
    )
    inputs = _inputs(logits)
    states = [replace(inputs, offsets=inputs.offsets + step * 32) for step in range(4)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for state in states:
            sampler.sample_tensors(logits, state)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [sampler.sample_tensors(logits, state) for state in states]
    assert len({output.data_ptr() for output in outputs}) == 4
    for _ in range(3):
        graph.replay()
        for output, state in zip(outputs, states, strict=True):
            _assert_reference(logits, state, output)


def test_filtered_invalid_rows_use_greedy_fallback():
    logits = torch.randn(5, 2051, device="cuda")
    logits[0].fill_(-torch.inf)
    logits[1, [2, 2000]] = torch.inf
    logits[2, [3, 1999]] = torch.nan
    inputs = _inputs(logits)
    inputs.temperatures[3] = float("nan")
    inputs.temperatures[4] = 0
    inputs.top_ks.fill_(7)
    inputs.top_ps.fill_(0.8)
    inputs.stochastic_mask.fill_(True)
    actual = FusedSampler().sample_tensors(logits, inputs)
    torch.testing.assert_close(actual, logits.argmax(-1), atol=0, rtol=0)


@pytest.mark.parametrize("field", ["top_ps", "top_ks"])
def test_policy_output_alias_is_rejected_before_any_kernel_write(field):
    logits = torch.randn(2, 257, device="cuda")
    state = _inputs(logits)
    source = getattr(state, field)
    storage = torch.full((4,), 1, dtype=source.dtype, device="cuda")
    policy = storage[::2]
    policy.copy_(source)
    state = replace(state, **{field: policy})
    before = storage.clone()
    output = storage.view(torch.int64)
    with pytest.raises(ValueError, match="overlap"):
        FusedSampler().sample_tensors(logits, state, out=output)
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


@pytest.mark.parametrize("vocab", [5, 2051])
def test_signed_zero_top_k_boundary_keeps_both_signs(vocab):
    logits = torch.full((64, vocab), -100.0, device="cuda")
    positions = [0, 1, vocab - 3, vocab - 2]
    logits[:, positions] = torch.tensor([-0.0, 0.0, -0.0, 0.0], device="cuda")
    state = _inputs(logits)
    state.stochastic_mask.fill_(True)
    state.top_ks.fill_(1)
    state.top_ps.fill_(1)
    actual = FusedSampler().sample_tensors(logits, state)
    _assert_reference(logits, state, actual)
    assert len(actual.unique()) == 4


@pytest.mark.parametrize("vocab", [17, 2051])
def test_filtered_finite_subtraction_overflow_preserves_mass(vocab):
    logits = torch.full((64, vocab), -2e38, device="cuda")
    logits[:, vocab // 2 :] = 2e38
    state = _inputs(logits)
    state.temperatures.fill_(3e38)
    state.top_ks.fill_(vocab)
    state.top_ps.fill_(0.9)
    state.stochastic_mask.fill_(True)
    actual = FusedSampler().sample_tensors(logits, state)
    _assert_reference(logits, state, actual)
    assert (actual < vocab // 2).any()


@pytest.mark.parametrize("top_k,top_p", [(7, 0.8), (2051, 0.9)])
def test_filtered_common_logit_translation_preserves_draw(top_k, top_p):
    generator = torch.Generator().manual_seed(291)
    logits = torch.randint(-8, 9, (64, 2051), generator=generator).float().cuda() * 8
    state = _inputs(logits)
    state.temperatures.fill_(8)
    state.top_ks.fill_(top_k)
    state.top_ps.fill_(top_p)
    state.stochastic_mask.fill_(True)
    sampler = FusedSampler()
    original = sampler.sample_tensors(logits, state)
    _assert_reference(logits, state, original)
    # Multiples of eight survive this FP32 translation exactly. Filtering and
    # CDF weights must be computed after removing the common offset.
    logits.add_(1e8)
    shifted = sampler.sample_tensors(logits, state)
    torch.testing.assert_close(shifted, original, atol=0, rtol=0)
    _assert_reference(logits, state, shifted)


def test_cutoff_tie_group_larger_than_candidate_scratch():
    logits = torch.ones(8, 8193, device="cuda")
    state = _inputs(logits)
    state.top_ks.fill_(7)
    state.top_ps.fill_(0.2)
    state.stochastic_mask.fill_(True)
    actual = FusedSampler().sample_tensors(logits, state)
    _assert_reference(logits, state, actual)
