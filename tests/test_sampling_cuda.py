"""Real CUDA regressions for the shared sampler's request-local random stream."""

from dataclasses import fields, replace

import pytest
import torch

from resin.backend.cuda.sampling import FusedSampler, SamplingTensors

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


@pytest.fixture(scope="module")
def sampler():
    pytest.importorskip("triton")
    return FusedSampler()


def _inputs(logits):
    rows, vocab = logits.shape
    row = torch.arange(rows, device=logits.device)
    return SamplingTensors(
        temperatures=0.5 + (row % 4).float() * 0.5,
        top_ps=0.6 + (row % 3).float() * 0.2,
        top_ks=torch.where(row % 3 == 0, vocab, 7 + row % 5).int(),
        seeds=(1 << 62) + 17 + row * (1 << 32),
        offsets=(1 << 40) + 32 + row * 32,
        stochastic_mask=row % 5 != 4,
    )


def _select(inputs, indices):
    return replace(
        inputs,
        **{
            field.name: getattr(inputs, field.name)[indices] for field in fields(inputs)
        },
    )


def _logits(rows, vocab, seed=71):
    return torch.randn(
        rows, vocab, generator=torch.Generator().manual_seed(seed)
    ).cuda()


@pytest.mark.parametrize("rows", [4, 32])
def test_request_stream_is_independent_of_batch_position(sampler, rows):
    logits = _logits(rows, 257)
    inputs = _inputs(logits)
    expected = sampler.sample_tensors(logits, inputs)
    # Moving a request cannot replace its seed/offset with row zero's values,
    # or change its RNG subsequence to the new physical batch index.
    order = torch.arange(rows - 1, -1, -1, device=logits.device)
    actual = sampler.sample_tensors(logits[order], _select(inputs, order))
    torch.testing.assert_close(actual, expected[order], rtol=0, atol=0)
    for row in range(rows):
        actual = sampler.sample_tensors(
            logits[row : row + 1], _select(inputs, slice(row, row + 1))
        )
        torch.testing.assert_close(actual, expected[row : row + 1], rtol=0, atol=0)
    padded = torch.cat((logits, _logits(8, logits.shape[1], seed=72)))
    padded_inputs = _inputs(padded)
    for field in fields(inputs):
        getattr(padded_inputs, field.name)[:rows].copy_(getattr(inputs, field.name))
    padded_inputs.stochastic_mask[rows:] = False
    actual = sampler.sample_tensors(padded, padded_inputs)
    torch.testing.assert_close(actual[:rows], expected, rtol=0, atol=0)


def _capture(sampler, logits, inputs, *, pool=None, stream=None):
    output = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
    if stream is None:
        stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sampler.sample_tensors(logits, inputs, out=output)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool, stream=stream):
        sampler.sample_tensors(logits, inputs, out=output)
    return graph, output


def test_graph_catalogue_replays_and_mutable_request_state(sampler):
    pool, stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream()
    cases = []
    for rows in (1, 4, 16, 32, 64, 128):
        logits = _logits(rows, 257)
        inputs = _inputs(logits)
        graph, output = _capture(sampler, logits, inputs, pool=pool, stream=stream)
        cases.append((logits, inputs, graph, output))
    # Retain the entire shared-pool catalogue, as model graph capture does.
    for logits, inputs, graph, output in cases:
        expected = sampler.sample_tensors(logits, inputs).cpu()
        for _ in range(30):
            graph.replay()
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        # Reuse captured addresses while changing every input's values,
        # including high seed/offset bits. Replays must read current values.
        logits.copy_(_logits(logits.shape[0], logits.shape[1], seed=73))
        inputs.seeds.add_(1 << 40)
        inputs.offsets.add_(1 << 41)
        inputs.temperatures.fill_(1.3)
        inputs.top_ps.fill_(0.87)
        inputs.top_ks.fill_(13)
        inputs.stochastic_mask.copy_(
            torch.arange(logits.shape[0], device=logits.device) % 3 != 1
        )
        expected = sampler.sample_tensors(logits, inputs).cpu()
        for _ in range(3):
            graph.replay()
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def _conditional_distribution(logits, temperature, top_k, top_p):
    """Independent FP64 oracle; probability-threshold cutoffs retain ties."""
    values = torch.tensor(logits, dtype=torch.float64) / temperature
    probabilities = values.softmax(0)
    descending = probabilities.sort(descending=True).values
    crossing = min(int((descending.cumsum(0) < top_p).sum()), len(descending) - 1)
    probabilities[probabilities < descending[crossing]] = 0
    probabilities[values < values.topk(top_k).values[-1]] = 0
    return probabilities / probabilities.sum()


@pytest.mark.parametrize(
    ("values", "temperature", "top_k", "top_p"),
    [
        ([-2.6, -1.7, -0.9, -0.15, 0.4, 1.1, 1.9], 1.0, 7, 1.0),
        ([-2.6, -1.7, -0.9, -0.15, 0.4, 1.1, 1.9], 0.7, 4, 0.8),
        ([-2.6, -1.7, -0.9, -0.15, 0.4, 1.1, 1.9], 1.7, 3, 0.95),
        ([2.0, 2.0, 2.0, 0.0, -1.0], 1.0, 5, 0.25),
    ],
)
def test_sampling_matches_conditional_distribution(
    sampler, values, temperature, top_k, top_p
):
    count = 8192
    logits = torch.tensor(values, device="cuda").expand(count, -1).contiguous()
    inputs = _inputs(logits)
    inputs.temperatures.fill_(temperature)
    inputs.top_ks.fill_(top_k)
    inputs.top_ps.fill_(top_p)
    inputs.seeds.fill_((1 << 62) + 41)
    inputs.stochastic_mask.fill_(True)
    sampled = sampler.sample_tensors(logits, inputs).cpu()
    expected = _conditional_distribution(values, temperature, top_k, top_p)
    observed = torch.bincount(sampled, minlength=len(values)).double() / count
    assert torch.all(observed[expected == 0] == 0), (observed, expected)
    # Six binomial standard deviations plus a small rounding allowance keeps
    # this statistical test well away from ordinary finite-sample variation.
    bounds = 6 * (expected * (1 - expected) / count).sqrt() + 2 / count
    assert torch.all((observed - expected).abs() <= bounds), (observed, expected)


def test_real_vocabulary_sampling_replays(sampler):
    logits = _logits(4, 201088)
    inputs = _inputs(logits)
    graph, output = _capture(sampler, logits, inputs)
    expected = sampler.sample_tensors(logits, inputs).cpu()
    assert torch.all((expected >= 0) & (expected < logits.shape[1]))
    for _ in range(5):
        graph.replay()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    inputs.offsets.add_(32)
    expected = sampler.sample_tensors(logits, inputs).cpu()
    graph.replay()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
