"""Sampling preserves raw scores and private state across asynchronous replays."""

from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from resin import BatchPlan, SamplingParams
from resin.backend.cuda.completion import DeviceAcceptance, DeviceCompletion
from resin.backend.cuda.sampling import FusedSampler, SamplingTensors

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _inputs(vocab, dtype, *, seed=171):
    rows, width = 5, 4
    generator = torch.Generator("cuda").manual_seed(seed)
    storage = torch.randn(
        rows, width, 2 * vocab + 3, device="cuda", dtype=dtype, generator=generator
    )
    logits = storage[..., 1 : 2 * vocab + 1 : 2]
    row = torch.arange(rows, device="cuda", dtype=torch.int64)
    sampling = SamplingTensors(
        temperatures=0.5 + (row % 3).float() * 0.5,
        top_ps=torch.where(row % 3 == 0, 1.0, 0.87),
        top_ks=torch.where(row % 2 == 0, vocab, 9).int(),
        seeds=(1 << 62) + seed + row * (1 << 32),
        offsets=(1 << 42) + 97 + row * (1 << 36),
        stochastic_mask=row % 4 != 3,
    )
    return logits, sampling


def _clone_sampling(sampling):
    return replace(
        sampling,
        **{
            field.name: getattr(sampling, field.name).clone()
            for field in fields(sampling)
        },
    )


def _capture(launch, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


@pytest.mark.parametrize("vocab", [2051, 262144])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_sampled_completion_keeps_raw_logprobs_across_changed_replay(vocab, dtype):
    sampler = FusedSampler()
    logits, sampling = _inputs(vocab, dtype)
    raw = logits.clone()
    metadata = _clone_sampling(sampling)
    targets = torch.empty(logits.shape[:2], device="cuda", dtype=torch.int64)
    sampler.sample_block(logits, sampling, out=targets)
    drafts = targets[:, :3].clone()
    # Vary accepted prefixes, including zero and all three drafts.
    for row, accepted in enumerate((0, 1, 3, 2, 0)):
        if accepted < 3:
            drafts[row, accepted] = (drafts[row, accepted] + 1) % vocab
    positions = torch.tensor([8191, 8192, 8703, 8704, 1], device="cuda")
    state = {}

    def launch():
        sampler.sample_block(logits, sampling, out=targets)
        state["accepted"] = DeviceAcceptance.from_samples(drafts, targets, positions)

    graph = _capture(launch, torch.cuda.Stream())
    graph.replay()
    torch.testing.assert_close(logits, raw, rtol=0, atol=0)
    for field in fields(sampling):
        torch.testing.assert_close(
            getattr(sampling, field.name), getattr(metadata, field.name), rtol=0, atol=0
        )
    accepted = state["accepted"]
    counts = accepted.counts[:4].cpu().numpy()
    np.testing.assert_array_equal(counts, [1, 2, 4, 3])
    tokens = accepted.tokens[:4].cpu()
    # Independent FP64 normalization of unscaled, unfiltered model logits.
    reference = raw[:4].cpu().double()
    reference -= reference.logsumexp(-1, keepdim=True)
    plan = BatchPlan(
        sequence_ids=[10, 11, 12, 13],
        context_lengths=[8192, 8193, 8704, 8705],
        token_ids=[1, 2, 3, 4],
        token_indptr=[0, 1, 2, 3, 4],
        sampling_params=[
            SamplingParams(temperature=0.5, top_logprobs=3),
            SamplingParams(temperature=1.0, top_p=0.87, top_k=9),
            SamplingParams(temperature=1.5, top_p=0.87, logprobs=True),
            SamplingParams(temperature=0.5, top_k=9, top_logprobs=1),
        ],
        sampling_mask=[True, True, True, False],
    )
    debug_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        pending = DeviceCompletion.from_acceptance(
            accepted, rows=4, plan=plan, logits=logits
        )
        # Queue a different generation before harvesting any score buffers.
        logits.mul_(-0.7)
        sampling.top_ks.fill_(vocab)
        sampling.top_ps.fill_(0.71)
        sampling.temperatures.fill_(1.3)
        sampling.seeds.add_(1 << 39)
        sampling.offsets.add_(1 << 44)
        sampling.stochastic_mask.logical_not_()
        drafts.zero_()
        positions.add_(4)
        graph.replay()
    finally:
        torch.cuda.set_sync_debug_mode(debug_mode)
    output = pending.result()
    np.testing.assert_array_equal(output.token_indptr, [0, 1, 3, 7, 10])
    np.testing.assert_array_equal(
        output.token_ids,
        np.concatenate(
            [tokens[row, :count].numpy() for row, count in enumerate(counts)]
        ),
    )
    np.testing.assert_array_equal(output.logprob_indptr, [0, 1, 1, 5, 5])
    expected_scores = np.concatenate(
        [
            reference[row, torch.arange(count), tokens[row, :count]].numpy()
            for row, count in enumerate(counts)
            if row in (0, 2)
        ]
    )
    assert output.logprobs is not None
    np.testing.assert_allclose(output.logprobs, expected_scores, atol=3e-6, rtol=0)
    np.testing.assert_array_equal(output.top_indptr, [0, 3, 3, 3, 3, 3])
    # Quantized logits can tie: verify each returned ID and the ordered top
    # values, without imposing an arbitrary CPU/GPU tie order on IDs.
    assert output.top_token_ids is not None
    assert output.top_logprobs is not None
    top_ids = torch.from_numpy(output.top_token_ids.astype(np.int64))
    np.testing.assert_allclose(
        output.top_logprobs, reference[0, 0, top_ids].numpy(), atol=3e-6, rtol=0
    )
    np.testing.assert_allclose(
        output.top_logprobs, reference[0, 0].topk(3).values.numpy(), atol=3e-6, rtol=0
    )


def _change_policy(sampling, step, vocab):
    k, p, temperature, stochastic = (
        (vocab, 1.0, 1.0, True),
        (9, 0.87, 0.7, True),
        (vocab, 0.95, 1.3, True),
        (vocab, 1.0, 1.0, False),
    )[step % 4]
    sampling.top_ks.fill_(k)
    sampling.top_ps.fill_(p)
    sampling.temperatures.fill_(temperature)
    sampling.stochastic_mask.fill_(stochastic)
    sampling.seeds.add_(1 << 39)
    sampling.offsets.add_(32)


@pytest.mark.parametrize("vocab", [2051, 262144])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_independent_sampling_graphs_replay_on_concurrent_streams(vocab, dtype):
    sampler = FusedSampler()
    rounds = 12
    cases = [_inputs(vocab, dtype, seed=seed) for seed in (171, 789)]
    expected, graphs, outputs, streams, snapshots = [], [], [], [], []
    for logits, sampling in cases:
        serial_sampling = _clone_sampling(sampling)
        serial = []
        for step in range(rounds):
            _change_policy(serial_sampling, step, vocab)
            serial.append(sampler.sample_block(logits, serial_sampling))
        expected.append(torch.stack(serial))
        stream = torch.cuda.Stream()
        output = torch.empty(logits.shape[:2], device="cuda", dtype=torch.int64)

        def launch(logits=logits, sampling=sampling, output=output):
            sampler.sample_block(logits, sampling, out=output)

        graphs.append(_capture(launch, stream))
        streams.append(stream)
        outputs.append(output)
        snapshots.append(
            torch.empty((rounds, *output.shape), device="cuda", dtype=output.dtype)
        )
    torch.cuda.synchronize()
    # Alternate submissions without waiting between streams or replay rounds.
    # Each graph must own native filter scratch, expanded policies and CDF state.
    for step in range(rounds):
        for index, (_, sampling) in enumerate(cases):
            with torch.cuda.stream(streams[index]):
                _change_policy(sampling, step, vocab)
                graphs[index].replay()
                snapshots[index][step].copy_(outputs[index])
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for actual, reference in zip(snapshots, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
