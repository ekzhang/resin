"""Captured score tails preserve row policy, accepted IDs, and owned results."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from resin import BatchPlan, SamplingParams
from resin.backend.cuda.completion import DeviceAcceptance, DeviceCompletion
from resin_models_cuda.scoring import (
    _PROMPT_SCORE_CHUNK_SIZE,
    prompt_scores,
    score_completion,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _decode_plan(tops, *, mask=(True, True, False)):
    return BatchPlan(
        sequence_ids=[1, 2, 3],
        context_lengths=[8, 8, 8],
        token_ids=[1, 2, 3],
        token_indptr=[0, 1, 2, 3],
        sampling_mask=mask,
        sampling_params=[
            SamplingParams(logprobs=True, top_logprobs=count, seed=row)
            for row, count in enumerate(tops)
        ],
    )


@pytest.mark.parametrize("rows", [1, 32])
@pytest.mark.parametrize("fixed", [None, 0, 3])
@pytest.mark.parametrize("scored", [False, True])
@torch.inference_mode()
def test_eager_acceptance_completion_owns_scored_and_unscored_blocks(
    rows, fixed, scored
):
    from resin.backend.speculative import SpeculativeStats, _accept_speculative_tokens

    drafts = torch.arange(rows * 3).view(rows, 3).remainder(31)
    targets = torch.cat((drafts.clone(), torch.full((rows, 1), 32)), dim=1)
    for row in range(rows):
        targets[row, row % 4 :] = 33
    expected = _accept_speculative_tokens(
        drafts.tolist(), targets.tolist(), benchmark_accepted_drafts=fixed
    )
    drafts, targets = drafts.cuda(), targets.cuda()
    positions = torch.arange(rows, device="cuda") * 7
    logits = torch.randn((rows, 4, 65), device="cuda", dtype=torch.bfloat16)
    plan = BatchPlan(
        sequence_ids=list(range(rows)),
        context_lengths=np.arange(rows, dtype=np.uint32) * 7,
        token_ids=[1] * rows,
        token_indptr=list(range(rows + 1)),
        sampling_params=[SamplingParams(logprobs=scored)] * rows,
    )
    acceptance = DeviceAcceptance.from_samples(
        drafts, targets, positions, fixed_acceptance=fixed
    )
    score_completion(logits, acceptance, plan)
    stats = SpeculativeStats(3)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        acceptance = DeviceAcceptance.from_samples(
            drafts, targets, positions, fixed_acceptance=fixed
        )
        pending = DeviceCompletion.from_acceptance(
            acceptance,
            rows=rows,
            stats=stats,
            plan=plan,
            scores=score_completion(logits, acceptance, plan),
        )
        acceptance.tokens.fill_(-1)
        acceptance.counts.zero_()
        logits.zero_()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    result = pending.result()
    assert pending.result() is result
    np.testing.assert_array_equal(result.token_ids, sum(expected.token_rows, []))
    np.testing.assert_array_equal(
        np.diff(result.token_indptr), expected.processed_counts
    )
    assert (result.logprobs is not None) == scored
    assert stats.blocks == rows and stats.accepted == sum(expected.accepted_counts)
    assert stats.emitted == sum(expected.processed_counts)


@torch.inference_mode()
def test_captured_mixed_scores_survive_replay_and_forced_acceptance(monkeypatch):
    logits = torch.randn((3, 4, 8193), device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([8, 8, 8], device="cuda")
    drafts = torch.tensor([[4, 7, 9], [2, 4, 8], [3, 1, 7]], device="cuda")
    targets = torch.tensor([[1, 2, 3, 4], [2, 7, 8, 1], [8, 4, 1, 2]], device="cuda")
    acceptance = DeviceAcceptance.from_samples(
        drafts, targets, positions, fixed_acceptance=2
    )
    plan = _decode_plan([5, 0, 32])
    # Warm capture before sync-debug; hot scoring must not read device scalars.
    score_completion(logits, acceptance, plan)
    reference = logits.cpu().double().log_softmax(-1)
    emitted = acceptance.tokens.cpu()
    expected = reference.gather(-1, emitted[..., None]).squeeze(-1).numpy()

    def forbid_normalize(*args, **kwargs):
        raise AssertionError("CUDA scoring must not fall back to Torch log_softmax")

    monkeypatch.setattr(torch.Tensor, "log_softmax", forbid_normalize)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        scores = score_completion(logits, acceptance, plan)
        assert scores is not None
        pending = DeviceCompletion.from_acceptance(
            acceptance, rows=3, plan=plan, scores=scores
        )
        logits.mul_(-0.3)
        next_acceptance = DeviceAcceptance.from_samples(drafts, targets, positions)
        changed = _decode_plan([32, 1, 9], mask=(True, False, True))
        score_completion(logits, next_acceptance, changed)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    result = pending.result()
    assert result.logprobs is not None
    np.testing.assert_array_equal(result.logprob_indptr, [0, 3, 6, 6])
    np.testing.assert_allclose(
        result.logprobs, expected[:2, :3].reshape(-1), atol=3e-6, rtol=0
    )
    assert result.top_indptr is not None
    np.testing.assert_array_equal(result.top_indptr, [0, 5, 10, 15, 15, 15, 15])
    assert len(getattr(logits, "_resin_score_graphs")) == 1


@torch.inference_mode()
def test_prompt_projection_scores_and_duplicate_targets_use_cuda(monkeypatch):
    count, hidden_size, vocab = 2 * _PROMPT_SCORE_CHUNK_SIZE + 1, 32, 2051
    hidden = torch.randn((count, hidden_size), device="cuda", dtype=torch.float32)
    head = torch.nn.Linear(hidden_size, vocab, bias=False, device="cuda")
    tops = np.asarray([0, 1, 5], np.uint32)[np.arange(count) % 3]
    target_counts = np.asarray(
        [2 if row % 7 == 0 else 0 for row in range(count)], np.uint32
    )
    target_indptr = np.concatenate(
        (np.zeros(1, np.uint32), target_counts.cumsum(dtype=np.uint32))
    )
    target_ids = np.full(int(target_indptr[-1]), 17, np.uint32)
    selected = np.arange(count, dtype=np.uint32) % vocab
    plan = BatchPlan(
        sequence_ids=[1],
        context_lengths=[0],
        token_ids=[1] * count,
        token_indptr=[0, count],
        sampling_params=[SamplingParams(seed=1, prompt_top_logprobs=5)],
        prompt_score_indices=np.arange(count, dtype=np.uint32),
        prompt_score_token_ids=selected,
        prompt_score_indptr=[0, count],
        prompt_score_top_counts=tops,
        prompt_target_indptr=target_indptr,
        prompt_target_token_ids=target_ids,
    )
    values = torch.cat(
        [head(part).double() for part in hidden.split(_PROMPT_SCORE_CHUNK_SIZE)]
    )
    reference = values.log_softmax(-1).cpu()
    expected = reference[
        torch.arange(count), torch.from_numpy(selected.astype(np.int64))
    ]
    projected = []

    def project(rows):
        projected.append(len(rows))
        return head(rows)

    first = prompt_scores(plan, hidden, project)
    assert first is not None
    assert max(projected) == _PROMPT_SCORE_CHUNK_SIZE
    assert set(projected) == {1, _PROMPT_SCORE_CHUNK_SIZE}
    projected.clear()

    def forbid_normalize(*args, **kwargs):
        raise AssertionError("CUDA prompt scoring must not call Torch log_softmax")

    monkeypatch.setattr(torch.Tensor, "log_softmax", forbid_normalize)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        result = prompt_scores(plan, hidden, project)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert result is not None and projected == []
    output = result.output(plan)
    np.testing.assert_allclose(
        output["prompt_logprobs"], expected.numpy(), atol=3e-6, rtol=0
    )
    np.testing.assert_array_equal(output["prompt_target_token_ids"], target_ids)
    expected_targets = reference[
        torch.from_numpy(np.repeat(np.arange(count), target_counts)), 17
    ]
    np.testing.assert_allclose(
        output["prompt_target_logprobs"], expected_targets.numpy(), atol=3e-6, rtol=0
    )
    top_ids = torch.from_numpy(output["prompt_top_token_ids"].astype(np.int64))
    top_rows = torch.from_numpy(np.repeat(np.arange(count), tops))
    np.testing.assert_allclose(
        output["prompt_top_logprobs"],
        reference[top_rows, top_ids].numpy(),
        atol=3e-6,
        rtol=0,
    )


@torch.inference_mode()
def test_ordinary_inference_scores_reuse_the_original_logits_owner(monkeypatch):
    from resin_models_cuda import scoring

    logits = torch.randn((4, 2051), device="cuda", dtype=torch.bfloat16)
    sampled = torch.tensor([7, 8, 9, 0], device="cuda")
    acceptance = DeviceAcceptance.from_samples(
        sampled[:, None][:, :0], sampled[:, None], torch.zeros_like(sampled)
    )
    plan = _decode_plan([5, 0, 0])
    first = score_completion(logits, acceptance, plan)
    assert first is not None
    cache = getattr(logits, "_resin_score_graphs")
    assert len(cache) == 1

    def forbid_capture(*args, **kwargs):
        raise AssertionError("ordinary decode must reuse its captured score tail")

    monkeypatch.setattr(scoring, "_capture", forbid_capture)
    assert score_completion(logits, acceptance, plan) is first
    output = scoring.format_output(logits, sampled, plan)
    assert len(cache) == 1
    np.testing.assert_array_equal(output.token_ids, sampled[:3].cpu().numpy())
    np.testing.assert_array_equal(output.logprob_indptr, [0, 1, 2, 2])


@torch.inference_mode()
def test_prompt_graph_cache_buckets_targets_and_bounds_owned_projections():
    logits = torch.randn((6, 65), device="cuda")

    def plan(rows, targets):
        return BatchPlan(
            sequence_ids=[1],
            context_lengths=[0],
            token_ids=[1] * rows,
            token_indptr=[0, rows],
            sampling_params=[SamplingParams(prompt_logprobs=True)],
            prompt_score_indices=np.arange(rows, dtype=np.uint32),
            prompt_score_token_ids=[1] * rows,
            prompt_score_indptr=[0, rows],
            prompt_score_top_counts=[0] * rows,
            prompt_target_indptr=[0] + [targets] * rows,
            prompt_target_token_ids=[17] * targets,
        )

    first_plan = plan(4, 3)
    first = prompt_scores(first_plan, logits[:4], cache_owner=logits)
    assert first is not None
    cache = getattr(logits, "_resin_prompt_score_graphs")
    first_graph = next(iter(cache.values())).graph
    second_plan = plan(4, 4)
    second = prompt_scores(second_plan, logits[:4], cache_owner=logits)
    assert second is not None
    assert len(cache) == 1 and next(iter(cache.values())).graph is first_graph

    # Each different projection shape owns a graph pool, but only four stay hot.
    # Earlier results must survive their source entry being evicted.
    for rows in range(1, 7):
        prompt_scores(plan(rows, 0), logits[:rows], cache_owner=logits)
        assert len(cache) <= 4
    reference = logits.double().log_softmax(-1).cpu()
    first_output = first.output(first_plan)
    second_output = second.output(second_plan)
    np.testing.assert_allclose(
        first_output["prompt_logprobs"], reference[:4, 1].numpy(), atol=3e-6, rtol=0
    )
    for output, count in ((first_output, 3), (second_output, 4)):
        np.testing.assert_array_equal(output["prompt_target_token_ids"], [17] * count)
        np.testing.assert_allclose(
            output["prompt_target_logprobs"],
            [reference[0, 17].item()] * count,
            atol=3e-6,
            rtol=0,
        )
