from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import torch  # noqa: E402

from resin import SamplingParams  # noqa: E402
from resin_models_mlx.sampling import MlxSampler, filtered_probabilities  # noqa: E402


def _torch_reference(logits, temperature, top_p, top_k):
    """The distribution TorchSampler draws from, scattered back to vocab order."""

    scores = torch.from_numpy(logits).float() / temperature
    scores, token_ids = torch.sort(scores, descending=True)
    probabilities = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        cumulative = probabilities.cumsum(dim=-1)
        probabilities = probabilities * (cumulative - probabilities < top_p)
    probabilities[top_k:] = 0
    probabilities = probabilities / probabilities.sum()
    result = torch.zeros_like(probabilities)
    result[token_ids] = probabilities
    return result.numpy()


@pytest.mark.parametrize(
    ("temperature", "top_p", "top_k"),
    [(1.0, 1.0, 64), (0.7, 0.9, 64), (1.3, 0.5, 5), (0.6, 0.95, 20), (1.0, 1.0, 3)],
)
def test_filtered_distribution_matches_torch_sampler(temperature, top_p, top_k):
    rng = np.random.default_rng(0)
    logits = (rng.standard_normal((3, 64)) * 3).astype(np.float32)
    order, probabilities = filtered_probabilities(
        mx.array(logits),
        mx.array([temperature] * 3),
        mx.array([top_p] * 3),
        mx.array([top_k] * 3),
    )
    actual = np.zeros_like(logits)
    np.put_along_axis(actual, np.array(order), np.array(probabilities), axis=-1)
    for row in range(3):
        expected = _torch_reference(logits[row], temperature, top_p, top_k)
        np.testing.assert_allclose(actual[row], expected, atol=1e-6)


def _plan(params, contexts, mask=None):
    rows = len(params)
    return SimpleNamespace(
        num_sequences=rows,
        sampling_params=params,
        context_lengths=np.asarray(contexts, dtype=np.uint32),
        token_indptr=np.arange(rows + 1, dtype=np.uint32),
        token_ids=np.zeros(rows, dtype=np.uint32),
        sampling_mask=np.ones(rows, np.bool_) if mask is None else np.asarray(mask),
    )


def test_greedy_rows_and_unsampled_rows_use_argmax():
    logits = mx.array(np.random.default_rng(1).standard_normal((3, 50)), mx.float32)
    params = (
        SamplingParams(temperature=0.0, seed=1),
        SamplingParams(temperature=1.0, seed=2),
        SamplingParams(temperature=1.0, seed=3),
    )
    tokens = MlxSampler()(logits, _plan(params, [4, 5, 6], [True, False, True]))
    greedy = np.array(mx.argmax(logits, axis=-1))
    assert tokens[0] == greedy[0] and tokens[1] == greedy[1]


def test_draws_depend_only_on_seed_and_position():
    rng = np.random.default_rng(2)
    row = rng.standard_normal(500).astype(np.float32)
    other = rng.standard_normal(500).astype(np.float32)
    policy = SamplingParams(temperature=1.0, top_p=0.9, seed=7)
    sampler = MlxSampler()
    alone = sampler(mx.array(row[None]), _plan((policy,), [10]))
    batched = sampler(
        mx.array(np.stack([other, row])),
        _plan((SamplingParams(temperature=1.0, seed=8), policy), [3, 10]),
    )
    assert alone[0] == batched[1]
    draws = {
        int(sampler(mx.array(row[None]), _plan((policy,), [position]))[0])
        for position in range(40)
    }
    assert len(draws) > 1


def test_samples_follow_the_filtered_distribution():
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32))
    policy = [
        SamplingParams(temperature=1.0, top_k=2, seed=seed) for seed in range(4000)
    ]
    tokens = MlxSampler()(
        mx.array(np.tile(logits, (len(policy), 1))), _plan(tuple(policy), [0] * 4000)
    )
    counts = np.bincount(tokens, minlength=4) / len(tokens)
    assert counts[2] == counts[3] == 0
    assert abs(counts[0] - 0.625) < 0.03
