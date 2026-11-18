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


def test_full_top_p_keeps_the_whole_vocabulary():
    # Float32 cumulative sums saturate before the tail of a 248K vocabulary;
    # top_p == 1 must still leave every token reachable, as in TorchSampler.
    vocab = 248_320
    logits = (np.random.default_rng(1).standard_normal((2, vocab)) * 8).astype(
        np.float32
    )
    _, probabilities = filtered_probabilities(
        mx.array(logits),
        mx.array([1.0, 1.0]),
        mx.array([1.0, 1.0]),
        mx.array([vocab] * 2),
    )
    assert np.all(np.array(probabilities) > 0)


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


def test_block_columns_and_gpu_chosen_shifts_match_single_positions():
    """Speculation samples column j as one-token decode at that position."""

    rng = np.random.default_rng(5)
    rows, columns, vocab = 3, 4, 300
    logits = rng.standard_normal((rows, columns, vocab)).astype(np.float32)
    params = tuple(
        SamplingParams(temperature=temperature, top_p=0.9, seed=seed)
        for temperature, seed in ((1.0, 4), (0.0, 5), (0.7, 6))
    )
    contexts = [17, 200, 3]
    sampler = MlxSampler()
    block = np.array(sampler.block_tokens(mx.array(logits), _plan(params, contexts)))
    for column in range(columns):
        single = sampler(
            mx.array(logits[:, column]),
            _plan(params, [context + column for context in contexts]),
        )
        np.testing.assert_array_equal(block[:, column], single)
    choice = np.array([2, 0, 3], np.int32)
    chosen = sampler.chosen_tokens(
        mx.array(logits[:, 0]), _plan(params, contexts), 1, columns, mx.array(choice)
    )
    expected = sampler.block_tokens(
        mx.array(logits[:, :1]), _plan(params, contexts), 1 + choice
    )[:, 0]
    np.testing.assert_array_equal(np.array(chosen), np.array(expected))


def test_gpu_positions_replace_a_bounding_plan():
    """A continuation's GPU positions draw what its exact plan would."""

    rng = np.random.default_rng(6)
    rows, columns, vocab = 3, 3, 300
    logits = mx.array(rng.standard_normal((rows, columns, vocab)).astype(np.float32))
    params = tuple(
        SamplingParams(temperature=0.8, top_p=0.95, seed=seed)
        for seed in (7, 2**63 - 9, 123456789)
    )
    contexts = [41, 8191, 70_000]
    exact = _plan(params, contexts)
    # The Engine's continuation plan only bounds each row's position.
    bound = _plan(params, [context + 4 for context in contexts])
    positions = mx.array(np.array(contexts, np.int32))
    sampler = MlxSampler()
    for shift in (0, 2):
        np.testing.assert_array_equal(
            np.array(sampler.block_tokens(logits, bound, shift, positions=positions)),
            np.array(sampler.block_tokens(logits, exact, shift)),
        )
    choice = mx.array(np.array([1, 0, 2], np.int32))
    np.testing.assert_array_equal(
        np.array(
            sampler.chosen_tokens(
                logits[:, 0], bound, 1, columns, choice, positions=positions
            )
        ),
        np.array(sampler.chosen_tokens(logits[:, 0], exact, 1, columns, choice)),
    )
