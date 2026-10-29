"""Batched token sampling on the GPU for MLX runners.

Filtering follows ``resin.backend.sampling.TorchSampler``: temperature scales the
logits, top-p selects the nucleus from the full scaled distribution, top-k caps
the retained tokens, and the remaining mass is renormalized. Each row draws from
a key derived from its request seed and absolute sampled position, so results
do not depend on batch composition. Random streams differ from other backends.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from resin.backend.sampling import TorchSampler, batch_sampling_inputs


def filtered_probabilities(
    logits: mx.array, temperatures: mx.array, top_ps: mx.array, top_ks: mx.array
) -> tuple[mx.array, mx.array]:
    """Return descending token ids and their retained, normalized probabilities."""

    scores = logits.astype(mx.float32) / temperatures[:, None]
    order = mx.argsort(-scores, axis=-1)
    probabilities = mx.softmax(mx.take_along_axis(scores, order, axis=-1), axis=-1)
    preceding = mx.cumsum(probabilities, axis=-1) - probabilities
    ranks = mx.arange(scores.shape[-1])[None, :]
    # top_p == 1 disables the nucleus, as in TorchSampler; float32 cumulative
    # sums reach 1 before the tail of a large vocabulary.
    nucleus = (top_ps[:, None] >= 1.0) | (preceding < top_ps[:, None])
    keep = nucleus & (ranks < top_ks[:, None])
    probabilities = mx.where(keep, probabilities, 0.0)
    return order, probabilities / probabilities.sum(axis=-1, keepdims=True)


class MlxSampler:
    """Sample one token per plan row from ``[rows, vocab]`` MLX logits."""

    def __call__(self, logits: mx.array, plan) -> np.ndarray:
        greedy = mx.argmax(logits, axis=-1)
        values = batch_sampling_inputs(plan, logits.shape[-1])
        sampled = [
            row
            for row, (temperature, should_sample) in enumerate(
                zip(values["temperatures"], values["sampling_mask"], strict=True)
            )
            if should_sample and temperature != 0.0
        ]
        if not sampled:
            return np.array(greedy).astype(np.int64)
        rows = mx.array(sampled)
        order, probabilities = filtered_probabilities(
            logits[rows],
            mx.array([values["temperatures"][row] for row in sampled]),
            mx.array([values["top_ps"][row] for row in sampled]),
            mx.array([values["top_ks"][row] for row in sampled]),
        )
        draws = mx.concatenate(
            [
                mx.random.uniform(
                    shape=(1,),
                    key=mx.random.key(
                        TorchSampler._mixed_seed(
                            values["seeds"][row], values["offsets"][row]
                        )
                    ),
                )
                for row in sampled
            ]
        )
        # Inverse CDF over the retained, descending distribution. Retained
        # tokens form a prefix, so rounding can never select a filtered token.
        cumulative = mx.cumsum(probabilities, axis=-1)
        retained = (probabilities > 0).sum(axis=-1)
        rank = mx.minimum((cumulative < draws[:, None]).sum(axis=-1), retained - 1)
        chosen = mx.take_along_axis(order, rank[:, None], axis=-1)[:, 0]
        result = np.array(greedy).astype(np.int64)
        result[sampled] = np.array(chosen)
        return result


__all__ = ["MlxSampler", "filtered_probabilities"]
