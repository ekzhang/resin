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

from resin.backend.sampling import (
    SAMPLING_OFFSET_STRIDE,
    batch_sampling_inputs,
    sampling_seed,
)


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
        return np.array(self.tokens(logits, plan)).astype(np.int64)

    def tokens(self, logits: mx.array, plan) -> mx.array:
        """Unevaluated token IDs, so they can feed the next step on the GPU."""

        return self.block_tokens(logits[:, None], plan)[:, 0]

    def chosen_tokens(
        self,
        logits: mx.array,
        plan,
        first: int,
        candidates: int,
        choice: mx.array,
        positions: mx.array | None = None,
    ) -> mx.array:
        """Sample ``[rows, vocab]`` logits at a shift chosen on the GPU.

        Row ``r`` samples as ``block_tokens`` would with shift
        ``first + choice[r]``, where ``choice`` is an unevaluated
        ``[rows]`` int array below ``candidates``. Every candidate's random
        draw is prepared, so no host round trip selects the position.
        """

        return self._sample(
            logits[:, None], plan, first, None, (candidates, choice), positions
        )[:, 0]

    def block_tokens(
        self,
        logits: mx.array,
        plan,
        shifts: np.ndarray | int = 0,
        rows: np.ndarray | None = None,
        positions: mx.array | None = None,
    ) -> mx.array:
        """Sample ``[rows, columns, vocab]`` logits into ``[rows, columns]`` IDs.

        Column ``j`` of row ``r`` samples the position ``shifts[r] + j`` past
        the plan's own: it draws exactly what decoding one token at a time
        would draw there, which speculative verification relies on. ``rows``
        names the plan row of each logits row when they are a subset.
        ``positions`` optionally gives each plan row's decode position on the
        GPU, replacing the plan's context lengths in the random streams.
        """

        return self._sample(logits, plan, shifts, rows, None, positions)

    def _sample(self, logits, plan, shifts, rows, choices, positions=None):
        count, columns = logits.shape[:2]
        tokens = mx.argmax(logits, axis=-1)
        values = batch_sampling_inputs(plan, logits.shape[-1])
        plan_rows = np.arange(count) if rows is None else np.asarray(rows)
        shifts = np.broadcast_to(np.asarray(shifts, dtype=np.int64), (count,))
        sampled = [
            index
            for index, row in enumerate(plan_rows.tolist())
            if values["sampling_mask"][row] and values["temperatures"][row] != 0.0
        ]
        if not sampled:
            return tokens
        selected = mx.array(sampled)
        flat = logits[selected].reshape(len(sampled) * columns, -1)

        def repeat(name):
            return mx.array(
                np.repeat([values[name][plan_rows[i]] for i in sampled], columns)
            )

        order, probabilities = filtered_probabilities(
            flat, repeat("temperatures"), repeat("top_ps"), repeat("top_ks")
        )
        candidates = 1 if choices is None else choices[0]
        draw_rows = [
            (index, int(shifts[index]) + column + candidate)
            for index in sampled
            for column in range(columns)
            for candidate in range(candidates)
        ]
        if positions is None:
            keys = [
                mx.random.key(
                    sampling_seed(
                        values["seeds"][plan_rows[index]],
                        values["offsets"][plan_rows[index]]
                        + shift * SAMPLING_OFFSET_STRIDE,
                    )
                )
                for index, shift in draw_rows
            ]
        else:
            keys = _device_keys(
                np.array(
                    [values["seeds"][plan_rows[i]] for i, _ in draw_rows],
                    dtype=np.uint64,
                ),
                positions[mx.array([int(plan_rows[i]) for i, _ in draw_rows])],
                np.array([shift for _, shift in draw_rows]),
            )
        draws = mx.concatenate([mx.random.uniform(shape=(1,), key=key) for key in keys])
        if choices is not None:
            draws = mx.take_along_axis(
                draws.reshape(len(sampled) * columns, candidates),
                mx.repeat(choices[1][selected].astype(mx.int32), columns)[:, None],
                axis=-1,
            )[:, 0]
        # Inverse CDF over the retained, descending distribution. Retained
        # tokens form a prefix, so rounding can never select a filtered token.
        cumulative = mx.cumsum(probabilities, axis=-1)
        retained = (probabilities > 0).sum(axis=-1)
        rank = mx.minimum((cumulative < draws[:, None]).sum(axis=-1), retained - 1)
        chosen = mx.take_along_axis(order, rank[:, None], axis=-1)[:, 0]
        tokens[selected] = chosen.reshape(len(sampled), columns).astype(tokens.dtype)
        return tokens


def _u64(values) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.uint64))


def _device_keys(seeds: np.ndarray, positions: mx.array, shifts: np.ndarray):
    """``sampling_seed`` keys for decode positions held on the GPU.

    The sampled position past ``positions[i]`` is ``positions[i] + 1 +
    shifts[i]``, as in a decode plan's offsets. The 64-bit mix matches the
    host function bit for bit, and MLX keys are its high and low words.
    """

    offsets = (positions.astype(mx.uint64) + _u64(shifts + 1)) * _u64(
        SAMPLING_OFFSET_STRIDE
    )
    value = _u64(seeds) + offsets * _u64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> _u64(30))) * _u64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> _u64(27))) * _u64(0x94D049BB133111EB)
    value = (value ^ (value >> _u64(31))) & _u64((1 << 63) - 1)
    keys = mx.stack(
        [(value >> _u64(32)).astype(mx.uint32), value.astype(mx.uint32)], axis=1
    )
    return [keys[index] for index in range(keys.shape[0])]


__all__ = ["MlxSampler", "filtered_probabilities"]
