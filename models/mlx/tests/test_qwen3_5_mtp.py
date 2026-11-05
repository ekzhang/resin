"""MTP speculative decoding against target-only decoding of the same model."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from test_qwen3_5_mlx import _model, _prompt  # noqa: E402

from resin import BatchPhase, Engine, SamplingParams  # noqa: E402
from resin_models_mlx.qwen3_5.model import (  # noqa: E402
    AttentionHistory,
    BlockInputs,
)
from resin_models_mlx.qwen3_5.runner import Qwen35MlxRunner  # noqa: E402

# Distinct lengths identify prompts; generation crosses a 128-token page.
PROMPTS = (5, 126, 131)


def _runner(model, drafts: int, slabs: int = 96) -> Qwen35MlxRunner:
    return Qwen35MlxRunner(
        model,
        cache_slab_capacity=slabs,
        workspace_bytes=64 << 20,
        speculative_draft_tokens=drafts,
    )


@pytest.fixture(scope="module")
def model():
    # FP32 weights and every expert keep greedy choices away from rounding ties.
    return _model(seed=7, dtype=mx.float32, all_experts=True, mtp=True)


def _generate(runner, prompts, params, *, prefix_cache=False, context=8192):
    with Engine(runner, context_length=context, prefix_cache=prefix_cache) as engine:
        requests = [engine.submit(prompt, sampling_params=params) for prompt in prompts]
        while engine.has_work():
            engine.step()
        return [request.take_result() for request in requests]


class _Oracle:
    """Synthetic drafts: the known continuation, except at ``miss`` positions.

    This drives full and partial acceptance through the real runner; it is
    not evidence of draft quality. Prompts are recognized by their lengths.
    """

    def __init__(self, sampler, continuations, miss=lambda position: False):
        self._sampler = sampler
        self._continuations = continuations
        self._miss = miss
        self._sequences: dict[int, list[int]] = {}

    def __call__(self, logits, plan):
        return np.array(self.tokens(logits, plan)).astype(np.int64)

    def tokens(self, logits, plan):
        return self.block_tokens(logits[:, None], plan)[:, 0]

    def chosen_tokens(self, logits, plan, first, candidates, choice):
        shifts = first + np.array(choice)
        return self.block_tokens(logits[:, None], plan, shifts)[:, 0]

    def block_tokens(self, logits, plan, shifts=0, rows=None):
        tokens = self._sampler.block_tokens(logits, plan, shifts, rows)
        if plan.phase == BatchPhase.PREFILL:
            for row, sequence in enumerate(plan.sequence_ids):
                length = int(plan.context_lengths[row]) + int(
                    plan.token_indptr[row + 1] - plan.token_indptr[row]
                )
                if length in self._continuations:
                    self._sequences[int(sequence)] = self._continuations[length]
            return tokens
        if logits.shape[1] != 1:
            return tokens  # Target verification samples normally.
        plan_rows = np.arange(logits.shape[0]) if rows is None else np.asarray(rows)
        shifts = np.broadcast_to(np.asarray(shifts), (len(plan_rows),))
        proposals = np.array(tokens).copy()
        for index, row in enumerate(plan_rows.tolist()):
            full = self._sequences[int(plan.sequence_ids[row])]
            position = int(plan.context_lengths[row]) + 1 + int(shifts[index])
            if position < len(full) and not self._miss(position):
                proposals[index, 0] = full[position]
            else:
                proposals[index, 0] = (full[-1] + 1) % 256
        return mx.array(proposals)


def _with_oracle(runner, prompts, expected, miss=lambda position: False):
    continuations = {
        len(prompt): prompt + output
        for prompt, output in zip(prompts, expected, strict=True)
    }
    runner._sampler = _Oracle(runner._sampler, continuations, miss)
    return runner


def test_block_commit_matches_sequential_decode(model):
    config = model.config
    prompt, block = _prompt(19, seed=1), _prompt(4, seed=2)
    first = model(
        mx.array([prompt]),
        mx.array([0]),
        model.initial_state(1),
        (None,) * config.full_layer_count,
    )
    history = [
        AttentionHistory(key, value)
        for key, value in zip(first.keys, first.values, strict=True)
    ]
    retained: list[BlockInputs] = []
    verified = model(
        mx.array([block]),
        mx.array([len(prompt)]),
        first.state,
        history,
        retain=retained,
    )
    state, keys, values = first.state, first.keys, first.values
    for accepted in range(len(block)):
        step = model(
            mx.array([[block[accepted]]]),
            mx.array([len(prompt) + accepted]),
            state,
            [
                AttentionHistory(key, value)
                for key, value in zip(keys, values, strict=True)
            ],
        )
        np.testing.assert_allclose(
            np.array(model.logits(verified.hidden[0, accepted])),
            np.array(model.logits(step.hidden[0, 0])),
            atol=1e-4,
            rtol=1e-4,
        )
        state = step.state
        keys = tuple(
            mx.concatenate([key, new], axis=1)
            for key, new in zip(keys, step.keys, strict=True)
        )
        values = tuple(
            mx.concatenate([value, new], axis=1)
            for value, new in zip(values, step.values, strict=True)
        )
        committed = model.commit_block(retained, mx.array([accepted]))
        for actual, expected in zip(
            (*committed.recurrent, *committed.convolution),
            (*state.recurrent, *state.convolution),
            strict=True,
        ):
            np.testing.assert_allclose(
                np.array(actual), np.array(expected), atol=1e-5, rtol=1e-5
            )


@pytest.mark.parametrize("drafts", [1, 2, 3])
@pytest.mark.parametrize("proposals", ["model", "all", "partial"])
def test_greedy_speculation_matches_target_only(model, drafts, proposals):
    prompts = [_prompt(length, seed=length) for length in PROMPTS]
    params = SamplingParams(max_output_tokens=24, temperature=0.0)
    expected = _generate(_runner(model, 0), prompts, params)
    runner = _runner(model, drafts)
    if proposals != "model":
        miss = (lambda position: position % 3 == 0) if proposals == "partial" else None
        _with_oracle(runner, prompts, expected, *(() if miss is None else (miss,)))
    assert _generate(runner, prompts, params) == expected
    stats = runner.speculative_stats()
    if proposals == "all":
        assert stats["acceptance_rate"] > 0.85
    elif proposals == "partial":
        assert 0.1 < stats["acceptance_rate"] < 0.9
    assert stats["emitted"] >= stats["blocks"]
    assert not runner._drafts and not runner._last_hidden


@pytest.mark.parametrize("proposals", ["model", "partial"])
def test_seeded_sampling_speculation_matches_target_only(model, proposals):
    prompts = [_prompt(length, seed=length + 1) for length in PROMPTS]
    params = SamplingParams(
        max_output_tokens=20, temperature=0.9, top_p=0.95, top_k=40, seed=1234
    )
    expected = _generate(_runner(model, 0), prompts, params)
    runner = _runner(model, 2)
    if proposals == "partial":
        _with_oracle(runner, prompts, expected, lambda position: position % 4 == 1)
    assert _generate(runner, prompts, params) == expected


def test_stop_token_inside_an_accepted_block(model):
    prompts = [_prompt(length, seed=length + 2) for length in PROMPTS]
    params = SamplingParams(max_output_tokens=24, temperature=0.0)
    full = _generate(_runner(model, 0), prompts, params)
    stops = SamplingParams(
        max_output_tokens=24,
        temperature=0.0,
        stop_token_ids=[full[0][5], full[2][9]],
    )
    expected = _generate(_runner(model, 0), prompts, stops)
    runner = _with_oracle(_runner(model, 3), prompts, full)
    assert _generate(runner, prompts, stops) == expected


def test_checkpoint_crossing_and_prefix_hit(model):
    # Accepted blocks cross the 4096-token state checkpoint during decode. A
    # prompt crossing it publishes a prefix, which a later request restores;
    # that hit has no stored target hidden state for its first MTP pair.
    crossing = _prompt(4090, seed=11)
    publisher = _prompt(4150, seed=12)
    continuation = publisher[:4096] + _prompt(7, seed=13)
    prompts = [crossing, publisher, continuation]
    params = SamplingParams(max_output_tokens=16, temperature=0.0)
    expected = [
        _generate(_runner(model, 0, slabs=192), [prompt], params, context=16384)[0]
        for prompt in prompts
    ]
    runner = _with_oracle(_runner(model, 3, slabs=192), prompts, expected)
    with Engine(runner, context_length=16384, prefix_cache=True) as engine:
        actual = []
        for prompt in prompts:
            request = engine.submit(prompt, sampling_params=params)
            while engine.has_work():
                engine.step()
            actual.append((request.cached_tokens, request.take_result()))
    assert [output for _, output in actual] == expected
    assert actual[2][0] == 4096
    assert runner.speculative_stats()["acceptance_rate"] > 0.85


def test_mtp_proposal_matches_dense_reference(model, monkeypatch):
    """The cached MTP state for the next anchor equals a no-cache MTP pass."""

    prompt = _prompt(131, seed=21)
    runner = _runner(model, 1)
    proposals = {}
    release = runner.release

    def keep(sequence_ids):
        proposals.update(runner._drafts)
        release(sequence_ids)

    monkeypatch.setattr(runner, "release", keep)
    params = SamplingParams(max_output_tokens=6, temperature=0.0)
    output = _generate(runner, [prompt], params)[0]
    (token, hidden), *_ = proposals.values()

    config = model.config
    sequence = prompt + output
    target = model(
        mx.array([sequence]),
        mx.array([0]),
        model.initial_state(1),
        (None,) * config.full_layer_count,
    )
    # Pairs (h[p], token[p + 1]) for every p before the final anchor.
    dense, _, _ = model.mtp(
        model.embed_tokens(mx.array([sequence[1:]])),
        target.hidden[:, :-1],
        mx.array([0]),
        None,
    )
    np.testing.assert_allclose(
        np.array(hidden), np.array(dense[0, -1]), atol=3e-2, rtol=3e-2
    )
    assert int(token) == int(mx.argmax(model.logits(dense[0, -1])))


@pytest.mark.parametrize("context,prompt_length", [(256, 240), (256, 250), (384, 254)])
def test_speculation_runs_to_the_context_limit(model, context, prompt_length):
    prompts = [_prompt(prompt_length, seed=3)]
    params = SamplingParams(max_output_tokens=context - prompt_length, temperature=0.0)
    expected = _generate(_runner(model, 0), prompts, params, context=context)
    runner = _with_oracle(_runner(model, 3), prompts, expected)
    assert _generate(runner, prompts, params, context=context) == expected
    assert len(expected[0]) == context - prompt_length
