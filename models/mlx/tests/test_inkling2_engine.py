"""Inkling 2 MLX runner through the Engine: cache plumbing, overlap and prefix reuse.

Random tiny Inkling models are numerically chaotic (FP4 activations amplify one
ULP into different tokens), so these tests check exact invariants instead of
comparing against a separately rounded dense model: every history row the arena
returns is byte-identical to the row written for that token prefix, and runs
that perform identical arithmetic produce identical tokens.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("torch")

from test_inkling2_model import tiny_config, write_checkpoint  # noqa: E402

from resin import Engine, SamplingParams  # noqa: E402
from resin_models_mlx.inkling2 import runner as runner_module  # noqa: E402
from resin_models_mlx.inkling2.attention import dense_attention  # noqa: E402
from resin_models_mlx.inkling2.cache import GLOBAL_GROUP, LOCAL_GROUP  # noqa: E402
from resin_models_mlx.inkling2.checkpoint import load_model  # noqa: E402
from resin_models_mlx.inkling2.runner import Inkling2MlxRunner  # noqa: E402

CHUNK = 128
CONTEXT = 1024


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    root = tmp_path_factory.mktemp("inkling2")
    write_checkpoint(root, tiny_config())
    return load_model(root)


def _runner(weights, slabs: int = 256) -> Inkling2MlxRunner:
    return Inkling2MlxRunner(
        weights,
        cache_slab_capacity=slabs,
        max_seq_len=CONTEXT,
        prefill_chunk_size=CHUNK,
        workspace_bytes=64 << 20,
        unembed_f32=False,
    )


def _prompt(length: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).integers(0, 300, size=length).tolist()


def _generate(engine, prompts, count):
    params = SamplingParams(max_output_tokens=count, temperature=0.0)
    requests = [engine.submit(prompt, sampling_params=params) for prompt in prompts]
    while engine.has_work():
        engine.step()
    return [list(request.take_result()) for request in requests]


class _Mirror:
    """Reference K/V by token prefix, checked against every arena history read."""

    def __init__(self, prompts, window):
        self.prompts = prompts
        self.window = window
        self.tokens: dict[int, list[int]] = {}
        # A request's own writes, and the first write of each token prefix,
        # which is what a prefix hit restores.
        self.own: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.shared: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.checked = 0
        self.attention_checks = 0

    def start_step(self, plan, rows, tokens):
        batch = plan.batch
        for index, row in enumerate(rows.tolist()):
            sequence = int(batch.sequence_ids[row])
            context = int(batch.context_lengths[row])
            step = batch.token_ids[tokens[index]].tolist()
            known = self.tokens.get(sequence)
            if known is None:
                # A prefix hit starts mid-prompt: recover its restored tokens.
                known = next(
                    p[:context]
                    for p in self.prompts
                    if p[context : context + len(step)] == step
                )
                self.tokens[sequence] = known
            assert len(known) == context
            known += step

    def key(self, sequence, position, group, slot):
        return (tuple(self.tokens[sequence][: position + 1]), group, slot)

    def check(self, store, group, slot, k, v):
        history = store.history[group]
        contexts = store.contexts
        if history is not None:
            keys, values, key_positions, key_valid = history.gather(slot)
            hk = np.array(keys.view(mx.uint16))
            hv = np.array(values.view(mx.uint16))
            positions, valid = np.array(key_positions), np.array(key_valid)
            for row, sequence in enumerate(store.sequence_ids.tolist()):
                present = {}
                for column in np.flatnonzero(valid[row]):
                    present[int(positions[row, column])] = column
                context = int(contexts[row])
                first = (
                    max(0, context - (self.window - 1)) if group == LOCAL_GROUP else 0
                )
                needed = range(first, context)
                assert set(needed) <= present.keys(), (group, slot, context)
                for position in needed:
                    expected = self.own.get((int(sequence), position, group, slot))
                    if expected is None:
                        expected = self.shared[
                            self.key(int(sequence), position, group, slot)
                        ]
                    column = present[position]
                    assert np.array_equal(hk[row, column], expected[0]), position
                    assert np.array_equal(hv[row, column], expected[1]), position
                    self.checked += 1
        k_rows = np.array(k.astype(mx.bfloat16).view(mx.uint16))
        v_rows = np.array(v.astype(mx.bfloat16).view(mx.uint16))
        positions = np.array(store.positions)
        for row, sequence in enumerate(store.sequence_ids.tolist()):
            for column, position in enumerate(positions[row].tolist()):
                written = (k_rows[row, column], v_rows[row, column])
                self.own[int(sequence), position, group, slot] = written
                self.shared.setdefault(
                    self.key(int(sequence), position, group, slot), written
                )

    def reference(self_, store, group, slot, q, k, v, bias, scale, window):
        """Unsliced dense attention over every mirrored key of each row."""
        positions = np.array(store.positions)
        rows, tokens = positions.shape
        longest = int(positions.max()) + 1
        kv_heads, dim = k.shape[2], k.shape[3]
        keys = np.zeros((rows, longest, kv_heads * dim), np.uint16)
        values = np.zeros_like(keys)
        valid = np.zeros((rows, longest), bool)
        for row, sequence in enumerate(store.sequence_ids.tolist()):
            for position in range(int(positions[row, -1]) + 1):
                written = self_.own.get((int(sequence), position, group, slot))
                if written is None:
                    written = self_.shared[
                        self_.key(int(sequence), position, group, slot)
                    ]
                keys[row, position] = written[0].reshape(-1)
                values[row, position] = written[1].reshape(-1)
                valid[row, position] = True
        shape = (rows, longest, kv_heads, dim)
        return dense_attention(
            q,
            mx.array(keys).view(mx.bfloat16).reshape(shape),
            mx.array(values).view(mx.bfloat16).reshape(shape),
            mx.broadcast_to(mx.arange(longest)[None], (rows, longest)).astype(mx.int32),
            mx.array(valid),
            store.positions,
            bias,
            scale,
            window if group == LOCAL_GROUP else None,
        )


def _instrument(monkeypatch, runner, mirror):
    launch = runner._launch

    def recorded_launch(plan, rows, tokens, token_ids=None):
        assert token_ids is None, "plumbing checks need plan token IDs (no overlap)"
        mirror.start_step(plan, rows, tokens)
        return launch(plan, rows, tokens, token_ids)

    monkeypatch.setattr(runner, "_launch", recorded_launch)
    attend = runner_module._StepStore.attend

    def checked_attend(self, local, slot, q, k, v, bias, scale, window):
        group = LOCAL_GROUP if local else GLOBAL_GROUP
        mirror.check(self, group, slot, k, v)
        output, lse = attend(self, local, slot, q, k, v, bias, scale, window)
        expected, expected_lse = mirror.reference(
            self, group, slot, q, k, v, bias, scale, window
        )
        np.testing.assert_allclose(
            np.array(output), np.array(expected), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.array(lse), np.array(expected_lse), rtol=1e-4, atol=1e-4
        )
        mirror.attention_checks += 1
        return output, lse

    monkeypatch.setattr(runner_module._StepStore, "attend", checked_attend)


def test_history_reads_return_exactly_what_was_written(weights, monkeypatch):
    config = weights[0]
    # Prompts straddle page, chunk and window boundaries; decode crosses more.
    prompts = [_prompt(n, seed) for seed, n in enumerate((5, 127, 128, 129, 300))]
    runner = _runner(weights)
    mirror = _Mirror(prompts, config.sliding_window_size)
    _instrument(monkeypatch, runner, mirror)
    with Engine(runner, context_length=CONTEXT, enable_overlap=False) as engine:
        outputs = _generate(engine, prompts, 20)
    assert all(len(output) == 20 for output in outputs)
    assert mirror.checked > 10_000
    assert mirror.attention_checks > 400
    assert len(runner._states) == 0


def test_prefix_hit_reads_restored_history_exactly(weights, monkeypatch):
    config = weights[0]
    shared = _prompt(400, seed=11)
    prompts = [shared + _prompt(30, seed=12), shared + _prompt(50, seed=13)]
    runner = _runner(weights)
    mirror = _Mirror(prompts, config.sliding_window_size)
    _instrument(monkeypatch, runner, mirror)
    contexts = []
    launch = runner._launch

    def first_context(plan, rows, tokens, token_ids=None):
        contexts.append(int(plan.batch.context_lengths[rows[0]]))
        return launch(plan, rows, tokens, token_ids)

    monkeypatch.setattr(runner, "_launch", first_context)
    with Engine(
        runner, context_length=CONTEXT, enable_overlap=False, prefix_cache=True
    ) as engine:
        _generate(engine, prompts[:1], 4)
        contexts.clear()
        _generate(engine, prompts[1:], 4)
    # The second prompt restored a snapshot inside the shared prefix.
    assert 0 < contexts[0] <= len(shared) and contexts[0] % CHUNK == 0
    assert mirror.checked > 1_000


def test_cached_continuation_matches_fresh_and_overlap_matches_sync(weights):
    shared = _prompt(384, seed=21)
    prompt = shared + _prompt(40, seed=22)
    with Engine(
        _runner(weights), context_length=CONTEXT, enable_overlap=False
    ) as engine:
        fresh = _generate(engine, [prompt], 24)
    with Engine(_runner(weights), context_length=CONTEXT, prefix_cache=True) as engine:
        _generate(engine, [shared + [1, 2, 3]], 2)
        cached = _generate(engine, [prompt], 24)
    assert cached == fresh
    prompts = [_prompt(n, seed) for seed, n in enumerate((7, 130, 260), start=30)]
    with Engine(
        _runner(weights), context_length=CONTEXT, enable_overlap=False
    ) as engine:
        sync = _generate(engine, prompts, 24)
    runner = _runner(weights)
    with Engine(runner, context_length=CONTEXT, enable_overlap=True) as engine:
        overlapped = _generate(engine, prompts, 24)
        assert engine.overlap_stats[1] > 0
    assert overlapped == sync
    assert len(runner._states) == 0


def test_cancel_releases_state_and_context_limit_is_enforced(weights):
    runner = _runner(weights)
    with Engine(runner, context_length=CONTEXT) as engine:
        params = SamplingParams(max_output_tokens=50, temperature=0.0)
        request = engine.submit(_prompt(140, seed=40), sampling_params=params)
        for _ in range(4):
            engine.step()
        assert engine.cancel(request)
        while engine.has_work():
            engine.step()
        assert len(runner._states) == 0
        # The final sampled token is never stored, so a prompt three short of
        # the limit yields four tokens; the Engine rejects a fifth.
        long_prompt = _prompt(CONTEXT - 3, seed=41)
        assert len(_generate(engine, [long_prompt], 4)[0]) == 4
        with pytest.raises(ValueError, match="context"):
            _generate(engine, [long_prompt], 5)
