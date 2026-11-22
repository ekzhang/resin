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

from _attention_pages import dense_reference  # noqa: E402
from test_inkling2_model import tiny_config, write_checkpoint  # noqa: E402

from resin import Engine, SamplingParams  # noqa: E402
from resin_models_mlx.inkling2 import runner as runner_module  # noqa: E402
from resin_models_mlx.inkling2.cache import (  # noqa: E402
    GLOBAL_GROUP,
    LOCAL_GROUP,
    PAGE_TOKENS,
)
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
        self.positions = batch.positions[tokens]
        self.sequence_ids = batch.sequence_ids[rows].tolist()
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
        for row, sequence in enumerate(self.sequence_ids):
            context = int(self.positions[row, 0])
            first = max(0, context - (self.window - 1)) if group == LOCAL_GROUP else 0
            if first == context:
                continue
            keys, values = history.gather(slot, row)
            hk = np.array(keys.view(mx.uint16))[0]
            hv = np.array(values.view(mx.uint16))[0]
            start = int(history.starts[row]) * PAGE_TOKENS
            assert start <= first and start + len(hk) == context
            for position in range(first, context):
                expected = self.own.get((sequence, position, group, slot))
                if expected is None:
                    expected = self.shared[self.key(sequence, position, group, slot)]
                assert np.array_equal(hk[position - start], expected[0]), position
                assert np.array_equal(hv[position - start], expected[1]), position
                self.checked += 1
        k_rows = np.array(k.astype(mx.bfloat16).view(mx.uint16))
        v_rows = np.array(v.astype(mx.bfloat16).view(mx.uint16))
        for row, sequence in enumerate(self.sequence_ids):
            for column, position in enumerate(self.positions[row].tolist()):
                written = (k_rows[row, column], v_rows[row, column])
                self.own[int(sequence), position, group, slot] = written
                self.shared.setdefault(
                    self.key(int(sequence), position, group, slot), written
                )

    def reference(self, group, slot, q, k, v, bias, scale, window):
        """Unsliced dense attention over every mirrored key of each row."""
        positions = self.positions
        rows, tokens = positions.shape
        longest = int(positions.max()) + 1
        kv_heads, dim = k.shape[2], k.shape[3]
        keys = np.zeros((rows, longest, kv_heads * dim), np.uint16)
        values = np.zeros_like(keys)
        for row, sequence in enumerate(self.sequence_ids):
            for position in range(int(positions[row, -1]) + 1):
                written = self.own.get((int(sequence), position, group, slot))
                if written is None:
                    written = self.shared[
                        self.key(int(sequence), position, group, slot)
                    ]
                keys[row, position] = written[0].reshape(-1)
                values[row, position] = written[1].reshape(-1)
        shape = (rows, longest, kv_heads, dim)
        return dense_reference(
            q,
            mx.array(keys).view(mx.bfloat16).reshape(shape),
            mx.array(values).view(mx.bfloat16).reshape(shape),
            mx.array(positions, mx.int32),
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
            group, slot, q, k, v, bias, scale, window
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


def test_serving_thread_can_use_weights_loaded_on_another_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    # The HTTP server steps the Engine on its own actor thread; lazily loaded
    # safetensors arrays would need the loading thread's CPU stream there.
    write_checkpoint(tmp_path, tiny_config())
    weights = load_model(tmp_path)
    runner = _runner(weights)
    with Engine(runner, context_length=CONTEXT) as engine:
        with ThreadPoolExecutor(max_workers=1) as executor:
            outputs = executor.submit(_generate, engine, [_prompt(20, seed=50)], 4)
            assert len(outputs.result(timeout=120)[0]) == 4


@pytest.mark.parametrize("window,extent", [(300, 200), (200, 300)])
def test_multi_page_window_and_extent(tmp_path, monkeypatch, window, extent):
    from dataclasses import replace

    # Windows and bias extents spanning several pages, unequal to each other,
    # with chunks larger than a page and a prefix hit inside the window.
    config = replace(tiny_config(), sliding_window_size=window, rel_extent=extent)
    write_checkpoint(tmp_path, config)
    weights = load_model(tmp_path)
    shared = _prompt(700, seed=60)
    prompts = [_prompt(n, seed) for seed, n in enumerate((5, 257, 640), start=61)]
    prompts += [shared + _prompt(30, seed=64), shared + _prompt(90, seed=65)]
    runner = Inkling2MlxRunner(
        weights,
        cache_slab_capacity=512,
        max_seq_len=2048,
        prefill_chunk_size=256,
        workspace_bytes=64 << 20,
    )
    mirror = _Mirror(prompts, window)
    _instrument(monkeypatch, runner, mirror)
    with Engine(
        runner, context_length=2048, enable_overlap=False, prefix_cache=True
    ) as engine:
        assert all(len(output) == 24 for output in _generate(engine, prompts[:4], 24))
        _generate(engine, prompts[4:], 24)
    assert mirror.attention_checks > 400


def test_padded_vocabulary_ids_and_small_caches_are_rejected(weights):
    config = weights[0]
    runner = _runner(weights)
    # Embedding rows past the unpadded vocabulary are padding with no logits.
    assert runner.vocab_size == config.unpadded_vocab_size < config.vocab_size
    with Engine(runner, context_length=CONTEXT) as engine:
        padded = [1, 2, config.unpadded_vocab_size, 3]
        params = SamplingParams(max_output_tokens=2, temperature=0.0)
        # Rejected while validating the plan, before any cache write.
        with pytest.raises(ValueError, match="out of range"):
            engine.submit(padded, sampling_params=params)
            while engine.has_work():
                engine.step()
        assert len(runner._states) == 0
    with pytest.raises(ValueError, match="fewer than one"):
        Inkling2MlxRunner(
            weights,
            cache_slab_capacity=2,
            max_seq_len=CONTEXT,
            prefill_chunk_size=CHUNK,
            workspace_bytes=64 << 20,
        )


def test_unseen_lengths_and_sampling_reuse_compiled_kernels(weights, monkeypatch):
    from resin_models_mlx import fused_norm, paged_attention
    from resin_models_mlx.inkling2 import attention, kernels, quantization

    # Record every custom Metal kernel variant: its factory, template and dtypes.
    variants: set = set()
    for module in (attention, kernels, quantization, paged_attention, fused_norm):
        for name in dir(module):
            factory = getattr(module, name)
            if not (
                name.startswith("_") and name.endswith("_kernel") and callable(factory)
            ):
                continue

            def recording(*args, factory=factory, name=name, **options):
                kernel = factory(*args, **options)

                def call(**launch):
                    dtypes = tuple(
                        str(getattr(x, "dtype", type(x))) for x in launch["inputs"]
                    )
                    variants.add(
                        (name, args, tuple(launch.get("template", ())), dtypes)
                    )
                    return kernel(**launch)

                return call

            monkeypatch.setattr(module, name, recording)

    def run(engine, lengths, params):
        requests = [
            engine.submit(_prompt(n, seed), sampling_params=params)
            for seed, n in enumerate(lengths, start=80)
        ]
        while engine.has_work():
            engine.step()
        for request in requests:
            request.take_result()

    greedy = SamplingParams(max_output_tokens=6, temperature=0.0)
    with Engine(
        _runner(weights), context_length=CONTEXT, max_running_requests=4
    ) as engine:
        # Warm the bounded regimes: prefill, decode at each batch size, sampling.
        for batch in range(1, 5):
            run(engine, [40 + 9 * i for i in range(batch)], greedy)
        run(
            engine,
            [50],
            SamplingParams(max_output_tokens=6, temperature=0.7, top_p=0.9, seed=1),
        )
        run(
            engine,
            [50],
            SamplingParams(max_output_tokens=6, temperature=0.7, top_k=5, seed=2),
        )
        warm = set(variants)
        # Unseen prompt and chunk lengths, contexts past the window and page,
        # batch mixes and sampling values.
        run(engine, [173, 3, 301], greedy)
        run(
            engine,
            [77, 129],
            SamplingParams(max_output_tokens=9, temperature=1.3, top_k=3, seed=4),
        )
        run(
            engine,
            [250],
            SamplingParams(max_output_tokens=20, temperature=0.31, top_p=0.6, seed=5),
        )
    # Every kernel ran during warmup (the GEMV needs widths of 256 or more).
    assert {variant[0] for variant in warm} == {
        "_activate_kernel",
        "_combine_kernel",
        "_convolution_kernel",
        "_kernel",
        "_mix_kernel",
        "_mxfp8_kernel",
        "_rms_kernel",
        "_nvfp4_kernel",
        "_prefill_kernel",
        "_route_kernel",
    }
    assert variants == warm, sorted(map(str, variants - warm))
