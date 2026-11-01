from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from resin import Engine, SamplingParams, SchedulerConfig  # noqa: E402
from resin.models.qwen3_5.config import Qwen35Config  # noqa: E402
from resin_models_mlx.qwen3_5_moe.cache import (  # noqa: E402
    KV_GROUP,
    PAGE_TOKENS,
    PREFILL_CHUNK_SIZE,
    STATE_GROUP,
    CacheArena,
    CacheGeometry,
)
from resin_models_mlx.qwen3_5_moe.model import (  # noqa: E402
    LinearState,
    Qwen35MoeModel,
)
from resin_models_mlx.qwen3_5_moe.runner import Qwen3_5MoeMlxRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _tiny_model_workspace(monkeypatch):
    # The 64-wide test model does not need the production runner's 4 GiB
    # workspace reserve, which exceeds the budget on small Apple CI runners.
    monkeypatch.setattr(
        "resin_models_mlx.qwen3_5_moe.runner._WORKSPACE_BYTES", 64 << 20
    )


def _text_config() -> dict[str, Any]:
    layers = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(8)
    ]
    return {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 256,
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 8,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 16384,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 2],
            "mrope_interleaved": True,
        },
        "layer_types": layers,
        "full_attention_interval": 4,
        "mtp_num_hidden_layers": 0,
        "eos_token_id": 1,
        "bos_token_id": 1,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 48,
    }


def _model(seed: int = 0, dtype=None, *, all_experts: bool = False) -> Qwen35MoeModel:
    mx.random.seed(seed)
    model = Qwen35MoeModel(Qwen35Config.from_hugging_face(_text_config()))
    # Give norms and DeltaNet decay realistic, non-trivial values.
    updates = []
    for name, value in cast(list[tuple[str, Any]], tree_flatten(model.parameters())):
        if name.endswith("linear_attn.norm.weight"):
            updates.append((name, 1.0 + 0.1 * mx.random.normal(value.shape)))
        elif name.endswith("norm.weight"):
            # Qwen RMSNorms store an offset from one.
            updates.append((name, 0.1 * mx.random.normal(value.shape)))
        elif name.endswith("A_log"):
            # Slowly decaying heads remember a request's state over a whole
            # test prompt, so stale or misrouted state changes its output.
            updates.append((name, mx.random.uniform(-8.0, 1.0, value.shape)))
        elif name.endswith("dt_bias"):
            updates.append((name, 0.1 * mx.random.normal(value.shape)))
    model.load_weights(updates, strict=False)
    selected = mx.bfloat16 if dtype is None else dtype
    model.set_dtype(selected, lambda dtype: dtype == mx.float32)
    mx.eval(model.parameters())
    if all_experts:
        # Random routers have near-tied experts, so BF16 rounding can flip a
        # top-k choice; routing to every expert keeps logits continuous.
        for layer in model.layers:
            layer.mlp.top_k = model.config.num_experts
    return model


def _full_logits(model: Qwen35MoeModel, tokens: list[int]) -> mx.array:
    output = model(
        mx.array([tokens]),
        mx.array([0]),
        model.initial_state(1),
        (None,) * model.config.full_layer_count,
    )
    return model.logits(output.hidden)[0].astype(mx.float32)


def _assert_greedy(
    model: Qwen35MoeModel, prompt: list[int], generated: list[int], count: int
) -> None:
    """Each token must be a reference argmax up to BF16 tie tolerance.

    The reference is a one-pass BF16 forward, so cache layout, chunking and
    kernel choice may legitimately break near-ties. A stale state or wrong
    page moves logits far beyond this tolerance.
    """

    assert len(generated) == count
    for index, token in enumerate(generated):
        logits = _full_logits(model, prompt + generated[:index])[-1]
        assert logits[token].item() >= logits.max().item() - 0.05, (index, token)


def _generate(engine: Engine, prompts: list[list[int]], count: int):
    params = SamplingParams(max_output_tokens=count, temperature=0.0)
    requests = [engine.submit(prompt, sampling_params=params) for prompt in prompts]
    while engine.has_work():
        engine.step()
    return [request.take_result() for request in requests]


def _runner(model: Qwen35MoeModel, slabs: int = 64) -> Qwen3_5MoeMlxRunner:
    return Qwen3_5MoeMlxRunner(model, cache_slab_capacity=slabs)


def _prompt(length: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(2, 256, size=length).tolist()


def test_geometry_packs_ornith_state_into_whole_kv_units() -> None:
    raw = _text_config()
    raw.update(
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=[
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(40)
        ],
        num_experts=256,
        num_experts_per_tok=8,
        rope_parameters={"rope_theta": 1e7, "partial_rotary_factor": 0.25},
    )
    geometry = CacheGeometry.from_config(Qwen35Config.from_hugging_face(raw))

    # 30 layers of 32x128x128 FP32 state plus 3x8192 BF16 conv tails.
    assert geometry.state_units == 246
    assert geometry.slab_units == 250
    assert geometry.kv_blocks_per_slab == 25
    assert geometry.state_blocks_per_slab == 1
    assert geometry.slab_bytes == 250 * 256 * 1024
    assert geometry.unit_elements * 2 == 256 * 1024


def _mlx_lm_reference(model: Qwen35MoeModel):
    from mlx_lm.models.qwen3_5_moe import Model, ModelArgs

    reference = Model(
        ModelArgs.from_dict(
            {"model_type": "qwen3_5_moe", "text_config": _text_config()}
        )
    )
    weights = []
    for name, value in cast(list[tuple[str, Any]], tree_flatten(model.parameters())):
        prefix = (
            "language_model."
            if name.startswith("lm_head")
            else ("language_model.model.")
        )
        if name.endswith("norm.weight") and not name.endswith(
            "linear_attn.norm.weight"
        ):
            # mlx-lm folds Qwen's offset into the stored RMSNorm weight.
            value = 1.0 + value
        weights.append((prefix + name, value))
    reference.load_weights(weights)
    mx.eval(reference.parameters())
    return reference


def test_forward_matches_mlx_lm_reference() -> None:
    # mlx-lm folds Qwen's RMSNorm offset into its weight; FP32 makes folding
    # exact, so this compares the composition rather than BF16 rounding, which
    # this randomly initialized network amplifies.
    model = _model(dtype=mx.float32)
    reference = _mlx_lm_reference(model)
    tokens = _prompt(37, seed=1)

    actual = _full_logits(model, tokens)
    expected = reference(mx.array([tokens]))[0].astype(mx.float32)

    assert mx.allclose(actual, expected, atol=1e-3, rtol=1e-3).item()
    assert (mx.argmax(actual, axis=-1) == mx.argmax(expected, axis=-1)).all().item()


def test_incremental_state_matches_full_forward_exactly() -> None:
    # FP32 isolates state and history plumbing from BF16 rounding.
    model = _model(dtype=mx.float32)
    tokens = _prompt(21, seed=2)
    config = model.config
    expected = _full_logits(model, tokens)

    first = model(
        mx.array([tokens[:13]]),
        mx.array([0]),
        model.initial_state(1),
        (None,) * config.full_layer_count,
    )
    from resin_models_mlx.qwen3_5_moe.model import AttentionHistory

    history = tuple(
        AttentionHistory(key, value)
        for key, value in zip(first.keys, first.values, strict=True)
    )
    second = model(mx.array([tokens[13:]]), mx.array([13]), first.state, history)
    actual = model.logits(second.hidden)[0].astype(mx.float32)

    assert mx.allclose(actual, expected[13:], atol=1e-5, rtol=1e-5).item()


def test_engine_greedy_matches_reference_across_page_boundaries() -> None:
    model = _model(all_experts=True)
    prompts = [_prompt(length, seed=length) for length in (5, 127, 128, 129)]

    with Engine(_runner(model), context_length=8192) as engine:
        results = _generate(engine, prompts, 6)
    for prompt, generated in zip(prompts, results, strict=True):
        _assert_greedy(model, prompt, generated, 6)


@pytest.mark.parametrize("staggered", [True, False])
def test_live_state_follows_changing_batches(staggered: bool) -> None:
    model = _model(all_experts=True)
    prompts = [_prompt(length, seed=20 + length) for length in (4, 33, 90, 150)]
    counts = [3, 9, 5, 12]
    runner = _runner(model)
    with Engine(runner, context_length=8192) as engine:
        leases, continuations = [], []
        for prompt, count in zip(prompts, counts, strict=True):
            params = SamplingParams(max_output_tokens=count, temperature=0.0)
            continuations.append(engine.overlap_stats[1])
            leases.append(engine.submit(prompt, sampling_params=params))
            # Later requests join a decode batch already in flight.
            for _ in range(4 if staggered else 0):
                engine.step()
        while engine.has_work():
            engine.step()
        # The last request joins while decode steps overlap.
        assert continuations[-1] > 0 or not staggered
        # Finished requests release their live state.
        assert len(runner._states) == 0
    # Batch shapes change as requests join and leave, which may break BF16
    # ties; a row reading another row's state would not stay within them.
    for prompt, count, lease in zip(prompts, counts, leases, strict=True):
        _assert_greedy(model, prompt, lease.take_result(), count)


def test_cancel_releases_live_state_and_others_continue() -> None:
    model = _model(all_experts=True)
    prompts = [_prompt(length, seed=30 + length) for length in (20, 50, 70)]
    runner = _runner(model)
    with Engine(runner, context_length=8192) as engine:
        params = SamplingParams(max_output_tokens=10, temperature=0.0)
        leases = [engine.submit(p, sampling_params=params) for p in prompts]
        for _ in range(4):
            engine.step()
        engine.cancel(leases[1])
        assert len(runner._states) == 2
        while engine.has_work():
            engine.step()
        assert len(runner._states) == 0
    for index in (0, 2):
        _assert_greedy(model, prompts[index], leases[index].take_result(), 10)


def test_live_states_regroup_rows_held_by_different_batches() -> None:
    model = _model()
    runner = _runner(model)
    states = runner._states

    def plan(sequences: list[int], contexts: list[int], restores=()) -> Any:
        state = SimpleNamespace(restore_copies=restores, checkpoint_slots=())
        return SimpleNamespace(
            batch=SimpleNamespace(
                sequence_ids=np.array(sequences), context_lengths=np.array(contexts)
            ),
            groups={STATE_GROUP: SimpleNamespace(state=state)},
        )

    def marked(rows: list[int]) -> LinearState:
        # Every element of a row's state holds that row's marker.
        def fill(value: mx.array) -> mx.array:
            marks = mx.array(rows).reshape(-1, *[1] * (value.ndim - 1))
            return (value + marks).astype(value.dtype)

        state = model.initial_state(len(rows))
        return LinearState(
            tuple(map(fill, state.recurrent)), tuple(map(fill, state.convolution))
        )

    runner._cache.write_states([runner._cache.state_start(3, 0)], marked([7]))
    states.commit(
        plan([1, 3], [9, 9]), np.arange(2), np.array([10, 10]), marked([1, 3])
    )
    states.commit(
        plan([5, 2], [19, 19]), np.arange(2), np.array([20, 20]), marked([5, 2])
    )
    # Sequences 1 and 2 come from different batches whose other rows left. A
    # new request restores the null slab and another a snapshot in slab 3.
    gathered = states.gather(
        plan([4, 1, 2, 6], [0, 10, 20, 4096], restores=((0, 0, 0), (3, 3, 0))),
        np.arange(4),
    )
    expected = mx.array([[0], [1], [2], [7]])
    for value in (*gathered.recurrent, *gathered.convolution):
        assert (value.reshape(4, -1) == expected).all().item()


def test_retracted_requests_replay_from_fresh_state() -> None:
    model = _model(all_experts=True)
    prompts = [_prompt(500, seed=40), _prompt(500, seed=41)]
    runner = _runner(model, slabs=4)
    released = []
    release = runner.release

    def recording(ids):
        released.extend(int(i) for i in ids)
        release(ids)

    cast(Any, runner).release = recording
    config = SchedulerConfig(schedule_conservativeness=0.0)
    with Engine(runner, context_length=8192, scheduler_config=config) as engine:
        results = _generate(engine, prompts, 40)
    # Some request was released and resumed before it finished.
    assert len(released) > len(set(released))
    # Replay prefills generated tokens that were first decoded, so compare with
    # the reference within BF16 tie tolerance rather than token for token.
    for prompt, generated in zip(prompts, results, strict=True):
        _assert_greedy(model, prompt, generated, 40)


def _decode_with_stops(
    overlap: bool, model: Qwen35MoeModel, temperature: float, stops: tuple[int, ...]
) -> tuple[list[list[int]], tuple[int, int]]:
    prompts = [_prompt(length, seed=60 + length) for length in (9, 40, 100)]
    counts = [16, 7, 20]
    with Engine(_runner(model), context_length=8192, enable_overlap=overlap) as engine:
        leases = [
            engine.submit(
                prompt,
                sampling_params=SamplingParams(
                    max_output_tokens=count,
                    temperature=temperature,
                    seed=row,
                    stop_token_ids=stops,
                ),
            )
            for row, (prompt, count) in enumerate(zip(prompts, counts, strict=True))
        ]
        while engine.has_work():
            engine.step()
        stats = engine.overlap_stats
    return [lease.take_result() for lease in leases], stats


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_overlapped_decode_matches_synchronous_steps(temperature: float) -> None:
    model = _model(all_experts=True)
    synchronous, stats = _decode_with_stops(False, model, temperature, ())
    assert stats == (0, 0)
    # The first row stops partway through while its next step is in flight.
    stop = synchronous[0][9]
    synchronous, _ = _decode_with_stops(False, model, temperature, (stop,))
    overlapped, (_, continuations) = _decode_with_stops(
        True, model, temperature, (stop,)
    )
    assert continuations > 0
    assert len(synchronous[0]) < 16
    assert overlapped == synchronous


def test_snapshot_written_during_decode_serves_the_next_turn() -> None:
    model = _model(all_experts=True)
    prompt = _prompt(PREFILL_CHUNK_SIZE - 3, seed=8)
    with Engine(_runner(model), context_length=16384, prefix_cache=True) as engine:
        # Decode crosses the 4096-token checkpoint and publishes it from the
        # second row of the batch.
        _, first = _generate(engine, [_prompt(30, seed=10), prompt], 8)
        turn = prompt + first[:3] + _prompt(5, seed=9)
        request = engine.submit(
            turn, sampling_params=SamplingParams(max_output_tokens=4, temperature=0.0)
        )
        while engine.has_work():
            engine.step()
        assert request.cached_tokens == PREFILL_CHUNK_SIZE
        cached = request.take_result()
    # The snapshot's last positions were decoded, not prefilled, which may
    # break BF16 ties against a fresh run; a wrong snapshot would not.
    _assert_greedy(model, turn, cached, 4)


def test_chunked_prefill_crosses_state_checkpoint() -> None:
    model = _model(all_experts=True)
    prompt = _prompt(PREFILL_CHUNK_SIZE + 37, seed=3)

    with Engine(_runner(model), context_length=16384) as engine:
        (generated,) = _generate(engine, [prompt], 4)
    _assert_greedy(model, prompt, generated, 4)


def test_prefix_cache_hit_matches_fresh_run() -> None:
    model = _model(all_experts=True)
    shared = _prompt(PREFILL_CHUNK_SIZE, seed=4)
    suffix = _prompt(9, seed=5)

    with Engine(_runner(model), context_length=16384) as engine:
        (fresh,) = _generate(engine, [shared + suffix], 5)
    with Engine(_runner(model), context_length=16384, prefix_cache=True) as engine:
        # The shared prefix's checkpoint is published from the second row of
        # a prefill chunk, after the rest of a longer prompt.
        prompts = [_prompt(100, seed=6), _prompt(4146, seed=7), shared + [3] * 50]
        _generate(engine, prompts, 1)
        request = engine.submit(
            shared + suffix,
            sampling_params=SamplingParams(max_output_tokens=5, temperature=0.0),
        )
        while engine.has_work():
            engine.step()
        assert request.cached_tokens == PREFILL_CHUNK_SIZE
        # The restored prefix runs exactly the computation of a fresh request.
        assert request.take_result() == fresh
    _assert_greedy(model, shared + suffix, fresh, 5)


def test_arena_parts_serve_the_same_tokens(monkeypatch) -> None:
    import resin_models_mlx.qwen3_5_moe.cache as cache_module

    model = _model(all_experts=True)
    shared = _prompt(PREFILL_CHUNK_SIZE + 104, seed=11)

    def run():
        runner = _runner(model)
        with Engine(runner, context_length=16384, prefix_cache=True) as engine:
            # Requests span parts: prefill gathers history across them,
            # decode reads pages from several, and the checkpoint published
            # at 4096 tokens serves the last prompt from its part.
            prompts = [_prompt(n, seed=n) for n in (129, 700, 1500)] + [shared]
            generated = _generate(engine, prompts, 8)
            request = engine.submit(
                shared[:PREFILL_CHUNK_SIZE] + _prompt(30, seed=12),
                sampling_params=SamplingParams(max_output_tokens=8, temperature=0),
            )
            while engine.has_work():
                engine.step()
            assert request.cached_tokens == PREFILL_CHUNK_SIZE
        return runner._cache, [*generated, request.take_result()]

    arena, expected = run()
    assert len(arena.parts) == 1
    monkeypatch.setattr(cache_module, "PART_BYTES", 1)
    arena, actual = run()
    # Each part holds five 512-token slabs, so every long request spans parts.
    assert len(arena.parts) == 13
    assert actual == expected


def _arena(monkeypatch, slabs: int, part_bytes: int) -> CacheArena:
    monkeypatch.setattr("resin_models_mlx.qwen3_5_moe.cache.PART_BYTES", part_bytes)
    geometry = CacheGeometry.from_config(Qwen35Config.from_hugging_face(_text_config()))
    return CacheArena(geometry.layout(slabs), geometry)


def test_decode_binds_only_the_parts_its_pages_occupy(monkeypatch) -> None:
    arena = _arena(monkeypatch, 64, 1)
    assert len(arena.parts) == 13
    assert arena.part_units == 40
    tables = {
        0: ([12, 3], [1, 0]),
        1: ([], []),
        2: ([12], [3]),
    }
    group = SimpleNamespace(
        block_slabs=lambda row: np.array(tables[row][0], dtype=np.int64),
        block_slots=lambda row: np.array(tables[row][1], dtype=np.int64),
    )
    plan = SimpleNamespace(groups={KV_GROUP: group})
    bound = []
    units = arena._units

    def spy(part: int) -> mx.array:
        bound.append(part)
        return units(part)

    monkeypatch.setattr(arena, "_units", spy)
    history = arena.paged_history(
        cast(Any, plan), np.arange(3), np.array([200, 0, 100])
    )

    assert bound == [0, 2]
    # One kernel serves every batch, so the bound parts repeat to a fixed count.
    assert len(history.parts) == len(arena.parts)
    # Slabs 12 and 3 are in parts 2 and 0, renumbered 1 and 0; padding reads
    # nothing.
    assert history.pages.tolist() == [[58, 24], [0, 0], [62, 0]]
    assert history.part_units == 40

    bound.clear()
    empty = arena.paged_history(cast(Any, plan), np.array([1]), np.array([0]))
    assert bound == [0]
    assert len(empty.parts) == len(arena.parts)


def test_states_round_trip_across_parts_in_row_order(monkeypatch) -> None:
    arena = _arena(monkeypatch, 64, 1)
    geometry = arena.geometry
    mx.random.seed(0)
    state = LinearState(
        tuple(
            mx.random.normal((3, *geometry.recurrent_shape[1:]))
            for _ in range(geometry.linear_layers)
        ),
        tuple(
            mx.random.normal((3, *geometry.convolution_shape[1:])).astype(mx.bfloat16)
            for _ in range(geometry.linear_layers)
        ),
    )
    # Slabs 40, 4 and 63 lie in different parts, out of order.
    starts = [arena.state_start(slab, 0) for slab in (40, 4, 63)]
    arena.write_states(starts, state)
    mx.eval(*arena.parts)

    actual = arena.read_states(np.array(starts))

    for written, read in zip(
        (*state.recurrent, *state.convolution),
        (*actual.recurrent, *actual.convolution),
        strict=True,
    ):
        assert mx.array_equal(written, read).item()


def test_scatter_across_parts_updates_each_in_place(monkeypatch) -> None:
    arena = _arena(monkeypatch, 4095, 512 << 20)
    assert len(arena.parts) == 2
    first = arena.parts[0].shape[0] // arena.geometry.slab_units
    # One token lands in each part.
    plan = SimpleNamespace(
        groups={
            KV_GROUP: SimpleNamespace(
                append_slab_ids=np.array([1, first + 1]),
                append_slotnums=np.array([0, 0]),
                append_block_offsets=np.array([3, 4]),
            )
        }
    )
    geometry = arena.geometry
    # Every token and layer writes its own value.
    tokens = mx.array([20.0, 40.0]).reshape(2, 1, 1)
    shape = (2, geometry.kv_heads, geometry.head_dim)
    keys = tuple(
        mx.broadcast_to(tokens + layer + 1, shape).astype(mx.bfloat16)
        for layer in range(geometry.full_layers)
    )
    values = tuple(-key for key in keys)
    mx.eval(keys, values)
    mx.synchronize()
    mx.reset_peak_memory()
    active = mx.get_active_memory()

    arena.write_tokens(cast(Any, plan), keys, values, np.arange(2))
    mx.eval(*arena.parts)

    # A copied part would add half a GiB.
    assert mx.get_peak_memory() - active < 64 << 20
    for part, offset, token in ((0, 3, 20), (1, 4, 40)):
        for layer in range(geometry.full_layers):
            unit = arena.parts[part][geometry.slab_units + layer]
            assert unit[offset, 0].item() == token + layer + 1
            assert unit[PAGE_TOKENS + offset, 0].item() == -(token + layer + 1)


def test_prompt_logprobs_match_one_pass_forward() -> None:
    model = _model()
    prompts = [_prompt(37, seed=8), _prompt(21, seed=9)]
    params = SamplingParams(max_output_tokens=1, prompt_logprobs=True, temperature=0)

    with Engine(_runner(model), context_length=4096) as engine:
        requests = [engine.submit(p, sampling_params=params) for p in prompts]
        while engine.has_work():
            engine.step()
    for prompt, request in zip(prompts, requests):
        scores = request.prompt_logprobs
        assert scores is not None and scores[0] is None
        logits = _full_logits(model, prompt)[:-1]
        expected = (logits - mx.logsumexp(logits, axis=-1, keepdims=True))[
            mx.arange(len(prompt) - 1), mx.array(prompt[1:])
        ]
        actual = [item.logprob for item in scores[1:] if item is not None]
        assert actual == pytest.approx(expected.tolist(), abs=0.05)


def test_decode_crosses_kv_page_boundary() -> None:
    model = _model(all_experts=True)
    prompt = _prompt(PAGE_TOKENS - 2, seed=7)

    with Engine(_runner(model), context_length=8192) as engine:
        (generated,) = _generate(engine, [prompt], 6)
    _assert_greedy(model, prompt, generated, 6)


def _assert_checkpoint_loader_matches_transformers(layout: str) -> None:
    import json
    import tempfile

    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    from resin_models_mlx.qwen3_5_moe.checkpoint import load_model

    values = _text_config()
    torch.manual_seed(0)
    reference = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**values)).float().eval()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith("norm.weight"):
                parameter.normal_(0.0, 0.1)
            elif name.endswith("A_log"):
                parameter.uniform_(-1.0, 1.0)
    # Checkpoints store the DeltaNet decay parameters in float32.
    decay = ("A_log", "dt_bias")
    reference_decay = {
        name: tensor.clone()
        for name, tensor in reference.state_dict().items()
        if name.endswith(decay)
    }
    state = {
        name: tensor if name.endswith(decay) else tensor.to(torch.bfloat16)
        for name, tensor in reference.state_dict().items()
    }
    if layout != "text-only":
        # Multimodal exports nest the text model.
        state = {
            name.replace("model.", "model.language_model.", 1): tensor
            for name, tensor in state.items()
        }
    if layout == "per-expert":
        # Ornith-1.0 stores one tensor per expert instead of fused experts.
        for name in [n for n in state if n.endswith("mlp.experts.gate_up_proj")]:
            prefix = name.removesuffix("gate_up_proj")
            gate_up = state.pop(name)
            down = state.pop(prefix + "down_proj")
            middle = gate_up.shape[1] // 2
            for expert in range(gate_up.shape[0]):
                state[f"{prefix}{expert}.gate_proj.weight"] = gate_up[expert, :middle]
                state[f"{prefix}{expert}.up_proj.weight"] = gate_up[expert, middle:]
                state[f"{prefix}{expert}.down_proj.weight"] = down[expert]
    reference = reference.to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith(decay):
                parameter.data = parameter.data.float()
                parameter.data.copy_(reference_decay[name])
    tokens = _prompt(19, seed=8)
    with torch.no_grad():
        expected = reference(torch.tensor([tokens])).logits[0].float().numpy()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        tensors = {name: tensor.contiguous() for name, tensor in state.items()}
        if layout == "per-expert":
            # Exercise the sharded index path as well.
            names = sorted(tensors)
            shards = {
                "model-1.safetensors": names[::2],
                "model-2.safetensors": names[1::2],
            }
            for file, members in shards.items():
                save_file({name: tensors[name] for name in members}, path / file)
            weight_map = {
                name: file for file, members in shards.items() for name in members
            }
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map})
            )
        else:
            save_file(tensors, path / "model.safetensors")
        config = (
            values
            if layout == "text-only"
            else {"model_type": "qwen3_5_moe", "text_config": values}
        )
        (path / "config.json").write_text(json.dumps(config))
        model, _, _ = load_model(path, bits=None)
        for layer in model.layers:
            if hasattr(layer, "linear_attn"):
                assert layer.linear_attn.A_log.dtype == mx.float32
                assert layer.linear_attn.dt_bias.dtype == mx.float32
        actual = np.array(_full_logits(model, tokens))
    np.testing.assert_allclose(actual, expected, atol=3e-2, rtol=3e-2)
    assert (actual.argmax(-1) == expected.argmax(-1)).mean() > 0.9


@pytest.mark.parametrize("layout", ["per-expert", "fused", "text-only"])
def test_checkpoint_loader_matches_transformers_reference(layout: str) -> None:
    tests = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(tests)!r}); "
                "from test_qwen3_5_moe_mlx import "
                "_assert_checkpoint_loader_matches_transformers as check; "
                f"check({layout!r})"
            ),
        ],
        check=True,
    )


def test_cache_writes_update_the_arena_in_place() -> None:
    model = _model()
    # Keep a >1 GiB copy detector without occupying most of a small CI runner.
    runner = _runner(model, slabs=5_000)
    arena = sum(part.nbytes for part in runner._cache.parts)
    with Engine(runner, context_length=8192) as engine:
        params = SamplingParams(max_output_tokens=12, temperature=0.0)
        for seed in range(3):
            engine.submit(_prompt(300, seed=seed), sampling_params=params)
        engine.step()
        mx.synchronize()
        mx.reset_peak_memory()
        active = mx.get_active_memory()
        while engine.has_work():
            engine.step()
        growth = mx.get_peak_memory() - active
        # Overlapped decode steps also issue their writes in place.
        assert engine.overlap_stats[1] > 0
    # Prefill and decode both write; copying any part would exceed this bound.
    assert arena > 1 << 30
    assert growth < arena // 8


def test_decode_writes_every_bound_part_in_place(monkeypatch) -> None:
    part_bytes = 16 << 20
    monkeypatch.setattr("resin_models_mlx.qwen3_5_moe.cache.PART_BYTES", part_bytes)
    runner = _runner(_model(), slabs=250)
    assert len(runner._cache.parts) > 1
    bound = []
    paged_history = runner._cache.paged_history

    def spy(plan, rows, lengths):
        history = paged_history(plan, rows, lengths)
        bound.append(len({id(part) for part in history.parts}))
        return history

    monkeypatch.setattr(runner._cache, "paged_history", spy)
    params = SamplingParams(max_output_tokens=12, temperature=0.0)
    growth = 0
    with Engine(runner, context_length=8192) as engine:
        # The free list walks upward, so a later request straddles two parts.
        for seed in range(60):
            bound.clear()
            engine.submit(_prompt(2000, seed=seed), sampling_params=params)
            engine.step()
            mx.synchronize()
            mx.reset_peak_memory()
            active = mx.get_active_memory()
            while engine.has_work():
                engine.step()
            growth = max(growth, mx.get_peak_memory() - active)
            if max(bound) > 1:
                break
        assert max(bound) > 1
        assert engine.overlap_stats[1] > 0
    # A copied part shows here as at least 7 MiB; in-place writes stay under 1.
    assert growth < part_bytes // 4


def test_capacity_allows_one_buffer_per_part(monkeypatch) -> None:
    geometry = CacheGeometry.from_config(Qwen35Config.from_hugging_face(_text_config()))
    limit = 4 * geometry.slab_bytes
    device = {**mx.device_info(), "max_buffer_length": limit}
    monkeypatch.setattr(mx, "device_info", lambda: device)
    monkeypatch.setattr("resin_models_mlx.qwen3_5_moe.cache.PART_BYTES", 1)
    runner = Qwen3_5MoeMlxRunner(_model())
    # The reserved null slab counts toward the buffers.
    assert runner.cache_layout.parent_capacity == 16 * 4 - 1
    assert [part.nbytes for part in runner._cache.parts] == [limit] * 16


def test_last_part_holds_the_remaining_slabs(monkeypatch) -> None:
    geometry = CacheGeometry.from_config(Qwen35Config.from_hugging_face(_text_config()))
    arena = _arena(monkeypatch, 6, 2 * geometry.slab_bytes)
    assert [part.nbytes // geometry.slab_bytes for part in arena.parts] == [2, 2, 2, 1]


def test_context_is_capped_by_checkpoint_positions() -> None:
    model = _model()
    limit = model.config.max_position_embeddings
    with pytest.raises(ValueError, match="checkpoint's"):
        Qwen3_5MoeMlxRunner(model, cache_slab_capacity=8, max_seq_len=limit + 1)
    runner = Qwen3_5MoeMlxRunner(model, cache_slab_capacity=64, max_seq_len=300)
    assert runner.benchmark_metadata["quant"] == "preloaded"
    with Engine(runner, context_length=8192) as engine:
        engine.submit(
            _prompt(400, seed=11),
            sampling_params=SamplingParams(max_output_tokens=2, temperature=0.0),
        )
        with pytest.raises(Exception, match="300-token context"):
            while engine.has_work():
                engine.step()


def test_overlapped_decode_drains_before_the_context_cap() -> None:
    runner = Qwen3_5MoeMlxRunner(_model(), cache_slab_capacity=64, max_seq_len=300)
    params = SamplingParams(max_output_tokens=60, temperature=0.0)
    with Engine(runner, context_length=8192) as engine:
        long = engine.submit(_prompt(280, seed=1), sampling_params=params)
        other = engine.submit(_prompt(20, seed=2), sampling_params=params)
        with pytest.raises(Exception, match="300-token context"):
            while engine.has_work():
                engine.step()
        assert engine.overlap_stats[1] > 0
        # The failure surfaces from forward, so the Engine is not poisoned.
        engine.cancel(long)
        while engine.has_work():
            engine.step()
        assert len(other.take_result()) == 60


def test_query_blocked_prefill_attention_matches_one_block(monkeypatch) -> None:
    import resin_models_mlx.qwen3_5_moe.model as model_module

    model = _model(dtype=mx.float32)
    attention = model.layers[3].self_attn
    keys = mx.random.split(mx.random.key(3), 3)
    query = mx.random.normal((1, attention.heads, 37, attention.head_dim), key=keys[0])
    key = mx.random.normal(
        (1, attention.kv_heads, 100, attention.head_dim), key=keys[1]
    )
    value = mx.random.normal(
        (1, attention.kv_heads, 100, attention.head_dim), key=keys[2]
    )
    whole = attention._dense_attention(query, key, value)
    # Force blocks of three queries over the 100 keys.
    monkeypatch.setattr(model_module, "SCORE_BUDGET", attention.heads * 100 * 3)
    blocked = attention._dense_attention(query, key, value)
    assert mx.allclose(whole, blocked, atol=1e-5).item()
