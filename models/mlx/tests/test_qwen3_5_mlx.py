from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from resin import Engine, SamplingParams, SchedulerConfig  # noqa: E402
from resin.models.qwen3_5.config import Qwen35Config  # noqa: E402
from resin_models_mlx.qwen3_5.cache import (  # noqa: E402
    PAGE_TOKENS,
    PREFILL_CHUNK_SIZE,
    CacheGeometry,
)
from resin_models_mlx.qwen3_5.model import (  # noqa: E402
    Qwen35Model,
)
from resin_models_mlx.qwen3_5.runner import Qwen35MlxRunner  # noqa: E402


def _text_config(
    *, dense: bool = False, tie_word_embeddings: bool = False
) -> dict[str, Any]:
    layers = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(8)
    ]
    config = {
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
        "tie_word_embeddings": tie_word_embeddings,
    }
    if dense:
        config.update(
            model_type="qwen3_5_text", intermediate_size=128, bos_token_id=None
        )
        for key in (
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ):
            config.pop(key)
    return config


def _model(
    seed: int = 0,
    dtype=None,
    *,
    all_experts: bool = False,
    dense: bool = False,
    tie_word_embeddings: bool = False,
) -> Qwen35Model:
    mx.random.seed(seed)
    model = Qwen35Model(
        Qwen35Config.from_hugging_face(
            _text_config(dense=dense, tie_word_embeddings=tie_word_embeddings)
        )
    )
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
    if all_experts and model.config.is_moe:
        # Random routers have near-tied experts, so BF16 rounding can flip a
        # top-k choice; routing to every expert keeps logits continuous.
        for layer in model.layers:
            layer.mlp.top_k = model.config.num_experts
    return model


def _full_logits(model: Qwen35Model, tokens: list[int]) -> mx.array:
    output = model(
        mx.array([tokens]),
        mx.array([0]),
        model.initial_state(1),
        (None,) * model.config.full_layer_count,
    )
    return model.logits(output.hidden)[0].astype(mx.float32)


def _assert_greedy(
    model: Qwen35Model, prompt: list[int], generated: list[int], count: int
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


def _runner(model: Qwen35Model, slabs: int = 64) -> Qwen35MlxRunner:
    return Qwen35MlxRunner(model, cache_slab_capacity=slabs, workspace_bytes=64 << 20)


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


def _mlx_lm_reference(model: Qwen35Model):
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


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_dense_output_projection_respects_embedding_tying(tied: bool) -> None:
    model = _model(dtype=mx.float32, dense=True, tie_word_embeddings=tied)
    hidden = mx.random.normal((2, 3, model.config.hidden_size))
    parameters = dict(tree_flatten(model.parameters()))
    assert ("lm_head.weight" in parameters) is not tied

    # An independent head keeps its own weights; tied logits follow later
    # embedding updates without maintaining a second parameter copy.
    before = model.logits(hidden)
    mx.eval(before)
    model.embed_tokens.weight = model.embed_tokens.weight + 0.25
    after = model.logits(hidden)
    if tied:
        expected = hidden @ model.embed_tokens.weight.T
        assert mx.allclose(after, expected, atol=1e-5).item()
        assert not mx.allclose(before, after).item()
    else:
        assert mx.array_equal(before, after).item()


def test_dense_tied_logits_support_quantized_embeddings() -> None:
    model = _model(dense=True, tie_word_embeddings=True)
    hidden = mx.random.normal((2, 3, model.config.hidden_size)).astype(mx.bfloat16)
    expected = model.logits(hidden)
    mx.eval(expected)
    nn.quantize(
        model,
        bits=8,
        group_size=64,
        class_predicate=lambda path, _module: path == "embed_tokens",
    )
    assert "lm_head.weight" not in dict(tree_flatten(model.parameters()))
    np.testing.assert_allclose(
        np.array(model.logits(hidden).astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=0.05,
        rtol=0.03,
    )


@pytest.mark.parametrize(
    "precision,rows", [("fp32", 1), ("bf16", 4), ("int8", 1), ("int8", 5)]
)
def test_preparation_preserves_forward_bits_and_checkpoint_parameters(precision, rows):
    model = _model(
        seed=87, dtype=mx.float32 if precision == "fp32" else mx.bfloat16, dense=True
    )
    if precision == "int8":
        nn.quantize(model, bits=8, group_size=64)
    tokens = mx.array(_prompt(rows, seed=86), mx.int32)[:, None]
    state, history = model.initial_state(rows), (None,) * model.config.full_layer_count

    def arrays(output):
        return (
            output.hidden,
            model.logits(output.hidden),
            *output.keys,
            *output.values,
            *output.state.recurrent,
            *output.state.convolution,
        )

    def bits(value):
        integer = mx.uint16 if value.dtype in (mx.bfloat16, mx.float16) else mx.uint32
        return np.array(value.view(integer))

    expected = arrays(model(tokens, mx.zeros(rows, mx.int32), state, history))
    mx.eval(expected)
    before = {name: bits(value) for name, value in tree_flatten(model.parameters())}
    model.prepare_for_inference()
    actual = arrays(model(tokens, mx.zeros(rows, mx.int32), state, history))
    mx.eval(actual)
    for value, reference in zip(actual, expected, strict=True):
        assert value.shape == reference.shape and value.dtype == reference.dtype
        np.testing.assert_array_equal(bits(value), bits(reference))
    parameters = dict(tree_flatten(model.parameters()))
    assert parameters.keys() == before.keys()
    for name, value in parameters.items():
        np.testing.assert_array_equal(bits(value), before[name])


def test_load_weights_updates_projection_outputs():
    model = _model(seed=84, dense=True)
    nn.quantize(model, bits=8, group_size=64)
    model.prepare_for_inference()
    mlp = model.layers[0].mlp
    hidden = mx.random.normal((2, 1, model.config.hidden_size)).astype(mx.bfloat16)
    original = np.array(mlp(hidden).astype(mx.float32))
    assert np.any(original != 0)
    # A zero affine gate has zero output regardless of its packed payload.
    # Public weight loading must replace the values used by the model.
    model.load_weights(
        [
            ("layers.0.mlp.gate_proj.scales", mx.zeros_like(mlp.gate_proj.scales)),
            ("layers.0.mlp.gate_proj.biases", mx.zeros_like(mlp.gate_proj.biases)),
        ],
        strict=False,
    )
    np.testing.assert_array_equal(
        np.array(mlp(hidden).astype(mx.float32)), np.zeros_like(original)
    )


def test_set_dtype_updates_projection_outputs():
    model = _model(seed=85, dtype=mx.float32, dense=True)
    model.prepare_for_inference()
    model.set_dtype(mx.bfloat16)
    mlp = model.layers[0].mlp
    hidden = mx.random.normal((2, 3, model.config.hidden_size)).astype(mx.bfloat16)
    expected = mlp.down_proj(nn.silu(mlp.gate_proj(hidden)) * mlp.up_proj(hidden))
    actual = mlp(hidden)
    assert actual.dtype == expected.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
    )


@pytest.mark.parametrize("dense", [False, True], ids=["moe", "dense"])
def test_incremental_state_matches_full_forward_exactly(dense: bool) -> None:
    # FP32 isolates state and history plumbing from BF16 rounding.
    model = _model(dtype=mx.float32, dense=dense)
    tokens = _prompt(21, seed=2)
    config = model.config
    expected = _full_logits(model, tokens)

    first = model(
        mx.array([tokens[:13]]),
        mx.array([0]),
        model.initial_state(1),
        (None,) * config.full_layer_count,
    )
    from resin_models_mlx.qwen3_5.model import AttentionHistory

    history = tuple(
        AttentionHistory(key, value)
        for key, value in zip(first.keys, first.values, strict=True)
    )
    second = model(mx.array([tokens[13:]]), mx.array([13]), first.state, history)
    actual = model.logits(second.hidden)[0].astype(mx.float32)

    assert mx.allclose(actual, expected[13:], atol=1e-5, rtol=1e-5).item()


@pytest.mark.parametrize("dense", [False, True], ids=["moe", "dense"])
def test_engine_greedy_matches_reference_across_page_boundaries(dense: bool) -> None:
    model = _model(all_experts=True, dense=dense)
    lengths = (5, PAGE_TOKENS - 2, PAGE_TOKENS - 1, PAGE_TOKENS, PAGE_TOKENS + 1)
    prompts = [_prompt(length, seed=length) for length in lengths]

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
    overlap: bool, model: Qwen35Model, temperature: float, stops: tuple[int, ...]
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


@pytest.mark.parametrize("dense", [False, True], ids=["moe", "dense"])
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_overlapped_decode_matches_synchronous_steps(
    temperature: float, dense: bool
) -> None:
    model = _model(all_experts=True, dense=dense)
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


@pytest.mark.parametrize("dense", [False, True], ids=["moe", "dense"])
def test_chunked_prefill_crosses_state_checkpoint(dense: bool) -> None:
    model = _model(all_experts=True, dense=dense)
    prompt = _prompt(PREFILL_CHUNK_SIZE + 37, seed=3)

    with Engine(_runner(model), context_length=16384) as engine:
        (generated,) = _generate(engine, [prompt], 4)
    _assert_greedy(model, prompt, generated, 4)


@pytest.mark.parametrize("dense", [False, True], ids=["moe", "dense"])
def test_prefix_cache_hit_matches_fresh_run(dense: bool) -> None:
    model = _model(all_experts=True, dense=dense)
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


@pytest.mark.parametrize("dense", [False, True], ids=["moe", "dense-tied"])
def test_prompt_logprobs_match_one_pass_forward(dense: bool) -> None:
    model = _model(dense=dense, tie_word_embeddings=dense)
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


@pytest.mark.parametrize(
    "layout,dense,tied",
    [
        pytest.param("per-expert", False, False, id="moe-per-expert-sharded"),
        pytest.param("fused", False, False, id="moe-fused"),
        pytest.param("fused", True, True, id="dense-nested-tied"),
        pytest.param("text-only", True, False, id="dense-text-only-untied"),
    ],
)
def test_checkpoint_loader_matches_transformers_reference(layout, dense, tied):
    import json
    import tempfile

    import torch
    from safetensors.torch import save_file
    from transformers import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeTextConfig,
        Qwen3_5TextConfig,
    )

    from resin_models_mlx.qwen3_5.checkpoint import load_model

    values = _text_config(dense=dense, tie_word_embeddings=tied)
    torch.manual_seed(0)
    config_type = Qwen3_5TextConfig if dense else Qwen3_5MoeTextConfig
    model_type = Qwen3_5ForCausalLM if dense else Qwen3_5MoeForCausalLM
    reference = model_type(config_type(**values)).float().eval()
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
    if tied:
        # Hugging Face saves only the embedding tensor for tied weights.
        state.pop("lm_head.weight")
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
            else {
                "model_type": "qwen3_5" if dense else "qwen3_5_moe",
                "text_config": values,
            }
        )
        (path / "config.json").write_text(json.dumps(config))
        model, _, _ = load_model(path, bits=None)
        assert ("lm_head.weight" in dict(tree_flatten(model.parameters()))) is not tied
        for layer in model.layers:
            if hasattr(layer, "linear_attn"):
                assert layer.linear_attn.A_log.dtype == mx.float32
                assert layer.linear_attn.dt_bias.dtype == mx.float32
        actual = np.array(_full_logits(model, tokens))
    np.testing.assert_allclose(actual, expected, atol=3e-2, rtol=3e-2)
    assert (actual.argmax(-1) == expected.argmax(-1)).mean() > 0.9


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_prequantized_checkpoint_preserves_packed_weights_and_reference_logits(
    tmp_path, tied: bool
) -> None:
    import copy
    import json

    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    import resin_models_mlx.qwen3_5.checkpoint as checkpoint

    values = _text_config(dense=True, tie_word_embeddings=tied)
    quantization = {"bits": 8, "group_size": 64, "mode": "affine"}
    raw = {
        "model_type": "qwen3_5",
        "text_config": values,
        # MLX exports can carry both spellings with identical settings.
        "quantization": quantization,
        "quantization_config": quantization.copy(),
    }
    mx.random.seed(17)
    reference = Model(ModelArgs.from_dict(copy.deepcopy(raw)))
    updates = []
    for name, value in cast(
        list[tuple[str, Any]], tree_flatten(reference.parameters())
    ):
        if name.endswith("norm.weight"):
            # mlx-lm stores sanitized RMSNorm scales rather than HF offsets.
            updates.append((name, 1.0 + 0.1 * mx.random.normal(value.shape)))
        elif name.endswith("A_log"):
            updates.append((name, mx.random.uniform(-8.0, 1.0, value.shape)))
        elif name.endswith("dt_bias"):
            updates.append((name, 0.1 * mx.random.normal(value.shape)))
    reference.load_weights(updates, strict=False)
    nn.quantize(reference, bits=8, group_size=64, mode="affine")
    reference.eval()
    saved = dict(cast(list[tuple[str, Any]], tree_flatten(reference.parameters())))
    mx.eval(saved)
    # Independent mlx-lm parameter names and depthwise-convolution layout are
    # the ones found in converted community checkpoints.
    assert saved["language_model.model.embed_tokens.weight"].dtype == mx.uint32
    mx.save_safetensors(str(tmp_path / "model.safetensors"), saved)
    (tmp_path / "config.json").write_text(json.dumps(raw))

    model, config, _ = checkpoint.load_model(tmp_path)
    parameters = dict(tree_flatten(model.parameters()))
    assert config.tie_word_embeddings is tied
    assert isinstance(model.embed_tokens, nn.QuantizedEmbedding)
    assert ("lm_head.weight" in parameters) is not tied

    for name, value in saved.items():
        target = name.removeprefix("language_model.").removeprefix("model.")
        if value.dtype == mx.uint32 or name.endswith((".scales", ".biases")):
            assert parameters[target].dtype == value.dtype
            np.testing.assert_array_equal(np.array(parameters[target]), np.array(value))
    np.testing.assert_array_equal(
        np.array(model.norm.weight),
        np.array(saved["language_model.model.norm.weight"]),
    )

    tokens = _prompt(29, seed=17)
    expected = reference(mx.array([tokens]))[0].astype(mx.float32)
    actual = _full_logits(model, tokens)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=1e-3, rtol=1e-3
    )
    assert mx.array_equal(
        mx.argmax(actual, axis=-1), mx.argmax(expected, axis=-1)
    ).item()


def _assert_first_inference_in_native_http_actor(checkpoint: Path) -> None:
    import copy
    import json
    import socket
    import threading
    import time
    import urllib.error
    import urllib.request

    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    from resin.recipe import RenderedPrompt, TextDelta
    from resin.server import HttpServer
    from resin_models_mlx.qwen3_5.vision import Qwen35VisionModel

    values = _text_config(dense=True, tie_word_embeddings=True)
    values["linear_key_head_dim"] = 128
    visual_config = dict(
        model_type="qwen3_5_vision",
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=values["hidden_size"],
        num_heads=4,
        depth=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        num_position_embeddings=16,
        hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=[],
    )
    raw = dict(
        model_type="qwen3_5",
        text_config=values,
        quantization=dict(bits=8, group_size=64, mode="affine"),
        vision_config=visual_config,
    )
    mx.random.seed(94)
    reference = Model(ModelArgs.from_dict(copy.deepcopy(raw)))
    nn.quantize(reference, bits=8, group_size=64)
    visual = Qwen35VisionModel(visual_config)
    visual.set_dtype(mx.bfloat16)
    saved = dict(cast(list[tuple[str, Any]], tree_flatten(reference.parameters())))
    saved.update(
        ("vision_tower." + name, value)
        for name, value in cast(
            list[tuple[str, Any]], tree_flatten(visual.parameters())
        )
    )
    mx.eval(saved)
    mx.save_safetensors(str(checkpoint / "model.safetensors"), saved)
    (checkpoint / "config.json").write_text(json.dumps(raw))

    constructed_on = threading.get_native_id()
    forwarded_on = []

    class Runner(Qwen35MlxRunner):
        def forward(self, plan):
            forwarded_on.append(threading.get_native_id())
            return super().forward(plan)

    # Load real packed tensors and warm the
    # default compiled vision encoder on the constructor thread. The first text
    # forward must then run on the actual Rust HTTP inference actor.
    runner = Runner(checkpoint, cache_slab_capacity=64, workspace_bytes=64 << 20)
    assert runner._media is not None
    from resin_models_mlx.qwen3_5.normalization import normalize_qk

    # The first native actor forward also reuses constants first published on
    # the constructor thread, as happens after main-thread model warmup.
    for dtype in (mx.bfloat16, mx.float32):
        qk = mx.zeros((1, 1, 2, 128), dtype=dtype)
        mx.eval(normalize_qk(qk, qk, inv_scale=128**-0.5, eps=1e-6))
    prompts = [_prompt(19, seed=95), _prompt(131, seed=96)]

    class Recipe:
        tokenizer = None
        stop_token_ids = []

        def render(self, messages, *, tools=(), options=None):
            return RenderedPrompt(prompts[int(messages[0].content[0].text)])

        def create_parser(self, prompt):
            class Parser:
                def feed(self, token_ids):
                    return [
                        TextDelta(
                            " ".join(str(int(token)) for token in token_ids) + " "
                        )
                    ]

            return Parser()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    engine = Engine(runner, context_length=8192)
    server = HttpServer(engine, recipe=Recipe(), model="tiny-packed")
    errors = []

    def serve():
        try:
            server.serve(port=port)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=0.2) as response:
                    assert response.status == 200
                break
            except (OSError, urllib.error.URLError):
                assert not errors, errors
                assert time.monotonic() < deadline, "native HTTP actor did not start"
                time.sleep(0.01)
        actual = []
        for index in range(len(prompts)):
            request = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps(
                    dict(
                        model="tiny-packed",
                        messages=[dict(role="user", content=str(index))],
                        max_tokens=8,
                        temperature=0,
                    )
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.status == 200
                body = json.load(response)
            actual.append(
                [
                    int(token)
                    for token in body["choices"][0]["message"]["content"].split()
                ]
            )
            assert len(actual[-1]) == 8
        assert not engine.poisoned
        assert engine.overlap_stats[1] > 0
    finally:
        server.shutdown()
        thread.join(timeout=10)
        assert not thread.is_alive(), "native HTTP actor failed to shut down"
        assert errors == []
    assert forwarded_on and all(
        native_id != constructed_on for native_id in forwarded_on
    )
    # Compare the actor's outputs with the same loaded checkpoint running on the
    # constructor thread, including the second prompt's page-crossing history.
    direct = Qwen35MlxRunner(
        checkpoint,
        include_vision=False,
        cache_slab_capacity=64,
        workspace_bytes=64 << 20,
    )
    with Engine(direct, context_length=8192) as engine:
        expected = [_generate(engine, [prompt], 8)[0] for prompt in prompts]
    assert actual == expected


def test_first_inference_uses_fresh_native_http_actor_after_checkpoint_loading(
    tmp_path,
):
    # A separate process prevents an earlier test from initializing the actor's
    # MLX stream registry or warming the target model before its first request.
    tests = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(tests)!r}); "
                "from test_qwen3_5_mlx import "
                "_assert_first_inference_in_native_http_actor as check; "
                f"check(Path({str(tmp_path)!r}))"
            ),
        ],
        check=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(
            {
                "quantization": {"bits": 8, "group_size": 64},
                "quantization_config": {"bits": 4, "group_size": 64},
            },
            id="conflicting-aliases",
        ),
        pytest.param(
            {"quantization": {"bits": 8, "group_size": 64, "mode": "mxfp8"}},
            id="unsupported-format",
        ),
        pytest.param(
            {"quantization_config": {"quant_method": "awq", "bits": 8}},
            id="foreign-format",
        ),
    ],
)
def test_prequantized_config_rejects_ambiguous_formats(tmp_path, metadata) -> None:
    import json

    from resin_models_mlx.qwen3_5.checkpoint import load_config

    raw = {
        "model_type": "qwen3_5",
        "text_config": _text_config(dense=True),
        **metadata,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="quantization"):
        load_config(tmp_path)


def test_cache_writes_update_the_arena_in_place() -> None:
    model = _model()
    # Keep a >1 GiB copy detector without occupying most of a small CI runner.
    runner = _runner(model, slabs=5_000)
    arena = runner._cache.nbytes
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
    # Prefill and decode both write; a copied arena would double its size.
    assert arena > 1 << 30
    assert growth < arena // 8


def test_context_is_capped_by_checkpoint_positions() -> None:
    model = _model()
    limit = model.config.max_position_embeddings
    with pytest.raises(ValueError, match="checkpoint's"):
        Qwen35MlxRunner(
            model,
            cache_slab_capacity=8,
            max_seq_len=limit + 1,
            workspace_bytes=64 << 20,
        )
    runner = Qwen35MlxRunner(
        model, cache_slab_capacity=64, max_seq_len=300, workspace_bytes=64 << 20
    )
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
    runner = Qwen35MlxRunner(
        _model(), cache_slab_capacity=64, max_seq_len=300, workspace_bytes=64 << 20
    )
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
    import resin_models_mlx.qwen3_5.model as model_module

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
