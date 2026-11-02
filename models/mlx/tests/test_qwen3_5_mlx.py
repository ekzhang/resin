from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from resin import Engine, SamplingParams  # noqa: E402
from resin.models.qwen3_5.config import Qwen35Config  # noqa: E402
from resin_models_mlx.qwen3_5.cache import PREFILL_CHUNK_SIZE  # noqa: E402
from resin_models_mlx.qwen3_5.model import Qwen35Model  # noqa: E402
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


def test_preparation_preserves_forward_bits_and_checkpoint_parameters():
    model = _model(seed=87, dense=True)
    nn.quantize(model, bits=8, group_size=64)
    rows = 1
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


def test_prequantized_checkpoint_preserves_packed_weights_and_reference_logits(
    tmp_path,
) -> None:
    import copy

    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    import resin_models_mlx.qwen3_5.checkpoint as checkpoint

    tied = True

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


@pytest.mark.parametrize("dense", [False, True], ids=["moe", "dense-tied"])
def test_checkpoint_loader_matches_transformers_reference(tmp_path, dense):
    import torch
    from safetensors.torch import save_file
    from transformers import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeTextConfig,
        Qwen3_5TextConfig,
    )

    from resin_models_mlx.qwen3_5.checkpoint import load_model

    values = _text_config(dense=dense, tie_word_embeddings=dense)
    torch.manual_seed(0)
    config_type = Qwen3_5TextConfig if dense else Qwen3_5MoeTextConfig
    model_type = Qwen3_5ForCausalLM if dense else Qwen3_5MoeForCausalLM
    reference = model_type(config_type(**values)).float().eval()
    decay = ("A_log", "dt_bias")
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith("norm.weight"):
                parameter.normal_(0.0, 0.1)
            elif name.endswith("A_log"):
                parameter.uniform_(-1.0, 1.0)
    reference_decay = {
        name: tensor.clone()
        for name, tensor in reference.state_dict().items()
        if name.endswith(decay)
    }
    reference = reference.to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith(decay):
                parameter.data = reference_decay[name]
    state = {
        name.replace("model.", "model.language_model.", 1): tensor.contiguous()
        for name, tensor in reference.state_dict().items()
        if not (dense and name == "lm_head.weight")
    }
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_5" if dense else "qwen3_5_moe", "text_config": values}
        )
    )
    model, _, _ = load_model(tmp_path, bits=None)
    assert ("lm_head.weight" in dict(tree_flatten(model.parameters()))) is not dense
    for layer in model.layers:
        if hasattr(layer, "linear_attn"):
            assert layer.linear_attn.A_log.dtype == mx.float32
            assert layer.linear_attn.dt_bias.dtype == mx.float32
    tokens = _prompt(19, seed=8)
    with torch.inference_mode():
        expected = reference(torch.tensor([tokens])).logits[0].float().numpy()
    actual = np.array(_full_logits(model, tokens))
    np.testing.assert_allclose(actual, expected, atol=3e-2, rtol=3e-2)
    assert (actual.argmax(-1) == expected.argmax(-1)).mean() > 0.9


def test_prefix_restore_and_overlap_match_fresh_synchronous_decode():
    model = _model(all_experts=True)
    shared = _prompt(PREFILL_CHUNK_SIZE, seed=4)
    prompts = [shared + _prompt(9, seed=5), _prompt(37, seed=6)]
    with Engine(_runner(model), context_length=16384, enable_overlap=False) as engine:
        expected = _generate(engine, prompts, 5)
    with Engine(_runner(model), context_length=16384, prefix_cache=True) as engine:
        _generate(engine, [shared + [3] * 50], 1)
        params = SamplingParams(max_output_tokens=5, temperature=0.0)
        requests = [engine.submit(prompt, sampling_params=params) for prompt in prompts]
        while engine.has_work():
            engine.step()
        assert requests[0].cached_tokens == PREFILL_CHUNK_SIZE
        assert engine.overlap_stats[1] > 0
        assert [request.take_result() for request in requests] == expected


def test_first_inference_after_preparation_can_run_on_a_fresh_thread():
    model = _model(dense=True)
    nn.quantize(model, bits=8, group_size=64)
    mx.eval(model.parameters())
    model.prepare_for_inference()
    runner = _runner(model)
    prompts = [_prompt(19, seed=95), _prompt(131, seed=96)]
    with Engine(runner, context_length=8192) as engine:
        with ThreadPoolExecutor(max_workers=1) as executor:
            actual = executor.submit(_generate, engine, prompts, 8).result(timeout=30)
        assert engine.overlap_stats[1] > 0
    with Engine(_runner(model), context_length=8192) as engine:
        assert actual == _generate(engine, prompts, 8)


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
