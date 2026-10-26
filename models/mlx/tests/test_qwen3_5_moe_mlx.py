from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from resin import Engine, SamplingParams  # noqa: E402
from resin.models.qwen3_5.config import Qwen35Config  # noqa: E402
from resin_models_mlx.qwen3_5_moe.cache import (  # noqa: E402
    PAGE_TOKENS,
    PREFILL_CHUNK_SIZE,
    CacheGeometry,
)
from resin_models_mlx.qwen3_5_moe.model import Qwen35MoeModel  # noqa: E402
from resin_models_mlx.qwen3_5_moe.runner import Qwen3_5MoeMlxRunner  # noqa: E402


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


def _model(seed: int = 0, dtype=None) -> Qwen35MoeModel:
    mx.random.seed(seed)
    model = Qwen35MoeModel(Qwen35Config.from_hugging_face(_text_config()))
    # Give norms and DeltaNet decay realistic, non-trivial values.
    updates = []
    for name, value in cast(list[tuple[str, Any]], tree_flatten(model.parameters())):
        if name.endswith("norm.weight"):
            updates.append((name, 1.0 + 0.1 * mx.random.normal(value.shape)))
        elif name.endswith("A_log"):
            updates.append((name, mx.random.uniform(-1.0, 1.0, value.shape)))
        elif name.endswith("dt_bias"):
            updates.append((name, 0.1 * mx.random.normal(value.shape)))
    model.load_weights(updates, strict=False)
    selected = mx.bfloat16 if dtype is None else dtype
    model.set_dtype(selected, lambda dtype: dtype == mx.float32)
    mx.eval(model.parameters())
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
        weights.append((prefix + name, value))
    reference.load_weights(weights)
    mx.eval(reference.parameters())
    return reference


def test_forward_matches_mlx_lm_reference() -> None:
    model = _model()
    reference = _mlx_lm_reference(model)
    tokens = _prompt(37, seed=1)

    actual = _full_logits(model, tokens)
    expected = reference(mx.array([tokens]))[0].astype(mx.float32)

    assert mx.allclose(actual, expected, atol=2e-2, rtol=2e-2).item()
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
        AttentionHistory(key, value, mx.ones((1, 13), dtype=mx.bool_))
        for key, value in zip(first.keys, first.values, strict=True)
    )
    second = model(mx.array([tokens[13:]]), mx.array([13]), first.state, history)
    actual = model.logits(second.hidden)[0].astype(mx.float32)

    assert mx.allclose(actual, expected[13:], atol=1e-5, rtol=1e-5).item()


def test_engine_greedy_matches_reference_across_page_boundaries() -> None:
    model = _model()
    prompts = [_prompt(length, seed=length) for length in (5, 127, 128, 129)]

    with Engine(_runner(model), context_length=8192) as engine:
        results = _generate(engine, prompts, 6)
    for prompt, generated in zip(prompts, results, strict=True):
        _assert_greedy(model, prompt, generated, 6)


def test_batched_decode_matches_individual_requests() -> None:
    model = _model()
    prompts = [_prompt(length, seed=10 + length) for length in (3, 40, 200)]
    with Engine(_runner(model), context_length=8192) as engine:
        alone = [_generate(engine, [prompt], 8)[0] for prompt in prompts]
    with Engine(_runner(model), context_length=8192) as engine:
        together = _generate(engine, prompts, 8)
    assert together == alone


def test_chunked_prefill_crosses_state_checkpoint() -> None:
    model = _model()
    prompt = _prompt(PREFILL_CHUNK_SIZE + 37, seed=3)

    with Engine(_runner(model), context_length=16384) as engine:
        (generated,) = _generate(engine, [prompt], 4)
    _assert_greedy(model, prompt, generated, 4)


def test_prefix_cache_hit_matches_fresh_run() -> None:
    model = _model()
    shared = _prompt(PREFILL_CHUNK_SIZE, seed=4)
    suffix = _prompt(9, seed=5)

    with Engine(_runner(model), context_length=16384) as engine:
        (fresh,) = _generate(engine, [shared + suffix], 5)
    with Engine(_runner(model), context_length=16384, prefix_cache=True) as engine:
        _generate(engine, [shared + _prompt(3, seed=6)], 1)
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


def test_decode_crosses_kv_page_boundary() -> None:
    model = _model()
    prompt = _prompt(PAGE_TOKENS - 2, seed=7)

    with Engine(_runner(model), context_length=8192) as engine:
        (generated,) = _generate(engine, [prompt], 6)
    _assert_greedy(model, prompt, generated, 6)


def _assert_checkpoint_loader_matches_transformers() -> None:
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
    state = {
        name.replace("model.", "model.language_model.", 1): tensor.to(torch.bfloat16)
        for name, tensor in reference.state_dict().items()
    }
    # Hugging Face's fused expert tensors are split into per-expert checkpoint
    # names, matching the published Ornith safetensors.
    for name in [name for name in state if name.endswith("mlp.experts.gate_up_proj")]:
        prefix = name.removesuffix("gate_up_proj")
        gate_up = state.pop(name)
        down = state.pop(prefix + "down_proj")
        middle = gate_up.shape[1] // 2
        for expert in range(gate_up.shape[0]):
            state[f"{prefix}{expert}.gate_proj.weight"] = gate_up[expert, :middle]
            state[f"{prefix}{expert}.up_proj.weight"] = gate_up[expert, middle:]
            state[f"{prefix}{expert}.down_proj.weight"] = down[expert]
    reference = reference.to(torch.bfloat16)
    tokens = _prompt(19, seed=8)
    with torch.no_grad():
        expected = reference(torch.tensor([tokens])).logits[0].float().numpy()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        save_file(
            {k: v.contiguous() for k, v in state.items()}, path / "model.safetensors"
        )
        import json

        (path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5_moe", "text_config": values})
        )
        model, _, _ = load_model(path, bits=None)
        actual = np.array(_full_logits(model, tokens))
    np.testing.assert_allclose(actual, expected, atol=5e-2, rtol=5e-2)
    assert (actual.argmax(-1) == expected.argmax(-1)).mean() > 0.9


def test_checkpoint_loader_matches_transformers_reference() -> None:
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
                "check()"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cache_writes_update_the_arena_in_place() -> None:
    model = _model()
    runner = _runner(model, slabs=10_000)
    arena = runner._cache.storage.nbytes
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
    # Prefill and decode both write; a copied arena would double its size.
    assert arena > 1 << 30
    assert growth < arena // 8
