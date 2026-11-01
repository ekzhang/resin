from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from torch.nn import functional as F

from resin.models.gemma4.edge_config import Gemma4EdgeConfig
from resin_models_cuda.gemma4.cuda import plan_gemma4_kv_capacity
from resin_models_cuda.gemma4.edge_assistant import (
    Gemma4EdgeAssistantConfig,
    Gemma4EdgeAssistantForCausalLM,
    Gemma4EdgeMaskedEmbedder,
)
from resin_models_cuda.gemma4.edge_model import Gemma4EdgeForCausalLM


def _config() -> Gemma4EdgeConfig:
    return Gemma4EdgeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=8,
        global_head_dim=16,
        sliding_window=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        layer_types=(
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ),
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=64,
        num_kv_shared_layers=2,
        use_double_wide_mlp=True,
        pad_token_id=0,
        eos_token_ids=(1,),
        final_logit_softcapping=30.0,
        local_rope_theta=10000.0,
        global_rope_theta=1000000.0,
        global_partial_rotary_factor=0.25,
    )


def _hf_reference(config: Gemma4EdgeConfig, *, dtype=torch.bfloat16):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    hf_config = Gemma4TextConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        num_global_key_value_heads=config.num_global_key_value_heads,
        head_dim=config.head_dim,
        global_head_dim=config.global_head_dim,
        sliding_window=config.sliding_window,
        max_position_embeddings=config.max_position_embeddings,
        layer_types=list(config.layer_types),
        hidden_size_per_layer_input=config.hidden_size_per_layer_input,
        vocab_size_per_layer_input=config.vocab_size_per_layer_input,
        num_kv_shared_layers=config.num_kv_shared_layers,
        use_double_wide_mlp=True,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )
    hf_config._attn_implementation = "eager"
    reference = Gemma4TextModel(hf_config)
    cast(torch.nn.Module, reference).to(dtype=dtype)
    return reference.eval()


def _forward_without_cache(model, token_ids, *, only_last_consumer=False):
    config = model.config
    hidden, per_layer = model.prepare_embeddings(token_ids)
    positions = torch.arange(len(token_ids))
    caches = {}
    for index, layer in enumerate(model.language_model.layers):
        if only_last_consumer and index == config.num_kv_producer_layers:
            hidden, per_layer, positions = hidden[-1:], per_layer[-1:], positions[-1:]
        cos, sin = model.language_model.rotary_emb(
            positions, layer.self_attn.layer_type, hidden.dtype
        )
        query, key, value = layer.self_attn.project_qkv(
            layer.input_layernorm(hidden), cos, sin
        )
        if key is not None:
            caches[index] = key, value
        key, value = caches[layer.self_attn.cache_source_index]
        key = key.repeat_interleave(config.num_attention_heads, dim=1)
        value = value.repeat_interleave(config.num_attention_heads, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query, key)
        key_positions = torch.arange(len(key))
        allowed = key_positions[None, :] <= positions[:, None]
        if layer.self_attn.is_local:
            allowed &= (
                key_positions[None, :] > positions[:, None] - config.sliding_window
            )
        scores = scores.masked_fill(~allowed[None], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attended = torch.einsum("hqk,khd->qhd", weights, value)
        contribution = layer.self_attn.o_proj(attended.reshape(len(hidden), -1))
        hidden = hidden + layer.post_attention_layernorm(contribution)
        hidden = layer.feed_forward(hidden, per_layer[:, index])
    return model.language_model.norm(hidden)


@torch.inference_mode()
def test_e2b_matches_independent_transformers_reference_and_fast_consumer_rows():
    torch.manual_seed(7)
    config = _config()
    reference = _hf_reference(config, dtype=torch.float32)
    model = Gemma4EdgeForCausalLM(config).float().eval()
    model.language_model.load_state_dict(reference.state_dict(), strict=True)
    model.materialize_buffers(torch.device("cpu"))
    model.prepare_attention()
    token_ids = torch.tensor([2, 9, 6, 15, 3, 18, 31])

    expected = reference(token_ids[None], use_cache=False).last_hidden_state[0]
    observed = _forward_without_cache(model, token_ids)
    torch.testing.assert_close(observed, expected, atol=2e-5, rtol=2e-5)
    reduced = _forward_without_cache(model, token_ids, only_last_consumer=True)
    torch.testing.assert_close(reduced, observed[-1:], atol=2e-5, rtol=2e-5)
    expected_logits = F.linear(observed[-1:], model.language_model.embed_tokens.weight)
    expected_logits = torch.tanh(expected_logits / 30.0) * 30.0
    torch.testing.assert_close(model.logits(observed[-1:]), expected_logits)


def test_e2b_consumer_reuses_last_producer_of_its_type_and_doubles_mlp():
    model = Gemma4EdgeForCausalLM(_config())
    layers = model.language_model.layers
    assert [layer.self_attn.cache_source_index for layer in layers] == [0, 1, 2, 2, 1]
    assert [layer.mlp.intermediate_size for layer in layers] == [32, 32, 32, 64, 64]
    assert not hasattr(layers[3].self_attn, "k_proj")
    assert not hasattr(layers[4].self_attn, "v_proj")
    assert "model.language_model.embed_tokens_per_layer.weight" in model.state_dict()


@torch.inference_mode()
def test_e2b_embedding_scale_and_per_layer_inputs_match_reference():
    torch.manual_seed(11)
    config = _config()
    reference = _hf_reference(config)
    model = Gemma4EdgeForCausalLM(config).eval()
    model.language_model.load_state_dict(reference.state_dict(), strict=True)
    model.materialize_buffers(torch.device("cpu"))
    ids = torch.tensor([2, 7, 29])
    hidden, per_layer = model.prepare_embeddings(ids)
    expected_hidden = reference.embed_tokens(ids[None])
    expected_ple = reference.project_per_layer_inputs(
        expected_hidden, reference.get_per_layer_inputs(ids[None], expected_hidden)
    )
    torch.testing.assert_close(hidden, expected_hidden[0], atol=0, rtol=0)
    torch.testing.assert_close(per_layer, expected_ple[0], atol=0, rtol=0)


def raw_config():
    # google/gemma-4-E2B-it, revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7.
    return json.loads(
        (Path(__file__).parent / "fixtures/gemma4_e2b_config.json").read_text()
    )


def test_original_bf16_e2b_configuration_and_shared_cache_geometry():
    config = Gemma4EdgeConfig.from_dict(raw_config())
    assert config.num_kv_producer_layers == 15
    assert config.local_layer_count == 12
    assert config.global_layer_count == 3
    assert config.history_blocks_per_parent == 2
    assert config.num_global_key_value_heads == 1
    assert config.sliding_window == 512
    assert config.cache_source_layer(15) == 13
    assert config.cache_source_layer(19) == 14
    assert config.cache_layer_index(15) == 11
    assert config.cache_layer_index(19) == 2
    capacity = plan_gemma4_kv_capacity(
        80 * 1024**3,
        70 * 1024**3,
        0.9,
        config=config,
        cache_element_bytes=2,
    )
    assert capacity.bytes_per_parent == 2 * 12 * 64 * 1 * 256 * 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("attention_k_eq_v", True),
        ("attention_bias", True),
        ("hidden_activation", "silu"),
        ("enable_moe_block", True),
    ],
)
def test_e2b_rejects_unsupported_checkpoint_computations(field, value):
    raw = copy.deepcopy(raw_config())
    raw["text_config"][field] = value
    with pytest.raises(ValueError, match="unsupported E2B text options"):
        Gemma4EdgeConfig.from_dict(raw)


def test_edge_default_mtp_uses_matching_assistant_and_supported_bf16_cache(monkeypatch):
    import torch

    from resin.backend.cuda.runner import _resolve_kv_cache_dtype
    from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner
    from resin_models_cuda.gemma4.edge_assistant import (
        DEFAULT_ASSISTANT,
        DEFAULT_ASSISTANT_REVISION,
    )
    from resin_models_cuda.gemma4.edge_cuda import Gemma4EdgeCudaRunner

    received = {}
    monkeypatch.setattr(
        Gemma4CudaRunner,
        "__init__",
        lambda self, model, **kwargs: received.update(kwargs),
    )
    Gemma4EdgeCudaRunner(device="cuda", speculative_draft_tokens=3)
    assert received["assistant_model"] == DEFAULT_ASSISTANT
    assert received["assistant_revision"] == DEFAULT_ASSISTANT_REVISION
    assert _resolve_kv_cache_dtype(received["kv_cache_dtype"]) == torch.bfloat16
    with pytest.raises(ValueError, match="explicit assistant_model"):
        Gemma4EdgeCudaRunner(
            "/local/custom-edge-target", device="cuda", speculative_draft_tokens=3
        )


def _assistant_config() -> Gemma4EdgeAssistantConfig:
    return Gemma4EdgeAssistantConfig(
        backbone_hidden_size=16,
        vocab_size=64,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=8,
        global_head_dim=16,
        sliding_window=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        layer_types=("sliding_attention",) * 3 + ("full_attention",),
        local_rope_theta=10000.0,
        global_rope_theta=1000000.0,
        global_partial_rotary_factor=0.25,
        num_centroids=8,
        centroid_intermediate_top_k=2,
    )


def _assistant_hf_reference(config, *, dtype=torch.float32):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4_assistant.configuration_gemma4_assistant import (
        Gemma4AssistantConfig,
    )
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantForCausalLM,
    )

    text = Gemma4TextConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=config.head_dim,
        global_head_dim=config.global_head_dim,
        sliding_window=config.sliding_window,
        max_position_embeddings=config.max_position_embeddings,
        layer_types=list(config.layer_types),
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        num_kv_shared_layers=4,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )
    hf_config = Gemma4AssistantConfig(
        text_config=text,
        backbone_hidden_size=config.backbone_hidden_size,
        use_ordered_embeddings=True,
        num_centroids=config.num_centroids,
        centroid_intermediate_top_k=config.centroid_intermediate_top_k,
    )
    hf_config._attn_implementation = "eager"
    text._attn_implementation = "eager"
    reference = Gemma4AssistantForCausalLM(hf_config)
    cast(torch.nn.Module, reference).to(dtype=dtype)
    return reference.eval()


class _assistant_Attention:
    def __init__(self, window_left=None):
        self.window_left = window_left

    def run(self, query, kv):
        key, value = kv
        if self.window_left is not None:
            key = key[:, -self.window_left - 1 :]
            value = value[:, -self.window_left - 1 :]
        key = key.repeat_interleave(query.shape[1], dim=2)
        value = value.repeat_interleave(query.shape[1], dim=2)
        scores = torch.einsum("bhd,bkhd->bhk", query, key)
        weights = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
        return torch.einsum("bhk,bkhd->bhd", weights, value)


@torch.inference_mode()
@pytest.mark.parametrize("local_length", [3, 5, 6, 12])
def test_e2b_mtp_assistant_matches_independent_transformers_reference(local_length):
    torch.manual_seed(13)
    config = _assistant_config()
    reference = _assistant_hf_reference(config)
    reference.masked_embedding.token_ordering.copy_(torch.randperm(config.vocab_size))
    assistant = Gemma4EdgeAssistantForCausalLM(config).float().eval()
    state = {k: v for k, v in reference.state_dict().items() if k != "lm_head.weight"}
    assistant.load_state_dict(state, strict=True)
    assistant.materialize_buffers(torch.device("cpu"))
    inputs = torch.randn(2, 2 * config.backbone_hidden_size)
    positions = torch.tensor([5, 7])
    local_key, local_value = (
        torch.randn(2, local_length, 1, config.head_dim) for _ in range(2)
    )
    global_key, global_value = (
        torch.randn(2, 6, 1, config.global_head_dim) for _ in range(2)
    )
    local_kv = (local_key, local_value)
    global_kv = (global_key, global_value)
    shared = {
        "sliding_attention": tuple(t.transpose(1, 2) for t in local_kv),
        "full_attention": tuple(t.transpose(1, 2) for t in global_kv),
    }
    expected = reference(
        inputs_embeds=inputs[:, None],
        position_ids=positions[:, None],
        shared_kv_states=shared,
    )
    logits, hidden = assistant.forward_step(
        inputs,
        positions,
        # The official assistant uses a flipped bidirectional window with
        # inclusive distance <= sliding_window. With one cross-attention query
        # this admits sliding_window + 1 shared keys, unlike target attention.
        local_wrapper=_assistant_Attention(window_left=config.sliding_window),
        global_wrapper=_assistant_Attention(),
        local_kv=local_kv,
        global_kv=global_kv,
    )
    torch.testing.assert_close(
        hidden, expected.last_hidden_state[:, 0], atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(logits, expected.logits[:, 0], atol=2e-5, rtol=2e-5)
    assert torch.equal(logits.argmax(-1), expected.logits[:, 0].argmax(-1))


@torch.inference_mode()
def test_ordered_head_matches_huggingface_selected_token_and_mask_semantics():
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantMaskedEmbedder,
    )

    torch.manual_seed(19)
    config = _assistant_config()
    hf_config = _assistant_hf_reference(config).config
    reference = Gemma4AssistantMaskedEmbedder(hf_config).bfloat16()
    reference.token_ordering.copy_(torch.randperm(config.vocab_size))
    head = Gemma4EdgeMaskedEmbedder(config)
    head.load_state_dict(reference.state_dict())
    hidden = torch.randn(2, 3, config.hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
    expected = reference(hidden, weight)
    actual = head(hidden, weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert set(head.state_dict()) == {"centroids.weight", "token_ordering"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@torch.inference_mode()
def test_e2b_assistant_graph_replay_uses_changed_tokens_positions_and_kv():
    torch.manual_seed(23)
    config = _assistant_config()
    reference = _assistant_hf_reference(config, dtype=torch.bfloat16)
    reference.masked_embedding.token_ordering.copy_(torch.randperm(config.vocab_size))
    assistant = Gemma4EdgeAssistantForCausalLM(config)
    state = {k: v for k, v in reference.state_dict().items() if k != "lm_head.weight"}
    assistant.load_state_dict(state)
    assistant = assistant.to("cuda").eval()
    assistant.materialize_buffers(torch.device("cuda"))
    inputs = torch.randn(
        2, 2 * config.backbone_hidden_size, device="cuda", dtype=torch.bfloat16
    )
    positions = torch.tensor([5, 7], device="cuda")
    local_key, local_value = (
        torch.randn(2, 3, 1, config.head_dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    global_key, global_value = (
        torch.randn(
            2, 6, 1, config.global_head_dim, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(2)
    )

    local_kv = (local_key, local_value)
    global_kv = (global_key, global_value)

    def forward():
        return assistant.forward_step(
            inputs,
            positions,
            local_wrapper=_assistant_Attention(),
            global_wrapper=_assistant_Attention(),
            local_kv=local_kv,
            global_kv=global_kv,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_logits, captured_hidden = forward()
    for step in range(3):
        inputs.copy_(torch.randn_like(inputs))
        positions.copy_(torch.tensor([63 + step, 511 + step], device="cuda"))
        for tensor in (*local_kv, *global_kv):
            tensor.copy_(torch.randn_like(tensor))
        expected_logits, expected_hidden = forward()
        expected_logits, expected_hidden = (
            expected_logits.clone(),
            expected_hidden.clone(),
        )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured_logits, expected_logits, atol=0, rtol=0)
        torch.testing.assert_close(captured_hidden, expected_hidden, atol=0, rtol=0)
