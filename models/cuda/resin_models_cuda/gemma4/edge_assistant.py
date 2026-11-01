"""Official Gemma 4 E2B/E4B BF16 MTP assistants with ordered vocabulary heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from resin.backend.loading import assign_model_tensor, read_config
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.models.gemma4.assistant import (
    Gemma4AssistantConfig,
    Gemma4AssistantForCausalLM,
)
from resin.models.gemma4.config import FULL_ATTENTION, SLIDING_ATTENTION

DEFAULT_ASSISTANT = "google/gemma-4-E2B-it-assistant"
DEFAULT_ASSISTANT_REVISION = "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df"


@dataclass(frozen=True, slots=True)
class Gemma4EdgeAssistantConfig(Gemma4AssistantConfig):
    num_centroids: int
    centroid_intermediate_top_k: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Gemma4EdgeAssistantConfig:
        text = raw.get("text_config") or {}
        required = {
            "model_type": "gemma4_text",
            "attention_bias": False,
            "attention_k_eq_v": False,
            "enable_moe_block": False,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 4,
            "tie_word_embeddings": True,
            "use_double_wide_mlp": False,
        }
        if raw.get("model_type") != "gemma4_assistant":
            raise ValueError("expected a Gemma 4 assistant checkpoint")
        mismatched = {
            key: (text.get(key), expected)
            for key, expected in required.items()
            if text.get(key) != expected
        }
        if mismatched:
            raise ValueError(f"unsupported E2B assistant options: {mismatched}")
        if not raw.get("use_ordered_embeddings"):
            raise ValueError("the E2B assistant requires its ordered vocabulary head")
        rope = text.get("rope_parameters") or {}
        local = rope.get(SLIDING_ATTENTION) or {}
        global_ = rope.get(FULL_ATTENTION) or {}
        config = cls(
            backbone_hidden_size=int(raw["backbone_hidden_size"]),
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            num_global_key_value_heads=int(
                text.get("num_global_key_value_heads") or text["num_key_value_heads"]
            ),
            head_dim=int(text["head_dim"]),
            global_head_dim=int(text["global_head_dim"]),
            sliding_window=int(text["sliding_window"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            layer_types=tuple(text["layer_types"]),
            local_rope_theta=float(local["rope_theta"]),
            global_rope_theta=float(global_["rope_theta"]),
            global_partial_rotary_factor=float(global_["partial_rotary_factor"]),
            num_centroids=int(raw["num_centroids"]),
            centroid_intermediate_top_k=int(raw["centroid_intermediate_top_k"]),
        )
        if (
            config.backbone_hidden_size != 1536
            or config.vocab_size != 262144
            or config.hidden_size != 256
            or config.intermediate_size != 2048
            or config.num_hidden_layers != 4
            or config.num_attention_heads != 4
            or config.num_key_value_heads != 1
            or config.num_global_key_value_heads != 1
            or config.head_dim != 256
            or config.global_head_dim != 512
            or config.sliding_window != 512
            or config.layer_types != (SLIDING_ATTENTION,) * 3 + (FULL_ATTENTION,)
            or config.num_centroids != 2048
            or config.centroid_intermediate_top_k != 32
        ):
            raise ValueError("assistant geometry does not match the official E2B model")
        return config


class Gemma4EdgeMaskedEmbedder(nn.Module):
    """Select Google's top centroid clusters, then score their canonical tokens.

    The masked value matches Transformers' global minimum-minus-one rule, but
    stays a device tensor so selection can be captured and replayed.
    """

    token_ordering: Tensor

    def __init__(
        self, config: Gemma4EdgeAssistantConfig, *, backend: str = "reference"
    ) -> None:
        super().__init__()
        if config.vocab_size % config.num_centroids:
            raise ValueError("assistant vocabulary must divide into centroid groups")
        if not 0 < config.centroid_intermediate_top_k <= config.num_centroids:
            raise ValueError("invalid assistant centroid selection width")
        self.vocab_size = config.vocab_size
        self.num_centroids = config.num_centroids
        self.top_k = config.centroid_intermediate_top_k
        self.hidden_size = config.hidden_size
        if backend not in {"reference", "triton"}:
            raise ValueError(
                "assistant sparse head backend must be reference or triton"
            )
        self.backend = backend
        self.centroids = nn.Linear(
            config.hidden_size,
            config.num_centroids,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.register_buffer(
            "token_ordering", torch.empty(config.vocab_size, dtype=torch.int64)
        )

    def forward(self, hidden: Tensor, lm_head_weight: Tensor) -> Tensor:
        leading = hidden.shape[:-1]
        flat = hidden.reshape(-1, self.hidden_size)
        selected_centroids = self.centroids(flat).topk(self.top_k, dim=-1).indices
        ordering = self.token_ordering.reshape(self.num_centroids, -1)
        selected_tokens = ordering[selected_centroids].reshape(len(flat), -1)
        if self.backend == "triton":
            from resin_models_cuda.gemma4.edge_kernels import sparse_gather_dot

            selected_logits = sparse_gather_dot(flat, lm_head_weight, selected_tokens)
        else:
            embeddings = lm_head_weight[selected_tokens]
            selected_logits = torch.bmm(
                flat[:, None, :], embeddings.transpose(1, 2)
            ).squeeze(1)
        mask_value = selected_logits.amin() - 1.0
        logits = mask_value.expand(len(flat), self.vocab_size).clone()
        logits.scatter_(1, selected_tokens, selected_logits)
        return logits.reshape(*leading, self.vocab_size)


class Gemma4EdgeAssistantForCausalLM(Gemma4AssistantForCausalLM):
    def __init__(self, config: Gemma4EdgeAssistantConfig) -> None:
        super().__init__(config)
        self.masked_embedding = Gemma4EdgeMaskedEmbedder(config)

    def forward_step(
        self,
        inputs_embeds: Tensor,
        positions: Tensor,
        *,
        local_wrapper,
        global_wrapper,
        local_kv: tuple[Tensor, Tensor],
        global_kv: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        hidden = self.pre_projection(inputs_embeds)
        embeddings = {
            kind: self.model.rotary_emb(positions, kind, hidden.dtype)
            for kind in set(self.config.layer_types)
        }
        for layer in self.model.layers:
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            cos, sin = embeddings[layer.self_attn.layer_type]
            query = layer.self_attn.project_q(normalized, cos, sin)
            local = layer.self_attn.is_local
            wrapper = local_wrapper if local else global_wrapper
            attended = wrapper.run(query, local_kv if local else global_kv)
            hidden = layer.self_attn.o_proj(attended.reshape(len(hidden), -1))
            hidden = residual + layer.post_attention_layernorm(hidden)
            residual = hidden
            hidden = layer.mlp(layer.pre_feedforward_layernorm(hidden))
            hidden = residual + layer.post_feedforward_layernorm(hidden)
            hidden = hidden * layer.layer_scalar
        hidden = self.model.norm(hidden)
        projected = self.post_projection(hidden)
        logits = self.masked_embedding(hidden, self.model.embed_tokens.weight)
        return logits, projected


def load_edge_assistant(model_path: str | Path, *, device, **_kwargs):
    root = Path(model_path).expanduser().resolve()
    config = Gemma4EdgeAssistantConfig.from_dict(read_config(root))
    with torch.device("meta"):
        model = Gemma4EdgeAssistantForCausalLM(config)
    tensors = indexed_tensor_files(root)
    expected = set(model.state_dict())
    if missing := expected - set(tensors):
        raise ValueError(f"E2B assistant is missing tensors: {sorted(missing)[:8]}")
    if unexpected := set(tensors) - expected:
        raise ValueError(
            f"E2B assistant has unexpected tensors: {sorted(unexpected)[:8]}"
        )
    with SafetensorReader() as reader:
        for key in sorted(expected):
            assign_model_tensor(
                model, key, reader.read(tensors[key], key), device=device
            )
    model.materialize_buffers(torch.device(device))
    model.eval()
    return model, config


__all__ = [
    "DEFAULT_ASSISTANT",
    "DEFAULT_ASSISTANT_REVISION",
    "Gemma4EdgeAssistantConfig",
    "Gemma4EdgeAssistantForCausalLM",
    "Gemma4EdgeMaskedEmbedder",
    "load_edge_assistant",
]
