"""Backend-neutral Gemma 4 assistant over injected paged-attention calls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin.backend.layers import RMSNorm
from resin.models.gemma4.config import FULL_ATTENTION, SLIDING_ATTENTION
from resin.models.gemma4.model import RotaryEmbedding, apply_rotary


@dataclass(frozen=True, slots=True)
class Gemma4AssistantConfig:
    backbone_hidden_size: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int
    global_head_dim: int
    sliding_window: int
    max_position_embeddings: int
    rms_norm_eps: float
    layer_types: tuple[str, ...]
    local_rope_theta: float
    global_rope_theta: float
    global_partial_rotary_factor: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Gemma4AssistantConfig:
        text = raw.get("text_config") or {}
        required = {
            "model_type": "gemma4_text",
            "attention_bias": False,
            "attention_k_eq_v": True,
            "enable_moe_block": False,
            "hidden_activation": "gelu_pytorch_tanh",
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
            raise ValueError(f"unsupported Gemma 4 assistant options: {mismatched}")
        if raw.get("use_ordered_embeddings", False):
            raise ValueError("ordered Gemma 4 assistant embeddings are unsupported")
        layer_types = tuple(text.get("layer_types") or ())
        if layer_types != (
            SLIDING_ATTENTION,
            SLIDING_ATTENTION,
            SLIDING_ATTENTION,
            FULL_ATTENTION,
        ):
            raise ValueError(
                "the Gemma 4 assistant must have three local and one global layer"
            )
        rope = text.get("rope_parameters") or {}
        local_rope = rope.get(SLIDING_ATTENTION) or {}
        global_rope = rope.get(FULL_ATTENTION) or {}
        config = cls(
            backbone_hidden_size=int(raw["backbone_hidden_size"]),
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            num_global_key_value_heads=int(text["num_global_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            global_head_dim=int(text["global_head_dim"]),
            sliding_window=int(text["sliding_window"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            layer_types=layer_types,
            local_rope_theta=float(local_rope["rope_theta"]),
            global_rope_theta=float(global_rope["rope_theta"]),
            global_partial_rotary_factor=float(global_rope["partial_rotary_factor"]),
        )
        if (
            config.backbone_hidden_size != 5376
            or config.vocab_size != 262144
            or config.hidden_size != 1024
            or config.intermediate_size != 8192
            or config.num_hidden_layers != 4
            or config.num_attention_heads != 32
            or config.num_key_value_heads != 16
            or config.num_global_key_value_heads != 4
            or config.head_dim != 256
            or config.global_head_dim != 512
            or config.sliding_window != 1024
        ):
            raise ValueError("the Gemma 4 assistant geometry does not match 31B")
        return config


class Gemma4AssistantAttention(nn.Module):
    def __init__(self, config: Gemma4AssistantConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        self.is_local = self.layer_type == SLIDING_ATTENTION
        self.head_dim = config.head_dim if self.is_local else config.global_head_dim
        self.num_heads = config.num_attention_heads
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )

    def project_q(self, hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        query = self.q_proj(hidden).view(-1, self.num_heads, self.head_dim)
        query = self.q_norm(query)
        return apply_rotary(query, cos, sin)


class Gemma4AssistantMLP(nn.Module):
    def __init__(self, config: Gemma4AssistantConfig) -> None:
        super().__init__()
        options = {"bias": False, "dtype": torch.bfloat16}
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, **options
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, **options
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, **options
        )

    def forward(self, hidden: Tensor) -> Tensor:
        gate = F.gelu(self.gate_proj(hidden), approximate="tanh")
        return self.down_proj(gate * self.up_proj(hidden))


class Gemma4AssistantLayer(nn.Module):
    layer_scalar: Tensor

    def __init__(self, config: Gemma4AssistantConfig, layer_index: int) -> None:
        super().__init__()
        self.self_attn = Gemma4AssistantAttention(config, layer_index)
        self.mlp = Gemma4AssistantMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.register_buffer(
            "layer_scalar", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )


class Gemma4AssistantTextModel(nn.Module):
    def __init__(self, config: Gemma4AssistantConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16
        )
        self.layers = cast(
            Sequence[Gemma4AssistantLayer],
            nn.ModuleList(
                Gemma4AssistantLayer(config, index)
                for index in range(config.num_hidden_layers)
            ),
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)


class Gemma4AssistantForCausalLM(nn.Module):
    """Four dense Q-only layers sharing the target model's last local/global K/V."""

    def __init__(self, config: Gemma4AssistantConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma4AssistantTextModel(config)
        self.pre_projection = nn.Linear(
            2 * config.backbone_hidden_size,
            config.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.post_projection = nn.Linear(
            config.hidden_size,
            config.backbone_hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )

    def materialize_buffers(self, device: torch.device) -> None:
        self.model.rotary_emb.materialize(device)

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
            wrapper = local_wrapper if layer.self_attn.is_local else global_wrapper
            kv = local_kv if layer.self_attn.is_local else global_kv
            attended = wrapper.run(query, kv)
            hidden = layer.self_attn.o_proj(attended.reshape(len(hidden), -1))
            hidden = residual + layer.post_attention_layernorm(hidden)

            residual = hidden
            hidden = layer.pre_feedforward_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + layer.post_feedforward_layernorm(hidden)
            hidden *= layer.layer_scalar

        hidden = self.model.norm(hidden)
        projected = self.post_projection(hidden)
        logits = F.linear(hidden, self.model.embed_tokens.weight)
        return logits, projected


__all__ = [
    "Gemma4AssistantConfig",
    "Gemma4AssistantForCausalLM",
]
