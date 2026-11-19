"""Backend-neutral GPT-OSS checkpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from resin.models.hf_config import PositiveInt, invalid_dimensions, parse_hf_config


@dataclass(frozen=True, slots=True)
class GptOssConfig:
    vocab_size: PositiveInt
    hidden_size: PositiveInt
    intermediate_size: PositiveInt
    num_hidden_layers: PositiveInt
    num_attention_heads: PositiveInt
    num_key_value_heads: PositiveInt
    num_local_experts: PositiveInt
    num_experts_per_tok: PositiveInt
    head_dim: PositiveInt
    sliding_window: PositiveInt
    rms_norm_eps: float
    rope_theta: float
    initial_context_length: PositiveInt
    rope_scaling_factor: float
    rope_beta_fast: float
    rope_beta_slow: float
    max_position_embeddings: PositiveInt
    swiglu_limit: float
    attention_bias: bool
    layer_types: tuple[str, ...]
    attention_dropout: float = 0.0
    pad_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        invalid = list(invalid_dimensions(self))
        if invalid:
            raise ValueError(f"GPT-OSS dimensions must be positive: {invalid}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_local_experts")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        unsupported = set(self.layer_types) - {
            "sliding_attention",
            "full_attention",
        }
        if unsupported:
            raise ValueError(f"unsupported GPT-OSS layer types: {sorted(unsupported)}")
        if self.rope_scaling_factor < 1.0:
            raise ValueError("rope scaling factor must be at least one")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.attention_dropout != 0.0:
            raise ValueError("the GPT-OSS inference runner requires zero dropout")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GptOssConfig:
        if config.get("model_type") != "gpt_oss":
            raise ValueError(
                f"expected a GPT-OSS checkpoint, got {config.get('model_type')!r}"
            )

        num_experts = _same_value(
            config,
            "num_local_experts",
            "num_experts",
            missing="GPT-OSS config is missing num_local_experts",
            conflict="conflicting GPT-OSS expert counts in config",
        )
        top_k = _same_value(
            config,
            "num_experts_per_tok",
            "experts_per_token",
            missing="GPT-OSS config is missing num_experts_per_tok",
            conflict="conflicting GPT-OSS top-k values in config",
        )
        rope = config.get("rope_scaling") or config.get("rope_parameters") or {}
        layer_types = config.get("layer_types")
        if layer_types is None:
            layer_types = [
                "sliding_attention" if index % 2 == 0 else "full_attention"
                for index in range(config["num_hidden_layers"])
            ]
        if config.get("tie_word_embeddings", False):
            raise ValueError("GPT-OSS requires untied input and output embeddings")

        return parse_hf_config(
            cls,
            config,
            num_local_experts=num_experts,
            num_experts_per_tok=top_k,
            head_dim=config.get(
                "head_dim", config["hidden_size"] // config["num_attention_heads"]
            ),
            rms_norm_eps=config.get("rms_norm_eps", 1e-5),
            rope_theta=rope.get("rope_theta", config.get("rope_theta", 150_000.0)),
            initial_context_length=rope.get(
                "original_max_position_embeddings",
                config.get("initial_context_length", 4096),
            ),
            rope_scaling_factor=rope.get(
                "factor", config.get("rope_scaling_factor", 1.0)
            ),
            rope_beta_fast=rope.get("beta_fast", config.get("rope_ntk_beta", 32.0)),
            rope_beta_slow=rope.get("beta_slow", config.get("rope_ntk_alpha", 1.0)),
            max_position_embeddings=config.get("max_position_embeddings", 131_072),
            swiglu_limit=config.get("swiglu_limit", 7.0),
            attention_bias=config.get("attention_bias", True),
            layer_types=tuple(layer_types),
            pad_token_id=config.get("pad_token_id"),
            eos_token_id=config.get("eos_token_id"),
        )


def _same_value(
    config: dict[str, Any],
    primary: str,
    alias: str,
    *,
    missing: str,
    conflict: str,
):
    first = config.get(primary)
    second = config.get(alias)
    if first is not None and second is not None and first != second:
        raise ValueError(conflict)
    value = first if first is not None else second
    if value is None:
        raise ValueError(missing)
    return value
