"""Backend-neutral text configuration for Qwen3.5-family checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self, cast


@dataclass(frozen=True, slots=True)
class Qwen35Config:
    """Hybrid-attention geometry used by Qwen3.5, Qwen3.8 and Qwen3.5-MoE.

    Dense checkpoints set ``intermediate_size``. Mixture-of-experts checkpoints
    set the expert fields instead and leave ``intermediate_size`` at zero.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, int, int] | None
    layer_types: tuple[str, ...]
    mtp_num_hidden_layers: int
    mtp_use_dedicated_embeddings: bool
    eos_token_id: int
    bos_token_id: int
    dtype: str = "bfloat16"
    mamba_ssm_dtype: str = "float32"
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    output_gate_type: str = "swish"
    tie_word_embeddings: bool = False
    is_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        """Parse either the multimodal root config or its text sub-config."""

        root_type = values.get("model_type")
        if "text_config" in values:
            if root_type not in _TEXT_TYPES:
                raise ValueError(
                    f"expected a Qwen3.5-family checkpoint, got {root_type!r}"
                )
            text = values["text_config"]
            expected = (_TEXT_TYPES[root_type],)
        else:
            text = values
            expected = tuple(_TEXT_TYPES.values())
        if not isinstance(text, dict) or text.get("model_type") not in expected:
            raise ValueError("Qwen3.5 checkpoint is missing qwen3_5_text config")
        moe = text["model_type"] == _TEXT_TYPES["qwen3_5_moe"]

        rope = text.get("rope_parameters") or {}
        if not isinstance(rope, dict):
            raise ValueError("Qwen3.5 rope_parameters must be an object")
        config = cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=0 if moe else int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            linear_num_key_heads=int(text["linear_num_key_heads"]),
            linear_num_value_heads=int(text["linear_num_value_heads"]),
            linear_key_head_dim=int(text["linear_key_head_dim"]),
            linear_value_head_dim=int(text["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(text["linear_conv_kernel_dim"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10_000.0))),
            partial_rotary_factor=float(
                rope.get(
                    "partial_rotary_factor",
                    text.get("partial_rotary_factor", 1.0),
                )
            ),
            mrope_section=(
                cast(
                    tuple[int, int, int],
                    tuple(int(section) for section in rope["mrope_section"]),
                )
                if "mrope_section" in rope
                else None
            ),
            layer_types=tuple(str(kind) for kind in text["layer_types"]),
            mtp_num_hidden_layers=int(text.get("mtp_num_hidden_layers", 0)),
            mtp_use_dedicated_embeddings=bool(
                text.get("mtp_use_dedicated_embeddings", False)
            ),
            eos_token_id=int(text["eos_token_id"]),
            bos_token_id=int(text["bos_token_id"]),
            dtype=str(text.get("dtype", text.get("torch_dtype", "bfloat16"))),
            mamba_ssm_dtype=str(text.get("mamba_ssm_dtype", "float32")),
            hidden_act=str(text.get("hidden_act", "silu")),
            attention_bias=bool(text.get("attention_bias", False)),
            attention_dropout=float(text.get("attention_dropout", 0.0)),
            attn_output_gate=bool(text.get("attn_output_gate", True)),
            output_gate_type=str(text.get("output_gate_type", "swish")),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            num_experts=int(text["num_experts"]) if moe else 0,
            num_experts_per_tok=int(text["num_experts_per_tok"]) if moe else 0,
            moe_intermediate_size=int(text["moe_intermediate_size"]) if moe else 0,
            shared_expert_intermediate_size=(
                int(text["shared_expert_intermediate_size"]) if moe else 0
            ),
            norm_topk_prob=bool(text.get("norm_topk_prob", True)),
            is_moe=moe,
        )
        config.validate()
        return config

    def validate(self) -> None:
        dimensions = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "max_position_embeddings": self.max_position_embeddings,
        }
        if self.is_moe:
            dimensions |= {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_intermediate_size": self.moe_intermediate_size,
                "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            }
        else:
            dimensions["intermediate_size"] = self.intermediate_size
        if invalid := [name for name, value in dimensions.items() if value <= 0]:
            raise ValueError(f"Qwen3.5 dimensions must be positive: {invalid}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        unsupported_layers = set(self.layer_types) - {
            "linear_attention",
            "full_attention",
        }
        if unsupported_layers:
            raise ValueError(
                f"unsupported Qwen3.5 layer types: {sorted(unsupported_layers)}"
            )
        if "linear_attention" not in self.layer_types:
            raise ValueError("Qwen3.5 runner requires linear-attention layers")
        if "full_attention" not in self.layer_types:
            raise ValueError("Qwen3.5 runner requires full-attention layers")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("experts per token cannot exceed the expert count")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear value heads must be divisible by key heads")
        rotary_dim = self.rotary_dim
        if rotary_dim <= 0 or rotary_dim > self.head_dim or rotary_dim % 2:
            raise ValueError("partial rotary dimension must be positive and even")
        if self.mrope_section is not None and (
            len(self.mrope_section) != 3
            or any(section <= 0 for section in self.mrope_section)
            or sum(self.mrope_section) != rotary_dim // 2
        ):
            raise ValueError("mrope_section must partition half the rotary dimension")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("Qwen3.5 norm epsilon and RoPE theta must be positive")

        required = {
            "dtype": (self.dtype, "bfloat16"),
            "mamba_ssm_dtype": (self.mamba_ssm_dtype, "float32"),
            "hidden_act": (self.hidden_act, "silu"),
            "attention_bias": (self.attention_bias, False),
            "attention_dropout": (self.attention_dropout, 0.0),
            "attn_output_gate": (self.attn_output_gate, True),
            "output_gate_type": (self.output_gate_type, "swish"),
            "tie_word_embeddings": (self.tie_word_embeddings, False),
        }
        if unsupported := {
            name: actual
            for name, (actual, expected) in required.items()
            if actual != expected
        }:
            raise ValueError(f"unsupported Qwen3.5 options: {unsupported}")
        if self.mtp_num_hidden_layers not in (0, 1):
            raise ValueError("Qwen3.5 runner supports at most one bundled MTP layer")
        if self.mtp_use_dedicated_embeddings:
            raise ValueError("dedicated Qwen3.5 MTP embeddings are not supported")

    def require_dense(self) -> None:
        if self.is_moe:
            raise ValueError("this runner does not support Qwen3.5-MoE checkpoints")

    def require_mtp(self) -> None:
        if self.mtp_num_hidden_layers != 1:
            raise ValueError("this Qwen3.5 checkpoint does not contain an MTP head")

    @property
    def full_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, kind in enumerate(self.layer_types)
            if kind == "full_attention"
        )

    @property
    def linear_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, kind in enumerate(self.layer_types)
            if kind == "linear_attention"
        )

    @property
    def full_layer_count(self) -> int:
        return len(self.full_layer_ids)

    @property
    def linear_layer_count(self) -> int:
        return len(self.linear_layer_ids)

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim


# Root multimodal model types and their text sub-config types.
_TEXT_TYPES = {"qwen3_5": "qwen3_5_text", "qwen3_5_moe": "qwen3_5_moe_text"}


__all__ = ["Qwen35Config"]
