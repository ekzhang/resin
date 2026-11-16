"""Checkpoint geometry for Inkling 2 Small."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from resin.models.hf_config import (
    PositiveFloat,
    PositiveInt,
    invalid_dimensions,
    parse_hf_config,
    require_options,
)


@dataclass(frozen=True, slots=True)
class Inkling2Config:
    vocab_size: PositiveInt
    unpadded_vocab_size: PositiveInt
    hidden_size: PositiveInt
    num_hidden_layers: PositiveInt
    num_attention_heads: PositiveInt
    num_key_value_heads: PositiveInt
    head_dim: PositiveInt
    swa_num_attention_heads: PositiveInt
    swa_num_key_value_heads: PositiveInt
    swa_head_dim: PositiveInt
    d_rel: PositiveInt
    rel_extent: PositiveInt
    local_layer_ids: tuple[int, ...]
    sliding_window_size: PositiveInt
    sconv_kernel_size: PositiveInt
    n_layers_per_attn_res_block: PositiveInt
    attn_res_normalize_values: bool
    intermediate_size: PositiveInt
    num_routed_experts: PositiveInt
    experts_per_token: PositiveInt
    route_scale: PositiveFloat
    mlp_gate_softcap: PositiveFloat
    mlp_up_softcap: PositiveFloat
    qk_l2_clip_threshold: PositiveFloat
    rel_l2_clip_threshold: PositiveFloat
    rms_norm_eps: PositiveFloat
    max_position_embeddings: PositiveInt
    log_scaling_n_floor: PositiveInt
    log_scaling_alpha: float
    final_logit_softcapping: PositiveFloat
    eos_token_id: int

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        if values.get("model_type") != "inkling2" or values.get("architectures") != [
            "Inkling2ForConditionalGeneration"
        ]:
            raise ValueError(
                "Inkling 2 requires an Inkling2ForConditionalGeneration checkpoint"
            )
        text = values.get("text_config")
        if not isinstance(text, dict):
            raise ValueError("Inkling 2 text_config must be an object")
        require_options(
            text,
            {
                "torch_dtype": "bfloat16",
                "q_bias": False,
                "o_bias": False,
                "pre_norm_pre_affine": False,
                "final_norm_pre_affine": False,
                "use_value_norm_no_gamma": True,
                "n_shared_experts": 0,
                "use_gate_bias": True,
            },
            "Inkling 2",
        )
        result = parse_hf_config(
            cls,
            text,
            coerce=True,
            local_layer_ids=tuple(int(i) for i in text["local_layer_ids"]),
            num_routed_experts=int(text["n_routed_experts"]),
            experts_per_token=int(text["num_experts_per_tok"]),
            max_position_embeddings=int(text["model_max_length"]),
            eos_token_id=int(values["eos_token_id"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if invalid := list(invalid_dimensions(self)):
            raise ValueError(f"Inkling 2 fields must be positive: {invalid}")
        if self.unpadded_vocab_size > self.vocab_size:
            raise ValueError(
                "Inkling 2 unpadded vocabulary exceeds its embedding table"
            )
        if not 0 <= self.eos_token_id < self.unpadded_vocab_size:
            raise ValueError("Inkling 2 EOS token lies outside its output vocabulary")
        if len(set(self.local_layer_ids)) != len(self.local_layer_ids) or any(
            not 0 <= i < self.num_hidden_layers for i in self.local_layer_ids
        ):
            raise ValueError("Inkling 2 local layer IDs must be unique and in range")
        if (
            self.num_attention_heads % self.num_key_value_heads
            or self.swa_num_attention_heads % self.swa_num_key_value_heads
        ):
            raise ValueError("Inkling 2 query heads must be divisible by KV heads")
        if (self.swa_head_dim, self.swa_num_key_value_heads) != (
            self.head_dim,
            self.num_key_value_heads,
        ):
            raise ValueError(
                "Inkling 2 cache requires equal local and global KV geometry"
            )
        if self.experts_per_token > self.num_routed_experts:
            raise ValueError("Inkling 2 routes more experts than it owns")
        if self.sconv_kernel_size != 4:
            raise ValueError("Inkling 2 kernels require four convolution taps")
        if self.n_layers_per_attn_res_block > self.num_hidden_layers:
            raise ValueError("Inkling 2 residual blocks exceed the layer count")

    @property
    def local_layer_count(self) -> int:
        return len(self.local_layer_ids)

    @property
    def global_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            i for i in range(self.num_hidden_layers) if i not in self.local_layer_ids
        )

    @property
    def global_layer_count(self) -> int:
        return self.num_hidden_layers - self.local_layer_count

    def state_width(self, attention_tp_size: int = 1) -> int:
        if (
            type(attention_tp_size) is not int
            or attention_tp_size <= 0
            or self.num_key_value_heads % attention_tp_size
        ):
            raise ValueError("attention TP must divide Inkling 2 KV heads")
        return (
            2 * (self.num_key_value_heads // attention_tp_size) * self.head_dim
            + 2 * self.hidden_size
        )


__all__ = ["Inkling2Config"]
