"""Backend-neutral configuration for Gemma 4 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from resin.models.hf_config import (
    PositiveInt,
    invalid_dimensions,
    parse_hf_config,
    require_options,
)

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"


@dataclass(frozen=True, slots=True)
class Gemma4Nvfp4Config:
    vocab_size: PositiveInt
    hidden_size: PositiveInt
    intermediate_size: PositiveInt
    num_hidden_layers: PositiveInt
    num_attention_heads: PositiveInt
    num_key_value_heads: PositiveInt
    num_global_key_value_heads: PositiveInt
    head_dim: PositiveInt
    global_head_dim: PositiveInt
    sliding_window: PositiveInt
    max_position_embeddings: PositiveInt
    rms_norm_eps: float
    layer_types: tuple[str, ...]
    image_token_id: int
    video_token_id: int
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    final_logit_softcapping: float
    local_rope_theta: float
    global_rope_theta: float
    global_partial_rotary_factor: float
    group_size: PositiveInt

    def __post_init__(self) -> None:
        invalid = list(invalid_dimensions(self))
        if invalid:
            raise ValueError(f"Gemma 4 dimensions must be positive: {invalid}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        unsupported = set(self.layer_types) - {SLIDING_ATTENTION, FULL_ATTENTION}
        if unsupported:
            raise ValueError(f"unsupported Gemma 4 layer types: {sorted(unsupported)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("local query heads must be divisible by KV heads")
        if self.num_attention_heads % self.num_global_key_value_heads:
            raise ValueError("global query heads must be divisible by KV heads")
        if self.head_dim % 2 or self.global_head_dim % 2:
            raise ValueError("attention head dimensions must be even")
        if not 0.0 < self.global_partial_rotary_factor <= 1.0:
            raise ValueError("global_partial_rotary_factor must be in (0, 1]")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if not self.eos_token_ids:
            raise ValueError("eos_token_ids must not be empty")

    @property
    def local_layer_count(self) -> int:
        return self.layer_types.count(SLIDING_ATTENTION)

    @property
    def global_layer_count(self) -> int:
        return self.layer_types.count(FULL_ATTENTION)

    @property
    def history_blocks_per_parent(self) -> int:
        """Pack global pages into a parent the same size as one local page."""

        return _history_blocks_per_parent(self)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Gemma4Nvfp4Config:
        if config.get("model_type") != "gemma4":
            raise ValueError(
                f"expected a Gemma 4 checkpoint, got {config.get('model_type')!r}"
            )
        if config.get("audio_config") is not None:
            raise ValueError("the CUDA runner does not implement Gemma 4 audio inputs")
        text = config.get("text_config") or {}
        if text.get("model_type") != "gemma4_text":
            raise ValueError("checkpoint is missing a Gemma 4 text configuration")
        required = {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_k_eq_v": True,
            "enable_moe_block": False,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "tie_word_embeddings": True,
            "use_bidirectional_attention": "vision",
        }
        require_options(text, required, "Gemma 4 text")
        if not config.get("tie_word_embeddings", False):
            raise ValueError("the NVFP4 runner requires tied input/output embeddings")

        quantization = config.get("quantization_config") or {}
        groups = quantization.get("config_groups") or {}
        group = groups.get("group_0") or {}
        weights = group.get("weights") or {}
        activations = group.get("input_activations") or {}
        if (
            weights.get("num_bits") != 4
            or weights.get("type") != "float"
            or weights.get("dynamic") is not False
            or activations.get("num_bits") != 4
            or activations.get("type") != "float"
            or activations.get("dynamic") is not False
            or weights.get("group_size") != activations.get("group_size")
        ):
            raise ValueError(
                "checkpoint is not static ModelOpt NVFP4 with matching groups"
            )

        rope = text.get("rope_parameters") or {}
        local_rope = rope.get(SLIDING_ATTENTION) or {}
        global_rope = rope.get(FULL_ATTENTION) or {}
        if local_rope.get("rope_type") != "default":
            raise ValueError("local attention requires default RoPE")
        if global_rope.get("rope_type") != "proportional":
            raise ValueError("global attention requires proportional RoPE")

        eos = config.get("eos_token_id", text.get("eos_token_id"))
        eos_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
        return parse_hf_config(
            cls,
            text,
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            layer_types=tuple(text["layer_types"]),
            image_token_id=config["image_token_id"],
            video_token_id=config["video_token_id"],
            eos_token_ids=eos_ids,
            local_rope_theta=local_rope.get("rope_theta", 10_000.0),
            global_rope_theta=global_rope["rope_theta"],
            global_partial_rotary_factor=global_rope.get("partial_rotary_factor", 1.0),
            group_size=weights["group_size"],
        )


def _history_blocks_per_parent(config) -> int:
    local = config.local_layer_count * config.num_key_value_heads * config.head_dim
    global_ = (
        config.global_layer_count
        * config.num_global_key_value_heads
        * config.global_head_dim
    )
    if not global_ or local % global_:
        raise ValueError("local/global KV geometry does not share an integral parent")
    return local // global_


__all__ = [
    "FULL_ATTENTION",
    "Gemma4Nvfp4Config",
    "SLIDING_ATTENTION",
]
