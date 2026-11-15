"""Text execution geometry of the original BF16 Gemma 4 E2B/E4B checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from resin.models.gemma4.config import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    _history_blocks_per_parent,
)
from resin.models.hf_config import (
    PositiveInt,
    invalid_dimensions,
    parse_hf_config,
    require_options,
)


@dataclass(frozen=True, slots=True)
class Gemma4EdgeConfig:
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
    hidden_size_per_layer_input: PositiveInt
    vocab_size_per_layer_input: PositiveInt
    num_kv_shared_layers: int
    use_double_wide_mlp: bool
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    final_logit_softcapping: float
    local_rope_theta: float
    global_rope_theta: float
    global_partial_rotary_factor: float
    image_token_id: int = 258880
    video_token_id: int = 258884

    def __post_init__(self) -> None:
        for name in invalid_dimensions(self):
            raise ValueError(f"{name} must be positive")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        if set(self.layer_types) - {SLIDING_ATTENTION, FULL_ATTENTION}:
            raise ValueError("unsupported Gemma 4 attention type")
        if not 0 <= self.num_kv_shared_layers < self.num_hidden_layers:
            raise ValueError("num_kv_shared_layers must leave at least one producer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by local KV heads")
        if self.num_attention_heads % self.num_global_key_value_heads:
            raise ValueError("query heads must be divisible by global KV heads")
        if not self.eos_token_ids or self.rms_norm_eps <= 0:
            raise ValueError("invalid EOS or normalization configuration")
        if not 0 < self.global_partial_rotary_factor <= 1:
            raise ValueError("invalid global rotary fraction")
        for kind in set(self.layer_types):
            if kind not in self.layer_types[: self.num_kv_producer_layers]:
                raise ValueError("shared attention type has no KV producer")

    @property
    def num_kv_producer_layers(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    @property
    def local_layer_count(self) -> int:
        return self.layer_types[: self.num_kv_producer_layers].count(SLIDING_ATTENTION)

    @property
    def global_layer_count(self) -> int:
        return self.layer_types[: self.num_kv_producer_layers].count(FULL_ATTENTION)

    @property
    def history_blocks_per_parent(self) -> int:
        return _history_blocks_per_parent(self)

    def cache_source_layer(self, index: int) -> int:
        if index < self.num_kv_producer_layers:
            return index
        kind = self.layer_types[index]
        return max(
            i for i in range(self.num_kv_producer_layers) if self.layer_types[i] == kind
        )

    def cache_layer_index(self, index: int) -> int:
        source = self.cache_source_layer(index)
        return self.layer_types[:source].count(self.layer_types[source])

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Gemma4EdgeConfig:
        if raw.get("model_type") != "gemma4":
            raise ValueError("expected a Gemma 4 checkpoint")
        text = raw.get("text_config") or {}
        required = {
            "model_type": "gemma4_text",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_k_eq_v": False,
            "enable_moe_block": False,
            "hidden_activation": "gelu_pytorch_tanh",
            "tie_word_embeddings": True,
            "use_bidirectional_attention": None,
        }
        require_options(text, required, "E2B text")
        if raw.get("quantization_config") or not raw.get("tie_word_embeddings"):
            raise ValueError("E2B runner requires original BF16 tied-embedding weights")
        rope = text.get("rope_parameters") or {}
        local = rope.get(SLIDING_ATTENTION) or {}
        global_ = rope.get(FULL_ATTENTION) or {}
        if (
            local.get("rope_type") != "default"
            or global_.get("rope_type") != "proportional"
        ):
            raise ValueError("unsupported E2B rotary configuration")
        eos = raw.get("eos_token_id", text.get("eos_token_id"))
        return parse_hf_config(
            cls,
            text,
            num_global_key_value_heads=text.get("num_global_key_value_heads")
            or text["num_key_value_heads"],
            layer_types=tuple(text["layer_types"]),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            eos_token_ids=(eos,) if isinstance(eos, int) else tuple(eos or ()),
            final_logit_softcapping=text.get("final_logit_softcapping", 0.0),
            use_double_wide_mlp=text.get("use_double_wide_mlp", False),
            local_rope_theta=local.get("rope_theta", 10000.0),
            global_rope_theta=global_["rope_theta"],
            global_partial_rotary_factor=global_.get("partial_rotary_factor", 1.0),
            image_token_id=raw.get("image_token_id", 258880),
            video_token_id=raw.get("video_token_id", 258884),
        )
