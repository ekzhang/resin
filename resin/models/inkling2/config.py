"""Checkpoint geometry for Inkling 2 Small and Mini."""

from __future__ import annotations

import math
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
class Inkling2LoopConfig:
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int


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
    loop: Inkling2LoopConfig | None = None

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        model_type = values.get("model_type")
        looped = model_type == "inkling2_loop"
        architecture = (
            "Inkling2LoopForConditionalGeneration"
            if looped
            else "Inkling2ForConditionalGeneration"
        )
        if model_type not in ("inkling2", "inkling2_loop") or values.get(
            "architectures"
        ) != [architecture]:
            raise ValueError("unsupported Inkling 2 checkpoint architecture")
        text = values.get("text_config")
        if not isinstance(text, dict):
            raise ValueError("Inkling 2 text_config must be an object")
        loop = None
        if looped:
            loop_values = text.get("loop_config")
            if not isinstance(loop_values, dict):
                raise ValueError("Inkling 2 Mini loop_config must be an object")
            if loop_values.get("is_varloop", False) is not False:
                raise ValueError("Inkling 2 supports fixed loops only")
            loop = parse_hf_config(Inkling2LoopConfig, loop_values)
        pre_affine_options = {
            "pre_norm_pre_affine": False,
            "final_norm_pre_affine": False,
        }
        require_options(
            pre_affine_options | text if looped else text,
            {
                "torch_dtype": "bfloat16",
                "q_bias": False,
                "o_bias": False,
                **pre_affine_options,
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
            loop=loop,
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
        if (loop := self.loop) is not None:
            if any(
                type(value) is not int
                for value in (
                    loop.core_start,
                    loop.core_end,
                    loop.num_passes,
                    loop.gate_rank,
                )
            ) or not (
                0 < loop.core_start < loop.core_end < self.num_hidden_layers
                and loop.num_passes >= 2
                and loop.gate_rank > 0
            ):
                raise ValueError("invalid Inkling 2 middle-layer loop geometry")
            if (
                loop.core_start % self.n_layers_per_attn_res_block
                or loop.core_end % self.n_layers_per_attn_res_block
            ):
                raise ValueError(
                    "Inkling 2 loop must align to residual block boundaries"
                )

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

    @property
    def layer_order(self) -> tuple[int, ...]:
        """Checkpoint layer IDs in execution order, including repeated visits."""
        layers = tuple(range(self.num_hidden_layers))
        if (loop := self.loop) is None:
            return layers
        start, end = loop.core_start, loop.core_end
        return layers[:start] + layers[start:end] * loop.num_passes + layers[end:]

    @property
    def execution_layer_count(self) -> int:
        return len(self.layer_order)

    @property
    def execution_local_layer_ids(self) -> tuple[int, ...]:
        """Execution indices of visits using sliding-window attention."""
        if self.loop is None:
            return self.local_layer_ids
        return tuple(
            i
            for i, layer in enumerate(self.layer_order)
            if layer in self.local_layer_ids
        )

    @property
    def execution_global_layer_ids(self) -> tuple[int, ...]:
        if self.loop is None:
            return self.global_layer_ids
        return tuple(
            i
            for i, layer in enumerate(self.layer_order)
            if layer not in self.local_layer_ids
        )

    @property
    def execution_local_layer_count(self) -> int:
        return len(self.execution_local_layer_ids)

    @property
    def execution_global_layer_count(self) -> int:
        return self.execution_layer_count - self.execution_local_layer_count

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


def validate_vision_config(vision_config: dict, text_hidden_size: int) -> None:
    """Validate the native Inkling vision tower and its text projection."""
    for key in (
        "temporal_patch_size",
        "native_patch_size",
        "adapter_merge_size",
        "projection_hidden_dim",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "posemb_grid_size",
    ):
        value = vision_config[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"vision_config.{key} must be a positive integer")
    if type(text_hidden_size) is not int or text_hidden_size <= 0:
        raise ValueError("text_hidden_size must be a positive integer")
    for key in ("mlp_ratio", "layer_norm_eps"):
        value = vision_config[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"vision_config.{key} must be finite and positive")
    width, heads = (
        vision_config["hidden_size"],
        vision_config["num_attention_heads"],
    )
    if width % heads or width // heads not in (4, 8, 16, 32, 64, 128):
        raise ValueError("Vision head dimension must be one of 4, 8, 16, 32, 64, 128")
    intermediate_size = width * vision_config["mlp_ratio"]
    if not math.isfinite(intermediate_size) or intermediate_size != int(
        intermediate_size
    ):
        raise ValueError("Vision MLP intermediate size must be an integer")
    if type(vision_config["projector_bias"]) is not bool:
        raise ValueError("vision_config.projector_bias must be a boolean")


__all__ = ["Inkling2Config", "Inkling2LoopConfig", "validate_vision_config"]
