"""Geometry of the BF16 DSpark head bundled with Inkling-2-Small."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from resin.models.hf_config import (
    PositiveFloat,
    PositiveInt,
    invalid_dimensions,
    parse_hf_config,
    require_options,
)

if TYPE_CHECKING:
    from resin.models.inkling2.config import Inkling2Config


@dataclass(frozen=True, slots=True)
class Inkling2DSparkConfig:
    vocab_size: PositiveInt
    hidden_size: PositiveInt
    intermediate_size: PositiveInt
    num_hidden_layers: PositiveInt
    num_attention_heads: PositiveInt
    num_key_value_heads: PositiveInt
    head_dim: PositiveInt
    rms_norm_eps: PositiveFloat
    max_position_embeddings: PositiveInt
    rope_theta: PositiveFloat
    sliding_window_size: PositiveInt
    block_size: PositiveInt
    markov_rank: PositiveInt
    local_layer_ids: tuple[int, ...]
    target_layer_ids: tuple[int, ...]

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        if values.get("model_type") != "inkling2":
            raise ValueError("Inkling2 DSpark requires the bundled Inkling2 config")
        draft, text = values["mtp_config"], values["text_config"]
        require_options(
            draft,
            {
                "torch_dtype": "bfloat16",
                "q_bias": False,
                "o_bias": False,
                "markov_head_type": "rnn_hidden",
                "block_size": 8,
                "num_hidden_layers": 5,
            },
            "Inkling2 DSpark",
        )
        result = parse_hf_config(
            cls,
            draft,
            coerce=True,
            vocab_size=int(text["vocab_size"]),
            local_layer_ids=tuple(int(i) for i in draft["local_layer_ids"]),
            target_layer_ids=tuple(int(i) for i in draft["target_layer_ids"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if invalid := list(invalid_dimensions(self)):
            raise ValueError(f"Inkling2 DSpark fields must be positive: {invalid}")
        if self.block_size < 2 or self.head_dim % 2:
            raise ValueError("DSpark requires an anchor, drafts, and even rotary width")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("DSpark query heads must be divisible by KV heads")
        if (
            len(set(self.local_layer_ids)) != len(self.local_layer_ids)
            or any(not 0 <= i < self.num_hidden_layers for i in self.local_layer_ids)
            or not self.target_layer_ids
            or tuple(sorted(set(self.target_layer_ids))) != self.target_layer_ids
            or self.target_layer_ids[0] < 0
        ):
            raise ValueError("invalid Inkling2 DSpark layer IDs")

    def validate_target(self, target: Inkling2Config) -> None:
        self.validate()
        if (self.vocab_size, self.hidden_size) != (
            target.vocab_size,
            target.hidden_size,
        ):
            raise ValueError("Inkling2 target and DSpark embedding geometries differ")
        # The index equal to the layer count denotes the final normalized output.
        if self.target_layer_ids[-1] > target.num_hidden_layers:
            raise ValueError("DSpark target tap lies outside Inkling2")

    @property
    def gamma(self) -> int:
        return self.block_size - 1

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
