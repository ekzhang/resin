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
from resin.models.hf_config import parse_hf_config, require_options

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
        require_options(text, required, "E2B assistant")
        if not raw.get("use_ordered_embeddings"):
            raise ValueError("the E2B assistant requires its ordered vocabulary head")
        rope = text.get("rope_parameters") or {}
        local = rope.get(SLIDING_ATTENTION) or {}
        global_ = rope.get(FULL_ATTENTION) or {}
        config = parse_hf_config(
            cls,
            text,
            coerce=True,
            backbone_hidden_size=int(raw["backbone_hidden_size"]),
            num_global_key_value_heads=int(
                text.get("num_global_key_value_heads") or text["num_key_value_heads"]
            ),
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

    _scale_inplace = False

    def logits(self, hidden: Tensor) -> Tensor:
        return self.masked_embedding(hidden, self.model.embed_tokens.weight)


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
