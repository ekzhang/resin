"""Checkpoint-shaped Gemma 4 text modules and Hugging Face vision tower."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin.backend.layers import RMSNorm
from resin.models.gemma4.config import SLIDING_ATTENTION, Gemma4Nvfp4Config
from resin.models.gemma4.model import RotaryEmbedding, apply_rotary
from resin_models_cuda.nvfp4 import Nvfp4LinearWeights

if TYPE_CHECKING:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4Nvfp4Config, layer_index: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        self.is_local = self.layer_type == SLIDING_ATTENTION
        self.head_dim = config.head_dim if self.is_local else config.global_head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = (
            config.num_key_value_heads
            if self.is_local
            else config.num_global_key_value_heads
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.v_proj = (
            nn.Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=False,
                dtype=torch.bfloat16,
            )
            if self.is_local
            else None
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)
        self.qkv_proj: nn.Linear | None = None

    def pack_projections(self) -> None:
        if self.qkv_proj is not None:
            raise RuntimeError("attention projections were already packed")
        projections = [self.q_proj, self.k_proj]
        if self.v_proj is not None:
            projections.append(self.v_proj)
        weight = torch.cat([projection.weight for projection in projections])
        packed = nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=False,
            device=weight.device,
            dtype=weight.dtype,
        )
        packed.weight = nn.Parameter(weight, requires_grad=False)
        self.qkv_proj = packed
        del self.q_proj
        del self.k_proj
        del self.v_proj

    def project_qkv(
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.qkv_proj is None:
            raise RuntimeError("attention projections must be packed before inference")
        projected = self.qkv_proj(hidden)
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        if self.is_local:
            query, key, value = projected.split((q_width, kv_width, kv_width), dim=-1)
        else:
            query, key = projected.split((q_width, kv_width), dim=-1)
            value = key
        query = self.q_norm(query.view(-1, self.num_heads, self.head_dim))
        key = self.k_norm(key.view(-1, self.num_kv_heads, self.head_dim))
        value = self.v_norm(value.view(-1, self.num_kv_heads, self.head_dim))
        return (
            apply_rotary(query, cos, sin),
            apply_rotary(key, cos, sin),
            value,
        )


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4Nvfp4Config) -> None:
        super().__init__()
        self.gate_proj = Nvfp4LinearWeights(
            config.hidden_size, config.intermediate_size, config.group_size
        )
        self.up_proj = Nvfp4LinearWeights(
            config.hidden_size, config.intermediate_size, config.group_size
        )
        self.down_proj = Nvfp4LinearWeights(
            config.intermediate_size, config.hidden_size, config.group_size
        )

    def forward(self, hidden: Tensor) -> Tensor:
        gate = F.gelu(self.gate_proj(hidden), approximate="tanh")
        return self.down_proj(gate * self.up_proj(hidden))


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4Nvfp4Config, layer_index: int) -> None:
        super().__init__()
        self.self_attn = Gemma4Attention(config, layer_index)
        self.mlp = Gemma4MLP(config)
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


class ScaledEmbedding(nn.Embedding):
    def __init__(self, config: Gemma4Nvfp4Config) -> None:
        super().__init__(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=torch.bfloat16,
        )
        self.scale = math.sqrt(config.hidden_size)
        self.register_buffer("scale_tensor", None, persistent=False)

    def materialize_scale(self) -> None:
        self.scale_tensor = self.weight.new_tensor(self.scale)

    def forward(self, input: Tensor) -> Tensor:
        if self.scale_tensor is None:
            raise RuntimeError("embedding scale must be materialized before inference")
        return super().forward(input) * self.scale_tensor


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4Nvfp4Config) -> None:
        super().__init__()
        self.embed_tokens = ScaledEmbedding(config)
        self.layers = cast(
            Sequence[Gemma4DecoderLayer],
            nn.ModuleList(
                Gemma4DecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ),
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)


class _Gemma4ModelContainer(nn.Module):
    def __init__(
        self, config: Gemma4Nvfp4Config, transformers_config: Gemma4Config
    ) -> None:
        super().__init__()
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4TextConfig,
            Gemma4VisionConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4MultimodalEmbedder,
            Gemma4VisionModel,
        )

        self.language_model = Gemma4TextModel(config)
        vision_config = transformers_config.vision_config
        text_config = transformers_config.text_config
        assert isinstance(vision_config, Gemma4VisionConfig)
        assert isinstance(text_config, Gemma4TextConfig)
        self.vision_tower = Gemma4VisionModel(vision_config)
        self.embed_vision = Gemma4MultimodalEmbedder(vision_config, text_config)


class Gemma4ForConditionalGeneration(nn.Module):
    """Only the modules present in the NVIDIA image/video checkpoint."""

    def __init__(
        self, config: Gemma4Nvfp4Config, transformers_config: Gemma4Config
    ) -> None:
        super().__init__()
        self.config = config
        self.transformers_config = transformers_config
        self.model = _Gemma4ModelContainer(config, transformers_config)

    @property
    def language_model(self) -> Gemma4TextModel:
        return self.model.language_model

    def materialize_buffers(self, device: torch.device) -> None:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4VisionRotaryEmbedding,
        )

        self.language_model.rotary_emb.materialize(device)
        vision_config = self.transformers_config.vision_config
        assert isinstance(vision_config, Gemma4VisionConfig)
        self.model.vision_tower.encoder.rotary_emb = Gemma4VisionRotaryEmbedding(
            vision_config
        ).to(device)

    def prepare_attention(self) -> None:
        self.language_model.embed_tokens.materialize_scale()
        for layer in self.language_model.layers:
            layer.self_attn.pack_projections()

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        multimodal = (token_ids == self.config.image_token_id) | (
            token_ids == self.config.video_token_id
        )
        llm_ids = torch.where(multimodal, self.config.pad_token_id, token_ids)
        return self.language_model.embed_tokens(llm_ids)

    def get_image_features(
        self, pixel_values: Tensor, image_position_ids: Tensor
    ) -> Tensor:
        features = self.model.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            return_dict=True,
        ).last_hidden_state
        return self.model.embed_vision(features)

    def get_video_features(
        self, pixel_values: Tensor, video_position_ids: Tensor
    ) -> Tensor:
        features = self.model.vision_tower(
            pixel_values=pixel_values.flatten(0, 1),
            pixel_position_ids=video_position_ids.flatten(0, 1),
            return_dict=True,
        ).last_hidden_state
        return self.model.embed_vision(features)

    def logits(self, hidden: Tensor) -> Tensor:
        logits = F.linear(hidden, self.language_model.embed_tokens.weight)
        cap = self.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap


__all__ = [
    "Gemma4Attention",
    "Gemma4DecoderLayer",
    "Gemma4ForConditionalGeneration",
    "RotaryEmbedding",
    "apply_rotary",
]
