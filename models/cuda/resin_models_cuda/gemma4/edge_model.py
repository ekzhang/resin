"""BF16 Gemma 4 E2B/E4B text modules, shaped like the Google checkpoints.

The E2B/E4B architecture has per-layer inputs and a KV-sharing consumer stack; it
does not use the 31B NVFP4 model's projection or cache geometry. Attention cache
ownership and attention execution remain in the runner.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin.backend.layers import RMSNorm
from resin.models.gemma4.config import SLIDING_ATTENTION
from resin.models.gemma4.model import RotaryEmbedding, apply_rotary

if TYPE_CHECKING:
    from resin.models.gemma4.edge_config import Gemma4EdgeConfig


class EdgeScaledEmbedding(nn.Embedding):
    def __init__(
        self, count: int, width: int, padding_idx: int, *, scale: float
    ) -> None:
        super().__init__(count, width, padding_idx, dtype=torch.bfloat16)
        self.scale = scale
        self.register_buffer("scale_tensor", None, persistent=False)

    def materialize_scale(self) -> None:
        self.scale_tensor = self.weight.new_tensor(self.scale)

    def forward(self, input: Tensor) -> Tensor:
        if self.scale_tensor is None:
            raise RuntimeError("embedding scale must be prepared before capture")
        return F.embedding(input, self.weight, self.padding_idx) * self.scale_tensor


class Gemma4EdgeAttention(nn.Module):
    def __init__(self, config: Gemma4EdgeConfig, layer_index: int) -> None:
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
        first_consumer = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_index >= first_consumer
        self.cache_source_index = (
            max(
                index
                for index, kind in enumerate(config.layer_types[:first_consumer])
                if kind == self.layer_type
            )
            if self.is_kv_shared_layer
            else layer_index
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        if not self.is_kv_shared_layer:
            self.k_proj = nn.Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.v_proj = nn.Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.v_norm = RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.qkv_proj: nn.Linear | None = None

    def pack_projections(self) -> None:
        if self.qkv_proj is not None:
            raise RuntimeError("attention projections were already packed")
        if self.is_kv_shared_layer:
            return
        weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0
        )
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
        self, hidden: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if hidden.is_cuda:
            from resin_models_cuda.gemma4.edge_kernels import strided_qk_rmsnorm_rope

            if self.is_kv_shared_layer:
                query = self.q_proj(hidden).view(-1, self.num_heads, self.head_dim)
                query, _ = strided_qk_rmsnorm_rope(
                    query,
                    None,
                    self.q_norm.weight,
                    None,
                    cos,
                    sin,
                    eps=self.q_norm.eps,
                )
                return query, None, None
            if self.qkv_proj is None:
                raise RuntimeError(
                    "attention projections must be packed before inference"
                )
            q_width = self.num_heads * self.head_dim
            kv_width = self.num_kv_heads * self.head_dim
            query, key, value = self.qkv_proj(hidden).split(
                (q_width, kv_width, kv_width), dim=-1
            )
            query, key = strided_qk_rmsnorm_rope(
                query.view(-1, self.num_heads, self.head_dim),
                key.view(-1, self.num_kv_heads, self.head_dim),
                self.q_norm.weight,
                self.k_norm.weight,
                cos,
                sin,
                eps=self.q_norm.eps,
            )
            value = self.v_norm(value.view(-1, self.num_kv_heads, self.head_dim))
            return query, key, value
        if self.is_kv_shared_layer:
            query = self.q_proj(hidden)
            query = self.q_norm(query.view(-1, self.num_heads, self.head_dim))
            return apply_rotary(query, cos, sin), None, None
        if self.qkv_proj is None:
            raise RuntimeError("attention projections must be packed before inference")
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        query, key, value = self.qkv_proj(hidden).split(
            (q_width, kv_width, kv_width), dim=-1
        )
        query = self.q_norm(query.view(-1, self.num_heads, self.head_dim))
        key = self.k_norm(key.view(-1, self.num_kv_heads, self.head_dim))
        value = self.v_norm(value.view(-1, self.num_kv_heads, self.head_dim))
        return apply_rotary(query, cos, sin), apply_rotary(key, cos, sin), value


class Gemma4EdgeMLP(nn.Module):
    gelu_zero_bias: Tensor | None

    def __init__(self, config: Gemma4EdgeConfig, layer_index: int) -> None:
        super().__init__()
        first_consumer = config.num_hidden_layers - config.num_kv_shared_layers
        double = config.use_double_wide_mlp and layer_index >= first_consumer > 0
        self.intermediate_size = config.intermediate_size * (2 if double else 1)
        self.is_kv_producer = layer_index < first_consumer
        self.gate_proj = nn.Linear(
            config.hidden_size,
            self.intermediate_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            self.intermediate_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.register_buffer("gelu_zero_bias", None, persistent=False)
        self.down_proj = nn.Linear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        if (
            hidden.is_cuda
            and self.is_kv_producer
            and hidden.numel() // hidden.shape[-1] >= 8192
        ):
            from resin_models_cuda.gemma4.edge_kernels import linear_gelu

            if self.gelu_zero_bias is None:
                raise RuntimeError("GELU bias must be prepared before capture")
            gate = linear_gelu(
                hidden.reshape(-1, hidden.shape[-1]),
                self.gate_proj.weight,
                zero_bias=self.gelu_zero_bias,
            ).reshape(*hidden.shape[:-1], self.intermediate_size)
        else:
            gate = F.gelu(self.gate_proj(hidden), approximate="tanh")
        return self.down_proj(gate * self.up_proj(hidden))


class Gemma4EdgeDecoderLayer(nn.Module):
    layer_scalar: Tensor

    def __init__(self, config: Gemma4EdgeConfig, layer_index: int) -> None:
        super().__init__()
        self.self_attn = Gemma4EdgeAttention(config, layer_index)
        self.mlp = Gemma4EdgeMLP(config, layer_index)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.register_buffer("layer_scalar", torch.ones(1, dtype=torch.bfloat16))
        self.per_layer_input_gate = nn.Linear(
            config.hidden_size,
            config.hidden_size_per_layer_input,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.per_layer_projection = nn.Linear(
            config.hidden_size_per_layer_input,
            config.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.post_per_layer_input_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def feed_forward(self, hidden: Tensor, per_layer_input: Tensor) -> Tensor:
        if hidden.is_cuda:
            from resin_models_cuda.gemma4.edge_kernels import norm_residual

            residual = hidden
            hidden = self.mlp(self.pre_feedforward_layernorm(hidden))
            hidden = norm_residual(
                hidden,
                self.post_feedforward_layernorm.weight,
                residual,
                eps=self.post_feedforward_layernorm.eps,
            )
            residual = hidden
            gate = F.gelu(self.per_layer_input_gate(hidden), approximate="tanh")
            hidden = self.per_layer_projection(gate * per_layer_input)
            return norm_residual(
                hidden,
                self.post_per_layer_input_norm.weight,
                residual,
                scale=self.layer_scalar,
                eps=self.post_per_layer_input_norm.eps,
            )
        residual = hidden
        hidden = self.mlp(self.pre_feedforward_layernorm(hidden))
        hidden = residual + self.post_feedforward_layernorm(hidden)
        residual = hidden
        gate = F.gelu(self.per_layer_input_gate(hidden), approximate="tanh")
        hidden = self.per_layer_projection(gate * per_layer_input)
        hidden = residual + self.post_per_layer_input_norm(hidden)
        return hidden * self.layer_scalar


class Gemma4EdgeTextModel(nn.Module):
    def __init__(self, config: Gemma4EdgeConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = EdgeScaledEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            scale=math.sqrt(config.hidden_size),
        )
        self.layers = cast(
            Sequence[Gemma4EdgeDecoderLayer],
            nn.ModuleList(
                Gemma4EdgeDecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ),
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.embed_tokens_per_layer = EdgeScaledEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            config.pad_token_id,
            scale=math.sqrt(config.hidden_size_per_layer_input),
        )
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.per_layer_projection_norm = RMSNorm(
            config.hidden_size_per_layer_input, config.rms_norm_eps
        )

    def prepare_embeddings(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.embed_tokens(token_ids)
        return hidden, self.prepare_per_layer_inputs(hidden, token_ids)

    def prepare_per_layer_inputs(self, hidden: Tensor, token_ids: Tensor) -> Tensor:
        shape = (
            *token_ids.shape,
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        token_inputs = self.embed_tokens_per_layer(token_ids).reshape(shape)
        projected = (
            self.per_layer_model_projection(hidden) * self.config.hidden_size**-0.5
        ).reshape(shape)
        projected = self.per_layer_projection_norm(projected)
        return (projected + token_inputs) * 2.0**-0.5


class _EdgeModelContainer(nn.Module):
    def __init__(self, config: Gemma4EdgeConfig) -> None:
        super().__init__()
        self.language_model = Gemma4EdgeTextModel(config)


class Gemma4EdgeForCausalLM(nn.Module):
    """Text-only E2B backbone; vision/audio weights are intentionally not loaded."""

    def __init__(self, config: Gemma4EdgeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _EdgeModelContainer(config)

    @property
    def language_model(self) -> Gemma4EdgeTextModel:
        return self.model.language_model

    def materialize_buffers(self, device: torch.device) -> None:
        self.language_model.rotary_emb.materialize(device)
        for module in self.modules():
            if isinstance(module, EdgeScaledEmbedding):
                module.materialize_scale()
            elif isinstance(module, Gemma4EdgeMLP):
                module.gelu_zero_bias = module.gate_proj.weight.new_zeros(
                    module.intermediate_size
                )

    def prepare_attention(self) -> None:
        for layer in self.language_model.layers:
            layer.self_attn.pack_projections()

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        return self.language_model.embed_tokens(token_ids)

    def prepare_embeddings(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.language_model.prepare_embeddings(token_ids)

    def prepare_per_layer_inputs(self, hidden: Tensor, token_ids: Tensor) -> Tensor:
        return self.language_model.prepare_per_layer_inputs(hidden, token_ids)

    def logits(self, hidden: Tensor) -> Tensor:
        logits = F.linear(hidden, self.language_model.embed_tokens.weight)
        cap = self.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap if cap else logits


__all__ = [
    "Gemma4EdgeAttention",
    "Gemma4EdgeDecoderLayer",
    "Gemma4EdgeForCausalLM",
    "Gemma4EdgeMLP",
    "Gemma4EdgeTextModel",
]
