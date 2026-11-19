# Portions are adapted from OpenAI gpt-oss and Hugging Face Transformers
# (Apache-2.0).

"""GPT-OSS model semantics and checkpoint-shaped modules."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from resin.backend.balanced_routing import BalancedRoutingMixin
from resin.backend.layers import RMSNorm
from resin.models.gpt_oss.config import GptOssConfig

if TYPE_CHECKING:
    from resin_models_cuda.gpt_oss.mxfp4 import KernelExperts


class YarnRotaryEmbedding(nn.Module):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.base = config.rope_theta
        self.initial_context = config.initial_context_length
        self.factor = config.rope_scaling_factor
        self.beta_fast = config.rope_beta_fast
        self.beta_slow = config.rope_beta_slow
        self.concentration = (
            0.1 * math.log(self.factor) + 1.0 if self.factor > 1 else 1.0
        )
        self.register_buffer(
            "inv_freq", self._frequencies(torch.device("cpu")), persistent=False
        )

    def _frequencies(self, device) -> Tensor:
        frequencies = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
            / self.head_dim
        )
        if self.factor <= 1:
            return 1.0 / frequencies
        half_dim = self.head_dim / 2
        low = (
            half_dim
            * math.log(self.initial_context / (self.beta_fast * 2 * math.pi))
            / math.log(self.base)
        )
        high = (
            half_dim
            * math.log(self.initial_context / (self.beta_slow * 2 * math.pi))
            / math.log(self.base)
        )
        if not 0 < low < high < half_dim - 1:
            raise ValueError("invalid YaRN parameters for the GPT-OSS head size")
        interpolation = 1.0 / (self.factor * frequencies)
        extrapolation = 1.0 / frequencies
        ramp = (torch.arange(half_dim, device=device) - low) / (high - low)
        mask = 1 - ramp.clamp(0, 1)
        return interpolation * (1 - mask) + extrapolation * mask

    def materialize(self, device) -> None:
        self.inv_freq = self._frequencies(device)

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        angles = torch.einsum("bl,d->bld", positions.float(), self.inv_freq.float())
        return (
            (angles.cos() * self.concentration).to(dtype),
            (angles.sin() * self.concentration).to(dtype),
        )


class SinkAttention(nn.Module):
    def __init__(self, config: GptOssConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        query = config.num_attention_heads * config.head_dim
        kv = config.num_key_value_heads * config.head_dim
        self.qkv_widths = (query, kv, kv)
        self.qkv_proj = nn.Linear(
            config.hidden_size, sum(self.qkv_widths), bias=config.attention_bias
        )
        self.o_proj = nn.Linear(query, config.hidden_size, bias=config.attention_bias)
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))


class TopKRouter(BalancedRoutingMixin, nn.Module):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(
            torch.empty(config.num_local_experts, config.hidden_size)
        )
        self.bias = nn.Parameter(torch.empty(config.num_local_experts))


class MLP(nn.Module):
    def __init__(self, config: GptOssConfig, experts: KernelExperts) -> None:
        super().__init__()
        self.router = TopKRouter(config)
        self.experts = experts


class MoEDecoderLayer(nn.Module):
    def __init__(
        self, config: GptOssConfig, layer_index: int, experts: KernelExperts
    ) -> None:
        super().__init__()
        self.self_attn = SinkAttention(config, layer_index)
        self.mlp = MLP(config, experts)
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, weight_dtype=torch.float32
        )


class MoETransformerModel(nn.Module):
    def __init__(
        self, config: GptOssConfig, make_experts: Callable[[], KernelExperts]
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = cast(
            Sequence[MoEDecoderLayer],
            nn.ModuleList(
                MoEDecoderLayer(config, index, make_experts())
                for index in range(config.num_hidden_layers)
            ),
        )
        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.rotary_emb = YarnRotaryEmbedding(config)


class MoEForCausalLM(nn.Module):
    def __init__(
        self, config: GptOssConfig, make_experts: Callable[[], KernelExperts]
    ) -> None:
        super().__init__()
        self.args = config
        self.model = MoETransformerModel(config, make_experts)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def materialize_buffers(self, device) -> None:
        self.model.rotary_emb.materialize(device)


__all__ = [
    "MLP",
    "MoEDecoderLayer",
    "MoEForCausalLM",
    "MoETransformerModel",
    "SinkAttention",
    "TopKRouter",
    "YarnRotaryEmbedding",
]
