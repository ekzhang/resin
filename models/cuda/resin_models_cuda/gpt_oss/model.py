# Portions are adapted from OpenAI gpt-oss and Hugging Face Transformers
# (Apache-2.0).

"""GPT-OSS model semantics and checkpoint-shaped modules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin.backend.layers import RMSNorm
from resin.backend.torch import benchmark_expert_indices
from resin.models.gpt_oss.config import GptOssConfig


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


def apply_rotary(hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class SinkAttention(nn.Module):
    qkv_proj: nn.Linear

    def __init__(self, config: GptOssConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )
        width = config.hidden_size
        self.q_proj = nn.Linear(
            width, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            width, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            width, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, width, bias=config.attention_bias
        )
        self.sinks = nn.Parameter(torch.empty(self.num_heads))

    def pack_qkv_projection(self) -> None:
        """Replace the loaded local Q/K/V shards with one packed projection."""

        projections = (self.q_proj, self.k_proj, self.v_proj)
        weight = torch.cat(tuple(projection.weight for projection in projections))
        packed = nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=self.q_proj.bias is not None,
            device="meta",
            dtype=weight.dtype,
        )
        packed.weight = nn.Parameter(
            weight,
            requires_grad=self.q_proj.weight.requires_grad,
        )
        if self.q_proj.bias is not None:
            packed.bias = nn.Parameter(
                torch.cat(
                    tuple(
                        projection.bias
                        for projection in projections
                        if projection.bias is not None
                    )
                ),
                requires_grad=self.q_proj.bias.requires_grad,
            )
        self.qkv_proj = packed
        del self.q_proj
        del self.k_proj
        del self.v_proj


class TopKRouter(nn.Module):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.benchmark_balanced_experts = False
        self.benchmark_replica = 0
        self.benchmark_expert_parallel_size = 1
        self.weight = nn.Parameter(
            torch.empty(config.num_local_experts, config.hidden_size)
        )
        self.bias = nn.Parameter(torch.empty(config.num_local_experts))

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = F.linear(hidden, self.weight, self.bias)
        if self.benchmark_balanced_experts:
            indices = benchmark_expert_indices(
                logits,
                self.top_k,
                self.benchmark_replica,
                self.benchmark_expert_parallel_size,
            )
            values = logits.gather(1, indices)
        else:
            values, indices = torch.topk(logits, self.top_k, dim=-1, sorted=True)
        return logits, torch.softmax(values, dim=-1), indices


class DenseExperts(nn.Module):
    """Readable dense experts used by tests and unquantized checkpoints."""

    storage_format = "dense"

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        experts, hidden, intermediate = (
            config.num_local_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        self.intermediate_size = intermediate
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Parameter(torch.empty(experts, hidden, 2 * intermediate))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(experts, 2 * intermediate))
        self.down_proj = nn.Parameter(torch.empty(experts, intermediate, hidden))
        self.down_proj_bias = nn.Parameter(torch.empty(experts, hidden))

    def forward(
        self,
        hidden: Tensor,
        expert_indices: Tensor,
        routing_weights: Tensor,
    ) -> Tensor:
        gate_up = torch.einsum(
            "tkhi,th->tki", self.gate_up_proj[expert_indices], hidden
        )
        gate_up += self.gate_up_proj_bias[expert_indices]
        gate = gate_up[..., ::2].clamp(max=self.swiglu_limit)
        up = gate_up[..., 1::2].clamp(-self.swiglu_limit, self.swiglu_limit)
        activated = gate * torch.sigmoid(1.702 * gate) * (up + 1)
        output = torch.einsum(
            "tkih,tki->tkh", self.down_proj[expert_indices], activated
        )
        output += self.down_proj_bias[expert_indices]
        return torch.einsum("tkh,tk->th", output, routing_weights)


class MLP(nn.Module):
    experts: nn.Module

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.router = TopKRouter(config)
        self.experts = DenseExperts(config)

    def forward(self, hidden: Tensor) -> Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        _, weights, indices = self.router(flat)
        return self.experts(flat, indices, weights).reshape(shape)


class MoEDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, layer_index: int) -> None:
        super().__init__()
        self.self_attn = SinkAttention(config, layer_index)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, weight_dtype=torch.float32
        )


class MoETransformerModel(nn.Module):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = cast(
            Sequence[MoEDecoderLayer],
            nn.ModuleList(
                MoEDecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ),
        )
        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.rotary_emb = YarnRotaryEmbedding(config)


class MoEForCausalLM(nn.Module):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.config = config
        self.args = config
        self.model = MoETransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
        elif isinstance(module, SinkAttention):
            nn.init.normal_(module.sinks, std=std)
        elif isinstance(module, TopKRouter):
            nn.init.normal_(module.weight, std=std)
            nn.init.normal_(module.bias, std=std)
        elif isinstance(module, DenseExperts):
            nn.init.normal_(module.gate_up_proj, std=std)
            nn.init.zeros_(module.gate_up_proj_bias)
            nn.init.normal_(module.down_proj, std=std)
            nn.init.zeros_(module.down_proj_bias)

    def materialize_buffers(self, device) -> None:
        self.model.rotary_emb.materialize(device)


__all__ = [
    "DenseExperts",
    "MLP",
    "MoEDecoderLayer",
    "MoEForCausalLM",
    "MoETransformerModel",
    "SinkAttention",
    "TopKRouter",
    "YarnRotaryEmbedding",
    "apply_rotary",
]
