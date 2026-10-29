"""Native dense Qwen3.5 hybrid model and its optional bundled MTP head."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_cuda.nvfp4 import Nvfp4LinearWeights
from resin_models_cuda.qwen3_5.attention import (
    PagedAttentionRuntime,
    PreparedPagedAttention,
    append_kv,
)
from resin_models_cuda.qwen3_5.cache import (
    MTP_GROUP,
    TARGET_GROUP,
    DeviceCache,
    RecurrentStatePool,
)
from resin_models_cuda.qwen3_5.quantization import ModelOptFp8Linear

if TYPE_CHECKING:
    from transformers import Qwen3_5VisionModel


def _projection(in_features: int, out_features: int, *, quantized: bool) -> nn.Module:
    if quantized:
        return ModelOptFp8Linear(in_features, out_features)
    return nn.Linear(in_features, out_features, bias=False)


def _feed_forward(in_features: int, out_features: int, *, quantized: bool) -> nn.Module:
    if quantized:
        return Nvfp4LinearWeights(in_features, out_features)
    return nn.Linear(in_features, out_features, bias=False)


def _fla_kernels():
    try:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except ImportError:
        return None
    return chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


class Qwen35RMSNorm(nn.Module):
    """Qwen's one-centered RMSNorm checkpoint convention."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.is_cuda:
            from flashinfer.norm import gemma_rmsnorm

            # This kernel uses Qwen's FP32 (1 + checkpoint_weight) convention.
            return gemma_rmsnorm(
                hidden.reshape(-1, hidden.shape[-1]), self.weight, eps=self.eps
            ).view_as(hidden)
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * (1.0 + self.weight.float())).to(hidden.dtype)


class Qwen35GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden: Tensor, gate: Tensor) -> Tensor:
        if hidden.is_cuda:
            from fla.modules.fused_norm_gate import rms_norm_gated

            # FLA supports bias=None despite its public Tensor annotation.
            return rms_norm_gated(hidden, gate, self.weight, None, eps=self.eps)  # ty: ignore[invalid-argument-type]
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float() * F.silu(gate.float())).to(
            hidden.dtype
        )


def _l2_normalize(hidden: Tensor) -> Tensor:
    return hidden * torch.rsqrt(hidden.square().sum(-1, keepdim=True) + 1e-6)


def _reference_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
    beta: Tensor,
    initial_state: Tensor,
) -> tuple[Tensor, Tensor]:
    """Small, exact fallback used by CPU tests and unsupported CUDA hosts."""

    output_dtype = value.dtype
    query = _l2_normalize(query.float()) * (query.shape[-1] ** -0.5)
    key = _l2_normalize(key.float())
    value = value.float()
    state = initial_state.float()
    outputs = []
    for step in range(query.shape[1]):
        state = state * decay[:, step].exp()[..., None, None]
        key_step = key[:, step]
        memory = (state * key_step.unsqueeze(-1)).sum(-2)
        delta = (value[:, step] - memory) * beta[:, step].unsqueeze(-1)
        state = state + key_step.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * query[:, step].unsqueeze(-1)).sum(-2))
    return torch.stack(outputs, dim=1).to(output_dtype), state


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35Config, *, quantized: bool = False) -> None:
        super().__init__()
        self.config = config
        self.key_dim = config.linear_key_dim
        self.value_dim = config.linear_value_dim
        self.conv_dim = config.linear_conv_dim
        self.in_proj_qkv = _projection(
            config.hidden_size, self.conv_dim, quantized=quantized
        )
        self.in_proj_z = _projection(
            config.hidden_size, self.value_dim, quantized=quantized
        )
        self.in_proj_b = nn.Linear(
            config.hidden_size, config.linear_num_value_heads, bias=False
        )
        self.in_proj_a = nn.Linear(
            config.hidden_size, config.linear_num_value_heads, bias=False
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            config.linear_conv_kernel_dim,
            groups=self.conv_dim,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(config.linear_num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(config.linear_num_value_heads))
        self.norm = Qwen35GatedRMSNorm(
            config.linear_value_head_dim, config.rms_norm_eps
        )
        self.out_proj = _projection(
            self.value_dim, config.hidden_size, quantized=quantized
        )

    def _pooled_recurrence(
        self, hidden: Tensor, mixed: Tensor, pool: RecurrentStatePool, layer: int
    ) -> Tensor:
        from resin_models_cuda.qwen3_5.gdn_decode import gdn_decode

        batch, tokens = mixed.shape[:2]
        return gdn_decode(
            mixed.reshape(batch, tokens, self.conv_dim).contiguous(),
            self.in_proj_a(hidden).reshape(batch, tokens, -1),
            self.in_proj_b(hidden).reshape(batch, tokens, -1),
            self.A_log,
            self.dt_bias,
            pool,
            layer,
            num_key_heads=self.config.linear_num_key_heads,
            num_value_heads=self.config.linear_num_value_heads,
            key_dim=self.config.linear_key_head_dim,
            value_dim=self.config.linear_value_head_dim,
        ).reshape(-1, self.config.linear_value_head_dim)

    def forward(
        self,
        hidden: Tensor,
        recurrent_state: Tensor | None,
        convolution_state: Tensor | None,
        state_pool: tuple[RecurrentStatePool, int] | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """With ``state_pool``, recurrent (and for one token, convolution) state
        is read from and written to the cache arena in place."""
        batch, tokens = hidden.shape[:2]
        if (
            state_pool is not None
            and tokens == 1
            and state_pool[0].convolution_states is not None
        ):
            from resin_models_cuda.qwen3_5.gdn_decode import conv_decode

            pool, linear_layer = state_pool
            mixed = conv_decode(
                self.in_proj_qkv(hidden).reshape(batch, self.conv_dim),
                self.conv1d.weight,
                pool,
                linear_layer,
            )
            core = self._pooled_recurrence(hidden, mixed[:, None], pool, linear_layer)
            gate = self.in_proj_z(hidden).reshape_as(core)
            core = self.norm(core, gate).view(batch, tokens, self.value_dim)
            return self.out_proj(core), None, None
        mixed = self.in_proj_qkv(hidden).transpose(1, 2)
        state_length = self.config.linear_conv_kernel_dim - 1
        if convolution_state is None:
            convolution_state = torch.zeros(
                batch,
                self.conv_dim,
                state_length,
                dtype=mixed.dtype,
                device=mixed.device,
            )
        convolution_input = torch.cat((convolution_state, mixed), dim=-1)
        mixed = F.silu(self.conv1d(convolution_input)).transpose(1, 2)
        convolution_state = convolution_input[..., -state_length:]
        if state_pool is not None and hidden.is_cuda:
            # Multi-token verification: the commit selects the accepted window.
            pool, linear_layer = state_pool
            core = self._pooled_recurrence(hidden, mixed, pool, linear_layer)
            gate = self.in_proj_z(hidden).reshape_as(core)
            core = self.norm(core, gate).view(batch, tokens, self.value_dim)
            return self.out_proj(core), None, convolution_input

        query, key, value = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.view(
            batch,
            tokens,
            self.config.linear_num_key_heads,
            self.config.linear_key_head_dim,
        )
        key = key.view_as(query)
        value = value.view(
            batch,
            tokens,
            self.config.linear_num_value_heads,
            self.config.linear_value_head_dim,
        )
        beta = self.in_proj_b(hidden).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden).float() + self.dt_bias
        )
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                batch,
                self.config.linear_num_value_heads,
                self.config.linear_key_head_dim,
                self.config.linear_value_head_dim,
                dtype=torch.float32,
                device=hidden.device,
            )

        kernels = _fla_kernels() if hidden.is_cuda else None
        if kernels is None:
            repeats = (
                self.config.linear_num_value_heads // self.config.linear_num_key_heads
            )
            core, recurrent_state = _reference_delta_rule(
                query.repeat_interleave(repeats, dim=2),
                key.repeat_interleave(repeats, dim=2),
                value,
                decay,
                beta,
                recurrent_state,
            )
        else:
            # FLA handles grouped value heads without materializing repeated Q/K.
            chunk, recurrent = kernels
            kernel = recurrent if tokens == 1 else chunk
            core, recurrent_state = kernel(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        core = core.reshape(-1, self.config.linear_value_head_dim)
        gate = self.in_proj_z(hidden).reshape_as(core)
        core = self.norm(core, gate).view(batch, tokens, self.value_dim)
        return self.out_proj(core), recurrent_state.float(), convolution_state


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor):
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query_rotary = query_rotary * cos + _rotate_half(query_rotary) * sin
    key_rotary = key_rotary * cos + _rotate_half(key_rotary) * sin
    return (
        torch.cat((query_rotary, query_pass), dim=-1),
        torch.cat((key_rotary, key_pass), dim=-1),
    )


class Qwen35Attention(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        *,
        cache_group: int,
        cache_layer: int,
        cache_layer_count: int,
        quantized: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.cache_group = cache_group
        self.cache_layer = cache_layer
        self.cache_layer_count = cache_layer_count
        self.scaling = config.head_dim**-0.5
        self.q_proj = _projection(
            config.hidden_size,
            config.num_attention_heads * config.head_dim * 2,
            quantized=quantized,
        )
        self.k_proj = _projection(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            quantized=quantized,
        )
        self.v_proj = _projection(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            quantized=quantized,
        )
        self.o_proj = _projection(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            quantized=quantized,
        )
        self.q_norm = Qwen35RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: DeviceCache | None,
        attention: PreparedPagedAttention | None,
        sequence_lengths: Tensor,
        max_sequence_length: int,
    ) -> Tensor:
        batch, tokens = hidden.shape[:2]
        query, gate = (
            self.q_proj(hidden)
            .view(
                batch, tokens, self.config.num_attention_heads, 2 * self.config.head_dim
            )
            .chunk(2, dim=-1)
        )
        key = self.k_proj(hidden).view(
            batch, tokens, self.config.num_key_value_heads, self.config.head_dim
        )
        value = self.v_proj(hidden).view_as(key)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_rotary(query, key, cos, sin)

        if cache is not None:
            append_kv(
                cache,
                group_index=self.cache_group,
                layer=self.cache_layer,
                layer_count=self.cache_layer_count,
                positions=positions + int(self.cache_group == MTP_GROUP),
                keys=key,
                values=value,
            )
            if attention is None:
                raise RuntimeError("cached Qwen3.5 attention was not prepared")
            output = attention.run(query, self.cache_layer)
        else:
            repeats = self.config.num_attention_heads // self.config.num_key_value_heads
            key = key.repeat_interleave(repeats, dim=2).transpose(1, 2)
            value = value.repeat_interleave(repeats, dim=2).transpose(1, 2)
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key,
                value,
                is_causal=True,
                scale=self.scaling,
            ).transpose(1, 2)
        output = output.reshape(batch, tokens, -1) * torch.sigmoid(
            gate.reshape(batch, tokens, -1)
        )
        return self.o_proj(output)


class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen35Config, *, quantized: bool = False) -> None:
        super().__init__()
        self.gate_proj = _feed_forward(
            config.hidden_size, config.intermediate_size, quantized=quantized
        )
        self.up_proj = _feed_forward(
            config.hidden_size, config.intermediate_size, quantized=quantized
        )
        self.down_proj = _feed_forward(
            config.intermediate_size, config.hidden_size, quantized=quantized
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Qwen35DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        layer_index: int,
        *,
        full_layer_index: int | None = None,
        cache_group: int = TARGET_GROUP,
        cache_layer_count: int | None = None,
        quantized: bool = False,
    ) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_index]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, quantized=quantized)
        else:
            if full_layer_index is None:
                raise ValueError("full Qwen3.5 layer requires its cache-layer index")
            self.self_attn = Qwen35Attention(
                config,
                cache_group=cache_group,
                cache_layer=full_layer_index,
                cache_layer_count=cache_layer_count or config.full_layer_count,
                quantized=quantized,
            )
        self.mlp = Qwen35MLP(config, quantized=quantized)
        self.prefill_ffn: Callable | None = None
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def feed_forward(self, hidden: Tensor, _ids: Tensor | None = None) -> Tensor:
        return self.mlp(self.post_attention_layernorm(hidden))


class Qwen35TextModel(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Qwen35Config, *, quantized: bool = False) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        full_indices = {
            layer: index for index, layer in enumerate(config.full_layer_ids)
        }
        self.layers = nn.ModuleList(
            Qwen35DecoderLayer(
                config,
                layer,
                full_layer_index=full_indices.get(layer),
                quantized=quantized,
            )
            for layer in range(config.num_hidden_layers)
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, config.rms_norm_eps)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.rotary_dim, 2, dtype=torch.float32)
                / config.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotary(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if positions.ndim == 2:
            frequencies = positions.float().unsqueeze(-1) * self.inv_freq
        elif positions.ndim == 3 and positions.shape[0] == 3:
            section = self.config.mrope_section
            if section is None:
                raise ValueError("3-D Qwen positions require mrope_section")
            axes = positions.float().unsqueeze(-1) * self.inv_freq
            frequencies = axes[0].clone()
            for dimension, offset in enumerate((1, 2), start=1):
                indices = slice(offset, section[dimension] * 3, 3)
                frequencies[..., indices] = axes[dimension, ..., indices]
        else:
            raise ValueError("Qwen positions must have shape [B, T] or [3, B, T]")
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)

    def materialize_buffers(self, device: torch.device) -> None:
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(
                    0,
                    self.config.rotary_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / self.config.rotary_dim
            )
        )
        self.inv_freq = inv_freq

    def forward(
        self,
        token_ids: Tensor,
        positions: Tensor,
        *,
        cache: DeviceCache | None = None,
        recurrent_states: Tensor | None = None,
        convolution_states: Tensor | None = None,
        sequence_lengths: Tensor | None = None,
        max_sequence_length: int | None = None,
        input_embeddings: Tensor | None = None,
        attention_runtime: PagedAttentionRuntime | None = None,
        rotary_positions: Tensor | None = None,
        recurrent_pool: RecurrentStatePool | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        hidden = (
            self.embed_tokens(token_ids)
            if input_embeddings is None
            else input_embeddings
        )
        if sequence_lengths is None:
            sequence_lengths = positions[:, -1] + 1
        if max_sequence_length is None:
            max_sequence_length = int(sequence_lengths.max().item())
        cos, sin = self.rotary(
            positions if rotary_positions is None else rotary_positions,
            hidden.dtype,
        )
        attention = (
            attention_runtime.prepare(
                cache,
                group_index=TARGET_GROUP,
                layer_count=self.config.full_layer_count,
                sequence_lengths=sequence_lengths,
                tokens=positions.shape[1],
                softmax_scale=self.config.head_dim**-0.5,
            )
            if cache is not None and attention_runtime is not None
            else None
        )
        recurrent_out = []
        convolution_out = []
        linear_index = 0
        for layer in cast(Sequence[Qwen35DecoderLayer], self.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            if layer.block_type == "linear_attention":
                recurrent = (
                    None
                    if recurrent_states is None
                    else recurrent_states[:, linear_index]
                )
                convolution = (
                    None
                    if convolution_states is None
                    else convolution_states[:, linear_index]
                )
                mixed, recurrent, convolution = layer.linear_attn(
                    normalized,
                    recurrent,
                    convolution,
                    None if recurrent_pool is None else (recurrent_pool, linear_index),
                )
                recurrent_out.append(recurrent)
                convolution_out.append(convolution)
                linear_index += 1
            else:
                mixed = layer.self_attn(
                    normalized,
                    positions,
                    cos,
                    sin,
                    cache,
                    attention,
                    sequence_lengths,
                    max_sequence_length,
                )
            hidden = residual + mixed
            hidden = hidden + (layer.prefill_ffn or layer.feed_forward)(hidden)
        return (
            self.norm(hidden),
            None if recurrent_pool is not None else torch.stack(recurrent_out, dim=1),
            None if convolution_out[0] is None else torch.stack(convolution_out, dim=1),
        )


class _Qwen35Backbone(nn.Module):
    visual: Qwen3_5VisionModel | None

    def __init__(
        self,
        config: Qwen35Config,
        *,
        vision_config: dict | None = None,
        quantized: bool = False,
    ) -> None:
        super().__init__()
        self.language_model = Qwen35TextModel(config, quantized=quantized)
        if vision_config is None:
            self.visual = None
        else:
            from transformers import Qwen3_5VisionConfig, Qwen3_5VisionModel

            self.visual = Qwen3_5VisionModel(
                Qwen3_5VisionConfig.from_dict(vision_config)
            )


class Qwen35MTP(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        config.require_mtp()
        self.config = config
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_embedding = Qwen35RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen35RMSNorm(config.hidden_size, config.rms_norm_eps)
        mtp_config = replace(
            config,
            num_hidden_layers=1,
            layer_types=("full_attention",),
            mtp_num_hidden_layers=0,
        )
        self.layers = nn.ModuleList(
            [
                Qwen35DecoderLayer(
                    mtp_config,
                    0,
                    full_layer_index=0,
                    cache_group=MTP_GROUP,
                    cache_layer_count=1,
                )
            ]
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_embeddings: Tensor,
        previous_hidden: Tensor,
        positions: Tensor,
        *,
        rotary: tuple[Tensor, Tensor],
        cache: DeviceCache | None,
        sequence_lengths: Tensor,
        max_sequence_length: int,
        attention_runtime: PagedAttentionRuntime | None = None,
    ) -> Tensor:
        hidden = self.fc(
            torch.cat(
                (
                    self.pre_fc_norm_embedding(input_embeddings),
                    self.pre_fc_norm_hidden(previous_hidden),
                ),
                dim=-1,
            )
        )
        layer = cast(Qwen35DecoderLayer, self.layers[0])
        attention = (
            attention_runtime.prepare(
                cache,
                group_index=MTP_GROUP,
                layer_count=1,
                sequence_lengths=sequence_lengths + 1,
                tokens=positions.shape[1],
                softmax_scale=self.config.head_dim**-0.5,
            )
            if cache is not None and attention_runtime is not None
            else None
        )
        residual = hidden
        hidden = residual + layer.self_attn(
            layer.input_layernorm(hidden),
            positions,
            *rotary,
            cache,
            attention,
            sequence_lengths,
            max_sequence_length,
        )
        hidden = hidden + (layer.prefill_ffn or layer.feed_forward)(hidden)
        return self.norm(hidden)


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Text-only target whose parameter names match the public checkpoint."""

    def __init__(
        self,
        config: Qwen35Config,
        *,
        include_mtp: bool,
        vision_config: dict | None = None,
        quantized: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = _Qwen35Backbone(
            config, vision_config=vision_config, quantized=quantized
        )
        self.lm_head = _feed_forward(
            config.hidden_size, config.vocab_size, quantized=quantized
        )
        self.mtp = Qwen35MTP(config) if include_mtp else None

    @property
    def language_model(self) -> Qwen35TextModel:
        return self.model.language_model

    @property
    def visual(self) -> Qwen3_5VisionModel | None:
        return self.model.visual

    def logits(self, hidden: Tensor) -> Tensor:
        return self.lm_head(hidden)


__all__ = [
    "Qwen35Attention",
    "Qwen3_5ForConditionalGeneration",
    "Qwen35GatedDeltaNet",
    "Qwen35MTP",
    "Qwen35RMSNorm",
    "Qwen35TextModel",
]
