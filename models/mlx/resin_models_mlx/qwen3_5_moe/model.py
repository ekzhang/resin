"""Qwen3.5-MoE text model composed from MLX kernels.

The forward pass is stateless: callers pass the previous recurrent state and
already-cached attention history, and receive the new state plus this step's
keys and values. The runner owns writing them back into the Engine's slabs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
from mlx_lm.models.switch_layers import SwitchGLU

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.paged_attention import paged_decode_attention

# Query-key scores materialized per prefill attention call (512 MiB in BF16).
SCORE_BUDGET = 1 << 28


@dataclass(frozen=True, slots=True)
class AttentionHistory:
    """Every earlier key and value of one full-attention layer.

    ``keys`` and ``values`` have shape ``[rows, history, kv_heads, head_dim]``
    and hold exactly the rows' history, with no padding, so ordinary causal
    masking applies after the step's own tokens are appended.
    """

    keys: mx.array
    values: mx.array


@dataclass(frozen=True, slots=True)
class PagedHistory:
    """Decode history read in place from the arena by every attention layer.

    ``units`` is the arena viewed as ``[units, 2, page, kv_heads * head_dim]``,
    ``pages`` holds each row's int32 base unit per block (add the layer index),
    and ``lengths`` is each row's int32 history length.
    """

    units: mx.array
    pages: mx.array
    lengths: mx.array


@dataclass(frozen=True, slots=True)
class LinearState:
    """DeltaNet state for every linear-attention layer of a batch.

    Each layer keeps its own contiguous arrays so kernels consume them without
    copies: ``recurrent[i]`` is ``[rows, value_heads, value_dim, key_dim]`` FP32
    and ``convolution[i]`` is ``[rows, kernel - 1, conv_dim]`` BF16.
    """

    recurrent: tuple[mx.array, ...]
    convolution: tuple[mx.array, ...]


@dataclass(frozen=True, slots=True)
class StepOutput:
    hidden: mx.array
    keys: tuple[mx.array, ...]
    values: tuple[mx.array, ...]
    state: LinearState


class OffsetRMSNorm(nn.Module):
    """Qwen RMSNorm whose weight is stored as an offset from one.

    The scale ``1 + w`` is applied in FP32, as in the reference implementation;
    folding it into a BF16 weight would round it by up to 2^-8.
    """

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((width,))

    def __call__(self, hidden: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(hidden.astype(mx.float32), None, self.eps)
        return (normed * (1.0 + self.weight.astype(mx.float32))).astype(hidden.dtype)


class GatedRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((width,))

    def __call__(self, hidden: mx.array, gate: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(hidden, self.weight, self.eps)
        gate = nn.silu(gate.astype(mx.float32))
        return (gate * normed.astype(mx.float32)).astype(hidden.dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.key_heads = config.linear_num_key_heads
        self.value_heads = config.linear_num_value_heads
        self.key_dim = config.linear_key_head_dim
        self.value_dim = config.linear_value_head_dim
        self.conv_dim = config.linear_conv_dim
        self.kernel = config.linear_conv_kernel_dim
        hidden = config.hidden_size
        self.in_proj_qkv = nn.Linear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, config.linear_value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.value_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.kernel,
            groups=self.conv_dim,
            bias=False,
        )
        self.dt_bias = mx.ones((self.value_heads,))
        self.A_log = mx.zeros((self.value_heads,))
        self.norm = GatedRMSNorm(self.value_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(config.linear_value_dim, hidden, bias=False)

    def __call__(
        self, hidden: mx.array, recurrent: mx.array, convolution: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        rows, tokens, _ = hidden.shape
        mixed = self.in_proj_qkv(hidden)
        z = self.in_proj_z(hidden).reshape(
            rows, tokens, self.value_heads, self.value_dim
        )
        b = self.in_proj_b(hidden)
        a = self.in_proj_a(hidden)

        window = mx.concatenate([convolution.astype(mixed.dtype), mixed], axis=1)
        convolution = window[:, -(self.kernel - 1) :]
        mixed = nn.silu(self.conv1d(window))
        key_width = self.key_heads * self.key_dim
        q, k, v = mx.split(mixed, [key_width, 2 * key_width], axis=-1)
        q = q.reshape(rows, tokens, self.key_heads, self.key_dim)
        k = k.reshape(rows, tokens, self.key_heads, self.key_dim)
        v = v.reshape(rows, tokens, self.value_heads, self.value_dim)
        q, k = normalize_qk(q, k, inv_scale=self.key_dim**-0.5, eps=1e-6)
        # The decay is computed in float32, as in the reference model.
        output, recurrent = gated_delta_update(
            q, k, v, a.astype(mx.float32), b, self.A_log, self.dt_bias, recurrent
        )
        output = self.norm(output, z).reshape(rows, tokens, -1)
        return self.out_proj(output), recurrent, convolution


class GatedAttention(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.rope_theta = config.rope_theta
        self.scale = self.head_dim**-0.5
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, 2 * self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False)
        self.q_norm = OffsetRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(self.head_dim, config.rms_norm_eps)

    def _rope(self, x: mx.array, offsets: mx.array) -> mx.array:
        # Text-only positions make every interleaved mRoPE axis equal, which
        # reduces to ordinary partial RoPE on the leading rotary dimensions.
        return mx.fast.rope(
            x,
            self.rotary_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offsets,
        )

    def __call__(
        self,
        hidden: mx.array,
        offsets: mx.array,
        history: AttentionHistory | PagedHistory | None,
        layer: int = 0,
    ) -> tuple[mx.array, mx.array, mx.array]:
        rows, tokens, _ = hidden.shape
        query, gate = mx.split(
            self.q_proj(hidden).reshape(rows, tokens, self.heads, -1), 2, axis=-1
        )
        key = self.k_proj(hidden).reshape(rows, tokens, self.kv_heads, -1)
        value = self.v_proj(hidden).reshape(rows, tokens, self.kv_heads, -1)
        query = self._rope(self.q_norm(query).transpose(0, 2, 1, 3), offsets)
        key = self._rope(self.k_norm(key).transpose(0, 2, 1, 3), offsets)
        value = value.transpose(0, 2, 1, 3)
        new_key, new_value = key, value

        if isinstance(history, PagedHistory):
            if tokens != 1:
                raise ValueError("paged history attends one decode token per row")
            output = paged_decode_attention(
                query[:, :, 0],
                key[:, :, 0],
                value[:, :, 0],
                history.units,
                history.pages,
                history.lengths,
                layer,
                self.scale,
            )[:, :, None]
            return self._project(output, gate, rows, tokens, new_key, new_value)
        if history is not None:
            key = mx.concatenate([history.keys.transpose(0, 2, 1, 3), key], axis=2)
            value = mx.concatenate(
                [history.values.transpose(0, 2, 1, 3), value], axis=2
            )
        output = self._dense_attention(query, key, value)
        return self._project(output, gate, rows, tokens, new_key, new_value)

    def _dense_attention(self, query: mx.array, key: mx.array, value: mx.array):
        """Causal attention for queries that end the key sequence.

        MLX's fused kernel streams scores only for head dimensions up to 128;
        wider heads materialize every query-key score. Splitting the queries
        bounds that buffer. Each block attends to the keys up to its own last
        position, and MLX aligns a causal mask to the last query. Blocks are
        evaluated one at a time: lazily, MLX would allocate them together. On
        an M4 Max at 98K keys this holds 0.55 GiB instead of 10 GiB, at the
        same speed.
        """

        tokens, length = query.shape[2], key.shape[2]
        if tokens == 1:
            return mx.fast.scaled_dot_product_attention(
                query, key, value, scale=self.scale
            )
        block = max(1, SCORE_BUDGET // (self.heads * length))
        if block >= tokens:
            return mx.fast.scaled_dot_product_attention(
                query, key, value, scale=self.scale, mask="causal"
            )
        outputs = []
        for start in range(0, tokens, block):
            end = min(start + block, tokens)
            visible = length - tokens + end
            output = mx.fast.scaled_dot_product_attention(
                query[:, :, start:end],
                key[:, :, :visible],
                value[:, :, :visible],
                scale=self.scale,
                mask="causal",
            )
            mx.eval(output)
            outputs.append(output)
        return mx.concatenate(outputs, axis=2)

    def _project(self, output, gate, rows, tokens, key, value):
        output = output.transpose(0, 2, 1, 3).reshape(rows, tokens, -1)
        output = output * mx.sigmoid(gate.reshape(rows, tokens, -1))
        return (
            self.o_proj(output),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
        )


class FeedForward(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class SparseMoe(nn.Module):
    """Softmax top-k routed experts plus a sigmoid-gated shared expert."""

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(hidden, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            hidden, config.moe_intermediate_size, config.num_experts
        )
        self.shared_expert = FeedForward(hidden, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        probabilities = mx.softmax(self.gate(hidden), axis=-1, precise=True)
        experts = mx.argpartition(probabilities, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        scores = mx.take_along_axis(probabilities, experts, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        routed = (self.switch_mlp(hidden, experts) * scores[..., None]).sum(axis=-2)
        shared = mx.sigmoid(self.shared_expert_gate(hidden)) * self.shared_expert(
            hidden
        )
        return routed + shared


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35Config, kind: str) -> None:
        super().__init__()
        self.is_linear = kind == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = GatedAttention(config)
        self.mlp = SparseMoe(config)
        self.input_layernorm = OffsetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )


class Qwen35MoeModel(nn.Module):
    """Text decoder over Qwen3.5-MoE checkpoint weights."""

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        if not config.is_moe:
            raise ValueError("Qwen35MoeModel requires a Qwen3.5-MoE configuration")
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, kind) for kind in config.layer_types]
        self.norm = OffsetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        token_ids: mx.array,
        offsets: mx.array,
        state: LinearState,
        history: Sequence[AttentionHistory | None] | PagedHistory,
    ) -> StepOutput:
        """Run ``[rows, tokens]`` inputs starting at per-row position ``offsets``."""

        hidden = self.embed_tokens(token_ids)
        keys, values, recurrent, convolution = [], [], [], []
        linear_index = full_index = 0
        for layer in self.layers:
            normed = layer.input_layernorm(hidden)
            if layer.is_linear:
                mixed, layer_recurrent, layer_convolution = layer.linear_attn(
                    normed,
                    state.recurrent[linear_index],
                    state.convolution[linear_index],
                )
                recurrent.append(layer_recurrent)
                convolution.append(layer_convolution)
                linear_index += 1
            else:
                layer_history = (
                    history
                    if isinstance(history, PagedHistory)
                    else history[full_index]
                )
                mixed, key, value = layer.self_attn(
                    normed, offsets, layer_history, full_index
                )
                keys.append(key)
                values.append(value)
                full_index += 1
            hidden = hidden + mixed
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return StepOutput(
            self.norm(hidden),
            tuple(keys),
            tuple(values),
            LinearState(tuple(recurrent), tuple(convolution)),
        )

    def logits(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden)

    def initial_state(self, rows: int) -> LinearState:
        config = self.config
        layers = config.linear_layer_count
        recurrent = (
            rows,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            config.linear_key_head_dim,
        )
        convolution = (
            rows,
            config.linear_conv_kernel_dim - 1,
            config.linear_conv_dim,
        )
        return LinearState(
            tuple(mx.zeros(recurrent, dtype=mx.float32) for _ in range(layers)),
            tuple(mx.zeros(convolution, dtype=mx.bfloat16) for _ in range(layers)),
        )


__all__ = [
    "AttentionHistory",
    "LinearState",
    "PagedHistory",
    "Qwen35MoeModel",
    "StepOutput",
]
