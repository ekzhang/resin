"""Qwen3.5 text model composed from MLX kernels.

The forward pass is stateless: callers pass the previous recurrent state and
already-cached attention history, and receive the new state plus this step's
keys and values. The runner owns writing them back into the Engine's slabs.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from typing import NamedTuple, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3_5 import GatedDeltaNet as _GatedDeltaNet
from mlx_lm.models.qwen3_5 import TextModelArgs as _TextArgs
from mlx_lm.models.qwen3_next import ModelArgs as _MoeArgs
from mlx_lm.models.qwen3_next import Qwen3NextMLP as _FeedForward
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock as _SparseMoe

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.qwen3_5.fused_conv import conv4_silu_qkv
from resin_models_mlx.qwen3_5.fused_norm import mixed_rms_norm, residual_rms_norm
from resin_models_mlx.qwen3_5.gated_delta import gated_delta_update
from resin_models_mlx.qwen3_5.normalization import normalize_qk

# Query-key scores materialized per prefill attention call (512 MiB in BF16).
SCORE_BUDGET = 1 << 28


# Elementwise tails are compiled into single kernels, as in mlx-lm. Decode is
# launch-bound, and each unfused cast or multiply is a separate launch.
@partial(mx.compile, shapeless=True)
def _sigmoid_gate(value: mx.array, gate: mx.array) -> mx.array:
    return value * mx.sigmoid(gate)


@dataclass(frozen=True, slots=True)
class _JoinedLinear:
    linear: nn.Module
    projections: tuple[nn.Module, ...]
    splits: tuple[int, ...]

    def __call__(self, hidden: mx.array):
        # Joining larger GEMMs can change their reduction; only small GEMVs
        # use the shared packed rows. Prefill retains the original projections.
        if hidden.size // hidden.shape[-1] > 4:
            return [module(hidden) for module in self.projections]
        return mx.split(self.linear(hidden), self.splits, axis=-1)


def _join_linears(modules: tuple[nn.Module, ...]) -> _JoinedLinear | None:
    first = modules[0]
    if type(first) not in (nn.Linear, nn.QuantizedLinear):
        return None
    quantized = isinstance(first, nn.QuantizedLinear)

    def signature(module):
        affine = (
            (
                module.group_size,
                module.bits,
                module.mode,
                module.scales.dtype,
                None if module.biases is None else module.biases.dtype,
            )
            if quantized
            else ()
        )
        return module.weight.dtype, module.weight.shape[1:], affine

    if any(type(module) is not type(first) or "bias" in module for module in modules):
        return None
    if any(signature(module) != signature(first) for module in modules):
        return None
    widths = tuple(accumulate(module.weight.shape[0] for module in modules))
    fields = ("weight", "scales", "biases") if quantized else ("weight",)
    linear = copy(first)
    for name in fields:
        if getattr(first, name) is not None:
            setattr(linear, name, mx.concatenate([getattr(m, name) for m in modules]))
    mx.eval(linear.parameters())
    # Keep checkpoint paths as row views of the joined allocation. Finish the
    # views too: lazy main-thread graphs cannot escape into the HTTP actor.
    for module, start, end in zip(modules, (0, *widths[:-1]), widths, strict=True):
        for name in fields:
            if (value := getattr(linear, name)) is not None:
                setattr(module, name, value[start:end])
    mx.eval([module.parameters() for module in modules])
    return _JoinedLinear(linear, modules, widths[:-1])


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
    ``pages`` holds each row's int32 layer-zero unit per block. Add the
    logical layer index times ``layer_stride`` to address another layer.
    ``lengths`` is each row's int32 history length. Metadata may contain
    unused padded rows; queries determine the logical batch size. The
    default stride preserves consecutive-layer external page tables.
    """

    units: mx.array
    pages: mx.array
    lengths: mx.array
    layer_stride: int = 1


class LinearState(NamedTuple):
    """DeltaNet state for every linear-attention layer of a batch.

    Each layer keeps its own contiguous arrays so kernels consume them without
    copies: ``recurrent[i]`` is ``[rows, value_heads, value_dim, key_dim]`` FP32
    and ``convolution[i]`` is ``[rows, kernel - 1, conv_dim]`` BF16.
    """

    recurrent: tuple[mx.array, ...]
    convolution: tuple[mx.array, ...]


@dataclass(frozen=True, slots=True)
class StepOutput:
    """Model outputs and the updated recurrent state."""

    hidden: mx.array
    keys: tuple[mx.array, ...]
    values: tuple[mx.array, ...]
    state: LinearState
    input_embeddings: mx.array | None = None
    rotary_positions: mx.array | None = None


class OffsetRMSNorm(nn.Module):
    """Qwen RMSNorm accepting HF offsets or MLX's already-folded scales.

    The scale ``1 + w`` is applied in FP32, as in the reference implementation;
    folding it into a BF16 weight would round it by up to 2^-8. MLX's fused
    kernel normalizes in FP32 and, given an FP32 scale, matches the reference
    bit for bit. The scale is computed once per loaded weight.
    """

    def __init__(self, width: int, eps: float, *, offset: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.offset = offset
        self.weight = mx.zeros((width,))
        self._scale: tuple[mx.array, mx.array] | None = None

    def _effective_scale(self) -> mx.array:
        if self._scale is None or self._scale[0] is not self.weight:
            scale = self.weight.astype(mx.float32)
            if self.offset:
                scale = 1.0 + scale
            mx.eval(scale)
            self._scale = (self.weight, scale)
        return self._scale[1]

    def __call__(self, hidden: mx.array) -> mx.array:
        return mixed_rms_norm(hidden, self._effective_scale(), self.eps)

    def add_and_normalize(
        self, hidden: mx.array, residual: mx.array
    ) -> tuple[mx.array, mx.array]:
        return residual_rms_norm(hidden, residual, self._effective_scale(), self.eps)


class GatedDeltaNet(_GatedDeltaNet):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__(cast(_TextArgs, config))
        self.A_log = mx.zeros((self.num_v_heads,))

    def __call__(
        self, hidden: mx.array, recurrent: mx.array, convolution: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        rows, tokens, _ = hidden.shape
        if (joined := getattr(self, "_joined_input", None)) is None:
            mixed, z = self.in_proj_qkv(hidden), self.in_proj_z(hidden)
            b, a = self.in_proj_b(hidden), self.in_proj_a(hidden)
        else:
            mixed, z, b, a = joined(hidden)
        z = z.reshape(rows, tokens, self.num_v_heads, self.head_v_dim)

        if tokens == 1 and self.conv_kernel_size == 4:
            q, k, v, convolution = conv4_silu_qkv(
                mixed,
                convolution,
                self.conv1d.weight,
                key_heads=self.num_k_heads,
                key_dim=self.head_k_dim,
                value_heads=self.num_v_heads,
                value_dim=self.head_v_dim,
            )
        else:
            window = mx.concatenate([convolution.astype(mixed.dtype), mixed], axis=1)
            convolution = window[:, -(self.conv_kernel_size - 1) :]
            mixed = nn.silu(self.conv1d(window))
            key_width = self.key_dim
            q = mixed[..., :key_width]
            k = mixed[..., key_width : 2 * key_width]
            v = mixed[..., 2 * key_width :]
            q = q.reshape(rows, tokens, self.num_k_heads, self.head_k_dim)
            k = k.reshape(rows, tokens, self.num_k_heads, self.head_k_dim)
            v = v.reshape(rows, tokens, self.num_v_heads, self.head_v_dim)
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
        # The decay is computed in float32, as in the reference model.
        output, recurrent = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, recurrent
        )
        output = self.norm(output, z).reshape(rows, tokens, -1)
        return self.out_proj(output), recurrent, convolution


class GatedAttention(nn.Module):
    def __init__(
        self, config: Qwen35Config, *, norm_weights_are_offset: bool = True
    ) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.rope_theta = config.rope_theta
        self.mrope_section = config.mrope_section
        self.scale = self.head_dim**-0.5
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, 2 * self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False)
        self.q_norm = OffsetRMSNorm(
            self.head_dim, config.rms_norm_eps, offset=norm_weights_are_offset
        )
        self.k_norm = OffsetRMSNorm(
            self.head_dim, config.rms_norm_eps, offset=norm_weights_are_offset
        )

    def _rope(self, x: mx.array, offsets: mx.array, rotary_positions=None) -> mx.array:
        if rotary_positions is not None:
            from resin_models_mlx.qwen3_5.media import apply_mrope

            return apply_mrope(
                x,
                rotary_positions,
                self.rotary_dim,
                self.rope_theta,
                self.mrope_section,
            )
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
        *,
        rotary_positions=None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        rows, tokens, _ = hidden.shape
        if (joined := getattr(self, "_joined_input", None)) is None:
            projected_query, key, value = (
                self.q_proj(hidden),
                self.k_proj(hidden),
                self.v_proj(hidden),
            )
        else:
            projected_query, key, value = joined(hidden)
        projected_query = projected_query.reshape(rows, tokens, self.heads, -1)
        query = projected_query[..., : self.head_dim]
        gate = projected_query[..., self.head_dim :]
        key = key.reshape(rows, tokens, self.kv_heads, -1)
        value = value.reshape(rows, tokens, self.kv_heads, -1)
        query = self._rope(
            self.q_norm(query).transpose(0, 2, 1, 3), offsets, rotary_positions
        )
        key = self._rope(
            self.k_norm(key).transpose(0, 2, 1, 3), offsets, rotary_positions
        )
        value = value.transpose(0, 2, 1, 3)
        new_key, new_value = key, value

        if isinstance(history, PagedHistory):
            from resin_models_mlx.paged_attention import paged_decode_attention

            output = paged_decode_attention(
                query[:, :, 0],
                key[:, :, 0],
                value[:, :, 0],
                history.units,
                history.pages,
                history.lengths,
                layer * history.layer_stride,
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
        output = _sigmoid_gate(output, gate.reshape(rows, tokens, -1))
        return (
            self.o_proj(output),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
        )


class FeedForward(_FeedForward):
    def __call__(self, hidden: mx.array) -> mx.array:
        if (joined := getattr(self, "_joined_input", None)) is None:
            return super().__call__(hidden)
        gate, up = joined(hidden)
        return self.down_proj(swiglu(gate, up))


class SparseMoe(_SparseMoe):
    """Softmax top-k routed experts plus a sigmoid-gated shared expert."""

    def __init__(self, config: _MoeArgs):
        super().__init__(config)
        self.shared_expert = FeedForward(
            config.hidden_size, config.shared_expert_intermediate_size
        )

    def __call__(self, hidden: mx.array) -> mx.array:
        probabilities = mx.softmax(self.gate(hidden), axis=-1, precise=True)
        experts = mx.argpartition(probabilities, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        scores = mx.take_along_axis(probabilities, experts, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        routed = (self.switch_mlp(hidden, experts) * scores[..., None]).sum(axis=-2)
        shared = _sigmoid_gate(
            self.shared_expert(hidden), self.shared_expert_gate(hidden)
        )
        return routed + shared


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        kind: str,
        *,
        norm_weights_are_offset: bool = True,
    ) -> None:
        super().__init__()
        self.is_linear = kind == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = GatedAttention(
                config, norm_weights_are_offset=norm_weights_are_offset
            )
        self.mlp = (
            SparseMoe(cast(_MoeArgs, config))
            if config.is_moe
            else FeedForward(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = OffsetRMSNorm(
            config.hidden_size, config.rms_norm_eps, offset=norm_weights_are_offset
        )
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, config.rms_norm_eps, offset=norm_weights_are_offset
        )


class Qwen35Model(nn.Module):
    """Text decoder over Qwen3.5 checkpoint weights."""

    def __init__(
        self,
        config: Qwen35Config,
        *,
        norm_weights_are_offset: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.norm_weights_are_offset = norm_weights_are_offset
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(config, kind, norm_weights_are_offset=norm_weights_are_offset)
            for kind in config.layer_types
        ]
        self.norm = OffsetRMSNorm(
            config.hidden_size, config.rms_norm_eps, offset=norm_weights_are_offset
        )
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _projection_groups(self):
        for layer in self.layers:
            attention = layer.linear_attn if layer.is_linear else layer.self_attn
            names = (
                ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
                if layer.is_linear
                else ("q_proj", "k_proj", "v_proj")
            )
            yield attention, tuple(getattr(attention, name) for name in names)
            mlp = (
                layer.mlp.shared_expert
                if isinstance(layer.mlp, SparseMoe)
                else layer.mlp
            )
            yield mlp, (mlp.gate_proj, mlp.up_proj)

    def prepare_for_inference(self):
        for module, projections in self._projection_groups():
            module._joined_input = _join_linears(projections)

    def _clear_prepared_projections(self):
        for module, _ in self._projection_groups():
            module._joined_input = None

    def update(self, parameters: dict, strict: bool = True):
        self._clear_prepared_projections()
        return super().update(parameters, strict=strict)

    def update_modules(self, modules: dict, strict: bool = True):
        self._clear_prepared_projections()
        return super().update_modules(modules, strict=strict)

    def __call__(
        self,
        token_ids: mx.array,
        offsets: mx.array,
        state: LinearState,
        history: Sequence[AttentionHistory | None] | PagedHistory,
        *,
        input_embeddings=None,
        rotary_positions=None,
    ) -> StepOutput:
        """Run ``[rows, tokens]`` inputs starting at per-row position ``offsets``."""

        hidden = (
            self.embed_tokens(token_ids)
            if input_embeddings is None
            else input_embeddings
        )
        keys, values, recurrent, convolution = [], [], [], []
        linear_index = full_index = 0
        pending = None
        for layer in self.layers:
            if pending is None:
                normed = layer.input_layernorm(hidden)
            else:
                hidden, normed = layer.input_layernorm.add_and_normalize(
                    hidden, pending
                )
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
                    normed,
                    offsets,
                    layer_history,
                    full_index,
                    rotary_positions=rotary_positions,
                )
                keys.append(key)
                values.append(value)
                full_index += 1
            hidden, normed = layer.post_attention_layernorm.add_and_normalize(
                hidden, mixed
            )
            # Keep the MLP branch separate until the following normalization,
            # where one kernel produces both its rounded residual and norm.
            pending = layer.mlp(normed)
        if pending is None:
            hidden = self.norm(hidden)
        else:
            _, hidden = self.norm.add_and_normalize(hidden, pending)
        return StepOutput(
            hidden,
            tuple(keys),
            tuple(values),
            LinearState(tuple(recurrent), tuple(convolution)),
            input_embeddings,
            rotary_positions,
        )

    def logits(self, hidden: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.embed_tokens.as_linear(hidden)
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
    "Qwen35Model",
    "StepOutput",
]
