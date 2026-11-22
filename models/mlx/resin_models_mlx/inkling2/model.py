"""Inkling 2 target forward pass on MLX: looped layers, depth mixes and sparse MLPs.

The arithmetic follows the CUDA runner (``resin_models_cuda/inkling2/model.py``
and its CPU references in ``kernels.py``), including its BF16 rounding points.
Attention storage is supplied by the caller through :class:`AttentionStore`, so
the same forward pass runs over a dense test cache or the Engine's paged arena.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import update_wrapper
from typing import Protocol

import mlx.core as mx
import numpy as np

from resin.models.inkling2.config import Inkling2Config
from resin_models_mlx.inkling2.kernels import depth_mix as fused_depth_mix
from resin_models_mlx.inkling2.kernels import (
    expert_activation,
    expert_combine,
    gemv_f32,
    pad_to_device,
    rms_norm,
    route_experts,
    short_convolution,
)
from resin_models_mlx.inkling2.quantization import quantize_activations

BF16 = mx.bfloat16
F32 = mx.float32


def _decode_compiled(function):
    """Fuse ``function`` with ``mx.compile`` for one-token (decode) inputs only.

    Decode shapes depend only on the batch size, a small bounded set. Prefill
    chunk lengths vary with every prompt, so prefill runs uncompiled rather
    than compiling a new graph for each new length.
    """
    compiled = mx.compile(function)

    def call(first, *args):
        return (compiled if first.shape[1] == 1 else function)(first, *args)

    return update_wrapper(call, function)


class AttentionStore(Protocol):
    """Write one visit's keys and values, then attend over its retained history."""

    def attend(
        self,
        local: bool,
        slot: int,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        bias: mx.array,
        scale: float,
        window: int,
    ) -> tuple[mx.array, mx.array]:
        """Return FP32 ``[B, T, H, D]`` outputs and ``[B, T, H]`` log-sum-exps.

        ``q`` is ``[B, T, H, D]``, ``k``/``v`` are ``[B, T, KV, D]`` and ``bias``
        is ``[B, T, H, E]``: entry ``e`` applies to the key ``e`` positions
        before the query. ``slot`` indexes the visit within its local or global
        cache group. Local visits see at most ``window`` keys.
        """
        ...


@dataclass(slots=True)
class Experts:
    """Packed expert weights for one layer and their quantization modes."""

    w13: mx.array
    w13_scales: mx.array
    w2: mx.array
    w2_scales: mx.array
    mode: str
    # NVFP4 only: per-expert weight amax and the output scale denominator.
    w13_amax: mx.array | None = None
    w2_amax: mx.array | None = None
    w13_denominator: mx.array | None = None
    w2_denominator: mx.array | None = None

    @property
    def group_size(self) -> int:
        return 16 if self.mode == "nvfp4" else 32

    @property
    def bits(self) -> int:
        return 4 if self.mode == "nvfp4" else 8


@dataclass(slots=True)
class Controller:
    down: mx.array
    gain: mx.array
    forget: mx.array


@dataclass(slots=True)
class Layer:
    local: bool
    # Local layers use ``attn_norm`` for every stream; global layers use q/k/v norms.
    norms: tuple[mx.array, ...]
    mlp_norm: mx.array
    # Packed projections, as in the CUDA runner. Local layers: rows
    # [K | V | Q | rel | gate | sink] of one shared source, the last four
    # contiguous for a separate query stream. Global layers: [Q | rel | gate |
    # sink] here, with K and V packed beside their dynamic-convolution taps.
    projection: mx.array
    k_projection: mx.array | None
    v_projection: mx.array | None
    wo: mx.array
    q_scale: mx.array
    rel_scale: mx.array
    rel_proj: mx.array
    # Local: chronological [C, 4] static taps. Global: FP32 [4, C] static taps
    # (tap zero is the current token); their dynamic part is in K/V projections.
    k_conv: mx.array
    v_conv: mx.array
    attn_conv: mx.array
    mlp_conv: mx.array
    router: mx.array
    router_bias: mx.array
    experts: Experts
    controllers: dict[str, Controller] | None = None


@dataclass(slots=True)
class Inkling2Weights:
    embed: mx.array
    embed_norm: mx.array
    norm: mx.array
    unembed: mx.array
    attn_queries: list[mx.array]
    mlp_queries: list[mx.array]
    output_query: mx.array
    layers: list[Layer]
    # Loop message queries, stacked [core layers, 4, H]: q, k, v and MLP streams.
    loop_queries: mx.array | None = None


def linear(x: mx.array, weight: mx.array) -> mx.array:
    """BF16 GEMM with FP32 accumulation and a BF16 result."""
    return x @ weight.T


def linear_f32(x: mx.array, weight: mx.array) -> mx.array:
    """FP32 GEMM against a weight stored in FP32 at load."""
    return x.astype(F32) @ weight.T


def depth_mix(
    sources: list[mx.array],
    query: mx.array,
    addend: mx.array | None = None,
) -> list[mx.array]:
    """Attend over residual blocks with normalized values; FP32 outputs per query row."""
    shape = sources[0].shape
    width = shape[-1]
    mixed = fused_depth_mix(
        [x.reshape(-1, width) for x in sources],
        query,
        None if addend is None else addend.reshape(-1, width),
    )
    return [mixed[:, row].reshape(shape) for row in range(mixed.shape[1])]


@_decode_compiled
def l2_clip(
    x: mx.array, scale: mx.array | None, threshold: float, tau: mx.array | None
) -> mx.array:
    values = x.astype(F32) if scale is None else x.astype(F32) * scale.astype(F32)
    norm = mx.sqrt(mx.sum(values * values, axis=-1, keepdims=True))
    values = values * mx.minimum(threshold / (norm + 1e-6), 1.0)
    values = values.astype(x.dtype)
    if tau is not None:
        values = (values.astype(F32) * tau[..., None, None]).astype(x.dtype)
    return values


def log_scaling_tau(positions: mx.array, n_floor: int, alpha: float) -> mx.array:
    effective = (positions + 1).astype(F32)
    return 1.0 + alpha * mx.log(mx.maximum(effective / float(n_floor), 1.0))


# From this many routes (512 tokens), prefill runs one dense FP32 GEMM per
# expert over its sorted segment of exactly dequantized weights. On an M4 Max
# the two paths tie at 512 tokens; at 2,048 tokens (16K routes) the segments are
# 2.3x (NVFP4) to 3.5x (MXFP8) faster, dequantization included. Below the
# threshold, dequantizing every expert costs more than the gathered GEMM saves.
SEGMENTED_ROUTES = 4096


def _expert_matmul(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    experts: Experts,
    rows: mx.array | None,
    ids: mx.array,
    segments: list[tuple[int, int, int]] | None = None,
) -> tuple[mx.array, mx.array | None]:
    """Quantize activations as the checkpoint requires, then run the packed GEMM.

    Returns the FP32 product of dequantized codes, one row per route in ``ids``
    order (sorted by expert), and the NVFP4 activation amax per row of ``x``;
    the output scale is applied with the next rounding. ``rows`` maps each route
    to its row of ``x``; ``None`` means one row per route. ``segments`` lists
    ``(expert, start, stop)`` route ranges; given, each expert's weights are
    dequantized to FP32 (exactly) for a dense GEMM.
    """
    values, activation_amax = quantize_activations(x, experts.mode)
    if segments is None:
        output = mx.gather_qmm(
            values[:, None, :],
            weight,
            scales,
            None,
            lhs_indices=rows,
            rhs_indices=ids,
            transpose=True,
            group_size=experts.group_size,
            bits=experts.bits,
            mode=experts.mode,
            sorted_indices=True,
        )[:, 0, :]
    else:
        routed = values if rows is None else values[rows]
        parts = []
        for expert, start, stop in segments:
            dense = mx.dequantize(
                weight[expert],
                scales[expert],
                None,
                group_size=experts.group_size,
                bits=experts.bits,
                mode=experts.mode,
                dtype=F32,
            )
            parts.append(routed[start:stop] @ dense.T)
        output = mx.concatenate(parts) if len(parts) > 1 else parts[0]
    if activation_amax is not None:
        activation_amax = pad_to_device(activation_amax)
    return output, activation_amax


def _segments(sorted_ids: mx.array, experts: int) -> list[tuple[int, int, int]]:
    """Route ranges per expert; reading the routing back waits for the router."""
    counts = np.bincount(np.array(sorted_ids), minlength=experts)
    stops = np.cumsum(counts)
    return [
        (expert, int(stops[expert] - counts[expert]), int(stops[expert]))
        for expert in np.flatnonzero(counts).tolist()
    ]


@_decode_compiled
def _sort_routes(ids, weights, top_k: int):
    """Order routes by expert; ``ids``/``weights`` are ``[B, T, top_k]``."""
    flat_ids = ids.reshape(-1)
    order = mx.argsort(flat_ids)
    source_rows = (order // top_k).astype(mx.uint32)
    inverse = mx.argsort(order).astype(mx.uint32)
    return flat_ids[order], source_rows, weights.reshape(-1)[order], inverse


def sparse_mlp(x: mx.array, layer: Layer, config: Inkling2Config) -> mx.array:
    """Route each row to its experts; returns the BF16 sum of BF16-rounded routes."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    top_k = config.experts_per_token
    ids, weights = route_experts(
        flat.astype(F32) @ layer.router.T, layer.router_bias, top_k, config.route_scale
    )
    sorted_ids, source_rows, gamma, inverse = _sort_routes(
        ids.reshape(*shape[:2], top_k), weights.reshape(*shape[:2], top_k), top_k
    )
    experts = layer.experts
    nvfp4 = experts.mode == "nvfp4"
    segments = (
        _segments(sorted_ids, experts.w13.shape[0])
        if sorted_ids.size >= SEGMENTED_ROUTES
        else None
    )
    gate_up, gate_up_amax = _expert_matmul(
        flat,
        experts.w13,
        experts.w13_scales,
        experts,
        source_rows,
        sorted_ids,
        segments,
    )
    activated = expert_activation(
        gate_up,
        gamma,
        config.mlp_gate_softcap,
        config.mlp_up_softcap,
        source_rows,
        sorted_ids,
        gate_up_amax if nvfp4 else None,
        experts.w13_amax,
        experts.w13_denominator,
    )
    down, down_amax = _expert_matmul(
        activated, experts.w2, experts.w2_scales, experts, None, sorted_ids, segments
    )
    return expert_combine(
        down,
        inverse,
        top_k,
        sorted_ids,
        down_amax if nvfp4 else None,
        experts.w2_amax,
        experts.w2_denominator,
    ).reshape(shape)


@_decode_compiled
def _controller(
    dest: mx.array, message: mx.array, down: mx.array, gain: mx.array, forget: mx.array
) -> mx.array:
    pair = mx.stack([dest, message], axis=-2)
    normalized = rms_norm(pair, None, 1e-6).reshape(*dest.shape[:-1], -1)
    hidden = linear(normalized, down)
    gain_values = mx.tanh(linear_f32(hidden, gain))
    forget_values = mx.tanh(linear_f32(hidden, forget))
    return (1 - forget_values) * dest.astype(F32) + gain_values * message.astype(F32)


def controller(dest: mx.array, message: mx.array, weights: Controller) -> mx.array:
    """Blend a pass message into a residual stream through a rank-limited gate."""
    return _controller(dest, message, weights.down, weights.gain, weights.forget)


def logits(
    hidden: mx.array, weights: Inkling2Weights, config: Inkling2Config
) -> mx.array:
    """FP32 logits from BF16 weights, sliced to the real vocabulary and softcapped."""
    shape = hidden.shape
    values = gemv_f32(hidden.reshape(-1, shape[-1]), weights.unembed)
    values = values.reshape(*shape[:-1], -1)[..., : config.unpadded_vocab_size]
    cap = config.final_logit_softcapping
    return cap * mx.tanh(values * (1.0 / cap))


class Inkling2Model:
    """Inkling 2 target over an :class:`AttentionStore`.

    ``state`` holds each visit's raw convolution history, ``[B, visits, 3, W]``
    with channels ``[K inputs | V inputs | attention-conv | MLP-conv]``.
    """

    def __init__(self, config: Inkling2Config, weights: Inkling2Weights) -> None:
        if not config.attn_res_normalize_values:
            raise ValueError("the Inkling 2 MLX depth mix requires normalized values")
        self.config, self.weights = config, weights
        order = config.layer_order
        slots = {}
        for group in (
            config.execution_local_layer_ids,
            config.execution_global_layer_ids,
        ):
            slots.update((visit, slot) for slot, visit in enumerate(group))
        self.visits = tuple((layer, slots[visit]) for visit, layer in enumerate(order))
        self.kv_width = config.num_key_value_heads * config.head_dim
        self.state_width = config.state_width()

    def initial_state(self, batch: int) -> mx.array:
        shape = (batch, self.config.execution_layer_count, 3, self.state_width)
        return mx.zeros(shape, dtype=BF16)

    def embed(self, token_ids: mx.array) -> mx.array:
        weights = self.weights
        return rms_norm(
            weights.embed[token_ids], weights.embed_norm, self.config.rms_norm_eps
        )

    def __call__(
        self,
        token_ids: mx.array,
        positions: mx.array,
        state: mx.array,
        store: AttentionStore,
        *,
        last_only: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Return softcapped FP32 logits and the next convolution state."""
        hidden, next_state = self.hidden(token_ids, positions, state, store)
        if last_only:
            hidden = hidden[:, -1:]
        return logits(hidden, self.weights, self.config), next_state

    def hidden(
        self,
        token_ids: mx.array,
        positions: mx.array,
        state: mx.array,
        store: AttentionStore,
    ) -> tuple[mx.array, mx.array]:
        config, weights = self.config, self.weights
        eps = config.rms_norm_eps
        tau = log_scaling_tau(
            positions, config.log_scaling_n_floor, config.log_scaling_alpha
        )
        partial = self.embed(token_ids)
        blocks: list[mx.array] = []
        loop = config.loop
        prelude: tuple[list[mx.array], mx.array] | None = None
        message_bank: list[mx.array] = []
        kv, hidden_size = self.kv_width, config.hidden_size
        next_states = []
        for visit, (index, slot) in enumerate(self.visits):
            layer = weights.layers[index]
            if loop is not None:
                if index == loop.core_start:
                    if prelude is None:
                        prelude = (blocks.copy(), partial)
                    else:
                        prelude_blocks, prelude_partial = prelude
                        # Only blocks produced inside the previous pass carry messages.
                        message_bank = [*blocks[len(prelude_blocks) + 1 :], partial]
                        blocks, partial = prelude_blocks.copy(), prelude_partial
                elif index == loop.core_end:
                    message_bank = []
            streams = (
                depth_mix([*blocks, partial], weights.attn_queries[index - 1])
                if blocks
                else [partial] * (1 if layer.local else 3)
            )
            messages = None
            if message_bank:
                assert loop is not None and layer.controllers is not None
                assert weights.loop_queries is not None
                query = weights.loop_queries[index - loop.core_start]
                messages = depth_mix(message_bank, query)
                if layer.local:
                    streams = streams * 3
                streams = [
                    controller(stream, message, layer.controllers[name])
                    for name, stream, message in zip(
                        ("attn", "k", "v"), streams, messages[:3], strict=True
                    )
                ]
            if layer.local and messages is None:
                q_source = rms_norm(streams[0], layer.norms[0], eps)
                sources = (q_source, q_source, q_source)
            else:
                norms = layer.norms * 3 if layer.local else layer.norms
                sources = tuple(
                    rms_norm(x, norm, eps)
                    for norm, x in zip(norms, streams, strict=True)
                )
            visit_state = state[:, visit]
            attention, k_history, v_history = self._attention(
                layer, sources, visit_state, tau, visit, slot, store
            )
            attention_state = visit_state[..., 2 * kv : 2 * kv + hidden_size]
            attention, attention_history = short_convolution(
                attention, attention_state, layer.attn_conv
            )
            boundary = index % config.n_layers_per_attn_res_block == 0
            mlp_sources = (
                [*blocks, partial, attention] if boundary else [*blocks, partial]
            )
            mlp_input = depth_mix(
                mlp_sources,
                weights.mlp_queries[index],
                None if boundary else attention,
            )[0]
            if messages is not None:
                assert layer.controllers is not None
                mlp_input = controller(mlp_input, messages[3], layer.controllers["mlp"])
            normalized = rms_norm(mlp_input, layer.mlp_norm, eps)
            expert_output = sparse_mlp(normalized, layer, config)
            mlp_state = visit_state[..., -hidden_size:]
            mlp, mlp_history = short_convolution(
                expert_output, mlp_state, layer.mlp_conv
            )
            if boundary:
                blocks.append(partial)
                partial = (attention.astype(F32) + mlp.astype(F32)).astype(BF16)
            else:
                partial = (
                    partial.astype(F32) + attention.astype(F32) + mlp.astype(F32)
                ).astype(BF16)
            next_states.append(
                mx.concatenate(
                    [k_history, v_history, attention_history, mlp_history], axis=-1
                )
            )
        hidden = rms_norm(
            depth_mix([*blocks, partial], weights.output_query)[0], weights.norm, eps
        )
        return hidden, mx.stack(next_states, axis=1)

    def _attention(
        self,
        layer: Layer,
        sources: tuple[mx.array, ...],
        state: mx.array,
        tau: mx.array,
        visit: int,
        slot: int,
        store: AttentionStore,
    ) -> tuple[mx.array, mx.array, mx.array]:
        config = self.config
        q_source, k_source, v_source = sources
        batch, tokens, _ = q_source.shape
        heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
        if layer.local:
            heads, kv_heads = (
                config.swa_num_attention_heads,
                config.swa_num_key_value_heads,
            )
        d, r, width = config.head_dim, config.d_rel, self.kv_width
        query_sizes = (heads * d, heads * r, heads * d)
        k_dynamic = v_dynamic = None
        if layer.local:
            if k_source is q_source and v_source is q_source:
                projected = linear(q_source, layer.projection)
                k_input, v_input = (
                    projected[..., :width],
                    projected[..., width : 2 * width],
                )
                query = projected[..., 2 * width :]
            else:
                k_input = linear(k_source, layer.projection[:width])
                v_input = linear(v_source, layer.projection[width : 2 * width])
                query = linear(q_source, layer.projection[2 * width :])
        else:
            assert layer.k_projection is not None and layer.v_projection is not None
            query = linear(q_source, layer.projection)
            k_packed = linear(k_source, layer.k_projection)
            v_packed = linear(v_source, layer.v_projection)
            k_input, v_input = k_packed[..., :width], v_packed[..., :width]
            k_dynamic = k_packed[..., width:].reshape(batch, tokens, kv_heads, 4)
            v_dynamic = v_packed[..., width:].reshape(batch, tokens, kv_heads, 4)
        q, relative, gate, sink = mx.split(
            query,
            [sum(query_sizes[:1]), sum(query_sizes[:2]), sum(query_sizes)],
            axis=-1,
        )
        k, k_history = short_convolution(
            k_input, state[..., :width], layer.k_conv, k_dynamic
        )
        v, v_history = short_convolution(
            v_input, state[..., width : 2 * width], layer.v_conv, v_dynamic
        )
        visit_tau = None if layer.local else tau
        q = l2_clip(
            q.reshape(batch, tokens, heads, d),
            layer.q_scale,
            config.qk_l2_clip_threshold,
            visit_tau,
        )
        k = l2_clip(
            k.reshape(batch, tokens, kv_heads, d),
            None,
            config.qk_l2_clip_threshold,
            None,
        )
        v = rms_norm(v.reshape(batch, tokens, kv_heads, d), None, config.rms_norm_eps)
        relative = l2_clip(
            relative.reshape(batch, tokens, heads, r),
            layer.rel_scale,
            config.rel_l2_clip_threshold,
            visit_tau,
        )
        bias = relative @ layer.rel_proj
        output, lse = store.attend(
            layer.local,
            slot,
            q,
            k,
            v,
            bias,
            d**-0.5,
            config.sliding_window_size,
        )
        gated = output * mx.sigmoid(gate.astype(F32).reshape(batch, tokens, heads, d))
        gated = gated * mx.sigmoid(lse - sink.astype(F32))[..., None]
        attended = gated.astype(BF16).reshape(batch, tokens, heads * d)
        return linear(attended, layer.wo), k_history, v_history


__all__ = [
    "AttentionStore",
    "Controller",
    "Experts",
    "Inkling2Model",
    "Inkling2Weights",
    "Layer",
    "depth_mix",
    "l2_clip",
    "log_scaling_tau",
    "logits",
    "sparse_mlp",
]
