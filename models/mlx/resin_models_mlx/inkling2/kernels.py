"""Fused Inkling 2 elementwise kernels for Apple GPUs.

Each replaces a chain of small MLX operations whose launches, not their
arithmetic, dominated batch-one decode. Results follow the CPU references in
``resin_models_cuda/inkling2/kernels.py``.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

_CONVOLUTION_SOURCE = r"""
    // One thread per (row, channel). State holds three raw inputs, oldest first.
    const uint index = thread_position_in_grid.x;
    const uint channel = index % C;
    const uint row = index / C;
    if (row >= uint(geometry[0])) return;
    const int tokens = geometry[1];
    float history[3];
    for (int i = 0; i < 3; ++i)
        history[i] = float(state[(ulong(row) * 3 + i) * C + channel]);
    for (int t = 0; t < tokens; ++t) {
        const ulong at = (ulong(row) * tokens + t) * C + channel;
        const float current = float(x[at]);
        float output = 0.0f;
        if (DYNAMIC) {
            // Tap zero is the current token; coefficients stay FP32.
            const uint head = channel / (C / HEADS);
            const ulong taps = ((ulong(row) * tokens + t) * HEADS + head) * 4;
            const float values[4] = {current, history[2], history[1], history[0]};
            for (int tap = 0; tap < 4; ++tap) {
                const float coefficient =
                    float(weight[tap * C + channel]) + float(dynamic[taps + tap]);
                output = metal::fma(values[tap], coefficient, output);
            }
        } else {
            // Chronological taps rounded to the input dtype; tap three is current.
            const float values[4] = {history[0], history[1], history[2], current};
            for (int tap = 0; tap < 4; ++tap) {
                const float coefficient = float(T(weight[channel * 4 + tap]));
                output = metal::fma(values[tap], coefficient, output);
            }
        }
        // FP32 roundings of 1 / sqrt(9) and 1 / sqrt(5), as the reference uses.
        const float scale = DYNAMIC ? 0.3333333432674408f : 0.4472135901451111f;
        y[at] = T((output + current) * scale);
        history[0] = history[1];
        history[1] = history[2];
        history[2] = current;
    }
    for (int i = 0; i < 3; ++i)
        next_state[(ulong(row) * 3 + i) * C + channel] = T(history[i]);
"""


@cache
def _convolution_kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_short_convolution",
        input_names=["x", "state", "weight", "dynamic", "geometry"],
        output_names=["y", "next_state"],
        source=_CONVOLUTION_SOURCE,
    )


def short_convolution(
    x: mx.array,
    state: mx.array,
    weight: mx.array,
    dynamic: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Four residual taps over three raw history inputs; output and new history.

    ``x`` is ``[B, T, C]`` and ``state`` ``[B, 3, C]`` (oldest first), both BF16.
    Static ``weight`` is ``[C, 4]`` in chronological order; dynamic convolutions
    pass FP32 ``weight [4, C]`` (tap zero current) and ``dynamic [B, T, H, 4]``.
    """
    batch, tokens, channels = x.shape
    is_dynamic = dynamic is not None
    heads = dynamic.shape[-2] if dynamic is not None else 1
    if dynamic is None:
        dynamic = mx.zeros((8,), dtype=mx.float32)
    geometry = mx.array([batch, tokens, 0, 0, 0, 0, 0, 0], dtype=mx.int32)
    threads = batch * channels
    return tuple(
        _convolution_kernel()(
            inputs=[x, state, weight, dynamic, geometry],
            template=[
                ("C", channels),
                ("HEADS", heads),
                ("DYNAMIC", is_dynamic),
                ("T", x.dtype),
            ],
            grid=(-(-threads // 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape, (batch, 3, channels)],
            output_dtypes=[x.dtype, x.dtype],
        )
    )  # type: ignore[return-value]


_SIGMOID = r"""
    // MLX's sigmoid (mlx/backend/metal/kernels/unary_ops.h, MIT), bit-for-bit.
    inline float resin_sigmoid(float x) {
        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
        return x < 0.0f ? y : 1.0f - y;
    }
"""

_ROUTE_SOURCE = r"""
    // One threadgroup per row; thread e owns expert e.
    const uint row = threadgroup_position_in_grid.x;
    const uint e = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    threadgroup float best_key[32];
    threadgroup uint best_id[32];
    threadgroup uint chosen[TOP_K];
    const float z = e < E ? float(logits[row * E + e]) : 0.0f;
    float key = e < E ? resin_sigmoid(z) + float(bias[e]) : -INFINITY;
    for (int k = 0; k < TOP_K; ++k) {
        // Largest key; equal keys choose the lower expert ID.
        float value = key;
        uint id = e;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            const float other = simd_shuffle_down(value, offset);
            const uint other_id = simd_shuffle_down(id, offset);
            if (other > value || (other == value && other_id < id)) {
                value = other;
                id = other_id;
            }
        }
        if (lane == 0) {
            best_key[simd] = value;
            best_id[simd] = id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == 0) {
            float top = best_key[0];
            uint top_id = best_id[0];
            for (uint s = 1; s < (E + 31) / 32; ++s) {
                if (best_key[s] > top || (best_key[s] == top && best_id[s] < top_id)) {
                    top = best_key[s];
                    top_id = best_id[s];
                }
            }
            chosen[k] = top_id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == chosen[k]) key = -INFINITY;
    }
    if (e == 0) {
        // Unbiased log-sigmoids, normalized stably over the selection.
        float p[TOP_K];
        float peak = -INFINITY;
        for (int k = 0; k < TOP_K; ++k) {
            const float value = -float(logits[row * E + chosen[k]]);
            // log-sigmoid(z) = -logaddexp(0, -z); within an FP32 ULP of MLX's ops.
            const float high = metal::max(value, 0.0f);
            const float low = metal::min(value, 0.0f);
            p[k] = -(high + log1p(metal::precise::exp(low - high)));
            peak = metal::max(peak, p[k]);
        }
        float shifted[TOP_K];
        for (int k = 0; k < TOP_K; ++k) shifted[k] = metal::precise::exp(p[k] - peak);
        float total;
        if (TOP_K == 8) {
            total = ((shifted[0] + shifted[2]) + (shifted[1] + shifted[3]))
                  + ((shifted[4] + shifted[6]) + (shifted[5] + shifted[7]));
        } else {
            total = 0.0f;
            for (int k = 0; k < TOP_K; ++k) total += shifted[k];
        }
        const float normalizer = peak + metal::precise::log(total);
        for (int k = 0; k < TOP_K; ++k) {
            ids[row * TOP_K + k] = chosen[k];
            weights[row * TOP_K + k] = metal::precise::exp(p[k] - normalizer) * scale[0];
        }
    }
"""

_FACTOR = r"""
    // NVFP4 output scale: activation amax times weight amax over 1536 * denominator.
    inline float resin_factor(float activation_amax, float weight_amax, float denominator) {
        return (activation_amax * weight_amax) * metal::precise::divide(1.0f, 1536.0f * denominator);
    }
"""

_ACTIVATE_SOURCE = r"""
    // One thread per (route, intermediate channel); gate and up rows interleave.
    const uint index = thread_position_in_grid.x;
    const uint route = index / I;
    const uint j = index % I;
    if (route >= uint(geometry[0])) return;
    float factor = 1.0f;
    if (NVFP4) {
        factor = resin_factor(
            float(activation_amax[rows[route]]), float(weight_amax[ids[route]]),
            float(denominator[0]));
    }
    const ulong at = ulong(route) * (2 * I) + 2 * j;
    // The packed GEMM rounds each scaled output to BF16.
    const float gate = float(bfloat(float(gate_up[at]) * factor));
    const float up = float(bfloat(float(gate_up[at + 1]) * factor));
    const float capped_gate = GATE_CAP * metal::precise::tanh(gate / GATE_CAP);
    const float capped_up = UP_CAP * metal::precise::tanh(up / UP_CAP);
    const float sigmoid = resin_sigmoid(gate);
    activated[ulong(route) * I + j] =
        bfloat(((capped_gate * sigmoid) * capped_up) * float(gamma[route]));
"""

_COMBINE_SOURCE = r"""
    // One thread per (row, hidden channel): round each route, sum slots in order.
    const uint index = thread_position_in_grid.x;
    const uint row = index / H;
    const uint c = index % H;
    if (row >= uint(geometry[0])) return;
    float total = 0.0f;
    for (int slot = 0; slot < TOP_K; ++slot) {
        const uint route = inverse[row * TOP_K + slot];
        float factor = 1.0f;
        if (NVFP4) {
            factor = resin_factor(
                float(activation_amax[route]), float(weight_amax[ids[route]]),
                float(denominator[0]));
        }
        const float value = float(bfloat(float(down[ulong(route) * H + c]) * factor));
        total = slot == 0 ? value : total + value;
    }
    output[ulong(row) * H + c] = bfloat(total);
"""


@cache
def _route_kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_route_experts",
        input_names=["logits", "bias", "scale"],
        output_names=["ids", "weights"],
        header=_SIGMOID,
        source=_ROUTE_SOURCE,
    )


@cache
def _activate_kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_expert_activation",
        input_names=[
            "gate_up",
            "gamma",
            "rows",
            "ids",
            "activation_amax",
            "weight_amax",
            "denominator",
            "geometry",
        ],
        output_names=["activated"],
        header=_FACTOR + _SIGMOID,
        source=_ACTIVATE_SOURCE,
    )


@cache
def _combine_kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_expert_combine",
        input_names=[
            "down",
            "inverse",
            "ids",
            "activation_amax",
            "weight_amax",
            "denominator",
            "geometry",
        ],
        output_names=["output"],
        header=_FACTOR,
        source=_COMBINE_SOURCE,
    )


def _geometry(*values: int) -> mx.array:
    return mx.array([*values, *([0] * (8 - len(values)))], dtype=mx.int32)


def _placeholder(dtype=mx.float32) -> mx.array:
    return mx.zeros((8,), dtype=dtype)


def route_experts(
    logits: mx.array, bias: mx.array, top_k: int, scale: float
) -> tuple[mx.array, mx.array]:
    """Top-k by biased sigmoid (lower ID wins ties); unbiased normalized weights."""
    rows, experts = logits.shape
    if experts > 1024:
        raise ValueError("Inkling 2 routing supports at most 1024 experts")
    threads = max(32, -(-experts // 32) * 32)
    ids, weights = _route_kernel()(
        inputs=[logits, bias, mx.full((8,), scale, dtype=mx.float32)],
        template=[("E", experts), ("TOP_K", top_k)],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return ids, weights


def expert_activation(
    gate_up: mx.array,
    gamma: mx.array,
    gate_cap: float,
    up_cap: float,
    rows: mx.array | None = None,
    ids: mx.array | None = None,
    activation_amax: mx.array | None = None,
    weight_amax: mx.array | None = None,
    denominator: mx.array | None = None,
) -> mx.array:
    """Scale (NVFP4), round, then capped SwiGLU times the route weight, in BF16."""
    routes, width = gate_up.shape
    intermediate = width // 2
    nvfp4 = activation_amax is not None
    if not float(gate_cap).is_integer() or not float(up_cap).is_integer():
        raise ValueError("Inkling 2 expert softcaps must be whole numbers")
    placeholder = _placeholder()
    threads = routes * intermediate
    return _activate_kernel()(
        inputs=[
            gate_up,
            gamma,
            rows if rows is not None else _placeholder(mx.uint32),
            ids if ids is not None else _placeholder(mx.uint32),
            activation_amax if activation_amax is not None else placeholder,
            weight_amax if weight_amax is not None else placeholder,
            denominator if denominator is not None else placeholder,
            _geometry(routes),
        ],
        template=[
            ("I", intermediate),
            ("NVFP4", nvfp4),
            ("GATE_CAP", int(gate_cap)),
            ("UP_CAP", int(up_cap)),
        ],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(routes, intermediate)],
        output_dtypes=[mx.bfloat16],
    )[0]


def expert_combine(
    down: mx.array,
    inverse: mx.array,
    top_k: int,
    ids: mx.array | None = None,
    activation_amax: mx.array | None = None,
    weight_amax: mx.array | None = None,
    denominator: mx.array | None = None,
) -> mx.array:
    """Scale (NVFP4) and round each route to BF16; sum each row's slots in order."""
    routes, hidden = down.shape
    rows = routes // top_k
    placeholder = _placeholder()
    threads = rows * hidden
    return _combine_kernel()(
        inputs=[
            down,
            inverse,
            ids if ids is not None else _placeholder(mx.uint32),
            activation_amax if activation_amax is not None else placeholder,
            weight_amax if weight_amax is not None else placeholder,
            denominator if denominator is not None else placeholder,
            _geometry(rows),
        ],
        template=[
            ("H", hidden),
            ("TOP_K", top_k),
            ("NVFP4", activation_amax is not None),
        ],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]


__all__ = ["expert_activation", "expert_combine", "route_experts", "short_convolution"]
