"""Fused Inkling 2 elementwise kernels for Apple GPUs.

Each replaces a chain of small MLX operations whose launches, not their
arithmetic, dominated batch-one decode. Results follow the CPU references in
``resin_models_cuda/inkling2/kernels.py``.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from resin_models_mlx.metal import constant

_CONV_TOKENS_PER_THREAD = 16
_CONVOLUTION_SOURCE = r"""
    const uint index = thread_position_in_grid.x;
    const uint channel = index % C;
    const uint row = index / C;
    if (row >= uint(geometry[0])) return;
    const int tokens = geometry[1];
    const int start = thread_position_in_grid.y * TOKENS_PER_THREAD;
    const int stop = min(tokens, start + TOKENS_PER_THREAD);
    // The three preceding raw inputs seed each independent time tile.
    float inputs[4];
    for (int i = 0; i < 3; ++i) {
        inputs[i] = start == 0
            ? float(state[(ulong(row) * 3 + i) * C + channel])
            : float(x[(ulong(row) * tokens + start - 3 + i) * C + channel]);
    }
    for (int t = start; t < stop; ++t) {
        const ulong at = (ulong(row) * tokens + t) * C + channel;
        inputs[3] = float(x[at]);
        const ulong taps = ((ulong(row) * tokens + t) * HEADS + channel / (C / HEADS)) * 4;
        float output = 0.0f;
        for (int tap = 0; tap < 4; ++tap) {
            const float coefficient = DYNAMIC
                ? float(weight[tap * C + channel]) + float(dynamic[taps + tap])
                : float(T(weight[channel * 4 + tap]));
            output = metal::fma(inputs[DYNAMIC ? 3 - tap : tap], coefficient, output);
        }
        // FP32 roundings of 1 / sqrt(9) and 1 / sqrt(5), as the reference uses.
        const float scale = DYNAMIC ? 0.3333333432674408f : 0.4472135901451111f;
        y[at] = T((output + inputs[3]) * scale);
        for (int i = 0; i < 3; ++i) inputs[i] = inputs[i + 1];
    }
    if (stop == tokens) {
        for (int i = 0; i < 3; ++i)
            next_state[(ulong(row) * 3 + i) * C + channel] = T(inputs[i]);
    }
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
        dynamic = constant((), "float32")
    geometry = constant((batch, tokens), "int32")
    threads = batch * channels
    return tuple(
        _convolution_kernel()(
            inputs=[x, state, weight, dynamic, geometry],
            template=[
                ("C", channels),
                ("HEADS", heads),
                ("DYNAMIC", is_dynamic),
                ("TOKENS_PER_THREAD", _CONV_TOKENS_PER_THREAD),
                ("T", x.dtype),
            ],
            grid=(-(-threads // 256) * 256, -(-tokens // _CONV_TOKENS_PER_THREAD), 1),
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
    return constant(tuple(values), "int32")


def _placeholder(dtype: str = "float32") -> mx.array:
    return constant((), dtype)


def route_experts(
    logits: mx.array, bias: mx.array, top_k: int, scale: float
) -> tuple[mx.array, mx.array]:
    """Top-k by biased sigmoid (lower ID wins ties); unbiased normalized weights."""
    rows, experts = logits.shape
    if experts > 1024:
        raise ValueError("Inkling 2 routing supports at most 1024 experts")
    threads = max(32, -(-experts // 32) * 32)
    ids, weights = _route_kernel()(
        inputs=[logits, bias, constant((float(scale),) * 8, "float32")],
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
            rows if rows is not None else _placeholder("uint32"),
            ids if ids is not None else _placeholder("uint32"),
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
            ids if ids is not None else _placeholder("uint32"),
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


_THREADS = 256

_MIX_TEMPLATE = r"""
    // One threadgroup per token row. Normalize each source over H, score it
    // against each query row, softmax over sources, and mix normalized values.
    constexpr int PER = (H + THREADS - 1) / THREADS;
    constexpr int SIMDS = THREADS / 32;
    constexpr int SUMS = S * (R + 1);
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    threadgroup float partial_sums[SUMS][SIMDS];
    threadgroup float totals[SUMS];
    float values[S][PER];
    float sums[SUMS];
    for (int i = 0; i < SUMS; ++i) sums[i] = 0.0f;
    $LOADS$
    for (int s = 0; s < S; ++s)
        for (int k = 0; k < PER; ++k) sums[s] += values[s][k] * values[s][k];
    $REDUCE$
    float inverse[S];
    for (int s = 0; s < S; ++s)
        inverse[s] = metal::precise::rsqrt(totals[s] / float(H) + 1e-6f);
    for (int s = 0; s < S; ++s)
        for (int k = 0; k < PER; ++k) values[s][k] *= inverse[s];
    for (int i = 0; i < SUMS; ++i) sums[i] = 0.0f;
    for (int k = 0; k < PER; ++k) {
        const uint c = tid + k * THREADS;
        if (c >= H) continue;
        for (int r = 0; r < R; ++r) {
            const float q = float(queries[r * H + c]);
            for (int s = 0; s < S; ++s) sums[S + r * S + s] += values[s][k] * q;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    $REDUCE$
    const float scale = metal::precise::rsqrt(float(H));
    for (int r = 0; r < R; ++r) {
        float scores[S];
        float peak = -INFINITY;
        for (int s = 0; s < S; ++s) {
            scores[s] = totals[S + r * S + s] * scale;
            peak = metal::max(peak, scores[s]);
        }
        float denominator = 0.0f;
        for (int s = 0; s < S; ++s) {
            scores[s] = metal::precise::exp(scores[s] - peak);
            denominator += scores[s];
        }
        for (int k = 0; k < PER; ++k) {
            const uint c = tid + k * THREADS;
            if (c >= H) continue;
            float mixed = 0.0f;
            for (int s = 0; s < S; ++s) mixed += (scores[s] / denominator) * values[s][k];
            output[(ulong(row) * R + r) * H + c] = mixed;
        }
    }
"""

_MIX_REDUCE = r"""
    for (int i = 0; i < SUMS; ++i) {
        const float value = simd_sum(sums[i]);
        if (lane == 0) partial_sums[i][simd] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < SUMS) {
        float total = 0.0f;
        for (int g = 0; g < SIMDS; ++g) total += partial_sums[tid][g];
        totals[tid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""


@cache
def _mix_kernel(sources: int, addend: bool):
    loads = []
    for index in range(sources):
        extra = (
            " + float(addend[ulong(row) * H + c])"
            if addend and index == sources - 1
            else ""
        )
        loads.append(
            f"for (int k = 0; k < PER; ++k) {{ const uint c = tid + k * THREADS; "
            f"values[{index}][k] = c < H ? float(s{index}[ulong(row) * H + c]){extra} : 0.0f; }}"
        )
    source = _MIX_TEMPLATE.replace("$LOADS$", "\n    ".join(loads)).replace(
        "$REDUCE$", _MIX_REDUCE
    )
    names = (
        [f"s{i}" for i in range(sources)] + (["addend"] if addend else []) + ["queries"]
    )
    return mx.fast.metal_kernel(
        name=f"resin_inkling2_depth_mix_{sources}{'a' if addend else ''}",
        input_names=names,
        output_names=["output"],
        source=source,
    )


def depth_mix(
    sources: list[mx.array], queries: mx.array, addend: mx.array | None = None
) -> mx.array:
    """Attention over residual blocks; FP32 ``[rows, R, H]`` for query rows ``R``.

    Each source is ``[rows, H]``; ``addend`` (optional) adds to the last source
    before normalization. Values are RMS-normalized (no weight, eps 1e-6).
    """
    rows, width = sources[0].shape
    queries = queries.reshape(-1, width)
    if queries.dtype != mx.float32:
        queries = queries.astype(mx.float32)
    inputs = [*sources, *([addend] if addend is not None else []), queries]
    return _mix_kernel(len(sources), addend is not None)(
        inputs=inputs,
        template=[
            ("H", width),
            ("R", queries.shape[0]),
            ("S", len(sources)),
            ("THREADS", _THREADS),
        ],
        grid=(rows * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, queries.shape[0], width)],
        output_dtypes=[mx.float32],
    )[0]


_NORM_SOURCE = r"""
    // One threadgroup per row: FP32 sum of squares, then one rounding to the output.
    constexpr int PER = (H + THREADS - 1) / THREADS;
    constexpr int SIMDS = THREADS / 32;
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    threadgroup float partial_sums[SIMDS];
    threadgroup float total;
    float values[PER];
    float sum = 0.0f;
    for (int k = 0; k < PER; ++k) {
        const uint c = tid + k * THREADS;
        values[k] = c < H ? float(x[ulong(row) * H + c]) : 0.0f;
        sum += values[k] * values[k];
    }
    sum = simd_sum(sum);
    if (lane == 0) partial_sums[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float value = 0.0f;
        for (int g = 0; g < SIMDS; ++g) value += partial_sums[g];
        total = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = metal::precise::rsqrt(total / float(H) + eps[0]);
    for (int k = 0; k < PER; ++k) {
        const uint c = tid + k * THREADS;
        if (c >= H) continue;
        float value = values[k] * inverse;
        if (WEIGHTED) value *= float(weight[c]);
        output[ulong(row) * H + c] = bfloat(value);
    }
"""


@cache
def _norm_kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_rms_norm",
        input_names=["x", "weight", "eps"],
        output_names=["output"],
        source=_NORM_SOURCE,
    )


def rms_norm(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    """FP32 RMS normalization times an optional FP32 weight, rounded once to BF16."""
    width = x.shape[-1]
    rows = x.size // width
    threads = 256 if width >= 256 else max(32, -(-width // 32) * 32)
    return _norm_kernel()(
        inputs=[
            x,
            weight if weight is not None else constant((), "float32"),
            constant((float(eps),) * 8, "float32"),
        ],
        template=[("H", width), ("THREADS", threads), ("WEIGHTED", weight is not None)],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


_GEMV_SOURCE = r"""
    // FP32 output = x[ROWS, K] . W[N, K]^T with BF16 W, FP32 accumulation.
    // Each simdgroup owns OUTPUTS consecutive weight rows; every batch row
    // reuses each loaded weight vector.
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint first = (threadgroup_position_in_grid.x * SIMDS + simd) * OUTPUTS;
    const int rows = geometry[0];
    float acc[OUTPUTS][MAX_ROWS];
    for (int o = 0; o < OUTPUTS; ++o)
        for (int r = 0; r < MAX_ROWS; ++r) acc[o][r] = 0.0f;
    for (uint k = lane * 8; k < K; k += 32 * 8) {
        float xs[MAX_ROWS][8];
        for (int r = 0; r < MAX_ROWS; ++r)
            for (int i = 0; i < 8; ++i)
                xs[r][i] = r < rows ? float(x[ulong(r) * K + k + i]) : 0.0f;
        for (int o = 0; o < OUTPUTS; ++o) {
            const uint n = first + o;
            if (n >= N) break;
            const device bfloat* w = weight + ulong(n) * K + k;
            float ws[8];
            for (int i = 0; i < 8; ++i) ws[i] = float(w[i]);
            for (int r = 0; r < MAX_ROWS; ++r) {
                float dot = 0.0f;
                for (int i = 0; i < 8; ++i) dot = metal::fma(xs[r][i], ws[i], dot);
                acc[o][r] += dot;
            }
        }
    }
    for (int o = 0; o < OUTPUTS; ++o) {
        const uint n = first + o;
        for (int r = 0; r < MAX_ROWS; ++r) {
            const float total = simd_sum(acc[o][r]);
            if (lane == 0 && n < N && r < rows) output[ulong(r) * N + n] = total;
        }
    }
"""


@cache
def _gemv_kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_bf16_gemv_f32",
        input_names=["x", "weight", "geometry"],
        output_names=["output"],
        source=_GEMV_SOURCE,
    )


# Rows per GEMV launch: up to four rows reuse each weight load at full bandwidth
# (2.2 ms for the 201K-row vocabulary on an M4 Max); eight rows in one launch
# run slower than two launches of four.
GEMV_ROWS = 4


def gemv_f32(x: mx.array, weight: mx.array) -> mx.array:
    """FP32 ``x @ weight.T`` for BF16 rows; ``weight`` is BF16 ``[N, K]``.

    Reads the BF16 weights at full bandwidth instead of casting them to FP32
    (or keeping an FP32 copy). Each launch serves at most four rows and is
    specialized by its row count; more rows, such as prompt scores, run in
    groups of four.
    """
    rows, width = x.shape
    if width % 256:
        return x.astype(mx.float32) @ weight.astype(mx.float32).T
    if rows > GEMV_ROWS:
        return mx.concatenate(
            [
                gemv_f32(x[start : start + GEMV_ROWS], weight)
                for start in range(0, rows, GEMV_ROWS)
            ]
        )
    outputs_per_simd, simds = 4, 8
    count = weight.shape[0]
    groups = -(-count // (outputs_per_simd * simds))
    return _gemv_kernel()(
        inputs=[x.astype(mx.bfloat16), weight, constant((rows,), "int32")],
        template=[
            ("K", width),
            ("N", count),
            ("OUTPUTS", outputs_per_simd),
            ("SIMDS", simds),
            ("MAX_ROWS", rows),
        ],
        grid=(groups * 32 * simds, 1, 1),
        threadgroup=(32 * simds, 1, 1),
        output_shapes=[(rows, count)],
        output_dtypes=[mx.float32],
    )[0]


__all__ = [
    "constant",
    "gemv_f32",
    "rms_norm",
    "depth_mix",
    "expert_activation",
    "expert_combine",
    "route_experts",
    "short_convolution",
]
