"""Inkling 2 decode attention read in place from paged MLX slabs.

Flash decoding as in ``resin_models_mlx/paged_attention.py`` (one threadgroup
per partition of positions for one row and the query heads of one KV head,
online softmax, log-sum-exp merge), extended with Inkling's semantics:

* a per-row lower bound, so local visits see only their last ``window`` keys;
* page tables whose first entry is the row's first retained block;
* a relative bias ``bias[row, head, distance]`` added within its extent.

Positions, bounds and page-table widths are runtime data: new context lengths
or batch sizes do not compile new kernels. The merged output and log-sum-exp
stay FP32 for Inkling's sink and gate.
"""

from __future__ import annotations

from functools import cache, partial

import mlx.core as mx

PARTITION_TOKENS = 256
SIMDGROUPS = 4
# MLX 0.32.4 binds arrays under this size as constants, a different kernel ABI;
# padding keeps one compiled variant for every batch size and context.
_MIN_DEVICE_ELEMENTS = 8

_HEADER = r"""
template <int GROUP, int LANE_DIMS>
inline void resin_attend(
    thread const float (&q)[GROUP][LANE_DIMS],
    thread const float (&k)[LANE_DIMS],
    thread const float (&v)[LANE_DIMS],
    thread const float (&bias)[GROUP],
    thread float (&m)[GROUP],
    thread float (&l)[GROUP],
    thread float (&acc)[GROUP][LANE_DIMS]) {
    for (int h = 0; h < GROUP; ++h) {
        float dot = 0.0f;
        for (int i = 0; i < LANE_DIMS; ++i) dot += q[h][i] * k[i];
        const float score = simd_sum(dot) + bias[h];
        const float peak = max(m[h], score);
        const float correction = fast::exp(m[h] - peak);
        const float weight = fast::exp(score - peak);
        l[h] = l[h] * correction + weight;
        for (int i = 0; i < LANE_DIMS; ++i) {
            acc[h][i] = acc[h][i] * correction + weight * v[i];
        }
        m[h] = peak;
    }
}
"""

_SOURCE = r"""
    constexpr int LANE_DIMS = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const int PAGES = geometry[0];
    const int PARTITIONS = geometry[1];
    const int layer = geometry[2];
    const int EXTENT = geometry[3];

    const int work = threadgroup_position_in_grid.z;
    const int partition = work % PARTITIONS;
    const int kv_head = (work / PARTITIONS) % KV_HEADS;
    const int row = work / (PARTITIONS * KV_HEADS);
    const int first_head = kv_head * GROUP;

    // Absolute positions: history [first, context) then the step's own key.
    const int context = meta[row * 3 + 0];
    const int first = meta[row * 3 + 1];
    const int start_block = meta[row * 3 + 2];
    const int total = context + 1;
    const int start = first + partition * PART;
    const int stop = min(start + PART, total);

    float q[GROUP][LANE_DIMS];
    for (int h = 0; h < GROUP; ++h) {
        for (int i = 0; i < LANE_DIMS; ++i) {
            q[h][i] = float(queries[((row * HEADS) + first_head + h) * D
                                    + lane * LANE_DIMS + i]) * scale[0];
        }
    }
    float m[GROUP];
    float l[GROUP];
    float acc[GROUP][LANE_DIMS];
    for (int h = 0; h < GROUP; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        for (int i = 0; i < LANE_DIMS; ++i) acc[h][i] = 0.0f;
    }

    const int width = KV_HEADS * D;
    const int run = PART / SIMDGROUPS;
    const int lo = start + int(group) * run;
    const int hi = min(lo + run, stop);
    const int cached = min(hi, context);
    float bias_values[GROUP];
    int p = lo;
    while (p < cached) {
        const int page_end = min(cached, (p / PAGE + 1) * PAGE);
        const long unit = long(pages[row * PAGES + (p / PAGE - start_block)]) + layer;
        const device T* key = units + (unit * 2 * PAGE + p % PAGE) * width
                              + kv_head * D + lane * LANE_DIMS;
        const device T* value = key + long(PAGE) * width;
        for (; p < page_end; ++p) {
            float k[LANE_DIMS];
            float v[LANE_DIMS];
            for (int i = 0; i < LANE_DIMS; ++i) {
                k[i] = float(key[i]);
                v[i] = float(value[i]);
            }
            const int distance = context - p;
            for (int h = 0; h < GROUP; ++h) {
                bias_values[h] = distance < EXTENT
                    ? float(bias[((row * HEADS) + first_head + h) * EXTENT + distance])
                    : 0.0f;
            }
            resin_attend(q, k, v, bias_values, m, l, acc);
            key += width;
            value += width;
        }
    }
    if (lo <= context && context < hi) {
        const int base = (row * KV_HEADS + kv_head) * D + lane * LANE_DIMS;
        float k[LANE_DIMS];
        float v[LANE_DIMS];
        for (int i = 0; i < LANE_DIMS; ++i) {
            k[i] = float(current_keys[base + i]);
            v[i] = float(current_values[base + i]);
        }
        for (int h = 0; h < GROUP; ++h) {
            bias_values[h] = float(bias[((row * HEADS) + first_head + h) * EXTENT]);
        }
        resin_attend(q, k, v, bias_values, m, l, acc);
    }

    threadgroup float shared_m[SIMDGROUPS][GROUP];
    threadgroup float shared_l[SIMDGROUPS][GROUP];
    threadgroup float shared_acc[SIMDGROUPS][GROUP][D];
    for (int h = 0; h < GROUP; ++h) {
        if (lane == 0) {
            shared_m[group][h] = m[h];
            shared_l[group][h] = l[h];
        }
        for (int i = 0; i < LANE_DIMS; ++i) {
            shared_acc[group][h][lane * LANE_DIMS + i] = acc[h][i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int index = tid; index < GROUP * D; index += SIMDGROUPS * 32) {
        const int h = index / D;
        const int d = index % D;
        float peak = -INFINITY;
        for (int s = 0; s < SIMDGROUPS; ++s) peak = max(peak, shared_m[s][h]);
        float sum = 0.0f;
        float value = 0.0f;
        if (peak != -INFINITY) {
            for (int s = 0; s < SIMDGROUPS; ++s) {
                const float scaled = fast::exp(shared_m[s][h] - peak);
                sum += shared_l[s][h] * scaled;
                value += shared_acc[s][h][d] * scaled;
            }
        }
        const int out = ((row * HEADS) + first_head + h) * PARTITIONS + partition;
        partial[out * D + d] = value;
        if (d == 0) {
            maxima[out] = peak;
            sums[out] = sum;
        }
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_decode_attention",
        input_names=[
            "queries",
            "current_keys",
            "current_values",
            "units",
            "pages",
            "meta",
            "bias",
            "geometry",
            "scale",
        ],
        output_names=["partial", "maxima", "sums"],
        source=_SOURCE,
        header=_HEADER,
    )


@partial(mx.compile, shapeless=True)
def _merge(outputs: mx.array, maxima: mx.array, sums: mx.array):
    peak = maxima.max(axis=-1, keepdims=True)
    weights = mx.where(maxima == -mx.inf, 0.0, mx.exp(maxima - peak))
    numerator = (outputs * weights[..., None]).sum(axis=2)
    denominator = (sums * weights).sum(axis=-1)
    return numerator / denominator[..., None], peak[..., 0] + mx.log(denominator)


def decode_attention(
    queries: mx.array,
    current_keys: mx.array,
    current_values: mx.array,
    units: mx.array,
    pages: mx.array,
    meta: mx.array,
    bias: mx.array,
    layer: int,
    scale: float,
    longest: int,
) -> tuple[mx.array, mx.array]:
    """One new token per row over paged history plus itself.

    ``queries`` is ``[rows, heads, D]``; current keys/values ``[rows, KV, D]``;
    ``units`` the arena as ``[units, 2, page, KV * D]``; ``pages[row, j]`` the
    visit-zero unit of the row's ``j``th retained block, offset by ``layer``
    units; ``meta[row] = (context, first visible position, first block)``;
    ``bias`` ``[rows, heads, extent]``. ``longest`` bounds ``context + 1 -
    first`` over rows and sets the partition count. Returns FP32
    ``[rows, heads, D]`` outputs and ``[rows, heads]`` log-sum-exps.
    """
    rows, heads, dim = queries.shape
    kv_heads = current_keys.shape[1]
    page = units.shape[2]
    group = heads // kv_heads
    if dim % 32 or heads % kv_heads:
        raise ValueError("unsupported Inkling 2 decode attention geometry")
    partitions = max(1, -(-longest // PARTITION_TOKENS))
    pages = _pad_rows(pages)
    meta = _pad_rows(meta)
    outputs, maxima, sums = _kernel()(
        inputs=[
            queries,
            current_keys,
            current_values,
            units,
            pages,
            meta,
            bias,
            mx.array(
                [pages.shape[1], partitions, layer, bias.shape[-1], 0, 0, 0, 0],
                dtype=mx.int32,
            ),
            mx.full((_MIN_DEVICE_ELEMENTS,), scale, dtype=mx.float32),
        ],
        template=[
            ("D", dim),
            ("HEADS", heads),
            ("KV_HEADS", kv_heads),
            ("GROUP", group),
            ("PAGE", page),
            ("PART", PARTITION_TOKENS),
            ("SIMDGROUPS", SIMDGROUPS),
            ("T", units.dtype),
        ],
        grid=(32 * SIMDGROUPS, 1, rows * kv_heads * partitions),
        threadgroup=(32 * SIMDGROUPS, 1, 1),
        output_shapes=[
            (rows, heads, partitions, dim),
            (rows, heads, partitions),
            (rows, heads, partitions),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _merge(outputs, maxima, sums)


def _pad_rows(values: mx.array) -> mx.array:
    rows, width = values.shape
    needed = -(-_MIN_DEVICE_ELEMENTS // max(1, width))
    if rows * width >= _MIN_DEVICE_ELEMENTS:
        return values
    padding = mx.zeros((needed - rows, width), dtype=values.dtype)
    return mx.concatenate([values, padding])


__all__ = ["PARTITION_TOKENS", "decode_attention"]
