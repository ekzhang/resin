"""Decode attention that reads keys and values directly from paged MLX slabs.

The arena stores one 128-token page of one layer as a *unit*:
``units[unit, 0, offset]`` holds keys and ``units[unit, 1, offset]`` values,
each ``kv_heads * head_dim`` wide. ``pages[row, block]`` names the unit holding
logical block ``block`` of the row's history for the current layer.

This is flash decoding: each threadgroup scores one partition of positions for
one row and one group of query heads sharing a KV head, with an online softmax.
The partitions are then merged by their log-sum-exp. The step's own key and
value, not yet written to the arena, are the final position of every row, so
history padding is never read. Context length, page-table width and batch
size are runtime data; only the model geometry selects a compiled kernel.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

PARTITION_TOKENS = 512
SIMDGROUPS = 4
_HEADS_PER_THREADGROUP = 4

_SOURCE = """
    constexpr int LANE_DIMS = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    // Runtime geometry: page-table width, partition count and layer offset.
    const int PAGES = geometry[0];
    const int PARTITIONS = geometry[1];
    const int layer = geometry[2];

    const int work = threadgroup_position_in_grid.z;
    const int partition = work % PARTITIONS;
    const int head_group = (work / PARTITIONS) % HEAD_GROUPS;
    const int kv_head = (work / (PARTITIONS * HEAD_GROUPS)) % KV_HEADS;
    const int row = work / (PARTITIONS * HEAD_GROUPS * KV_HEADS);
    const int first_head = kv_head * (HEADS / KV_HEADS) + head_group * GROUP;

    const int history = lengths[row];
    const int total = history + 1;
    const int start = partition * PART;
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
    for (int p = start + int(group); p < stop; p += SIMDGROUPS) {
        float k[LANE_DIMS];
        float v[LANE_DIMS];
        if (p < history) {
            const long unit = long(pages[row * PAGES + p / PAGE]) + layer;
            const long base = (unit * 2 * PAGE + (p % PAGE)) * width
                              + kv_head * D + lane * LANE_DIMS;
            const long value = base + long(PAGE) * width;
            for (int i = 0; i < LANE_DIMS; ++i) {
                k[i] = float(units[base + i]);
                v[i] = float(units[value + i]);
            }
        } else {
            const int base = (row * KV_HEADS + kv_head) * D + lane * LANE_DIMS;
            for (int i = 0; i < LANE_DIMS; ++i) {
                k[i] = float(current_keys[base + i]);
                v[i] = float(current_values[base + i]);
            }
        }
        for (int h = 0; h < GROUP; ++h) {
            float dot = 0.0f;
            for (int i = 0; i < LANE_DIMS; ++i) dot += q[h][i] * k[i];
            const float score = simd_sum(dot);
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
        name="resin_paged_decode_attention",
        input_names=[
            "queries",
            "current_keys",
            "current_values",
            "units",
            "pages",
            "lengths",
            "geometry",
            "scale",
        ],
        output_names=["partial", "maxima", "sums"],
        source=_SOURCE,
    )


def paged_decode_attention(
    queries: mx.array,
    current_keys: mx.array,
    current_values: mx.array,
    units: mx.array,
    pages: mx.array,
    lengths: mx.array,
    layer: int,
    scale: float,
) -> mx.array:
    """Attend one new token per row over paged history plus itself.

    ``queries`` is ``[rows, heads, head_dim]``; ``current_keys`` and
    ``current_values`` are ``[rows, kv_heads, head_dim]``; ``units`` is the
    arena viewed as ``[units, 2, page, kv_heads * head_dim]``; ``pages`` is
    ``[rows, blocks]`` int32 base units, offset by ``layer``; ``lengths`` is the
    int32 history length of each row. Returns ``[rows, heads, head_dim]``.
    """

    rows, heads, head_dim = queries.shape
    kv_heads = current_keys.shape[1]
    page = units.shape[2]
    group = min(_HEADS_PER_THREADGROUP, heads // kv_heads)
    head_groups = heads // kv_heads // group
    longest = pages.shape[1] * page + 1
    partitions = -(-longest // PARTITION_TOKENS)
    partial, maxima, sums = _kernel()(
        inputs=[
            queries,
            current_keys,
            current_values,
            units,
            pages,
            lengths,
            mx.array([pages.shape[1], partitions, layer], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("D", head_dim),
            ("HEADS", heads),
            ("KV_HEADS", kv_heads),
            ("GROUP", group),
            ("HEAD_GROUPS", head_groups),
            ("PAGE", page),
            ("PART", PARTITION_TOKENS),
            ("SIMDGROUPS", SIMDGROUPS),
        ],
        grid=(32 * SIMDGROUPS, 1, rows * kv_heads * head_groups * partitions),
        threadgroup=(32 * SIMDGROUPS, 1, 1),
        output_shapes=[
            (rows, heads, partitions, head_dim),
            (rows, heads, partitions),
            (rows, heads, partitions),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    peak = maxima.max(axis=-1, keepdims=True)
    weights = mx.where(maxima == -mx.inf, 0.0, mx.exp(maxima - peak))
    numerator = (partial * weights[..., None]).sum(axis=2)
    denominator = (sums * weights).sum(axis=-1)[..., None]
    return (numerator / denominator).astype(queries.dtype)


__all__ = ["PARTITION_TOKENS", "paged_decode_attention"]
