"""Decode attention that reads keys and values directly from paged MLX slabs.

The arena stores one 128-token page of one layer as a *unit*:
``units[unit, 0, offset]`` holds keys and ``units[unit, 1, offset]`` values,
each ``kv_heads * head_dim`` wide. The arena is split into parts of
``part_units`` units; ``pages[row, block]`` names the unit holding logical
block ``block`` of the row's history for the current layer, as
``part * part_units + unit``.

This is flash decoding: each threadgroup scores one partition of positions for
one row and one group of query heads sharing a KV head, with an online softmax.
The partitions are then merged by their log-sum-exp. The step's own key and
value, not yet written to the arena, are the final position of every row, so
history padding is never read. Context length, page-table width and batch
size are runtime data, except for a few small-batch kernel variants that each
compile once. The model geometry and the arena's part count also select a kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache, partial

import mlx.core as mx

PARTITION_TOKENS = 512
# Below this many rows, half-size partitions keep more GPU cores busy.
SMALL_BATCH_ROWS = 4
SIMDGROUPS = 4
PREFETCH = 4
_HEADS_PER_THREADGROUP = 4
# Metal guarantees 32 KiB of threadgroup memory.
_THREADGROUP_BYTES = 32 * 1024

_HEADER = """
template <int GROUP, int LANE_DIMS>
inline void attend(
    thread const float (&q)[GROUP][LANE_DIMS],
    thread const float (&k)[LANE_DIMS],
    thread const float (&v)[LANE_DIMS],
    thread float (&m)[GROUP],
    thread float (&l)[GROUP],
    thread float (&acc)[GROUP][LANE_DIMS]) {
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
"""

_SOURCE = """
    constexpr int LANE_DIMS = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    // Runtime geometry: page-table width, partition count, layer offset and
    // units per part.
    const int PAGES = geometry[0];
    const int PARTITIONS = geometry[1];
    const int layer = geometry[2];
    const int part_units = geometry[3];

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

    // Each simdgroup walks its own contiguous run of positions, page by page,
    // loading PREFETCH positions ahead so their memory latency overlaps.
    const int width = KV_HEADS * D;
    const int run = PART / SIMDGROUPS;
    const int first = start + int(group) * run;
    const int last = min(first + run, stop);
    const int cached = min(last, history);
    int p = first;
    while (p < cached) {
        const int page_end = min(cached, (p / PAGE + 1) * PAGE);
        const int entry = pages[row * PAGES + p / PAGE] + layer;
        const device T* units = units_0;
        switch (entry / part_units) {
PARTS        }
        const long unit = entry % part_units;
        const device T* key = units + (unit * 2 * PAGE + p % PAGE) * width
                              + kv_head * D + lane * LANE_DIMS;
        const device T* value = key + long(PAGE) * width;
        for (; p + PREFETCH <= page_end; p += PREFETCH) {
            float k[PREFETCH][LANE_DIMS];
            float v[PREFETCH][LANE_DIMS];
            for (int t = 0; t < PREFETCH; ++t) {
                for (int i = 0; i < LANE_DIMS; ++i) k[t][i] = float(key[t * width + i]);
            }
            for (int t = 0; t < PREFETCH; ++t) {
                for (int i = 0; i < LANE_DIMS; ++i) v[t][i] = float(value[t * width + i]);
            }
            for (int t = 0; t < PREFETCH; ++t) attend(q, k[t], v[t], m, l, acc);
            key += PREFETCH * width;
            value += PREFETCH * width;
        }
        for (; p < page_end; ++p) {
            float k[LANE_DIMS];
            float v[LANE_DIMS];
            for (int i = 0; i < LANE_DIMS; ++i) {
                k[i] = float(key[i]);
                v[i] = float(value[i]);
            }
            attend(q, k, v, m, l, acc);
            key += width;
            value += width;
        }
    }
    // The step's own key and value follow the history.
    if (first <= history && history < last) {
        const int base = (row * KV_HEADS + kv_head) * D + lane * LANE_DIMS;
        float k[LANE_DIMS];
        float v[LANE_DIMS];
        for (int i = 0; i < LANE_DIMS; ++i) {
            k[i] = float(current_keys[base + i]);
            v[i] = float(current_values[base + i]);
        }
        attend(q, k, v, m, l, acc);
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
def _kernel(parts: int):
    cases = "".join(
        f"            case {part}: units = units_{part}; break;\n"
        for part in range(1, parts)
    )
    return mx.fast.metal_kernel(
        name=f"resin_paged_decode_attention_{parts}",
        input_names=[
            "queries",
            "current_keys",
            "current_values",
            *(f"units_{part}" for part in range(parts)),
            "pages",
            "lengths",
            "geometry",
            "scale",
        ],
        output_names=["partial", "maxima", "sums"],
        source=_SOURCE.replace("PARTS", cases),
        header=_HEADER,
    )


def paged_decode_attention(
    queries: mx.array,
    current_keys: mx.array,
    current_values: mx.array,
    parts: Sequence[mx.array],
    part_units: int,
    pages: mx.array,
    lengths: mx.array,
    layer: int,
    scale: float,
) -> mx.array:
    """Attend one new token per row over paged history plus itself.

    ``queries`` is ``[rows, heads, head_dim]``; ``current_keys`` and
    ``current_values`` are ``[rows, kv_heads, head_dim]``; ``parts`` are arena
    parts viewed as ``[units, 2, page, kv_heads * head_dim]``; ``pages`` is
    ``[rows, blocks]`` int32 base units numbered ``part * part_units + unit``,
    offset by ``layer``; ``lengths`` is the int32 history length of each row.
    Returns ``[rows, heads, head_dim]``.
    """

    rows, heads, head_dim = queries.shape
    kv_heads = current_keys.shape[1]
    page = parts[0].shape[2]
    if heads % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    if head_dim % 32:
        raise ValueError("paged decode attention needs head_dim divisible by 32")
    # Each threadgroup serves an equal share of the query heads of a KV head.
    shared = heads // kv_heads
    # The cross-simdgroup reduction stores float maxima, sums and outputs.
    fits = [
        g
        for g in range(1, _HEADS_PER_THREADGROUP + 1)
        if shared % g == 0 and SIMDGROUPS * g * (head_dim + 2) * 4 <= _THREADGROUP_BYTES
    ]
    if not fits:
        raise ValueError(f"head_dim {head_dim} exceeds paged decode threadgroup memory")
    group = max(fits)
    head_groups = shared // group
    longest = pages.shape[1] * page + 1
    part = PARTITION_TOKENS // (2 if rows < SMALL_BATCH_ROWS else 1)
    partitions = -(-longest // part)
    outputs, maxima, sums = _kernel(len(parts))(
        inputs=[
            queries,
            current_keys,
            current_values,
            *parts,
            pages,
            lengths,
            mx.array([pages.shape[1], partitions, layer, part_units], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("D", head_dim),
            ("HEADS", heads),
            ("KV_HEADS", kv_heads),
            ("GROUP", group),
            ("HEAD_GROUPS", head_groups),
            ("PAGE", page),
            ("PART", part),
            ("SIMDGROUPS", SIMDGROUPS),
            ("PREFETCH", PREFETCH),
            ("T", parts[0].dtype),
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
    return _merge(outputs, maxima, sums).astype(queries.dtype)


@partial(mx.compile, shapeless=True)
def _merge(outputs: mx.array, maxima: mx.array, sums: mx.array) -> mx.array:
    """Combine partitions by their log-sum-exp; empty partitions weigh zero."""

    peak = maxima.max(axis=-1, keepdims=True)
    weights = mx.where(maxima == -mx.inf, 0.0, mx.exp(maxima - peak))
    numerator = (outputs * weights[..., None]).sum(axis=2)
    denominator = (sums * weights).sum(axis=-1)[..., None]
    return numerator / denominator


__all__ = ["PARTITION_TOKENS", "paged_decode_attention"]
