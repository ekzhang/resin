"""Inkling 2 paged attention with query-relative bias and a local or global window."""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from resin_models_mlx.metal import constant
from resin_models_mlx.paged_attention import _merge, _paged_partitions

_PREFILL_HEADER = """
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

template <int D, int KV_WIDTH, int PAGE, int BK, int SLICE, typename K, typename V,
          typename Units, typename Pages>
inline void load_prefill_kv(
    K k, V v, Units units, Pages pages, threadgroup float* ks, threadgroup float* vs,
    int tid, int key_tile, int last_key, int context, int first_block, int layer_offset,
    int kv_head, int d0) {
    constexpr int STRIDE = SLICE + 4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A tile can span two retained pages or the transition to the current chunk.
    for (int j = tid * 4; j < BK * SLICE; j += 128 * 4) {
        const int r = j / SLICE;
        const int channel = j % SLICE;
        const int key_pos = key_tile + r;
        float4 k4(0.f), v4(0.f);
        const bool cached = key_pos < context;
        if (key_pos < last_key) {
            size_t idx;
            if (cached) {
                const size_t unit = size_t(pages[key_pos / PAGE - first_block] + layer_offset);
                idx = (unit * 2 * PAGE + key_pos % PAGE) * KV_WIDTH;
            } else {
                idx = size_t(key_pos - context) * KV_WIDTH;
            }
            idx += kv_head * D + d0 + channel;
            #pragma clang loop unroll(full)
            for (int c = 0; c < 4; ++c) {
                if (D % SLICE == 0 || d0 + channel + c < D) {
                    k4[c] = cached ? float(units[idx + c]) : float(k[idx + c]);
                    v4[c] = cached ? float(units[idx + PAGE * KV_WIDTH + c]) : float(v[idx + c]);
                }
            }
        }
        *(threadgroup float4*)(ks + r * STRIDE + channel) = k4;
        *(threadgroup float4*)(vs + r * STRIDE + channel) = v4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

_PREFILL = """
// Each of four simdgroups handles eight queries for a common GQA head.
constexpr int BQ = 32;
// Wider heads stream the channel dimension too, keeping threadgroup memory bounded.
constexpr int SLICES = (D + 255) / 256;
constexpr int SLICE = ((D + 8 * SLICES - 1) / (8 * SLICES)) * 8;
constexpr int TILED_D = SLICES * SLICE;
constexpr int STRIDE = SLICE + 4;
constexpr int KV_WIDTH = KV_HEADS * D;
const int lane = thread_index_in_simdgroup;
const int group = simdgroup_index_in_threadgroup;
const int tid = thread_position_in_threadgroup.x;
const int head = threadgroup_position_in_grid.y;
const int kv_head = head / (HEADS / KV_HEADS);
const int q_tile = threadgroup_position_in_grid.x * BQ;
const int tokens = geometry[0];
const int context = meta[0];
const int first_block = meta[2];
const int layer_offset = geometry[1];
const int window = geometry[2];
const int extent = geometry[3];
const int total = context + tokens;

// Fragment coordinates match MLX Steel BaseMMAFrag<float, 8, 8>.
const int quad = lane / 4;
const int row = (quad & 4) + ((lane / 2) % 4);
const int col = (quad & 2) * 2 + (lane % 2) * 2;
const int qi = q_tile + group * 8 + row;
const int pos = context + qi;
const size_t query_idx = size_t(min(qi, tokens - 1)) * HEADS + head;
simdgroup_float8x8 qm[TILED_D / 8], om[TILED_D / 8];
#pragma clang loop unroll(full)
for (int d = 0; d < TILED_D / 8; ++d) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < 2; ++j) {
        const int c = d * 8 + col + j;
        qm[d].thread_elements()[j] = (TILED_D == D || c < D) ? float(q[query_idx * D + c]) : 0.f;
    }
    om[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
}

alignas(16) threadgroup float ks[BK * STRIDE];
alignas(16) threadgroup float vs[BK * STRIDE];
// A fully masked first tile must contribute zero to the online softmax.
float peak = -FLT_MAX;
float denom = 0.f;
const int first_key = max(first_block * PAGE, window ? context + q_tile - window + 1 : 0);
const int last_key = min(total, context + q_tile + BQ);
for (int key_tile = first_key; key_tile < last_key; key_tile += BK) {
    simdgroup_float8x8 scores[BK / 8];
    #pragma clang loop unroll(full)
    for (int b = 0; b < BK / 8; ++b)
        scores[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (int d0 = 0; d0 < TILED_D; d0 += SLICE) {
        load_prefill_kv<D, KV_WIDTH, PAGE, BK, SLICE>(
            k, v, units, pages, ks, vs, tid, key_tile, last_key, context,
            first_block, layer_offset, kv_head, d0);
        #pragma clang loop unroll(full)
        for (int b = 0; b < BK / 8; ++b) {
            #pragma clang loop unroll(full)
            for (int d = 0; d < SLICE / 8; ++d) {
                simdgroup_float8x8 km;
                simdgroup_load(km, ks + b * 8 * STRIDE + d * 8, STRIDE, ulong2(0), true);
                simdgroup_multiply_accumulate(scores[b], qm[d0 / 8 + d], km, scores[b]);
            }
        }
    }
    // Each lane holds two scores. Four lanes share a query row.
    float new_peak = peak;
    #pragma clang loop unroll(full)
    for (int b = 0; b < BK / 8; ++b) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 2; ++j) {
            const int kp = key_tile + b * 8 + col + j;
            const int distance = pos - kp;
            float s = -INFINITY;
            if (kp < last_key && distance >= 0 && (!window || distance < window)) {
                s = scores[b].thread_elements()[j] * scale[0];
                if (distance < extent) s += float(bias[query_idx * extent + distance]);
            }
            scores[b].thread_elements()[j] = s;
            new_peak = max(new_peak, s);
        }
    }
    new_peak = max(new_peak, simd_shuffle_xor(new_peak, 1));
    new_peak = max(new_peak, simd_shuffle_xor(new_peak, 8));
    const float correction = fast::exp(peak - new_peak);
    float tile_sum = 0.f;
    #pragma clang loop unroll(full)
    for (int b = 0; b < BK / 8; ++b) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 2; ++j) {
            const float weight = fast::exp(scores[b].thread_elements()[j] - new_peak);
            scores[b].thread_elements()[j] = weight;
            tile_sum += weight;
        }
    }
    tile_sum += simd_shuffle_xor(tile_sum, 1);
    tile_sum += simd_shuffle_xor(tile_sum, 8);
    denom = denom * correction + tile_sum;
    peak = new_peak;
    for (int d0 = 0; d0 < TILED_D; d0 += SLICE) {
        // For a single slice, V is already staged beside K.
        if (SLICES > 1) {
            load_prefill_kv<D, KV_WIDTH, PAGE, BK, SLICE>(
                k, v, units, pages, ks, vs, tid, key_tile, last_key, context,
                first_block, layer_offset, kv_head, d0);
        }
        #pragma clang loop unroll(full)
        for (int d = 0; d < SLICE / 8; ++d) {
            om[d0 / 8 + d].thread_elements()[0] *= correction;
            om[d0 / 8 + d].thread_elements()[1] *= correction;
            #pragma clang loop unroll(full)
            for (int b = 0; b < BK / 8; ++b) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, vs + b * 8 * STRIDE + d * 8, STRIDE);
                simdgroup_multiply_accumulate(om[d0 / 8 + d], scores[b], vm, om[d0 / 8 + d]);
            }
        }
    }
}
if (qi < tokens) {
    if (col == 0) lse[query_idx] = peak + fast::log(denom);
    #pragma clang loop unroll(full)
    for (int d = 0; d < TILED_D / 8; ++d) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 2; ++j) {
            const int c = d * 8 + col + j;
            if (TILED_D == D || c < D) out[query_idx * D + c] = om[d].thread_elements()[j] / denom;
        }
    }
}
"""


@cache
def _prefill_kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_paged_prefill",
        input_names=[
            "q",
            "k",
            "v",
            "bias",
            "units",
            "pages",
            "meta",
            "geometry",
            "scale",
        ],
        output_names=["out", "lse"],
        header=_PREFILL_HEADER,
        source=_PREFILL,
    )


def paged_prefill(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    bias: mx.array,
    units: mx.array,
    pages: mx.array,
    meta: mx.array,
    layer: int,
    scale: float,
    window: int | None,
) -> tuple[mx.array, mx.array]:
    """Attend one row's current chunk to its retained pages and its own K/V.

    ``meta`` and ``pages`` use the decode layout. ``layer`` offsets page-table
    entries in arena units. Queries, keys and values stay in their storage
    dtype until a tile is read. Output and natural-log LSE are both FP32.
    """
    batch, tokens, heads, dim = q.shape
    if batch != 1:
        raise ValueError("Inkling 2 prefill processes one row at a time")
    args = (tokens, layer, window or 0, bias.shape[-1])
    out, lse = _prefill_kernel()(
        inputs=[
            q,
            k,
            v,
            bias,
            units,
            pages,
            meta,
            constant(args, "int32"),
            constant((float(scale),), "float32"),
        ],
        template=[
            ("D", dim),
            ("HEADS", heads),
            ("KV_HEADS", k.shape[2]),
            ("BK", 16 if dim <= 128 else 8),
            ("PAGE", units.shape[2]),
        ],
        grid=((tokens + 31) // 32 * 128, heads, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[q.shape, q.shape[:-1]],
        output_dtypes=[mx.float32, mx.float32],
    )
    return out, lse


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
    partials = _paged_partitions(
        queries,
        current_keys,
        current_values,
        units,
        pages,
        meta,
        layer,
        scale,
        bias=bias,
        longest=longest,
    )
    return _merge(*partials, return_lse=True)


__all__ = ["decode_attention", "paged_prefill"]
