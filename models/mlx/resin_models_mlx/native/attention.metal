// Artifact binding only: canonical _HEADER and _SOURCE remain unchanged.
// MLX helpers retain their upstream MIT license, documented by repository NOTICE.
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
using namespace metal;
using T = bfloat16_t;

#include "attention_kernel.h"

kernel void resin_attention_partition_bf16(
    device const bfloat16_t* queries [[buffer(0)]],
    device const bfloat16_t* current_keys [[buffer(1)]],
    device const bfloat16_t* current_values [[buffer(2)]],
    device const bfloat16_t* units [[buffer(3)]],
    device const int* pages [[buffer(4)]],
    device const int* lengths [[buffer(5)]],
    constant int* geometry [[buffer(6)]],
    constant float* scale [[buffer(7)]],
    device float* partial [[buffer(8)]],
    device float* maxima [[buffer(9)]],
    device float* sums [[buffer(10)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
constexpr int D = 256;
constexpr int HEADS = 8;
constexpr int KV_HEADS = 2;
constexpr int GROUP = 2;
constexpr int HEAD_GROUPS = 2;
constexpr int PAGE = 128;
constexpr int PART = 512;
constexpr int SIMDGROUPS = 4;
constexpr int PREFETCH = 1;

#define RESIN_ATTENTION_BODY
#include "attention_kernel.h"
#undef RESIN_ATTENTION_BODY

}
