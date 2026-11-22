"""Inkling 2 expert activation quantization, emitted as exact dequantized values.

The checkpoint requires NVFP4 or MXFP8 activations before each expert GEMM.
This follows the CUDA quantizer (adapted from SGLang e62ac627, Apache-2.0):
NVFP4 picks the four- or six-level block scale with lower pairwise-summed
reconstruction error, under a per-row outer scale; MXFP8 uses a power-of-two
block scale. A code times its block scale is exact in BF16, so the kernel
writes those products and leaves the per-row outer scale to the GEMM output.
"""

from functools import cache

import mlx.core as mx

FLOAT_MAX = "3.4028234663852886e38f"

_HEADER = r"""
// Round a finite magnitude to FP8 E4M3FN with round-to-nearest-even, saturating
// at 448. Subnormals step by 2^-9.
inline float resin_e4m3(float x) {
    float a = metal::abs(x);
    float q;
    if (a >= 448.0f) {
        q = 448.0f;
    } else if (a < 0.015625f) {
        q = metal::rint(a * 512.0f) * 0.001953125f;
    } else {
        int e;
        metal::frexp(a, e);  // a = m * 2^e, m in [0.5, 1)
        float step = metal::ldexp(1.0f, e - 4);
        q = metal::min(metal::rint(a / step) * step, 448.0f);
    }
    return metal::copysign(q, x);
}

// Round to FP4 E2M1 with round-to-nearest-even, saturating at 6.
inline float resin_e2m1(float x) {
    float a = metal::abs(x);
    float q;
    if (a < 2.0f) {
        q = metal::rint(a * 2.0f) * 0.5f;
    } else if (a < 4.0f) {
        q = metal::rint(a);
    } else {
        q = metal::min(metal::rint(a * 0.5f) * 2.0f, 6.0f);
    }
    return metal::copysign(q, x);
}
"""

_NVFP4_SOURCE = r"""
    #pragma METAL fp contract(off)
    const uint row = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    threadgroup float maxima[32];
    threadgroup float row_amax[1];
    constexpr int BLOCKS = K / 16;
    const bool active = int(lid) < BLOCKS;
    const long base = long(row) * K + long(lid) * 16;
    float v[16];
    float block_max = 0.0f;
    for (int i = 0; i < 16; ++i) {
        v[i] = active ? float(x[base + i]) : 0.0f;
        block_max = metal::max(block_max, metal::abs(v[i]));
    }
    float m = simd_max(block_max);
    if (simd == 0) maxima[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) maxima[simd] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float total = simd_max(maxima[lane]);
        if (lane == 0) row_amax[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float amax = row_amax[0];
    if (lid == 0) row_amax_out[row] = amax;
    if (!active) return;
    const float encoder = metal::min(metal::precise::divide(1536.0f, amax), FLOAT_MAX);
    const float decoder = metal::precise::divide(1.0f, encoder);
    const float outer_scale = amax * (1.0f / 1536.0f);
    float scale6 = resin_e4m3(metal::min(block_max * (encoder * (1.0f / 6.0f)), 448.0f));
    float scale4 = resin_e4m3(metal::min(block_max * (encoder * 0.25f), 448.0f));
    float inverse6 = metal::min(metal::precise::divide(1.0f, scale6 * decoder), FLOAT_MAX);
    float inverse4 = metal::min(metal::precise::divide(1.0f, scale4 * decoder), FLOAT_MAX);
    float r6[16], r4[16], e6[16], e4[16];
    for (int i = 0; i < 16; ++i) {
        r6[i] = resin_e2m1(v[i] * inverse6) * scale6;
        r4[i] = resin_e2m1(v[i] * inverse4) * scale4;
        e6[i] = metal::abs(r6[i] * outer_scale - v[i]);
        e4[i] = metal::abs(r4[i] * outer_scale - v[i]);
    }
    for (int width = 8; width >= 1; width >>= 1) {
        for (int i = 0; i < width; ++i) {
            e6[i] = e6[2 * i] + e6[2 * i + 1];
            e4[i] = e4[2 * i] + e4[2 * i + 1];
        }
    }
    const bool choose4 = e4[0] < e6[0];
    for (int i = 0; i < 16; ++i) {
        y[base + i] = choose4 ? r4[i] : r6[i];
    }
"""

_MXFP8_SOURCE = r"""
    #pragma METAL fp contract(off)
    const uint row = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    constexpr int BLOCKS = K / 32;
    if (int(lid) >= BLOCKS) return;
    const long base = long(row) * K + long(lid) * 32;
    float v[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        v[i] = float(x[base + i]);
        amax = metal::max(amax, metal::abs(v[i]));
    }
    const float raw = amax * (1.0f / 448.0f);
    int exponent = int(((as_type<uint>(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23);
    exponent = raw <= 5.877471754111438e-39f ? 0 : metal::min(exponent, 254);
    // Blocks with amax <= 448 * 2^-127 (in practice all-zero) quantize to
    // zero; CUDA keeps 2^-127-scaled denormals there.
    const float scale = as_type<float>(uint(exponent) << 23);
    const float inverse = exponent == 0 ? 0.0f : metal::precise::divide(1.0f, scale);
    for (int i = 0; i < 32; ++i) {
        float q = metal::clamp(v[i] * inverse, -448.0f, 448.0f);
        y[base + i] = resin_e4m3(q) * scale;
    }
"""


@cache
def _nvfp4_kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_nvfp4_activation",
        input_names=["x"],
        output_names=["y", "row_amax_out"],
        header=_HEADER,
        source=_NVFP4_SOURCE.replace("FLOAT_MAX", FLOAT_MAX),
    )


@cache
def _mxfp8_kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_mxfp8_activation",
        input_names=["x"],
        output_names=["y"],
        header=_HEADER,
        source=_MXFP8_SOURCE,
    )


def _threads(blocks: int) -> int:
    return max(32, (blocks + 31) // 32 * 32)


def quantize_activations(x: mx.array, mode: str) -> tuple[mx.array, mx.array | None]:
    """Return FP32 dequantized activations without the outer scale, and row amax.

    ``x`` is a contiguous BF16 ``[rows, K]`` matrix. NVFP4 values exclude the
    per-row outer scale ``amax / 1536``; the returned ``amax`` lets the GEMM
    apply it with the weight's scale. MXFP8 has no outer scale and returns ``None``.
    """

    rows, width = x.shape
    if mode == "nvfp4":
        if width % 16 or width // 16 > 1024:
            raise ValueError("NVFP4 activations need a width divisible by 16")
        threads = _threads(width // 16)
        y, amax = _nvfp4_kernel()(
            inputs=[x],
            template=[("K", width)],
            grid=(rows * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(rows, width), (rows,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        return y, amax
    if mode == "mxfp8":
        if width % 32 or width // 32 > 1024:
            raise ValueError("MXFP8 activations need a width divisible by 32")
        threads = _threads(width // 32)
        (y,) = _mxfp8_kernel()(
            inputs=[x],
            template=[("K", width)],
            grid=(rows * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(rows, width)],
            output_dtypes=[mx.float32],
        )
        return y, None
    raise ValueError(f"unsupported Inkling 2 activation precision {mode!r}")


__all__ = ["quantize_activations"]
