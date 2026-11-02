"""Exact BF16 RMS and rounded residual+RMS with FP32 effective scales.

Reduction adapted from MLX v0.32.4 rms_norm.metal (Apple Inc., 2024, MIT).
The native four-values-per-thread reduction and BF16 boundaries are retained.
"""

from functools import cache

import mlx.core as mx

_HEADER = """
inline long resin_row_offset(uint row, constant const int* shape,
                             constant const long* strides, int ndim, int width) {
    long expected_stride = width;
    bool contiguous = true;
    for (int axis = ndim - 2; axis >= 0; --axis) {
        if (shape[axis] != 1 && strides[axis] != expected_stride)
            contiguous = false;
        expected_stride *= shape[axis];
    }
    if (contiguous) return long(row) * width;
    long offset = 0;
    for (int axis = ndim - 2; axis >= 0 && row; --axis) {
        offset += long(row % shape[axis]) * strides[axis];
        row /= shape[axis];
    }
    return offset;
}
"""

_ROW_SOURCE = r"""
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    const uint axis_size = x_shape[x_ndim - 1];
    const uint gid = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];
    const long base = resin_row_offset(gid, x_shape, x_strides, x_ndim, D);
    const long stride = x_strides[x_ndim - 1];
    $EXTRA_SETUP$
    float acc = 0;
    float thread_x[N_READS];
    if (lid * N_READS + N_READS <= D) {
        for (int i = 0; i < N_READS; i++) {
            thread_x[i] = $LOAD$;
            acc += thread_x[i] * thread_x[i];
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            thread_x[i] = (lid * N_READS + i < D) ? $LOAD$ : 0;
            acc += thread_x[i] * thread_x[i];
        }
    }
    acc = simd_sum(acc);
    float inv_mean;
    if constexpr (D <= 128) {
        // Native's second reduction adds only zeros for a one-SIMD row.
        inv_mean = metal::precise::rsqrt(acc / axis_size + eps);
    } else {
        if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
            acc = simd_sum(local_sums[simd_lane_id]);
            if (simd_lane_id == 0)
                local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv_mean = local_inv_mean[0];
    }
    for (int i = 0; i < N_READS; i++) {
        const uint column = lid * N_READS + i;
        if (column < D) {
            $STORE$
        }
    }
"""


@cache
def _kernel(residual: bool):
    load = "float(x[base + (lid * N_READS + i) * stride])"
    setup = ""
    inputs, outputs = ["x", "scale", "eps"], ["out"]
    store = """
        float normalized = thread_x[i] * inv_mean;
        out[long(gid) * D + column] =
            bfloat16_t(scale[column * scale_strides[0]] * normalized);
    """
    if residual:
        inputs, outputs = ["x", "residual", "scale", "eps"], ["summed", "out"]
        setup = """
            const long residual_base = resin_row_offset(
                gid, residual_shape, residual_strides, residual_ndim, D);
            const long residual_stride = residual_strides[residual_ndim - 1];
        """
        load = """float(bfloat16_t(
            float(x[base + (lid * N_READS + i) * stride]) +
            float(residual[residual_base + (lid * N_READS + i) * residual_stride])))"""
        store = "summed[long(gid) * D + column] = bfloat16_t(thread_x[i]);" + store
    source = (
        _ROW_SOURCE.replace("$EXTRA_SETUP$", setup)
        .replace("$LOAD$", load)
        .replace("$STORE$", store)
    )
    return mx.fast.metal_kernel(
        name="resin_qwen35_" + ("residual" if residual else "mixed") + "_rms",
        input_names=inputs,
        output_names=outputs,
        source=source,
        header=_HEADER,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )


def _supported(x, scale):
    return (
        scale is not None
        and x.dtype == mx.bfloat16
        and scale.dtype == mx.float32
        and x.ndim > 0
        and scale.ndim == 1
        and scale.size == x.shape[-1]
        and 8 <= x.shape[-1] <= 4096
        and x.size > 0
        and x.size // x.shape[-1] <= 0xFFFF_FFFF
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


def _launch(x, inputs, residual=False):
    threads = -(-x.shape[-1] // 128) * 32
    return _kernel(residual)(
        inputs=inputs,
        template=[("D", x.shape[-1])],
        grid=(threads * (x.size // x.shape[-1]), 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape] * (2 if residual else 1),
        output_dtypes=[mx.bfloat16] * (2 if residual else 1),
    )


def mixed_rms_norm(x, scale, eps):
    if not _supported(x, scale):
        return mx.fast.rms_norm(x, scale, eps).astype(x.dtype)
    return _launch(x, [x, scale, eps])[0]


def residual_rms_norm(x, residual, scale, eps):
    if x.dtype == residual.dtype == mx.bfloat16:
        left, right = mx.broadcast_arrays(x, residual)
        if _supported(left, scale):
            return tuple(_launch(left, [left, right, scale, eps], residual=True))
    summed = x + residual
    return summed, mx.fast.rms_norm(summed, scale, eps).astype(summed.dtype)
