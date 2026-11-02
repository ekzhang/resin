"""One-token depthwise Conv4, BF16 SiLU and Q/K/V stores in one dispatch.

Preserves MLX's FP32 FMA and BF16 activation roundings. The sigmoid follows
MLX unary_ops.h (Apple Inc., MIT); other geometries use native convolution.
"""

from functools import cache

import mlx.core as mx
import mlx.nn as nn

_HEADER = """
template <typename T>
inline T resin_conv_sigmoid(T value) {
    auto tail = 1 / (1 + metal::precise::exp(metal::abs(value)));
    return value < 0 ? tail : 1 - tail;
}
"""

_SOURCE = """
    const uint index = thread_position_in_grid.x;
    const uint channel = index % CHANNELS;
    const uint row = index / CHANNELS;
    const long previous_base = long(row) * previous_strides[0]
                               + long(channel) * previous_strides[2];
    const long current_base = long(row) * current_strides[0]
                              + long(channel) * current_strides[2];
    const long weight_base = long(channel) * weight_strides[0];
    const T x0 = previous[previous_base];
    const T x1 = previous[previous_base + previous_strides[1]];
    const T x2 = previous[previous_base + 2 * previous_strides[1]];
    const T x3 = current[current_base];

    float value = 0.0f;
    value = metal::fma(float(x0), float(weight[weight_base]), value);
    value = metal::fma(float(x1), float(weight[weight_base + weight_strides[1]]), value);
    value = metal::fma(float(x2), float(weight[weight_base + 2 * weight_strides[1]]), value);
    value = metal::fma(float(x3), float(weight[weight_base + 3 * weight_strides[1]]), value);
    const T rounded = T(value);
    const T sigmoid = resin_conv_sigmoid(rounded);

    const T activated = T(rounded * sigmoid);
    if (channel < KEY_WIDTH)
        query[row * KEY_WIDTH + channel] = activated;
    else if (channel < 2 * KEY_WIDTH)
        key[row * KEY_WIDTH + channel - KEY_WIDTH] = activated;
    else
        value_out[row * VALUE_WIDTH + channel - 2 * KEY_WIDTH] = activated;

    carry[(row * 3) * CHANNELS + channel] = x1;
    carry[(row * 3 + 1) * CHANNELS + channel] = x2;
    carry[(row * 3 + 2) * CHANNELS + channel] = x3;
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="resin_decode_conv4_silu_qkv",
        input_names=["current", "previous", "weight"],
        output_names=["query", "key", "value_out", "carry"],
        source=_SOURCE,
        header=_HEADER,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )


def conv4_silu_qkv(
    current, previous, weight, *, key_heads, key_dim, value_heads, value_dim
):
    """Return Q/K/V and convolution carry; strided inputs remain runtime data."""
    if current.ndim != 3 or previous.ndim != 3 or weight.ndim != 3:
        raise ValueError("depthwise convolution inputs and weights must have rank 3")
    if any(
        type(n) is not int or n < 1
        for n in (key_heads, key_dim, value_heads, value_dim)
    ):
        raise ValueError("Q/K/V dimensions must be positive integers")
    rows, tokens, channels = current.shape
    kernel = weight.shape[1]
    key_width, value_width = key_heads * key_dim, value_heads * value_dim
    if not rows or not tokens or channels != 2 * key_width + value_width:
        raise ValueError("Q/K/V dimensions must cover nonempty convolution channels")
    if (
        kernel < 1
        or weight.shape != (channels, kernel, 1)
        or previous.shape != (rows, kernel - 1, channels)
    ):
        raise ValueError(
            "depthwise weights/carry must match the input channels and kernel"
        )
    if (
        tokens == 1
        and kernel == 4
        and channels >= 8
        and current.dtype == previous.dtype == weight.dtype == mx.bfloat16
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        return tuple(
            _kernel()(
                inputs=[current, previous, weight],
                template=[
                    ("T", current.dtype),
                    ("CHANNELS", channels),
                    ("KEY_WIDTH", key_width),
                    ("VALUE_WIDTH", value_width),
                ],
                grid=(rows * channels, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[
                    (rows, 1, key_heads, key_dim),
                    (rows, 1, key_heads, key_dim),
                    (rows, 1, value_heads, value_dim),
                    previous.shape,
                ],
                output_dtypes=[current.dtype] * 4,
            )
        )
    window = mx.concatenate([previous.astype(current.dtype), current], axis=1)
    carry = window[:, -(kernel - 1) :] if kernel > 1 else window[:, :0]
    query, key, value = mx.split(
        nn.silu(mx.conv1d(window, weight, groups=channels)),
        [key_width, 2 * key_width],
        axis=-1,
    )
    return (
        query.reshape(rows, tokens, key_heads, key_dim),
        key.reshape(rows, tokens, key_heads, key_dim),
        value.reshape(rows, tokens, value_heads, value_dim),
        carry,
    )
