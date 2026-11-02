"""Copy strided per-layer K/V into the arena's token/layer/KV write order."""

from functools import cache

import mlx.core as mx


@cache
def _kernel(layers):
    names = [name for i in range(layers) for name in (f"k{i}", f"v{i}")]
    source = """
        const uint flat = thread_position_in_grid.x;
        const uint kind = thread_position_in_grid.y;
        const uint layer = thread_position_in_grid.z;
        const uint width = geometry[0], dim = geometry[1];
        const uint token = flat / width, column = flat % width;
        const ulong output = ((ulong(token) * LAYERS + layer) * 2 + kind) * width + column;
    """
    for i, name in enumerate(names):
        source += f"""
        if (layer == {i // 2} && kind == {i % 2})
            packed[output] = {name}[long(token) * {name}_strides[0]
                + long(column / dim) * {name}_strides[1]
                + long(column % dim) * {name}_strides[2]];
        """
    return mx.fast.metal_kernel(
        name=f"resin_pack_kv_layers_{layers}",
        input_names=names + ["geometry"],
        output_names=["packed"],
        source=source,
        ensure_row_contiguous=False,
    )


def pack_key_values(keys, values, count, width):
    layers = len(keys)
    inputs = [value for pair in zip(keys, values, strict=True) for value in pair]
    # Two data/stride pairs per layer fit Metal's 31 buffer slots up to seven.
    if (
        1 <= layers <= 7
        and 0 < count * width <= 0xFFFF_FFFF
        and all(
            v.ndim == 3
            and v.shape == inputs[0].shape
            and v.dtype == mx.bfloat16
            and v.size == count * width
            and v.size >= 8
            and v.shape[-2] * v.shape[-1] == width
            for v in inputs
        )
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        return _kernel(layers)(
            inputs=inputs + [mx.array([width, inputs[0].shape[-1]], mx.int32)],
            template=[("LAYERS", layers)],
            grid=(count * width, 2, layers),
            threadgroup=(256, 1, 1),
            output_shapes=[(count, layers, 2, width)],
            output_dtypes=[mx.bfloat16],
        )[0]
    return mx.stack(
        [
            mx.stack([k.reshape(count, -1), v.reshape(count, -1)], axis=1)
            for k, v in zip(keys, values, strict=True)
        ],
        axis=1,
    )
