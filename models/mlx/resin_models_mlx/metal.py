"""Small host-built Metal arguments with a stable MLX buffer binding."""

from functools import lru_cache
from math import prod

import mlx.core as mx
import numpy as np


def device_array(values, dtype: str = "int32") -> mx.array:
    """Pad leading rows to eight elements so MLX always binds a device array."""
    values = np.asarray(values, dtype=dtype)
    if values.size < 8:
        width = prod(values.shape[1:])
        padded = np.zeros((-(-8 // width), *values.shape[1:]), dtype=values.dtype)
        padded[: len(values)] = values
        values = padded
    return mx.array(values)


@lru_cache(maxsize=4096)
def constant(values: tuple, dtype: str) -> mx.array:
    # Host data carries no pending graph or stream, so serving threads can share it.
    return device_array(values, dtype)
