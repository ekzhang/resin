"""DeltaNet gates with FP32 decay and the upstream recurrent kernels."""

from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gated_delta as official


@partial(mx.compile, shapeless=True)
def decay_beta(
    A_log: mx.array, a: mx.array, b: mx.array, dt_bias: mx.array
) -> tuple[mx.array, mx.array]:
    """Fuse the FP32 decay preparation and the original-dtype beta sigmoid."""
    decay = mx.exp(
        -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a.astype(mx.float32) + dt_bias)
    )
    return decay, mx.sigmoid(b)


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    decay, beta = decay_beta(A_log, a, b, dt_bias)
    if state is None:
        rows, _, _, key_dim = q.shape
        value_heads, value_dim = v.shape[-2:]
        state = mx.zeros((rows, value_heads, value_dim, key_dim), mx.float32)
    if (
        mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return official.gated_delta_ops(q, k, v, decay, beta, state, mask)
    return official.gated_delta_kernel(q, k, v, decay, beta, state, mask)
