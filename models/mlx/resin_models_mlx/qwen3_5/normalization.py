"""GDN Q/K normalization using MLX's fused weighted RMS operation."""

from functools import cache

import mlx.core as mx


@cache
def _weights(width: int, dtype: mx.Dtype, inv_scale: float):
    query = mx.full((width,), inv_scale**2, dtype=dtype)
    key = mx.full((width,), inv_scale, dtype=dtype)
    # A runner can load on one OS thread and serve on another. Publish only
    # evaluated constants, without primitives tied to the creating stream.
    mx.eval(query, key)
    return query, key


def normalize_qk(
    query: mx.array, key: mx.array, *, inv_scale: float, eps: float
) -> tuple[mx.array, mx.array]:
    """Preserve normalization's dtype rounding before applying its scale.

    Native RMS rounds the normalized value to its output dtype before
    multiplying the weight, exactly as the separate RMS-plus-scale path does.
    """
    query_weight, _ = _weights(query.shape[-1], query.dtype, inv_scale)
    _, key_weight = _weights(key.shape[-1], key.dtype, inv_scale)
    rms_eps = eps * inv_scale**2
    return (
        mx.fast.rms_norm(query, query_weight, rms_eps),
        mx.fast.rms_norm(key, key_weight, rms_eps),
    )
