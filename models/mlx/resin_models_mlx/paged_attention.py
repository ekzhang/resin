"""Decode attention that reads keys and values directly from paged MLX slabs.

The arena stores one 128-token page of one layer as a *unit*:
``units[unit, 0, offset]`` holds keys and ``units[unit, 1, offset]`` values,
each ``kv_heads * head_dim`` wide. ``pages[row, block]`` names the unit holding
logical block ``block`` of the row's history for the current layer.

This is flash decoding: each threadgroup scores one partition of positions for
one row and one group of query heads sharing a KV head, with an online softmax.
The partitions are then merged by their log-sum-exp. The step's own key and
value, not yet written to the arena, are the final position of every row, so
history padding is never read. Context length, page-table width and batch
size are runtime data; only the model geometry selects a compiled kernel.
"""

from __future__ import annotations

from functools import cache, partial
from pathlib import Path

import mlx.core as mx

PARTITION_TOKENS = 512
# Below this many rows, half-size partitions keep more GPU cores busy.
SMALL_BATCH_ROWS = 4
SIMDGROUPS = 4
PREFETCH = 4
# MLX 0.32.4 binds kernel arguments below eight elements as constants.
_MIN_METADATA_ROWS = 8
_HEADS_PER_THREADGROUP = 4
# Bound the query and output registers held by each lane. Four 256-wide
# heads plus prefetched K/V spill on M1 Pro; two heads with one K/V pair
# also expose more independent threadgroups without duplicating the cache.
_HEAD_DIMENSIONS_PER_THREADGROUP = 512
# Metal guarantees 32 KiB of threadgroup memory.
_THREADGROUP_BYTES = 32 * 1024

# The preprocessor selects declarations outside the generated entrypoint and
# the same body inside it; the native metallib includes this header too.
_HEADER = (Path(__file__).parent / "_native/attention_kernel.h").read_text()
_SOURCE = "#define RESIN_ATTENTION_BODY\n" + _HEADER + "\n#undef RESIN_ATTENTION_BODY\n"


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="resin_paged_decode_attention",
        input_names=[
            "queries",
            "current_keys",
            "current_values",
            "units",
            "pages",
            "lengths",
            "geometry",
            "scale",
        ],
        output_names=["partial", "maxima", "sums"],
        source=_SOURCE,
        header=_HEADER,
    )


def paged_decode_attention(
    queries: mx.array,
    current_keys: mx.array,
    current_values: mx.array,
    units: mx.array,
    pages: mx.array,
    lengths: mx.array,
    layer: int,
    scale: float,
    skip: int = 0,
) -> mx.array:
    """Attend one new token per row over paged history plus itself.

    ``queries`` is ``[rows, heads, head_dim]``; ``current_keys`` and
    ``current_values`` are ``[rows, kv_heads, head_dim]``; ``units`` is the
    arena viewed as ``[units, 2, page, kv_heads * head_dim]``; ``pages`` is
    ``[rows, blocks]`` int32 base units, offset by ``layer``; ``lengths`` is the
    int32 history length of each row. History positions before ``skip`` are
    not read. Returns ``[rows, heads, head_dim]``.
    """

    return _merge_output(
        *_partials(
            queries,
            current_keys,
            current_values,
            units,
            pages,
            lengths,
            layer,
            scale,
            skip,
        ),
        dtype=queries.dtype,
    )


def paged_block_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    units: mx.array,
    pages: mx.array,
    lengths: mx.array,
    layer: int,
    scale: float,
    skip: int = 0,
    prior: tuple[mx.array, mx.array] | None = None,
) -> mx.array:
    """Attend a causal block of new tokens per row over paged history.

    ``queries`` is ``[rows, tokens, heads, head_dim]`` and ``keys``/``values``
    are ``[rows, tokens, kv_heads, head_dim]``, none of them written to the
    arena yet. Token ``j`` attends to the row's ``lengths`` history, then the
    optional ``prior`` keys and values (``[rows, count, kv_heads, head_dim]``,
    also unwritten), then block tokens ``0..j``. Each token is one row of the
    decode kernel, with its own key as the current position; the earlier
    unwritten keys form one more partition, merged by log-sum-exp. Returns
    ``[rows, tokens, heads, head_dim]``.
    """

    rows, tokens, heads, head_dim = queries.shape
    kv_heads = keys.shape[2]
    flat = rows * tokens
    flat_pages = mx.repeat(pages[:rows], tokens, axis=0)
    flat_lengths = mx.repeat(lengths[:rows], tokens, axis=0)
    if flat < _MIN_METADATA_ROWS:
        # Unread padding rows keep metadata out of constant address space,
        # whose different kernel ABI would compile another variant.
        padding = _MIN_METADATA_ROWS - flat
        flat_pages = mx.concatenate(
            [flat_pages, mx.zeros((padding, pages.shape[1]), pages.dtype)]
        )
        flat_lengths = mx.concatenate(
            [flat_lengths, mx.zeros((padding,), lengths.dtype)]
        )
    outputs, maxima, sums = _partials(
        queries.reshape(flat, heads, head_dim),
        keys.reshape(flat, kv_heads, head_dim),
        values.reshape(flat, kv_heads, head_dim),
        units,
        flat_pages,
        flat_lengths,
        layer,
        scale,
        skip,
    )
    if prior is not None:
        keys = mx.concatenate([prior[0], keys], axis=1)
        values = mx.concatenate([prior[1], values], axis=1)
    if keys.shape[1] > 1:
        block = _earlier_partition(scale)(queries, keys, values)
        outputs, maxima, sums = (
            mx.concatenate([part, extra], axis=2)
            for part, extra in zip((outputs, maxima, sums), block, strict=True)
        )
    output = _merge_output(outputs, maxima, sums, dtype=queries.dtype)
    return output.reshape(rows, tokens, heads, head_dim)


@cache
def _earlier_partition(scale: float):
    """One fused graph per softmax scale; shapes follow the block and batch."""

    return mx.compile(partial(_earlier_block_partition, scale=scale))


def _earlier_block_partition(queries, keys, values, *, scale):
    """One partition per query over the unwritten keys strictly before it.

    The last ``tokens`` keys belong to the queries, in order; any before them
    precede every query. Returns ``[rows * tokens, heads, 1, head_dim]``
    unnormalized outputs and ``[rows * tokens, heads, 1]`` maxima and sums in
    the kernel's format; a query without earlier keys contributes an empty
    partition.
    """

    rows, tokens, heads, head_dim = queries.shape
    prior = keys.shape[1] - tokens
    shared = heads // keys.shape[2]
    query = queries.astype(mx.float32).transpose(0, 2, 1, 3) * scale
    key = mx.repeat(keys.astype(mx.float32), shared, axis=2).transpose(0, 2, 3, 1)
    value = mx.repeat(values.astype(mx.float32), shared, axis=2).transpose(0, 2, 1, 3)
    scores = query @ key  # [rows, heads, tokens, tokens]
    earlier = mx.arange(prior + tokens)[None, :] < (prior + mx.arange(tokens))[:, None]
    scores = mx.where(earlier, scores, -mx.inf)
    peak = scores.max(axis=-1, keepdims=True)
    weights = mx.where(earlier, mx.exp(scores - mx.where(earlier, peak, 0.0)), 0.0)
    output = weights @ value  # [rows, heads, tokens, head_dim]

    def flatten(array):
        return array.transpose(0, 2, 1, *range(3, array.ndim)).reshape(
            rows * tokens, heads, 1, *array.shape[3:]
        )

    return (
        flatten(output),
        flatten(peak).squeeze(-1),
        flatten(weights.sum(axis=-1, keepdims=True)).squeeze(-1),
    )


def _partials(
    queries: mx.array,
    current_keys: mx.array,
    current_values: mx.array,
    units: mx.array,
    pages: mx.array,
    lengths: mx.array,
    layer: int,
    scale: float,
    skip: int,
) -> tuple[mx.array, mx.array, mx.array]:

    rows, heads, head_dim = queries.shape
    kv_heads = current_keys.shape[1]
    page = units.shape[2]
    if heads % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    if head_dim % 32:
        raise ValueError("paged decode attention needs head_dim divisible by 32")
    # Bind sparse cache reads without charging the entire arena to MLX's
    # command-buffer packing heuristic. Other geometries retain the same body.
    if (
        rows > 0
        and (heads, kv_heads, head_dim, page) == (8, 2, 256, 128)
        and units.ndim == 4
        and units.shape[0] > 0
        and units.shape[1:] == (2, 128, 512)
        and queries.dtype
        == current_keys.dtype
        == current_values.dtype
        == units.dtype
        == mx.bfloat16
        and pages.dtype == lengths.dtype == mx.int32
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        from resin_models_mlx._native_cache import _paged_attention_partials

        outputs, maxima, sums = _paged_attention_partials(
            queries,
            current_keys,
            current_values,
            units,
            pages,
            lengths,
            layer,
            scale,
            skip,
        )
        return outputs, maxima, sums
    # Each threadgroup serves an equal share of the query heads of a KV head.
    shared = heads // kv_heads
    # The cross-simdgroup reduction stores float maxima, sums and outputs.
    fits = [
        g
        for g in range(1, _HEADS_PER_THREADGROUP + 1)
        if shared % g == 0
        and (g == 1 or g * head_dim <= _HEAD_DIMENSIONS_PER_THREADGROUP)
        and SIMDGROUPS * g * (head_dim + 2) * 4 <= _THREADGROUP_BYTES
    ]
    if not fits:
        raise ValueError(f"head_dim {head_dim} exceeds paged decode threadgroup memory")
    group = max(fits)
    head_groups = shared // group
    longest = pages.shape[1] * page + 1
    wide_head = head_dim >= 256
    part = PARTITION_TOKENS // (2 if rows < SMALL_BATCH_ROWS and not wide_head else 1)
    prefetch = 1 if wide_head else PREFETCH
    partitions = -(-longest // part)
    outputs, maxima, sums = _kernel()(
        inputs=[
            queries,
            current_keys,
            current_values,
            units,
            pages,
            lengths,
            mx.array([pages.shape[1], partitions, layer, skip], dtype=mx.int32),
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
            ("PREFETCH", prefetch),
            ("T", units.dtype),
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
    return outputs, maxima, sums


@partial(mx.compile, shapeless=True)
def _merge(outputs: mx.array, maxima: mx.array, sums: mx.array) -> mx.array:
    """Combine partitions by their log-sum-exp; empty partitions weigh zero."""

    peak = maxima.max(axis=-1, keepdims=True)
    weights = mx.where(maxima == -mx.inf, 0.0, mx.exp(maxima - peak))
    numerator = (outputs * weights[..., None]).sum(axis=2)
    denominator = (sums * weights).sum(axis=-1)[..., None]
    return numerator / denominator


__all__ = ["PARTITION_TOKENS", "paged_block_attention", "paged_decode_attention"]


# MLX v0.32.4's small-column reduction order, Apple Inc., MIT (see NOTICE).
_MERGE_HEADER = r"""
#pragma clang fp contract(off)
inline float resin_max(float a, float b) {
    return (metal::isnan(a) || metal::isnan(b)) ? float(NAN) : (a > b ? a : b);
}
"""
_MERGE_SMALL_SOURCE = r"""
    const uint x = thread_position_in_threadgroup.x;
    const uint y = thread_position_in_threadgroup.y;
    const uint tile = threadgroup_position_in_grid.x;
    const uint head = threadgroup_position_in_grid.z;
    const uint P = uint(geometry[0]);
    const uint Y = min(P, 8u);
    const uint d = tile * 128 + x * 4;
    const ulong base = ulong(head) * P;
    threadgroup float peak;
    threadgroup float weights[32];
    threadgroup float denominator;
    threadgroup float intermediate[8 * 128];
    if (x == 0 && y == 0) {
        float value = -INFINITY;
        for (uint p = 0; p < P; ++p)
            value = resin_max(maxima[base + p], value);
        peak = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint index = y * 32 + x;
    if (index < P) {
        const float maximum = maxima[base + index];
        const float shifted = maximum - peak;
        weights[index] = maximum == -INFINITY ? 0.0f : metal::precise::exp(shifted);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (x == 0 && y == 0) {
        float value = 0.0f;
        for (uint p = 0; p < P; ++p) {
            // The original multiply materializes F32 before row_reduce_small.
            const float product = sums[base + p] * weights[p];
            value = product + value;
        }
        denominator = value;
    }
    float value[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // col_reduce_small: y,y+Y,..., then serial y0,y1,...,y(Y-1).
    for (uint p = y; p < P; p += Y) {
        const ulong input = (base + p) * D + d;
        for (uint i = 0; i < 4; ++i) {
            const float product = float(partials[input + i]) * weights[p];
            value[i] = product + value[i];
        }
    }
    for (uint i = 0; i < 4; ++i)
        intermediate[y * 128 + x * 4 + i] = value[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (y == 0) {
        for (uint i = 0; i < 4; ++i) {
            float total = intermediate[x * 4 + i];
            for (uint row = 1; row < Y; ++row)
                total = intermediate[row * 128 + x * 4 + i] + total;
            const float result = total / denominator;
            output[ulong(head) * D + d + i] = T(result);
        }
    }
"""


@cache
def _merge_kernel():
    return mx.fast.metal_kernel(
        name="resin_partition_merge_small",
        input_names=["partials", "maxima", "sums", "geometry"],
        output_names=["output"],
        source=_MERGE_SMALL_SOURCE,
        header=_MERGE_HEADER,
        compile_options={"math_mode": "safe"},
    )


def _fused_merge(outputs, maxima, sums, *, dtype):
    """Fuse D256/F32 partitions 2..31; other regimes keep native MLX."""
    if outputs.ndim != 4:
        return None
    rows, heads, partitions, dim = outputs.shape
    if not (
        dim == 256
        and 2 <= partitions < 32
        and rows * heads * partitions >= 8
        and maxima.shape == sums.shape == (rows, heads, partitions)
        and outputs.dtype == maxima.dtype == sums.dtype == mx.float32
        and dtype in (mx.float32, mx.bfloat16)
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        return None
    return _merge_kernel()(
        inputs=[outputs, maxima, sums, mx.array([partitions], mx.int32)],
        template=[("D", 256), ("T", dtype)],
        grid=(64, min(partitions, 8), rows * heads),
        threadgroup=(32, min(partitions, 8), 1),
        output_shapes=[(rows, heads, dim)],
        output_dtypes=[dtype],
    )[0]


def _merge_output(outputs, maxima, sums, *, dtype):
    fused = _fused_merge(outputs, maxima, sums, dtype=dtype)
    return _merge(outputs, maxima, sums).astype(dtype) if fused is None else fused


@cache
def _warm_small_merge():
    # Cache only completion, never arrays or streams created on this thread.
    partials = mx.zeros((1, 8, 2, 256), mx.float32)
    maxima = mx.zeros((1, 8, 2), mx.float32)
    sums = mx.ones((1, 8, 2), mx.float32)
    mx.eval(
        [
            _fused_merge(partials, maxima, sums, dtype=dtype)
            for dtype in (mx.bfloat16, mx.float32)
        ]
    )


def _warm_fused_merge():
    if mx.default_device() == mx.gpu and mx.metal.is_available():
        _warm_small_merge()
