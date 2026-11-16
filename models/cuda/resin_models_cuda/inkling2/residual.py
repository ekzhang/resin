"""Parallel source scores and ordered residual mixing for small Inkling2 batches."""

from typing import Any, cast

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["Rows"])
def _residual_scores(
    Sources,
    Addend,
    Q,
    Values,
    Scores,
    Rows,
    D: tl.constexpr,
    NQ: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, source, query = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    col = tl.arange(0, BLOCK)
    x = tl.full((BLOCK,), 0, tl.float32)
    for i in tl.static_range(len(Sources)):
        if source == i:
            x = tl.load(Sources[i] + row * D + col, col < D, 0).to(tl.float32)
    if HAS_ADDEND:
        if source == len(Sources) - 1:
            x += tl.load(Addend + row * D + col, col < D, 0).to(tl.float32)
    normalized = x * tl.rsqrt(tl.sum(x * x, 0) / D + 1e-6)
    q = tl.load(Q + query * D + col, col < D, 0).to(tl.float32) * (D**-0.5)
    score = tl.sum(normalized * q, 0)
    tl.store(Scores + (query * Rows + row) * len(Sources) + source, score)
    if query == 0:
        tl.store(
            Values + (source * Rows + row) * D + col,
            normalized if NORMALIZE else x,
            col < D,
        )


@triton.jit(do_not_specialize=["Rows"])
def _residual_combine(
    Values, Scores, Y, Rows, D: tl.constexpr, SOURCES: tl.constexpr, BLOCK: tl.constexpr
):
    row, query, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    output = tl.full((BLOCK,), 0, tl.float32)
    for i in tl.static_range(SOURCES):
        score = tl.load(Scores + (query * Rows + row) * SOURCES + i)
        x = tl.load(Values + (i * Rows + row) * D + col, col < D, 0)
        next_max = tl.maximum(maximum, score)
        correction, probability = tl.exp(maximum - next_max), tl.exp(score - next_max)
        output = output * correction + x * probability
        denominator = denominator * correction + probability
        maximum = next_max
    tl.store(Y + (query * Rows + row) * D + col, output / denominator, col < D)


def split_mix(sources, query, normalize_values, partial_addend):
    shape = sources[0].shape
    rows, width = sources[0].numel() // shape[-1], shape[-1]
    queries = query.numel() // width
    values = torch.empty(
        (len(sources), *shape), device=query.device, dtype=torch.float32
    )
    scores = torch.empty(
        (queries, rows, len(sources)), device=query.device, dtype=torch.float32
    )
    out = torch.empty((queries, *shape), device=query.device, dtype=torch.float32)
    cast(Any, _residual_scores)[(rows, len(sources), queries)](
        tuple(source.contiguous() for source in sources),
        partial_addend,
        query,
        values,
        scores,
        rows,
        width,
        queries,
        normalize_values,
        partial_addend is not None,
        triton.next_power_of_2(width),
        num_warps=4,
        enable_fp_fusion=False,
    )
    cast(Any, _residual_combine)[(rows, queries, triton.cdiv(width, 256))](
        values,
        scores,
        out,
        rows,
        width,
        len(sources),
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out.unbind(0)
