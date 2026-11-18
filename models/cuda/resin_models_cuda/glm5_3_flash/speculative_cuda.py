"""Bit-preserving convolution history and hidden selection for GLM rollback.

Counts and source strides are device/runtime metadata, independent of context
length or sampling.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["width", "h0", "h1", "h2", "h3", "r0", "r1", "r2", "c0"])
def _select_history(
    History,
    Raw,
    Counts,
    Output,
    width,
    h0,
    h1,
    h2,
    h3,
    r0,
    r1,
    r2,
    c0,
    GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HISTORY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < GROUPS * CHANNELS * HISTORY
    position = index % HISTORY
    channel = index // HISTORY % CHANNELS
    group = index // (HISTORY * CHANNELS)
    source_position = tl.load(Counts + row * c0).to(tl.int64) + position
    saved = source_position < HISTORY
    prior = tl.load(
        History + row * h0 + group * h1 + channel * h2 + source_position * h3,
        valid & saved,
        other=0,
    )
    current = tl.load(
        Raw
        + (row * width + source_position - HISTORY) * r0
        + group * r1
        + channel * r2,
        valid & ~saved,
        other=0,
    )
    tl.store(
        Output + row * GROUPS * CHANNELS * HISTORY + index,
        tl.where(saved, prior, current),
        valid,
    )


def select_history(counts, width, history, raw):
    """Select the three accepted convolution samples without preparing KDA logits."""
    rows = _counts(counts, width)
    if history.ndim != 4 or raw.ndim != 3:
        raise ValueError("convolution history/raw QKV must have rank 4/3")
    groups, channels, length = history.shape[1:]
    if history.shape[0] != rows or raw.shape != (rows * width, groups, channels):
        raise ValueError("convolution history and packed QKV shapes disagree")
    if length != 3:
        raise ValueError("GLM convolution repair requires three saved samples")
    if any(x.device != counts.device for x in (history, raw)):
        raise ValueError("convolution history and counts must share a device")
    if history.dtype != raw.dtype:
        raise ValueError("saved and current convolution samples must share a dtype")
    output = torch.empty(history.shape, device=history.device, dtype=history.dtype)
    if rows:
        with torch.cuda.device(counts.device):
            cast(Any, _select_history)[
                (triton.cdiv(groups * channels * length, 256), rows)
            ](
                history,
                raw,
                counts,
                output,
                width,
                *history.stride(),
                *raw.stride(),
                counts.stride(0),
                groups,
                channels,
                length,
                256,
                num_warps=4,
            )
    return output


@triton.jit(do_not_specialize=["width", "h0", "h1", "c0", "p0"])
def _finish_commit(
    Hidden,
    Counts,
    Contexts,
    Selected,
    Lengths,
    width,
    h0,
    h1,
    c0,
    p0,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    column = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(Counts + row * c0).to(tl.int64)
    value = tl.load(
        Hidden + (row * width + count - 1) * h0 + column * h1, column < HIDDEN, other=0
    )
    tl.store(Selected + row * HIDDEN + column, value, column < HIDDEN)
    if tl.program_id(0) == 0:
        context = tl.load(Contexts + row * p0)
        tl.store(Lengths + row, context.to(tl.int64) + count)


def _counts(counts, width):
    if (
        not counts.is_cuda
        or counts.ndim != 1
        or counts.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("accepted counts must be a CUDA integer vector")
    if type(width) is not int or not 1 <= width <= 8:
        raise ValueError("KDA speculative blocks must contain 1–8 tokens")
    return counts.shape[0]


def finish_commit(counts, width, contexts, hidden):
    """Select the last accepted hidden row and update lengths in one launch."""
    rows = _counts(counts, width)
    if (
        hidden.ndim != 2
        or hidden.shape[0] != rows * width
        or contexts.shape != counts.shape
    ):
        raise ValueError("hidden/context shapes disagree with accepted counts")
    if contexts.device != counts.device or hidden.device != counts.device:
        raise ValueError("commit operands must share a CUDA device")
    if contexts.dtype not in (torch.int32, torch.int64):
        raise ValueError("context lengths must be integers")
    if any(stride <= 0 for tensor in (hidden, contexts) for stride in tensor.stride()):
        raise ValueError("commit operands require positive strides")
    selected = torch.empty(
        (rows, hidden.shape[1]), dtype=hidden.dtype, device=hidden.device
    )
    lengths = torch.empty(
        (rows,),
        dtype=torch.promote_types(contexts.dtype, counts.dtype),
        device=counts.device,
    )
    if rows:
        cast(Any, _finish_commit)[(triton.cdiv(hidden.shape[1], 256), rows)](
            hidden,
            counts,
            contexts,
            selected,
            lengths,
            width,
            *hidden.stride(),
            counts.stride(0),
            contexts.stride(0),
            hidden.shape[1],
            256,
            num_warps=4,
        )
    return selected, lengths
