"""Bit-preserving convolution history and hidden selection for GLM rollback.

Counts and source strides are device/runtime metadata, independent of context
length or sampling. Masked-logit preparation remains an independent test control
for the original FLA recurrence.
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


@triton.jit(
    do_not_specialize=[
        "width",
        "f0",
        "f1",
        "f2",
        "b0",
        "b1",
        "h0",
        "h1",
        "h2",
        "h3",
        "r0",
        "r1",
        "r2",
        "c0",
    ]
)
def _prepare_replay(
    Forget,
    Beta,
    History,
    Raw,
    Counts,
    MaskedForget,
    MaskedBeta,
    NextHistory,
    width,
    f0,
    f1,
    f2,
    b0,
    b1,
    h0,
    h1,
    h2,
    h3,
    r0,
    r1,
    r2,
    c0,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HISTORY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(Counts + row * c0).to(tl.int64)
    forget_size = width * HEADS * DIM
    beta_size = width * HEADS
    history_size = GROUPS * CHANNELS * HISTORY

    token = index // (HEADS * DIM)
    head = index // DIM % HEADS
    channel = index % DIM
    forget_valid = index < forget_size
    value = tl.load(
        Forget + (row * width + token) * f0 + head * f1 + channel * f2,
        forget_valid,
        other=0,
    )
    # Select in the source dtype: accepted BF16/FP32 bits pass through unchanged.
    negative_inf = tl.full((BLOCK,), float("-inf"), value.dtype)
    tl.store(
        MaskedForget + row * forget_size + index,
        tl.where(token < count, value, negative_inf),
        forget_valid,
    )

    beta_index = index - forget_size
    token = beta_index // HEADS
    head = beta_index % HEADS
    beta_valid = (beta_index >= 0) & (beta_index < beta_size)
    value = tl.load(Beta + (row * width + token) * b0 + head * b1, beta_valid, other=0)
    negative_inf = tl.full((BLOCK,), float("-inf"), value.dtype)
    tl.store(
        MaskedBeta + row * beta_size + beta_index,
        tl.where(token < count, value, negative_inf),
        beta_valid,
    )

    history_index = beta_index - beta_size
    history_valid = (history_index >= 0) & (history_index < history_size)
    position = history_index % HISTORY
    channel = history_index // HISTORY % CHANNELS
    group = history_index // (HISTORY * CHANNELS)
    source_position = count + position
    saved = source_position < HISTORY
    prior = tl.load(
        History + row * h0 + group * h1 + channel * h2 + source_position * h3,
        history_valid & saved,
        other=0,
    )
    current = tl.load(
        Raw
        + (row * width + source_position - HISTORY) * r0
        + group * r1
        + channel * r2,
        history_valid & ~saved,
        other=0,
    )
    tl.store(
        NextHistory + row * history_size + history_index,
        tl.where(saved, prior, current),
        history_valid,
    )


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


def prepare_replay(counts, width, forget, beta, history, raw):
    """Allocate independent masked logits and accepted convolution history.

    Values in counts must lie in [1,width], as guaranteed by speculative_step.
    No host read or mutation of the initial state, history or packed QKV occurs.
    """
    rows = _counts(counts, width)
    if (
        forget.ndim != 3
        or beta.shape != forget.shape[:2]
        or forget.shape[0] != rows * width
    ):
        raise ValueError(
            "replay logits must contain one sequence-major verification block"
        )
    if history.ndim != 4 or raw.ndim != 3:
        raise ValueError("convolution history/raw QKV must have rank 4/3")
    groups, channels, length = history.shape[1:]
    if history.shape[0] != rows or raw.shape != (rows * width, groups, channels):
        raise ValueError("convolution history and packed QKV shapes disagree")
    if length != 3:
        raise ValueError("GLM convolution repair requires three saved samples")
    for value in (forget, beta, history, raw):
        if value.device != counts.device or any(
            stride <= 0 for stride in value.stride()
        ):
            raise ValueError(
                "replay operands must share a CUDA device and have positive strides"
            )
    if forget.dtype not in (torch.bfloat16, torch.float32) or beta.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("replay logits must be BF16 or FP32")
    if raw.dtype != history.dtype or raw.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("convolution operands must have matching BF16/FP32 dtype")
    masked_forget = torch.empty(forget.shape, dtype=forget.dtype, device=forget.device)
    masked_beta = torch.empty(beta.shape, dtype=beta.dtype, device=beta.device)
    next_history = torch.empty(
        history.shape, dtype=history.dtype, device=history.device
    )
    if rows:
        heads, dim = forget.shape[1:]
        elements = width * heads * (dim + 1) + groups * channels * length
        cast(Any, _prepare_replay)[(triton.cdiv(elements, 256), rows)](
            forget,
            beta,
            history,
            raw,
            counts,
            masked_forget,
            masked_beta,
            next_history,
            width,
            *forget.stride(),
            *beta.stride(),
            *history.stride(),
            *raw.stride(),
            counts.stride(0),
            heads,
            dim,
            groups,
            channels,
            length,
            256,
            num_warps=4,
        )
    return masked_forget, masked_beta, next_history


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
