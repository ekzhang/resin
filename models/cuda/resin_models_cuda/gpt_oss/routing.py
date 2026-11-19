"""Fused GPT-OSS top-four selection and softmax on Hopper."""

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice


@triton.jit
def _top4(Logits, Indices, Weights, EXPERTS: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, EXPERTS)
    scores = tl.load(Logits + row * EXPERTS + columns)

    # Sort BF16 by value, breaking ties (including signed zero) by expert ID.
    bits = scores.to(tl.uint16, bitcast=True).to(tl.uint32)
    bits = tl.where((bits & 0x7FFF) == 0, 0, bits)
    ordered = bits ^ tl.where((bits & 0x8000) != 0, 0xFFFF, 0x8000)
    keys = (ordered << 7) | (127 - columns).to(tl.uint32)
    winners = tl.topk(keys, 4)
    ids = 127 - (winners & 127).to(tl.int32)
    values = tl.load(Logits + row * EXPERTS + ids).to(tl.float32)

    exponentials = libdevice.exp(values - tl.max(values, 0))
    probabilities = tl.div_rn(exponentials, tl.sum(exponentials, 0))
    slots = row * 4 + tl.arange(0, 4)
    tl.store(Indices + slots, ids)
    # Preserve BF16 softmax rounding before widening for the MoE kernel.
    tl.store(Weights + slots, probabilities.to(tl.bfloat16).to(tl.float32))


def top4_softmax(logits: Tensor) -> tuple[Tensor, Tensor]:
    rows, experts = logits.shape
    indices = torch.empty((rows, 4), dtype=torch.int32, device=logits.device)
    weights = torch.empty((rows, 4), dtype=torch.float32, device=logits.device)
    cast(Any, _top4)[(rows,)](logits, indices, weights, experts, num_warps=1)
    return indices, weights
