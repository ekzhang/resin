# ty: ignore
"""Owner-masked FP8 embedding lookup with rowwise UE8M0 block32 scales."""

import torch
import triton
import triton.language as tl


@triton.jit
def _lookup(
    ids,
    weight,
    scales,
    output,
    start: tl.constexpr,
    stop: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.load(ids + row)
    owned = (token >= start) & (token < stop)
    column = tl.arange(0, BLOCK)
    local = token - start
    value = tl.load(
        weight + local * DIM + column, mask=owned & (column < DIM), other=0.0
    ).to(tl.float32)
    exponent = tl.load(
        scales + local * (DIM // 32) + column // 32,
        mask=owned & (column < DIM),
        other=127,
    ).to(tl.uint32)
    bits = tl.where(
        exponent == 0, 0x00400000, tl.where(exponent == 255, 0x7FC00000, exponent << 23)
    )
    scale = bits.to(tl.float32, bitcast=True)
    tl.store(output + row * DIM + column, value * scale, mask=column < DIM)


def fp8_embedding(
    indices: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    start: int,
    stop: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    indices = indices.contiguous()
    if output is None:
        output = torch.empty(
            (*indices.shape, weight.shape[1]),
            device=indices.device,
            dtype=torch.bfloat16,
        )
    if indices.numel():
        with torch.cuda.device(indices.device):
            _lookup[(indices.numel(),)](
                indices,
                weight,
                scales.view(torch.uint8),
                output,
                start,
                stop,
                weight.shape[1],
                triton.next_power_of_2(weight.shape[1]),
            )
    return output
