"""Expand request-local sampling metadata for a verified block in one launch."""

import torch
import triton
import triton.language as tl

from resin.backend.cuda.sampling import SamplingTensors
from resin.backend.sampling import SAMPLING_OFFSET_STRIDE


@triton.jit(
    do_not_specialize=[
        "count",
        "width",
        "start_step",
        "st",
        "sp",
        "sk",
        "ss",
        "so",
        "sm",
    ],
    do_not_specialize_on_alignment=[
        "Temperature",
        "TopP",
        "TopK",
        "Seeds",
        "Offsets",
        "Mask",
        "ExpandedTemperature",
        "ExpandedTopP",
        "ExpandedTopK",
        "ExpandedSeeds",
        "ExpandedOffsets",
        "ExpandedMask",
    ],
)
def _expand_metadata(
    Temperature,
    TopP,
    TopK,
    Seeds,
    Offsets,
    Mask,
    ExpandedTemperature,
    ExpandedTopP,
    ExpandedTopK,
    ExpandedSeeds,
    ExpandedOffsets,
    ExpandedMask,
    count,
    width,
    start_step,
    st,
    sp,
    sk,
    ss,
    so,
    sm,
    OFFSET_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < count
    row = index // width
    column = index % width
    tl.store(
        ExpandedTemperature + index,
        tl.load(Temperature + row * st, valid, other=1.0),
        valid,
    )
    tl.store(ExpandedTopP + index, tl.load(TopP + row * sp, valid, other=1.0), valid)
    tl.store(ExpandedTopK + index, tl.load(TopK + row * sk, valid, other=1), valid)
    tl.store(ExpandedSeeds + index, tl.load(Seeds + row * ss, valid, other=0), valid)
    base_offset = tl.load(Offsets + row * so, valid, other=0).to(tl.uint64)
    offset = (
        base_offset + (start_step.to(tl.uint64) + column.to(tl.uint64)) * OFFSET_STRIDE
    )
    tl.store(ExpandedOffsets + index, offset.to(tl.int64), valid)
    tl.store(ExpandedMask + index, tl.load(Mask + row * sm, valid, other=False), valid)


def expand_sampling(
    tensors: SamplingTensors, width: int, start_step: int
) -> SamplingTensors:
    """Allocate invocation-owned metadata; inputs are validated by sample_block."""
    count = tensors.temperatures.numel() * width
    device = tensors.temperatures.device
    expanded = SamplingTensors(
        temperatures=torch.empty(count, dtype=torch.float32, device=device),
        top_ps=torch.empty(count, dtype=torch.float32, device=device),
        top_ks=torch.empty(count, dtype=torch.int32, device=device),
        seeds=torch.empty(count, dtype=torch.int64, device=device),
        offsets=torch.empty(count, dtype=torch.int64, device=device),
        stochastic_mask=torch.empty(count, dtype=torch.bool, device=device),
    )
    if count:
        source = (
            tensors.temperatures,
            tensors.top_ps,
            tensors.top_ks,
            tensors.seeds,
            tensors.offsets,
            tensors.stochastic_mask,
        )
        destination = (
            expanded.temperatures,
            expanded.top_ps,
            expanded.top_ks,
            expanded.seeds,
            expanded.offsets,
            expanded.stochastic_mask,
        )
        with torch.cuda.device(device):
            _expand_metadata.run(
                *source,
                *destination,
                count,
                width,
                start_step,
                *(value.stride(0) for value in source),
                SAMPLING_OFFSET_STRIDE,
                256,
                grid=(triton.cdiv(count, 256),),
                num_warps=4,
                warmup=False,
            )
    return expanded


__all__ = ["expand_sampling"]
