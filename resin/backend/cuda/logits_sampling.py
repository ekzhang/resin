"""Two-kernel CUDA categorical sampling with request-local Philox streams.

Parallel tiles compute stable FP32 masses. One CTA per row selects a tile and
inverts its local CDF using a single full-width Philox draw. The scalar target
and conditional fraction use FP64, preserving all 32 random bits without FP64
vocabulary arithmetic. Seeded choices differ from Gumbel-max and FlashInfer;
request seeds and offsets remain invariant under batching and reordering.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.random import philox


@triton.jit
def _scaled_difference(values, center, temperature):
    """Center in FP32, scaling first only for finite subtraction overflow."""
    difference = values - center
    result = tl.div_rn(difference, temperature)
    # Half an ulp at FP32's maximum magnitude is 2**103. Smaller centers
    # cannot make any finite value overflow negative, so ordinary logits skip
    # the extra reduction entirely. All callers use a scalar maximum center.
    if center >= 2.0**103:
        overflow = (
            (difference == -float("inf"))
            & (values > -float("inf"))
            & (values < float("inf"))
            & (center < float("inf"))
        )
        if tl.constexpr(len(difference.shape) == 0):
            any_overflow = overflow
        else:
            any_overflow = tl.sum(overflow.to(tl.int32), 0) > 0
        if any_overflow:
            # Opposite signs cannot produce an inf - inf cancellation; a tiny
            # temperature retains a negative infinite normalized gap.
            result = tl.where(
                overflow,
                tl.div_rn(values, temperature) - tl.div_rn(center, temperature),
                result,
            )
    return result


@triton.jit
def _filtered_values(
    Logits, FilterState, row, token, valid, s0, s1, temperature, FILTERED: tl.constexpr
):
    values = tl.load(
        Logits + row * s0 + token.to(tl.int64) * s1, valid, other=-float("inf")
    ).to(tl.float32)
    if FILTERED:
        cutoff = tl.load(FilterState + row * 3)
        probability_cutoff = tl.load(FilterState + row * 3 + 1)
        if cutoff > -float("inf"):
            values = tl.where(values >= cutoff, values, -float("inf"))
        if probability_cutoff > 0:
            center = tl.load(FilterState + row * 3 + 2)
            weights = tl.exp(_scaled_difference(values, center, temperature))
            values = tl.where(weights >= probability_cutoff, values, -float("inf"))
    return values


@triton.jit(
    do_not_specialize=["s0", "s1", "st", "sm"],
    do_not_specialize_on_alignment=[
        "Logits",
        "Temperature",
        "Mask",
        "FilterState",
        "Maxima",
        "Greedy",
        "Mass",
    ],
)
def _partials(
    Logits,
    Temperature,
    Mask,
    FilterState,
    Maxima,
    Greedy,
    Mass,
    s0,
    s1,
    st,
    sm,
    V: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    FILTERED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    token = part * BLOCK + lane
    valid = lane < (V - part * BLOCK)
    temperature = tl.load(Temperature + row * st)
    stochastic = tl.load(Mask + row * sm)
    values = _filtered_values(
        Logits, FilterState, row, token, valid, s0, s1, temperature, FILTERED
    )
    first_nan = tl.min(tl.where(valid & (values != values), token, 2147483647), 0)
    maximum = tl.max(tl.where(valid & (values == values), values, -float("inf")), 0)
    greedy = tl.min(tl.where(valid & (values == maximum), token, 2147483647), 0)
    greedy = tl.where(first_nan < 2147483647, first_nan, greedy)
    maximum = tl.where(first_nan < 2147483647, float("nan"), maximum)
    mass = tl.full((), 0.0, tl.float32)
    if stochastic & (temperature > 0) & (temperature < float("inf")):
        if (maximum > -float("inf")) & (maximum < float("inf")):
            weights = tl.exp(_scaled_difference(values, maximum, temperature))
            weights = tl.where(valid & (values > -float("inf")), weights, 0.0)
            mass = tl.sum(weights, 0)
    index = row * PARTS + part
    tl.store(Maxima + index, maximum)
    tl.store(Greedy + index, greedy)
    tl.store(Mass + index, mass)


@triton.jit
def _below_total(value, total):
    # Both are nonnegative finite FP64 values, and total is strictly positive.
    # u<1 can still round u*total up to total; exclude that endpoint explicitly.
    previous = (total.to(tl.uint64, bitcast=True) - 1).to(tl.float64, bitcast=True)
    return tl.minimum(tl.maximum(value, 0.0), previous)


@triton.jit
def _request_uniform(seed, offset):
    random, _, _, _ = philox(
        seed,
        tl.full((), 0, tl.uint32),
        tl.full((), 0, tl.uint32),
        (offset & 0xFFFFFFFF).to(tl.uint32),
        (offset >> 32).to(tl.uint32),
    )
    return (random.to(tl.float64) + 0.5) * 2.3283064365386963e-10


@triton.jit(
    do_not_specialize=["s0", "s1", "st", "ss", "so", "sm"],
    do_not_specialize_on_alignment=[
        "Logits",
        "Temperature",
        "Seeds",
        "Offsets",
        "Mask",
        "FilterState",
        "Maxima",
        "Greedy",
        "Mass",
        "Output",
    ],
)
def _finish(
    Logits,
    Temperature,
    Seeds,
    Offsets,
    Mask,
    FilterState,
    Maxima,
    Greedy,
    Mass,
    Output,
    s0,
    s1,
    st,
    ss,
    so,
    sm,
    V: tl.constexpr,
    PARTS: tl.constexpr,
    PART_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    FILTERED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.arange(0, PART_BLOCK)
    maxima = tl.load(Maxima + row * PARTS + part, part < PARTS, other=-float("inf"))
    ids = tl.load(Greedy + row * PARTS + part, part < PARTS, other=2147483647)
    first_nan = tl.min(
        tl.where((part < PARTS) & (maxima != maxima), ids, 2147483647), 0
    )
    maximum = tl.max(tl.where(maxima == maxima, maxima, -float("inf")), 0)
    greedy = tl.min(tl.where(maxima == maximum, ids, 2147483647), 0)
    selected = tl.where(first_nan < 2147483647, first_nan, greedy)
    temperature = tl.load(Temperature + row * st)
    stochastic = tl.load(Mask + row * sm)
    if stochastic & (temperature > 0) & (temperature < float("inf")):
        if (
            (first_nan == 2147483647)
            & (maximum > -float("inf"))
            & (maximum < float("inf"))
        ):
            local_mass = tl.load(Mass + row * PARTS + part, part < PARTS, other=0.0)
            scale = tl.exp(_scaled_difference(maxima, maximum, temperature))
            masses = tl.where(
                (part < PARTS) & (maxima > -float("inf")), local_mass * scale, 0.0
            )
            cdf = tl.cumsum(masses, 0)
            total = tl.sum(tl.where(part == PARTS - 1, cdf, 0.0), 0)
            seed = tl.load(Seeds + row * ss).to(tl.uint64)
            offset = tl.load(Offsets + row * so).to(tl.uint64)
            uniform = _request_uniform(seed, offset)
            target = _below_total(uniform * total.to(tl.float64), total.to(tl.float64))
            selected_part = tl.min(
                tl.where(
                    (part < PARTS) & (cdf.to(tl.float64) > target), part, 2147483647
                ),
                0,
            )
            before = tl.sum(tl.where(part == selected_part - 1, cdf, 0.0), 0)
            selected_mass = tl.sum(tl.where(part == selected_part, masses, 0.0), 0)
            tile_max = tl.sum(tl.where(part == selected_part, maxima, 0.0), 0)
            fraction = (target - before.to(tl.float64)) / selected_mass.to(tl.float64)
            # Recompute just the chosen tile. Its CDF total, rather than its
            # earlier reduction total, closes any FP32 tree-rounding gap.
            lane = tl.arange(0, BLOCK)
            token = selected_part * BLOCK + lane
            valid = lane < (V - selected_part * BLOCK)
            values = _filtered_values(
                Logits, FilterState, row, token, valid, s0, s1, temperature, FILTERED
            )
            weights = tl.exp(_scaled_difference(values, tile_max, temperature))
            weights = tl.where(valid & (values > -float("inf")), weights, 0.0)
            local_cdf = tl.cumsum(weights, 0)
            local_total = tl.sum(tl.where(lane == BLOCK - 1, local_cdf, 0.0), 0)
            local_target = _below_total(
                fraction * local_total.to(tl.float64), local_total.to(tl.float64)
            )
            selected = tl.min(
                tl.where(
                    valid & (local_cdf.to(tl.float64) > local_target), token, 2147483647
                ),
                0,
            )
    tl.store(Output + row, selected.to(tl.int64))


@dataclass(frozen=True)
class LogitsSamplingWorkspace:
    """Private scratch and output storage, optionally retained across replays."""

    shape: tuple[int, int]
    maxima: torch.Tensor
    greedy: torch.Tensor
    mass: torch.Tensor
    output: torch.Tensor


def _geometry(logits: torch.Tensor) -> tuple[int, int]:
    if not logits.is_cuda or logits.ndim != 2:
        raise ValueError("logits must be a CUDA matrix")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("logits must use float16, bfloat16 or float32")
    rows, vocab = logits.shape
    if not 0 < vocab < 2**31:
        raise ValueError("vocabulary must contain between one and 2**31-1 tokens")
    return rows, vocab


def prepare_workspace(logits: torch.Tensor) -> LogitsSamplingWorkspace:
    """Allocate fixed scratch outside capture when explicitly managing lifetime."""
    rows, vocab = _geometry(logits)
    shape = (rows, triton.cdiv(vocab, 1024))
    return LogitsSamplingWorkspace(
        (rows, vocab),
        torch.empty(shape, device=logits.device, dtype=torch.float32),
        torch.empty(shape, device=logits.device, dtype=torch.int32),
        torch.empty(shape, device=logits.device, dtype=torch.float32),
        torch.empty(rows, device=logits.device, dtype=torch.int64),
    )


def _byte_span(value: torch.Tensor) -> tuple[int, int] | None:
    """Conservative address bounds from tensor metadata; never reads device data."""
    if value.numel() == 0:
        return None
    extents = [(size - 1) * stride for size, stride in zip(value.shape, value.stride())]
    first = value.data_ptr()
    itemsize = value.element_size()
    return (
        first + sum(min(extent, 0) for extent in extents) * itemsize,
        first + (1 + sum(max(extent, 0) for extent in extents)) * itemsize,
    )


def _check_no_overlap(
    inputs: Iterable[tuple[str, torch.Tensor]],
    scratch: Iterable[tuple[str, torch.Tensor]],
    out: torch.Tensor,
) -> None:
    reads = [(name, _byte_span(value)) for name, value in inputs]
    writes = [("output", _byte_span(out))]
    writes.extend((name, _byte_span(value)) for name, value in scratch)
    for index, (name, span) in enumerate(writes):
        if span is None:
            continue
        for other_name, other_span in reads + writes[:index]:
            if (
                other_span is not None
                and span[0] < other_span[1]
                and other_span[0] < span[1]
            ):
                raise ValueError(f"{name} must not overlap {other_name}")


def sample_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    stochastic_mask: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    workspace: LogitsSamplingWorkspace | None = None,
    filter_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample logits using optional inclusive device filter thresholds.

    Positive finite temperatures and usable logits define a categorical draw.
    NaN/+inf rows, all-negative-infinity rows, and invalid temperatures use
    PyTorch-compatible
    greedy fallback; isolated -inf logits have zero mass. No input or RNG
    state is mutated. Callers own offset advancement and workspace reuse.

    Parameters, row count, strides, and pointer alignments stay runtime inputs.
    Vocabulary, dtype, and the presence of filter storage determine geometry.
    FP32 exponentials and cumulative masses can move draws near a boundary;
    request-local 32-bit uniform targets and fractions are evaluated in FP64.
    """
    rows, vocab = _geometry(logits)
    inputs = (
        ("logits", logits),
        ("temperatures", temperatures),
        ("seeds", seeds),
        ("offsets", offsets),
        ("stochastic_mask", stochastic_mask),
    )
    for (name, value), dtype in zip(
        inputs[1:], (torch.float32, torch.int64, torch.int64, torch.bool), strict=True
    ):
        if (
            value.shape != (rows,)
            or value.dtype != dtype
            or value.device != logits.device
        ):
            raise ValueError(
                f"{name} must be a row-aligned {dtype} tensor on the logits device"
            )
    if filter_state is not None:
        if (
            filter_state.shape != (rows, 3)
            or filter_state.device != logits.device
            or filter_state.dtype != torch.float32
            or not filter_state.is_contiguous()
        ):
            raise ValueError("filter state requires contiguous CUDA float32 [rows,3]")
        inputs += (("filter_state", filter_state),)
    if workspace is None:
        workspace = prepare_workspace(logits)
    if workspace.shape != (rows, vocab):
        raise ValueError("sampling workspace geometry differs from the input")
    parts = triton.cdiv(vocab, 1024)
    scratch = (
        ("workspace.maxima", workspace.maxima),
        ("workspace.greedy", workspace.greedy),
        ("workspace.mass", workspace.mass),
    )
    for (_, value), dtype in zip(
        scratch, (torch.float32, torch.int32, torch.float32), strict=True
    ):
        if (
            value.shape != (rows, parts)
            or value.device != logits.device
            or value.dtype != dtype
            or not value.is_contiguous()
        ):
            raise ValueError("sampling workspace buffers have invalid metadata")
    out = workspace.output if out is None else out
    if (
        out.shape != (rows,)
        or out.dtype != torch.int64
        or out.device != logits.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "output must be a contiguous row-aligned int64 tensor on the logits device"
        )
    _check_no_overlap(inputs, scratch, out)
    if rows == 0:
        return out
    filter_input = stochastic_mask if filter_state is None else filter_state
    with torch.cuda.device(logits.device):
        _partials.run(
            logits,
            temperatures,
            stochastic_mask,
            filter_input,
            workspace.maxima,
            workspace.greedy,
            workspace.mass,
            *logits.stride(),
            temperatures.stride(0),
            stochastic_mask.stride(0),
            vocab,
            parts,
            1024,
            filter_state is not None,
            grid=(rows, parts),
            num_warps=4,
            warmup=False,
        )
        _finish.run(
            logits,
            temperatures,
            seeds,
            offsets,
            stochastic_mask,
            filter_input,
            workspace.maxima,
            workspace.greedy,
            workspace.mass,
            out,
            *logits.stride(),
            temperatures.stride(0),
            seeds.stride(0),
            offsets.stride(0),
            stochastic_mask.stride(0),
            vocab,
            parts,
            triton.next_power_of_2(parts),
            1024,
            filter_state is not None,
            grid=(rows,),
            num_warps=4,
            warmup=False,
        )
    return out


__all__ = ["LogitsSamplingWorkspace", "prepare_workspace", "sample_logits"]
