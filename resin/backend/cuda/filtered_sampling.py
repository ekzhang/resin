"""Device-selected top-k/top-p thresholds before tiled Philox sampling.

Large vocabularies use parallel native radix selection; small vocabularies use
the single-block selector below. All cutoffs retain ties. Runtime policies can
change inside one captured graph, and inactive rows skip vocabulary work.
"""

import torch
import triton
import triton.language as tl

from resin.backend.cuda.logits_sampling import (
    _check_no_overlap,
    _geometry,
    _scaled_difference,
)


@triton.jit
def _ordered_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    return bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)


@triton.jit
def _from_key(key):
    bits = key ^ tl.where((key & 0x80000000) != 0, 0x80000000, 0xFFFFFFFF)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _sorted_cutoff(Buffer, count, target, SIZE: tl.constexpr, WEIGHTED: tl.constexpr):
    index = tl.arange(0, SIZE)
    if WEIGHTED:
        value = tl.load(Buffer + index, index < count, other=0.0)
    else:
        value = tl.load(Buffer + index, index < count, other=-float("inf"))
    descending = tl.sort(value, descending=True)
    if WEIGHTED:
        cumulative = tl.cumsum(descending.to(tl.float64), 0)
        selected = tl.min(
            tl.where((index < count) & (cumulative >= target), index, SIZE), 0
        )
        # A different FP64 reduction order can put the target one ulp above
        # this bucket's total; retain its last actual value, never padded zero.
        selected = tl.minimum(selected, count - 1)
    else:
        selected = target - 1
    return tl.sum(tl.where(index == selected, descending, 0.0), 0)


@triton.jit
def _compact_cutoff(
    Buffer, count, target, CAPACITY: tl.constexpr, WEIGHTED: tl.constexpr
):
    # Limit bitonic sorts to small sets: larger sorts inflate compile time and
    # register pressure. Larger compact buckets keep using exact radix scans.
    if CAPACITY <= 64:
        cutoff = _sorted_cutoff(Buffer, count, target, 64, WEIGHTED)
    else:
        if count <= 64:
            cutoff = _sorted_cutoff(Buffer, count, target, 64, WEIGHTED)
        else:
            cutoff = _sorted_cutoff(Buffer, count, target, 256, WEIGHTED)
    return cutoff


@triton.jit
def _source_values(
    Row,
    Buffer,
    index,
    valid,
    stride,
    compact,
    cutoff,
    maximum,
    temperature,
    WEIGHTED: tl.constexpr,
):
    if compact:
        value = tl.load(Buffer + index, valid, other=0.0)
    else:
        value = tl.load(
            Row + index.to(tl.int64) * stride, valid, other=-float("inf")
        ).to(tl.float32)
        valid = valid & (value >= cutoff) & (value > -float("inf"))
        if WEIGHTED:
            value = tl.exp(_scaled_difference(value, maximum, temperature))
            valid = valid & (value > 0)
    return value, valid


@triton.jit
def _radix_cutoff(
    Row,
    Buffer,
    stride,
    cutoff,
    maximum,
    temperature,
    target,
    initial_count,
    initial_compact,
    V: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    WEIGHTED: tl.constexpr,
):
    """Select a boundary value by count or by positive FP32 probability mass."""
    lane = tl.arange(0, BLOCK)
    if WEIGHTED:
        # Positive FP32 weights <= 1 share their highest two ordered-key bits.
        prefix = tl.full((), 0x80000000, tl.uint32)
        prefix_mask = tl.full((), 0xC0000000, tl.uint32)
        shift = 29
        bits = 1
    else:
        bins = tl.arange(0, 16)
        prefix = tl.full((), 0, tl.uint32)
        prefix_mask = tl.full((), 0, tl.uint32)
        shift = 28
        bits = 4
    answer = tl.full((), 0, tl.uint32)
    count = initial_count
    compact = tl.full((), False, tl.int1) | initial_compact
    done = tl.full((), False, tl.int1)
    if compact & (count <= 256):
        answer = _ordered_key(
            _compact_cutoff(Buffer, count, target, CAPACITY, WEIGHTED)
        )
        done = True
    while (shift >= 0) & (done == 0):
        if WEIGHTED:
            # Only the upper half's mass is needed. A scalar reduction avoids
            # the large FP64 [BLOCK, bins] matrix and register spilling.
            upper_mass_lanes = tl.full((BLOCK,), 0, tl.float64)
            upper_count_lanes = tl.full((BLOCK,), 0, tl.int32)
            valid_count_lanes = tl.full((BLOCK,), 0, tl.int32)
        else:
            counts = tl.full((16,), 0, tl.int32)
        smallest_lanes = tl.full((BLOCK,), 0xFFFFFFFF, tl.uint32)
        largest_lanes = tl.full((BLOCK,), 0, tl.uint32)
        for start in range(0, count, BLOCK):
            index = start + lane
            value, valid = _source_values(
                Row,
                Buffer,
                index,
                lane < (count - start),
                stride,
                compact,
                cutoff,
                maximum,
                temperature,
                WEIGHTED,
            )
            key = _ordered_key(value)
            valid = valid & ((key & prefix_mask) == prefix)
            smallest_lanes = tl.minimum(
                smallest_lanes, tl.where(valid, key, 0xFFFFFFFF)
            )
            largest_lanes = tl.maximum(largest_lanes, tl.where(valid, key, 0))
            if WEIGHTED:
                upper = valid & (((key >> shift) & 1) != 0)
                upper_mass_lanes += tl.where(upper, value, 0.0).to(tl.float64)
                upper_count_lanes += upper.to(tl.int32)
                valid_count_lanes += valid.to(tl.int32)
            else:
                # Bin zero contains the highest values, so cumsum gives exact
                # descending integer counts.
                digit = 15 - ((key >> shift) & 15).to(tl.int32)
                counts += tl.histogram(digit, 16, mask=valid)
        # Reduce once per scan instead of synchronizing every vocabulary tile.
        smallest = tl.min(smallest_lanes, 0)
        largest = tl.max(largest_lanes, 0)
        if WEIGHTED:
            upper_mass = tl.sum(upper_mass_lanes, 0)
            upper_count = tl.sum(upper_count_lanes, 0)
            valid_count = tl.sum(valid_count_lanes, 0)
        if smallest == largest:
            # Includes huge repeated boundary groups: one scan resolves them.
            answer = smallest
            done = True
        else:
            if WEIGHTED:
                take_upper = upper_mass >= target
                target -= tl.where(take_upper, 0.0, upper_mass)
                bucket_count = tl.where(
                    take_upper, upper_count, valid_count - upper_count
                )
                digit = take_upper.to(tl.uint32)
            else:
                cumulative = tl.cumsum(counts, 0)
                selected = tl.min(tl.where(cumulative >= target, bins, 15), 0)
                target -= tl.sum(tl.where(bins < selected, counts, 0), 0)
                bucket_count = tl.sum(tl.where(bins == selected, counts, 0), 0)
                digit = (15 - selected).to(tl.uint32)
            prefix |= digit << shift
            prefix_mask |= tl.full((), (1 << bits) - 1, tl.uint32) << shift
            # Skip shared leading digits exactly. The two shifts avoid a
            # 32-bit shift by 32 when the highest differing bit is bit 31.
            leading = tl.inline_asm_elementwise(
                "clz.b32 $0, $1;",
                constraints="=r,r",
                args=[smallest ^ largest],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            highest = 31 - leading
            common = (tl.full((), 0xFFFFFFFF, tl.uint32) << highest) << 1
            prefix |= smallest & common
            prefix_mask |= common
            answer = prefix
            # Large buckets remain a scan of the original logits. Compact
            # buckets can then use radix scans, or one sort for <=256 values.
            if (compact == 0) & (bucket_count <= CAPACITY):
                written = 0
                for start in range(0, V, BLOCK):
                    index = start + lane
                    value, valid = _source_values(
                        Row,
                        Buffer,
                        index,
                        lane < (V - start),
                        stride,
                        False,
                        cutoff,
                        maximum,
                        temperature,
                        WEIGHTED,
                    )
                    key = _ordered_key(value)
                    keep = valid & ((key & prefix_mask) == prefix)
                    position = written + tl.cumsum(keep.to(tl.int32), 0) - 1
                    tl.store(Buffer + position, value, keep)
                    written += tl.sum(keep.to(tl.int32), 0)
                tl.debug_barrier()
                count = written
                compact = True
                if count <= 256:
                    answer = _ordered_key(
                        _compact_cutoff(Buffer, count, target, CAPACITY, WEIGHTED)
                    )
                    done = True
            shift = tl.minimum(shift - bits, (highest // bits) * bits)
    return _from_key(answer)


@triton.jit(
    do_not_specialize=["s0", "s1", "st", "sk", "sp", "sm"],
    do_not_specialize_on_alignment=[
        "Logits",
        "Temperature",
        "TopK",
        "TopP",
        "Mask",
        "Buffer",
        "State",
    ],
)
def _filter_thresholds(
    Logits,
    Temperature,
    TopK,
    TopP,
    Mask,
    Buffer,
    State,
    s0,
    s1,
    st,
    sk,
    sp,
    sm,
    V: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    # Reset on every replay, including transitions back to unfiltered/greedy.
    tl.store(State + row * 3, -float("inf"))
    tl.store(State + row * 3 + 1, 0.0)
    tl.store(State + row * 3 + 2, 0.0)
    stochastic = tl.load(Mask + row * sm)
    k = tl.load(TopK + row * sk)
    p = tl.load(TopP + row * sp)
    temperature = tl.load(Temperature + row * st)
    if (
        ~stochastic
        | ((k >= V) & (p >= 1))
        | ~(temperature > 0)
        | ~(temperature < float("inf"))
    ):
        return
    Row = Logits + row * s0
    Scratch = Buffer + row * CAPACITY
    lane = tl.arange(0, BLOCK)
    # Accumulate independently in each lane across the vocabulary. Reducing
    # every chunk would introduce hundreds of CTA barriers per row scan.
    lane_maximum = tl.full((BLOCK,), -float("inf"), tl.float32)
    lane_invalid = tl.full((BLOCK,), False, tl.int1)
    lane_finite_count = tl.full((BLOCK,), 0, tl.int32)
    for start in range(0, V, BLOCK):
        index = start + lane
        valid = lane < (V - start)
        value = tl.load(Row + index.to(tl.int64) * s1, valid, other=-float("inf")).to(
            tl.float32
        )
        lane_invalid |= valid & ((value != value) | (value == float("inf")))
        lane_maximum = tl.maximum(lane_maximum, value)
        lane_finite_count += (valid & (value > -float("inf"))).to(tl.int32)
    maximum = tl.max(lane_maximum, 0)
    invalid = tl.sum(lane_invalid.to(tl.int32), 0) > 0
    finite_count = tl.sum(lane_finite_count, 0)
    # Permissive defaults leave invalid rows untouched for the tiled sampler's
    # original NaN/+inf/all-inf greedy fallback.
    if invalid | (finite_count == 0):
        return
    cutoff = -float("inf")
    if k < finite_count:
        cutoff = _radix_cutoff(
            Row,
            Scratch,
            s1,
            cutoff,
            maximum,
            temperature,
            k,
            V,
            False,
            V,
            CAPACITY,
            BLOCK,
            False,
        )
    probability_cutoff = 0.0
    if p < 1:
        lane_total = tl.full((BLOCK,), 0, tl.float64)
        # Nucleus mass comes from the full temperature-scaled distribution.
        # Top-k is an independent final cap, never a renormalization before p.
        try_compact = finite_count <= CAPACITY
        kept_count = 0
        for start in range(0, V, BLOCK):
            index = start + lane
            valid = lane < (V - start)
            value = tl.load(
                Row + index.to(tl.int64) * s1, valid, other=-float("inf")
            ).to(tl.float32)
            keep = valid & (value > -float("inf"))
            weight = tl.exp(_scaled_difference(value, maximum, temperature))
            lane_total += tl.where(keep, weight, 0.0).to(tl.float64)
            if try_compact:
                keep = keep & (weight > 0)
                chunk_count = tl.sum(keep.to(tl.int32), 0)
                if kept_count + chunk_count > CAPACITY:
                    # Once full, later chunks need only contribute total mass.
                    try_compact = False
                elif chunk_count > 0:
                    position = kept_count + tl.cumsum(keep.to(tl.int32), 0) - 1
                    tl.store(Scratch + position, weight, keep)
                    kept_count += chunk_count
        total = tl.sum(lane_total, 0)
        compact = try_compact & (kept_count <= CAPACITY)
        count = tl.where(compact, kept_count, V)
        tl.debug_barrier()
        probability_cutoff = _radix_cutoff(
            Row,
            Scratch,
            s1,
            -float("inf"),
            maximum,
            temperature,
            p.to(tl.float64) * total,
            count,
            compact,
            V,
            CAPACITY,
            BLOCK,
            True,
        )
    tl.store(State + row * 3, cutoff)
    tl.store(State + row * 3 + 1, probability_cutoff)
    tl.store(State + row * 3 + 2, maximum)


def validate_filtered_inputs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    stochastic_mask: torch.Tensor,
    out: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Validate metadata before any sampling launch can overwrite a policy view."""
    rows, vocab = _geometry(logits)
    inputs = [("logits", logits)]
    for name, value, dtype in (
        ("temperatures", temperatures, torch.float32),
        ("top_ks", top_ks, torch.int32),
        ("top_ps", top_ps, torch.float32),
        ("seeds", seeds, torch.int64),
        ("offsets", offsets, torch.int64),
        ("stochastic_mask", stochastic_mask, torch.bool),
    ):
        if (
            value.shape != (rows,)
            or value.dtype != dtype
            or value.device != logits.device
        ):
            raise ValueError(
                f"{name} must be a row-aligned {dtype} tensor on the logits device"
            )
        inputs.append((name, value))
    if out is not None:
        if (
            out.shape != (rows,)
            or out.dtype != torch.int64
            or out.device != logits.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "sampling output must be a contiguous row-aligned CUDA int64 tensor"
            )
        _check_no_overlap(inputs, (), out)
    return rows, vocab


@triton.jit(
    do_not_specialize=["s0", "s1", "st", "sk", "sp", "sm"],
    do_not_specialize_on_alignment=[
        "Logits",
        "Temperature",
        "TopK",
        "TopP",
        "Mask",
        "Maxima",
        "Invalid",
    ],
)
def _filter_stats(
    Logits,
    Temperature,
    TopK,
    TopP,
    Mask,
    Maxima,
    Invalid,
    s0,
    s1,
    st,
    sk,
    sp,
    sm,
    V: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    temperature = tl.load(Temperature + row * st)
    active = (
        tl.load(Mask + row * sm)
        & (temperature > 0)
        & (temperature < float("inf"))
        & ((tl.load(TopK + row * sk) < V) | (tl.load(TopP + row * sp) < 1))
    )
    maximum = tl.full((), -float("inf"), tl.float32)
    invalid = tl.full((), 0, tl.int32)
    if active:
        lane = tl.arange(0, BLOCK)
        start = part * BLOCK
        valid = lane < (V - start)
        value = tl.load(
            Logits + row * s0 + (start + lane).to(tl.int64) * s1,
            valid,
            other=-float("inf"),
        ).to(tl.float32)
        bad = (value != value) | (value == float("inf"))
        maximum = tl.max(tl.where(bad, -float("inf"), value), 0)
        invalid = tl.max((valid & bad).to(tl.int32), 0)
    tl.store(Maxima + row * PARTS + part, maximum)
    tl.store(Invalid + row * PARTS + part, invalid)


@triton.jit(
    do_not_specialize=["sk", "sp"],
    do_not_specialize_on_alignment=[
        "Maxima",
        "Invalid",
        "TopK",
        "TopP",
        "State",
        "ActiveK",
        "ActiveP",
        "P",
    ],
)
def _filter_stats_finish(
    Maxima,
    Invalid,
    TopK,
    TopP,
    State,
    ActiveK,
    ActiveP,
    P,
    sk,
    sp,
    V: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.arange(0, BLOCK)
    maximum = tl.max(
        tl.load(Maxima + row * PARTS + part, part < PARTS, other=-float("inf")), 0
    )
    invalid = tl.max(tl.load(Invalid + row * PARTS + part, part < PARTS, other=0), 0)
    active = (invalid == 0) & (maximum > -float("inf"))
    k = tl.load(TopK + row * sk)
    p = tl.load(TopP + row * sp)
    tl.store(State + row * 3, -float("inf"))
    tl.store(State + row * 3 + 1, 0.0)
    tl.store(State + row * 3 + 2, tl.where(active, maximum, 0.0))
    tl.store(ActiveK + row, active & (k < V))
    tl.store(ActiveP + row, active & (p < 1))
    tl.store(P + row, p)


@triton.jit(
    do_not_specialize=["s0", "s1", "st"],
    do_not_specialize_on_alignment=[
        "Logits",
        "Temperature",
        "State",
        "ActiveK",
        "ActiveP",
        "Weights",
        "GuardMass",
    ],
)
def _filter_weights(
    Logits,
    Temperature,
    State,
    ActiveK,
    ActiveP,
    Weights,
    GuardMass,
    s0,
    s1,
    st,
    V: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if tl.load(ActiveP + row):
        start = tl.program_id(1) * BLOCK
        lane = tl.arange(0, BLOCK)
        index = start + lane
        valid = lane < (V - start)
        value = tl.load(
            Logits + row * s0 + index.to(tl.int64) * s1, valid, other=-float("inf")
        ).to(tl.float32)
        maximum = tl.load(State + row * 3 + 2)
        temperature = tl.load(Temperature + row * st)
        weight = tl.exp(_scaled_difference(value, maximum, temperature))
        weight = tl.where(valid & (value > -float("inf")), weight, 0.0)
        tl.store(Weights + row * V + index, weight, valid)
        if tl.load(ActiveK + row):
            # A nucleus that extends beyond the entire top-k set cannot
            # further restrict that set. Leave a wide margin for AIR's FP32
            # bucket sums; these FP64 partials only enable that early exit.
            cutoff = tl.load(State + row * 3)
            mass = weight.to(tl.float64)
            total = tl.sum(mass, 0)
            retained = tl.sum(tl.where(value >= cutoff, mass, 0.0), 0)
            slot = (row * PARTS + tl.program_id(1)) * 2
            tl.store(GuardMass + slot, total)
            tl.store(GuardMass + slot + 1, retained)


def _prepare_parallel_filters(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    stochastic_mask: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    from resin.backend.cuda.sampling_native import air_top_p_cutoff, topk_cutoffs

    rows, vocab = logits.shape
    block = 1024
    parts = triton.cdiv(vocab, block)
    maxima = torch.empty((rows, parts), dtype=torch.float32, device=logits.device)
    invalid = torch.empty((rows, parts), dtype=torch.int32, device=logits.device)
    active_k = torch.empty(rows, dtype=torch.bool, device=logits.device)
    active_p = torch.empty_like(active_k)
    p = torch.empty(rows, dtype=torch.float32, device=logits.device)
    weights = torch.empty((rows, vocab), dtype=torch.float32, device=logits.device)
    guard_mass = torch.empty(
        (rows, parts, 2), dtype=torch.float64, device=logits.device
    )
    _filter_stats.run(
        logits,
        temperatures,
        top_ks,
        top_ps,
        stochastic_mask,
        maxima,
        invalid,
        logits.stride(0),
        logits.stride(1),
        temperatures.stride(0),
        top_ks.stride(0),
        top_ps.stride(0),
        stochastic_mask.stride(0),
        vocab,
        parts,
        block,
        grid=(rows, parts),
        warmup=False,
        num_warps=4,
    )
    _filter_stats_finish.run(
        maxima,
        invalid,
        top_ks,
        top_ps,
        state,
        active_k,
        active_p,
        p,
        top_ks.stride(0),
        top_ps.stride(0),
        vocab,
        parts,
        triton.next_power_of_2(parts),
        grid=(rows,),
        warmup=False,
        num_warps=4,
    )
    topk_cutoffs(logits, top_ks, active_k, out=state[:, 0])
    _filter_weights.run(
        logits,
        temperatures,
        state,
        active_k,
        active_p,
        weights,
        guard_mass,
        logits.stride(0),
        logits.stride(1),
        temperatures.stride(0),
        vocab,
        parts,
        block,
        grid=(rows, parts),
        warmup=False,
        num_warps=4,
    )
    air_top_p_cutoff(
        weights,
        p,
        active_p,
        out=state[:, 1],
        guard_mass=guard_mass,
        guard_active=active_k,
    )
    return state


def prepare_filters(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    stochastic_mask: torch.Tensor,
    *,
    block_size: int = 1024,
    num_warps: int = 8,
) -> torch.Tensor:
    """Return FP32 [rows, 3] logit cutoff, probability cutoff, and global maximum.

    Policies use positive top-k and 0 < top-p <= 1, as validated by the engine.
    Top-p uses the full temperature-scaled distribution before the top-k cap.
    Parameters, strides and pointer alignments do not specialize kernels. Scratch
    belongs to this invocation/capture. Large vocabularies use a FP32 weight
    matrix and native radix workspaces; small vocabularies use bounded compact
    scratch. AIR reconstructs deterministic FP32 bucket masses; the small-vocab
    selector accumulates FP64 mass. Both use FP32 exponential weights, retain
    every cutoff tie, and can differ at floating-point nucleus boundaries.
    Launch geometry depends on vocabulary and hardware, never policy values.
    """
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("filter block size must be a positive power of two")
    if num_warps not in (4, 8, 16, 32):
        raise ValueError("filter launch must use 4, 8, 16, or 32 warps")
    rows, vocab = _geometry(logits)
    for name, value, dtype in (
        ("temperatures", temperatures, torch.float32),
        ("top_ks", top_ks, torch.int32),
        ("top_ps", top_ps, torch.float32),
        ("stochastic_mask", stochastic_mask, torch.bool),
    ):
        if (
            value.shape != (rows,)
            or value.dtype != dtype
            or value.device != logits.device
        ):
            raise ValueError(
                f"{name} must be a row-aligned {dtype} tensor on the logits device"
            )
    state = torch.empty((rows, 3), dtype=torch.float32, device=logits.device)
    if rows == 0:
        return state
    if vocab >= 2048:
        with torch.cuda.device(logits.device):
            return _prepare_parallel_filters(
                logits, temperatures, top_ks, top_ps, stochastic_mask, state
            )
    capacity = min(vocab, 4096)
    scratch = torch.empty((rows, capacity), dtype=torch.float32, device=logits.device)
    with torch.cuda.device(logits.device):
        _filter_thresholds.run(
            logits,
            temperatures,
            top_ks,
            top_ps,
            stochastic_mask,
            scratch,
            state,
            logits.stride(0),
            logits.stride(1),
            temperatures.stride(0),
            top_ks.stride(0),
            top_ps.stride(0),
            stochastic_mask.stride(0),
            vocab,
            capacity,
            block_size,
            grid=(rows,),
            warmup=False,
            num_warps=num_warps,
        )
    return state


__all__ = ["prepare_filters", "validate_filtered_inputs"]
