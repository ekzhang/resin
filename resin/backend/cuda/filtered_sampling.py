"""Device-selected top-k/top-p thresholds before tiled Philox sampling.

CuTe mass histograms select top-p and native radix selection handles raw top-k.
All vocabularies use this same path, retaining every cutoff tie.
Runtime policies can change inside one graph, and inactive rows skip work.
"""

import torch
import triton
import triton.language as tl

from resin.backend.cuda.logits_sampling import _check_no_overlap, _geometry


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


def prepare_filters(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    stochastic_mask: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 [rows, 3] logit cutoff, probability cutoff, and global maximum.

    Policies use positive top-k and 0 < top-p <= 1, as validated by the engine.
    Top-p uses the full temperature-scaled distribution before the top-k cap.
    Parameters, strides and pointer alignments do not specialize kernels. Scratch
    belongs to this invocation/capture. Integer mass histograms reconstruct
    FP32 exponential weights in FP64; the native top-k workspace selects raw
    logits. Both cutoffs retain every tie.
    Batch size and hardware choose runtime launch geometry; policy values do
    not specialize kernels.
    """
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
    from resin.backend.cuda._sampling_cutedsl import nucleus_cutoffs
    from resin.backend.cuda.sampling_native import topk_cutoffs

    with torch.cuda.device(logits.device):
        block = 1024
        parts = triton.cdiv(vocab, block)
        maxima = torch.empty((rows, parts), dtype=torch.float32, device=logits.device)
        invalid = torch.empty((rows, parts), dtype=torch.int32, device=logits.device)
        active_k = torch.empty(rows, dtype=torch.bool, device=logits.device)
        active_p = torch.empty_like(active_k)
        p = torch.empty(rows, dtype=torch.float32, device=logits.device)
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
        # Rank raw logits before temperature/weight rounding can create ties.
        topk_cutoffs(logits, top_ks, active_k, out=state[:, 0])
        nucleus_cutoffs(logits, temperatures, state, active_k, active_p, p)
    return state


__all__ = ["prepare_filters", "validate_filtered_inputs"]
