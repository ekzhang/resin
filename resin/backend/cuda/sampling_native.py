"""Native threshold selectors using the installed FlashInfer CUDA headers.

Only compiled modules are cached. Every invocation owns its workspace, including
invocations captured into independent CUDA graphs. Request policies remain
device tensors and never enter a compilation key.
"""

import hashlib
from functools import cache
from pathlib import Path

import torch


@cache
def _load_native_module(source_name: str):
    """Build once during eager warmup, using FlashInfer's existing toolchain."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm the native sampler once before CUDA graph capture")
    import flashinfer
    from flashinfer.jit import gen_jit_spec

    source = Path(__file__).with_name(source_name)
    digest = hashlib.sha256(
        source.read_bytes() + flashinfer.__version__.encode()
    ).hexdigest()[:16]
    return gen_jit_spec(f"resin_{source.stem}_{digest}", [source]).build_and_load()


def _air_workspace_size(rows: int, vocab: int) -> int:
    # Matches air_top_p::Counter<float>, NUM_BUCKETS and calcBufLen<float>.
    # The native binding checks this against sizeof(Counter), so a changed
    # installed FlashInfer ABI fails clearly instead of overrunning scratch.
    def align256(size: int) -> int:
        return (size + 255) // 256 * 256

    buffer_length = align256(vocab // 32)
    return (
        align256(384 * rows)
        + align256(8 * 2048 * rows)
        + align256(4 * 2048 * rows)
        + 2 * align256(4 * buffer_length * rows)
    )


def air_top_p_cutoff(
    weights: torch.Tensor,
    top_ps: torch.Tensor,
    active: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    guard_mass: torch.Tensor | None = None,
    guard_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Find inclusive top-p cutoffs for contiguous FP32 CUDA weights [B,V].

    Callers provide nonnegative finite weights, with at least one positive
    weight in every active row. Weights need not sum to one: AIR multiplies p
    by the row's actual mass. Inactive rows emit zero and skip all radix work.
    Vocabularies below 2048 must use the small-vocabulary selector, matching
    FlashInfer's own dispatch. The output may be a strided column of [B,3].

    AIR's deterministic integer histograms reconstruct FP32 bucket masses.
    Cutoffs close to a nucleus boundary can therefore differ from the earlier
    FP64-mass selector. This function neither draws nor changes RNG state.

    Optional guard_mass is contiguous FP64 [B,tiles,2], containing full and
    retained-top-k mass per tile; guard_active identifies rows with top-k.
    Those rows skip AIR when retained mass < (p - 0.001) * full mass. Callers
    must still apply the top-k cutoff. Other rows never read guard_mass.
    """
    from resin.backend.cuda.logits_sampling import _check_no_overlap

    if (
        not weights.is_cuda
        or weights.ndim != 2
        or weights.dtype != torch.float32
        or not weights.is_contiguous()
    ):
        raise ValueError("AIR weights must be a contiguous FP32 CUDA matrix")
    rows, vocab = weights.shape
    if not 2048 <= vocab < 2**31 or rows >= 2**31:
        raise ValueError("AIR requires 2048 <= vocab < 2**31 and rows < 2**31")
    for name, value, dtype in (
        ("top_ps", top_ps, torch.float32),
        ("active", active, torch.bool),
    ):
        if (
            value.shape != (rows,)
            or value.dtype != dtype
            or value.device != weights.device
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous row-aligned {dtype} on weights device"
            )
    if (guard_mass is None) != (guard_active is None):
        raise ValueError("AIR guard_mass and guard_active must be provided together")
    guard_inputs = ()
    if guard_mass is not None and guard_active is not None:
        if (
            guard_mass.ndim != 3
            or guard_mass.shape[0] != rows
            or not 0 < guard_mass.shape[1] < 2**31
            or guard_mass.shape[2] != 2
            or guard_mass.dtype != torch.float64
            or guard_mass.device != weights.device
            or not guard_mass.is_contiguous()
        ):
            raise ValueError(
                "AIR guard_mass must be contiguous FP64 [rows,tiles,2] on weights device"
            )
        if (
            guard_active.shape != (rows,)
            or guard_active.dtype != torch.bool
            or guard_active.device != weights.device
            or not guard_active.is_contiguous()
        ):
            raise ValueError(
                "AIR guard_active must be contiguous row-aligned bool on weights device"
            )
        guard_inputs = (("guard_mass", guard_mass), ("guard_active", guard_active))
    if out is None:
        out = torch.empty(rows, dtype=torch.float32, device=weights.device)
    elif (
        out.shape != (rows,)
        or out.dtype != torch.float32
        or out.device != weights.device
        or out.stride(0) <= 0
    ):
        raise ValueError(
            "AIR output must be row-aligned FP32 with positive stride on weights device"
        )
    _check_no_overlap(
        (("weights", weights), ("top_ps", top_ps), ("active", active)) + guard_inputs,
        (),
        out,
    )
    if rows == 0:
        return out
    with torch.cuda.device(weights.device):
        module = _load_native_module("sampling_air.cu")
        workspace = torch.empty(
            _air_workspace_size(rows, vocab), dtype=torch.uint8, device=weights.device
        )
        module.air_top_p_cutoff(
            weights, top_ps, active, out, workspace, guard_mass, guard_active
        )
    return out


def topk_cutoffs(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    active: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Find inclusive raw-logit top-k cutoffs; inactive rows emit -inf.

    Logits may use FP16/BF16/FP32 and arbitrary nonnegative strides. Policies
    and k remain device inputs. The result can write directly to state[:,0].
    """
    from resin.backend.cuda.logits_sampling import _check_no_overlap, _geometry

    rows, vocab = _geometry(logits)
    if rows >= 2**31 or any(stride < 0 for stride in logits.stride()):
        raise ValueError(
            "native top-k requires rows < 2**31 and nonnegative logits strides"
        )
    for name, value, dtype in (
        ("top_ks", top_ks, torch.int32),
        ("active", active, torch.bool),
    ):
        if (
            value.shape != (rows,)
            or value.dtype != dtype
            or value.device != logits.device
            or value.stride(0) < 0
        ):
            raise ValueError(f"{name} must be row-aligned {dtype} on logits device")
    if out is None:
        out = torch.empty(rows, dtype=torch.float32, device=logits.device)
    elif (
        out.shape != (rows,)
        or out.dtype != torch.float32
        or out.device != logits.device
        or out.stride(0) <= 0
    ):
        raise ValueError(
            "top-k output must be row-aligned FP32 with positive stride on logits device"
        )
    _check_no_overlap(
        (("logits", logits), ("top_ks", top_ks), ("active", active)), (), out
    )
    if rows == 0:
        return out
    with torch.cuda.device(logits.device):
        module = _load_native_module("sampling_topk.cu")
        # The native binding asserts RadixRowState fits within4096 bytes, and
        # never needs more groups than rows. Storage is private to this call.
        workspace = torch.empty(rows * 4096, dtype=torch.uint8, device=logits.device)
        module.topk_cutoffs(logits, top_ks, active, out, workspace)
    return out


__all__ = ["air_top_p_cutoff", "topk_cutoffs"]
