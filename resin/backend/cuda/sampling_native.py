"""Native raw-logit top-k thresholds using FlashInfer CUDA headers.

Only compiled modules are cached. Every invocation owns its workspace, including
invocations captured into independent CUDA graphs. Request policies remain
device tensors and never enter a compilation key.
"""

import hashlib
from functools import cache
from pathlib import Path

import torch


@cache
def _load_native_module():
    """Build once during eager warmup, using FlashInfer's existing toolchain."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm the native sampler once before CUDA graph capture")
    import flashinfer
    from flashinfer.jit import gen_jit_spec

    source = Path(__file__).with_name("sampling_topk.cu")
    digest = hashlib.sha256(
        source.read_bytes() + flashinfer.__version__.encode()
    ).hexdigest()[:16]
    return gen_jit_spec(f"resin_{source.stem}_{digest}", [source]).build_and_load()


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
        module = _load_native_module()
        # The native binding asserts RadixRowState fits within4096 bytes, and
        # never needs more groups than rows. Storage is private to this call.
        workspace = torch.empty(rows * 4096, dtype=torch.uint8, device=logits.device)
        module.topk_cutoffs(logits, top_ks, active, out, workspace)
    return out


__all__ = ["topk_cutoffs"]
