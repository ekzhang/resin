"""Shared PyTorch layers with some device-specific kernel selection."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@torch.compile(dynamic=True, fullgraph=True, recompile_limit=128)  # ty: ignore[no-matching-overload]
def _rms_norm(hidden: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    # Mark the norm dimension as static (constant) for specialization, but try
    # not to specialize leading dimensions.
    torch._dynamo.mark_static(hidden, -1)
    if weight is not None:
        torch._dynamo.mark_static(weight)

    return F.rms_norm(
        hidden.float(),
        (hidden.shape[-1],),
        weight.float() if weight is not None else None,
        eps,
    ).to(hidden.dtype)


class RMSNorm(nn.Module):
    """RMSNorm using Quack on SM90+ and FlashInfer on older CUDA devices.

    Non-CUDA tensors and FP32 inputs unsupported by FlashInfer use compiled
    PyTorch normalization. ``with_scale=False`` omits the learned multiplier.
    """

    unit_weight: Tensor | None

    def __init__(
        self,
        width: int,
        eps: float = 1e-6,
        *,
        weight_dtype: torch.dtype = torch.bfloat16,
        with_scale: bool = True,
    ) -> None:
        super().__init__()
        self.width = width
        self.eps = eps
        if with_scale:
            self.weight = nn.Parameter(torch.ones(width, dtype=weight_dtype))
        else:
            self.register_parameter("weight", None)

        # This is only for used for caching an all-ones dummy tensor.
        self.register_buffer("unit_weight", None, persistent=False)

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.is_cuda:
            if torch.cuda.get_device_capability(hidden.device) >= (9, 0):
                from quack import rmsnorm

                return rmsnorm(hidden, self.weight, eps=self.eps)
            if hidden.dtype in (torch.float16, torch.bfloat16):
                from flashinfer import rmsnorm

                # FlashInfer requires a scale tensor with the input dtype.
                weight = self.weight
                if weight is None:
                    if (
                        self.unit_weight is None
                        or self.unit_weight.device != hidden.device
                        or self.unit_weight.dtype != hidden.dtype
                    ):
                        self.unit_weight = hidden.new_ones((self.width,))
                    weight = self.unit_weight
                else:
                    weight = weight.to(hidden.dtype)
                flat = hidden.reshape(-1, self.width)
                return rmsnorm(flat, weight, eps=self.eps).reshape_as(hidden)
        return _rms_norm(hidden, self.weight, self.eps)


__all__ = ["RMSNorm"]
