# Portions are adapted from OpenAI gpt-oss and Hugging Face Transformers
# (Apache-2.0).

"""Small PyTorch transformer primitives shared by model families."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Reference/CUDA RMSNorm with checkpoint-controlled weight storage."""

    def __init__(
        self,
        width: int,
        eps: float,
        *,
        weight_dtype: torch.dtype = torch.float32,
        with_scale: bool = True,
    ) -> None:
        super().__init__()
        self.width = width
        self.eps = eps
        if with_scale:
            self.weight = nn.Parameter(torch.ones(width, dtype=weight_dtype))
        else:
            self.register_parameter("weight", None)
        self.register_buffer("unit_weight", None, persistent=False)

    def forward(self, hidden: Tensor) -> Tensor:
        weight = self.weight
        if hidden.is_cuda:
            from quack import rmsnorm

            if weight is None:
                if self.unit_weight is None:
                    self.unit_weight = torch.ones(
                        self.width, dtype=hidden.dtype, device=hidden.device
                    )
                weight = self.unit_weight
            return rmsnorm(hidden, weight, eps=self.eps)
        normalized = hidden.float()
        normalized *= torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        if weight is not None:
            normalized *= weight.float()
        return normalized.to(hidden.dtype)


__all__ = ["RMSNorm"]
