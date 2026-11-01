"""Backend-neutral Gemma 4 rotary primitives."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn

from resin.models.gemma4.config import SLIDING_ATTENTION


class Gemma4RotaryConfig(Protocol):
    @property
    def head_dim(self) -> int: ...
    @property
    def global_head_dim(self) -> int: ...
    @property
    def local_rope_theta(self) -> float: ...
    @property
    def global_rope_theta(self) -> float: ...
    @property
    def global_partial_rotary_factor(self) -> float: ...


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4RotaryConfig) -> None:
        super().__init__()
        self.local_head_dim = config.head_dim
        self.global_head_dim = config.global_head_dim
        self.local_theta = config.local_rope_theta
        self.global_theta = config.global_rope_theta
        self.global_partial = config.global_partial_rotary_factor
        self.register_buffer(
            "local_inv_freq", torch.empty(config.head_dim // 2), persistent=False
        )
        self.register_buffer(
            "global_inv_freq",
            torch.empty(config.global_head_dim // 2),
            persistent=False,
        )

    @staticmethod
    def _frequencies(
        head_dim: int, theta: float, proportion: float, device: torch.device
    ) -> Tensor:
        rotated = int(proportion * head_dim // 2)
        inverse = 1.0 / (
            theta
            ** (
                torch.arange(0, 2 * rotated, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        unrotated = head_dim // 2 - rotated
        if unrotated:
            inverse = torch.cat(
                (inverse, torch.zeros(unrotated, dtype=torch.float32, device=device))
            )
        return inverse

    def materialize(self, device: torch.device) -> None:
        self.local_inv_freq = self._frequencies(
            self.local_head_dim, self.local_theta, 1.0, device
        )
        self.global_inv_freq = self._frequencies(
            self.global_head_dim, self.global_theta, self.global_partial, device
        )

    def forward(
        self, positions: Tensor, layer_type: str, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        inverse = (
            self.local_inv_freq
            if layer_type == SLIDING_ATTENTION
            else self.global_inv_freq
        )
        frequencies = positions.float().unsqueeze(-1) * inverse.float()
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def apply_rotary(hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return hidden * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


__all__ = ["RotaryEmbedding", "apply_rotary"]
