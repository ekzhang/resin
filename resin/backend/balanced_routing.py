"""Synthetic expert routing and opt-in configuration for model gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BalancedExpertRouting:
    """Synthetic routing configuration shared by eager and fused router paths."""

    replica: int = 0
    expert_parallel_size: int = 1

    def __post_init__(self) -> None:
        if type(self.replica) is not int or self.replica < 0:
            raise ValueError("routing replica must be a nonnegative integer")
        if type(self.expert_parallel_size) is not int or self.expert_parallel_size < 1:
            raise ValueError("expert_parallel_size must be a positive integer")

    def indices(self, scores: torch.Tensor, top_k: int) -> torch.Tensor:
        """Balance routes across contiguous expert owners, then their experts.

        This permutation balances communication even for small batches while
        retaining distinct token routes and even coverage of individual experts.
        """

        experts_per_owner = scores.shape[-1] // self.expert_parallel_size
        ordinal = (
            torch.arange(scores.shape[0] * top_k, device=scores.device)
            + self.replica * scores.shape[0] * top_k
        )
        return (
            (ordinal % self.expert_parallel_size) * experts_per_owner
            + (ordinal // self.expert_parallel_size) % experts_per_owner
        ).reshape(-1, top_k)


class BalancedRoutingMixin:
    """Cooperative mixin for gates that support synthetic balanced routing.

    Model preparation records the expert owner count. Benchmarks enable the
    override before graph capture, supplying the attention replica and topology.
    Concrete gates consume the configuration in their own routing operations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.expert_parallel_size = 1
        self.balanced_routing: BalancedExpertRouting | None = None

    def enable_balanced_routing(
        self, *, replica: int = 0, expert_parallel_size: int | None = None
    ) -> None:
        self.balanced_routing = BalancedExpertRouting(
            replica,
            self.expert_parallel_size
            if expert_parallel_size is None
            else expert_parallel_size,
        )


__all__ = ["BalancedExpertRouting", "BalancedRoutingMixin"]
