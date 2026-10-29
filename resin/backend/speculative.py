"""Shared speculative-decoding acceptance bookkeeping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeAcceptance:
    """Accepted output rows and the target-state depth they require."""

    token_rows: list[list[int]]
    accepted_counts: list[int]

    @property
    def processed_counts(self) -> list[int]:
        """Target input tokens whose state must become visible."""

        return [count + 1 for count in self.accepted_counts]


@dataclass(slots=True)
class SpeculativeStats:
    """Mutable acceptance counters shared by speculative model runners."""

    draft_tokens: int
    blocks: int = 0
    drafts: int = 0
    accepted: int = 0
    emitted: int = 0
    benchmark_accepted_drafts: int | None = None

    def __post_init__(self) -> None:
        if self.draft_tokens < 0:
            raise ValueError("draft_tokens cannot be negative")

    def accept(
        self,
        draft_rows: Sequence[Sequence[int]],
        target_rows: Sequence[Sequence[int]],
    ) -> SpeculativeAcceptance:
        fixed = self.validated_benchmark_accepted_drafts
        acceptance = accept_speculative_tokens(
            draft_rows, target_rows, benchmark_accepted_drafts=fixed
        )
        self.blocks += len(draft_rows)
        self.drafts += sum(map(len, draft_rows))
        self.accepted += sum(acceptance.accepted_counts)
        self.emitted += sum(map(len, acceptance.token_rows))
        return acceptance

    @property
    def validated_benchmark_accepted_drafts(self) -> int | None:
        fixed = self.benchmark_accepted_drafts
        if fixed is not None and (
            type(fixed) is not int or not 0 <= fixed <= self.draft_tokens
        ):
            raise ValueError("benchmark accepted drafts must fit the draft block")
        return fixed

    def snapshot(self, *, include_draft_tokens: bool = False) -> dict[str, int | float]:
        values: dict[str, int | float] = {
            "blocks": self.blocks,
            "drafts": self.drafts,
            "accepted": self.accepted,
            "emitted": self.emitted,
            "acceptance_rate": self.accepted / self.drafts if self.drafts else 0.0,
        }
        if include_draft_tokens:
            values = {"draft_tokens": self.draft_tokens, **values}
        return values


def accept_speculative_tokens(
    draft_rows: Sequence[Sequence[int]],
    target_rows: Sequence[Sequence[int]],
    *,
    benchmark_accepted_drafts: int | None = None,
) -> SpeculativeAcceptance:
    """Accept each matching draft prefix and append one target bonus token.

    A target row contains one sample per verified input: one for every draft
    comparison plus the final bonus/correction token. Explicit benchmark mode
    substitutes a fixed acceptance depth while retaining that target correction.
    """

    if len(draft_rows) != len(target_rows):
        raise ValueError("draft and target batches must contain the same rows")
    fixed = benchmark_accepted_drafts
    if fixed is not None and (type(fixed) is not int or fixed < 0):
        raise ValueError("benchmark accepted drafts must be a nonnegative integer")

    token_rows: list[list[int]] = []
    accepted_counts: list[int] = []
    for row, (drafts, targets) in enumerate(zip(draft_rows, target_rows)):
        if len(targets) != len(drafts) + 1:
            raise ValueError(
                f"target row {row} must contain one more token than its drafts"
            )
        if fixed is not None:
            if fixed > len(drafts):
                raise ValueError("benchmark accepted drafts exceed a verified row")
            # Synthetic acceptance still determines real state rollback depth.
            accepted = fixed
        else:
            accepted = 0
            while accepted < len(drafts) and targets[accepted] == drafts[accepted]:
                accepted += 1
        token_rows.append([*drafts[:accepted], targets[accepted]])
        accepted_counts.append(accepted)

    return SpeculativeAcceptance(token_rows, accepted_counts)


__all__ = ["SpeculativeAcceptance", "SpeculativeStats", "accept_speculative_tokens"]
