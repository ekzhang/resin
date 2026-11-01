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

    def __post_init__(self) -> None:
        if self.draft_tokens < 0:
            raise ValueError("draft_tokens cannot be negative")

    def accept(
        self,
        draft_rows: Sequence[Sequence[int]],
        target_rows: Sequence[Sequence[int]],
    ) -> SpeculativeAcceptance:
        acceptance = _accept_speculative_tokens(draft_rows, target_rows)
        self.blocks += len(draft_rows)
        self.drafts += sum(map(len, draft_rows))
        self.accepted += sum(acceptance.accepted_counts)
        self.emitted += sum(map(len, acceptance.token_rows))
        return acceptance

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


def _accept_speculative_tokens(
    draft_rows: Sequence[Sequence[int]],
    target_rows: Sequence[Sequence[int]],
) -> SpeculativeAcceptance:
    """Accept each matching draft prefix and append one target bonus token.

    A target row contains one sample per verified input: one for every draft
    comparison plus the final bonus/correction token.
    """

    if len(draft_rows) != len(target_rows):
        raise ValueError("draft and target batches must contain the same rows")

    token_rows: list[list[int]] = []
    accepted_counts: list[int] = []
    for row, (drafts, targets) in enumerate(zip(draft_rows, target_rows)):
        if len(targets) != len(drafts) + 1:
            raise ValueError(
                f"target row {row} must contain one more token than its drafts"
            )
        accepted = 0
        while accepted < len(drafts) and targets[accepted] == drafts[accepted]:
            accepted += 1
        token_rows.append([*drafts[:accepted], targets[accepted]])
        accepted_counts.append(accepted)

    return SpeculativeAcceptance(token_rows, accepted_counts)


__all__ = ["SpeculativeAcceptance", "SpeculativeStats"]
