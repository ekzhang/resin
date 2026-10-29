"""Owned CUDA results and device feedback for Engine overlap scheduling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
import torch

from resin.backend.cuda.failures import GraphInputError
from resin.backend.cuda.staging import TensorStager
from resin.backend.sampling import SAMPLING_OFFSET_STRIDE, token_output

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput
    from resin.backend.speculative import SpeculativeStats


@dataclass(frozen=True, slots=True)
class DeviceFeedback:
    """Exact next decode inputs, independent of reusable graph output buffers."""

    token_ids: torch.Tensor
    positions: torch.Tensor

    def restrict(self, rows: int) -> Self:
        # Scratch rows must remain at their reserved initial position across
        # arbitrarily many continuations; they have no logical request progress.
        self.token_ids[rows:].zero_()
        self.positions[rows:].fill_(1)
        return self

    def inputs(self) -> dict[str, torch.Tensor]:
        return {
            "token_ids": self.token_ids[:, None],
            "positions": self.positions,
            "sampling_offsets": (self.positions + 1) * SAMPLING_OFFSET_STRIDE,
        }


@dataclass(frozen=True, slots=True)
class DeviceAcceptance:
    """One GPU decision shared by state commit, output, feedback and statistics.

    Captured tensors belong to the graph and change on replay. Consume state
    selections and create the completion before another replay overwrites them.
    """

    tokens: torch.Tensor
    accepted_drafts: torch.Tensor
    counts: torch.Tensor
    positions: torch.Tensor
    draft_tokens: int

    @classmethod
    def from_samples(
        cls,
        drafts: torch.Tensor,
        targets: torch.Tensor,
        positions: torch.Tensor,
        *,
        fixed_acceptance: int | torch.Tensor | None = None,
    ) -> Self:
        if drafts.ndim != 2 or targets.shape != (len(drafts), drafts.shape[1] + 1):
            raise ValueError(
                "target samples must contain one bonus after the draft block"
            )
        draft_tokens = drafts.shape[1]
        dynamic_fixed = isinstance(fixed_acceptance, torch.Tensor)
        if dynamic_fixed and (
            fixed_acceptance.ndim != 0
            or fixed_acceptance.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "runtime fixed acceptance must be an integer scalar tensor"
            )
        if (
            fixed_acceptance is not None
            and not dynamic_fixed
            and (
                type(fixed_acceptance) is not int
                or not 0 <= fixed_acceptance <= draft_tokens
            )
        ):
            raise ValueError("fixed acceptance must fit the verified draft block")
        accepted = (
            (drafts == targets[:, :draft_tokens]).int().cumprod(1).sum(1)
            if fixed_acceptance is None or dynamic_fixed
            else torch.full_like(positions, fixed_acceptance)
        )
        if dynamic_fixed:
            accepted = torch.where(fixed_acceptance >= 0, fixed_acceptance, accepted)
        bonus = targets.gather(1, accepted[:, None]).squeeze(1)
        columns = torch.arange(draft_tokens + 1, device=targets.device)[None, :]
        candidate = torch.cat((drafts, targets[:, -1:]), dim=1)
        tokens = torch.where(columns < accepted[:, None], candidate, bonus[:, None])
        return cls.from_accepted(tokens, accepted, positions, draft_tokens=draft_tokens)

    @classmethod
    def from_accepted(
        cls,
        tokens: torch.Tensor,
        accepted_drafts: torch.Tensor,
        positions: torch.Tensor,
        *,
        draft_tokens: int,
    ) -> Self:
        if (
            type(draft_tokens) is not int
            or draft_tokens < 0
            or positions.ndim != 1
            or tokens.shape != (len(positions), draft_tokens + 1)
            or accepted_drafts.shape != positions.shape
        ):
            raise ValueError("accepted GPU block does not match its draft/row geometry")
        accepted_drafts = accepted_drafts.long()
        return cls(
            tokens,
            accepted_drafts,
            (accepted_drafts + 1).int(),
            positions,
            draft_tokens,
        )

    def feedback(self) -> DeviceFeedback:
        bonus = self.tokens.gather(1, self.accepted_drafts[:, None]).squeeze(1)
        return DeviceFeedback(bonus.clone(), self.positions + self.counts)


def accepted_block(
    output: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: int,
    *,
    fixed_acceptance: int | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DeviceFeedback]:
    """Compatibility wrapper for runners adopting DeviceAcceptance separately."""
    if not draft_tokens:
        output = output.reshape(-1, 1)
    accepted = DeviceAcceptance.from_samples(
        output[:, :draft_tokens],
        output[:, draft_tokens:],
        positions,
        fixed_acceptance=fixed_acceptance,
    )
    return accepted.tokens, accepted.counts, accepted.feedback()


@dataclass(frozen=True, slots=True)
class ContinuationState:
    """An admission generation owning one row order, graph bucket and lease view."""

    sequence_ids: tuple[int, ...]
    shape: tuple[int, ...]
    inputs: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class GraphDeviceFeedback(DeviceFeedback):
    """Shared feedback for graphs with no additional model-specific payload."""

    continuation: ContinuationState


class CudaContinuation:
    """Validate retained graph ownership independently of model feedback payloads."""

    def __init__(self) -> None:
        self._active: ContinuationState | None = None

    def begin(
        self,
        sequence_ids: Iterable[int],
        shape: Sequence[int],
        inputs: Mapping[str, torch.Tensor],
    ) -> ContinuationState:
        ids = tuple(map(int, sequence_ids))
        if len(set(ids)) != len(ids) or not shape or shape[0] < len(ids):
            raise GraphInputError("continuation rows do not fit their graph bucket")
        if not all(isinstance(value, torch.Tensor) for value in inputs.values()):
            raise GraphInputError("continuations require graph-owned tensor inputs")
        state = ContinuationState(ids, tuple(shape), MappingProxyType(dict(inputs)))
        self._active = state
        return state

    def resume(
        self,
        sequence_ids: Iterable[int],
        state: ContinuationState,
        *,
        shape: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(state, ContinuationState) or state is not self._active:
            raise GraphInputError(
                "continuation no longer owns its admitted cache lease"
            )
        if tuple(map(int, sequence_ids)) != state.sequence_ids:
            raise GraphInputError("continuation changed its row ownership")
        if shape is not None and tuple(shape) != state.shape:
            raise GraphInputError("continuation changed its graph bucket")
        return dict(state.inputs)

    def drain(self, state: ContinuationState | None = None) -> None:
        if state is None or self._active is state:
            self._active = None


class DeviceCompletion:
    """One accepted step; CPU harvest never reruns model or acceptance work.

    Copies are queued before another replay may reuse graph buffers. The Engine
    retains the native writer lease until ``result`` has established completion.
    """

    @classmethod
    def from_acceptance(
        cls, acceptance: DeviceAcceptance, *, feedback=None, **kwargs
    ) -> Self:
        stats = kwargs.get("stats")
        if stats is not None and stats.draft_tokens != acceptance.draft_tokens:
            raise ValueError(
                "completion statistics do not match the accepted draft block"
            )
        return cls(
            acceptance.tokens,
            acceptance.counts,
            acceptance.feedback() if feedback is None else feedback,
            accepted_end_positions=acceptance.positions + acceptance.counts,
            **kwargs,
        )

    def __init__(
        self,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        feedback,
        *,
        rows: int,
        stats: SpeculativeStats | None = None,
        device: str | torch.device | None = None,
        plan: BatchPlan | None = None,
        logits: torch.Tensor | None = None,
        accepted_end_positions: torch.Tensor | None = None,
        state_checkpoints=None,
    ) -> None:
        self.accepted_end_positions = (
            feedback.positions
            if accepted_end_positions is None
            else accepted_end_positions
        )
        self.feedback = feedback.restrict(rows)
        self._rows, self._stats = rows, stats
        self._staging = TensorStager(device or tokens.device)
        self._tokens = self._staging.output("tokens", tokens)
        self._counts = self._staging.output("counts", counts)
        self._positions = self._staging.output("positions", feedback.positions - counts)
        self._logprobs = None
        self._top_ids = self._top_logprobs = None
        if plan is not None and plan.needs_logprobs:
            self._stage_scores(tokens, plan, logits)
        self._device = tokens.device
        self._event = None
        self._result = None
        self._state_checkpoints = None
        self._prepared_state_checkpoints = state_checkpoints
        if state_checkpoints is not None:
            self._state_checkpoints = state_checkpoints.stage(self._staging)
        self._record_event()

    def _record_event(self) -> None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self._device))
        self._event = event

    def attach_state_checkpoints(self, prepared_report) -> Self:
        """Queue a report before exposing this completion to the Engine.

        Snapshot copies must already follow model state commit on the same
        stream. Republishing the event includes their report readback; failure
        leaves this completion quarantined instead of exposing an older fence.
        """
        if self._result is not None:
            raise RuntimeError("cannot attach state checkpoints after result harvest")
        if self._state_checkpoints is not None:
            raise RuntimeError("completion already has a state checkpoint report")
        self._event = None
        self._prepared_state_checkpoints = prepared_report
        self._state_checkpoints = prepared_report.stage(self._staging)
        self._record_event()
        return self

    def _stage_scores(
        self, tokens: torch.Tensor, plan: BatchPlan, logits: torch.Tensor | None
    ) -> None:
        if logits is None:
            raise ValueError("requested logprobs require target logits")
        if logits.ndim == 2:
            logits = logits[:, None, :]
        width = tokens.shape[1]
        if logits.ndim != 3 or logits.shape[0] < self._rows or logits.shape[1] < width:
            raise ValueError("target logits must cover the accepted token grid")
        policies = plan.sampling_params
        self._requested = np.fromiter(
            (
                sample and params.logprobs
                for sample, params in zip(plan.sampling_mask, policies, strict=True)
            ),
            dtype=np.bool_,
        )
        if len(self._requested) != self._rows:
            raise ValueError("sampling policies must match active completion rows")
        self._score_rows = np.flatnonzero(self._requested)
        self._top_counts = np.fromiter(
            (
                min(policies[row].top_logprobs, logits.shape[-1])
                for row in self._score_rows
            ),
            dtype=np.uint32,
        )
        # Keep a fixed grid until harvest: accepted counts live on the device,
        # and CUDA boolean compaction would synchronize before the next launch.
        values, selected_tokens = logits[: self._rows, :width], tokens[: self._rows]
        if len(self._score_rows) != self._rows:
            row_indices = self._staging.device(
                "score_rows", self._score_rows, torch.long
            )
            values = values.index_select(0, row_indices)
            selected_tokens = selected_tokens.index_select(0, row_indices)
        values = values.float().log_softmax(dim=-1)
        selected = values.gather(-1, selected_tokens.long()[..., None]).squeeze(-1)
        self._logprobs = self._staging.output("logprobs", selected)
        top_k = int(max(self._top_counts, default=0))
        if top_k:
            top_values, top_ids = values.topk(top_k, dim=-1)
            self._top_ids = self._staging.output("top_ids", top_ids)
            self._top_logprobs = self._staging.output("top_logprobs", top_values)

    def _scored_output(
        self, output: ForwardOutput, counts: npt.NDArray[np.integer]
    ) -> ForwardOutput:
        from resin import ForwardOutput

        assert self._logprobs is not None
        width = self._logprobs.shape[1]
        scored_counts = counts.astype(np.uint32) * self._requested
        kept_counts = counts[self._score_rows].astype(np.uint32)
        keep = np.arange(width)[None, :] < kept_counts[:, None]
        columns: dict[str, Any] = {}
        if self._top_ids is not None:
            assert self._top_logprobs is not None
            top_counts = np.repeat(self._top_counts, kept_counts)
            top_keep = keep[..., None] & (
                np.arange(self._top_ids.shape[-1])[None, None, :]
                < self._top_counts[:, None, None]
            )
            columns = {
                "top_indptr": np.concatenate(
                    (np.zeros(1, np.uint32), top_counts.cumsum(dtype=np.uint32))
                ),
                "top_token_ids": self._top_ids.numpy()[top_keep].astype(np.uint32),
                "top_logprobs": self._top_logprobs.numpy()[top_keep],
            }
        return ForwardOutput(
            token_ids=output.token_ids,
            token_indptr=output.token_indptr,
            logprobs=self._logprobs.numpy()[keep],
            logprob_indptr=np.concatenate(
                (np.zeros(1, np.uint32), scored_counts.cumsum(dtype=np.uint32))
            ),
            **columns,
        )

    @property
    def positions(self) -> npt.NDArray[np.integer]:
        self.result()
        return self._positions[: self._rows].numpy()

    def result(self) -> ForwardOutput:
        if self._result is None:
            if self._event is None:
                raise RuntimeError(
                    "completion has an unfinished state checkpoint report"
                )
            self._event.synchronize()
            counts = self._counts[: self._rows].numpy()
            output = token_output(self._tokens[: self._rows].numpy(), counts=counts)
            if self._logprobs is not None:
                output = self._scored_output(output, counts)
            if self._state_checkpoints is not None:
                output = self._state_checkpoints.attach(output)
            self._result = output
            if self._stats is not None:
                self._stats.blocks += self._rows
                self._stats.drafts += self._rows * self._stats.draft_tokens
                emitted = int(counts.sum())
                self._stats.accepted += emitted - self._rows
                self._stats.emitted += emitted
        return self._result
