"""Execute native live-state restores and optional immutable checkpoint copies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

import numpy as np
import torch

from resin import BatchPhase, ForwardOutput
from resin.backend.cuda.completion import DeviceCompletion
from resin.backend.cuda.failures import GraphInputError
from resin.backend.cuda.staging import TensorStager


@dataclass
class _CheckpointReport:
    reservation_ids: tuple[int, ...]
    flags: torch.Tensor
    owner: object

    def stage(self, staging: TensorStager) -> _CheckpointReport:
        return _CheckpointReport(
            self.reservation_ids, staging.output("state_checkpoints", self.flags), self
        )

    def attach(self, output: ForwardOutput) -> ForwardOutput:
        # One native reservation covers every STATE group at a row/prefix.
        # A receipt certifies the complete snapshot, not an individual layer.
        snapshots = dict.fromkeys(self.reservation_ids, True)
        for reservation_id, materialized in zip(
            self.reservation_ids, self.flags.tolist(), strict=True
        ):
            snapshots[reservation_id] &= materialized
        records = tuple(key for key, ready in snapshots.items() if ready)
        return output.with_state_checkpoints((*output.state_checkpoints, *records))


class _StateExecution:
    def __init__(self, plan, arena) -> None:
        self.plan, self.arena = plan, arena
        self.staging = TensorStager(arena.device)
        self.groups = tuple(
            group for group in plan.cache_groups if group.state is not None
        )
        for group in self.groups:
            storage = arena.state_blocks[group.group_index]
            state = group.state

            def address(
                slab: int | np.integer,
                slot: int | np.integer,
                *,
                destination: bool = False,
            ) -> None:
                if not (
                    int(destination) <= slab <= arena.layout.parent_capacity
                    and slab < storage.shape[0]
                    and 0 <= slot < storage.shape[1]
                ):
                    raise ValueError(
                        "state transfer address is outside its cache arena"
                    )

            for row, source_slab, source_slot in state.restore_copies:
                if not 0 <= row < plan.num_sequences:
                    raise ValueError("state restore row is outside its batch")
                address(source_slab, source_slot)
                address(
                    group.block_slab_ids[row],
                    group.block_slotnums[row],
                    destination=True,
                )
            for _, row, _, slab, slot in state.checkpoint_slots:
                if not 0 <= row < plan.num_sequences:
                    raise ValueError("state checkpoint row is outside its batch")
                address(group.block_slab_ids[row], group.block_slotnums[row])
                address(slab, slot, destination=True)

    def restore(self) -> None:
        from resin.backend.cuda.state_triton import copy_state

        for group in self.groups:
            copies = group.state.restore_copies
            if not copies:
                continue
            addresses = self.staging.device(
                f"state.{group.group_index}.restore",
                [
                    (
                        slab,
                        slot,
                        group.block_slab_ids[row],
                        group.block_slotnums[row],
                    )
                    for row, slab, slot in copies
                ],
                torch.int64,
            )
            copy_state(self.arena.state_blocks[group.group_index], addresses)

    def finish(
        self, result: DeviceCompletion | ForwardOutput
    ) -> DeviceCompletion | ForwardOutput:
        from resin.backend.cuda.state_triton import copy_state

        if isinstance(result, DeviceCompletion):
            ends = result.accepted_end_positions
        elif isinstance(result, ForwardOutput):
            counts = (
                np.diff(self.plan.token_indptr)
                if self.plan.phase == BatchPhase.PREFILL
                else np.diff(result.token_indptr)
            )
            ends = self.staging.device(
                "state.ends",
                self.plan.context_lengths.astype(np.int64) + counts,
                torch.int64,
            )
        else:
            raise TypeError(
                "state execution requires ForwardOutput or DeviceCompletion"
            )
        records, flags = [], []
        for group in self.groups:
            state = group.state
            slots = state.checkpoint_slots
            if not slots:
                continue
            addresses = self.staging.device(
                f"state.{group.group_index}.snapshots",
                [
                    (
                        group.block_slab_ids[row],
                        group.block_slotnums[row],
                        slab,
                        slot,
                    )
                    for _, row, _, slab, slot in slots
                ],
                torch.int64,
            )
            candidates = self.staging.device(
                f"state.{group.group_index}.candidates",
                [(row, prefix) for _, row, prefix, _, _ in slots],
                torch.int64,
            )
            flags.append(
                copy_state(
                    self.arena.state_blocks[group.group_index],
                    addresses,
                    candidates=candidates,
                    ends=ends,
                )
            )
            records.extend(reservation_id for reservation_id, *_ in slots)
        # Retain restore staging even when there was no optional snapshot space.
        materialized = (
            torch.cat(flags)
            if flags
            else torch.empty(0, dtype=torch.bool, device=self.arena.device)
        )
        report = _CheckpointReport(tuple(records), materialized, self)
        if isinstance(result, DeviceCompletion):
            result.attach_state_checkpoints(report)
            return result
        staged = report.stage(self.staging)
        torch.cuda.current_stream(self.arena.device).synchronize()
        return staged.attach(result)


def execute_state(function):
    """Apply plan-owned state transfers once around a runner submission.

    Nested fallback calls share the outer execution. The native continuation
    clears restore instructions; snapshot copies run only when the accepted
    cache prefix exactly matches a reserved checkpoint, before live state is
    reused by another submission.
    """

    @wraps(function)
    def wrapped(self, plan, *args, **kwargs):
        if (
            plan is None
            or getattr(self, "_state_execution_depth", 0)
            or not any(
                group.state is not None
                and (group.state.restore_copies or group.state.checkpoint_slots)
                for group in getattr(plan, "cache_groups", ())
            )
        ):
            return function(self, plan, *args, **kwargs)
        self._check_cuda_graphs(plan.phase)
        execution = _StateExecution(plan, self._cache)
        self._state_execution_depth = 1
        try:
            with self._cuda_failure, torch.cuda.device(self.device):
                try:
                    execution.restore()
                    return execution.finish(function(self, plan, *args, **kwargs))
                except GraphInputError as error:
                    if any(group.state.restore_copies for group in execution.groups):
                        # Model validation may reject after our restore launch.
                        # Quiesce and quarantine instead of releasing its lease
                        # through the ordinary pre-launch rejection exception.
                        raise RuntimeError(
                            "state restore was submitted before graph input rejection"
                        ) from error
                    raise
        finally:
            self._state_execution_depth = 0

    return wrapped
