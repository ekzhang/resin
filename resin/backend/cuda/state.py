"""Execute native live-state restores and optional immutable checkpoint copies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

import numpy as np
import torch

from resin import BatchPhase, ForwardOutput, StateCheckpoint
from resin.backend.cuda.completion import DeviceCompletion
from resin.backend.cuda.failures import GraphInputError
from resin.backend.cuda.staging import TensorStager


@dataclass
class _CheckpointReport:
    records: tuple
    flags: torch.Tensor
    owner: object

    def stage(self, staging):
        return _StagedReport(
            self.records, staging.output("state_checkpoints", self.flags), self
        )


@dataclass
class _StagedReport:
    records: tuple
    flags: torch.Tensor
    owner: object

    def attach(self, output):
        # One native reservation covers every STATE group at a row/prefix.
        # A receipt certifies the complete snapshot, not an individual layer.
        snapshots = {}
        for record, materialized in zip(self.records, self.flags.tolist(), strict=True):
            previous = snapshots.get(record.reservation_id)
            snapshots[record.reservation_id] = (
                record,
                materialized and (previous is None or previous[1]),
            )
        records = tuple(
            record for record, materialized in snapshots.values() if materialized
        )
        return output.with_state_checkpoints((*output.state_checkpoints, *records))


class _StateExecution:
    def __init__(self, plan, arena):
        self.plan, self.arena = plan, arena
        self.staging = TensorStager(arena.device)
        self.groups = tuple(
            group for group in plan.cache_groups if group.state is not None
        )
        for group in self.groups:
            storage = arena.state_blocks[group.group_index]
            state = group.state

            def address(slab, slot, *, destination=False):
                if not (
                    int(destination) <= slab <= arena.layout.parent_capacity
                    and slab < storage.shape[0]
                    and 0 <= slot < storage.shape[1]
                ):
                    raise ValueError(
                        "state transfer address is outside its cache arena"
                    )

            for copy in state.restore_copies:
                if not 0 <= copy.row < plan.num_sequences:
                    raise ValueError("state restore row is outside its batch")
                address(copy.source_slab_id, copy.source_slotnum)
                address(
                    copy.destination_slab_id, copy.destination_slotnum, destination=True
                )
                if (copy.destination_slab_id, copy.destination_slotnum) != (
                    int(state.live_slab_ids[copy.row]),
                    int(state.live_slotnums[copy.row]),
                ):
                    raise ValueError(
                        "state restore destination is not its row's live slot"
                    )
            for slot in state.checkpoint_slots:
                if not 0 <= slot.row < plan.num_sequences:
                    raise ValueError("state checkpoint row is outside its batch")
                address(
                    int(state.live_slab_ids[slot.row]),
                    int(state.live_slotnums[slot.row]),
                )
                address(
                    slot.destination_slab_id, slot.destination_slotnum, destination=True
                )

    def restore(self):
        from resin.backend.cuda.state_triton import copy_state

        for group in self.groups:
            copies = group.state.restore_copies
            if not copies:
                continue
            addresses = self.staging.device(
                f"state.{group.group_index}.restore",
                [
                    (
                        copy.source_slab_id,
                        copy.source_slotnum,
                        copy.destination_slab_id,
                        copy.destination_slotnum,
                    )
                    for copy in copies
                ],
                torch.int64,
            )
            copy_state(self.arena.state_blocks[group.group_index], addresses)

    def finish(self, result):
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
                        int(state.live_slab_ids[slot.row]),
                        int(state.live_slotnums[slot.row]),
                        slot.destination_slab_id,
                        slot.destination_slotnum,
                    )
                    for slot in slots
                ],
                torch.int64,
            )
            candidates = self.staging.device(
                f"state.{group.group_index}.candidates",
                [(slot.row, slot.prefix_length) for slot in slots],
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
            records.extend(
                StateCheckpoint(
                    reservation_id=slot.reservation_id, prefix_length=slot.prefix_length
                )
                for slot in slots
            )
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
