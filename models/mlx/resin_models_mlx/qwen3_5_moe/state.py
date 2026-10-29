"""Live DeltaNet state kept on the GPU between steps.

Each running request's recurrent and convolution state stays in the MLX arrays
the model returned for its last step, so an unchanged decode batch feeds them
straight back without touching the arena. The Engine's STATE group still
decides when state is restored from a prefix snapshot, and where snapshots are
published; only those transitions copy through the arena. Requests leave when
the Engine releases them.
"""

from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx
import numpy as np

from resin.backend.plans import PagedCachePlan
from resin_models_mlx.qwen3_5_moe.cache import STATE_GROUP, CacheArena
from resin_models_mlx.qwen3_5_moe.model import LinearState, Qwen35MoeModel


def _rows(state: LinearState) -> int:
    return state.recurrent[0].shape[0]


def _take(state: LinearState, indices: list[int]) -> LinearState:
    if indices == list(range(_rows(state))):
        return state
    selected = mx.array(indices)
    return LinearState(
        tuple(mx.take(value, selected, axis=0) for value in state.recurrent),
        tuple(mx.take(value, selected, axis=0) for value in state.convolution),
    )


def _concatenate(parts: list[LinearState]) -> LinearState:
    if len(parts) == 1:
        return parts[0]
    return LinearState(
        tuple(
            mx.concatenate(list(layer)) for layer in zip(*(p.recurrent for p in parts))
        ),
        tuple(
            mx.concatenate(list(layer))
            for layer in zip(*(p.convolution for p in parts))
        ),
    )


class LiveStates:
    """Per-request DeltaNet state, indexed into the batch arrays that hold it."""

    def __init__(self, arena: CacheArena, model: Qwen35MoeModel) -> None:
        self._arena = arena
        self._model = model
        # sequence ID -> (tokens processed, batch state, row in that batch)
        self._rows: dict[int, tuple[int, LinearState, int]] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def gather(self, plan: PagedCachePlan, rows: np.ndarray) -> LinearState:
        """State before each of ``rows`` runs, in plan row order."""

        batch = plan.batch
        restore = plan.groups[STATE_GROUP].state
        sources = {
            int(row): (int(slab), int(slot))
            for row, slab, slot in (restore.restore_copies if restore else ())
        }
        snapshots, fresh = [], []
        held: dict[int, tuple[LinearState, list[int], list[int]]] = {}
        for position, row in enumerate(rows.tolist()):
            context = int(batch.context_lengths[row])
            if row in sources:
                snapshots.append((position, *sources[row]))
            elif context == 0:
                fresh.append(position)
            else:
                sequence = int(batch.sequence_ids[row])
                entry = self._rows.get(sequence)
                if entry is None or entry[0] != context:
                    raise RuntimeError(
                        f"sequence {sequence} has no live recurrent state at "
                        f"position {context}"
                    )
                _, state, index = entry
                group = held.setdefault(id(state), (state, [], []))
                group[1].append(position)
                group[2].append(index)

        parts, order = [], []
        for state, positions, indices in held.values():
            parts.append(_take(state, indices))
            order += positions
        if snapshots:
            parts.append(
                self._arena.read_states(
                    np.array([self._arena.state_start(s, t) for _, s, t in snapshots])
                )
            )
            order += [position for position, *_ in snapshots]
        if fresh:
            parts.append(self._model.initial_state(len(fresh)))
            order += fresh
        state = _concatenate(parts)
        if order != list(range(len(order))):
            state = _take(state, np.argsort(order).tolist())
        return state

    def commit(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        ends: np.ndarray,
        state: LinearState,
    ) -> tuple[int, ...]:
        """Hold each row's state after ``ends`` tokens and publish due snapshots.

        Returns the reservation IDs of the snapshots written.
        """

        sequences = plan.batch.sequence_ids
        for index, row in enumerate(rows.tolist()):
            self._rows[int(sequences[row])] = (int(ends[index]), state, index)
        checkpoints = plan.groups[STATE_GROUP].state
        positions = {row: index for index, row in enumerate(rows.tolist())}
        written, starts, indices = [], [], []
        for reservation, row, prefix, slab, slot in (
            checkpoints.checkpoint_slots if checkpoints else ()
        ):
            index = positions.get(int(row))
            # A snapshot is valid only at its exact prefix.
            if index is not None and int(ends[index]) == int(prefix):
                written.append(int(reservation))
                starts.append(self._arena.state_start(int(slab), int(slot)))
                indices.append(index)
        if written:
            self._arena.write_states(starts, _take(state, indices))
        return tuple(written)

    def release(self, sequence_ids: Iterable[int]) -> None:
        for sequence in sequence_ids:
            self._rows.pop(int(sequence), None)


__all__ = ["LiveStates"]
