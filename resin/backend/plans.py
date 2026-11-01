"""Small framework-neutral plan adapters for readable model runners."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from resin._resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupPlan,
    CacheLayout,
    CacheStatePlan,
)


@dataclass(frozen=True, slots=True)
class BatchView:
    """Cached NumPy views owned by an immutable, validated native plan."""

    _plan: BatchPlan
    phase: BatchPhase = field(init=False)
    sequence_ids: npt.NDArray[np.uint64] = field(init=False)
    context_lengths: npt.NDArray[np.uint32] = field(init=False)
    token_ids: npt.NDArray[np.uint32] = field(init=False)
    token_indptr: npt.NDArray[np.uint32] = field(init=False)
    positions: npt.NDArray[np.uint32] = field(init=False)
    sampling_mask: npt.NDArray[np.bool_] = field(init=False)
    cache_groups: tuple[CacheGroupPlan, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self._plan, BatchPlan):
            raise TypeError("BatchView requires a native BatchPlan")
        # Native construction and archive restore validate these fields. Keeping
        # their owner prevents callers from mixing unrelated arrays into a view.
        for name in (
            "phase",
            "sequence_ids",
            "context_lengths",
            "token_ids",
            "token_indptr",
            "positions",
            "sampling_mask",
            "cache_groups",
        ):
            object.__setattr__(self, name, getattr(self._plan, name))

    @classmethod
    def from_plan(cls, plan: BatchPlan) -> BatchView:
        return cls(plan)

    @property
    def num_sequences(self) -> int:
        return len(self.sequence_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def query_lengths(self) -> npt.NDArray[np.uint32]:
        return np.diff(self.token_indptr)

    @property
    def post_lengths(self) -> npt.NDArray[np.uint32]:
        return self.context_lengths + self.query_lengths

    @property
    def last_token_indices(self) -> npt.NDArray[np.uint32]:
        return self.token_indptr[1:] - 1

    def token_slice(self, row: int) -> slice:
        return slice(*self._plan.token_range(row))


@dataclass(frozen=True, slots=True)
class CacheGroupView:
    """Validated native cache addresses for one ordered cache group."""

    group_index: int
    blocks_per_slab: int
    block_indptr: npt.NDArray[np.uint32]
    block_slab_ids: npt.NDArray[np.uint32]
    block_slotnums: npt.NDArray[np.uint32]
    block_table_start_indices: npt.NDArray[np.uint32]
    append_slab_ids: npt.NDArray[np.uint32]
    append_slotnums: npt.NDArray[np.uint32]
    append_block_offsets: npt.NDArray[np.uint32]
    state: CacheStatePlan | None

    @classmethod
    def from_plan(
        cls,
        batch: BatchView,
        *,
        group_index: int,
        blocks_per_slab: int | None = None,
        parent_capacity: int | None = None,
        block_tokens: int | None = None,
    ) -> CacheGroupView:
        """Check the batch's native group against the physical cache layout."""

        if not isinstance(batch, BatchView):
            raise TypeError("cache group views require a native-backed BatchView")
        if not 0 <= group_index < len(batch.cache_groups):
            raise ValueError(f"cache group {group_index} is outside the native plan")
        native = batch.cache_groups[group_index]
        packing = native.blocks_per_slab
        block_slab_ids = native.block_slab_ids
        append_slab_ids = native.append_slab_ids
        append_offsets = native.append_block_offsets
        if blocks_per_slab is not None and packing != blocks_per_slab:
            raise ValueError(f"cache group {group_index} has the wrong slab packing")
        if (block_tokens is not None and np.any(append_offsets >= block_tokens)) or (
            parent_capacity is not None
            and (
                np.any(block_slab_ids > parent_capacity)
                or np.any(append_slab_ids > parent_capacity)
            )
        ):
            raise ValueError(
                f"cache group {group_index} has a physical address outside its layout"
            )
        return cls(
            group_index,
            packing,
            native.block_indptr,
            block_slab_ids,
            native.block_slotnums,
            native.block_table_start_indices,
            append_slab_ids,
            native.append_slotnums,
            append_offsets,
            native.state,
        )

    def block_slice(self, row: int) -> slice:
        start, end = map(int, self.block_indptr[row : row + 2])
        return slice(start, end)

    def block_slabs(self, row: int) -> npt.NDArray[np.uint32]:
        return self.block_slab_ids[self.block_slice(row)]

    def block_slots(self, row: int) -> npt.NDArray[np.uint32]:
        return self.block_slotnums[self.block_slice(row)]

    @property
    def packed_block_ids(self) -> npt.NDArray[np.uint64]:
        return (
            self.block_slab_ids.astype(np.uint64) * self.blocks_per_slab
            + self.block_slotnums
        )

    @property
    def packed_append_ids(self) -> npt.NDArray[np.uint64]:
        return (
            self.append_slab_ids.astype(np.uint64) * self.blocks_per_slab
            + self.append_slotnums
        )


@dataclass(frozen=True, slots=True)
class PagedCachePlan:
    """Validated physical cache leases for one native batch plan."""

    batch: BatchView
    layout: CacheLayout
    groups: tuple[CacheGroupView, ...]

    @classmethod
    def from_plan(
        cls,
        plan: BatchPlan,
        layout: CacheLayout,
        *,
        decode_lookahead: int = 0,
    ) -> PagedCachePlan:
        """Validate group order, physical bounds, extents, and append ownership.

        Decode tables describe the Engine's frozen lease, which can include
        future write destinations. Readable history still ends at the actual
        device position. Prefill tables describe the exact active range.
        """

        from resin._resin import CacheGroupFamily

        if decode_lookahead < 0:
            raise ValueError("decode_lookahead cannot be negative")
        if not isinstance(layout, CacheLayout):
            raise TypeError("paged cache plans require a native CacheLayout")
        batch = BatchView.from_plan(plan)
        if len(batch.cache_groups) != len(layout.groups):
            raise ValueError("BatchPlan cache groups do not match CacheLayout")

        query_lengths = batch.query_lengths.astype(np.int64)
        row_ids = np.repeat(np.arange(batch.num_sequences), query_lengths)
        positions = batch.positions.astype(np.int64)
        lookahead = decode_lookahead if batch.phase == BatchPhase.DECODE else 0
        # Native fields are u32, but a leased decode span can extend beyond u32.
        # Keep even oversized caller-supplied lookahead from wrapping arithmetic.
        post_lengths = batch.context_lengths.astype(np.int64) + query_lengths
        arithmetic = (
            object
            if lookahead > np.iinfo(np.int64).max - int(post_lengths.max())
            else np.int64
        )
        write_tokens = query_lengths.astype(arithmetic) + lookahead
        leased = batch.context_lengths.astype(arithmetic) + write_tokens

        groups = []
        for index, descriptor in enumerate(layout.groups):
            block_tokens = descriptor.block_tokens
            group = CacheGroupView.from_plan(
                batch,
                group_index=index,
                blocks_per_slab=descriptor.blocks_per_slab,
                parent_capacity=layout.parent_capacity,
                block_tokens=block_tokens,
            )
            window_blocks = descriptor.window_resident_block_count
            if descriptor.family == CacheGroupFamily.STATE:
                if group.state is None:
                    raise ValueError(f"cache group {index} requires private live state")
                if (
                    np.any(np.diff(group.block_indptr) != 1)
                    or np.any(group.block_table_start_indices != 0)
                    or not np.array_equal(
                        group.append_slab_ids, group.block_slab_ids[row_ids]
                    )
                    or not np.array_equal(
                        group.append_slotnums, group.block_slotnums[row_ids]
                    )
                ):
                    raise ValueError(
                        f"cache group {index} must address its private live state"
                    )
                groups.append(group)
                continue
            if group.state is not None:
                raise ValueError(f"cache group {index} is not a STATE group")
            invalid_span = np.zeros(batch.num_sequences, dtype=np.bool_)
            maximum = 0
            if window_blocks is not None:
                assert descriptor.window_tokens is not None
                assert descriptor.read_window_tokens is not None
                maximum = descriptor.window_tokens - descriptor.read_window_tokens + 1
                invalid_span = write_tokens > maximum
            last_blocks = (leased - 1) // block_tokens
            if window_blocks is not None:
                expected_first = np.maximum(0, last_blocks + 1 - window_blocks)
            else:
                expected_first = np.zeros(batch.num_sequences, dtype=np.int64)
            invalid_range = (group.block_table_start_indices != expected_first) | (
                np.diff(group.block_indptr) != last_blocks - expected_first + 1
            )
            if batch.phase == BatchPhase.DECODE:
                # The native table is the lease authority, including any tail
                # reserved for queued steps. Validate coverage without requiring
                # the runner to reproduce the Engine's reservation policy.
                expected_first = group.block_table_start_indices.astype(np.int64)
                count = np.diff(group.block_indptr).astype(np.int64)
                table_end = expected_first + count
                invalid_range = table_end <= last_blocks
                if window_blocks is None:
                    invalid_range |= expected_first != 0
                else:
                    assert descriptor.read_window_tokens is not None
                    earliest_read = np.maximum(
                        0,
                        batch.context_lengths.astype(np.int64)
                        - (descriptor.read_window_tokens - 1),
                    )
                    invalid_range |= (
                        expected_first != np.maximum(0, table_end - window_blocks)
                    ) | (expected_first > earliest_read // block_tokens)

            logical_blocks = positions // block_tokens
            # Layout-inconsistent extents must be rejected before indexing.
            retained = (logical_blocks >= expected_first[row_ids]) & ~invalid_range[
                row_ids
            ]
            retained_rows = row_ids[retained]
            columns = (logical_blocks[retained] - expected_first[retained_rows]).astype(
                np.intp
            )
            addresses = group.block_indptr[:-1][retained_rows] + columns
            invalid_append = (
                group.append_slab_ids[retained] != group.block_slab_ids[addresses]
            ) | (group.append_slotnums[retained] != group.block_slotnums[addresses])
            invalid_append |= (
                group.append_block_offsets[retained]
                != positions[retained] % block_tokens
            )
            invalid_append_rows = np.zeros(batch.num_sequences, dtype=np.bool_)
            invalid_append_rows[retained_rows[invalid_append]] = True
            # Keep the original row-first failure order across all three checks.
            invalid_rows = np.flatnonzero(
                invalid_span | invalid_range | invalid_append_rows
            )
            if len(invalid_rows):
                row = int(invalid_rows[0])
                if invalid_span[row]:
                    raise ValueError(
                        f"cache group {index} row {row} write span {write_tokens[row]} "
                        f"exceeds {maximum}-token Window read/write capacity"
                    )
                if invalid_range[row]:
                    raise ValueError(
                        f"cache group {index} row {row} has an invalid active range"
                    )
                raise ValueError(
                    f"cache group {index} row {row} append target is not row-owned"
                )
            groups.append(group)
        return cls(batch, layout, tuple(groups))

    @property
    def phase(self) -> BatchPhase:
        return self.batch.phase

    @property
    def rows(self) -> int:
        return self.batch.num_sequences

    @property
    def token_ids(self) -> npt.NDArray[np.uint32]:
        return self.batch.token_ids

    @property
    def positions(self) -> npt.NDArray[np.uint32]:
        return self.batch.positions

    @property
    def token_indptr(self) -> npt.NDArray[np.uint32]:
        return self.batch.token_indptr

    @property
    def context_lengths(self) -> npt.NDArray[np.uint32]:
        return self.batch.context_lengths
