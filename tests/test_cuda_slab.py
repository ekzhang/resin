from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from resin import CacheGroupLayout, CacheLayout
from resin.backend.cuda.slab import GraphPageTableArena
from resin.backend.plans import PagedCachePlan


@pytest.mark.parametrize("private_scratch", [False, True])
def test_graph_replay_pads_unused_rows_with_the_capture_scratch_parents(
    private_scratch,
):
    class Arena(GraphPageTableArena):
        device = torch.device("cpu")
        layout = CacheLayout(
            parent_capacity=10, groups=[CacheGroupLayout(128, blocks_per_slab=2)] * 2
        )

        def _make_graph_cache(self, groups):
            return SimpleNamespace(groups=groups)

        def _graph_padding_address(self, index: int, row: int) -> tuple[int, int]:
            return (
                (self.layout.parent_capacity + index + 1, 0)
                if private_scratch
                else (0, 0)
            )

    arena = Arena()
    cache = arena.graph_cache(2, 256)
    plan = cast(
        PagedCachePlan,
        SimpleNamespace(
            rows=1,
            groups=tuple(
                SimpleNamespace(
                    block_slabs=lambda row: np.array([7], dtype=np.uint32),
                    block_slots=lambda row: np.array([1], dtype=np.uint32),
                    block_table_start_indices=np.array([3], dtype=np.uint32),
                    block_indptr=np.array([0, 1], dtype=np.uint32),
                )
                for _ in range(2)
            ),
        ),
    )

    assert arena.graph_can_fit_plan(plan, (2, 256))
    assert not arena.graph_can_fit_plan(plan, (0, 256))
    for group in plan.groups:
        group.block_indptr[-1] = 3
    assert not arena.graph_can_fit_plan(plan, (2, 256))
    for group in plan.groups:
        group.block_indptr[-1] = 1

    capture = arena.graph_shape_capture_inputs(2, 256)
    replay = arena.graph_plan_inputs(cache, plan)

    for group in range(2):
        prefix = f"cache.{group}"
        parent, slot = arena._graph_padding_address(group, 1)
        np.testing.assert_array_equal(
            replay[f"{prefix}.slabs"], [[7, 0], [parent, parent]]
        )
        np.testing.assert_array_equal(replay[f"{prefix}.slots"], [[1, 0], [0, 0]])
        np.testing.assert_array_equal(replay[f"{prefix}.first"], [3, 0])
        # Only whole padded requests use scratch storage. A real request's
        # absent history columns must remain zero rather than becoming valid KV.
        assert np.asarray(replay[f"{prefix}.slabs"])[1].tolist() == list(
            np.asarray(capture[f"{prefix}.slabs"])[-2:]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_retained_page_tables_follow_actual_positions_across_graph_replays():
    from resin.backend.cuda.completion import CudaContinuation
    from resin.backend.cuda.slab import DevicePageGroup, slab_locations

    # A frozen lease includes future pages; logical order need not match physical
    # order. Row 2 is a retained WINDOW suffix and row 3 is graph padding.
    slabs = torch.tensor(
        [[7, 13, 5], [11, 3, 19], [17, 2, 23], [0, 0, 0]], device="cuda"
    )
    slots = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0]], device="cuda")
    first = torch.tensor([0, 0, 1, 0], device="cuda")
    positions = torch.tensor([126, 127, 255, 1], device="cuda")
    group = DevicePageGroup(slabs, slots, first)
    manager = CudaContinuation()
    state = manager.begin(
        [10, 20, 30],
        (4, 512),
        {"positions": positions, "slabs": slabs, "slots": slots, "first": first},
    )
    result = ()

    def launch():
        nonlocal result
        result = slab_locations(
            (group,), 0, positions[:, None] + torch.arange(4, device="cuda"), 128
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    source_slabs, source_slots, source_first = slabs.cpu(), slots.cpu(), first.cpu()
    for anchors in (
        [126, 127, 255, 1],
        [127, 130, 256, 1],
        [131, 131, 260, 1],
        [254, 255, 383, 1],
    ):
        owned = manager.resume([10, 20, 30], state)
        owned["positions"].copy_(torch.tensor(anchors, device="cuda"))
        graph.replay()
        expected = [torch.zeros(4, 4, dtype=torch.long) for _ in range(3)]
        for row, anchor in enumerate(anchors):
            for query in range(4):
                position = anchor + query
                column = position // 128 - int(source_first[row])
                expected[0][row, query] = source_slabs[row, column]
                expected[1][row, query] = source_slots[row, column]
                expected[2][row, query] = position % 128
        for actual, reference in zip(result, expected, strict=True):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
    # Reacquisition with the same row IDs invalidates the previous lease owner.
    manager.begin([10, 20, 30], (4, 512), state.inputs)
    with pytest.raises(ValueError, match="cache lease"):
        manager.resume([10, 20, 30], state)
