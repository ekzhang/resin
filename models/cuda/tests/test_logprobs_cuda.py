"""Sparse CUDA scores keep raw semantics and fixed-shape graph ownership."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def score_logits(*args, **kwargs):
    # CPU collection must not import the CUDA-only DSL runtime.
    from resin_models_cuda._scoring_cutedsl import score_logits as implementation

    return implementation(*args, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [0, 1, 5, 32, 65])
def test_raw_scores_match_independent_normalization(dtype, top_k):
    generator = torch.Generator("cuda").manual_seed(407)
    storage = torch.randn(
        (4, 3, 2 * 8193 + 1), generator=generator, dtype=dtype, device="cuda"
    )
    logits = storage[..., 1::2]
    ids = torch.tensor([[3, 0, 5], [2, 4, 7], [8, 3, 6], [1, 8, 2]], device="cuda")
    mask = torch.tensor([True, False, True, True], device="cuda")
    counts = torch.tensor([1, 3, 2, 3], dtype=torch.int32, device="cuda")
    top_counts = torch.tensor(
        [top_k, 0, top_k // 2, 1 if top_k else 0], dtype=torch.int32, device="cuda"
    )
    target_rows = torch.tensor([0, 0, 6, 7, 9, 10], device="cuda")
    target_ids = torch.tensor([2, 2, 4, 5, 1, 2], device="cuda")
    scores, top_ids, top_scores, targets = score_logits(
        logits,
        ids,
        score_mask=mask,
        valid_counts=counts,
        top_counts=top_counts,
        max_top_k=top_k,
        target_rows=target_rows,
        target_ids=target_ids,
    )
    reference = logits.double().log_softmax(-1)
    expected = reference.gather(-1, ids[..., None]).squeeze(-1).float()
    active = mask[:, None] & (torch.arange(3, device="cuda")[None, :] < counts[:, None])
    torch.testing.assert_close(
        scores, torch.where(active, expected, 0), atol=3e-6, rtol=0
    )
    flat = reference.reshape(-1, reference.shape[-1])
    expected_targets = flat[target_rows, target_ids].float()
    target_active = active.reshape(-1)[target_rows]
    torch.testing.assert_close(
        targets, torch.where(target_active, expected_targets, 0), atol=3e-6, rtol=0
    )
    for request in range(4):
        for column in range(3):
            count = int(top_counts[request]) if active[request, column] else 0
            if count:
                selected = top_ids[request, column, :count]
                torch.testing.assert_close(
                    top_scores[request, column, :count],
                    reference[request, column, selected].float(),
                    atol=3e-6,
                    rtol=0,
                )
                torch.testing.assert_close(
                    top_scores[request, column, :count],
                    reference[request, column].topk(count).values.float(),
                    atol=3e-6,
                    rtol=0,
                )
                assert selected.unique().numel() == count
            assert (top_ids[request, column, count:] == -1).all()
            assert (top_scores[request, column, count:] == 0).all()


def test_strided_integer_metadata_and_per_position_policy():
    logits = torch.randn((3, 2, 8193), device="cuda", dtype=torch.bfloat16)
    selected = torch.tensor(
        [[3, 0, 5, 0], [2, 0, 4, 0], [8, 0, 3, 0]], device="cuda", dtype=torch.int32
    )[:, ::2]
    mask = torch.tensor(
        [
            [True, False, False, False],
            [True, False, True, False],
            [False, False, True, False],
        ],
        device="cuda",
    )[:, ::2]
    counts = torch.tensor([2, 99, 1, 99, 2, 99], device="cuda")[::2]
    tops = torch.tensor([[5, 0, 0, 0], [1, 0, 32, 0], [0, 0, 5, 0]], device="cuda")[
        :, ::2
    ]
    rows = torch.tensor([0, 0, 1, 2, 5, -1, 6, 0, 1], device="cuda", dtype=torch.int32)
    tokens = torch.tensor(
        [7, 7, 8, 11, 8192, 0, 0, 8193, 8193], device="cuda", dtype=torch.int32
    )
    chosen, ids, scores, targets = score_logits(
        logits,
        selected,
        score_mask=mask,
        valid_counts=counts,
        top_counts=tops,
        max_top_k=32,
        target_rows=rows,
        target_ids=tokens,
    )
    reference = logits.double().log_softmax(-1)
    active = mask & (torch.arange(2, device="cuda")[None, :] < counts[:, None])
    torch.testing.assert_close(
        chosen,
        torch.where(
            active, reference.gather(-1, selected.long()[..., None]).squeeze(-1), 0
        ).float(),
        atol=3e-6,
        rtol=0,
    )
    expected = torch.stack(
        (
            reference[0, 0, 7],
            reference[0, 0, 7],
            reference.new_zeros(()),
            reference[1, 0, 11],
            reference[2, 1, 8192],
            reference.new_full((), float("nan")),
            reference.new_full((), float("nan")),
            reference.new_full((), float("nan")),
            reference.new_zeros(()),
        )
    ).float()
    torch.testing.assert_close(targets, expected, atol=3e-6, rtol=0, equal_nan=True)
    for row in range(3):
        for column in range(2):
            count = int(tops[row, column]) if active[row, column] else 0
            torch.testing.assert_close(
                scores[row, column, :count],
                reference[row, column].topk(count).values.float(),
                atol=3e-6,
                rtol=0,
            )
            assert ids[row, column, :count].unique().numel() == count
            assert (ids[row, column, count:] == -1).all()
            assert (scores[row, column, count:] == 0).all()


@pytest.mark.parametrize("top_k", [65, 1000, 2048, 2051])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_large_top_count_and_ties_have_unique_ids(top_k, dtype):
    logits = torch.zeros((2, 2051), dtype=dtype, device="cuda")
    selected = torch.tensor([4, 9], device="cuda")
    _, top_ids, scores, _ = score_logits(logits, selected, max_top_k=top_k)
    assert top_ids.shape == (2, top_k)
    torch.testing.assert_close(
        scores,
        torch.full_like(scores, -torch.log(torch.tensor(2051.0)).item()),
        atol=2e-6,
        rtol=0,
    )
    for row in top_ids:
        assert row.unique().numel() == top_k
        assert (row < 2051).all() and (row >= 0).all()


def test_native_top_scores_preserve_nonfinite_and_mask_semantics():
    logits = torch.randn((5, 8193), dtype=torch.float32, device="cuda")
    logits[0, :4096] = -float("inf")
    logits[1, 0] = float("nan")
    logits[2, 0] = float("inf")
    logits[3, :] = -float("inf")
    logits[4, 0] = float("inf")
    selected = torch.full((5,), 8192, dtype=torch.int64, device="cuda")
    mask = torch.tensor([True, True, True, True, False], device="cuda")
    scores, ids, tops, _ = score_logits(
        logits, selected, score_mask=mask, max_top_k=1000
    )
    reference = logits.double().log_softmax(-1)
    torch.testing.assert_close(scores[0], reference[0, 8192].float(), atol=2e-6, rtol=0)
    torch.testing.assert_close(tops[0], reference[0, ids[0]].float(), atol=2e-6, rtol=0)
    assert ids[0].unique().numel() == 1000
    assert scores[1:4].isnan().all() and tops[1:4].isnan().all()
    assert scores[4] == 0 and (ids[4] == -1).all() and (tops[4] == 0).all()


def test_centered_scores_preserve_large_offsets_and_invalid_distributions():
    logits = torch.tensor(
        [
            [1e4, 1e4 + 1, 1e4 - 1],
            [0, -float("inf"), 1],
            [float("inf"), 1, 2],
            [float("nan"), 1, 2],
            [-float("inf")] * 3,
        ],
        device="cuda",
    )
    selected = torch.tensor([1, 1, 1, 1, 1], device="cuda")
    scores, _, _, _ = score_logits(logits, selected)
    reference = (
        logits.double().log_softmax(-1).gather(1, selected[:, None])[:, 0].float()
    )
    torch.testing.assert_close(scores, reference, atol=2e-6, rtol=0, equal_nan=True)


def test_an_empty_tile_does_not_contaminate_an_otherwise_finite_row():
    logits = torch.full((2, 8193), -float("inf"), device="cuda")
    logits[0, 8192] = 0
    logits[1, 4096:4100] = torch.tensor([1, 2, 3, 4], device="cuda")
    selected = torch.tensor([8192, 4099], device="cuda")
    scores, _, _, targets = score_logits(
        logits,
        selected,
        target_rows=torch.tensor([0, 1], device="cuda"),
        target_ids=torch.tensor([8192, 4097], device="cuda"),
    )
    reference = logits.double().log_softmax(-1)
    torch.testing.assert_close(
        scores,
        reference[torch.arange(2, device="cuda"), selected].float(),
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        targets,
        reference[
            torch.arange(2, device="cuda"), torch.tensor([8192, 4097], device="cuda")
        ].float(),
        atol=2e-6,
        rtol=0,
    )


@pytest.mark.parametrize("capacity", [32, 1000])
def test_changed_policy_and_accepted_counts_are_replayed_without_host_reads(capacity):
    logits = torch.randn((4, 3, 8193), dtype=torch.bfloat16, device="cuda")
    selected = torch.zeros((4, 3), dtype=torch.int64, device="cuda")
    mask = torch.ones(4, dtype=torch.bool, device="cuda")
    counts = torch.full((4,), 3, dtype=torch.int32, device="cuda")
    tops = torch.full((4,), 5, dtype=torch.int32, device="cuda")
    state = []

    def launch():
        state[:] = score_logits(
            logits,
            selected,
            score_mask=mask,
            valid_counts=counts,
            top_counts=tops,
            max_top_k=capacity,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    logits.add_(2)
    selected.fill_(7)
    counts.copy_(torch.tensor([1, 2, 0, 3], device="cuda", dtype=torch.int32))
    tops.copy_(torch.tensor([capacity, 0, 1, 9], device="cuda", dtype=torch.int32))
    mask.copy_(torch.tensor([True, False, True, True], device="cuda"))
    debug = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        graph.replay()
    finally:
        torch.cuda.set_sync_debug_mode(debug)
    expected = score_logits(
        logits,
        selected,
        score_mask=mask,
        valid_counts=counts,
        top_counts=tops,
        max_top_k=capacity,
    )
    for actual, reference in zip(state, expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_batch_and_top_count_changes_reuse_compiled_kernels(monkeypatch):
    import cutlass.cute as cute

    def run(batch, top):
        logits = torch.randn((batch, 3, 8193), device="cuda", dtype=torch.bfloat16)
        ids = torch.zeros((batch, 3), device="cuda", dtype=torch.int64)
        score_logits(
            logits,
            ids,
            score_mask=torch.ones(batch, dtype=torch.bool, device="cuda"),
            valid_counts=torch.ones(batch, dtype=torch.int32, device="cuda"),
            top_counts=torch.full((batch,), top, dtype=torch.int32, device="cuda"),
            max_top_k=top,
            target_rows=torch.tensor([0, 0, 2], device="cuda"),
            target_ids=torch.tensor([1, 1, 7], device="cuda"),
        )

    run(3, 0)
    run(3, 1)
    run(3, 65)

    def forbid_recompile(*args, **kwargs):
        raise AssertionError("batch/top-count changes must reuse compiled kernels")

    monkeypatch.setattr(cute, "compile", forbid_recompile)
    for batch, top in (
        (1, 0),
        (128, 0),
        (2, 1),
        (7, 32),
        (16, 65),
        (64, 1000),
        (3, 2051),
        (17, 5),
    ):
        run(batch, top)
