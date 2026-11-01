from __future__ import annotations

import numpy as np
import pytest
import torch

from resin.backend.cuda.completion import (
    CudaContinuation,
    DeviceAcceptance,
)
from resin.backend.speculative import _accept_speculative_tokens


@pytest.mark.parametrize("draft_tokens", [0, 1, 3, 4, 8])
def test_device_acceptance_matches_shared_reference(draft_tokens):
    generator = torch.Generator().manual_seed(15)
    drafts = torch.randint(0, 5, (128, draft_tokens), generator=generator)
    targets = torch.randint(0, 5, (128, draft_tokens + 1), generator=generator)
    for row in range(128):
        prefix = row % (draft_tokens + 1)
        targets[row, :prefix] = drafts[row, :prefix]
    positions = torch.arange(128, dtype=torch.int64) * 7
    decision = DeviceAcceptance.from_samples(drafts, targets, positions)
    tokens, counts, feedback = decision.tokens, decision.counts, decision.feedback()
    expected = _accept_speculative_tokens(drafts.tolist(), targets.tolist())
    actual = [
        row[:count] for row, count in zip(tokens.tolist(), counts.tolist(), strict=True)
    ]
    assert actual == expected.token_rows
    assert counts.tolist() == expected.processed_counts
    assert decision.accepted_drafts.tolist() == expected.accepted_counts
    assert feedback.token_ids.tolist() == [row[-1] for row in actual]
    assert feedback.positions.tolist() == (positions + counts).tolist()
    assert (
        feedback.inputs()["sampling_offsets"].tolist()
        == ((positions + counts + 1) * 32).tolist()
    )
    # Next replay is allowed to overwrite its source output immediately.
    targets.fill_(-1)
    drafts.fill_(-1)
    assert feedback.token_ids.tolist() == [row[-1] for row in actual]


def test_padded_feedback_never_advances_outside_reserved_scratch_positions():
    positions = torch.ones(8, dtype=torch.int64)
    output = torch.tensor([[3, 4, 5, 3, 4, 5, 6]] * 8, dtype=torch.int64)
    for _ in range(256):
        feedback = DeviceAcceptance.from_samples(
            output[:, :3], output[:, 3:], positions
        ).feedback()
        feedback.restrict(5)
        positions = feedback.positions
        assert positions[5:].tolist() == [1, 1, 1]
        assert feedback.token_ids[5:].tolist() == [0, 0, 0]
    assert positions[:5].tolist() == [1025] * 5


def test_completion_owns_copied_results_and_counts_stats_once(monkeypatch):
    from resin.backend.cuda import completion as runtime
    from resin.backend.speculative import SpeculativeStats

    class Stager:
        def __init__(self, device):
            pass

        def output(self, name, source):
            return source.clone()

    class Event:
        syncs = 0

        def record(self, stream):
            pass

        def synchronize(self):
            self.syncs += 1

    monkeypatch.setattr(runtime, "TensorStager", Stager)
    monkeypatch.setattr(runtime.torch.cuda, "Event", Event)
    monkeypatch.setattr(runtime.torch.cuda, "current_stream", lambda device: None)
    output = torch.tensor([[3, 4, 5, 3, 9, 7, 6]] * 4, dtype=torch.int64)
    acceptance = DeviceAcceptance.from_samples(
        output[:, :3],
        output[:, 3:],
        torch.ones(4, dtype=torch.int64),
    )
    tokens, counts = acceptance.tokens, acceptance.counts
    stats = SpeculativeStats(3)
    pending = runtime.DeviceCompletion.from_acceptance(acceptance, rows=3, stats=stats)
    tokens.fill_(-1)
    counts.fill_(-1)

    def forbid_list(tensor):
        raise AssertionError("completion must not box tokens into Python lists")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", forbid_list)
        result = pending.result()
        assert pending.result() is result
    np.testing.assert_array_equal(result.token_ids, [3, 9] * 3)
    np.testing.assert_array_equal(result.token_indptr, [0, 2, 4, 6])
    assert pending.positions.tolist() == [1, 1, 1]
    assert pending.feedback.positions.tolist() == [3, 3, 3, 1]
    assert isinstance(pending._event, Event)
    assert pending._event.syncs == 1
    assert stats.snapshot() == {
        "blocks": 3,
        "drafts": 9,
        "accepted": 3,
        "emitted": 6,
        "acceptance_rate": 1 / 3,
    }


def test_continuation_owns_admission_rows_bucket_and_fixed_inputs():
    manager = CudaContinuation()
    inputs = {"positions": torch.tensor([127, 255, 1, 1])}
    state = manager.begin([10, 20], (4, 1024), inputs)
    assert (
        manager.resume([10, 20], state, shape=(4, 1024))["positions"]
        is inputs["positions"]
    )
    with pytest.raises(ValueError, match="row ownership"):
        manager.resume([20, 10], state)
    with pytest.raises(ValueError, match="graph bucket"):
        manager.resume([10, 20], state, shape=(8, 1024))
    # Reacquiring even the same IDs/bucket creates a fresh immutable lease view.
    replacement = manager.begin([10, 20], (4, 1024), inputs)
    with pytest.raises(ValueError, match="cache lease"):
        manager.resume([10, 20], state)
    manager.drain(state)  # A stale completion cannot invalidate the new owner.
    manager.resume([10, 20], replacement)
    manager.drain()
    with pytest.raises(ValueError, match="cache lease"):
        manager.resume([10, 20], replacement)
    # Empty attention replicas still own an agreed scratch bucket while peers
    # execute real rows in the same captured expert-communication schedule.
    idle = manager.begin([], (4, 1024), inputs)
    assert manager.resume([], idle)["positions"] is inputs["positions"]


def test_state_report_extends_completion_fence_and_attaches_once(monkeypatch):
    from resin.backend.cuda import completion as runtime

    operations = []

    class Stager:
        def __init__(self, device):
            pass

        def output(self, name, source):
            operations.append(name)
            return source.clone()

    class Event:
        def record(self, stream):
            operations.append("record")

        def synchronize(self):
            operations.append("synchronize")

    class Report:
        def stage(self, staging):
            self.records = staging.output("state_checkpoints", torch.tensor([7]))
            return self

        def attach(self, output):
            operations.append("attach")
            assert self.records.tolist() == [7]
            return output.with_state_checkpoints(tuple(self.records.tolist()))

    monkeypatch.setattr(runtime, "TensorStager", Stager)
    monkeypatch.setattr(runtime.torch.cuda, "Event", Event)
    monkeypatch.setattr(runtime.torch.cuda, "current_stream", lambda device: None)
    acceptance = DeviceAcceptance.from_samples(
        torch.tensor([[5], [6]]),
        torch.tensor([[5, 9], [8, 9]]),
        torch.tensor([127, 255]),
    )
    pending = runtime.DeviceCompletion.from_acceptance(acceptance, rows=2)
    pending.feedback = None  # Draining does not lose the accepted valid prefix.
    assert pending.accepted_end_positions.tolist() == [129, 256]
    old_event = pending._event
    assert pending.attach_state_checkpoints(Report()) is pending
    assert pending._event is not old_event
    result = pending.result()
    assert result is pending.result()
    assert result.state_checkpoints == (7,)
    assert operations[-2:] == ["synchronize", "attach"]
    assert operations.count("record") == 2 and operations.count("attach") == 1
    with pytest.raises(RuntimeError, match="after result harvest"):
        pending.attach_state_checkpoints(Report())

    class BrokenReport:
        def stage(self, _staging):
            raise RuntimeError("report copy failed")

    pending = runtime.DeviceCompletion.from_acceptance(acceptance, rows=2)
    with pytest.raises(RuntimeError, match="report copy failed"):
        pending.attach_state_checkpoints(BrokenReport())
    with pytest.raises(RuntimeError, match="unfinished state checkpoint"):
        pending.result()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_runtime_acceptance_replay_shares_state_output_and_feedback():
    from resin.backend.cuda.completion import DeviceCompletion

    drafts = torch.tensor([[1, 2, 3]] * 4, device="cuda")
    targets = torch.tensor(
        [[9, 8, 7, 6], [1, 8, 7, 6], [1, 2, 7, 6], [1, 2, 3, 6]], device="cuda"
    )
    positions = torch.tensor([63, 127, 511, 8191], device="cuda")
    hidden = torch.arange(4 * 4 * 8, device="cuda").reshape(4, 4, 8)
    initial_targets = targets.clone()
    state = {}

    def launch():
        accepted = DeviceAcceptance.from_samples(drafts, targets, positions)
        state["accepted"] = accepted
        state["selected"] = hidden.gather(
            1, accepted.accepted_drafts[:, None, None].expand(-1, 1, 8)
        ).squeeze(1)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    decision = state["accepted"]
    for depth in (None, 0, 3):
        anchors = positions.cpu().tolist()
        targets.copy_(initial_targets)
        if depth is not None:
            targets[:, :depth].copy_(drafts[:, :depth])
            targets[:, depth].fill_(9)
        graph.replay()
        expected = _accept_speculative_tokens(
            drafts.cpu().tolist(),
            targets.cpu().tolist(),
        )
        selected = hidden[
            torch.arange(4, device="cuda"),
            torch.tensor(expected.accepted_counts, device="cuda"),
        ]
        torch.testing.assert_close(state["selected"], selected, rtol=0, atol=0)
        pending = DeviceCompletion.from_acceptance(decision, rows=4)
        feedback = pending.feedback
        # The next replay may replace every graph-owned decision tensor.
        targets.fill_(-1)
        positions.add_(4)
        graph.replay()
        result = pending.result()
        actual = [
            result.token_ids[
                result.token_indptr[row] : result.token_indptr[row + 1]
            ].tolist()
            for row in range(4)
        ]
        assert actual == expected.token_rows
        ends = [
            anchor + count
            for anchor, count in zip(anchors, expected.processed_counts, strict=True)
        ]
        assert pending.accepted_end_positions.tolist() == ends
        assert feedback.positions.tolist() == ends
        assert feedback.token_ids.tolist() == [row[-1] for row in expected.token_rows]


def test_feedback_fits_the_captured_decode_token_column_and_metadata_buffers():
    from resin.backend.cuda.staging import TensorStager

    output = torch.tensor([[3, 4, 5, 3, 9, 7, 6]] * 12, dtype=torch.int64)
    feedback = DeviceAcceptance.from_samples(
        output[:, :3], output[:, 3:], torch.ones(12, dtype=torch.int64)
    ).feedback()
    feedback.restrict(8)
    # These are the actual shapes/dtypes allocated by the shared CUDA runner.
    targets = {
        "token_ids": torch.empty((12, 1), dtype=torch.int64),
        "positions": torch.empty(12, dtype=torch.int64),
        "sampling_offsets": torch.empty(12, dtype=torch.int64),
    }
    TensorStager("cpu").copy_into(targets, feedback.inputs())
    assert targets["token_ids"][:, 0].tolist() == [9] * 8 + [0] * 4
    assert targets["positions"].tolist() == [3] * 8 + [1] * 4
    assert targets["sampling_offsets"].tolist() == [128] * 8 + [64] * 4


def test_staging_error_identifies_the_device_field_shape_and_dtype():
    from resin.backend.cuda.staging import StagingError, TensorStager

    targets = {"positions": torch.zeros(4, dtype=torch.int64)}
    with pytest.raises(
        StagingError,
        match=r"positions.*shape \(4,\).*torch.int32.*expected shape \(4,\).*torch.int64",
    ):
        TensorStager("cpu").copy_into(
            targets, {"positions": torch.ones(4, dtype=torch.int32)}
        )
    assert targets["positions"].tolist() == [0] * 4


def test_scored_completion_preserves_speculative_outputs_across_replay(monkeypatch):
    from resin import BatchPlan, SamplingParams
    from resin.backend.cuda import completion as runtime
    from resin.backend.sampling import format_output

    class Stager:
        def __init__(self, device):
            pass

        def output(self, name, source):
            return source.clone()

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            pass

    monkeypatch.setattr(runtime, "TensorStager", Stager)
    monkeypatch.setattr(runtime.torch.cuda, "Event", Event)
    monkeypatch.setattr(runtime.torch.cuda, "current_stream", lambda device: None)

    plan = BatchPlan(
        sequence_ids=[1, 2, 3, 4],
        context_lengths=[10, 20, 30, 40],
        token_ids=[0, 0, 0, 0],
        token_indptr=[0, 1, 2, 3, 4],
        sampling_params=[
            SamplingParams(top_logprobs=2, temperature=0.1, top_k=1, top_p=0.2),
            SamplingParams(),
            SamplingParams(top_logprobs=1),
            SamplingParams(top_logprobs=3),
        ],
        sampling_mask=[True, True, True, False],
    )
    output = torch.tensor(
        [
            [1, 2, 3, 1, 6, 4, 2],
            [0, 1, 2, 0, 1, 2, 5],
            [2, 3, 4, 6, 1, 2, 3],
            [5, 4, 3, 5, 4, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    decision = DeviceAcceptance.from_samples(
        output[:, :3], output[:, 3:], torch.ones(5, dtype=torch.int64)
    )
    tokens, counts, feedback = decision.tokens, decision.counts, decision.feedback()
    logits = torch.randn(5, 4, 7, generator=torch.Generator().manual_seed(42)) + 1e4
    accepted = [
        row[:count]
        for row, count in zip(tokens[:4].tolist(), counts[:4].tolist(), strict=True)
    ]
    expected = format_output(logits[:4], accepted, plan)
    values = logits.log_softmax(-1)
    top_values, top_ids = values.topk(2, dim=-1)
    scores = runtime.DeviceScores(
        values.gather(-1, tokens[..., None]).squeeze(-1),
        top_ids,
        top_values,
        logits.shape[-1],
    )
    with pytest.raises(ValueError, match="require prepared scores"):
        runtime.DeviceCompletion(tokens, counts, feedback, rows=4, plan=plan)

    def forbid_normalize(*args, **kwargs):
        raise AssertionError("completion must only stage prepared scores")

    monkeypatch.setattr(torch.Tensor, "log_softmax", forbid_normalize)
    pending = runtime.DeviceCompletion(
        tokens, counts, feedback, rows=4, plan=plan, scores=scores
    )
    # The next replay may replace every graph output before CPU harvest.
    tokens.fill_(-1)
    counts.zero_()
    logits.fill_(100)
    scores.logprobs.fill_(100)
    scores.top_ids.fill_(0)
    scores.top_logprobs.fill_(100)
    actual = pending.result()
    assert pending.result() is actual
    for name in (
        "token_ids",
        "token_indptr",
        "logprob_indptr",
        "logprobs",
        "top_indptr",
        "top_token_ids",
        "top_logprobs",
    ):
        values, reference = getattr(actual, name), getattr(expected, name)
        assert values.dtype == reference.dtype
        np.testing.assert_allclose(values, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16, torch.float32])
def test_native_state_copy_zero_restore_and_exact_checkpoint(dtype):
    from resin.backend.cuda.state_triton import copy_state

    # Non-contiguous slab strides and a partial last tile exercise complete-slot
    # copies without touching adjacent states or padding in the shared slab.
    width = 8197
    arena = torch.full((7, 3 * width + 17), 13, dtype=dtype, device="cuda")
    states = arena.as_strided((7, 3, width), (arena.stride(0), width, 1))
    states[1, 1].copy_(torch.arange(width, device="cuda").remainder(127).to(dtype))
    states[2, 2].fill_(7)
    before = arena.clone()
    restores = torch.tensor([[0, 0, 3, 0], [1, 1, 4, 2]], device="cuda")
    copy_state(states, restores)
    torch.testing.assert_close(
        states[3, 0], torch.zeros_like(states[3, 0]), rtol=0, atol=0
    )
    torch.testing.assert_close(states[4, 2], states[1, 1], rtol=0, atol=0)
    # Crossing 4096 is not an exactly materialized 4096 checkpoint.
    destinations = torch.tensor([[4, 2, 5, 1], [2, 2, 6, 0]], device="cuda")
    candidates = torch.tensor([[0, 4096], [1, 4096]], device="cuda")
    ends = torch.tensor([4097, 4096], device="cuda")
    flags = copy_state(states, destinations, candidates=candidates, ends=ends)
    assert flags.tolist() == [False, True]
    torch.testing.assert_close(
        states[5, 1], torch.full_like(states[5, 1], 13), rtol=0, atol=0
    )
    torch.testing.assert_close(states[6, 0], states[2, 2], rtol=0, atol=0)
    for slab, slot in ((0, 0), (1, 1), (2, 2), (3, 1), (4, 1), (5, 0), (6, 2)):
        torch.testing.assert_close(
            states[slab, slot],
            before[slab, slot * width : (slot + 1) * width],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        arena[:, 3 * width :], before[:, 3 * width :], rtol=0, atol=0
    )
