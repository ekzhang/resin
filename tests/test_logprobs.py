import weakref
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from output_helpers import forward_output

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    ForwardOutput,
    RunnerConfig,
    SamplingParams,
)
from resin.backend.plan_transport import PlanTransport
from resin.backend.sampling import TorchSampler, format_output, token_output


def test_forward_output_requires_keyword_arguments():
    ids = np.array([3], np.uint32)
    offsets = np.array([0, 1], np.uint32)
    output = ForwardOutput(token_ids=ids, token_indptr=offsets)
    assert output.token_ids is ids and output.token_indptr is offsets
    with pytest.raises(TypeError, match="positional"):
        ForwardOutput(ids, offsets)  # ty: ignore[too-many-positional-arguments, missing-argument]


def test_token_only_output_owns_tensor_storage_without_python_scalars(monkeypatch):
    tokens = torch.tensor([[3, 4], [5, 6]])
    plan = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[0, 0],
        token_ids=[0, 0],
        token_indptr=[0, 1, 2],
    )

    def forbid_list(tensor):
        raise AssertionError("output must not box tokens into Python lists")

    monkeypatch.setattr(torch.Tensor, "tolist", forbid_list)
    output = format_output(None, tokens, plan)
    tokens.fill_(99)  # The next graph replay can overwrite its output buffer.
    np.testing.assert_array_equal(output.token_ids, [3, 4, 5, 6])
    np.testing.assert_array_equal(output.token_indptr, [0, 2, 4])
    assert output.token_ids.dtype == np.uint32
    assert output.logprobs is None


@pytest.mark.parametrize(
    ("tokens", "counts", "ids", "offsets"),
    [
        (np.array([[3, 4, 99], [5, 99, 99]]), [2, 1], [3, 4, 5], [0, 2, 3]),
        (np.array([[3, 99], [5, 6]]), [0, 2], [5, 6], [0, 0, 2]),
        (np.empty((0, 4), np.int64), np.empty(0, np.uint32), [], [0]),
        (np.array([3, 4]), None, [3, 4], [0, 1, 2]),
        ([[3, 4], [5]], None, [3, 4, 5], [0, 2, 3]),
    ],
)
def test_token_output_packs_only_accepted_positions(tokens, counts, ids, offsets):
    output = token_output(tokens, counts=counts)
    np.testing.assert_array_equal(output.token_ids, ids)
    np.testing.assert_array_equal(output.token_indptr, offsets)


@pytest.mark.parametrize("counts", [[-1, 1], [3, 1], [1], [[1, 1]], [1.5, 1]])
def test_token_output_rejects_invalid_acceptance_counts(counts):
    with pytest.raises(ValueError, match="accepted counts"):
        token_output(np.array([[3, 4], [5, 6]]), counts=counts)


@pytest.mark.parametrize(
    "wrap",
    [np.asarray, torch.tensor, list, lambda values: [values]],
    ids=["numpy", "torch", "flat-list", "ragged-list"],
)
@pytest.mark.parametrize("token", [-1, 2**32, 2**40])
def test_token_output_rejects_overflow_before_narrowing(wrap, token):
    with pytest.raises(ValueError, match="token ids must fit uint32"):
        token_output(wrap([3, token]))


@pytest.mark.parametrize(
    "tokens",
    [
        np.array([1.0, 2.5]),
        np.array([True, False]),
        torch.tensor([1.0, 2.5]),
        torch.tensor([True, False]),
        [1, 2.5],
        [True, 2],
        [[1], [2.5]],
        [[1], [np.bool_(False)]],
    ],
)
def test_token_output_rejects_noninteger_tokens(tokens):
    with pytest.raises(TypeError, match="token ids must be integers"):
        token_output(tokens)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.uint32])
def test_token_output_preserves_full_uint32_range(dtype):
    output = token_output(np.array([0, 2**31, 2**32 - 1], dtype=dtype))
    np.testing.assert_array_equal(output.token_ids, [0, 2**31, 2**32 - 1])


@pytest.mark.parametrize(
    "tokens",
    [
        np.array([[3, -1], [-1, 2**32]], np.int64),
        np.array([[3, 2**64 - 1], [2**32, 2**64 - 1]], np.uint64),
        torch.tensor([[3, -1], [-1, 2**32]]),
    ],
)
def test_token_output_ignores_unaccepted_padding_values(tokens):
    output = token_output(tokens, counts=[1, 0])
    np.testing.assert_array_equal(output.token_ids, [3])
    np.testing.assert_array_equal(output.token_indptr, [0, 1, 1])


@dataclass
class OutputRunner:
    forward: Callable[[BatchPlan], ForwardOutput]
    release: Callable[[list[int]], None] = lambda _: None


def scored_output(rows, scores, **kwargs):
    return ForwardOutput(
        token_ids=np.asarray([token for row in rows for token in row], np.uint32),
        token_indptr=np.asarray([0, *np.cumsum(list(map(len, rows)))], np.uint32),
        logprobs=np.asarray([value for row in scores for value in row], np.float32),
        logprob_indptr=np.asarray([0, *np.cumsum(list(map(len, scores)))], np.uint32),
        **kwargs,
    )


def test_scores_use_model_distribution_and_cover_only_accepted_positions():
    logits = torch.tensor(
        [
            [[1.0, 3.0, 0.0], [2.0, 1.0, 4.0], [4.0, 2.0, 1.0]],
            [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    plan = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[0, 0],
        token_ids=[0, 0],
        token_indptr=[0, 1, 2],
        sampling_params=[
            SamplingParams(temperature=0.1, top_k=1, top_logprobs=2),
            SamplingParams(logprobs=True),
        ],
        sampling_mask=[True, False],
    )
    output = format_output(logits, [[1, 2], [2]], plan)
    assert output.logprobs is not None
    assert output.top_indptr is not None
    assert output.top_token_ids is not None
    assert output.top_logprobs is not None
    expected = logits.log_softmax(-1)
    np.testing.assert_array_equal(output.token_ids, [1, 2, 2])
    np.testing.assert_array_equal(output.token_indptr, [0, 2, 3])
    np.testing.assert_array_equal(output.logprob_indptr, [0, 2, 2])
    for column, token in enumerate([1, 2]):
        assert output.logprobs[column] == pytest.approx(
            expected[0, column, token].item()
        )
        start, end = output.top_indptr[column : column + 2]
        np.testing.assert_array_equal(
            output.top_token_ids[start:end], expected[0, column].topk(2).indices.numpy()
        )
        np.testing.assert_allclose(
            output.top_logprobs[start:end], expected[0, column].topk(2).values.numpy()
        )


def test_logprob_options_survive_broadcast_and_do_not_change_sampling():
    params = SamplingParams(seed=7, top_k=2, top_logprobs=3)
    assert params.logprobs
    plan = BatchPlan(
        sequence_ids=[1],
        context_lengths=[0],
        token_ids=[0],
        token_indptr=[0, 1],
        sampling_params=[params],
    )
    restored = PlanTransport().restore(PlanTransport().prepare(plan))
    assert restored.sampling_params[0].top_logprobs == 3
    assert restored.needs_logprobs
    assert not BatchPlan(
        sequence_ids=[1],
        context_lengths=[0],
        token_ids=[0],
        token_indptr=[0, 1],
        sampling_params=[params],
        sampling_mask=[False],
    ).needs_logprobs
    logits = torch.tensor([[2.0, 4.0, 1.0]])
    tokens = TorchSampler()(logits, plan)
    output = format_output(logits, tokens, plan)
    np.testing.assert_array_equal(output.token_ids, tokens.numpy())
    assert output.top_logprobs is not None
    assert len(output.top_logprobs) == 3
    with pytest.raises(ValueError, match="top_logprobs"):
        SamplingParams(top_logprobs=1001)


def test_mixed_top_k_requests_use_packed_columns():
    logits = torch.tensor([[2.0, 4.0, 1.0], [0.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    plan = BatchPlan(
        sequence_ids=[1, 2, 3],
        context_lengths=[0, 0, 0],
        token_ids=[0, 0, 0],
        token_indptr=[0, 1, 2, 3],
        sampling_params=[
            SamplingParams(top_logprobs=1),
            SamplingParams(logprobs=True),
            SamplingParams(top_logprobs=9),
        ],
    )
    output = format_output(logits, torch.tensor([1, 2, 0]), plan)
    assert output.logprobs is not None
    np.testing.assert_array_equal(output.top_indptr, [0, 1, 1, 4])
    np.testing.assert_allclose(
        output.logprobs, logits.log_softmax(-1)[range(3), [1, 2, 0]].numpy()
    )


class ScoredRunner:
    def __init__(self):
        self.runner_config = RunnerConfig(
            cache_layout=CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(2)]),
            max_accepted_per_step=3,
        )
        self.calls = 0
        self.released = []

    def bind_resources(self, binding):
        pass

    def release(self, ids):
        self.released.extend(ids)

    def forward(self, plan):
        self.calls += 1
        if self.calls == 1:
            return scored_output([[2]], [[-0.1]])
        return scored_output([[7, 99, 8]], [[-0.2, -0.3, -0.4]])


def test_native_events_keep_speculative_scores_aligned_through_eos():
    runner = ScoredRunner()
    engine = Engine(runner)
    request = engine.submit(
        [1],
        sampling_params=SamplingParams(
            max_output_tokens=5, stop_token_ids=[99], logprobs=True
        ),
    )
    events = engine.step() + engine.step()
    assert [event.token_id for event in events] == [2, 7, 99]
    assert all(event.logprobs is not None for event in events)
    assert [
        event.logprobs.logprob for event in events if event.logprobs is not None
    ] == pytest.approx([-0.1, -0.2, -0.3])
    assert [event.index for event in events] == [0, 1, 2]
    assert request.take_result() == [2, 7, 99]
    assert runner.released == [request]
    assert engine.cache_free_parent_count == 4


@pytest.mark.parametrize(
    "scores", [[[-0.2]], [[-0.2, -0.3, float("nan")]], [[-0.2, -0.3, 0.1]], []]
)
def test_invalid_scores_after_stop_still_quarantine_the_whole_batch(scores):
    runner = ScoredRunner()
    original = runner.forward

    def forward(plan):
        return scored_output([[7, 99, 8]], scores) if runner.calls else original(plan)

    setattr(runner, "forward", forward)
    engine = Engine(runner)
    request = engine.submit(
        [1],
        sampling_params=SamplingParams(
            max_output_tokens=5, stop_token_ids=[99], logprobs=True
        ),
    )
    engine.step()
    with pytest.raises(RuntimeError):
        engine.step()
    assert engine.poisoned and not request.done
    assert runner.released == []
    assert engine.cache_occupied_block_count is not None
    assert engine.cache_occupied_block_count > 0


@pytest.mark.parametrize("offsets", [[1, 3], [0, 4], [0, 2], [0, 3, 2]])
def test_invalid_output_offsets_fail_before_commit(offsets):
    output = ForwardOutput(
        token_ids=np.array([3, 4, 5], np.uint32),
        token_indptr=np.array(offsets, np.uint32),
    )
    engine = Engine(OutputRunner(forward=lambda plan: output))
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=1))
    with pytest.raises(RuntimeError):
        engine.step()
    assert engine.poisoned and not request.done


def test_array_handoff_retains_storage_and_events_own_their_scores():
    values = np.array([-0.125], np.float32)
    ids = np.array([3], np.uint32)
    output = ForwardOutput(
        token_ids=ids, token_indptr=np.array([0, 1], np.uint32), logprobs=values
    )
    assert output.token_ids is ids and output.logprobs is values
    assert np.shares_memory(output.logprobs, values)
    sources = []

    def forward(plan):
        local_ids = np.array([3], np.uint32)
        local_offsets = np.array([0, 1], np.uint32)
        local_scores = np.array([-0.5], np.float32)
        sources.extend(map(weakref.ref, (local_ids, local_offsets, local_scores)))
        return ForwardOutput(
            token_ids=local_ids, token_indptr=local_offsets, logprobs=local_scores
        )

    def release(sequence_ids):
        # Storage stays alive until the complete transaction has released the row.
        assert all(source() is not None for source in sources)

    engine = Engine(OutputRunner(forward=forward, release=release))
    engine.submit(
        [1], sampling_params=SamplingParams(max_output_tokens=1, logprobs=True)
    )
    (event,) = engine.step()
    assert all(source() is None for source in sources)
    assert event.logprobs is not None
    assert event.logprobs.logprob == -0.5

    engine = Engine(OutputRunner(forward=lambda plan: output))
    request = engine.submit(
        [1], sampling_params=SamplingParams(max_output_tokens=1, logprobs=True)
    )
    (event,) = engine.step()
    ids.fill(99)
    values.fill(-9)
    output.token_indptr.fill(0)
    assert request.take_result() == [3] and event.token_id == 3
    assert event.logprobs is not None and event.logprobs.logprob == -0.125


def test_exact_array_dtypes_and_contiguity_are_required_without_coercion():
    with pytest.raises(TypeError):
        # Wrong array dtype must fail without coercion.
        ForwardOutput(
            token_ids=np.array([3], np.int64),  # ty: ignore[invalid-argument-type]
            token_indptr=np.array([0, 1], np.uint32),
        )
    output = ForwardOutput(
        token_ids=np.array([3], np.uint32),
        token_indptr=np.array([0, 1], np.uint32),
        logprobs=np.zeros(4, np.float32)[::2],
    )
    engine = Engine(OutputRunner(forward=lambda plan: output))
    engine.submit([1], sampling_params=SamplingParams(max_output_tokens=1))
    with pytest.raises(ValueError, match="contiguous"):
        engine.step()
    assert engine.poisoned


def test_top_alternatives_are_read_from_arrays_and_validated_even_past_eos():
    good = scored_output(
        [[3]],
        [[-0.1]],
        top_indptr=np.array([0, 2], np.uint32),
        top_token_ids=np.array([3, 4], np.uint32),
        top_logprobs=np.array([-0.1, -2.0], np.float32),
    )
    engine = Engine(OutputRunner(forward=lambda plan: good))
    engine.submit(
        [1], sampling_params=SamplingParams(max_output_tokens=1, top_logprobs=2)
    )
    (event,) = engine.step()
    assert event.logprobs is not None
    assert np.array(event.logprobs.top_logprobs) == pytest.approx(
        np.array([[3, -0.1], [4, -2.0]])
    )

    bad = scored_output(
        [[3]],
        [[-0.1]],
        top_indptr=np.array([0, 2], np.uint32),
        top_token_ids=np.array([3], np.uint32),
        top_logprobs=np.array([-0.1, float("nan")], np.float32),
    )
    engine = Engine(OutputRunner(forward=lambda plan: bad))
    engine.submit(
        [1], sampling_params=SamplingParams(max_output_tokens=1, top_logprobs=2)
    )
    with pytest.raises(RuntimeError):
        engine.step()
    assert engine.poisoned


def test_requested_scores_cannot_silently_disappear():
    engine = Engine(OutputRunner(forward=lambda plan: forward_output([3])))
    engine.submit(
        [1], sampling_params=SamplingParams(max_output_tokens=1, logprobs=True)
    )
    with pytest.raises(RuntimeError, match="requested token logprobs"):
        engine.step()
    assert engine.poisoned


def test_token_only_native_outputs_need_no_logprob_columns():
    engine = Engine(OutputRunner(forward=lambda plan: forward_output([3])))
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=1))
    (event,) = engine.step()
    assert event.logprobs is None and request.take_result() == [3]
