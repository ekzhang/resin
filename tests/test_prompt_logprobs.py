"""Input scoring shares prefill computation, including chunk-boundary logits."""

import numpy as np
import pytest
import torch
from output_helpers import forward_output

from resin import (
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    RunnerConfig,
    SamplingParams,
    SchedulerConfig,
)
from resin.backend.plan_transport import PlanTransport
from resin.backend.sampling import PromptScores, format_output, prompt_scores


class TransitionRunner:
    def __init__(self, *, cached=False, malformed=False):
        self.runner_config = RunnerConfig(
            continuation_prefill=True,
            cache_layout=CacheLayout(parent_capacity=32, groups=[CacheGroupLayout(2)])
            if cached
            else None,
        )
        self.calls = []
        self.malformed = malformed
        self.table = torch.arange(64, dtype=torch.float32).view(8, 8).sin()

    def bind_resources(self, binding):
        pass

    def release(self, ids):
        pass

    def forward(self, plan):
        self.calls.append(plan)
        hidden = self.table[torch.as_tensor(plan.token_ids.astype(np.int64))]
        last = torch.as_tensor(plan.token_indptr[1:].astype(np.int64) - 1)
        logits = hidden[last]
        if self.malformed:
            return forward_output(logits.argmax(-1).tolist())
        return format_output(
            logits, logits.argmax(-1), plan, prompt_logprobs=prompt_scores(plan, hidden)
        )


@pytest.mark.parametrize("chunk", [1, 2, 3, 8])
def test_prompt_scores_cover_chunk_successor_and_do_not_change_generation(chunk):
    runner = TransitionRunner()
    engine = Engine(runner, scheduler_config=SchedulerConfig(prefill_chunk_size=chunk))
    tokens = [1, 3, 5, 2, 6]
    request = engine.submit(
        tokens,
        sampling_params=SamplingParams(
            max_output_tokens=2, prompt_logprobs=True, logprobs=True, temperature=0
        ),
    )
    while engine.has_work():
        engine.step()
    scores = request.prompt_logprobs
    assert scores is not None and scores[0] is None
    expected = runner.table[tokens[:-1]].log_softmax(-1)[
        range(len(tokens) - 1), tokens[1:]
    ]
    assert [item.logprob for item in scores[1:] if item is not None] == pytest.approx(
        expected.tolist(), abs=1e-6
    )
    assert len(runner.calls) == (len(tokens) + chunk - 1) // chunk + 1
    generated = request.take_result()
    assert generated == [
        runner.table[tokens[-1]].argmax().item(),
        runner.table[generated[0]].argmax().item(),
    ]


def test_prompt_cache_reuse_is_bypassed_only_when_scores_are_requested():
    runner = TransitionRunner(cached=True)
    engine = Engine(runner, prefix_cache=True)
    for scoring in (False, False, True, False):
        request = engine.submit(
            [1, 2, 3, 4, 5],
            sampling_params=SamplingParams(
                max_output_tokens=1, prompt_logprobs=scoring, temperature=0
            ),
        )
        if scoring:
            assert request.cached_tokens == 0
        elif len(runner.calls):
            assert request.cached_tokens > 0
        while engine.has_work():
            engine.step()
        scores = request.prompt_logprobs
        assert (scores is not None) == scoring
        if scores is not None:
            assert scores[0] is None and all(value is not None for value in scores[1:])


def test_single_token_prompt_has_no_distribution_to_score():
    runner = TransitionRunner()
    engine = Engine(runner)
    request = engine.submit(
        [1], sampling_params=SamplingParams(max_output_tokens=1, prompt_logprobs=True)
    )
    engine.step()
    assert request.prompt_logprobs == [None]


@pytest.mark.parametrize("tokens", [[1], [1, 3, 2, 4]])
def test_score_only_request_retires_without_generated_events(tokens):
    runner = TransitionRunner()
    engine = Engine(runner, scheduler_config=SchedulerConfig(prefill_chunk_size=2))
    request = engine.submit(
        tokens,
        sampling_params=SamplingParams(max_output_tokens=0, prompt_logprobs=True),
    )
    while engine.has_work():
        assert engine.step() == []
    assert request.done and request.take_result() == []
    assert request.prompt_logprobs is not None
    assert request.prompt_logprobs[0] is None
    assert all(score is not None for score in request.prompt_logprobs[1:])


def test_score_only_full_prompt_does_not_require_continuation_support():
    runner = TransitionRunner()
    runner.runner_config = RunnerConfig()
    engine = Engine(runner)
    request = engine.submit(
        [1, 2, 3],
        sampling_params=SamplingParams(max_output_tokens=0, prompt_logprobs=True),
    )
    assert engine.step() == []
    assert request.done and len(runner.calls) == 1
    assert request.prompt_logprobs is not None
    assert request.prompt_logprobs[1] is not None


def test_missing_requested_prompt_scores_poison_without_partial_progress():
    engine = Engine(TransitionRunner(malformed=True))
    request = engine.submit(
        [1, 2],
        sampling_params=SamplingParams(max_output_tokens=1, prompt_logprobs=True),
    )
    with pytest.raises(RuntimeError, match="requested prompt logprobs"):
        engine.step()
    assert engine.poisoned and not request.done
    assert request.prompt_logprobs == [None, None]


def test_prompt_metadata_survives_follower_transport():
    plan = BatchPlan(
        sequence_ids=[1],
        context_lengths=[0],
        token_ids=[1, 2],
        token_indptr=[0, 2],
        sampling_params=[SamplingParams(prompt_logprobs=True, prompt_top_logprobs=3)],
        prompt_score_indices=[0, 1],
        prompt_score_token_ids=[2, 3],
        prompt_score_indptr=[0, 2],
        prompt_score_top_counts=[1, 3],
        prompt_target_indptr=[0, 2, 3],
        prompt_target_token_ids=[7, 2, 4],
    )
    restored = PlanTransport().restore(PlanTransport().prepare(plan))
    assert restored.sampling_params[0].prompt_logprobs
    assert restored.sampling_params[0].prompt_top_logprobs == 3
    assert restored.needs_prompt_logprobs
    np.testing.assert_array_equal(restored.prompt_score_token_ids, [2, 3])
    np.testing.assert_array_equal(restored.prompt_score_indices, [0, 1])
    np.testing.assert_array_equal(restored.prompt_score_top_counts, [1, 3])
    np.testing.assert_array_equal(restored.prompt_target_indptr, [0, 2, 3])
    np.testing.assert_array_equal(restored.prompt_target_token_ids, [7, 2, 4])


def test_mixed_rows_keep_input_and_output_scores_separate():
    runner = TransitionRunner()
    engine = Engine(runner, scheduler_config=SchedulerConfig(prefill_chunk_size=2))
    scored = engine.submit(
        [1, 3, 5],
        sampling_params=SamplingParams(
            max_output_tokens=2, prompt_logprobs=True, logprobs=True
        ),
    )
    ordinary = engine.submit(
        [2, 4], sampling_params=SamplingParams(max_output_tokens=2)
    )
    while engine.has_work():
        engine.step()
    scores = scored.prompt_logprobs
    assert scores is not None and scores[0] is None
    expected = runner.table[[1, 3]].log_softmax(-1)[[0, 1], [3, 5]]
    assert [
        score.logprob for score in scores[1:] if score is not None
    ] == pytest.approx(expected.tolist(), abs=1e-6)
    assert ordinary.prompt_logprobs is None


@pytest.mark.parametrize(
    "bad_scores",
    [
        np.array([], np.float32),
        np.array([0.1], np.float32),
        np.array([float("nan")], np.float32),
    ],
)
def test_invalid_prompt_columns_leave_scores_unpublished(bad_scores, monkeypatch):
    from resin import ForwardOutput

    runner = TransitionRunner()
    monkeypatch.setattr(
        runner,
        "forward",
        lambda plan: ForwardOutput(
            token_ids=np.array([3], np.uint32),
            token_indptr=np.array([0, 1], np.uint32),
            prompt_logprobs=bad_scores,
            prompt_logprob_indptr=np.array([0, len(bad_scores)], np.uint32),
        ),
    )
    engine = Engine(runner)
    request = engine.submit(
        [1, 2],
        sampling_params=SamplingParams(max_output_tokens=1, prompt_logprobs=True),
    )
    with pytest.raises(RuntimeError, match="prompt"):
        engine.step()
    assert engine.poisoned and not request.done
    assert request.prompt_logprobs == [None, None]


def test_projection_and_normalization_are_shared_and_bounded(monkeypatch):
    count = 513
    plan = BatchPlan(
        sequence_ids=[1],
        context_lengths=[0],
        token_ids=[0] * count,
        token_indptr=[0, count],
        prompt_score_indices=list(range(count)),
        prompt_score_token_ids=[1] * count,
        prompt_score_indptr=[0, count],
        prompt_score_top_counts=[9] * count,
        prompt_target_indptr=list(range(0, 3 * count + 1, 3)),
        prompt_target_token_ids=[6, 0, 6] * count,
    )
    rows = []
    hidden = torch.randn(count, 4)
    weight = torch.randn(4, 7)
    expected = (hidden @ weight).log_softmax(-1)
    normalizations = []
    original = torch.Tensor.log_softmax

    def normalize(values, *args, **kwargs):
        normalizations.append(len(values))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "log_softmax", normalize)

    def project(values):
        rows.append(len(values))
        return values @ weight

    actual = prompt_scores(plan, hidden, project)
    assert isinstance(actual, PromptScores)
    assert rows == normalizations == [128, 128, 128, 128, 1]
    torch.testing.assert_close(actual.logprobs, expected[:, 1])
    assert actual.top is not None and actual.targets is not None
    np.testing.assert_array_equal(actual.top.counts, np.full(count, 7, np.uint32))
    torch.testing.assert_close(actual.top.logprobs, expected.topk(7).values.flatten())
    torch.testing.assert_close(
        actual.targets.logprobs, expected[:, [6, 0, 6]].flatten()
    )
    assert actual.targets.token_ids.tolist() == [6, 0, 6] * count


def test_shared_columns_join_per_sequence_rows_without_losing_empty_score_groups():
    plan = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[0, 0],
        token_ids=[1, 2, 3, 4],
        token_indptr=[0, 2, 4],
        sampling_params=[
            SamplingParams(prompt_logprobs=True),
            SamplingParams(prompt_top_logprobs=2),
        ],
        prompt_score_indices=[0, 1, 2, 3],
        prompt_score_token_ids=[2, 5, 4, 6],
        prompt_score_indptr=[0, 2, 4],
        prompt_score_top_counts=[0, 0, 2, 2],
        prompt_target_indptr=[0, 1, 3, 3, 3],
        prompt_target_token_ids=[7, 0, 7],
    )
    hidden = TransitionRunner().table[[1, 2, 3, 4]]
    whole = prompt_scores(plan, hidden)
    first = prompt_scores(plan, hidden[:2])
    second = prompt_scores(plan, hidden[2:], token_start=2)
    assert first is not None and second is not None
    parts = PromptScores.concat([first, second])
    assert whole is not None and parts is not None
    expected, actual = whole.output(plan), parts.output(plan)
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    np.testing.assert_array_equal(actual["prompt_top_indptr"], [0, 0, 0, 2, 4])
    np.testing.assert_array_equal(actual["prompt_target_indptr"], [0, 1, 3, 3, 3])


@pytest.mark.parametrize("chunk", [1, 2, 8])
@pytest.mark.parametrize("mode", ["top", "targets", "both"])
def test_prompt_top_and_per_position_targets_enable_raw_score_only_requests(
    chunk, mode
):
    runner = TransitionRunner()
    engine = Engine(runner, scheduler_config=SchedulerConfig(prefill_chunk_size=chunk))
    tokens = [1, 3, 5, 2, 6]
    targets = [[], [7, 3, 7], [], [0], [2, 1]]
    top = 3 if mode in ("top", "both") else 0
    requested_targets = targets if mode in ("targets", "both") else None
    request = engine.submit(
        tokens,
        sampling_params=SamplingParams(
            max_output_tokens=0,
            prompt_top_logprobs=top,
            temperature=0.13,
            top_k=1,
            top_p=0.2,
        ),
        prompt_target_logprobs=requested_targets,
    )
    while engine.has_work():
        assert engine.step() == []
    scores = request.prompt_logprobs
    assert scores is not None and scores[0] is None
    expected = runner.table[tokens[:-1]].log_softmax(-1)
    for position, score in enumerate(scores[1:], 1):
        assert score is not None
        distribution = expected[position - 1]
        assert score.logprob == pytest.approx(
            distribution[tokens[position]].item(), abs=1e-6
        )
        assert [token for token, _ in score.top_logprobs] == distribution.topk(
            top
        ).indices.tolist()
        assert [value for _, value in score.top_logprobs] == pytest.approx(
            distribution.topk(top).values.tolist(), abs=1e-6
        )
        ids = [] if requested_targets is None else targets[position]
        assert [token for token, _ in score.target_logprobs] == ids
        assert [value for _, value in score.target_logprobs] == pytest.approx(
            distribution[ids].tolist(), abs=1e-6
        )
    assert request.done and request.take_result() == []
    assert len(runner.calls) == (len(tokens) + chunk - 1) // chunk
    assert sum(len(plan.prompt_target_token_ids) for plan in runner.calls) == (
        sum(map(len, targets)) if requested_targets is not None else 0
    )


def test_mixed_prompt_views_preserve_row_ownership_and_target_request_storage():
    runner = TransitionRunner()
    engine = Engine(runner, scheduler_config=SchedulerConfig(prefill_chunk_size=2))
    top = engine.submit(
        [1, 3, 2],
        sampling_params=SamplingParams(max_output_tokens=1, prompt_top_logprobs=20),
    )
    targets = [[], [7, 2], [1], [3, 3]]
    target = engine.submit(
        [2, 4, 1, 5],
        sampling_params=SamplingParams(max_output_tokens=1),
        prompt_target_logprobs=targets,
    )
    ordinary = engine.submit(
        [6, 3], sampling_params=SamplingParams(max_output_tokens=1)
    )
    targets[1].clear()
    while engine.has_work():
        engine.step()
    assert top.prompt_logprobs is not None and target.prompt_logprobs is not None
    top_score, target_score = top.prompt_logprobs[1], target.prompt_logprobs[1]
    assert top_score is not None and target_score is not None
    assert len(top_score.top_logprobs) == 8
    assert top_score.target_logprobs == []
    assert [token for token, _ in target_score.target_logprobs] == [7, 2]
    assert target_score.top_logprobs == []
    assert ordinary.prompt_logprobs is None
    assert any(plan.num_sequences > 1 for plan in runner.calls)


def test_empty_target_rows_still_enable_input_scoring():
    runner = TransitionRunner()
    engine = Engine(runner)
    request = engine.submit(
        [1, 2],
        sampling_params=SamplingParams(max_output_tokens=0),
        prompt_target_logprobs=[[], []],
    )
    assert engine.step() == []
    assert request.done and request.prompt_logprobs is not None
    score = request.prompt_logprobs[1]
    assert score is not None and score.target_logprobs == []


@pytest.mark.parametrize(
    "targets", [[[]], [[], [], []], [[1], []], [[], [-1]], [[], [2**32]]]
)
def test_prompt_target_request_shape_and_ids_are_validated_before_admission(targets):
    engine = Engine(TransitionRunner())
    with pytest.raises((ValueError, TypeError, OverflowError)):
        engine.submit(
            [1, 2],
            sampling_params=SamplingParams(max_output_tokens=0),
            prompt_target_logprobs=targets,
        )
    assert not engine.has_work()


def test_out_of_vocabulary_targets_are_rejected_before_any_gather(monkeypatch):
    engine = Engine(TransitionRunner())
    request = engine.submit(
        [1, 2],
        sampling_params=SamplingParams(max_output_tokens=0),
        prompt_target_logprobs=[[], [8]],
    )
    gathers = []
    original = torch.Tensor.gather

    def gather(values, *args, **kwargs):
        gathers.append(True)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "gather", gather)
    with pytest.raises(ValueError, match="target token IDs.*vocabulary size"):
        engine.step()
    assert gathers == []
    assert not request.done
    assert request.prompt_logprobs == [None, None]
