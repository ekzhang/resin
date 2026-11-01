"""Verification samples one complete block without copying vocabulary slices."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from resin import BatchPhase, BatchPlan, SamplingParams
from resin.backend.cuda.sampling import FusedSampler
from resin.backend.speculative import SpeculativeStats
from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner
from resin_models_cuda.inkling import cuda as inkling_runtime
from resin_models_cuda.inkling.cuda import InklingCudaRunner


class _BlockSampler:
    graph_inputs = staticmethod(FusedSampler.graph_inputs)

    def __init__(self, expected, targets):
        self.expected = expected
        self.targets = targets
        self.calls = []

    def empty(self, rows, device):
        return object.__new__(FusedSampler).empty(rows, device)

    def __call__(self, logits, plan):
        self._record(logits)
        assert plan.num_sequences == logits.shape[0]
        return self.targets.clone()

    def sample_block(self, logits, sampling, *, out=None, start_step=0):
        self._record(logits)
        assert sampling.seeds.shape == logits.shape[:1]
        assert out is None and start_step == 0
        return self.targets.clone()

    def sample_tensors(self, *_args, **_kwargs):
        raise AssertionError("verification must use one block sampler call")

    def _record(self, logits):
        assert logits.shape == self.expected.shape
        assert logits.stride() == self.expected.stride()
        assert logits.data_ptr() == self.expected.data_ptr()
        self.calls.append(tuple(logits.shape))


def _fixture(runner_type, *, scored=False):
    runner: Any = object.__new__(runner_type)
    runner.device = torch.device("cpu")
    runner.speculative_draft_tokens = 2
    runner.vocab_size = 10
    runner.config = SimpleNamespace(vocab_size=10, hidden_size=2, num_hidden_layers=1)
    runner._speculation = SpeculativeStats(2)
    runner._graph_logits = {}
    runner._graph_acceptance = {}
    runner._eager_phase_timing = lambda *_args: None
    plan = BatchPlan(
        sequence_ids=[7, 8],
        context_lengths=[14, 25],
        token_ids=[2, 3],
        token_indptr=[0, 1, 2],
        sampling_params=[SamplingParams(logprobs=scored)] * 2,
        phase=BatchPhase.DECODE,
    )
    logits = torch.arange(2 * 3 * 20, dtype=torch.float32).view(2, 3, 20)[..., ::2]
    drafts = torch.tensor([[2, 6], [3, 4]])
    targets = torch.tensor([[2, 4, 7], [3, 4, 5]])
    runner._sampler = _BlockSampler(logits, targets)
    return runner, plan, logits, drafts, targets


@pytest.mark.parametrize("runner_type", [DeepSeekV41CudaRunner, InklingCudaRunner])
@pytest.mark.parametrize("scored,fixed", [(False, None), (True, 0)])
def test_eager_verification_routes_scores_and_rolls_back_accepted_state(
    monkeypatch, runner_type, scored, fixed
):
    runner, plan, logits, drafts, _ = _fixture(runner_type, scored=scored)
    runner._speculation.benchmark_accepted_drafts = fixed
    positions = torch.tensor(plan.positions, dtype=torch.int64)
    counts, stored = [], []
    if runner_type is DeepSeekV41CudaRunner:
        from resin_models_cuda.deepseek_v41 import cuda as runtime

        runner._broadcast_tokens = lambda value: value
        runner._model = SimpleNamespace(
            draft=lambda *a, **kw: drafts,
            forward_verify=lambda *a, **kw: (logits, object()),
            commit_dspark_main_block=lambda *a, **kw: None,
            clear_rejected_speculative_state=lambda _positions, processed, *a, **kw: (
                counts.append(processed.tolist())
            ),
        )
    else:
        runtime = inkling_runtime
        runner._draft = SimpleNamespace(
            propose=lambda *a: (drafts, None), inject_target=lambda *a: None
        )
        runner._model = SimpleNamespace(
            target=lambda *a, **kw: (logits, object(), object(), None)
        )
        runner._initial_state = lambda *a: object()
        runner._store_state = lambda _cache, ends, _state: stored.append(ends)
        monkeypatch.setattr(
            runtime,
            "select_convolution_state",
            lambda _initial, _inputs, processed, **kw: counts.append(
                processed.tolist()
            ),
        )

    prepared, result = object(), object()

    def score(raw, decision, native):
        assert scored and raw is logits and native is plan
        return prepared

    def complete(decision, **kwargs):
        assert kwargs["scores"] is prepared and kwargs["plan"] is plan
        return SimpleNamespace(result=lambda: result)

    monkeypatch.setattr(runtime, "score_completion", score)
    monkeypatch.setattr(
        runtime, "DeviceCompletion", SimpleNamespace(from_acceptance=complete)
    )
    output = runner._speculative_decode(
        plan, object(), torch.tensor(plan.token_ids, dtype=torch.int64), positions
    )
    expected = [1, 1] if scored else [2, 3]
    assert counts == [expected]
    if stored:
        torch.testing.assert_close(stored[0], positions + torch.tensor(expected) - 1)
    if scored:
        assert output is result
    else:
        assert output.token_ids.tolist() == [2, 4, 3, 4, 5]
    assert runner._sampler.calls == [(2, 3, 10)]


@pytest.mark.parametrize("fixed", [None, 1])
def test_deepseek_unscored_graph_harvests_directly_and_rolls_back(monkeypatch, fixed):
    from resin.backend.speculative import _accept_speculative_tokens
    from resin_models_cuda.deepseek_v41 import cuda as runtime

    runner, plan, _logits, drafts, targets = _fixture(DeepSeekV41CudaRunner)
    runner._speculation.benchmark_accepted_drafts = fixed
    runner.parallel_groups = None
    runner._graph_device_caches = {(2, 128): object()}
    runner._dspark_phase_events = {}
    runner._cache = SimpleNamespace(graph_plan_inputs=lambda *_args: {})
    runner._run_cuda_graph = lambda *_args: torch.cat((drafts, targets), dim=1)
    processed = []
    runner._model = SimpleNamespace(
        clear_rejected_speculative_state=lambda _positions, counts, *_args, **_kwargs: (
            processed.append(counts.tolist())
        )
    )

    def forbid_completion(*_args):
        raise AssertionError("unscored synchronous graph must harvest directly")

    runner._enqueue_graph_output = forbid_completion
    monkeypatch.setattr(runtime, "pad_sampling_inputs", lambda *_args: {})
    result = runner._graph_forward(plan, object(), shape=(2, 128))
    expected = _accept_speculative_tokens(
        drafts.tolist(), targets.tolist(), benchmark_accepted_drafts=fixed
    )
    assert result is not None
    assert result.token_ids.tolist() == sum(expected.token_rows, [])
    assert processed == [expected.processed_counts]
    assert runner._speculation.blocks == 2
    assert runner._speculation.accepted == sum(expected.accepted_counts)


def test_deepseek_graph_verification_samples_and_broadcasts_one_block():
    runner, plan, logits, drafts, targets = _fixture(DeepSeekV41CudaRunner)
    runner.parallel_groups = None
    runner.profile_graph_phases = False
    runner._graph_device_caches = {}
    runner._graph_inputs = {}
    runner._cache = SimpleNamespace(
        graph_cache=lambda *_args: object(),
        graph_device_inputs=lambda _cache: {},
    )
    runner._new_phase_timing = lambda *_args: None
    runner._parallel_graph_capacity = lambda *_args, **_kwargs: nullcontext()
    runner._reset_model_state = lambda: None
    runner._model = SimpleNamespace(
        draft=lambda *_args, **_kwargs: drafts,
        forward_verify=lambda *_args, **_kwargs: (logits, object()),
        commit_dspark_main_block=lambda *_args, **_kwargs: None,
    )
    broadcasts = []
    runner._broadcast_tokens = lambda value: (
        broadcasts.append(tuple(value.shape)) or value
    )
    # The fake uses the real CPU metadata allocation without loading CUDA kernels.
    step = runner._new_dspark_graph_step((2, 128))
    step.inputs["token_ids"].copy_(
        torch.tensor(plan.token_ids, dtype=torch.int64).view(2, 1)
    )
    step.inputs["positions"].copy_(torch.tensor(plan.positions, dtype=torch.int64))
    step.launch()
    assert runner._sampler.calls == [(2, 3, 10)]
    assert broadcasts == [(2, 2), (2, 3)]
    torch.testing.assert_close(step.output, torch.cat((drafts, targets), dim=1))
    assert runner._graph_logits[(2, 128)] is logits


@pytest.mark.parametrize("fixed", [-1, 0])
def test_inkling_graph_keeps_target_logits_and_one_device_acceptance(
    monkeypatch, fixed
):
    runner, plan, logits, drafts, targets = _fixture(InklingCudaRunner)
    runner._graph_device_caches = {}
    runner._cache = SimpleNamespace(
        graph_cache=lambda *_args: object(),
        graph_device_inputs=lambda _cache: {},
        reset_null=lambda: None,
    )
    runner._new_phase_timing = lambda *_args: None
    runner._graph_state_inputs = torch.empty(2)
    runner._draft = SimpleNamespace(
        propose=lambda *_args: (drafts, None),
        inject_target=lambda *_args: None,
    )
    runner._model = SimpleNamespace(
        target=lambda *_args, **_kwargs: (logits, object(), object(), None)
    )
    runner._initial_state = lambda *_args: object()
    state_positions = []
    runner._store_state = lambda _cache, positions, _state: state_positions.append(
        positions.clone()
    )
    state_counts = []
    monkeypatch.setattr(
        inkling_runtime,
        "select_convolution_state",
        lambda _initial, _inputs, counts, **_kwargs: state_counts.append(
            counts.clone()
        ),
    )
    step = runner._new_decode_graph_step((2, 128))
    step.inputs["token_ids"].copy_(torch.tensor(plan.token_ids, dtype=torch.int64))
    positions = torch.tensor(plan.positions, dtype=torch.int64)
    step.inputs["positions"].copy_(positions)
    step.inputs["fixed_acceptance"].fill_(fixed)
    step.launch()

    acceptance = runner._graph_acceptance[(2, 128)]
    expected = [2, 3] if fixed < 0 else [fixed + 1] * 2
    assert acceptance.counts.tolist() == expected
    assert state_counts[0].tolist() == expected
    torch.testing.assert_close(state_positions[0], positions + acceptance.counts - 1)
    assert runner._graph_logits[(2, 128)] is logits
    assert runner._sampler.calls == [(2, 3, 10)]
    torch.testing.assert_close(step.output, torch.cat((drafts, targets), dim=1))
