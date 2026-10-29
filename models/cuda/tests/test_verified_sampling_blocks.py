"""Verification samples one complete block without copying vocabulary slices."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from resin import BatchPhase, BatchPlan
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


def _fixture(runner_type):
    runner: Any = object.__new__(runner_type)
    runner.device = torch.device("cpu")
    runner.speculative_draft_tokens = 2
    runner.vocab_size = 10
    runner.config = SimpleNamespace(vocab_size=10, hidden_size=2, num_hidden_layers=1)
    runner._speculation = SpeculativeStats(2)
    runner._eager_phase_timing = lambda *_args: None
    plan = BatchPlan(
        sequence_ids=[7, 8],
        context_lengths=[14, 25],
        token_ids=[2, 3],
        token_indptr=[0, 1, 2],
        phase=BatchPhase.DECODE,
    )
    logits = torch.arange(2 * 3 * 20, dtype=torch.float32).view(2, 3, 20)[..., ::2]
    drafts = torch.tensor([[2, 6], [3, 4]])
    targets = torch.tensor([[2, 4, 7], [3, 4, 5]])
    runner._sampler = _BlockSampler(logits, targets)
    return runner, plan, logits, drafts, targets


@pytest.mark.parametrize("runner_type", [DeepSeekV41CudaRunner, InklingCudaRunner])
def test_eager_verification_samples_entire_strided_block_once(monkeypatch, runner_type):
    runner, plan, logits, drafts, _ = _fixture(runner_type)
    positions = torch.tensor(plan.positions, dtype=torch.int64)
    anchors = torch.tensor(plan.token_ids, dtype=torch.int64)
    broadcasts = []
    if runner_type is DeepSeekV41CudaRunner:
        runner._broadcast_tokens = lambda value: (
            broadcasts.append(tuple(value.shape)) or value
        )
        runner._model = SimpleNamespace(
            draft=lambda *_args, **_kwargs: drafts,
            forward_verify=lambda *_args, **_kwargs: (logits, object()),
            commit_dspark_main_block=lambda *_args, **_kwargs: None,
            clear_rejected_speculative_state=lambda *_args, **_kwargs: None,
        )
        output = runner._speculative_decode(plan, object(), anchors, positions)
        assert broadcasts == [(2, 2), (2, 3)]
    else:
        runner._draft = SimpleNamespace(
            propose=lambda *_args: (drafts, None),
            inject_target=lambda *_args: None,
        )
        runner._model = SimpleNamespace(
            target=lambda *_args, **_kwargs: (logits, object(), object(), None),
        )
        runner._initial_state = lambda *_args: object()
        runner._store_state = lambda *_args: None
        monkeypatch.setattr(
            inkling_runtime, "select_convolution_state", lambda *_args, **_kwargs: None
        )
        output = runner._speculative_decode(plan, object(), anchors, positions)
    assert runner._sampler.calls == [(2, 3, 10)]
    assert output.token_ids.tolist() == [2, 4, 3, 4, 5]
    assert output.token_indptr.tolist() == [0, 2, 5]
    assert runner._speculation.accepted == 3


def test_deepseek_graph_verification_samples_and_broadcasts_one_block():
    runner, plan, logits, drafts, targets = _fixture(DeepSeekV41CudaRunner)
    runner.parallel_groups = None
    runner.profile_graph_phases = False
    runner._graph_device_caches = {}
    runner._graph_inputs = {}
    runner._graph_logits = {}
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
