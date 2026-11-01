from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest
import torch

from resin import BatchPhase, BatchPlan, ForwardOutput, Parallelism, SamplingParams
from resin.backend.cuda.graphs import CudaGraphMode
from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.cuda.runner import CudaRunner
from resin.backend.cuda.sampling import FusedSampler
from resin.backend.distributed import StepContext
from resin.backend.sampling import token_output
from resin.backend.topology import RankGroup
from resin.models.gpt_oss.plan import AttentionBatch, AttentionGroup
from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner


def _groups(parallelism, rank=0):
    broadcasts = []

    def broadcast(tensor, *, src, group):
        broadcasts.append((src, group, tensor.tolist()))

    groups = SimpleNamespace(
        topology=parallelism,
        rank=rank,
        distributed=SimpleNamespace(broadcast=broadcast),
        broadcasts=broadcasts,
    )
    for name, partitions in parallelism.partitions().items():
        for ranks in partitions:
            if rank in ranks:
                setattr(
                    groups, name, RankGroup(ranks, (name, ranks), ranks.index(rank))
                )
    groups.device_domain = groups.domain
    groups.device_replica = groups.replica
    return groups


class _Guard(nullcontext):
    def __init__(self):
        super().__init__()
        self.synchronizations = 0

    def __call__(self):
        self.synchronizations += 1


def _runner(monkeypatch, groups):
    import resin_models_cuda.gpt_oss.cuda as implementation

    runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
    runner.parallel_groups = groups
    runner.tensor_parallel = TensorParallelContext()
    runner.device = torch.device("cpu")
    runner.hidden_size = 32
    setattr(runner, "cache_layout", object())
    runner.cuda_graph_prefill = "disabled"
    runner.cuda_graph_decode = "disabled"
    runner._prefill_graph_active = False
    runner._graph_logits = {}
    setattr(runner, "_cuda_failure", _Guard())
    setattr(runner, "_check_cuda_graphs", lambda _phase: None)
    setattr(runner, "_graph_forward", lambda *_args, **kwargs: None)
    setattr(runner, "_select_distributed_cuda_graph", lambda *args, **kwargs: None)
    setattr(runner, "_distributed_prefill_graphs", lambda context: nullcontext())
    setattr(runner, "_sampler", lambda logits, plan: logits.argmax(dim=-1))
    calls, tensorized, lowered = [], [], []
    setattr(runner, "calls", calls)
    setattr(runner, "tensorized", tensorized)
    setattr(runner, "lowered", lowered)
    setattr(
        runner,
        "_model",
        SimpleNamespace(
            lm_head=lambda hidden: hidden,
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        mlp=SimpleNamespace(
                            experts=SimpleNamespace(
                                forward_empty=lambda index=index: calls.append(
                                    ("empty", index)
                                )
                            )
                        )
                    )
                    for index in range(3)
                ]
            ),
        ),
    )

    def batches(plan, cache_layout):
        assert cache_layout is runner.cache_layout
        lowered.append(plan)
        batch = SimpleNamespace(
            token_indices=torch.arange(len(plan.token_ids)),
            sequence_indices=tuple(range(plan.num_sequences)),
            last_token_indices=torch.tensor(plan.token_indptr[1:].astype(np.int64) - 1),
            phase=plan.phase,
            query_lengths=tuple(np.diff(plan.token_indptr)),
            group=lambda name: SimpleNamespace(
                width=1 if name == "full" else runner.window_table_width
            ),
        )
        return ((batch,), None) if plan.phase == BatchPhase.PREFILL else ((), batch)

    def tensorize(host, staging, prefix):
        tensorized.append(prefix)
        return SimpleNamespace(
            token_indices=host.token_indices,
            sequence_indices=torch.tensor(host.sequence_indices),
            last_token_indices=host.last_token_indices,
        )

    def forward(
        token_ids, positions, batch, *, context_parallel=None, full_hidden=False
    ):
        calls.append(("batch", len(token_ids), positions.tolist()))
        return torch.nn.functional.one_hot(
            token_ids if full_hidden else token_ids[batch.last_token_indices],
            num_classes=runner.hidden_size,
        ).bfloat16()

    monkeypatch.setattr(implementation, "build_attention_batches", batches)
    monkeypatch.setattr(
        implementation,
        "TensorStager",
        lambda _device: SimpleNamespace(
            device=lambda _name, values, dtype: torch.tensor(
                values.tolist(), dtype=dtype
            )
        ),
    )
    setattr(runner, "_tensorize", tensorize)
    setattr(runner, "_forward_batch", forward)
    return runner


def _context(groups, phase, sizes, plan):
    return StepContext(
        groups=groups,
        step_id=0,
        phase=phase,
        replica_sizes=sizes,
        num_sequences=0 if plan is None else plan.num_sequences,
        num_tokens=0 if plan is None else len(plan.token_ids),
    )


@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
def test_distributed_forward_packs_unequal_replica_rows_once(monkeypatch, phase):
    topology = Parallelism(world_size=2, ep=2, attn_dp=2)
    if phase == BatchPhase.PREFILL:
        plans = (
            BatchPlan(
                sequence_ids=[7, 8],
                context_lengths=[0, 9],
                token_ids=[2, 3, 4, 5, 6],
                token_indptr=[0, 3, 5],
                phase=phase,
            ),
            BatchPlan(
                sequence_ids=[7],
                context_lengths=[0],
                token_ids=[9],
                token_indptr=[0, 1],
                phase=phase,
            ),
        )
        sizes = ((2, 5), (1, 1))
    else:
        plans = (
            BatchPlan(
                sequence_ids=[7, 8],
                context_lengths=[11, 15],
                token_ids=[4, 6],
                token_indptr=[0, 1, 2],
                phase=phase,
            ),
            BatchPlan(
                sequence_ids=[7],
                context_lengths=[3],
                token_ids=[9],
                token_indptr=[0, 1],
                phase=phase,
            ),
        )
        sizes = ((2, 2), (1, 1))
    for rank, plan in enumerate(plans):
        groups = _groups(topology, rank)
        runner = _runner(monkeypatch, groups)
        assert runner.forward_distributed(
            plan, _context(groups, phase, sizes, plan)
        ).token_ids.tolist() == ([4, 6] if rank == 0 else [9])
        assert runner.calls == [("batch", len(plan.token_ids), plan.positions.tolist())]
        assert runner.tensorized == [
            "prefill.0" if phase == BatchPhase.PREFILL else "decode"
        ]
        assert groups.broadcasts == []
        assert runner.lowered == [plan]


@pytest.mark.parametrize(
    "phase,captured",
    [
        (BatchPhase.PREFILL, False),
        (BatchPhase.DECODE, False),
        (BatchPhase.DECODE, True),
    ],
)
@pytest.mark.parametrize("idle", [False, True])
def test_worker_prepared_attention_is_reused_for_every_execution_path(
    monkeypatch, phase, captured, idle
):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2))
    runner = _runner(monkeypatch, groups)
    plan = (
        None
        if idle
        else BatchPlan(
            sequence_ids=[7],
            context_lengths=[3],
            token_ids=[9],
            token_indptr=[0, 1],
            phase=phase,
        )
    )
    context = _context(groups, phase, ((0, 0) if idle else (1, 1), (1, 1)), plan)
    prepared = runner.validate_distributed(plan, context)
    assert runner.lowered == ([] if idle else [plan])

    def repeated(*args, **kwargs) -> NoReturn:
        raise AssertionError("worker-prepared host metadata must be reused")

    monkeypatch.setattr(runner, "validate_distributed", repeated)
    monkeypatch.setattr(runner, "_attention_batches", repeated)
    replayed = []
    if captured:
        monkeypatch.setattr(
            runner, "_select_distributed_cuda_graph", lambda *args, **kwargs: (2, 1)
        )

        def replay(plan_arg, host, *, shape):
            assert plan_arg is plan and host is prepared[1]
            replayed.append(shape)
            return token_output([9])

        monkeypatch.setattr(runner, "_graph_forward", replay)
        monkeypatch.setattr(
            runner, "_decode_graph_capture_inputs", lambda shape: {"ep_valid_rows": 2}
        )

        def empty_replay(mode, shape, inputs):
            assert mode == "decode" and inputs["ep_valid_rows"] == 0
            replayed.append(shape)

        monkeypatch.setattr(runner, "_run_cuda_graph", empty_replay)

    output = runner.forward_distributed(plan, context, prepared=prepared)
    if idle:
        assert output is None
    else:
        assert isinstance(output, ForwardOutput)
        assert output.token_ids.tolist() == [9]
    assert runner.lowered == ([] if idle else [plan])
    assert replayed == ([(2, 1)] if captured else [])
    assert runner.tensorized == (
        []
        if idle or captured
        else ["prefill.0" if phase == BatchPhase.PREFILL else "decode"]
    )


def test_distributed_prefill_scores_all_predecessors_from_one_forward(monkeypatch):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2))
    runner = _runner(monkeypatch, groups)
    plan = BatchPlan(
        sequence_ids=[7, 8],
        context_lengths=[0, 9],
        token_ids=[2, 2, 4, 5, 5],
        token_indptr=[0, 3, 5],
        sampling_params=[SamplingParams(prompt_logprobs=True)] * 2,
        prompt_score_indices=[0, 1, 2, 3],
        prompt_score_token_ids=[2, 4, 9, 5],
        prompt_score_indptr=[0, 3, 4],
    )
    output = runner.forward_distributed(
        plan, _context(groups, BatchPhase.PREFILL, ((2, 5), (0, 0)), plan)
    )
    assert isinstance(output, ForwardOutput)
    logits = (
        torch.nn.functional.one_hot(
            torch.tensor(plan.token_ids.astype(np.int64)),
            num_classes=runner.hidden_size,
        )
        .float()
        .log_softmax(-1)
    )
    np.testing.assert_allclose(
        output.prompt_logprobs, logits[[0, 1, 2, 3], [2, 4, 9, 5]].numpy(), atol=1e-6
    )
    np.testing.assert_array_equal(output.prompt_logprob_indptr, [0, 3, 4])
    np.testing.assert_array_equal(output.token_ids, [4, 5])
    assert len(runner.calls) == 1


@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
def test_idle_replica_joins_each_expert_layer_without_attention_or_cache(
    monkeypatch, phase
):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2))
    runner = _runner(monkeypatch, groups)

    assert (
        runner.forward_distributed(
            None, _context(groups, phase, ((0, 0), (1, 1)), None)
        )
        is None
    )

    assert runner.calls == [("empty", 0), ("empty", 1), ("empty", 2)]
    assert runner.tensorized == []
    assert groups.broadcasts == []
    assert runner._cuda_failure.synchronizations == 1


def test_sample_broadcast_uses_complete_attention_replica_group(monkeypatch):
    groups = _groups(Parallelism(world_size=2, ep=2), rank=1)
    runner = _runner(monkeypatch, groups)
    plan = BatchPlan(
        sequence_ids=[7],
        context_lengths=[3],
        token_ids=[9],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
    )

    assert runner.forward_distributed(
        plan, _context(groups, BatchPhase.DECODE, ((1, 1),), plan)
    ).token_ids.tolist() == [9]

    assert groups.broadcasts == [(0, groups.device_replica.handle, [9])]


def test_distributed_metadata_mismatch_is_rejected_before_execution(monkeypatch):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2))
    runner = _runner(monkeypatch, groups)
    plan = BatchPlan(
        sequence_ids=[7],
        context_lengths=[0],
        token_ids=[2],
        token_indptr=[0, 1],
        phase=BatchPhase.PREFILL,
    )

    with pytest.raises(ValueError, match="metadata"):
        runner.forward_distributed(
            plan, _context(groups, BatchPhase.DECODE, ((1, 1), (0, 0)), plan)
        )
    assert runner.calls == []
    assert runner.tensorized == []


@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
@pytest.mark.parametrize("idle", [False, True])
def test_missing_graph_capture_is_rejected_in_worker_readiness_validation(
    monkeypatch, phase, idle
):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2))
    runner = _runner(monkeypatch, groups)
    runner.cuda_graph_decode = "full"
    runner.cuda_graph_prefill = "breakable"
    runner._cuda_graph_executor = None
    pending = "prefill" if phase == BatchPhase.PREFILL else "decode"

    def unexpected_capture(_shape) -> NoReturn:
        raise AssertionError("readiness validation must precede graph construction")

    runner._configure_cuda_graphs(
        {
            name: CudaGraphMode(
                ((1, 1),) if name == pending else (),
                unexpected_capture,
                lambda _: {},
            )
            for name in ("prefill", "decode")
        }
    )
    runner._check_cuda_graphs = CudaRunner._check_cuda_graphs.__get__(runner)
    plan = (
        None
        if idle
        else BatchPlan(
            sequence_ids=[7],
            context_lengths=[0],
            token_ids=[2],
            token_indptr=[0, 1],
            phase=phase,
        )
    )
    context = _context(groups, phase, ((0, 0) if idle else (1, 1), (1, 1)), plan)

    with pytest.raises(RuntimeError, match="not been captured"):
        runner.validate_distributed(plan, context)

    assert runner.calls == runner.tensorized == []
    assert runner._cuda_failure.synchronizations == 0


@pytest.mark.parametrize(
    ("parallelism", "options", "message"),
    [
        (
            Parallelism(world_size=2, ep=2),
            {"tensor_parallel": TensorParallelContext()},
            "either",
        ),
    ],
)
def test_unsupported_distributed_preparation_fails_before_cuda(
    parallelism, options, message
):
    kwargs: dict[str, Any] = dict(
        cuda_graph_prefill="disabled", cuda_graph_decode="disabled"
    )
    kwargs.update(options)
    with pytest.raises(ValueError, match=message):
        GptOssCudaRunner(device="cuda", parallel_groups=_groups(parallelism), **kwargs)


@pytest.mark.parametrize(
    "parallelism",
    [
        Parallelism(world_size=2, ep=2, cp=2),
        Parallelism(world_size=2),
        Parallelism(world_size=2, ep=2, attn_tp=2),
        Parallelism(world_size=8, ep=2, attn_dp=2, cp=2, attn_tp=2),
    ],
)
def test_distributed_graph_policies_pass_topology_preparation(monkeypatch, parallelism):
    import resin_models_cuda.gpt_oss.cuda as implementation

    class CudaEntry(Exception):
        pass

    def enter_cuda(self, **kwargs):
        assert kwargs["cuda_graph_prefill"] == "breakable"
        assert kwargs["cuda_graph_decode"] == "full"
        raise CudaEntry

    monkeypatch.setattr(implementation.CudaRunner, "__init__", enter_cuda)
    with pytest.raises(CudaEntry):
        GptOssCudaRunner(
            device="cuda",
            parallel_groups=_groups(parallelism),
            kernel_attention=lambda *args: None,
        )


def test_distributed_decode_replays_the_common_graph_bucket(monkeypatch):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2), rank=1)
    runner = _runner(monkeypatch, groups)
    plan = BatchPlan(
        sequence_ids=[7, 8],
        context_lengths=[11, 15],
        token_ids=[4, 6],
        token_indptr=[0, 1, 2],
        phase=BatchPhase.DECODE,
    )
    context = _context(groups, BatchPhase.DECODE, ((1, 1), (2, 2)), plan)
    selected, replayed = [], []

    def select(mode, context_arg, **kwargs):
        selected.append((mode, context_arg, kwargs))
        return (4, 2)

    def replay(plan_arg, host, *, shape):
        replayed.append((plan_arg, host.phase, shape))
        return token_output([4, 6])

    runner._select_distributed_cuda_graph = select
    runner._graph_forward = replay

    assert runner.forward_distributed(plan, context).token_ids.tolist() == [4, 6]
    assert selected == [("decode", context, {"required_shape": (2, 1)})]
    assert replayed == [(plan, BatchPhase.DECODE, (4, 2))]
    assert runner.calls == runner.tensorized == []


def test_idle_graph_replica_rejoins_with_zero_valid_expert_rows(monkeypatch):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2))
    runner = _runner(monkeypatch, groups)
    replayed = []
    runner._select_distributed_cuda_graph = lambda *args, **kwargs: (2, 1)
    runner._decode_graph_capture_inputs = lambda shape: {
        "shape": shape,
        "ep_valid_rows": shape[0],
    }
    runner._run_cuda_graph = lambda mode, shape, inputs: replayed.append(inputs)
    idle = _context(groups, BatchPhase.DECODE, ((0, 0), (2, 2)), None)

    assert runner.forward_distributed(None, idle) is None
    assert replayed == [{"shape": (2, 1), "ep_valid_rows": 0}]
    assert runner.calls == runner.tensorized == []

    plan = BatchPlan(
        sequence_ids=[7],
        context_lengths=[3],
        token_ids=[9],
        token_indptr=[0, 1],
        phase=BatchPhase.DECODE,
    )
    runner._graph_forward = lambda *args, **kwargs: token_output([9])
    assert runner.forward_distributed(
        plan, _context(groups, BatchPhase.DECODE, ((1, 1), (2, 2)), plan)
    ).token_ids.tolist() == [9]


def test_idle_prefill_replica_joins_captured_ffns(monkeypatch):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2))
    runner = _runner(monkeypatch, groups)
    runner.cuda_graph_prefill = "breakable"
    captured = []

    @contextmanager
    def aligned(context):
        runner._prefill_graph_active = True
        try:
            yield
        finally:
            runner._prefill_graph_active = False

    runner._distributed_prefill_graphs = aligned
    runner._empty_prefill_regions = lambda: captured.append("ffns")

    assert (
        runner.forward_distributed(
            None, _context(groups, BatchPhase.PREFILL, ((0, 0), (2, 5)), None)
        )
        is None
    )
    assert captured == ["ffns"]
    assert runner.calls == runner.tensorized == []
    assert runner._cuda_failure.synchronizations == 1


def _decode_host(runner, plan):
    rows = plan.num_sequences
    return AttentionBatch(
        phase=BatchPhase.DECODE,
        token_indices=np.arange(rows, dtype=np.int64),
        sequence_indices=tuple(range(rows)),
        query_lengths=(1,) * rows,
        groups=tuple(
            AttentionGroup(
                name=name,
                width=width,
                page_table=np.ones(rows * width, dtype=np.uint64),
                lengths=tuple(int(value) + 1 for value in plan.context_lengths),
                append_slots=np.arange(rows, dtype=np.int64),
            )
            for name, width in (("full", 1), ("sliding", runner.window_table_width))
        ),
    )


def test_decode_graph_retains_logits_and_stages_active_row_count(monkeypatch):
    groups = _groups(Parallelism(world_size=2, ep=2, attn_dp=2))
    runner = _runner(monkeypatch, groups)
    runner.vocab_size = 3
    runner._sampler = FusedSampler
    runner._cuda_graph_mode_enabled = lambda mode: True
    shape = (4, 2)
    logits = torch.tensor([[1.0, 3.0, 0.0], [2.0, 1.0, 4.0], [0.0] * 3, [0.0] * 3])
    runner._graph_logits = {shape: logits}
    replayed = []

    def replay(mode, shape_arg, inputs):
        replayed.append((mode, shape_arg, inputs))
        return torch.tensor([1, 2, 0, 0])

    runner._run_cuda_graph = replay
    plan = BatchPlan(
        sequence_ids=[7, 8],
        context_lengths=[11, 15],
        token_ids=[0, 0],
        token_indptr=[0, 1, 2],
        phase=BatchPhase.DECODE,
        sampling_params=[SamplingParams(logprobs=True, seed=1)] * 2,
    )

    output = GptOssCudaRunner._graph_forward(
        runner, plan, _decode_host(runner, plan), shape=shape
    )

    assert isinstance(output, ForwardOutput)
    assert output.logprobs is not None
    assert output.token_ids.tolist() == [1, 2]
    expected = logits[:2].log_softmax(-1)
    assert output.logprobs.tolist() == pytest.approx([expected[0, 1], expected[1, 2]])
    assert len(replayed) == 1
    assert replayed[0][2]["ep_valid_rows"] == 2
    assert replayed[0][2]["full_page_table"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    torch.testing.assert_close(runner._last_graph_logits, logits[:2])


def test_decode_capture_wraps_experts_and_replica_sampling_broadcast(monkeypatch):
    groups = _groups(Parallelism(world_size=2, ep=2), rank=1)
    runner = _runner(monkeypatch, groups)
    runner.vocab_size = runner.hidden_size
    runner._sampler = SimpleNamespace(
        empty=lambda *args: object(),
        graph_inputs=lambda sampling: {},
        sample_tensors=lambda logits, sampling, *, out: out.copy_(logits.argmax(-1)),
    )
    scopes = []

    @contextmanager
    def capacity(rows, *, valid_rows):
        scopes.append((rows, int(valid_rows)))
        yield

    runner._parallel_graph_capacity = capacity
    step = runner._new_decode_graph_step((2, 1))
    step.inputs["token_ids"].copy_(torch.tensor([4, 6]))
    step.inputs["positions"].copy_(torch.tensor([3, 5]))
    step.inputs["sequence_indices"].copy_(torch.arange(2))
    step.inputs["ep_valid_rows"].fill_(1)

    step.launch()

    assert scopes == [(2, 1)]
    assert step.output.tolist() == [4, 6]
    assert groups.broadcasts == [(0, groups.device_replica.handle, [4, 6])]
    assert runner._graph_logits[(2, 1)].argmax(-1).tolist() == [4, 6]
