from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest
import torch

from resin import BatchPhase
from resin.backend.cuda.graphs import (
    CudaGraphExecutor,
    CudaGraphMode,
    CudaGraphProgram,
    CudaGraphStep,
)
from resin.backend.cuda.runner import CudaRunner
from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner
from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner
from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner
from resin_models_cuda.inkling2.cache import CacheArena, CacheGeometry
from resin_models_cuda.inkling2.cuda import Inkling2CudaRunner
from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner


def _fail(message: str) -> NoReturn:
    raise AssertionError(message)


@pytest.mark.parametrize("enqueue", [False, True])
@pytest.mark.parametrize("larger_graph", [False, True])
def test_inkling_distributed_graph_selection_covers_the_complete_native_lease(
    enqueue, larger_graph
):
    runner: Any = object.__new__(Inkling2CudaRunner)
    runner.speculative_draft_tokens = 7
    runner._sampling_graph_algorithm = lambda _plan: 1
    geometry = CacheGeometry(2, 3, 1, 2, 2, 8, 32, 3, 128, 128, prefill_chunk_size=128)
    runner._cache = CacheArena("cpu", geometry.layout(8), geometry)
    # Position 119 and this step's eight writes fit in one page. The native
    # continuation lease reserves a second page which a 120-token graph lacks.
    plan = SimpleNamespace(num_sequences=1, positions=np.array([119]))
    cache_plan = SimpleNamespace(
        rows=1,
        groups=tuple(
            SimpleNamespace(block_indptr=np.array([0, count]))
            for count in (2, 2, 2, 2, 1)
        ),
    )
    shapes = ((1, 120, 1), (1, 256, 1)) if larger_graph else ((1, 120, 1),)
    runner._select_cuda_graph = lambda _mode, *, match: next(
        (shape for shape in shapes if match(shape)), None
    )
    votes, launched = [], []

    def vote(_mode, _context, **options):
        votes.append(options)
        return options["required_shape"] if options["eligible"] else None

    runner._select_distributed_cuda_graph = vote
    runner._graph_forward = lambda *_args, **options: launched.append(options)
    used, _ = runner._distributed_graph(
        plan, object(), SimpleNamespace(cache_plan=cache_plan), enqueue=enqueue
    )
    assert used is larger_graph
    assert votes[0]["eligible"] is larger_graph
    if larger_graph:
        assert votes[0]["required_shape"] == (1, 256, 1)
        assert launched[0]["shape"] == (1, 256, 1)
    else:
        assert launched == []


@pytest.mark.parametrize(
    "runner_type,args,kwargs",
    [
        (
            DeepSeekV41CudaRunner,
            ("unused",),
            {"max_seq_len": 32, "tensor_parallel": SimpleNamespace(rank=0, size=1)},
        ),
        (Inkling2CudaRunner, ("unused",), {"max_seq_len": 32}),
        (Qwen3_5CudaRunner, (), {}),
        (GptOssCudaRunner, (), {}),
        (Gemma4CudaRunner, (), {}),
    ],
)
def test_model_runner_passes_graph_policies_to_shared_runtime(
    monkeypatch, runner_type, args, kwargs
):
    class ReachedSharedRuntime(Exception):
        pass

    received = []

    def initialize(_self, **options):
        received.append((options["cuda_graph_prefill"], options["cuda_graph_decode"]))
        raise ReachedSharedRuntime

    monkeypatch.setattr(CudaRunner, "__init__", initialize)
    with pytest.raises(ReachedSharedRuntime):
        runner_type(
            *args,
            device="cuda",
            **kwargs,
            cuda_graph_prefill="disabled",
            cuda_graph_decode="breakable",
        )
    assert received == [("disabled", "breakable")]


@pytest.mark.parametrize(
    "runner_type",
    [
        DeepSeekV41CudaRunner,
        Inkling2CudaRunner,
        Qwen3_5CudaRunner,
        GptOssCudaRunner,
        Gemma4CudaRunner,
    ],
)
def test_model_prefill_rejects_uncaptured_graphs_before_accessing_model_state(
    runner_type,
):
    runner = object.__new__(runner_type)
    runner._cuda_graph_executor = CudaGraphExecutor(
        "cpu",
        {
            "prefill": CudaGraphMode(
                ((4, 1),),
                lambda _: _fail("uncaptured graph must not be constructed"),
                lambda _: {},
            )
        },
        synchronize_failure=lambda: _fail("no CUDA work should have begun"),
    )
    with pytest.raises(RuntimeError, match="call runner.capture_cuda_graphs") as error:
        runner.forward(SimpleNamespace(phase=BatchPhase.PREFILL))
    assert not getattr(error.value, "_resin_post_accept", False)
    assert not runner._cuda_graph_executor.stats["prefill"]["poisoned"]


@pytest.mark.parametrize("policy", ["full", "breakable"])
@pytest.mark.parametrize("parallel", [False, True])
def test_deepseek_decode_policy_places_broadcast_inside_or_between_graphs(
    monkeypatch, policy, parallel
):
    runner = object.__new__(DeepSeekV41CudaRunner)
    runner.speculative_draft_tokens = 0
    runner.parallel_groups = None
    runner.vocab_size = 2
    runner.cuda_graph_decode = policy
    runner.device = torch.device("cpu")
    setattr(
        runner,
        "tensor_parallel",
        SimpleNamespace(enabled=parallel, process_group=object()),
    )
    runner._graph_device_caches = {}
    runner._graph_inputs = {}
    runner._graph_logits = {}
    setattr(
        runner,
        "_cache",
        SimpleNamespace(
            graph_cache=lambda *_args: object(),
            graph_device_inputs=lambda _cache: {},
        ),
    )
    events = []

    def forward(token_ids, positions, _capacity, *, cache):
        events.append("model")
        return torch.stack((token_ids[:, 0], positions), dim=1)

    def sample(logits, _sampling, *, out):
        events.append("sample")
        out.copy_(logits.argmax(dim=1))

    def broadcast(output, *, src, group):
        events.append("broadcast")
        assert src == 0 and group is runner.tensor_parallel.process_group
        output.fill_(7)

    setattr(runner, "_model", SimpleNamespace(forward_decode=forward))
    setattr(
        runner,
        "_sampler",
        SimpleNamespace(
            empty=lambda *_args: object(),
            sample_tensors=sample,
            graph_inputs=lambda _sampling: {},
        ),
    )
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    program = runner._new_decode_graph_step((2, 32))
    program.inputs["token_ids"].zero_()
    program.inputs["positions"].fill_(1)

    if parallel and policy == "breakable":
        assert isinstance(program, CudaGraphProgram)
        assert [region.capture for region in program.regions] == [True, False]
        program.regions[0].launch()
        assert events == ["model", "sample"]
        program.regions[1].launch()
    else:
        assert isinstance(program, CudaGraphStep)
        program.launch()

    assert events == ["model", "sample", *(["broadcast"] if parallel else [])]
    assert program.output.tolist() == ([7, 7] if parallel else [1, 1])


def cpu_prefill_regions(
    operations,
    hidden_size,
    *,
    max_tokens=8,
    token_ids=False,
    capture_context=nullcontext,
):
    """Exercise real staging/launch closures with shared CPU buffers, without CUDA."""
    runner = object.__new__(CudaRunner)
    runner.device = torch.device("cpu")
    runner.cuda_graph_prefill = "breakable"
    runner._prefill_graph_active = False
    runner._prefill_operations = ()
    runner._prefill_tensor_graphs = {}
    setattr(
        runner, "_cuda_graph_executor", SimpleNamespace(add_mode=lambda *_args: None)
    )
    callbacks = runner._configure_prefill_graphs(
        tuple(operations),
        hidden_size=hidden_size,
        max_tokens=max_tokens,
        token_ids=token_ids,
        capture_context=capture_context,
    )
    programs, replays = {}, []

    def replay(mode, shape, _inputs):
        assert mode == "prefill"
        if shape not in programs:
            programs[shape] = runner._prefill_tensor_graphs["prefill"].build(shape)
        program = programs[shape]
        program.launch()
        replays.append(shape)
        return program.output

    setattr(runner, "_run_cuda_graph", replay)
    return runner, callbacks, replays
