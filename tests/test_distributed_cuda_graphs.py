"""Graph buckets are collective choices across the expert communication domain."""

from __future__ import annotations

import multiprocessing
import os
import socket
import time
from datetime import timedelta
from types import SimpleNamespace
from typing import NoReturn

import pytest
import torch
import torch.distributed as dist

from resin import BatchPhase, CacheGroupLayout, CacheLayout, Parallelism, RunnerConfig
from resin.backend.cuda.graphs import CudaGraphMode
from resin.backend.cuda.parallel import (
    TensorParallelContext,
    replicated_cache_capacity,
)
from resin.backend.cuda.runner import CudaRunner
from resin.backend.distributed import StepContext
from resin.backend.expert_parallel import ExpertParallelDispatch
from resin.backend.topology import ParallelGroups, RankGroup


def _unreachable() -> NoReturn:
    raise AssertionError("fixture graph builder must not run")


class _CpuRunner(CudaRunner):
    """Exercise coordination and graph ownership without constructing CUDA."""

    def forward(self, plan):
        raise AssertionError("this fixture only tests graph coordination")

    def __init__(self, groups, shapes=((1, 4), (2, 4), (4, 4), (4, 8))):
        self.device = torch.device("cpu")
        self.cuda_graph_decode = "full"
        self.cuda_graph_prefill = "breakable"
        self.prefill_chunk_size = 4
        self._prefill_graph_active = False
        self._prefill_operations = ()
        self._prefill_tensor_graphs = {}
        self._cuda_failure = lambda: None
        self._cuda_graph_executor = None
        self._model = torch.nn.Module()
        self.runner_config = RunnerConfig()
        self._configure_parallel_graphs(self._model, groups)
        self._configure_cuda_graphs(
            {"decode": CudaGraphMode(shapes, lambda _: _unreachable(), lambda _: {})}
        )


def _context(groups, *, phase=BatchPhase.DECODE, sizes=((1, 2), (3, 6))):
    rows, tokens = sizes[groups.rank]
    return StepContext(groups, 0, phase, sizes, rows, tokens)


def _prefill_fixture(groups, *, regions=2):
    """Use real exchanges with CPU launches of the persistent graph closures."""

    runner = _CpuRunner(groups, shapes=())
    received = []

    def local(hidden, indices, weights):
        received.append((hidden.shape[0], int(weights[:, 0].count_nonzero())))
        coefficient = ((indices == groups.rank) * weights).sum(1) * (groups.rank + 1)
        # Nonzero expert output bias makes padded rows observable unless the
        # persistent valid-row scalar suppresses their routing probabilities.
        return (hidden.float() + 1) * coefficient[:, None]

    dispatch = ExpertParallelDispatch(local, num_experts=2, group=groups.ep)
    setattr(runner._model, "dispatch", dispatch)

    def operation(hidden, _ids):
        indices = torch.tensor([0, 1], dtype=torch.int32).expand(hidden.shape[0], 2)
        weights = torch.tensor([0.25, 0.75]).expand(hidden.shape[0], 2)
        return dispatch(hidden, indices, weights)

    runner._configure_prefill_graphs(
        (operation,) * regions, hidden_size=2, max_tokens=4
    )
    steps = {
        shape: runner._prefill_tensor_graphs["prefill"].build(shape)
        for shape in runner._require_cuda_graph_executor().mode("prefill").shapes
    }
    replays = []

    def replay(mode, shape, inputs):
        assert mode == "prefill"
        step = steps[shape]
        assert step.inputs is inputs
        replays.append((shape, int(inputs["ep_valid_rows"])))
        step.launch()
        return step.output

    setattr(runner, "_run_cuda_graph", replay)
    return runner, replays, received


def _prefill_rounds(groups):
    runner, replays, received = _prefill_fixture(groups)
    for sizes, capacity in (
        (((1, 1), (1, 3)), 4),
        (((0, 0), (1, 2)), 2),
        (((1, 3), (1, 1)), 4),
    ):
        context = _context(groups, phase=BatchPhase.PREFILL, sizes=sizes)
        tokens = context.num_tokens
        hidden = torch.arange(tokens * 2, dtype=torch.bfloat16).reshape(tokens, 2)
        hidden += groups.rank
        with runner._distributed_prefill_graphs(context):
            assert runner._distributed_prefill_bucket == capacity
            if context.placeholder:
                runner._empty_prefill_regions()
            else:
                expected = hidden
                output = hidden
                for region in range(2):
                    output = runner._run_prefill_region(region, output)
                    expected = ((expected.float() + 1) * 1.75).bfloat16()
                torch.testing.assert_close(output, expected)
        assert not runner._prefill_graph_active
        assert runner._distributed_prefill_bucket is None
        assert replays[-2:] == [((capacity, 1), tokens), ((capacity, 2), tokens)]
        assert received[-2:] == [(2 * capacity, sum(n for _, n in sizes))] * 2

    # Packed prefill can exceed a per-sequence chunk limit. Clamping this
    # request to four would select a graph too small for the five live rows.
    # Nested model scopes must preserve the collective decision to run eager.
    context = _context(groups, phase=BatchPhase.PREFILL, sizes=((1, 5), (0, 0)))
    hidden = torch.ones(context.num_tokens, 2, dtype=torch.bfloat16)
    replay_count = len(replays)
    with runner._distributed_prefill_graphs(context):
        assert runner._distributed_prefill_bucket is None
        assert not runner._prefill_graph_active
        with runner._prefill_graphs():
            assert not runner._prefill_graph_active
            expected = hidden
            output = hidden
            for region in range(2):
                output = runner._run_prefill_region(region, output)
                expected = ((expected.float() + 1) * 1.75).bfloat16()
            torch.testing.assert_close(output, expected)
    assert len(replays) == replay_count
    assert runner.cuda_graph_stats["prefill"]["eager_fallback_count"] == 1
    assert received[-2:] == [(5, 5)] * 2
    assert not runner._prefill_graph_active
    assert not runner._distributed_prefill_scope

    # A model that executes one sequence at a time explicitly requests the
    # bounded chunk bucket instead of the total packed token count.
    with runner._distributed_prefill_graphs(context, per_sequence=True):
        assert runner._distributed_prefill_bucket == 4
        assert runner._prefill_graph_active
    assert runner._distributed_prefill_bucket is None


def _compact_prefill_rounds(groups):
    runner, replays, received = _prefill_fixture(groups, regions=4)
    for sizes, retained, capacities in (
        (((1, 4), (1, 3)), (1, 2), (4, 2)),
        (((0, 0), (1, 4)), (0, 1), (4, 1)),
        # A peer needing full prompt outputs keeps every consumer at full size.
        (((1, 4), (1, 3)), (4, 1), (4, 4)),
    ):
        context = _context(groups, phase=BatchPhase.PREFILL, sizes=sizes)
        count = retained[groups.rank]
        hidden = torch.arange(context.num_tokens * 2, dtype=torch.bfloat16).reshape(
            -1, 2
        )
        with runner._distributed_prefill_graphs(
            context, region_tokens={range(2, 4): count}
        ):
            assert runner._distributed_prefill_bucket == capacities[0]
            assert runner._distributed_prefill_region_buckets == {
                2: capacities[1],
                3: capacities[1],
            }
            if context.placeholder:
                runner._empty_prefill_regions()
            else:
                expected = hidden
                output = hidden
                for region in range(4):
                    if region == 2:
                        expected, output = expected[-count:], output[-count:]
                    output = runner._run_prefill_region(region, output)
                    expected = ((expected.float() + 1) * 1.75).bfloat16()
                torch.testing.assert_close(output, expected)
        assert replays[-4:] == [
            ((capacity, region + 1), context.num_tokens if region < 2 else count)
            for region, capacity in enumerate(
                (capacities[0],) * 2 + (capacities[1],) * 2
            )
        ]
        assert received[-4:] == [
            (2 * capacity, sum(n for _, n in sizes) if region < 2 else sum(retained))
            for region, capacity in enumerate(
                (capacities[0],) * 2 + (capacities[1],) * 2
            )
        ]
        assert not runner._prefill_graph_active
        assert not runner._distributed_prefill_scope
        assert runner._distributed_prefill_bucket is None
        assert runner._distributed_prefill_region_buckets == {}

    # Layout disagreements fail before layers, rather than sending an idle
    # peer into a different number of graph-selection or expert collectives.
    context = _context(groups, phase=BatchPhase.PREFILL, sizes=((1, 4), (1, 4)))
    before = len(replays)
    for regions in (
        {range(2, 4): 1} if groups.rank else {},
        {range(1 if groups.rank else 2, 4): 1},
    ):
        with pytest.raises(RuntimeError, match="different orders"):
            with runner._distributed_prefill_graphs(context, region_tokens=regions):
                raise AssertionError("a mismatched region schedule entered forward")
    with pytest.raises(ValueError, match="nonnegative integers"):
        with runner._distributed_prefill_graphs(
            context, region_tokens={range(2, 4): -1 if groups.rank else 1}
        ):
            raise AssertionError("an invalid token demand entered forward")
    assert len(replays) == before
    with runner._distributed_prefill_graphs(context, region_tokens={range(2, 4): 1}):
        assert runner._distributed_prefill_region_buckets == {2: 1, 3: 1}


def _selection_worker(rank, rendezvous):
    torch.set_num_threads(1)
    interfaces = {name for _, name in socket.if_nameindex()}
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if "lo0" in interfaces else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    dist.barrier()
    groups = None
    try:
        world = RankGroup((0, 1), dist.group.WORLD, rank)
        context = TensorParallelContext.from_group(world)
        assert (context.rank, context.size) == (rank, 2)
        torch.testing.assert_close(
            context.sum(torch.tensor([rank + 1])), torch.tensor([3])
        )
        with torch.inference_mode():
            assert replicated_cache_capacity(100 - rank * 7, world) == 93
        assert not TensorParallelContext.from_group(None).enabled
        assert not TensorParallelContext.from_group(RankGroup((rank,), None, 0)).enabled
        assert replicated_cache_capacity(17, None) == 17
        groups = ParallelGroups.create(
            Parallelism(world_size=2, ep=2, attn_dp=2), device_backend="gloo"
        )

        class CountTransport:
            fixed = 0
            objects = 0

            def gather(self, *args, **kwargs):
                self.fixed += 1
                return dist.gather(*args, **kwargs)

            def broadcast(self, *args, **kwargs):
                self.fixed += 1
                return dist.broadcast(*args, **kwargs)

            def all_gather_object(self, *args, **kwargs):
                self.objects += 1
                return dist.all_gather_object(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(dist, name)

        transport = CountTransport()
        groups.distributed = transport
        runner = _CpuRunner(groups)
        context = _context(groups)

        # A rank with a skinny local batch must launch the same graph as a
        # peer needing a larger batch and wider metadata buffers.
        required = (1, 2) if rank == 0 else (3, 6)
        shape = runner._select_distributed_cuda_graph(
            "decode", context, required_shape=required
        )
        assert shape == (4, 8)
        assert (transport.fixed, transport.objects) == (2, 1)
        for _ in range(3):
            before = transport.fixed, transport.objects
            assert runner._select_distributed_cuda_graph(
                "decode", context, required_shape=required
            ) == (4, 8)
            assert (transport.fixed, transport.objects) == (before[0] + 2, before[1])

        # An identical live catalog update is revalidated once on every rank.
        # A later one-rank mutation must invalidate warm trust on all peers.
        mutable = _CpuRunner(groups)
        assert mutable._select_distributed_cuda_graph(
            "decode", context, required_shape=(1, 1)
        ) == (1, 4)
        cache = mutable._require_cuda_graph_executor().mode("decode")
        cache.shapes = ((1, 4), (4, 8))
        before = transport.fixed, transport.objects
        assert mutable._select_distributed_cuda_graph(
            "decode", context, required_shape=(1, 1)
        ) == (1, 4)
        assert (transport.fixed, transport.objects) == (before[0] + 2, before[1] + 1)
        if rank == 0:
            cache.shapes = ((1, 4),)
        with pytest.raises(ValueError, match="catalogs must agree"):
            mutable._select_distributed_cuda_graph(
                "decode", context, required_shape=(1, 1)
            )
        assert mutable._parallel_graph_catalogs["decode"] == ((1, 4), (4, 8))

        with pytest.raises(ValueError, match="dimensions must agree"):
            runner._select_distributed_cuda_graph(
                "decode", context, required_shape=(1,) if rank else (1, 1)
            )
        modes = _CpuRunner(groups)
        modes._configure_prefill_graphs(
            (lambda hidden, ids: hidden,), hidden_size=2, max_tokens=4
        )
        with pytest.raises(RuntimeError, match="different orders"):
            modes._select_distributed_cuda_graph(
                "decode" if rank else "prefill", context, required_shape=(1, 1)
            )

        # Model-level graph eligibility is also collective. Mixing a replay
        # and eager variable-size exchange would deadlock the NCCL domain.
        shape = runner._select_distributed_cuda_graph(
            "decode",
            context,
            required_shape=required,
            eligible=rank != 0,
            fallback="eligibility",
        )
        assert shape is None
        assert runner.cuda_graph_stats["decode"]["eager_fallback_count"] == 1

        required = (5, 1) if rank == 1 else (1, 1)
        assert (
            runner._select_distributed_cuda_graph(
                "decode", context, required_shape=required
            )
            is None
        )
        assert runner.cuda_graph_stats["decode"]["eager_fallback_count"] == 2

        # Idle attention replicas still vote and select their active peer's
        # physical bucket, then can resume with a smaller logical demand.
        context = _context(groups, sizes=((0, 0), (2, 3)))
        required = (0, 0) if rank == 0 else (2, 3)
        assert runner._select_distributed_cuda_graph(
            "decode", context, required_shape=required
        ) == (
            2,
            4,
        )
        assert runner._select_distributed_cuda_graph(
            "decode", context, required_shape=(1, 1)
        ) == (
            1,
            4,
        )

        mismatched = _CpuRunner(
            groups, shapes=((1, 4), (4, 8)) if rank else ((1, 4), (2, 4), (4, 8))
        )
        with pytest.raises((ValueError, RuntimeError), match="agree|catalog|same"):
            mismatched._select_distributed_cuda_graph(
                "decode", context, required_shape=(1, 1)
            )

        # Startup must reject incompatible capture catalogs before any rank
        # enters the executor's first CUDA/NCCL warmup or capture operation.
        launched = []
        mismatched._cuda_graph_executor.capture_all = lambda **_: launched.append(True)
        with pytest.raises((ValueError, RuntimeError), match="agree|catalog|same"):
            mismatched.capture_cuda_graphs()
        assert not launched

        policy = _CpuRunner(groups)
        if rank:
            policy.cuda_graph_decode = "breakable"
        policy._cuda_graph_executor.capture_all = lambda **_: launched.append(True)
        with pytest.raises((ValueError, RuntimeError), match="agree|policy|same"):
            policy.capture_cuda_graphs()
        assert not launched

        # Equal physical graph shapes do not imply the same expert schedule:
        # ordinary decode and different draft lengths have different rounds.
        for max_accepted in ((1, 3), (3, 6)):
            speculative = _CpuRunner(groups)
            speculative.runner_config = RunnerConfig(
                cache_layout=CacheLayout(
                    parent_capacity=64, groups=[CacheGroupLayout(2)]
                ),
                max_accepted_per_step=max_accepted[rank],
            )
            speculative._cuda_graph_executor.capture_all = lambda **_: launched.append(
                True
            )
            with pytest.raises(ValueError, match="agree"):
                speculative.capture_cuda_graphs()
            assert not launched

        # An agreed catalog reaches capture on both peers, rather than being
        # rejected simply because the attention replicas have unequal work.
        runner._cuda_graph_executor.capture_all = lambda **_: launched.append(True)
        runner.capture_cuda_graphs()
        assert launched == [True]
        # A live policy switch votes eager without editing captured catalogs;
        # followers keep their original configured policy and still participate.
        if rank == 0:
            runner.cuda_graph_decode = "disabled"
        catalog = runner._require_cuda_graph_executor().mode("decode").shapes
        assert (
            runner._select_distributed_cuda_graph(
                "decode", context, required_shape=(1, 1)
            )
            is None
        )
        assert runner._require_cuda_graph_executor().mode("decode").shapes == catalog
        runner.cuda_graph_decode = "full"
        assert runner._select_distributed_cuda_graph(
            "decode", context, required_shape=(1, 1)
        ) == (1, 4)
        _prefill_rounds(groups)
        _compact_prefill_rounds(groups)
    finally:
        if groups is not None:
            groups.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_graph_selection_and_startup_capture_agree_across_attention_replicas(tmp_path):
    context = multiprocessing.get_context("spawn")
    rendezvous = str(tmp_path / "graph-domain")
    processes = [
        context.Process(target=_selection_worker, args=(rank, rendezvous))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_driver_policy_disables_every_peer_without_hiding_catalogs(
    monkeypatch, rank, mode
):
    runner = _CpuRunner(None)
    runner._configure_prefill_graphs(
        (lambda hidden, ids: hidden,), hidden_size=2, max_tokens=4
    )
    setattr(
        runner,
        f"cuda_graph_{mode}",
        "disabled" if rank == 0 else "full" if mode == "decode" else "breakable",
    )
    cache = runner._require_cuda_graph_executor().mode(mode)
    catalog = cache.shapes

    def gather(group, operation, value, *, error=None):
        assert error is None
        assert bool(value[0]) == (rank == 1)
        return [(0, *value[1:]), (1, *value[1:])], [None, None]

    runner._parallel_graph_catalogs[mode] = catalog
    runner._parallel_graph_groups = SimpleNamespace(
        domain=RankGroup((0, 1), None, rank)
    )
    monkeypatch.setattr(
        runner, "_parallel_graph_control", SimpleNamespace(gather=gather)
    )
    context = SimpleNamespace(max_num_tokens=2)
    assert (
        runner._select_distributed_cuda_graph(mode, context, required_shape=catalog[0])
        is None
    )
    assert cache.shapes == catalog
    assert cache.stats["fallback_counts"] == {"distributed_eligibility": 1}
    if mode == "prefill":
        with runner._distributed_prefill_graphs(context):
            assert runner._distributed_prefill_bucket is None
            assert not runner._prefill_graph_active
            with runner._prefill_graphs():
                assert not runner._prefill_graph_active
    if rank == 0:

        def cannot_select(_shape):
            raise AssertionError(
                "disabled local policy must not consider a captured shape"
            )

        assert runner._select_cuda_graph(mode, match=cannot_select) is None


def test_wide_graph_demands_keep_every_dimension():
    runner = _CpuRunner(None, shapes=((1, 2, 3, 4), (2, 3, 4, 5)))
    for _ in range(2):
        assert runner._select_distributed_cuda_graph(
            "decode", None, required_shape=(2, 2, 2, 2)
        ) == (2, 3, 4, 5)


@pytest.mark.parametrize(
    "regions",
    [
        {range(2, 2): 1},
        {range(0, 4, 2): 1},
        {range(0, 5): 1},
        {range(0, 3): 1, range(2, 4): 1},
        {range(2, 4): -1},
        {range(2, 4): True},
    ],
)
def test_prefill_region_demands_are_validated_before_forward(regions):
    runner = _CpuRunner(None)
    runner._configure_prefill_graphs(
        (lambda hidden, ids: hidden,) * 4, hidden_size=2, max_tokens=4
    )
    with pytest.raises(ValueError, match="prefill graph region"):
        with runner._distributed_prefill_graphs(
            SimpleNamespace(max_num_tokens=4), region_tokens=regions
        ):
            raise AssertionError("invalid region demands entered forward")
    assert not runner._prefill_graph_active
    assert not getattr(runner, "_distributed_prefill_scope", False)


def test_nested_prefill_region_buckets_restore_after_failure():
    runner = _CpuRunner(None)
    runner._configure_prefill_graphs(
        (lambda hidden, ids: hidden,) * 4, hidden_size=2, max_tokens=4
    )
    with runner._distributed_prefill_graphs(
        SimpleNamespace(max_num_tokens=4), region_tokens={range(2, 4): 1}
    ):
        with pytest.raises(RuntimeError, match="operation failed"):
            with runner._distributed_prefill_graphs(
                SimpleNamespace(max_num_tokens=2), region_tokens={range(1, 3): 2}
            ):
                assert runner._distributed_prefill_bucket == 2
                assert runner._distributed_prefill_region_buckets == {1: 2, 2: 2}
                raise RuntimeError("operation failed")
        assert runner._prefill_graph_active
        assert runner._distributed_prefill_scope
        assert runner._distributed_prefill_bucket == 4
        assert runner._distributed_prefill_region_buckets == {2: 1, 3: 1}
    assert not runner._prefill_graph_active
    assert not runner._distributed_prefill_scope
    assert runner._distributed_prefill_bucket is None
    assert runner._distributed_prefill_region_buckets == {}


def test_prefill_region_fallback_keeps_the_other_agreed_bucket(monkeypatch):
    runner = _CpuRunner(None)
    runner._configure_prefill_graphs(
        (lambda hidden, ids: hidden + 1,) * 2, hidden_size=2, max_tokens=4
    )
    replays = []

    def replay(mode, shape, inputs):
        replays.append(shape)
        step = runner._prefill_tensor_graphs["prefill"].build(shape)
        assert step.inputs is inputs
        step.launch()
        return step.output

    monkeypatch.setattr(runner, "_run_cuda_graph", replay)
    for full_count, compact_count in ((5, 1), (4, 5)):
        hidden = torch.zeros((full_count, 2), dtype=torch.bfloat16)
        with runner._distributed_prefill_graphs(
            SimpleNamespace(max_num_tokens=full_count),
            region_tokens={range(1, 2): compact_count},
        ):
            assert runner._prefill_graph_active
            with runner._prefill_graphs():
                output = runner._run_prefill_region(0, hidden)
                torch.testing.assert_close(output, hidden + 1)
                compact = torch.ones((compact_count, 2), dtype=torch.bfloat16)
                output = runner._run_prefill_region(1, compact)
                torch.testing.assert_close(output, compact + 1)
    assert replays == [(1, 2), (4, 1)]


def test_prefill_region_cannot_silently_exceed_its_agreed_capacity():
    runner = _CpuRunner(None)
    invoked = []

    def operation(hidden, _ids):
        invoked.append(True)
        return hidden

    runner._configure_prefill_graphs((operation,) * 2, hidden_size=2, max_tokens=4)
    with runner._distributed_prefill_graphs(
        SimpleNamespace(max_num_tokens=4), region_tokens={range(1, 2): 1}
    ):
        with pytest.raises(ValueError, match="exceeds its agreed"):
            runner._run_prefill_region(1, torch.ones(2, 2, dtype=torch.bfloat16))
    assert invoked == []
