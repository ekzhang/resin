from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import NoReturn, TypedDict

import numpy as np
import pytest

from resin.backend.cuda import graphs as graph_runtime
from resin.backend.cuda import runner as runner_runtime
from resin.backend.cuda import staging
from resin.backend.cuda.failures import GraphInputError, PostAcceptCudaError
from resin.backend.cuda.graphs import (
    DEFAULT_DECODE_GRAPH_MAX_BATCH_SIZE,
    DEFAULT_MAX_NUM_SEQS,
    CudaGraphExecutor,
    CudaGraphMode,
    CudaGraphProgram,
    CudaGraphRegion,
    CudaGraphStep,
    _CudaGraphCache,
    _graph_shapes,
    default_decode_graph_shapes,
    resolve_decode_graph_shapes,
)
from resin.backend.cuda.runner import CudaRunner, _resolve_kv_cache_dtype
from resin.backend.expert_parallel import ExpertParallelDispatch
from resin.backend.resources import CachePoolBinding
from resin.backend.topology import RankGroup


# Empty shape catalogs must never invoke their builder.
def _unreachable(_shape) -> NoReturn:
    raise AssertionError("empty graph catalog must not build")


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Tensor:
    def __init__(self, values, *, dtype="int"):
        self.values = np.array(values, copy=True)
        self.device = SimpleNamespace(type="cuda")
        self.dtype = dtype
        self.shape = self.values.shape
        self.copy_error = None

    def clone(self):
        return _Tensor(self.values, dtype=self.dtype)

    def float(self):
        return _Tensor(self.values.astype(np.float32), dtype="float")

    def __sub__(self, other):
        return _Tensor(self.values - other.values, dtype=self.dtype)

    def __ne__(self, other):
        return _Tensor(self.values != other.values)

    def abs(self):
        return _Tensor(np.abs(self.values), dtype=self.dtype)

    def nan_to_num(self, **options):
        return _Tensor(np.nan_to_num(self.values, **options), dtype=self.dtype)

    def max(self):
        return _Tensor(self.values.max(), dtype=self.dtype)

    def sum(self):
        return _Tensor(self.values.sum(), dtype=self.dtype)

    def item(self):
        return self.values.item()

    def numel(self):
        return self.values.size

    def reshape(self, *shape):
        return _Tensor(self.values.reshape(*shape), dtype=self.dtype)

    def __getitem__(self, key):
        return _Tensor(self.values[key], dtype=self.dtype)

    def view(self, shape):
        size = 1
        for dimension in shape:
            size *= dimension
        if size != len(self.values):
            raise ValueError("shape mismatch")
        self.shape = shape
        return self

    def copy_(self, source, *, non_blocking=False):
        del non_blocking
        if self.copy_error is not None:
            raise self.copy_error
        self.values[...] = source.values

    def numpy(self):
        return self.values.reshape(self.shape)

    def tolist(self):
        return self.values.tolist()


class _Stream:
    def wait_stream(self, _other):
        pass


class _Event:
    def record(self, _stream):
        pass

    def query(self):
        return True


class _Graph:
    def __init__(self):
        self.error = None
        self.replay_count = 0
        self.reset_count = 0

    def replay(self):
        if self.error is not None:
            raise self.error
        self.replay_count += 1

    def reset(self):
        self.reset_count += 1


class _Cuda:
    Event = _Event

    def __init__(self):
        self.graphs = []
        self.pools = []
        self._current_stream = _Stream()
        self.capturing = False

    def memory_allocated(self, _device):
        return 100

    def current_stream(self, _device):
        return self._current_stream

    def Stream(self, *, device):
        del device
        return _Stream()

    @contextmanager
    def stream(self, stream):
        previous, self._current_stream = self._current_stream, stream
        try:
            yield
        finally:
            self._current_stream = previous

    def synchronize(self, _device):
        pass

    def device(self, _device):
        return _Context()

    def CUDAGraph(self):
        graph = _Graph()
        self.graphs.append(graph)
        return graph

    def graph_pool_handle(self):
        pool = object()
        self.pools.append(pool)
        return pool

    @contextmanager
    def graph(self, graph, *, pool=None, stream=None):
        graph.pool = pool
        graph.stream = stream
        with self.stream(stream if stream is not None else _Stream()):
            previous, self.capturing = self.capturing, True
            try:
                yield
            finally:
                self.capturing = previous


class _Torch:
    Tensor = _Tensor

    def __init__(self):
        self.cuda = _Cuda()
        self.host_allocations = 0

    def inference_mode(self, enabled=True):
        return _Context()

    def tensor(self, values, *, dtype, device, pin_memory=False):
        assert device == "cpu"
        assert pin_memory
        return _Tensor(values, dtype=dtype)

    def empty(self, shape, *, dtype, device, pin_memory=False):
        assert device == "cpu"
        assert pin_memory
        self.host_allocations += 1
        return _Tensor(np.empty(shape), dtype=dtype)

    @staticmethod
    def equal(left, right):
        return np.array_equal(left.values, right.values)


@pytest.fixture
def graph_torch(monkeypatch):
    torch = _Torch()
    monkeypatch.setattr(staging, "torch", torch)
    monkeypatch.setattr(graph_runtime, "torch", torch)
    return torch


def _cache(shapes, build, synchronize_failure=lambda: None):
    return _CudaGraphCache(
        "cuda:0",
        _graph_shapes(shapes, dimensions=2),
        build=build,
        synchronize_failure=synchronize_failure,
    )


def test_mismatch_diagnostics_measure_full_tensor_after_identical_prefix():
    import torch

    expected = torch.zeros(1024, 32, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[-1, -1] = 0.0078125
    assert graph_runtime._output_summary(expected) == graph_runtime._output_summary(
        actual
    )
    assert graph_runtime._output_difference(expected, actual) == (
        "mismatched=1/32768, max_abs=0.0078125"
    )


class _ConcreteCudaRunner(CudaRunner):
    def forward(self, plan):
        raise AssertionError("this fixture only tests graph ownership")


class _DecodeOptions(TypedDict):
    policy: str
    width: int
    speculative: bool
    max_batch_size: int


class TestCudaGraph:
    @pytest.mark.parametrize("kind", ["dispatch", "shared", "module"])
    @pytest.mark.parametrize("allocated", [False, True])
    def test_expert_workspace_cleanup_is_collective_once_and_only_if_allocated(
        self, monkeypatch, allocated, kind
    ):
        from torch import nn

        events = []
        group_closed = False

        class Workspace:
            def __del__(self):
                events.append("unmap")

        def barrier(*, group):
            assert not group_closed, "cleanup reused a destroyed process group"
            events.append("barrier")

        from resin.backend.collectives import LayerCollective

        class Manager(LayerCollective):
            def __init__(self):
                super().__init__()
                self.workspace = Workspace() if allocated else None

            @property
            def allocated(self):
                return self.workspace is not None

            def close(self):
                self.workspace = None

        class ModuleManager(nn.Module, Manager):
            def __init__(self):
                nn.Module.__init__(self)
                Manager.__init__(self)

        model = nn.ModuleList([nn.Module(), nn.Module()])
        if kind == "dispatch":
            collectives = [
                ExpertParallelDispatch(
                    lambda hidden, indices, weights: hidden,
                    num_experts=2,
                    group=RankGroup((0, 1), object(), 0),
                )
                for _ in model
            ]
            if allocated:
                # A later layer owns storage even though the first layer is empty.
                collectives[-1]._transports["test"] = Workspace()
        else:
            manager = ModuleManager() if kind == "module" else Manager()
            collectives = [manager, manager]
        for module, collective in zip(model, collectives, strict=True):
            setattr(module, "collective", collective)
        monkeypatch.setattr(
            runner_runtime.torch.cuda,
            "synchronize",
            lambda device: events.append("sync"),
        )
        runner = CudaRunner.__new__(CudaRunner)
        runner.device = runner_runtime.torch.device("cuda:0")
        runner._cuda_graph_executor = None
        runner._prefill_tensor_graphs = {}
        runner._parallel_graph_model = model
        runner._parallel_graph_groups = SimpleNamespace(
            distributed=SimpleNamespace(barrier=barrier),
            domain=SimpleNamespace(handle=object()),
        )

        runner.close_cuda_graphs()
        group_closed = True
        runner.close_cuda_graphs()

        assert events == (["sync", "barrier", "unmap"] if allocated else [])
        assert all(not collective.allocated for collective in collectives)

    def test_cuda_runner_owns_device_precision_sampler_and_cache_binding(
        self, monkeypatch
    ) -> None:
        sampler = object()
        monkeypatch.setattr(runner_runtime.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(runner_runtime.torch.cuda, "current_device", lambda: 3)
        monkeypatch.setattr(runner_runtime, "FusedSampler", lambda: sampler)

        runner = _ConcreteCudaRunner(device="cuda", label="test", kv_cache_dtype="fp8")

        assert str(runner.device) == "cuda:3"
        assert runner.kv_cache_dtype == "fp8"
        assert runner.kv_cache_torch_dtype is runner_runtime.torch.float8_e4m3fn
        assert runner._sampler is sampler
        layout = object()
        binding = CachePoolBinding(layout, resource="test KV")
        runner._configure_cache_binding(binding)
        # Minimal native binding double isolates cache ownership.
        runner.bind_resources(SimpleNamespace(cache_pool_id=7, cache_layout=layout))  # ty: ignore[invalid-argument-type]
        assert binding.pool_id == 7
        with pytest.raises(RuntimeError, match="already configured"):
            runner._configure_cache_binding(CachePoolBinding(layout))

    def test_model_runner_owns_modes_capture_hooks_selection_and_stats(
        self, graph_torch
    ) -> None:
        class ModelRunner(_ConcreteCudaRunner):
            device = "cuda:0"

            def __init__(self) -> None:
                self.events = []
                self.cuda_graph_decode = "full"
                self._prefill_operations = ()
                self._cuda_failure = lambda: None
                self._cuda_graph_executor = None

            def _before_cuda_graph_capture(self) -> None:
                self.events.append("before")

            def _after_cuda_graph_capture(self) -> None:
                self.events.append("after")

        def mode(value):
            def build(_shape):
                state = _Tensor([0])
                return CudaGraphStep({"value": state}, state, lambda: None)

            return CudaGraphMode(((1, 4),), build, lambda _shape: {"value": [value]})

        runner = ModelRunner()
        runner._configure_cuda_graphs({"prefill": mode(1), "decode": mode(2)})

        stats = runner.capture_cuda_graphs()

        assert not hasattr(CudaRunner, "primary_cuda_graph_mode")
        assert runner.events == ["before", "after"]
        assert stats == runner.cuda_graph_stats
        assert set(stats) == {"prefill", "decode"}
        assert stats["prefill"]["capture_count"] == 1
        shape = runner._select_cuda_graph(
            "decode", match=lambda candidate: candidate[0] == 1
        )
        assert shape == (1, 4)
        assert runner._run_cuda_graph("decode", shape, {"value": [3]}).tolist() == [3]
        runner._check_cuda_graphs()

    def test_model_runner_requires_one_nonempty_mode_configuration(self) -> None:
        class ModelRunner(_ConcreteCudaRunner):
            device = "cuda:0"

        runner = ModelRunner.__new__(ModelRunner)
        runner._cuda_graph_executor = None
        runner.cuda_graph_decode = "full"
        with pytest.raises(ValueError, match="must not be empty"):
            runner._configure_cuda_graphs({})
        with pytest.raises(RuntimeError, match="not been configured"):
            runner._require_cuda_graph_executor()

        mode = CudaGraphMode((), _unreachable, lambda _shape: {})
        setattr(runner, "_cuda_failure", lambda: None)
        runner._configure_cuda_graphs({"decode": mode})
        with pytest.raises(RuntimeError, match="already configured"):
            runner._configure_cuda_graphs({"decode": mode})

    def test_kv_cache_dtype_is_one_shared_runner_option(self) -> None:
        import torch

        assert _resolve_kv_cache_dtype("fp8") is torch.float8_e4m3fn
        assert _resolve_kv_cache_dtype("bfloat16") is torch.bfloat16
        with pytest.raises(ValueError, match="kv_cache_dtype"):
            _resolve_kv_cache_dtype("float16")

    def test_runner_reuses_one_mechanism_across_execution_modes(
        self, graph_torch
    ) -> None:
        events = []

        def build_prefill(_shape):
            state = _Tensor([0])
            return CudaGraphProgram(
                inputs={"value": state},
                output=state,
                regions=(
                    CudaGraphRegion("attention", lambda: None),
                    CudaGraphRegion("collective", lambda: None, capture=False),
                    CudaGraphRegion("moe", lambda: None),
                ),
            )

        def build_decode(_shape):
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = CudaGraphExecutor(
            "cuda:0",
            {
                "prefill": CudaGraphMode(
                    ((4, 128),), build_prefill, lambda _shape: {"value": [1]}
                ),
                "decode": CudaGraphMode(
                    ((4, 4096),), build_decode, lambda _shape: {"value": [2]}
                ),
            },
            synchronize_failure=lambda: None,
        )

        stats = graphs.capture_all(
            before_capture=lambda: events.append("before"),
            after_capture=lambda: events.append("after"),
        )

        assert events == ["before", "after"]
        assert stats["prefill"]["graph_region_count"] == 2
        assert stats["prefill"]["graph_break_count"] == 1
        assert stats["decode"]["graph_region_count"] == 1
        assert graphs.mode("prefill").run((4, 128), {"value": [3]}).tolist() == [3]
        assert graphs.mode("decode").run((4, 4096), {"value": [4]}).tolist() == [4]

    def test_shared_decode_defaults_use_one_history_capacity(self) -> None:
        assert DEFAULT_MAX_NUM_SEQS == 128
        assert DEFAULT_DECODE_GRAPH_MAX_BATCH_SIZE == 1024
        assert default_decode_graph_shapes(4096, speculative=True) == tuple(
            (batch_size, 4096)
            for batch_size in (
                *range(1, 9),
                *range(12, 33, 4),
                *range(40, 65, 8),
                *range(96, 257, 32),
                *range(384, 1025, 128),
            )
        )
        assert default_decode_graph_shapes(4096, speculative=False) == tuple(
            (batch_size, 4096)
            for batch_size in (
                1,
                2,
                4,
                8,
                *range(12, 17, 4),
                *range(24, 33, 8),
                *range(64, 129, 32),
                *range(256, 1025, 128),
            )
        )

    def test_shared_decode_defaults_respect_lower_sequence_limit(self) -> None:
        assert default_decode_graph_shapes(
            128, speculative=True, max_batch_size=16
        ) == tuple((batch_size, 128) for batch_size in (*range(1, 9), 12, 16))
        for name, value in (("width", 0), ("max_batch_size", True)):
            arguments = {"width": 8, "max_batch_size": 128, name: value}
            with pytest.raises(ValueError, match=name):
                default_decode_graph_shapes(speculative=True, **arguments)

    @pytest.mark.parametrize("policy", ["disabled", "full", "breakable"])
    def test_decode_policy_resolves_defaults_and_explicit_overrides(self, policy):
        arguments: _DecodeOptions = dict(
            policy=policy, width=128, speculative=True, max_batch_size=9
        )
        expected = (
            ()
            if policy == "disabled"
            else default_decode_graph_shapes(128, speculative=True, max_batch_size=9)
        )
        assert resolve_decode_graph_shapes(None, **arguments) == expected
        assert resolve_decode_graph_shapes((), **arguments) == ()
        if policy == "disabled":
            with pytest.raises(ValueError, match="nonempty"):
                resolve_decode_graph_shapes(((1, 128),), **arguments)
        else:
            assert resolve_decode_graph_shapes({2: 64, 1: 128}, **arguments) == (
                (1, 128),
                (2, 64),
            )
        with pytest.raises(ValueError, match="duplicate"):
            resolve_decode_graph_shapes(((1, 128), (1, 128)), **arguments)

    def test_decode_policy_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="policy"):
            resolve_decode_graph_shapes(
                None, policy="automatic", width=128, speculative=False, max_batch_size=8
            )

    def test_shapes_are_frozen_sorted_and_validated(self) -> None:
        assert _graph_shapes(None, dimensions=2) == ()
        assert _graph_shapes({2: 8, 1: 4}, dimensions=2) == ((1, 4), (2, 8))
        assert _graph_shapes(((2, 8), (1, 8)), dimensions=2) == ((1, 8), (2, 8))
        for invalid in ("1x4", ((0, 4),), ((True, 4),), ((1,),)):
            with (
                pytest.raises((TypeError, ValueError)),
            ):
                _graph_shapes(invalid, dimensions=2)
        with pytest.raises(ValueError, match="duplicate"):
            _graph_shapes(((1, 4), (1, 4)), dimensions=2)

    def test_matches_the_first_compatible_physical_shape(self) -> None:
        graphs = _cache(
            ((2, 8), (1, 4), (2, 4)),
            lambda _shape: CudaGraphStep({}, _Tensor([0]), lambda: None),
        )
        assert graphs.match(lambda shape: shape[0] == 2 and shape[1] >= 3) == (2, 4)
        assert graphs.match(lambda shape: shape[0] == 2 and shape[1] >= 5) == (2, 8)
        assert graphs.match(lambda shape: shape[0] == 1 and shape[1] >= 5) is None

    def test_explicit_capture_precedes_replay_of_the_same_static_step(
        self, graph_torch
    ) -> None:
        staged = [7]
        allocations = []

        def build(_shape):
            state = _Tensor([0])
            allocations.append(state)
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(((1, 4),), build)

        assert graphs.capture((1, 4), {"value": staged}).tolist() == [7]
        staged[:] = [8]
        assert graphs.run((1, 4), {"value": staged}).tolist() == [8]
        assert len(allocations) == 1
        assert graphs.stats["capture_count"] == 1
        assert graphs.stats["replay_count"] == 1
        assert graphs.stats["captured_shapes"] == ((1, 4),)

    @pytest.mark.parametrize("builder", [CudaGraphStep, CudaGraphProgram])
    def test_device_output_preserves_identity_without_host_conversion(
        self, monkeypatch, graph_torch, builder
    ):
        torch = graph_torch
        monkeypatch.setattr(
            _Tensor,
            "tolist",
            # ty misbinds pytest's decorated helper and drops its reason parameter.
            lambda _: pytest.fail("unexpected host conversion"),  # ty: ignore[invalid-argument-type]
        )
        output = _Tensor([0])
        operation = (
            (CudaGraphRegion("compute", lambda: None),)
            if builder is CudaGraphProgram
            else lambda: None
        )
        graphs = _cache(
            ((1, 4),),
            lambda _: builder({"value": output}, output, operation),
        )
        assert graphs.capture((1, 4), {"value": _Tensor([7])}) is output
        assert graphs.run((1, 4), {"value": _Tensor([8])}) is output
        assert output.values.tolist() == [8]
        assert torch.host_allocations == 0

    def test_capture_all_owns_shape_iteration_and_model_state_hooks(
        self, graph_torch
    ) -> None:
        events = []

        def build(shape):
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(((1, 4), (2, 4)), build)

        stats = graphs.capture_all(
            lambda shape: events.append(("inputs", shape)) or {"value": [1]},
            before_capture=lambda: events.append("before"),
            after_capture=lambda: events.append("after"),
        )
        assert stats["captured_shapes"] == ((1, 4), (2, 4))
        assert events == [
            "before",
            ("inputs", (2, 4)),
            ("inputs", (1, 4)),
            "after",
        ]

        events.clear()
        graphs.capture_all(
            lambda _shape: pytest.fail("captured shapes must not rebuild inputs"),  # ty: ignore[invalid-argument-type]
            before_capture=lambda: events.append("before"),
            after_capture=lambda: events.append("after"),
        )
        assert events == ["before", "after"]

    def test_all_modes_share_one_lazy_pool_and_stream_and_capture_largest_first(
        self, graph_torch
    ):
        torch = graph_torch
        order = []

        def mode(name):
            def build(shape):
                order.append((name, shape))
                state = _Tensor([0])
                return CudaGraphProgram(
                    {"value": state},
                    state,
                    (
                        CudaGraphRegion("first", lambda: None),
                        CudaGraphRegion("last", lambda: None),
                    ),
                )

            return CudaGraphMode(((1, 4), (4, 4)), build, lambda _: {"value": [1]})

        graphs = CudaGraphExecutor(
            "cuda:0",
            {
                "prefill": mode("prefill"),
                "decode": mode("decode"),
                "other": mode("other"),
            },
            synchronize_failure=lambda: None,
        )
        assert torch.cuda.pools == []
        graphs.capture_all()
        assert order == [
            (name, shape)
            for name in ("prefill", "decode", "other")
            for shape in ((4, 4), (1, 4))
        ]
        (pool,) = torch.cuda.pools
        assert [graph.pool for graph in torch.cuda.graphs] == [pool] * 12
        assert len({graph.stream for graph in torch.cuda.graphs}) == 1
        graphs.capture_all()
        assert len(torch.cuda.pools) == 1
        assert graphs.mode("prefill").run((1, 4), {"value": [2]}).tolist() == [2]

        # Modes registered after startup join the same workspace.
        graphs.add_mode("later", mode("later"))
        graphs.capture_all()
        assert len(torch.cuda.pools) == 1
        assert all(graph.pool is pool for graph in torch.cuda.graphs)
        assert len({graph.stream for graph in torch.cuda.graphs}) == 1

    def test_pool_lifetime_is_per_executor_and_closing_one_mode_keeps_siblings(
        self, graph_torch
    ):
        torch = graph_torch

        def build(_shape):
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        mode = CudaGraphMode(((1, 4),), build, lambda _: {"value": [1]})
        first = CudaGraphExecutor(
            "cuda:0",
            {"prefill": mode, "decode": mode},
            synchronize_failure=lambda: None,
        )
        second = CudaGraphExecutor(
            "cuda:0",
            {"decode": mode},
            synchronize_failure=lambda: None,
        )
        standalone = _cache(((1, 4),), build)
        first.capture_all()
        second.capture_all()
        standalone.capture_all(mode.capture_inputs)
        assert len(set(torch.cuda.pools)) == 3
        before = first.mode("decode")._workspace
        first.mode("prefill").close()
        assert first.mode("prefill")._workspace is None
        assert first.mode("decode")._workspace is before
        assert first.mode("decode").run((1, 4), {"value": [7]}).tolist() == [7]
        first.close()
        assert first._workspace is None
        assert second.mode("decode").run((1, 4), {"value": [9]}).tolist() == [9]
        assert standalone.run((1, 4), {"value": [11]}).tolist() == [11]
        second.close()
        standalone.close()
        assert all(graph.reset_count == 1 for graph in torch.cuda.graphs)

    def test_capture_warms_stream_local_kernel_state_and_preserves_breaks_and_pool(
        self, graph_torch
    ):
        torch = graph_torch
        caller = torch.cuda.current_stream("cuda:0")
        initialized, events = set(), []

        def launch(name):
            stream = torch.cuda.current_stream("cuda:0")
            if stream not in initialized:
                assert not torch.cuda.capturing, (
                    "first kernel use cannot allocate in capture"
                )
                initialized.add(stream)
            events.append((name, stream))

        cache = _CudaGraphCache(
            "cuda:0",
            ((1, 4), (2, 4)),
            build=lambda _: CudaGraphProgram(
                {},
                _Tensor([0]),
                (
                    CudaGraphRegion("first", lambda: launch("first")),
                    CudaGraphRegion("break", lambda: launch("break"), capture=False),
                    CudaGraphRegion("last", lambda: launch("last")),
                ),
            ),
            synchronize_failure=lambda: None,
        )
        cache.capture_all(lambda _: {})
        assert len(initialized) == 1
        stream = next(iter(initialized))
        assert stream is not caller
        assert all(graph.stream is stream for graph in torch.cuda.graphs)
        assert all(graph.pool is torch.cuda.pools[0] for graph in torch.cuda.graphs)
        # Three warmups, then capture in region order, then one eager break
        # during parity replay. The fake graphs themselves record only replays.
        assert [name for name, _ in events] == (
            ["first", "break", "last"] * 4 + ["break"]
        ) * 2
        assert all(current is stream for _, current in events)
        assert torch.cuda.current_stream("cuda:0") is caller
        cache.close()
        assert cache._workspace is None

    def test_cache_close_resets_graphs_after_synchronizing_and_is_idempotent(
        self, monkeypatch, graph_torch
    ):
        torch = graph_torch

        def build(_shape):
            state = _Tensor([0])
            return CudaGraphProgram(
                {"value": state},
                state,
                (
                    CudaGraphRegion("first", lambda: None),
                    CudaGraphRegion("break", lambda: None, capture=False),
                    CudaGraphRegion("last", lambda: None),
                ),
            )

        cache = _CudaGraphCache(
            "cuda:0",
            ((1, 4), (2, 4)),
            build=build,
            synchronize_failure=lambda: None,
        )
        cache.capture_all(lambda _: {"value": [1]})
        stats = cache.stats
        events = []
        monkeypatch.setattr(
            torch.cuda, "synchronize", lambda device: events.append("sync")
        )
        original_reset = _Graph.reset

        def reset(graph):
            events.append("reset")
            original_reset(graph)

        monkeypatch.setattr(_Graph, "reset", reset)
        cache.close()
        cache.close()

        assert events == ["sync", "reset", "reset", "reset", "reset"]
        assert [graph.reset_count for graph in torch.cuda.graphs] == [1] * 4
        assert cache._workspace is None
        assert all(
            not entry.graphs and entry.program is None and entry.staging is None
            for entry in cache._entries.values()
        )
        assert cache.stats == stats
        for operation in (
            cache.check_ready,
            lambda: cache.match(lambda _: True),
            lambda: cache.fallback("shape"),
            lambda: cache.capture((1, 4), {"value": [1]}),
            lambda: cache.capture_all(lambda _: {"value": [1]}),
            lambda: cache.run((1, 4), {"value": [1]}),
        ):
            with pytest.raises(RuntimeError, match="closed"):
                operation()

    def test_runner_close_synchronizes_once_across_modes_before_group_teardown(
        self, monkeypatch, graph_torch
    ):
        torch = graph_torch

        def mode():
            def build(_shape):
                state = _Tensor([0])
                return CudaGraphStep({"value": state}, state, lambda: None)

            return CudaGraphMode(((1, 4),), build, lambda _: {"value": [1]})

        executor = CudaGraphExecutor(
            "cuda:0",
            {"prefill": mode(), "decode": mode()},
            synchronize_failure=lambda: None,
        )
        executor.capture_all()
        caches = [executor.mode(name) for name in ("prefill", "decode")]
        stats = executor.stats
        synchronized = []
        monkeypatch.setattr(
            torch.cuda, "synchronize", lambda device: synchronized.append(device)
        )
        runner = CudaRunner.__new__(CudaRunner)
        runner._cuda_graph_executor = executor
        runner._prefill_tensor_graphs = {
            "prefill": SimpleNamespace(buffers={"retained": object()})
        }
        runner._parallel_graph_model = None
        runner._parallel_graph_groups = None

        runner.close_cuda_graphs()
        runner.close_cuda_graphs()

        assert synchronized == ["cuda:0"]
        assert [graph.reset_count for graph in torch.cuda.graphs] == [1, 1]
        assert not runner._prefill_tensor_graphs["prefill"].buffers
        assert executor.stats == stats
        for operation in (
            executor.check,
            lambda: executor.mode("decode"),
            lambda: executor.add_mode("other", mode()),
            executor.capture_all,
            *(
                lambda cache=cache: cache.run((1, 4), {"value": [1]})
                for cache in caches
            ),
        ):
            with pytest.raises(RuntimeError, match="closed"):
                operation()

    def test_runtime_counts_track_success_and_survive_close_without_catalog_scan(
        self, graph_torch
    ):
        torch = graph_torch

        def build(_shape):
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        synchronized = []
        executor = CudaGraphExecutor(
            "cuda:0",
            {
                "decode": CudaGraphMode(
                    ((1, 4), (2, 4)), build, lambda _: {"value": [1]}
                )
            },
            synchronize_failure=lambda: synchronized.append(True),
        )
        cache = executor.mode("decode")
        cache.capture((1, 4), {"value": [1]})
        assert cache.runtime_counts["replay_count"] == 0
        cache.run((1, 4), {"value": [2]})
        cache.run((1, 4), {"value": [3]})
        cache.fallback("padding")
        cache.fallback("padding")
        expected = {
            "decode": {
                "replay_count": 2,
                "eager_fallback_count": 2,
                "fallback_counts": {"padding": 2},
            }
        }
        assert executor.runtime_counts == expected
        assert {
            "replay_count": cache.stats["replay_count"],
            "eager_fallback_count": cache.stats["eager_fallback_count"],
            "fallback_counts": cache.stats["fallback_counts"],
        } == cache.runtime_counts
        torch.cuda.graphs[0].error = RuntimeError("illegal replay")
        with pytest.raises(PostAcceptCudaError, match="illegal replay") as caught:
            cache.run((1, 4), {"value": [4]})
        assert caught.value._resin_post_accept is True
        assert synchronized == [True]
        assert "graph replay" in (cache.stats["poison_error"] or "")
        assert executor.runtime_counts == expected
        executor.close()
        assert executor.runtime_counts == expected
        assert cache.stats["shape_stats"][0]["replay_count"] == 2

        class UnreadableCatalog(dict):
            def values(self):
                raise AssertionError("runtime counters scanned graph shapes")

        cache._entries = UnreadableCatalog(cache._entries)
        snapshot = executor.runtime_counts
        snapshot["decode"]["fallback_counts"]["padding"] = -1
        assert executor.runtime_counts == expected

    def test_closing_uncaptured_modes_needs_no_cuda_context(self, monkeypatch):
        torch = _Torch()
        monkeypatch.setattr(graph_runtime, "torch", torch)
        monkeypatch.setattr(
            torch.cuda,
            "synchronize",
            lambda _: pytest.fail("no graph or pending device work exists"),  # ty: ignore[invalid-argument-type]
        )
        cache = _cache(((1, 4),), lambda _: pytest.fail("must not build"))  # ty: ignore[invalid-argument-type]
        cache.close()
        cache.close()
        executor = CudaGraphExecutor(
            "cuda:0",
            {"decode": CudaGraphMode((), _unreachable, lambda _: {})},
            synchronize_failure=lambda: None,
        )
        executor.close()
        executor.close()
        with pytest.raises(RuntimeError, match="closed"):
            executor.capture_all()

    def test_replay_reuses_persistent_pinned_staging(self, graph_torch) -> None:
        torch = graph_torch

        def build(_shape):
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(((1, 4),), build)
        graphs.capture((1, 4), {"value": [1]})
        allocations_after_capture = torch.host_allocations

        graphs.run((1, 4), {"value": [2]})
        graphs.run((1, 4), {"value": [3]})

        assert torch.host_allocations == allocations_after_capture

    def test_program_replays_captured_regions_around_an_eager_break(
        self, graph_torch
    ) -> None:
        torch = graph_torch
        events = []

        def build(_shape):
            state = _Tensor([0])
            return CudaGraphProgram(
                inputs={"value": state},
                output=state,
                regions=(
                    CudaGraphRegion("before", lambda: events.append("before")),
                    CudaGraphRegion(
                        "unsupported",
                        lambda: events.append("break"),
                        capture=False,
                    ),
                    CudaGraphRegion("after", lambda: events.append("after")),
                ),
            )

        graphs = _cache(((1, 4),), build)
        assert graphs.capture((1, 4), {"value": [7]}).tolist() == [7]
        assert len(torch.cuda.graphs) == 2
        events.clear()

        assert graphs.run((1, 4), {"value": [8]}).tolist() == [8]
        assert events == ["break"]
        assert [graph.replay_count for graph in torch.cuda.graphs] == [2, 2]
        assert graphs.stats["graph_region_count"] == 2
        assert graphs.stats["graph_break_count"] == 1

    def test_program_requires_at_least_one_captured_region(self) -> None:
        with pytest.raises(ValueError, match="capture at least one"):
            CudaGraphProgram(
                inputs={},
                output=_Tensor([0]),
                regions=(CudaGraphRegion("eager", lambda: None, capture=False),),
            )

    def test_capture_can_prepare_mutable_state_before_each_parity_run(
        self, graph_torch
    ) -> None:
        prepared = []

        def build(_shape):
            state = _Tensor([0])
            return CudaGraphStep(
                {"value": state},
                state,
                lambda: None,
                prepare=lambda: prepared.append(True),
            )

        graphs = _cache(((1, 4),), build)

        graphs.capture((1, 4), {"value": [7]})

        assert len(prepared) == 5

    def test_graph_inputs_can_copy_directly_from_device_tensors(
        self, graph_torch
    ) -> None:
        def build(_shape):
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(((1, 4),), build)
        source = _Tensor([9])

        assert graphs.capture((1, 4), {"value": source}).tolist() == [9]

    def test_staging_the_same_input_mapping_does_not_copy_or_allocate(self) -> None:
        tensor = _Tensor([3, 7])
        tensor.copy_error = AssertionError("unnecessary self-copy")
        inputs = {"value": tensor}
        stager = staging.TensorStager("cuda:0")
        # CPU tensor double proves no CUDA allocation or copy occurs.
        stager.copy_into(inputs, inputs)  # ty: ignore[invalid-argument-type]
        assert tensor.tolist() == [3, 7]
        assert not stager.host_tensors

    def test_run_never_captures_dynamically(self) -> None:
        graphs = _cache(
            ((1, 4),),
            lambda _shape: CudaGraphStep({}, _Tensor([0]), lambda: None),
        )
        with pytest.raises(RuntimeError, match="has not been captured"):
            graphs.run((1, 4), {})
        assert graphs.stats["capture_count"] == 0

    def test_readiness_requires_successful_capture_of_every_configured_shape(
        self, monkeypatch
    ) -> None:
        torch = _Torch()
        monkeypatch.setattr(graph_runtime, "torch", torch)
        graphs = _cache(
            ((1, 4), (2, 4)),
            lambda _shape: CudaGraphStep({}, _Tensor([0]), lambda: None),
        )
        with pytest.raises(RuntimeError, match="2 configured.*not been captured"):
            graphs.check_ready()
        graphs.capture((1, 4), {})
        with pytest.raises(RuntimeError, match="1 configured.*not been captured"):
            graphs.check_ready()
        assert not graphs.stats["poisoned"]
        graphs.capture((2, 4), {})
        graphs.check_ready()
        graphs.capture_all(lambda _: {})
        graphs.check_ready()

    def test_host_shape_error_is_retryable_and_drops_fresh_step(
        self, graph_torch
    ) -> None:
        builds = []

        def build(_shape):
            builds.append(True)
            state = _Tensor([0])
            return CudaGraphStep({"value": state}, state, lambda: None)

        graphs = _cache(((1, 4),), build)
        with pytest.raises(GraphInputError):
            graphs.capture((1, 4), {})
        assert not graphs.stats["poisoned"]
        assert graphs.capture((1, 4), {"value": [1]}).tolist() == [1]
        assert len(builds) == 2

    def test_capture_mismatch_poisons_before_future_work(self, graph_torch) -> None:
        synchronized = []

        def build(_shape):
            state = _Tensor([0])

            def launch():
                state.values[0] += 1

            return CudaGraphStep({"value": state}, state, launch)

        torch = graph_torch
        graphs = _cache(((1, 4),), build, lambda: synchronized.append(True))
        with pytest.raises(RuntimeError, match="differs"):
            graphs.capture((1, 4), {"value": [0]})
        assert synchronized == [True]
        assert graphs.stats["poisoned"]
        with pytest.raises(RuntimeError, match="poisoned"):
            graphs.check()
        # Even failed capture retains its graph until explicit cleanup, so a
        # captured NCCL user object cannot outlive communicator teardown.
        graphs.close()
        assert [graph.reset_count for graph in torch.cuda.graphs] == [1]

    @pytest.mark.parametrize("validate_eager_output", [False, True])
    def test_large_parity_failures_report_only_bounded_samples(
        self, monkeypatch, validate_eager_output
    ) -> None:
        output = _Tensor(np.zeros((128, 4096)), dtype="bfloat16")
        torch = _Torch()
        monkeypatch.setattr(graph_runtime, "torch", torch)
        original_tolist = _Tensor.tolist
        sampled_sizes = []

        def bounded_tolist(tensor):
            sampled_sizes.append(tensor.numel())
            assert tensor.numel() <= 8, "diagnostics copied an entire hidden tensor"
            return original_tolist(tensor)

        monkeypatch.setattr(_Tensor, "tolist", bounded_tolist)

        def increment():
            output.values += 1

        monkeypatch.setattr(
            torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=increment)
        )
        graphs = _cache(
            ((1, 4),),
            lambda _shape: CudaGraphStep(
                {},
                output,
                increment,
                validate_eager_output=validate_eager_output,
            ),
        )
        with pytest.raises(PostAcceptCudaError) as caught:
            graphs.capture((1, 4), {})
        message = str(caught.value)
        assert "shape=(128, 4096), dtype=bfloat16, elements=524288" in message
        assert len(message) < 700
        assert sampled_sizes == [8, 8]
        assert graphs.stats["poisoned"]

    def test_device_copy_failure_poisons_before_capture(self, graph_torch) -> None:
        state = _Tensor([0])
        state.copy_error = RuntimeError("copy failed")
        graphs = _cache(
            ((1, 4),),
            lambda _shape: CudaGraphStep({"value": state}, state, lambda: None),
        )

        with pytest.raises(RuntimeError, match="copy failed"):
            graphs.capture((1, 4), {"value": [1]})
        assert graphs.stats["capture_count"] == 0
        assert "graph input copy" in graphs.stats["poison_error"]
