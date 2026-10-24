"""Model-neutral host mechanics shared by CUDA runners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from time import perf_counter

import torch

from resin import BatchPhase, BatchPlan, ResourceBinding, RunnerConfig
from resin.backend.collectives import (
    close_layer_collectives,
    fixed_layer_capacity,
    layer_collectives,
)
from resin.backend.control_plane import ControlPlane
from resin.backend.cuda.failures import CudaFailureGuard
from resin.backend.cuda.graphs import (
    CudaGraphCounts,
    CudaGraphExecutor,
    CudaGraphMode,
    CudaGraphProgram,
    CudaGraphStats,
    GraphShape,
)
from resin.backend.cuda.metrics import CudaMetrics, GraphPhaseTiming
from resin.backend.cuda.sampling import (
    SAMPLING_OFFSET_STRIDE,
    FlashInferSampler,
    SamplingTensors,
)
from resin.backend.cuda.tensor_graphs import (
    CudaGraphTensorRegion,
    PrefillTensorGraphs,
    TensorSpec,
)
from resin.backend.metrics import RunnerMetrics
from resin.backend.resources import CachePoolBinding
from resin.backend.topology import ParallelismCapabilities
from resin.protocols import Runner

_SAMPLING_DEFAULTS = (1.0, 1.0, None, 0, 0, False)


def resolve_kv_cache_dtype(value: str) -> torch.dtype:
    """Resolve the cache precision shared by CUDA model runners."""

    if value == "fp8":
        return torch.float8_e4m3fn
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError("kv_cache_dtype must be 'fp8' or 'bfloat16'")


class CudaRunner(Runner):
    """Common device, cache-precision, failure, and graph runtime boundary.

    Model implementations configure semantic modes and supply their graph-step
    builders. This base owns device normalization, KV precision naming, failure
    containment, capture, poisoning checks, mode-scoped statistics, physical
    shape selection, fallback accounting, and replay without knowing anything
    about transformer layers, cache geometry, kernels, or quantization.
    """

    runner_config: RunnerConfig
    prefill_chunk_size: int
    parallelism_capabilities = ParallelismCapabilities()

    def __init__(
        self,
        *,
        device: str | torch.device,
        label: str,
        kv_cache_dtype: str,
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
    ) -> None:
        if cuda_graph_prefill not in {"disabled", "breakable"}:
            raise ValueError("cuda_graph_prefill must be 'disabled' or 'breakable'")
        if cuda_graph_decode not in {"disabled", "full", "breakable"}:
            raise ValueError(
                "cuda_graph_decode must be 'disabled', 'full', or 'breakable'"
            )
        self.cuda_graph_prefill = cuda_graph_prefill
        self.cuda_graph_decode = cuda_graph_decode
        self._prefill_graph_active = False
        self._prefill_operations = ()
        self._prefill_tensor_graphs = {}
        cache_dtype = resolve_kv_cache_dtype(kv_cache_dtype)
        if not torch.cuda.is_available():
            raise RuntimeError(f"{label} requires CUDA")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(f"{label} requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_torch_dtype = cache_dtype
        self._cuda_failure = CudaFailureGuard(
            lambda: torch.cuda.synchronize(self.device),
            label=f"{label} eager execution",
        )
        self._sampler = FlashInferSampler()
        self._cache_binding: CachePoolBinding | None = None
        self._cuda_graph_executor: CudaGraphExecutor | None = None
        self._parallel_graph_model = None
        self._parallel_graph_groups = None
        self._parallel_graph_catalogs = {}
        self._parallel_graph_control = None
        self._distributed_prefill_bucket = None
        self._distributed_prefill_region_buckets = {}
        self._distributed_prefill_scope = False
        self._metrics: CudaMetrics | None = None

    def configure_metrics(self, *, enabled: bool = True) -> None:
        """Enable optional execution observations without changing forward results."""

        if enabled and getattr(self, "_metrics", None) is None:
            self._metrics = CudaMetrics(
                self.device, speculation=getattr(self, "_speculation", None)
            )
        elif not enabled:
            self._metrics = None

    def metrics_snapshot(self) -> RunnerMetrics:
        """Read completed observations without waiting for outstanding GPU work."""

        metrics = getattr(self, "_metrics", None)
        return (
            RunnerMetrics()
            if metrics is None
            else metrics.snapshot(getattr(self, "_speculation", None))
        )

    def _new_phase_timing(self, *categories: str) -> GraphPhaseTiming | None:
        """Capture optional semantic markers when metrics precede graph capture."""

        return (
            GraphPhaseTiming(categories, device=self.device)
            if getattr(self, "_metrics", None) is not None
            else None
        )

    def _eager_phase_timing(self, *categories: str):
        metrics = getattr(self, "_metrics", None)
        return None if metrics is None else metrics.eager_phases(categories)

    def _configure_parallel_graphs(self, model, groups) -> None:
        """Coordinate capture catalogs and collectives across the execution mesh."""

        self._parallel_graph_model, self._parallel_graph_groups = model, groups
        self._parallel_graph_catalogs = {}
        self._parallel_graph_control = (
            None if groups is None else ControlPlane(groups.distributed)
        )

    def _parallel_graph_capacity(self, capacity: int, *, valid_rows=None):
        if getattr(self, "_parallel_graph_groups", None) is None:
            return nullcontext()
        model = self._parallel_graph_model
        assert model is not None
        return fixed_layer_capacity(model, capacity, valid_rows=valid_rows)

    def _graph_domain_values(self, value):
        groups = getattr(self, "_parallel_graph_groups", None)
        if groups is None or groups.domain.size == 1:
            return [value]
        values = [None] * groups.domain.size
        groups.distributed.all_gather_object(values, value, group=groups.domain.handle)
        return values

    def _validate_distributed_graph_catalogs(self, request=None) -> None:
        if getattr(self, "_parallel_graph_groups", None) is None:
            return
        catalog = tuple(
            (mode, stats["configured_shapes"], stats["captured_shapes"])
            for mode, stats in self.cuda_graph_stats.items()
        )
        signature = (
            type(self).__qualname__,
            self.cuda_graph_prefill,
            self.cuda_graph_decode,
            self.runner_config.cache_write_lookahead_tokens,
            request,
            catalog,
        )
        values = self._graph_domain_values(signature)
        if any(value != values[0] for value in values):
            raise ValueError(
                "distributed CUDA graph catalogs and capture order must agree"
            )
        self._parallel_graph_catalogs = {
            mode: tuple(
                tuple(shape)
                for shape in self._require_cuda_graph_executor().mode(mode).shapes
            )
            for mode in self.cuda_graph_stats
        }

    def _select_distributed_cuda_graph(
        self,
        mode,
        context,
        *,
        required_shape,
        eligible=True,
        fallback="shape",
        agreement=None,
        validation_error=None,
    ):
        """Vote before GPU work so every attention replica chooses the same path.

        The hot vote carries eligibility, dimensions and local catalog changes.
        Any changed or untrusted catalog makes every rank join an exact object
        comparison before selection; only successful agreement updates trust.
        Wider demands also keep their complete shape through that cold path.
        An optional schedule tag rejects incompatible region sequences in
        their first vote, while validation errors fail every peer together.
        """

        eligible = eligible and (
            self.cuda_graph_decode != "disabled"
            if mode == "decode"
            else self.cuda_graph_prefill == "breakable"
            if mode == "prefill"
            else True
        )
        cache = self._require_cuda_graph_executor().mode(mode)
        required_shape = tuple(required_shape)
        catalog = tuple(tuple(shape) for shape in cache.shapes)
        trusted = self._parallel_graph_catalogs
        changed = trusted.get(mode) != catalog
        # Ordinary graph demands fit the fixed control record. Wider generic
        # GraphShapes retain the full object path rather than losing dimensions.
        record = (
            int(bool(eligible)),
            int(changed),
            len(required_shape),
            *required_shape[:3],
            *(0 for _ in range(max(0, 3 - len(required_shape)))),
        )
        groups = getattr(self, "_parallel_graph_groups", None)
        if groups is None:
            records = [record]
            errors = [validation_error]
        else:
            control = self._parallel_graph_control
            assert control is not None
            operation = f"cuda_graph_select:{mode}"
            if agreement is not None:
                operation += f":{agreement!r}"
            metrics = getattr(self, "_metrics", None)
            timed = metrics is not None and groups.topology.attn_dp > 1
            started = perf_counter() if timed else 0.0
            try:
                records, errors = control.gather(
                    groups.domain, operation, record, error=validation_error
                )
            finally:
                if timed:
                    assert metrics is not None
                    metrics.dp_attention_sync_wait_seconds.observe(
                        perf_counter() - started
                    )
        if any(error is not None for error in errors):
            raise ValueError("; ".join(error for error in errors if error is not None))
        if any(row[2] != len(required_shape) for row in records):
            raise ValueError("distributed CUDA graph demand dimensions must agree")
        if any(row[1] for row in records) or len(required_shape) > 3:
            values = self._graph_domain_values((required_shape, catalog))
            if any(value[1] != values[0][1] for value in values):
                raise ValueError("distributed CUDA graph catalogs must agree")
            trusted[mode] = catalog
            demands = [value[0] for value in values]
        else:
            demands = [row[3 : 3 + len(required_shape)] for row in records]
        if not all(row[0] for row in records):
            cache.fallback("distributed_eligibility")
            return None
        required = tuple(max(parts) for parts in zip(*demands))
        return self._select_cuda_graph(
            mode,
            match=lambda shape: all(a >= b for a, b in zip(shape, required)),
            fallback=fallback,
        )

    @contextmanager
    def _distributed_prefill_graphs(
        self,
        context,
        *,
        per_sequence=False,
        region_tokens: Mapping[range, int] | None = None,
    ):
        """Agree FFN capacities before unequal or empty replicas enter forward.

        A model can give consecutive regions a smaller token demand after a
        row-reducing operation. Each range keeps its own common bucket across
        the communication domain, including replicas with no local rows. No
        votes are needed between regions.
        """

        tokens = context.max_num_tokens
        if per_sequence:
            tokens = min(tokens, self.prefill_chunk_size)
        demands = []
        validation_error = None
        try:
            for regions, count in (region_tokens or {}).items():
                if (
                    not isinstance(regions, range)
                    or regions.step != 1
                    or not 0
                    <= regions.start
                    < regions.stop
                    <= len(self._prefill_operations)
                ):
                    raise ValueError(
                        "prefill graph regions must be nonempty contiguous ranges within the configured operations"
                    )
                demands.append((regions.start, regions.stop, count))
                if type(count) is not int or count < 0:
                    raise ValueError(
                        "prefill graph region token demands must be nonnegative integers"
                    )
            demands.sort()
            if any(left[1] > right[0] for left, right in zip(demands, demands[1:])):
                raise ValueError("prefill graph region ranges must not overlap")
        except ValueError as error:
            validation_error = str(error)
        layout = tuple((start, stop) for start, stop, _ in demands)
        agreement = None if not layout else ("prefill_regions", layout)
        shape = self._select_distributed_cuda_graph(
            "prefill",
            context,
            required_shape=(tokens, 1),
            agreement=agreement,
            validation_error=validation_error,
        )
        region_buckets = {}
        for start, stop, count in demands:
            region_shape = self._select_distributed_cuda_graph(
                "prefill",
                context,
                required_shape=(count, 1),
                agreement=("prefill_regions", layout, start, stop),
            )
            bucket = None if region_shape is None else region_shape[0]
            region_buckets.update((region, bucket) for region in range(start, stop))
        previous = getattr(self, "_distributed_prefill_bucket", None)
        previous_regions = getattr(self, "_distributed_prefill_region_buckets", {})
        previous_scope = getattr(self, "_distributed_prefill_scope", False)
        self._distributed_prefill_bucket = None if shape is None else shape[0]
        self._distributed_prefill_region_buckets = region_buckets
        self._distributed_prefill_scope = True
        try:
            with self._prefill_graphs():
                yield
        finally:
            self._distributed_prefill_bucket = previous
            self._distributed_prefill_region_buckets = previous_regions
            self._distributed_prefill_scope = previous_scope

    def _empty_prefill_regions(self) -> None:
        """Join each captured FFN without running attention or touching live KV."""

        if not self._prefill_graph_active:
            raise RuntimeError("empty prefill regions require an aligned graph bucket")
        hidden = torch.empty(
            (0, self._prefill_hidden_size), dtype=torch.bfloat16, device=self.device
        )
        ids = torch.empty(0, dtype=torch.long, device=self.device)
        for region in range(len(self._prefill_operations)):
            self._run_prefill_region(region, hidden, ids)

    def _configure_cache_binding(self, binding: CachePoolBinding) -> None:
        """Attach the model-owned physical cache to the Engine resource seam."""

        if not isinstance(binding, CachePoolBinding):
            raise TypeError("CUDA cache binding must be a CachePoolBinding")
        if self._cache_binding is not None:
            raise RuntimeError("CUDA cache binding is already configured")
        self._cache_binding = binding

    def bind_resources(self, binding: ResourceBinding) -> None:
        """Bind the model cache to its Engine-owned physical allocation."""

        if self._cache_binding is None:
            raise RuntimeError("CUDA cache binding has not been configured")
        self._cache_binding.bind_resources(binding)

    def _configure_cuda_graphs(
        self,
        modes: Mapping[str, CudaGraphMode],
    ) -> None:
        if not modes:
            raise ValueError("CUDA graph modes must not be empty")
        if self._cuda_graph_executor is not None:
            raise RuntimeError("CUDA graph modes are already configured")
        self._cuda_graph_executor = CudaGraphExecutor(
            self.device,
            {
                name: replace(mode, build=self._full_graph_builder(mode.build))
                if name == "decode" and self.cuda_graph_decode == "full"
                else mode
                for name, mode in modes.items()
            },
            synchronize_failure=self._cuda_failure,
        )

    @staticmethod
    def _full_graph_builder(build):
        def checked(shape):
            program = build(shape)
            if isinstance(program, CudaGraphProgram) and any(
                not region.capture for region in program.regions
            ):
                raise ValueError("full CUDA graph policy cannot contain eager regions")
            return program

        return checked

    def _configure_prefill_graphs(
        self,
        operations: tuple[Callable, ...],
        *,
        hidden_size: int,
        max_tokens: int,
        token_ids: bool = False,
        pad_up_to: int = 128,
        capture_context: Callable = nullcontext,
    ) -> tuple[Callable, ...]:
        """Capture row-independent compute regions used by the existing forward.

        Operations accept flattened BF16 hidden states and optional token IDs.
        Only these stateless regions are padded; attention, recurrent updates,
        and sampling keep their actual request lengths and execute eagerly.
        Local bucket selection leaves large partial chunks eager; ``pad_up_to=0``
        requires an exact local bucket. Explicit distributed capacities override
        this padding policy so every participant executes the same graph shape.
        """

        if self._prefill_operations:
            raise RuntimeError("prefill graph operations are already configured")
        self._prefill_operations = tuple(operations)
        self._prefill_hidden_size = hidden_size
        self._prefill_token_ids = token_ids
        hidden = TensorSpec((hidden_size,))
        inputs = (hidden, TensorSpec((), torch.int64)) if token_ids else (hidden,)
        regions = tuple(
            CudaGraphTensorRegion(
                str(index),
                lambda hidden, ids=None, op=operation: op(hidden, ids),
                inputs,
                hidden,
            )
            for index, operation in enumerate(operations)
        )
        self._configure_prefill_tensor_graphs(
            regions,
            max_tokens=max_tokens,
            pad_up_to=pad_up_to,
            capture_context=capture_context,
            mode="prefill",
            coordinated=True,
        )
        return tuple(
            lambda hidden, ids=None, region=region: self._run_prefill_region(
                region, hidden, ids
            )
            for region in range(len(operations))
        )

    def _configure_prefill_tensor_graphs(self, regions, **options):
        """Capture pure tensor regions while keeping attention metadata eager."""

        graphs = PrefillTensorGraphs(self, tuple(regions), **options)
        if not hasattr(self, "_prefill_tensor_graphs"):
            self._prefill_tensor_graphs = {}
        self._prefill_tensor_graphs[graphs.mode] = graphs
        return graphs.callbacks

    @contextmanager
    def _prefill_graphs(self):
        active = self._prefill_graph_active
        self._prefill_graph_active = self.cuda_graph_prefill == "breakable" and (
            not getattr(self, "_distributed_prefill_scope", False)
            or self._distributed_prefill_bucket is not None
            or any(
                bucket is not None
                for bucket in getattr(
                    self, "_distributed_prefill_region_buckets", {}
                ).values()
            )
        )
        try:
            yield
        finally:
            self._prefill_graph_active = active

    def _run_prefill_region(self, region: int, hidden, token_ids=None):
        flat_input = hidden.ndim == 2
        flattened = hidden if flat_input else hidden.reshape(-1, hidden.shape[-1])
        ids = (
            token_ids
            if token_ids is None or token_ids.ndim == 1
            else token_ids.reshape(-1)
        )
        if not self._prefill_graph_active:
            output = self._prefill_operations[region](flattened, ids)
        else:
            values = (flattened, ids) if self._prefill_token_ids else (flattened,)
            output = self._prefill_tensor_graphs["prefill"].invoke(region, *values)
        return output if flat_input else output.view_as(hidden)

    @property
    def cuda_graph_counts(self) -> dict[str, CudaGraphCounts]:
        """Return cheap runtime counters without rebuilding graph diagnostics."""
        return self._require_cuda_graph_executor().runtime_counts

    @property
    def cuda_graph_stats(self) -> dict[str, CudaGraphStats]:
        """Return statistics keyed by semantic graph mode."""

        stats = self._require_cuda_graph_executor().stats
        for mode, graphs in getattr(self, "_prefill_tensor_graphs", {}).items():
            stats[mode]["capture_scope"] = (
                "feed_forward_regions"
                if graphs.coordinated
                else "attention_projections"
            )
            if not graphs.coordinated:
                stats[mode]["captured_operations"] = tuple(
                    r.name for r in graphs.regions
                )
        return stats

    def _before_cuda_graph_capture(self) -> None:
        """Prepare model-owned state immediately before graph capture."""

    def _after_cuda_graph_capture(self) -> None:
        """Restore model-owned state immediately after graph capture."""

    def capture_cuda_graphs(self) -> dict[str, CudaGraphStats]:
        """Capture every configured mode and return mode-scoped statistics."""

        self._validate_distributed_graph_catalogs()
        self._require_cuda_graph_executor().capture_all(
            before_capture=self._before_cuda_graph_capture,
            after_capture=self._after_cuda_graph_capture,
        )
        return self.cuda_graph_stats

    def close_cuda_graphs(self) -> None:
        """Release captured collectives before closing distributed process groups."""

        if self._cuda_graph_executor is not None:
            self._cuda_graph_executor.close()
        for graphs in getattr(self, "_prefill_tensor_graphs", {}).values():
            graphs.buffers.clear()
        if (
            self._parallel_graph_model is not None
            and self._parallel_graph_groups is not None
            and any(
                value.allocated
                for value in layer_collectives(self._parallel_graph_model)
            )
        ):
            # Every peer must finish reading shared layer buffers before any
            # rank unmaps them, including when a model is reloaded in-process.
            torch.cuda.synchronize(self.device)
            self._parallel_graph_groups.distributed.barrier(
                group=self._parallel_graph_groups.domain.handle
            )
            close_layer_collectives(self._parallel_graph_model)

    def capture_cuda_graph(self, mode: str, shape: GraphShape) -> CudaGraphStats:
        """Capture one mode shape while honoring model-state hooks."""

        self._validate_distributed_graph_catalogs((mode, shape))
        executor = self._require_cuda_graph_executor()
        self._before_cuda_graph_capture()
        try:
            executor.capture(mode, shape)
        finally:
            self._after_cuda_graph_capture()
        return self.cuda_graph_stats[mode]

    def _check_cuda_graphs(self, phase: BatchPhase | None = None) -> None:
        """Reject poisoned graphs and missing prefill capture before model work."""

        executor = self._require_cuda_graph_executor()
        executor.check()
        if phase == BatchPhase.PREFILL:
            executor.mode("prefill").check_ready()
            for mode in getattr(self, "_prefill_tensor_graphs", {}):
                if mode != "prefill":
                    executor.mode(mode).check_ready()

    def _select_cuda_graph(
        self,
        mode: str,
        *,
        match: Callable[[GraphShape], bool],
        fallback: str = "shape",
    ) -> GraphShape | None:
        """Select one physical shape and account for an eager fallback."""

        cache = self._require_cuda_graph_executor().mode(mode)
        if (
            mode == "decode"
            and self.cuda_graph_decode == "disabled"
            or mode == "prefill"
            and self.cuda_graph_prefill != "breakable"
        ):
            return None
        if not cache.shapes:
            return None
        shape = cache.match(match)
        if shape is None:
            cache.fallback(fallback)
            return None
        return shape

    def _run_cuda_graph(
        self,
        mode: str,
        shape: GraphShape,
        inputs: Mapping[str, object],
    ):
        """Stage model-owned inputs and replay one selected physical shape."""

        metrics = getattr(self, "_metrics", None)
        cache = self._require_cuda_graph_executor().mode(mode)
        output = (
            cache.run(shape, inputs, metrics=metrics)
            if metrics is not None
            else cache.run(shape, inputs)
        )
        if metrics is not None:
            metrics.graph(mode)
        return output

    def _cuda_graph_fallback(self, mode: str, reason: str) -> None:
        """Record a model-level reason that a configured mode ran eagerly."""

        self._require_cuda_graph_executor().mode(mode).fallback(reason)

    def _cuda_graph_mode_enabled(self, mode: str) -> bool:
        """Return whether a semantic mode has any configured physical shapes."""

        return bool(self._require_cuda_graph_executor().mode(mode).shapes)

    def _require_cuda_graph_executor(self) -> CudaGraphExecutor:
        executor = self._cuda_graph_executor
        if executor is None:
            raise RuntimeError("CUDA graph modes have not been configured")
        return executor


def capture_sampling_inputs(
    rows: int, vocab_size: int, *, stochastic: bool = False, unique_seeds: bool = False
) -> dict[str, tuple]:
    zeros = (0,) * rows
    return {
        "sampling_temperatures": (1.0,) * rows,
        "sampling_top_ps": (1.0,) * rows,
        "sampling_top_ks": (vocab_size,) * rows,
        "sampling_seeds": tuple(range(rows)) if unique_seeds else zeros,
        "sampling_offsets": zeros,
        "sampling_stochastic_mask": (stochastic,) * rows,
    }


def pad_sampling_inputs(sampler, plan, vocab_size: int, padding: int):
    inputs = sampler.host_inputs(plan, vocab_size)
    defaults = (*_SAMPLING_DEFAULTS[:2], vocab_size, *_SAMPLING_DEFAULTS[3:])
    for (name, values), default in zip(inputs.items(), defaults, strict=True):
        inputs[name] = (*values, *((default,) * padding))
    return inputs


def graph_sample(sampler, logits, sampling: SamplingTensors, step: int):
    return sampler.sample_tensors(
        logits.float(),
        SamplingTensors(
            sampling.temperatures,
            sampling.top_ps,
            sampling.top_ks,
            sampling.seeds,
            sampling.offsets + step * SAMPLING_OFFSET_STRIDE,
            sampling.stochastic_mask,
        ),
    )


def validate_hybrid_plan(
    plan: BatchPlan,
    *,
    label: str,
    max_batch_size: int,
    max_seq_len: int,
    decode_lookahead: int,
) -> tuple[tuple[int, int], ...]:
    """Check runner limits; native construction already validates plan metadata."""

    rows = plan.num_sequences
    if rows > max_batch_size:
        raise ValueError(f"{label} batch size is outside runner capacity")
    slices = tuple(plan.token_range(row) for row in range(rows))
    contexts = plan.context_lengths
    extra = decode_lookahead if plan.phase == BatchPhase.DECODE else 0
    for context, (start, end) in zip(contexts, slices, strict=True):
        if int(context) + end - start + extra > max_seq_len:
            raise ValueError(f"{label} sequence exceeds max_seq_len")
    if plan.phase == BatchPhase.DECODE:
        if any(end - start != 1 for start, end in slices):
            raise ValueError(f"{label} decode rows must contain one anchor token")
        if any(int(context) == 0 for context in contexts):
            raise ValueError(f"{label} decode requires a cached prefix")
    return slices


__all__ = [
    "CudaRunner",
    "capture_sampling_inputs",
    "graph_sample",
    "pad_sampling_inputs",
    "resolve_kv_cache_dtype",
    "validate_hybrid_plan",
]
