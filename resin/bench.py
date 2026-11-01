"""Resident-engine benchmarks with optional synthetic expert routing.

Install BenchmarkRunner before constructing the Engine and capturing graphs.
Each point owns an uncached request cohort; loading, capture, and admission
are untimed. Pipelined decode measures continuous completion intervals, with
metrics reduced only after the window so in-flight GPU work stays timed.
"""

from __future__ import annotations

import gzip
import json
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from resin import BatchPhase, BatchPlan, SamplingParams

_TRACE_HOST_SCOPES = ContextVar("resin_benchmark_trace_host_scopes", default=False)


def _host_scope(name: str):
    """Annotate only untimed profiler callbacks; normal sweeps need no Torch."""
    if not _TRACE_HOST_SCOPES.get():
        return nullcontext()
    import torch

    return torch.profiler.record_function(name)


@dataclass(frozen=True, kw_only=True)
class BenchCase:
    phase: Literal["prefill", "decode"]
    context_length: int
    batch_size: int
    shared_prefix_tokens: int = 0

    def __post_init__(self):
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        if any(
            type(n) is not int or n < 1 for n in (self.context_length, self.batch_size)
        ):
            raise ValueError("context_length and batch_size must be positive integers")
        if (
            type(self.shared_prefix_tokens) is not int
            or not 0 <= self.shared_prefix_tokens < self.context_length
        ):
            raise ValueError(
                "shared_prefix_tokens must be an integer in [0, context_length)"
            )


def decode_output_budget(
    *,
    warmup_steps: int,
    measure_steps: int,
    draft_tokens: int,
    overlap: bool,
    trace_passes: int = 0,
    max_accepted_per_step: int | None = None,
) -> int:
    """Reserve output tokens without making a trace drain the decode pipeline.

    The Engine retains a continuation only with more than two maximum output
    steps remaining. Traces need that reservation *after* their last completion,
    in addition to the worst-case accepted output count. Before constructing a
    runner, the default is the draft count plus one; callers with a runner use
    its declared output bound. Physical cache writes do not consume output budget.
    Untraced requests keep their existing budget.
    """
    for name, value, minimum in (
        ("warmup_steps", warmup_steps, 0),
        ("measure_steps", measure_steps, 1),
        ("draft_tokens", draft_tokens, 0),
        ("trace_passes", trace_passes, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max_accepted_per_step is None:
        max_accepted_per_step = draft_tokens + 1
    if type(max_accepted_per_step) is not int or max_accepted_per_step < 1:
        raise ValueError("max_accepted_per_step must be an integer >= 1")
    if not trace_passes:
        return 1 + (warmup_steps + measure_steps + 1 + int(overlap)) * (
            draft_tokens + 1
        )
    generated = 1 + (warmup_steps + measure_steps + trace_passes) * (draft_tokens + 1)
    reservation = 2 * max_accepted_per_step + 1 if overlap else 0
    return generated + reservation


def perfetto_trace(
    directory: str | Path,
    rank: int = 0,
    metadata: dict | None = None,
    *,
    decode_passes: int = 2,
    synchronize: Callable[[], None] | None = None,
    cuda: bool = True,
):
    """Capture two untimed prefill cohorts or ``decode_passes`` completions.

    Pass the returned callback as ``run_sweep(trace_step=...)``. Each rank writes
    a compressed Chrome trace that can be opened in Perfetto or combined with
    ``scripts/merge_bench_traces.py``. CUDA imports occur only when tracing.
    Raw export and inspection use local temporary storage; only the compressed
    trace is written to the requested directory, which may be a shared volume.
    Use three decode completions for an overlapping Engine: one may harvest
    already-completed work submitted before the profiler started. This captures
    at least two fresh forwards while preserving the normal overlap path.
    Pass ``cuda=False`` and the device's ``synchronize`` for non-CUDA runners;
    their traces then contain host spans only.
    """
    if type(decode_passes) is not int or decode_passes < 1:
        raise ValueError("decode_passes must be a positive integer")
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if synchronize is None:
        synchronize = torch.cuda.synchronize
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def trace(case, run_pass):
        name = f"{case.phase}-ctx{case.context_length}-b{case.batch_size}-rank{rank:03}"
        target = directory / f"{name}.json.gz"
        passes = decode_passes if case.phase == "decode" else 2
        unit = "decode completions" if case.phase == "decode" else "prefill cohorts"
        trace_metadata = {
            **(metadata or {}),
            "case": asdict(case),
            "trace_passes": passes,
            "trace_unit": unit,
            "host_scope_policy": "untimed-perfetto-callbacks-only",
        }
        synchronize()
        with torch.profiler.profile(activities=activities) as profiler:
            profiler.add_metadata("resin_rank", str(rank))
            profiler.add_metadata(
                "resin_timing", f"{passes} untimed {unit} after measurement"
            )
            with torch.profiler.record_function(f"resin.{name}"):
                for _ in range(passes):
                    token = _TRACE_HOST_SCOPES.set(True)
                    try:
                        with _host_scope("benchmark.trace_callback"):
                            run_pass()
                    finally:
                        _TRACE_HOST_SCOPES.reset(token)
                synchronize()
        with tempfile.TemporaryDirectory(prefix="resin-bench-trace-") as staging:
            raw = Path(staging) / f"{name}.json"
            profiler.export_chrome_trace(str(raw))
            with raw.open() as source:
                data = json.load(source)
            # Kineto rewrites backslashes in add_metadata_json values, which
            # corrupts escaped quotes in token strings. Add structured metadata
            # with Python's JSON encoder after the profiler export instead.
            data["resin_benchmark"] = trace_metadata
            with gzip.open(target, "wt") as destination:
                json.dump(data, destination)
            kernels = sum(event.get("cat") == "kernel" for event in data["traceEvents"])
        print(
            f"[bench-trace] rank={rank} case={name} CUDA kernel events={kernels}",
            flush=True,
        )
        if cuda and not kernels:
            raise RuntimeError(f"{case.phase} trace contains no CUDA kernel activity")

    # run_sweep reserves the advertised count, including custom trace lengths.
    # Plain user callbacks can instead pass trace_decode_passes explicitly.
    setattr(trace, "decode_passes", decode_passes)
    return trace


class _ObservedCompletion:
    """Attribute a pipelined sample to its harvested plan, not the next launch."""

    def __init__(self, runner, plan, completion):
        self._runner, self._plan, self._completion = runner, plan, completion

    @property
    def feedback(self):
        return self._completion.feedback

    def result(self):
        with _host_scope("benchmark.runner.result"):
            output = self._completion.result()
            self._runner._observe(self._plan)
            self._runner._completed_positions = getattr(
                self._completion, "positions", None
            )
            return output


class BenchmarkRunner:
    """Delegate the runner contract while observing actual scheduled input rows.

    Synthetic routing must be configured before graph capture and never implies
    model quality. Speculative acceptance follows the model's sampled tokens.
    """

    def __init__(self, runner, *, balanced_experts=False):
        self.runner = runner
        self.draft_tokens = int(getattr(runner, "speculative_draft_tokens", 0))
        if type(balanced_experts) is not bool:
            raise ValueError("balanced_experts must be a bool")
        self.balanced_experts = balanced_experts
        if balanced_experts:
            from resin.backend.balanced_routing import BalancedRoutingMixin

            model = getattr(runner, "_model", None)
            routers = (
                []
                if model is None
                else [m for m in model.modules() if isinstance(m, BalancedRoutingMixin)]
            )
            if not routers:
                raise ValueError(
                    "balanced experts are unsupported by this model runner"
                )
            groups = getattr(runner, "parallel_groups", None)
            for router in routers:
                router.enable_balanced_routing(
                    replica=0 if groups is None else groups.attn_dp.local_rank,
                    expert_parallel_size=None if groups is None else groups.ep.size,
                )
        self._plan: BatchPlan | None = None
        self._completed_positions = None

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def _observe(self, plan: BatchPlan | None):
        self._plan = plan
        self._completed_positions = None

    @property
    def completed_context_lengths(self):
        if self._plan is None:
            raise RuntimeError("no benchmark result has been harvested")
        return (
            self._completed_positions
            if self._completed_positions is not None
            else self._plan.context_lengths
        )

    @property
    def observation(self) -> dict[str, Any]:
        plan = self._plan
        return (
            {}
            if plan is None
            else {
                "phase": "prefill" if plan.phase == BatchPhase.PREFILL else "decode",
                "input_tokens": len(plan.token_ids),
                "sequences": plan.num_sequences,
            }
        )

    def graph_counts(self):
        counts = {}
        stats = getattr(self.runner, "cuda_graph_counts", None)
        if stats is None:
            stats = getattr(self.runner, "cuda_graph_stats", {})
        for mode, stats in stats.items():
            for name in ("replay_count", "eager_fallback_count"):
                counts[f"{mode}.{name}"] = stats[name]
            for reason, value in stats["fallback_counts"].items():
                counts[f"{mode}.fallback.{reason}"] = value
        return counts

    def runner_counts(self):
        """Optional host-only monotonic backend counters, read outside timing."""
        counts = dict(getattr(self.runner, "benchmark_counters", {}))
        if any(
            not isinstance(name, str) or type(value) is not int or value < 0
            for name, value in counts.items()
        ):
            raise ValueError(
                "benchmark_counters must map names to nonnegative integers"
            )
        return counts

    def forward(self, plan):
        with _host_scope("benchmark.runner.forward"):
            self._observe(plan)
            return self.runner.forward(plan)

    def forward_distributed(self, plan, context, *, prepared=None):
        with _host_scope("benchmark.runner.forward_distributed"):
            self._observe(plan)
            return (
                self.runner.forward_distributed(plan, context)
                if prepared is None
                else self.runner.forward_distributed(plan, context, prepared=prepared)
            )

    def enqueue(self, plan, *, feedback=None):
        with _host_scope("benchmark.runner.enqueue"):
            enqueue = getattr(self.runner, "enqueue", None)
            if enqueue is None:
                return self.forward(plan)
            output = enqueue(plan, feedback=feedback)
            if callable(getattr(output, "result", None)):
                return _ObservedCompletion(self, plan, output)
            self._observe(plan)
            return output

    def enqueue_distributed(self, plan, context, *, prepared=None, feedback=None):
        with _host_scope("benchmark.runner.enqueue_distributed"):
            enqueue = getattr(self.runner, "enqueue_distributed", None)
            if enqueue is None:
                return self.forward_distributed(plan, context, prepared=prepared)
            kwargs = {"feedback": feedback}
            if prepared is not None:
                kwargs["prepared"] = prepared
            output = enqueue(plan, context, **kwargs)
            if callable(getattr(output, "result", None)):
                return _ObservedCompletion(self, plan, output)
            self._observe(plan)
            return output

    def speculation(self):
        stats = getattr(self.runner, "_speculation", None)
        return (0, 0) if stats is None else (stats.drafts, stats.accepted)


def distributed_reducer(groups):
    """Reduce unique attention-replica driver counters after each timed step."""

    drivers = groups.drivers
    if drivers is None:
        raise ValueError("only attention-replica drivers run benchmark sweeps")

    def reduce_step(local):
        values = [local]
        if drivers.size > 1:
            values = [None] * drivers.size
            groups.distributed.all_gather_object(values, local, group=drivers.handle)
        return {
            "seconds": max(v["seconds"] for v in values),
            "replica_sequences": [v["sequences"] for v in values],
            **{
                key: sum(v[key] for v in values)
                for key in (
                    "input_tokens",
                    "committed_tokens",
                    "drafts",
                    "accepted",
                    "sequences",
                )
            },
            "graph_counts": {
                key: sum(v.get("graph_counts", {}).get(key, 0) for v in values)
                for key in {key for v in values for key in v.get("graph_counts", {})}
            },
            **(
                {
                    "runner_counts": {
                        key: sum(v.get("runner_counts", {}).get(key, 0) for v in values)
                        for key in {
                            key for v in values for key in v.get("runner_counts", {})
                        }
                    }
                }
                if any(v.get("runner_counts") for v in values)
                else {}
            ),
            "overlap_counts": {
                key: sum(v.get("overlap_counts", {}).get(key, 0) for v in values)
                for key in ("enqueues", "continuations")
            },
            "context_start_min": min(v["context_start_min"] for v in values),
            "context_start_max": max(v["context_start_max"] for v in values),
            "context_end_min": min(v["context_end_min"] for v in values),
            "context_end_max": max(v["context_end_max"] for v in values),
            "device_phase_ms": {
                phase: max(v.get("device_phase_ms", {}).get(phase, 0) for v in values)
                for phase in {
                    key for v in values for key in v.get("device_phase_ms", {})
                }
            },
        }

    return reduce_step


def run_sweep(
    engine,
    cases: Iterable[BenchCase],
    *,
    runner: BenchmarkRunner,
    prompt_tokens: Sequence[int] | None = None,
    prompt_batches: Mapping[int, Sequence[Sequence[int]]] | None = None,
    sampling: Mapping[str, Any] | None = None,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    prefill_warmup_cohorts: int | None = None,
    prefill_measure_cohorts: int | None = None,
    gpu_count: int = 1,
    attn_dp: int = 1,
    synchronize: Callable[[], None] = lambda: None,
    reduce_step: Callable[[dict], dict] | None = None,
    metadata: dict | None = None,
    on_result: Callable[[dict], None] | None = None,
    trace_step: Callable[[BenchCase, Callable[[], dict]], None] | None = None,
    trace_decode_passes: int | None = None,
) -> list[dict]:
    """Measure prefill cohorts or a steady decode window on an existing Engine.

    batch_size counts requests per attention-DP replica. Prefill finishes whole
    cohorts, so its actual step count can exceed the requested minimum when
    scheduling chunks. Decode records exactly the requested number of steps;
    its context grows naturally during warmup and measurement. All DP drivers
    call together with distributed_reducer; records contain global token counts
    and the slowest driver's completion interval. Overlapped decode defers these
    reductions until the continuous measurement window ends. The caller retains
    engine ownership.
    on_result receives each completed record after request cleanup, outside timing.
    Prefill records ``prefill_seconds`` for each complete measured cohort, so
    chunking does not turn a context's prefill latency into a single chunk's time.
    trace_step receives an untimed workload callback: one fresh complete cohort
    for prefill, or one live completion for decode. It may invoke that callback
    twice, or three times for overlapped decode to allow for a completion
    submitted before tracing. perfetto_trace advertises its exact decode count;
    custom callbacks use these defaults or declare trace_decode_passes. More
    calls than reserved fail before executing extra work. All distributed
    drivers must use the same passes. Trace output headroom preserves the normal
    continuation guard; cache-page boundaries can still interrupt overlap.
    """

    sampling = dict[str, Any]({"temperature": 0.0, **(sampling or {})})
    if set(sampling) - {"temperature", "top_p", "top_k", "seed"}:
        raise ValueError(
            "benchmark sampling accepts temperature, top_p, top_k and seed"
        )
    SamplingParams(max_output_tokens=1, stop_token_ids=[], **sampling)
    for name, value, minimum in (
        ("warmup_steps", warmup_steps, 0),
        ("measure_steps", measure_steps, 1),
        ("gpu_count", gpu_count, 1),
        ("attn_dp", attn_dp, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value, minimum in (
        ("prefill_warmup_cohorts", prefill_warmup_cohorts, 0),
        ("prefill_measure_cohorts", prefill_measure_cohorts, 1),
    ):
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if gpu_count % attn_dp:
        raise ValueError("gpu_count must be divisible by attn_dp")
    if attn_dp > 1 and reduce_step is None:
        raise ValueError("attention-DP benchmarks require a step reducer")
    if prompt_batches is None:
        if prompt_tokens is None or not len(prompt_tokens):
            raise ValueError("prompt_tokens must not be empty")
    elif prompt_tokens is not None:
        raise ValueError("provide prompt_batches or prompt_tokens, not both")
    local = getattr(engine, "local", engine)
    overlap = bool(getattr(local, "overlap", False)) and callable(
        getattr(runner.runner, "enqueue", None)
    )
    declared_passes = getattr(trace_step, "decode_passes", None)
    if trace_step is None:
        if trace_decode_passes is not None:
            raise ValueError("trace_decode_passes requires trace_step")
        trace_passes = 0
    else:
        if (
            trace_decode_passes is not None
            and declared_passes is not None
            and trace_decode_passes != declared_passes
        ):
            raise ValueError("trace_decode_passes disagrees with the trace callback")
        trace_passes = (
            trace_decode_passes
            if trace_decode_passes is not None
            else declared_passes
            if declared_passes is not None
            else 2 + int(overlap)
        )
        if type(trace_passes) is not int or trace_passes < 1:
            raise ValueError("trace_decode_passes must be a positive integer")
    if local.has_work():
        raise ValueError("benchmark engine must not have active requests")
    reduce = reduce_step or (lambda value: value)
    records = []
    previous_completion = None

    def step(*, defer_reduce=False):
        nonlocal previous_completion
        before_drafts, before_accepted = runner.speculation()
        before_graphs = runner.graph_counts()
        before_counters = runner.runner_counts()
        before_overlap = getattr(local, "overlap_stats", (0, 0))
        runner._plan = None
        if not overlap:
            synchronize()
        started = (
            previous_completion
            if defer_reduce and previous_completion is not None
            else time.perf_counter()
        )
        # Filling an empty pipeline produces no host result. Include any such
        # submission in the next completion interval, including boundary bubbles.
        while True:
            with _host_scope("benchmark.engine.step"):
                events = engine.step()
            if runner._plan is not None:
                break
            if not getattr(local, "overlap_pending", False):
                raise RuntimeError("Engine must use the supplied BenchmarkRunner")
        if not overlap:
            synchronize()
        finished = time.perf_counter()
        elapsed = finished - started
        previous_completion = finished
        with _host_scope("benchmark.post_timer_bookkeeping"):
            plan = runner._plan
            if plan is None:
                raise RuntimeError("Engine must use the supplied BenchmarkRunner")
            drafts, accepted = runner.speculation()
            graph_counts = runner.graph_counts()
            counters = runner.runner_counts()
            counter_deltas = {
                key: value - before_counters.get(key, 0)
                for key, value in counters.items()
            }
            if set(before_counters) - set(counters) or any(
                value < 0 for value in counter_deltas.values()
            ):
                raise ValueError("benchmark_counters must be monotonic during a sweep")
            after_overlap = getattr(local, "overlap_stats", (0, 0))
            starts = getattr(
                runner, "completed_context_lengths", plan.context_lengths
            ).tolist()
            if plan.phase == BatchPhase.PREFILL:
                offsets = plan.token_indptr.tolist()
                ends = [
                    start + b - a for start, a, b in zip(starts, offsets, offsets[1:])
                ]
            else:
                counts = Counter(event.request_id for event in events)
                ends = [
                    start + counts[int(sequence)]
                    for start, sequence in zip(starts, plan.sequence_ids)
                ]
            measured = {
                "seconds": elapsed,
                "input_tokens": runner.observation["input_tokens"],
                "sequences": runner.observation["sequences"],
                "replica_sequences": [runner.observation["sequences"]],
                "committed_tokens": len(events),
                "drafts": drafts - before_drafts,
                "accepted": accepted - before_accepted,
                "context_start_min": min(starts),
                "context_start_max": max(starts),
                "context_end_min": min(ends),
                "context_end_max": max(ends),
                "device_phase_ms": (
                    getattr(runner, "last_dspark_phase_ms", None) or {}
                    if plan.phase == BatchPhase.DECODE
                    else {}
                ),
                "graph_counts": {
                    key: value - before_graphs.get(key, 0)
                    for key, value in graph_counts.items()
                },
                **({"runner_counts": counter_deltas} if counter_deltas else {}),
                "overlap_counts": dict(
                    zip(
                        ("enqueues", "continuations"),
                        (
                            after - before
                            for before, after in zip(before_overlap, after_overlap)
                        ),
                        strict=True,
                    )
                ),
            }
            return measured if defer_reduce else reduce(measured)

    for case in cases:
        if case.batch_size > local.max_running_requests:
            raise ValueError("batch_size exceeds the engine request capacity")
        if case.context_length >= local.context_length:
            raise ValueError("context_length leaves no space for generated tokens")
        cohort = None
        if prompt_batches is not None:
            cohort = prompt_batches.get(case.context_length)
            if cohort is None or len(cohort) < case.batch_size:
                raise ValueError("prompt corpus does not cover the requested cohort")
            cohort = cohort[: case.batch_size]
            if any(len(tokens) != case.context_length for tokens in cohort):
                raise ValueError("corpus prompts must match the exact context length")
            tokens = list(cohort[0])
            if case.shared_prefix_tokens and any(
                list(row[: case.shared_prefix_tokens])
                != tokens[: case.shared_prefix_tokens]
                for row in cohort
            ):
                raise ValueError("corpus prompts do not share the requested prefix")
        else:
            assert prompt_tokens is not None
            tokens = (
                list(prompt_tokens)
                * ((case.context_length + len(prompt_tokens) - 1) // len(prompt_tokens))
            )[: case.context_length]
        leases = []
        samples = []
        prefill_seconds = []
        warmed = 0
        warmed_cohorts = 0
        result = None
        nonce = 0
        cached_tokens = []
        shared = case.shared_prefix_tokens

        def admit(max_output_tokens):
            nonlocal nonce
            if not shared:
                local.clear_prefix_cache()
            leases.clear()
            for index in range(case.batch_size):
                request_tokens = tokens if cohort is None else cohort[index]
                if shared:
                    if nonce >= runner.vocab_size:
                        raise ValueError(
                            "shared-prefix sweep exhausted distinct private tokens"
                        )
                    request_tokens = list(request_tokens)
                    request_tokens[shared] = nonce
                    nonce += 1
                lease = engine.submit(
                    request_tokens,
                    sampling_params=SamplingParams(
                        max_output_tokens=max_output_tokens,
                        stop_token_ids=[],
                        **sampling,
                    ),
                )
                leases.append(lease)
                cached_tokens.append(lease.cached_tokens)
                if lease.cached_tokens != shared:
                    raise ValueError(
                        f"shared prefix reuse differs from requested checkpoint: "
                        f"requested={shared}, cached={lease.cached_tokens}"
                    )

        try:
            local.clear_prefix_cache()
            if shared:
                if local.prefix_cache_stats is None:
                    raise ValueError(
                        "shared-prefix benchmarks require Engine(prefix_cache=True)"
                    )
                leases.append(
                    engine.submit(
                        tokens[:shared],
                        sampling_params=SamplingParams(
                            max_output_tokens=1, stop_token_ids=[], **sampling
                        ),
                    )
                )
                while not leases[0].done:
                    step()  # Seed the one shared prefix outside warmup/measurement.
            if case.phase == "prefill":
                for target, measured, complete_cohorts in (
                    (
                        prefill_warmup_cohorts
                        if prefill_warmup_cohorts is not None
                        else warmup_steps,
                        False,
                        prefill_warmup_cohorts is not None,
                    ),
                    (
                        prefill_measure_cohorts
                        if prefill_measure_cohorts is not None
                        else measure_steps,
                        True,
                        prefill_measure_cohorts is not None,
                    ),
                ):
                    count = 0
                    while count < target:
                        admit(1)
                        cohort_seconds = 0.0
                        while not all(lease.done for lease in leases):
                            result = step()
                            if runner.observation["phase"] != "prefill":
                                raise RuntimeError("prefill benchmark entered decode")
                            if not complete_cohorts:
                                count += 1
                            if not measured:
                                warmed += 1
                            if measured:
                                samples.append(result)
                                cohort_seconds += result["seconds"]
                        if measured:
                            prefill_seconds.append(cohort_seconds)
                        else:
                            warmed_cohorts += 1
                        if complete_cohorts:
                            count += 1
            else:
                maximum = decode_output_budget(
                    warmup_steps=warmup_steps,
                    measure_steps=measure_steps,
                    draft_tokens=runner.draft_tokens,
                    overlap=overlap,
                    trace_passes=trace_passes,
                    max_accepted_per_step=getattr(
                        getattr(runner, "runner_config", None),
                        "max_accepted_per_step",
                        None,
                    ),
                )
                if case.context_length + maximum > local.context_length:
                    raise ValueError("engine context cannot fit the full decode window")
                admit(maximum)
                while local.next_phase() == BatchPhase.PREFILL:
                    step()
                for index in range(warmup_steps + measure_steps):
                    measuring = index >= warmup_steps
                    if overlap and measuring and warmup_steps == 0 and index == 0:
                        previous_completion = None
                    result = step(defer_reduce=overlap)
                    if runner.observation["phase"] != "decode":
                        raise RuntimeError("decode benchmark did not execute decode")
                    expected = case.batch_size * (1 if overlap else attn_dp)
                    if result["sequences"] != expected:
                        raise RuntimeError(
                            "decode batch was not fully scheduled: "
                            f"scheduled_global={result['sequences']}, "
                            f"expected_global={case.batch_size * attn_dp}, "
                            f"scheduled_local={runner.observation['sequences']}, "
                            f"expected_per_replica={case.batch_size}, "
                            f"context_length={case.context_length}, phase={case.phase}"
                        )
                    if any(lease.done for lease in leases):
                        raise RuntimeError("decode cohort retired during measurement")
                    if measuring:
                        samples.append(result)
                if overlap:
                    # A per-step reduction or CUDA synchronization lets future
                    # GPU work run outside the timer. Reduce only after the
                    # continuous completion window has finished.
                    samples = [reduce(sample) for sample in samples]
                warmed = warmup_steps
            seconds = [sample["seconds"] for sample in samples]
            inputs = (
                sum(s["input_tokens"] for s in samples)
                if case.phase == "prefill"
                else 0
            )
            drafts = sum(s["drafts"] for s in samples)
            records.append(
                {
                    "schema_version": 1,
                    "phase": case.phase,
                    "context_length": case.context_length,
                    "shared_prefix_tokens": shared,
                    "private_prompt_tokens": case.context_length - shared,
                    "cached_tokens_min": min(cached_tokens),
                    "cached_tokens_max": max(cached_tokens),
                    "batch_size": case.batch_size,
                    "global_batch_size": case.batch_size * attn_dp,
                    "measured_global_batch_size_min": min(
                        s["sequences"] for s in samples
                    ),
                    "measured_global_batch_size_max": max(
                        s["sequences"] for s in samples
                    ),
                    "gpu_count": gpu_count,
                    "attn_dp": attn_dp,
                    "prefill_tokens": inputs / len(samples),
                    "input_tokens": inputs,
                    "draft_tokens": runner.draft_tokens,
                    "measured_accept_rate": sum(s["accepted"] for s in samples) / drafts
                    if drafts
                    else None,
                    "balanced_experts": runner.balanced_experts,
                    "balanced_experts_scope": "physical_routing_rows"
                    if runner.balanced_experts
                    else None,
                    "warmup_steps": warmed,
                    "measure_steps": len(samples),
                    "step_seconds": seconds,
                    **(
                        {
                            "prefill_seconds": prefill_seconds,
                            "prefill_warmup_cohorts": warmed_cohorts,
                            "prefill_measure_cohorts": len(prefill_seconds),
                        }
                        if prefill_seconds
                        else {}
                    ),
                    **(
                        {
                            "device_phase_ms": [
                                sample["device_phase_ms"] for sample in samples
                            ]
                        }
                        if any(sample.get("device_phase_ms") for sample in samples)
                        else {}
                    ),
                    "window_seconds": sum(seconds),
                    "committed_tokens": sum(s["committed_tokens"] for s in samples),
                    "measured_context_start_min": samples[0]["context_start_min"],
                    "measured_context_start_max": samples[0]["context_start_max"],
                    "measured_context_end_min": samples[-1]["context_end_min"],
                    "measured_context_end_max": samples[-1]["context_end_max"],
                    "graph_counts": {
                        key: sum(s["graph_counts"].get(key, 0) for s in samples)
                        for key in {key for s in samples for key in s["graph_counts"]}
                    },
                    **(
                        {
                            "runner_counts": {
                                key: sum(
                                    s.get("runner_counts", {}).get(key, 0)
                                    for s in samples
                                )
                                for key in {
                                    key
                                    for s in samples
                                    for key in s.get("runner_counts", {})
                                }
                            }
                        }
                        if any(s.get("runner_counts") for s in samples)
                        else {}
                    ),
                    "overlap_counts": {
                        key: sum(
                            s.get("overlap_counts", {}).get(key, 0) for s in samples
                        )
                        for key in ("enqueues", "continuations")
                    },
                    "metadata": dict(metadata or {}),
                }
            )
            if trace_step is not None:
                synchronize()
                previous_completion = None
                traced_completions = 0

                def trace_pass():
                    nonlocal traced_completions
                    if case.phase == "decode":
                        if traced_completions >= trace_passes:
                            raise ValueError(
                                "trace exceeded reserved decode completions"
                            )
                        traced_completions += 1
                        return step(defer_reduce=True)
                    # A complete cohort exercises all chunks of this context.
                    # Admission clears private prefix state before each pass.
                    admit(1)
                    traced = []
                    while not all(lease.done for lease in leases):
                        traced.append(step(defer_reduce=True))
                        if runner.observation["phase"] != "prefill":
                            raise RuntimeError("prefill trace entered decode")
                    return {
                        "seconds": sum(item["seconds"] for item in traced),
                        "input_tokens": sum(item["input_tokens"] for item in traced),
                        "sequences": case.batch_size,
                        "steps": len(traced),
                    }

                trace_step(case, trace_pass)
        except Exception as error:
            diagnostic = dict(
                **asdict(case),
                error=f"{type(error).__name__}: {error}",
                attn_dp=attn_dp,
                actual_global=(result or {}).get("sequences"),
                expected_global=case.batch_size * attn_dp,
                observation=runner.observation,
                per_replica=(result or {}).get("replica_sequences"),
                cache={
                    name: getattr(local, name, None)
                    for name in (
                        "ready_request_count",
                        "active_request_count",
                        "new_token_ratio",
                        "cache_parent_capacity",
                        "cache_free_parent_count",
                        "cache_occupied_block_count",
                    )
                },
            )
            print("[bench-error] " + json.dumps(diagnostic), flush=True)
            raise
        finally:
            if not local.poisoned:
                for lease in leases:
                    if not lease.done:
                        engine.cancel(lease)
                local.clear_prefix_cache()
        if on_result is not None:
            on_result(records[-1])
    return records
