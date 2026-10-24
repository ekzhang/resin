"""Load one CUDA model per rank and run a resident, synchronous Engine sweep."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import resin
from deploy.bench_trace import decode_trace
from resin import Engine, Parallelism, SchedulerConfig
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups
from resin.recipe import DeepSeekV41Recipe


def _runner(model, checkpoint, *, device, groups, cases, options, context_capacity):
    common: dict[str, Any] = dict(
        device=device,
        parallel_groups=groups,
        cuda_graph_prefill=options["prefill_graph"],
        cuda_graph_decode=options["decode_graph"],
        gpu_memory_utilization=options["gpu_memory_utilization"],
        cache_parent_capacity=128 if options["validate_model"] else None,
        fast_allreduce=False,
    )
    batches = sorted({case.batch_size for case in cases})
    if model.startswith("gpt-oss"):
        from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner
        from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

        if options["draft_tokens"]:
            raise ValueError("GPT-OSS has no draft model; pass --draft-tokens 0")

        class BenchGptRunner(GptOssCudaRunner):
            prefill_chunk_size = max(batches) * options["prefill_chunk_size"]

        return BenchGptRunner(
            checkpoint,
            kernel_moe=FlashInferSM90MXFP4Kernel(),
            cuda_graph_decode_shapes=(
                tuple((batch, (context_capacity + 127) // 128) for batch in batches)
                if options["decode_graph"] != "disabled"
                else ()
            ),
            **common,
        )
    if model == "deepseek-v4":
        from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner

        runner_type = DeepSeekV4CudaRunner
    elif model == "deepseek-v41":
        from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner

        runner_type = DeepSeekV41CudaRunner
    else:
        raise ValueError(f"unknown benchmark model: {model}")
    return runner_type(
        checkpoint,
        **(
            {"expert_backend": options["expert_backend"]}
            if model == "deepseek-v41"
            else {}
        ),
        max_seq_len=context_capacity,
        max_batch_size=max(batches),
        speculative_tokens=options["draft_tokens"],
        prefill_chunk_size=options["prefill_chunk_size"],
        profile_graph_phases=options.get("profile_graph_phases", False),
        cuda_graph_decode_shapes=(
            tuple((batch, context_capacity) for batch in batches)
            if options["decode_graph"] != "disabled"
            else ()
        ),
        **common,
    )


def _result_callback(report_path):
    """Keep a valid snapshot of completed cases and print untimed progress."""
    records = []
    path = Path(report_path)

    def save(record):
        records.append(record)
        pending = path.with_suffix(path.suffix + ".tmp")
        pending.write_text(json.dumps(records))
        pending.replace(path)
        print("[bench-result] " + json.dumps(record), flush=True)

    return save


def _cache_preflight(raw, cases, options, groups=None):
    """Reject a partial decode cohort before spending time on capture or prefill."""
    plan = getattr(raw, "cache_capacity_plan", None)
    if plan is None:
        return
    layout = raw.cache_layout
    requirements = []
    for case in cases:
        context = case.context_length
        if case.phase == "decode":
            context += (options["warmup_steps"] + options["measure_steps"] + 1) * (
                options["draft_tokens"] + 1
            ) + options["draft_tokens"]
        parents = layout.estimated_parent_count(
            [context] * case.batch_size, shared_prefix_tokens=case.shared_prefix_tokens
        )
        requirements.append(
            dict(
                **asdict(case),
                peak_context=context,
                required_parents=parents,
            )
        )
    rank = 0 if groups is None else groups.rank
    print(
        "[bench-cache] "
        + json.dumps(
            dict(
                rank=rank,
                capacity=asdict(plan),
                cases=requirements,
            )
        ),
        flush=True,
    )
    failures = [
        f"rank {rank} {row['phase']} ctx={row['context_length']} "
        f"batch={row['batch_size']} needs {row['required_parents']} cache parents, "
        f"capacity={layout.parent_capacity} (peak context {row['peak_context']})"
        for row in requirements
        if row["required_parents"] > layout.parent_capacity
    ]
    errors = [failures]
    if groups is not None and groups.domain.size > 1:
        errors = [None] * groups.domain.size
        groups.distributed.all_gather_object(
            errors, failures, group=groups.domain.handle
        )
    failures = [failure for values in errors if values for failure in values]
    if failures:
        raise ValueError("benchmark cache preflight failed: " + "; ".join(failures))


def _checkpoint_validation(raw, engine, checkpoint, metadata, groups):
    from tokenizers import Tokenizer

    from deploy.deepseek_v41_validation import _validation_stage, validate_model

    drivers = None if groups is None else groups.drivers
    rank = 0 if groups is None else groups.rank

    def gather(stage, value):
        if drivers is None or drivers.size == 1:
            return [value]
        values = [None] * drivers.size
        groups.distributed.all_gather_object(
            values, (stage, value), group=drivers.handle
        )
        if any(value is None or value[0] != stage for value in values):
            raise RuntimeError("validation drivers reached different stages")
        return [value[1] for value in values if value is not None]

    def coordinate(stage, error):
        errors = gather(
            stage, None if error is None else f"{type(error).__name__}: {error}"
        )
        failures = [
            f"rank {source}: {error}"
            for source, error in zip(
                (rank,) if drivers is None else drivers.ranks, errors, strict=True
            )
            if error is not None
        ]
        if failures:
            raise RuntimeError(f"validation {stage}: " + "; ".join(failures))

    try:
        with _validation_stage("prompt", coordinate):
            recipe = DeepSeekV41Recipe(
                Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
            )
            prompt = recipe.render_ids(
                [
                    resin.recipe.Message(
                        resin.recipe.Role.USER,
                        [
                            resin.recipe.Text(
                                "Count from 1 to 20, putting each number on its own line. Do not add an introduction."
                            )
                        ],
                    )
                ]
            )
        local = {
            "rank": rank,
            **validate_model(
                raw,
                engine,
                prompt_tokens=prompt,
                synchronize=torch.cuda.synchronize,
                coordinate=coordinate,
            ),
        }
    except Exception as error:
        local = {"rank": rank, "ok": False, "error": f"{type(error).__name__}: {error}"}
    reports = gather("reports", local)
    if len(reports) == 1:
        return [
            {key: value for key, value in reports[0].items() if key != "rank"}
            | {"metadata": metadata}
        ]
    return [
        {
            "ok": all(report["ok"] for report in reports),
            "drivers": reports,
            "metadata": metadata,
        }
    ]


def _rank(
    rank,
    port,
    report_path,
    model,
    checkpoint,
    case_values,
    topology,
    options,
    *,
    runner_factory=None,
    validation=None,
):
    from resin.bench import (
        BenchCase,
        BenchmarkRunner,
        distributed_reducer,
        run_sweep,
    )

    if model.startswith("gpt-oss"):
        from deploy.jit_cache import install_flashinfer_hash_cache

        install_flashinfer_hash_cache()
    torch.cuda.set_device(rank)
    groups = None
    if topology.world_size > 1:
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=topology.world_size,
            device_id=torch.device("cuda", rank),
            timeout=timedelta(minutes=30),
        )
        groups = ParallelGroups.create(topology)
    cases = [BenchCase(**case) for case in case_values]
    prefill_budget = (
        max(case.batch_size for case in cases) * options["prefill_chunk_size"]
    )
    # Reserve real generated positions, including draft write lookahead. Context
    # lengths on the charts remain the prompt lengths recorded by the harness.
    headroom = (options["warmup_steps"] + options["measure_steps"] + 4) * (
        options["draft_tokens"] + 1
    )
    capacity = max(case.context_length for case in cases) + headroom
    if rank == 0:
        print(f"[bench] loading {model} over {topology.world_size} GPUs", flush=True)
    started = time.perf_counter()
    raw = (runner_factory or _runner)(
        model,
        checkpoint,
        device=f"cuda:{rank}",
        groups=groups,
        cases=cases,
        options=options,
        context_capacity=capacity,
    )
    _cache_preflight(raw, cases, options, groups)
    observed = BenchmarkRunner(
        raw,
        accepted_drafts=options["accepted_drafts"],
        balanced_experts=options["balanced_experts"],
    )
    torch.cuda.synchronize()
    loaded_seconds = time.perf_counter() - started
    if rank == 0:
        print(
            f"[bench] weights ready in {loaded_seconds:.1f}s; capturing graphs",
            flush=True,
        )
    started = time.perf_counter()
    raw.capture_cuda_graphs()
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - started
    if rank == 0:
        print(f"[bench] graphs ready in {capture_seconds:.1f}s", flush=True)
    overlap = options.get("overlap", True) and callable(getattr(raw, "enqueue", None))
    metadata = {
        **options["metadata"],
        "model": model,
        "checkpoint": str(checkpoint),
        "hardware": torch.cuda.get_device_name(rank),
        "parallelism": {
            axis: getattr(topology, axis)
            for axis in ("world_size", "attn_dp", "attn_tp", "cp", "ep")
        },
        "graphs": {
            "prefill": options["prefill_graph"],
            "decode": options["decode_graph"],
        },
        "quant": "checkpoint",
        "expert_backend": options["expert_backend"],
        "load_seconds": loaded_seconds,
        "capture_seconds": capture_seconds,
        "profile_graph_phases": options.get("profile_graph_phases", False),
        "overlap": overlap,
        "timing": "continuous_completions" if overlap else "synchronized_steps",
        "prefill_chunk_size": options["prefill_chunk_size"],
        "max_batch_tokens": prefill_budget,
    }
    engine_options: dict[str, Any] = dict(
        max_running_requests=max(case.batch_size for case in cases),
        context_length=capacity,
        prefix_cache=any(case.shared_prefix_tokens for case in cases),
        enable_overlap=options.get("overlap", True),
        scheduler_config=SchedulerConfig(
            max_batch_tokens=prefill_budget,
            prefill_chunk_size=options["prefill_chunk_size"],
        ),
    )
    worker = None if groups is None else DistributedWorker(observed, groups)
    if groups is not None and worker is not None and not groups.is_driver:
        worker.serve()
        os._exit(0)
    engine = (
        Engine(observed, **engine_options)
        if worker is None
        else DistributedEngine(worker, **engine_options)
    )
    try:
        # A seeded synthetic prompt avoids tokenization overhead and special
        # token stops. Every replica has distinct, reproducible input tokens.
        generator = torch.Generator().manual_seed(41 + rank)
        prompt = torch.randint(
            1000, min(30_000, raw.vocab_size), (capacity,), generator=generator
        ).tolist()
        if options["validate_model"]:
            try:
                records = (validation or _checkpoint_validation)(
                    raw, engine, checkpoint, metadata, groups
                )
            except Exception as error:
                diagnostics = getattr(error, "validation_records", None)
                if rank == 0 and diagnostics is not None:
                    Path(report_path).write_text(json.dumps(diagnostics))
                raise
        else:
            records = run_sweep(
                engine,
                cases,
                runner=observed,
                prompt_tokens=prompt,
                warmup_steps=options["warmup_steps"],
                measure_steps=options["measure_steps"],
                gpu_count=topology.world_size,
                attn_dp=topology.attn_dp,
                synchronize=torch.cuda.synchronize,
                reduce_step=None if groups is None else distributed_reducer(groups),
                metadata=metadata,
                on_result=_result_callback(report_path) if rank == 0 else None,
                trace_step=(
                    None
                    if options.get("trace_dir") is None
                    else decode_trace(options["trace_dir"], rank, metadata)
                ),
            )
        if rank == 0:
            Path(report_path).write_text(json.dumps(records))
    finally:
        # Closing a poisoned domain still tells followers to stop. Preserve the
        # original failure/report if close repeats that same domain error.
        failed = engine.poisoned or sys.exc_info()[0] is not None
        try:
            engine.close()
        except Exception:
            if not failed:
                raise
    # Spawned process exit releases graph-owned NCCL communicators together.
    if groups is not None:
        os._exit(0)


def run_benchmark(
    model: str,
    checkpoint: str | Path,
    cases: list[dict],
    *,
    parallelism: Parallelism | None = None,
    draft_tokens: int = 4,
    accepted_drafts: int | None = 3,
    balanced_experts: bool = True,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    prefill_chunk_size: int = 4096,
    prefill_graph: str = "breakable",
    decode_graph: str = "full",
    gpu_memory_utilization: float = 0.9,
    metadata: dict | None = None,
    validate_model: bool = False,
    expert_backend: str = "flashinfer",
    profile_graph_phases: bool = False,
    overlap: bool = True,
    trace_dir: str | None = None,
) -> list[dict]:
    """One load and graph catalog for all compatible points in this sweep."""
    from resin.bench import BenchCase

    topology = parallelism or Parallelism()
    if validate_model and model != "deepseek-v41":
        raise ValueError("checkpoint validation currently requires V4.1")
    if expert_backend not in ("flashinfer", "deepgemm"):
        raise ValueError(f"unknown expert backend: {expert_backend}")
    if expert_backend == "deepgemm":
        if model != "deepseek-v41":
            raise ValueError("the DeepGEMM expert backend currently requires V4.1")
        if topology.world_size == 1:
            raise ValueError(
                "DeepGEMM sweeps require at least two ranks with explicit parallel groups"
            )
        if (
            topology.attn_tp != 1
            or topology.cp != 1
            or topology.attn_dp != topology.world_size
            or topology.ep != topology.world_size
        ):
            raise ValueError("DeepGEMM requires ADP=EP=world_size, attn_tp=1, and cp=1")
    if validate_model:
        accepted_drafts, balanced_experts = None, False
    if not cases:
        raise ValueError("the benchmark needs at least one case")
    for values in cases:
        BenchCase(**values)
    if trace_dir and (validate_model or not any(c["phase"] == "decode" for c in cases)):
        raise ValueError(
            "traces require a decode benchmark, outside checkpoint validation"
        )
    if (
        draft_tokens < 0
        or accepted_drafts is not None
        and not 0 <= accepted_drafts <= draft_tokens
    ):
        raise ValueError("accepted drafts must be between zero and draft tokens")
    if model.startswith("gpt-oss") and draft_tokens:
        raise ValueError("GPT-OSS has no draft model; pass --draft-tokens 0")
    if model.startswith("gpt-oss") and topology.world_size == 1:
        from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

        limit = FlashInferSM90MXFP4Kernel.max_tokens
        budget = max(case["batch_size"] for case in cases) * prefill_chunk_size
        if budget > limit:
            raise ValueError(
                f"single-GPU GPT-OSS prefill budget {budget} exceeds the kernel's "
                f"{limit}-token limit; reduce batch size or prefill_chunk_size"
            )
    if torch.cuda.device_count() < topology.world_size:
        raise ValueError(f"the sweep requires {topology.world_size} visible GPUs")
    options = dict(
        draft_tokens=draft_tokens,
        accepted_drafts=accepted_drafts,
        balanced_experts=balanced_experts,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        prefill_chunk_size=prefill_chunk_size,
        prefill_graph=prefill_graph,
        decode_graph=decode_graph,
        gpu_memory_utilization=gpu_memory_utilization,
        metadata=metadata or {},
        validate_model=validate_model,
        expert_backend=expert_backend,
        profile_graph_phases=profile_graph_phases,
        overlap=overlap,
        trace_dir=trace_dir,
    )
    with tempfile.TemporaryDirectory(prefix="resin-bench-") as directory:
        output = Path(directory) / "results.json"
        if topology.world_size == 1:
            _rank(0, 0, output, model, checkpoint, cases, topology, options)
        else:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            torch.multiprocessing.spawn(
                _rank,
                args=(port, output, model, checkpoint, cases, topology, options),
                nprocs=topology.world_size,
                join=True,
            )
        return json.loads(output.read_text())


def write_results(path: str | Path, records: list[dict]) -> None:
    """Append raw points; plotting deliberately resolves repeated points later."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["run_benchmark", "write_results"]
