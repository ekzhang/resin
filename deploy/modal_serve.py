"""One Modal serving application, configured by RESIN_MODEL.

Download: RESIN_MODEL=deepseek-v41 uv run modal run deploy/modal_serve.py
Serve:    RESIN_MODEL=deepseek-v41 uv run modal deploy deploy/modal_serve.py
Each selection has its own app/endpoint; adding a model requires a registry entry.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback
from datetime import timedelta
from multiprocessing.connection import wait
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from deploy.modal_image import build_modal_image
from deploy.models import (
    create_recipe,
    create_runner,
    download_checkpoint,
    get_model,
    image_environment,
    prepare_runtime,
    require_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = get_model(os.environ.get("RESIN_MODEL", "gpt-oss-20b"))
if MODEL.backend != "cuda" or MODEL.model_mount is None or MODEL.model_volume is None:
    raise ValueError(f"{MODEL.name} is not a CUDA model")
model_volume = modal.Volume.from_name(
    MODEL.model_volume, version=MODEL.model_volume_version, create_if_missing=True
)
kernel_volume = (
    modal.Volume.from_name(
        MODEL.kernel_volume, version=MODEL.kernel_volume_version, create_if_missing=True
    )
    if MODEL.kernel_volume
    else None
)
volumes: dict[str | PurePosixPath, modal.Volume] = {MODEL.model_mount: model_volume}
if MODEL.kernel_mount is not None and kernel_volume is not None:
    volumes[MODEL.kernel_mount] = kernel_volume
image = build_modal_image(
    ROOT, env={**image_environment(MODEL), "RESIN_MODEL": MODEL.name}
).add_local_file(
    ROOT / "deploy/modal_serve.py", "/root/deploy/modal_serve.py", copy=True
)
app = modal.App(MODEL.app_name or f"resin-{MODEL.label}", image=image, volumes=volumes)
_http_server_processes: list[object] = []


@app.function(cpu=16, memory=32 * 1024, timeout=24 * 60 * 60)
def download(metadata_only: bool = False) -> dict:
    model_volume.reload()
    report = download_checkpoint(MODEL, metadata_only=metadata_only)
    model_volume.commit()
    return report


def _warmup(engine, recipe, spec, rank: int) -> None:
    import numpy as np

    from resin import SamplingParams
    from resin.recipe import Image, Message, Role, Text

    # Exercise several decode steps, rather than ending after a one-word reply.
    content = [Text("Count from 1 to 100.")]
    for image_input in (False, True) if spec.image_warmup else (False,):
        lease = None
        # This function runs only on attention replica drivers. Warm each
        # replica's image tower; round-robin or sticky HTTP routing may use any
        # of them immediately after startup.
        if rank == 0 or image_input:
            parts = content
            if image_input:
                axis = np.linspace(0, 1, 96, dtype=np.float32)
                horizontal = np.broadcast_to(axis[None], (96, 96))
                pixels = np.stack(
                    (horizontal, horizontal.T, 1 - horizontal), axis=-1
                ).copy()
                parts = [Image(source=pixels), Text("Name one visible color.")]
            prompt = recipe.render([Message(Role.USER, parts)])
            inputs = (
                {"media": prompt.media}
                if image_input
                else {"token_ids": prompt.token_ids}
            )
            lease = engine.submit(
                **inputs,
                sampling_params=SamplingParams(
                    max_output_tokens=64,
                    stop_token_ids=recipe.stop_token_ids,
                    temperature=0.0,
                ),
            )
        while engine.has_work():
            engine.step()
        if lease is not None:
            lease.take_result()


def _serve_rank(rank: int, port: int, model: str) -> None:
    import torch
    import torch.distributed as dist

    from resin import Engine, Parallelism, SchedulerConfig
    from resin.backend.distributed import DistributedEngine, DistributedWorker
    from resin.backend.topology import ParallelGroups
    from resin.server import HttpServer

    spec = get_model(model)
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    prepare_runtime(spec)
    topology = Parallelism(
        world_size=spec.world_size,
        attn_dp=spec.attn_dp,
        attn_tp=spec.attn_tp,
        ep=spec.ep,
    )
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
    runner = create_runner(
        spec,
        device=f"cuda:{rank}",
        groups=groups,
        context_length=spec.context_length + (spec.family == "gpt_oss"),
        max_batch_size=spec.max_batch_size,
        prefill_chunk_size=spec.prefill_chunk_size,
        draft_tokens=spec.draft_tokens,
        fast_allreduce=spec.family == "gpt_oss",
    )
    runner.capture_cuda_graphs()
    worker = DistributedWorker(runner, groups) if groups is not None else None
    if worker is not None and groups is not None and not groups.is_driver:
        worker.serve()
        os._exit(0)
    recipe = create_recipe(spec, runner=runner)
    # Respect the model's fixed prefill workspace; DeepSeek explicitly supports
    # a chunk per sequence, while dense/recurrent runners cap total input rows.
    batch_tokens = getattr(
        runner.runner_config, "max_batch_tokens", spec.prefill_chunk_size
    )
    options: dict[str, Any] = dict(
        max_running_requests=spec.max_batch_size,
        context_length=spec.context_length,
        prefix_cache=True,
        enable_overlap=True,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=batch_tokens, prefill_chunk_size=spec.prefill_chunk_size
        ),
    )
    engine = (
        Engine(runner, **options)
        if worker is None
        else DistributedEngine(worker, **options)
    )
    try:
        _warmup(engine, recipe, spec, rank)
        torch.cuda.synchronize(runner.device)
        if rank == 0 and kernel_volume is not None:
            kernel_volume.commit()
        print(f"[serve] {spec.name} rank {rank} ready", flush=True)
        HttpServer(
            engine,
            recipe=recipe,
            model=spec.repo_id,
            enable_anthropic=spec.enable_anthropic,
        ).serve(host="0.0.0.0", port=8000)
    finally:
        if not engine.closed:
            engine.close()
    if groups is not None:
        os._exit(0)


def _monitor_process_context(process_context: Any, label: str) -> None:
    """Report a spawned worker failure and terminate its hosting process."""

    def monitor() -> None:
        try:
            # A serving worker returning normally is still a group failure.
            # Resin's join waits for every worker; observe the first exit before
            # asking either Resin's or Torch's context to surface its error.
            wait([process.sentinel for process in process_context.processes])
            process_context.join(timeout=0)
        except BaseException:
            print(f"[{label}] worker process failed:", flush=True)
            traceback.print_exc()
        else:
            print(f"[{label}] worker process exited unexpectedly", flush=True)
        try:
            close = getattr(process_context, "close", None)
            if close is not None:
                close()
            else:
                # Torch contexts do not expose close(). Reap their peers before
                # exiting the host rather than relying on parent-death signals.
                processes = process_context.processes
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                deadline = time.monotonic() + 1.0
                for process in processes:
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
                for process in processes:
                    if process.is_alive():
                        process.kill()
                for process in processes:
                    process.join()
        finally:
            os._exit(1)

    threading.Thread(
        target=monitor,
        name=f"{label}-process-monitor",
        daemon=True,
    ).start()


@app.function(
    gpu=f"{MODEL.gpu}:{MODEL.world_size}",
    cpu=32 if MODEL.world_size > 1 else 16,
    memory=MODEL.memory,
    timeout=24 * 60 * 60,
    startup_timeout=60 * 60,
    scaledown_window=15 * 60,
    max_containers=1,
)
@modal.concurrent(
    max_inputs=MODEL.max_batch_size * MODEL.attn_dp,
    target_inputs=MODEL.max_batch_size * MODEL.attn_dp,
)
@modal.web_server(
    8000, startup_timeout=60 * 60, label=MODEL.label, requires_proxy_auth=False
)
def serve() -> None:
    import resin

    model_volume.reload()
    require_checkpoint(MODEL, draft_tokens=MODEL.draft_tokens)
    if kernel_volume is not None:
        kernel_volume.reload()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes = resin.process.spawn(
        _serve_rank,
        args=(port, MODEL.name),
        devices=[f"cuda:{rank}" for rank in range(MODEL.world_size)],
        join=False,
    )
    _http_server_processes.append(processes)
    _monitor_process_context(processes, f"{MODEL.name}-http")


@app.local_entrypoint()
def main(publish_only: bool = False) -> None:
    print(json.dumps(download.remote(publish_only), indent=2))
