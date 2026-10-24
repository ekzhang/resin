"""Public DeepSeek V4.1 Flash endpoint: four B300s, EP4 and attention DP4.

Cache weights with deploy/modal_bench.py's download command, then:
    uv run modal deploy deploy/modal_deepseek_v41.py
"""

from __future__ import annotations

import os
import socket
import time
import warnings
from datetime import timedelta
from pathlib import Path

import modal

import resin
from deploy.checkpoint_cache import checkpoint_is_complete
from deploy.modal_image import build_modal_image
from resin.recipe import DeepSeekV41Recipe

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
CHECKPOINT = Path("/models/deepseek/flash-1017") / REVISION
WORLD_SIZE = 4
MAX_SEQUENCES = 32  # Per attention-DP replica; 128 requests across the engine.
CONTEXT_LENGTH = 1_048_576
PREFILL_CHUNK_SIZE = 8192

model_volume = modal.Volume.from_name("resin-deepseek-v4-models-v1", version=1)
kernel_volume = modal.Volume.from_name("resin-deepseek-v4-kernels-v2", version=1)
image = build_modal_image(
    ROOT,
    env={
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "DG_JIT_CACHE_DIR": "/kernel-cache/.cache/deep_gemm",
        "TILELANG_CACHE_DIR": "/kernel-cache/.cache/tilelang",
        "CUDA_TILE_CACHE_DIR": "/kernel-cache/.cache/cutile-python",
        "TRITON_CACHE_DIR": "/kernel-cache/.cache/triton",
        "FLASHINFER_WORKSPACE_BASE": "/tmp/resin-flashinfer-build",
    },
).add_local_file(
    ROOT / "deploy/modal_deepseek_v41.py",
    "/root/deploy/modal_deepseek_v41.py",
    copy=True,
)
app = modal.App(
    "resin-deepseek-v41",
    image=image,
    volumes={"/models/deepseek": model_volume, "/kernel-cache/.cache": kernel_volume},
)


def _serve_rank(rank: int, port: int) -> None:
    # Keep compatibility with older Torch releases without logging its renamed
    # collective aliases on every serving-thread callback in Torch 2.13.
    warnings.filterwarnings(
        "ignore",
        message=r"`torch.distributed.(all_gather_into_tensor|reduce_scatter_tensor)` is deprecated.*",
        category=FutureWarning,
    )
    import torch
    import torch.distributed as dist
    from tokenizers import Tokenizer

    from resin import Parallelism, SamplingParams, SchedulerConfig
    from resin.backend.distributed import DistributedEngine, DistributedWorker
    from resin.backend.topology import ParallelGroups
    from resin.server import HttpServer
    from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner

    torch.cuda.set_device(rank)
    # Leave CPU capacity for HTTP, tokenization, and the other model processes.
    torch.set_num_threads(4)
    topology = Parallelism(world_size=WORLD_SIZE, attn_dp=WORLD_SIZE, ep=WORLD_SIZE)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=topology.world_size,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(minutes=10),
    )
    groups = ParallelGroups.create(topology)
    started = time.perf_counter()
    print(f"[deepseek-http rank={rank} pid={os.getpid()}] loading weights", flush=True)
    runner = DeepSeekV41CudaRunner(
        CHECKPOINT,
        device=f"cuda:{rank}",
        parallel_groups=groups,
        max_seq_len=CONTEXT_LENGTH,
        max_batch_size=MAX_SEQUENCES,
        prefill_chunk_size=PREFILL_CHUNK_SIZE,
        cuda_graph_prefill="breakable",
        cuda_graph_decode="full",
        speculative_tokens=3,
        expert_backend="deepgemm",
        fast_allreduce=False,
        gpu_memory_utilization=0.85,
    )
    print(
        f"[deepseek-http rank={rank}] weights ready in "
        f"{time.perf_counter() - started:.1f}s; capturing graphs",
        flush=True,
    )
    runner.capture_cuda_graphs()
    codec = DeepSeekV41Recipe(Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json")))
    engine = DistributedEngine(
        DistributedWorker(runner, groups),
        max_running_requests=MAX_SEQUENCES,
        context_length=CONTEXT_LENGTH,
        prefix_cache=True,
        enable_overlap=True,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=MAX_SEQUENCES * PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
        ),
    )
    try:
        # Exercise the web-chat path: one active replica, idle EP peers, and
        # ordinary EOS handling while device-feedback decode overlaps.
        warmup = (
            engine.submit(
                codec.render_ids(
                    [
                        resin.recipe.Message(
                            resin.recipe.Role.USER,
                            [resin.recipe.Text("Count from 1 to 100.")],
                        )
                    ]
                ),
                sampling_params=SamplingParams(
                    max_output_tokens=64,
                    stop_token_ids=codec.get_stop_token_ids(),
                    temperature=0.0,
                ),
            )
            if rank == 0
            else None
        )
        while engine.has_work():
            engine.step()
        if warmup is not None:
            warmup.take_result()
            if engine.local.overlap_stats[1] == 0:
                raise RuntimeError("single-replica warmup did not overlap decode")
        torch.cuda.synchronize()
        captures = {
            mode: stats["capture_count"]
            for mode, stats in runner.cuda_graph_stats.items()
        }
        print(
            f"[deepseek-http rank={rank}] ready after "
            f"{time.perf_counter() - started:.1f}s; "
            f"captured graphs={captures}; overlap={engine.local.overlap_stats}",
            flush=True,
        )
        if rank == 0:
            kernel_volume.commit()
        # All four drivers participate; the shared coordinator listens on rank 0.
        HttpServer(engine, recipe=codec, model=MODEL).serve(host="0.0.0.0", port=8000)
    finally:
        if not engine.closed:
            engine.close()


_http_server_processes: list[object] = []


@app.function(
    gpu="B300:4",
    cpu=32,
    memory=256 * 1024,
    timeout=24 * 60 * 60,
    startup_timeout=60 * 60,
    scaledown_window=15 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=256, target_inputs=256)
@modal.web_server(
    8000, startup_timeout=60 * 60, label="deepseek-v41", requires_proxy_auth=False
)
def serve_deepseek_v41() -> None:
    import torch.multiprocessing

    from resin.backend.parallel import monitor_process_context

    model_volume.reload()
    if not checkpoint_is_complete(CHECKPOINT, MODEL, REVISION):
        raise RuntimeError("Download the pinned deepseek-v41 checkpoint first")
    kernel_volume.reload()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes = torch.multiprocessing.spawn(
        _serve_rank, args=(port,), nprocs=WORLD_SIZE, join=False
    )
    _http_server_processes.append(processes)
    monitor_process_context(processes, "deepseek-v41-http")
