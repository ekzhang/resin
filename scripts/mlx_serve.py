"""Serve a local Apple Silicon (MLX) registry preset through Resin's HTTP APIs.

Pass the snapshot path returned by ``hf download`` through ``--checkpoint``.
The model README includes commands using Hugging Face's normal cache.

CUDA presets need graph capture, warmup and runtime preparation; serve them with
deploy/modal_serve.py.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

from deploy.models import MODEL_SPECS, create_recipe, create_runner, get_model


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    local = [name for name, spec in MODEL_SPECS.items() if spec.backend == "mlx"]
    parser.add_argument("--model", choices=local, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument(
        "--draft-tokens",
        type=int,
        help="MTP draft tokens per decode step (default: the preset's; 0 disables)",
    )
    parser.add_argument(
        "--wired-limit-gib",
        type=float,
        help="MLX wired budget in GiB (default: live memory + 1 GiB, capped at Metal's recommended limit; 0 disables)",
    )
    args = parser.parse_args(argv)
    if args.wired_limit_gib is not None and (
        not math.isfinite(args.wired_limit_gib) or args.wired_limit_gib < 0
    ):
        parser.error("--wired-limit-gib must be finite and nonnegative")

    os.environ.setdefault("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", "500")
    import mlx.core as mx

    from resin import Engine, SchedulerConfig
    from resin.server import HttpServer

    spec = get_model(args.model)
    context_length = args.context_length or spec.context_length
    max_batch_size = args.max_batch_size or spec.max_batch_size
    started = time.perf_counter()
    runner = create_runner(
        spec,
        args.checkpoint.expanduser(),
        device="mps",
        context_length=context_length,
        max_batch_size=max_batch_size,
        prefill_chunk_size=spec.prefill_chunk_size,
        draft_tokens=spec.draft_tokens
        if args.draft_tokens is None
        else args.draft_tokens,
    )
    recipe = create_recipe(spec, args.checkpoint.expanduser(), runner=runner)
    batch_tokens = getattr(
        runner.runner_config, "max_batch_tokens", spec.prefill_chunk_size
    )
    engine = Engine(
        runner,
        max_running_requests=max_batch_size,
        context_length=context_length,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=batch_tokens, prefill_chunk_size=spec.prefill_chunk_size
        ),
    )
    previous_wired_limit = None
    try:
        mx.synchronize()
        mx.clear_cache()
        active = mx.get_active_memory()
        if args.wired_limit_gib is not None:
            wired_limit = int(args.wired_limit_gib * (1 << 30))
        else:
            wired_limit = min(
                active + (1 << 30),
                int(mx.device_info()["max_recommended_working_set_size"]),
            )
        previous_wired_limit = mx.set_wired_limit(wired_limit)
        print(
            f"[serve] wired budget {wired_limit / (1 << 30):.2f} GiB; "
            f"MLX active {active / (1 << 30):.2f} GiB; "
            f"residency refresh {os.environ['MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS']} ms",
            flush=True,
        )
        print(
            f"[serve] {spec.name} ready in {time.perf_counter() - started:.1f}s "
            f"at http://{args.host}:{args.port}",
            flush=True,
        )
        HttpServer(
            engine,
            recipe=recipe,
            model=spec.repo_id,
            enable_anthropic=spec.enable_anthropic,
        ).serve(host=args.host, port=args.port)
    finally:
        try:
            if not engine.closed:
                engine.close()
        finally:
            if previous_wired_limit is not None:
                mx.set_wired_limit(previous_wired_limit)


if __name__ == "__main__":
    main()
