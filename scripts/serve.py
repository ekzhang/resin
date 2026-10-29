"""Serve a local Apple Silicon (MLX) registry preset through Resin's HTTP APIs.

uv run scripts/serve.py --model ornith-1.0-35b --checkpoint ~/models/Ornith-1.0-35B

CUDA presets need graph capture, warmup and runtime preparation; serve them with
deploy/modal_serve.py.
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args(argv)

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
        draft_tokens=spec.draft_tokens,
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
    print(
        f"[serve] {spec.name} ready in {time.perf_counter() - started:.1f}s "
        f"at http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        HttpServer(
            engine,
            recipe=recipe,
            model=spec.repo_id,
            enable_anthropic=spec.enable_anthropic,
        ).serve(host=args.host, port=args.port)
    finally:
        if not engine.closed:
            engine.close()


if __name__ == "__main__":
    main()
