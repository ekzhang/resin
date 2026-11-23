# Modal serving and benchmarks

`models.py` is the shared model registry. It owns pinned target/draft
checkpoints, hardware and parallelism defaults, and model-specific runner/recipe
construction. The shared launchers handle downloads, startup, and result storage.

## Serve a model

Select `RESIN_MODEL` when invoking the single serving app:

```bash
RESIN_MODEL=deepseek-v41 uv run modal run deploy/modal_serve.py
RESIN_MODEL=deepseek-v41 uv run modal deploy deploy/modal_serve.py
```

The first command downloads and publishes the checkpoint on a CPU worker; the
second publishes the endpoint. Use `--publish-only` on the first command to
publish completion markers for complete existing files. Every model uses the
same checkpoint validation contract for serving and benchmarking.

Run `uv run scripts/bench.py --help` to see the current model keys. Serving and
benchmarking both read [`MODEL_SPECS`](models.py), which defines each key's pinned
checkpoint, hardware, parallelism, and serving limits. Each selection gets its
own Modal app (`resin-<preset>`) and endpoint label (`<preset>`).

## Benchmark a model

Run the common CLI from the repository root:

```bash
uv run scripts/bench.py --model deepseek-v41 --backend modal --download-only
uv run scripts/bench.py --model deepseek-v41 --backend modal \
  --context-lengths 4096,8192 --batch-sizes 1,2,4,8 \
  --output bench/deepseek-v41.jsonl --trace-dir bench/traces --plot
```

`modal_bench.py` is the remote adapter used by this CLI. For a local CUDA host,
use `--backend local --checkpoint PATH`, plus `--assistant-checkpoint PATH` when
the model needs separate draft weights. `resin.bench` owns measurement and
untimed prefill/decode tracing; `scripts/plot_bench.py` plots the JSONL records.
See [the benchmark protocol](../docs/benchmarks.md) for workload options,
normalization, and interpreting traces.

The former one-off Modal validation/probe launchers have been retired. Runtime
correctness tests remain under `tests/` and `models/cuda/tests/`; historical
measurement artifacts retain their original source revisions and scope.
