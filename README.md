# Resin

Resin is not an inference engine. It's a library of _building blocks_ for LLM
inference.

At its core is the headless `Engine`: a fast Rust scheduler, KV cache, and
metrics server. You then write the forward pass as **model-specific kernels,
optimized for your hardware.**

```text
token IDs ──► Rust Engine ──► frozen BatchPlan ──► Python Runner ───► token IDs
                  ▲                                    │
                  └──────── ownership & commit ────────┘
```

Resin also has optional HTTP serving and Prometheus metrics.

> [!WARNING]
> This is not production-quality software. If you're looking for an actual
> inference engine, please use SGLang or vLLM. Read on if you want to hack on
> LLM inference systems, new models or hardware.

With the right kernels, this makes it possible to autonomously "vibe code" even
the most complex models. In 12 hours, we used Resin to build standalone engines
for [DeepSeek-V4.1-Flash](models/cuda/resin_models_cuda/deepseek_v41/) and
[GLM-5.3-Flash](models/cuda/resin_models_cuda/glm5_3_flash/) on Blackwell GPUs
that rival vLLM and SGLang.

<center>
<table>
<thead>
<tr>
<th>Prefill</th>
<th>Decode</th>
</tr>
</thead>
<tbody>
<tr>
<td><img width="100%" alt="DeepSeek V4.1 Flash prefill" src="https://i.imgur.com/wFCTaMD.png"></td>
<td><img width="100%" alt="DeepSeek V4.1 Flash interactivity-throughput curve" src="https://i.imgur.com/yVw8HCy.png"></td>
</tr>
</tbody>
</table>
<i>Prefill and decode for DeepSeek V4.1 Flash, within 5-10% of DeepInfer.</i>
</center>

## Features

Start with the two-page [architecture overview](docs/architecture.md).

- [Rust batch scheduler](docs/kernel-boundary.md)
  - [Unified slab-based cache pool (KV, SWA, Recurrent)](docs/cache-pool.md)
  - [Speculative decoding (MTP, DSpark)](docs/model-forward-pass.md)
  - [Zero-overhead overlap scheduling](docs/scheduler.md)
  - [Engine metrics and Prometheus export](docs/metrics.md)
- [Parallelisms: TP, EP, CP, and DP Attention](docs/distributed.md)
- [Prefill and decode CUDA graphs](docs/cuda-graphs.md)
- [Native renderer and incremental parser](crates/resin-recipe/README.md)
  - [Recipe differential fuzzing](tests/recipe_fuzz/README.md)
- [NUMA pinning on worker startup](docs/numa.md)
- [Shared kernel cache](docs/kernel-cache.md)

## Getting started

Either fork the repository or import `resin` and `resin_models_cuda` as standalone
packages. Then use the following skills:

- [**add-resin-model**](skills/add-resin-model/SKILL.md) -
  Create an inference engine for a new model from scratch, using Resin.
- [**optimize-resin-performance**](skills/optimize-resin-performance/SKILL.md) -
  Iterate on your inference engine to improve performance with an "autoresearch"
  workflow, diagnose bottlenecks and validate improvements, drawing on past
  learnings.

Importing `resin` does not import PyTorch or initialize CUDA.

Model runners are not tied to CUDA. `resin_models_mlx` runs
[Qwen3.5-MoE models such as Ornith](models/mlx/resin_models_mlx/qwen3_5_moe/README.md)
on Apple Silicon with MLX, on the same Rust Engine:

```bash
uv run scripts/serve.py --model ornith-1.0-35b --checkpoint ~/models/Ornith-1.0-35B
```

## Installation

Install the core library directly from GitHub with Python 3.11+ and Rust installed:

```bash
pip install "git+https://github.com/ekzhang/resin.git"
```

Pip builds the Rust extension locally. The import remains `resin`.
Use `@<commit>` after `.git` to pin a revision.

Model runners are separate packages: [`resin-models-cuda`](models/cuda/pyproject.toml)
for optimized CUDA implementations and
[`resin-models-torch`](models/torch/pyproject.toml) for reference implementations.
The repository's development workspace installs both; the examples below assume
that workspace or a separately installed runner package.

## Development

Resin requires Python 3.11+ and Rust.

```bash
uv run scripts/rust_build_and_types.py
```

`uv run` syncs the project dependencies automatically. The build command then
syncs type-library sources and builds the Rust extension with generated stubs.

The Modal harnesses are launched from the repository root. Install the
development dependencies first, then authenticate with Modal once if this is a
new machine:

```bash
uv run modal setup
```

`modal setup` is interactive and only needs to be repeated when the local
Modal credentials change.

All models share one benchmark CLI and one parameterized serving app. The model
registry in [`deploy/models.py`](deploy/models.py) supplies checkpoint revisions,
hardware, parallelism, and model-specific runner construction.

```bash
# Populate the pinned checkpoint without allocating a GPU.
uv run scripts/bench.py --backend modal --model gpt-oss-20b --download-only

# Sweep prefill and decode, save traces, and plot the measured results.
uv run scripts/bench.py --backend modal --model gpt-oss-20b \
  --context-lengths 4096,8192 --batch-sizes 1,2,4,8 \
  --output bench/gpt-oss-20b.jsonl --trace-dir bench/traces --plot
```

For local GPUs, use `--backend local --checkpoint /path/to/checkpoint`; models
with a separate draft checkpoint also accept `--assistant-checkpoint`. The
local backend is the default. Modal submits the repository's source into the
shared CUDA image, so the client machine does not need CUDA installed.

Prefill charts compare complete-context latency with input tokens/s/GPU; decode
charts compare per-user output tokens/s with output tokens/s/GPU. Natural draft
acceptance and model routing are the defaults. `--accepted-drafts K` and
`--balanced-experts` explicitly enable synthetic workloads. See
[benchmarking](docs/benchmarks.md) for the timing protocol and trace scope.

For an already running endpoint, `scripts/check_server.py URL --output REPORT`
checks serving correctness, prefix-cache reuse, context-size TTFT, and optional
image/reasoning/stress cases. It can also evaluate a small local MMLU CSV or
JSONL subset with `--eval-file` (defaults: `--eval-limit 100`,
`--eval-concurrency 32`); see
[server checks](docs/benchmarks.md#http-correctness-and-latency-checks).

## Minimal use

```python
from resin import Engine, SamplingParams
from resin_models_torch.lfm2 import Lfm2TorchRunner

runner = Lfm2TorchRunner(device="cpu")

with Engine(runner, context_length=32768) as engine:
    request = engine.submit(
        [1, 6, 24131],
        sampling_params=SamplingParams(
            max_output_tokens=8,
            stop_token_ids=[runner.eos_token_id],
            temperature=1.0,
            top_p=0.95,
        ),
    )
    while engine.has_work():
        for event in engine.step():
            print(event)
    generated_ids = request.take_result()
```

`SamplingParams` is the frozen request policy for generation limits, stop token
IDs, sampling, and logprob options. Use the recipe's `stop_token_ids`
for chat requests, or supply the model's stop tokens explicitly for raw token
requests. `Engine` has no default EOS token.

A custom synchronous runner only needs one method:

```python
import numpy as np

from resin import BatchPlan, ForwardOutput


class Runner:
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        return ForwardOutput(
            token_ids=np.zeros(plan.num_sequences, dtype=np.uint32),
            token_indptr=np.arange(plan.num_sequences + 1, dtype=np.uint32),
        )
```

`ForwardOutput` carries flat token IDs and row offsets, with optional logprob
columns. The same result type covers ordinary and speculative decoding.

Managed history and recurrent state are cache groups declared through a frozen
`RunnerConfig`. The Engine binds their one physical arena during construction.
Pass `prefix_cache=True` to reuse aligned prefixes across requests.

## Local HTTP server

The built-in native server exposes one Engine through OpenAI Chat Completions and
Responses, Anthropic Messages, and a small streaming chat page. Model-specific
prompt rendering and incremental response parsing live in the independent
`resin-recipe` Rust crate. Native model recipes implement the same `Recipe`
protocol as application-provided recipes. Stock image/video preprocessing also
runs in Rust; the model runner performs the GPU vision encoder.

Native recipes reuse preparation for repeated immutable encoded images, with an
LRU limited to 8 image sets and 128 MiB per recipe. CUDA vision runners separately
retain up to 64 encoded feature sets and 256 MiB per runner. Both caches use
content identities; prompt positions are rebuilt for each request. Mutable image
arrays and video inputs bypass the CPU preparation cache. Prepared native media
exposes read-only snapshots, so pixels and geometry cannot diverge from their
cache identity; render a new payload when changing an input image.

The bridge bounds queued jobs with `HttpServer(request_queue_capacity=1024)`.
This shared intake queue absorbs bursts while requests await engine admission;
it does not reserve GPU cache or change the engine's active batch size.
`max_active_requests` bounds admitted requests, including prompts being rendered,
and defaults to `engine.max_running_requests + render_workers`. Set it explicitly
to control how many additional requests can wait inside the Engine. Local prompt
rendering defaults to the CPU parallelism reported by
`std::thread::available_parallelism()` (falling back to one thread), capped at 16,
or an explicit `render_workers` count. `admissions_per_step=32` limits admissions
between inference steps. For distributed engines, the total active
and admission limits scale with the number of attention-DP replicas; rendering
runs on the replica drivers. Buffered events per client remain bounded at 64.
A full intake queue returns HTTP 429; slow or disconnected clients are cancelled
without blocking other requests. Actor failure stops the listener, and
`HttpServer.shutdown()` requests shutdown after the current model step.

```bash
# Run one LFM2 prompt in the terminal.
uv run python scripts/lfm2_chat.py

# Serve LFM2 locally over HTTP.
uv run python scripts/lfm2_serve.py
```

Then open <http://127.0.0.1:8000>. The page reports prompt and generated token
counts, elapsed generation time, and output tokens per second. The same process
serves `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and `/v1/models`.
Both examples explicitly use the CPU Torch runner.

## Modal HTTP endpoints

[`deploy/modal_serve.py`](deploy/modal_serve.py) serves every registered model.
Select `RESIN_MODEL` at launch; each selection has a separate app and endpoint,
with shared download, warmup, process management, and HTTP startup code.
Populate its pinned checkpoint before deploying:

```bash
RESIN_MODEL=gpt-oss-20b uv run modal run deploy/modal_serve.py
RESIN_MODEL=gpt-oss-20b uv run modal deploy deploy/modal_serve.py

RESIN_MODEL=deepseek-v41 uv run modal run deploy/modal_serve.py
RESIN_MODEL=deepseek-v41 uv run modal deploy deploy/modal_serve.py
```

Use `uv run scripts/bench.py --help` to see the current model keys accepted by
`RESIN_MODEL` and the benchmark CLI. The shared
[`MODEL_SPECS` registry](deploy/models.py) defines each selection's checkpoint,
hardware, and serving limits. Add `--publish-only` to the download command to
adopt complete files already in the checkpoint volume without downloading their
contents. Serving and benchmarking use the same checkpoint inventory and
completion marker.

Multi-rank deployments use the shared distributed worker, Engine, and HTTP
coordinator. For example, DeepSeek uses attention DP and expert parallelism,
while GPT-OSS 120B uses attention and expert tensor parallelism. Requests are
routed round-robin across attention-DP replicas unless `routed_dp_rank` in the
JSON body or `resin-routed-dp-rank` in the header selects a replica. Single-rank
deployments use a local Engine through the same HTTP frontend. The built-in chat
page chooses one replica per page session and keeps later turns there so they can
reuse its prefix and image caches. API clients can select a stable rank likewise.

Each deployed URL serves `/`, `/health`, `/v1/models`,
`/v1/chat/completions`, and `/v1/responses`; model configuration controls
`/v1/messages` support. The functions
scale to zero after 15 idle minutes and are capped at one GPU container apiece.
The chat demo renders Markdown and TeX math and requests up to 16,384 output
tokens by default. Model loading, CUDA graph capture, and prompt warmup all
complete before the endpoint starts accepting traffic. Models with image warmup
enabled also warm the vision path on every attention replica.
Because `requires_proxy_auth=False`, anyone with a URL can submit inference
requests and incur GPU usage; remove the deployment when it is no longer
needed.

Select a native recipe by name and supply its tokenizer or multimodal processor.
Messages and content blocks are Rust-owned typed objects:

```python
import resin
from resin.recipe import get_native_recipe
from resin.server import HttpServer

recipe = get_native_recipe("gpt_oss", tokenizer)
prompt = recipe.render(
    [
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("Hello!")]),
    ]
)
parser = recipe.create_parser(prompt)

HttpServer(engine, recipe=recipe, model="my-model", stream_interval=4).serve()
```

## Current shape

The native layer currently provides:

- sticky continual batching with prefill priority and memory retraction;
- phase-homogeneous prefill or decode batches, token budgets, and prefill chunks;
- one flat physical cache pool shared by History, Window, and State groups;
- sparse, chronological `CacheGroupPlan` rows and exact append destinations;
- opt-in exact-token LRU prefix caching with immutable page sharing and
  sliding-window and recurrent-state copy-on-write;
- retryable pre-accept failure and fail-stop post-accept quarantine.

`crates/resin-core` is Python-independent Rust containing request
lifecycle, scheduling, token-range planning, grouped cache allocation, and
execution transactions. It uses `rkyv` for the distributed
plan wire format. `crates/resin-py` converts Python inputs and adapts runner
callbacks, plans, errors, handles, and events to that core. `crates/resin-recipe`
owns conversation formats and CPU media preparation; `crates/resin-http` owns
the serving protocols and streaming responses.

History cache grows on demand to each frozen plan. Window cache grows toward
bounded cyclic storage whose plan row is projected into chronological order.
With prefix caching enabled, a block trie indexes exact token prefixes, cached
pages become immutable readers, and Window pages are copied on write when a
request advances its ring. State groups retain a two-checkpoint ring. At an
aligned prefix hit, the input checkpoint stays immutable while the next page is
writable; the runner copies only when input and output pages differ. LRU entries
use spare capacity in the same pool and are evicted when live requests need it.

## Public surface

The top-level `resin` package exposes the Engine, immutable plans and request
policies, runner protocols, cache layouts, and metrics types. See
[`resin/__init__.py`](resin/__init__.py) for the current exports and
[`resin/protocols.py`](resin/protocols.py) for the runner contracts.

Distributed coordination lives in `resin.backend.distributed`, with explicit
process groups in `resin.backend.topology`. Concrete runners are imports from
separately installed architecture submodules such as
`resin_models_cuda.deepseek_v41`, `resin_models_cuda.gpt_oss`, and
`resin_models_torch.lfm2`.

## Checks

```bash
cargo test --all-targets
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run python scripts/cloc.py
```

GPU validation is model- and machine-specific; local tests use fakes or skip
unavailable accelerator libraries.

The code-size report uses Tokei to count production Rust/Python by component,
excluding comments, blank lines, tests, deployment tools, and generated stubs.
Use the current report instead of a checked-in line-count snapshot.

For resident prefill/decode sweeps, optional fixed speculative acceptance and
balanced routing, Perfetto traces, and throughput/latency charts, see
[benchmarking](docs/benchmarks.md).
`Engine` collects native request/cache metrics and optional runner observations
by default (`enable_metrics=True`). Use `enable_metrics=False` to opt out.
The HTTP server publishes `/metrics`; see [metrics](docs/metrics.md)
for Prometheus queries and forwarding through an OTLP Collector.

## Repository map

- `crates/resin-core/src/engine.rs` — request lifecycle and execution transactions
- `crates/resin-py/src/lib.rs` — Python objects and the native Engine adapter
- `crates/resin-py/src/planning.rs` — conversion from core batch/resource views to `BatchPlan`
- `crates/resin-py/src/runner.rs` — execution discovery and one-time resource binding
- `crates/resin-py/src/cache.rs` — Python adapters for core KV descriptors
- `crates/resin-core/src/scheduler.rs` — cohort selection and queue-age policy
- `crates/resin-core/src/planning.rs` — typed token-range planning
- `crates/resin-core/src/resources/` — grouped cache allocation, prefix reuse, and leases
- `crates/resin-recipe/` — native conversation formats, parsing, and CPU media preparation
- `crates/resin-http/` — HTTP protocols, streaming, and serving lifecycle
- `resin/backend/` — reusable execution and resource gadgets
- `models/` — separately installable model definitions, loaders, and runners
- `scripts/` — examples, benchmarks, and local maintenance tools
- `deploy/` — shared model registry, checkpoint/cache helpers, and parameterized Modal serving/benchmark adapters
- `tests/` — native and runner-contract coverage

## Acknowledgements

Resin was based on and inspired by systems from SGLang, TokenSpeed, and DeepInfer.
