# Scripts and deployment modules

Run commands from the repository root. The shared entry points are
`scripts/bench.py` for Engine performance, `scripts/check_server.py` for endpoint
qualification, and `deploy/modal_serve.py` for Modal serving. Model-specific
configuration belongs in `deploy/models.py`.

## Scripts

| File | Purpose |
| --- | --- |
| [bench.py](../scripts/bench.py) | Construct a registered runner locally or on Modal; sweep prefill contexts and decode batches through `resin.bench`; save JSONL, plots and Perfetto traces. Also download or publish checkpoint inventories. |
| [plot_bench.py](../scripts/plot_bench.py) | Plot prefill latency versus input throughput/GPU and decode interactivity versus output throughput/GPU. Actual committed output is the default; acceptance projections are explicit. |
| [merge_bench_traces.py](../scripts/merge_bench_traces.py) | Merge per-rank prefill/decode traces, align timestamp bases and isolate process/flow IDs for Perfetto. |
| [check_server.py](../scripts/check_server.py) | Qualify an existing HTTP endpoint: JSON/SSE, logprobs, concurrency, prefix hits, context/TTFT, optional media/reasoning, MMLU subset and stress/recovery. Writes a pass/fail/skip report. |
| [stress_http.py](../scripts/stress_http.py) | Standalone bounded HTTP load/protocol/disconnect/recovery checks, plus the reusable client/parser/reporting machinery used by `check_server.py`. |
| [bench_startup.py](../scripts/bench_startup.py) | Measure cold `resin` and Torch backend imports in separate fresh processes. This measures Python startup, not model loading or inference. |
| [lfm2_chat.py](../scripts/lfm2_chat.py) | Minimal CPU example: render a fixed LFM2 conversation with the native recipe and stream one Engine completion. |
| [lfm2_serve.py](../scripts/lfm2_serve.py) | Minimal CPU example: serve the Torch LFM2 runner through the native HTTP server on localhost. |
| [serve.py](../scripts/serve.py) | Serve a local registered preset (currently the Apple Silicon MLX presets) through the native HTTP server with the prefix cache enabled. |
| [rust_build_and_types.py](../scripts/rust_build_and_types.py) | Development build: sync type-library sources, build the Rust extension with generated stubs, normalize stubs and format them. |
| [sync_type_libraries.py](../scripts/sync_type_libraries.py) | Make locked Linux dependency sources available to `ty` on the development host without importing those libraries or running their installation hooks. |
| [normalize_stubs.py](../scripts/normalize_stubs.py) | Add property types, protocol bases and imports missing from generated native bindings. Called by the build script. |
| [cloc.py](../scripts/cloc.py) | Count production Rust/Python code using Tokei, excluding tests and grouping the report by package. |

## Deployment modules

| File | Purpose |
| --- | --- |
| [models.py](../deploy/models.py) | Shared checkpoint revisions/inventories, GPU/topology defaults, runner and recipe factories, and model-specific runtime preparation. |
| [modal_serve.py](../deploy/modal_serve.py) | One parameterized serving app selected by `RESIN_MODEL`; checkpoint publication, worker startup/warmup and native HTTP serving. Each model selection has its own endpoint. |
| [modal_bench.py](../deploy/modal_bench.py) | Thin remote adapter for `scripts/bench.py`, including CPU checkpoint actions and GPU sweep execution. |
| [modal_image.py](../deploy/modal_image.py) | Build the shared CUDA image and its pinned native/kernel dependencies. |
| [checkpoint_cache.py](../deploy/checkpoint_cache.py) | Validate checkpoint file inventories and atomically publish/require completion markers. |
| [jit_cache.py](../deploy/jit_cache.py) | Configure persistent compiler namespaces and content-validated FlashInfer artifact loading. |
| [__init__.py](../deploy/__init__.py) | Package marker for shared deployment imports. |

The artifact checker, one-off kernel/parallelism probes, Modal validation wrappers
and per-model deployment files have been retired. Reusable numerical tests remain
in `tests/` and `models/cuda/tests/`; performance measurement lives in `resin.bench`,
with command-line orchestration in `scripts/`. Historical reports retain the
revisions and scope of their original measurements.

See [benchmarking and endpoint checks](benchmarks.md) for command examples and
[Modal serving](../deploy/README.md) for checkpoint and deployment commands.
