# CUDA model package boundaries

CUDA model families are deliberately small accelerator frontends. Backend-
neutral configuration, prompt recipes, prepared media, and scheduler lowering
live in `resin.models.<family>` so CUDA and AMD implementations can share them.
A CUDA family owns its runner, optimized model, physical cache geometry,
checkpoint materialization, draft implementation, and concrete kernels. All
of those files live together in `resin_models_cuda.<family>`; the package does
not split implementations into `runners`, `kernels`, `drafts`, `media`, or
`loading` directories. Importers use the owning module directly, without
compatibility aliases or implementation registries.

Resin itself owns backend-neutral model math and CUDA host mechanics:
graph capture/replay, fixed-buffer staging, sampling inputs, and slab page-
table addressing. Concrete CUDA, cuTile, TileLang, FlashInfer, and model-
specific fused kernels remain directly importable from
`resin_models_cuda.<family>`; they are never moved into `resin.backend.cuda` or
selected through a registry. A small cross-model CUDA primitive may live as a
root module when it has multiple model owners and does not justify another
package layer.

For example, `RMSNorm` lives in `resin.backend.layers` and selects Quack on
SM90+ or FlashInfer on older CUDA devices, with compiled PyTorch math for
non-CUDA tensors and FP32 inputs unsupported by FlashInfer. Gemma's targets
and assistants use this same layer. GPT-OSS YaRN, sink attention, routers, and
experts live in `resin_models_cuda.gpt_oss.model` and consume `GptOssConfig`
directly. Sharing a layer does not move a model family's composition or
configuration contract into the generic backend.

Every CUDA model runner derives from `resin.backend.cuda.runner.CudaRunner`.
That base owns device normalization, KV precision selection, failure
containment, sampling, Engine cache binding, graph modes, capture hooks, shape
selection, fallback accounting, replay, poisoning checks, mode-scoped graph
statistics while remaining unaware of model layers and physical cache layout.
Each model-owned cache arena decides which precision and heterogeneous state
layouts it can actually support.

`GptOssCudaRunner` owns GPT-OSS paging, FA4, router, sampling, and
tensor-parallel execution directly in the GPT-OSS model folder. Gemma 4,
Qwen3.5, Inkling, and DeepSeek V4.1 likewise derive directly from `CudaRunner`;
there is no intermediate runner named for paging or MoE unless a second model
actually shares its complete execution contract.

## Graph execution modes

A `CudaRunner` configures one internal `CudaGraphExecutor` with semantic
modes such as `prefill` and `decode`. Each `CudaGraphMode` supplies only its
shape buckets, fixed-input factory, and program builder. The builder returns
either one `CudaGraphStep` (a full graph) or a `CudaGraphProgram` containing
captured and eager regions (a breakable graph). Capture, staging, replay,
poisoning, validation, and metrics are shared by every model, every mode, and
both graph styles.

Phase code must keep the model operation separate from graph orchestration.
Eager execution and graph builders call the same model/attention helpers; a
graph region is a capture boundary around those helpers, not a second
implementation of the phase. Consequently, adding breakable prefill means
registering a `prefill` mode and choosing its region boundaries. It must not
add a second graph runner, duplicate the model forward pass, or fork cache and
sampling semantics. Decode can independently remain full-graph or become
breakable in the same runner.

## Deployment configuration

`deploy/models.py` is a small operational registry for pinned checkpoints,
hardware defaults, and runner/recipe construction. `deploy/modal_serve.py` and
`scripts/bench.py` use it to share startup and download behavior across models.
This registry does not dispatch model kernels or change the direct model-module
ownership described above. Numerical contracts belong in the normal test suite;
one-off Modal operator and diagnostic scripts have been retired.
