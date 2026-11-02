# Qwen3.5 on Apple Silicon

`Qwen35MlxRunner` serves dense and MoE Qwen3.5 checkpoints through Resin's
Engine on one Apple Silicon GPU. It supports tied embeddings, paged attention,
chunked prefill, prefix reuse, overlapping decode and image input. Speculative
and MTP decoding are not supported.

## Run locally

Use the prequantized
[`mlx-community/Qwen3.5-0.8B-8bit`](https://huggingface.co/mlx-community/Qwen3.5-0.8B-8bit)
checkpoint, pinned to revision `87e768fbfa03994095f3d14527c80c5ae70c5758`:

```bash
QWEN_CHECKPOINT="$(uv run hf download mlx-community/Qwen3.5-0.8B-8bit \
  --revision 87e768fbfa03994095f3d14527c80c5ae70c5758 --quiet)"
uv run scripts/mlx_serve.py --model qwen3.5-0.8b-mlx --checkpoint "$QWEN_CHECKPOINT"
```

This uses Hugging Face's normal cache, respecting `HF_HOME` and `HF_HUB_CACHE`.
The 1.00 GB safetensors file contains packed affine language weights and BF16
vision weights. Loading preserves the packed weights and metadata; it does not
require a BF16 language checkpoint or startup requantization. The preset uses an
8,192-token context and at most four requests; `--context-length` and
`--max-batch-size` override those limits.

The loader also accepts original BF16 Hugging Face checkpoints, including
per-expert or fused MoE tensors, and quantizes language weights to 8-bit affine
weights with group size 64. For
[`ornith-ai/Ornith-1.0-35B`](https://huggingface.co/ornith-ai/Ornith-1.0-35B),
use revision `5df2ed3f675c7beaa490328cc70bb573b65fb660` and the
`ornith-1.0-35b` preset. That preset serves text only and requires substantially
more memory than the 0.8B model.

Building the native extension requires Xcode's C++ and Metal command-line tools
and CMake 3.25 or newer. Build and runtime use MLX 0.32.4 because the extension
uses its C++ ABI. Vision reuses MLX-VLM 0.7.11.

## Vision

The native Qwen recipe supplies normalized image patches and multimodal rotary
positions. A small adapter uses MLX-VLM's Qwen3.5 encoder for patch convolution,
learned positions, rotary attention and spatial merging, then inserts features
at image placeholders. A bounded feature cache reuses identical images
independently of the Engine's prefix cache.

Original HF and MLX-VLM vision weight layouts are supported without changing
checkpoint precision. DeepStack, packed quantized vision weights and video
input are unsupported.

The runner compiles the per-frame encoder by default; pass
`compile_vision=False` to opt out. Standalone `Qwen35VisionModel` and
`load_vision_model` are eager unless passed `compiled=True`. Geometry preparation
and frame slicing stay outside compilation. Upstream Split operations need
concrete sizes, so compiled functions cache by shape; this does not promise one
trace across image sizes. Startup warms one canonical grid with zero patches.

## Runtime

The Engine owns cache allocation, prefix leases and request lifetime. Its cache
has 128-token attention pages and recurrent-state snapshots at 4,096-token
checkpoints. Live DeltaNet arrays stay outside the arena between decode steps;
prefix hits restore a snapshot, and cancellation or retraction releases held
state through the Engine's normal lifecycle.

A native cache resource keeps one allocation across mutations. Dependency
epochs and computed reader outputs order reads before later writes, and Metal
completion handlers retain buffers until GPU work finishes. An explicit snapshot
makes an independent copy; metadata queries do not copy the arena. See the
[native cache contract](../_native/README.md) for ordering and lifetime details.
Streams are selected on the calling thread so startup and HTTP inference can
run on different threads.

Decode walks page tables directly, without gathering history. Layer-major pages
improve locality within each slab. A native reader for BF16 H8/KV2/D256 avoids
charging the entire arena to MLX's command-buffer packing heuristic; other
geometries use the generic reader with the same Metal body. Prefill gathers
history per layer and limits materialized attention scores to a 512 MiB budget.

Decode uses the Engine's enqueue/completion contract: a continuation submits its
GPU work before the preceding result is collected. GPU dependencies protect
cache reuse without a host wait between ordinary decode steps. Terminal steps,
ownership transitions and logprob results drain pending writes.

The compact runtime retains the useful fusions:

- RMS normalization and residual addition preserve BF16 rounding with FP32
  effective scales; unsupported regimes use MLX operations.
- Small decode projections share materialized packed weight rows without
  dequantization; larger GEMMs retain separate projections. Weight, dtype or
  module updates invalidate prepared views.
- One-token Conv4 fuses SiLU, Q/K/V splitting and the convolution carry.
- D256 attention merges use a fused reduction for 2–31 partitions, with a
  compiled MLX fallback. K/V packing preserves raw bits for strided layer rows.

Model constructors, activations and recurrent kernels reuse MLX LM. Runtime
source totals 3,712 lines of Python, type stubs, C++ and Metal, versus 4,817 in
the larger implementation and 1,906 before these additions. Tests, docs and
packaging are excluded.

## Measured performance

November 2, 2026, Apple M1 Pro with 16 GiB, MLX 0.32.4 and mlx-lm 0.32.2,
using the pinned 0.8B 8-bit checkpoint above. These are medians of four matched
AB/BA windows per backend, with greedy sampling and 320 completions after five
warmup steps. Decode starts from an 8K prompt:

| Aggregate output tokens/s | Batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| Resin Engine | 135.2 | 212.7 | 259.3 |
| Native mlx-lm decoder | 123.5 | 213.0 | 249.7 |

Batches 1 and 4 are faster in all four pairs. Batch 2 is effectively tied, with
two wins and two losses. The ratio of medians is +9.5%, −0.1% and +3.8%; these
four windows do not establish confidence intervals or performance elsewhere.

| Complete prefill latency, seconds | 4K prompt | 8K prompt |
| --- | ---: | ---: |
| Resin Engine | 2.047 | 4.417 |
| Native mlx-lm decoder | 2.036 | 4.464 |

Both paths retain the same 694 parameter arrays, vision tower, 512 MiB
allocator-cache limit, recommended wired-memory limit and 674,758,656-byte arena.
The benchmark explicitly uses 32 cache parents and a 16K context ceiling so the
8K prompt plus completions fit; measured decode contexts are [8197, 8517).
Prefill includes the whole prompt in 4K execution chunks. The reference uses
native mlx-lm kernels/caches and one-lookahead token delivery, excluding
text detokenization and logprob bookkeeping. These are warm local Engine
measurements, not HTTP throughput measurements.

Matched Metal captures show 3,750 versus 6,910 dispatches over ten decode
forwards. Fresh HTTP checks return identical repeated text and image responses;
the repeated long text prompt reuses 4,096 prefix tokens. These functional
checks do not measure image quality or vision throughput.

Use the shared benchmark entry point for local Engine curves, following the
[benchmark guide](../../../../docs/benchmarks.md):

```bash
uv run scripts/bench.py --model qwen3.5-0.8b-mlx --backend local \
  --checkpoint "$QWEN_CHECKPOINT" --context-lengths 4096,8192 \
  --decode-batch-sizes 1,2,4 --output artifacts/qwen35.jsonl --plot
```

## Validation and limits

The retained tests compare model/checkpoint and vision outputs with independent
references, check essential fusion numerics and cache ordering, and exercise
prefix reuse and overlapped decoding through the Engine.

```bash
uv run pytest models/mlx/tests tests/test_deploy_models.py \
  tests/test_qwen3_5_moe_config.py
uv build --package resin-models-mlx
```

Metal tests require Apple Silicon; CUDA-only checks run on CUDA hosts. The
native binding and vision adapter have also been checked from installed wheels.

Startup compilation stability remains unqualified: upstream MLX may create
additional pipelines for first unseen inputs. Vision compilation is shape
cached. The current performance measurements cover only the 0.8B checkpoint,
4K/8K prompts and batches 1/2/4 on this M1 Pro. They do not qualify larger
contexts, other Apple GPUs, the large Ornith preset or speculative decoding.
