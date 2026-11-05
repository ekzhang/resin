# Qwen3.5 on Apple Silicon

`Qwen35MlxRunner` serves dense and MoE Qwen3.5 checkpoints through Resin's
Engine on one Apple Silicon GPU. It supports tied embeddings, paged attention,
chunked prefill, prefix reuse, overlapping decode, image input and speculative
decoding with the checkpoint's bundled MTP head.

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

## MTP speculative decoding

`speculative_draft_tokens=K` (`--draft-tokens` for `scripts/mlx_serve.py`)
loads the bundled `mtp.*` head, which must be present: original Qwen
checkpoints ship it, while `mlx-community` conversions and Ornith do not. The
`qwen3.5-0.8b-mtp-mlx` and `qwen3.8-27b-mlx` presets use pinned BF16 sources
quantized to 8-bit at load.

Each decode step follows the CUDA runner. The previous step proposed the first
draft; MTP chains `K - 1` more on its own hidden state. One target forward
verifies `[anchor, drafts]`, sampling column `j` exactly as one-token decoding
would sample that position, so greedy and seeded outputs match target-only
decoding up to BF16 rounding: verification multiplies `K + 1` rows per
sequence, so a near-tie can resolve differently, just as target-only decoding
changes with batch size. On the 0.8B checkpoint, every greedy divergence from
one to three drafts falls on a token whose top-two logit gap is at most 0.25,
and at batch 1 each is a position where target-only decoding itself differs
between batches 1 and 4. The matching draft prefix and one correction or bonus token are
returned. Verification writes K/V for every leased position, and only accepted
positions become readable. DeltaNet state for the accepted prefix is replayed
from the block's retained recurrence inputs with mlx-lm's masked kernel, so no
projection or MLP runs twice. MTP then reads the verified hidden states to fill
its own cache and propose the next anchor's draft.

The MTP layer keeps a third cache group with one unit per 128-token block. Slot
`p` pairs the target's hidden state at `p - 1` with token `p`, so every pair
lands in the block that leases its token, and the layout declares one extra
decode write. Slot 0 holds no pair; the paged reader skips it. Multi-token
verification runs each block token as a row of the paged decode kernel, plus
one partition over the block's earlier, unwritten keys, merged by log-sum-exp.

Speculative steps overlap through the Engine's enqueue path. A step's
feedback holds each row's next anchor and position on the GPU; the Engine's
continuation plan only bounds those positions and leases pages for the
longest possible output. History lengths, cache write addresses, rotary
offsets and sampling streams therefore all derive from GPU positions, and the
page tables cover the whole lease. Logprob, image and state-snapshot steps run
synchronously.

A prefix hit restores no target hidden state, so the hit boundary's MTP pair
uses zeros. That pair stays in the MTP cache and slightly weakens later
proposals. Image requests give MTP the image placeholder embeddings and text
rotary positions. Both cost only acceptance, never output. Each verified token also rereads its row's history,
so long contexts pay proportionally more attention bandwidth.

November 4, 2026, Apple M4 Max with 128 GiB, MLX 0.32.4 and mlx-lm 0.32.2,
`qwen3.8-27b-mlx` (8-bit). One process loads the model once and alternates
target-only and speculative Engines in four ABBA windows on the same 8K
ShareGPT cohort, 64 greedy tokens per request, prefill excluded. Cells are
median aggregate output tokens/s, with the speedup and draft acceptance:

| Batch | Target only | 1 draft             | 2 drafts            | 3 drafts            |
| ----: | ----------: | ------------------: | ------------------: | ------------------: |
|     1 |         8.7 | 14.3 (1.64×, 91%)   | 16.6 (1.89×, 80%)   | 17.5 (1.98×, 67%)   |
|     2 |        15.7 | 22.4 (1.42×, 90%)   | 23.6 (1.50×, 83%)   | 23.9 (1.52×, 70%)   |
|     4 |        24.7 | 29.1 (1.18×, 90%)   | 29.7 (1.19×, 84%)   | 33.1 (1.34×, 73%)   |
|     8 |        32.1 | 38.1 (1.19×, 87%)   | 45.3 (1.41×, 79%)   | 42.1 (1.33×, 69%)   |

This 14-inch laptop throttles its GPU after about two minutes of sustained
decode, so these are throttled rates and only matched ratios are compared.
From a cool start, target-only decoding runs at 14.9 tokens/s at batch 1,
and two drafts run at 31.3 (2.10×). Outputs are identical at batches 1 and 2. At batch
8, every speculative divergence falls on a token where target-only decoding
itself changes between batch 1 and batch 8, with a top-two logit gap of 0 or
0.125.

The table times synchronous speculative steps. A second matched run added
overlapped steps: on the 27B they stayed within 1% of synchronous steps
wherever windows were stable, because each step is GPU-bound. The two noisy
batch-8 cells, with one and three drafts, measured 0.85× and 1.08×. The 0.8B
checkpoint is launch-bound and its 248K-token `lm_head` dominates, so overlap
matters there. With overlap, speculation reaches 1.10× (one draft) and 1.22×
(two drafts) over target-only decoding at batch 1, against 0.83× and 0.96×
for synchronous steps. At batch 4 it reaches 0.94–0.95×.

## Measured performance

November 2, 2026, Apple M1 Pro with 16 GiB, MLX 0.32.4 and mlx-lm 0.32.2,
using the pinned 0.8B 8-bit checkpoint above. These are medians of four matched
AB/BA windows per backend, with greedy sampling and 320 completions after five
warmup steps. Decode starts from an 8K prompt:

| Aggregate output tokens/s | Batch 1 | Batch 2 | Batch 4 |
| ------------------------- | ------: | ------: | ------: |
| Resin Engine              |   135.2 |   212.7 |   259.3 |
| Native mlx-lm decoder     |   123.5 |   213.0 |   249.7 |

Batches 1 and 4 are faster in all four pairs. Batch 2 is effectively tied, with
two wins and two losses. The ratio of medians is +9.5%, −0.1% and +3.8%; these
four windows do not establish confidence intervals or performance elsewhere.

| Complete prefill latency, seconds | 4K prompt | 8K prompt |
| --------------------------------- | --------: | --------: |
| Resin Engine                      |     2.047 |     4.417 |
| Native mlx-lm decoder             |     2.036 |     4.464 |

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
prefix reuse and overlapped decoding through the Engine. MTP tests use an FP32 model and require
exact token parity with target-only decoding for one to three drafts, natural
and synthetic full or partial acceptance, seeded sampling, stop tokens inside a
block, checkpoint crossings, prefix hits and the context limit.

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
contexts, other Apple GPUs or the large Ornith preset; speculative decoding is
measured separately above.
