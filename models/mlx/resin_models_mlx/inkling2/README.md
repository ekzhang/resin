# Inkling 2 Mini on Apple Silicon

`Inkling2MlxRunner` serves **Inkling 2 Mini** text and image prompts on one
Apple Silicon GPU. It supports paged global and sliding-window attention,
chunked prefill, prefix reuse and overlapped decode.

## Run locally

The `inkling2-mini-mlx` preset pins
[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7)
at revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`:

```bash
INKLING_CHECKPOINT="$(uv run hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --quiet)"
uv run scripts/mlx_serve.py --model inkling2-mini-mlx --checkpoint "$INKLING_CHECKPOINT"
```

The preset loads the language model and vision tower from the target checkpoint.
It uses 8,192-token prefill chunks and defaults to eight concurrent requests.
Set the context and batch limits with `--context-length` and `--max-batch-size`.

## Precision

Experts stay packed in the checkpoint's formats: NVFP4 in layers 0–19 and MXFP8
in layers 20–23. Their bytes are viewed directly as MLX `nvfp4` and `mxfp8`
weights; dequantized values match the CUDA reference bit for bit on the real
checkpoint. The NVFP4 output scale, activation amax times the expert's weight
amax over `1536 · denominator`, is applied to the FP32 GEMM output as the CUDA
kernel does, rather than through MLX's global scale.

The checkpoint requires quantized expert activations. A Metal kernel reproduces
the CUDA quantizer bit for bit (NVFP4's four-or-six block scale chosen by summed
reconstruction error and MXFP8's power-of-two scale) and emits FP32 inputs
for the expert GEMMs. The NVFP4 row maximum stays in the activation dtype
until the FP32 output scaling step. The one exception to quantizer parity is an MXFP8 block whose
largest value is below 448 · 2⁻¹²⁷, which becomes zero where CUDA keeps
denormals. The packed GEMM therefore forms the same products as CUDA's
W4A4/W8A8 MMA, up to FP32 summation order. Attention and dense projection
weights, convolution history and KV are BF16; attention scores, softmax and
outputs are FP32. Depth mixes, controller gates, routing and logits are FP32,
and BF16 rounding points follow the CUDA runner (`resin_models_cuda/inkling2`).

## Execution

Mini runs 48 layer visits over 24 stored layers: layers 0–5, layers 6–17 three
times, then layers 18–23. Each pass restarts from the prelude's residual blocks
and receives the previous pass through four message controllers per layer.
Every visit has its own KV cache and convolution history.

The Engine owns three cache groups, mirroring the CUDA runner without DSpark:

- global KV for the 8 global visits (full history);
- local KV for the 40 sliding-window visits, retaining the 1,024 readable
  tokens plus one prefill chunk of writes;
- convolution history (`48 × 3 × 6656` BF16) for prefix snapshots.

Live convolution history stays on the GPU between steps, and prefix hits restore
a snapshot. Both prefill and decode read retained BF16 KV directly from the
page table and new KV from the current chunk. The prefill kernel streams tiles
of visible keys through an online softmax, applies the learned relative bias
to the most recent 1,024 positions, and restricts local visits to their window.
Both paths return FP32 attention outputs and log-sum-exp for the sink.

Decode overlaps through the Engine's enqueue contract. Fused Metal kernels cover
four-tap convolutions over independent 16-token tiles, depth mixes, RMS norms,
expert routing, the capped SwiGLU
with its output scale, the route combine and the BF16 vocabulary projection
with FP32 output. Decode also compiles small FP32 helpers with `mx.compile`;
their shapes depend only on the batch size. Prefill runs uncompiled, and chunks
of 512 or more tokens run one dense FP32 GEMM per expert segment over exactly
dequantized weights; shorter chunks keep the gathered quantized GEMM. At 2,048 tokens (16K routes) that is 2.3× (NVFP4) to 3.5× (MXFP8)
faster than MLX's gathered quantized GEMM, dequantization included.

## Validation

No independent MLX or Transformers implementation of Inkling 2 exists, so the
reference is the CUDA package's `Inkling2Target` on CPU PyTorch, with only its
GPU kernels replaced: dense attention, and packed experts computed by the CUDA
tests' independent NVFP4/MXFP8 arithmetic.

- On a tiny looped checkpoint with both expert formats, every visit's mixing,
  attention and experts match the reference within about one BF16 rounding step
  over a prefill and six decode steps, including prefill chunks large enough for
  the per-expert GEMMs (teacher-forced: random tiny models with FP4 activations
  amplify one rounding step into different logits).
- On the real checkpoint, the opt-in test (a 24-token prompt from the model
  card, then three greedy decode steps) finds each visit's attention and
  experts within 0.53% relative, teacher-forced logits within about 1×10⁻⁵, and
  the same top-1 token at every step. Ad hoc runs of the same comparison on
  16-, 1,100- and 2,100-token prompts with four decode steps each also agreed on
  all 15 top-1 tokens; mean absolute logit differences were 0.25–1.05 (logits
  are capped at ±30). Moving 40 values by one BF16 step at the first visit moves
  the reference's own logits by a mean of 0.64. The same amplification means greedy output can change with
  the prefill chunk size or the batch a request runs in, whose different GEMM
  shapes round differently; it is reproducible for a fixed configuration.
- Engine tests check that every arena history read is byte-identical to what
  was written across page, chunk and window boundaries and prefix restores;
  that every runner attention call matches an unsliced dense reference,
  including windows and bias extents spanning several pages; that a
  cached continuation equals a fresh one; and that overlapped decode equals
  synchronous decode. Image tests exercise a checkpoint-backed request and
  verify embedding placement across chunks and prefix-cache hits.
- On a rectangular synthetic image and Mini's published vision weights, MLX
  projector output matched the PyTorch CPU reference to 0.99% relative L2
  error (cosine similarity 0.99995).
- `scripts/check_server.py --stress` passes chat, logprobs, concurrency, prefix
  reuse, stress, robustness and recovery, and context checks up to 2,048 tokens.
  The original runner took 15 seconds on the 8,192-token check, exceeding its
  default 10-second TTFT limit. These measurements used text prompts.
- On 100 MMLU questions (`cais/mmlu` revision `c30699e8`, 50 subjects × 2,
  seed 1017; default reasoning, 4,096-token cap, eight concurrent), 86 answers
  are correct. Of the 14 others, 4 are wrong, 2 reached the token cap and 8
  exceeded the checker's 180-second client timeout while still reasoning.

After warmup over batch sizes 1–8 and both sampling paths, unseen prompt
lengths (up to 9,000 tokens), contexts, chunk lengths and sampling values create
no new custom Metal kernel variants and no new `mx.compile` traces on the real
checkpoint. An Engine test repeats the custom-kernel check on the tiny model.
MLX's own kernel JIT (GEMMs) is not observable this way and remains
unqualified.

```bash
uv run pytest models/mlx/tests/test_inkling2_model.py \
  models/mlx/tests/test_inkling2_engine.py models/mlx/tests/test_inkling2_attention.py \
  models/mlx/tests/test_inkling2_vision.py tests/test_deploy_models.py
# Real checkpoint against the CPU reference (about 50 GB of memory, under a minute):
RESIN_TEST_CHECKPOINT="$INKLING_CHECKPOINT" \
  uv run pytest models/mlx/tests/test_inkling2_checkpoint.py
```

## Measured performance

### Image encoding on M1 Pro

With the pinned Mini vision weights and MLX 0.32.4, one synthetic normalized
RGB image took 77 ms at 320 × 320, 253 ms at 640 × 640, and 639 ms at 960 × 960
(median of three runs after warmup). These times include patch packing and
projection but exclude preprocessing, language prefill, and HTTP overhead.
Cached feature lookup took about 4–5 µs.

### Prefill attention on M1 Pro

Each cell shows the time for one attention visit, from the old dense path to
the integrated paged kernel. The two implementations ran in separate sessions
on the same 16 GiB M1 Pro with MLX 0.32.4, identical synthetic BF16 inputs and
Mini's head geometry. Each time is the median of three calls after warmup.

| Prior tokens | New tokens | Global, dense → paged (ms) | Local, dense → paged (ms) |
| -----------: | ---------: | -------------------------: | ------------------------: |
|            0 |        128 |                0.96 → 0.73 |               0.68 → 0.70 |
|            0 |      2,048 |              56.83 → 14.75 |             42.18 → 11.60 |
|        8,193 |        128 |               16.64 → 7.36 |               3.46 → 1.29 |
|            0 |      8,192 |          1,093.39 → 208.42 |            236.09 → 53.31 |
|        8,192 |      8,192 |          3,039.08 → 637.95 |            224.47 → 57.32 |

Dense timings include gathering retained KV. For 8,192-token chunks, the
benchmark synchronized after each dense query block to bound memory. Paged
attention used one dispatch per visit. In all ten cases the integrated
kernel matched the prototype bit for bit. Compared with the dense reference,
the maximum absolute error was 3.34 × 10⁻⁶ for the output and 2.86 × 10⁻⁶
for log-sum-exp.

### Historical full-model throughput on M4 Max

The original runner (`e2f020b`) was benchmarked on November 22, 2026 on a
14-inch MacBook Pro with M4 Max and 128 GiB, using MLX 0.32.4, ShareGPT
prompts tokenized for Inkling and greedy sampling. Prefill figures are
medians of three cohorts; decode figures are medians of 20 overlapped steps after
five warmup steps. This laptop throttles under sustained load.

| Decode at 8K context | Batch 1 | Batch 2 | Batch 4 | Batch 8 | Batch 16 |
| -------------------- | ------: | ------: | ------: | ------: | -------: |
| Output tokens/s      |    35.6 |    60.6 |    97.6 |   129.6 |    140.1 |
| Tokens/s per request |    35.6 |    30.3 |    24.4 |    16.2 |      8.8 |

Every measured decode step overlapped with the next through `enqueue`. Batch 16
was measured in a separate run after a change that left decode code untouched. At batch
eight, step times rose from 56 to 66 ms across the window.

| Complete prefill | 4K prompt | 8K prompt | 16K prompt | 32K prompt |
| ---------------- | --------: | --------: | ---------: | ---------: |
| Seconds          |      7.60 |     15.48 |      38.79 |     103.63 |
| Input tokens/s   |       539 |       529 |        422 |        316 |

Prefill runs in 8,192-token chunks and includes the whole prompt. Reproduce
with the shared benchmark (raw JSONL, plots and Perfetto traces for both phases):

```bash
uv run python -m resin.bench_prompts prepare \
  --corpus ShareGPT_V3_unfiltered_cleaned_split.json --source-metadata source.json \
  --tokenizer "$INKLING_CHECKPOINT" --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --contexts 4096,8192,16384,32768 --context-counts 4096:16,8192:16,16384:8,32768:8 \
  --output inkling2-mini-32k.resinprompts
uv run scripts/bench.py --model inkling2-mini-mlx --backend local \
  --checkpoint "$INKLING_CHECKPOINT" --prompt-source saved \
  --prompt-corpus inkling2-mini-32k.resinprompts \
  --prefill-context-lengths 4096,8192,16384,32768 --decode-context-lengths 8192 \
  --decode-batch-sizes 1,2,4,8 --output artifacts/inkling2-mini-mlx-bench.jsonl \
  --trace-dir artifacts/inkling2-mini-mlx-traces --plot
```

For comparison, llama.cpp at commit `8d60873dfb8619f2b5ed31adfbab07cccb40db5c`
on the same machine, with a Q4_K_M GGUF (16.3 GiB) converted from the same
checkpoint by its own `convert_hf_to_gguf.py` and `llama-quantize` (Unsloth's
published GGUFs use a different tensor layout that upstream llama.cpp does not
load). Metal offload and flash attention are enabled. Q4_K_M stores most
weights, attention included, in about 4.5 bits, where Resin keeps BF16 attention
and the checkpoint's NVFP4/MXFP8 experts.

| llama.cpp, 8K context | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --------------------- | ------: | ------: | ------: | ------: |
| Output tokens/s       |    32.2 |    47.1 |    64.8 |    81.9 |
| Resin speedup         |   ~1.1× |   1.29× |   1.51× |   1.58× |

`llama-batched-bench -npp 8192 -ntg 64` measured those rates; `llama-bench`
(`tg128 @ d8192`) reports 28.0 tokens/s at batch one. Resin's rates are median
steps; its batch-one window average was 30.9 tokens/s because of one slow
step, so batch one is roughly parity to 1.1×. llama.cpp's prefill
reaches 404 tokens/s for a 4K prompt and 318 for 8K (`llama-bench`), and
332–356 tokens/s per 8K prompt when batched. These are single sessions on a
laptop that throttles; they compare implementations, not precision.

Batch-one decode reads about 2.6 GB of BF16 attention weights, 1.8 GB of expert
weights and 0.9 GB of vocabulary projection per token, about 5.3 GB. At the
330–410 GB/s chained GEMVs reach here, that bounds it near 62–77 tokens/s. The
remaining gap is the dispatch of many small per-visit operations.

## Limits

- No DSpark speculative decoding (the CUDA preset uses seven proposals). This
  PR is scoped to the target model; speculation needs the bundled draft model's
  own cache groups, block verification and rollback, and is left for a
  follow-up, so the speculative-decoding step of the model checklist is open.
- One GPU; no tensor or expert parallelism.
- Prefill expert GEMMs reach about 7.8 TFLOPS (versus 13.5 for one dense GEMM)
  because each expert sees about 128 rows per 2,048 tokens.
- Prompt logprobs project 128 prompt rows in about 60 ms (four rows per GEMV
  launch), adding about 4 seconds per 8,192-token chunk when requested.

## Source references

- The CUDA runner `models/cuda/resin_models_cuda/inkling2`, which follows
  [SGLang's Inkling 2 integration](https://github.com/sgl-project/sglang/tree/e62ac6279646c686d3b328f58d9b5c2c2d5927dc),
  defines the arithmetic and supplies the CPU reference. The activation
  quantizer adapts SGLang's Apache-2.0 kernel.
- The routing and activation kernels reuse MLX v0.32.4's sigmoid formula.
  Prefill uses its Steel SIMD fragment coordinates. The MIT notice is in
  this directory's LICENSE.
