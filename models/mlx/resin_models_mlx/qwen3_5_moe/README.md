# Qwen3.5-MoE runner for Apple Silicon (MLX)

`Qwen3_5MoeMlxRunner` serves Qwen3.5-MoE checkpoints, such as
[`ornith-ai/Ornith-1.0-35B`](https://huggingface.co/ornith-ai/Ornith-1.0-35B)
(revision `5df2ed3f675c7beaa490328cc70bb573b65fb660`), on one Apple Silicon GPU
through Metal. It loads BF16 Hugging Face safetensors (multimodal or text-only
exports, with per-expert or fused expert tensors) and quantizes them at load time
to MLX 8-bit affine weights (group size 64). Text only: the vision tower and any
MTP head are not loaded, and the preset's recipe rejects image and video input.

```bash
uv run scripts/serve.py --model ornith-1.0-35b --checkpoint ~/models/Ornith-1.0-35B
uv run scripts/bench.py --model ornith-1.0-35b --backend local \
  --checkpoint ~/models/Ornith-1.0-35B --context-lengths 8192 \
  --decode-batch-sizes 1,2,4,8,16 --output artifacts/ornith.jsonl --plot
```

## Boundaries

| Component | Responsibility |
| --- | --- |
| `resin.models.qwen3_5.config` | Backend-neutral Qwen3.5 and Qwen3.5-MoE geometry. |
| `checkpoint.py` | Map checkpoint names, stack or split experts, quantize layer by layer. |
| `model.py` | Stateless forward: gated DeltaNet, gated attention, routed and shared experts. Qwen's `1 + w` RMSNorm scale and the DeltaNet decay are computed in FP32, as in the reference. |
| `cache.py` | Slab geometry and typed reads and writes over the Engine-owned arena. |
| `../paged_attention.py` | Metal flash-decoding kernel that reads K/V directly from slabs. |
| `../sampling.py` | Batched GPU sampling with `TorchSampler`'s filtering semantics. |
| `runner.py` | `ManagedRunner` adapter: plans in, tokens and scores out. |
| Shared Resin runtime | Scheduling, cache admission, prefix reuse, retraction, HTTP and metrics. |

The model imports mlx-lm's gated delta-rule Metal kernel and switched-expert
`gather_qmm` layers, as CUDA families import FlashInfer. Its composition follows
the CUDA `qwen3_5` family and mlx-lm's `qwen3_5`/`qwen3_next` modules.

## Cache

The Engine sees the same two groups as the CUDA Qwen family: 128-token
full-attention pages (all ten attention layers per block) and a STATE group of
4096-token DeltaNet checkpoints using the two-page ring. Physically the arena is
one BF16 MLX array of 256 KiB *units*; a unit is one layer's keys and values for
one page. A slab is 250 units (65.5 MB) and holds 25 KV blocks or one checkpoint
(240 units of FP32 recurrent state, then the convolution tails of all 30 layers).

MLX arrays are immutable values, so in-place updates rely on buffer donation.
Every step evaluates its reads before issuing writes, and drops any live view of
the arena, so MLX updates the buffer in place rather than copying all of it;
`test_cache_writes_update_the_arena_in_place` guards this. Writes use the
fastest MLX primitive for each payload: keys and values scatter single rows,
recurrent states are contiguous slice updates per layer and row, and the
convolution tails of a row are one padded slice. Capacity is bounded by Metal's
maximum buffer length as well as the memory budget.

Decode attention never gathers history. `paged_decode_attention` walks each
row's page table inside a Metal kernel with an online softmax per partition of
512 positions, then merges partitions by log-sum-exp; the step's own key and
value are its final position. Grouped query heads, including counts that are not
multiples of the threadgroup's four, and any head dimension divisible by 32 are
supported.

Prefill gathers one row's earlier pages a layer at a time, only as each layer
reads them. MLX's fused attention streams scores only up to head dimension 128;
for Ornith's 256 it would materialize every query-key score, 4 GiB per layer at
32K keys. The runner therefore splits each chunk's queries into blocks within a
512 MiB score budget and evaluates them one at a time. Prefill memory stays
about 3.5 GiB above the resident model from 32K to 98K tokens.

## Correctness

`models/mlx/tests` checks, on a small random configuration:

- forward logits against mlx-lm, and loader plus forward against Transformers;
- FP32 incremental state and history against a one-pass forward, exactly;
- Engine greedy decoding across page boundaries (127/128/129 tokens), chunked
  prefill across the 4096-token checkpoint, and decode across a page boundary,
  within BF16 tie tolerance of a one-pass reference (these route every token to
  all experts, since BF16 rounding flips near-tied top-k choices of a random
  router);
- token-identical continuation after a 4096-token prefix-cache hit;
- batched decode identical to each request alone;
- the paged kernel against gathered attention, including NaN-poisoned padding;
- sampler filtering against `TorchSampler`, and row-stable seeded draws.

## Results on an M4 Max (128 GB)

October 28, 2026, mlx 0.32.4 and mlx-lm 0.32.2, Ornith-1.0-35B at the pinned
revision. Loading the BF16 checkpoint and quantizing it takes 4–14 s and leaves
about 50 GiB resident.

mlx-lm 0.32.2 cannot load Ornith-1.0's per-expert checkpoint directly, so the
baseline builds mlx-lm's `qwen3_5_moe` model from the same 8-bit weights, in
eval mode so its DeltaNet layers use the Metal kernel, and times its bare forward
pass with a plain KV cache and greedy argmax. Resin is timed through the Engine,
with paged cache, scheduling and sampling. Each engine runs in its own process,
three rounds in alternating order, and idles 75 s before every timed window so
neither inherits the other's heat. Decode cells are the median over rounds of the
median of 20 synchronized steps after 5 warmup steps, with distinct tokens per row.
Prefill runs in 4,096-token chunks for Resin and mlx-lm's default 2,048 for mlx-lm.

| Decode, 8K context | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resin step (ms) | 18.0 | 22.3 | 30.4 | 47.0 | 79.0 |
| mlx-lm step (ms) | 12.7 | 15.6 | 21.2 | 31.2 | 50.5 |
| Resin output tokens/s | 55.6 | 89.6 | 131.7 | 170.2 | 202.5 |
| mlx-lm output tokens/s | 78.7 | 128.3 | 188.9 | 256.7 | 317.1 |

| Prefill, one request | 4K | 8K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: |
| Resin input tokens/s | 1,612 | 1,663 | 1,527 | 1,207 |
| mlx-lm input tokens/s | 2,009 | 1,925 | 1,667 | 1,312 |

Rounds agree within 4%. Resin's batch-16 setup prefills 16 rows of 8K tokens,
which heat-soaks this laptop: after a 75 s pause its steps still climbed from 54
to 137 ms. Its batch-16 cell is the median of two runs that idle 240 s instead.
Resin's decode steps take 1.4–1.6× as long as mlx-lm's, and prefill is 8–20%
behind, narrowing with length. At batch 16 the model forward is 56 ms; copying
each row's DeltaNet state (63 MB) out of the arena and back adds 18 ms, and
logits, sampling and host work about 6 ms. That copy is the main open
performance item.

- **Parity.** On three chat prompts, with full argmax agreement over prompt
  positions, two 64-token greedy continuations match mlx-lm exactly. The third
  departs at token 42 on a one-BF16-step tie (`iter`+`atively` against
  `iter`+`ative`+`ly`). Logit differences come from mlx-lm computing `1 + w`
  and the DeltaNet decay in BF16, where the reference model uses FP32.
- **Serving.** `scripts/check_server.py` passes chat, logprobs, concurrency,
  stress, robustness and recovery. A repeated 5,245-token prompt reuses 4,096
  cached tokens. No prompt up to 16K tokens shows a first-request TTFT spike.
  Measured separately from a cool start, steady TTFT is 0.14 s, 1.0 s, 5.0 s
  and 11.9 s for prompts of 172, 2,094, 8,235 and 16,421 tokens. On 100 seeded
  MMLU questions (`cais/mmlu` `c30699e`, 50 subjects × 2), the model
  answers 93 correctly with a 16,384-token reasoning budget; one answer is
  truncated by it and one names two options. Image and video requests are
  refused.
- **Memory.** Prefill holds 3.3–3.7 GiB above the resident model from 32K to
  98K tokens.
- **Compilation.** After warmup, three sets of unseen context lengths (77 to
  9,001 tokens) with varied sampling settings add 0.00 s of
  `MTLCompilerService` CPU time, as do idle windows; 20 new kernels add 0.55 s.

## Limitations

- The published Ornith checkpoints declare one MTP layer but ship no MTP
  weights, so there is no speculative decoding.
- `enqueue` overlap is not implemented. Writes must follow the step's reads to
  update the arena in place, which serializes each step on the GPU; the
  runner already queues writes asynchronously behind host sampling and commit.
- Vision input, multiple devices and prompt-chunk sizes other than 4096.
- The fused-expert and text-only loaders are tested against Transformers on a
  small configuration; only Ornith-1.0-35B has been run at full size.
- `scripts/serve.py` serves MLX presets only; CUDA presets need the graph
  capture and warmup in `deploy/modal_serve.py`. MLX presets have no Modal
  volume and are rejected by the Modal launchers.
