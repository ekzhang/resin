# Qwen3.5-MoE runner for Apple Silicon (MLX)

`Qwen3_5MoeMlxRunner` serves Qwen3.5-MoE checkpoints, such as
[`ornith-ai/Ornith-1.0-35B`](https://huggingface.co/ornith-ai/Ornith-1.0-35B)
(revision `5df2ed3f675c7beaa490328cc70bb573b65fb660`), on one Apple Silicon GPU
through Metal. It loads BF16 Hugging Face safetensors (multimodal or text-only
exports, with per-expert or fused expert tensors) and quantizes them at load time
to MLX 8-bit affine weights (group size 64). Text only: the vision tower and any
MTP head are not loaded, and the preset's recipe rejects image and video input.

```bash
uv run scripts/mlx_serve.py --model ornith-1.0-35b --checkpoint ~/models/Ornith-1.0-35B
uv run scripts/bench.py --model ornith-1.0-35b --backend local \
  --checkpoint ~/models/Ornith-1.0-35B --context-lengths 8192 \
  --decode-batch-sizes 1,2,4,8,16 --output artifacts/ornith.jsonl --plot
```

## Cache

The Engine sees the same two groups as the CUDA Qwen family: 128-token
full-attention pages (all ten attention layers per block) and a STATE group
holding each request's live slot and its 4096-token prefix snapshots.
Physically the arena is one BF16 MLX array of 256 KiB _units_; a unit is one
layer's keys and values for one page. A slab is 250 units (65.5 MB) and holds 25
KV blocks or one STATE block (240 units of FP32 recurrent state, then the
convolution tails of all 30 layers).

Live DeltaNet state does not travel through the arena. `LiveStates` keeps each
request's state in the arrays the model returned for its last step, so an
unchanged decode batch feeds them straight back, and a changed batch selects
and concatenates rows once. The arena is touched only at the STATE plan's
transitions: a prefix hit reads its snapshot instead of starting from zero, and
a row whose accepted length reaches a reserved checkpoint writes one. A request
leaves when the Engine releases it (finish, cancellation or retraction), and a
row that arrives without held state, a snapshot or position zero is an error.
Held state is reserved in the memory budget for `max_batch_size` rows, twice
over for a step's input and output. The live slots the Engine allocates in the
arena go unused.

MLX arrays are immutable values, so in-place updates rely on buffer donation.
Every step finishes its reads before issuing writes, and drops any live view of
the arena, so MLX updates the buffer in place rather than copying all of it;
`test_cache_writes_update_the_arena_in_place` guards this.

Decode implements the Engine's `enqueue` overlap. Each step samples on the GPU
and hands its token IDs to the next step as feedback. A continuation builds its
graph while the previous step runs, waits for that step's reads, then submits
the previous step's arena writes followed by its own forward pass, so the
Engine commits the previous step while the GPU is busy. A step with no
continuation issues its writes when its result is collected. Prefill and steps
that return logprobs run synchronously. Writes use the
fastest MLX primitive for each payload: keys and values scatter single rows,
snapshot recurrent states are contiguous slice updates per layer and row, and
the convolution tails of a row are one padded slice. Capacity is bounded by Metal's
maximum buffer length as well as the memory budget.

Decode attention never gathers history. `paged_decode_attention` walks each
row's page table inside a Metal kernel with an online softmax per partition of
512 positions (256 below four rows, to occupy more GPU cores), then merges
partitions by log-sum-exp; the step's own key and value are its final position.
Each SIMD group reads a contiguous run of positions page by page, loading four
positions ahead so their memory latency overlaps. Grouped query heads,
including counts that are not multiples of the threadgroup's four, and any head
dimension divisible by 32 are supported.

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
- token-identical continuation after a 4096-token prefix-cache hit, including
  a snapshot written during decode and reused by the next turn;
- batched decode identical to each request alone, and unchanged when requests
  join and leave a running batch, are cancelled, or are retracted and replayed;
- overlapped decode token-identical to synchronous steps, greedy and seeded
  sampling, including a request that stops while its next step is in flight;
- the paged kernel against gathered attention, including NaN-poisoned padding;
- prompt logprobs of a multi-row prefill against the one-pass forward;
- sampler filtering against `TorchSampler`, and row-stable seeded draws.

## Results on an M4 Max (128 GB)

October 29 and 31 (decode), 2026, mlx 0.32.4 and mlx-lm 0.32.2, Ornith-1.0-35B
at the pinned revision. Loading the BF16 checkpoint and quantizing it takes
4–14 s and leaves about 50 GiB resident.

mlx-lm 0.32.2 cannot load Ornith-1.0's per-expert checkpoint directly, so the
baseline builds mlx-lm's `qwen3_5_moe` model from the same 8-bit weights, in
eval mode so its DeltaNet layers use the Metal kernel, and times its bare forward
pass with a plain KV cache and greedy argmax. Resin is timed through the Engine,
with paged cache, live state, scheduling and sampling. Each engine runs in its
own process and idles 240 s before every timed window, since a batch-16 setup
(16 prefills of 8K tokens) otherwise heat-soaks this laptop. After 5 warmup
steps, a Resin decode cell times 20 committed steps back to back, without
synchronizing between them so overlap is kept, and reports the median of three
such windows; an mlx-lm cell is the median of 20 synchronized steps. Rows decode
distinct tokens. Prefill runs in 4,096-token chunks for Resin and mlx-lm's default
2,048 for mlx-lm.

| Decode, 8K context     |    1 |     2 |     4 |     8 |    16 |
| ---------------------- | ---: | ----: | ----: | ----: | ----: |
| Resin step (ms)        | 12.6 |  15.9 |  22.7 |  34.0 |  52.5 |
| mlx-lm step (ms)       | 12.7 |  15.7 |  21.0 |  31.0 |  50.4 |
| Resin / mlx-lm         | 1.00 |  1.02 |  1.08 |  1.10 |  1.04 |
| Resin output tokens/s  | 79.1 | 125.6 | 176.6 | 235.2 | 305.0 |
| mlx-lm output tokens/s | 79.0 | 127.8 | 190.7 | 258.0 | 317.7 |

| Prefill, one request  |    4K |    8K |   16K |   32K |
| --------------------- | ----: | ----: | ----: | ----: |
| Resin input tokens/s  | 1,665 | 1,716 | 1,587 | 1,239 |
| mlx-lm input tokens/s | 2,017 | 1,929 | 1,662 | 1,321 |

Without `enqueue` overlap, timed the same way, the steps are 14.7, 18.1, 24.9,
36.6 and 55.0 ms: overlap hides 2.1–2.6 ms of host work per step (graph
construction and the Engine's plan, sampling and commit). Before live state, the
steps were 18.0, 22.3, 30.4, 47.0 and 79.0 ms; copying each row's DeltaNet state
through the arena cost 18 ms at batch 16. Swapping mlx-lm's modules into
Resin's forward one at a time, only attention moves the batch-4 step (by
1.5 ms); norms, DeltaNet and experts match. Serving every query head of a KV
head from one threadgroup, or launching head groups side by side, does not
recover it.

Without overlap, with eight rows decoding at 8K, a step takes 36.5 ms; the first step after a
row leaves takes 37.0–39.0 ms and the first after a row joins 39.4 ms, while
the batch's held state is regathered. An 8,292-token prompt's first token takes
5.03 s cold and 0.68 s when its first 8,192 tokens, with their DeltaNet
snapshot, are cached.

- **Parity.** On three chat prompts, Resin's prompt logits have the same argmax
  as mlx-lm's at every position, and 64-token greedy continuations match
  exactly. Logit differences come from mlx-lm computing `1 + w` and the
  DeltaNet decay in BF16, where the reference model uses FP32.
- **Serving.** `scripts/check_server.py` passes chat, logprobs, concurrency,
  stress, robustness and recovery. A repeated 5,247-token prompt reuses 4,096
  cached tokens, cutting its latency from 2.70 s to 0.68 s. No prompt up to 16K
  tokens shows a first-request TTFT spike. Measured in the same run, after the
  stress checks, steady TTFT is 0.14 s, 1.1 s, 5.6 s and 12.7 s for prompts of
  172, 2,091, 8,231 and 16,429 tokens. On 100 seeded MMLU questions
  (`cais/mmlu` `c30699e`, 50 subjects × 2), the model answers 92 correctly with
  a 16,384-token reasoning budget; three reasoning traces exhaust it before
  answering. Rerun with `enqueue` overlap, the same checks pass and the model
  answers 91, again with three truncations. The previous runner scored 93 with
  one truncation. Image and video
  requests are refused.
- **Memory.** Prefill holds 3.3–3.7 GiB above the resident model from 32K to
  98K tokens.
- **Compilation.** After warmup, three sets of unseen context lengths (77 to
  9,001 tokens) with varied sampling settings add 0.00 s of
  `MTLCompilerService` CPU time, as do idle windows; 20 new kernels add 0.55 s.

## Limitations

- The published Ornith checkpoints declare one MTP layer but ship no MTP
  weights, so there is no speculative decoding.
- Vision input, multiple devices and prompt-chunk sizes other than 4096.
- The fused-expert and text-only loaders are tested against Transformers on a
  small configuration; only Ornith-1.0-35B has been run at full size.
