# DeepSeek V4.1 Flash runner

`DeepSeekV41CudaRunner` loads indexed safetensors from
`deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Serving imports Resin code;
checkpoint Python is a development reference.

The prepared forward pass follows [DeepInfer 774ce244](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01).
Its CUDA operator sources in `upstream/` retain Apache-2.0 attribution and license.
The pinned kernel stack is DeepGEMM `1fc9c669`, FlashMLA `2e5429fc`,
DeepSelect `bfa4507d`, and TileKernels 1.0.0; full revisions live in
[the CUDA package configuration](../../pyproject.toml).

## Boundaries

| Component | Responsibility |
| --- | --- |
| `model.py` | Shifted HC, shared KV/index ownership, Engram, and DSpark architecture. |
| `vision.py` | Checkpoint-shaped ViT and 3x3 patch-to-language projection. |
| `linear.py`, `attention.py`, `compressor.py`, `experts.py` | Lower loaded checkpoint tensors into fused projections, packed scales and prepared operators. |
| `indexer.py`, `upstream/` | Paged FP4 MQA, candidate selection and fused cache transforms. |
| `batch.py` | Prepare shared store slots, attention indices, compression lengths and state-page mappings once per forward. |
| `yoco.py` | Select retained consumer queries and bound their SWA visibility over complete producer KV. |
| `cache.py` | Typed zero-copy views over Engine-owned byte slabs and live page tables. |
| `cuda.py` | Adapt the serving runtime and prepare graph regions. |
| Shared Resin runtime | Scheduling, cache admission, graph lifecycle, sampling, scores and distributed steps. |

`resin.recipe.get_native_recipe("deepseek_v41", tokenizer)` prepares typed image content independently of HTTP.
Its resize, padding, normalization and patch order follow the pinned checkpoint's
image processor. The Rust recipe expands trusted image slots; `Media`
owns the normalized CPU bitmap and includes its identity in prefix-cache keys. The runner
packs and encodes each uncached image once and slices its embeddings across causal
prefill chunks, including partial prefix-cache restores. Image rows use the
checkpoint's vision routing bias and are masked from Engram; generated tokens
keep the ordinary text path. Vision encoding runs outside the language-model
graph regions, and decode retains full CUDA graphs and DSpark.

Preparation fuses Q-A/KV and Q-B/index-Q. BF16 vocabulary/Markov GEMMs return FP32
logits; FP8 projections use per-token block32 casts and packed UE8M0 TMA scales.
C2 combines BF16 gate/value projections with FP32 accumulation and pools on the
Engine's paged FP32 state. Only completed compression windows update packed KV
or index keys; rejected speculative rows cannot overwrite accepted earlier state.

Each target or draft forward prepares one batch metadata object from live positions
and Engine parent/child page tables. Native kernels expand SWA windows, prepare
compression/state mappings and adapt selected compressed indices once per KV
source; consuming layers reuse those tensors. Graph replay reads updated inputs,
including page tables and masks. Full-history table width is a runtime dimension,
so growing context reuses the same compiled metadata and index-conversion kernels.
Prefill also shares address geometry and unpacked compressed KV across consumers
of the same owner.

FlashMLA fuses query RoPE, sparse attention, inverse output RoPE and FP8 output
casting. KV stores fuse RMS normalization, RoPE and quantization. Sliding KV uses
528-byte FP8 rows, compressed KV 288-byte FP4 rows, and index keys 68-byte FP4
rows. Decode uses DeepGEMM's paged MQA scorer; CED consumers score candidate blocks
without gathering complete key histories or forming per-head history matrices.

With `expert_backend="deepgemm"`, DeepGEMM fuses gate selection, sparse EP
transport, routed/shared experts and combine. Lazy shifted HC carries residual
updates across both attention and FFN boundaries, materializing them for Engram,
DSpark feature taps and the final output. Serial layers reuse prepared collective
workspaces. `expert_backend="flashinfer"` retains the existing expert adapter;
selecting DeepGEMM never silently falls back.

Engram tables are row-sharded over the device domain. Ordered fetch hooks issue
all lookups together. FlashInfer execution waits at each consuming layer;
DeepGEMM joins all NCCL fetches before persistent computation because its grid
barriers require the entire GPU. These are device-stream dependencies, without
host synchronization. Modal uses NCCL owner lookups with GPU-resident shards;
host-resident RDMA Engram fetch is not implemented.

## Parallelism and graphs

The runner declares attention DP, CP, EP and expert TP support. Attention TP means
KV-head sharding; this one-KV-head MLA implementation requires `attn_tp=1`.
The DeepGEMM expert backend currently requires `attn_dp=ep=world_size`, CP1 and
expert TP1. Other supported meshes use the FlashInfer expert path.

Decode uses full CUDA graphs. Prefill graphs cover projections and FFNs;
attention, indexing, cache updates and Engram hashing execute between regions.
CP partitions query attention while retaining replicated KV, compressor and FFN
work. There is no pipeline or sequence parallelism.

## Bounded prefill replay

The runner defaults to `swa_replay_tokens=140`. Layers 0–19 process every input;
layer 20 writes complete shared C1 main/index KV before layers 20–39 keep only
the final replay suffix. The bound refers to the request's final prefill endpoint,
including reconstructed generated inputs, rather than each chunk's endpoint.
Earlier chunks use one placeholder consumer row with no SWA reads or writes.
DSpark commits retained rows at their absolute positions.

This follows DeepInfer's approximate SWA reconstruction: it preserves complete
producer KV but truncates consumer SWA dependencies. It does not reproduce a full
40-layer prefill exactly. Prompt-scoring requests execute every row. Set
`swa_replay_mode="off"` for ordinary full execution; `"mask"` retains every query
with the same bounded SWA visibility as `"compact"` for implementation comparisons.

Shared machinery knows only the final prefill lengths, a cache-layout replay
suffix, and graph region row demands. V4.1 chooses its layer boundary and masks.
Prefix hits stop before the required suffix; restored consumer bytes are ignored
until their tail is rebuilt. Producer and consumer FFNs agree separate graph
capacities across the expert domain, including idle attention replicas. CP
repartitions retained attention queries and keeps its existing output gather.

## Validation

CPU tests cover geometry, physical aliases, child slots, rollback, shifted HC,
prepared weight layouts, projection fusion and independent paged-address equations.
Run the family tests with:

```bash
uv run pytest models/cuda/tests/test_deepseek_v41*.py -q
```

The [bounded-replay checkpoint qualification](https://modal.com/apps/ekzhang/main/ap-uGTh6NFXHkXqPg0FN0LDNo)
at `43c3db4` passes on four B300s with EP4/ADP4, 1K chunks, K=3 and natural
routing/acceptance. Masked and compact execution match exactly for sampled
producer/consumer cache state, generated tokens, selected/top scores and prompt
scores on 1,281- and 2,177-token prompts with 160 generated tokens per request.
Fresh, restored, scored, unequal-active and idle-rank cohorts all pass; active
drivers each execute 44 real continuations per mode. Fresh/restored observations
also match exactly despite different chunk boundaries. These checks enable
DeepGEMM deterministic arithmetic; the locked TileKernels has no global
deterministic switch. Production timings use ordinary fast arithmetic.

The layer-36–38 diagnostic also matches every retained hidden state and HC mix.
Materialized HC boundaries use DeepInfer's fixed 16-way prenorm reduction, avoiding
the row-count-dependent arithmetic of the previous general FP32 matmul. This is
DeepInfer's TF32 path: a separate B300 operator check verifies exact row slicing
and nine graph replays, with relative L2 error `2.07e-4` against an FP64 reference.
Reports and exact source manifests are in `validation-prenorm-ep4.*`
and `hc-prenorm-probe.*` under
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-control/bounded-replay/`.
The later `63cba9e` restore gate is independently certified against the unchanged
GPU report in `validation-prenorm-ep4-restore-certification.json`. This is
implementation parity for bounded replay, not equivalence to full-history SWA
or a long-context model-quality evaluation.

The [bounded-replay CP2 probe](https://modal.com/apps/ekzhang/main/ap-3CqOIEsvN90Xzj60QQAnw2)
at `2973faf` passes on two B300s. Seven cases retain 1, 11 or 140 rows at page
boundaries and contexts beyond 128K, including an empty CP peer and an early
placeholder with no consumer SWA access. All 16 metadata fields and physical
compressed-KV indices match CPU equations. Each rank replays 28 collective graph
regions at distinct producer/consumer capacities without eager fallback. This
operator check uses synthetic tensors, not checkpoint logits; its report is
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-control/bounded-replay/cp-probe.json`.

Historical B300 operator checks covered paged indexing/CED, projections,
compressor pooling, packed attention, fused stores, vocabulary heads,
HC/Engram residuals, and shared batch metadata, including graph replay. The
one-off forward, kernel, and checkpoint-validation launchers have been retired;
the linked reports below retain their measured scope and original revisions.

Current performance measurements use the unified benchmark CLI with natural
expert routing and acceptance by default:

```bash
uv run scripts/bench.py --backend modal --model deepseek-v41 \
  --gpu B300 --world-size 4 --attn-dp 4 --attn-tp 1 --ep 4 \
  --draft-tokens 3 --prefill-chunk-size 1024 \
  --context-lengths 512,4096 --batch-sizes 1,2,4 \
  --gpu-memory-utilization 0.75 --trace-dir bench/v41-traces \
  --output bench/v41.jsonl --plot
```

This produces prefill/decode measurements and traces. It does not repeat the
retired eager-versus-captured checkpoint qualification.

The [B300 forward probes](https://modal.com/apps/ekzhang/main/ap-AMkTFWk8QDJEM5Dg1cRdQd)
and [head probes](https://modal.com/apps/ekzhang/main/ap-CRERPEvWiUcxDXtsYWV0Wn) pass
all six families, including merged/unmerged Q layouts, C1/C2 paged indexing, C2
state across chunk boundaries, packed attention, stores, FP32 logits and exact replay.
The [direct FP8 output probe](https://modal.com/apps/ekzhang/main/ap-AeDmb4KbJfjlNfVVL0wA0M)
checks H64/G8/R1024 at 1/5/1,024/1,280 rows: O-A values, packed scales and O-B
outputs match BF16-plus-cast bitwise, including five graph replays.

The [shared metadata B300 probe](https://modal.com/apps/ekzhang/main/ap-m1vDGbDRZUwEdde9ROc8Fd)
at `a4e62a1` checks seven shapes, including batch256 with four target queries or
three draft queries at 128K context and a 1,024-query masked prefill. All 18 output
fields match the CPU equations, including irregular masks and both position
dtypes. Five graph replays per case update positions, parent/child tables, masks
and selected indices. Table widths 1, 3 and 1,025 also match the equations while
reusing identical compiled metadata and conversion kernels. Its report is
`artifacts/benchmarks/deepinfer-v41-dynamic-metadata-probe.json`.

At clean commit `40a44fd`, the [full-checkpoint EP4/ADP4 B300 check](https://modal.com/apps/ekzhang/main/ap-LJDJ9gxQ1FTvfB51xQOh9U)
passes on all four drivers with the released 384 experts and natural routing.
Each driver runs one prefill and three decode steps, replaying 40 prefill compute
regions and three full decode graphs. Eager and captured execution produce
identical greedy IDs, prompt logprobs, generated-token logprobs and top-3 logprobs
(maximum absolute difference zero). This includes prepared metadata, compact
graph-control votes and event-guarded host staging. Natural K3 verification accepts
9/9 drafts on this short counting prompt; this does not establish model quality
or representative acceptance.

The same run checks Engine-owned GPU feedback with greedy sampling, scores and
token stops disabled. All four drivers emit 160 tokens identical to synchronous
captured execution, with 54 enqueues and 52 GPU continuations per driver. Generated
positions 26 through 185 cross a 128-token cache-page boundary, exercising the
transition back through page planning as well as continued graph replay. The raw
report, audited summary and completed-run log are
`artifacts/benchmarks/2026-10-22-deepseek-v41-prepared/final-validation.json`,
`final-validation-summary.json` and `final-validation.log` in the same directory.

The [native FP32 route EP2 probe](https://modal.com/apps/ekzhang/main/ap-72WXKqszkwgN9zSWQIn3gv)
at `14af36b` covers synthetic 16-expert, top-6 routing with widths 5,120/2,304 and
1,280 rows/rank, rather than the full 384-expert checkpoint. It checks route
precision, FFN numerical error and exact graph replay; its reports live beside
the joined-Engram probes below.

A B300 EP2 composition test with 384 experts and hidden/intermediate widths
5,120/2,304 reproduces the NCCL/persistent-grid conflict after all ranks finish
serial warmup: asynchronous Engram fetches cause a DeepGEMM grid-sync timeout
at four tokens. Joining both fetches first passes
eager execution, capture and 30 graph replays at four and 1,280 tokens, with exact
MoE and Engram outputs. [Joined results](https://modal.com/apps/ekzhang/main/ap-rZUgGEwWmUWHh5XBuCBcxC),
[overlap failure](https://modal.com/apps/ekzhang/main/ap-XVDWkUOmoD41DVkMtwSvL2), traces
and reproduction sources are archived under
`artifacts/benchmarks/2026-10-22-v41-forward-port-microprobes/`.

## Benchmark status and remaining work

The [historical 128K EP8 sweep](https://modal.com/apps/ekzhang/main/ap-zYfGn4MBfzHHNDRSsHqnXO)
used commit `41e7814`, K4 with three forced draft acceptances and balanced experts.
It reached 1,430.90 output tokens/s/GPU at batch256/GPU. Those measurements predate
the paged FP4 indexer, fused dense projections, cross-block lazy HC and Engram
prefetch; they do not measure the new forward pass. Raw records and PNG/SVG plots
remain under `artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-128k/`.
The [prepared-forward EP8 baseline](https://modal.com/apps/ekzhang/main/ap-lfw5WzD1XFQrU1UZ06dAXN)
at `de438e4` completes all six 128K points with K3, two forced draft acceptances
and balanced experts. At batch256/GPU, Engine steps take 140.85 ms and device work
takes 46.47 ms; projecting 75% acceptance yields 5,906.86 output tokens/s/GPU.
At batch8/GPU, the corresponding times are 60.92/21.64 ms, exposing substantial
host coordination and planning overhead. Raw records, PNG/SVG plots and all-rank
Perfetto traces live under
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-deepinfer/`.
The [Engine-overlap EP8 sweep](https://modal.com/apps/ekzhang/main/ap-gnaa8Li1vQmHzzujs8tZT4)
at `cd5ecfd` measures the same six workloads with prepared batch metadata and
continuous completion timing. Batch256/GPU takes 39.79 ms: 20,907.62 projected
tokens/s/GPU at 75% acceptance, or 19,299.34 actual tokens/s/GPU with two forced
acceptances. This is 3.54× the historical baseline. Each point records 160 native
continuations across eight drivers and zero eager fallbacks. The 48 raw rank
traces each contain two forwards; batch8/GPU drops from 5,116 to 1,436 CUDA
kernels per forward. Records, plots, source provenance and merged Perfetto traces
are under `artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-overlap/`.
See [the benchmark protocol](../../../../docs/benchmarks.md) for all six points,
timing scope and the distinction between measured and projected acceptance.

DSpark uses fixed-width greedy proposals up to the released five-token block; its confidence head is loaded
but does not choose proposal length. Adaptive draft length, host RDMA Engram,
wider EP and broader quality workloads remain separate work.
