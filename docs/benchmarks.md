# Benchmark sweeps

The sweep loads one model per GPU, keeps it resident across cases, and measures
engine completions after warmup. It writes raw measurements to JSONL;
plotting runs locally without importing PyTorch. Startup, weight loading, and
graph capture are outside the measured windows and recorded separately.
Overlap is enabled by default for capable runners; pass `--no-overlap` to time
the synchronized reference path. Overlapped decode measures continuous completion
intervals, including CPU work between completions, without per-step CUDA
synchronization or metric reductions. Warmup reductions are deferred too, so
the first measured completion cannot advance in an untimed reduction. Each point
records actual enqueue/continuation counts; static graph phase events are omitted
when a following replay could overwrite them. Perfetto traces contain two live
completions after timing, with no benchmark metric collectives between them.
The requested result file is written after normal sweep completion. Each completed
point is also flushed as a `[bench-result] {json}` log line, so earlier points can
be recovered from captured logs if a later case fails.
Before capture, the launcher logs each rank's cache budget and checks every
cohort's peak context against the native `CacheLayout.estimated_parent_count`
estimator. Decode includes warmup, measured generation, and draft write
lookahead. A case that cannot remain fully resident fails before prefill rather
than measuring a smaller admitted batch.
The chart axes and speculative projection follow
[DeepInfer's sweep plotter](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/sweep/plot_pareto.py).

Download the pinned checkpoint before allocating GPUs:

```bash
uv run modal run deploy/modal_bench.py --model deepseek-v4 --download-only
```

Run prefill and decode sweeps, then produce both charts:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v4 --gpu B300 --world-size 8 --attn-dp 8 --attn-tp 1 --ep 8 \
  --context-lengths 4096,8192 --batch-sizes 1,2,4,8 \
  --output bench/deepseek-v4.jsonl
uv run scripts/plot_bench.py bench/deepseek-v4.jsonl
```

To reuse one long prefix within each attention replica, set `--shared-prefix-tokens`.
`--context-lengths` still specifies the total context; the remainder is private:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v41 --gpu B300 --world-size 8 --attn-dp 8 --attn-tp 1 --ep 8 \
  --expert-backend deepgemm --context-lengths 131072 \
  --shared-prefix-tokens 126976 --batch-sizes 1,4,16,64,256 \
  --phases decode --prefill-chunk-size 1024 --gpu-memory-utilization 0.5 \
  --output bench/deepseek-v41-shared.jsonl
```

The shared prefix is seeded once per case outside timing. Each request receives
a distinct private suffix, and its reported cached length must exactly match the
requested prefix. Unsupported cache/checkpoint reuse fails explicitly. Records
include shared/private token counts and actual cached-length bounds; prefill
throughput counts newly processed inputs. Independent and shared-prefix points
must use separate chart files. The default is zero shared tokens.
The lower cache budget in this long-context example leaves workspace for the
captured decode graphs and uncaptured prefill operations. Cache preflight still
requires the entire requested cohort to fit; unused arena capacity does not
improve throughput.

The default DeepSeek workload uses four draft tokens, forces three accepted
drafts per verification step, and balances expert routing. These are controlled
performance measurements, rather than measurements of natural acceptance or
routing. Use `--natural-acceptance` to run the actual verifier and
`--no-balanced-experts` to preserve model routing. The JSONL records both the
configured acceptance policy and the observed acceptance rate.

V4.1 uses the pinned `DeepSeek-V4.1-Flash-1017` checkpoint and its own native text
architecture and recipe. Download and sweep it with:

```bash
uv run modal run deploy/modal_bench.py --model deepseek-v41 --download-only
uv run modal run deploy/modal_bench.py \
  --model deepseek-v41 --gpu B300 --world-size 8 --attn-dp 8 --attn-tp 1 --ep 8 \
  --expert-backend deepgemm \
  --context-lengths 4096,8192 --batch-sizes 1,2,4,8 \
  --output bench/deepseek-v41.jsonl
uv run scripts/plot_bench.py bench/deepseek-v41.jsonl
```

`--expert-backend deepgemm` selects the upstream fused MoE operation, including
sparse EP exchange and shared experts. Its initial mesh is ADP=EP=world size,
attention TP1, CP1, and expert TP1. The default `flashinfer` backend retains the
existing prepared expert path. Backend identity is recorded and checked when
combining chart points. The two-GPU B300 kernel/adapter probe passes numerical,
idle-rank, changing-route, and CUDA graph checks, including 1,280 physical rows
per rank (batch 256 with four draft positions). The separate Engine smoke needs
only two B300s and generates deterministic synthetic weights locally:

```bash
uv run modal run deploy/modal_deepinfer_kernels.py --mode engine-smoke \
  --output artifacts/benchmarks/deepseek-v41-synthetic-engine-validation.json
```

It runs the production V4.1 runner, pages, EP/ADP coordinator, scoring, and graph
paths at K0 and K4. Its unrelated draft weights exercise rejection and rollback;
the real-checkpoint check below additionally exercises accepted drafts. This
synthetic smoke does not test model quality or Engram tables.

Both DeepSeek launchers share the same checkpoint volume and completion marker.
For an older generic download whose files are already complete, publish only
the small inventory marker with
`uv run modal run deploy/modal_bench.py --model deepseek-v41 --publish-only`.
This CPU operation stats the indexed files and publishes metadata atomically;
it does not download or read the weight bodies. New downloads publish the marker
automatically. GPU sweeps require that complete inventory. The legacy
`modal_deepseek_v4.py` run/serve/benchmark paths use attention TP and now reject
these one-KV-head MLA models; their checkpoint download remains usable.

Attention TP means KV-head sharding, so both DeepSeek MLA families require
`attn_tp=1`. Earlier four-GPU TP4 debug reports sliced query heads while
replicating KV; their results do not qualify this contract. The initial V4.1
target used eight B300s with attention DP8/EP8 and replicated Engram tables.
The current runner shards Engram rows across the device domain and stores packed
FP8 sliding KV and FP4 compressed KV. On eight ranks, Engram table storage drops
from 188.83 to 23.60 GiB per GPU. Earlier full-checkpoint reports passed
eager/captured K0 on ADP2/EP2 and ADP8/EP8, and K4 on ADP8/EP8; those reports
predate the prepared DeepInfer forward operators. See the
[family memory and qualification details](../models/cuda/resin_models_cuda/deepseek_v41/README.md)
for exact allocations and the distinction between the current path and older
replicated-table/BF16-cache measurements.

Run the current prepared forward comparison with natural routing and K3:

```bash
uv run modal run deploy/modal_bench.py \
  --model deepseek-v41 --gpu B300 --world-size 4 --attn-dp 4 --attn-tp 1 --ep 4 \
  --expert-backend deepgemm \
  --draft-tokens 3 --natural-acceptance --no-balanced-experts --validate-model \
  --prefill-chunk-size 8 --context-lengths 64 --batch-sizes 1 \
  --output artifacts/benchmarks/deepinfer-v41-forward-ep4-k3-validation.json
```

Validation writes a JSON report rather than chart measurements. At clean commit
`7087a31`, this [four-B300 run](https://modal.com/apps/ekzhang/main/ap-RGV18BLKlCCRqSzjNVwMBm)
passes with the released 384 experts on all four drivers. Each driver runs four
prefill and three decode steps, including three full decode graph replays. Eager
and captured greedy IDs, prompt logprobs and generated-token logprobs match
exactly (maximum absolute difference zero). K3 naturally accepts all nine drafts
on this short counting prompt. The report is
`artifacts/benchmarks/deepinfer-v41-forward-ep4-k3-validation.json`; this focused
execution check does not establish model quality or representative acceptance.

The [FP8 output-boundary probe](https://modal.com/apps/ekzhang/main/ap-AeDmb4KbJfjlNfVVL0wA0M)
uses the released H64/G8/R1024 geometry at 1/5/1,024/1,280 rows. Its direct FP8
O-A values, packed scales and final O-B outputs equal BF16-plus-cast bitwise,
with five exact graph replays. The separate
[FP32 route EP2 probe](https://modal.com/apps/ekzhang/main/ap-72WXKqszkwgN9zSWQIn3gv)
at `14af36b` uses synthetic 16-expert top-6 routing, widths 5,120/2,304 and
1,280 rows/rank; it qualifies route precision and captured FFN execution, rather
than the full 384-expert checkpoint. Artifacts live under
`artifacts/benchmarks/2026-10-22-v41-forward-port-microprobes/`, alongside
`artifacts/benchmarks/deepinfer-v41-output-probe.json`.

The [prepared-forward EP8 baseline](https://modal.com/apps/ekzhang/main/ap-lfw5WzD1XFQrU1UZ06dAXN)
at clean `de438e4` completes six B300 points at 128K context: 126,976 shared tokens
and a 4,096-token private suffix, K3 with two forced draft acceptances, balanced
experts, five warmup steps and 20 measured steps. Device times use CUDA phase
events; Engine times include native planning and distributed coordination.
Throughput below projects 75% draft acceptance (3.25 tokens/step); measured
acceptance was fixed at two of three drafts (3 tokens/step).

| Batch/GPU | Engine ms | Device ms | Projected tokens/s/GPU |
| ---: | ---: | ---: | ---: |
| 8 | 60.920 | 21.640 | 426.79 |
| 16 | 65.611 | 22.727 | 792.55 |
| 32 | 70.023 | 23.920 | 1,485.24 |
| 64 | 77.823 | 26.926 | 2,672.71 |
| 128 | 99.315 | 33.906 | 4,188.70 |
| 256 | 140.853 | 46.469 | 5,906.86 |

At batch8/GPU, the comparable four-GPU baseline takes 41.94 ms in the Engine and
26.75 ms on device. Eight GPUs reduce device time but increase the interval
outside those device phases to 39.28 ms; at batch256/GPU that interval is
94.38 ms. This difference includes host work and synchronization waits, so it
does not measure isolated CPU execution. Raw records, source provenance, plots
and all-rank CPU/CUDA Perfetto traces are archived under
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-deepinfer/`; four-GPU
records use the sibling `2026-10-22-deepseek-v41-b300-ep4-deepinfer/` directory.
The B200 attempt failed during Quack RMSNorm graph capture with
`cudaErrorNoKernelImageForDevice`, producing no throughput result; its log and
failure metadata are under the sibling `2026-10-22-deepseek-v41-b200-ep8-deepinfer/`.

The [Engine-overlap EP8 sweep](https://modal.com/apps/ekzhang/main/ap-gnaa8Li1vQmHzzujs8tZT4)
at clean `cd5ecfd` measures the same six workloads after the CPU planning,
prepared metadata and native overlap changes. Timing covers continuous completion
intervals, including intervening host work; metric reductions run after the
window. Per-step CUDA timing events are disabled because another replay can
overwrite them before harvest.

| Batch/GPU | Engine ms | Projected tokens/s/GPU | Actual tokens/s/GPU | Baseline speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 31.099 | 836.03 | 771.72 | 1.96× |
| 16 | 32.209 | 1,614.47 | 1,490.28 | 2.04× |
| 32 | 32.784 | 3,172.28 | 2,928.26 | 2.14× |
| 64 | 32.621 | 6,376.32 | 5,885.84 | 2.39× |
| 128 | 33.660 | 12,358.75 | 11,408.08 | 2.95× |
| 256 | 39.794 | 20,907.62 | 19,299.34 | 3.54× |

Every point records 160 enqueues, 160 continuations and 160 full graph replays
across eight drivers, with no eager fallbacks. Each sequence commits 20 three-token
results; measured live context grows from 131,087 to 131,147. Projected and actual
throughput use the same measured completion intervals and differ only in their
acceptance factor (3.25 versus 3 tokens/step).

All 48 rank traces contain two complete forwards. After normalizing by that
count, batch8/GPU uses 1,436 CUDA kernels per forward versus 5,116 in the baseline;
PyTorch tensor kernels drop from 4,283 to 670. These are extra profiler traces
outside throughput timing. Their category comparison preserves per-rank results
and distinguishes summed kernel work from elapsed GPU activity. Records, plots,
source provenance, the comparison and six merged Perfetto traces are under
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-overlap/`.
These archived `cd5ecfd` results precede the compact graph-selection vote added
afterward. That change retains exact catalog revalidation on mutation; it needs
its own GPU measurement and does not change the recorded plots above.

The [compact-vote EP8 sweep](https://modal.com/apps/ekzhang/main/ap-a7qZSjXVMRlWgDgyLccAtF)
at clean `40a44fd` qualifies that change with 50 warmup and 60 measured
completions per case, using memory utilization 0.5. All six cases pass the token,
context and graph-count audit. Timing includes the safe overlap drains at a cache
page boundary.

| Batch/GPU | Engine ms | Projected tokens/s/GPU | Actual tokens/s/GPU | Published DI EP8 ms, approximate |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 25.683 | 1,012.33 | 934.46 | ~13.6 |
| 16 | 24.722 | 2,103.39 | 1,941.59 | ~14.1 |
| 32 | 26.781 | 3,883.32 | 3,584.60 | ~14.8 |
| 64 | 28.241 | 7,365.06 | 6,798.51 | ~16.4 |
| 128 | 28.526 | 14,583.18 | 13,461.40 | ~21.7 |
| 256 | 42.507 | 19,573.12 | 18,067.50 | ~33.3 |

Each case records 480 graph replays, 464 enqueues and 456 continuations across
eight drivers, with no eager fallback. Each sequence commits 180 tokens while its
context grows from 131,222 to 131,402. At B256, the three intervals around the
page crossing take 89.96, 97.68 and 87.20 ms; the median interval is 39.95 ms.
The reported 42.507 ms mean includes them. The earlier `cd5ecfd` window stayed
within one page, so its 39.794 ms B256 mean covers a different drain pattern.
Extending prepared write leases across this boundary is the first large-batch
lifecycle target, while preserving writer ownership, page retirement and failure
quarantine.

All 48 new rank traces contain two complete forwards. The graph vote now uses one
fixed record, and every rank records six fixed control gathers per forward with
no object gathers. CUDA kernel counts remain 1,436 at B8 and 1,428 at B256.
The extra traces measure GPU-active union durations of 15.63 and 38.74 ms,
respectively; these separate profiler captures cannot isolate CPU time by
subtraction from Engine timing. B8 still points to CPU coordination as the first
latency target. Coordinated decode intervals, as in DeepInfer's periodic phase
checks, could reduce phase arbitration. Completion and commit failure agreement
must preserve the Engine's page leases, prefix publication and request results.

The DeepInfer column is read from its
[published EP8 image](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/images/wide_ep_decode_pareto.png),
not raw measurements. Both sources really prefill the same 126,976 shared and
4,096 private tokens, and both charts use K3 and τ=3.25. The comparison is
indicative: DeepInfer specifies SM100 rather than an exact B300 model, uses
384+32 experts with random physical routing, natural acceptance, temperature 0.3,
top-p 0.95, forced sampler top-k 1024 and five logprobs. Resin uses 384 experts,
balanced routes, greedy sampling without scores and forced two-draft acceptance.
DeepInfer reports rank0's mean after 50 warmup, 60 logged and 60 silent intervals,
excluding periodic load-balance-stat flush intervals; Resin includes drains and
takes the DP maximum for each measured completion interval.

The remaining device opportunities below use frozen `cd5ecfd` profiler work,
averaged across eight ranks. These are bundle footprints and upper bounds on
removable work, not promised wall-time savings; some bundles overlap.

| Candidate | Kernels/forward | B8 work ms | B256 work ms | Concrete boundary |
| --- | ---: | ---: | ---: | --- |
| Sampling | 47 / 43 | 1.501 | 5.405 | Fuse runtime greedy and filtering branches; four stochastic paths currently run even for greedy rows. |
| Seven materialized HC boundaries | 98 | 0.350 | 2.331 | Use DeepInfer's shifted prenorm and pre-mix operators instead of cast/SGEMM/reduction chains. |
| Route setup | 430 | 0.661 | 0.839 | Prepare balanced layouts before capture and share the live-token mask; 344 kernels are synthetic balanced-route setup. |
| Verification-logit slice copies | 4 | 0.018 | 0.347 | Let sampling read the existing row strides. |
| Markov vocabulary additions | 3 | 0.015 | 0.304 | Accumulate the projection into logits through DeepGEMM's `out_acc` contract. |
| FP8 casts, total footprint | 92 | 0.161 | 0.287 | Emit attention normalization's FP8 values/scales directly where the consumer permits it. |

Sampling changes must retain Resin's top-k-before-top-p distribution and per-request
Philox seed/offset stream. DeepInfer's sampler uses different filtering and RNG
contracts, so copying it requires an explicit adapter and independent validation.
Shifted HC must apply the previous pre-mix; the unshifted fused operator is not a
drop-in replacement. Route tensors must be prepared outside capture and retained
through graph teardown.

Raw records, exact token/counter audits, source provenance, plots, image
digitization, protocol details and six merged Perfetto traces are under
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-compact/`. The full failed
`d6074ac` attempt is preserved there separately: it produced no timing point and
exposed reusable inference-tensor buffers, corrected before `40a44fd`.

The subsequent [EP8 control-path sweep](https://modal.com/apps/ekzhang/main/ap-sSIr6M6gIVKDIHJSURJzMn)
at clean `43b9333` retains the same six rank agreements, but transports each fixed
record with gather-to-source followed by broadcast. It also reads aggregate
graph counters instead of rebuilding all per-shape diagnostics twice per timed
step. The latter removes benchmark instrumentation overhead; the full diagnostics
API remains available. Neither change alters model kernels or native completion,
cache ownership, and failure agreement.

| Batch/GPU | Engine ms | Projected tokens/s/user | Projected tokens/s/GPU |
| ---: | ---: | ---: | ---: |
| 8 | 15.897 | 204.44 | 1,635.55 |
| 16 | 16.835 | 193.05 | 3,088.88 |
| 32 | 17.900 | 181.56 | 5,809.95 |
| 64 | 20.667 | 157.25 | 10,064.17 |
| 128 | 27.246 | 119.28 | 15,268.43 |
| 256 | 41.626 | 78.08 | 19,987.31 |

All cases use eight B300s, K3, forced two-draft acceptance, balanced routes,
50 warmup and 60 measured completions, with the same 128K shared/private prompt
split and all page-boundary drains included. Every case retains 480 decode
replays, zero eager fallback, and 464 enqueues / 456 continuations across drivers.
Charts still project 75% acceptance with tau 3.25; actual B256 throughput is
18,449.82 tokens/s/GPU at three committed tokens per step. Raw records, the
updated interactivity/throughput plot, and rank traces are in
`artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-control/frontier/`.

A separate B8-only qualification on the same source measured 16.474 ms versus
25.683 ms in the previous sweep. Its extra traces reduce the mean gap between
graph bodies across eight ranks from 10.895 to 0.170 ms, with no Gloo activity
remaining inside that gap. All ranks retain the same graph kernel names and
launch dimensions, and 1,436 kernels per forward. These are separate profiler
captures and allocations; profiled graph-body durations are not subtracted from
unprofiled completion timing. That focused run, CPU transport/counter probes,
and gap analysis are retained in the parent `ep8-control/` directory.

The same source's warmed, uncached prefill probe uses one prompt per GPU at
EP8/ADP8 and 1,024-token chunks. It measures 0.300 seconds for 4,096 tokens and
9.774 seconds for 131,072 tokens (about 13,410 input tokens/s/GPU). These are sums
of timed Engine steps after one complete warmup prompt at each length, excluding
weight loading, graph capture and benchmark metric reductions; they are not HTTP
TTFT measurements. Raw prefill records are in `ep8-control/prefill.jsonl`.

At clean `a77b14a`, the same full-model probe with larger chunks measures:

| Chunk tokens | 4K prompt seconds | 128K prompt seconds | 128K input tokens/s/GPU |
| ---: | ---: | ---: | ---: |
| 1,024 (baseline) | 0.300 | 9.774 | 13,409.65 |
| 4,096 | 0.119 | 4.009 | 32,696.57 |
| 8,192 | 0.117 | 3.691 | 35,511.46 |

The larger-chunk source removes history/table widths from metadata and candidate
kernel specialization, and parallelizes prefill table expansion across queries.
A separate B300 qualification checks exact outputs and mutable graph replay
while context grows to 128K, with unchanged compiled-kernel counts. Chunk/query
sizes remain legitimate specializations. All prefill points still execute the
full backbone; records and qualification are in `ep8-control/prefill-chunks/`
and `ep8-control/runtime-width/` respectively.

This prefill result executes all 40 backbone layers over every input token.
DeepInfer's published prefill curve instead uses EP4/ADP4, CP1, 4K–64K packed
tokens per rank, and 140-token bounded replay. Its layers 0–19 process the full
prompt; layer 20 prepares full compressed main/index KV, while layers 20–39
process retained queries only. The upstream
[deployment notes](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md#kv-cache-hits-and-recomputation)
describe this as approximate consumer SWA reconstruction, nearly halving prefill
computation; exact reconstruction would require a 2,560-token tail. The
approximate consumer state is used for decode and excluded from prefix caching.
Matching that protocol in Resin also requires a consumer-sized distributed
graph bucket, full KV writes at the boundary, absolute DSpark positions, and
full execution when prompt scores are requested. Increasing chunk size alone
does not implement bounded replay or establish parity with that curve.

At `43c3db4`, [bounded consumer prefill on eight B300s](https://modal.com/apps/ekzhang/main/ap-mQBVegCNI0UvzOL1ncAMmq)
uses the same EP8/ADP8, 8K chunks, balanced routing and warmed whole-prompt timing
as the preceding full-model measurement:

| Prompt tokens per GPU | Full prefill (s) | Bounded prefill (s) | Bounded input tokens/s/GPU |
| ---: | ---: | ---: | ---: |
| 4,096 | 0.1166 | 0.0903 | 45,348.55 |
| 131,072 | 3.6910 | 1.9509 | 67,184.86 |

The 128K result is 1.892× faster. It includes all 16 sequential chunks, full
producer/C1 writes, retained consumer queries and DSpark state commits. It uses
ordinary fast kernel arithmetic, with no deterministic validation overrides.
Every FFN region replays its agreed graph; the final 140-row consumer projection
regions use their intentional eager `padding` fallback. The model's bounded
SWA reconstruction is approximate; these timings do not establish numerical
equivalence or a controlled match to DeepInfer's EP4 curve. Raw records and
provenance are in `ep8-control/bounded-replay/prefill-prenorm.jsonl` and
`prefill-prenorm-summary.json`.

The same source passes separate EP4/ADP4 checkpoint qualification with natural
routing and acceptance, 1K chunks and DeepGEMM deterministic arithmetic. Masked
and compact replay match exactly across fresh, restored, prompt-scored,
unequal-active and idle cohorts; prefix restoration also matches across changed
chunk boundaries. The late-layer HC projection uses DeepInfer's fixed-reduction
TF32 kernel to avoid the former row-count-dependent FP32 matmul arithmetic.
See the [model validation notes](../models/cuda/resin_models_cuda/deepseek_v41/README.md#validation)
for measured scope, CP2 coverage and raw evidence.

The implementation uses fixed-width greedy proposals, loads the confidence head
without applying its length policy, and supports up to the released five-token
block. Benchmark defaults remain four drafts. Wider meshes and longer contexts
need separate qualification; see the
[family limits](../models/cuda/resin_models_cuda/deepseek_v41/README.md).

Available model names are `deepseek-v4`, `deepseek-v41`, `gpt-oss-20b`, and
`gpt-oss-120b`. The GPT-OSS benchmark path currently uses the Hopper expert kernel
and has no draft model:

```bash
uv run modal run deploy/modal_bench.py \
  --model gpt-oss-20b --gpu 'H100!' --world-size 1 --attn-dp 1 --attn-tp 1 --ep 1 \
  --draft-tokens 0 --prefill-chunk-size 1024 --output bench/gpt-oss-20b.jsonl
uv run scripts/plot_bench.py bench/gpt-oss-20b.jsonl --draft-tokens 0
```

Use `--gpu 'H100!'` for an H100 comparison. Modal can automatically upgrade
plain `H100` requests to H200; the exclamation mark
[disables that upgrade](https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s).
Records always retain the actual device name.

Use `--phases decode` or `--phases prefill` for one phase and the matching
`--mode decode` or `--mode prefill` when plotting. Parallelism is configured with
`--world-size`, `--attn-dp`, `--attn-tp`, `--cp`, and `--ep`. The launcher defaults
to eight GPUs, attention DP8, attention TP1, CP1, and EP8 for DeepSeek; override
all relevant fields when selecting another model. Batch sizes count
requests **per attention-DP replica**; the global batch is `batch_size * attn_dp`.
`--prefill-chunk-size`, `--prefill-graph`, and `--decode-graph` select execution
policies. Defaults are a 4,096-token prefill chunk, breakable prefill graphs,
full decode graphs, five warmup steps, and twenty measured steps. Prefill may
measure additional steps to finish its request cohort.

## Decode traces

Add `--trace-dir bench/traces` to save Perfetto-compatible Chrome JSON traces.
After each decode timing window, every attention-DP driver profiles one extra
full-cohort step with CPU and CUDA activities, including individual CUDA graph
nodes. This step is excluded from measured durations, acceptance, counters and
context bounds. Prefill and checkpoint-validation runs are not traced.

The launcher downloads compressed raw rank traces and a merged
`decode-ctx<length>-b<batch>-all-ranks.json.gz` for each case. The merger aligns
PyTorch timestamp bases and separates rank process/flow IDs. Open the merged
file in [Perfetto](https://ui.perfetto.dev/). Source and workload metadata stay
in the trace. Raw files can also be merged again locally:

```bash
uv run scripts/merge_bench_traces.py bench/traces
```

## Reading the charts

The plotter saves `decode.png`, `decode.svg`, `prefill.png`, and `prefill.svg`
under the JSONL directory's `plots/` folder. Use `--output-dir` to change it.
Different context lengths form separate curves.

To compare meshes for the same model, pass their result files together with
`--compare-topologies`. Each topology becomes a separate curve; model revision,
source, kernels, graph and workload policies must still match. Decode comparisons
allow different prefill capacities between meshes because prefill runs outside
the timed decode window:

```bash
uv run scripts/plot_bench.py bench/gpt-oss-120b-{tp1,tp2,tp4,ep2,ep4}.jsonl \
  --compare-topologies --draft-tokens 0 --mode decode \
  --output-dir bench/gpt-oss-120b-comparison
```

The decode chart has per-user output tokens/s on the horizontal axis and output
tokens/s/GPU on the vertical axis. Point labels are requests per attention-DP
replica. Its acceptance setting is an explicit projection of measured step
latency:

```text
τ = 1 + draft_tokens * accept_rate
per-user tokens/s = measured_steps / window_seconds * τ
output tokens/s/GPU = per-user tokens/s * global_batch_size / gpu_count
```

Plot defaults are `--draft-tokens 4 --accept-rate 0.75`, hence four committed
tokens per step. The draft count must match the measured verification width.
Changing `--accept-rate` rescales throughput using the same measured latency;
it does not rerun verification or change the recorded observed acceptance.
The chart footer shows both projected and observed acceptance. Actual measured
output throughput remains available as
`committed_tokens / window_seconds / gpu_count` in the raw data.

The prefill chart uses mean milliseconds per measured prefill step horizontally
and actual input tokens/s/GPU vertically. Point labels show actual unique input
tokens per global step, rather than the configured token budget. These counts
exclude replication across attention TP/CP ranks.

## Raw data and comparisons

Each JSONL row has `schema_version: 1`, a phase, context length, local/global
batch sizes, GPU count, attention-DP size, every measured step duration, and the
sum of those durations as `window_seconds`. `input_tokens` counts actual unique
prefill inputs; `prefill_tokens` is their measured per-step mean.
`committed_tokens` always counts generated outputs. `draft_tokens`,
`accepted_drafts`, `measured_accept_rate`, and `balanced_experts` describe the
workload. A null `accepted_drafts` means natural verification.

Balanced routing covers physical router rows, including any padding used by a
captured prefill bucket. Successive assignments cycle through EP owners first,
then through experts within each owner, so even a small batch exercises balanced
communication. Attention-DP replicas receive offset portions of that assignment;
other attention ranks apply the same ordinal rule to their router rows. The real router GEMM still runs,
and selected router scores still determine weights.
`graph_counts` sums replay and fallback counter changes over unique attention-DP
drivers during measurement, excluding warmup. Distributed timings use the slowest replica driver per step;
token counts are summed over unique drivers. `window_seconds` sums timed
`Engine.step` calls, excluding admission and benchmark metric reductions.
`context_length` is the requested prompt length; the
`measured_context_start_min/max` and `measured_context_end_min/max` fields record
the actual contexts after warmup and measurement as decode advances.

Metadata records the model, checkpoint revision, Git commit and source digest,
hardware, parallelism, graph policies, checkpoint quantization, and prefill
chunk size. The plotter rejects incompatible configurations within a phase
instead of connecting them as one curve. Keep separate result files for
different models, meshes, quantization, graph policies, or acceptance policies.
The source digest also prevents combining different uncommitted code under the
same Git commit. Repeated identical sweep points use their latest record;
validation still checks all earlier rows.

The reusable measurement API is `resin.bench.run_sweep`; the Modal launcher is
`deploy/modal_bench.py`. The harness observes the real runner and scheduler;
it does not substitute a synthetic latency model. Plot tests use explicitly
synthetic fixtures only.

## Recorded decode sweeps

The 2026-10-22 GPT-OSS-120B comparison uses actual H100 80GB HBM3 GPUs,
checkpoint revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, and clean runtime
commit `b2f7c3c`. All five meshes use independent 8,192-token prompts, balanced
experts, no drafts, full decode graphs, a 32-token prefill chunk, and 0.95 memory
utilization. Each point retains its complete 20-warmup/100-measurement window;
measured contexts advance from 8,212 to 8,312 tokens. All 25 points retain the
full requested cohort and report zero measured decode fallbacks.

TP meshes use attention TP equal to world size, attention DP1, and EP1. EP meshes
use attention DP and EP equal to world size, with attention TP1. CP is 1 throughout.
TP1 sweeps batches 1/2/4/8/14; the other meshes sweep 1/4/16/32/64. Point labels
count requests per attention-DP replica, so EP4 batch 64 means 256 global requests.

| Mesh | Batch at best measured throughput | Output tokens/s/GPU | Per-user tokens/s |
| --- | ---: | ---: | ---: |
| TP1 | 14 | 574.750 | 41.054 |
| TP2 | 64 | 488.915 | 15.279 |
| TP4 | 64 | 286.481 | 17.905 |
| EP2 | 64 | 988.479 | 15.445 |
| EP4 | 64 | 1,013.632 | 15.838 |

The largest per-point timing CV is 4.05%; the largest half-window drift is
4.93%. These are complete fixed-window means. Raw JSONL files and PNG/SVG charts
are saved locally under `artifacts/benchmarks/2026-10-22-gpt-oss-120b-h100-*`;
the combined chart is in the `comparison` directory and the summary is
`2026-10-22-gpt-oss-120b-h100-summary.json`.

The separate V4.1 EP8/ADP8 sweep uses eight B300s at runtime commit `41e7814`,
131,072-token contexts with a physically shared 126,976-token prefix and a
4,096-token private suffix, and fixed three-of-four draft acceptance. Its
five-warmup/20-measurement windows reach 1,430.895 output tokens/s/GPU at batch
256 per replica, or 5.589 tokens/s/user. Its source and protocol are retained
separately in `artifacts/benchmarks/2026-10-22-deepseek-v41-b300-ep8-128k` and the
[family qualification record](../models/cuda/resin_models_cuda/deepseek_v41/README.md).

## CUDA model primitive checks

The 2026-10-23 B300 probes cover GPT-OSS routing, DeepSeek-V4 indexer preparation,
Inkling short convolution, and Qwen normalization/DeltaNet preparation. These
are synthetic primitive measurements, not full-model throughput or interactivity
results. CUDA-event intervals include repeated submissions from Python, so they
can also include submission gaps. Compilation and graph capture precede timing.

| Primitive | Shape | Timing mode | Before (µs) | After (µs) |
| --- | --- | --- | ---: | ---: |
| GPT-OSS 20B top-4 router | 8 rows, 32 experts | Graph replay | 14.39 | 4.12 |
| GPT-OSS 20B top-4 router | 1,024 rows, 32 experts | Graph replay | 18.45 | 12.31 |
| DeepSeek-V4 selected page rows | 1,024 queries, 131,072 context | Graph replay | 16.40 | 4.13 |
| DeepSeek-V4 indexer RoPE | 1,024 queries, 131,072 context | Graph replay | 20.50 | 14.38 |
| Qwen RMSNorm | 1,024 × 5,120 | Graph replay | 77.63 | 6.18 |
| Qwen gated RMSNorm | 1,024 × 5,120 | Graph replay | 112.62 | 10.28 |
| Qwen DeltaNet block | Batch 1, 1,024 tokens, synthetic width 256 | Graph replay | 186.30 | 151.40 |
| Inkling short convolution | Batch 1, 4,096 tokens, width 4,096 | Eager launches | 496.74 | 122.11 |

The 128-expert GPT router was already fused; the new production path is the
32-expert router. V4 reuses its existing decode RoPE and page-map kernels during
prefill. Qwen uses fused normalization and grouped value-head kernels without
materializing repeated Q/K heads. Numerical probes compare independent reference
paths and changed-input graph replays; Inkling also verifies chunk continuation
and exact recurrent-state updates.

The context-length audit found no kernel factory specializing on every exact
context length in these paths. Inkling's new short-convolution kernel was checked
across odd lengths 127/255/1,023/4,095/8,191 and power-of-two lengths
128/256/1,024/4,096/8,192. Each layout class compiled once, then reused that
specialization for the remaining lengths. This does not imply that different
layouts, graph buckets, or kernel scheduling choices never compile separately.

Raw results and source digests are in
[`artifacts/cuda-model-improvements/result.json`](../artifacts/cuda-model-improvements/result.json)
and the successful Qwen rerun
[`qwen_result.json`](../artifacts/cuda-model-improvements/qwen_result.json).
The first file records the initial Qwen probe failure; the rerun is the Qwen
qualification record. The reproducible probes are
`deploy/cuda_index_routing_probe.py`, `deploy/inkling_prefill_probe.py`, and
`deploy/qwen3_5_kernel_probe.py`.

## Shared sampler checks

On a CUDA environment with deployment dependencies, run:

```bash
uv run pytest tests/test_sampling_cuda.py -q
```

These checks cover request-local random streams across batch permutations and
padding, high seed and offset bits, conditional top-k/top-p distributions, and
mutable graph inputs. The graph catalogue captures batches 1 through 128 in a
shared pool, and the vocabulary smoke uses 201,088 classes. Environments without
CUDA skip these tests; the CPU sampler contract tests remain in the normal suite.

Filtering uses FlashInfer with deterministic nucleus normalization. A shared
categorical kernel owns the random draw: the full request seed is its Philox key,
while sampling offset and vocabulary index occupy separate counter words. This
keeps the stream independent of physical batch position, including when a graph
replays with new request state.
