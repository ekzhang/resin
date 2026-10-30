# Benchmark sweeps

`resin.bench` supplies the reusable measurement framework. Run it through
`scripts/bench.py` on a local CUDA host or with `--backend modal`; both backends
use the same model registry, runner construction, measurements, and JSONL schema.
`scripts/plot_bench.py` renders charts, and `scripts/merge_bench_traces.py` combines
rank traces for Perfetto. `deploy/modal_bench.py` is a thin remote adapter.

## Prompt workloads

Benchmarks default to chat-templated ShareGPT conversations from a saved token
artifact. Prepare one on CPU with `python -m resin.bench_prompts prepare --help`,
then pass `--prompt-corpus PATH` or set `RESIN_BENCH_PROMPT_CORPUS=PATH`. Artifacts
pin the corpus, tokenizer, chat template and exact token IDs. Missing artifacts
fail before GPU dispatch. Use `--prompt-source synthetic` for random-token
controls, and keep those results separate from ShareGPT measurements.

Sampling defaults to greedy. Set `--temperature`, `--top-p`, `--top-k` and `--seed`
explicitly for sampled runs, for example `--temperature 1.0 --top-p 0.95
--top-k 64 --seed 41`. Resin applies top-p before top-k. Natural draft acceptance
remains the default.

Gemma models default to 8,192-token prefill chunks with breakable prefill graphs
and full decode graphs. Override these with `--prefill-chunk-size`,
`--prefill-graph` and `--decode-graph`.

## Run a sweep

Download pinned weights on a Modal CPU worker, then sweep prefill and decode:

```bash
uv run scripts/bench.py --model deepseek-v41 --backend modal --download-only
uv run scripts/bench.py --model deepseek-v41 --backend modal \
  --prefill-context-lengths 4096,8192 --decode-context-lengths 8192 \
  --output bench/deepseek-v41.jsonl --trace-dir bench/traces --plot
```

For local GPUs, point to an existing checkpoint:

```bash
uv run scripts/bench.py --model gpt-oss-20b --backend local \
  --checkpoint /path/to/gpt-oss-20b \
  --prefill-context-lengths 4096,8192 --decode-context-lengths 8192 \
  --output bench/gpt-oss-20b.jsonl --trace-dir bench/gpt-oss-traces
uv run scripts/plot_bench.py bench/gpt-oss-20b.jsonl
```

The default backend is `local`, which requires `--checkpoint`. Models using a
separate assistant also accept `--assistant-checkpoint PATH`. Modal uses pinned
checkpoint volumes from `deploy/models.py`; `--publish-only` adopts complete
existing files by validating and atomically publishing their inventory markers.
It does not download or read weight bodies. Serving uses the same inventories.

Run `uv run scripts/bench.py --help` for the available model keys and options.
The registry in [`deploy/models.py`](../deploy/models.py) defines each preset's
pinned checkpoints, hardware, parallelism, draft count, and memory defaults.
Override topology with `--world-size`, `--attn-dp`, `--attn-tp`, `--cp`, and `--ep`,
and Modal hardware with `--gpu`. The registry's `validate_gpu` checks hardware
overrides against the preset's kernel architecture; runners validate the selected
topology. Record files retain the actual device name and parallel layout.

Use `--phases prefill`, `--phases decode`, or `--phases prefill,decode` (default).
The default prefill curve uses **4K, 8K, 16K, 32K, and 64K prompt tokens**, with
one request per attention replica. Decode defaults to an **8K initial context**
and batches of **1, 2, 4, 8, 16, 32, and 64**. During iteration, use
`--prefill-context-lengths 4096,8192` before expanding to the full curve. Set
`--prefill-context-lengths` / `--decode-context-lengths` and
`--prefill-batch-sizes` / `--decode-batch-sizes` independently. The existing
`--context-lengths` and `--batch-sizes` set both phases; phase-specific options
take precedence. Choose a sweep that fits the model and available cache memory.
Use `--decode-batch-sizes 1,2,4,8` for a smaller initial sweep, then extend through
16/32/64 to find the throughput plateau and its interactivity cost. Cache
preflight rejects an oversized cohort instead of silently measuring fewer users.
`--skip-capacity-exceeded` records and skips cases that cannot fit, while keeping
the requested engine capacity. Prefill cohorts are capped at 131,072 input tokens
by default: `context_length * batch_size` per attention replica. Adjust this with
`--max-prefill-tokens`; zero disables the cap. Budget exclusions are recorded
separately from KV-capacity failures.
One resident sweep shares a context capacity large enough for its longest case
plus generation headroom; records include `context_capacity`. Run the full
prefill curve and the 8K decode curve separately when a larger sparse-index or
graph capacity would change decode cost:

```bash
uv run scripts/bench.py --model glm-5.3-flash --backend modal \
  --phases prefill --prefill-chunk-size 8192 --output bench/glm-prefill.jsonl --plot
uv run scripts/bench.py --model glm-5.3-flash --backend modal \
  --phases decode --output bench/glm-decode.jsonl --plot
```

Batch sizes count requests **per attention-DP replica**; the global batch is
`batch_size * attn_dp`. Throughput is divided by total GPU count, including TP/CP
participants. For example, attention-DP4 with batch one at 8K runs four concurrent
8192-token prompts, totaling 32768 input tokens per complete cohort.
`--prefill-chunk-size`, `--prefill-graph`, and `--decode-graph`
control execution policy. Prompt length is the total input; chunk size is the
maximum work submitted per request in an Engine step. The benchmark CLI defaults
to the model's chunk size or 4K, whichever is larger, subject to kernel limits
(single-GPU GPT-OSS also limits the flattened batch). This benchmark default is
independent of a smaller serving chunk chosen for prefix reuse or scheduling.
Use `--prefill-chunk-size 4096` or `8192` to compare chunk policies where supported;
each plotted latency still covers the **whole prompt**, including every chunk.
The default graph policies are breakable prefill and
full decode. Decode uses five warmup steps and twenty measured steps unless
overridden with `--warmup-steps` and `--measure-steps`. Prefill separately uses
one complete warmup cohort and three complete measured cohorts, controlled by
`--prefill-warmup-steps` and `--prefill-measure-steps`. These prefill options count
whole cohorts, independently of how many Engine chunks each cohort needs.

## Timing and synthetic workloads

Each sweep loads the model once per GPU and keeps it resident across cases.
Startup, weight loading, graph capture, and request admission are untimed.
Prefill processes complete request cohorts at each context length. The low-level
`run_sweep` API also retains the legacy minimum-step mode when no explicit
prefill cohort counts are supplied. Prefill latency is
the sum of timed Engine steps for a complete cohort, excluding admission and
between-step benchmark reductions. Prefill throughput counts actual new input
tokens, excluding attention-TP/CP replication and any reused prefix.

Decode prefills outside timing, warms up, then measures the requested number
of completions while the full cohort stays resident. Context grows during
warmup and measurement. Overlap is enabled where supported; `--no-overlap`
selects synchronized execution. Overlapped decode measures continuous completion
intervals, including CPU work between completions, without per-step CUDA
synchronization or benchmark reductions. Metrics are reduced after the window.

Before capture, the launcher checks the full cohort against the cache budget,
including generated positions, speculative lookahead, and trace headroom. A
case that cannot fit fails before timing rather than benchmarking a smaller
admitted batch. Each completed point is printed as `[bench-result] {json}` for
recovery from logs if a later case fails.

Natural draft acceptance and model expert routing are the defaults. To measure
a controlled speculative workload, explicitly select the number of draft
positions and accepted drafts:

```bash
uv run scripts/bench.py --model deepseek-v41 --backend modal \
  --phases decode --context-lengths 8192 --batch-sizes 1,4,16 \
  --draft-tokens 4 --accepted-drafts 3 --balanced-experts \
  --output bench/deepseek-v41-fixed.jsonl
```

`--accepted-drafts` changes actual verification/commit and rollback behavior;
`--balanced-experts` changes routing while retaining router computation and
weights. These are synthetic performance workloads, not model-quality results.
Their settings and observed acceptance are recorded. Use separate result files
for natural and synthetic policies. `--draft-tokens 0` measures ordinary decode
when the model supports disabling its draft path.

To reuse a long prefix within each attention replica:

```bash
uv run scripts/bench.py --model deepseek-v41 --backend modal \
  --gpu B300 --world-size 8 --attn-dp 8 --attn-tp 1 --cp 1 --ep 8 \
  --phases decode --context-lengths 131072 --shared-prefix-tokens 126976 \
  --batch-sizes 1,4,16,64,256 --prefill-chunk-size 1024 \
  --gpu-memory-utilization 0.5 --output bench/deepseek-v41-shared.jsonl
```

`context_length` remains the total prompt length. The prefix is seeded once
outside timing, and every request gets a distinct private suffix. Each admitted
request must report exactly the requested cached length. Independent and
shared-prefix measurements must use separate chart files. The default is zero
shared tokens. DeepSeek requires attention TP1; the DeepGEMM expert path
requires ADP=EP=world size, CP1, and at least two GPUs.

## Prefill and decode traces

`--trace-dir DIR` captures extra passes after each measured case. Prefill
traces run two fresh complete cohorts, including every chunk of the requested
context. Decode traces run two live completions, or three with overlap. Traced
work does not alter measured timing, acceptance, graph counters, or context
bounds, and no benchmark metric collectives run between traced passes.

Decode traces reserve output and context headroom beyond the requested completion
count, so reaching `max_output_tokens` does not drain overlap during capture.
The Engine's termination guard is unchanged. A final forward may remain pending
after the last completion and is included by the profiler's final synchronization;
attribute costs to observed graph/kernel counts, not completion count alone.
Historical traces near termination retain their original scope and measurements.

Traces include CPU/CUDA activity from **attention-DP driver processes**. This
covers all ranks for DeepSeek's ADP=world-size layout. TP/CP follower processes
remain in their worker loop and are not independently profiled. The records
should not be interpreted as every GPU's trace for those layouts.

Each case produces raw `<phase>-ctx<length>-b<batch>-rank<rank>.json.gz` files
and a merged `<phase>-ctx<length>-b<batch>-all-ranks.json.gz`. The Modal backend
downloads them to the local directory. Local sweeps create a fresh run subdirectory
beneath `--trace-dir`, preventing stale ranks from an earlier sweep from being
merged into new traces. Pass that subdirectory to the merger when remerging.
The merger aligns PyTorch timestamp
bases and isolates process/flow IDs; open the merged file in
[Perfetto](https://ui.perfetto.dev/). Source/workload metadata and raw files are
retained. Merge downloaded or local traces again with:

```bash
uv run scripts/merge_bench_traces.py bench/traces
```

## Reading the charts

`--plot` renders the measured phases after the sweep. The standalone plotter
writes `prefill.png/svg` and `decode.png/svg` beneath the input directory's
`plots/` folder; use `--output-dir` to choose another destination. For a
single-phase result, pass the matching `--mode prefill` or `--mode decode`.
Different context lengths form separate curves; point labels count requests
per attention-DP replica.

Prefill compares mean complete-cohort latency in milliseconds against actual
input tokens/s/GPU. `prefill_seconds` records one duration per measured cohort,
so a long chunked prompt is not mislabeled with one chunk's latency. Older
schema-1 records without this field still plot their original step timings,
with a legacy label, and cannot be mixed with cohort timings.

Decode plots use actual committed output by default:

```text
output tokens/s/GPU = committed_tokens / window_seconds / gpu_count
per-user tokens/s = committed_tokens / window_seconds / global_batch_size
```

An explicit `--accept-rate` instead projects throughput from measured step
latency at an assumed acceptance rate:

```bash
uv run scripts/plot_bench.py bench/deepseek-v41-fixed.jsonl \
  --mode decode --accept-rate 0.75
```

```text
τ = 1 + recorded_draft_tokens * projected_accept_rate
per-user tokens/s = measured_steps / window_seconds * τ
output tokens/s/GPU = per-user tokens/s * global_batch_size / gpu_count
```

The draft width is read from the measurements; `--draft-tokens` can optionally
assert that width. Projection rescales the same timings and does not rerun
verification or change observed acceptance. Its footer identifies projected
and observed acceptance. Conditional work, such as accepted-prefix rollback, may
cost more at higher acceptance; a fixed-cycle-time projection does not simulate
that additional work. The chart conventions follow
[DeepInfer's sweep plotter](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/sweep/plot_pareto.py).

To compare topologies with matching model/source/kernel/workload policies:

```bash
uv run scripts/plot_bench.py bench/gpt-oss-120b-{tp1,tp2,tp4,ep2,ep4}.jsonl \
  --compare-topologies --mode decode --output-dir bench/gpt-oss-120b-comparison
```

Decode comparisons can use different prefill capacities between meshes because
prefill runs outside the decode timing window. Within a curve, incompatible
configurations are rejected rather than connected. Repeated identical points
use their latest record; earlier rows are still validated.

## Raw records and reusable API

Each JSONL row has `schema_version: 1`, phase, requested context length,
local/global batch sizes, GPU count, attention-DP count, and every measured
`step_seconds`. `window_seconds` is their sum. Prefill adds `prefill_seconds`
for complete cohorts. `input_tokens` counts actual unique prefill inputs;
`prefill_tokens` remains their per-step mean for compatibility.
`committed_tokens` counts generated output.

`draft_tokens`, `accepted_drafts`, `measured_accept_rate`, and
`balanced_experts` describe the workload. Null `accepted_drafts` means natural
verification. Shared-prefix fields record requested and observed cached tokens.
`graph_counts` and `overlap_counts` exclude warmup/tracing. Distributed timings
use the slowest replica driver and sum tokens over unique drivers. Actual
context bounds are retained as `measured_context_start_min/max` and
`measured_context_end_min/max`.

Metadata contains model/checkpoint/source identity, actual hardware, topology,
graph policies, expert backend, quantization, prefill budget, and startup times.
The source digest prevents combining different uncommitted source under one
Git commit. Timing measures real Engine execution, not a synthetic latency model.

For a resident sweep, `scripts.bench.run_benchmark(..., progress_path=...)` can
retain completed cases as an atomic JSON array if a later worker or case fails.
The path must be new; successful runs also add NUMA placement to that snapshot.
This optional progress file is separate from the CLI's appended JSONL results.

For an existing Engine, wrap the model runner in `BenchmarkRunner` before
Engine construction and graph capture, then call `resin.bench.run_sweep` with
`BenchCase` values. The caller owns Engine lifetime. Supply
`resin.bench.perfetto_trace(directory, rank, metadata)` as `trace_step` to enable
the same untimed traces. Multi-driver runs must coordinate with
`distributed_reducer` and invoke the same trace passes on every driver.
`perfetto_trace(..., decode_passes=N)` advertises its count automatically. For a
custom callback, pass `trace_decode_passes=N` to `run_sweep` when using a different
count; exceeding the reserved count fails before another forward is submitted.

## HTTP correctness and latency checks

Use the reusable server checker against an already running endpoint:

```bash
uv run scripts/check_server.py https://YOUR-ENDPOINT \
  --output artifacts/server-check.json
```

It checks JSON/SSE responses, logprobs, concurrent requests, repeated-prefix
cache hits, and a context-size TTFT sweep. `--image` adds a built-in image test
(or accepts a path/URL); `--reasoning-effort low high` tests those request
settings. `--stress` adds the existing HTTP stress/robustness suite. Unsupported
capabilities are reported as skipped where evidence cannot be collected; use
`--skip-prefix-cache` or `--skip-logprobs` for intentionally disabled features.
For recurrent models, the repeated prompt must extend past a reusable state
checkpoint. Increase `--prefix-cache-repeats` from its default 256 phrases when
needed, keeping the prompt inside the server's context limit. The report records
the repetition count, actual prompt tokens, and cached tokens for both requests;
a zero-hit repeat still fails the check.

The context sweep uses fresh early prefixes to avoid cache hits hiding prefill
work. Requested sizes are repeated-word counts; records include actual server
`prompt_tokens`. Adjust sizes and latency limits explicitly:

```bash
uv run scripts/check_server.py https://YOUR-ENDPOINT \
  --context-lengths 128 512 2048 --repeats 2 \
  --max-ttft-ms 10000 --ttft-ratio 3 --ttft-slack-ms 2000 \
  --output artifacts/server-contexts.json
```

TTFT includes HTTP/network overhead and does not establish whether kernels
recompiled. If the server exports a cumulative compilation counter, pass
`--compilation-metric NAME` and run against an idle deployment. The checker
requires the counter to remain unchanged across the sweep; without that
counter the compilation assertion is explicitly skipped. Exact context-limit
checks require `--context-length LIMIT` and a compatible chat-tokenization
endpoint; Resin currently has no `/tokenize` endpoint, so these checks skip.

For a small MMLU accuracy check, supply a standard subject CSV from your chosen
source/split, with provenance saved alongside it. CSV rows have no header and
contain the question, choices A–D, and the answer letter. No benchmark dataset
is bundled or downloaded:

```bash
uv run scripts/check_server.py https://YOUR-ENDPOINT \
  --eval-file eval/mmlu100.jsonl --eval-limit 100 --eval-concurrency 32 \
  --eval-max-tokens 256 \
  --min-accuracy 0.5 --output artifacts/server-eval.json
```

JSONL is also supported: each line contains
`{"question": "...", "choices": ["...", "..."], "answer": "A"}`; answers may
be zero-based indices. The checker records answers and accuracy, fails missing/unparseable answers,
and applies the selected minimum. A small subset is a regression check, not a
full MMLU score. By default the checker evaluates up to 100 questions with at most
32 requests in flight; both settings are configurable with `--eval-limit` and
`--eval-concurrency`. Supply at least 100 questions for a 100-question check,
preferably a fixed sample across subjects. Reports retain input question order,
the actual question count, configured concurrency, and per-request errors even
when responses finish out of order. JSON reports separate passed, failed, and skipped checks.

## Recorded V4.1 measurements

The following qualification and performance reports describe their original
source revisions. The one-off Modal validation/probe launchers have been
retired; these artifacts are historical evidence, not results of the current
benchmark CLI.

At clean commit
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

That historical Resin sampler used top-k-before-top-p and a per-request Philox
seed/offset stream. The later [fused sampler](#shared-sampler-checks) deliberately
changes filtering order and stochastic choices; comparisons must identify which
contract they use. DeepInfer's sampler also has different filtering and RNG
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
block. These historical benchmark runs used four drafts; current defaults come from
the shared model registry. Wider meshes and longer contexts
need separate qualification; see the
[family limits](../models/cuda/resin_models_cuda/deepseek_v41/README.md).

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
qualification record. The one-off primitive probes have been retired. Current numerical and graph
contracts live in `models/cuda/tests/` and `tests/`; the archived results above
retain their original scope and source provenance.

## Shared sampler checks

On a CUDA environment with deployment dependencies, run:

```bash
uv run pytest tests/test_sampling_cuda.py tests/test_sampling_fused_cuda.py \
  tests/test_sampling_filters_cuda.py tests/test_sampling_blocks_cuda.py \
  tests/test_sampling_guard_cuda.py -q
```

These checks cover request-local random streams across batch permutations and
padding, high seed and offset bits, conditional top-k/top-p distributions, and
mutable graph inputs. The graph catalogue captures batches 1 through 128 in a
shared pool, and the vocabulary smoke uses 201,088 classes. Environments without
CUDA skip these tests; the CPU sampler contract tests remain in the normal suite.

`FusedSampler` is the shared CUDA runner default. Parallel radix selectors handle
runtime top-k/top-p for large vocabularies, while a single-block selector handles
small vocabularies. Inactive rows skip vocabulary filtering. Two tiled kernels
then choose a vocabulary tile and token using one per-request Philox CDF draw.
Top-p uses the full temperature-scaled distribution, followed by the top-k cap,
retaining all cutoff ties. This changes the previous top-k-before-top-p behavior. There
are no CUDA graph conditional nodes or host reads of sampling policy tensors.

The full request seed is the Philox key, with counter words
`(0, 0, offset_low32, offset_high32)`. All 32 bits of the first random word form
the uniform draw; scalar targets and conditional fractions use FP64, while
vocabulary weights and cumulative sums use FP32. The stream is independent of
physical batch position. The engine owns offset advancement, including speculative
columns. Each call owns its scratch, so captured columns and separate streams do
not share mutable sampler buffers. Parameters and strides stay runtime inputs;
changing temperature, k, p, seed, offset, or mask does not compile another variant.

CDF sampling replaces the former vocabulary-wide Gumbel draws and intentionally
changes seeded token choices. A request still produces the same choices when
sampled alone, reordered, padded, or combined into a verification block. Top-p
uses FP32 exponential weights. The native AIR selector reconstructs deterministic
FP32 bucket masses; the small-vocabulary selector accumulates FP64 mass. Tests
compare against independent sorting, FP64 distributions and Philox arithmetic,
including FP16/BF16 logits, cutoff ties, nonfinite fallback, large finite logits, shifted
pointers, changed-input graph replay and the two real vocabulary sizes.
When the inclusive top-k set has less than `(top_p - 0.001)` of the full mass,
top-p cannot remove any of those tokens. FP64 mass partials certify this shortcut;
AIR skips its vocabulary work for that row. Rows near the boundary still run
the complete selector.

`FusedSampler.sample_block(logits, sampling, start_step=0)` handles verification
logits shaped `[requests, positions, vocabulary]` in one sampler invocation.
The plan-based `sampler(logits, plan)` entry point also accepts this block shape
and prepares the request policies once.
The first two dimensions flatten as a view, including padded vocabulary strides;
non-viewable storage is rejected instead of silently copied. One small kernel
expands all request metadata and sets each column's offset to
`base + (start_step + column) * SAMPLING_OFFSET_STRIDE`. The filtering and draw
kernels then process the complete block. `graph_sample_block` exposes this at the
shared runner boundary. GLM, DeepSeek, Gemma, Inkling, and Qwen verification
callers use the block path in eager execution and CUDA graphs. Draft calls that
depend on previously sampled tokens still run sequentially. Qwen verification
begins at `start_step=1`; the other verification blocks begin at zero. Qwen
retains the states produced during verification and selects the accepted prefix
after block sampling. DeepSeek broadcasts the complete sampled block once.

### Sampler qualification (2026-10-25)

The final CDF snapshot `f402251c` was tested on an NVIDIA B200 and an NVIDIA
H100 80GB HBM3 (SM90). Each passed all 133 CUDA tests with zero skips, all
64 paired benchmark cases, and five verification-block comparisons. The CPU
sampler and runner suites passed 302 tests; one CUDA-dependent test was skipped
locally. These are sampler microbenchmarks, not end-to-end model speedups.

The matrix uses FP32 logits, vocabularies 154,880 and 201,088, batches 1/8/64/256,
and eight policies covering greedy, unfiltered, top-k, top-p and mixed requests.
Times are the median of five event-timed trials of 16 calls per CUDA graph,
excluding compilation, model execution and host scheduling. The immutable
baseline is commit `56176e5`. Every measured median improved against that
baseline; 48 cases per GPU preserve the filtering policy. The other 16 include
the requested top-p-before-top-k behavior change.

Representative B8, vocabulary 154,880, microseconds per complete batch call:

| GPU | Policy | Previous sampler | Fused CDF sampler |
| --- | --- | ---: | ---: |
| B200 | Temperature 1, no filters | 252.34 | 29.45 |
| B200 | Top-k 40 | 194.22 | 67.68 |
| B200 | Top-p 0.9 | 181.02 | 93.13 |
| B200 | Top-k 40 + top-p 0.9† | 194.10 | 70.50 |
| H100 | Temperature 1, no filters | 345.56 | 33.15 |
| H100 | Top-k 40 | 512.37 | 68.12 |
| H100 | Top-p 0.9 | 184.02 | 105.18 |
| H100 | Top-k 40 + top-p 0.9† | 200.76 | 72.85 |

† Joint filtering changed order, so this row compares old and new behavior rather
than claiming a speedup for identical filtering semantics. Seeded token choices
also change with the CDF draw; request-local batching invariance is tested.

Four-position unfiltered verification, vocabulary 154,880, microseconds per
complete block. B8 means eight requests and 32 sampled positions. The previous
parallel Gumbel block uses the same current filters and metadata as CDF, so that
comparison isolates the draw implementation:

| GPU | Requests | Legacy columns | Current CDF columns | Parallel Gumbel block | CDF block |
| --- | ---: | ---: | ---: | ---: | ---: |
| B200 | 1 | 623.82 | 100.74 | 25.11 | 24.27 |
| B200 | 8 | 961.99 | 143.05 | 61.45 | 44.41 |
| B200 | 64 | 2527.06 | 350.08 | 305.58 | 162.06 |
| H100 | 1 | 557.06 | 107.51 | 27.79 | 26.79 |
| H100 | 8 | 1286.54 | 167.37 | 69.30 | 49.45 |
| H100 | 64 | 3316.52 | 413.11 | 347.08 | 179.61 |

CDF won all five paired block comparisons on both GPUs. The B8 unfiltered block
uses 12 CUDA kernels versus 53 for current separate columns and 81 for legacy
columns. The model callers above now use the block API; full-model performance
curves were not rerun for this sampler-only qualification.

Independent checks cover policy support, FP64 CDF references, full-width Philox
counters, cutoff ties, replay with changed parameters, and absence of new native
or Triton variants after policy/stride changes. Exact oracle comparisons omit
numerically ambiguous draws within a documented `2e-7` CDF boundary band;
all sampled tokens still receive support checks. Each GPU's 28 saved traces,
source hashes, compiler/resource logs and raw timing records are retained under
[`artifacts/shared-sampler/f402251c/`](../artifacts/shared-sampler/f402251c/).
