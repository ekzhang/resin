---
name: optimize-resin-performance
description: Profile and optimize Resin inference performance using reproducible prefill/decode curves, Perfetto traces, compilation evidence, and repository lessons. Use for slow TTFT, decode latency or throughput, CPU/collective gaps, kernel fusion, graph coverage, speculative overlap, or a measured performance regression in an existing integration.
---

# Optimize Resin performance

Find the limiting work in the requested workload, change it, and demonstrate the
result without weakening correctness. Start from the current implementation and
the user's target hardware, latency/throughput goals and compute budget. New model
integration and qualification are covered by
[add-resin-model](../add-resin-model/SKILL.md); this skill supplies its performance
iteration. A focused optimization does not require reimplementing the whole model.

Run commands from the repository root. Read
[the recorded lessons](references/lessons.md) for cases relevant to the observed
symptom. Those are dated evidence and leads, not current performance promises or
instructions to recreate retired probes. Inspect the cited commits with
`git show HASH` and check what survived on the current branch.

## Establish the comparison

Record source commit and dirty changes, checkpoint/kernel revisions, exact GPU
model/count, precision/layouts, TP/attention-DP/CP/EP, graph policy, overlap, context
and shared-prefix lengths, batch per attention replica, chunk size, sampling,
draft width and observed acceptance, routing policy and timing boundaries.
Include CPU quota/affinity, NUMA placement and NCCL settings when investigating
host or communication behavior. Prefer paired A/B on the same allocation; if the
machine or environment changes, label the comparison accordingly.

Use [`scripts/bench.py`](../../scripts/bench.py) and
[`resin.bench`](../../resin/bench.py); extend shared tooling when necessary instead
of adding a model-specific Modal diagnostic file. Consult `--help` and
[the timing protocol](../../docs/benchmarks.md). For example, on an existing GPU
host with `MODEL_KEY` and `CHECKPOINT` set:

```bash
uv run scripts/bench.py --model "$MODEL_KEY" --backend local \
  --checkpoint "$CHECKPOINT" --context-lengths 8192 \
  --decode-batch-sizes 1,2,4,8 --output artifacts/perf-baseline.jsonl \
  --trace-dir artifacts/perf-baseline-traces --plot
```

Use a separate output path for each candidate. Add an assistant checkpoint or
choose Modal and topology options as the workload requires. Start with a bounded
representative case, then expand to the affected curve. Prefill uses complete
cohorts; do not call one chunk's time full-context latency. Decode uses continuous
completion intervals, including intervening host work, with no per-step metric
reductions or synchronization that drains overlap. Include page-boundary drains
in the measured window. Report startup/compilation/capture separately from warm
prefill and also test first-use TTFT when investigating serving latency.

Start decode tuning and Perfetto captures at **8,192 context tokens**.
Iterate prefill at **4K and 8K prompts**, then expand to the default
**4K, 8K, 16K, 32K, 64K** curve. Compare 4K/8K execution chunks where supported;
report prompt length and chunk size separately and time complete prompts.
Serving defaults such as a 512-token chunk are not the prefill performance target.
For decode, extend batches per attention replica through **16, 32, and 64** when
memory permits; 1/2/4/8 alone can miss the throughput plateau. Record configured
batch/graph capacity as well as live batch, since padding can change small-batch
cost. Use consistent capacity for paired comparisons and preflight the full cohort.
Get 8K decode fast before expanding to 32K or 131K when useful and feasible. Keep short
contexts and nearby boundaries for correctness checks; avoid redundant 128/512/2K
performance traces unless they expose a distinct bottleneck or reproduce a bug.

Primary throughput counts actual committed output tokens with natural acceptance.
Plot-only `--accept-rate` projections estimate other acceptance rates from measured
cycle time; label them separately from measured throughput. Historical forced
acceptance runs used a retired runtime option and require separate comparisons.
Batch is per attention-DP replica and GPU normalization includes TP/CP workers. For external
baselines, read current primary sources and match actual computation, hardware,
precision, routing, sampling and timing. A digitized curve with mismatched policy
is indicative, not proof of parity or state-of-the-art performance.

A plot-only acceptance projection holds measured cycle time fixed. It does not
simulate changed execution cost: accepted-prefix rollback, cache repair or other
conditional work may become more expensive as more drafts are accepted. State
this limit when optimizing such a path, and use measured before/after throughput
for the primary speedup claim.

## Locate the limiting interval

Capture a few complete prefill/decode passes outside throughput timing and inspect
Perfetto across participating ranks. The current profiler traces attention-DP
drivers; TP/CP followers require separate instrumentation when material. Put
follower profiling around their matching worker forwards, identify rank/process
and case, and coordinate start/stop outside the timed window. Retain raw rank
files before merging; report missing ranks rather than treating a driver trace as
the whole mesh. Verify
how many complete forwards a trace contains before normalizing counts. Separate
summed kernel durations, GPU-active union, critical path and wall-clock completion
intervals: overlapping kernel work is not additive savings. Engine minus GPU
timing is unattributed time, not a CPU measurement; especially do not subtract
different profiler and benchmark runs.

Keep steady-decode traces away from request termination. Reserve output and
context headroom beyond the traced completions and verify continuation counters
during capture: reaching the output-budget guard intentionally drains overlap.
That tail can expose CPU coordination and graph-launch gaps even when the timed
window used normal overlap. Use separate labeled captures to study retirement.
Correlate CPU graph launches with actual GPU graph intervals before treating a
Gloo span as exposed latency; the next launch may already be queued behind work.

Use the evidence to choose the next experiment:

| Observed symptom | Check before changing code |
| --- | --- |
| Long gap between fast graph bodies | Host staging/boxing, repeated plan validation or metadata preparation, control collectives, admission work, detailed diagnostics in the timed loop. |
| New context/sampling values stall only once | Compiler events, specialization keys, cache misses and per-rank artifact identity; distinguish JIT from new graph capture and allocation. |
| Many tiny kernels around large kernels | Repeated masks, casts, table expansion, routing setup, sampler branches, copies and strided slices; prepare once or fuse a measured bundle. |
| Periodic decode spikes | Page crossings, speculative lookahead capacity, overlap drains, prefix publication, cancellation/EOS and uneven/idle replicas. |
| Prefill throughput far below a baseline | Chunk size, graph row geometry, cache reuse, producer/consumer work and algorithmic approximation before kernel substitutions. |
| Fewer kernels but slower completion | NCCL waiting/topology/channels, memory pressure, launch contention, changed acceptance or instrumentation; compare the same workload and allocation. |
| HTTP much slower than engine curves | Actual acceptance, overlap counters with stop tokens and idle ranks, prompt preparation, parser work, queueing and errors. Queue capacity absorbs bursts; it does not raise GPU throughput. |

Rank candidates by removable work on the critical path and likely whole-workload
effect. Keep observations, hypotheses and measured wins separate. Do not optimize
a kernel solely because its name looks expensive or add all profiler bundle
durations as a promised speedup. Use a small ablation when changing multiple
layers would otherwise hide which change helped.

## Apply Resin's proven boundaries

**Prepare once, transport compactly.** Reuse batch metadata across layers and
target/draft paths when their semantics match. Keep native plan/CSR arrays and
flat output columns through Python/Rust boundaries; use reusable pinned staging
instead of lists, scalar boxing and repeated validation. Send immutable request
media/sampling once and typed mutable step data thereafter. A metadata cache needs
explicit invalidation on changed inputs, layout or lifecycle. Keep Engine-owned
allocation, leases and commit in the Engine; model-specific attention layout and
replay policy belong in the runner. See [the boundary](../../docs/kernel-boundary.md).

**Keep context and sampling out of JIT keys.** Logical history lengths, page-table
widths, positions, temperature, top-k/top-p and seeds are runtime data. Audit
Triton `constexpr`, TileLang factory/cache keys, launch dimensions and upstream
wrappers. Distinguish bounded algorithm/dtype regimes from request-value
specialization. After startup, run unseen context lengths and sampling combinations
on every participating rank and demonstrate no extra compilation with compiler
events/counters or attributable artifact identities/digests. Snapshot after startup
warmup and after each unseen case; record rank/process, cache namespace and the
kernel/compile event involved. A shared-volume file count alone cannot identify
which rank compiled. If artifacts do not expose compilation reliably, instrument
the compiler/cache-miss entry point or leave that claim unresolved. Warm repeat TTFT,
mtime-only comparisons, and an exhaustive warmup grid do not establish this.
Also verify numerical parity on unseen lengths; see
[kernel caching](../../docs/kernel-cache.md).

**Graph useful regions without changing the computation.** GPU kernel paths must
be graphable, but the selected full/breakable policy determines actual capture
coverage. Verify replay/fallback counters. Measure staging cost and padded work;
larger buckets may change GEMM rounding as well as runtime. An intentional exact-row
prefill fallback is different from an unprepared graph or an eager-only kernel.
If target/speculative parity requires matching projection shapes, select that
shape from the executing batch bucket, not the server's concurrency limit; raising
the limit should not enlarge existing small-batch work. Check both numerical
parity and padding cost across the supported buckets.
Keep graph inputs/outputs at stable addresses, update mutable metadata, and consume
or copy shared output views before replay overwrites them. Shared prefill-region
buffers execute serially; overlapping streams cannot replay them concurrently.
See [graph policy and ownership](../../docs/cuda-graphs.md).

**Separate matched-batch parity from cross-batch variation.** Compare target-only
and speculative outputs with matched physical batch buckets, prompts, sampling,
source and cache state. Different batch shapes or retirement schedules can change
floating-point reductions and greedy wording even in target-only decoding. Check
that target-only control before spending effort forcing cross-batch bitwise
identity; require such invariance only when the user or workload needs it. Keep
matched-batch parity, cache ownership, graph coverage and completion checks strict.
Ordinary numerical variation does not excuse dropped tokens, stale state or
incorrect stop handling.

**Overlap through the existing completion contract.** Use the Engine's
enqueue/completion and device-feedback path rather than weakening synchronous
`forward`. Prove continuations actually occur with natural acceptance, EOS and
idle attention replicas. Retain stopped/cancelled rows and page leases until
already-launched work finishes; reserve speculative writes and readable history
across boundaries. Preserve collective participation, rank agreement on readiness,
execution/commit failures, and quarantine after uncertain mutation. Stable decode
may reduce transport work, not erase those agreements. Some persistent kernels
need the whole GPU: obey measured stream dependencies instead of overlapping every
NCCL operation. See [distributed execution](../../docs/distributed.md).

**Fuse with the real numerical contract.** Preserve accumulator precision, packed
scale/weight strides, sentinel ordering, masks, recurrent state and rollback.
Prefer existing fused kernels or implement a fused GPU operation for repeated
pointwise, routing, packing and state-update chains. CUDA graphs still launch each
unfused operation; capture alone does not make raw Torch chains efficient. Keep
independent Torch references for testing and use proven GEMM/library kernels where
appropriate.
Stage quantization/layout conversion at load time where possible. For sampling,
preserve the documented filtering order (currently top-p on the full scaled
distribution, then the top-k cap, retaining cutoff ties), normalization and
request-local Philox seed/offset semantics under batch permutation and padding.
A copied upstream kernel may use a
different distribution or memory layout. Microbenchmark an independent reference,
then measure the full workload; fewer kernels alone do not qualify a win.

**Keep instrumentation from becoming the bottleneck.** Read lightweight aggregate
counters during timing; retain detailed diagnostics outside it. GPU event storage
must survive until its corresponding completion, rather than being overwritten by
the next replay. Metrics collection/scraping must not synchronize or introduce
collectives. Enable capture-time instrumentation before capture and quantify its
overhead when interpreting small differences. See [metrics](../../docs/metrics.md).

## Qualify the change and close the loop

Use two stages within the agreed budget: a representative paired experiment and
affected-contract checks first, then wider coverage for a candidate that survives.
Repeat paired measurements when a claimed gain is close to run-to-run variation;
show the spread and avoid calling an unstable or noise-sized difference a win.
Stop a losing hypothesis and choose the next trace-supported experiment.

Rerun the baseline and candidate with matching measurement windows. Include
representative small/large batches, short/long contexts and nearby tile/page
boundaries; target the tests to the code touched. For speculation/cache/layout or
graph changes, exercise eager/replay, fresh/restored prefix, changed-input replay,
target-only/speculative greedy tokens, acceptance extremes, EOS/cancellation and
unequal/idle ranks. Include logprobs and prompt scoring when affected. Compare
independent numerical references, not just two paths sharing the changed kernel.
Do not loosen tolerances to hide graph padding or parallel reduction drift.
Approximate algorithms need an explicit quality comparison and must remain labeled
as approximate; they cannot establish exact full-compute parity.

For serving changes, use [`scripts/check_server.py`](../../scripts/check_server.py)
against the test endpoint for prefix reuse, varying-context TTFT, concurrent
requests, media/reasoning where supported, stress/recovery and the relevant MMLU
subset. The checker's prefix test verifies telemetry; separately compare
cached-versus-fresh outputs when qualifying cache semantics.
HTTP TTFT does not prove absence of JIT, and a small MMLU subset does not establish
full model quality. A missing capability or skipped check stays unresolved.

Expand successful candidates to both prefill and interactivity/throughput curves
where relevant, retaining raw results and traces. Keep a failed or cancelled run
as such; never interpolate an unmeasured point into a claimed win. If the proposed
change regresses the requested workload, revert your experiment or keep it clearly
optional rather than generalizing it. Do not revert unrelated user changes.

Report the bottleneck evidence, what changed, before/after absolute values and
relative differences, correctness/compilation checks and remaining gaps. Link raw
JSONL, plots and Perfetto traces for the measured phases, with exact commands and
source/environment provenance. A focused decode investigation may report prefill
as unmeasured; full model qualification still requires both curves. State the next
decisive experiment if access or the agreed budget prevents completion. Do not
publish a new endpoint merely to finish a performance investigation.
