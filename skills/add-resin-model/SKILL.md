---
name: add-resin-model
description: Add or port a Resin model runner for specified GPU hardware, then qualify correctness, graph-safe speculative decoding, compilation stability, and performance. Use for a new model/hardware integration or completing its qualification, including optional Modal serving.
---

# Add a Resin model

Deliver a working Engine integration and evidence for its correctness and speed on
specified hardware. Follow the workflow below, continuing from existing work when
appropriate. A working forward pass alone does not complete model integration.

Run repository commands from the repository root. Links below are relative to this
skill directory; inspect the current implementations and command `--help` before
using them.

## 1. Research the model and target hardware

Establish the exact checkpoint/revision, weight and KV precision, GPU architecture
and count, intended context lengths/concurrency, media support, and available
compute budget from the request and repository. If a larger model needs multiple
GPUs or the parallelism choices are unresolved, ask the user which hardware counts
and TP/attention-DP/CP/EP topologies are in scope. Continue architecture research and
memory estimates while awaiting that choice; do not ask again for settled choices.

Research current primary sources: checkpoint configuration/reference code, model
papers, upstream kernel implementations, and available draft models, MTP heads or
other speculators. Record revisions, supported GPU architectures/dtypes/layouts,
licenses, and the baseline implementations to compare. Check attention, positional
encoding, MoE routing, recurrent/sliding state, vocabulary, and multimodal details.

Reuse an existing recipe when its formatting, parsing and media semantics match.
When working in the Resin repository or a fork, implement a new native Rust recipe
in [`resin-recipe`](../../crates/resin-recipe/README.md) and its bindings. When
importing Resin as a library without its source, implement the
[`resin.recipe.Recipe`](../../resin/recipe.py) protocol in Python instead; a Rust
change is not required. Keep formatting and parsing in the recipe rather than
duplicating them inside a deployment script.

Before implementation, read the relevant parts of:

- [Runner authoring](../../docs/model-forward-pass.md): model preparation,
  shapes, graphs, cache ownership, speculation, and validation.
- [CUDA package boundaries](../../models/cuda/ARCHITECTURE.md): where model and
  hardware code belongs.
- [Distributed execution](../../docs/distributed.md), when using several GPUs.

## 2. Implement and qualify the forward-pass Runner

Implement the reference-correct forward path and checkpoint loading in the owning
model package, importing or implementing the best suitable kernels. Keep cache
allocation, scheduling, prefix ownership and request lifetime in the Engine. Use
existing runtime staging, graph, sampling and distributed machinery. Extend the
shared [model registry](../../deploy/models.py) for hardware presets and runtime
construction; do not create a per-model deployment file or a new Modal probe suite.

Compare logits and greedy tokens against an independent trusted reference before
optimizing. Exercise prefill, decode, continuation, cache/page boundaries, and the
agreed parallelisms. Distributed tests must include unequal and idle attention
replicas while every required collective participant remains active. Keep durable
kernel/model regression tests in the ordinary test suite.

**Every GPU kernel path must be CUDA graphable, and changing context length or
sampling parameters must not cause recompilation.** Treat these as acceptance
requirements, including imported upstream kernels:

- Keep logical lengths, positions, page tables, routing inputs and sampling values
  in mutable runtime data. Do not use context length, temperature, top-p, top-k,
  seed or other ordinary request values as JIT specialization keys.
- Prepare plans, libraries and graph-owned buffers before capture. Replay must
  use current data without allocation, compilation, hidden host synchronization
  or stale state. Check eager/replay parity and changed-input replay at short,
  tile/page/checkpoint-boundary and near-capacity histories.
- Distinguish kernel graphability from the configured capture policy. Full and
  breakable graphs may cover different regions; report actual replay coverage
  and remaining eager regions. An existing eager-only kernel does not waive the
  graphability requirement or qualify itself through a neighboring graphed region.
- Warm the bounded, genuinely distinct algorithm/dtype regimes during startup.
  Warming an exhaustive grid of context lengths or sampling combinations does
  not satisfy the no-recompilation requirement.
- After startup, run previously unseen context lengths, batch shapes and sampling
  combinations, then repeat them. Observe compiler invocations/cache misses and
  stable artifact identities on every participating rank. A warm second request
  does not excuse compilation on the first unseen shape. Avoid timing-based
  artifact signatures such as mtimes.
- Measure TTFT spikes too, but use compiler events/counters or attributable
  before/after artifact evidence to establish compilation stability. Similar TTFT
  is not proof. If tooling lacks this evidence, add reusable instrumentation or
  leave that qualification explicitly unresolved.

Read [CUDA graph contracts](../../docs/cuda-graphs.md) and
[kernel artifact caching](../../docs/kernel-cache.md) for the relevant invariants.

## 3. Implement fast, graphed and overlapped speculative decoding

Use the research to select the supported speculator and implement draft generation,
target verification, acceptance, commit and rollback inside the runner. Reuse
[`resin.backend.speculative`](../../resin/backend/speculative.py) and the shared
CUDA completion/overlap machinery; do not introduce a separate scheduler.

Keep the GPU draft/verify path graphable and prepare its buffers before capture.
Use the runner's asynchronous enqueue/completion contract to overlap work without
host round trips or synchronization that serializes the pipeline. Demonstrate real
graph replays and enqueue/continuation overlap in counters and traces; setting a
flag is insufficient. Keep synchronous forward callbacks faithful to their
contract; use the distinct enqueue/completion path for asynchronous execution.
Use explicit events and buffer ownership, and serialize access to shared graph
workspaces. In particular, overlap does not permit concurrent cross-stream replay
of the shared prefill-region buffers.

Account for maximum uncommitted verification lookahead in cache retention. Commit
only accepted tokens and roll back rejected full-attention, sliding and recurrent
state before reuse. Test zero/all/partial acceptance, EOS within a draft block,
cache-boundary crossings, cached prefixes, and concurrent requests. At temperature
zero, require exact token parity with target-only decoding. Compare block
verification logits/state with sequential target steps; also test seeded
stochastic sampling and its acceptance semantics.

Measure natural acceptance and committed output tokens/s against target-only
execution. Tune draft width and overlap using phase timings and traces; a lower
verification latency with worse acceptance is not a speedup. Forced acceptance is
an explicitly labeled synthetic benchmark mode, never correctness or model-quality
evidence. If the model has no usable speculator or cannot support overlap yet,
explain the concrete limitation and keep this step open instead of silently
claiming the full integration is qualified.

## 4. Validate the Engine through its server

Place the runner in the Engine and start its native HTTP server on the agreed GPU
host. This can be a temporary test server; a public Modal deployment is not needed
for validation. Use [`scripts/check_server.py`](../../scripts/check_server.py)
and the normal runtime tests, rather than adding model-specific diagnostic CLIs.

Run ordinary JSON/streaming requests, token usage/logprobs, concurrency, context
sweeps, prefix reuse, supported media, reasoning effort, and stress/recovery. Include
cached-versus-fresh continuation parity, not just a positive cache-hit counter.
Select media/reasoning options that the model actually supports and record skips.
Validate context boundaries with tokenization evidence where available.

For example, after setting `SERVER_URL` to the test server and `MMLU_JSONL` to a
locally available fixed MMLU sample with at least 100 questions across subjects:

```bash
uv run scripts/check_server.py "$SERVER_URL" \
  --output artifacts/model-check.json --stress \
  --context-lengths 128 512 2048 8192 \
  --eval-file "$MMLU_JSONL" --eval-limit 100 --eval-concurrency 32 --min-accuracy 0.5
```

Choose longer contexts, supported `--image` and `--reasoning-effort` checks, and an
accuracy floor appropriate to the model and reference. The example floor is not a
universal quality bar. Record dataset provenance, selected questions, prompt
format, answer extraction, score and reference score; a small subset is a smoke
evaluation, not a full MMLU claim. Inspect wrong/missing answers before tuning speed.

The checker reports cached tokens and TTFT, but this server does not currently
expose a universal tokenizer endpoint or compilation counter. Use
`--context-length` with an available `--tokenize-path`, and `--compilation-metric`
only with an actual counter. A skipped check is not a passed requirement: use the
runner-level evidence from step 2 to close gaps in HTTP observability.

## 5. Reach competitive prefill and decode curves

Use the [performance optimization skill](../optimize-resin-performance/SKILL.md)
for the profiling and iteration loop, including lessons from earlier Resin work.

Use [`scripts/bench.py`](../../scripts/bench.py), backed by `resin.bench`, and
[`scripts/plot_bench.py`](../../scripts/plot_bench.py). Read the
[benchmark guide](../../docs/benchmarks.md) for timing and topology semantics.
Start with an 8K curve and traces on the target GPU configuration:

```bash
uv run scripts/bench.py --model "$MODEL_KEY" --backend modal \
  --context-lengths 8192 --decode-batch-sizes 1,2,4,8 \
  --output artifacts/model-bench.jsonl --trace-dir artifacts/model-traces --plot
```

Use `--backend local --checkpoint PATH` on an existing GPU host; provide a separate
assistant checkpoint when needed. Modal benchmarks use the shared pinned model
registry. Download/publish its checkpoint first with the shared tooling. Choose
sizes that fit the agreed hardware; the example sweep is not a fixed target.
Iterate prefill at 4K/8K first, then measure the default 4K/8K/16K/32K/64K prompt
curve. Use 4K/8K chunks where supported and record them separately from total
prompt length; each prefill timing must include the complete prompt.

Deliver both complete-cohort prefill latency versus input tokens/s/GPU and decode
interactivity (output tokens/s/user) versus output tokens/s/GPU. Batch counts are
per attention-DP replica; normalization includes every TP/CP GPU. Cover target-only
and natural speculative execution. Optionally use the plotter's `--accept-rate`
for an analytical projection, labeling it separately from observed natural
acceptance.

Capture and inspect Perfetto traces for both phases: normally two untimed complete
prefill passes and two decode completions after measurement. The existing profiler
covers attention-DP drivers, not TP/CP follower processes. Where follower behavior
is material, obtain corresponding traces/instrumentation and state their scope.

**The target is state-of-the-art performance for both curves on the specified
hardware.** Compare with the fastest verified applicable baselines using matching
checkpoint/precision, context, concurrency, topology, cache reuse, speculation and
timing boundaries. Research current results and record source dates/versions;
unmatched published numbers do not establish a win. Establish the comparison
workloads and baseline set before tuning, and report quantitative gaps across both
curves rather than inventing a universal tolerance. Iterate from traces on
launches, memory traffic, synchronization, collective costs and acceptance until
both curves meet that target. Re-run affected correctness checks after kernel or
speculation changes. If hardware access, an agreed budget or remaining defects
prevent completion, report the gap and next experiment rather than claim success.

The final report must link the prefill and decode plots, raw JSONL and both phases'
Perfetto traces, alongside correctness/evaluation and compilation-stability
evidence. Include checkpoint/source/kernel revisions, hardware/topology, benchmark
commands, natural acceptance, baseline comparisons, and unresolved limitations.

## 6. Deploy to Modal if requested

Use [`deploy/modal_serve.py`](../../deploy/modal_serve.py) with `RESIN_MODEL`
selecting the shared registry entry. Keep checkpoint download/publication separate
from GPU startup; add configuration to the registry instead of a new deployment
script. When deployment is already authorized, proceed after qualification without
asking again. Otherwise stop at the local implementation and report; do not infer
public deployment intent from a request to add a model.

```bash
RESIN_MODEL="$MODEL_KEY" uv run modal run deploy/modal_serve.py
RESIN_MODEL="$MODEL_KEY" uv run modal deploy deploy/modal_serve.py
```

After deployment, check the endpoint with the same server-validation command and
include its URL in the report. Distinguish a passing local runner from a verified
remote endpoint.
