# Architecture

Resin separates request and resource ownership from model-specific execution.
The Python-independent `resin-core` crate contains the complete transactional
`EngineCore<P>` lifecycle. `P` is opaque frontend metadata: native code schedules
tokens without knowing about Python sampling objects or model media. Payloads
are cloned once per dispatched row; native applications should use `Arc` for
large shared data.

`resin-py` converts Python inputs, binds the runner once, and adapts native plans,
errors, handles, and events. The core uses `rkyv` for the
distributed plan wire format. Native applications can use `resin-core` without
Python, Torch, or HTTP.

Shared Resin also owns native conversation rendering, incremental parsing and
CPU media preparation in `resin-recipe`, plus serving protocols and streaming
in `resin-http`. These components sit above the token-based Engine; they are
part of the library, while GPU vision encoding remains in model runners.

The Rust Engine owns:

- request admission, priority, batching, and prefill chunking;
- request progress and completion;
- one unified paged cache allocator;
- full-history, sliding-window, and recurrent-state retention;
- exact-token LRU prefix reuse;
- cache reader/writer leases across runner execution and commit;
- retry or quarantine decisions after runner failures.

The Python runner owns:

- model weights and device allocations;
- conversion of `BatchPlan` page indices into kernel metadata;
- attention, recurrent, collective, sampling, and speculative-decode kernels;
- CUDA graphs and execution synchronization.

The boundary is intentionally asymmetric: scheduling complexity belongs in the
Engine, while a forward pass receives immutable token and page metadata and can
execute straightforwardly.

Separately installable model packages already use this boundary to add models
and select kernels. The [model authoring reference](model-forward-pass.md)
documents preparation and execution requirements. Step-by-step tutorials and
an authoring skill are planned.

## Lifecycle

1. The runner publishes a frozen `RunnerConfig`.
2. The Engine creates the declared `CacheLayout` and calls
   `bind_resources(ResourceBinding)` once.
3. `submit()` records tokens, generation limits, priority, batch key, and
   sampling policy. With prefix caching enabled, it attaches the longest aligned
   cached prefix.
4. The scheduler selects a phase- and key-compatible cohort, respecting token
   and cache capacity.
5. The cache coordinator grows each request, creates ordered group page tables,
   and acquires one batch lease.
6. The Engine emits a `BatchPlan` and either calls `forward()` or enqueues an
   owned device completion while retaining the exact requests and page lease.
7. The Engine validates the complete raw result against the leased write span
   before applying stop tokens or generation limits. It prepares events, releases
   terminal model state, publishes aligned prefixes, retires cache leases, and
   commits progress before requeuing survivors.

Internally, preparation owns the selected requests and cache capabilities until
completion or rollback. `step()` drives submission and completion; there is no
public API that commits arbitrary page IDs. Rust runners implement
`engine::Runner<P>` and receive an owned `ExecutionPlan<P>`. The native
[conformance tests](../crates/resin-core/tests/engine.rs) demonstrate this API
without a Python interpreter. Native applications bind their arena to the core's
cache pool and use the same logical runner throughout its lifetime. They call
`close(&mut runner)` for orderly shutdown; the core does not own the device
runtime or call back into a runner when it is dropped.

Multiple requests within a batch execute together. With device overlap, an
initial `step()` can return no events while its batch runs; subsequent steps
submit a continuation before harvesting the previous batch.

### Engine overlap

`Engine(enable_overlap=True)` is the default. An optional runner `enqueue()`
returns an owned completion with `result()` and device `feedback`; the
synchronous runner contract remains sufficient. The shared DeepSeek CUDA runner
implements this boundary. Model forward methods receive ordinary device inputs
and cache views; they do not track pending batches or schedule continuations.

The Engine keeps one batch in flight. Exact accepted tokens, positions and
sampling offsets feed the next graph on the GPU; CPU progress and events commit
from the completed output. A single writer lease transfers between safe decode
continuations. Arrivals, cancellation, close and page boundaries drain before
changing ownership. Failures after enqueue retain owners in quarantine.

The fast path requires a stable decode cohort, no generated-token logprobs,
sufficient remaining budget, and both write spans inside the same owned blocks
in every cache group. Stop tokens, including EOS, can finish a request while its
successor step is in flight: its result is published immediately, but its row
and cache lease remain alive until that step completes. Successor outputs for
finished rows are discarded. Other steps use the synchronous path.
`enable_overlap=False` or `step(allow_overlap=False)` selects the reference path.
Distributed replicas vote together before enqueue or drain.

## Cache geometry

Every group uses equal-sized physical slabs. `blocks_per_slab` lets smaller
states share one slab, while the allocator remains free to assign each slab
to whichever group needs it. There is no fixed VRAM percentage for recurrent
state versus attention history.

`CacheGroupFamily.HISTORY` covers full and sliding token history.
`CacheGroupFamily.STATE` covers recurrent checkpoints and retains a two-page
ring per live sequence. A Window group declares both physical `window_tokens`
and logical `read_window_tokens`. A step may write at most
`window_tokens - read_window_tokens + 1` tokens so its earliest query can still
read the preceding window. The Engine caps prefill and checks speculative
lookahead against this bound. See [unified paged cache](cache-pool.md).

CUDA runners normally determine slab capacity after loading weights and
capturing graphs: they reserve the configured model/static memory fraction,
measure remaining VRAM, divide by bytes per slab, and construct the final
layout. This mirrors TokenSpeed's geometry-first sizing rather than guessing
separate cache ratios.

## Prefix reuse

The prefix cache stores immutable page snapshots in an exact-token block trie.
Its checkpoint alignment is the least common multiple of every group block size,
so a hit restores attention, sliding-window, and recurrent state at the same
token boundary. Cached pages consume the same fungible pool and are evicted LRU
when live allocation needs capacity. Successive History snapshots share immutable
page-table segments instead of copying every ancestor's references. Window and
State snapshots remain bounded. Trie nodes carry direct LRU links, so eviction
does not scan all cached entries.

## Speculative decoding

The generic Engine schedules one decode anchor per sequence. A runner such as
DeepSeek-V4 may draft and verify several tokens, then return the accepted token
row. `cache_write_lookahead_tokens` reserves enough destinations for the maximum
verified block. DSpark state updates and target verification remain inside the
model runner and its fused kernels.

## Failure boundary

A rejected execution must have no persistent effects, or must have rolled them
back. It restores the exact selected cohort and permits a retry with a new batch
ID. After execution may have changed state, failure poisons the Engine and
quarantines its resources. Successful device synchronization cannot undo an
in-place recurrent update. Native callbacks that panic also take this failure
path. See the [runner contract](kernel-boundary.md) for Python error handling.

## Distributed coordination and output

The [distributed prototype](distributed.md) adds explicit attention TP/CP/DP and
expert EP/TP meshes, one request Engine per attention-DP replica, and synchronous
phase agreement across communicating replicas. It leaves model partitioning and
kernel collectives in prepared runtimes. GPT-OSS and DeepSeek-V4 prepare
attention head shards separately from expert ownership and intermediate shards.
Shared expert dispatch returns complete origin rows across EP and expert TP;
models do not reduce that result again across attention TP. Sampling uses the
full device request-replica group.

Their CP prefill boundaries differ: GPT-OSS partitions hidden/FFN token rows and
gathers K/V, while DeepSeek partitions attention query work around replicated
MLA compressor updates and full FFNs. Both replicate attention during decode
and retain complete KV/state copies across CP ranks. CP does not multiply
logical cache capacity. Shared CUDA graph machinery captures FFN regions and
decode, coordinates buckets and empty ranks, and explicitly releases captured
collectives before process-group shutdown.

[Generated-token logprobs](logprobs.md) cross the Python-to-Rust boundary as
borrowed NumPy columns.

## Observability

`Engine(enable_metrics=True)` enables native request timing, committed token
counters, scheduler state, and cache inventory. Optional runner hooks supply device pass
durations, graph decisions, speculative counters, and distributed wait time.
They leave `forward`, `enqueue`, and `ForwardOutput` unchanged. Collection never
synchronizes the accelerator; the HTTP server publishes periodic snapshots at
`/metrics`, and scrapes read the last published buffer without entering inference.
See [metrics](metrics.md) for ownership, timing definitions, Prometheus queries,
and OTLP forwarding. [Benchmark harnesses](benchmarks.md) also record timings and
Perfetto traces.

## Current limits

- overlapped decode currently excludes generated-token logprobs;
- no prefill/decode disaggregation;
- sparse graph-safe expert dispatch requires a shared NVLink domain and retains
  static receive storage; RDMA graph transport and communication overlap are pending;
- model dimensions, quantization blocks, and kernel capabilities constrain layouts;
- sparse transport passes composed meshes on up to eight H100s; DeepSeek-specific
  Blackwell validation of the new transport is pending;
- no multi-node serving;
- model runtimes implement device collectives; the native scheduler owns local
  request and page lifetimes.
