# Distributed execution

`resin.Parallelism` defines rank placement in the Rust core and exposes that
immutable configuration to Python. `resin.backend.topology.ParallelGroups` owns
the Python process-group handles. `resin.backend.distributed` coordinates
Engine steps and eligible decode continuations. Each rank owns a model runtime;
each attention-DP replica has one native request Engine, on its lowest global
rank. The other ranks in that replica follow its driver.

The boundaries are:

| Component           | Responsibility                                                                       |
| ------------------- | ------------------------------------------------------------------------------------ |
| `Parallelism`       | Rust-defined dimensions, validation, rank placement, and partitions                  |
| `ParallelGroups`    | Explicit device groups and Gloo control groups                                       |
| `DistributedEngine` | Phase agreement, normal local Engine steps, empty participation, failure propagation |
| Native `Engine`     | Local admission, requests, pages, leases, output validation, commit                  |
| `DistributedWorker` | Replica commands and model callback invocation                                       |
| `DistributedRunner` | Weight placement, CP partitioning, expert dispatch/combine, device synchronization   |

This preserves the existing checkpoint-to-runtime preparation boundary. Create
groups before preparing model weights and inject the appropriate group into each
prepared runtime. Attention weights use `groups.attn_tp`; prefill query ownership
uses `groups.cp`; experts use the independent `groups.ep` and `groups.expert_tp`
coordinates. Expert payloads travel over `groups.device_domain`, the device group
covering the complete serving world. The coordinator never
infers model weight layouts or uses the default world group for those operations.

CUDA runners declare supported axes in `parallelism_capabilities`, a
[`ParallelismCapabilities`](../resin/backend/topology.py) value with flags for
`attn_tp`, `attn_dp`, `cp`, `ep`, and `expert_tp`. The shared CUDA base disables
these flags by default; each runner opts into the axes its implementation
supports. Check the selected runner's declaration in
[`models/cuda/resin_models_cuda/`](../models/cuda/resin_models_cuda/) for its
supported layouts. Deployment presets in
[`deploy/models.py`](../deploy/models.py) supply concrete starting topologies.

Attention replication is always allowed; it repeats the same attention
computation without introducing another sharding axis.
These axes compose through the explicit `parallel_groups=` preparation path.
Model dimensions, quantization blocks, and kernel capabilities still constrain
valid layouts. For example, DeepSeek-V4.1 DSpark uses the same meshes for draft
and target verification, including
empty-replica participation and captured expert exchanges.
The GPT-OSS and DeepSeek explicit-group paths provide
captured prefill FFN and attention-projection regions, plus full decode graphs.
GPT-OSS's FlashInfer implementation returns FP32 local expert partials so the
cross-rank combine does not add an intermediate activation rounding step.
`DistributedWorker` validates a declared capability set collectively before
serving begins. Custom distributed runners may supply the same declaration.

See the [DeepSeek-V4.1 model notes](../models/cuda/resin_models_cuda/deepseek_v41/README.md)
and [GPT-OSS runner](../models/cuda/resin_models_cuda/gpt_oss/cuda.py) for concrete
implementations. [Benchmarking](benchmarks.md) documents the shared CLI for
measuring a model with explicit topology options.

### Dense model tensor parallelism

Gemma and Qwen accept `Parallelism(world_size=2, attn_tp=2)` for attention and
FFN TP together. `Parallelism(world_size=2)` instead replicates attention and
shards the FFNs. Both configurations keep one request Engine and one logical
batch; neither introduces attention-DP request routing. These models require
`attn_dp=cp=ep=1`, so dense FFN TP equals `world_size`.

Attention TP partitions complete KV heads. Gemma requires the TP size to divide
both its 16 sliding-window and four global KV heads, as well as its query heads;
the assistant uses the corresponding target KV shards. Qwen requires divisibility
of its full-attention query/KV heads and DeltaNet key/value heads. Qwen also shards
the recurrent state, convolution channels, and bundled MTP attention. Remaining
attention copies replicate their assigned head shard.

The shared dense preparation partitions gate/up output rows and down input
columns, then sums BF16 partial outputs before the model's residual or output
normalization. It covers target and assistant/MTP FFNs. Intermediate widths must
divide evenly by FFN TP; NVFP4 shards must also align to 64 channels. NVFP4 uses
W4A16 kernels, retaining the checkpoint's packed weights and quantization scales.
Checkpoint tensors are validated against their original shapes on CPU, sharded
there, and transferred to CUDA before kernel layout preparation, avoiding a full
unsharded model allocation on each GPU.

These dense paths capture prefill FFN regions and the full decode forward,
including attention and TP reductions. Prefill attention and recurrent-state
updates remain eager.

CPU tests cover head ordering, continued DeltaNet states, packed quantization
coordinates, and reductions. The [two-B300 checkpoint run](https://modal.com/apps/ekzhang/main/ap-zoL3hOYaVgLsjvc54Nb6uC)
exercised Gemma 4 31B NVFP4 and Qwen3.8 27B NVFP4, with attention TP1/TP2,
FFN TP2, and speculative lengths zero/two. It covers two-request decode,
chunked prefill, assistant/MTP verification, generation and prompt scores,
cancellation, and cache reuse. That historical run used eager decode for
generation logprobs and captured decode for separate unscored requests. Both
runners now retain captured decode when scores are requested; scoring uses a
separate captured tail over the model's raw logits.

The [eager/graph controls](https://modal.com/apps/ekzhang/main/ap-MScJzgLu2mX3n4eTB1KyLT)
produce bit-exact tokens and scores at each fixed topology, including the
unsharded baseline. Cross-topology BF16 arithmetic is not bit-exact. On this
short workload, Gemma preserves generated IDs but prompt observed-logprob
maximum differences reach 2.81 nats with attention TP2 and 4.68 with replicated
attention. Qwen attention TP2 turns a 0.125-logit lead into a tie, changing the
first generated token of one prompt; its prompt observed-logprob maximum
difference is 0.117 nats. These are execution and arithmetic checks, not a
model-quality evaluation. The validation report keeps these differences visible
instead of accepting them through a blanket tolerance.

A [Gemma projection diagnostic](https://modal.com/apps/ekzhang/main/ap-pSi5v0KJxB1RH3uswRKunI)
uses actual checkpoint weights and prefill inputs from layers 0, 29 and 59.
Gate/up projections and gated activations are bit-exact after sharding. The
down-projection split differs by 0.268–0.303% relative L2; an independent
dequantized reference reproduces this when each partial is rounded to BF16
before addition. Packed weight/scales retain the original quantization groups.
A [single-GPU split-FFN control](https://modal.com/apps/ekzhang/main/ap-sJJhAtGlzOUBvYLiiAYUE5)
then reproduces every distributed FFN-TP2 prompt score exactly, including the
4.68-nat maximum difference. It runs both shards locally and adds their BF16
outputs, establishing that this drift comes from the split FFN arithmetic rather
than distributed communication, cache mapping, or graph execution.

The dedicated dense-parallel checkpoint validator has been retired. Raw
checkpoint and control records remain under `artifacts/cuda-model-improvements/`
as historical evidence. Current performance runs use `scripts/bench.py` with
`--model gemma4` or `--model qwen3_5` and explicit topology options; local speculative
Gemma runs also need `--assistant-checkpoint`. The regular tests retain
distributed transport, cache, and model contracts.

## CUDA graphs across ranks

Graph-enabled CUDA runners call `runner.capture_cuda_graphs()` on every rank
before constructing workers or entering serving loops. All ranks must configure
the same graph catalog and capture shapes in the same order. The shared CUDA
runner checks those catalogs collectively, owns capture/replay, and chooses
common physical buckets from the logical sizes in `StepContext`.

Prefill captures FFNs in `prefill` and row-independent attention projections in
`prefill_attention`. The latter reports `capture_scope="attention_projections"`
and the captured operation names. Attention kernels, page-table lowering and
request-dependent cache/state updates remain eager between these callbacks.
Decode captures the model forward and sampling. Each step votes on graph
eligibility and capacity across the serving domain before issuing device work.
If any replica needs an unsupported expert/decode shape, every replica uses the eager path
for that step. This keeps expert collectives in the same order even when request
queues and batch sizes differ. Graph counters expose captures, replays, and
eager fallback reasons.

After workers stop, call `runner.close_cuda_graphs()` on every rank before
destroying process groups. This waits for device work and releases captured
NCCL references; keeping graphs alive during group destruction can hang shutdown.

`ExpertParallelDispatch` has two transports behind the same local expert
callback. Eager execution sends each selected row to every expert-TP lane of its
expert owner with variable-size NCCL all-to-all, reading split counts on the host.
CUDA graphs use FlashInfer's GPU-driven sparse dispatch/combine over a shared
NVLink workspace. Only selected destination ranks receive token payloads; counts
and routes stay on the GPU. Static receive storage has `world_size * capacity`
slots, but unused slots carry the inactive expert sentinel and execute no expert
GEMMs. Padded source rows stay local with inactive routes and zero weights.
Returned rows are already complete across EP and expert TP;
the model must not apply another attention-TP reduction to them. GPT-OSS retains
FP32 partials through the combine. Sampling broadcasts use
`groups.device_replica`, covering every rank executing one logical request
replica. The older `group=` dispatcher API retains its optional single-source
attention-replication behavior. GPT-OSS dispatches one origin per attention TP and
replication group, preserving distinct CP prefill segments. DeepSeek's MLA gathers
CP outputs before FFN, so it dispatches one origin per complete attention replica.
The combined result is broadcast back to the matching input copies.

Eager combine groups returned partials by their origin while preserving owner
rank order, then reduces each segment in that order. This avoids unordered CUDA
atomic additions changing repeated greedy outputs. Both transports preserve the
local callback's partial dtype until the final cast to the hidden-state dtype.

The sparse graph backend supports at most 64 physical ranks in one NVLink domain.
FlashInfer's `SymmDeviceMemory` exchanges CUDA allocations cooperatively with Unix
sockets, without `SYS_PTRACE`, or uses MNNVL fabric handles when available.
Without MNNVL fabric, all ranks must share one host.
Cross-host InfiniBand/RDMA graphs are unsupported and fail explicitly;
eager NCCL remains available. More than 22 distinct transport routes use multiple
rank-partitioned sparse passes. These limits describe the primitive, not an EP64
or cross-node performance qualification. Static receive storage and routing work
still scale with world size; communication/computation overlap is separate work.
The model and kernel adapters remain responsible for capture-safe local expert
execution and their own weight layouts. Kernel token limits apply to the entire
static receive shape, including inactive slots. Workspaces are shared across
layers and released after `close_cuda_graphs()` synchronizes all ranks, before
group teardown.

Real H100 validation on 2026-10-21 covers four-GPU eager/graph transport with
unequal and empty origins, changed routes and valid-row scalars on replay,
attention replication, independent attention TP, attention DP, CP and expert TP.
A forced two-partition case checks actual multi-pass dispatch/combine arithmetic.
The [eight-H100 all-axes run](https://modal.com/apps/ekzhang/main/ap-mAC3PAVTWVWPU38FUCZMeF)
also passes EP2/expert-TP4 with attention-DP2, CP2 and attention-TP2 together.
Every graph comparison is exact against independent arithmetic. Device counters
in the fully active EP4 fixture record 30 inter-GPU token copies versus the former
dense transport's 60; deduplicated attention replication reduces that fixture to 10. Idle replay transfers zero token payloads. Corresponding logical payloads
are 3,600/1,200 versus 7,200 bytes, excluding synchronization and protocol traffic.
These are transport checks, not full-model throughput or MFU measurements.

Idle replicas replay the same expert collectives. Empty prefill replays FFN
regions without attention. Empty decode replays padded model rows using reserved
null or scratch cache storage, leaving live request pages untouched. Graph
decode can still return the normal NumPy logprob columns after replay.

## Context-parallel prefill

`ContextParallelTokens` partitions a packed token axis into balanced contiguous
segments and reconstructs global order with a padded all-gather. Prepared maps
and fixed collective shapes permit capture without host reads of GPU counts.
Decode executes replicated attention on CP ranks.

GPT-OSS keeps hidden states and FFN rows local to each CP shard during prefill.
It gathers rotated K/V before writing the complete cache on every copy, then
runs local queries against those pages. Shortening each K length to the end of
that rank's query segment preserves causal and sliding-window positions. Empty
query sequences are excluded from FA4 metadata. Final hidden rows are gathered
before selecting each request's last token.

DeepSeek-V4.1 preserves full hidden states, MLA KV writes, and ordered recurrent
compressor updates on each CP rank. It partitions indexer query scoring, sparse
attention, and query/output projection work, then gathers attention outputs
before the full FFN. This preserves the compressor's ordered state updates.

These implementations retain complete KV/state copies across CP ranks. Their
CP layouts distribute prefill computation without increasing logical KV-cache
capacity.

## Rank placement

`world_size` is the total number of ranks. The five configuration fields are
`world_size`, `attn_dp`, `attn_tp`, `cp`, and `ep`, each defaulting to one.
`Parallelism` is immutable and accepts keyword arguments only:

```python
from resin import Parallelism

parallelism = Parallelism(world_size=4, attn_dp=4)
```

Attention TP is explicit; attention replication and expert TP are derived:

```python
attn_replication = world_size // (attn_dp * cp * attn_tp)
expert_tp = world_size // ep
```

All sizes must be positive integers. Both `attn_dp * cp * attn_tp` and `ep` must
divide `world_size`. Attention uses an
`[ATTN_DP, CP, ATTN_REPLICATION, ATTN_TP]` mesh, keeping each attention-TP group
contiguous. The derived replication axis repeats the same attention shard and
receives the same logical plan; it does not create additional request Engines.
Experts use `[EP, EXPERT_TP]` over the same ranks. Expert weights are sharded
across that complete mesh, with no separate expert-replica axis. CP describes
attention token partitioning within those ranks; it is not an additional
multiplier in the expert mesh. For example,
`Parallelism(world_size=8, ep=8, attn_tp=8)` gives attention TP eight and
expert TP one on eight ranks.

`parallelism.coordinates(rank)` returns `(dp_index, cp_index, replication_index,
tp_index)`, and `parallelism.rank(...)` maps those four coordinates back to the
global rank. `parallelism.attn_replication` is the derived replication count;
`groups.attn_replication` contains ranks with the same attention shard at fixed
DP, CP, and TP coordinates. The runner owns any device communication across
that group.

For `Parallelism(world_size=8, ep=4, cp=2, attn_dp=2, attn_tp=2)`, both
attention TP and expert TP are two, and attention replication is one. The groups
containing rank zero are:

| Group                  | Ranks      |
| ---------------------- | ---------- |
| Attention TP           | 0, 1       |
| CP                     | 0, 2       |
| Attention DP           | 0, 4       |
| Attention replication  | 0          |
| EP                     | 0, 2, 4, 6 |
| Expert TP              | 0, 1       |
| Request replica        | 0, 1, 2, 3 |
| Serving domain         | 0–7        |
| Device request replica | 0, 1, 2, 3 |
| Device serving domain  | 0–7        |
| Domain drivers         | 0, 4       |

`Parallelism(world_size=4, attn_dp=4)` gives four attention replicas with
attention TP one, CP one, and expert TP four. Each rank drives its own request
Engine while the experts span all four ranks.

`Parallelism(world_size=8, attn_tp=2)` gives attention replication four and
expert TP eight. There is one request replica and one driver for all eight
ranks. Its four attention-TP groups receive the same logical plan. In general,
each request replica contains `cp * attn_replication * attn_tp` ranks, and there
are exactly `attn_dp` request Engines.

The complete expert mesh connects every attention replica, so all drivers
synchronize their serving loops. This also holds with `ep=1`, where expert TP
spans the entire world. A replica with no requests participates in empty
execution while its peers have work. The serving domain is the complete world;
no connected-component inference is needed.

The Rust `Parallelism` value contains numeric configuration and rank layouts;
Python `ParallelGroups` creates and owns the communication handles.
`ParallelGroups.create(parallelism)` is collective over an initialized default
world group, checks agreement on configuration/backend, creates groups in
deterministic order, and reuses identical groups. Its `.topology` field refers
to the supplied `Parallelism`. Device collectives default to NCCL; control
collectives use Gloo. The CPU conformance test uses Gloo for both.
`groups.close()` destroys only the groups it created, after workers stop; the
caller owns the default world.

Each attention-DP replica still has its own native scheduler, request queue, and
logical cache pool. Request limits, token budgets, and cache-parent capacity are
per replica; `attn_dp` does not scale one shared cache pool. The runner supplies
a logical cache layout backed by every rank in that replica. Attention TP can
reduce per-rank KV head storage; CP currently retains complete KV copies and
does not multiply logical capacity. The native configuration type does not add a global
request router or move the current Python serving coordinator into Rust. The
HTTP bridge below adds explicit replica routing above these local engines.

## One coordinated step

Drivers in the same world call `has_work()`, `step()`, and `close()` in matching
order. `submit()` and `cancel()` remain local operations. Applications can drive
this loop directly or use the HTTP coordinator to route ingress and keep all
drivers stepping together.

1. Drivers exchange read-only `Engine.next_phase()` proposals. When replicas
   propose different phases, drivers alternate prefill and decode. The proposal shares the native admission
   fallback used by actual dispatch and reserves nothing.
2. Each driver calls `Engine.step(phase=agreed_phase)`. A mismatched local cohort
   remains queued. Cache pressure and retraction still belong to the native Engine.
3. Actual plans rendezvous with empty or failed preparation on other drivers.
   This second exchange determines whether there is real work, computes logical
   replica capacities, and catches plan serialization errors before execution.
4. Each driver broadcasts its logical plan to all ranks in its request replica. All workers
   validate inputs and vote readiness before calling the model. An idle replica
   receives `plan=None` and participates in the model's empty execution path.
5. Workers execute or enqueue the forward callback and exchange execution status.
   The native Engine validates and commits completed results, including NumPy
   [logprob columns](logprobs.md). Eligible decode steps can enqueue the next
   continuation before collecting the previous completion.
6. Drivers exchange commit status before returning events. A failure permanently
   poisons every native Engine in the domain and stops further execution.

Applications retain the same `step()` API. The Engine owns pending plans and
cache leases during overlap; workers retain per-rank completions and drain them
before resource release or close. Model tensor forwards do not schedule batches.

Replica commands also propagate resource binding and release. Page allocation
exists only on the driver; followers bind their device arenas to the same logical
pool identity and receive its page tables. Request IDs and pool IDs are local to
the request replica; cross-replica device metadata must preserve that namespace.
Bind and release callbacks are replica-local commands and must not issue
collectives spanning other drivers. Cross-replica collectives belong in forward.

Replica transport retains request state by sequence ID. The first scheduled plan
registers each request's sampling policy and prepared media on the followers.
Native media uses a versioned descriptor and separate numeric buffers; subsequent
prefill chunks and decode steps reuse the exact same follower object. Completion,
cancellation and retraction release that state after outstanding work has drained.
Plans already in flight retain their own references and array storage.

Successful step commands use typed Gloo broadcasts: a fixed integer header and
a native plan archive. The archive schema is derived directly from Rust's
`BatchPlan` and cache-group types using rkyv; Python does not repeat their field
lists, sizes or alignment rules. Request sampling, media and derived positions
are excluded from the archive. Receivers validate the archive, attach retained
request state, and apply the ordinary plan validation before execution. Restored
arrays own their storage, so subsequent receives cannot overwrite an in-flight
plan. The private format is versioned and requires the same Resin build on every
rank. This is not a zero-copy receive path or a persistent storage format.
Successful step commands do not use pickle or object collectives. Media buffers
and JSON request descriptors are sent only on registration, never on repeated
decode steps. Request policy and media stay unchanged until release. Distributed
sampling `extra` values must be JSON data, and distributed media must use Resin's
native payloads. Local execution accepts custom Python `MediaInput`
implementations with the same required cache identity and token-span fields.

## Model callback

A CP, EP, or attention-DP runtime implements:

```python
def forward_distributed(
    self, plan: BatchPlan | None, context: StepContext
) -> ForwardOutput | None:
    # Model-owned CP partitioning and expert communication, including plan=None.
    # Return ForwardOutput for a local batch; idle ranks may return None.
    ...
```

`context` carries the local groups, agreed phase, step ID, local logical sequence
and token counts, and `(sequence_count, token_count)` for each replica in driver
rank order. `max_num_sequences` and `max_num_tokens` describe logical per-replica
capacity. The model derives physical shard sizes and graph buckets from them.
CP groups are static across phases. The runner uses `context.phase` to shard
attention tokens across CP ranks during prefill and replicate attention
execution across those ranks during decode. The coordinator broadcasts the same
logical plan within a replica; phase-specific CP behavior and expert routing
belong to the model callback.

An optional `validate_distributed(plan, context)` performs host checks before the
readiness vote. If it returns a non-`None` prepared value, the worker passes that
value to the execution callback as the keyword-only `prepared` argument. This
lets runners reuse validated metadata without retaining pending state.

The synchronous forward callback must preserve collective ordering and finish
its device work before returning. A runner can instead provide
`enqueue_distributed(plan, context, *, prepared=..., feedback=...)` for deferred
completion and eligible decode continuations. It returns an owned completion
with `result()` and device feedback for queued work, or the ordinary synchronous
result on fallback. The Engine retains the plan and cache leases until
completion, and every replica must enqueue together. See
[overlap scheduling](scheduler.md) for the completion contract.

Mid-collective process/device failures rely on
process-group timeouts and process teardown. The coordinator propagates errors;
it cannot roll back in-place model state or globally undo a peer's completed local
commit. Failure marks all native Engines poisoned, preserving resources and
preventing cleanup/reuse through normal Engine operations.

TP-only runtimes can implement ordinary `forward(plan)` whenever
`cp=ep=attn_dp=1`, including attention replication greater than one. They use
the same `DistributedWorker` and `DistributedEngine` lifecycle; the runtime
owns weight preparation and communication across its groups.

The GPT-OSS 120B HTTP deployment constructs
`Parallelism(world_size=2, attn_tp=2)` and passes `parallel_groups=` to its
runner. Attention TP and expert TP both equal two, and its distributed callback
uses the same coordinated validation and execution path as composed layouts.
The separate TP server wrapper/follower loop has been removed. Model-level
`tensor_parallel=` contexts remain available for diagnostic callers; they are
mutually exclusive with `parallel_groups=`. DeepSeek's single KV head still
requires `attn_tp=1`, and the retired query-head-only TP HTTP route fails before
loading a model.

## Process integration

For HTTP serving, every attention-DP driver constructs the same server and
calls `serve()`. Non-driver ranks stay in the existing worker loop:

```python
worker = DistributedWorker(runner, groups)
if groups.is_driver:
    engine = DistributedEngine(worker, max_running_requests=256)
    HttpServer(engine, recipe=recipe, model=model_name).serve(host="0.0.0.0", port=8000)
else:
    worker.serve()
```

Only the first driver binds a listener. Requests select an attention-DP replica
using `routed_dp_rank` in JSON or the `resin-routed-dp-rank` header. The header takes
precedence; invalid or out-of-range ranks return HTTP 400. Unassigned requests
use round robin. Each selected driver owns the actual native request lease,
recipe, media preprocessing, incremental parser, scheduler, and cache. Client
disconnects and stalled response channels cancel the request on its owner.
Generated and prompt logprobs travel through the same response bridge.

The built-in chat page chooses a random replica once per page session and sends
that rank on every turn. This preserves prefix and image-cache locality for a
conversation; it does not alter automatic routing for other clients. The page's
first-token timer includes upload, queueing, preparation and inference, and does
not identify any one of those phases.

Set `HttpServer(..., stream_interval=N)` to batch later streaming output until N
generated model tokens accumulate, checked at step boundaries after the first
semantic output. Completion flushes the remainder. It defaults to 1 and must be
the same positive integer on every driver, along with the model and route
configuration. Every driver's native
session uses that interval; coordinated inference, stop matching, and
cancellation continue on every step. Text and score records stay aligned.

`DistributedHttpCoordinator` exchanges ingress before each coordinated step,
including steps where a replica is idle. Native `_HttpSession` instances reuse
the ordinary HTTP lifecycle on each driver; the frontend retains bounded client
channels and receives native event packets. Session initialization,
admission infrastructure, execution, and parser errors are voted before the
next phase. Request rendering failures remain local request errors. Frontend
bind failure or shutdown closes waiting driver loops and model followers.
Ingress and events remain typed Rust packets while crossing Python. Remote
packets use Rust's existing JSON encoding over Gloo byte buffers; local packets
need no serialization, and empty packets carry no payload. Fixed integer votes
exchange lengths, shutdown state, and errors before payload collectives. Python
does not decode requests or events into dictionaries on this path.

For direct engine integration, applications can continue to own local
admission and event consumption explicitly:

After initializing distributed communication and preparing each rank's runtime:

```python
# For a CUDA runner with graphs enabled, all ranks capture before serving.
runner.capture_cuda_graphs()
worker = DistributedWorker(runner, groups)
if groups.is_driver:
    engine = DistributedEngine(worker, max_running_requests=256)
    try:
        # Admit this replica's requests here. Every driver stays in the same loop.
        while engine.has_work():
            events = engine.step()
            consume_local_events(events)
    finally:
        engine.close()
else:
    worker.serve()
# Once all ranks have left their serving loops:
runner.close_cuda_graphs()
groups.close()
```

Initialization and error diagnostics may still use object collectives;
successful execution votes, plan transport, and HTTP rounds use typed buffers.
See [recorded benchmark measurements](benchmarks.md#recorded-decode-sweeps) for
measured multi-GPU performance and its workload limits. Shared-memory command
transport, expert load balancing, and prefix-aware replica routing remain
separate work.

## Source references and validation

The design follows DeepInfer's explicit device/CPU group separation in
[`parallel_state.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/parallel_utils/parallel_state.py),
its read-only phase vote followed by an actual-scheduling vote in
[`scheduler.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp#L299),
and model-level empty execution in
[`worker.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py).
Resin uses explicit injected groups. Its native Engine owns pending plans and
cache leases, including when a decode continuation overlaps completion of the
previous step.

[`test_distributed.py`](../tests/test_distributed.py) launches eight real processes
and checks numerical TP/CP reductions and variable-sized EP dispatch/combine,
including empty senders, mixed phases, unequal request queues, cancellation,
array score output, shutdown, idle DP replicas sharing expert TP, TP-only
compatibility, and failures in initialization, preparation, serialization,
preflight, execution, output validation, release, and control failure after model
execution. These are CPU Gloo conformance tests; GPU kernel and multi-node
behavior are not established by them.

[`test_composable_expert_parallel.py`](../tests/test_composable_expert_parallel.py)
compares a dense SwiGLU reference with EP1/ETP4, EP2/ETP2, and EP4/ETP1 across
different attention meshes, including unequal intermediate shards, output bias,
padding masks, idle origins, and FP32 cancellation.
[`test_context_parallel.py`](../tests/test_context_parallel.py) checks real token
exchange and global causal/window coordinates.
[`test_distributed_overlap.py`](../tests/test_distributed_overlap.py) checks
coordinated continuations across replicas, including idle participation and
synchronous fallback. Historical two-H200 eager/graph runs covered EP2 paths;
the expanded two/eight-H100 matrices executed composed full-model
graphs, including all axes together, with matching greedy probes and reclaimed
request caches. Those tensor-parallel qualification runs exceeded the fixed
same-history logit-error limits; executing the mesh was not an accuracy pass.
Later [DeepSeek full-checkpoint validation](../models/cuda/resin_models_cuda/deepseek_v41/README.md#validation)
at `40a44fd` established exact eager/graph tokens and logprobs on a bounded
EP4/attention-DP4 workload, plus matching synchronous and overlapped output.
That result does not establish cross-topology parity or broader model quality.
The linked model notes and benchmark records retain the revisions and measured
scope of each run.
