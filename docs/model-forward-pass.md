# Authoring a CUDA model runner

A model runner receives a frozen `BatchPlan`, stages its token and page-table
arrays, executes the model, and returns `ForwardOutput` with token rows and any
requested scores. The Engine already owns admission, batching, cache allocation,
prefix reuse, and request lifetime; do not recreate those policies in the forward
pass.

## Weight preparation

Checkpoint storage and executable layers may have different structures. Build
checkpoint containers on meta, validate and load their named tensors, then lower
them with `resin.backend.loading.prepare_model`. Its child-first traversal
preserves shared module identity. A lowering rule can pack a projection or fuse
loaded children; ordinary modules pass through unchanged. Return the prepared
model from the checkpoint loader before allocating execution workspaces or
capturing graphs.

Qwen and Gemma use `Nvfp4LinearWeights` for ModelOpt names and linear scales,
then replace them with `Nvfp4Linear` containing the selected kernel and its final
layout. Runtime linears retain only execution weights. GPT-OSS already prepares
expert weights incrementally while streaming its checkpoint; large weights need
not be duplicated to establish this boundary.

This follows DeepInfer's parameter-container to runtime adaptation in
[`DeepseekForCausalLM.load_model`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/deepseek.py#L445)
and its separate
[`containers.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/containers.py).
Keep the model's lowering rule explicit: checkpoint encoding, kernel support,
and transformations such as projection fusion are different decisions.

DeepSeek's `MoEWeights` similarly exposes checkpoint expert names and aliased
packed storage. `Transformer.prepare_weights()` lowers it to architectural
`MoE(gate, routed, shared)` composition. The prepared `RoutedExperts` adopts the
loaded buffers without copying, packs them once and owns the injected reduction
policy. Checkpoint expert wrappers do not remain in the executable tree.

Its call is `routed(hidden_rows, selected_experts, routing_weights)`, returning
complete rows in the caller's order. Prepared `ExpertParallelDispatch` owns
communication buffers and explicit EP/expert-TP groups behind that layer call;
the scheduler and cache plan do not perform expert dispatch. Group participation
is mandatory even when a rank has no selected local experts. The shared
`DistributedWorker` coordinator supplies placeholder steps when an attention-DP
replica has no requests. See [distributed execution](distributed.md) for the
worker lifecycle, supported transports, and idle-rank contract.

This ownership follows DeepInfer's
[`DeepseekMoE`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/mlp.py)
and
[`FusedExperts`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/moe.py).
Resin keeps reduction injection explicit rather than reading a process-global
group from the architectural forward.

## Shape rules

- Write persistent kernels whose work is bounded by runtime metadata. Attention
  must not specialize or do useful work up to the configured maximum sequence
  length, and MoE kernels must not specialize on the current token count.
- A persistent kernel still needs an explicit, testable work-distribution
  scheme. Do not replace a correct fixed tile grid with a runtime loop merely
  to reduce dispatch geometry; prove that every tile is visited exactly once
  at the minimum, boundary, and maximum logical lengths. A fixed maximum grid
  whose programs immediately mask inactive tiles is often the safer baseline.
- Do not make batch size 1 a separate architecture. A small number of measured
  batch regimes is acceptable when the implementation genuinely changes, such
  as switching tiny projections to a TinyGEMM path.
- Padding may establish a safe tile boundary, but it must not turn logical
  sequence length into work. Pass logical lengths to the kernel and mask or
  terminate persistent work at those lengths.
- Treat new JIT artifacts for novel prompt lengths, token counts, or ordinary
  batch sizes as a bug. Exercise several unseen shapes while recording compiler
  logs and artifact-cache entry counts.
- Keep `BatchPlan` and cache CSR fields as NumPy arrays. Stage them through
  reusable pinned buffers with nonblocking copies instead of converting large
  prefills or page tables to Python lists.
- Use `cuda.tile.assume_divisible_by` for true runtime divisibility facts. It is
  an unchecked compiler hint, not a runtime assertion; a false claim is
  undefined behavior.

## Kernel selection and fusion

- Prefer FlashInfer and Quack kernels when they implement the operation and
  layout directly. Prefer FlashInfer collectives; its all-reduce selects the
  one-shot path for small tensors automatically.
- Use FlashAttention/FlashInfer attention plans rather than rebuilding generic
  attention from PyTorch operations.
- Write custom cuTile or TileLang when fusing a long architecture-specific
  chain materially removes launches or memory traffic. MHC normalization and
  DeepSeek-V4.1 compressor/state updates are representative examples.
- Keep the launch sequence short. Compare a Perfetto trace with the reference
  implementation and account for every material dispatch, synchronization, and
  collective. Avoid host reads, `.item()`, implicit stream waits, and temporary
  allocations in the steady-state path.
- When streams overlap work, record the exact data dependency with events and
  test it under replay. Do not rely on incidental default-stream ordering.
- Preserve numerical intent when fusing: accumulation dtype, quantization scale
  layout, causal masking, top-k sentinel handling, and collective placement are
  part of model correctness.
- Sort or compact sparse-selection candidates before converting logical
  indices to physical cache rows. Padding sentinels such as `-inf` must never
  displace a finite candidate because a top-k primitive returns an unspecified
  order. Test histories with fewer valid entries than the configured top-k.

## CUDA graphs

Every decode path must be CUDA graphable. Allocate graph-owned workspaces and
stage mutable values into fixed buffers; do not compile kernels, allocate CUDA
memory, create plans, or initialize libraries during replay. Capture graphs at
startup and fail startup if capture or replay parity fails.

Graph capacity is not logical work size. A maximum-context graph must still
read the current positions and execute only the active attention/indexing work.
Use graph variants only for a small number of measured execution regimes, not
as a lookup table for sequence lengths.

Never key a graph or graph-owned buffer layout by live sequence length. Replaying
that graph with different cache metadata can violate the captured pointer and
extent assumptions; the result may be silent KV/state corruption rather than a
clean graph error.

Test eager-versus-graph output parity, repeated replay, the largest supported
shape, and transitions across any graph regime boundary. Do not validate only
the synthetic positions used during capture: replay the same graph at short,
tile-boundary, page-boundary, recurrent-checkpoint, and near-capacity history
lengths. Long-context bugs can remain invisible when capture always starts at
position one. A replay failure poisons the runner; never silently fall back
after the GPU may have accepted work.

## Paged state and prefix reuse

Declare one `CacheLayout` for full attention, sliding-window attention, and
recurrent state. The Engine allocates the pages and passes their physical
IDs in `BatchPlan.cache_groups`; the model only maps those IDs to its device
arena. Do not keep a second request-to-page allocator in Python.

STATE groups use one private live slot per request, independent of token position.
The ordinary block table identifies that slot; `CacheGroupPlan.state` carries
one-time zero/restore copies and optional checkpoint destinations. Apply restores
before model execution. Copy a checkpoint only at its exact accepted cache-valid
prefix, and return its reservation ID in `ForwardOutput.state_checkpoints`.
All STATE groups must materialize that prefix before the Engine publishes it.
Never relabel terminal state after EOS or output-length truncation as an earlier
prefix. CUDA runners use the shared
`execute_state` wrapper; their arena exposes complete `[slab, slot, elements]`
state views and their model commits only the accepted state before returning.
Snapshots and restore sources remain pinned until queued consumers finish.

Consume each cache block's structured slab ID and slot number. Do not
invent or decode a flattened kernel page ID independently in every runner or
kernel. Validate the slab and slot bounds once at the runner boundary, then
stage the structured address through graph-owned page tables.

Initialize every newly allocated or reused cache block before a kernel can
vector-load its padding. A logical length mask does not make uninitialized
bytes safe if a fused attention kernel performs arithmetic before applying the
mask: a NaN in an inactive page tail can poison valid output. Test attention
with deliberately dirty and NaN-filled padding; kernels should mask before
arithmetic, while the arena should still clear fresh blocks defensively.

Include the runner's maximum uncommitted verification lookahead in every
sliding-window retention budget. Verification may write the whole candidate
block before acceptance; the Engine must keep the logical attention history and
that future tail resident simultaneously, including when their union straddles
an extra page. Leasing write destinations without extending retention can drop
a still-readable history page near a block boundary.

Test prefix caching with real continuation requests. A hit must prefill the
uncached suffix within the Engine's chunk and token budgets, restore recurrent
and sliding-window state at the same checkpoint, and copy on write before
mutating shared pages. Cover:

- a fixture long enough to publish at least one aligned cache checkpoint; a
  short prompt below the model's shared block alignment cannot produce a hit;
- zero-hit and exact-hit requests;
- hits immediately before and after every cache block/checkpoint boundary;
- a long generated response followed by another user turn;
- shared prefixes decoded concurrently, then diverging;
- LRU eviction under cache pressure and physical page reuse.

Compare cached continuation output with a fresh full prefill under greedy
sampling. Run the comparison repeatedly; coherent first output is not enough to
detect stale recurrent state or a bad sliding-window offset.

## Speculative decode

Speculative decoding belongs inside the runner. Set
`speculative_draft_tokens` (default zero) and derive
`RunnerConfig.max_accepted_per_step = speculative_draft_tokens + 1`. The Engine
limits each raw returned row to this total, before applying stop or length limits.
Use the ordinary `draft -> target_verify [anchor, drafts] -> commit` flow, without
an extra target-decode pass that silently increases the output bound.
`CacheLayout.decode_write_extra_tokens` accounts for model-specific shifted
persistent writes beyond that block; it is physical geometry, not a second
runner output setting. Write drafts only into leased destinations, commit exactly
the accepted prefix, and clear or logically hide rejected recurrent/SWA state
before the next step.

The Engine chooses overlap reservations from the per-step advance and cache
geometry; runners do not configure a second token bound. The plan's page tables
are the actual immutable lease, including destinations for queued steps. Runners
derive readable history from device positions and select graphs that cover the
leased tables. For example, three drafts permit at most four returned tokens per
invocation even when the Engine reserves another page. Stable live recurrent
state has no rotating checkpoint boundary that forces a drain. The Engine falls
back to a one-step reservation if the wider span cannot fit window, context,
output, or capacity limits. Runners must accept both lease shapes.

The Engine allocates, copies on write, and leases the complete envelope before
enqueueing work. Its physical page tables remain immutable until all accepted
consumers finish, including consumers of stopped or cancelled rows. A reserved
table may contain unwritten future pages: derive actual readable CSR ranges,
last-page lengths, and physical append page IDs from accepted GPU positions.
Updating only append offsets is insufficient at page crossings. Qualify the
attention plan with changing page counts; do not read the full reserved extent
as though it were committed history. The Engine drains before another maximum
step would leave the lease or violate its read/write guard.

An `enqueue` completion may publish `feedback=None` to decline its next
continuation. The Engine reports `overlap_ready=False` and harvests that work
normally before preparing another batch. Missing or throwing feedback attributes
remain contract failures; they are not a normal drain signal.

Track acceptance rate and time draft, target verification, commit, and sampling
separately. Always report output-token throughput as well as block latency: a
fast kernel can still produce slow generation when acceptance collapses. Test
greedy and stochastic sampling with fixed seeds, EOS inside a draft block, and
page/checkpoint crossings inside the verified block.

At temperature zero, speculative decoding must produce exactly the same token
sequence as target-only decoding. Compare tokens, not whether both answers look
reasonable. Run this parity check for short and long generations, every
verification-block size, fresh and prefix-cached prompts, and blocks that cross
full-attention, sliding-window, or recurrent-checkpoint boundaries. Any first
divergence is a correctness bug in verification, acceptance, commit, rollback,
or cache addressing; a later repetition loop is only a symptom.

Give every block-verification forward path a sequential oracle: compare each
column's target logits and the final full-attention, sliding-window, and
recurrent state with repeated one-token target decode from the same starting
state. End-to-end acceptance can conceal an incorrect later column until a
particular draft pattern exercises it.

## Isolating regressions

Do not infer model quality from one plausible response, or attribute a
regression to tensor parallelism, the GPU node, or deployment state before
isolating the runner. Hold the checkpoint, prompt, sampling parameters, and seed
fixed, then compare target-only against speculative decoding, eager against
graph replay, and fresh prefill against prefix reuse. Compare exact greedy token
IDs and locate the first divergence. This turns intermittent gibberish into a
small forward-path or state-management problem instead of a subjective quality
judgment.

Repeat deterministic probes from a clean runner and after unrelated requests.
First-turn success does not rule out stale workspace, recurrent state, page
reuse, or graph-buffer bugs. Conversely, a correct target-only run with a broken
speculative run localizes the fault to verification or commit semantics rather
than the checkpoint or renderer.

## Minimum validation

Before publishing a runner:

1. Compare logits or greedy tokens with the checkpoint reference on short and
   long prefills, decode, and continuation.
2. Run novel prompt lengths and batch sizes twice; the second run must not hide
   shape-specialized compilation required by the first.
3. Verify CUDA graph capture/replay and prefix-cache parity across page and
   recurrent-state boundaries.
4. Verify exact greedy token parity between target-only and speculative decode
   for every supported verification-block size and cache-boundary regime.
5. Include a deterministic long-form oracle, such as an exact count or copied
   token pattern, that crosses many attention tiles and must terminate with EOS.
   Short conversational answers are too forgiving to expose cache-index drift.
6. Stress multiple requests, cancellation, EOS, cache pressure, and page reuse.
7. Record TTFT, output tok/s, speculative acceptance, phase timings, and a
   Perfetto trace. Compare kernel dispatches with the fastest trusted runner.
8. Warm every unavoidable library/kernel regime during startup, then confirm a
   warm server has no lazy compilation stalls.
