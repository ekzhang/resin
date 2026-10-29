# Runner boundary

The Engine owns request scheduling, cache allocation, cache pressure, prefix
reuse, and resource lifetime. A runner owns model execution and kernel
scheduling.

## `BatchPlan`

The independent sequence metadata is deliberately small:

- `sequence_ids`: stable request IDs, one per sequence.
- `context_lengths`: tokens processed before this step, one per sequence.
- `prefill_lengths`: final prompt or replay endpoints, stable across chunks.
- `token_ids`: scheduled inputs flattened in sequence order.
- `token_indptr`: offsets that slice `token_ids` for each sequence.
- `phase`: prefill or decode. A step never mixes phases.

Positions are derived rather than transported independently. `plan.positions`
memoizes the flattened ranges beginning at each `context_lengths[i]`.

The numeric fields on `BatchPlan` are read-only NumPy arrays backed directly by
the frozen native plan: `sequence_ids` is `uint64`; token, position, pointer,
and sequence-index arrays are `uint32`. Repeated property access creates a new
view, not a new data buffer. A runner should pass these arrays directly to its
pinned host staging path rather than calling `list()` or `tuple()` first.
Native construction and wire restoration validate structural invariants once.
`resin.backend.plans.BatchView` retains those native arrays without rescanning
their CSR structure. `PagedCachePlan` checks them against the runner's physical
layout: packing, bounds, active ranges, and exact append ownership.

`batch_id` is a monotonic tracing identifier for runner steps.
`batch_key` is an optional caller-defined compatibility key; the scheduler does
not mix different keys in one batch.

`sampling_mask` has one Boolean per sequence and marks which sequences should
generate a sampled token. It is used for chunked prefill, when some sequences
only advance the cache. `sampling_params` contains one effective sampling policy
per sequence. Random draw offsets are derived from the absolute sampled
position rather than transported separately.

Prompt scoring uses separate predecessor-row and target-token arrays, so runners
can project only the requested hidden rows. See the
[logprob contract](logprobs.md#array-boundary) for these fields and the matching
`ForwardOutput` score columns.

## Cache metadata

`cache_groups` has the same order as the runner's bound `CacheLayout`: there is
one `CacheGroupPlan` per declared cache-block type, and each plan has one CSR
block-table row per batch sequence. Every equally sized physical slab in the
cache is assigned to one block type while in use and holds that group's
`blocks_per_slab` blocks. A block address is `(slab_id, slotnum)`, exposed by
the parallel `block_slab_ids` and `block_slotnums` arrays. A cached token location is
`(block, block_offset)`, equivalently `(slab_id, slotnum, block_offset)`.
Each flattened scheduled token also has an exact append triple: slab ID, slot
number, and in-block token offset. The Engine has already allocated, retained,
and leased those blocks. The runner must not allocate blocks or keep a
request-ID cache map.

`block_table_start_indices[i]` aligns sequence `i`'s possibly cropped table row with
absolute token positions. Logical block `b` begins at
`b * CacheGroupLayout.block_tokens`; full-history rows begin at zero, while a
sliding-window row advances as old blocks are evicted. Logical block numbers
are sequence positions, not physical slab IDs.

Its pointer, block, and append fields are likewise read-only, zero-copy
`uint32` arrays. Constructors accept ordinary integer sequences or arrays with
the exact native dtype; the Engine-generated execution path does not perform a
Python element-by-element conversion.

The cache pool ID appears only in `ResourceBinding`, which permanently binds the
runner's physical arena to one Engine. Repeating it in every `BatchPlan` would
not add information.

History, sliding-window history, and recurrent checkpoints use the same page
allocator. Their different retention and packing rules are properties of the
bound `CacheLayout`, not parallel plan formats.

## Execution contract

A runner implements:

```python
def forward(plan: BatchPlan) -> ForwardOutput: ...
```

Runners return `ForwardOutput`: flat `uint32` token IDs and row
offsets, plus optional logprob columns. Shared `token_output` and `format_output`
helpers materialize these arrays without boxing tensor tokens into Python
scalars. This is the single output contract for token-only and scored results;
raw integer sequences are rejected after execution and quarantine the batch.

The native adapter borrows token and score columns through validation and
commit. It does not allocate an intermediate token vector for each row; only
committed history and events take ownership. Runners must keep the arrays
unchanged until completion returns, including during `release()` callbacks.
Native Rust runners use the same flat representation through `OwnedTokenRows`
and its borrowed `TokenRows` view.

Prefill runners return one token for each sequence, including a placeholder ID
for unsampled continuation or replay rows. A decode runner may
return multiple tokens for each sequence; this is how speculative decoding
commits an accepted block. `RunnerConfig.max_accepted_per_step` is the total
output bound, normally `speculative_draft_tokens + 1`. Cache geometry declares
any additional shifted writes with `CacheLayout.decode_write_extra_tokens`.
The Engine reserves the combined physical write span before execution. DSpark and
other speculative algorithms remain model-runner details rather than fields in
the generic plan. The Engine validates raw output
rows before applying stop tokens and length limits, so truncation cannot hide
an oversized verified block.

`resin.ManagedRunner` describes the managed lifecycle: publish a frozen
`runner_config`, then accept one `bind_resources(ResourceBinding)` call before
execution. An optional `release(sequence_ids)` callback clears model-local
state on completion, cancellation, and retraction. Cache pages always remain
Engine-owned. The one-method `Runner` protocol remains sufficient for a runner
that does not declare managed resources.

An exception before work is accepted is retryable. An exception marked
`_resin_post_accept = True` or a malformed result after execution quarantines
the batch resources because execution may have changed persistent state or GPU
use may still exist. An unmarked Python exception is an assertion by the runner
that the step had no persistent effects or rolled them back. Native Rust
runners express this explicitly with `engine::Error::rejected`; other native
execution failures are fatal.
CUDA runners validate their plans before entering
`resin.backend.cuda.failures.CudaFailureGuard`. Once execution begins, any
failure quarantines resources even if CUDA synchronization succeeds: an
in-place recurrent-state or sliding-window update cannot be undone by waiting
for the device. Graph host-input validation remains retryable because it occurs
before device copies or model execution.
