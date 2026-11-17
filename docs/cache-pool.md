# Unified paged cache

Each Engine owns a pool of equal-sized physical slabs shared by all of a model's
cache groups. The Engine manages allocation, prefix reuse, and resource
lifetimes. The runner allocates the device arena and maps physical cache
addresses into model tensors.

## Cache layout

A runner declares an ordered `CacheLayout` with one `CacheGroupLayout` for each
kind of model state:

| `family`  | `retention` | Stored state                                                                       |
| --------- | ----------- | ---------------------------------------------------------------------------------- |
| `HISTORY` | `HISTORY`   | Full token history, such as attention keys and values.                             |
| `HISTORY` | `WINDOW`    | A bounded chronological suffix of attention or rolling history.                    |
| `STATE`   | `HISTORY`   | One private live recurrent state per request, with optional immutable checkpoints. |

For History groups, `block_tokens` is the number of token positions covered by a
block. For State groups, it specifies the minimum checkpoint alignment. `blocks_per_slab`
declares how many blocks of that group fit in one physical slab.

An allocated slab belongs to one group. When its blocks are released, the slab
returns to the common pool and can serve any group. Admission accounts for each
request's demand across all groups, including their packing into slabs.

The runner receives the pool ID and frozen layout through `bind_resources`.
Subsequent batch plans address this bound arena.

Use `CacheLayout(replay_suffix_tokens=N)` when a model needs at least `N` prompt
inputs recomputed after restoring a cached prefix. The Engine always leaves at
least one input for execution.

### Window retention

A Window group declares two lengths:

- `read_window_tokens` is the logical read window, including the current token.
- `window_tokens` is the physical retention budget. `read_window_tokens` defaults
  to this value when omitted.

A chunk of `Q` writes with a read window of `W` requires
`window_tokens >= W + Q - 1`. The retained span contains both the earliest
query's history and every write in the chunk. For example, a 128-token read
window and 64-token chunk require at least 191 retained tokens.

The Engine caps prefill chunks at the smallest
`window_tokens - read_window_tokens + 1` across Window groups and rejects decode
lookahead that exceeds it. Larger chunks therefore require a larger retention
budget per live request.

## Batch plans

`BatchPlan.cache_groups` contains one `CacheGroupPlan` per declared group, in
layout order. Each group plan describes all sequences in the batch. A physical
block has address `(slab_id, slotnum)`; a token location adds `block_offset`.
Slab zero is reserved for inactive/null rows, and live slab IDs are one-based.

Each plan contains a CSR block table and explicit write locations:

- `block_indptr` slices the parallel `block_slab_ids` and `block_slotnums`
  arrays by sequence.
- `block_table_start_indices[i]` is the absolute logical block index of sequence
  `i`'s first table entry. With 64-token blocks, index 37 covers positions
  2368–2431. Full-history rows begin at zero; Window rows advance as older
  blocks are evicted.
- `append_slab_ids`, `append_slotnums`, and `append_block_offsets` specify the
  write location for each scheduled token.

`BatchPlan.prefill_lengths` gives each request's final prefill endpoint across
chunks. It includes generated inputs when reconstructing a retracted request.

## Execution and resource lifetimes

The Engine allocates blocks, applies copy-on-write to shared history, and holds
reader/writer leases through execution and native commit. Physical addresses
remain stable for the lifetime of a lease. Completion and cancellation release
resources only after outstanding accesses have finished.

Overlapped decode can reserve a write span covering several steps. The Engine
bounds it by cache geometry, window retention, context length, output budget,
and available capacity. Page tables describe the reserved span; the runner uses
accepted GPU positions to determine which pages and token positions are readable.

A recurrent group's private live slot stays fixed across checkpoint boundaries.
Its state descriptor supplies initialization or restore copies and optional
snapshot destinations. The runner updates the live state and reports snapshots
only for destinations matching the exact accepted prefix. The shared CUDA
`execute_state` wrapper performs these transfers outside model graphs and
attaches completion receipts.

## Prefix caching

A checkpoint holds the model state needed to continue from a token prefix. With
`Engine(..., prefix_cache=True)`, a new request can reuse a matching checkpoint
and compute only the remaining suffix.

The Engine finds the longest usable checkpoint through a flat SHA-256 index of
prefix identities. Reuse requires matching text or media identity and the same
cache namespace. Checkpoint endpoints align to all groups' `block_tokens`, so
every group resumes at the same token position.

For models with recurrent state or sliding-window attention, state tracking and
checkpoint retention are separate settings:

- `state_tracking_interval=256` sets the recurrent-state tracking interval in
  tokens, independently of the prefill chunk size. The effective interval is the
  least common multiple of this value and the cache groups' block sizes.
- `prefix_cache_checkpoint_interval=0` disables periodic retention. Set it to
  `N` to retain additional checkpoints at multiples of `N` tokens, aligned with
  the effective checkpoint interval.

For these models, an active request keeps its latest usable checkpoint, replacing
older tracking snapshots as it advances. Generation completion preserves that
checkpoint, which may precede the final token because of alignment. The Engine
also preserves checkpoints at detected shared prompt boundaries. If a newly
detected boundary has no cached state, the incoming request computes it before
it can be reused. Checkpointing never splits an atomic media span.

Models with only full-context KV retain ordinary prefix reuse: earlier aligned
prefixes remain available through shared history pages, subject to LRU eviction.

On a hit, the request shares the checkpoint's immutable history pages and copies
any recurrent state into its own private live slot.

Cached checkpoints use the same slab pool as live requests. When requests need
more capacity, the Engine evicts least-recently-used checkpoints. Pages remain
allocated until their last owner releases them.

[Prefix storage](cache-storage.md) extends reuse beyond the device pool with
asynchronous storage tiers.

## Physical sizing

The runner chooses a slab byte size that accommodates its cache groups and
declares each group's `blocks_per_slab`. Packing may include padding; the Engine
accounts for all demand in slab units.

CUDA runners normally use
[`plan_cache_capacity`](../resin/backend/capacity.py) after loading weights and
before allocating the cache or capturing CUDA graphs. The helper uses free VRAM,
the memory-utilization target, workspace reservations, and slab size to determine
usable capacity after reserving the null slab. Runners can also impose an
explicit capacity limit.
