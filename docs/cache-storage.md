# Prefix cache storage

Prefix cache storage lets Resin reuse model state after it has been evicted from
GPU memory. The Engine manages prefix matching and the lifetime of device pages;
a storage provider keeps immutable checkpoints outside the
[device cache pool](cache-pool.md). Resin includes a bounded CPU DRAM provider, and
custom providers can implement other storage tiers through the same interface.

## Using the DRAM cache

Construct a provider and pass it to an Engine with prefix caching enabled. Given a
runner and prompt token IDs:

```python
from resin import Engine, MemoryCacheStorage, SamplingParams

storage = MemoryCacheStorage(max_bytes=8 * 1024**3)  # 8 GiB
try:
    with Engine(
        runner,
        prefix_cache=True,
        prefix_cache_storage=storage,
    ) as engine:
        request = engine.submit(
            token_ids,
            sampling_params=SamplingParams(max_output_tokens=128),
        )
        while not request.done:
            engine.step()
        output_tokens = request.take_result()
finally:
    storage.close()
```

`max_bytes` is required. The provider reserves its arena at construction, using
pinned host memory when CUDA is available. Its LRU budget charges object keys,
region names and payload bytes; general Python and allocator overhead is outside
that budget. One background worker owns a private CUDA copy stream per device.
It queues pending reads before fencing the stream and completing their futures.
Writes retire earlier reads before evicting or reusing their host memory.

The caller owns the provider. Close every Engine using it before calling
`storage.close()`. Stored objects can outlive an Engine, so compatible Engines can
reuse the same provider.

## How reuse works

Resin first looks for a usable checkpoint in the GPU prefix cache. If there is no
GPU hit, it asks the storage provider which checkpoint objects are available. A
storage hit is loaded into private device pages before the request resumes from
that checkpoint. The unmatched suffix, including any model-required replay
inputs, then runs normally. A partial GPU hit takes precedence over storage;
Resin does not search storage for a longer match in that case.
Requests for prompt logprobs run a full prefill instead of reusing checkpoints.

While a request waits for a load, the Engine can schedule other requests if cache
capacity permits. Transfers retain device pages, so memory pressure can still
limit dispatch. If an object has been evicted or a load fails, the request falls
back to prefill. Storage is an optional cache, so a successful presence check does
not guarantee that a later load will hit.

A checkpoint contains the state needed to resume at an aligned token boundary.
Full-history data is divided into objects for aligned spans, which can be shared
by several checkpoints. Sliding-window and recurrent state belong to a specific
checkpoint endpoint. Providers receive these objects as opaque keys and byte
regions; the Engine decides which combination forms a usable prefix.

## Choosing a write policy

Set `prefix_cache_storage_policy` when constructing the Engine:

| Policy | When Resin offers a checkpoint to storage |
|---|---|
| `"write_through_selective"` (default) | When another request reuses that checkpoint from the GPU cache. |
| `"write_through"` | When the checkpoint is published, including its first use. |

With the selective policy, the first request creates a GPU checkpoint without
writing it to storage. A second request that reuses it triggers a backup of that
exact checkpoint. Later requests can restore the backup after GPU eviction.
Newly computed suffixes remain in the GPU cache until another request reuses
their checkpoints.

Selective writes avoid transferring prefixes that are used only once. Choose
`"write_through"` when prefixes are likely to be evicted from the GPU before their
first reuse: the selective policy cannot promote a checkpoint that is already
gone. Under either policy, providers skip copies of objects that remain resident
and coalesce duplicate pending writes. Eviction can make a later copy necessary.

## Concurrency and shutdown

`prefix_cache_max_inflight` bounds outstanding lookup, load and store **batches**
for one Engine. It counts operations, not keys or CUDA streams. Omitting it, or
passing `None`, uses `max(2, max_running_requests)`; an explicit value must be
positive. A larger limit permits more outstanding work and can retain more GPU
cache memory while transfers finish.

At the limit, new storage lookups and writes are skipped rather than queued by
the Engine. Requests that skip lookup proceed with prefill. A selective write
skipped at one GPU hit can be offered again on a later hit. The provider may
apply its own concurrency limits within the work it accepts.

Request completion or cancellation does not mean its storage transfers have
finished. The Engine keeps borrowed pages alive until the provider reports that
they are safe to recycle. Engine shutdown waits for those transfers; the provider
can continue to retain its own cached data afterward.

`engine.metrics_snapshot().cache["storage_pending"]` reports outstanding batches;
Prometheus exposes the same count as `resin:cache_storage_pending`. Restored prefix
lengths contribute to request cached-token metrics.

## Implementing a provider

The [`CacheStorage` protocol](../resin/cache_storage.py) has three methods. Each
submits work and returns a `concurrent.futures.Future` promptly, without waiting
for I/O. These are ordinary methods returning futures, not `async def` methods.

| Method | Result |
|---|---|
| `lookup(keys)` | A tuple of presence booleans in key order. |
| `store(keys, buffers)` | Transfer success, or unknown completion as described below. |
| `load(keys, buffers)` | Transfer success, or unknown completion as described below. |

Keys are opaque `bytes` values. For transfers, `keys` and `buffers` have equal
lengths, and `buffers[i]` maps string region names to borrowed, contiguous
`torch.uint8` tensors for `keys[i]`. Tensors may be on the CPU or GPU.
A store reads those views; a load writes into them. Match fields by name, not by
dictionary insertion order. An empty mapping is valid: it records a checkpoint
manifest with no payload.

Objects are immutable. Before reading source buffers, a provider must skip keys
it already holds and coalesce writes of the same pending key. The Engine may
offer an object more than once, including on repeated GPU cache hits.

### Keys and sharing

Keys identify token prefixes and checkpoint parts. They do not include a model
name, weight version or cache-layout fingerprint. The provider's instance or
backing-store namespace determines which Engines share data. Use separate
providers or partitions for incompatible model or cache representations.

### Transfer completion

A transfer future also defines when Resin may recycle its buffers:

- `True` means the transfer succeeded and all accesses to the borrowed buffers
  have finished. A successful load must fill every requested region.
- `False`, or a terminal exception, means the operation failed but all buffer
  accesses have finished. The Engine can discard a failed restore safely.
- `None` means buffer access may still be outstanding. Resin retains the owners;
  Engine shutdown raises instead of releasing memory that may still be in use.

Providers must order GPU copies after the work that produced their inputs. A
separate CUDA stream alone does not establish that ordering. Resolve a transfer
with `True`, `False` or an exception only after all borrowed-buffer accesses have
finished. `future.done()` must also return promptly. Cancellation is not proof of
completion.

Completion covers the borrowed buffers. A provider can manage several tiers by
staging data in its own allocation, then continuing remote I/O against that copy
after the GPU transfer has finished. The provider owns that copy and its lifetime.

## Runner support

Built-in CUDA runners implement
`cache_storage_buffers(groups, *, prefix_length)`. Custom runners return one
named-region mapping per supplied physical block, in group and block order.
These must be views of the bytes needed to resume that checkpoint; constructing
the views must not copy data. Guard-only blocks return `{}`. Include every byte
that a resumed kernel can read, even when that requires exporting a complete
intersecting page rather than only individual live rows.

The hook must accept empty groups with `prefix_length=0` and return no blocks.
The Engine uses this capability check during construction. Missing or
unsupported exporters raise `NotImplementedError` before resources are bound.
