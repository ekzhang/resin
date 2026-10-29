# Async prefix storage

Pass `Engine(runner, prefix_cache=True, prefix_cache_storage=provider)`.
`MemoryCacheStorage(max_bytes=...)` provides an asynchronous, bounded CPU DRAM tier.
The caller must set `max_bytes`; its LRU budget counts key, region-name and payload bytes.
The caller owns the provider and calls `provider.close()` after closing its engines.

Resin hashes token prefixes, selects published checkpoints, and maps cache groups to bytes.
Full-history chunks are keyed per aligned span; providers skip copies while those objects
remain resident. Window and recurrent state accompany each checkpoint. Providers implement
three methods over opaque immutable byte keys:

- `lookup(keys)` returns a future yielding one presence boolean per key.
- `load(keys, buffers)` and `store(keys, buffers)` return futures yielding a success boolean.

`buffers[i]` maps stable string names to borrowed contiguous `torch.uint8` views for
`keys[i]`; an empty dict is a checkpoint manifest. Names label parts of one object,
and providers restore by name rather than depending on dict insertion order.
Runners implement `cache_storage_buffers(groups, *, prefix_length)`, returning one
dict per supplied physical block, in group/block order. Export only bytes needed to
resume that checkpoint; guard-only blocks return `{}`. Every byte read after loading
must be restored or overwritten before use. Constructing views must not copy data.
Built-in CUDA runners supply this hook. Unsupported runners raise `NotImplementedError`
at Engine construction, before resource binding; the hook must accept empty groups
with `prefix_length=0` and return no blocks for this capability check.
Providers own partitioning: storage keys depend only on token prefixes and object kind.
Use separate provider instances or backing-store partitions for incompatible model/cache
representations. Provider `store` must skip resident objects and coalesce pending puts
before reading GPU buffers.
Submissions and `future.done()` must not block. Completion, including an exception or
failure, must prove every buffer access has retired; providers must order GPU streams.
A result of `None` means retirement is unknown, so Resin retains those buffers.

Loads gate only dependent requests; stores pin immutable snapshots. Cancellation does
not retire transfers. Engine close waits; unknown retirement raises and retains buffers.
`prefix_cache_max_inflight` sets the positive per-engine limit on outstanding
lookup/load/store batches (default: 2). Storage and eviction may outlive engines.
Metrics expose `cache.storage_pending`, `resin:cache_storage_pending`, and restored
request cached tokens.
