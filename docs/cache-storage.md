# Async prefix storage

Pass `Engine(runner, prefix_cache=True, prefix_cache_storage=provider)`.
`MemoryCacheStorage(max_bytes=...)` provides an asynchronous, bounded CPU DRAM tier;
its LRU budget counts key and payload bytes. The caller owns the provider and calls
`provider.close()` after closing its engines.

Resin hashes token prefixes, selects published checkpoints, and maps cache groups to bytes.
Full-history chunks are stored once per aligned span; window and recurrent state accompany
each checkpoint. Providers implement three methods over opaque immutable byte keys:

- `lookup(keys)` returns a future yielding one presence boolean per key.
- `load(keys, buffers)` and `store(keys, buffers)` return futures yielding a success boolean.

`buffers[i]` is a tuple of borrowed contiguous byte regions for `keys[i]`, including
empty tuples for checkpoint manifests. Runners expose regions through
`cache_storage_buffers(groups)`; built-in CUDA runners supply this hook.
Submissions and `future.done()` must not block. Completion, including an exception or
failure, must prove every buffer access has retired; providers must order GPU streams.
A result of `None` means retirement is unknown, so Resin retains those buffers.

Loads gate only dependent requests; stores pin immutable snapshots. Cancellation does
not retire transfers, and engine close waits. At most two operations retain resources
per engine. Storage and eviction may outlive engines. The default namespace is private;
set `prefix_cache_namespace` to share compatible weights/adapters/dtypes/layout/shards.
Metrics expose `cache.storage_pending`, `resin:cache_storage_pending`, and restored
request cached tokens.
