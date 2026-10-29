# Async prefix storage

Use `Engine(runner, prefix_cache=True, prefix_cache_storage=provider)`.
`lookup(tokens, maximum, alignment)` returns an operation yielding an aligned prefix length or zero.
`load(tokens, groups, buffers)` and `store(...)` return operations yielding success (one) or failure (zero).
Groups contain `(first_logical_block, [(slab, slot), ...])`; buffers contain tuples of borrowed byte regions.
Runners expose those regions through `cache_storage_buffers(groups)`; providers own matching and transport.
Submissions and `operation.poll(False)` must never block; `None` means pending. `poll(True)` waits.
Terminal results persist and, including failures, prove all accesses retired; providers must order GPU streams.
Unknown retirement retains buffers. Loads gate only dependent requests; stores pin immutable snapshots.
Cancellation is not retirement; close waits. At most two storage operations retain resources per Engine.
Providers own eviction, lifetimes, and compatibility namespaces covering weights/adapters/dtypes/layout/shards.
Local Engine metrics expose `cache.storage_pending`, `resin:cache_storage_pending`, and restored request cached tokens.
