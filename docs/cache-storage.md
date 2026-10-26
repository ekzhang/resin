# Immutable prefix storage

Prefix storage extends Resin's device prefix cache with independently owned
objects. A stored object contains opaque cache bytes for a reusable prefix;
loading it can avoid recomputing that prefix. The Engine still selects legal
checkpoint boundaries, allocates cache pages, and owns request progress.

Storage is optional. Without `prefix_cache_storage`, the Engine runs the existing
native cache path without storage callbacks, transfer queues, or storage I/O.
This feature does not route requests or implement prefill/decode disaggregation.

## Using a provider

The reference `MemoryCacheStorage` is a bounded, process-local LRU store:

```python
from resin import Engine
from resin.cache_storage import MemoryCacheStorage

store = MemoryCacheStorage(max_bytes=1 << 30)

with Engine(runner, prefix_cache=True, prefix_cache_storage=store) as engine:
    # Submit and step requests normally.
    ...
```

The provider and its objects belong to the application, independently of any
one Engine. Closing an Engine must retire its outstanding transfers before
releasing their buffers; it does not clear a shared provider. Reuse a provider
across compatible engines when cross-engine prefix reuse is useful.

Namespaces default to an Engine-private identity. Sharing the same provider
alone therefore does not enable cross-engine hits. Supply the same explicit
`prefix_cache_namespace` only to engines whose stored cache bytes are compatible.
`prefix_cache_max_pending` defaults to `2` and bounds accepted load/store
transactions and their retained device pages. It must be positive. Lookup
requests reserve no device pages and are bounded by `max_running_requests`.

`step()` may return no events while a lookup or transfer is pending. Continue
using `has_work()` to drive the Engine; outstanding physical transfers remain
work even after their request finishes or is cancelled. `close()` drains
accepted transfers before returning resources.

The prototype integrates storage with the local Engine. A distributed runner
needs an adapter that coordinates transfers across every rank holding the
request's cache; `DistributedWorker` does not yet provide that adapter. A shared
namespace describes compatibility and does not establish replica coordination.

Promotion currently materializes a complete grouped checkpoint into fresh
device pages. An existing shorter device prefix stays valid until promotion
commits, so the temporary reservation can be larger than copying only a missing
suffix. If it cannot fit, the request continues from its existing device prefix
through ordinary prefill. The storage protocol does not require this particular
promotion strategy.

`MemoryCacheStorage` copies host data into its own immutable objects. Its CUDA
path stages through owned pinned host memory and uses per-operation events to
establish retirement. That implementation choice does not require other
providers to use host staging. Providers may implement files, object storage,
remote memory, or direct GPU transfers through the same buffer contract.

## The storage boundary

`resin.cache_storage` defines a small batched interface:

```python
class CacheStorage:
    def lookup(self, keys: tuple[bytes, ...]) -> CacheOperation: ...

    def load(
        self, keys: tuple[bytes, ...], buffers: tuple[CacheBuffer, ...]
    ) -> CacheOperation: ...

    def store(
        self, keys: tuple[bytes, ...], buffers: tuple[CacheBuffer, ...]
    ) -> CacheOperation: ...


class CacheOperation:
    def poll(self) -> CacheResult | None: ...
    def wait(self) -> CacheResult: ...
```

Keys are immutable opaque bytes. Load and store batches have one buffer per key;
result success flags preserve that order. A lookup hit is advisory: eviction can
occur before a later load. A completed miss is different from an operation that
has not completed.

`CacheBuffer(regions, nbytes, device)` describes one object's byte ranges in
payload order. Regions can be contiguous host byte buffers or contiguous
`torch.uint8` tensors. Every region belongs to the declared device, and their
sizes sum to `nbytes`. Several regions allow scatter/gather transfers without
requiring the Engine to concatenate an object into Python `bytes`. Load
destinations must be writable. A provider must explicitly support the memory
kind it receives; host staging is a provider concern.

The storage provider does not interpret token IDs, attention layers, recurrent
state, page tables, or model-specific replay rules. Those remain Engine and
runner concerns. A provider only transfers the supplied byte ranges under their
keys. It does not choose a shorter prefix, repair missing model state, or publish
cache pages into the Engine.

## Completion and ownership

An operation retains all buffers and other native owners it needs until it
returns a terminal `CacheResult(success, error=None)`:

1. `poll()` returns `None` immediately while work remains pending. It must not
   wait for network, disk, or device completion.
2. A returned result proves that all native accesses to the batch's buffers have
   retired. This guarantee applies even when some success flags are false or
   `error` is set.
3. `wait()` may block to establish that same guarantee. Repeated completion
   reads must preserve the terminal result.

Holding a Python reference alone does not protect a buffer's contents. Store
sources must not be modified, and load destinations must not be read, published,
or recycled until the operation retires. GPU providers must also preserve the
producer/transfer stream ordering needed before reading a source and before
reporting destination readiness.

The Engine holds the corresponding native page owners through retirement.
Logical cancellation, timeout, request completion, or an attempted operation
cancellation cannot by itself release those owners. A load or store exception
does not prove that accepted native data access stopped; if retirement cannot be
established, resources must remain quarantined rather than reenter the reusable
pool. Lookup operations receive metadata keys without cache buffers, so lookup
submission, polling, or waiting failures can fall back to misses without
quarantining cache memory. Malformed completion results still violate the
provider protocol.
If a data transfer cannot prove retirement, its quarantined buffer owners remain
alive even after the Engine is destroyed.

Ordinary missing objects are cache misses. A transfer failure must not publish
partly filled destinations as reusable prefixes. A stale positive lookup must
not cause the same failed load to be retried indefinitely. Only a complete,
validated snapshot may be materialized into an Engine request, including all
required history, window, and recurrent-state components.

## Immutability and compatibility

Store is idempotent: an already published key may succeed without rewriting its
object. Once visible, an object's bytes do not change. New objects become visible
only when the entire payload is ready; temporary files, incomplete network
writes, and in-progress GPU copies cannot count as hits. Eviction removes
availability for future operations without invalidating buffers retained by
already submitted operations.

A stable namespace is a compatibility promise, not merely a model display name.
It must distinguish every setting that changes the meaning or arrangement of
stored bytes, including:

- weight revision and adapter identity;
- input identity and position rules, including media and positional transforms;
- cache data types, group geometry, and layout/format version;
- rank or shard identity when the format depends on the parallel layout.

Do not reuse a namespace after changing one of these without proving byte
compatibility. The provider cannot infer that proof by inspecting an opaque
payload. Content identity must cover the preceding prefix, rather than treating
an isolated token block as sufficient context.

## Observations

The Engine's storage manager owns observations independently of the provider.
`engine.metrics_snapshot().storage` exposes cumulative lookup, load, store,
error, byte, restored-token, pending-batch, and latency observations when metrics
are enabled. A storage backend needs no metrics callback. The HTTP endpoint
exports the last published snapshot without polling storage during a scrape.
See [metrics](metrics.md#prefix-storage) for names and counting rules.

## Design references and tradeoffs

These are design references, not provider compatibility claims. No upstream code
is required to implement the Resin protocol.

- [vLLM's native offloading interface at `2a2717d9`](https://github.com/vllm-project/vllm/blob/2a2717d9b8e418088000d9d66c427d9351e531fd/vllm/v1/kv_offload/base.py)
  separates lookup readiness, pinning, asynchronous transfer submission, and
  completion. Resin retains explicit completion and ownership while keeping the
  provider API smaller than vLLM's scheduler/worker management hierarchy.
- [SGLang's HiCache storage interface at `d7f832d3`](https://github.com/sgl-project/sglang/blob/d7f832d30ac3d88cf111ba242e3217dbf6ee6212/python/sglang/srt/mem_cache/hicache_storage.py)
  provides batched operations with per-entry outcomes and registered host pools.
  Its pool-aware prefix checks show why all required cache components must agree
  on a reusable boundary. Resin keeps model geometry outside the store and
  accepts device-specific buffers without requiring a registered host tier.
- [TokenSpeed's L3 store at `6b238003`](https://github.com/lightseekorg/tokenspeed/blob/6b2380031602ce67bd94c209cd5e6a650fb529c5/tokenspeed-engine/crates/ts-l3/src/store.rs)
  transfers immutable keys and host byte ranges in batches. Its result separates
  success, object failures, and native retirement; its store lifetime survives
  client close. Resin uses terminal operation completion as the retirement proof
  and supports byte regions beyond host memory.
- [DeepInfer's README at `55374155`](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md)
  explicitly lists a complete KV cache system among components not included in
  that release. Its [production deployment notes](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#kv-cache-storage)
  describe metadata hash-chain matching followed by asynchronous transfers from
  3FS through shared host memory to HBM. Those notes inform the separation of
  matching, storage, and device execution; they do not establish a published
  reusable storage API.

Batching and independent completion ownership avoid per-page scheduler waits.
Keeping the provider unaware of model semantics avoids duplicating the cache
allocator inside each transport. A mandatory host hierarchy would simplify some
backends but would also impose staging and extra copies on GPU-capable ones.
Resin leaves that tradeoff to the provider and runner while keeping page ownership
and legal-prefix publication in the Engine.
