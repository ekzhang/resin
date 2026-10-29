# Prefix storage

```python
from resin import Engine, MemoryCacheStorage

store = MemoryCacheStorage(max_bytes=1 << 30)
engine = Engine(runner, prefix_cache=True, prefix_cache_storage=store)
```

Storage extends the device prefix cache with immutable, opaque byte objects.
The provider owns its objects and eviction policy independently of the Engine;
closing the Engine does not clear the provider. `MemoryCacheStorage` is a bounded
DRAM LRU, with blocking copies for CUDA buffers. Other providers can use files,
RDMA/Mooncake, or 3FS without changing Resin's allocator.

The deliberately synchronous interface is:

```python
lookup(keys: tuple[bytes, ...]) -> tuple[bool, ...]
load(keys: tuple[bytes, ...], buffers: tuple) -> tuple[bool, ...]
store(keys: tuple[bytes, ...], buffers: tuple) -> tuple[bool, ...]
```

Each buffer is a tuple of contiguous byte regions: writable memoryviews or
`torch.uint8` tensors. Several regions permit scatter/gather without forcing
concatenation or host staging. Results contain one boolean per key, in order.
Writes are immutable and idempotent; objects become visible atomically.
Lookups are advisory because an object can disappear before a load.

**All buffer accesses must finish before returning or raising.** This includes
GPU stream ordering, transport failures, and background workers. Providers may
use asynchronous I/O internally, but must wait before returning to Resin.
Buffers cannot be retained for later access. This contract lets Resin reuse its
existing checkpoint owners without another transfer scheduler or lease system.
Storage latency therefore blocks Engine dispatch; this prototype does not
overlap storage transfers with model execution.

On a device-cache miss, Resin probes context-dependent block keys and imports a
complete grouped checkpoint into fresh pages. Partial loads and provider errors
fall back to prefill. An existing shorter device prefix takes priority; storage
does not promote its missing suffix. Only authenticated immutable checkpoints
are exported, including sliding-window and recurrent state. Existing objects
are probed before export so repeated checkpoints do not retransmit old blocks.
Key enumeration still scans the checkpoint's token history.

The default namespace is private to each Engine. For cross-engine reuse, pass
the same explicit `prefix_cache_namespace` (nonempty `str` or `bytes`) to
compatible engines. This is the caller's promise that weights, adapters, input
identity, position rules, cache dtype/layout, and shard identity all match.
Resin cannot infer compatibility from a model name. Model runners expose exact
block regions through `cache_storage_buffers(groups)`, where groups are in
layout order and contain `(first_logical_block, [(slab, slot), ...])`.

`engine.metrics_snapshot().storage` contains cumulative object `lookups` and
`hits` (including export deduplication), successful `load_bytes`/`store_bytes`,
and failed callback `errors`. Successful object bytes count even if another
object in that checkpoint misses. HTTP exports the same counters as
`resin:cache_storage_<name>_total`. Collection never calls the provider.
Request cached-token accounting includes restored prefixes already attached to
the device cache. Metrics remain disabled with `enable_metrics=False`.

This integrates with the local Engine. Distributed replica coordination and
prefill/decode disaggregation are outside this storage prototype.

The design takes immutable batched objects from
[TokenSpeed](https://github.com/lightseekorg/tokenspeed/blob/6b2380031602ce67bd94c209cd5e6a650fb529c5/tokenspeed-engine/crates/ts-l3/src/store.rs),
explicit buffer lifetime boundaries from
[vLLM](https://github.com/vllm-project/vllm/blob/2a2717d9b8e418088000d9d66c427d9351e531fd/vllm/v1/kv_offload/base.py),
and per-object outcomes from
[SGLang](https://github.com/sgl-project/sglang/blob/d7f832d30ac3d88cf111ba242e3217dbf6ee6212/python/sglang/srt/mem_cache/hicache_storage.py).
[DeepInfer's deployment notes](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#kv-cache-storage)
describe 3FS/host/HBM transfers; its public release does not include a complete
KV cache system. Resin keeps transport policy in the provider and starts with a
synchronous boundary to keep the prototype small.
