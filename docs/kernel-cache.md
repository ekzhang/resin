# Persistent kernel-artifact cache

Compiled kernel persistence is an execution-runtime concern, not Engine cache
policy. The Rust control plane does not select compilers or know where generated
libraries live. A Python runner or deployment harness may opt into
`resin.kernel_cache` before importing its accelerator framework.

Keep these stores conceptually separate:

- model checkpoints contain weights;
- the native flat cache pool contains live history and recurrent state;
- the kernel-artifact store contains compiled libraries and tuning records.

Tokenizer files may accompany a model checkpoint. Live tokenizer and
conversation-parser state belongs to the recipe layer, independently of the
device and kernel caches.

## Portable API

`resin.kernel_cache` uses the Python standard library and has no import-time
dependency on CUDA, PyTorch, FlashInfer, or Modal.

```python
from resin.kernel_cache import ArtifactCache

cache = ArtifactCache(storage_root, abi_namespace)
```

The host supplies both values. `abi_namespace` is one safe path component that
should encode every compatibility input relevant to the caller: GPU
architecture, Python, CUDA, framework, compiler, and kernel-library versions.
Changing any such input should select another namespace rather than mutate an
old one.

The main methods are:

```python
cache.validate(artifact, identity=...)
cache.publish(artifact, identity=...)
cache.publish_from(local_source, durable_artifact, identity=...)
cache.load_or_build(artifact, identity=..., load=..., build=...)
cache.load_or_build_from(
    durable_artifact,
    source=local_source,
    identity=...,
    load=...,
    build=...,
)
```

`hits`, `misses`, `root`, `abi`, and `record_path()` expose diagnostics. Paths
must remain inside the selected ABI namespace.

## Validation and publication

An artifact is reusable only when its sidecar record exactly matches the record
schema, ABI namespace, logical identity, relative path, byte count, and current
content digest. The current `ArtifactRecord` schema stores that digest in its
`blake3` field. Legacy schema 1 sidecars are misses; there is no legacy digest
compatibility path. This artifact identity belongs only to the persistent
kernel-artifact store; the live request cache has no content index.

Resin never uses mtime as proof that compiled bytes are reusable. Missing,
malformed, wrong-ABI, wrong-identity, truncated, or corrupted artifacts are
misses and enter the caller's normal build path.

`publish_from` supports building on compiler-friendly local scratch and copying
the finished result into durable storage. It flushes and atomically replaces the
artifact, then publishes the readiness sidecar last. A reader may see a miss
during publication but cannot validate a partial destination. A failed build or
publication does not create a new readiness record.

## Compiler adapters

The generic cache does not turn every compiler directory into a safe portable
cache. Each backend needs an adapter that knows how to build, publish, and load
its artifact.

`wrap_flashinfer_build_and_load` wraps a selected FlashInfer runtime-JIT method.
A covered AOT specialization remains upstream-owned. On a runtime-JIT hit, the
wrapper directly loads the content-validated durable library. On a miss,
FlashInfer may build locally and Resin publishes the completed library. The
caller remains responsible for version compatibility and installation policy.

FA4/CuTe DSL, Triton, compiler-extension helpers, and MoE tactic files each have
their own upstream reuse semantics. Routing all of their directories beneath one
mount is not equivalent to validating them with `ArtifactCache`.

The CUDA scorer uses QuACK's persistent CuTe object cache. With
`QUACK_CACHE_ENABLED=1`, compiled objects are exported under `QUACK_CACHE_DIR`
and loaded directly on subsequent processes. Serving routes this directory into
the shared kernel volume alongside the lower-level `CUTE_DSL_CACHE_DIR`.
Scoring keys include its source digest, target architecture, compiler/runtime
versions, and kernel configuration; batch size and requested top count remain
runtime arguments. In-memory executors are isolated by CUDA context, while disk
objects remain reusable across processes. QuACK owns locking, publication, and
unloadable-object recovery; these objects do not use Resin's `ArtifactCache`
sidecars.

## Runtime integration

The shared deployment registry selects compiler environment variables and
persistent cache volumes. Before importing GPU libraries, runtime setup creates
the compiler directories and installs the FlashInfer content-addressed loading
wrapper where applicable. A covered FlashInfer cache miss builds in local scratch
and publishes the completed library through `ArtifactCache`; later workers
validate that specific artifact before loading it.

The launchers capture graphs and warm up their model before serving, then commit
the kernel volume. They do not run a separate prewarm/reuse-verification job or
publish a whole-workload qualification marker. The former one-off cache probes
and saved-autotune harness have been retired; the portable artifact API remains.

Atomic artifact publication is not a distributed compiler lock. Multiple cold
writers can duplicate work or contend inside an upstream build directory.

Valid cache hits can avoid compilation for covered specializations. They do not
remove framework import, device-context creation, weight loading, graph capture,
or model warmup. A new layout, bucket, or backend may still compile.

The parameterized `deploy/modal_serve.py` and the Modal adapter used by
`scripts/bench.py` share registry-selected cache volumes and `deploy/jit_cache.py`
helpers. A benchmark trace records execution after warmup; it is not by itself
a cold-start or no-recompilation qualification.

## Current limitations

- The portable store is filesystem-backed; an object-store API needs an adapter.
- The FlashInfer wrapper is version-specific and must be revalidated on upgrade.
- Concurrent cache misses have no distributed compiler lock.
- A validated compiled artifact says nothing about live cache correctness.
- Historical GPT-OSS cache qualification covered a single-sequence H200 run;
  it does not establish current cold-worker or arbitrary-shape behavior.
- Similar TTFT values do not prove absence of compilation. That claim requires
  compilation-event instrumentation or before/after artifact evidence, with
  the measured cache paths and process scope recorded.
