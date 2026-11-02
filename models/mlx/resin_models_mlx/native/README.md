# Private Metal cache storage

`CacheStorage` owns one MLX-allocated buffer for an Engine cache arena. It is a
mutable inference resource, rather than an immutable tensor value. BF16 and
FP32 storage are supported; writes copy their raw bits without floating-point
conversion. The Qwen adapter converts checkpoint values to the arena's BF16
storage format before writing.

## Ordering contract

`initial_epoch()` returns an eight-element `uint32` dependency token. Its first
evaluation allocates and zeroes the arena, including unused and null storage.
Repeated calls return the same initial token; they do not reset the arena.
Every mutation takes the preceding epoch and returns the next epoch. Epochs
express execution order, not historical versions of the buffer.

`read_epoch(epoch)` produces a private, transient alias for model attention or
gather operations. Before a later write, the adapter passes **all computed
reader outputs** through the writer's `after=` argument. These become ordinary
primitive inputs and bound GPU resources. An alias-only `mx.depends` edge is
insufficient across command-encoder boundaries. Keeping the alias alone as a
dependency is also insufficient: the model operations which consume it must
precede mutation. Old read views must never escape as immutable values or be
reused after advancing the epoch. The Engine retains its writer lease across
continuations and drains writes before releasing or reassigning cache ownership.

`snapshot(epoch)` returns an independent, immutable copy of the entire arena.
The wrapper records each issued snapshot as a pending reader, makes the next
mutation consume its real output buffer, then clears the pending list after
successful construction. Thus a snapshot created
before a write preserves the earlier contents even if it has not yet been
evaluated. Calling `snapshot` with an old epoch after later writes have already
been issued cannot recover historical contents. Snapshots are explicit debug
or inspection operations; metadata queries do not allocate copies.

`scatter_rows` treats the arena as rows of `shape[-1]` elements. Indices must be
rank-one `int32` or `int64`, and valid target rows must be unique. Negative and
out-of-range device indices are ignored by guards. `write_slice` addresses a
contiguous flat interval, checked on the host. Both accept strided updates by
constructing a lazy contiguous input; neither evaluates inputs on the CPU.

## GPU lifetime and synchronization

The native write primitives bind the existing arena through MLX's Metal encoder
as an output resource. Attention and gather consumers bind private views of the
same buffer as inputs. MLX therefore sees the actual read/write hazards, while
the epoch inputs carry graph and stream dependencies. Writes explicitly reuse
the private allocation; they do not depend on immutable-array donation or unique
reference counts.

The writer binds each computed-reader and pending-snapshot buffer at an unused
argument slot. This registers their completion frontiers with MLX even when an
event has ended the previous encoder. The binding does not copy data or add a
dispatch. The writer's fresh epoch token carries that order to later writes.

Each submitted cache operation also retains the arena's `Data` in a Metal
command-buffer completion handler. This keeps the allocation alive while GPU
commands use it, even if Python objects or graph nodes have been released.
`buffer_identity` reports the actual Metal buffer identity without evaluating an
array; `allocation_count` counts arena allocations, excluding tokens and copies.

The Qwen runner evaluates the initial epoch during startup, before handing the
runner to a serving actor. Subsequent operations select the calling thread's
current MLX GPU stream. No stream object is cached across threads. Ordinary
decode continuations use GPU dependencies instead of a CPU wait between passes;
terminal and ownership transitions wait for the cache epoch to complete.

BF16 decode with eight query heads, two KV heads and dimension 256 uses the
native partition reader. It binds the real arena and byte offset without
charging its entire extent to MLX's command-buffer packing heuristic. Explicit
barriers and completion-held arena data preserve ordering and lifetime. The
native and generic MLX readers share one Metal body; other geometries use the
generic reader. D256 merges with 2–31 partitions use the exact small-reduction
kernel; other merges use the compiled MLX reduction.

Linear shader addresses are 64-bit. Row scatter uses one grid axis per row
and column; both axes fit uint32 because MLX dimensions are int32. Other
dispatches above 1,048,576 elements use a bounded two-dimensional grid and guard
the padded tail. Shape and byte counts are checked for `size_t` overflow;
allocation remains subject to Metal's buffer and memory limits. Initialization
warms five fixed pipelines for the owner's byte width: zero, snapshot, slice,
and row scatter with each index width. Batch size, context length and runtime
addresses do not select new cache pipelines.

## Build and scope

The extension uses MLX's `MetalExtension` build helper and native C++ API. Build
and runtime are pinned to **MLX 0.32.4**, whose C++ ABI must match. Building on
macOS requires Xcode's C++ and Metal command-line tools and CMake 3.25 or later.
The package builds the extension and its adjacent Metal library during normal
installation; non-macOS metadata builds skip the Metal extension. This change
does not upgrade MLX or the package version.

Cache operations require a GPU stream and do not support `compile`, automatic
differentiation, vectorization, or export. Pure model regions can still use
MLX compilation outside this resource boundary. The Python wrapper imports
`mlx.core` before the extension to load MLX and register its array types,
including for editable installations.
