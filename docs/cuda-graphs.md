# CUDA graph policies

CUDA runners built on the shared `resin.backend.cuda.runner.CudaRunner` expose
separate prefill and decode policies. For example:

```python
from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner

runner = Gemma4CudaRunner(
    "/path/to/checkpoint",
    device="cuda",
    cuda_graph_prefill="breakable",
    cuda_graph_decode="full",
)
runner.capture_cuda_graphs()
# Construct the Engine and start accepting requests after capture succeeds.
```

These are the default policies. Capture is an explicit startup operation: it
warms kernels, allocates fixed buffers, captures every configured shape, and
checks replay against eager execution. A selected but uncaptured graph raises
an error. The Modal harnesses perform capture before inference or HTTP startup.

Both policies accept `"disabled"`. Decode also accepts `"breakable"`; prefill
does not yet support `"full"`. For fully eager execution, disable both policies.

## Scope

Full decode captures the model's fixed-shape forward and sampling path. Models
with tensor parallelism include their token broadcast. Input staging, attention
planning, and Engine scheduling remain host work before replay. A full-decode
program cannot contain an explicit eager region.

Breakable decode permits explicit eager regions between captured regions. In
DeepSeek's ordinary tensor-parallel decode, this keeps the token broadcast
outside capture. Other paths currently use the same graph for both policies.

Breakable prefill captures **stateless tensor regions** within the existing
model forward. Each layer calls the shared graph runner for its MLP or MoE;
normalization and residual operations are included where the model already
groups them with that computation. GPT-OSS additionally captures QKV projection
and RoPE. DeepSeek captures the MLA input
projection and Q normalization, the local query projection after CP selection,
and inverse RoPE with the output projections. These use a shared tensor-region
contract: row shapes and dtypes describe fixed-address inputs and outputs,
including tuples of tensors. The ordinary forward calls the same operations
with graph capture disabled.

Attention planning, attention kernels, cache writes,
recurrent updates, media processing, and sampling retain their existing paths.
DeepSeek passes token IDs into its hash-routed experts. Qwen3.5 and Inkling keep
their recurrent/convolution state outside these graphs.

Prefill capture is bounded to these tensor regions. It uses the shared
`CudaGraphStep`/`CudaGraphProgram` runtime and the model's ordinary forward
implementation. The policy names follow
[SGLang's CUDA graph configuration](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_config.py),
but Resin's captured regions and model coverage are its own.

## Shapes and memory

When `cuda_graph_decode_shapes` is omitted, a shared batch bucket catalog uses
the runner's physical history capacity. Larger buckets can serve smaller
batches; inactive rows use reserved cache storage. Speculative decode uses
denser batch buckets to limit repeated verification work. Attention still reads
the live sequence lengths rather than doing work through the history capacity.

An explicit shape list overrides the catalog. An explicit empty tuple preserves
the older decode opt-out; it does not disable prefill. A nonempty list with
`cuda_graph_decode="disabled"` is an error.

Prefill uses power-of-two token buckets through the model's prefill chunk size.
Regions of at most 128 tokens may pad their stateless inputs to the next bucket.
Larger regions replay only exact buckets; partial chunks run at their actual
size and record a `padding` fallback. This avoids doubling large feed-forward
work merely to fit a graph. Attention and recurrent state always see the real
token count. Inputs outside the captured token or dtype range also run eagerly
and record a fallback. Graph inputs and outputs stay on the GPU.

Model integrations can require exact buckets even for short prefills through
the private region configuration. DeepSeek and GLM use this policy; padding
DeepSeek's 17-token input to 32 failed its numerical comparison. Check the
runner's region configuration for its padding limit.

Models that change row counts between regions can pass `region_tokens` to the
distributed prefill scope: a mapping from callback index ranges to local row
demands. The coordinator agrees a bucket for each range before the forward
pass, including idle replicas, and each region uses its agreed bucket. Model
code owns the selection policy; the graph runner only handles ranges and sizes.
V4.1 uses this to give its full producer and compact consumer regions separate
capacities.

All modes in one `CudaGraphExecutor`, including prefill and decode, share one
private capture memory pool and one capture stream. Sharing is automatic;
there is no per-mode opt-in. Each mode captures larger buckets first, while
separate executors and standalone caches keep independent pools.
Prefill regions additionally share input/output buffers per token bucket and
tensor layout.
Regions execute serially: a caller must consume a region's output before
another region overwrites the shared buffer. Concurrent replay on multiple
streams is unsupported. Each current model consumes the output into a new
attention/residual tensor or persistent state before the next region.

Each program must initialize its temporary workspace on every replay. Persistent
state, inputs, outputs, and values passed across graph breaks must live in
fixed-address storage allocated outside capture. This permits alternating modes
and batch buckets without depending on another capture's temporary contents.
Closing one mode releases its graphs without invalidating the remaining modes;
executor shutdown synchronizes and releases every graph before its shared
workspace. These ownership constraints follow
[PyTorch's shared-pool guidance](https://docs.pytorch.org/docs/2.14/notes/cuda.html#sharing-memory-across-captures).

Graph replay always returns its device buffers. The caller consumes or copies
them before replay can overwrite them; the graph cache does not materialize
host token lists. Synchronous output helpers return owned `ForwardOutput`
columns. Deferred `DeviceCompletion` stages accepted tokens and counts, records
an event, and constructs those columns after that event completes.

`PrefillTensorGraphs` owns buffer allocation, padding and replay for both
feed-forward and projection regions. Feed-forward regions use coordinated
capacities and valid-row counts across the execution mesh, including idle
replicas. Projection regions select local buckets.

## Validation and statistics

`runner.cuda_graph_stats` reports capture counts, replay counts, eager fallback
reasons, shape capacities, capture time, and allocation deltas per mode. These
deltas are not total retained/reserved graph memory. Prefill
includes `capture_scope="feed_forward_regions"`. GPT-OSS and DeepSeek also report
`prefill_attention`, with `capture_scope="attention_projections"` and named
`captured_operations`. Both modes use private shape keys
`(token_bucket, one_based_region_index)`. Explicit graph-break counts describe
eager regions inside a `CudaGraphProgram`; they do not count host/model work
between separate tensor-region graph calls. In particular, eager attention
between projection callbacks does not increment that counter.

Projection graphs capture no request metadata or page tables. They consume the
actual local rows selected by the ordinary CP path. Expert dispatch retains its
shared bucket across attention-DP replicas; pure projections choose their own
local bucket, because they have no collectives across those replicas. Both
models use exact projection buckets to preserve eager GEMM row geometry:
padding can change rounding, even when each operation is row-independent.

Capturing a single GEMM can cost more in staging than it saves in launches.
GPT-OSS keeps its attention output projection eager: the 32-token Hopper probes
measured 16–19 µs eager versus 24–30 µs through a separate graph in the CP/DP
layouts. QKV plus RoPE amortizes that boundary across more kernels. These are
region measurements, not an end-to-end throughput result.

On a CUDA host, exercise shared-buffer reuse, changed inputs, padding, and replay:

```bash
uv run pytest tests/test_cuda_prefill_graphs.py tests/test_cuda_tensor_graphs.py -q
```

The former Modal validation wrapper was retired.
For end-to-end performance, use `scripts/bench.py` with prefill/decode graph
policies and `--trace-dir`; see [benchmarking](benchmarks.md).

A missing prefill capture is rejected before model state changes. A capture or
replay failure poisons the runner. Construct a fresh runner rather
than falling back after CUDA may have changed persistent state. See
[the model authoring contract](model-forward-pass.md#cuda-graphs) for cache,
continuation, speculative decode, and real-history validation requirements.

The initial projection validation passed all 16 shared-runtime checks on a
[B300](https://modal.com/apps/ekzhang/main/ap-uO1BoGsDsOcdWhBlbmFcvm), plus bounded
full-checkpoint workloads with 32-token chunks on two GPUs. Current regression
coverage includes the [GPT-OSS projection tests](../models/cuda/tests/test_gpt_oss_prefill_projections.py)
and the shared tests above. See also the
[DeepSeek validation record](../models/cuda/resin_models_cuda/deepseek_v41/README.md#validation).
The GPU probes compare nonzero eager and captured projections exactly, and
exercise changing contexts, chunk sizes, page ownership, and idle replicas.
They do not establish full-model numerical parity across quantized layouts.

## Initial measurements

The September 11, 2026 validation uses the pinned checkpoints and environments
in the Modal scripts. The local suite passes 405 tests (40 GPU skips). All 12
shared-runtime tests pass on
[B300](https://modal.com/apps/ekzhang/main/ap-yQTbA9eHl5d4fqnRx58Nlz).

The first policy padded every prefill to a power of two. That was a regression:
Qwen's first feed-forward region at 129 live tokens took 789 µs with a 256-token
graph versus 585 µs eager. Gemma on RTX PRO 6000 similarly took 743 µs versus
617 µs. Inkling's final feed-forward region also failed numerical comparison
between 129 and 256 rows. The final policy keeps these large partial chunks at
their actual size rather than loosening the numerical checks.

[GPT-OSS 120B on two H200s](https://modal.com/apps/ekzhang/main/ap-JilfRMfGHPmKsxGPHzr6Xs)
passes first/last-region comparisons, exact large-bucket replay, a 9,001-token
chunk-crossing workload, and exact greedy output parity for the first three
requests when a fourth live request replaces a padded row. The parity probe
uses 129 prompt tokens and eight output tokens, with seven decode graph replays
for each batch size. Capture takes 7.73 seconds for three decode shapes and 504
prefill region shapes with saved kernel artifacts.

First-region median timings in microseconds, including configured-policy input
staging, with immutable reuse of the FlashInfer tuning cache:

| GPT-OSS live tokens | Eager at live size | Configured policy | Selected path |
| ---: | ---: | ---: | --- |
| 3 | 249.43 | 116.27 | Graph padded to 4 |
| 128 | 491.87 | 483.85 | Exact graph |
| 129 | 471.30 | 470.99 | Eager |
| 4,097 | 4,067.96 | 4,064.13 | Eager |
| 8,192 | 7,473.16 | 7,484.78 | Exact graph |

Small cases use seven alternating rounds of 20 calls; large cases use three
rounds of five calls. These are region measurements, not an end-to-end throughput
claim. Small launch-bound work benefits; the largest compute-bound case is
effectively unchanged. Validation replays are included in reported replay totals.

### Initial generation measurements and remaining limits

The five integrations measured in this initial validation passed the twelve
first/last-region numerical probes, including exact graph buckets and awkward
lengths that use the eager fallback. Completed generation runs from that
validation are:

| Model / hardware | Capture time, decode + prefill shapes | Completed workload |
| --- | --- | --- |
| [GPT-OSS 120B / H200 × 2](https://modal.com/apps/ekzhang/main/ap-JilfRMfGHPmKsxGPHzr6Xs) | 7.73 s, 3 + 504 | Batches 3 and 4, 9,001 prompt tokens, 16 output tokens each; separate padded-batch greedy parity probe |
| [Gemma 4 / RTX PRO 6000](https://modal.com/apps/ekzhang/main/ap-dL2VaSUIp14Mughxsr04kb) | 65.88 s, 20 + 780 | Image input, 16 output tokens, speculative decoding |
| [Qwen3.8-27B / B200](https://modal.com/apps/ekzhang/main/ap-GsBjupN2h3dfy8Rjmia3eb) | 32.47 s, 6 + 845 | Qwen3.5-family runner; text and image input, 16 output tokens each, speculative decoding |
| [Inkling / B300](https://modal.com/apps/ekzhang/main/ap-4WsdpFmDjGCcjhxGrfwwnk) | 65.53 s, 20 + 546 | 16 output tokens, speculative decoding |

These are startup costs in the deployed environments, including kernel warmup;
they are not isolated graph-capture costs or a cold-cache comparison. GPT-OSS,
Qwen, and the completed Inkling smoke preceded the final removal of redundant
views and a self-copy in the shared host path. The final shared GPU suite passes;
[Inkling's final host path](https://modal.com/apps/ekzhang/main/ap-NIDezJfwd8H7Yirug8bimc)
also passes all twelve region probes, but that run's final generation result was
not recovered after the client disconnected. Its later reruns were cancelled.

The eager fallback still has a small dispatch cost. For example, Inkling's final
129-token probes measured 123.16 → 130.78 µs at the first region and
708.16 → 725.11 µs at the last region. At three tokens, the same regions measured
127.59 → 89.90 µs and 696.90 → 119.13 µs with graph replay. This is not evidence
of a speedup at every shape, and the smoke runs do not establish end-to-end
throughput improvement.

DeepSeek's final [ordinary](https://modal.com/apps/ekzhang/main/ap-au8nBe6WqQIaxgv96dNGul)
and [speculative](https://modal.com/apps/ekzhang/main/ap-KcO7QwGNp2KdP5Ywqfj7ut)
profiling runs capture 559 prefill region shapes and pass all twelve numerical
probes. They exercise a 4,113-token prompt (a 4,096-token graph chunk followed
by a 17-token eager tail), drain the profiled generation requests, and pass
cached-prefix continuation against fresh execution. Both reach the subsequent
clean timed request, but Modal cancels the calls before a complete timing and
final tensor-parallel token-agreement report is returned. Server logs report
cancellation without identifying its source. No numerical or CUDA failure
preceded those cancellations. Complete final DeepSeek timing and rank-agreement
validation therefore remain outstanding; the partial runs are not counted as
completed model benchmarks.
