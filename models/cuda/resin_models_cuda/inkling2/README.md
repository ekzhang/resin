# Inkling 2 on Blackwell

`Inkling2CudaRunner` runs **Inkling 2 Small**, including its bundled DSpark
speculator and image encoder. The model family and deployment key are `inkling2`.

The deployment registry pins
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small/tree/2ad1157016e1aac77b6ff1b770e7c05229403134)
at `2ad1157016e1aac77b6ff1b770e7c05229403134`. The checkpoint mixes NVFP4 and
MXFP8 expert matrices; attention, dense projections, recurrent state and KV use
BF16. Loading follows each matrix's quantization metadata and materializes only
the rank's experts and attention shards.

## Execution

Attention tensor parallelism, attention data parallelism, expert parallelism and
expert tensor parallelism compose through Resin's shared distributed worker.
On two B300s, attention TP2 can use either EP2 (whole experts on each GPU) or
EP1/ETP2 (each expert's intermediate channels split across both GPUs). Attention
DP2/EP2 and four-GPU TP2/DP2/EP4 are also supported configurations. Idle attention
replicas still participate in expert communication. Context parallelism is not
implemented.

The `inkling2` Modal preset uses two B300s with TP2/EP2, a 1,048,576-token
context limit, up to 256 concurrent requests and seven DSpark proposals. It enables
the CuTe expert backend and fused attention prologue.

ETP preserves the full expert's activation scale and reduces W2 partial sums in
FP32 before rounding each route to BF16. With EP1 and matching attention TP,
both GPUs already have the inputs, so expert dispatch needs no all-to-all.

The target has 48 layers: 40 use a 1,024-token attention window and eight use
global attention. Attention uses the pinned FlashAttention 4 Blackwell kernel
with an FP32 output accumulator, followed by Inkling's sink and channel gate.
Short convolution and attention-residual operations use model-local fused
kernels. Expert execution preserves the checkpoint's NVFP4/MXFP8 scale layouts.

`expert_backend="triton"` is the default. `"cute-dsl"` selects SGLang's compact
CuTe-DSL NVFP4 GEMM; MXFP8 still uses Triton. Small, eligible NVFP4 batches use
native fused quantization and activation kernels. Their GEMM consumers can
prefetch immutable weights before the activation producer finishes, using
programmatic dependent launch to preserve the data dependency. Other shapes use
the general adapter. Both backends preserve Inkling's route weight before W2
activation quantization. The runner records the selected backend and actual
kernels in benchmark metadata.

Draft attention packs grouped queries and skips cached tiles outside its fixed
attention window. Split counts depend on batch and model geometry; history lengths
remain runtime inputs.

`fused_attention_prologue=True` combines clipping, convolution, normalization and
KV writes and prepares padded queries and relative bias for native single-request
and batched decode attention.

Inkling uses FlashInfer for eligible small attention and replicated-EP reductions
when the attention TP ranks share a host and support NVLink multicast. Other
distributed reductions use NCCL. Single-rank attention runs locally.

The Engine owns page allocation and prefix checkpoints. Separate cache groups
hold target global/local KV, optional draft global/local KV, and recurrent state.
The recurrent snapshot includes both convolution history and DSpark's target
feature. A restored prefix therefore supplies all state needed to continue.
Cache pages contain 128 tokens; prefill chunks must be multiples of 128.
Sliding retention includes the speculative verification tail and prefill chunk.

`speculative_draft_tokens=7` enables the released eight-position DSpark block:
seven proposals followed by a target bonus token. `0` selects target-only
decoding. Verification commits the accepted convolution state and target feature;
rejected KV positions remain beyond the accepted logical length. Decode capture
includes proposal, verification, acceptance and state commit. The shared Engine
completion path supplies device feedback for overlapping continuations.

The native `inkling2` recipe owns conversation formatting and image preprocessing.
The runner caches immutable vision features by media identity and inserts them
after text embedding normalization.

## Source references

- [SGLang's Inkling 2 integration](https://github.com/sgl-project/sglang/tree/e62ac6279646c686d3b328f58d9b5c2c2d5927dc),
  pinned at `e62ac6279646c686d3b328f58d9b5c2c2d5927dc`, supplies the architectural,
  numerical and kernel reference. Adapted model code retains its Apache-2.0
  attribution; the vendored CuTe NVFP4 GEMM retains its BSD-3-Clause license.
- Transformers at `3a134bc37f7c22c801b618e289b0255425f22647` supplies the independent
  model and image-processing reference used by small numerical tests.
- FlashAttention `4.0.0b26` supplies the Blackwell attention kernel. The relative
  bias specialization inherits its MMA, correction and output methods; local
  code handles bias loading and softmax. The launcher retains FP32 output;
  its BSD notice is in [LICENSE.flash-attention](LICENSE.flash-attention).
- FlashInfer `0.7.1` supplies the CuTe expert kernel's constructor and memory
  partition helpers. Local code retains compact expert routing, per-route FP32
  output and PDL weight prefetch. Both specializations depend on these pinned
  library internals and need GPU qualification when those dependencies change.
- [DeepSeek Ultrafast](https://github.com/basetenlabs/deepseek-ultrafast/tree/76766a97308bfd1658fb055e835d0f92de737dc3)
  was inspected for batch-one performance techniques: fused small expert
  operations, weight-load overlap, projection specialization and device-resident
  speculation. No implementation was copied. Its different model, speculator and
  GPU topology make its published throughput unsuitable as an Inkling 2 baseline.

## Qualification

The small regression suite covers checkpoint loading, chunked target execution,
DSpark state and cache ownership, and representative attention and expert kernels.
`test_inkling2_engine_gpu.py` exercises a real checkpoint under torchrun with
captured DSpark, overlap, cached/fresh prefixes, idle replicas and slot reuse.
Use two ranks with `RESIN_TEST_ATTN_TP=1` or `2`, or four ranks with TP=2, and set
`RESIN_TEST_CHECKPOINT` to the local checkpoint directory. Kernel tests require
CUDA; the Engine smoke test requires Blackwell GPUs.

Use the shared `deploy/modal_bench.py` qualification runner and
`scripts/bench.py --model inkling2`. The [32K decode comparison](../../../../artifacts/inkling2-qualification/batched-final-pipeline/comparison/README.md)
records measured throughput, kernel controls and the final TP2/DP2/EP4 Engine
regression. That study does not measure prefill or serving performance.
Comparisons must match precision, topology, context, batch capacity and natural
speculative acceptance.
