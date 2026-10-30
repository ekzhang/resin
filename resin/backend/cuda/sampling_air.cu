/*
 * Copyright (c) 2026 Resin contributors.
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * The initialization, workspace layout and first-pass radix kernel are adapted
 * from FlashInfer's air_top_p.cuh. Helpers and later passes use its installed
 * header. The radix algorithm derives from TensorRT-LLM's
 * samplingAirTopPKernels.cu.
 * https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/include/flashinfer/air_top_p.cuh
 */

#include <flashinfer/air_top_p.cuh>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <math_constants.h>
#include <type_traits>
#include "tvm_ffi_utils.h"

namespace resin_sampling {

namespace air = flashinfer::sampling::air_top_p;
using Counter = air::Counter<float>;
using Histogram = air::HisT<true, float>;
using tvm::ffi::Optional;

constexpr int kPrivateGroups = 4;
constexpr int kPrivateBuckets = 512;
constexpr size_t kPrivateHistogramBytes =
    kPrivateGroups * kPrivateBuckets * (sizeof(Histogram) + sizeof(air::IdxT));
// The first-pass bucket range and reduction below require this upstream layout.
static_assert(air::BITS_PER_PASS == 11 && air::NUM_BUCKETS == 2048 && air::BLOCK_SIZE == 512);
static_assert(std::is_same_v<Histogram, uint64_t>);

__device__ __forceinline__ void BuildFirstPassHistogram(
    const float* input, air::IdxT length, Histogram* histogram, air::IdxT* counts,
    Histogram* shared_histogram, air::IdxT* shared_counts) {
  for (air::IdxT i = threadIdx.x; i < air::NUM_BUCKETS; i += blockDim.x) {
    shared_histogram[i] = 0;
    shared_counts[i] = 0;
  }
  __syncthreads();
  const int start_bit = air::calcStartBit<float>(0);
  const uint32_t mask = air::calcMask<float>(0);
  extern __shared__ __align__(16) unsigned char private_histogram[];
  auto* private_mass = reinterpret_cast<Histogram*>(private_histogram);
  auto* private_counts = reinterpret_cast<air::IdxT*>(
      private_mass + kPrivateGroups * kPrivateBuckets);
  for (int i = threadIdx.x; i < kPrivateGroups * kPrivateBuckets; i += blockDim.x) {
    private_mass[i] = 0;
    private_counts[i] = 0;
  }
  __syncthreads();
  const int group = (threadIdx.x / 32) % kPrivateGroups;
  auto add = [start_bit, mask, shared_histogram, shared_counts,
              private_mass, private_counts, group](float value, air::IdxT) {
    const int bucket = air::calcBucket<float>(value, start_bit, mask);
    // Privatize the populated positive range to reduce inter-warp contention.
    // Other buckets retain the full histogram path for large raw weights.
    if (bucket >= 512 && bucket < 1024) {
      const int slot = group * kPrivateBuckets + bucket - 512;
      air::histAtomicAdd<true, float>(private_mass + slot, value);
      atomicAdd(private_counts + slot, static_cast<air::IdxT>(1));
    } else {
      air::histAtomicAdd<true, float>(shared_histogram + bucket, value);
      atomicAdd(shared_counts + bucket, static_cast<air::IdxT>(1));
    }
  };
  air::vectorizedProcess<float>(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                               static_cast<size_t>(blockDim.x) * gridDim.x, input, length, add);
  __syncthreads();
  for (int bin = threadIdx.x; bin < kPrivateBuckets; bin += blockDim.x) {
    Histogram mass = 0;
    air::IdxT count = 0;
#pragma unroll
    for (int group = 0; group < kPrivateGroups; ++group) {
      mass += private_mass[group * kPrivateBuckets + bin];
      count += private_counts[group * kPrivateBuckets + bin];
    }
    shared_histogram[bin + 512] = mass;
    shared_counts[bin + 512] = count;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < air::NUM_BUCKETS; i += blockDim.x) {
    if (shared_histogram[i] != 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(histogram + i),
                static_cast<unsigned long long>(shared_histogram[i]));
    }
    if (shared_counts[i] != 0) atomicAdd(counts + i, shared_counts[i]);
  }
}

__global__ void AirTopPFirstPassPrivateKernel(Counter* counters, Histogram* histograms,
                                            air::IdxT* count_histograms) {
  const int row = blockIdx.y;
  Counter* counter = counters + row;
  const air::IdxT current_length = counter->len;
  if (current_length == 0) return;
  auto* histogram = histograms + row * air::NUM_BUCKETS;
  auto* counts = count_histograms + row * air::NUM_BUCKETS;
  __shared__ Histogram shared_histogram[air::NUM_BUCKETS];
  __shared__ air::IdxT shared_counts[air::NUM_BUCKETS];
  BuildFirstPassHistogram(counter->in, counter->oriLen, histogram, counts,
                          shared_histogram, shared_counts);
  __syncthreads();
  __threadfence();
  bool is_last_block = false;
  if (threadIdx.x == 0) {
    const uint32_t finished = atomicInc(&counter->finishedBlockCnt, gridDim.x - 1);
    is_last_block = (finished == gridDim.x - 1);
  }
  // Only the last finisher reduces; no CTA waits for another CTA to be resident.
  if (__syncthreads_or(is_last_block)) {
    float* histogram_values = reinterpret_cast<float*>(shared_histogram);
    for (int i = threadIdx.x; i < air::NUM_BUCKETS; i += blockDim.x) {
      const uint64_t value = histogram[i];
      const air::IdxT count = counts[i];
      if (count != 0) {
        const uint32_t bucket_bits = i << air::calcStartBit<float>(0);
        histogram_values[i] = air::calcFloatValue(
            static_cast<uint32_t>(count),
            air::calcExponent(air::twiddleOut<float>(bucket_bits, false)), value);
      } else {
        histogram_values[i] = 0.0f;
      }
    }
    __syncthreads();
    // Keep upstream's two-stage FP32 reduction and boundary search order.
    constexpr int kWarpSize = 32;
    constexpr int kWarpCount = air::NUM_BUCKETS / kWarpSize;
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float warp_sum[kWarpCount];
    __shared__ cuda::atomic<float, cuda::thread_scope_block> block_sum;
    for (int i = threadIdx.x; i < air::NUM_BUCKETS; i += air::BLOCK_SIZE)
      cg::reduce_store_async(warp, warp_sum + i / kWarpSize, histogram_values[i], cg::plus<float>{});
    __syncthreads();
    if (threadIdx.x < kWarpSize) {
      cg::reduce_store_async(warp, block_sum, warp_sum[threadIdx.x], cg::plus<float>{});
      cg::reduce_update_async(warp, block_sum, warp_sum[threadIdx.x + kWarpSize], cg::plus<float>{});
    }
    __syncthreads();
    const float current_sum = block_sum * counter->p;
    if (threadIdx.x == 0) {
      float previous = 0;
      int target_step = 0;
      for (int i = 0; i < kWarpCount; i++) {
        if (warp_sum[i]) {
          target_step = i;
          if ((previous + warp_sum[i]) >= current_sum) break;
          previous += warp_sum[i];
        }
      }
      int target_index = 0;
      for (int i = target_step * kWarpSize; i < air::NUM_BUCKETS; i++) {
        if (counts[i]) {
          target_index = i;
          if ((previous + histogram_values[i]) >= current_sum) break;
          previous += histogram_values[i];
        }
      }
      counter->sum = current_sum - previous;
      counter->len = counts[target_index];
      counter->kthValueBits |= static_cast<uint32_t>(target_index) << air::calcStartBit<float>(0);
      // A nonempty bucket with zero integer mantissa mass consists entirely
      // of powers of two: the <2^31 input bound keeps the sum below 2^54.
      // Publish its full exact cutoff and let the original later passes exit.
      if (counts[target_index] > 0 && histogram[target_index] == 0) {
        const uint32_t exponent_bits = air::calcExponent(
            air::twiddleOut<float>(counter->kthValueBits, false));
        const float exact_value = __uint_as_float(exponent_bits);
        // Zero/subnormal-zero and nonfinite boundaries cannot be certified.
        if (exact_value > 0.0f && exact_value < CUDART_INF_F) {
          counter->kthValueBits = air::twiddleIn<float>(exact_value, false);
          counter->len = 0;
          counter->previousLen = -1;
        }
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < air::NUM_BUCKETS; i += blockDim.x) {
      histogram[i] = 0;
      counts[i] = 0;
    }
    if (threadIdx.x == 0) {
      if (counter->previousLen != -1) counter->previousLen = current_length;
      counter->filterCnt = 0;
    }
  }
}

__global__ void InitAirThresholds(Counter* counters, const float* weights,
                                  const float* top_ps, const bool* active,
                                  Histogram* histograms, air::IdxT* count_histograms,
                                  int vocab, const double* guard_mass,
                                  const bool* guard_active, int guard_parts) {
  const int row = blockIdx.x;
  Counter* counter = counters + row;
  const float p = top_ps[row];
  bool enabled = active[row] && p > 0.0f && p < 1.0f;
  // Top-p is redundant when the full inclusive top-k set lies well before
  // its mass boundary. FP64 partials and a 0.001 probability margin keep
  // this decision far from AIR's FP32 histogram rounding. P-only rows and
  // inactive rows do not read scratch or execute these reductions.
  if (enabled && guard_mass && guard_active[row] && p > 0.001f) {
    using Reduce = cub::BlockReduce<double, 256>;
    __shared__ typename Reduce::TempStorage total_storage;
    __shared__ typename Reduce::TempStorage retained_storage;
    __shared__ bool redundant;
    double total = 0.0;
    double retained = 0.0;
    const int64_t base = static_cast<int64_t>(row) * guard_parts * 2;
    for (int part = threadIdx.x; part < guard_parts; part += blockDim.x) {
      total += guard_mass[base + part * 2];
      retained += guard_mass[base + part * 2 + 1];
    }
    total = Reduce(total_storage).Sum(total);
    retained = Reduce(retained_storage).Sum(retained);
    if (threadIdx.x == 0) {
      redundant = total > 0.0 && retained >= 0.0 &&
                  retained < (static_cast<double>(p) - 0.001) * total;
    }
    __syncthreads();
    enabled = !redundant;
  }
  if (threadIdx.x == 0) {
    counter->in = weights + static_cast<int64_t>(row) * vocab;
    counter->oriLen = vocab;
    counter->len = enabled ? vocab : 0;
    counter->previousLen = vocab;
    counter->p = p;
    counter->sum = 0;
    counter->kthValueBits = 0;
    counter->finishedBlockCnt = 0;
    counter->filterCnt = 0;
  }
  // All three AIR passes check len before touching histogram or vocabulary
  // memory. A later replay that enables this row initializes every bin here.
  if (!enabled) return;
  const int64_t base = static_cast<int64_t>(row) * air::NUM_BUCKETS;
  for (int i = threadIdx.x; i < air::NUM_BUCKETS; i += blockDim.x) {
    histograms[base + i] = 0;
    count_histograms[base + i] = 0;
  }
}

__global__ void EmitAirThresholds(const Counter* counters, float* output,
                                  int64_t output_stride, int rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float cutoff = 0.0f;
  if (counters[row].len > 0 || counters[row].previousLen == -1) {
    // Init resets this marker on every call, including inactive graph replays.
    cutoff = air::twiddleOut<float>(counters[row].kthValueBits, false);
  }
  output[static_cast<int64_t>(row) * output_stride] = cutoff;
}

inline size_t Align256(size_t size) { return (size + 255) / 256 * 256; }

inline void CheckCuda(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess) << "AIR cutoff failed: " << cudaGetErrorString(status);
}

void AirTopPCutoff(TensorView weights, TensorView top_ps, TensorView active,
                    TensorView output, TensorView workspace,
                    Optional<TensorView> guard_mass, Optional<TensorView> guard_active) {
  CHECK_INPUT_AND_TYPE(weights, dl_float32);
  CHECK_INPUT_AND_TYPE(top_ps, dl_float32);
  CHECK_INPUT_AND_TYPE(active, dl_bool);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_CUDA(output);
  CHECK_INPUT_TYPE(output, dl_float32);
  CHECK_DIM(2, weights);
  CHECK_DIM(1, top_ps);
  CHECK_DIM(1, active);
  CHECK_DIM(1, output);
  CHECK_DIM(1, workspace);
  CHECK_DEVICE(weights, top_ps);
  CHECK_DEVICE(weights, active);
  CHECK_DEVICE(weights, output);
  CHECK_DEVICE(weights, workspace);
  const int64_t rows64 = weights.size(0);
  const int64_t vocab64 = weights.size(1);
  TVM_FFI_ICHECK(rows64 >= 0 && rows64 < INT32_MAX);
  TVM_FFI_ICHECK(vocab64 >= air::NUM_BUCKETS && vocab64 < INT32_MAX);
  TVM_FFI_ICHECK_EQ(top_ps.size(0), rows64);
  TVM_FFI_ICHECK_EQ(active.size(0), rows64);
  TVM_FFI_ICHECK_EQ(output.size(0), rows64);
  TVM_FFI_ICHECK_GT(output.stride(0), 0);
  TVM_FFI_ICHECK_EQ(guard_mass.has_value(), guard_active.has_value());
  const double* guard_mass_ptr = nullptr;
  const bool* guard_active_ptr = nullptr;
  int guard_parts = 0;
  if (guard_mass.has_value()) {
    const auto mass = guard_mass.value();
    const auto enabled = guard_active.value();
    CHECK_INPUT_AND_TYPE(mass, dl_float64);
    CHECK_INPUT_AND_TYPE(enabled, dl_bool);
    CHECK_DIM(3, mass);
    CHECK_DIM(1, enabled);
    CHECK_DEVICE(weights, mass);
    CHECK_DEVICE(weights, enabled);
    TVM_FFI_ICHECK_EQ(mass.size(0), rows64);
    TVM_FFI_ICHECK_GT(mass.size(1), 0);
    TVM_FFI_ICHECK_LT(mass.size(1), INT32_MAX);
    TVM_FFI_ICHECK_EQ(mass.size(2), 2);
    TVM_FFI_ICHECK_EQ(enabled.size(0), rows64);
    guard_parts = static_cast<int>(mass.size(1));
    guard_mass_ptr = static_cast<const double*>(mass.data_ptr());
    guard_active_ptr = static_cast<const bool*>(enabled.data_ptr());
  }
  if (rows64 == 0) return;
  const int rows = static_cast<int>(rows64);
  const int vocab = static_cast<int>(vocab64);
  const int buffer_length = air::calcBufLen<float>(vocab);
  const size_t counters_size = Align256(sizeof(Counter) * rows);
  const size_t histogram_size = Align256(sizeof(Histogram) * air::NUM_BUCKETS * rows);
  const size_t count_size = Align256(sizeof(air::IdxT) * air::NUM_BUCKETS * rows);
  const size_t buffer_size = Align256(sizeof(float) * buffer_length * rows);
  const size_t workspace_size = counters_size + histogram_size + count_size + 2 * buffer_size;
  TVM_FFI_ICHECK_GE(static_cast<size_t>(workspace.size(0)), workspace_size);
  auto* base = static_cast<uint8_t*>(workspace.data_ptr());
  auto* counters = reinterpret_cast<Counter*>(base);
  auto* histograms = reinterpret_cast<Histogram*>(base + counters_size);
  auto* counts = reinterpret_cast<air::IdxT*>(base + counters_size + histogram_size);
  auto* buffer1 = reinterpret_cast<float*>(base + counters_size + histogram_size + count_size);
  auto* buffer2 = reinterpret_cast<float*>(base + counters_size + histogram_size + count_size + buffer_size);

  ffi::CUDADeviceGuard device_guard(weights.device().device_id);
  cudaStream_t stream = get_stream(weights.device());
  int sm_count;
  CheckCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                   weights.device().device_id));
  const uint32_t blocks = air::CalcAirTopPBlockNum<true, float>(rows, vocab, sm_count);
  const auto* weights_ptr = static_cast<const float*>(weights.data_ptr());
  const auto* ps_ptr = static_cast<const float*>(top_ps.data_ptr());
  const auto* active_ptr = static_cast<const bool*>(active.data_ptr());
  InitAirThresholds<<<rows, 256, 0, stream>>>(
      counters, weights_ptr, ps_ptr, active_ptr, histograms, counts, vocab,
      guard_mass_ptr, guard_active_ptr, guard_parts);
  CheckCuda(cudaGetLastError());
  const dim3 grid(blocks, rows);
  if (rows >= 16 && vocab >= 131072) {
    // Only pass 0 pays the private histogram's extra 24 KiB shared memory.
    // Later passes keep the upstream kernel, occupancy and original grid.
    CheckCuda(cudaFuncSetAttribute(AirTopPFirstPassPrivateKernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(kPrivateHistogramBytes)));
    const uint32_t first_blocks = std::max(blocks, (static_cast<uint32_t>(vocab) + 8191u) / 8192u);
    AirTopPFirstPassPrivateKernel<<<dim3(first_blocks, rows), air::BLOCK_SIZE,
                                   kPrivateHistogramBytes, stream>>>(counters, histograms, counts);
    CheckCuda(cudaGetLastError());
  } else {
    air::AirTopPRenormRadixKernel<true, float><<<grid, air::BLOCK_SIZE, 0, stream>>>(
        counters, histograms, counts, 0, buffer1, buffer2);
    CheckCuda(cudaGetLastError());
  }
  for (int pass = 1; pass < air::NUM_PASSES<float>; ++pass) {
    air::AirTopPRenormRadixKernel<true, float><<<grid, air::BLOCK_SIZE, 0, stream>>>(
        counters, histograms, counts, pass, buffer1, buffer2);
    CheckCuda(cudaGetLastError());
  }
  EmitAirThresholds<<<(rows + 255) / 256, 256, 0, stream>>>(
      counters, static_cast<float*>(output.data_ptr()), output.stride(0), rows);
  CheckCuda(cudaGetLastError());
}

}  // namespace resin_sampling

TVM_FFI_DLL_EXPORT_TYPED_FUNC(air_top_p_cutoff, resin_sampling::AirTopPCutoff);
