// Private native binding for the current Resin partition-attention body.
// The honest full UNITS array remains an ordinary dependency input.
#include "common.h"
#include <nanobind/stl/vector.h>
#include <mlx/allocator.h>
#include <mlx/array.h>
#include <mlx/backend/metal/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace resin_models_mlx::native::attention {
constexpr int kHeads = 8, kKVHeads = 2, kHeadDim = 256;
constexpr int kPage = 128, kPartition = 512, kHeadGroups = 2;

bool supported_units(const mx::array& units) {
  return units.ndim() == 4 && units.shape(0) > 0 && units.shape(1) == 2 &&
         units.shape(2) == kPage && units.shape(3) == kKVHeads * kHeadDim &&
         units.dtype() == mx::bfloat16;
}

class AttentionPartitions : public mx::Primitive {
 public:
  AttentionPartitions(mx::Stream stream, int partitions, int layer, float scale,
                      int skip)
      : Primitive(stream), partitions_(partitions), layer_(layer), scale_(scale),
        skip_(skip) {}

  const char* name() const override { return "ResinPagedAttentionPartitions"; }
  // Primitive defaults to is_equivalent=false. No transform is admitted by
  // the host guard; this reader consumes an explicitly versioned mutable cache.
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("Native paged attention has no CPU implementation");
  }

  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    auto& device = mx::metal::device(stream().device);
    auto library = device.get_library(MLX_METAL_LIBRARY_NAME, library_directory());
    auto kernel = device.get_kernel("resin_attention_partition_bf16", library);
    if (kernel->maxTotalThreadsPerThreadgroup() < 128) {
      throw std::runtime_error("Native paged attention needs a 128-thread group");
    }
    for (auto& output : outputs) {
      output.set_data(mx::allocator::malloc(output.nbytes()));
    }
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    const auto& units = inputs[3];
    if (!units.flags().row_contiguous) {
      throw std::invalid_argument("Native attention UNITS must materialize contiguously");
    }
    // offset() is in BYTES. Preserve the real allocation and view offset;
    // never reshape/falsify the arena extent or compact the full buffer.
    auto units_buffer = static_cast<const MTL::Buffer*>(units.buffer().ptr());
    encoder.set_buffer(units_buffer, 3, units.offset());
    encoder.set_input_array(inputs[4], 4);
    encoder.set_input_array(inputs[5], 5);
    const std::array<int32_t, 4> geometry = {
        int32_t(inputs[4].shape(1)), int32_t(partitions_), int32_t(layer_),
        int32_t(skip_)};
    encoder.set_bytes(geometry.data(), 4, 6);
    encoder.set_bytes(scale_, 7);
    encoder.set_output_array(outputs[0], 8);
    encoder.set_output_array(outputs[1], 9);
    encoder.set_output_array(outputs[2], 10);
    // Also retain explicitly although ordinary input retention covers UNITS.
    // Retain Data, not merely MTLBuffer, to prevent allocator reuse.
    encoder.get_command_buffer()->addCompletedHandler(
        [keep = units.data_shared_ptr()](MTL::CommandBuffer*) {});
    encoder.barrier();
    const uint64_t groups = uint64_t(inputs[0].shape(0)) * kKVHeads *
                            kHeadGroups * uint64_t(partitions_);
    encoder.dispatch_threadgroups(MTL::Size(1, 1, groups), MTL::Size(128, 1, 1));
    encoder.barrier();
  }

 private:
  int partitions_;
  int layer_;
  float scale_;
  int skip_;
};

std::vector<mx::array> attention_partials(
    const mx::array& queries, const mx::array& current_keys,
    const mx::array& current_values, const mx::array& units,
    const mx::array& pages, const mx::array& lengths,
    int layer, float scale, int skip) {
  auto stream = checked_stream();
  if (queries.ndim() != 3 || queries.shape(0) < 1 ||
      queries.shape(1) != kHeads || queries.shape(2) != kHeadDim ||
      current_keys.ndim() != 3 || current_values.ndim() != 3 ||
      current_keys.shape() != current_values.shape() ||
      current_keys.shape(0) != queries.shape(0) ||
      current_keys.shape(1) != kKVHeads || current_keys.shape(2) != kHeadDim) {
    throw std::invalid_argument("Native paged attention requires Q[B,8,256], K/V[B,2,256]");
  }
  if (!supported_units(units)) {
    throw std::invalid_argument("Native paged attention needs UNITS[units,2,128,512]");
  }
  if (queries.dtype() != mx::bfloat16 || current_keys.dtype() != mx::bfloat16 ||
      current_values.dtype() != mx::bfloat16 || units.dtype() != mx::bfloat16) {
    throw std::invalid_argument("Native paged attention requires BF16 Q/K/V and UNITS; no narrowing");
  }
  const int rows = queries.shape(0);
  if (pages.ndim() != 2 || pages.shape(0) < rows || pages.shape(1) < 1 ||
      lengths.ndim() != 1 || lengths.size() < size_t(rows) ||
      pages.dtype() != mx::int32 || lengths.dtype() != mx::int32 ||
      layer < 0 || layer >= units.shape(0) || !std::isfinite(scale) ||
      skip < 0) {
    throw std::invalid_argument("Native paged attention has invalid metadata geometry/dtype/layer/scale");
  }
  const uint64_t longest = uint64_t(pages.shape(1)) * kPage + 1;
  const uint64_t partitions = (longest + kPartition - 1) / kPartition;
  const uint64_t groups = uint64_t(rows) * kKVHeads * kHeadGroups * partitions;
  if (longest > uint64_t(INT32_MAX) || groups > uint64_t(INT32_MAX)) {
    throw std::invalid_argument("Native paged attention exceeds current shader int geometry");
  }
  // Layout flags of lazy slices are not final. Contiguous is a no-copy alias
  // for the Engine arena, while unsupported strided values materialize before
  // this primitive, matching the canonical Metal adapter. Shape stays honest.
  std::vector<mx::array> inputs = {
      mx::contiguous(queries, false, stream),
      mx::contiguous(current_keys, false, stream),
      mx::contiguous(current_values, false, stream),
      mx::contiguous(units, false, stream),
      mx::contiguous(pages, false, stream),
      mx::contiguous(lengths, false, stream)};
  const int count = int(partitions);
  return mx::array::make_arrays(
      {{rows, kHeads, count, kHeadDim}, {rows, kHeads, count},
       {rows, kHeads, count}},
      {mx::float32, mx::float32, mx::float32},
      std::make_shared<AttentionPartitions>(stream, count, layer, scale, skip),
      inputs);
}
void warm_paged_attention() {
  // Warm a device-wide pipeline on the caller's current GPU stream. No
  // arrays, graph nodes, stream objects or request geometry are retained.
  auto stream = mx::to_stream({});
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument("Native attention warmup requires the current GPU stream");
  }
  auto& device = mx::metal::device(stream.device);
  auto library = device.get_library(MLX_METAL_LIBRARY_NAME, library_directory());
  device.get_kernel("resin_attention_partition_bf16", library);
}
}  // namespace resin_models_mlx::native::attention

namespace resin_models_mlx::native {
void bind_paged_attention(nb::module_& module) {
  module.def("_paged_attention_partials", &attention::attention_partials,
             "queries"_a, "current_keys"_a, "current_values"_a,
             "units"_a, "pages"_a, "lengths"_a, "layer"_a, "scale"_a,
             "skip"_a = 0);
  module.def("_warm_paged_attention", &attention::warm_paged_attention);
}
}  // namespace resin_models_mlx::native
