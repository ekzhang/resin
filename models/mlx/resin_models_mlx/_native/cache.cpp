// Engine-private mutable cache storage using MLX's Metal stream and hazards.
// This is an inference resource, not an immutable MLX tensor operation.
#include "common.h"
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <mlx/array.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/transforms.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {
using resin_models_mlx::native::checked_stream;
using resin_models_mlx::native::library_directory;

constexpr int kEpochElements = 8;

enum class Kind { Read, Rows, Slice };

mx::array allocate_arena(const std::vector<int>& dimensions) {
  mx::Shape shape(dimensions.begin(), dimensions.end());
  if (shape.empty()) {
    throw std::invalid_argument("CacheStorage requires a nonempty byte shape");
  }
  size_t bytes = 1;
  for (auto dimension : shape) {
    if (dimension < 1 || bytes > std::numeric_limits<size_t>::max() / size_t(dimension)) {
      throw std::invalid_argument("CacheStorage shape must be positive and fit size_t indexing");
    }
    bytes *= size_t(dimension);
  }
  return mx::zeros(shape, mx::uint8, checked_stream());
}

void check_epoch(const mx::array& epoch) {
  if (epoch.dtype() != mx::uint32 || epoch.ndim() != 1 || epoch.size() != kEpochElements) {
    throw std::invalid_argument("CacheStorage epoch must be an eight-element uint32 array");
  }
}

class CacheOperation : public mx::Primitive {
 public:
  CacheOperation(mx::Stream stream, std::shared_ptr<mx::array> storage, Kind kind,
                 uint64_t byte_start, uint64_t row_bytes)
      : Primitive(stream), storage_(std::move(storage)), kind_(kind),
        byte_start_(byte_start), row_bytes_(row_bytes) {}

  const char* name() const override { return "ResinPrivateCache"; }
  // Primitive defaults to is_equivalent=false. These resource operations must
  // not be CSE'd, differentiated, vectorized or graph-compiled.
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("CacheStorage has no CPU implementation");
  }

  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    auto& arena = *storage_;
    auto& output = outputs[0];
    if (kind_ == Kind::Read) {
      // Private read view only. The eventual attention/gather primitive binds
      // this exact resource as its input. No memory access occurs here.
      output.copy_shared_buffer(arena);
      return;
    }
    output.set_data(mx::allocator::malloc(output.nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto library = device.get_library(MLX_METAL_LIBRARY_NAME, library_directory());
    const auto& updates = inputs[kind_ == Kind::Rows ? 2 : 1];
    // Sliced contiguous inputs can have a nonzero buffer offset. Use word
    // copies only when both source and destination are aligned for every row.
    const bool aligned = updates.offset() % 4 == 0 && updates.nbytes() % 4 == 0 &&
                         byte_start_ % 4 == 0 && row_bytes_ % 4 == 0;
    const uint64_t word_bytes = aligned ? 4 : 1;
    std::string kernel_name = kind_ == Kind::Rows ? "resin_cache_rows" : "resin_cache_slice";
    kernel_name += aligned ? "_b4" : "_b1";
    if (kind_ == Kind::Rows) {
      kernel_name += inputs[1].dtype() == mx::int32 ? "_i4" : "_i8";
    }
    auto kernel = device.get_kernel(kernel_name, library);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    const size_t base_inputs = kind_ == Kind::Rows ? 3 : 2;
    // Keep every completed reader in the writer's resource frontier. An
    // alias-only Depends node can lose this edge across encoder boundaries.
    // Slot 30 is outside the shaders' arguments; repeated binding records
    // every real input buffer without copying any of its contents.
    for (size_t i = base_inputs; i < inputs.size(); ++i) {
      encoder.set_input_array(inputs[i], 30);
    }
    encoder.set_output_array(arena, 0);
    encoder.set_input_array(inputs[0], 1);
    encoder.set_output_array(output, 2);
    uint64_t work = updates.nbytes() / word_bytes;
    if (kind_ == Kind::Rows) {
      encoder.set_input_array(inputs[1], 3);
      encoder.set_input_array(inputs[2], 4);
      uint64_t width = row_bytes_ / word_bytes;
      uint64_t count = inputs[1].size();
      uint64_t total_rows = storage_->nbytes() / row_bytes_;
      encoder.set_bytes(width, 5);
      encoder.set_bytes(count, 6);
      encoder.set_bytes(total_rows, 7);
    } else {
      encoder.set_input_array(inputs[1], 3);
      uint64_t start = byte_start_ / word_bytes;
      encoder.set_bytes(start, 4);
      encoder.set_bytes(work, 5);
    }
    // arena is a resource captured by the primitive, not an ordinary input or
    // output array. Keep its Data alive explicitly until the GPU completes.
    // add_temporary() is unsuitable: it removes resources from hazard tracking.
    encoder.get_command_buffer()->addCompletedHandler(
        [keep = arena.data_shared_ptr()](MTL::CommandBuffer*) {});
    auto threads = std::min<size_t>(256, kernel->maxTotalThreadsPerThreadgroup());
    // Row byte width and rank-one index count fit the grid axes. The
    // shader uses 64-bit addresses for arenas larger than 4 GiB.
    if (kind_ == Kind::Rows) {
      const uint64_t width = row_bytes_ / word_bytes;
      const uint64_t count = inputs[1].size();
      const uint64_t columns = count ? std::max<uint64_t>(width, 8) : 8;
      encoder.dispatch_threads(MTL::Size(columns, std::max<uint64_t>(count, 1), 1),
                               MTL::Size(threads, 1, 1));
      return;
    }
    // Metal grid coordinates are uint32 per axis. Keep a bounded first
    // dimension and reconstruct the full linear address as ulong in MSL.
    work = std::max<uint64_t>(work, kEpochElements);
    constexpr uint64_t max_grid_width = 1 << 20;
    auto grid_width = std::min<uint64_t>(work, max_grid_width);
    auto grid_height = work / grid_width + (work % grid_width != 0);
    if (grid_height > UINT32_MAX) {
      throw std::invalid_argument("CacheStorage work exceeds the Metal grid capacity");
    }
    encoder.dispatch_threads(MTL::Size(grid_width, grid_height, 1),
                             MTL::Size(threads, 1, 1));
  }

 private:
  std::shared_ptr<mx::array> storage_;
  Kind kind_;
  uint64_t byte_start_;
  uint64_t row_bytes_;
};

class CacheStorage {
 public:
  explicit CacheStorage(const std::vector<int>& byte_shape)
      : storage_(std::make_shared<mx::array>(allocate_arena(byte_shape))),
        initial_(mx::zeros({kEpochElements}, mx::uint32, checked_stream())) {
    // Materialize on the creating thread; serving may use another MLX stream.
    mx::eval(*storage_, initial_);
    auto& device = mx::metal::device(checked_stream().device);
    auto library = device.get_library(MLX_METAL_LIBRARY_NAME, library_directory());
    for (const auto* suffix : {"_b1", "_b4"}) {
      device.get_kernel(std::string("resin_cache_slice") + suffix, library);
      device.get_kernel(std::string("resin_cache_rows") + suffix + "_i4", library);
      device.get_kernel(std::string("resin_cache_rows") + suffix + "_i8", library);
    }
  }

  mx::array initial_epoch() { return initial_; }
  mx::array read_epoch(const mx::array& epoch) {
    check_epoch(epoch);
    return operation(Kind::Read, {epoch});
  }
  mx::array scatter_rows(const mx::array& epoch, const mx::array& indices,
                         const mx::array& updates, int row_bytes,
                         const std::vector<mx::array>& after) {
    check_epoch(epoch);
    if (indices.ndim() != 1 || (indices.dtype() != mx::int32 && indices.dtype() != mx::int64)) {
      throw std::invalid_argument("CacheStorage row indices must be rank-one int32/int64");
    }
    if (row_bytes <= 0 || indices.size() > std::numeric_limits<size_t>::max() / size_t(row_bytes) ||
        updates.nbytes() != indices.size() * size_t(row_bytes)) {
      throw std::invalid_argument("CacheStorage row updates must supply row_bytes bytes per index");
    }
    return operation(Kind::Rows, {epoch, mx::contiguous(indices), mx::contiguous(updates)},
                     0, row_bytes, after);
  }
  mx::array write_slice(const mx::array& epoch, int64_t byte_start,
                        const mx::array& updates, const std::vector<mx::array>& after) {
    check_epoch(epoch);
    if (byte_start < 0 || uint64_t(byte_start) > storage_->nbytes() ||
        updates.nbytes() > storage_->nbytes() - uint64_t(byte_start)) {
      throw std::invalid_argument("CacheStorage slice is outside the allocation");
    }
    return operation(Kind::Slice, {epoch, mx::contiguous(updates)}, byte_start, 0, after);
  }
  size_t nbytes() const { return storage_->nbytes(); }
  uintptr_t buffer_identity() const { return reinterpret_cast<uintptr_t>(storage_->buffer().ptr()); }

 private:
  mx::array operation(Kind kind, std::vector<mx::array> inputs, uint64_t byte_start = 0,
                      uint64_t row_bytes = 0, const std::vector<mx::array>& after = {}) {
    inputs.insert(inputs.end(), after.begin(), after.end());
    const bool read = kind == Kind::Read;
    return mx::array(read ? storage_->shape() : mx::Shape{kEpochElements},
                     read ? storage_->dtype() : mx::uint32,
                     std::make_shared<CacheOperation>(checked_stream(), storage_, kind, byte_start, row_bytes), inputs);
  }
  std::shared_ptr<mx::array> storage_;
  mx::array initial_;
};
}  // namespace

NB_MODULE(MLX_EXTENSION_NAME, module) {
  resin_models_mlx::native::bind_paged_attention(module);
  nb::class_<CacheStorage>(module, "CacheStorage")
      .def(nb::init<const std::vector<int>&>(), "byte_shape"_a)
      .def("initial_epoch", &CacheStorage::initial_epoch)
      .def("read_epoch", &CacheStorage::read_epoch, "epoch"_a)
      .def("scatter_rows", &CacheStorage::scatter_rows, "epoch"_a, "row_indices"_a, "updates"_a,
           nb::kw_only(), "row_bytes"_a, "after"_a = std::vector<mx::array>{})
      .def("write_slice", &CacheStorage::write_slice, "epoch"_a, "byte_start"_a, "updates"_a,
           nb::kw_only(), "after"_a = std::vector<mx::array>{})
      .def_prop_ro("nbytes", &CacheStorage::nbytes)
      .def_prop_ro("buffer_identity", &CacheStorage::buffer_identity);
}
