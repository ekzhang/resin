// Engine-private mutable cache storage using MLX's Metal stream and hazards.
// This is an inference resource, not an immutable MLX tensor operation.
#include "common.h"
#include <nanobind/stl/vector.h>
#include <mlx/array.h>
#include <mlx/allocator.h>
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
using resin_models_mlx::native::load_kernel;

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

class CacheOperation : public mx::Primitive {
 public:
  CacheOperation(mx::Stream stream, mx::array arena, Kind kind,
                 uint64_t byte_start, uint64_t row_bytes)
      : Primitive(stream), arena_(std::move(arena)), kind_(kind),
        byte_start_(byte_start), row_bytes_(row_bytes) {}

  const char* name() const override { return "ResinPrivateCache"; }
  // Primitive defaults to is_equivalent=false. These resource operations must
  // not be CSE'd, differentiated, vectorized or graph-compiled.
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("CacheStorage has no CPU implementation");
  }

  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    auto& output = outputs[0];
    if (kind_ == Kind::Read) {
      // Private read view only. The eventual attention/gather primitive binds
      // this exact resource as its input. No memory access occurs here.
      output.copy_shared_buffer(arena_);
      return;
    }
    output.set_data(mx::allocator::malloc(output.nbytes()));
    const bool scatter = kind_ == Kind::Rows;
    const size_t update_index = scatter ? 2 : 1;
    const auto& updates = inputs[update_index];
    // Sliced contiguous inputs can have a nonzero buffer offset. Use word
    // copies only when both source and destination are aligned for every row.
    const bool aligned = updates.offset() % 4 == 0 && updates.nbytes() % 4 == 0 &&
                         byte_start_ % 4 == 0 && row_bytes_ % 4 == 0;
    const uint64_t word_bytes = aligned ? 4 : 1;
    std::string kernel_name = scatter ? "resin_cache_rows" : "resin_cache_slice";
    kernel_name += aligned ? "_b4" : "_b1";
    if (scatter) {
      kernel_name += inputs[1].dtype() == mx::int32 ? "_i4" : "_i8";
    }
    auto kernel = load_kernel(stream(), kernel_name);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    // Keep every completed reader in the writer's resource frontier. An
    // alias-only Depends node can lose this edge across encoder boundaries.
    // Slot 30 is outside the shaders' arguments; repeated binding records
    // every real input buffer without copying any of its contents.
    for (size_t i = update_index + 1; i < inputs.size(); ++i) {
      encoder.set_input_array(inputs[i], 30);
    }
    encoder.set_output_array(arena_, 0);
    encoder.set_input_array(inputs[0], 1);
    encoder.set_output_array(output, 2);
    // The arena is captured by the primitive. Retain its Data until the GPU
    // finishes without removing it from MLX hazard tracking (add_temporary).
    encoder.get_command_buffer()->addCompletedHandler(
        [keep = arena_.data_shared_ptr()](MTL::CommandBuffer*) {});
    MTL::Size grid;
    if (scatter) {
      const uint64_t width = row_bytes_ / word_bytes;
      const uint64_t count = inputs[1].size();
      encoder.set_input_array(inputs[1], 3);
      encoder.set_input_array(updates, 4);
      encoder.set_bytes(width, 5);
      encoder.set_bytes(count, 6);
      encoder.set_bytes(arena_.nbytes() / row_bytes_, 7);
      // One axis per row and column. Even an empty write advances the epoch.
      const auto columns = std::max<uint64_t>(kEpochElements, count ? width : 0);
      grid = MTL::Size(columns, std::max<uint64_t>(count, 1), 1);
    } else {
      const uint64_t count = updates.nbytes() / word_bytes;
      encoder.set_input_array(updates, 3);
      encoder.set_bytes(byte_start_ / word_bytes, 4);
      encoder.set_bytes(count, 5);
      // Bound the first axis and reconstruct the linear address as ulong in MSL.
      const auto work = std::max<uint64_t>(count, kEpochElements);
      const auto width = std::min<uint64_t>(work, 1 << 20);
      const auto height = work / width + (work % width != 0);
      if (height > UINT32_MAX) {
        throw std::invalid_argument("CacheStorage work exceeds the Metal grid capacity");
      }
      grid = MTL::Size(width, height, 1);
    }
    auto threads = std::min<size_t>(256, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(grid, MTL::Size(threads, 1, 1));
  }

 private:
  mx::array arena_;
  Kind kind_;
  uint64_t byte_start_;
  uint64_t row_bytes_;
};

class CacheStorage {
 public:
  explicit CacheStorage(const std::vector<int>& byte_shape)
      : arena_(allocate_arena(byte_shape)),
        initial_(mx::zeros({kEpochElements}, mx::uint32, checked_stream())) {
    // Materialize on the creating thread; serving may use another MLX stream.
    mx::eval(arena_, initial_);
    auto stream = checked_stream();
    for (const auto* suffix : {"_b1", "_b4"}) {
      load_kernel(stream, std::string("resin_cache_slice") + suffix);
      load_kernel(stream, std::string("resin_cache_rows") + suffix + "_i4");
      load_kernel(stream, std::string("resin_cache_rows") + suffix + "_i8");
    }
  }

  mx::array initial_epoch() { return initial_; }
  mx::array read_epoch(const mx::array& epoch) {
    return operation(Kind::Read, {epoch});
  }
  mx::array scatter_rows(const mx::array& epoch, const mx::array& indices,
                         const mx::array& updates, int row_bytes,
                         const std::vector<mx::array>& after) {
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
    if (byte_start < 0 || uint64_t(byte_start) > arena_.nbytes() ||
        updates.nbytes() > arena_.nbytes() - uint64_t(byte_start)) {
      throw std::invalid_argument("CacheStorage slice is outside the allocation");
    }
    return operation(Kind::Slice, {epoch, mx::contiguous(updates)}, byte_start, 0, after);
  }
  size_t nbytes() const { return arena_.nbytes(); }
  uintptr_t buffer_identity() const { return reinterpret_cast<uintptr_t>(arena_.buffer().ptr()); }

 private:
  mx::array operation(Kind kind, std::vector<mx::array> inputs, uint64_t byte_start = 0,
                      uint64_t row_bytes = 0, const std::vector<mx::array>& after = {}) {
    const auto& epoch = inputs.front();
    if (epoch.dtype() != mx::uint32 || epoch.ndim() != 1 || epoch.size() != kEpochElements) {
      throw std::invalid_argument("CacheStorage epoch must be an eight-element uint32 array");
    }
    inputs.insert(inputs.end(), after.begin(), after.end());
    const bool read = kind == Kind::Read;
    return mx::array(read ? arena_.shape() : mx::Shape{kEpochElements},
                     read ? arena_.dtype() : mx::uint32,
                     std::make_shared<CacheOperation>(checked_stream(), arena_, kind, byte_start, row_bytes),
                     std::move(inputs));
  }
  mx::array arena_;
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
